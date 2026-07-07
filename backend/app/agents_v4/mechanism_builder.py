# BioDynamics Agent v4 - Mechanism Builder Agent（Phase 6 / Task 6.6.1）
#
# 强制 MM/Hill/DDE 动力学机制选择。
#
# 职责：
# 1. 遍历 PathwayGraph 的每条 edge，根据 mechanism 推断动力学类型
# 2. 酶催化机制（phosphorylation/dephosphorylation/cleavage）→ 强制 Michaelis_Menten（不可降级）
# 3. 转录机制 → Hill 动力学
# 4. 振荡通路（p53/NF_KB）+ 反馈延迟 → DDE 强制
# 5. 每条 assignment 记录 pathway_tag，保证通路隔离
#
# 设计原则（铁律）：
# 1. Feature Flag V4_DYNAMIC_ROUTING_ENABLED=false → 返回 {}（不执行）
# 2. 不修改 v3 任何字段；仅新增 v4_mechanism_assignments
# 3. 失败降级：任何异常返回 {"v4_mechanism_assignments": [], "warnings": [...]}
# 4. pathway_tag 隔离：每条 assignment 记录其 pathway_tag，无跨通路泄漏
# 5. MM 不可降级：酶催化机制被误标 mass_action 时强制恢复 MM 并记录 warning
#
# 参考：
# - app.reaction_ir_v2.mechanism_types: is_enzymatic_mechanism() / MechanismType
# - app.ode_renderer_v2._OSCILLATORY_PATHWAYS
# - tasks.md SubTask 6.6.1

from __future__ import annotations

import logging
from typing import Any

# app.config 无循环依赖风险（不导入 agents_v4），可在模块级导入
from app.config import settings

logger = logging.getLogger(__name__)

# =============================================================================
# 常量
# =============================================================================
# 振荡通路类别集合（需 DDE 强制）
# 兼容 specialist 命名（p53/NF_KB/TGF_BETA/JAK_STAT）与 ode_renderer 命名
# （p53_signaling/NF_kB/TGF_beta/JAK_STAT），大小写均覆盖
_OSCILLATORY_PATHWAYS: set[str] = {
    "p53", "p53_signaling",
    "NF_KB", "NF_kB", "nf_kb",
    "TGF_BETA", "TGF_beta", "tgf_beta",
    "JAK_STAT", "JAK-STAT",
}

# 合法的 kinetics_type 白名单（与 PathwayEdge._validate_kinetics 对齐）
_VALID_KINETICS: set[str] = {
    "mass_action", "Michaelis_Menten", "Hill", "Boolean", "hybrid",
}


class MechanismBuilderAgent:
    """v4 机制构建 Agent：强制 MM/Hill/DDE 动力学机制选择。

    从 PathwayGraph 的 edges 推断每条反应的动力学类型，强制执行科学约束：
    - 酶催化机制 → Michaelis_Menten（不可降级为 mass_action）
    - 转录机制 → Hill 动力学
    - 振荡通路 + 反馈延迟 → DDE 强制
    - 其他机制 → mass_action（或保留 edge 已有 kinetics_type）

    用法::

        agent = MechanismBuilderAgent()
        update = agent.build(state)
        # update = {"v4_mechanism_assignments": [...], "warnings": [...]}
    """

    AGENT_VERSION: str = "v4.0"

    # -------------------------------------------------------------------------
    # 主入口
    # -------------------------------------------------------------------------
    def build(self, state: dict) -> dict:
        """主入口：从 PathwayGraph 推断机制类型，强制科学约束。

        Args:
            state: LangGraph 全局状态，读取：
                - ``v4_pathway_graph``: PathwayGraph 序列化 dict（含 edges/feedback_loops）
                - ``v4_pathway_class``: 通路类别字符串（如 "p53_signaling"）

        Returns:
            flag=false 时返回 {}
            正常时返回 ``{"v4_mechanism_assignments": [...], "warnings": [...]}``
            失败时返回 ``{"v4_mechanism_assignments": [], "warnings": [...]}``

            每条 assignment 结构::

                {
                    "reaction_id": str,       # edge.id
                    "mechanism": str,         # 17 类机制之一
                    "kinetics_type": str,     # mass_action/Michaelis_Menten/Hill/...
                    "pathway_tag": str,       # 通路标签（隔离用）
                    "dde_required": bool,     # 是否需要 DDE 求解器
                }
        """
        # 1. Feature Flag 检查（铁律：flag=false 不执行）
        if not getattr(settings, "V4_DYNAMIC_ROUTING_ENABLED", False):
            logger.debug("V4_DYNAMIC_ROUTING_ENABLED=false，MechanismBuilder 跳过")
            return {}

        try:
            # 2. 提取输入
            pathway_graph = state.get("v4_pathway_graph") or {}
            pathway_class = state.get("v4_pathway_class", "") or ""

            if not pathway_graph:
                logger.warning("MechanismBuilder: v4_pathway_graph 为空，降级返回空 assignments")
                return {
                    "v4_mechanism_assignments": [],
                    "warnings": ["v4_pathway_graph 为空，无法推断机制类型"],
                }

            edges = pathway_graph.get("edges", []) or []
            if not edges:
                logger.warning("MechanismBuilder: pathway_graph.edges 为空")
                return {
                    "v4_mechanism_assignments": [],
                    "warnings": ["pathway_graph 无 edges，无法推断机制类型"],
                }

            # 3. 判断振荡通路（DDE 强制前提）
            is_oscillatory = self._is_oscillatory(pathway_class)

            # 4. 提取反馈环延迟（DDE 强制条件之一）
            feedback_delay_positive = self._has_positive_feedback_delay(pathway_graph)

            # 5. 遍历 edges，推断动力学类型
            assignments: list[dict[str, Any]] = []
            warnings: list[str] = []
            seen_pathway_tags: set[str] = set()

            for edge in edges:
                if not isinstance(edge, dict):
                    continue

                reaction_id = edge.get("id", "") or edge.get("reaction_id", "")
                mechanism = edge.get("mechanism", "activation") or "activation"
                # pathway_tag 优先取 edge 自身，其次取 pathway_class
                pathway_tag = edge.get("pathway_tag", "") or pathway_class
                existing_kinetics = edge.get("kinetics_type", "mass_action") or "mass_action"

                # 确定正确的动力学类型（强制约束）
                kinetics_type = self._determine_kinetics(mechanism, existing_kinetics)

                # MM 不可降级验证：酶催化机制的 existing kinetics 不是 MM → 记录警告
                if self._is_enzymatic(mechanism) and existing_kinetics != "Michaelis_Menten":
                    warnings.append(
                        f"反应 {reaction_id}: 酶催化机制 '{mechanism}' 原始 "
                        f"kinetics_type='{existing_kinetics}'，已强制恢复为 "
                        f"Michaelis_Menten（MM 不可降级）"
                    )

                # DDE 强制：振荡通路 + 反馈延迟 + 当前 edge 是反馈边
                dde_required = False
                if is_oscillatory and feedback_delay_positive:
                    if edge.get("is_feedback", False):
                        dde_required = True

                # 记录 pathway_tag（隔离检查）
                seen_pathway_tags.add(pathway_tag)

                assignments.append({
                    "reaction_id": reaction_id,
                    "mechanism": mechanism,
                    "kinetics_type": kinetics_type,
                    "pathway_tag": pathway_tag,
                    "dde_required": dde_required,
                })

            # 6. DDE 强制验证：振荡通路至少有一条 dde_required=True
            if is_oscillatory and feedback_delay_positive:
                if not any(a["dde_required"] for a in assignments):
                    warnings.append(
                        f"振荡通路 '{pathway_class}' 检测到反馈延迟但无 DDE 边，"
                        f"请检查 feedback_loops 标记"
                    )

            logger.info(
                "MechanismBuilder: 推断 %d 条机制分配，%d 条警告",
                len(assignments), len(warnings),
            )

            return {
                "v4_mechanism_assignments": assignments,
                "warnings": warnings,
            }

        except Exception as exc:
            # 失败降级：返回空 assignments + warning，不阻塞流水线
            logger.warning("MechanismBuilder.build 失败，降级返回空: %s", exc)
            return {
                "v4_mechanism_assignments": [],
                "warnings": [f"MechanismBuilder 构建失败: {exc}"],
            }

    def generate(self, state: dict) -> dict:
        """DynamicRouter 调度入口（别名，委托给 build）。

        DynamicRouter.execute_agent 优先调用 generate/run/execute 方法，
        此方法作为 build 的别名以兼容 Router 调度约定。
        """
        return self.build(state)

    # -------------------------------------------------------------------------
    # 内部辅助方法
    # -------------------------------------------------------------------------
    @staticmethod
    def _is_oscillatory(pathway_class: str) -> bool:
        """判断是否为振荡通路（p53/NF_KB/TGF_beta/JAK_STAT）。

        兼容多种命名变体（specialist 命名与 ode_renderer 命名）。
        """
        if not pathway_class:
            return False
        # 大小写不敏感匹配，并覆盖 MULTI: 前缀（多通路场景包含振荡通路）
        pc_lower = pathway_class.lower()
        for osc in _OSCILLATORY_PATHWAYS:
            if osc.lower() in pc_lower:
                return True
        return False

    @staticmethod
    def _is_enzymatic(mechanism: str) -> bool:
        """判断是否为酶催化机制（强制 Michaelis_Menten）。

        磷酸化 / 去磷酸化 / 切割 必须用 MM，禁止降级为 mass-action。
        与 app.reaction_ir_v2.mechanism_types.is_enzymatic_mechanism 对齐。
        """
        return mechanism.lower() in {"phosphorylation", "dephosphorylation", "cleavage"}

    @staticmethod
    def _determine_kinetics(mechanism: str, existing: str) -> str:
        """根据机制类型确定正确的动力学类型（强制约束）。

        规则：
        1. 酶催化（phosphorylation/dephosphorylation/cleavage）→ Michaelis_Menten（强制）
        2. 转录（transcription）→ Hill
        3. 其他机制 → 保留 edge 已有 kinetics_type（若合法）或 mass_action
        """
        m = mechanism.lower()
        # 酶催化 → 强制 MM（不可降级）
        if m in {"phosphorylation", "dephosphorylation", "cleavage"}:
            return "Michaelis_Menten"
        # 转录 → Hill 动力学
        if m == "transcription":
            return "Hill"
        # 其他机制：保留 existing（若合法）或默认 mass_action
        if existing in _VALID_KINETICS:
            return existing
        return "mass_action"

    @staticmethod
    def _has_positive_feedback_delay(pathway_graph: dict) -> bool:
        """检查 pathway_graph 中是否存在延迟 > 0 的反馈环。

        DDE 强制条件之一：振荡通路 + 反馈延迟 > 0。
        """
        feedback_loops = pathway_graph.get("feedback_loops", []) or []
        for fl in feedback_loops:
            if not isinstance(fl, dict):
                continue
            try:
                delay = float(fl.get("delay_minutes", 0.0))
                if delay > 0:
                    return True
            except (TypeError, ValueError):
                continue
        return False


# =============================================================================
# DynamicRouter 兼容别名
# =============================================================================
# DynamicRouter._get_class_name 期望短名 MechanismBuilder（agent_registry_v4 约定），
# 此别名保证 Router.execute_agent 能正确发现并实例化 Agent 类。
MechanismBuilder = MechanismBuilderAgent


__all__ = [
    "MechanismBuilderAgent",
    "MechanismBuilder",
]
