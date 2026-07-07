# BioDynamics Agent v4 - ODE Builder Agent（Phase 6 / Task 6.6.2）
#
# 从 Reaction IR 渲染 ODE 代码，强制 DDE 验证与 pathway_tag 隔离。
#
# 职责：
# 1. 调用 app.ode_renderer_v2.ODERendererV2.render() 从 Reaction IR 生成 ODE 代码
# 2. 振荡通路（p53/NF_KB）→ 强制 dde_delay_minutes > 0（DDE 强制）
# 3. 根据 pathway_class 选择模板（oscillatory_feedback.j2 / bistable_switch.j2）
# 4. pathway_tag 隔离：ODE 参数不得混入其他通路的参数
# 5. 结构化输出 v4_ode_system（不修改 v3 ode_model）
#
# 设计原则（铁律）：
# 1. Feature Flag V4_DYNAMIC_ROUTING_ENABLED=false → 返回 {}（不执行）
# 2. 不修改 v3 ode_model / python_code；仅新增 v4_ode_system
# 3. 失败降级：任何异常返回 {"v4_ode_system": {}, "warnings": [...]}
# 4. 不创建新的 ODE 生成逻辑，仅调用现有 ode_renderer_v2 并结构化输出
# 5. DDE 强制：振荡通路 dde_delay_minutes 必须 > 0
#
# 参考：
# - app.ode_renderer_v2: ODERendererV2 / render_ode_v2 / _OSCILLATORY_PATHWAYS
# - app.reaction_ir_v2.schema: ReactionIRv2
# - tasks.md SubTask 6.6.2

from __future__ import annotations

import logging
from typing import Any

# app.config 无循环依赖风险（不导入 agents_v4），可在模块级导入
from app.config import settings

logger = logging.getLogger(__name__)

# =============================================================================
# 常量
# =============================================================================
# 振荡通路类别集合（需 DDE 强制），与 mechanism_builder.py 对齐
_OSCILLATORY_PATHWAYS: set[str] = {
    "p53", "p53_signaling",
    "NF_KB", "NF_kB", "nf_kb",
    "TGF_BETA", "TGF_beta", "tgf_beta",
    "JAK_STAT", "JAK-STAT",
}

# 双稳态通路类别（需 bistable_switch.j2）
_BISTABLE_PATHWAYS: set[str] = {"Apoptosis", "Cell_Cycle", "APOPTOSIS", "CELL_CYCLE"}

# 合法的 kinetics_type 白名单
_VALID_KINETICS: set[str] = {
    "mass_action", "Michaelis_Menten", "Hill", "Boolean", "hybrid",
}

# DDE 默认延迟（分钟），振荡通路未配置延迟时使用
_DEFAULT_DDE_DELAY_MINUTES: float = 30.0


class ODEBuilderAgent:
    """v4 ODE 构建 Agent：从 Reaction IR 渲染 ODE 代码。

    调用现有 ``ODERendererV2`` 从 ReactionIRv2 + PathwayGraph 渲染 ODE 代码，
    并强制执行科学约束（DDE 强制、pathway_tag 隔离），结构化输出 v4_ode_system。

    用法::

        agent = ODEBuilderAgent()
        update = agent.build(state)
        # update = {"v4_ode_system": {ode_code, equations, ...}, "warnings": [...]}
    """

    AGENT_VERSION: str = "v4.0"

    # -------------------------------------------------------------------------
    # 主入口
    # -------------------------------------------------------------------------
    def build(self, state: dict) -> dict:
        """主入口：从 Reaction IR 渲染 ODE 代码，强制 DDE 验证。

        Args:
            state: LangGraph 全局状态，读取：
                - ``v4_reaction_ir``: ReactionIRv2 序列化 dict（含 species/reactions）
                - ``v4_pathway_class``: 通路类别字符串
                - ``v4_pathway_graph``: PathwayGraph 序列化 dict（含 temporal/DDE 信息）

        Returns:
            flag=false 时返回 {}
            正常时返回 ``{"v4_ode_system": {...}, "warnings": [...]}``
            失败时返回 ``{"v4_ode_system": {}, "warnings": [...]}``

            v4_ode_system 结构::

                {
                    "ode_code": str,              # 可执行 ODE Python 代码
                    "equations": list[str],       # ODE 方程列表
                    "parameters": dict,           # 动力学参数
                    "pathway_class": str,         # 通路类别
                    "dde_delay_minutes": float,   # DDE 延迟（分钟）
                    "template_name": str,         # 使用的模板名
                }
        """
        # 1. Feature Flag 检查（铁律：flag=false 不执行）
        if not getattr(settings, "V4_DYNAMIC_ROUTING_ENABLED", False):
            logger.debug("V4_DYNAMIC_ROUTING_ENABLED=false，ODEBuilder 跳过")
            return {}

        try:
            # 2. 提取输入
            reaction_ir = state.get("v4_reaction_ir") or {}
            pathway_class = state.get("v4_pathway_class", "") or ""
            pathway_graph = state.get("v4_pathway_graph") or {}

            if not reaction_ir:
                logger.warning("ODEBuilder: v4_reaction_ir 为空，降级返回空 ode_system")
                return {
                    "v4_ode_system": {},
                    "warnings": ["v4_reaction_ir 为空，无法渲染 ODE"],
                }

            # 3. 提取 DDE / temporal 信息
            temporal = pathway_graph.get("temporal") or {}
            requires_dde = bool(temporal.get("requires_dde", False))
            dde_delay_minutes = float(temporal.get("dde_delay_minutes", 0.0) or 0.0)
            t_end = float(temporal.get("t_end_minutes", 60.0) or 60.0)

            # 4. 判断振荡通路 → DDE 强制
            is_oscillatory = self._is_oscillatory(pathway_class)
            warnings: list[str] = []

            if is_oscillatory:
                # DDE 强制：振荡通路 dde_delay_minutes 必须 > 0
                if dde_delay_minutes <= 0:
                    warnings.append(
                        f"振荡通路 '{pathway_class}' 的 dde_delay_minutes={dde_delay_minutes} <= 0，"
                        f"已强制设置为默认延迟 {_DEFAULT_DDE_DELAY_MINUTES} 分钟（DDE 强制）"
                    )
                    dde_delay_minutes = _DEFAULT_DDE_DELAY_MINUTES
                requires_dde = True

            # 5. 选择模板
            template_name = self._select_template(pathway_class, requires_dde)

            # 6. 调用现有 ODERendererV2 渲染 ODE 代码
            ode_code = self._render_ode_code(
                reaction_ir=reaction_ir,
                pathway_graph=pathway_graph,
                pathway_class=pathway_class,
                t_end=t_end,
            )

            if not ode_code:
                warnings.append("ODERendererV2 返回空代码，ODE 渲染可能未启用或模板缺失")

            # 7. 提取方程与参数（从 reaction_ir 结构化）
            equations = self._extract_equations(reaction_ir)
            parameters = self._extract_parameters(reaction_ir, pathway_class)

            # 8. pathway_tag 隔离验证：检查参数是否标注了 pathway_tag
            isolation_warnings = self._check_pathway_isolation(parameters, pathway_class)
            warnings.extend(isolation_warnings)

            # 9. DDE 验证：振荡通路的 ode_code 应包含 DDE 相关内容
            if is_oscillatory and ode_code:
                dde_check = self._validate_dde_in_code(ode_code, dde_delay_minutes)
                if dde_check:
                    warnings.append(dde_check)

            ode_system: dict[str, Any] = {
                "ode_code": ode_code,
                "equations": equations,
                "parameters": parameters,
                "pathway_class": pathway_class,
                "dde_delay_minutes": dde_delay_minutes,
                "template_name": template_name,
            }

            logger.info(
                "ODEBuilder: 渲染完成 pathway=%s template=%s dde_delay=%.1f equations=%d",
                pathway_class, template_name, dde_delay_minutes, len(equations),
            )

            return {
                "v4_ode_system": ode_system,
                "warnings": warnings,
            }

        except Exception as exc:
            # 失败降级：返回空 ode_system + warning，不阻塞流水线
            logger.warning("ODEBuilder.build 失败，降级返回空: %s", exc)
            return {
                "v4_ode_system": {},
                "warnings": [f"ODEBuilder 构建失败: {exc}"],
            }

    def generate(self, state: dict) -> dict:
        """DynamicRouter 调度入口（别名，委托给 build）。"""
        return self.build(state)

    # -------------------------------------------------------------------------
    # 内部辅助方法
    # -------------------------------------------------------------------------
    @staticmethod
    def _is_oscillatory(pathway_class: str) -> bool:
        """判断是否为振荡通路（p53/NF_KB/TGF_beta/JAK_STAT）。"""
        if not pathway_class:
            return False
        pc_lower = pathway_class.lower()
        for osc in _OSCILLATORY_PATHWAYS:
            if osc.lower() in pc_lower:
                return True
        return False

    @staticmethod
    def _select_template(pathway_class: str, requires_dde: bool) -> str:
        """根据通路类别与 DDE 需求选择 v4 模板。

        选择规则（与 ODERendererV2._select_template 对齐）：
        1. 振荡通路 / requires_dde → oscillatory_feedback.j2
        2. 双稳态通路 → bistable_switch.j2
        3. 其他 → oscillatory_feedback.j2（默认，支持所有机制）
        """
        # 大小写不敏感匹配振荡通路
        pc = pathway_class or ""
        for osc in _OSCILLATORY_PATHWAYS:
            if osc.lower() in pc.lower():
                return "oscillatory_feedback.j2"
        if requires_dde:
            return "oscillatory_feedback.j2"
        # 双稳态通路
        for bs in _BISTABLE_PATHWAYS:
            if bs.lower() in pc.lower():
                return "bistable_switch.j2"
        # 默认（支持所有机制）
        return "oscillatory_feedback.j2"

    @staticmethod
    def _render_ode_code(
        reaction_ir: dict,
        pathway_graph: dict,
        pathway_class: str,
        t_end: float,
    ) -> str:
        """调用 ODERendererV2 渲染 ODE 代码。

        封装对现有 ode_renderer_v2 的调用，处理导入失败与渲染异常。
        不创建新的 ODE 生成逻辑（铁律 4）。
        """
        try:
            from app.ode_renderer_v2 import ODERendererV2

            renderer = ODERendererV2()
            code = renderer.render(
                pathway_class=pathway_class,
                reaction_ir=reaction_ir,
                pathway_graph=pathway_graph,
                t_end=t_end,
            )
            return code or ""
        except ImportError as exc:
            logger.debug("ODEBuilder: ode_renderer_v2 不可导入: %s", exc)
            return ""
        except Exception as exc:
            logger.warning("ODEBuilder: ODERendererV2.render 异常: %s", exc)
            return ""

    @staticmethod
    def _extract_equations(reaction_ir: dict) -> list[str]:
        """从 reaction_ir 提取 ODE 方程列表。

        从 reactions 的 reaction_eq / equation 字段提取定性方程描述。
        """
        equations: list[str] = []
        reactions = reaction_ir.get("reactions", []) or []
        for rxn in reactions:
            if not isinstance(rxn, dict):
                continue
            eq = rxn.get("reaction_eq") or rxn.get("equation") or ""
            if eq and eq not in equations:
                equations.append(eq)
        return equations

    @staticmethod
    def _extract_parameters(reaction_ir: dict, pathway_class: str) -> dict:
        """从 reaction_ir 提取动力学参数，标注 pathway_tag。

        返回格式：{param_name: {value, pathway_tag, source}}
        每个 参数标注其来源通路，保证 pathway_tag 隔离可追溯。
        """
        params: dict[str, dict[str, Any]] = {}

        # 从 species.parameters 提取
        for sp in reaction_ir.get("species", []) or []:
            if not isinstance(sp, dict):
                continue
            sp_name = sp.get("name") or sp.get("id", "")
            sp_tag = sp.get("pathway_tag", "") or pathway_class
            sp_params = sp.get("parameters", {})
            if isinstance(sp_params, dict):
                for k, v in sp_params.items():
                    if isinstance(v, (int, float)):
                        params[k] = {
                            "value": float(v),
                            "pathway_tag": sp_tag,
                            "source": "reaction_ir",
                        }

        # 从 reactions.parameters 提取
        for rxn in reaction_ir.get("reactions", []) or []:
            if not isinstance(rxn, dict):
                continue
            rxn_tag = rxn.get("pathway_tag", "") or pathway_class
            rxn_params = rxn.get("parameters", {})
            if isinstance(rxn_params, dict):
                for k, v in rxn_params.items():
                    if isinstance(v, (int, float)):
                        if k not in params:
                            params[k] = {
                                "value": float(v),
                                "pathway_tag": rxn_tag,
                                "source": "reaction_ir",
                            }

        return params

    @staticmethod
    def _check_pathway_isolation(parameters: dict, pathway_class: str) -> list[str]:
        """检查参数的 pathway_tag 隔离。

        验证：参数标注的 pathway_tag 应与当前通路一致，
        若出现不同 pathway_tag 的参数，记录 warning（潜在跨通路泄漏）。
        """
        warnings: list[str] = []
        seen_tags: set[str] = set()
        for p_name, p_info in parameters.items():
            if isinstance(p_info, dict):
                tag = p_info.get("pathway_tag", "") or pathway_class
                if tag and tag != pathway_class:
                    seen_tags.add(tag)

        for foreign_tag in seen_tags:
            warnings.append(
                f"pathway_tag 隔离警告：ODE 参数中存在来自通路 '{foreign_tag}' 的参数，"
                f"当前通路为 '{pathway_class}'，请确认是否为跨通路共享参数"
            )
        return warnings

    @staticmethod
    def _validate_dde_in_code(ode_code: str, dde_delay_minutes: float) -> str:
        """验证 ODE 代码是否包含 DDE 相关内容。

        振荡通路的 ODE 代码应包含延迟相关变量（dde_delay / DDE_DELAY / delay）。
        若缺失，返回警告字符串；否则返回空字符串。
        """
        code_lower = ode_code.lower()
        dde_keywords = ["dde_delay", "dde", "delay", "history", "past_value"]
        has_dde = any(kw in code_lower for kw in dde_keywords)
        if not has_dde and dde_delay_minutes > 0:
            return (
                f"DDE 验证警告：振荡通路 ODE 代码中未检测到 DDE 相关变量，"
                f"但 dde_delay_minutes={dde_delay_minutes} > 0"
            )
        return ""


# =============================================================================
# DynamicRouter 兼容别名
# =============================================================================
# DynamicRouter._get_class_name 期望短名 ODEBuilder（agent_registry_v4 约定）
ODEBuilder = ODEBuilderAgent


__all__ = [
    "ODEBuilderAgent",
    "ODEBuilder",
]
