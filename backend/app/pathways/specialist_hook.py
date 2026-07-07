# BioDynamics Agent v4 - Pathway Specialist Hook (Phase 4 / Task 4.14)
# LangGraph 节点：在 worker_mechanism 后调用 10 Pathway Specialist，
# 收集通路特异 Reaction IR 片段写入 state.v4_specialist_outputs。
#
# 设计原则（铁律）：
# 1. Feature Flag V4_PATHWAY_SPECIALIST_ENABLED=false 时返回 {}，不执行任何逻辑
# 2. 不修改 v3 任何字段（network_json / entities / mechanism / parameters 等）
# 3. 不生成 ODE / 不调用 RAG / 不做 SBML 验证（职责边界严格）
# 4. 失败降级：任何异常都返回 {}，不阻塞主流水线
# 5. 不重新设计 P1/P2/P3，仅消费其产出（v4_pathway_class / v4_pathway_graph）
# 6. flag 隐含规则：V4_PATHWAY_SPECIALIST_ENABLED=true 隐含
#    V4_PATHWAY_PLANNER_ENABLED=true（Specialist 依赖 Planner 输出）；
#    若 v4_pathway_class 缺失，降级返回 {}（不抛异常）
#
# 依赖：
# - P4 Pathway Planner 输出 state.v4_pathway_class（单通路 / 多通路 / UNKNOWN）
# - P4 Pathway Planner 输出 state.v4_pathway_graph（含预识别 crosstalk_edges）
# - P4 pathway_registry.SPECIALIST_REGISTRY + get_specialist
# - P4 pathway_specialist_base.PathwaySpecialistBase（apply_* 接口）
# - P4 pathway_planner.parse_pathway_class（解析 MULTI: 字符串）
#
# 参考：
# - spec.md Part 3 Specialist 编排（第 269-272 行）
# - tasks.md SubTask 4.14.1

from __future__ import annotations

import logging
from typing import Any

from app.state import set_v4_state

logger = logging.getLogger(__name__)


# =============================================================================
# Specialist 子模块懒加载（避免启动时全量导入）
# =============================================================================
# specialists/__init__.py 不自动导入子模块（避免循环依赖与启动开销）。
# 此处按需导入 10 个 Specialist 模块，触发 @register_specialist 装饰器注册。
# 导入失败时记录 warning 但不抛异常（部分 Specialist 缺失仍可运行）。
_SPECIALISTS_IMPORTED: bool = False


def _ensure_specialists_imported() -> None:
    """懒加载 10 个 Specialist 子模块，触发 @register_specialist 注册。

    首次调用时执行 import，后续调用直接返回（_SPECIALISTS_IMPORTED 标记）。
    任一 Specialist 导入失败时记录 warning 但不阻塞（部分注册仍可用）。
    """
    global _SPECIALISTS_IMPORTED
    if _SPECIALISTS_IMPORTED:
        return
    _SPECIALISTS_IMPORTED = True
    _specialist_modules = [
        "app.pathways.specialists.egfr_specialist",
        "app.pathways.specialists.mapk_specialist",
        "app.pathways.specialists.pi3k_akt_mtor_specialist",
        "app.pathways.specialists.p53_specialist",
        "app.pathways.specialists.apoptosis_specialist",
        "app.pathways.specialists.cell_cycle_specialist",
        "app.pathways.specialists.jak_stat_specialist",
        "app.pathways.specialists.nf_kappa_b_specialist",
        "app.pathways.specialists.wnt_specialist",
        "app.pathways.specialists.tgf_beta_specialist",
    ]
    import importlib

    for mod_name in _specialist_modules:
        try:
            importlib.import_module(mod_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "specialist_hook: 导入 %s 失败: %s（该通路 Specialist 不可用）",
                mod_name,
                exc,
            )


# =============================================================================
# LangGraph 节点 hook（feature flag 隔离）
# =============================================================================
def specialist_hook_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph 节点：Pathway Specialist hook。

    行为：
    - V4_PATHWAY_SPECIALIST_ENABLED=false：直接返回空 dict（不修改 state）
    - V4_PATHWAY_SPECIALIST_ENABLED=true + v4_pathway_class 缺失：返回 {}
    - V4_PATHWAY_SPECIALIST_ENABLED=true + v4_pathway_class 存在：
      - parse_pathway_class 解析为 pathway 列表
      - 遍历每个 pathway，从 SPECIALIST_REGISTRY 获取 Specialist 实例
      - 调用 apply_core / apply_feedback / apply_crosstalk / apply_validation
      - 合并所有 Specialist 输出，写入 state["v4_specialist_outputs"]

    严格遵守不可碰清单：
    - 不修改 v3 任何字段
    - 不生成 ODE / 不调用 RAG / 不做 SBML 验证
    - 失败时降级返回空 dict，不抛异常

    Args:
        state: LangGraph 全局状态，读取：
            - ``v4_pathway_class``: 通路类别字符串（Pathway Planner 输出）
            - ``v4_pathway_graph``: 通路图（含预识别 crosstalk_edges，可选）

    Returns:
        flag=false 时返回 {}
        flag=true 时返回 {"v4_specialist_outputs": list[dict]}（每条含
        pathway_class / species / reactions / feedback_loops /
        crosstalk_reactions / validation_rules 等字段）
    """
    # 延迟导入 config 避免循环依赖
    from app.config import settings

    # Feature Flag 检查：默认 false，跳过所有逻辑
    if not getattr(settings, "V4_PATHWAY_SPECIALIST_ENABLED", False):
        logger.debug("V4_PATHWAY_SPECIALIST_ENABLED=false，跳过 Specialist")
        return {}

    try:
        pathway_class = state.get("v4_pathway_class", "") or ""
        if not pathway_class or pathway_class == "UNKNOWN":
            logger.info(
                "specialist_hook: v4_pathway_class 为空或 UNKNOWN，跳过 Specialist"
            )
            return {}

        # 延迟导入 parse_pathway_class + get_specialist
        from app.pathways.pathway_planner import parse_pathway_class
        from app.pathways.pathway_registry import get_specialist

        # 懒加载 10 个 Specialist 子模块（触发 @register_specialist）
        _ensure_specialists_imported()

        pathways = parse_pathway_class(pathway_class)
        if not pathways:
            logger.info(
                "specialist_hook: parse_pathway_class 返回空列表，pathway_class=%s",
                pathway_class,
            )
            return {}

        # 从 v4_pathway_graph 提取预识别 crosstalk_edges（供 apply_crosstalk 使用）
        pathway_graph = state.get("v4_pathway_graph") or {}
        crosstalk_edges: list[dict] = []
        if isinstance(pathway_graph, dict):
            # Pathway Planner 输出的预识别载荷含 crosstalk_edges 字段
            crosstalk_edges = pathway_graph.get("crosstalk_edges", []) or []

        specialist_outputs: list[dict[str, Any]] = []
        for pwc in pathways:
            specialist = get_specialist(pwc)
            if specialist is None:
                logger.warning(
                    "specialist_hook: pathway_class='%s' 未注册 Specialist，跳过",
                    pwc,
                )
                continue

            # 调用 4 个 apply_* 方法（每个方法内部已捕获异常，返回空 list/dict）
            core_output = specialist.apply_core(pathway_graph)
            feedback_loops = specialist.apply_feedback(pathway_graph)
            crosstalk_reactions = specialist.apply_crosstalk(
                pathway_graph, crosstalk_edges
            )
            validation_rules = specialist.apply_validation()

            # 合并为单个 Specialist 的输出条目
            # 字段约定与 CrossTalkCoordinator 的 specialist_outputs 对齐
            # （coordinator.py 模块顶部字段约定）
            entry: dict[str, Any] = {
                "pathway_class": pwc,
                "species": core_output.get("species", []) if isinstance(core_output, dict) else [],
                "reactions": core_output.get("reactions", []) if isinstance(core_output, dict) else [],
                "kinetics_overrides": core_output.get("kinetics_overrides", {}) if isinstance(core_output, dict) else {},
                "feedback_loops": feedback_loops,
                "crosstalk_reactions": crosstalk_reactions,
                "crosstalk_edges": crosstalk_edges,
                "validation_rules": validation_rules,
                "source_sbml": core_output.get("source_sbml", "") if isinstance(core_output, dict) else "",
            }

            # 从 crosstalk module data 提取 shared_species（若 Specialist 提供）
            try:
                crosstalk_module = specialist.load_module("crosstalk")
                if crosstalk_module is not None:
                    shared_sp = getattr(crosstalk_module, "shared_species", []) or []
                    entry["shared_species"] = list(shared_sp)
                else:
                    entry["shared_species"] = []
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "specialist_hook: %s load_module('crosstalk') 失败: %s",
                    pwc,
                    exc,
                )
                entry["shared_species"] = []

            specialist_outputs.append(entry)
            logger.info(
                "specialist_hook: pathway=%s species=%d reactions=%d "
                "crosstalk_reactions=%d validation_rules=%d",
                pwc,
                len(entry["species"]),
                len(entry["reactions"]),
                len(entry["crosstalk_reactions"]),
                len(entry["validation_rules"]),
            )

        if not specialist_outputs:
            logger.info(
                "specialist_hook: 无 Specialist 输出（pathway_class=%s）",
                pathway_class,
            )
            return {}

        # Task B.2: 双写 v4_specialist_outputs → v4_state["specialist"]["outputs"]
        result_update: dict[str, Any] = {}
        set_v4_state(result_update, "specialist", "outputs", specialist_outputs)
        return result_update
    except Exception as exc:
        # 任何失败都不阻塞流水线，记录 warning 并返回空更新
        logger.warning("specialist_hook 失败，降级跳过: %s", exc)
        return {}


__all__ = [
    "specialist_hook_node",
]
