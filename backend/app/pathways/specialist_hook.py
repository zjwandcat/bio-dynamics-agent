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
    if not settings.effective_v4_pathway_specialist_enabled():
        logger.debug("V4_PATHWAY_SPECIALIST_ENABLED effective=false，跳过 Specialist")
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

            # 调用 5 个 apply_* 方法（每个方法内部已捕获异常，返回空 list/dict）
            # [v5 Recovery Sprint 2 / RC14] 新增 apply_perturbation 调用
            # 旧实现仅调用 apply_core/apply_feedback/apply_crosstalk/apply_validation，
            # 药物扰动 Reaction 片段（Gefitinib/Trametinib/Rapamycin 等）不进入 specialist_outputs。
            core_output = specialist.apply_core(pathway_graph)
            feedback_loops = specialist.apply_feedback(pathway_graph)
            crosstalk_reactions = specialist.apply_crosstalk(
                pathway_graph, crosstalk_edges
            )
            validation_rules = specialist.apply_validation()
            # RC14: 从 pathway_graph 提取 perturbation_points（若 Planner 预识别则用，否则空列表）
            perturbation_points: list[dict] = []
            if isinstance(pathway_graph, dict):
                perturbation_points = pathway_graph.get("perturbation_points", []) or []
            perturbation_reactions = specialist.apply_perturbation(
                pathway_graph, perturbation_points
            )

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
                # [v5 Recovery Sprint 2 / RC14] 药物扰动 Reaction 片段接入
                "perturbation_reactions": perturbation_reactions,
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
                "crosstalk_reactions=%d validation_rules=%d "
                "perturbation_reactions=%d",
                pwc,
                len(entry["species"]),
                len(entry["reactions"]),
                len(entry["crosstalk_reactions"]),
                len(entry["validation_rules"]),
                len(entry["perturbation_reactions"]),
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

        # [BM2-BM8 修复 / Mode A] Specialist 核心 species/reactions 回写 v3 KG
        # 当 V4_SPECIALIST_KG_WRITEBACK_MODE in (mode_a, both) 时，将 Specialist
        # apply_core() 返回的 species（MDM2/β-catenin/MOMP/STAT3_dimer 等）+ reactions
        # 转换为 KG nodes/edges 注入 knowledge_graph + network_json，使 N6 ODE 生成器
        # 看到完整通路拓扑（而非仅 LLM 稀疏识别的 4n/1e）
        #
        # [RC21] 修复：调换执行顺序——core 先于 feedback。
        #   原顺序：feedback 先 → 创建 EGFR→pEGFR (mechanism=negative_feedback)，
        #   core 后 → 发现 (EGFR,pEGFR) 已存在而跳过，导致 specialist 的 phosphorylation
        #   机制被 feedback 的 negative_feedback 覆盖，ODE 模板无此分支→pEGFR=0。
        #   新顺序：core 先注入正确的 mechanism（phosphorylation/binding 等），
        #   feedback 后注入闭环边（DUSP→pERK 等），跳过已存在的 (source,target) 对。
        core_kg_writeback = _apply_specialist_core_kg_writeback(
            state, specialist_outputs
        )
        if core_kg_writeback:
            result_update.update(core_kg_writeback)

        # [P1-4] Specialist feedback_loops 回写 v3 KG（受控突破不可碰清单）
        # 当 V4_SPECIALIST_KG_FEEDBACK_ENABLED=true 时，将 DUSP/MDM2/IkBa/Axin2 等
        # 负反馈环注入 knowledge_graph + network_json，使 N6 ODE 生成器可见
        # [RC21] 基于已合并 core 的 state 运行，跳过 core 已注入的 (source,target) 对
        merged_state_for_feedback = {**state, **result_update}
        kg_writeback = _apply_specialist_kg_writeback(merged_state_for_feedback, specialist_outputs)
        if kg_writeback:
            result_update.update(kg_writeback)

        return result_update
    except Exception as exc:
        # 任何失败都不阻塞流水线，记录 warning 并返回空更新
        logger.warning("specialist_hook 失败，降级跳过: %s", exc)
        return {}


# =============================================================================
# [P1-4] Specialist feedback_loops → v3 KG 回写适配器
# =============================================================================
# 设计目的：突破"不可碰清单"铁律的受控开口。当 V4_SPECIALIST_KG_FEEDBACK_ENABLED=true
# 时，将 Specialist 的 feedback_loops（DUSP-ERK / MDM2-p53 / IkBa-NFkB / Axin2-βCat
# 等延迟负反馈环）转换为 KG edges 注入 v3 knowledge_graph + network_json，
# 使 LLM ODE 生成器（N6）能看到这些关键负反馈环，从而生成带衰减/振荡的 ODE。
#
# 回写规则：
# 1. 每个 feedback_loop 的 node_ids 转为顺序 activation edges（信号传播链）
# 2. 额外添加 1 条 inhibition edge：effector → source（反馈调控边）
#    - effector = node_ids[len//2]（链中间节点，通常是执行反馈的蛋白）
#    - source = node_ids[0]（反馈环起点）
#    - loop_type=negative → interaction=inhibition
#    - loop_type=positive → interaction=activation
# 3. 所有反馈边标注 feedback_loop_id / delay_minutes / loop_type（供 DDE 模板识别）
# 4. node_ids 中的新物种加入 KG nodes（去重）
#
# 已知限制：feedback_loops 使用规范名（pERK/ppERK），而 LLM NER 可能产出
# phosphorylated_ERK 等变体名，存在名称不匹配风险（由 P1-5 NER 规范化解决）。
def _infer_node_type(name: str) -> str:
    """根据节点名推断类型（mRNA / Protein / Complex）。"""
    n = (name or "").lower()
    if "mrna" in n or "_rna" in n:
        return "mRNA"
    if "complex" in n or "_" in n and any(x in n for x in ["egf_egfr"]):
        return "Complex"
    return "Protein"


def _feedback_loops_to_kg_updates(
    feedback_loops: list[dict[str, Any]],
    existing_nodes: list[dict[str, Any]],
    existing_edges: list[dict[str, Any]],
) -> dict[str, Any]:
    """将 feedback_loops 转换为 KG 增量 nodes/edges（已去重）。

    Args:
        feedback_loops: Specialist apply_feedback() 返回的反馈环列表
        existing_nodes: 当前 KG nodes（用于去重）
        existing_edges: 当前 KG edges（用于去重）

    Returns:
        {"nodes": [...新增节点...], "edges": [...新增边...]}
    """
    existing_node_names: set[str] = set()
    for n in existing_nodes:
        existing_node_names.add(
            n.get("name") or n.get("id") or n.get("entity_id") or ""
        )
    existing_edge_keys: set[tuple[str, str]] = set()
    for e in existing_edges:
        existing_edge_keys.add((e.get("source", ""), e.get("target", "")))

    new_nodes: list[dict[str, Any]] = []
    new_edges: list[dict[str, Any]] = []
    added_node_names: set[str] = set()
    added_edge_keys: set[tuple[str, str]] = set()

    for loop in feedback_loops:
        if not isinstance(loop, dict):
            continue
        node_ids = loop.get("node_ids", []) or []
        if len(node_ids) < 2:
            continue
        loop_type = loop.get("loop_type", "negative")
        loop_id = loop.get("id", "")
        delay = loop.get("delay_minutes", 0.0)
        description = loop.get("description", "")
        closing_interaction = "inhibition" if loop_type == "negative" else "activation"

        # 1. 添加新物种节点（去重）
        for nid in node_ids:
            if not nid or nid in existing_node_names or nid in added_node_names:
                continue
            added_node_names.add(nid)
            new_nodes.append({
                "name": nid,
                "id": nid,
                "entity_id": nid,
                "type": _infer_node_type(nid),
            })

        # 2. 顺序 edges（信号传播链 A→B→C→D）
        #    effector 之前的边：activation（信号传播至效应蛋白）
        #    effector → 下游 source 变体：inhibition（负反馈调控，如 DUSP 去磷酸化 pERK）
        #    effector 之后的边：activation（前向级联，如 pERK→ppERK 磷酸化）
        effector_idx = len(node_ids) // 2
        for i in range(len(node_ids) - 1):
            src, tgt = node_ids[i], node_ids[i + 1]
            ek = (src, tgt)
            if ek in existing_edge_keys or ek in added_edge_keys:
                continue
            added_edge_keys.add(ek)
            # 负反馈环中 effector→next 边标记为 inhibition（效应蛋白抑制 source 变体）
            is_effector_edge = (i == effector_idx and loop_type == "negative")
            edge_interaction = "inhibition" if is_effector_edge else "activation"
            arrow = "-|" if is_effector_edge else "->"
            new_edges.append({
                "source": src,
                "target": tgt,
                "interaction": edge_interaction,
                "mechanism": "feedback_regulation" if is_effector_edge else "feedback_propagation",
                "reaction_equation": f"{src} {arrow} {tgt}",
                "feedback_loop_id": loop_id,
                "delay_minutes": delay,
                "loop_type": loop_type,
            })

        # 3. 反馈调控边：effector → source（闭环）
        #    effector = node_ids[len//2]（链中间节点，通常是 DUSP/MDM2/IkBa/Axin2）
        effector = node_ids[effector_idx] if effector_idx < len(node_ids) else node_ids[-1]
        source = node_ids[0]
        ek = (effector, source)
        if ek not in existing_edge_keys and ek not in added_edge_keys:
            added_edge_keys.add(ek)
            arrow = "-|" if loop_type == "negative" else "->"
            new_edges.append({
                "source": effector,
                "target": source,
                "interaction": closing_interaction,
                "mechanism": f"{loop_type}_feedback",
                "reaction_equation": f"{effector} {arrow} {source}",
                "feedback_loop_id": loop_id,
                "delay_minutes": delay,
                "loop_type": loop_type,
                "description": description,
            })

    return {"nodes": new_nodes, "edges": new_edges}


def _apply_specialist_kg_writeback(
    state: dict[str, Any],
    specialist_outputs: list[dict[str, Any]],
) -> dict[str, Any]:
    """[P1-4] 将 specialist feedback_loops 回写到 v3 KG + network_json。

    Args:
        state: 当前 LangGraph state
        specialist_outputs: specialist_hook 收集的输出列表

    Returns:
        包含更新后 knowledge_graph / network_json 的 dict（若无增量则空 dict）
    """
    from app.config import settings

    if not settings.effective_v4_specialist_kg_feedback_enabled():
        return {}

    # 收集所有 feedback_loops
    all_feedback_loops: list[dict[str, Any]] = []
    for entry in specialist_outputs:
        loops = entry.get("feedback_loops", []) or []
        all_feedback_loops.extend(loops)
    if not all_feedback_loops:
        return {}

    # 读取现有 KG
    kg = state.get("knowledge_graph", {}) or {}
    kg_nodes = kg.get("nodes", []) or []
    kg_edges = kg.get("edges", []) or []
    nj = state.get("network_json", {}) or {}
    nj_nodes = nj.get("nodes", []) or []
    nj_edges = nj.get("edges", []) or []

    updates = _feedback_loops_to_kg_updates(all_feedback_loops, kg_nodes, kg_edges)
    new_nodes = updates["nodes"]
    new_edges = updates["edges"]
    if not new_nodes and not new_edges:
        return {}

    logger.info(
        "[P1-4] Specialist KG 回写: +%d nodes, +%d edges (feedback_loops=%d)",
        len(new_nodes), len(new_edges), len(all_feedback_loops),
    )

    # 合并到 knowledge_graph
    merged_kg = dict(kg)
    merged_kg["nodes"] = list(kg_nodes) + new_nodes
    merged_kg["edges"] = list(kg_edges) + new_edges
    merged_kg["node_count"] = len(merged_kg["nodes"])
    merged_kg["edge_count"] = len(merged_kg["edges"])

    # 合并到 network_json（N6 回退路径使用）
    merged_nj = dict(nj)
    merged_nj["nodes"] = list(nj_nodes) + [
        {"id": n.get("name") or n.get("id"), "name": n.get("name", ""), "type": n.get("type", "")}
        for n in new_nodes
    ]
    merged_nj["edges"] = list(nj_edges) + [
        {"source": e.get("source", ""), "target": e.get("target", ""), "interaction": e.get("interaction") or e.get("mechanism") or "activation"}
        for e in new_edges
    ]

    return {
        "knowledge_graph": merged_kg,
        "network_json": merged_nj,
    }


# =============================================================================
# [BM2-BM8 修复 / Mode A] Specialist 核心 species/reactions → v3 KG 回写
# =============================================================================
# 设计目的：将 Specialist apply_core() 返回的丰富通路拓扑（p53 Specialist 的
# 15 species + 12 reactions，含 MDM2 转录/翻译/泛素化等关键负反馈环节）注入
# v3 knowledge_graph + network_json，使 N6 ODE 生成器看到完整通路拓扑，
# 而非仅 LLM NER 产出的稀疏 4n/1e KG。
#
# 与 _apply_specialist_kg_writeback（feedback_loops 回写）的区别：
# - feedback_loops 回写：仅注入延迟负反馈环（DUSP/MDM2/IkBa/Axin2 等少数节点）
# - core 回写：注入 Specialist 完整 species + reactions（MDM2_mRNA/MDM2/p53_ubi/
#   p21_mRNA/p21/p300/CBP/p53_ac/MDM4 等所有核心物种 + 所有反应边）
#
# 回写规则：
# 1. species → KG nodes（name/aliases/species_type/compartment）
# 2. reactions → KG edges（source/target/mechanism/kinetics_type/reaction_equation）
# 3. 按 name 去重（与现有 KG nodes 合并，避免重复）
# 4. 按 (source, target) 去重 edges
# 5. interaction 字段：activation 边默认 activation；inhibition 类 mechanism → inhibition
#
# 已知限制：core species 使用规范名（如 Mdm2_mRNA），LLM NER 可能产出变体名
# （如 MDM2 mRNA），存在名称不匹配风险。但 Specialist 的 species_name 是通路
# 标准命名，作为 ODE 变量名是合理的。
def _infer_edge_interaction(mechanism: str) -> str:
    """根据反应机制推断 edge interaction 字段。

    Args:
        mechanism: 反应机制（phosphorylation/binding/inhibition/...）

    Returns:
        "activation" 或 "inhibition"
    """
    if not mechanism:
        return "activation"
    m = mechanism.lower()
    # 抑制类机制
    if m in ("inhibition", "ubiquitination", "proteasomal_degradation",
             "dephosphorylation", "degradation"):
        return "inhibition"
    # 其他机制默认 activation（phosphorylation/binding/exchange/...）
    return "activation"


def _specialist_core_to_kg_updates(
    specialist_outputs: list[dict[str, Any]],
    existing_nodes: list[dict[str, Any]],
    existing_edges: list[dict[str, Any]],
) -> dict[str, Any]:
    """将 Specialist core species + reactions 转换为 KG 增量 nodes/edges（已去重）。

    Args:
        specialist_outputs: specialist_hook_node 收集的输出列表（含 species + reactions）
        existing_nodes: 当前 KG nodes（用于去重）
        existing_edges: 当前 KG edges（用于去重）

    Returns:
        {"nodes": [...新增节点...], "edges": [...新增边...]}
    """
    # 收集现有节点名（小写比较，避免大小写差异导致重复）
    existing_node_names: set[str] = set()
    for n in existing_nodes:
        name = (n.get("name") or n.get("id") or n.get("entity_id") or "").lower()
        if name:
            existing_node_names.add(name)

    # [BENCHMARK CLOSURE / Gap-EGFR-PeakTime] 收集现有边的 (src, tgt) → mechanism 映射。
    #   旧代码仅收集 edge_keys 集合，当 LLM 已生成 (EGFR, pEGFR, activation) 时，
    #   specialist 的 (EGFR, pEGFR, phosphorylation) 因 pair 已存在被跳过，
    #   导致 pEGFR 由 activation 分支（k_act=0.05，慢）而非 phosphorylation 分支
    #   （k_cat=2.0，快）产生，pEGFR 达峰时间 15-19 min（期望 5-10 min）。
    #   修复：同时记录 mechanism，当 specialist 的 mechanism 更具体（非 activation）
    #   而现有边为 activation 时，REPLACE 现有边而非跳过。
    existing_edge_keys: set[tuple[str, str]] = set()
    existing_edge_mechanisms: dict[tuple[str, str], str] = {}
    for e in existing_edges:
        src = (e.get("source") or "").lower()
        tgt = (e.get("target") or "").lower()
        if src and tgt:
            existing_edge_keys.add((src, tgt))
            mech = (e.get("mechanism") or e.get("interaction") or "activation").lower()
            if (src, tgt) not in existing_edge_mechanisms:
                existing_edge_mechanisms[(src, tgt)] = mech

    new_nodes: list[dict[str, Any]] = []
    new_edges: list[dict[str, Any]] = []
    added_node_names: set[str] = set()
    added_edge_keys: set[tuple[str, str]] = set()
    # [BENCHMARK CLOSURE / Gap-EGFR-PeakTime] 被替换的 (src, tgt) 对列表，
    #   供 caller 过滤掉现有 activation 边（避免 ODE 模板看到两条同 pair 边）。
    replaced_edge_keys: list[tuple[str, str]] = []

    for entry in specialist_outputs:
        if not isinstance(entry, dict):
            continue
        pathway_class = entry.get("pathway_class", "")
        species_list = entry.get("species", []) or []
        reactions_list = entry.get("reactions", []) or []

        # 1. species → KG nodes
        for sp in species_list:
            if not isinstance(sp, dict):
                continue
            name = sp.get("name", "")
            if not name:
                continue
            name_lower = name.lower()
            if name_lower in existing_node_names or name_lower in added_node_names:
                continue
            added_node_names.add(name_lower)
            species_type = sp.get("species_type", "Protein")
            compartment = sp.get("compartment", "cytoplasm")
            # 推断 type 字段（与 v3 KG node schema 对齐）
            # [N6 缺口 1] 新增 "drug" → "Drug" 映射：含药物抑制的 case 必须使
            #   Drug 节点出现在 network_json，触发 _extract_drug_candidates_fallback
            #   → drug_candidates → node1_6_pkpd_inference → pkpd_profile（非空）。
            #   canonical drug_library 驱动（specialist 不再硬编码单 case 药物）。
            if isinstance(species_type, str):
                st = species_type.lower()
                if "mrna" in st or "_rna" in st:
                    node_type = "mRNA"
                elif "complex" in st:
                    node_type = "Complex"
                elif "damage" in st:
                    node_type = "Damage"
                elif st == "drug":
                    node_type = "Drug"
                else:
                    node_type = "Protein"
            else:
                node_type = "Protein"
            new_nodes.append({
                "name": name,
                "id": name,
                "entity_id": name,
                "type": node_type,
                "species_type": species_type,
                "compartment": compartment,
                "pathway_tag": pathway_class,
                "_source": "specialist_core",
                # [IC-Pipeline Fix 1] 透传 specialist 的 initial_concentration 到 KG node。
                #   旧 bug：仅复制 8 个字段，遗漏 IC，导致下游 SpeciesV2.initial_concentration=0.0，
                #   _extract_y0 回退到 _default_initial_concentration 使 EGF=1.0（应 680.0）。
                "initial_concentration": sp.get("initial_concentration", 0.0),
            })

        # 2. reactions → KG edges
        for rxn in reactions_list:
            if not isinstance(rxn, dict):
                continue
            # 优先使用 source/target（Reaction IR 字段）
            src = rxn.get("source", "")
            tgt = rxn.get("target", "")
            if not src or not tgt:
                # 回退到 substrate/product（部分反应用 substrate/product 字段）
                src = rxn.get("substrate", "")
                tgt = rxn.get("product", "")
            if not src or not tgt:
                continue
            ek_lower = (src.lower(), tgt.lower())
            mechanism = (rxn.get("mechanism") or "activation").lower()

            # [BENCHMARK CLOSURE / Gap-EGFR-PeakTime] 边替换策略：
            #   当 (src, tgt) 已存在且现有 mechanism 为 activation（LLM 生成的泛化机制），
            #   而 specialist 提供 more specific 的 mechanism（phosphorylation/binding/
            #   gtp_gdp_exchange/degradation 等），REPLACE 现有 activation 边而非跳过。
            #   原因：activation 分支 k_act=0.05（慢），phosphorylation 分支 k_cat=2.0（快），
            #   不替换会导致 pEGFR 达峰时间 19.26min（期望 5-10min）。
            existing_mech = existing_edge_mechanisms.get(ek_lower, "")
            should_replace_existing = (
                ek_lower in existing_edge_keys
                and existing_mech == "activation"
                and mechanism != "activation"
            )

            if should_replace_existing:
                # 记录被替换的 (src, tgt) 对，caller 据此过滤现有 activation 边
                if ek_lower not in replaced_edge_keys:
                    replaced_edge_keys.append(ek_lower)
                    logger.info(
                        "[BENCHMARK CLOSURE / Gap-EGFR-PeakTime] 边替换: "
                        "(%s, %s) activation → %s（specialist 提供更具体机制）",
                        src, tgt, mechanism,
                    )
                # 允许新边加入（不 continue），不加入 added_edge_keys 重复检查
                # 因为同一 specialist 内可能有同 (src, tgt) 多次出现，仍需去重
                if ek_lower in added_edge_keys:
                    continue
                added_edge_keys.add(ek_lower)
            elif ek_lower in existing_edge_keys or ek_lower in added_edge_keys:
                # 既有同 pair 边且非 replace 场景，跳过
                continue
            else:
                added_edge_keys.add(ek_lower)

            interaction = _infer_edge_interaction(mechanism)
            kinetics_type = rxn.get("kinetics_type", "")
            reaction_equation = rxn.get("reaction_equation", "") or f"{src} {mechanism} {tgt}"
            description = rxn.get("description", "")
            new_edges.append({
                "source": src,
                "target": tgt,
                "interaction": interaction,
                "mechanism": mechanism,
                "kinetics_type": kinetics_type,
                "reaction_equation": reaction_equation,
                "pathway_tag": pathway_class,
                "description": description,
                # [RC24] 修复：保留 substrate/modifier 字段供 ODE 模板正确使用
                # 原始代码丢弃了 specialist 反应中的 substrate/modifier，导致
                # 磷酸化分支错误地用 source(酶) 作底物并消耗酶。
                "substrate": rxn.get("substrate", ""),
                "modifier": rxn.get("modifier", "") or "",
                "_source": "specialist_core",
            })

    return {
        "nodes": new_nodes,
        "edges": new_edges,
        # [BENCHMARK CLOSURE / Gap-EGFR-PeakTime] 返回被替换的 (src, tgt) 对列表，
        # 供 caller 过滤掉现有 activation 边（避免 ODE 模板看到两条同 pair 边）。
        "replaced_edge_keys": replaced_edge_keys,
    }


def _apply_specialist_core_kg_writeback(
    state: dict[str, Any],
    specialist_outputs: list[dict[str, Any]],
) -> dict[str, Any]:
    """[BM2-BM8 修复 / Mode A] 将 Specialist core species/reactions 回写到 v3 KG + network_json。

    当 V4_SPECIALIST_KG_WRITEBACK_MODE in (mode_a, both) 时执行。
    将 Specialist apply_core() 输出的丰富通路拓扑（p53 的 15 species + 12 reactions
    等）注入 knowledge_graph + network_json，使 N6 ODE 生成器看到完整通路拓扑。

    Args:
        state: 当前 LangGraph state
        specialist_outputs: specialist_hook 收集的输出列表

    Returns:
        包含更新后 knowledge_graph / network_json 的 dict（若无增量则空 dict）
    """
    from app.config import settings

    # 仅在 mode_a 或 both 模式下执行
    if not settings.specialist_writeback_mode_a_enabled():
        return {}

    # 收集所有 species + reactions
    has_core_data = any(
        isinstance(e, dict) and (e.get("species") or e.get("reactions"))
        for e in specialist_outputs
    )
    if not has_core_data:
        return {}

    # 读取现有 KG
    kg = state.get("knowledge_graph", {}) or {}
    kg_nodes = kg.get("nodes", []) or []
    kg_edges = kg.get("edges", []) or []
    nj = state.get("network_json", {}) or {}
    nj_nodes = nj.get("nodes", []) or []
    nj_edges = nj.get("edges", []) or []

    updates = _specialist_core_to_kg_updates(specialist_outputs, kg_nodes, kg_edges)
    new_nodes = updates["nodes"]
    new_edges = updates["edges"]
    # [BENCHMARK CLOSURE / Gap-EGFR-PeakTime] 获取被替换的 (src, tgt) 对，
    #   过滤掉现有 KG 中的 activation 边（避免 ODE 模板看到两条同 pair 边）。
    replaced_edge_keys: set[tuple[str, str]] = set(updates.get("replaced_edge_keys", []))
    if not new_nodes and not new_edges:
        return {}

    logger.info(
        "[Mode A] Specialist core KG 回写: +%d nodes, +%d edges, "
        "替换 %d 条 activation 边 (pathways=%s)",
        len(new_nodes), len(new_edges), len(replaced_edge_keys),
        [e.get("pathway_class") for e in specialist_outputs if isinstance(e, dict)],
    )

    # [BENCHMARK CLOSURE / Gap-EGFR-PeakTime] 过滤掉被替换的 activation 边
    #   原因：当 specialist 提供 (EGFR, pEGFR, phosphorylation) 替换现有
    #   (EGFR, pEGFR, activation) 时，若不过滤，ODE 模板会同时看到两条同 (src, tgt)
    #   的边，phosphorylation 分支与 activation 分支都会执行，导致 pEGFR 双重生成。
    def _is_replaced_edge(e: dict[str, Any]) -> bool:
        src = (e.get("source") or "").lower()
        tgt = (e.get("target") or "").lower()
        return (src, tgt) in replaced_edge_keys

    filtered_kg_edges = [e for e in kg_edges if not _is_replaced_edge(e)]
    filtered_nj_edges = [e for e in nj_edges if not _is_replaced_edge(e)]

    # 合并到 knowledge_graph
    merged_kg = dict(kg)
    merged_kg["nodes"] = list(kg_nodes) + new_nodes
    merged_kg["edges"] = filtered_kg_edges + new_edges
    merged_kg["node_count"] = len(merged_kg["nodes"])
    merged_kg["edge_count"] = len(merged_kg["edges"])

    # 合并到 network_json（N6 回退路径使用）
    merged_nj = dict(nj)
    merged_nj["nodes"] = list(nj_nodes) + [
        {
            "id": n.get("name") or n.get("id"),
            "name": n.get("name", ""),
            "type": n.get("type", ""),
            "species_type": n.get("species_type", ""),
            "compartment": n.get("compartment", "cytoplasm"),
        }
        for n in new_nodes
    ]
    merged_nj["edges"] = filtered_nj_edges + [
        {
            "source": e.get("source", ""),
            "target": e.get("target", ""),
            "interaction": e.get("interaction") or e.get("mechanism") or "activation",
            "mechanism": e.get("mechanism", ""),
            "kinetics_type": e.get("kinetics_type", ""),
            "reaction_equation": e.get("reaction_equation", ""),
            # [BENCHMARK CLOSURE / Gap-EGFR-PeakOrder-RasGTP] 修复：保留 substrate/
            #   modifier 字段供 reaction_builder 与 ODE 模板正确使用真实酶（如 SOS
            #   作为 GEF 催化 RasGDP→RasGTP），而非退化为 source 占位符（RasGDP
            #   自催化），导致 RasGTP 达峰时间 29.3min（期望 2-8min）。
            #   原代码丢失这两个字段，使 reaction_builder L472 的 edge_modifier 恒为
            #   空字符串，触发 L476 降级分支用 source 作 placeholder modifier。
            "substrate": e.get("substrate", ""),
            "modifier": e.get("modifier", "") or "",
        }
        for e in new_edges
    ]

    return {
        "knowledge_graph": merged_kg,
        "network_json": merged_nj,
    }


__all__ = [
    "specialist_hook_node",
]
