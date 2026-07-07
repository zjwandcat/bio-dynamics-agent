# LEGACY BRIDGE — scheduled for removal in v4.1
# 此模块为 v3↔v4 双向兼容层（v3 network_json → v4 ReactionIRv2）。
# Task B.1 审计结论：确认 runtime 依赖（V4_REACTION_IR_ADAPTER_ENABLED=true 时被调用），
# 但属于 v3/v4 过渡期的 legacy bridge，v4.1 完成 v4 原生流水线后移除。
# 保守策略：RC 阶段保留并标注，不做任何逻辑改动。
# BioDynamics Agent v4 - v3 → v4 Adapter
# 对应 v4 Migration Plan §2.4 的 V3V4Adapter 核心逻辑。
#
# 职责：将 v3 的 network_json 转换为 v4 ReactionIRv2（降级模式）。
#
# 降级模式特性：
# - 无状态机（state_machines 为空）
# - 无组合反应（composite_reactions 为空）
# - compartment 默认 "cytoplasm"（ligand/receptor 按类型推断）
# - ontology.verified=False（未通过 P1 Ontology Agent 验证）
# - mechanism 由 v3 interaction 推断（activation→activation, inhibition→inhibition）
#
# 设计原则：
# 1. 转换失败时记录 warning + 返回 None（触发 fail-safe 降级到 v3 路径）
# 2. 不阻塞主流水线
# 3. 不调用 LLM，纯规则映射
# 4. 复用 reaction_ir_v2.reaction_builder.build_from_network_json

from __future__ import annotations

import logging
from typing import Any

from app.reaction_ir_v2.reaction_builder import build_from_network_json
from app.reaction_ir_v2.schema import ReactionIRv2

logger = logging.getLogger(__name__)


def v3_to_v4(
    network_json: dict[str, Any],
    ontology_entities: dict[str, Any] | None = None,
    pathway_tag: str = "",
    sbml_model_id: str | None = None,
) -> ReactionIRv2 | None:
    """将 v3 network_json 转换为 v4 ReactionIRv2（降级模式）。

    v3 network_json 格式：
        {
            "nodes": [{"id": str, "name": str, "type": str}],
            "edges": [{"source": str, "target": str, "interaction": str}]
        }

    转换规则：
    1. v3 nodes → v4 Species（缺失 compartment 默认 "cytoplasm"）
    2. v3 edges → v4 Reactions（interaction → mechanism 通过 v3_interaction_to_mechanism）
    3. v3 activation → v4 activation 机制（显式禁止强制映射为 phosphorylation）
    4. v3 inhibition → v4 inhibition 机制
    5. 缺失的 compartment 默认填 "cytoplasm"
    6. 转换失败时记录 warning + 返回 None（触发 fail-safe）

    Args:
        network_json: v3 格式的网络 JSON
        ontology_entities: P1 Ontology Agent 输出（可选）
        pathway_tag: 通路标签（可选）
        sbml_model_id: SBML model ID（可选，用于溯源）

    Returns:
        ReactionIRv2 对象，或 None（转换失败时）
    """
    if not network_json:
        logger.warning("v3_to_v4: network_json 为空，返回 None 触发 fail-safe")
        return None

    if not isinstance(network_json, dict):
        logger.warning(
            "v3_to_v4: network_json 类型 %s 非 dict，返回 None 触发 fail-safe",
            type(network_json).__name__,
        )
        return None

    nodes = network_json.get("nodes")
    edges = network_json.get("edges")
    if nodes is None and edges is None:
        logger.warning(
            "v3_to_v4: network_json 缺少 nodes/edges 字段，返回 None 触发 fail-safe"
        )
        return None

    try:
        ir = build_from_network_json(
            network_json=network_json,
            ontology_entities=ontology_entities,
            pathway_tag=pathway_tag,
            sbml_model_id=sbml_model_id,
        )
        # 标记为降级模式（v3→v4 转换的结果）
        ir.source = "v3_downgraded"
        # 补充 warning：降级模式无状态机/组合反应
        if not ir.warnings:
            ir.warnings = []
        ir.warnings.append(
            "v3_to_v4 降级模式：无状态机、无组合反应、ontology 未验证"
        )
        logger.info(
            "v3_to_v4 转换成功：%d species, %d reactions, %d constraints",
            len(ir.species), len(ir.reactions), len(ir.constraints),
        )
        return ir
    except Exception as exc:
        logger.warning(
            "v3_to_v4 转换失败：%s，返回 None 触发 fail-safe", exc, exc_info=True
        )
        return None


__all__ = ["v3_to_v4"]
