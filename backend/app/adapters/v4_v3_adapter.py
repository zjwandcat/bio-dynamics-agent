# BioDynamics Agent v4 - v4 → v3 Adapter
# 对应 v4 Migration Plan §2.4 的 V4V3Adapter 核心逻辑。
#
# 职责：将 v4 ReactionIRv2 转换为 v3 network_json 格式，供旧模板渲染。
#
# 兼容性策略：
# - 忽略 v4 特有字段（state_machine, composite_reaction, modifiers, constraints）
# - 将 v4 Species → v3 nodes（保留 id/name/type）
# - 将 v4 Reactions → v3 edges（保留 source/target/interaction）
# - reaction_type → interaction 的映射（phosphorylation → activation 等降级映射）
# - 保证转换后的 JSON 能被 v3 的 ode_templates/ 正常渲染
#
# 设计原则：
# 1. 转换失败时记录 warning + 返回 None（触发 fail-safe 降级到 v3 原始路径）
# 2. 不阻塞主流水线
# 3. 不调用 LLM，纯规则映射

from __future__ import annotations

import logging
from typing import Any

from app.reaction_ir_v2.mechanism_types import MechanismType
from app.reaction_ir_v2.schema import ReactionIRv2

logger = logging.getLogger(__name__)


# =============================================================================
# v4 mechanism → v3 interaction 映射
# =============================================================================
# v3 network_json 的 interaction 字段仅接受少量值（activation/inhibition 等），
# 需要将 v4 的 17 类机制降级映射。
_V4_MECHANISM_TO_V3_INTERACTION: dict[str, str] = {
    # 调控类直接映射
    MechanismType.ACTIVATION.value: "activation",
    MechanismType.INHIBITION.value: "inhibition",
    # 修饰类映射到 activation（激活下游）
    MechanismType.PHOSPHORYLATION.value: "activation",  # 磷酸化通常激活下游
    MechanismType.DEPHOSPHORYLATION.value: "inhibition",  # 去磷酸化通常抑制
    MechanismType.UBIQUITINATION.value: "inhibition",  # 泛素化通常导致降解=抑制
    # 结合/组装类映射到 binding
    MechanismType.BINDING.value: "binding",
    MechanismType.DIMERIZATION.value: "binding",
    MechanismType.COMPLEX_FORMATION.value: "binding",
    MechanismType.SEQUESTRATION.value: "inhibition",
    MechanismType.DISSOCIATION.value: "dissociation",
    # 切割/交换类
    MechanismType.CLEAVAGE.value: "activation",  # Caspase 切割激活下游
    MechanismType.GTP_GDP_EXCHANGE.value: "activation",  # GTP 交换激活
    # 基因表达类
    MechanismType.TRANSCRIPTION.value: "transcription",
    MechanismType.TRANSLATION.value: "translation",
    # 转运类
    MechanismType.NUCLEAR_IMPORT.value: "transport",
    MechanismType.NUCLEAR_EXPORT.value: "transport",
    MechanismType.CYTOPLASM_TRANSLOCATION.value: "transport",
    # 降解类
    MechanismType.DEGRADATION.value: "degradation",
    MechanismType.PROTEASOMAL_DEGRADATION.value: "degradation",
}


def v4_to_v3(reaction_ir: ReactionIRv2) -> dict[str, Any] | None:
    """将 v4 ReactionIRv2 转换为 v3 network_json 格式。

    转换规则：
    1. v4 Species → v3 nodes：保留 id/name/type
    2. v4 Reactions → v3 edges：
       - 从 reactants[0] 取 source
       - 从 products[0] 取 target（若 products 为空，取 modifiers 中的 inhibitor）
       - reaction_type → interaction 通过映射表降级
    3. 忽略 v4 特有字段：state_machine, composite_reaction, modifiers, constraints
    4. 保证转换后的 JSON 能被 v3 的 ode_templates/ 正常渲染

    Args:
        reaction_ir: ReactionIRv2 对象

    Returns:
        v3 network_json dict，或 None（转换失败时）
    """
    if reaction_ir is None:
        logger.warning("v4_to_v3: reaction_ir 为 None，返回 None 触发 fail-safe")
        return None

    if not isinstance(reaction_ir, ReactionIRv2):
        logger.warning(
            "v4_to_v3: reaction_ir 类型 %s 非 ReactionIRv2，返回 None 触发 fail-safe",
            type(reaction_ir).__name__,
        )
        return None

    try:
        # —— 1. Species → nodes ——
        nodes: list[dict[str, Any]] = []
        species_id_to_name: dict[str, str] = {}
        for sp in reaction_ir.species:
            name = sp.canonical_name or sp.id
            species_id_to_name[sp.id] = name
            nodes.append({
                "id": name,  # v3 用 name 作为 id（与 graph_v3.py _normalize_network_json 一致）
                "name": name,
                "type": sp.species_type,
            })

        # —— 2. Reactions → edges ——
        edges: list[dict[str, Any]] = []
        for rxn in reaction_ir.reactions:
            # source：reactants 的第一个 substrate
            source_name = _extract_source_name(rxn, species_id_to_name)
            # target：products 的第一个 product，或 modifiers 中的 inhibitor target
            target_name = _extract_target_name(rxn, species_id_to_name)

            if not source_name and not target_name:
                # 无法提取 source/target，跳过此反应
                logger.warning(
                    "v4_to_v3: 反应 %s 无法提取 source/target，跳过", rxn.id
                )
                continue

            # interaction：mechanism → v3 interaction
            interaction = _V4_MECHANISM_TO_V3_INTERACTION.get(
                rxn.reaction_type, "activation"  # 默认 activation
            )

            edges.append({
                "source": source_name,
                "target": target_name,
                "interaction": interaction,
            })

        network_json = {"nodes": nodes, "edges": edges}
        logger.info(
            "v4_to_v3 转换成功：%d nodes, %d edges",
            len(nodes), len(edges),
        )
        return network_json
    except Exception as exc:
        logger.warning(
            "v4_to_v3 转换失败：%s，返回 None 触发 fail-safe", exc, exc_info=True
        )
        return None


# =============================================================================
# 辅助：从 Reaction 提取 source / target 名称
# =============================================================================
def _extract_source_name(
    rxn: Any,
    species_id_to_name: dict[str, str],
) -> str:
    """从反应中提取 source 物种名。

    优先级：
    1. reactants 中第一个 role=substrate 的 species
    2. reactants 中第一个 species
    3. modifiers 中第一个 modifier_type=inhibitory 的 species（inhibition 情况）
    """
    # 1. reactants 中的 substrate
    for ref in rxn.reactants:
        if ref.role == "substrate":
            return species_id_to_name.get(ref.species_id, ref.species_id)
    # 2. reactants 中任意
    if rxn.reactants:
        ref = rxn.reactants[0]
        return species_id_to_name.get(ref.species_id, ref.species_id)
    # 3. modifiers 中的 inhibitor
    for mod in rxn.modifiers:
        if mod.modifier_type == "inhibitory":
            return species_id_to_name.get(mod.species_id, mod.species_id)
    return ""


def _extract_target_name(
    rxn: Any,
    species_id_to_name: dict[str, str],
) -> str:
    """从反应中提取 target 物种名。

    优先级：
    1. products 中第一个 role=product 的 species
    2. products 中第一个 species
    3. 若 products 为空，取 reactants 中的 substrate（如降解反应 A → ∅）
    4. modifiers 中的 inhibitor target（即被抑制的 substrate）
    """
    # 1. products 中的 product
    for ref in rxn.products:
        if ref.role == "product":
            return species_id_to_name.get(ref.species_id, ref.species_id)
    # 2. products 中任意
    if rxn.products:
        ref = rxn.products[0]
        return species_id_to_name.get(ref.species_id, ref.species_id)
    # 3. 降解类反应（products 为空）：target = source（substrate）
    if rxn.reactants:
        for ref in rxn.reactants:
            if ref.role == "substrate":
                return species_id_to_name.get(ref.species_id, ref.species_id)
    # 4. inhibition 的 target
    for mod in rxn.modifiers:
        if mod.modifier_type == "inhibitory":
            # inhibitor 是 modifier，被抑制的 target 在 reactants 中
            for ref in rxn.reactants:
                if ref.role == "substrate":
                    return species_id_to_name.get(ref.species_id, ref.species_id)
    return ""


__all__ = ["v4_to_v3"]
