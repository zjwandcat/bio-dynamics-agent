# BioDynamics Agent v4 - 组合反应辅助逻辑
# 对应 v4 Scientific Architecture Part 4 §4.2.3 + §4.7（Wnt destruction complex 示例）。
#
# 设计原则：
# 1. 组合反应用于表达多步耦合的复杂生物学结构（如 destruction complex 三步耦合）
# 2. 组合反应的 sub_reactions 必须有序（sequential 类型）
# 3. 组合反应不压扁为单一 reaction，保留中间产物与子反应语义
# 4. 净反应方程（net_reaction）描述组合反应的整体效果

from __future__ import annotations

from typing import Any

from app.reaction_ir_v2.schema import (
    CompositeReaction,
    ReactionIRv2,
    ReactionV2,
    SpeciesRef,
)
from app.reaction_ir_v2.mechanism_types import MechanismType


# =============================================================================
# 组合反应构建器
# =============================================================================
class CompositeReactionBuilder:
    """组合反应构建辅助器。

    用法：
        builder = CompositeReactionBuilder("CR_WNT_DESTRUCTION", "Wnt destruction complex")
        builder.set_coupling("sequential")
        builder.add_sub_reaction("RXN_DC_FORMATION")
        builder.add_sub_reaction("RXN_DC_BCAT_BINDING")
        builder.add_intermediate("SP_DC_BC3ATENIN")
        builder.set_net_reaction("β-catenin → ∅")
        cr = builder.build()
    """

    def __init__(self, cr_id: str, name: str) -> None:
        self.cr_id = cr_id
        self.name = name
        self._sub_reactions: list[str] = []
        self._coupling_type: str = "sequential"
        self._intermediates: list[str] = []
        self._net_reaction: str = ""

    def set_coupling(self, coupling: str) -> "CompositeReactionBuilder":
        """设置耦合类型：sequential / branched / cyclic。"""
        self._coupling_type = coupling
        return self

    def add_sub_reaction(self, reaction_id: str) -> "CompositeReactionBuilder":
        """添加子反应（按顺序追加）。"""
        self._sub_reactions.append(reaction_id)
        return self

    def add_intermediate(self, species_id: str) -> "CompositeReactionBuilder":
        """添加中间产物 species_id。"""
        self._intermediates.append(species_id)
        return self

    def set_net_reaction(self, net: str) -> "CompositeReactionBuilder":
        """设置净反应方程。"""
        self._net_reaction = net
        return self

    def build(self) -> CompositeReaction:
        """构建 CompositeReaction 对象。"""
        return CompositeReaction(
            id=self.cr_id,
            name=self.name,
            sub_reactions=list(self._sub_reactions),
            coupling_type=self._coupling_type,
            intermediate_species=list(self._intermediates),
            net_reaction=self._net_reaction,
        )


# =============================================================================
# Wnt Destruction Complex 示例（架构 §4.2.3）
# =============================================================================
def build_wnt_destruction_complex_reactions(
    axin_id: str = "SP_AXIN",
    apc_id: str = "SP_APC",
    gsk3b_id: str = "SP_GSK3B",
    ck1_id: str = "SP_CK1",
    dc_id: str = "SP_DESTRUCTION_COMPLEX",
    bcat_id: str = "SP_BCATENIN",
    dc_bcat_id: str = "SP_DC_BCATENIN",
    p_bcat_id: str = "SP_pBCATENIN",
    ub_bcat_id: str = "SP_UbBCATENIN",
    btrcp_id: str = "SP_BTRCP",
    pathway_tag: str = "WNT",
) -> tuple[list[ReactionV2], CompositeReaction]:
    """构建 Wnt destruction complex 的 5 步耦合反应（架构 §4.2.3 示例）。

    五步反应：
      1. complex_formation: Axin + APC + GSK3β + CK1 → destruction complex
      2. binding: destruction complex + β-catenin → DC-β-catenin
      3. phosphorylation: DC-β-catenin → destruction complex + p-β-catenin（GSK3β 为 enzyme）
      4. ubiquitination: p-β-catenin + β-TrCP → p-β-catenin-Ub
      5. proteasomal_degradation: p-β-catenin-Ub → ∅

    净反应：β-catenin → ∅（依赖 Wnt off 状态）

    Args:
        各 species_id 参数可定制，默认值对应标准命名

    Returns:
        (sub_reactions 列表, CompositeReaction 对象)
    """
    reactions: list[ReactionV2] = []

    # Step 1: destruction complex 组装
    r1 = ReactionV2(
        id="RXN_DC_FORMATION",
        reaction_type=MechanismType.COMPLEX_FORMATION.value,
        kinetics_type="mass_action",
        reactants=[
            SpeciesRef(species_id=axin_id, role="substrate"),
            SpeciesRef(species_id=apc_id, role="substrate"),
            SpeciesRef(species_id=gsk3b_id, role="substrate"),
            SpeciesRef(species_id=ck1_id, role="substrate"),
        ],
        products=[SpeciesRef(species_id=dc_id, role="product")],
        compartments=["cytoplasm"],
        parameter_context="destruction complex assembly kon/koff",
        pathway_tag=pathway_tag,
    )
    reactions.append(r1)

    # Step 2: β-catenin 结合到 destruction complex
    r2 = ReactionV2(
        id="RXN_DC_BCAT_BINDING",
        reaction_type=MechanismType.BINDING.value,
        kinetics_type="mass_action",
        reactants=[
            SpeciesRef(species_id=dc_id, role="substrate"),
            SpeciesRef(species_id=bcat_id, role="substrate"),
        ],
        products=[SpeciesRef(species_id=dc_bcat_id, role="product")],
        compartments=["cytoplasm"],
        parameter_context="DC-β-catenin binding kon/koff",
        pathway_tag=pathway_tag,
    )
    reactions.append(r2)

    # Step 3: GSK3β 磷酸化 β-catenin（强制 MM，审计 §3.1 修复）
    # 注意：destruction complex 在反应后释放（酶不被消耗）
    r3 = ReactionV2(
        id="RXN_DC_BCAT_PHOS",
        reaction_type=MechanismType.PHOSPHORYLATION.value,
        kinetics_type="Michaelis_Menten",  # 强制 MM
        reactants=[
            SpeciesRef(species_id=dc_bcat_id, role="substrate"),
        ],
        products=[
            SpeciesRef(species_id=dc_id, role="product"),  # DC 释放
            SpeciesRef(species_id=p_bcat_id, role="product"),  # p-β-catenin
        ],
        # GSK3β 作为 catalytic modifier（不被消耗）
        modifiers=[],
        compartments=["cytoplasm"],
        parameter_context="GSK3β phosphorylation of β-catenin (Vmax/Km)",
        pathway_tag=pathway_tag,
    )
    reactions.append(r3)

    # Step 4: β-TrCP 介导的泛素化
    r4 = ReactionV2(
        id="RXN_BCAT_UBIQ",
        reaction_type=MechanismType.UBIQUITINATION.value,
        kinetics_type="mass_action",
        reactants=[
            SpeciesRef(species_id=p_bcat_id, role="substrate"),
            SpeciesRef(species_id=btrcp_id, role="enzyme"),  # β-TrCP 作为 E3 ligase
        ],
        products=[
            SpeciesRef(species_id=ub_bcat_id, role="product"),
            SpeciesRef(species_id=btrcp_id, role="enzyme"),  # 酶释放
        ],
        compartments=["cytoplasm"],
        parameter_context="β-TrCP mediated ubiquitination of p-β-catenin",
        pathway_tag=pathway_tag,
    )
    reactions.append(r4)

    # Step 5: 蛋白酶体降解
    r5 = ReactionV2(
        id="RXN_BCAT_DEG",
        reaction_type=MechanismType.PROTEASOMAL_DEGRADATION.value,
        kinetics_type="mass_action",
        reactants=[SpeciesRef(species_id=ub_bcat_id, role="substrate")],
        products=[],  # 降解为 ∅
        compartments=["cytoplasm"],
        parameter_context="proteasomal degradation of Ub-β-catenin",
        pathway_tag=pathway_tag,
    )
    reactions.append(r5)

    # 组合反应容器
    cr = CompositeReaction(
        id="CR_WNT_DESTRUCTION",
        name="Wnt destruction complex",
        sub_reactions=[
            "RXN_DC_FORMATION",
            "RXN_DC_BCAT_BINDING",
            "RXN_DC_BCAT_PHOS",
            "RXN_BCAT_UBIQ",
            "RXN_BCAT_DEG",
        ],
        coupling_type="sequential",
        intermediate_species=[dc_id, dc_bcat_id, p_bcat_id, ub_bcat_id],
        net_reaction="β-catenin → ∅ (Wnt off state)",
    )
    return reactions, cr


# =============================================================================
# 组合反应查询辅助
# =============================================================================
def get_sub_reactions(cr: CompositeReaction, ir: ReactionIRv2) -> list[ReactionV2]:
    """获取组合反应的子反应对象列表（按顺序）。"""
    result: list[ReactionV2] = []
    for rid in cr.sub_reactions:
        rxn = ir.reaction_by_id(rid)
        if rxn:
            result.append(rxn)
    return result


def validate_composite_reaction(cr: CompositeReaction, ir: ReactionIRv2) -> list[str]:
    """校验单个组合反应（Validation Rule 7 的单机版本）。

    检查：
    1. sequential 类型必须有非空 sub_reactions
    2. 每个子反应 ID 必须存在于 ir.reactions
    3. 中间产物 species_id 应存在于 ir.species（若 ir 非空）
    """
    violations: list[str] = []
    # 1. 非空检查
    if cr.coupling_type == "sequential" and not cr.sub_reactions:
        violations.append(
            f"CompositeReaction {cr.id} ({cr.name}): sequential 但 sub_reactions 为空"
        )
    # 2. 子反应存在性
    reaction_ids = {rxn.id for rxn in ir.reactions}
    for sub_id in cr.sub_reactions:
        if sub_id not in reaction_ids:
            violations.append(
                f"CompositeReaction {cr.id}: 引用了不存在的子反应 {sub_id}"
            )
    # 3. 中间产物存在性（仅当 ir.species 非空时检查）
    if ir.species:
        species_ids = {sp.id for sp in ir.species}
        for sid in cr.intermediate_species:
            if sid not in species_ids:
                violations.append(
                    f"CompositeReaction {cr.id}: 中间产物 {sid} 不在 ir.species"
                )
    return violations


def composite_reaction_to_dict(cr: CompositeReaction) -> dict[str, Any]:
    """组合反应序列化为 dict（供日志 / 调试用）。"""
    return cr.model_dump()


__all__ = [
    "CompositeReactionBuilder",
    "build_wnt_destruction_complex_reactions",
    "get_sub_reactions",
    "validate_composite_reaction",
    "composite_reaction_to_dict",
]
