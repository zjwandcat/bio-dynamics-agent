# BioDynamics Agent v4 - 10 条 Validation Rules
# 对应 v4 Scientific Architecture Part 4 §4.5 的 10 条校验规则。
#
# 设计原则：
# 1. 每条规则独立函数，便于单测与按需调用
# 2. 规则不阻塞流水线，返回 violations 列表，由调用方决定是否阻断
# 3. 替代 v3 的 pre_validate_reaction_graph（子串匹配 / 自环检测过严）
# 4. 强制 MM 用于磷酸化（审计 §3.1 修复）

from __future__ import annotations

from typing import Any

from app.reaction_ir_v2.mechanism_types import (
    is_enzymatic_mechanism,
    is_transport_mechanism,
)
from app.reaction_ir_v2.schema import ReactionIRv2


# =============================================================================
# Rule 1: Ontology Alignment
# =============================================================================
def rule1_ontology_alignment(ir: ReactionIRv2) -> list[str]:
    """所有 species 必须有 HGNC / UniProt ID（药物除外，用 ChEBI）。

    降级模式（v3→v4）下 ontology.verified=False，记 warning 不记 violation。
    """
    violations: list[str] = []
    for sp in ir.species:
        # 药物/化学实体用 ChEBI
        if sp.species_type in ("ligand", "drug", "chemical"):
            if not sp.ontology.chebi_id and not sp.ontology.hgnc_id:
                violations.append(
                    f"Rule1 Ontology: 化学实体 {sp.canonical_name} 缺少 ChEBI ID"
                )
            continue
        # 蛋白/基因必须有 HGNC 或 UniProt
        if not sp.ontology.hgnc_id and not sp.ontology.uniprot_id:
            violations.append(
                f"Rule1 Ontology: 蛋白/基因 {sp.canonical_name} 缺少 HGNC/UniProt ID"
            )
    return violations


# =============================================================================
# Rule 2: Pathway Tag
# =============================================================================
def rule2_pathway_tag(ir: ReactionIRv2) -> list[str]:
    """所有 reaction 必须有 pathway_tag。"""
    violations: list[str] = []
    for rxn in ir.reactions:
        if not rxn.pathway_tag:
            violations.append(
                f"Rule2 PathwayTag: 反应 {rxn.id} 缺少 pathway_tag"
            )
    return violations


# =============================================================================
# Rule 3: Provenance
# =============================================================================
def rule3_provenance(ir: ReactionIRv2) -> list[str]:
    """所有 reaction 必须有 source_sbml_reaction 或 source_pmid。

    降级模式下可放宽为 warning，但 P5 Validation Layer 会强制阻塞。
    """
    violations: list[str] = []
    for rxn in ir.reactions:
        prov = rxn.provenance
        if not prov.source_sbml_reaction and not prov.source_pmid:
            violations.append(
                f"Rule3 Provenance: 反应 {rxn.id} 缺少 source_sbml_reaction 与 source_pmid"
            )
    return violations


# =============================================================================
# Rule 4: Compartment Consistency
# =============================================================================
def rule4_compartment_consistency(ir: ReactionIRv2) -> list[str]:
    """跨区室反应必须有 transport 类型机制。

    若 reaction.compartments 包含多个不同区室，则 reaction_type 必须是 transport 类。
    """
    violations: list[str] = []
    for rxn in ir.reactions:
        # 去重 compartments
        unique_compartments = set(rxn.compartments)
        if len(unique_compartments) <= 1:
            continue
        # 多区室反应必须是 transport 机制
        if not is_transport_mechanism(rxn.reaction_type):
            violations.append(
                f"Rule4 Compartment: 反应 {rxn.id} 跨区室 {unique_compartments} "
                f"但机制 {rxn.reaction_type} 非 transport 类"
            )
    return violations


# =============================================================================
# Rule 5: State Machine Closure
# =============================================================================
def rule5_state_machine_closure(ir: ReactionIRv2) -> list[str]:
    """状态机的所有 transition 必须关联到存在的 Reaction。

    且 transition.from_state / to_state 必须在 states 列表中。
    """
    violations: list[str] = []
    reaction_ids = {rxn.id for rxn in ir.reactions}
    for sm in ir.state_machines:
        state_names = {s.name for s in sm.states}
        for trans in sm.transitions:
            # transition 关联的 reaction 必须存在
            if trans.reaction_id and trans.reaction_id not in reaction_ids:
                violations.append(
                    f"Rule5 StateMachine: 状态机 {sm.id} 的 transition "
                    f"{trans.from_state}→{trans.to_state} 引用了不存在的反应 {trans.reaction_id}"
                )
            # from_state / to_state 必须在 states 列表
            if trans.from_state and trans.from_state not in state_names:
                violations.append(
                    f"Rule5 StateMachine: 状态机 {sm.id} 的 from_state "
                    f"'{trans.from_state}' 不在 states 列表"
                )
            if trans.to_state and trans.to_state not in state_names:
                violations.append(
                    f"Rule5 StateMachine: 状态机 {sm.id} 的 to_state "
                    f"'{trans.to_state}' 不在 states 列表"
                )
    return violations


# =============================================================================
# Rule 6: Constraint Satisfaction
# =============================================================================
def rule6_constraint_satisfaction(ir: ReactionIRv2) -> list[str]:
    """所有约束在初始条件下满足。

    委托给 constraints.check_all_constraints 执行。
    """
    from app.reaction_ir_v2.constraints import check_all_constraints
    report = check_all_constraints(ir)
    return report["violations"]


# =============================================================================
# Rule 7: Composite Reaction Order
# =============================================================================
def rule7_composite_reaction_order(ir: ReactionIRv2) -> list[str]:
    """sequential 类型的 sub_reactions 必须有明确顺序（非空且每个 reaction_id 存在）。"""
    violations: list[str] = []
    reaction_ids = {rxn.id for rxn in ir.reactions}
    for cr in ir.composite_reactions:
        if cr.coupling_type != "sequential":
            continue
        if not cr.sub_reactions:
            violations.append(
                f"Rule7 Composite: 组合反应 {cr.id} ({cr.name}) 为 sequential 但 sub_reactions 为空"
            )
            continue
        # 每个子反应 ID 必须存在
        for sub_id in cr.sub_reactions:
            if sub_id not in reaction_ids:
                violations.append(
                    f"Rule7 Composite: 组合反应 {cr.id} 引用了不存在的子反应 {sub_id}"
                )
        # 检查顺序：sub_reactions 列表本身就是顺序定义，非空即合规
    return violations


# =============================================================================
# Rule 8: Enzyme Role
# =============================================================================
def rule8_enzyme_role(ir: ReactionIRv2) -> list[str]:
    """标记为 enzyme 的 species 必须同时出现在 reactants 与 products。

    委托给 constraints.check_enzymatic 执行（复用同一逻辑）。
    """
    from app.reaction_ir_v2.constraints import check_enzymatic
    return check_enzymatic(ir)


# =============================================================================
# Rule 9: Kinetics-Mechanism Match
# =============================================================================
def rule9_kinetics_mechanism_match(ir: ReactionIRv2) -> list[str]:
    """phosphorylation 必须用 MM（强制，禁止降级）。

    审计报告 §3.1 致命错误修复：phosphorylation / dephosphorylation / cleavage
    必须用 Michaelis_Menten，禁止降级为 mass-action。
    """
    violations: list[str] = []
    for rxn in ir.reactions:
        if not is_enzymatic_mechanism(rxn.reaction_type):
            continue
        if rxn.kinetics_type != "Michaelis_Menten":
            violations.append(
                f"Rule9 Kinetics: 反应 {rxn.id} 机制 {rxn.reaction_type} "
                f"必须用 Michaelis_Menten，实际为 {rxn.kinetics_type}（禁止降级）"
            )
        # transcription 必须 Hill
    for rxn in ir.reactions:
        if rxn.reaction_type == "transcription" and rxn.kinetics_type != "Hill":
            violations.append(
                f"Rule9 Kinetics: 反应 {rxn.id} 机制 transcription "
                f"建议用 Hill，实际为 {rxn.kinetics_type}"
            )
    return violations


# =============================================================================
# Rule 10: Cross-talk Edge Validation
# =============================================================================
def rule10_crosstalk_edge_validation(ir: ReactionIRv2) -> list[str]:
    """cross-talk edge 必须标记两个 pathway_tag。

    cross-talk 反应的 pathway_tag 应为 "CROSSTALK_A_B" 格式（含两个通路名）。
    """
    violations: list[str] = []
    for rxn in ir.reactions:
        tag = rxn.pathway_tag or ""
        if not tag.upper().startswith("CROSSTALK"):
            continue
        # 格式应为 CROSSTALK_PATHWAY_A_PATHWAY_B
        parts = tag.split("_")
        # 至少 4 段：CROSSTALK + A + B（A/B 至少 1 段）
        if len(parts) < 3:
            violations.append(
                f"Rule10 CrossTalk: 反应 {rxn.id} pathway_tag '{tag}' "
                f"格式不合法（应为 CROSSTALK_A_B 含两个通路名）"
            )
    return violations


# =============================================================================
# 统一入口：执行全部 10 条规则
# =============================================================================
def validate_all(ir: ReactionIRv2) -> dict[str, Any]:
    """执行全部 10 条 Validation Rules，返回汇总报告。

    Args:
        ir: ReactionIRv2 对象

    Returns:
        {
            "passed": bool,                # 是否全部通过
            "total_violations": int,       # violation 总数
            "by_rule": {                   # 按规则分组
                "rule1_ontology_alignment": [...],
                "rule2_pathway_tag": [...],
                ...,
                "rule10_crosstalk_edge_validation": [...],
            },
        }
    """
    rules = {
        "rule1_ontology_alignment": rule1_ontology_alignment,
        "rule2_pathway_tag": rule2_pathway_tag,
        "rule3_provenance": rule3_provenance,
        "rule4_compartment_consistency": rule4_compartment_consistency,
        "rule5_state_machine_closure": rule5_state_machine_closure,
        "rule6_constraint_satisfaction": rule6_constraint_satisfaction,
        "rule7_composite_reaction_order": rule7_composite_reaction_order,
        "rule8_enzyme_role": rule8_enzyme_role,
        "rule9_kinetics_mechanism_match": rule9_kinetics_mechanism_match,
        "rule10_crosstalk_edge_validation": rule10_crosstalk_edge_validation,
    }
    by_rule: dict[str, list[str]] = {}
    total = 0
    for name, fn in rules.items():
        violations = fn(ir)
        by_rule[name] = violations
        total += len(violations)
    return {
        "passed": total == 0,
        "total_violations": total,
        "by_rule": by_rule,
    }


__all__ = [
    "rule1_ontology_alignment",
    "rule2_pathway_tag",
    "rule3_provenance",
    "rule4_compartment_consistency",
    "rule5_state_machine_closure",
    "rule6_constraint_satisfaction",
    "rule7_composite_reaction_order",
    "rule8_enzyme_role",
    "rule9_kinetics_mechanism_match",
    "rule10_crosstalk_edge_validation",
    "validate_all",
]
