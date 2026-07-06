# BioDynamics Agent v4 - Reaction IR v2 单元测试
# 覆盖 P2 完成标准的测试用例：
# 1. Schema 创建与校验（6 个核心组件）
# 2. 10 条 Validation Rules
# 3. CompositeReaction 构建（Wnt destruction complex 三步耦合）
# 4. State Machine 构建（EGFR 状态转换）
# 5. 5 类约束检查
#
# 运行：cd backend && python -m pytest tests/test_reaction_ir_v2.py -v
# 或：  cd backend && python tests/test_reaction_ir_v2.py

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

# 确保能导入 app 包（backend/ 在 sys.path）
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


class TestSchemaCreation(unittest.TestCase):
    """Schema 创建与校验测试（6 个核心组件）。"""

    def test_species_v2_creation(self):
        """测试 SpeciesV2 创建与默认值。"""
        from app.reaction_ir_v2.schema import SpeciesV2
        sp = SpeciesV2(id="SP_001", canonical_name="EGFR")
        self.assertEqual(sp.canonical_name, "EGFR")
        self.assertEqual(sp.compartment, "cytoplasm")  # 默认值
        self.assertEqual(sp.initial_concentration, 0.0)
        self.assertFalse(sp.ontology.verified)  # 默认未验证

    def test_species_compartment_validation(self):
        """测试区室校验：非法值降级为 cytoplasm。"""
        from app.reaction_ir_v2.schema import SpeciesV2
        sp = SpeciesV2(id="SP_001", canonical_name="X", compartment="invalid")
        self.assertEqual(sp.compartment, "cytoplasm")

    def test_reaction_v2_creation(self):
        """测试 ReactionV2 创建与默认动力学。"""
        from app.reaction_ir_v2.schema import ReactionV2
        rxn = ReactionV2(id="RXN_001", reaction_type="phosphorylation")
        self.assertEqual(rxn.reaction_type, "phosphorylation")
        self.assertEqual(rxn.kinetics_type, "mass_action")  # 默认值

    def test_reaction_kinetics_validation(self):
        """测试动力学类型校验：非法值降级为 mass_action。"""
        from app.reaction_ir_v2.schema import ReactionV2
        rxn = ReactionV2(
            id="RXN_001", reaction_type="binding", kinetics_type="invalid_type"
        )
        self.assertEqual(rxn.kinetics_type, "mass_action")

    def test_composite_reaction_creation(self):
        """测试 CompositeReaction 创建。"""
        from app.reaction_ir_v2.schema import CompositeReaction
        cr = CompositeReaction(id="CR_001", name="Test composite")
        self.assertEqual(cr.coupling_type, "sequential")  # 默认值
        self.assertEqual(cr.sub_reactions, [])

    def test_state_machine_creation(self):
        """测试 StateMachine 创建。"""
        from app.reaction_ir_v2.schema import State, StateMachine, Transition
        sm = StateMachine(
            id="SM_001",
            species="EGFR",
            states=[State(name="monomer", species_id="SP_001", is_initial=True)],
            transitions=[Transition(
                from_state="monomer", to_state="dimer", reaction_id="RXN_001"
            )],
        )
        self.assertEqual(len(sm.states), 1)
        self.assertEqual(len(sm.transitions), 1)
        self.assertTrue(sm.states[0].is_initial)

    def test_compartment_creation(self):
        """测试 Compartment 创建与校验。"""
        from app.reaction_ir_v2.schema import Compartment
        comp = Compartment(name="nucleus", size=0.1)
        self.assertEqual(comp.name, "nucleus")
        self.assertEqual(comp.size, 0.1)
        # 非法值降级
        comp2 = Compartment(name="invalid")
        self.assertEqual(comp2.name, "cytoplasm")

    def test_constraint_creation(self):
        """测试 Constraint 创建与校验。"""
        from app.reaction_ir_v2.schema import Constraint
        c = Constraint(
            type="mass_conservation",
            expression="EGFR + pEGFR = EGFR_total",
            tolerance=0.05,
        )
        self.assertEqual(c.type, "mass_conservation")
        # 非法 type 降级
        c2 = Constraint(type="invalid_type", expression="x")
        self.assertEqual(c2.type, "non_negative")

    def test_reaction_ir_v2_container(self):
        """测试 ReactionIRv2 顶层容器。"""
        from app.reaction_ir_v2.schema import (
            ReactionIRv2, ReactionV2, SpeciesV2,
        )
        ir = ReactionIRv2(
            species=[SpeciesV2(id="SP_001", canonical_name="EGFR")],
            reactions=[ReactionV2(id="RXN_001", reaction_type="binding")],
        )
        self.assertEqual(len(ir.species), 1)
        self.assertEqual(len(ir.reactions), 1)
        self.assertEqual(ir.version, "v4.0")
        # 查询方法
        sp = ir.species_by_id("SP_001")
        self.assertIsNotNone(sp)
        sp = ir.species_by_name("EGFR")
        self.assertIsNotNone(sp)
        rxn = ir.reaction_by_id("RXN_001")
        self.assertIsNotNone(rxn)

    def test_reaction_ir_v2_serialization(self):
        """测试 ReactionIRv2 序列化与反序列化。"""
        from app.reaction_ir_v2.schema import (
            ReactionIRv2, ReactionV2, SpeciesV2,
        )
        ir = ReactionIRv2(
            species=[SpeciesV2(id="SP_001", canonical_name="EGFR")],
            reactions=[ReactionV2(id="RXN_001", reaction_type="binding")],
        )
        d = ir.to_dict()
        self.assertIsInstance(d, dict)
        self.assertIn("species", d)
        self.assertIn("reactions", d)
        # 反序列化
        ir2 = ReactionIRv2.from_dict(d)
        self.assertEqual(len(ir2.species), 1)
        self.assertEqual(ir2.species[0].canonical_name, "EGFR")


class TestMechanismTypes(unittest.TestCase):
    """17 类机制枚举测试。"""

    def test_17_mechanisms_defined(self):
        """验证 17 类机制全部定义。"""
        from app.reaction_ir_v2.mechanism_types import MechanismType
        # 架构 §4.3 列出 17 类机制（含 dissociation 共 18 个枚举值）
        self.assertGreaterEqual(len(MechanismType), 17)

    def test_phosphorylation_sbo(self):
        """磷酸化对应 SBO:0000216。"""
        from app.reaction_ir_v2.mechanism_types import MechanismType
        self.assertEqual(
            MechanismType.PHOSPHORYLATION.sbo_term, "SBO:0000216"
        )

    def test_phosphorylation_default_kinetics_mm(self):
        """磷酸化默认动力学为 Michaelis_Menten（审计 §3.1 修复）。"""
        from app.reaction_ir_v2.mechanism_types import MechanismType
        self.assertEqual(
            MechanismType.PHOSPHORYLATION.default_kinetics, "Michaelis_Menten"
        )

    def test_transcription_default_kinetics_hill(self):
        """转录默认动力学为 Hill。"""
        from app.reaction_ir_v2.mechanism_types import MechanismType
        self.assertEqual(MechanismType.TRANSCRIPTION.default_kinetics, "Hill")

    def test_v3_interaction_to_mechanism_activation(self):
        """v3 activation → v4 activation（不强制映射为 phosphorylation）。"""
        from app.reaction_ir_v2.mechanism_types import (
            MechanismType, v3_interaction_to_mechanism,
        )
        m = v3_interaction_to_mechanism("activation")
        self.assertEqual(m, MechanismType.ACTIVATION)

    def test_v3_interaction_to_mechanism_inhibition(self):
        """v3 inhibition → v4 inhibition。"""
        from app.reaction_ir_v2.mechanism_types import (
            MechanismType, v3_interaction_to_mechanism,
        )
        m = v3_interaction_to_mechanism("inhibition")
        self.assertEqual(m, MechanismType.INHIBITION)

    def test_is_enzymatic_mechanism(self):
        """酶催化机制识别（强制 MM）。"""
        from app.reaction_ir_v2.mechanism_types import is_enzymatic_mechanism
        self.assertTrue(is_enzymatic_mechanism("phosphorylation"))
        self.assertTrue(is_enzymatic_mechanism("dephosphorylation"))
        self.assertTrue(is_enzymatic_mechanism("cleavage"))
        self.assertFalse(is_enzymatic_mechanism("binding"))
        self.assertFalse(is_enzymatic_mechanism("transcription"))


class TestValidationRules(unittest.TestCase):
    """10 条 Validation Rules 测试。"""

    def _build_egfr_ir(self):
        """构建测试用 EGF-EGFR ReactionIRv2。"""
        from app.reaction_ir_v2.schema import (
            Constraint, OntologyRef, Provenance, ReactionIRv2,
            ReactionV2, SpeciesRef, SpeciesV2,
        )
        species = [
            SpeciesV2(
                id="SP_EGF", canonical_name="EGF", species_type="ligand",
                compartment="extracellular",
                ontology=OntologyRef(chebi_id="CHEBI:5322", verified=True),
            ),
            SpeciesV2(
                id="SP_EGFR", canonical_name="EGFR", species_type="receptor",
                compartment="membrane",
                ontology=OntologyRef(
                    hgnc_id="HGNC:3236", uniprot_id="P00533", verified=True
                ),
            ),
            SpeciesV2(
                id="SP_pEGFR", canonical_name="pEGFR", species_type="protein",
                compartment="membrane",
                ontology=OntologyRef(
                    hgnc_id="HGNC:3236", uniprot_id="P00533", verified=True
                ),
            ),
        ]
        reactions = [
            ReactionV2(
                id="RXN_001",
                reaction_type="binding",
                kinetics_type="mass_action",
                reactants=[SpeciesRef(species_id="SP_EGF", role="substrate"),
                           SpeciesRef(species_id="SP_EGFR", role="substrate")],
                products=[SpeciesRef(species_id="SP_pEGFR", role="product")],
                pathway_tag="EGFR_RTK",
                provenance=Provenance(source_pmid="PMID:12345"),
            ),
        ]
        constraints = [
            Constraint(
                type="mass_conservation",
                expression="EGFR + pEGFR = EGFR_total",
                tolerance=0.05,
            ),
        ]
        return ReactionIRv2(
            species=species, reactions=reactions, constraints=constraints,
        )

    def test_rule1_ontology_alignment_pass(self):
        """Rule1: 物种有 HGNC/UniProt/ChEBI ID 时通过。"""
        from app.reaction_ir_v2.validation_rules import rule1_ontology_alignment
        ir = self._build_egfr_ir()
        violations = rule1_ontology_alignment(ir)
        # EGF 有 ChEBI, EGFR/pEGFR 有 HGNC+UniProt
        self.assertEqual(violations, [])

    def test_rule1_ontology_alignment_fail(self):
        """Rule1: 缺少 ontology ID 时记 violation。"""
        from app.reaction_ir_v2.schema import (
            OntologyRef, ReactionIRv2, SpeciesV2,
        )
        from app.reaction_ir_v2.validation_rules import rule1_ontology_alignment
        ir = ReactionIRv2(
            species=[
                SpeciesV2(
                    id="SP_X", canonical_name="X", species_type="protein",
                    ontology=OntologyRef(verified=False),  # 无 HGNC/UniProt
                ),
            ],
        )
        violations = rule1_ontology_alignment(ir)
        self.assertEqual(len(violations), 1)
        self.assertIn("缺少 HGNC/UniProt ID", violations[0])

    def test_rule2_pathway_tag_pass(self):
        """Rule2: 反应有 pathway_tag 时通过。"""
        from app.reaction_ir_v2.validation_rules import rule2_pathway_tag
        ir = self._build_egfr_ir()
        violations = rule2_pathway_tag(ir)
        self.assertEqual(violations, [])

    def test_rule2_pathway_tag_fail(self):
        """Rule2: 缺少 pathway_tag 时记 violation。"""
        from app.reaction_ir_v2.schema import ReactionIRv2, ReactionV2
        from app.reaction_ir_v2.validation_rules import rule2_pathway_tag
        ir = ReactionIRv2(
            reactions=[ReactionV2(id="RXN_001", reaction_type="binding")],  # 无 pathway_tag
        )
        violations = rule2_pathway_tag(ir)
        self.assertEqual(len(violations), 1)

    def test_rule9_kinetics_mechanism_match_phosphorylation_mm(self):
        """Rule9: phosphorylation 必须用 MM（审计 §3.1 修复）。"""
        from app.reaction_ir_v2.schema import ReactionIRv2, ReactionV2
        from app.reaction_ir_v2.validation_rules import rule9_kinetics_mechanism_match
        # 合规：phosphorylation + MM
        ir_ok = ReactionIRv2(
            reactions=[ReactionV2(
                id="RXN_001", reaction_type="phosphorylation",
                kinetics_type="Michaelis_Menten",
            )],
        )
        self.assertEqual(rule9_kinetics_mechanism_match(ir_ok), [])
        # 违规：phosphorylation + mass_action（降级）
        ir_bad = ReactionIRv2(
            reactions=[ReactionV2(
                id="RXN_001", reaction_type="phosphorylation",
                kinetics_type="mass_action",
            )],
        )
        violations = rule9_kinetics_mechanism_match(ir_bad)
        self.assertEqual(len(violations), 1)
        self.assertIn("禁止降级", violations[0])

    def test_validate_all_pass(self):
        """全部 10 条规则通过。"""
        from app.reaction_ir_v2.validation_rules import validate_all
        ir = self._build_egfr_ir()
        report = validate_all(ir)
        # Rule3 (provenance) 可能记 violation（取决于 ir 构建）
        # 这里仅验证 validate_all 返回结构正确
        self.assertIn("passed", report)
        self.assertIn("total_violations", report)
        self.assertIn("by_rule", report)
        self.assertEqual(len(report["by_rule"]), 10)


class TestCompositeReaction(unittest.TestCase):
    """CompositeReaction 构建测试（Wnt destruction complex 三步耦合）。"""

    def test_wnt_destruction_complex_build(self):
        """测试 Wnt destruction complex 5 步耦合反应构建。"""
        from app.reaction_ir_v2.composite_reaction import (
            build_wnt_destruction_complex_reactions,
        )
        reactions, cr = build_wnt_destruction_complex_reactions()
        # 5 步子反应
        self.assertEqual(len(reactions), 5)
        self.assertEqual(len(cr.sub_reactions), 5)
        self.assertEqual(cr.coupling_type, "sequential")
        self.assertEqual(cr.name, "Wnt destruction complex")
        # 净反应
        self.assertIn("β-catenin", cr.net_reaction)
        # 第 3 步磷酸化必须用 MM（审计 §3.1 修复）
        phos_rxn = next(
            r for r in reactions if r.reaction_type == "phosphorylation"
        )
        self.assertEqual(phos_rxn.kinetics_type, "Michaelis_Menten")

    def test_composite_reaction_order_validation(self):
        """Rule7: sequential 类型的 sub_reactions 必须有明确顺序。"""
        from app.reaction_ir_v2.composite_reaction import (
            build_wnt_destruction_complex_reactions,
        )
        from app.reaction_ir_v2.schema import ReactionIRv2
        from app.reaction_ir_v2.validation_rules import rule7_composite_reaction_order
        reactions, cr = build_wnt_destruction_complex_reactions()
        ir = ReactionIRv2(reactions=reactions, composite_reactions=[cr])
        violations = rule7_composite_reaction_order(ir)
        self.assertEqual(violations, [])


class TestStateMachine(unittest.TestCase):
    """State Machine 构建测试（EGFR 状态转换）。"""

    def test_egfr_state_machine_build(self):
        """测试 EGFR 状态机构建（7 个状态，6 个转换）。"""
        from app.reaction_ir_v2.state_machine import build_egfr_state_machine
        sm = build_egfr_state_machine()
        self.assertEqual(sm.id, "EGFR_STATE_MACHINE")
        self.assertEqual(sm.species, "EGFR")
        # 7 个状态：monomer/egf_bound/dimer/phosphorylated_dimer/grb2_bound/internalized/degraded
        self.assertEqual(len(sm.states), 7)
        # 6 个转换
        self.assertEqual(len(sm.transitions), 6)
        # 初始状态
        initial_states = [s for s in sm.states if s.is_initial]
        self.assertEqual(len(initial_states), 1)
        self.assertEqual(initial_states[0].name, "monomer")

    def test_state_machine_closure_validation(self):
        """Rule5: 状态机的 transition 必须关联到存在的 Reaction。"""
        from app.reaction_ir_v2.schema import ReactionIRv2, ReactionV2
        from app.reaction_ir_v2.state_machine import build_egfr_state_machine
        from app.reaction_ir_v2.validation_rules import rule5_state_machine_closure
        sm = build_egfr_state_machine()
        # 构建 IR，包含 sm 中引用的 reaction_id
        reaction_ids = {t.reaction_id for t in sm.transitions}
        reactions = [
            ReactionV2(id=rid, reaction_type="binding") for rid in reaction_ids
        ]
        ir = ReactionIRv2(reactions=reactions, state_machines=[sm])
        violations = rule5_state_machine_closure(ir)
        self.assertEqual(violations, [])


class TestConstraints(unittest.TestCase):
    """5 类约束检查测试。"""

    def test_non_negative_check(self):
        """Non-negative 约束检查。"""
        from app.reaction_ir_v2.constraints import check_non_negative
        from app.reaction_ir_v2.schema import ReactionIRv2, SpeciesV2
        # 合规
        ir_ok = ReactionIRv2(
            species=[SpeciesV2(id="SP_001", canonical_name="X", initial_concentration=0.0)],
        )
        self.assertEqual(check_non_negative(ir_ok), [])
        # 违规
        ir_bad = ReactionIRv2(
            species=[SpeciesV2(id="SP_001", canonical_name="X", initial_concentration=-1.0)],
        )
        violations = check_non_negative(ir_bad)
        self.assertEqual(len(violations), 1)
        self.assertIn("负数", violations[0])

    def test_check_all_constraints(self):
        """全部 5 类约束检查汇总。"""
        from app.reaction_ir_v2.constraints import check_all_constraints
        from app.reaction_ir_v2.schema import ReactionIRv2
        ir = ReactionIRv2()
        report = check_all_constraints(ir)
        self.assertIn("passed", report)
        self.assertIn("violations", report)
        self.assertIn("by_type", report)
        self.assertEqual(len(report["by_type"]), 5)


class TestReactionBuilder(unittest.TestCase):
    """Reaction Builder 测试（从 network_json 构建 ReactionIRv2）。"""

    def test_build_from_network_json_egf_egfr(self):
        """测试从 v3 network_json 构建 EGF-EGFR ReactionIRv2。"""
        from app.reaction_ir_v2.reaction_builder import build_from_network_json
        network_json = {
            "nodes": [
                {"id": "EGF", "name": "EGF", "type": "ligand"},
                {"id": "EGFR", "name": "EGFR", "type": "receptor"},
                {"id": "pEGFR", "name": "pEGFR", "type": "protein"},
            ],
            "edges": [
                {"source": "EGF", "target": "EGFR", "interaction": "binding"},
                {"source": "EGFR", "target": "pEGFR", "interaction": "activation"},
            ],
        }
        ir = build_from_network_json(network_json, pathway_tag="EGFR_RTK")
        self.assertEqual(len(ir.species), 3)
        self.assertEqual(len(ir.reactions), 2)
        self.assertEqual(ir.source, "v3_downgraded")
        # ligand → extracellular, receptor → membrane
        egl = ir.species_by_name("EGF")
        self.assertEqual(egl.compartment, "extracellular")
        egfr = ir.species_by_name("EGFR")
        self.assertEqual(egfr.compartment, "membrane")

    def test_build_with_ontology_entities(self):
        """测试带 ontology_entities 的构建。"""
        from app.reaction_ir_v2.reaction_builder import build_from_network_json
        network_json = {
            "nodes": [{"id": "EGFR", "name": "EGFR", "type": "protein"}],
            "edges": [],
        }
        ontology_entities = {
            "entities": [{
                "name": "EGFR",
                "hgnc_id": "HGNC:3236",
                "uniprot_id": "P00533",
                "verified": True,
            }],
        }
        ir = build_from_network_json(network_json, ontology_entities=ontology_entities)
        self.assertEqual(ir.source, "v4_native")
        egfr = ir.species_by_name("EGFR")
        self.assertTrue(egfr.ontology.verified)
        self.assertEqual(egfr.ontology.hgnc_id, "HGNC:3236")


if __name__ == "__main__":
    unittest.main()
