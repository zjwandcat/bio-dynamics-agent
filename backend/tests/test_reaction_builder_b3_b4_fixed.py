# BioDynamics Agent v4 - Task 4.0 B3+B4 修复验证测试
# 覆盖：
#   B3: PHOSPHORYLATION 自磷酸化与异磷酸化反应物构建（check_enzymatic 不再误判）
#   B4: build_from_pathway_graph 直接构造 MechanismType 枚举（8 种机制不 fallback）
#
# 运行：cd backend && python -m pytest tests/test_reaction_builder_b3_b4_fixed.py -v
# 或：  cd backend && python tests/test_reaction_builder_b3_b4_fixed.py

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# 确保能导入 app 包（backend/ 在 sys.path）
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


# =============================================================================
# B3: PHOSPHORYLATION 自磷酸化与异磷酸化
# =============================================================================
class TestB3PhosphorylationAutoVsHetero(unittest.TestCase):
    """B3: PHOSPHORYLATION 区分自磷酸化与异磷酸化。"""

    def test_autophosphorylation_egfr_to_pegfr(self):
        """B3 自磷酸化：EGFR → pEGFR (interaction=phosphorylation)。

        预期：
        - EGFR 在 reactants (substrate)
        - pEGFR 在 products
        - 无 modifier（自磷酸化无外部激酶）
        - check_enzymatic 不报违规

        修复前缺陷：source 同时作 substrate + catalytic modifier，
        check_enzymatic 因 enzyme 在 reactants 而不在 products 判定酶被消耗。
        """
        from app.reaction_ir_v2.reaction_builder import build_from_network_json
        from app.reaction_ir_v2.constraints import check_enzymatic

        network_json = {
            "nodes": [
                {"id": "EGFR", "name": "EGFR", "type": "receptor"},
                {"id": "pEGFR", "name": "pEGFR", "type": "protein"},
            ],
            "edges": [
                {"source": "EGFR", "target": "pEGFR",
                 "interaction": "phosphorylation"},
            ],
        }
        ir = build_from_network_json(network_json, pathway_tag="EGFR_RTK")

        self.assertEqual(len(ir.reactions), 1)
        rxn = ir.reactions[0]
        self.assertEqual(rxn.reaction_type, "phosphorylation")

        egfr_id = ir.species_by_name("EGFR").id
        pegfr_id = ir.species_by_name("pEGFR").id

        # EGFR 在 reactants (substrate)
        substrate_ids = {r.species_id for r in rxn.reactants
                         if r.role == "substrate"}
        self.assertIn(egfr_id, substrate_ids,
                      "EGFR 应在 reactants (substrate)")

        # pEGFR 在 products
        product_ids = {p.species_id for p in rxn.products
                       if p.role == "product"}
        self.assertIn(pegfr_id, product_ids, "pEGFR 应在 products")

        # 无 modifier（自磷酸化无外部激酶）
        self.assertEqual(len(rxn.modifiers), 0,
                         "自磷酸化不应有 modifier")

        # EGFR 不应同时在 reactants 和 modifiers（B3 缺陷核心）
        modifier_ids = {m.species_id for m in rxn.modifiers}
        self.assertNotIn(egfr_id, modifier_ids,
                         "EGFR 不应同时在 reactants 和 modifiers（B3 缺陷）")

        # check_enzymatic 不报违规
        violations = check_enzymatic(ir)
        self.assertEqual(violations, [],
                         f"check_enzymatic 不应报违规：{violations}")

    def test_hetero_phosphorylation_akt_to_ptsc2(self):
        """B3 异磷酸化：AKT → pTSC2 (interaction=phosphorylation)。

        预期：
        - TSC2 在 reactants (substrate)
        - pTSC2 在 products
        - AKT 仅在 modifiers (catalytic)
        - AKT 不在 reactants/products（酶不被消耗，B3 缺陷核心）
        - check_enzymatic 不报违规
        """
        from app.reaction_ir_v2.reaction_builder import build_from_network_json
        from app.reaction_ir_v2.constraints import check_enzymatic

        network_json = {
            "nodes": [
                {"id": "AKT", "name": "AKT", "type": "kinase"},
                {"id": "TSC2", "name": "TSC2", "type": "protein"},
                {"id": "pTSC2", "name": "pTSC2", "type": "protein"},
            ],
            "edges": [
                {"source": "AKT", "target": "pTSC2",
                 "interaction": "phosphorylation"},
            ],
        }
        ir = build_from_network_json(network_json, pathway_tag="PI3K_AKT_mTOR")

        self.assertEqual(len(ir.reactions), 1)
        rxn = ir.reactions[0]
        self.assertEqual(rxn.reaction_type, "phosphorylation")

        akt_id = ir.species_by_name("AKT").id
        tsc2_id = ir.species_by_name("TSC2").id
        ptsc2_id = ir.species_by_name("pTSC2").id

        # TSC2 在 reactants (substrate)
        substrate_ids = {r.species_id for r in rxn.reactants
                         if r.role == "substrate"}
        self.assertIn(tsc2_id, substrate_ids,
                      "TSC2 应在 reactants (substrate)")

        # pTSC2 在 products
        product_ids = {p.species_id for p in rxn.products
                       if p.role == "product"}
        self.assertIn(ptsc2_id, product_ids, "pTSC2 应在 products")

        # AKT 仅在 modifiers (catalytic)
        catalytic_ids = {m.species_id for m in rxn.modifiers
                         if m.modifier_type == "catalytic"}
        self.assertIn(akt_id, catalytic_ids,
                      "AKT 应在 modifiers (catalytic)")

        # AKT 不在 reactants/products（酶不被消耗，B3 缺陷核心）
        reactant_ids = {r.species_id for r in rxn.reactants}
        product_ids_all = {p.species_id for p in rxn.products}
        self.assertNotIn(akt_id, reactant_ids,
                         "AKT 不应在 reactants（异磷酸化酶不被消耗）")
        self.assertNotIn(akt_id, product_ids_all,
                         "AKT 不应在 products（异磷酸化酶不被消耗）")

        # check_enzymatic 不报违规
        violations = check_enzymatic(ir)
        self.assertEqual(violations, [],
                         f"check_enzymatic 不应报违规：{violations}")

    def test_hetero_phosphorylation_fallback_when_unphos_missing(self):
        """B3 fallback：异磷酸化场景下若未提供未磷酸化底物节点，
        回退为 target 自身作 substrate（不阻断流程）。

        例如 edge `AKT → pTSC2` 但 nodes 中只有 AKT 和 pTSC2（无 TSC2），
        则 substrate 使用 pTSC2 自身（语义不精确但不崩溃）。
        """
        from app.reaction_ir_v2.reaction_builder import build_from_network_json

        network_json = {
            "nodes": [
                {"id": "AKT", "name": "AKT", "type": "kinase"},
                {"id": "pTSC2", "name": "pTSC2", "type": "protein"},
            ],
            "edges": [
                {"source": "AKT", "target": "pTSC2",
                 "interaction": "phosphorylation"},
            ],
        }
        ir = build_from_network_json(network_json)
        self.assertEqual(len(ir.reactions), 1)
        rxn = ir.reactions[0]
        # 仍应有 catalytic modifier（AKT 是激酶）
        self.assertTrue(any(m.species_id == ir.species_by_name("AKT").id
                            for m in rxn.modifiers
                            if m.modifier_type == "catalytic"))


# =============================================================================
# B3: 质量守恒约束自动生成仍正常工作
# =============================================================================
class TestB3MassConservationStillWorks(unittest.TestCase):
    """B3 修复后，auto_generate_mass_conservation 仍正确生成守恒约束。"""

    def test_autophosphorylation_generates_mass_conservation(self):
        """自磷酸化 EGFR → pEGFR 仍生成 'EGFR + pEGFR = EGFR_total' 约束。"""
        from app.reaction_ir_v2.reaction_builder import build_from_network_json

        network_json = {
            "nodes": [
                {"id": "EGFR", "name": "EGFR", "type": "receptor"},
                {"id": "pEGFR", "name": "pEGFR", "type": "protein"},
            ],
            "edges": [
                {"source": "EGFR", "target": "pEGFR",
                 "interaction": "phosphorylation"},
            ],
        }
        ir = build_from_network_json(network_json, pathway_tag="EGFR_RTK")
        mc_constraints = [c for c in ir.constraints
                          if c.type == "mass_conservation"]
        self.assertGreaterEqual(len(mc_constraints), 1,
                               "应至少生成 1 条质量守恒约束")
        expr = mc_constraints[0].expression
        self.assertIn("EGFR", expr)
        self.assertIn("pEGFR", expr)
        self.assertIn("EGFR_total", expr)

    def test_hetero_phosphorylation_generates_mass_conservation(self):
        """异磷酸化 AKT → pTSC2 仍生成 'TSC2 + pTSC2 = TSC2_total' 约束。"""
        from app.reaction_ir_v2.reaction_builder import build_from_network_json

        network_json = {
            "nodes": [
                {"id": "AKT", "name": "AKT", "type": "kinase"},
                {"id": "TSC2", "name": "TSC2", "type": "protein"},
                {"id": "pTSC2", "name": "pTSC2", "type": "protein"},
            ],
            "edges": [
                {"source": "AKT", "target": "pTSC2",
                 "interaction": "phosphorylation"},
            ],
        }
        ir = build_from_network_json(network_json, pathway_tag="PI3K_AKT_mTOR")
        mc_constraints = [c for c in ir.constraints
                          if c.type == "mass_conservation"]
        self.assertGreaterEqual(len(mc_constraints), 1,
                               "应至少生成 1 条质量守恒约束")
        expr = mc_constraints[0].expression
        self.assertIn("TSC2", expr)
        self.assertIn("pTSC2", expr)
        self.assertIn("TSC2_total", expr)


# =============================================================================
# B4: build_from_pathway_graph 8 种机制直接构造
# =============================================================================
class TestB4MechanismDirectConstruction(unittest.TestCase):
    """B4: build_from_pathway_graph 直接构造 MechanismType 枚举。

    8 种机制（dimerization/complex_formation/sequestration/gtp_gdp_exchange/
    nuclear_import/nuclear_export/cytoplasm_translocation/proteasomal_degradation）
    不再被 fallback 到 ACTIVATION。
    """

    def _build_pathway_graph_with_mechanism(self, mechanism: str):
        """构造含单条指定 mechanism 边的 pathway_graph 并构建 IR。"""
        from app.reaction_ir_v2.reaction_builder import build_from_pathway_graph

        pathway_graph = {
            "nodes": [
                {"id": "A", "name": "A", "type": "protein"},
                {"id": "B", "name": "B", "type": "protein"},
            ],
            "edges": [
                {"source": "A", "target": "B", "mechanism": mechanism,
                 "pathway_tag": "TEST_PATHWAY"},
            ],
        }
        return build_from_pathway_graph(pathway_graph)

    def test_dimerization_not_fallback_to_activation(self):
        """B4: dimerization 不被 fallback 到 ACTIVATION。"""
        ir = self._build_pathway_graph_with_mechanism("dimerization")
        self.assertEqual(len(ir.reactions), 1)
        self.assertEqual(ir.reactions[0].reaction_type, "dimerization")

    def test_complex_formation_not_fallback_to_activation(self):
        """B4: complex_formation 不被 fallback 到 ACTIVATION。"""
        ir = self._build_pathway_graph_with_mechanism("complex_formation")
        self.assertEqual(len(ir.reactions), 1)
        self.assertEqual(ir.reactions[0].reaction_type, "complex_formation")

    def test_sequestration_not_fallback_to_activation(self):
        """B4: sequestration 不被 fallback 到 ACTIVATION。"""
        ir = self._build_pathway_graph_with_mechanism("sequestration")
        self.assertEqual(len(ir.reactions), 1)
        self.assertEqual(ir.reactions[0].reaction_type, "sequestration")

    def test_gtp_gdp_exchange_not_fallback_to_activation(self):
        """B4: gtp_gdp_exchange 不被 fallback 到 ACTIVATION。"""
        ir = self._build_pathway_graph_with_mechanism("gtp_gdp_exchange")
        self.assertEqual(len(ir.reactions), 1)
        self.assertEqual(ir.reactions[0].reaction_type, "gtp_gdp_exchange")

    def test_nuclear_import_not_fallback_to_activation(self):
        """B4: nuclear_import 不被 fallback 到 ACTIVATION。"""
        ir = self._build_pathway_graph_with_mechanism("nuclear_import")
        self.assertEqual(len(ir.reactions), 1)
        self.assertEqual(ir.reactions[0].reaction_type, "nuclear_import")

    def test_nuclear_export_not_fallback_to_activation(self):
        """B4: nuclear_export 不被 fallback 到 ACTIVATION。"""
        ir = self._build_pathway_graph_with_mechanism("nuclear_export")
        self.assertEqual(len(ir.reactions), 1)
        self.assertEqual(ir.reactions[0].reaction_type, "nuclear_export")

    def test_cytoplasm_translocation_not_fallback_to_activation(self):
        """B4: cytoplasm_translocation 不被 fallback 到 ACTIVATION。"""
        ir = self._build_pathway_graph_with_mechanism("cytoplasm_translocation")
        self.assertEqual(len(ir.reactions), 1)
        self.assertEqual(
            ir.reactions[0].reaction_type, "cytoplasm_translocation"
        )

    def test_proteasomal_degradation_not_fallback_to_activation(self):
        """B4: proteasomal_degradation 不被 fallback 到 ACTIVATION。"""
        ir = self._build_pathway_graph_with_mechanism("proteasomal_degradation")
        self.assertEqual(len(ir.reactions), 1)
        self.assertEqual(
            ir.reactions[0].reaction_type, "proteasomal_degradation"
        )

    def test_invalid_mechanism_fallback_to_activation_with_warning(self):
        """B4: 无效 mechanism 字符串优雅降级到 ACTIVATION 并记录 warning。"""
        ir = self._build_pathway_graph_with_mechanism("invalid_mechanism_xyz")
        self.assertEqual(len(ir.reactions), 1)
        self.assertEqual(ir.reactions[0].reaction_type, "activation")
        # 应有 warning
        self.assertTrue(
            any("invalid_mechanism_xyz" in w for w in ir.warnings),
            f"应有 warning 记录未知 mechanism，实际 warnings={ir.warnings}"
        )

    def test_mechanism_case_insensitive(self):
        """B4: mechanism 字符串大小写不敏感（'DIMERIZATION' → dimerization）。"""
        ir = self._build_pathway_graph_with_mechanism("DIMERIZATION")
        self.assertEqual(len(ir.reactions), 1)
        self.assertEqual(ir.reactions[0].reaction_type, "dimerization")

    def test_mechanism_with_whitespace(self):
        """B4: mechanism 字符串前后空白被 strip。"""
        ir = self._build_pathway_graph_with_mechanism("  dimerization  ")
        self.assertEqual(len(ir.reactions), 1)
        self.assertEqual(ir.reactions[0].reaction_type, "dimerization")

    def test_phosphorylation_in_pathway_graph_uses_b3_fix(self):
        """B4 重写后 PHOSPHORYLATION 仍应用 B3 修复（自磷酸化场景）。"""
        from app.reaction_ir_v2.reaction_builder import build_from_pathway_graph

        pathway_graph = {
            "nodes": [
                {"id": "EGFR", "name": "EGFR", "type": "receptor"},
                {"id": "pEGFR", "name": "pEGFR", "type": "protein"},
            ],
            "edges": [
                {"source": "EGFR", "target": "pEGFR",
                 "mechanism": "phosphorylation", "pathway_tag": "EGFR_RTK"},
            ],
        }
        ir = build_from_pathway_graph(pathway_graph)
        rxn = ir.reactions[0]
        self.assertEqual(rxn.reaction_type, "phosphorylation")
        # 自磷酸化：无 modifier
        self.assertEqual(len(rxn.modifiers), 0,
                         "自磷酸化不应有 modifier（B3 修复应在 pathway_graph 入口生效）")


# =============================================================================
# B4: cross_talk_edges 处理
# =============================================================================
class TestB4CrossTalkEdges(unittest.TestCase):
    """B4 重写后 cross_talk_edges 仍正常处理。"""

    def test_cross_talk_edges_appended_to_main_edges(self):
        """cross_talk_edges 应被追加到主 edges 列表并构建反应。"""
        from app.reaction_ir_v2.reaction_builder import build_from_pathway_graph

        pathway_graph = {
            "nodes": [
                {"id": "A", "name": "A", "type": "protein"},
                {"id": "B", "name": "B", "type": "protein"},
                {"id": "C", "name": "C", "type": "protein"},
            ],
            "edges": [
                {"source": "A", "target": "B", "mechanism": "activation"},
            ],
            "cross_talk_edges": [
                {"source": "A", "target": "C", "mechanism": "dimerization"},
            ],
        }
        ir = build_from_pathway_graph(pathway_graph)
        # 主 edges 1 + cross_talk 1 = 2 reactions
        self.assertEqual(len(ir.reactions), 2)
        # cross_talk edge 机制应保留为 dimerization（不被 fallback）
        dimer_rxns = [r for r in ir.reactions if r.reaction_type == "dimerization"]
        self.assertEqual(len(dimer_rxns), 1,
                         "cross_talk_edges 中的 dimerization 应被保留")


if __name__ == "__main__":
    unittest.main(verbosity=2)
