# BioDynamics Agent v4 - PI3K-AKT-mTOR Specialist 单元测试 (Phase 4 / Task 4.5)
#
# 测试用例（11 项，覆盖 5 模块 + flag 隔离 + 元数据 + PIP2/PIP3 守恒 + AKT 共享）：
#   1. Specialist 注册：get_specialist("PI3K_AKT_mTOR") 返回 PI3KAKTmTORSpecialist 实例
#   2. Core 模块：apply_core() 返回 9 条核心反应
#   3. PIP2/PIP3 质量守恒约束：Core 输出含 Constraint type="mass_conservation" expression 含 PIP2/PIP3
#   4. AKT shared 标记：species 中 AKT 标记 shared=True
#   5. Feedback 模块：apply_feedback() 返回 3 条反馈环（pS6K→IRS1 delay=30 / mTORC1→ULK1 / pAKT→mTORC2）
#   6. Crosstalk 模块：apply_crosstalk() 返回 4 条 cross-talk Reaction 片段
#   7. Perturbation 模块：apply_perturbation() 返回 6 个扰动
#   8. Validation 模块：apply_validation() 返回 3 条 benchmark（含 pmid）
#   9. 模板选择：phosphorylation → _mechanism_phosphorylation_mm；bistable → bistable_switch
#  10. Feature Flag 隔离：flag=false 时 hook 不调用 Specialist
#  11. 元数据：get_metadata() 返回 pathway_class / supported_modules 5 项
#
# 运行：cd backend && python -m pytest tests/test_pi3k_specialist.py -v

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# 确保能导入 app 包（backend/ 在 sys.path）
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


# =============================================================================
# 辅助：模拟 LangGraph specialist hook（验证 Feature Flag 隔离）
# =============================================================================
def _specialist_hook_mock(state, specialist_instance):
    """模拟 LangGraph specialist hook（测试用，验证 feature flag 隔离）。

    行为：
    - V4_PATHWAY_SPECIALIST_ENABLED=false：直接返回 {}，不调用 specialist
    - V4_PATHWAY_SPECIALIST_ENABLED=true：调用 specialist.apply_core 并返回结果
    """
    from app.config import settings

    if not getattr(settings, "V4_PATHWAY_SPECIALIST_ENABLED", False):
        return {}

    pathway_graph = state.get("v4_pathway_graph", {})
    return {"v4_specialist_core": specialist_instance.apply_core(pathway_graph)}


# =============================================================================
# 测试类
# =============================================================================
class TestPI3KSpecialistRegistration(unittest.TestCase):
    """测试 1：Specialist 注册到 SPECIALIST_REGISTRY。"""

    def setUp(self):
        """每个测试开始前清空 registry 并重新注册 PI3KAKTmTORSpecialist。"""
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.pi3k_akt_mtor_specialist import (
            PI3KAKTmTORSpecialist,
        )

        clear_registry()
        register_specialist(PI3KAKTmTORSpecialist)

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_get_specialist_returns_pi3k_instance(self):
        """get_specialist("PI3K_AKT_mTOR") 返回 PI3KAKTmTORSpecialist 实例。"""
        from app.pathways.pathway_registry import (
            get_specialist,
            is_specialist_available,
        )
        from app.pathways.specialists.pi3k_akt_mtor_specialist import (
            PI3KAKTmTORSpecialist,
        )

        # 已注册
        self.assertTrue(is_specialist_available("PI3K_AKT_mTOR"))

        # get_specialist 返回 PI3KAKTmTORSpecialist 实例
        instance = get_specialist("PI3K_AKT_mTOR")
        self.assertIsNotNone(instance)
        self.assertIsInstance(instance, PI3KAKTmTORSpecialist)

        # 多次调用返回不同实例（非单例）
        instance2 = get_specialist("PI3K_AKT_mTOR")
        self.assertIsNot(instance, instance2)

    def test_pathway_class_attribute(self):
        """PI3KAKTmTORSpecialist.pathway_class == 'PI3K_AKT_mTOR'。"""
        from app.pathways.specialists.pi3k_akt_mtor_specialist import (
            PI3KAKTmTORSpecialist,
        )

        self.assertEqual(PI3KAKTmTORSpecialist.pathway_class, "PI3K_AKT_mTOR")


class TestPI3KCoreModule(unittest.TestCase):
    """测试 2：Core 模块返回 9 条核心反应。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.pi3k_akt_mtor_specialist import (
            PI3KAKTmTORSpecialist,
        )

        clear_registry()
        register_specialist(PI3KAKTmTORSpecialist)
        self._PI3KSpecialist = PI3KAKTmTORSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_apply_core_returns_9_reactions(self):
        """apply_core() 返回 9 条核心反应。"""
        specialist = self._PI3KSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})

        self.assertIn("species", result)
        self.assertIn("reactions", result)
        reactions = result["reactions"]
        self.assertEqual(
            len(reactions),
            9,
            f"期望 9 条核心反应，实际 {len(reactions)} 条",
        )

    def test_core_reaction_pairs(self):
        """9 条核心反应的 source→target 对应正确（PI3K→PIP3→pAKT→...→PTEN→PIP2）。"""
        specialist = self._PI3KSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        pairs = [(r["source"], r["target"]) for r in result["reactions"]]
        expected_pairs = [
            ("PI3K", "PIP3"),       # 1. PI3K 催化 PIP2→PIP3
            ("PIP3", "pAKT"),       # 2. PIP3 招募 AKT 被磷酸化
            ("PDK1", "pAKT"),       # 3. PDK1 磷酸化 AKT Thr308
            ("pAKT", "pTSC2"),      # 4. pAKT 磷酸化 TSC2
            ("pTSC2", "RhebGTP"),   # 5. pTSC2 失活，Rheb 累积 GTP
            ("RhebGTP", "mTORC1"),  # 6. RhebGTP 激活 mTORC1
            ("mTORC1", "pS6K"),     # 7. mTORC1 磷酸化 S6K
            ("mTORC1", "p4EBP1"),   # 8. mTORC1 磷酸化 4E-BP1
            ("PTEN", "PIP2"),       # 9. PTEN 去磷酸化 PIP3→PIP2
        ]
        self.assertEqual(pairs, expected_pairs)

    def test_core_reactions_contain_pathway_tag(self):
        """每条核心反应含 pathway_tag='PI3K_AKT_mTOR'。"""
        specialist = self._PI3KSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        for r in result["reactions"]:
            self.assertEqual(r["pathway_tag"], "PI3K_AKT_mTOR")

    def test_core_reactions_contain_required_fields(self):
        """每条核心反应含 source / target / mechanism / kinetics_type /
        substrate / product / modifier / modifier_type。"""
        specialist = self._PI3KSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        for r in result["reactions"]:
            self.assertIn("source", r)
            self.assertIn("target", r)
            self.assertIn("mechanism", r)
            self.assertIn("kinetics_type", r)
            self.assertIn("substrate", r)
            self.assertIn("product", r)
            self.assertIn("modifier", r)
            self.assertIn("modifier_type", r)

    def test_core_hetero_phosphorylation_semantics(self):
        """异磷酸化反应（pAKT→pTSC2 / mTORC1→pS6K / mTORC1→p4EBP1）
        未磷酸化形式作 substrate，磷酸化形式作 product，source 作 catalytic modifier。"""
        specialist = self._PI3KSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        reactions = result["reactions"]

        # pAKT → pTSC2（第 4 条，index=3）
        rxn = reactions[3]
        self.assertEqual(rxn["source"], "pAKT")
        self.assertEqual(rxn["target"], "pTSC2")
        self.assertFalse(rxn["autophosphorylation"])
        self.assertEqual(rxn["substrate"], "TSC2")
        self.assertEqual(rxn["product"], "pTSC2")
        self.assertEqual(rxn["modifier"], "pAKT")
        self.assertEqual(rxn["modifier_type"], "catalytic")

        # mTORC1 → pS6K（第 7 条，index=6）
        rxn = reactions[6]
        self.assertEqual(rxn["source"], "mTORC1")
        self.assertEqual(rxn["target"], "pS6K")
        self.assertFalse(rxn["autophosphorylation"])
        self.assertEqual(rxn["substrate"], "S6K")
        self.assertEqual(rxn["product"], "pS6K")
        self.assertEqual(rxn["modifier"], "mTORC1")
        self.assertEqual(rxn["modifier_type"], "catalytic")

        # mTORC1 → p4EBP1（第 8 条，index=7）
        rxn = reactions[7]
        self.assertEqual(rxn["source"], "mTORC1")
        self.assertEqual(rxn["target"], "p4EBP1")
        self.assertFalse(rxn["autophosphorylation"])
        self.assertEqual(rxn["substrate"], "4E-BP1")
        self.assertEqual(rxn["product"], "p4EBP1")
        self.assertEqual(rxn["modifier"], "mTORC1")
        self.assertEqual(rxn["modifier_type"], "catalytic")


class TestPI3KMassConservationConstraint(unittest.TestCase):
    """测试 3：PIP2/PIP3 质量守恒约束。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.pi3k_akt_mtor_specialist import (
            PI3KAKTmTORSpecialist,
        )

        clear_registry()
        register_specialist(PI3KAKTmTORSpecialist)
        self._PI3KSpecialist = PI3KAKTmTORSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_core_contains_mass_conservation_constraint(self):
        """Core 输出含 Constraint type='mass_conservation'。"""
        specialist = self._PI3KSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})

        self.assertIn("constraints", result)
        constraints = result["constraints"]
        self.assertGreater(
            len(constraints), 0, "Core 输出应至少含 1 条约束"
        )

        # 查找 mass_conservation 约束
        mc = next(
            (c for c in constraints if c["type"] == "mass_conservation"),
            None,
        )
        self.assertIsNotNone(mc, "应含 type='mass_conservation' 约束")

    def test_mass_conservation_expression_contains_pip2_pip3(self):
        """mass_conservation 约束 expression 含 PIP2 与 PIP3。"""
        specialist = self._PI3KSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        constraints = result["constraints"]

        mc = next(
            (c for c in constraints if c["type"] == "mass_conservation"),
            None,
        )
        self.assertIsNotNone(mc)
        expr = mc["expression"]
        self.assertIn(
            "PIP2", expr, "mass_conservation expression 应含 PIP2"
        )
        self.assertIn(
            "PIP3", expr, "mass_conservation expression 应含 PIP3"
        )
        # expression 应为守恒形式（含 = 与 _total）
        self.assertIn("=", expr)
        self.assertIn("PIP_total", expr)

    def test_mass_conservation_constraint_fields(self):
        """mass_conservation 约束含 type / scope / expression / tolerance。"""
        specialist = self._PI3KSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        constraints = result["constraints"]

        mc = next(
            (c for c in constraints if c["type"] == "mass_conservation"),
            None,
        )
        self.assertIsNotNone(mc)
        self.assertEqual(mc["type"], "mass_conservation")
        self.assertEqual(mc["scope"], "species")
        self.assertIn("expression", mc)
        self.assertIn("tolerance", mc)


class TestPI3KAktSharedMarker(unittest.TestCase):
    """测试 4：AKT shared 标记。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.pi3k_akt_mtor_specialist import (
            PI3KAKTmTORSpecialist,
        )

        clear_registry()
        register_specialist(PI3KAKTmTORSpecialist)
        self._PI3KSpecialist = PI3KAKTmTORSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_akt_marked_as_shared(self):
        """species 中 AKT 标记 shared=True。"""
        specialist = self._PI3KSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        species = result["species"]

        akt = next(
            (s for s in species if s["name"] == "AKT"),
            None,
        )
        self.assertIsNotNone(akt, "Core 输出应含 AKT 物种")
        self.assertTrue(
            akt.get("shared"),
            "AKT 应标记 shared=True（与 Apoptosis Bad / p53 Mdm2 路径共享）",
        )

    def test_mtroc1_marked_as_shared(self):
        """mTORC1 也标记 shared=True（与 Apoptosis 自噬 / Cell Cycle 共享）。"""
        specialist = self._PI3KSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        species = result["species"]

        mtorc1 = next(
            (s for s in species if s["name"] == "mTORC1"),
            None,
        )
        self.assertIsNotNone(mtorc1, "Core 输出应含 mTORC1 物种")
        self.assertTrue(
            mtorc1.get("shared"),
            "mTORC1 应标记 shared=True",
        )


class TestPI3KFeedbackModule(unittest.TestCase):
    """测试 5：Feedback 模块返回 3 条反馈环。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.pi3k_akt_mtor_specialist import (
            PI3KAKTmTORSpecialist,
        )

        clear_registry()
        register_specialist(PI3KAKTmTORSpecialist)
        self._PI3KSpecialist = PI3KAKTmTORSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_apply_feedback_returns_3_loops(self):
        """apply_feedback() 返回 3 条反馈环。"""
        specialist = self._PI3KSpecialist()
        loops = specialist.apply_feedback({"nodes": [], "edges": []})

        self.assertEqual(
            len(loops),
            3,
            f"期望 3 条反馈环，实际 {len(loops)} 条",
        )

    def test_s6k_irs1_feedback_delay_30(self):
        """pS6K → IRS1 负反馈 delay=30 min。"""
        specialist = self._PI3KSpecialist()
        loops = specialist.apply_feedback({"nodes": [], "edges": []})

        irs1 = next(
            (l for l in loops if "IRS1" in l["id"]),
            None,
        )
        self.assertIsNotNone(irs1, "应含 pS6K→IRS1 反馈环")
        self.assertEqual(irs1["loop_type"], "negative")
        self.assertEqual(irs1["delay_minutes"], 30.0)
        # node_ids 应含 pS6K
        self.assertIn("pS6K", irs1["node_ids"])

    def test_mtroc1_ulk1_feedback_delay_zero(self):
        """mTORC1 → ULK1 负反馈 delay=0。"""
        specialist = self._PI3KSpecialist()
        loops = specialist.apply_feedback({"nodes": [], "edges": []})

        ulk1 = next(
            (l for l in loops if "ULK1" in l["id"]),
            None,
        )
        self.assertIsNotNone(ulk1, "应含 mTORC1→ULK1 反馈环")
        self.assertEqual(ulk1["loop_type"], "negative")
        self.assertEqual(ulk1["delay_minutes"], 0.0)
        # node_ids 应含 mTORC1 与 ULK1
        self.assertIn("mTORC1", ulk1["node_ids"])
        self.assertIn("ULK1", ulk1["node_ids"])

    def test_akt_mtroc2_positive_feedback(self):
        """pAKT → mTORC2 正反馈。"""
        specialist = self._PI3KSpecialist()
        loops = specialist.apply_feedback({"nodes": [], "edges": []})

        mtorc2 = next(
            (l for l in loops if "mTORC2" in l["id"]),
            None,
        )
        self.assertIsNotNone(mtorc2, "应含 pAKT→mTORC2 反馈环")
        self.assertEqual(
            mtorc2["loop_type"],
            "positive",
            "pAKT→mTORC2 应为正反馈",
        )
        # node_ids 应含 pAKT 与 mTORC2
        self.assertIn("pAKT", mtorc2["node_ids"])
        self.assertIn("mTORC2", mtorc2["node_ids"])

    def test_feedback_loops_contain_required_fields(self):
        """每条反馈环含 id / loop_type / node_ids / delay_minutes / description。"""
        specialist = self._PI3KSpecialist()
        loops = specialist.apply_feedback({"nodes": [], "edges": []})

        for loop in loops:
            self.assertIn("id", loop)
            self.assertIn("loop_type", loop)
            self.assertIn("node_ids", loop)
            self.assertIn("delay_minutes", loop)
            self.assertIn("description", loop)


class TestPI3KCrosstalkModule(unittest.TestCase):
    """测试 6：Crosstalk 模块返回 4 条 cross-talk Reaction 片段。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.pi3k_akt_mtor_specialist import (
            PI3KAKTmTORSpecialist,
        )

        clear_registry()
        register_specialist(PI3KAKTmTORSpecialist)
        self._PI3KSpecialist = PI3KAKTmTORSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_apply_crosstalk_returns_4_fragments(self):
        """apply_crosstalk() 返回 4 条 cross-talk Reaction 片段。"""
        specialist = self._PI3KSpecialist()
        fragments = specialist.apply_crosstalk(
            {"nodes": [], "edges": []}, []
        )

        self.assertEqual(
            len(fragments),
            4,
            f"期望 4 条 cross-talk 片段，实际 {len(fragments)} 条",
        )

    def test_crosstalk_contains_raf_inhibition(self):
        """含 pAKT → Raf Ser259 inhibition 片段。"""
        specialist = self._PI3KSpecialist()
        fragments = specialist.apply_crosstalk(
            {"nodes": [], "edges": []}, []
        )

        raf = next(
            (f for f in fragments if f["target"] == "Raf"),
            None,
        )
        self.assertIsNotNone(raf)
        self.assertEqual(raf["source"], "pAKT")
        self.assertEqual(raf["mechanism"], "inhibition")
        self.assertEqual(raf.get("site"), "Ser259")

    def test_crosstalk_contains_bad_inhibition(self):
        """含 pAKT → Bad inhibition 片段。"""
        specialist = self._PI3KSpecialist()
        fragments = specialist.apply_crosstalk(
            {"nodes": [], "edges": []}, []
        )

        bad = next(
            (f for f in fragments if f["target"] == "Bad"),
            None,
        )
        self.assertIsNotNone(bad)
        self.assertEqual(bad["source"], "pAKT")
        self.assertEqual(bad["mechanism"], "inhibition")
        self.assertEqual(bad.get("site"), "Ser136")

    def test_crosstalk_contains_mdm2_activation(self):
        """含 pAKT → Mdm2 activation 片段。"""
        specialist = self._PI3KSpecialist()
        fragments = specialist.apply_crosstalk(
            {"nodes": [], "edges": []}, []
        )

        mdm2 = next(
            (f for f in fragments if f["target"] == "Mdm2"),
            None,
        )
        self.assertIsNotNone(mdm2)
        self.assertEqual(mdm2["source"], "pAKT")
        self.assertEqual(mdm2["mechanism"], "activation")
        self.assertEqual(mdm2.get("site"), "Ser166")

    def test_crosstalk_contains_hif1a_activation(self):
        """含 mTORC1 → HIF-1α activation 片段。"""
        specialist = self._PI3KSpecialist()
        fragments = specialist.apply_crosstalk(
            {"nodes": [], "edges": []}, []
        )

        hif = next(
            (f for f in fragments if f["target"] == "HIF-1α"),
            None,
        )
        self.assertIsNotNone(hif)
        self.assertEqual(hif["source"], "mTORC1")
        self.assertEqual(hif["mechanism"], "activation")

    def test_crosstalk_fragments_contain_required_fields(self):
        """每条 cross-talk 片段含 source / target / mechanism / shared_species / description。"""
        specialist = self._PI3KSpecialist()
        fragments = specialist.apply_crosstalk(
            {"nodes": [], "edges": []}, []
        )

        for f in fragments:
            self.assertIn("source", f)
            self.assertIn("target", f)
            self.assertIn("mechanism", f)
            self.assertIn("shared_species", f)
            self.assertIn("description", f)


class TestPI3KPerturbationModule(unittest.TestCase):
    """测试 7：Perturbation 模块返回 6 个扰动。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.pi3k_akt_mtor_specialist import (
            PI3KAKTmTORSpecialist,
        )

        clear_registry()
        register_specialist(PI3KAKTmTORSpecialist)
        self._PI3KSpecialist = PI3KAKTmTORSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_apply_perturbation_returns_6(self):
        """apply_perturbation() 返回 6 个扰动。"""
        specialist = self._PI3KSpecialist()
        perturbations = specialist.apply_perturbation(
            {"nodes": [], "edges": []}, []
        )

        self.assertEqual(
            len(perturbations),
            6,
            f"期望 6 个扰动，实际 {len(perturbations)} 个",
        )

    def test_perturbation_contains_rapamycin(self):
        """含 Rapamycin（mTORC1 抑制剂）。"""
        specialist = self._PI3KSpecialist()
        perturbations = specialist.apply_perturbation(
            {"nodes": [], "edges": []}, []
        )

        rapa = next(
            (p for p in perturbations if p.get("drug") == "Rapamycin"),
            None,
        )
        self.assertIsNotNone(rapa)
        self.assertEqual(rapa["target"], "mTORC1")
        self.assertEqual(rapa["mechanism"], "inhibition")

    def test_perturbation_contains_everolimus(self):
        """含 Everolimus（mTORC1 抑制剂）。"""
        specialist = self._PI3KSpecialist()
        perturbations = specialist.apply_perturbation(
            {"nodes": [], "edges": []}, []
        )

        ever = next(
            (p for p in perturbations if p.get("drug") == "Everolimus"),
            None,
        )
        self.assertIsNotNone(ever)
        self.assertEqual(ever["target"], "mTORC1")
        self.assertEqual(ever["mechanism"], "inhibition")

    def test_perturbation_contains_bkm120(self):
        """含 BKM120（PI3K 抑制剂）。"""
        specialist = self._PI3KSpecialist()
        perturbations = specialist.apply_perturbation(
            {"nodes": [], "edges": []}, []
        )

        bkm = next(
            (p for p in perturbations if p.get("drug") == "BKM120"),
            None,
        )
        self.assertIsNotNone(bkm)
        self.assertEqual(bkm["target"], "PI3K")
        self.assertEqual(bkm["mechanism"], "inhibition")

    def test_perturbation_contains_idelalisib(self):
        """含 Idelalisib（PI3Kδ 抑制剂）。"""
        specialist = self._PI3KSpecialist()
        perturbations = specialist.apply_perturbation(
            {"nodes": [], "edges": []}, []
        )

        ide = next(
            (p for p in perturbations if p.get("drug") == "Idelalisib"),
            None,
        )
        self.assertIsNotNone(ide)
        self.assertEqual(ide["target"], "PI3K")
        self.assertEqual(ide["mechanism"], "inhibition")

    def test_perturbation_contains_mk2206(self):
        """含 MK-2206（AKT 抑制剂）。"""
        specialist = self._PI3KSpecialist()
        perturbations = specialist.apply_perturbation(
            {"nodes": [], "edges": []}, []
        )

        mk = next(
            (p for p in perturbations if p.get("drug") == "MK-2206"),
            None,
        )
        self.assertIsNotNone(mk)
        self.assertEqual(mk["target"], "AKT")
        self.assertEqual(mk["mechanism"], "inhibition")

    def test_perturbation_contains_pten_loss(self):
        """含 PTEN loss（PTEN 缺失突变）。"""
        specialist = self._PI3KSpecialist()
        perturbations = specialist.apply_perturbation(
            {"nodes": [], "edges": []}, []
        )

        pten = next(
            (p for p in perturbations if p.get("ko_target") == "PTEN_loss"),
            None,
        )
        self.assertIsNotNone(pten)
        self.assertEqual(pten["target"], "PTEN")
        self.assertEqual(pten["mechanism"], "knockout")

    def test_perturbations_contain_required_fields(self):
        """每个扰动含 target / drug / mechanism / ko_target / description。"""
        specialist = self._PI3KSpecialist()
        perturbations = specialist.apply_perturbation(
            {"nodes": [], "edges": []}, []
        )

        for p in perturbations:
            self.assertIn("target", p)
            self.assertIn("drug", p)
            self.assertIn("mechanism", p)
            self.assertIn("ko_target", p)
            self.assertIn("description", p)


class TestPI3KValidationModule(unittest.TestCase):
    """测试 8：Validation 模块返回 3 条 benchmark。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.pi3k_akt_mtor_specialist import (
            PI3KAKTmTORSpecialist,
        )

        clear_registry()
        register_specialist(PI3KAKTmTORSpecialist)
        self._PI3KSpecialist = PI3KAKTmTORSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_apply_validation_returns_3_rules(self):
        """apply_validation() 返回 3 条 Validation 规则。"""
        specialist = self._PI3KSpecialist()
        rules = specialist.apply_validation()

        self.assertEqual(
            len(rules),
            3,
            f"期望 3 条 Validation 规则，实际 {len(rules)} 条",
        )

    def test_pakt_peak_time_rule(self):
        """pAKT 达峰时间 30-60 min（PMID:19211571）。"""
        specialist = self._PI3KSpecialist()
        rules = specialist.apply_validation()

        peak = next(
            (r for r in rules if r["metric_name"] == "pAKT_peak_time"),
            None,
        )
        self.assertIsNotNone(peak)
        self.assertEqual(peak["expected_min"], 30.0)
        self.assertEqual(peak["expected_max"], 60.0)
        self.assertEqual(peak["unit"], "minutes")
        self.assertEqual(peak["pmid"], "PMID:19211571")

    def test_pip_pip3_mass_conservation_rule(self):
        """PIP2/PIP3 质量守恒比例 ≈1.0（tolerance 0.05）。"""
        specialist = self._PI3KSpecialist()
        rules = specialist.apply_validation()

        mc = next(
            (
                r
                for r in rules
                if r["metric_name"] == "PIP_PIP3_mass_conservation"
            ),
            None,
        )
        self.assertIsNotNone(mc)
        self.assertEqual(mc["expected"], 1.0)
        self.assertEqual(mc["tolerance"], 0.05)
        self.assertEqual(mc["unit"], "ratio")

    def test_s6k1_peak_delay_vs_akt_rule(self):
        """S6K1 达峰延迟 vs AKT 30-60 min（PMID:19211571）。"""
        specialist = self._PI3KSpecialist()
        rules = specialist.apply_validation()

        delay = next(
            (
                r
                for r in rules
                if r["metric_name"] == "S6K1_peak_delay_vs_AKT"
            ),
            None,
        )
        self.assertIsNotNone(delay)
        self.assertEqual(delay["expected_min"], 30.0)
        self.assertEqual(delay["expected_max"], 60.0)
        self.assertEqual(delay["unit"], "minutes_delay")
        self.assertEqual(delay["pmid"], "PMID:19211571")

    def test_validation_rules_contain_required_fields(self):
        """每条 Validation 规则含 rule_id / metric_name / expected /
        tolerance / pmid / description。"""
        specialist = self._PI3KSpecialist()
        rules = specialist.apply_validation()

        for r in rules:
            self.assertIn("rule_id", r)
            self.assertIn("metric_name", r)
            self.assertIn("expected", r)
            self.assertIn("tolerance", r)
            self.assertIn("pmid", r)
            self.assertIn("description", r)


class TestPI3KSelectTemplate(unittest.TestCase):
    """测试 9：模板选择（phosphorylation / bistable）。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.pi3k_akt_mtor_specialist import (
            PI3KAKTmTORSpecialist,
        )

        clear_registry()
        register_specialist(PI3KAKTmTORSpecialist)
        self._PI3KSpecialist = PI3KAKTmTORSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_select_template_phosphorylation(self):
        """select_template('phosphorylation') 返回 _mechanism_phosphorylation_mm。"""
        specialist = self._PI3KSpecialist()
        self.assertEqual(
            specialist.select_template("phosphorylation"),
            "_mechanism_phosphorylation_mm",
        )

    def test_select_template_bistable(self):
        """select_template('bistable') 返回 bistable_switch（mTORC1 双稳态）。"""
        specialist = self._PI3KSpecialist()
        self.assertEqual(
            specialist.select_template("bistable"),
            "bistable_switch",
        )


class TestPI3KFeatureFlagIsolation(unittest.TestCase):
    """测试 10：Feature Flag 隔离（flag=false 时 hook 不调用 Specialist）。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.pi3k_akt_mtor_specialist import (
            PI3KAKTmTORSpecialist,
        )

        clear_registry()
        register_specialist(PI3KAKTmTORSpecialist)
        self._PI3KSpecialist = PI3KAKTmTORSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_flag_false_hook_does_not_call_specialist(self):
        """flag=false 时 hook 返回空 dict 且不调用 Specialist 任何方法。"""
        specialist = MagicMock()
        specialist.apply_core.return_value = {
            "species": [],
            "reactions": [],
            "constraints": [],
        }
        state = {"v4_pathway_graph": {"nodes": [], "edges": []}}

        with patch("app.config.settings.V4_PATHWAY_SPECIALIST_ENABLED", False):
            result = _specialist_hook_mock(state, specialist)

        self.assertEqual(result, {})
        specialist.apply_core.assert_not_called()

    def test_flag_true_hook_calls_specialist(self):
        """flag=true 时 hook 调用 Specialist.apply_core 并返回结果。"""
        specialist = MagicMock()
        expected_core = {
            "species": [{"name": "AKT"}],
            "reactions": [],
            "constraints": [],
        }
        specialist.apply_core.return_value = expected_core
        state = {"v4_pathway_graph": {"nodes": [{"id": "AKT"}], "edges": []}}

        with patch("app.config.settings.V4_PATHWAY_SPECIALIST_ENABLED", True):
            result = _specialist_hook_mock(state, specialist)

        self.assertEqual(result, {"v4_specialist_core": expected_core})
        specialist.apply_core.assert_called_once()

    def test_flag_default_false(self):
        """settings.V4_PATHWAY_SPECIALIST_ENABLED 默认为 False。"""
        from app.config import settings

        self.assertFalse(
            settings.V4_PATHWAY_SPECIALIST_ENABLED,
            "V4_PATHWAY_SPECIALIST_ENABLED 应默认为 False",
        )


class TestPI3KMetadataAndInputValidation(unittest.TestCase):
    """测试 11：元数据 + 输入校验。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.pi3k_akt_mtor_specialist import (
            PI3KAKTmTORSpecialist,
        )

        clear_registry()
        register_specialist(PI3KAKTmTORSpecialist)
        self._PI3KSpecialist = PI3KAKTmTORSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_get_metadata_returns_pi3k_metadata(self):
        """get_metadata() 返回 pathway_class='PI3K_AKT_mTOR' + 5 模块。"""
        specialist = self._PI3KSpecialist()
        metadata = specialist.get_metadata()

        self.assertEqual(metadata["pathway_class"], "PI3K_AKT_mTOR")
        self.assertEqual(metadata["display_name"], "PI3K / AKT / mTOR Signaling")
        self.assertEqual(
            metadata["supported_modules"],
            ["core", "feedback", "crosstalk", "perturbation", "validation"],
        )
        self.assertEqual(len(metadata["supported_modules"]), 5)
        self.assertIn("version", metadata)

    def test_validate_input_empty_graph_returns_warnings(self):
        """validate_input({}) 返回非空 warning 列表。"""
        specialist = self._PI3KSpecialist()
        warnings = specialist.validate_input({})

        self.assertIsInstance(warnings, list)
        self.assertGreater(len(warnings), 0, "空 pathway_graph 应返回 warning")

    def test_validate_input_none_returns_warnings(self):
        """validate_input(None) 返回非空 warning 列表。"""
        specialist = self._PI3KSpecialist()
        warnings = specialist.validate_input(None)

        self.assertGreater(len(warnings), 0)

    def test_validate_input_valid_graph_returns_no_warnings(self):
        """含 nodes + edges 的 pathway_graph 无 warning。"""
        specialist = self._PI3KSpecialist()
        warnings = specialist.validate_input(
            {"nodes": [{"id": "AKT"}], "edges": []}
        )
        self.assertEqual(warnings, [])


if __name__ == "__main__":
    unittest.main()
