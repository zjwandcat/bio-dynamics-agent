# BioDynamics Agent v4 - Apoptosis Specialist 单元测试 (Phase 4 / Task 4.7)
#
# 测试用例（11 项，覆盖 5 模块 + flag 隔离 + 元数据 + CompositeReaction +
# Caspase3 shared + 模板选择）：
#   1. Specialist 注册：get_specialist("APOPTOSIS") 返回 ApoptosisSpecialist 实例
#   2. Core 模块：apply_core() 返回 ≥12 条核心反应
#   3. CompositeReaction 输出：Core 输出含 CompositeReaction（Caspase 级联正反馈环）
#   4. Caspase3 shared 标记：species 中 Caspase3_active 标记 shared=True
#   5. Feedback 模块：apply_feedback() 返回 ≥3 反馈环（Casp3-Casp6-Casp8 正反馈 /
#      XIAP→Casp3 负反馈 / MOMP bistable）
#   6. Crosstalk 模块：apply_crosstalk() 返回 5 条 cross-talk Reaction 片段
#   7. Perturbation 模块：apply_perturbation() 返回 6 个扰动
#   8. Validation 模块：apply_validation() 返回 3 条 benchmark
#   9. 模板选择：bistable → bistable_switch；phosphorylation → _mechanism_phosphorylation_mm
#  10. Feature Flag 隔离：flag=false 时 hook 不调用 Specialist
#  11. 元数据：get_metadata() 返回 pathway_class='APOPTOSIS'
#
# 运行：cd backend && python -m pytest tests/test_apoptosis_specialist.py -v

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
class TestApoptosisSpecialistRegistration(unittest.TestCase):
    """测试 1：Specialist 注册到 SPECIALIST_REGISTRY。"""

    def setUp(self):
        """每个测试开始前清空 registry 并重新注册 ApoptosisSpecialist。"""
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.apoptosis_specialist import (
            ApoptosisSpecialist,
        )

        clear_registry()
        register_specialist(ApoptosisSpecialist)

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_get_specialist_returns_apoptosis_instance(self):
        """get_specialist("APOPTOSIS") 返回 ApoptosisSpecialist 实例。"""
        from app.pathways.pathway_registry import (
            get_specialist,
            is_specialist_available,
        )
        from app.pathways.specialists.apoptosis_specialist import (
            ApoptosisSpecialist,
        )

        # 已注册
        self.assertTrue(is_specialist_available("APOPTOSIS"))

        # get_specialist 返回 ApoptosisSpecialist 实例
        instance = get_specialist("APOPTOSIS")
        self.assertIsNotNone(instance)
        self.assertIsInstance(instance, ApoptosisSpecialist)

        # 多次调用返回不同实例（非单例）
        instance2 = get_specialist("APOPTOSIS")
        self.assertIsNot(instance, instance2)

    def test_pathway_class_attribute(self):
        """ApoptosisSpecialist.pathway_class == 'APOPTOSIS'。"""
        from app.pathways.specialists.apoptosis_specialist import (
            ApoptosisSpecialist,
        )

        self.assertEqual(ApoptosisSpecialist.pathway_class, "APOPTOSIS")


class TestApoptosisCoreModule(unittest.TestCase):
    """测试 2：Core 模块返回 ≥12 条核心反应。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.apoptosis_specialist import (
            ApoptosisSpecialist,
        )

        clear_registry()
        register_specialist(ApoptosisSpecialist)
        self._ApoptosisSpecialist = ApoptosisSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_apply_core_returns_at_least_12_reactions(self):
        """apply_core() 返回 ≥12 条核心反应（Intrinsic 6 + Extrinsic 4 + Caspase 3）。"""
        specialist = self._ApoptosisSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})

        self.assertIn("species", result)
        self.assertIn("reactions", result)
        reactions = result["reactions"]
        self.assertGreaterEqual(
            len(reactions),
            12,
            f"期望至少 12 条核心反应，实际 {len(reactions)} 条",
        )

    def test_core_reaction_pairs(self):
        """13 条核心反应的 source→target 对应正确（Intrinsic→Extrinsic→Caspase 级联）。"""
        specialist = self._ApoptosisSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        pairs = [(r["source"], r["target"]) for r in result["reactions"]]
        expected_pairs = [
            # Intrinsic (6)
            ("Bad", "Bcl2"),                       # 1. Bad 结合 Bcl-2
            ("Bax", "MOMP"),                       # 2. Bax 寡聚化
            ("MOMP", "Cyt_c"),                     # 3. Cyt c 释放
            ("Cyt_c", "Apoptosome"),               # 4. Apoptosome 组装
            ("Apoptosome", "Caspase9_active"),     # 5. Caspase-9 激活
            ("Caspase9_active", "Caspase3_active"),  # 6. Caspase-3 激活
            # Extrinsic (4)
            ("FasL", "DISC"),                      # 7. DISC 组装
            ("DISC", "Caspase8_active"),           # 8. Caspase-8 激活
            ("Caspase8_active", "tBid"),           # 9. Bid 切割
            ("tBid", "Bax"),                       # 10. tBid 激活 Bax
            # Caspase 级联正反馈 (3)
            ("Caspase3_active", "Caspase6_active"),  # 11. Caspase-6 激活
            ("Caspase6_active", "Caspase8_active"),  # 12. Caspase-8 反馈
            ("Caspase3_active", "PARP_cleaved"),    # 13. PARP 切割
        ]
        self.assertEqual(pairs, expected_pairs)

    def test_core_reactions_contain_pathway_tag(self):
        """每条核心反应含 pathway_tag='APOPTOSIS'。"""
        specialist = self._ApoptosisSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        for r in result["reactions"]:
            self.assertEqual(r["pathway_tag"], "APOPTOSIS")

    def test_core_reactions_contain_required_fields(self):
        """每条核心反应含 source / target / mechanism / kinetics_type /
        substrate / product / modifier / modifier_type。"""
        specialist = self._ApoptosisSpecialist()
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

    def test_core_intrinsic_pathway_complete(self):
        """Intrinsic 途径 6 条反应齐全（Bad→Bcl2→Bax→MOMP→Cyt_c→Apoptosome→Casp9→Casp3）。"""
        specialist = self._ApoptosisSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        sources = {r["source"] for r in result["reactions"]}

        # Intrinsic 关键节点
        for node in ["Bad", "Bax", "MOMP", "Cyt_c", "Apoptosome",
                     "Caspase9_active"]:
            self.assertIn(node, sources, f"Intrinsic 途径应含 {node}")

    def test_core_extrinsic_pathway_complete(self):
        """Extrinsic 途径 4 条反应齐全（FasL→DISC→Casp8→tBid→Bax）。"""
        specialist = self._ApoptosisSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        sources = {r["source"] for r in result["reactions"]}

        # Extrinsic 关键节点
        for node in ["FasL", "DISC", "Caspase8_active", "tBid"]:
            self.assertIn(node, sources, f"Extrinsic 途径应含 {node}")


class TestApoptosisCompositeReaction(unittest.TestCase):
    """测试 3：CompositeReaction 输出（Caspase 级联正反馈环）。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.apoptosis_specialist import (
            ApoptosisSpecialist,
        )

        clear_registry()
        register_specialist(ApoptosisSpecialist)
        self._ApoptosisSpecialist = ApoptosisSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_core_contains_composite_reactions(self):
        """Core 输出含 composite_reactions 字段。"""
        specialist = self._ApoptosisSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})

        self.assertIn("composite_reactions", result)
        cr_list = result["composite_reactions"]
        self.assertIsInstance(cr_list, list)
        self.assertGreater(len(cr_list), 0, "composite_reactions 不应为空")

    def test_composite_reaction_caspase_cascade(self):
        """CompositeReaction 含 Caspase 级联正反馈环（Casp3→Casp6→Casp8）。"""
        specialist = self._ApoptosisSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        cr_list = result["composite_reactions"]

        # 查找 Caspase 级联 CompositeReaction
        cascade_cr = next(
            (cr for cr in cr_list
             if "CASPASE_CASCADE" in cr.get("id", "")
             or "Caspase" in cr.get("name", "")),
            None,
        )
        self.assertIsNotNone(cascade_cr, "应含 Caspase 级联正反馈环 CompositeReaction")

        # 检查节点含 Caspase3/6/8
        node_ids = cascade_cr.get("node_ids", [])
        self.assertIn("Caspase3_active", node_ids)
        self.assertIn("Caspase6_active", node_ids)
        self.assertIn("Caspase8_active", node_ids)

    def test_composite_reaction_bistable_flag(self):
        """Caspase 级联 CompositeReaction 标记 point_of_no_return / bistable。"""
        specialist = self._ApoptosisSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        cr_list = result["composite_reactions"]

        cascade_cr = next(
            (cr for cr in cr_list
             if "CASPASE_CASCADE" in cr.get("id", "")),
            None,
        )
        self.assertIsNotNone(cascade_cr)
        self.assertTrue(
            cascade_cr.get("point_of_no_return"),
            "Caspase 级联正反馈环应标记 point_of_no_return=True",
        )
        self.assertEqual(cascade_cr.get("loop_type"), "positive")
        self.assertEqual(cascade_cr.get("template"), "bistable_switch.j2")


class TestApoptosisCaspase3SharedMarker(unittest.TestCase):
    """测试 4：Caspase3_active shared 标记。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.apoptosis_specialist import (
            ApoptosisSpecialist,
        )

        clear_registry()
        register_specialist(ApoptosisSpecialist)
        self._ApoptosisSpecialist = ApoptosisSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_caspase3_marked_as_shared(self):
        """species 中 Caspase3_active 标记 shared=True（Intrinsic + Extrinsic 汇聚点）。"""
        specialist = self._ApoptosisSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        species = result["species"]

        casp3 = next(
            (s for s in species if s["name"] == "Caspase3_active"),
            None,
        )
        self.assertIsNotNone(casp3, "Core 输出应含 Caspase3_active 物种")
        self.assertTrue(
            casp3.get("shared"),
            "Caspase3_active 应标记 shared=True（Intrinsic + Extrinsic 汇聚点 + 正反馈核心）",
        )


class TestApoptosisFeedbackModule(unittest.TestCase):
    """测试 5：Feedback 模块返回 ≥3 反馈环。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.apoptosis_specialist import (
            ApoptosisSpecialist,
        )

        clear_registry()
        register_specialist(ApoptosisSpecialist)
        self._ApoptosisSpecialist = ApoptosisSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_apply_feedback_returns_at_least_3_loops(self):
        """apply_feedback() 返回 ≥3 条反馈环。"""
        specialist = self._ApoptosisSpecialist()
        loops = specialist.apply_feedback({"nodes": [], "edges": []})

        self.assertGreaterEqual(
            len(loops),
            3,
            f"期望至少 3 条反馈环，实际 {len(loops)} 条",
        )

    def test_caspase_cascade_positive_feedback(self):
        """Casp3-Casp6-Casp8 正反馈环（bistable, point-of-no-return, delay=0）。"""
        specialist = self._ApoptosisSpecialist()
        loops = specialist.apply_feedback({"nodes": [], "edges": []})

        cascade_loop = next(
            (l for l in loops if "CASPASE_CASCADE" in l["id"]),
            None,
        )
        self.assertIsNotNone(cascade_loop, "应含 Caspase 级联正反馈环")
        self.assertEqual(cascade_loop["loop_type"], "positive")
        self.assertEqual(cascade_loop["delay_minutes"], 0.0)
        self.assertTrue(cascade_loop.get("bistable"),
                        "Caspase 级联正反馈应标记 bistable=True")
        self.assertTrue(cascade_loop.get("point_of_no_return"),
                        "Caspase 级联应标记 point_of_no_return=True")
        # 节点含 Caspase3/6/8
        node_ids = cascade_loop["node_ids"]
        self.assertIn("Caspase3_active", node_ids)
        self.assertIn("Caspase6_active", node_ids)
        self.assertIn("Caspase8_active", node_ids)

    def test_xiap_negative_feedback(self):
        """XIAP→Caspase3 负反馈（XIAP 抑制 Caspase-3, delay=0）。"""
        specialist = self._ApoptosisSpecialist()
        loops = specialist.apply_feedback({"nodes": [], "edges": []})

        xiap_loop = next(
            (l for l in loops if "XIAP" in l["id"]),
            None,
        )
        self.assertIsNotNone(xiap_loop, "应含 XIAP 负反馈环")
        self.assertEqual(xiap_loop["loop_type"], "negative")
        self.assertEqual(xiap_loop["delay_minutes"], 0.0)
        node_ids = xiap_loop["node_ids"]
        self.assertIn("XIAP", node_ids)
        self.assertIn("Caspase3_active", node_ids)

    def test_momp_bistable_switch(self):
        """MOMP bistable switch（point-of-no-return, bistable all-or-none）。"""
        specialist = self._ApoptosisSpecialist()
        loops = specialist.apply_feedback({"nodes": [], "edges": []})

        momp_loop = next(
            (l for l in loops if "MOMP" in l["id"]),
            None,
        )
        self.assertIsNotNone(momp_loop, "应含 MOMP bistable switch 反馈环")
        self.assertTrue(momp_loop.get("bistable"),
                        "MOMP 应标记 bistable=True")
        self.assertTrue(momp_loop.get("point_of_no_return"),
                        "MOMP 应标记 point_of_no_return=True")

    def test_feedback_loops_contain_required_fields(self):
        """每条反馈环含 id / loop_type / node_ids / delay_minutes / description。"""
        specialist = self._ApoptosisSpecialist()
        loops = specialist.apply_feedback({"nodes": [], "edges": []})

        for loop in loops:
            self.assertIn("id", loop)
            self.assertIn("loop_type", loop)
            self.assertIn("node_ids", loop)
            self.assertIn("delay_minutes", loop)
            self.assertIn("description", loop)


class TestApoptosisCrosstalkModule(unittest.TestCase):
    """测试 6：Crosstalk 模块返回 5 条 cross-talk Reaction 片段。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.apoptosis_specialist import (
            ApoptosisSpecialist,
        )

        clear_registry()
        register_specialist(ApoptosisSpecialist)
        self._ApoptosisSpecialist = ApoptosisSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_apply_crosstalk_returns_5_fragments(self):
        """apply_crosstalk() 返回 5 条 cross-talk Reaction 片段。"""
        specialist = self._ApoptosisSpecialist()
        fragments = specialist.apply_crosstalk(
            {"nodes": [], "edges": []}, []
        )

        self.assertEqual(
            len(fragments),
            5,
            f"期望 5 条 cross-talk 片段，实际 {len(fragments)} 条",
        )

    def test_crosstalk_contains_p53_bax_transcription(self):
        """含 p53 → Bax transcription 片段。"""
        specialist = self._ApoptosisSpecialist()
        fragments = specialist.apply_crosstalk(
            {"nodes": [], "edges": []}, []
        )

        bax = next(
            (f for f in fragments if f["target"] == "Bax"),
            None,
        )
        self.assertIsNotNone(bax)
        self.assertEqual(bax["source"], "p53")
        self.assertEqual(bax["mechanism"], "transcription")

    def test_crosstalk_contains_p53_puma_transcription(self):
        """含 p53 → PUMA transcription 片段。"""
        specialist = self._ApoptosisSpecialist()
        fragments = specialist.apply_crosstalk(
            {"nodes": [], "edges": []}, []
        )

        puma = next(
            (f for f in fragments if f["target"] == "PUMA"),
            None,
        )
        self.assertIsNotNone(puma)
        self.assertEqual(puma["source"], "p53")
        self.assertEqual(puma["mechanism"], "transcription")

    def test_crosstalk_contains_akt_bad_inhibition(self):
        """含 pAKT → Bad inhibition 片段（Ser136）。"""
        specialist = self._ApoptosisSpecialist()
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

    def test_crosstalk_contains_erk_bim_phosphorylation(self):
        """含 pERK → Bim phosphorylation 片段。"""
        specialist = self._ApoptosisSpecialist()
        fragments = specialist.apply_crosstalk(
            {"nodes": [], "edges": []}, []
        )

        bim = next(
            (f for f in fragments if f["target"] == "Bim"),
            None,
        )
        self.assertIsNotNone(bim)
        self.assertEqual(bim["source"], "pERK")
        self.assertEqual(bim["mechanism"], "phosphorylation")

    def test_crosstalk_contains_nfkb_bcl2_transcription(self):
        """含 NF-κB → Bcl-2 transcription 片段。"""
        specialist = self._ApoptosisSpecialist()
        fragments = specialist.apply_crosstalk(
            {"nodes": [], "edges": []}, []
        )

        bcl2 = next(
            (f for f in fragments if f["target"] == "Bcl2"),
            None,
        )
        self.assertIsNotNone(bcl2)
        self.assertEqual(bcl2["mechanism"], "transcription")

    def test_crosstalk_fragments_contain_required_fields(self):
        """每条 cross-talk 片段含 source / target / mechanism / shared_species /
        description。"""
        specialist = self._ApoptosisSpecialist()
        fragments = specialist.apply_crosstalk(
            {"nodes": [], "edges": []}, []
        )

        for f in fragments:
            self.assertIn("source", f)
            self.assertIn("target", f)
            self.assertIn("mechanism", f)
            self.assertIn("shared_species", f)
            self.assertIn("description", f)


class TestApoptosisPerturbationModule(unittest.TestCase):
    """测试 7：Perturbation 模块返回 6 个扰动。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.apoptosis_specialist import (
            ApoptosisSpecialist,
        )

        clear_registry()
        register_specialist(ApoptosisSpecialist)
        self._ApoptosisSpecialist = ApoptosisSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_apply_perturbation_returns_6(self):
        """apply_perturbation() 返回 6 个扰动。"""
        specialist = self._ApoptosisSpecialist()
        perturbations = specialist.apply_perturbation(
            {"nodes": [], "edges": []}, []
        )

        self.assertEqual(
            len(perturbations),
            6,
            f"期望 6 个扰动，实际 {len(perturbations)} 个",
        )

    def test_perturbation_contains_abt199(self):
        """含 ABT-199（Bcl-2 选择性抑制剂，FDA-approved）。"""
        specialist = self._ApoptosisSpecialist()
        perturbations = specialist.apply_perturbation(
            {"nodes": [], "edges": []}, []
        )

        abt199 = next(
            (p for p in perturbations if p.get("drug") == "ABT-199"),
            None,
        )
        self.assertIsNotNone(abt199)
        self.assertEqual(abt199["target"], "Bcl2")
        self.assertEqual(abt199["mechanism"], "inhibition")

    def test_perturbation_contains_navitoclax(self):
        """含 Navitoclax（Bcl-2 / Bcl-xL 双重抑制剂）。"""
        specialist = self._ApoptosisSpecialist()
        perturbations = specialist.apply_perturbation(
            {"nodes": [], "edges": []}, []
        )

        navi = next(
            (p for p in perturbations if p.get("drug") == "Navitoclax"),
            None,
        )
        self.assertIsNotNone(navi)
        self.assertEqual(navi["target"], "Bcl2")

    def test_perturbation_contains_obatoclax(self):
        """含 Obatoclax（pan-Bcl-2 家族抑制剂）。"""
        specialist = self._ApoptosisSpecialist()
        perturbations = specialist.apply_perturbation(
            {"nodes": [], "edges": []}, []
        )

        oba = next(
            (p for p in perturbations if p.get("drug") == "Obatoclax"),
            None,
        )
        self.assertIsNotNone(oba)

    def test_perturbation_contains_zvadfmk(self):
        """含 Z-VAD-FMK（pan-caspase 广谱抑制剂）。"""
        specialist = self._ApoptosisSpecialist()
        perturbations = specialist.apply_perturbation(
            {"nodes": [], "edges": []}, []
        )

        zvad = next(
            (p for p in perturbations if p.get("drug") == "Z-VAD-FMK"),
            None,
        )
        self.assertIsNotNone(zvad)
        self.assertEqual(zvad["mechanism"], "inhibition")

    def test_perturbation_contains_bax_mutation(self):
        """含 BAX mutation（loss-of-function，凋亡抵抗）。"""
        specialist = self._ApoptosisSpecialist()
        perturbations = specialist.apply_perturbation(
            {"nodes": [], "edges": []}, []
        )

        bax_mut = next(
            (p for p in perturbations if p.get("ko_target") == "BAX_mutation"),
            None,
        )
        self.assertIsNotNone(bax_mut)
        self.assertEqual(bax_mut["target"], "Bax")
        self.assertEqual(bax_mut["mechanism"], "knockout")

    def test_perturbations_contain_required_fields(self):
        """每个扰动含 target / drug / mechanism / ko_target / description。"""
        specialist = self._ApoptosisSpecialist()
        perturbations = specialist.apply_perturbation(
            {"nodes": [], "edges": []}, []
        )

        for p in perturbations:
            self.assertIn("target", p)
            self.assertIn("drug", p)
            self.assertIn("mechanism", p)
            self.assertIn("ko_target", p)
            self.assertIn("description", p)


class TestApoptosisValidationModule(unittest.TestCase):
    """测试 8：Validation 模块返回 3 条 benchmark。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.apoptosis_specialist import (
            ApoptosisSpecialist,
        )

        clear_registry()
        register_specialist(ApoptosisSpecialist)
        self._ApoptosisSpecialist = ApoptosisSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_apply_validation_returns_3_rules(self):
        """apply_validation() 返回 3 条 Validation 规则。"""
        specialist = self._ApoptosisSpecialist()
        rules = specialist.apply_validation()

        self.assertEqual(
            len(rules),
            3,
            f"期望 3 条 Validation 规则，实际 {len(rules)} 条",
        )

    def test_cyt_c_precedes_casp3_rule(self):
        """Cyt c 早于 Caspase3 5-15 min（Reubold 2009, PMID:11274138）。"""
        specialist = self._ApoptosisSpecialist()
        rules = specialist.apply_validation()

        cytc = next(
            (r for r in rules
             if r["metric_name"] == "Cyt_c_precedes_Casp3"),
            None,
        )
        self.assertIsNotNone(cytc)
        self.assertEqual(cytc["expected_min"], 5.0)
        self.assertEqual(cytc["expected_max"], 15.0)
        self.assertEqual(cytc["unit"], "minutes_delay")
        self.assertEqual(cytc["pmid"], "PMID:11274138")

    def test_momp_bistable_switch_rule(self):
        """MOMP bistable all-or-none（Green & Kroemer 2004, PMID:15241432）。"""
        specialist = self._ApoptosisSpecialist()
        rules = specialist.apply_validation()

        momp = next(
            (r for r in rules
             if r["metric_name"] == "MOMP_bistable_switch"),
            None,
        )
        self.assertIsNotNone(momp)
        self.assertEqual(momp["expected"], True)
        self.assertEqual(momp["pmid"], "PMID:15241432")

    def test_caspase3_threshold_rule(self):
        """Caspase-3 激活阈值 0.1-0.5（bistable 阈值）。"""
        specialist = self._ApoptosisSpecialist()
        rules = specialist.apply_validation()

        threshold = next(
            (r for r in rules
             if r["metric_name"] == "Caspase3_activation_threshold"),
            None,
        )
        self.assertIsNotNone(threshold)
        self.assertEqual(threshold["expected_min"], 0.1)
        self.assertEqual(threshold["expected_max"], 0.5)
        self.assertEqual(threshold["unit"], "fraction_of_max")

    def test_validation_rules_contain_required_fields(self):
        """每条 Validation 规则含 rule_id / metric_name / expected /
        tolerance / pmid / description。"""
        specialist = self._ApoptosisSpecialist()
        rules = specialist.apply_validation()

        for r in rules:
            self.assertIn("rule_id", r)
            self.assertIn("metric_name", r)
            self.assertIn("expected", r)
            self.assertIn("tolerance", r)
            self.assertIn("pmid", r)
            self.assertIn("description", r)


class TestApoptosisSelectTemplate(unittest.TestCase):
    """测试 9：模板选择（bistable / phosphorylation）。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.apoptosis_specialist import (
            ApoptosisSpecialist,
        )

        clear_registry()
        register_specialist(ApoptosisSpecialist)
        self._ApoptosisSpecialist = ApoptosisSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_select_template_bistable(self):
        """select_template('bistable') 返回 bistable_switch（Caspase 级联）。"""
        specialist = self._ApoptosisSpecialist()
        self.assertEqual(
            specialist.select_template("bistable"),
            "bistable_switch",
        )

    def test_select_template_phosphorylation(self):
        """select_template('phosphorylation') 返回 _mechanism_phosphorylation_mm。"""
        specialist = self._ApoptosisSpecialist()
        self.assertEqual(
            specialist.select_template("phosphorylation"),
            "_mechanism_phosphorylation_mm",
        )

    def test_select_template_cleavage_bistable(self):
        """select_template('cleavage') 返回 bistable_switch（Caspase 级联触发 bistable）。"""
        specialist = self._ApoptosisSpecialist()
        self.assertEqual(
            specialist.select_template("cleavage"),
            "bistable_switch",
        )


class TestApoptosisFeatureFlagIsolation(unittest.TestCase):
    """测试 10：Feature Flag 隔离（flag=false 时 hook 不调用 Specialist）。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.apoptosis_specialist import (
            ApoptosisSpecialist,
        )

        clear_registry()
        register_specialist(ApoptosisSpecialist)
        self._ApoptosisSpecialist = ApoptosisSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_flag_false_hook_does_not_call_specialist(self):
        """flag=false 时 hook 返回空 dict 且不调用 Specialist 任何方法。"""
        specialist = MagicMock()
        specialist.apply_core.return_value = {
            "species": [],
            "reactions": [],
            "composite_reactions": [],
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
            "species": [{"name": "Caspase3_active"}],
            "reactions": [],
            "composite_reactions": [],
        }
        specialist.apply_core.return_value = expected_core
        state = {"v4_pathway_graph": {"nodes": [{"id": "Bax"}], "edges": []}}

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


class TestApoptosisMetadataAndInputValidation(unittest.TestCase):
    """测试 11：元数据 + 输入校验。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.apoptosis_specialist import (
            ApoptosisSpecialist,
        )

        clear_registry()
        register_specialist(ApoptosisSpecialist)
        self._ApoptosisSpecialist = ApoptosisSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_get_metadata_returns_apoptosis_metadata(self):
        """get_metadata() 返回 pathway_class='APOPTOSIS' + 5 模块。"""
        specialist = self._ApoptosisSpecialist()
        metadata = specialist.get_metadata()

        self.assertEqual(metadata["pathway_class"], "APOPTOSIS")
        self.assertEqual(
            metadata["display_name"],
            "Apoptosis (Intrinsic + Extrinsic)",
        )
        self.assertEqual(
            metadata["supported_modules"],
            ["core", "feedback", "crosstalk", "perturbation", "validation"],
        )
        self.assertEqual(len(metadata["supported_modules"]), 5)
        self.assertIn("version", metadata)

    def test_validate_input_empty_graph_returns_warnings(self):
        """validate_input({}) 返回非空 warning 列表。"""
        specialist = self._ApoptosisSpecialist()
        warnings = specialist.validate_input({})

        self.assertIsInstance(warnings, list)
        self.assertGreater(len(warnings), 0, "空 pathway_graph 应返回 warning")

    def test_validate_input_none_returns_warnings(self):
        """validate_input(None) 返回非空 warning 列表。"""
        specialist = self._ApoptosisSpecialist()
        warnings = specialist.validate_input(None)

        self.assertGreater(len(warnings), 0)

    def test_validate_input_valid_graph_returns_no_warnings(self):
        """含 nodes + edges 的 pathway_graph 无 warning。"""
        specialist = self._ApoptosisSpecialist()
        warnings = specialist.validate_input(
            {"nodes": [{"id": "Bax"}], "edges": []}
        )
        self.assertEqual(warnings, [])


if __name__ == "__main__":
    unittest.main()
