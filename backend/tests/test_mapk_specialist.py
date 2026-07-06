# BioDynamics Agent v4 - MAPK Specialist 单元测试 (Phase 4 / Task 4.4)
#
# 测试用例（11 项，覆盖 5 模块 + flag 隔离 + 元数据 + 双磷酸化 + RasGTP 共享）：
#   1. Specialist 注册：get_specialist("MAPK_ERK") 返回 MAPKSpecialist 实例
#   2. Core 模块：apply_core() 返回 5 条核心反应（三级双磷酸化级联）
#   3. 双磷酸化语义：pMEK→ppMEK / pERK→ppERK target 是 p+source 形式
#   4. RasGTP 共享标记：Core 输出中 RasGTP 标记 consumed_shared=True
#   5. Feedback 模块：apply_feedback() 返回 2 条反馈环（pERK→SOS / ppERK→pRaf）
#   6. Crosstalk 模块：apply_crosstalk() 返回 3 条 cross-talk Reaction 片段
#   7. Perturbation 模块：apply_perturbation() 返回 4 个药物
#   8. Validation 模块：apply_validation() 返回 3 条 benchmark（含 pmid）
#   9. 模板选择：select_template("phosphorylation") → _mechanism_phosphorylation_mm
#  10. Feature Flag 隔离：flag=false 时 hook 不调用 Specialist
#  11. 元数据：get_metadata() 返回 pathway_class / supported_modules 5 项
#
# 运行：cd backend && python -m pytest tests/test_mapk_specialist.py -v

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
class TestMAPKSpecialistRegistration(unittest.TestCase):
    """测试 1：Specialist 注册到 SPECIALIST_REGISTRY。"""

    def setUp(self):
        """每个测试开始前清空 registry 并重新注册 MAPKSpecialist。"""
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.mapk_specialist import MAPKSpecialist

        clear_registry()
        register_specialist(MAPKSpecialist)

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_get_specialist_returns_mapk_instance(self):
        """get_specialist("MAPK_ERK") 返回 MAPKSpecialist 实例。"""
        from app.pathways.pathway_registry import (
            get_specialist,
            is_specialist_available,
        )
        from app.pathways.specialists.mapk_specialist import MAPKSpecialist

        # 已注册
        self.assertTrue(is_specialist_available("MAPK_ERK"))

        # get_specialist 返回 MAPKSpecialist 实例
        instance = get_specialist("MAPK_ERK")
        self.assertIsNotNone(instance)
        self.assertIsInstance(instance, MAPKSpecialist)

        # 多次调用返回不同实例（非单例）
        instance2 = get_specialist("MAPK_ERK")
        self.assertIsNot(instance, instance2)

    def test_pathway_class_attribute(self):
        """MAPKSpecialist.pathway_class == 'MAPK_ERK'。"""
        from app.pathways.specialists.mapk_specialist import MAPKSpecialist

        self.assertEqual(MAPKSpecialist.pathway_class, "MAPK_ERK")


class TestMAPKCoreModule(unittest.TestCase):
    """测试 2：Core 模块返回 5 条核心反应（三级双磷酸化级联）。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.mapk_specialist import MAPKSpecialist

        clear_registry()
        register_specialist(MAPKSpecialist)
        self._MAPKSpecialist = MAPKSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_apply_core_returns_5_reactions(self):
        """apply_core() 返回 5 条核心反应。"""
        specialist = self._MAPKSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})

        self.assertIn("species", result)
        self.assertIn("reactions", result)
        reactions = result["reactions"]
        self.assertEqual(
            len(reactions),
            5,
            f"期望 5 条核心反应，实际 {len(reactions)} 条",
        )

    def test_apply_core_returns_9_species(self):
        """apply_core() 返回 9 个核心物种（含 RasGTP consumed_shared）。"""
        specialist = self._MAPKSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})

        species = result["species"]
        self.assertEqual(
            len(species),
            9,
            f"期望 9 个核心物种，实际 {len(species)} 个",
        )

    def test_core_reactions_contain_correct_mechanisms(self):
        """5 条核心反应 mechanism 均为 phosphorylation。"""
        specialist = self._MAPKSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        reactions = result["reactions"]

        # 所有反应均为 phosphorylation（MAPK 级联核心机制）
        for r in reactions:
            self.assertEqual(
                r["mechanism"],
                "phosphorylation",
                f"反应 {r['source']}→{r['target']} mechanism 应为 phosphorylation",
            )

    def test_core_reactions_force_michaelis_menten(self):
        """5 条核心反应 kinetics_type 强制为 Michaelis_Menten。"""
        specialist = self._MAPKSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        reactions = result["reactions"]

        for r in reactions:
            self.assertEqual(
                r["kinetics_type"],
                "Michaelis_Menten",
                f"反应 {r['source']}→{r['target']} kinetics_type 应为 Michaelis_Menten",
            )

    def test_core_reactions_contain_pathway_tag(self):
        """每条核心反应含 pathway_tag='MAPK_ERK'。"""
        specialist = self._MAPKSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        for r in result["reactions"]:
            self.assertEqual(r["pathway_tag"], "MAPK_ERK")

    def test_core_reactions_contain_source_target(self):
        """每条核心反应含 source / target / kinetics_type / substrate / product / modifier。"""
        specialist = self._MAPKSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        for r in result["reactions"]:
            self.assertIn("source", r)
            self.assertIn("target", r)
            self.assertIn("kinetics_type", r)
            self.assertIn("substrate", r)
            self.assertIn("product", r)
            self.assertIn("modifier", r)
            self.assertIn("modifier_type", r)

    def test_core_reaction_pairs(self):
        """5 条核心反应的 source→target 对应正确（三级双磷酸化级联）。"""
        specialist = self._MAPKSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        pairs = [(r["source"], r["target"]) for r in result["reactions"]]
        expected_pairs = [
            ("RasGTP", "pRaf"),    # 异磷酸化
            ("pRaf", "pMEK"),      # 异磷酸化
            ("pMEK", "ppMEK"),     # 双磷酸化第一步
            ("ppMEK", "pERK"),     # 异磷酸化
            ("pERK", "ppERK"),     # 双磷酸化第二步
        ]
        self.assertEqual(pairs, expected_pairs)


class TestMAPKDoublePhosphorylationSemantics(unittest.TestCase):
    """测试 3：双磷酸化语义（pMEK→ppMEK / pERK→ppERK）。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.mapk_specialist import MAPKSpecialist

        clear_registry()
        register_specialist(MAPKSpecialist)
        self._MAPKSpecialist = MAPKSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_double_phosphorylation_pmek_to_ppmek(self):
        """pMEK → ppMEK 是双磷酸化第一步：target 是 p+source 形式。"""
        specialist = self._MAPKSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        # 第 3 条反应（index=2）是 pMEK → ppMEK
        rxn = result["reactions"][2]

        self.assertEqual(rxn["source"], "pMEK")
        self.assertEqual(rxn["target"], "ppMEK")
        # 双磷酸化形式：target == "p" + source
        self.assertEqual(rxn["target"], "p" + rxn["source"])
        # autophosphorylation 标记（自催化形式）
        self.assertTrue(
            rxn["autophosphorylation"],
            "pMEK → ppMEK 应标记为自磷酸化形式（双磷酸化）",
        )
        # pMEK 作 substrate+modifier，ppMEK 作 product
        self.assertEqual(rxn["substrate"], "pMEK")
        self.assertEqual(rxn["product"], "ppMEK")
        self.assertEqual(rxn["modifier"], "pMEK")
        self.assertEqual(rxn["modifier_type"], "catalytic")

    def test_double_phosphorylation_perk_to_pperk(self):
        """pERK → ppERK 是双磷酸化第二步：target 是 p+source 形式。"""
        specialist = self._MAPKSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        # 第 5 条反应（index=4）是 pERK → ppERK
        rxn = result["reactions"][4]

        self.assertEqual(rxn["source"], "pERK")
        self.assertEqual(rxn["target"], "ppERK")
        # 双磷酸化形式：target == "p" + source
        self.assertEqual(rxn["target"], "p" + rxn["source"])
        # autophosphorylation 标记（自催化形式）
        self.assertTrue(
            rxn["autophosphorylation"],
            "pERK → ppERK 应标记为自磷酸化形式（双磷酸化）",
        )
        # pERK 作 substrate+modifier，ppERK 作 product
        self.assertEqual(rxn["substrate"], "pERK")
        self.assertEqual(rxn["product"], "ppERK")
        self.assertEqual(rxn["modifier"], "pERK")
        self.assertEqual(rxn["modifier_type"], "catalytic")

    def test_hetero_phosphorylation_rasgtp_to_praf(self):
        """RasGTP → pRaf 是异磷酸化：Raf 作 substrate，RasGTP 作 catalytic modifier。"""
        specialist = self._MAPKSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        # 第 1 条反应（index=0）是 RasGTP → pRaf
        rxn = result["reactions"][0]

        self.assertEqual(rxn["source"], "RasGTP")
        self.assertEqual(rxn["target"], "pRaf")
        self.assertFalse(rxn["autophosphorylation"])
        # 异磷酸化：Raf 作 substrate，pRaf 作 product，RasGTP 作 catalytic modifier
        self.assertEqual(rxn["substrate"], "Raf")
        self.assertEqual(rxn["product"], "pRaf")
        self.assertEqual(rxn["modifier"], "RasGTP")
        self.assertEqual(rxn["modifier_type"], "catalytic")

    def test_hetero_phosphorylation_ppmek_to_perk(self):
        """ppMEK → pERK 是异磷酸化：ERK 作 substrate，ppMEK 作 catalytic modifier。"""
        specialist = self._MAPKSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        # 第 4 条反应（index=3）是 ppMEK → pERK
        rxn = result["reactions"][3]

        self.assertEqual(rxn["source"], "ppMEK")
        self.assertEqual(rxn["target"], "pERK")
        self.assertFalse(rxn["autophosphorylation"])
        # 异磷酸化：ERK 作 substrate，pERK 作 product，ppMEK 作 catalytic modifier
        self.assertEqual(rxn["substrate"], "ERK")
        self.assertEqual(rxn["product"], "pERK")
        self.assertEqual(rxn["modifier"], "ppMEK")
        self.assertEqual(rxn["modifier_type"], "catalytic")


class TestMAPKRasGTPSharedMarker(unittest.TestCase):
    """测试 4：RasGTP 共享标记（consumed_shared=True）。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.mapk_specialist import MAPKSpecialist

        clear_registry()
        register_specialist(MAPKSpecialist)
        self._MAPKSpecialist = MAPKSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_rasgtp_marked_as_consumed_shared(self):
        """RasGTP 物种标记 consumed_shared=True（来自 EGFR，MAPK 仅消费）。"""
        specialist = self._MAPKSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        species = result["species"]

        rasgtp = next(
            (s for s in species if s["name"] == "RasGTP"),
            None,
        )
        self.assertIsNotNone(rasgtp, "Core 输出应含 RasGTP 物种")
        self.assertTrue(
            rasgtp.get("consumed_shared"),
            "RasGTP 应标记 consumed_shared=True（MAPK 仅消费，不创建）",
        )
        self.assertTrue(
            rasgtp.get("shared"),
            "RasGTP 应标记 shared=True",
        )

    def test_rasgtp_is_only_consumed_shared_species(self):
        """RasGTP 是唯一标记 consumed_shared 的物种。"""
        specialist = self._MAPKSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        species = result["species"]

        consumed = [s for s in species if s.get("consumed_shared")]
        self.assertEqual(
            len(consumed),
            1,
            f"期望仅 1 个 consumed_shared 物种，实际 {len(consumed)} 个",
        )
        self.assertEqual(consumed[0]["name"], "RasGTP")


class TestMAPKFeedbackModule(unittest.TestCase):
    """测试 5：Feedback 模块返回 2 条反馈环（均 delay=0 min）。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.mapk_specialist import MAPKSpecialist

        clear_registry()
        register_specialist(MAPKSpecialist)
        self._MAPKSpecialist = MAPKSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_apply_feedback_returns_2_loops(self):
        """apply_feedback() 返回 2 条反馈环。"""
        specialist = self._MAPKSpecialist()
        loops = specialist.apply_feedback({"nodes": [], "edges": []})

        self.assertEqual(
            len(loops),
            2,
            f"期望 2 条反馈环，实际 {len(loops)} 条",
        )

    def test_erk_sos_feedback_delay_zero(self):
        """pERK → SOS 负反馈 delay=0 min。"""
        specialist = self._MAPKSpecialist()
        loops = specialist.apply_feedback({"nodes": [], "edges": []})

        sos = next(
            (l for l in loops if "SOS" in l["id"]),
            None,
        )
        self.assertIsNotNone(sos, "应含 pERK→SOS 反馈环")
        self.assertEqual(sos["loop_type"], "negative")
        self.assertEqual(sos["delay_minutes"], 0.0)
        # node_ids 应含 pERK 与 SOS
        self.assertIn("pERK", sos["node_ids"])
        self.assertIn("SOS", sos["node_ids"])

    def test_erk_raf_s259_feedback_delay_zero(self):
        """ppERK → pRaf Ser259 负反馈 delay=0 min。"""
        specialist = self._MAPKSpecialist()
        loops = specialist.apply_feedback({"nodes": [], "edges": []})

        raf = next(
            (l for l in loops if "RAF" in l["id"] or "S259" in l["id"]),
            None,
        )
        self.assertIsNotNone(raf, "应含 ppERK→pRaf Ser259 反馈环")
        self.assertEqual(raf["loop_type"], "negative")
        self.assertEqual(raf["delay_minutes"], 0.0)
        # node_ids 应含 ppERK 与 pRaf
        self.assertIn("ppERK", raf["node_ids"])

    def test_feedback_loops_contain_required_fields(self):
        """每条反馈环含 id / loop_type / node_ids / delay_minutes / description。"""
        specialist = self._MAPKSpecialist()
        loops = specialist.apply_feedback({"nodes": [], "edges": []})

        for loop in loops:
            self.assertIn("id", loop)
            self.assertIn("loop_type", loop)
            self.assertIn("node_ids", loop)
            self.assertIn("delay_minutes", loop)
            self.assertIn("description", loop)


class TestMAPKCrosstalkModule(unittest.TestCase):
    """测试 6：Crosstalk 模块返回 3 条 cross-talk Reaction 片段。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.mapk_specialist import MAPKSpecialist

        clear_registry()
        register_specialist(MAPKSpecialist)
        self._MAPKSpecialist = MAPKSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_apply_crosstalk_returns_3_fragments(self):
        """apply_crosstalk() 返回 3 条 cross-talk Reaction 片段。"""
        specialist = self._MAPKSpecialist()
        fragments = specialist.apply_crosstalk(
            {"nodes": [], "edges": []}, []
        )

        self.assertEqual(
            len(fragments),
            3,
            f"期望 3 条 cross-talk 片段，实际 {len(fragments)} 条",
        )

    def test_crosstalk_contains_elk1_phosphorylation(self):
        """含 pERK → ELK1 phosphorylation 片段。"""
        specialist = self._MAPKSpecialist()
        fragments = specialist.apply_crosstalk(
            {"nodes": [], "edges": []}, []
        )

        elk1 = next(
            (f for f in fragments if f["target"] == "ELK1"),
            None,
        )
        self.assertIsNotNone(elk1)
        self.assertEqual(elk1["source"], "pERK")
        self.assertEqual(elk1["mechanism"], "phosphorylation")

    def test_crosstalk_contains_myc_transcription(self):
        """含 pERK → Myc transcription 片段。"""
        specialist = self._MAPKSpecialist()
        fragments = specialist.apply_crosstalk(
            {"nodes": [], "edges": []}, []
        )

        myc = next(
            (f for f in fragments if f["target"] == "Myc"),
            None,
        )
        self.assertIsNotNone(myc)
        self.assertEqual(myc["source"], "pERK")
        self.assertEqual(myc["mechanism"], "transcription")

    def test_crosstalk_contains_bim_phosphorylation(self):
        """含 pERK → Bim phosphorylation 片段。"""
        specialist = self._MAPKSpecialist()
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

    def test_crosstalk_fragments_contain_required_fields(self):
        """每条 cross-talk 片段含 source / target / mechanism / shared_species / description。"""
        specialist = self._MAPKSpecialist()
        fragments = specialist.apply_crosstalk(
            {"nodes": [], "edges": []}, []
        )

        for f in fragments:
            self.assertIn("source", f)
            self.assertIn("target", f)
            self.assertIn("mechanism", f)
            self.assertIn("shared_species", f)
            self.assertIn("description", f)


class TestMAPKPerturbationModule(unittest.TestCase):
    """测试 7：Perturbation 模块返回 4 个药物。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.mapk_specialist import MAPKSpecialist

        clear_registry()
        register_specialist(MAPKSpecialist)
        self._MAPKSpecialist = MAPKSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_apply_perturbation_returns_4(self):
        """apply_perturbation() 返回 4 个扰动。"""
        specialist = self._MAPKSpecialist()
        perturbations = specialist.apply_perturbation(
            {"nodes": [], "edges": []}, []
        )

        self.assertEqual(
            len(perturbations),
            4,
            f"期望 4 个扰动，实际 {len(perturbations)} 个",
        )

    def test_perturbation_contains_trametinib(self):
        """含 Trametinib（MEK 抑制剂）。"""
        specialist = self._MAPKSpecialist()
        perturbations = specialist.apply_perturbation(
            {"nodes": [], "edges": []}, []
        )

        trametinib = next(
            (p for p in perturbations if p.get("drug") == "Trametinib"),
            None,
        )
        self.assertIsNotNone(trametinib)
        self.assertEqual(trametinib["target"], "MEK")
        self.assertEqual(trametinib["mechanism"], "inhibition")

    def test_perturbation_contains_vemurafenib(self):
        """含 Vemurafenib（BRAF 抑制剂）。"""
        specialist = self._MAPKSpecialist()
        perturbations = specialist.apply_perturbation(
            {"nodes": [], "edges": []}, []
        )

        vemurafenib = next(
            (p for p in perturbations if p.get("drug") == "Vemurafenib"),
            None,
        )
        self.assertIsNotNone(vemurafenib)
        self.assertEqual(vemurafenib["target"], "BRAF")
        self.assertEqual(vemurafenib["mechanism"], "inhibition")

    def test_perturbation_contains_dabrafenib(self):
        """含 Dabrafenib（BRAF 抑制剂）。"""
        specialist = self._MAPKSpecialist()
        perturbations = specialist.apply_perturbation(
            {"nodes": [], "edges": []}, []
        )

        dabrafenib = next(
            (p for p in perturbations if p.get("drug") == "Dabrafenib"),
            None,
        )
        self.assertIsNotNone(dabrafenib)
        self.assertEqual(dabrafenib["target"], "BRAF")
        self.assertEqual(dabrafenib["mechanism"], "inhibition")

    def test_perturbation_contains_selumetinib(self):
        """含 Selumetinib（MEK 抑制剂）。"""
        specialist = self._MAPKSpecialist()
        perturbations = specialist.apply_perturbation(
            {"nodes": [], "edges": []}, []
        )

        selumetinib = next(
            (p for p in perturbations if p.get("drug") == "Selumetinib"),
            None,
        )
        self.assertIsNotNone(selumetinib)
        self.assertEqual(selumetinib["target"], "MEK")
        self.assertEqual(selumetinib["mechanism"], "inhibition")

    def test_perturbations_contain_required_fields(self):
        """每个扰动含 target / drug / mechanism / ko_target / description。"""
        specialist = self._MAPKSpecialist()
        perturbations = specialist.apply_perturbation(
            {"nodes": [], "edges": []}, []
        )

        for p in perturbations:
            self.assertIn("target", p)
            self.assertIn("drug", p)
            self.assertIn("mechanism", p)
            self.assertIn("ko_target", p)
            self.assertIn("description", p)


class TestMAPKValidationModule(unittest.TestCase):
    """测试 8：Validation 模块返回 3 条 benchmark。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.mapk_specialist import MAPKSpecialist

        clear_registry()
        register_specialist(MAPKSpecialist)
        self._MAPKSpecialist = MAPKSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_apply_validation_returns_3_rules(self):
        """apply_validation() 返回 3 条 Validation 规则。"""
        specialist = self._MAPKSpecialist()
        rules = specialist.apply_validation()

        self.assertEqual(
            len(rules),
            3,
            f"期望 3 条 Validation 规则，实际 {len(rules)} 条",
        )

    def test_mapk_amplification_rule(self):
        """MAPK 放大倍数 10-100x（PMID:11483517）。"""
        specialist = self._MAPKSpecialist()
        rules = specialist.apply_validation()

        amp = next(
            (r for r in rules if r["metric_name"] == "MAPK_amplification"),
            None,
        )
        self.assertIsNotNone(amp)
        self.assertEqual(amp["expected_min"], 10.0)
        self.assertEqual(amp["expected_max"], 100.0)
        self.assertEqual(amp["unit"], "fold")
        self.assertEqual(amp["pmid"], "PMID:11483517")

    def test_zero_order_ultrasensitivity_hill_rule(self):
        """零阶 ultrasensitivity Hill 系数 2-10（PMID:1941687）。"""
        specialist = self._MAPKSpecialist()
        rules = specialist.apply_validation()

        hill = next(
            (
                r
                for r in rules
                if r["metric_name"]
                == "zero_order_ultrasensitivity_hill_coefficient"
            ),
            None,
        )
        self.assertIsNotNone(hill)
        self.assertEqual(hill["expected_min"], 2.0)
        self.assertEqual(hill["expected_max"], 10.0)
        self.assertEqual(hill["unit"], "hill_n")
        self.assertEqual(hill["pmid"], "PMID:1941687")
        # 验证 Hill 系数 >=2（零阶 ultrasensitivity benchmark，spec expected_min=2.0）
        self.assertGreaterEqual(
            hill["expected_min"],
            2.0,
            "Hill 系数 expected_min 应 >=2（零阶 ultrasensitivity）",
        )

    def test_erk_peak_time_rule(self):
        """ERK 达峰时间 2-8 min（PMID:11483517）。"""
        specialist = self._MAPKSpecialist()
        rules = specialist.apply_validation()

        peak = next(
            (r for r in rules if r["metric_name"] == "ERK_peak_time"),
            None,
        )
        self.assertIsNotNone(peak)
        self.assertEqual(peak["expected_min"], 2.0)
        self.assertEqual(peak["expected_max"], 8.0)
        self.assertEqual(peak["unit"], "minutes")
        self.assertEqual(peak["pmid"], "PMID:11483517")

    def test_validation_rules_contain_required_fields(self):
        """每条 Validation 规则含 rule_id / metric_name / expected / tolerance / pmid / description。"""
        specialist = self._MAPKSpecialist()
        rules = specialist.apply_validation()

        for r in rules:
            self.assertIn("rule_id", r)
            self.assertIn("metric_name", r)
            self.assertIn("expected", r)
            self.assertIn("tolerance", r)
            self.assertIn("pmid", r)
            self.assertIn("description", r)


class TestMAPKSelectTemplate(unittest.TestCase):
    """测试 9：模板选择 select_template('phosphorylation') → _mechanism_phosphorylation_mm。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.mapk_specialist import MAPKSpecialist

        clear_registry()
        register_specialist(MAPKSpecialist)
        self._MAPKSpecialist = MAPKSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_select_template_phosphorylation(self):
        """select_template('phosphorylation') 返回 _mechanism_phosphorylation_mm。"""
        specialist = self._MAPKSpecialist()
        self.assertEqual(
            specialist.select_template("phosphorylation"),
            "_mechanism_phosphorylation_mm",
        )


class TestMAPKFeatureFlagIsolation(unittest.TestCase):
    """测试 10：Feature Flag 隔离（flag=false 时 hook 不调用 Specialist）。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.mapk_specialist import MAPKSpecialist

        clear_registry()
        register_specialist(MAPKSpecialist)
        self._MAPKSpecialist = MAPKSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_flag_false_hook_does_not_call_specialist(self):
        """flag=false 时 hook 返回空 dict 且不调用 Specialist 任何方法。"""
        specialist = MagicMock()
        specialist.apply_core.return_value = {"species": [], "reactions": []}
        state = {"v4_pathway_graph": {"nodes": [], "edges": []}}

        with patch("app.config.settings.V4_PATHWAY_SPECIALIST_ENABLED", False):
            result = _specialist_hook_mock(state, specialist)

        self.assertEqual(result, {})
        specialist.apply_core.assert_not_called()

    def test_flag_true_hook_calls_specialist(self):
        """flag=true 时 hook 调用 Specialist.apply_core 并返回结果。"""
        specialist = MagicMock()
        expected_core = {"species": [{"name": "RasGTP"}], "reactions": []}
        specialist.apply_core.return_value = expected_core
        state = {"v4_pathway_graph": {"nodes": [{"id": "RasGTP"}], "edges": []}}

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


class TestMAPKMetadataAndInputValidation(unittest.TestCase):
    """测试 11：元数据 + 输入校验。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.mapk_specialist import MAPKSpecialist

        clear_registry()
        register_specialist(MAPKSpecialist)
        self._MAPKSpecialist = MAPKSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_get_metadata_returns_mapk_metadata(self):
        """get_metadata() 返回 pathway_class='MAPK_ERK' + 5 模块。"""
        specialist = self._MAPKSpecialist()
        metadata = specialist.get_metadata()

        self.assertEqual(metadata["pathway_class"], "MAPK_ERK")
        self.assertEqual(metadata["display_name"], "MAPK / ERK Signaling Cascade")
        self.assertEqual(
            metadata["supported_modules"],
            ["core", "feedback", "crosstalk", "perturbation", "validation"],
        )
        self.assertEqual(len(metadata["supported_modules"]), 5)
        self.assertIn("version", metadata)

    def test_validate_input_empty_graph_returns_warnings(self):
        """validate_input({}) 返回非空 warning 列表。"""
        specialist = self._MAPKSpecialist()
        warnings = specialist.validate_input({})

        self.assertIsInstance(warnings, list)
        self.assertGreater(len(warnings), 0, "空 pathway_graph 应返回 warning")

    def test_validate_input_none_returns_warnings(self):
        """validate_input(None) 返回非空 warning 列表。"""
        specialist = self._MAPKSpecialist()
        warnings = specialist.validate_input(None)

        self.assertGreater(len(warnings), 0)

    def test_validate_input_valid_graph_returns_no_warnings(self):
        """含 nodes + edges 的 pathway_graph 无 warning。"""
        specialist = self._MAPKSpecialist()
        warnings = specialist.validate_input(
            {"nodes": [{"id": "RasGTP"}], "edges": []}
        )
        self.assertEqual(warnings, [])


if __name__ == "__main__":
    unittest.main()
