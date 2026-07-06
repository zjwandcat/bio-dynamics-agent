# BioDynamics Agent v4 - EGFR Specialist 单元测试 (Phase 4 / Task 4.3)
#
# 测试用例（11 项，覆盖 5 模块 + flag 隔离 + 元数据 + 输入校验）：
#   1. Specialist 注册：get_specialist("EGFR_RTK") 返回 EGFRSpecialist 实例
#   2. Core 模块：apply_core() 返回 7 条核心反应 + 11 物种
#   3. PHOSPHORYLATION 语义：自磷酸化(EGFR→pEGFR) vs 异磷酸化(pEGFR→pShc)
#   4. Feedback 模块：apply_feedback() 返回 2 条反馈环（delay=0 / delay=15）
#   5. Crosstalk 模块：apply_crosstalk() 返回 3 条 cross-talk Reaction 片段
#   6. Perturbation 模块：apply_perturbation() 返回 4 个药物/突变
#   7. Validation 模块：apply_validation() 返回 3 条 benchmark（含 pmid）
#   8. 模板选择：select_template("phosphorylation") → _mechanism_phosphorylation_mm
#   9. Feature Flag 隔离：flag=false 时 hook 不调用 Specialist
#  10. 元数据：get_metadata() 返回 pathway_class / display_name / supported_modules
#  11. 输入校验：validate_input({}) 返回非空 warning 列表
#
# 运行：cd backend && python -m pytest tests/test_egfr_specialist.py -v

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
class TestEGFRSpecialistRegistration(unittest.TestCase):
    """测试 1：Specialist 注册到 SPECIALIST_REGISTRY。"""

    def setUp(self):
        """每个测试开始前清空 registry 并重新注册 EGFRSpecialist。

        由于 @register_specialist 装饰器在模块导入时执行一次，registry
        可能被 test_specialist_base.py 的 setUp/tearDown 清空。本 setUp
        在 clear_registry 后重新注册，保证测试隔离。
        """
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.egfr_specialist import EGFRSpecialist

        clear_registry()
        register_specialist(EGFRSpecialist)

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_get_specialist_returns_egfr_instance(self):
        """get_specialist("EGFR_RTK") 返回 EGFRSpecialist 实例。"""
        from app.pathways.pathway_registry import (
            get_specialist,
            is_specialist_available,
        )
        from app.pathways.specialists.egfr_specialist import EGFRSpecialist

        # 已注册
        self.assertTrue(is_specialist_available("EGFR_RTK"))

        # get_specialist 返回 EGFRSpecialist 实例
        instance = get_specialist("EGFR_RTK")
        self.assertIsNotNone(instance)
        self.assertIsInstance(instance, EGFRSpecialist)

        # 多次调用返回不同实例（非单例）
        instance2 = get_specialist("EGFR_RTK")
        self.assertIsNot(instance, instance2)

    def test_pathway_class_attribute(self):
        """EGFRSpecialist.pathway_class == 'EGFR_RTK'。"""
        from app.pathways.specialists.egfr_specialist import EGFRSpecialist

        self.assertEqual(EGFRSpecialist.pathway_class, "EGFR_RTK")


class TestEGFRCoreModule(unittest.TestCase):
    """测试 2：Core 模块返回 7 条核心反应 + 11 物种。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.egfr_specialist import EGFRSpecialist

        clear_registry()
        register_specialist(EGFRSpecialist)
        self._EGFRSpecialist = EGFRSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_apply_core_returns_7_reactions(self):
        """apply_core() 返回 7 条核心反应。"""
        specialist = self._EGFRSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})

        self.assertIn("species", result)
        self.assertIn("reactions", result)
        reactions = result["reactions"]
        self.assertEqual(
            len(reactions),
            7,
            f"期望 7 条核心反应，实际 {len(reactions)} 条",
        )

    def test_apply_core_returns_11_species(self):
        """apply_core() 返回 11 个核心物种。"""
        specialist = self._EGFRSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})

        species = result["species"]
        self.assertEqual(
            len(species),
            11,
            f"期望 11 个核心物种，实际 {len(species)} 个",
        )

    def test_core_reactions_contain_correct_mechanisms(self):
        """7 条核心反应含正确的 mechanism 字段。"""
        specialist = self._EGFRSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        reactions = result["reactions"]

        # 期望的 mechanism 序列
        expected_mechanisms = [
            "binding",              # EGF → EGFR
            "phosphorylation",      # EGFR → pEGFR (autophosphorylation)
            "phosphorylation",      # pEGFR → pShc (hetero)
            "binding",              # pEGFR → Grb2
            "binding",              # Grb2 → SOS
            "gtp_gdp_exchange",     # SOS → RasGTP
            "phosphorylation",      # RasGTP → pRaf (hetero)
        ]
        actual_mechanisms = [r["mechanism"] for r in reactions]
        self.assertEqual(actual_mechanisms, expected_mechanisms)

    def test_core_reactions_contain_pathway_tag(self):
        """每条核心反应含 pathway_tag='EGFR_RTK'。"""
        specialist = self._EGFRSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        for r in result["reactions"]:
            self.assertEqual(r["pathway_tag"], "EGFR_RTK")

    def test_core_reactions_contain_source_target(self):
        """每条核心反应含 source / target / kinetics_type 字段。"""
        specialist = self._EGFRSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        for r in result["reactions"]:
            self.assertIn("source", r)
            self.assertIn("target", r)
            self.assertIn("kinetics_type", r)

    def test_core_reaction_pairs(self):
        """7 条核心反应的 source→target 对应正确。"""
        specialist = self._EGFRSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        pairs = [(r["source"], r["target"]) for r in result["reactions"]]
        expected_pairs = [
            ("EGF", "EGFR"),
            ("EGFR", "pEGFR"),
            ("pEGFR", "pShc"),
            ("pEGFR", "Grb2"),
            ("Grb2", "SOS"),
            ("SOS", "RasGTP"),
            ("RasGTP", "pRaf"),
        ]
        self.assertEqual(pairs, expected_pairs)


class TestEGFRPhosphorylationSemantics(unittest.TestCase):
    """测试 3：PHOSPHORYLATION 语义（自磷酸化 vs 异磷酸化）。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.egfr_specialist import EGFRSpecialist

        clear_registry()
        register_specialist(EGFRSpecialist)
        self._EGFRSpecialist = EGFRSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_autophosphorylation_egfr_to_pegfr(self):
        """EGFR → pEGFR 是自磷酸化：source 作 substrate+product，无 modifier。"""
        specialist = self._EGFRSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        # 第 2 条反应（index=1）是 EGFR → pEGFR
        rxn = result["reactions"][1]

        self.assertEqual(rxn["source"], "EGFR")
        self.assertEqual(rxn["target"], "pEGFR")
        self.assertTrue(
            rxn["autophosphorylation"],
            "EGFR → pEGFR 应标记为自磷酸化",
        )
        # 自磷酸化：source 作 substrate，target 作 product，无 modifier
        self.assertEqual(rxn["substrate"], "EGFR")
        self.assertEqual(rxn["product"], "pEGFR")
        self.assertIsNone(rxn["modifier"])

    def test_hetero_phosphorylation_pegfr_to_pshc(self):
        """pEGFR → pShc 是异磷酸化：Shc 作 substrate，pEGFR 作 catalytic modifier。"""
        specialist = self._EGFRSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        # 第 3 条反应（index=2）是 pEGFR → pShc
        rxn = result["reactions"][2]

        self.assertEqual(rxn["source"], "pEGFR")
        self.assertEqual(rxn["target"], "pShc")
        self.assertFalse(
            rxn["autophosphorylation"],
            "pEGFR → pShc 应标记为异磷酸化",
        )
        # 异磷酸化：Shc 作 substrate，pShc 作 product，pEGFR 作 catalytic modifier
        self.assertEqual(rxn["substrate"], "Shc")
        self.assertEqual(rxn["product"], "pShc")
        self.assertEqual(rxn["modifier"], "pEGFR")
        self.assertEqual(rxn["modifier_type"], "catalytic")

    def test_hetero_phosphorylation_rasgtp_to_praf(self):
        """RasGTP → pRaf 是异磷酸化：Raf 作 substrate，RasGTP 作 catalytic modifier。"""
        specialist = self._EGFRSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        # 第 7 条反应（index=6）是 RasGTP → pRaf
        rxn = result["reactions"][6]

        self.assertEqual(rxn["source"], "RasGTP")
        self.assertEqual(rxn["target"], "pRaf")
        self.assertFalse(rxn["autophosphorylation"])
        # 异磷酸化：Raf 作 substrate，pRaf 作 product，RasGTP 作 catalytic modifier
        self.assertEqual(rxn["substrate"], "Raf")
        self.assertEqual(rxn["product"], "pRaf")
        self.assertEqual(rxn["modifier"], "RasGTP")
        self.assertEqual(rxn["modifier_type"], "catalytic")


class TestEGFRFeedbackModule(unittest.TestCase):
    """测试 4：Feedback 模块返回 2 条反馈环。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.egfr_specialist import EGFRSpecialist

        clear_registry()
        register_specialist(EGFRSpecialist)
        self._EGFRSpecialist = EGFRSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_apply_feedback_returns_2_loops(self):
        """apply_feedback() 返回 2 条反馈环。"""
        specialist = self._EGFRSpecialist()
        loops = specialist.apply_feedback({"nodes": [], "edges": []})

        self.assertEqual(
            len(loops),
            2,
            f"期望 2 条反馈环，实际 {len(loops)} 条",
        )

    def test_erk_sos_feedback_delay_zero(self):
        """ERK → SOS 负反馈 delay=0 min。"""
        specialist = self._EGFRSpecialist()
        loops = specialist.apply_feedback({"nodes": [], "edges": []})

        erk_sos = next(
            (l for l in loops if "ERK" in l["id"] or "SOS" in l["id"]),
            None,
        )
        self.assertIsNotNone(erk_sos, "应含 ERK→SOS 反馈环")
        self.assertEqual(erk_sos["loop_type"], "negative")
        self.assertEqual(erk_sos["delay_minutes"], 0.0)

    def test_egfr_internalization_delay_15(self):
        """EGFR 内吞反馈 delay=15 min。"""
        specialist = self._EGFRSpecialist()
        loops = specialist.apply_feedback({"nodes": [], "edges": []})

        internalization = next(
            (l for l in loops if "INTERNALIZATION" in l["id"]),
            None,
        )
        self.assertIsNotNone(internalization, "应含 EGFR 内吞反馈环")
        self.assertEqual(internalization["loop_type"], "negative")
        self.assertEqual(internalization["delay_minutes"], 15.0)

    def test_feedback_loops_contain_required_fields(self):
        """每条反馈环含 id / loop_type / node_ids / delay_minutes / description。"""
        specialist = self._EGFRSpecialist()
        loops = specialist.apply_feedback({"nodes": [], "edges": []})

        for loop in loops:
            self.assertIn("id", loop)
            self.assertIn("loop_type", loop)
            self.assertIn("node_ids", loop)
            self.assertIn("delay_minutes", loop)
            self.assertIn("description", loop)


class TestEGFRCrosstalkModule(unittest.TestCase):
    """测试 5：Crosstalk 模块返回 3 条 cross-talk Reaction 片段。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.egfr_specialist import EGFRSpecialist

        clear_registry()
        register_specialist(EGFRSpecialist)
        self._EGFRSpecialist = EGFRSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_apply_crosstalk_returns_3_fragments(self):
        """apply_crosstalk() 返回 3 条 cross-talk Reaction 片段。"""
        specialist = self._EGFRSpecialist()
        fragments = specialist.apply_crosstalk(
            {"nodes": [], "edges": []}, []
        )

        self.assertEqual(
            len(fragments),
            3,
            f"期望 3 条 cross-talk 片段，实际 {len(fragments)} 条",
        )

    def test_crosstalk_contains_pi3k_activation(self):
        """含 pEGFR → PI3K activation 片段。"""
        specialist = self._EGFRSpecialist()
        fragments = specialist.apply_crosstalk(
            {"nodes": [], "edges": []}, []
        )

        pi3k = next(
            (f for f in fragments if f["target"] == "PI3K"),
            None,
        )
        self.assertIsNotNone(pi3k)
        self.assertEqual(pi3k["source"], "pEGFR")
        self.assertEqual(pi3k["mechanism"], "activation")

    def test_crosstalk_contains_fos_transcription(self):
        """含 pERK → Fos transcription 片段。"""
        specialist = self._EGFRSpecialist()
        fragments = specialist.apply_crosstalk(
            {"nodes": [], "edges": []}, []
        )

        fos = next(
            (f for f in fragments if f["target"] == "Fos"),
            None,
        )
        self.assertIsNotNone(fos)
        self.assertEqual(fos["source"], "pERK")
        self.assertEqual(fos["mechanism"], "transcription")

    def test_crosstalk_contains_akt_raf_inhibition(self):
        """含 AKT → Raf Ser259 inhibition 片段。"""
        specialist = self._EGFRSpecialist()
        fragments = specialist.apply_crosstalk(
            {"nodes": [], "edges": []}, []
        )

        akt = next(
            (f for f in fragments if f["source"] == "AKT"),
            None,
        )
        self.assertIsNotNone(akt)
        self.assertEqual(akt["target"], "Raf")
        self.assertEqual(akt["mechanism"], "inhibition")
        self.assertEqual(akt["site"], "Ser259")

    def test_crosstalk_fragments_contain_required_fields(self):
        """每条 cross-talk 片段含 source / target / mechanism / shared_species / description。"""
        specialist = self._EGFRSpecialist()
        fragments = specialist.apply_crosstalk(
            {"nodes": [], "edges": []}, []
        )

        for f in fragments:
            self.assertIn("source", f)
            self.assertIn("target", f)
            self.assertIn("mechanism", f)
            self.assertIn("shared_species", f)
            self.assertIn("description", f)


class TestEGFRPerturbationModule(unittest.TestCase):
    """测试 6：Perturbation 模块返回 4 个药物/突变。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.egfr_specialist import EGFRSpecialist

        clear_registry()
        register_specialist(EGFRSpecialist)
        self._EGFRSpecialist = EGFRSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_apply_perturbation_returns_4(self):
        """apply_perturbation() 返回 4 个扰动。"""
        specialist = self._EGFRSpecialist()
        perturbations = specialist.apply_perturbation(
            {"nodes": [], "edges": []}, []
        )

        self.assertEqual(
            len(perturbations),
            4,
            f"期望 4 个扰动，实际 {len(perturbations)} 个",
        )

    def test_perturbation_contains_gefitinib(self):
        """含 Gefitinib（EGFR 抑制剂）。"""
        specialist = self._EGFRSpecialist()
        perturbations = specialist.apply_perturbation(
            {"nodes": [], "edges": []}, []
        )

        gefitinib = next(
            (p for p in perturbations if p.get("drug") == "Gefitinib"),
            None,
        )
        self.assertIsNotNone(gefitinib)
        self.assertEqual(gefitinib["target"], "EGFR")
        self.assertEqual(gefitinib["mechanism"], "inhibition")

    def test_perturbation_contains_erlotinib(self):
        """含 Erlotinib（EGFR 抑制剂）。"""
        specialist = self._EGFRSpecialist()
        perturbations = specialist.apply_perturbation(
            {"nodes": [], "edges": []}, []
        )

        erlotinib = next(
            (p for p in perturbations if p.get("drug") == "Erlotinib"),
            None,
        )
        self.assertIsNotNone(erlotinib)
        self.assertEqual(erlotinib["target"], "EGFR")
        self.assertEqual(erlotinib["mechanism"], "inhibition")

    def test_perturbation_contains_cetuximab(self):
        """含 Cetuximab（EGFR 单克隆抗体）。"""
        specialist = self._EGFRSpecialist()
        perturbations = specialist.apply_perturbation(
            {"nodes": [], "edges": []}, []
        )

        cetuximab = next(
            (p for p in perturbations if p.get("drug") == "Cetuximab"),
            None,
        )
        self.assertIsNotNone(cetuximab)
        self.assertEqual(cetuximab["target"], "EGFR")
        self.assertEqual(cetuximab["mechanism"], "inhibition")

    def test_perturbation_contains_egfr_viii_mutation(self):
        """含 EGFR vIII 突变（组成性激活）。"""
        specialist = self._EGFRSpecialist()
        perturbations = specialist.apply_perturbation(
            {"nodes": [], "edges": []}, []
        )

        viii = next(
            (p for p in perturbations if p.get("ko_target") == "EGFR_vIII"),
            None,
        )
        self.assertIsNotNone(viii)
        self.assertEqual(viii["target"], "EGFR")
        self.assertEqual(viii["mechanism"], "activation")
        self.assertIsNone(viii["drug"])

    def test_perturbations_contain_required_fields(self):
        """每个扰动含 target / drug / mechanism / ko_target / description。"""
        specialist = self._EGFRSpecialist()
        perturbations = specialist.apply_perturbation(
            {"nodes": [], "edges": []}, []
        )

        for p in perturbations:
            self.assertIn("target", p)
            self.assertIn("drug", p)
            self.assertIn("mechanism", p)
            self.assertIn("ko_target", p)
            self.assertIn("description", p)


class TestEGFRValidationModule(unittest.TestCase):
    """测试 7：Validation 模块返回 3 条 benchmark。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.egfr_specialist import EGFRSpecialist

        clear_registry()
        register_specialist(EGFRSpecialist)
        self._EGFRSpecialist = EGFRSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_apply_validation_returns_3_rules(self):
        """apply_validation() 返回 3 条 Validation 规则。"""
        specialist = self._EGFRSpecialist()
        rules = specialist.apply_validation()

        self.assertEqual(
            len(rules),
            3,
            f"期望 3 条 Validation 规则，实际 {len(rules)} 条",
        )

    def test_pegfr_peak_time_rule(self):
        """pEGFR 达峰时间 5-10 min（PMID:11923475）。"""
        specialist = self._EGFRSpecialist()
        rules = specialist.apply_validation()

        peak = next(
            (r for r in rules if r["metric_name"] == "pEGFR_peak_time"),
            None,
        )
        self.assertIsNotNone(peak)
        self.assertEqual(peak["expected_min"], 5.0)
        self.assertEqual(peak["expected_max"], 10.0)
        self.assertEqual(peak["unit"], "minutes")
        self.assertEqual(peak["pmid"], "PMID:11923475")

    def test_mapk_amplification_rule(self):
        """MAPK 放大倍数 10-100x（PMID:11483517）。"""
        specialist = self._EGFRSpecialist()
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

    def test_egfr_internalization_half_life_rule(self):
        """EGFR 内吞半衰期 10-15 min（PMID:11923475）。"""
        specialist = self._EGFRSpecialist()
        rules = specialist.apply_validation()

        half_life = next(
            (
                r
                for r in rules
                if r["metric_name"] == "EGFR_internalization_half_life"
            ),
            None,
        )
        self.assertIsNotNone(half_life)
        self.assertEqual(half_life["expected_min"], 10.0)
        self.assertEqual(half_life["expected_max"], 15.0)
        self.assertEqual(half_life["unit"], "minutes")
        self.assertEqual(half_life["pmid"], "PMID:11923475")

    def test_validation_rules_contain_required_fields(self):
        """每条 Validation 规则含 rule_id / metric_name / expected / tolerance / pmid / description。"""
        specialist = self._EGFRSpecialist()
        rules = specialist.apply_validation()

        for r in rules:
            self.assertIn("rule_id", r)
            self.assertIn("metric_name", r)
            self.assertIn("expected", r)
            self.assertIn("tolerance", r)
            self.assertIn("pmid", r)
            self.assertIn("description", r)


class TestEGFRSelectTemplate(unittest.TestCase):
    """测试 8：模板选择 select_template('phosphorylation') → _mechanism_phosphorylation_mm。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.egfr_specialist import EGFRSpecialist

        clear_registry()
        register_specialist(EGFRSpecialist)
        self._EGFRSpecialist = EGFRSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_select_template_phosphorylation(self):
        """select_template('phosphorylation') 返回 _mechanism_phosphorylation_mm。"""
        specialist = self._EGFRSpecialist()
        self.assertEqual(
            specialist.select_template("phosphorylation"),
            "_mechanism_phosphorylation_mm",
        )


class TestEGFRFeatureFlagIsolation(unittest.TestCase):
    """测试 9：Feature Flag 隔离（flag=false 时 hook 不调用 Specialist）。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.egfr_specialist import EGFRSpecialist

        clear_registry()
        register_specialist(EGFRSpecialist)
        self._EGFRSpecialist = EGFRSpecialist

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
        expected_core = {"species": [{"name": "EGF"}], "reactions": []}
        specialist.apply_core.return_value = expected_core
        state = {"v4_pathway_graph": {"nodes": [{"id": "EGF"}], "edges": []}}

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


class TestEGFRMetadataAndInputValidation(unittest.TestCase):
    """测试 10 + 11：元数据 + 输入校验。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.egfr_specialist import EGFRSpecialist

        clear_registry()
        register_specialist(EGFRSpecialist)
        self._EGFRSpecialist = EGFRSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_get_metadata_returns_egfr_metadata(self):
        """get_metadata() 返回 pathway_class='EGFR_RTK' + 5 模块。"""
        specialist = self._EGFRSpecialist()
        metadata = specialist.get_metadata()

        self.assertEqual(metadata["pathway_class"], "EGFR_RTK")
        self.assertEqual(metadata["display_name"], "EGFR RTK Signaling")
        self.assertEqual(
            metadata["supported_modules"],
            ["core", "feedback", "crosstalk", "perturbation", "validation"],
        )
        self.assertIn("version", metadata)

    def test_validate_input_empty_graph_returns_warnings(self):
        """validate_input({}) 返回非空 warning 列表。"""
        specialist = self._EGFRSpecialist()
        warnings = specialist.validate_input({})

        self.assertIsInstance(warnings, list)
        self.assertGreater(len(warnings), 0, "空 pathway_graph 应返回 warning")

    def test_validate_input_none_returns_warnings(self):
        """validate_input(None) 返回非空 warning 列表。"""
        specialist = self._EGFRSpecialist()
        warnings = specialist.validate_input(None)

        self.assertGreater(len(warnings), 0)

    def test_validate_input_valid_graph_returns_no_warnings(self):
        """含 nodes + edges 的 pathway_graph 无 warning。"""
        specialist = self._EGFRSpecialist()
        warnings = specialist.validate_input(
            {"nodes": [{"id": "EGF"}], "edges": []}
        )
        self.assertEqual(warnings, [])


if __name__ == "__main__":
    unittest.main()
