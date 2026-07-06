# BioDynamics Agent v4 - JAK-STAT Specialist 单元测试 (Phase 4 / Task 4.9)
#
# 测试用例（11 项，覆盖 5 模块 + flag 隔离 + 元数据 + STAT5 状态机 +
# pSTAT5 shared + 模板选择）：
#   1. Specialist 注册：get_specialist("JAK_STAT") 返回 JakStatSpecialist 实例
#   2. Core 模块：apply_core() 返回 ≥9 条核心反应
#   3. STAT5 状态机：Core 输出含状态机（monomer→phosphorylated→dimer→nuclear）
#   4. pSTAT5 shared 标记：species 中 pSTAT5 标记 shared=True
#      （与 Apoptosis Specialist Bcl-xL 路径共享）
#   5. Feedback 模块：FL_STAT_SOCS delay=30min + SOCS→pJAK + PIAS→pSTAT5_dimer
#   6. Crosstalk 模块：apply_crosstalk() 返回 5 条 cross-talk Reaction 片段
#   7. Perturbation 模块：apply_perturbation() 返回 ≥5 个扰动
#   8. Validation 模块：apply_validation() 返回 3 条 benchmark
#   9. 模板选择：phosphorylation → _mechanism_phosphorylation_mm
#  10. Feature Flag 隔离：flag=false 时 hook 不调用 Specialist
#  11. 元数据：get_metadata() 返回 pathway_class='JAK_STAT'
#
# 运行：cd backend && python -m pytest tests/test_jak_stat_specialist.py -v

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
class TestJakStatSpecialistRegistration(unittest.TestCase):
    """测试 1：Specialist 注册到 SPECIALIST_REGISTRY。"""

    def setUp(self):
        """每个测试开始前清空 registry 并重新注册 JakStatSpecialist。"""
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.jak_stat_specialist import (
            JakStatSpecialist,
        )

        clear_registry()
        register_specialist(JakStatSpecialist)

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_get_specialist_returns_jak_stat_instance(self):
        """get_specialist("JAK_STAT") 返回 JakStatSpecialist 实例。"""
        from app.pathways.pathway_registry import (
            get_specialist,
            is_specialist_available,
        )
        from app.pathways.specialists.jak_stat_specialist import (
            JakStatSpecialist,
        )

        # 已注册
        self.assertTrue(is_specialist_available("JAK_STAT"))

        # get_specialist 返回 JakStatSpecialist 实例
        instance = get_specialist("JAK_STAT")
        self.assertIsNotNone(instance)
        self.assertIsInstance(instance, JakStatSpecialist)

        # 多次调用返回不同实例（非单例）
        instance2 = get_specialist("JAK_STAT")
        self.assertIsNot(instance, instance2)

    def test_pathway_class_attribute(self):
        """JakStatSpecialist.pathway_class == 'JAK_STAT'。"""
        from app.pathways.specialists.jak_stat_specialist import (
            JakStatSpecialist,
        )

        self.assertEqual(JakStatSpecialist.pathway_class, "JAK_STAT")


class TestJakStatCoreModule(unittest.TestCase):
    """测试 2：Core 模块返回 ≥9 条核心反应。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.jak_stat_specialist import (
            JakStatSpecialist,
        )

        clear_registry()
        register_specialist(JakStatSpecialist)
        self._JakStatSpecialist = JakStatSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_apply_core_returns_at_least_9_reactions(self):
        """apply_core() 返回 ≥9 条核心反应。"""
        specialist = self._JakStatSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})

        self.assertIn("species", result)
        self.assertIn("reactions", result)
        reactions = result["reactions"]
        self.assertGreaterEqual(
            len(reactions),
            9,
            f"期望至少 9 条核心反应，实际 {len(reactions)} 条",
        )

    def test_core_reaction_pairs(self):
        """9 条核心反应的 source→target 对应正确（IL6→JAK→STAT5→dimer→nuclear→转录）。"""
        specialist = self._JakStatSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        pairs = [(r["source"], r["target"]) for r in result["reactions"]]
        expected_pairs = [
            # IL-6 + 受体复合物 → JAK → STAT5 磷酸化级联
            ("IL6", "IL6_complex"),                       # 1. IL-6+IL6R+gp130 三元复合物
            ("IL6_complex", "pJAK"),                      # 2. IL6_complex 诱导 JAK 磷酸化
            ("pJAK", "pSTAT5"),                           # 3. pJAK 磷酸化 STAT5
            ("pSTAT5", "pSTAT5_dimer"),                   # 4. pSTAT5 二聚化
            ("pSTAT5_dimer", "pSTAT5_nuclear"),           # 5. pSTAT5_dimer 入核
            # 转录 3 条
            ("pSTAT5_nuclear", "SOCS_mRNA"),              # 6. STAT5 转录 SOCS
            ("SOCS_mRNA", "SOCS"),                        # 7. SOCS 翻译
            ("pSTAT5_nuclear", "Bcl_xL_mRNA"),            # 8. STAT3/5 转录 Bcl-xL
            ("pSTAT5_nuclear", "IRF1_mRNA"),              # 9. STAT3 转录 IRF1
        ]
        self.assertEqual(pairs, expected_pairs)

    def test_core_reactions_contain_pathway_tag(self):
        """每条核心反应含 pathway_tag='JAK_STAT'。"""
        specialist = self._JakStatSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        for r in result["reactions"]:
            self.assertEqual(r["pathway_tag"], "JAK_STAT")

    def test_core_reactions_contain_required_fields(self):
        """每条核心反应含 source / target / mechanism / kinetics_type /
        substrate / product / modifier / modifier_type。"""
        specialist = self._JakStatSpecialist()
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

    def test_core_jak_stat_cascade_complete(self):
        """JAK-STAT 级联完整（IL6→IL6_complex→pJAK→pSTAT5→dimer→nuclear→转录）。"""
        specialist = self._JakStatSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        # 同时收集 source 与 target
        nodes_in_reactions = set()
        for r in result["reactions"]:
            nodes_in_reactions.add(r["source"])
            nodes_in_reactions.add(r["target"])

        # JAK-STAT 级联关键节点
        for node in ["IL6", "IL6_complex", "pJAK", "pSTAT5",
                     "pSTAT5_dimer", "pSTAT5_nuclear", "SOCS_mRNA",
                     "Bcl_xL_mRNA", "IRF1_mRNA"]:
            self.assertIn(
                node, nodes_in_reactions,
                f"JAK-STAT 级联应含 {node}",
            )


class TestJakStatStat5StateMachine(unittest.TestCase):
    """测试 3：STAT5 状态机（monomer→phosphorylated→dimer→nuclear）。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.jak_stat_specialist import (
            JakStatSpecialist,
        )

        clear_registry()
        register_specialist(JakStatSpecialist)
        self._JakStatSpecialist = JakStatSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_core_contains_state_machine(self):
        """Core 输出含 state_machine 字段。"""
        specialist = self._JakStatSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})

        self.assertIn("state_machine", result)
        sm = result["state_machine"]
        self.assertIsInstance(sm, dict)
        self.assertGreater(len(sm), 0, "state_machine 不应为空")

    def test_stat5_state_machine_4_states(self):
        """STAT5 状态机含 4 状态：monomer / phosphorylated / dimer / nuclear。"""
        specialist = self._JakStatSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        sm = result["state_machine"]

        states = sm.get("states", [])
        state_names = [s["name"] for s in states]
        self.assertEqual(len(states), 4)
        self.assertIn("monomer", state_names)
        self.assertIn("phosphorylated", state_names)
        self.assertIn("dimer", state_names)
        self.assertIn("nuclear", state_names)

    def test_stat5_state_machine_initial_state(self):
        """STAT5 状态机初始状态为 monomer。"""
        specialist = self._JakStatSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        sm = result["state_machine"]

        initial_states = [s for s in sm.get("states", []) if s.get("is_initial")]
        self.assertEqual(len(initial_states), 1)
        self.assertEqual(initial_states[0]["name"], "monomer")

    def test_stat5_state_machine_transitions(self):
        """STAT5 状态机 3 个转换：monomer→phosphorylated→dimer→nuclear。"""
        specialist = self._JakStatSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        sm = result["state_machine"]

        transitions = sm.get("transitions", [])
        self.assertGreaterEqual(len(transitions), 3)

        # 检查转换链 monomer→phosphorylated→dimer→nuclear
        transition_pairs = [(t["from_state"], t["to_state"]) for t in transitions]
        self.assertIn(("monomer", "phosphorylated"), transition_pairs)
        self.assertIn(("phosphorylated", "dimer"), transition_pairs)
        self.assertIn(("dimer", "nuclear"), transition_pairs)


class TestJakStatPStat5SharedMarker(unittest.TestCase):
    """测试 4：pSTAT5 shared 标记（与 Apoptosis Specialist Bcl-xL 路径共享）。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.jak_stat_specialist import (
            JakStatSpecialist,
        )

        clear_registry()
        register_specialist(JakStatSpecialist)
        self._JakStatSpecialist = JakStatSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_pstat5_marked_as_shared(self):
        """species 中 pSTAT5 标记 shared=True（与 Apoptosis Specialist Bcl-xL 路径共享）。"""
        specialist = self._JakStatSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        species = result["species"]

        pstat5 = next(
            (s for s in species if s["name"] == "pSTAT5"),
            None,
        )
        self.assertIsNotNone(pstat5, "Core 输出应含 pSTAT5 物种")
        self.assertTrue(
            pstat5.get("shared"),
            "pSTAT5 应标记 shared=True（与 Apoptosis Specialist 的 Bcl-xL 抗凋亡路径共享）",
        )


class TestJakStatFeedbackModule(unittest.TestCase):
    """测试 5：Feedback 模块（FL_STAT_SOCS delay=30min + SOCS→pJAK + PIAS）。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.jak_stat_specialist import (
            JakStatSpecialist,
        )

        clear_registry()
        register_specialist(JakStatSpecialist)
        self._JakStatSpecialist = JakStatSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_apply_feedback_returns_at_least_3_loops(self):
        """apply_feedback() 返回 ≥3 条反馈环。"""
        specialist = self._JakStatSpecialist()
        loops = specialist.apply_feedback({"nodes": [], "edges": []})

        self.assertGreaterEqual(
            len(loops),
            3,
            f"期望至少 3 条反馈环，实际 {len(loops)} 条",
        )

    def test_fl_stat_socs_delayed_negative_feedback(self):
        """FL_STAT_SOCS 转录延迟负反馈（delay=30min）。"""
        specialist = self._JakStatSpecialist()
        loops = specialist.apply_feedback({"nodes": [], "edges": []})

        socs_loop = next(
            (l for l in loops if "STAT_SOCS" in l["id"]),
            None,
        )
        self.assertIsNotNone(socs_loop, "应含 FL_STAT_SOCS 反馈环")
        self.assertEqual(socs_loop["loop_type"], "negative")
        self.assertEqual(
            socs_loop["delay_minutes"],
            30.0,
            "FL_STAT_SOCS 延迟应为 30min（转录延迟负反馈）",
        )
        # 节点含 pSTAT5_nuclear / SOCS_mRNA / SOCS / pJAK
        node_ids = socs_loop["node_ids"]
        self.assertIn("pSTAT5_nuclear", node_ids)
        self.assertIn("SOCS_mRNA", node_ids)
        self.assertIn("SOCS", node_ids)
        self.assertIn("pJAK", node_ids)

    def test_socs_jak_inhibition_loop(self):
        """SOCS→pJAK 抑制反馈环（delay=0）。"""
        specialist = self._JakStatSpecialist()
        loops = specialist.apply_feedback({"nodes": [], "edges": []})

        socs_jak = next(
            (l for l in loops if "SOCS_JAK" in l["id"]),
            None,
        )
        self.assertIsNotNone(socs_jak, "应含 SOCS→pJAK 抑制反馈环")
        self.assertEqual(socs_jak["loop_type"], "negative")
        self.assertEqual(socs_jak["delay_minutes"], 0.0)
        # 节点含 SOCS + pJAK
        node_ids = socs_jak["node_ids"]
        self.assertIn("SOCS", node_ids)
        self.assertIn("pJAK", node_ids)

    def test_pias_stat_inhibition_loop(self):
        """PIAS→pSTAT5_dimer 抑制反馈环（delay=0）。"""
        specialist = self._JakStatSpecialist()
        loops = specialist.apply_feedback({"nodes": [], "edges": []})

        pias_loop = next(
            (l for l in loops if "PIAS_STAT" in l["id"]),
            None,
        )
        self.assertIsNotNone(pias_loop, "应含 PIAS→STAT 抑制反馈环")
        self.assertEqual(pias_loop["loop_type"], "negative")
        self.assertEqual(pias_loop["delay_minutes"], 0.0)
        # 节点含 PIAS + pSTAT5_dimer
        node_ids = pias_loop["node_ids"]
        self.assertIn("PIAS", node_ids)
        self.assertIn("pSTAT5_dimer", node_ids)

    def test_feedback_loops_contain_required_fields(self):
        """每条反馈环含 id / loop_type / node_ids / delay_minutes / description。"""
        specialist = self._JakStatSpecialist()
        loops = specialist.apply_feedback({"nodes": [], "edges": []})

        for loop in loops:
            self.assertIn("id", loop)
            self.assertIn("loop_type", loop)
            self.assertIn("node_ids", loop)
            self.assertIn("delay_minutes", loop)
            self.assertIn("description", loop)


class TestJakStatCrosstalkModule(unittest.TestCase):
    """测试 6：Crosstalk 模块返回 5 条 cross-talk Reaction 片段。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.jak_stat_specialist import (
            JakStatSpecialist,
        )

        clear_registry()
        register_specialist(JakStatSpecialist)
        self._JakStatSpecialist = JakStatSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_apply_crosstalk_returns_5_fragments(self):
        """apply_crosstalk() 返回 5 条 cross-talk Reaction 片段。"""
        specialist = self._JakStatSpecialist()
        fragments = specialist.apply_crosstalk(
            {"nodes": [], "edges": []}, []
        )

        self.assertEqual(
            len(fragments),
            5,
            f"期望 5 条 cross-talk 片段，实际 {len(fragments)} 条",
        )

    def test_crosstalk_contains_pstat3_bcl_xl_transcription(self):
        """含 pSTAT3 → Bcl_xL transcription 片段（STAT3 抗凋亡）。"""
        specialist = self._JakStatSpecialist()
        fragments = specialist.apply_crosstalk(
            {"nodes": [], "edges": []}, []
        )

        bclxl = next(
            (f for f in fragments
             if f["source"] == "pSTAT3" and f["target"] == "Bcl_xL"),
            None,
        )
        self.assertIsNotNone(bclxl)
        self.assertEqual(bclxl["mechanism"], "transcription")

    def test_crosstalk_contains_pstat3_bcl2_transcription(self):
        """含 pSTAT3 → Bcl2 transcription 片段。"""
        specialist = self._JakStatSpecialist()
        fragments = specialist.apply_crosstalk(
            {"nodes": [], "edges": []}, []
        )

        bcl2 = next(
            (f for f in fragments
             if f["source"] == "pSTAT3" and f["target"] == "Bcl2"),
            None,
        )
        self.assertIsNotNone(bcl2)
        self.assertEqual(bcl2["mechanism"], "transcription")

    def test_crosstalk_contains_egfr_pstat3_activation(self):
        """含 EGFR → pSTAT3 activation 片段（EGFR 旁路激活 STAT3）。"""
        specialist = self._JakStatSpecialist()
        fragments = specialist.apply_crosstalk(
            {"nodes": [], "edges": []}, []
        )

        egfr = next(
            (f for f in fragments
             if f["source"] == "EGFR" and f["target"] == "pSTAT3"),
            None,
        )
        self.assertIsNotNone(egfr)
        self.assertEqual(egfr["mechanism"], "activation")

    def test_crosstalk_contains_perk_pstat3_phosphorylation(self):
        """含 pERK → pSTAT3 phosphorylation 片段（Ser727）。"""
        specialist = self._JakStatSpecialist()
        fragments = specialist.apply_crosstalk(
            {"nodes": [], "edges": []}, []
        )

        perk = next(
            (f for f in fragments
             if f["source"] == "pERK" and f["target"] == "pSTAT3"),
            None,
        )
        self.assertIsNotNone(perk)
        self.assertEqual(perk["mechanism"], "phosphorylation")
        self.assertEqual(perk.get("site"), "Ser727")

    def test_crosstalk_contains_il6_pakt_activation(self):
        """含 IL6 → pAKT activation 片段（IL-6 激活 PI3K-AKT 旁路）。"""
        specialist = self._JakStatSpecialist()
        fragments = specialist.apply_crosstalk(
            {"nodes": [], "edges": []}, []
        )

        akt = next(
            (f for f in fragments
             if f["source"] == "IL6" and f["target"] == "pAKT"),
            None,
        )
        self.assertIsNotNone(akt)
        self.assertEqual(akt["mechanism"], "activation")

    def test_crosstalk_fragments_contain_required_fields(self):
        """每条 cross-talk 片段含 source / target / mechanism / shared_species /
        description。"""
        specialist = self._JakStatSpecialist()
        fragments = specialist.apply_crosstalk(
            {"nodes": [], "edges": []}, []
        )

        for f in fragments:
            self.assertIn("source", f)
            self.assertIn("target", f)
            self.assertIn("mechanism", f)
            self.assertIn("shared_species", f)
            self.assertIn("description", f)


class TestJakStatPerturbationModule(unittest.TestCase):
    """测试 7：Perturbation 模块返回 ≥5 个扰动。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.jak_stat_specialist import (
            JakStatSpecialist,
        )

        clear_registry()
        register_specialist(JakStatSpecialist)
        self._JakStatSpecialist = JakStatSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_apply_perturbation_returns_at_least_5(self):
        """apply_perturbation() 返回 ≥5 个扰动。"""
        specialist = self._JakStatSpecialist()
        perturbations = specialist.apply_perturbation(
            {"nodes": [], "edges": []}, []
        )

        self.assertGreaterEqual(
            len(perturbations),
            5,
            f"期望至少 5 个扰动，实际 {len(perturbations)} 个",
        )

    def test_perturbation_contains_tofacitinib(self):
        """含 Tofacitinib（JAK1/3 抑制剂, FDA-approved）。"""
        specialist = self._JakStatSpecialist()
        perturbations = specialist.apply_perturbation(
            {"nodes": [], "edges": []}, []
        )

        tofa = next(
            (p for p in perturbations if p.get("drug") == "Tofacitinib"),
            None,
        )
        self.assertIsNotNone(tofa)
        self.assertEqual(tofa["target"], "JAK")
        self.assertEqual(tofa["mechanism"], "inhibition")

    def test_perturbation_contains_ruxolitinib(self):
        """含 Ruxolitinib（JAK1/2 抑制剂, FDA-approved）。"""
        specialist = self._JakStatSpecialist()
        perturbations = specialist.apply_perturbation(
            {"nodes": [], "edges": []}, []
        )

        rux = next(
            (p for p in perturbations if p.get("drug") == "Ruxolitinib"),
            None,
        )
        self.assertIsNotNone(rux)
        self.assertEqual(rux["target"], "JAK")
        self.assertEqual(rux["mechanism"], "inhibition")

    def test_perturbation_contains_baricitinib(self):
        """含 Baricitinib（JAK1/2 抑制剂, FDA-approved）。"""
        specialist = self._JakStatSpecialist()
        perturbations = specialist.apply_perturbation(
            {"nodes": [], "edges": []}, []
        )

        bari = next(
            (p for p in perturbations if p.get("drug") == "Baricitinib"),
            None,
        )
        self.assertIsNotNone(bari)
        self.assertEqual(bari["target"], "JAK")

    def test_perturbation_contains_fedratinib(self):
        """含 Fedratinib（JAK2 抑制剂, FDA-approved）。"""
        specialist = self._JakStatSpecialist()
        perturbations = specialist.apply_perturbation(
            {"nodes": [], "edges": []}, []
        )

        fed = next(
            (p for p in perturbations if p.get("drug") == "Fedratinib"),
            None,
        )
        self.assertIsNotNone(fed)
        self.assertEqual(fed["target"], "JAK")
        self.assertEqual(fed["mechanism"], "inhibition")

    def test_perturbation_contains_socs1_oe(self):
        """含 SOCS1 overexpression（gain-of-function, 增强 JAK 抑制）。"""
        specialist = self._JakStatSpecialist()
        perturbations = specialist.apply_perturbation(
            {"nodes": [], "edges": []}, []
        )

        socs1 = next(
            (p for p in perturbations
             if p.get("ko_target") == "SOCS1_OE"),
            None,
        )
        self.assertIsNotNone(socs1)
        self.assertEqual(socs1["target"], "JAK")
        self.assertEqual(socs1["mechanism"], "knockout")

    def test_perturbations_contain_required_fields(self):
        """每个扰动含 target / drug / mechanism / ko_target / description。"""
        specialist = self._JakStatSpecialist()
        perturbations = specialist.apply_perturbation(
            {"nodes": [], "edges": []}, []
        )

        for p in perturbations:
            self.assertIn("target", p)
            self.assertIn("drug", p)
            self.assertIn("mechanism", p)
            self.assertIn("ko_target", p)
            self.assertIn("description", p)


class TestJakStatValidationModule(unittest.TestCase):
    """测试 8：Validation 模块返回 3 条 benchmark。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.jak_stat_specialist import (
            JakStatSpecialist,
        )

        clear_registry()
        register_specialist(JakStatSpecialist)
        self._JakStatSpecialist = JakStatSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_apply_validation_returns_3_rules(self):
        """apply_validation() 返回 3 条 Validation 规则。"""
        specialist = self._JakStatSpecialist()
        rules = specialist.apply_validation()

        self.assertEqual(
            len(rules),
            3,
            f"期望 3 条 Validation 规则，实际 {len(rules)} 条",
        )

    def test_pstat5_peak_time_rule(self):
        """pSTAT5 达峰时间 5-15 min（Schwartz 2003, PMID:15286703）。"""
        specialist = self._JakStatSpecialist()
        rules = specialist.apply_validation()

        peak = next(
            (r for r in rules
             if r["metric_name"] == "pSTAT5_peak_time"),
            None,
        )
        self.assertIsNotNone(peak)
        self.assertEqual(peak["expected_min"], 5.0)
        self.assertEqual(peak["expected_max"], 15.0)
        self.assertEqual(peak["unit"], "minutes")
        self.assertEqual(peak["pmid"], "PMID:15286703")

    def test_socs_mrna_delay_rule(self):
        """SOCS mRNA 延迟 30-60 min（Schwartz 2003, PMID:15286703）。"""
        specialist = self._JakStatSpecialist()
        rules = specialist.apply_validation()

        delay = next(
            (r for r in rules
             if r["metric_name"] == "SOCS_mRNA_delay"),
            None,
        )
        self.assertIsNotNone(delay)
        self.assertEqual(delay["expected_min"], 30.0)
        self.assertEqual(delay["expected_max"], 60.0)
        self.assertEqual(delay["unit"], "minutes")
        self.assertEqual(delay["pmid"], "PMID:15286703")

    def test_stat5_nuclear_pulse_rule(self):
        """STAT5 核质比单脉冲（Schwartz 2003, PMID:15286703）。"""
        specialist = self._JakStatSpecialist()
        rules = specialist.apply_validation()

        pulse = next(
            (r for r in rules
             if r["metric_name"] == "STAT5_nuclear_cytoplasmic_ratio_pulse"),
            None,
        )
        self.assertIsNotNone(pulse)
        self.assertEqual(pulse["expected"], True)
        self.assertEqual(pulse["pmid"], "PMID:15286703")

    def test_validation_rules_contain_required_fields(self):
        """每条 Validation 规则含 rule_id / metric_name / expected /
        tolerance / pmid / description。"""
        specialist = self._JakStatSpecialist()
        rules = specialist.apply_validation()

        for r in rules:
            self.assertIn("rule_id", r)
            self.assertIn("metric_name", r)
            self.assertIn("expected", r)
            self.assertIn("tolerance", r)
            self.assertIn("pmid", r)
            self.assertIn("description", r)


class TestJakStatSelectTemplate(unittest.TestCase):
    """测试 9：模板选择（phosphorylation → _mechanism_phosphorylation_mm）。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.jak_stat_specialist import (
            JakStatSpecialist,
        )

        clear_registry()
        register_specialist(JakStatSpecialist)
        self._JakStatSpecialist = JakStatSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_select_template_phosphorylation(self):
        """select_template('phosphorylation') 返回 _mechanism_phosphorylation_mm。"""
        specialist = self._JakStatSpecialist()
        self.assertEqual(
            specialist.select_template("phosphorylation"),
            "_mechanism_phosphorylation_mm",
        )

    def test_select_template_transcription_fallback(self):
        """select_template('transcription') 在 transcription_factor.j2 不存在时
        降级到 _mechanism_phosphorylation_mm。"""
        specialist = self._JakStatSpecialist()
        # P3 transcription_factor.j2 当前未实现，降级到 _mechanism_phosphorylation_mm
        self.assertEqual(
            specialist.select_template("transcription"),
            "_mechanism_phosphorylation_mm",
        )


class TestJakStatFeatureFlagIsolation(unittest.TestCase):
    """测试 10：Feature Flag 隔离（flag=false 时 hook 不调用 Specialist）。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.jak_stat_specialist import (
            JakStatSpecialist,
        )

        clear_registry()
        register_specialist(JakStatSpecialist)
        self._JakStatSpecialist = JakStatSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_flag_false_hook_does_not_call_specialist(self):
        """flag=false 时 hook 返回空 dict 且不调用 Specialist 任何方法。"""
        specialist = MagicMock()
        specialist.apply_core.return_value = {
            "species": [],
            "reactions": [],
            "state_machine": {},
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
            "species": [{"name": "pSTAT5"}],
            "reactions": [],
            "state_machine": {"id": "SM_STAT5"},
        }
        specialist.apply_core.return_value = expected_core
        state = {"v4_pathway_graph": {"nodes": [{"id": "pSTAT5"}], "edges": []}}

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


class TestJakStatMetadataAndInputValidation(unittest.TestCase):
    """测试 11：元数据 + 输入校验。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.jak_stat_specialist import (
            JakStatSpecialist,
        )

        clear_registry()
        register_specialist(JakStatSpecialist)
        self._JakStatSpecialist = JakStatSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_get_metadata_returns_jak_stat_metadata(self):
        """get_metadata() 返回 pathway_class='JAK_STAT' + 5 模块。"""
        specialist = self._JakStatSpecialist()
        metadata = specialist.get_metadata()

        self.assertEqual(metadata["pathway_class"], "JAK_STAT")
        self.assertEqual(
            metadata["display_name"],
            "JAK-STAT Signaling",
        )
        self.assertEqual(
            metadata["supported_modules"],
            ["core", "feedback", "crosstalk", "perturbation", "validation"],
        )
        self.assertEqual(len(metadata["supported_modules"]), 5)
        self.assertIn("version", metadata)

    def test_validate_input_empty_graph_returns_warnings(self):
        """validate_input({}) 返回非空 warning 列表。"""
        specialist = self._JakStatSpecialist()
        warnings = specialist.validate_input({})

        self.assertIsInstance(warnings, list)
        self.assertGreater(len(warnings), 0, "空 pathway_graph 应返回 warning")

    def test_validate_input_none_returns_warnings(self):
        """validate_input(None) 返回非空 warning 列表。"""
        specialist = self._JakStatSpecialist()
        warnings = specialist.validate_input(None)

        self.assertGreater(len(warnings), 0)

    def test_validate_input_valid_graph_returns_no_warnings(self):
        """含 nodes + edges 的 pathway_graph 无 warning。"""
        specialist = self._JakStatSpecialist()
        warnings = specialist.validate_input(
            {"nodes": [{"id": "pSTAT5"}], "edges": []}
        )
        self.assertEqual(warnings, [])


if __name__ == "__main__":
    unittest.main()
