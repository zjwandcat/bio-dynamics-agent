# BioDynamics Agent v4 - p53 Specialist 单元测试 (Phase 4 / Task 4.6)
#
# 测试用例（11 项，覆盖 5 模块 + flag 隔离 + 元数据 + p53 状态机 + p53/p21 共享）：
#   1. Specialist 注册：get_specialist("p53") 返回 P53Specialist 实例
#   2. Core 模块：apply_core() 返回 10 条核心反应
#   3. p53 状态机：Core 输出含状态机信息（monomer→phosphorylated→tetramer→nuclear）
#   4. p53/p21 shared 标记：species 中 p53 和 p21 标记 shared=True
#   5. Feedback 模块：apply_feedback() 返回 p53-Mdm2 反馈环 delay=60min（DDE 场景）
#   6. Crosstalk 模块：apply_crosstalk() 返回 4 条 cross-talk Reaction 片段
#   7. Perturbation 模块：apply_perturbation() 返回 5 个扰动
#   8. Validation 模块：apply_validation() 返回 3 条 benchmark（含 pmid）
#   9. 模板选择：phosphorylation → _mechanism_phosphorylation_mm；oscillatory → oscillatory_feedback
#  10. Feature Flag 隔离：flag=false 时 hook 不调用 Specialist
#  11. 元数据：get_metadata() 返回 pathway_class='p53' / supported_modules 5 项
#
# 运行：cd backend && python -m pytest tests/test_p53_specialist.py -v

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
class TestP53SpecialistRegistration(unittest.TestCase):
    """测试 1：Specialist 注册到 SPECIALIST_REGISTRY。"""

    def setUp(self):
        """每个测试开始前清空 registry 并重新注册 P53Specialist。"""
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.p53_specialist import P53Specialist

        clear_registry()
        register_specialist(P53Specialist)

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_get_specialist_returns_p53_instance(self):
        """get_specialist("p53") 返回 P53Specialist 实例。"""
        from app.pathways.pathway_registry import (
            get_specialist,
            is_specialist_available,
        )
        from app.pathways.specialists.p53_specialist import P53Specialist

        # 已注册
        self.assertTrue(is_specialist_available("p53"))

        # get_specialist 返回 P53Specialist 实例
        instance = get_specialist("p53")
        self.assertIsNotNone(instance)
        self.assertIsInstance(instance, P53Specialist)

        # 多次调用返回不同实例（非单例）
        instance2 = get_specialist("p53")
        self.assertIsNot(instance, instance2)

    def test_pathway_class_attribute(self):
        """P53Specialist.pathway_class == 'p53'。"""
        from app.pathways.specialists.p53_specialist import P53Specialist

        self.assertEqual(P53Specialist.pathway_class, "p53")


class TestP53CoreModule(unittest.TestCase):
    """测试 2：Core 模块返回 10 条核心反应。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.p53_specialist import P53Specialist

        clear_registry()
        register_specialist(P53Specialist)
        self._P53Specialist = P53Specialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_apply_core_returns_10_reactions(self):
        """apply_core() 返回 10 条核心反应。"""
        specialist = self._P53Specialist()
        result = specialist.apply_core({"nodes": [], "edges": []})

        self.assertIn("species", result)
        self.assertIn("reactions", result)
        reactions = result["reactions"]
        self.assertEqual(
            len(reactions),
            10,
            f"期望 10 条核心反应，实际 {len(reactions)} 条",
        )

    def test_core_reaction_pairs(self):
        """10 条核心反应的 source→target 对应正确（DNA damage→pATM→p53→...→p21）。"""
        specialist = self._P53Specialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        pairs = [(r["source"], r["target"]) for r in result["reactions"]]
        expected_pairs = [
            ("DNA_damage", "pATM"),       # 1. DNA damage 激活 ATM
            ("pATM", "p53"),              # 2. pATM 磷酸化 p53
            ("p53", "p53_tetramer"),      # 3. p53 四聚化
            ("p53_tetramer", "p53_nuclear"),  # 4. p53 tetramer 入核
            ("p53_nuclear", "Mdm2_mRNA"), # 5. p53 转录 Mdm2 mRNA
            ("Mdm2_mRNA", "Mdm2"),        # 6. Mdm2 mRNA 翻译
            ("Mdm2", "p53_ubi"),          # 7. Mdm2 泛素化 p53
            ("p53_ubi", "p53"),           # 8. p53_ubi 蛋白酶体降解
            ("p53_nuclear", "p21_mRNA"),  # 9. p53 转录 p21 mRNA
            ("p21_mRNA", "p21"),          # 10. p21 mRNA 翻译
        ]
        self.assertEqual(pairs, expected_pairs)

    def test_core_reactions_contain_pathway_tag(self):
        """每条核心反应含 pathway_tag='p53'。"""
        specialist = self._P53Specialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        for r in result["reactions"]:
            self.assertEqual(r["pathway_tag"], "p53")

    def test_core_reactions_contain_required_fields(self):
        """每条核心反应含 source / target / mechanism / kinetics_type /
        substrate / product / modifier / modifier_type。"""
        specialist = self._P53Specialist()
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

    def test_core_patm_p53_hetero_phosphorylation(self):
        """pATM → p53 反应是异磷酸化（p53 作 substrate/product，pATM 作 catalytic modifier）。"""
        specialist = self._P53Specialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        reactions = result["reactions"]

        # pATM → p53（第 2 条，index=1）
        rxn = reactions[1]
        self.assertEqual(rxn["source"], "pATM")
        self.assertEqual(rxn["target"], "p53")
        self.assertFalse(rxn["autophosphorylation"])
        self.assertEqual(rxn["substrate"], "p53")
        self.assertEqual(rxn["product"], "p53")
        self.assertEqual(rxn["modifier"], "pATM")
        self.assertEqual(rxn["modifier_type"], "catalytic")
        self.assertEqual(rxn.get("site"), "Ser15/Thr20")


class TestP53StateMachine(unittest.TestCase):
    """测试 3：p53 状态机（monomer→phosphorylated→tetramer→nuclear）。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.p53_specialist import P53Specialist

        clear_registry()
        register_specialist(P53Specialist)
        self._P53Specialist = P53Specialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_core_contains_state_machine(self):
        """Core 输出含 state_machine 字段。"""
        specialist = self._P53Specialist()
        result = specialist.apply_core({"nodes": [], "edges": []})

        self.assertIn("state_machine", result)
        sm = result["state_machine"]
        self.assertIsInstance(sm, dict)
        self.assertGreater(len(sm), 0, "state_machine 不应为空")

    def test_state_machine_states_contain_four_states(self):
        """状态机含 4 个状态（monomer / phosphorylated / tetramer / nuclear）。"""
        specialist = self._P53Specialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        sm = result["state_machine"]

        states = sm.get("states", [])
        state_names = [s["name"] for s in states]
        self.assertEqual(
            state_names,
            ["monomer", "phosphorylated", "tetramer", "nuclear"],
        )

    def test_state_machine_transitions(self):
        """状态机含 3 个转换（monomer→phosphorylated→tetramer→nuclear）。"""
        specialist = self._P53Specialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        sm = result["state_machine"]

        transitions = sm.get("transitions", [])
        self.assertEqual(len(transitions), 3)

        # 检查转换链：monomer→phosphorylated→tetramer→nuclear
        chain = [(t["from"], t["to"]) for t in transitions]
        self.assertIn(("monomer", "phosphorylated"), chain)
        self.assertIn(("phosphorylated", "tetramer"), chain)
        self.assertIn(("tetramer", "nuclear"), chain)

    def test_state_machine_initial_state(self):
        """状态机 initial_state='monomer'。"""
        specialist = self._P53Specialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        sm = result["state_machine"]

        self.assertEqual(sm.get("initial_state"), "monomer")


class TestP53P21SharedMarker(unittest.TestCase):
    """测试 4：p53/p21 shared 标记。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.p53_specialist import P53Specialist

        clear_registry()
        register_specialist(P53Specialist)
        self._P53Specialist = P53Specialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_p53_marked_as_shared(self):
        """species 中 p53 标记 shared=True（与 Apoptosis Bax/PUMA 路径共享）。"""
        specialist = self._P53Specialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        species = result["species"]

        p53 = next(
            (s for s in species if s["name"] == "p53"),
            None,
        )
        self.assertIsNotNone(p53, "Core 输出应含 p53 物种")
        self.assertTrue(
            p53.get("shared"),
            "p53 应标记 shared=True（与 Apoptosis Bax/PUMA 路径共享）",
        )

    def test_p21_marked_as_shared(self):
        """species 中 p21 标记 shared=True（与 Cell Cycle CDK2/CDK4 抑制路径共享）。"""
        specialist = self._P53Specialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        species = result["species"]

        p21 = next(
            (s for s in species if s["name"] == "p21"),
            None,
        )
        self.assertIsNotNone(p21, "Core 输出应含 p21 物种")
        self.assertTrue(
            p21.get("shared"),
            "p21 应标记 shared=True（与 Cell Cycle CDK2/CDK4 抑制路径共享）",
        )


class TestP53FeedbackModule(unittest.TestCase):
    """测试 5：Feedback 模块返回 p53-Mdm2 反馈环 delay=60min（DDE 场景）。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.p53_specialist import P53Specialist

        clear_registry()
        register_specialist(P53Specialist)
        self._P53Specialist = P53Specialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_apply_feedback_returns_loops(self):
        """apply_feedback() 返回至少 1 条反馈环（p53-Mdm2 DDE）。"""
        specialist = self._P53Specialist()
        loops = specialist.apply_feedback({"nodes": [], "edges": []})

        self.assertGreaterEqual(
            len(loops),
            1,
            f"期望至少 1 条反馈环，实际 {len(loops)} 条",
        )

    def test_p53_mdm2_feedback_delay_60(self):
        """p53-Mdm2 反馈环 delay=60 min（DDE 延迟负反馈）。"""
        specialist = self._P53Specialist()
        loops = specialist.apply_feedback({"nodes": [], "edges": []})

        # 查找 FL_P53_MDM2 反馈环
        mdm2_loop = next(
            (l for l in loops if "P53_MDM2" in l["id"] or "MDM2" in l["id"]),
            None,
        )
        self.assertIsNotNone(mdm2_loop, "应含 p53-Mdm2 反馈环")
        self.assertEqual(
            mdm2_loop["loop_type"],
            "negative",
            "p53-Mdm2 应为负反馈",
        )
        self.assertEqual(
            mdm2_loop["delay_minutes"],
            60.0,
            "p53-Mdm2 反馈环 delay 应为 60 min（DDE 转录延迟）",
        )

    def test_p53_mdm2_feedback_contains_dde_solver(self):
        """p53-Mdm2 反馈环标注 DDE 求解器（solvers/dde_solver.py）。"""
        specialist = self._P53Specialist()
        loops = specialist.apply_feedback({"nodes": [], "edges": []})

        mdm2_loop = next(
            (l for l in loops if "MDM2" in l["id"]),
            None,
        )
        self.assertIsNotNone(mdm2_loop)
        # DDE 求解器标注
        self.assertIn("dde_solver", mdm2_loop)
        self.assertIn("dde_solver.py", mdm2_loop["dde_solver"])

    def test_p53_mdm2_feedback_node_ids(self):
        """p53-Mdm2 反馈环节点含 p53_nuclear / Mdm2_mRNA / Mdm2。"""
        specialist = self._P53Specialist()
        loops = specialist.apply_feedback({"nodes": [], "edges": []})

        mdm2_loop = next(
            (l for l in loops if "MDM2" in l["id"]),
            None,
        )
        self.assertIsNotNone(mdm2_loop)
        node_ids = mdm2_loop["node_ids"]
        self.assertIn("p53_nuclear", node_ids)
        self.assertIn("Mdm2_mRNA", node_ids)
        self.assertIn("Mdm2", node_ids)

    def test_atm_bidirectional_feedback(self):
        """ATM 双向调控反馈环（ATM→p53 激活 + ATM→Mdm2 抑制）。"""
        specialist = self._P53Specialist()
        loops = specialist.apply_feedback({"nodes": [], "edges": []})

        atm_loop = next(
            (l for l in loops if "ATM" in l["id"]),
            None,
        )
        self.assertIsNotNone(atm_loop, "应含 ATM 双向调控反馈环")
        node_ids = atm_loop["node_ids"]
        self.assertIn("pATM", node_ids)
        self.assertIn("p53", node_ids)
        self.assertIn("Mdm2", node_ids)

    def test_feedback_loops_contain_required_fields(self):
        """每条反馈环含 id / loop_type / node_ids / delay_minutes / description。"""
        specialist = self._P53Specialist()
        loops = specialist.apply_feedback({"nodes": [], "edges": []})

        for loop in loops:
            self.assertIn("id", loop)
            self.assertIn("loop_type", loop)
            self.assertIn("node_ids", loop)
            self.assertIn("delay_minutes", loop)
            self.assertIn("description", loop)


class TestP53CrosstalkModule(unittest.TestCase):
    """测试 6：Crosstalk 模块返回 4 条 cross-talk Reaction 片段。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.p53_specialist import P53Specialist

        clear_registry()
        register_specialist(P53Specialist)
        self._P53Specialist = P53Specialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_apply_crosstalk_returns_4_fragments(self):
        """apply_crosstalk() 返回 4 条 cross-talk Reaction 片段。"""
        specialist = self._P53Specialist()
        fragments = specialist.apply_crosstalk(
            {"nodes": [], "edges": []}, []
        )

        self.assertEqual(
            len(fragments),
            4,
            f"期望 4 条 cross-talk 片段，实际 {len(fragments)} 条",
        )

    def test_crosstalk_contains_bax_transcription(self):
        """含 p53 → Bax transcription 片段。"""
        specialist = self._P53Specialist()
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

    def test_crosstalk_contains_puma_transcription(self):
        """含 p53 → PUMA transcription 片段。"""
        specialist = self._P53Specialist()
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

    def test_crosstalk_contains_p21_transcription(self):
        """含 p53 → p21 transcription 片段。"""
        specialist = self._P53Specialist()
        fragments = specialist.apply_crosstalk(
            {"nodes": [], "edges": []}, []
        )

        p21 = next(
            (f for f in fragments if f["target"] == "p21"),
            None,
        )
        self.assertIsNotNone(p21)
        self.assertEqual(p21["source"], "p53")
        self.assertEqual(p21["mechanism"], "transcription")

    def test_crosstalk_contains_akt_mdm2_activation(self):
        """含 AKT → Mdm2 activation 片段（Ser166）。"""
        specialist = self._P53Specialist()
        fragments = specialist.apply_crosstalk(
            {"nodes": [], "edges": []}, []
        )

        mdm2 = next(
            (f for f in fragments if f["target"] == "Mdm2"),
            None,
        )
        self.assertIsNotNone(mdm2)
        self.assertEqual(mdm2["source"], "AKT")
        self.assertEqual(mdm2["mechanism"], "activation")
        self.assertEqual(mdm2.get("site"), "Ser166")

    def test_crosstalk_fragments_contain_required_fields(self):
        """每条 cross-talk 片段含 source / target / mechanism / shared_species / description。"""
        specialist = self._P53Specialist()
        fragments = specialist.apply_crosstalk(
            {"nodes": [], "edges": []}, []
        )

        for f in fragments:
            self.assertIn("source", f)
            self.assertIn("target", f)
            self.assertIn("mechanism", f)
            self.assertIn("shared_species", f)
            self.assertIn("description", f)


class TestP53PerturbationModule(unittest.TestCase):
    """测试 7：Perturbation 模块返回 5 个扰动。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.p53_specialist import P53Specialist

        clear_registry()
        register_specialist(P53Specialist)
        self._P53Specialist = P53Specialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_apply_perturbation_returns_5(self):
        """apply_perturbation() 返回 5 个扰动。"""
        specialist = self._P53Specialist()
        perturbations = specialist.apply_perturbation(
            {"nodes": [], "edges": []}, []
        )

        self.assertEqual(
            len(perturbations),
            5,
            f"期望 5 个扰动，实际 {len(perturbations)} 个",
        )

    def test_perturbation_contains_nutlin3(self):
        """含 Nutlin-3（Mdm2-p53 相互作用抑制剂）。"""
        specialist = self._P53Specialist()
        perturbations = specialist.apply_perturbation(
            {"nodes": [], "edges": []}, []
        )

        nutlin = next(
            (p for p in perturbations if p.get("drug") == "Nutlin-3"),
            None,
        )
        self.assertIsNotNone(nutlin)
        self.assertEqual(nutlin["target"], "Mdm2")
        self.assertEqual(nutlin["mechanism"], "inhibition")

    def test_perturbation_contains_prima1(self):
        """含 PRIMA-1（p53 突变体再激活剂）。"""
        specialist = self._P53Specialist()
        perturbations = specialist.apply_perturbation(
            {"nodes": [], "edges": []}, []
        )

        prima = next(
            (p for p in perturbations if p.get("drug") == "PRIMA-1"),
            None,
        )
        self.assertIsNotNone(prima)
        self.assertEqual(prima["target"], "p53")
        self.assertEqual(prima["mechanism"], "activation")

    def test_perturbation_contains_tp53_r175h(self):
        """含 TP53 R175H（p53 DNA 结合域结构性突变）。"""
        specialist = self._P53Specialist()
        perturbations = specialist.apply_perturbation(
            {"nodes": [], "edges": []}, []
        )

        r175h = next(
            (p for p in perturbations if p.get("ko_target") == "TP53_R175H"),
            None,
        )
        self.assertIsNotNone(r175h)
        self.assertEqual(r175h["target"], "p53")
        self.assertEqual(r175h["mechanism"], "knockout")

    def test_perturbation_contains_tp53_r273h(self):
        """含 TP53 R273H（p53 DNA 结合域接触面突变）。"""
        specialist = self._P53Specialist()
        perturbations = specialist.apply_perturbation(
            {"nodes": [], "edges": []}, []
        )

        r273h = next(
            (p for p in perturbations if p.get("ko_target") == "TP53_R273H"),
            None,
        )
        self.assertIsNotNone(r273h)
        self.assertEqual(r273h["target"], "p53")
        self.assertEqual(r273h["mechanism"], "knockout")

    def test_perturbation_contains_5fluorouracil(self):
        """含 5-Fluorouracil（DNA damage 诱导剂，化疗药物）。"""
        specialist = self._P53Specialist()
        perturbations = specialist.apply_perturbation(
            {"nodes": [], "edges": []}, []
        )

        fu = next(
            (p for p in perturbations if p.get("drug") == "5-Fluorouracil"),
            None,
        )
        self.assertIsNotNone(fu)
        self.assertEqual(fu["target"], "DNA_damage")
        self.assertEqual(fu["mechanism"], "activation")

    def test_perturbations_contain_required_fields(self):
        """每个扰动含 target / drug / mechanism / ko_target / description。"""
        specialist = self._P53Specialist()
        perturbations = specialist.apply_perturbation(
            {"nodes": [], "edges": []}, []
        )

        for p in perturbations:
            self.assertIn("target", p)
            self.assertIn("drug", p)
            self.assertIn("mechanism", p)
            self.assertIn("ko_target", p)
            self.assertIn("description", p)


class TestP53ValidationModule(unittest.TestCase):
    """测试 8：Validation 模块返回 3 条 benchmark。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.p53_specialist import P53Specialist

        clear_registry()
        register_specialist(P53Specialist)
        self._P53Specialist = P53Specialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_apply_validation_returns_3_rules(self):
        """apply_validation() 返回 3 条 Validation 规则。"""
        specialist = self._P53Specialist()
        rules = specialist.apply_validation()

        self.assertEqual(
            len(rules),
            3,
            f"期望 3 条 Validation 规则，实际 {len(rules)} 条",
        )

    def test_p53_pulse_period_rule(self):
        """p53 脉冲振荡周期 5-7 hours（Lev Bar-Or 2000, PMID:10644692）。"""
        specialist = self._P53Specialist()
        rules = specialist.apply_validation()

        pulse = next(
            (r for r in rules if r["metric_name"] == "p53_pulse_period"),
            None,
        )
        self.assertIsNotNone(pulse)
        self.assertEqual(pulse["expected_min"], 5.0)
        self.assertEqual(pulse["expected_max"], 7.0)
        self.assertEqual(pulse["unit"], "hours")
        self.assertEqual(pulse["pmid"], "PMID:10644692")

    def test_mdm2_transcription_delay_rule(self):
        """Mdm2 转录延迟 60-120 min（Lev Bar-Or 2000, PMID:10644692）。"""
        specialist = self._P53Specialist()
        rules = specialist.apply_validation()

        delay = next(
            (
                r
                for r in rules
                if r["metric_name"] == "Mdm2_transcription_delay"
            ),
            None,
        )
        self.assertIsNotNone(delay)
        self.assertEqual(delay["expected_min"], 60.0)
        self.assertEqual(delay["expected_max"], 120.0)
        self.assertEqual(delay["unit"], "minutes")
        self.assertEqual(delay["pmid"], "PMID:10644692")

    def test_p53_phosphorylation_response_time_rule(self):
        """p53 磷酸化响应时间 5-30 min（Lev Bar-Or 2000, PMID:10644692）。"""
        specialist = self._P53Specialist()
        rules = specialist.apply_validation()

        resp = next(
            (
                r
                for r in rules
                if r["metric_name"] == "p53_phosphorylation_response_time"
            ),
            None,
        )
        self.assertIsNotNone(resp)
        self.assertEqual(resp["expected_min"], 5.0)
        self.assertEqual(resp["expected_max"], 30.0)
        self.assertEqual(resp["unit"], "minutes")
        self.assertEqual(resp["pmid"], "PMID:10644692")

    def test_validation_rules_contain_required_fields(self):
        """每条 Validation 规则含 rule_id / metric_name / expected /
        tolerance / pmid / description。"""
        specialist = self._P53Specialist()
        rules = specialist.apply_validation()

        for r in rules:
            self.assertIn("rule_id", r)
            self.assertIn("metric_name", r)
            self.assertIn("expected", r)
            self.assertIn("tolerance", r)
            self.assertIn("pmid", r)
            self.assertIn("description", r)


class TestP53SelectTemplate(unittest.TestCase):
    """测试 9：模板选择（phosphorylation / oscillatory）。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.p53_specialist import P53Specialist

        clear_registry()
        register_specialist(P53Specialist)
        self._P53Specialist = P53Specialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_select_template_phosphorylation(self):
        """select_template('phosphorylation') 返回 _mechanism_phosphorylation_mm。"""
        specialist = self._P53Specialist()
        self.assertEqual(
            specialist.select_template("phosphorylation"),
            "_mechanism_phosphorylation_mm",
        )

    def test_select_template_oscillatory(self):
        """select_template('oscillatory') 返回 oscillatory_feedback（p53-Mdm2 DDE）。"""
        specialist = self._P53Specialist()
        self.assertEqual(
            specialist.select_template("oscillatory"),
            "oscillatory_feedback",
        )

    def test_select_template_dde_mode(self):
        """select_template('dde') 返回 oscillatory_feedback（DDE 模式检测）。"""
        specialist = self._P53Specialist()
        self.assertEqual(
            specialist.select_template("dde"),
            "oscillatory_feedback",
        )


class TestP53FeatureFlagIsolation(unittest.TestCase):
    """测试 10：Feature Flag 隔离（flag=false 时 hook 不调用 Specialist）。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.p53_specialist import P53Specialist

        clear_registry()
        register_specialist(P53Specialist)
        self._P53Specialist = P53Specialist

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
            "species": [{"name": "p53"}],
            "reactions": [],
            "state_machine": {},
        }
        specialist.apply_core.return_value = expected_core
        state = {"v4_pathway_graph": {"nodes": [{"id": "p53"}], "edges": []}}

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


class TestP53MetadataAndInputValidation(unittest.TestCase):
    """测试 11：元数据 + 输入校验。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.p53_specialist import P53Specialist

        clear_registry()
        register_specialist(P53Specialist)
        self._P53Specialist = P53Specialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_get_metadata_returns_p53_metadata(self):
        """get_metadata() 返回 pathway_class='p53' + 5 模块。"""
        specialist = self._P53Specialist()
        metadata = specialist.get_metadata()

        self.assertEqual(metadata["pathway_class"], "p53")
        self.assertEqual(metadata["display_name"], "p53 Tumor Suppressor Signaling")
        self.assertEqual(
            metadata["supported_modules"],
            ["core", "feedback", "crosstalk", "perturbation", "validation"],
        )
        self.assertEqual(len(metadata["supported_modules"]), 5)
        self.assertIn("version", metadata)

    def test_validate_input_empty_graph_returns_warnings(self):
        """validate_input({}) 返回非空 warning 列表。"""
        specialist = self._P53Specialist()
        warnings = specialist.validate_input({})

        self.assertIsInstance(warnings, list)
        self.assertGreater(len(warnings), 0, "空 pathway_graph 应返回 warning")

    def test_validate_input_none_returns_warnings(self):
        """validate_input(None) 返回非空 warning 列表。"""
        specialist = self._P53Specialist()
        warnings = specialist.validate_input(None)

        self.assertGreater(len(warnings), 0)

    def test_validate_input_valid_graph_returns_no_warnings(self):
        """含 nodes + edges 的 pathway_graph 无 warning。"""
        specialist = self._P53Specialist()
        warnings = specialist.validate_input(
            {"nodes": [{"id": "p53"}], "edges": []}
        )
        self.assertEqual(warnings, [])


if __name__ == "__main__":
    unittest.main()
