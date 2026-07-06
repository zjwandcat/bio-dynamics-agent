# BioDynamics Agent v4 - Cell Cycle Specialist 单元测试 (Phase 4 / Task 4.8)
#
# 测试用例（11 项，覆盖 5 模块 + flag 隔离 + 元数据 + CompositeReaction +
# p21 shared + 模板选择）：
#   1. Specialist 注册：get_specialist("CELL_CYCLE") 返回 CellCycleSpecialist 实例
#   2. Core 模块：apply_core() 返回 ≥11 条核心反应
#   3. CompositeReaction 输出：Core 输出含 Rb-E2F toggle CompositeReaction
#   4. p21 shared 标记：species 中 p21 标记 shared=True（与 p53 Specialist 共享）
#   5. Feedback 模块：CyclinB-APC/C delay=30min + Rb-E2F bistable
#   6. Crosstalk 模块：apply_crosstalk() 返回 5 条 cross-talk Reaction 片段
#   7. Perturbation 模块：apply_perturbation() 返回 6 个扰动
#   8. Validation 模块：apply_validation() 返回 3 条 benchmark
#   9. 模板选择：oscillatory → oscillatory_feedback；bistable → bistable_switch
#  10. Feature Flag 隔离：flag=false 时 hook 不调用 Specialist
#  11. 元数据：get_metadata() 返回 pathway_class='CELL_CYCLE'
#
# 运行：cd backend && python -m pytest tests/test_cell_cycle_specialist.py -v

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
class TestCellCycleSpecialistRegistration(unittest.TestCase):
    """测试 1：Specialist 注册到 SPECIALIST_REGISTRY。"""

    def setUp(self):
        """每个测试开始前清空 registry 并重新注册 CellCycleSpecialist。"""
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.cell_cycle_specialist import (
            CellCycleSpecialist,
        )

        clear_registry()
        register_specialist(CellCycleSpecialist)

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_get_specialist_returns_cell_cycle_instance(self):
        """get_specialist("CELL_CYCLE") 返回 CellCycleSpecialist 实例。"""
        from app.pathways.pathway_registry import (
            get_specialist,
            is_specialist_available,
        )
        from app.pathways.specialists.cell_cycle_specialist import (
            CellCycleSpecialist,
        )

        # 已注册
        self.assertTrue(is_specialist_available("CELL_CYCLE"))

        # get_specialist 返回 CellCycleSpecialist 实例
        instance = get_specialist("CELL_CYCLE")
        self.assertIsNotNone(instance)
        self.assertIsInstance(instance, CellCycleSpecialist)

        # 多次调用返回不同实例（非单例）
        instance2 = get_specialist("CELL_CYCLE")
        self.assertIsNot(instance, instance2)

    def test_pathway_class_attribute(self):
        """CellCycleSpecialist.pathway_class == 'CELL_CYCLE'。"""
        from app.pathways.specialists.cell_cycle_specialist import (
            CellCycleSpecialist,
        )

        self.assertEqual(CellCycleSpecialist.pathway_class, "CELL_CYCLE")


class TestCellCycleCoreModule(unittest.TestCase):
    """测试 2：Core 模块返回 ≥11 条核心反应。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.cell_cycle_specialist import (
            CellCycleSpecialist,
        )

        clear_registry()
        register_specialist(CellCycleSpecialist)
        self._CellCycleSpecialist = CellCycleSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_apply_core_returns_at_least_11_reactions(self):
        """apply_core() 返回 ≥11 条核心反应（Cyclin-CDK 级联 + APC/C + Rb-E2F toggle）。"""
        specialist = self._CellCycleSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})

        self.assertIn("species", result)
        self.assertIn("reactions", result)
        reactions = result["reactions"]
        self.assertGreaterEqual(
            len(reactions),
            11,
            f"期望至少 11 条核心反应，实际 {len(reactions)} 条",
        )

    def test_core_reaction_pairs(self):
        """14 条核心反应的 source→target 对应正确（Cyclin-CDK 级联 + Rb-E2F toggle）。"""
        specialist = self._CellCycleSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        pairs = [(r["source"], r["target"]) for r in result["reactions"]]
        expected_pairs = [
            # Cyclin-CDK cascade (11)
            ("Cyclin_D", "CyclinD_CDK4"),                 # 1. CyclinD-CDK4 复合物
            ("CyclinD_CDK4", "pRb_phosphorylated"),       # 2. Rb 磷酸化
            ("pRb_phosphorylated", "E2F_free"),            # 3. E2F 释放
            ("Cyclin_E", "CyclinE_CDK2"),                  # 4. CyclinE-CDK2 复合物
            ("CyclinE_CDK2", "E2F_active"),               # 5. E2F 激活
            ("Cyclin_A", "CyclinA_CDK2"),                  # 6. CyclinA-CDK2 复合物
            ("Cyclin_A", "CyclinA_CDK1"),                 # 7. CyclinA-CDK1 复合物
            ("Cyclin_B", "CyclinB_CDK1"),                 # 8. CyclinB-CDK1 复合物
            ("CyclinB_CDK1", "APC_C_Cdc20_active"),       # 9. APC/C-Cdc20 激活
            ("APC_C_Cdc20_active", "CyclinB_degraded"),   # 10. Cyclin B 降解
            ("APC_C_Cdc20_active", "Securin_degraded"),   # 11. Securin 降解
            # Rb-E2F toggle (3)
            ("E2F_free", "Cyclin_E_mRNA"),                # 12. E2F 转录 Cyclin E
            ("Cyclin_E_mRNA", "Cyclin_E"),                # 13. Cyclin E 翻译
            ("CyclinE_CDK2", "pRb_phosphorylated"),       # 14. 正反馈磷酸化 Rb
        ]
        self.assertEqual(pairs, expected_pairs)

    def test_core_reactions_contain_pathway_tag(self):
        """每条核心反应含 pathway_tag='CELL_CYCLE'。"""
        specialist = self._CellCycleSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        for r in result["reactions"]:
            self.assertEqual(r["pathway_tag"], "CELL_CYCLE")

    def test_core_reactions_contain_required_fields(self):
        """每条核心反应含 source / target / mechanism / kinetics_type /
        substrate / product / modifier / modifier_type。"""
        specialist = self._CellCycleSpecialist()
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

    def test_core_cyclin_cdk_cascade_complete(self):
        """Cyclin-CDK 级联完整（Cyclin D/E/A/B + CDK1/2/4 + APC/C-Cdc20）。"""
        specialist = self._CellCycleSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        # 同时收集 source 与 target，覆盖级联中作为产物的复合物
        nodes_in_reactions = set()
        for r in result["reactions"]:
            nodes_in_reactions.add(r["source"])
            nodes_in_reactions.add(r["target"])

        # Cyclin-CDK 级联关键节点（含 source 形式与 target 复合物产物）
        for node in ["Cyclin_D", "Cyclin_E", "Cyclin_A", "Cyclin_B",
                     "CyclinD_CDK4", "CyclinE_CDK2", "CyclinA_CDK2",
                     "CyclinA_CDK1", "CyclinB_CDK1", "APC_C_Cdc20_active"]:
            self.assertIn(
                node, nodes_in_reactions,
                f"Cyclin-CDK 级联应含 {node}",
            )

    def test_core_rb_e2f_toggle_complete(self):
        """Rb-E2F toggle 完整（CyclinD_CDK4→pRb→E2F→CyclinE→CyclinE_CDK2→pRb）。"""
        specialist = self._CellCycleSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        sources = {r["source"] for r in result["reactions"]}

        # Rb-E2F toggle 关键节点
        for node in ["CyclinD_CDK4", "pRb_phosphorylated", "E2F_free",
                     "Cyclin_E_mRNA", "CyclinE_CDK2"]:
            self.assertIn(node, sources, f"Rb-E2F toggle 应含 {node}")


class TestCellCycleCompositeReaction(unittest.TestCase):
    """测试 3：CompositeReaction 输出（Rb-E2F G1/S toggle）。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.cell_cycle_specialist import (
            CellCycleSpecialist,
        )

        clear_registry()
        register_specialist(CellCycleSpecialist)
        self._CellCycleSpecialist = CellCycleSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_core_contains_composite_reactions(self):
        """Core 输出含 composite_reactions 字段。"""
        specialist = self._CellCycleSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})

        self.assertIn("composite_reactions", result)
        cr_list = result["composite_reactions"]
        self.assertIsInstance(cr_list, list)
        self.assertGreater(len(cr_list), 0, "composite_reactions 不应为空")

    def test_composite_reaction_rb_e2f_toggle(self):
        """CompositeReaction 含 Rb-E2F G1/S toggle（bistable）。"""
        specialist = self._CellCycleSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        cr_list = result["composite_reactions"]

        # 查找 Rb-E2F toggle CompositeReaction
        toggle_cr = next(
            (cr for cr in cr_list
             if "RB_E2F" in cr.get("id", "")
             or "Rb-E2F" in cr.get("name", "")),
            None,
        )
        self.assertIsNotNone(toggle_cr, "应含 Rb-E2F G1/S toggle CompositeReaction")

        # 检查节点含 Rb-E2F toggle 关键节点
        node_ids = toggle_cr.get("node_ids", [])
        self.assertIn("CyclinD_CDK4", node_ids)
        self.assertIn("pRb_phosphorylated", node_ids)
        self.assertIn("E2F_free", node_ids)
        self.assertIn("CyclinE_CDK2", node_ids)

    def test_composite_reaction_bistable_flag(self):
        """Rb-E2F toggle CompositeReaction 标记 point_of_no_return / bistable。"""
        specialist = self._CellCycleSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        cr_list = result["composite_reactions"]

        toggle_cr = next(
            (cr for cr in cr_list
             if "RB_E2F" in cr.get("id", "")),
            None,
        )
        self.assertIsNotNone(toggle_cr)
        self.assertTrue(
            toggle_cr.get("point_of_no_return"),
            "Rb-E2F G1/S toggle 应标记 point_of_no_return=True",
        )
        self.assertEqual(toggle_cr.get("loop_type"), "positive")
        self.assertEqual(toggle_cr.get("template"), "bistable_switch.j2")
        self.assertEqual(toggle_cr.get("mechanism"), "bistable")


class TestCellCycleP21SharedMarker(unittest.TestCase):
    """测试 4：p21 shared 标记（与 p53 Specialist 共享）。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.cell_cycle_specialist import (
            CellCycleSpecialist,
        )

        clear_registry()
        register_specialist(CellCycleSpecialist)
        self._CellCycleSpecialist = CellCycleSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_p21_marked_as_shared(self):
        """species 中 p21 标记 shared=True（与 p53 Specialist 共享）。"""
        specialist = self._CellCycleSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        species = result["species"]

        p21 = next(
            (s for s in species if s["name"] == "p21"),
            None,
        )
        self.assertIsNotNone(p21, "Core 输出应含 p21 物种")
        self.assertTrue(
            p21.get("shared"),
            "p21 应标记 shared=True（与 p53 Specialist 的 p53→p21 转录路径共享）",
        )


class TestCellCycleFeedbackModule(unittest.TestCase):
    """测试 5：Feedback 模块（CyclinB-APC/C delay=30min + Rb-E2F bistable）。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.cell_cycle_specialist import (
            CellCycleSpecialist,
        )

        clear_registry()
        register_specialist(CellCycleSpecialist)
        self._CellCycleSpecialist = CellCycleSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_apply_feedback_returns_at_least_4_loops(self):
        """apply_feedback() 返回 ≥4 条反馈环。"""
        specialist = self._CellCycleSpecialist()
        loops = specialist.apply_feedback({"nodes": [], "edges": []})

        self.assertGreaterEqual(
            len(loops),
            4,
            f"期望至少 4 条反馈环，实际 {len(loops)} 条",
        )

    def test_cyclinb_apc_delayed_negative_feedback(self):
        """CyclinB-APC/C 延迟负反馈振荡器（delay=30min, 振荡）。"""
        specialist = self._CellCycleSpecialist()
        loops = specialist.apply_feedback({"nodes": [], "edges": []})

        cyclinb_loop = next(
            (l for l in loops if "CYCLINB_APC" in l["id"]),
            None,
        )
        self.assertIsNotNone(cyclinb_loop, "应含 CyclinB-APC/C 反馈环")
        self.assertEqual(cyclinb_loop["loop_type"], "negative")
        self.assertEqual(
            cyclinb_loop["delay_minutes"],
            30.0,
            "CyclinB-APC/C 延迟应为 30min",
        )
        # 节点含 CyclinB-CDK1 / APC/C-Cdc20
        node_ids = cyclinb_loop["node_ids"]
        self.assertIn("CyclinB_CDK1", node_ids)
        self.assertIn("APC_C_Cdc20_active", node_ids)

    def test_rb_e2f_bistable_toggle(self):
        """Rb-E2F toggle bistable（正反馈, point-of-no-return）。"""
        specialist = self._CellCycleSpecialist()
        loops = specialist.apply_feedback({"nodes": [], "edges": []})

        rb_e2f_loop = next(
            (l for l in loops if "RB_E2F_TOGGLE" in l["id"]),
            None,
        )
        self.assertIsNotNone(rb_e2f_loop, "应含 Rb-E2F bistable toggle 反馈环")
        self.assertEqual(rb_e2f_loop["loop_type"], "positive")
        self.assertTrue(
            rb_e2f_loop.get("bistable"),
            "Rb-E2F toggle 应标记 bistable=True",
        )
        self.assertTrue(
            rb_e2f_loop.get("point_of_no_return"),
            "Rb-E2F toggle 应标记 point_of_no_return=True",
        )
        # 节点含 Rb-E2F toggle 关键节点
        node_ids = rb_e2f_loop["node_ids"]
        self.assertIn("CyclinD_CDK4", node_ids)
        self.assertIn("pRb_phosphorylated", node_ids)
        self.assertIn("E2F_free", node_ids)
        self.assertIn("CyclinE_CDK2", node_ids)

    def test_p53_p21_cdk_inhibition(self):
        """p53→p21 抑制 CDK2/4（负反馈, 来自 p53 Specialist cross-talk）。"""
        specialist = self._CellCycleSpecialist()
        loops = specialist.apply_feedback({"nodes": [], "edges": []})

        p53_loop = next(
            (l for l in loops if "P53_P21" in l["id"]),
            None,
        )
        self.assertIsNotNone(p53_loop, "应含 p53→p21 抑制 CDK 反馈环")
        self.assertEqual(p53_loop["loop_type"], "negative")
        # 节点含 p21 + CDK 复合物
        node_ids = p53_loop["node_ids"]
        self.assertIn("p21", node_ids)
        self.assertIn("CyclinE_CDK2", node_ids)

    def test_apc_cdh1_g1_maintenance(self):
        """APC/C-Cdh1 G1 早期维持（降解 Cyclin A/B）。"""
        specialist = self._CellCycleSpecialist()
        loops = specialist.apply_feedback({"nodes": [], "edges": []})

        cdh1_loop = next(
            (l for l in loops if "APC_CDH1" in l["id"]),
            None,
        )
        self.assertIsNotNone(cdh1_loop, "应含 APC/C-Cdh1 G1 维持反馈环")
        self.assertEqual(cdh1_loop["loop_type"], "negative")

    def test_feedback_loops_contain_required_fields(self):
        """每条反馈环含 id / loop_type / node_ids / delay_minutes / description。"""
        specialist = self._CellCycleSpecialist()
        loops = specialist.apply_feedback({"nodes": [], "edges": []})

        for loop in loops:
            self.assertIn("id", loop)
            self.assertIn("loop_type", loop)
            self.assertIn("node_ids", loop)
            self.assertIn("delay_minutes", loop)
            self.assertIn("description", loop)


class TestCellCycleCrosstalkModule(unittest.TestCase):
    """测试 6：Crosstalk 模块返回 5 条 cross-talk Reaction 片段。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.cell_cycle_specialist import (
            CellCycleSpecialist,
        )

        clear_registry()
        register_specialist(CellCycleSpecialist)
        self._CellCycleSpecialist = CellCycleSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_apply_crosstalk_returns_5_fragments(self):
        """apply_crosstalk() 返回 5 条 cross-talk Reaction 片段。"""
        specialist = self._CellCycleSpecialist()
        fragments = specialist.apply_crosstalk(
            {"nodes": [], "edges": []}, []
        )

        self.assertEqual(
            len(fragments),
            5,
            f"期望 5 条 cross-talk 片段，实际 {len(fragments)} 条",
        )

    def test_crosstalk_contains_perk_cyclin_d_transcription(self):
        """含 pERK → Cyclin_D transcription 片段（MAPK→Cell Cycle）。"""
        specialist = self._CellCycleSpecialist()
        fragments = specialist.apply_crosstalk(
            {"nodes": [], "edges": []}, []
        )

        perk = next(
            (f for f in fragments
             if f["source"] == "pERK" and f["target"] == "Cyclin_D"),
            None,
        )
        self.assertIsNotNone(perk)
        self.assertEqual(perk["mechanism"], "transcription")

    def test_crosstalk_contains_p53_p21_transcription(self):
        """含 p53 → p21 transcription 片段（来自 p53 Specialist）。"""
        specialist = self._CellCycleSpecialist()
        fragments = specialist.apply_crosstalk(
            {"nodes": [], "edges": []}, []
        )

        p21 = next(
            (f for f in fragments
             if f["source"] == "p53" and f["target"] == "p21"),
            None,
        )
        self.assertIsNotNone(p21)
        self.assertEqual(p21["mechanism"], "transcription")
        self.assertIn("p53", p21["shared_species"])
        self.assertIn("p21", p21["shared_species"])

    def test_crosstalk_contains_pakt_p21_inhibition(self):
        """含 pAKT → p21 inhibition 片段。"""
        specialist = self._CellCycleSpecialist()
        fragments = specialist.apply_crosstalk(
            {"nodes": [], "edges": []}, []
        )

        akt_p21 = next(
            (f for f in fragments
             if f["source"] == "pAKT" and f["target"] == "p21"),
            None,
        )
        self.assertIsNotNone(akt_p21)
        self.assertEqual(akt_p21["mechanism"], "inhibition")

    def test_crosstalk_contains_myc_cyclin_d_transcription(self):
        """含 Myc → Cyclin_D transcription 片段。"""
        specialist = self._CellCycleSpecialist()
        fragments = specialist.apply_crosstalk(
            {"nodes": [], "edges": []}, []
        )

        myc = next(
            (f for f in fragments
             if f["source"] == "Myc" and f["target"] == "Cyclin_D"),
            None,
        )
        self.assertIsNotNone(myc)
        self.assertEqual(myc["mechanism"], "transcription")

    def test_crosstalk_contains_pakt_gsk3b_cyclin_d(self):
        """含 pAKT → GSK3β → Cyclin_D inhibition 片段（pAKT 抑制 GSK3β）。"""
        specialist = self._CellCycleSpecialist()
        fragments = specialist.apply_crosstalk(
            {"nodes": [], "edges": []}, []
        )

        # 找到 pAKT → Cyclin_D 的 inhibition 片段（含 GSK3β intermediate）
        gsk3b = next(
            (f for f in fragments
             if f["source"] == "pAKT"
             and f["target"] == "Cyclin_D"
             and f["mechanism"] == "inhibition"
             and f.get("intermediate") == "GSK3β"),
            None,
        )
        self.assertIsNotNone(
            gsk3b,
            "应含 pAKT→GSK3β→Cyclin_D 片段（pAKT 抑制 GSK3β 阻止 Cyclin D 降解）",
        )

    def test_crosstalk_fragments_contain_required_fields(self):
        """每条 cross-talk 片段含 source / target / mechanism / shared_species /
        description。"""
        specialist = self._CellCycleSpecialist()
        fragments = specialist.apply_crosstalk(
            {"nodes": [], "edges": []}, []
        )

        for f in fragments:
            self.assertIn("source", f)
            self.assertIn("target", f)
            self.assertIn("mechanism", f)
            self.assertIn("shared_species", f)
            self.assertIn("description", f)


class TestCellCyclePerturbationModule(unittest.TestCase):
    """测试 7：Perturbation 模块返回 6 个扰动。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.cell_cycle_specialist import (
            CellCycleSpecialist,
        )

        clear_registry()
        register_specialist(CellCycleSpecialist)
        self._CellCycleSpecialist = CellCycleSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_apply_perturbation_returns_6(self):
        """apply_perturbation() 返回 6 个扰动。"""
        specialist = self._CellCycleSpecialist()
        perturbations = specialist.apply_perturbation(
            {"nodes": [], "edges": []}, []
        )

        self.assertEqual(
            len(perturbations),
            6,
            f"期望 6 个扰动，实际 {len(perturbations)} 个",
        )

    def test_perturbation_contains_palbociclib(self):
        """含 Palbociclib（CDK4/6 抑制剂, FDA-approved）。"""
        specialist = self._CellCycleSpecialist()
        perturbations = specialist.apply_perturbation(
            {"nodes": [], "edges": []}, []
        )

        palbo = next(
            (p for p in perturbations if p.get("drug") == "Palbociclib"),
            None,
        )
        self.assertIsNotNone(palbo)
        self.assertEqual(palbo["target"], "CDK4")
        self.assertEqual(palbo["mechanism"], "inhibition")

    def test_perturbation_contains_ribociclib(self):
        """含 Ribociclib（CDK4/6 抑制剂, FDA-approved）。"""
        specialist = self._CellCycleSpecialist()
        perturbations = specialist.apply_perturbation(
            {"nodes": [], "edges": []}, []
        )

        ribo = next(
            (p for p in perturbations if p.get("drug") == "Ribociclib"),
            None,
        )
        self.assertIsNotNone(ribo)
        self.assertEqual(ribo["target"], "CDK4")
        self.assertEqual(ribo["mechanism"], "inhibition")

    def test_perturbation_contains_abemaciclib(self):
        """含 Abemaciclib（CDK4/6 抑制剂, FDA-approved）。"""
        specialist = self._CellCycleSpecialist()
        perturbations = specialist.apply_perturbation(
            {"nodes": [], "edges": []}, []
        )

        abe = next(
            (p for p in perturbations if p.get("drug") == "Abemaciclib"),
            None,
        )
        self.assertIsNotNone(abe)
        self.assertEqual(abe["target"], "CDK4")

    def test_perturbation_contains_roscovitine(self):
        """含 Roscovitine / Seliciclib（CDK2 抑制剂）。"""
        specialist = self._CellCycleSpecialist()
        perturbations = specialist.apply_perturbation(
            {"nodes": [], "edges": []}, []
        )

        ros = next(
            (p for p in perturbations if p.get("drug") == "Roscovitine"),
            None,
        )
        self.assertIsNotNone(ros)
        self.assertEqual(ros["target"], "CDK2")
        self.assertEqual(ros["mechanism"], "inhibition")

    def test_perturbation_contains_flavopiridol(self):
        """含 Flavopiridol / Alvocidib（pan-CDK 抑制剂）。"""
        specialist = self._CellCycleSpecialist()
        perturbations = specialist.apply_perturbation(
            {"nodes": [], "edges": []}, []
        )

        fla = next(
            (p for p in perturbations if p.get("drug") == "Flavopiridol"),
            None,
        )
        self.assertIsNotNone(fla)
        self.assertEqual(fla["mechanism"], "inhibition")

    def test_perturbation_contains_cdkn2a_loss(self):
        """含 CDKN2A / p16 loss（loss-of-function, CDK4/6 失去抑制）。"""
        specialist = self._CellCycleSpecialist()
        perturbations = specialist.apply_perturbation(
            {"nodes": [], "edges": []}, []
        )

        cdkn2a = next(
            (p for p in perturbations
             if p.get("ko_target") == "CDKN2A_p16_loss"),
            None,
        )
        self.assertIsNotNone(cdkn2a)
        self.assertEqual(cdkn2a["target"], "CDK4")
        self.assertEqual(cdkn2a["mechanism"], "knockout")

    def test_perturbations_contain_required_fields(self):
        """每个扰动含 target / drug / mechanism / ko_target / description。"""
        specialist = self._CellCycleSpecialist()
        perturbations = specialist.apply_perturbation(
            {"nodes": [], "edges": []}, []
        )

        for p in perturbations:
            self.assertIn("target", p)
            self.assertIn("drug", p)
            self.assertIn("mechanism", p)
            self.assertIn("ko_target", p)
            self.assertIn("description", p)


class TestCellCycleValidationModule(unittest.TestCase):
    """测试 8：Validation 模块返回 3 条 benchmark。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.cell_cycle_specialist import (
            CellCycleSpecialist,
        )

        clear_registry()
        register_specialist(CellCycleSpecialist)
        self._CellCycleSpecialist = CellCycleSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_apply_validation_returns_3_rules(self):
        """apply_validation() 返回 3 条 Validation 规则。"""
        specialist = self._CellCycleSpecialist()
        rules = specialist.apply_validation()

        self.assertEqual(
            len(rules),
            3,
            f"期望 3 条 Validation 规则，实际 {len(rules)} 条",
        )

    def test_cyclinb_apc_oscillation_rule(self):
        """CyclinB-APC/C 振荡周期 8-12 hours（Pomerening 2005, PMID:11389814）。"""
        specialist = self._CellCycleSpecialist()
        rules = specialist.apply_validation()

        osc = next(
            (r for r in rules
             if r["metric_name"] == "CyclinB_APC_oscillation_period"),
            None,
        )
        self.assertIsNotNone(osc)
        self.assertEqual(osc["expected_min"], 8.0)
        self.assertEqual(osc["expected_max"], 12.0)
        self.assertEqual(osc["unit"], "hours")
        self.assertEqual(osc["pmid"], "PMID:11389814")

    def test_rb_e2f_bistable_switch_rule(self):
        """Rb-E2F bistable G1/S switch（Yao 2008, PMID:12064617）。"""
        specialist = self._CellCycleSpecialist()
        rules = specialist.apply_validation()

        bistable = next(
            (r for r in rules
             if r["metric_name"] == "Rb_E2F_bistable_switch"),
            None,
        )
        self.assertIsNotNone(bistable)
        self.assertEqual(bistable["expected"], True)
        self.assertEqual(bistable["pmid"], "PMID:12064617")

    def test_cyclin_d1_peak_time_rule(self):
        """Cyclin D1 达峰时间 60-240 minutes（G1 期响应）。"""
        specialist = self._CellCycleSpecialist()
        rules = specialist.apply_validation()

        peak = next(
            (r for r in rules
             if r["metric_name"] == "Cyclin_D1_peak_time"),
            None,
        )
        self.assertIsNotNone(peak)
        self.assertEqual(peak["expected_min"], 60.0)
        self.assertEqual(peak["expected_max"], 240.0)
        self.assertEqual(peak["unit"], "minutes")

    def test_validation_rules_contain_required_fields(self):
        """每条 Validation 规则含 rule_id / metric_name / expected /
        tolerance / pmid / description。"""
        specialist = self._CellCycleSpecialist()
        rules = specialist.apply_validation()

        for r in rules:
            self.assertIn("rule_id", r)
            self.assertIn("metric_name", r)
            self.assertIn("expected", r)
            self.assertIn("tolerance", r)
            self.assertIn("pmid", r)
            self.assertIn("description", r)


class TestCellCycleSelectTemplate(unittest.TestCase):
    """测试 9：模板选择（oscillatory / bistable）。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.cell_cycle_specialist import (
            CellCycleSpecialist,
        )

        clear_registry()
        register_specialist(CellCycleSpecialist)
        self._CellCycleSpecialist = CellCycleSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_select_template_oscillatory(self):
        """select_template('oscillatory') 返回 oscillatory_feedback（CyclinB-APC/C 振荡）。"""
        specialist = self._CellCycleSpecialist()
        self.assertEqual(
            specialist.select_template("oscillatory"),
            "oscillatory_feedback",
        )

    def test_select_template_bistable(self):
        """select_template('bistable') 返回 bistable_switch（Rb-E2F toggle）。"""
        specialist = self._CellCycleSpecialist()
        self.assertEqual(
            specialist.select_template("bistable"),
            "bistable_switch",
        )

    def test_select_template_phosphorylation(self):
        """select_template('phosphorylation') 返回 _mechanism_phosphorylation_mm。"""
        specialist = self._CellCycleSpecialist()
        self.assertEqual(
            specialist.select_template("phosphorylation"),
            "_mechanism_phosphorylation_mm",
        )

    def test_select_template_transcription_oscillatory(self):
        """select_template('transcription') 返回 oscillatory_feedback（E2F 转录振荡）。"""
        specialist = self._CellCycleSpecialist()
        self.assertEqual(
            specialist.select_template("transcription"),
            "oscillatory_feedback",
        )


class TestCellCycleFeatureFlagIsolation(unittest.TestCase):
    """测试 10：Feature Flag 隔离（flag=false 时 hook 不调用 Specialist）。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.cell_cycle_specialist import (
            CellCycleSpecialist,
        )

        clear_registry()
        register_specialist(CellCycleSpecialist)
        self._CellCycleSpecialist = CellCycleSpecialist

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
            "species": [{"name": "Cyclin_D"}],
            "reactions": [],
            "composite_reactions": [],
        }
        specialist.apply_core.return_value = expected_core
        state = {"v4_pathway_graph": {"nodes": [{"id": "Cyclin_D"}], "edges": []}}

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


class TestCellCycleMetadataAndInputValidation(unittest.TestCase):
    """测试 11：元数据 + 输入校验。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.cell_cycle_specialist import (
            CellCycleSpecialist,
        )

        clear_registry()
        register_specialist(CellCycleSpecialist)
        self._CellCycleSpecialist = CellCycleSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_get_metadata_returns_cell_cycle_metadata(self):
        """get_metadata() 返回 pathway_class='CELL_CYCLE' + 5 模块。"""
        specialist = self._CellCycleSpecialist()
        metadata = specialist.get_metadata()

        self.assertEqual(metadata["pathway_class"], "CELL_CYCLE")
        self.assertEqual(
            metadata["display_name"],
            "Cell Cycle Regulation",
        )
        self.assertEqual(
            metadata["supported_modules"],
            ["core", "feedback", "crosstalk", "perturbation", "validation"],
        )
        self.assertEqual(len(metadata["supported_modules"]), 5)
        self.assertIn("version", metadata)

    def test_validate_input_empty_graph_returns_warnings(self):
        """validate_input({}) 返回非空 warning 列表。"""
        specialist = self._CellCycleSpecialist()
        warnings = specialist.validate_input({})

        self.assertIsInstance(warnings, list)
        self.assertGreater(len(warnings), 0, "空 pathway_graph 应返回 warning")

    def test_validate_input_none_returns_warnings(self):
        """validate_input(None) 返回非空 warning 列表。"""
        specialist = self._CellCycleSpecialist()
        warnings = specialist.validate_input(None)

        self.assertGreater(len(warnings), 0)

    def test_validate_input_valid_graph_returns_no_warnings(self):
        """含 nodes + edges 的 pathway_graph 无 warning。"""
        specialist = self._CellCycleSpecialist()
        warnings = specialist.validate_input(
            {"nodes": [{"id": "Cyclin_D"}], "edges": []}
        )
        self.assertEqual(warnings, [])


if __name__ == "__main__":
    unittest.main()
