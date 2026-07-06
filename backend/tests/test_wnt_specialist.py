# BioDynamics Agent v4 - Wnt Specialist 单元测试 (Phase 4 / Task 4.11)
#
# 测试用例（11 项，覆盖 5 模块 + flag 隔离 + 元数据 + Destruction Complex 五步耦合
# CompositeReaction + bCatenin shared + 模板选择）：
#   1. Specialist 注册：get_specialist("WNT") 返回 WntSpecialist 实例
#   2. Core 模块：apply_core() 返回 ≥15 条核心反应（Off 6 + On 9+）
#   3. Destruction Complex 五步 CompositeReaction：Core 输出含
#      CR_DESTRUCTION_COMPLEX（binding→binding→phosphorylation→
#      ubiquitination→proteasomal_degradation）
#   4. bCatenin shared 标记：species 中 bCatenin 标记 shared=True
#      （与 Cell Cycle Specialist Cyclin D1 路径共享）
#   5. Feedback 模块：FL_BCAT_AXIN2 delay=30min + FL_AXIN2_DC_REFORMED
#   6. Crosstalk 模块：apply_crosstalk() 返回 5 条 cross-talk Reaction 片段
#   7. Perturbation 模块：apply_perturbation() 返回 6 个扰动
#   8. Validation 模块：apply_validation() 返回 3 条 benchmark
#   9. 模板选择：bistable → bistable_switch；
#      phosphorylation → _mechanism_phosphorylation_mm
#  10. Feature Flag 隔离：flag=false 时 hook 不调用 Specialist
#  11. 元数据：get_metadata() 返回 pathway_class='WNT'
#
# 运行：cd backend && python -m pytest tests/test_wnt_specialist.py -v

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
class TestWntSpecialistRegistration(unittest.TestCase):
    """测试 1：Specialist 注册到 SPECIALIST_REGISTRY。"""

    def setUp(self):
        """每个测试开始前清空 registry 并重新注册 WntSpecialist。"""
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.wnt_specialist import WntSpecialist

        clear_registry()
        register_specialist(WntSpecialist)

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_get_specialist_returns_wnt_instance(self):
        """get_specialist("WNT") 返回 WntSpecialist 实例。"""
        from app.pathways.pathway_registry import (
            get_specialist,
            is_specialist_available,
        )
        from app.pathways.specialists.wnt_specialist import WntSpecialist

        # 已注册
        self.assertTrue(is_specialist_available("WNT"))

        # get_specialist 返回 WntSpecialist 实例
        instance = get_specialist("WNT")
        self.assertIsNotNone(instance)
        self.assertIsInstance(instance, WntSpecialist)

        # 多次调用返回不同实例（非单例）
        instance2 = get_specialist("WNT")
        self.assertIsNot(instance, instance2)

    def test_pathway_class_attribute(self):
        """WntSpecialist.pathway_class == 'WNT'。"""
        from app.pathways.specialists.wnt_specialist import WntSpecialist

        self.assertEqual(WntSpecialist.pathway_class, "WNT")


class TestWntCoreModule(unittest.TestCase):
    """测试 2：Core 模块返回 ≥15 条核心反应（Off 6 + On 9+）。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.wnt_specialist import WntSpecialist

        clear_registry()
        register_specialist(WntSpecialist)
        self._WntSpecialist = WntSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_apply_core_returns_at_least_15_reactions(self):
        """apply_core() 返回 ≥15 条核心反应（Off 6 + On 9+）。"""
        specialist = self._WntSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})

        self.assertIn("species", result)
        self.assertIn("reactions", result)
        reactions = result["reactions"]
        self.assertGreaterEqual(
            len(reactions),
            15,
            f"期望至少 15 条核心反应（Off 6 + On 9+），实际 {len(reactions)} 条",
        )

    def test_core_reaction_pairs(self):
        """17 条核心反应的 source→target 对应正确
        （Off: Axin→Axin_APC→Axin_APC_GSK3b→Axin_APC_GSK3b_bcat→p_bcat→
        ub_bcat→bcat_degraded; On: Wnt→Wnt_Fz_LRP_complex→pDvl→pLRP6→
        Axin_recruited→destruction_complex_disrupted; bCatenin→bCatenin_nuclear→
        TCF_LEF_bcat_complex→Axin2_mRNA/Cyclin_D1_mRNA/cMyc_mRNA）。"""
        specialist = self._WntSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        pairs = [(r["source"], r["target"]) for r in result["reactions"]]
        expected_pairs = [
            # Off 状态（destruction complex 5 步降解 + 招募 β-catenin）
            ("Axin", "Axin_APC"),                             # 1. Axin+APC → Axin_APC (binding step 1)
            ("Axin_APC", "Axin_APC_GSK3b"),                   # 2. Axin_APC+GSK3β → Axin_APC_GSK3b (binding step 2)
            ("Axin_APC_GSK3b", "Axin_APC_GSK3b_bcat"),        # 3. 招募 β-catenin (complex_formation, 非 5 步成员)
            ("Axin_APC_GSK3b_bcat", "p_bcat"),                # 4. 磷酸化 β-catenin (step 3)
            ("p_bcat", "ub_bcat"),                             # 5. 泛素化 p_bcat (step 4)
            ("ub_bcat", "bcat_degraded"),                      # 6. 蛋白酶体降解 ub_bcat (step 5)
            # On 状态（Wnt 信号 β-catenin 累积）
            ("Wnt", "Wnt_Fz_LRP_complex"),                    # 7. Wnt+Frizzled+LRP5/6 复合物
            ("Wnt_Fz_LRP_complex", "pDvl"),                   # 8. Dvl 磷酸化激活
            ("pDvl", "pLRP6"),                                # 9. LRP6 磷酸化
            ("pLRP6", "Axin_recruited"),                      # 10. Axin 招募到膜
            ("Axin_recruited", "destruction_complex_disrupted"),  # 11. destruction complex 解离
            ("bCatenin", "bCatenin_nuclear"),                # 12. β-catenin 入核
            ("bCatenin_nuclear", "TCF_LEF_bcat_complex"),     # 13. 转录激活复合物
            ("TCF_LEF_bcat_complex", "Axin2_mRNA"),           # 14. Axin2 转录（负反馈）
            ("Axin2_mRNA", "Axin2"),                          # 15. Axin2 翻译
            ("TCF_LEF_bcat_complex", "Cyclin_D1_mRNA"),       # 16. Cyclin D1 转录（cross-talk）
            ("TCF_LEF_bcat_complex", "cMyc_mRNA"),            # 17. cMyc 转录（cross-talk）
        ]
        self.assertEqual(pairs, expected_pairs)

    def test_core_reactions_contain_pathway_tag(self):
        """每条核心反应含 pathway_tag='WNT'。"""
        specialist = self._WntSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        for r in result["reactions"]:
            self.assertEqual(r["pathway_tag"], "WNT")

    def test_core_reactions_contain_required_fields(self):
        """每条核心反应含 source / target / mechanism / kinetics_type /
        substrate / product / modifier / modifier_type。"""
        specialist = self._WntSpecialist()
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

    def test_core_wnt_cascade_complete(self):
        """Wnt 级联完整（Off: Axin→Axin_APC→Axin_APC_GSK3b→p_bcat→ub_bcat→
        bcat_degraded; On: Wnt→Wnt_Fz_LRP_complex→pDvl→pLRP6→Axin_recruited→
        destruction_complex_disrupted→bCatenin_nuclear→TCF_LEF_bcat_complex→转录）。"""
        specialist = self._WntSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        # 同时收集 source 与 target
        nodes_in_reactions = set()
        for r in result["reactions"]:
            nodes_in_reactions.add(r["source"])
            nodes_in_reactions.add(r["target"])

        # Wnt 级联关键节点
        for node in ["Axin", "Axin_APC", "Axin_APC_GSK3b", "Axin_APC_GSK3b_bcat",
                     "p_bcat", "ub_bcat", "bcat_degraded",
                     "Wnt", "Wnt_Fz_LRP_complex", "pDvl", "pLRP6",
                     "Axin_recruited", "destruction_complex_disrupted",
                     "bCatenin_nuclear", "TCF_LEF_bcat_complex",
                     "Axin2_mRNA", "Axin2"]:
            self.assertIn(
                node, nodes_in_reactions,
                f"Wnt 级联应含 {node}",
            )

    def test_core_destruction_complex_five_step_markers(self):
        """Destruction Complex 五步耦合反应含 composite_step 标记（1/2/3/4/5）。"""
        specialist = self._WntSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})

        # 提取含 composite_step 标记的反应
        composite_reactions = [
            r for r in result["reactions"] if "composite_step" in r
        ]
        # 至少 5 步耦合
        self.assertGreaterEqual(
            len(composite_reactions), 5,
            "Destruction Complex 五步耦合应含至少 5 步标记",
        )
        # 检查 1/2/3/4/5 步均存在
        steps = {r["composite_step"] for r in composite_reactions}
        self.assertIn(1, steps, "五步耦合第 1 步 binding 应存在")
        self.assertIn(2, steps, "五步耦合第 2 步 binding 应存在")
        self.assertIn(3, steps, "五步耦合第 3 步 phosphorylation 应存在")
        self.assertIn(4, steps, "五步耦合第 4 步 ubiquitination 应存在")
        self.assertIn(5, steps, "五步耦合第 5 步 proteasomal_degradation 应存在")
        # 检查机制对应正确
        step1 = next(r for r in composite_reactions if r["composite_step"] == 1)
        self.assertEqual(step1["mechanism"], "complex_formation")
        step2 = next(r for r in composite_reactions if r["composite_step"] == 2)
        self.assertEqual(step2["mechanism"], "complex_formation")
        step3 = next(r for r in composite_reactions if r["composite_step"] == 3)
        self.assertEqual(step3["mechanism"], "phosphorylation")
        step4 = next(r for r in composite_reactions if r["composite_step"] == 4)
        self.assertEqual(step4["mechanism"], "ubiquitination")
        step5 = next(r for r in composite_reactions if r["composite_step"] == 5)
        self.assertEqual(step5["mechanism"], "proteasomal_degradation")
        # 所有五步共享 composite_id
        composite_ids = {r["composite_id"] for r in composite_reactions}
        self.assertEqual(
            len(composite_ids), 1,
            "五步耦合应共享同一 composite_id",
        )
        self.assertEqual(
            composite_ids.pop(), "CR_DESTRUCTION_COMPLEX",
        )


class TestWntCompositeReaction(unittest.TestCase):
    """测试 3：Destruction Complex 五步耦合 CompositeReaction 输出。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.wnt_specialist import WntSpecialist

        clear_registry()
        register_specialist(WntSpecialist)
        self._WntSpecialist = WntSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_core_contains_composite_reactions(self):
        """Core 输出含 composite_reactions 字段。"""
        specialist = self._WntSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})

        self.assertIn("composite_reactions", result)
        cr_list = result["composite_reactions"]
        self.assertIsInstance(cr_list, list)
        self.assertGreater(len(cr_list), 0, "composite_reactions 不应为空")

    def test_composite_reaction_destruction_complex(self):
        """CompositeReaction 含 Destruction Complex 五步耦合降解
        （CR_DESTRUCTION_COMPLEX, binding→binding→phosphorylation→
        ubiquitination→proteasomal_degradation）。"""
        specialist = self._WntSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        cr_list = result["composite_reactions"]

        # 查找 Destruction Complex 五步耦合 CompositeReaction
        dc_cr = next(
            (cr for cr in cr_list
             if "DESTRUCTION_COMPLEX" in cr.get("id", "")
             or "Destruction" in cr.get("name", "")
             or "destruction" in cr.get("name", "").lower()),
            None,
        )
        self.assertIsNotNone(
            dc_cr,
            "应含 Destruction Complex 五步耦合 CompositeReaction",
        )

        # 检查字段
        self.assertEqual(dc_cr["id"], "CR_DESTRUCTION_COMPLEX")
        self.assertEqual(dc_cr["mechanism"], "sequential")
        self.assertEqual(dc_cr["coupling_type"], "sequential")

    def test_composite_reaction_intermediate_species(self):
        """CompositeReaction 中间产物
        Axin_APC / Axin_APC_GSK3b / Axin_APC_GSK3b_bcat / p_bcat /
        ub_bcat / bcat_degraded 完整。"""
        specialist = self._WntSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        cr_list = result["composite_reactions"]

        dc_cr = next(
            (cr for cr in cr_list
             if "DESTRUCTION_COMPLEX" in cr.get("id", "")),
            None,
        )
        self.assertIsNotNone(dc_cr)

        # 检查中间产物
        intermediate_species = dc_cr.get("intermediate_species", [])
        self.assertIn("Axin_APC", intermediate_species)
        self.assertIn("Axin_APC_GSK3b", intermediate_species)
        self.assertIn("Axin_APC_GSK3b_bcat", intermediate_species)
        self.assertIn("p_bcat", intermediate_species)
        self.assertIn("ub_bcat", intermediate_species)
        self.assertIn("bcat_degraded", intermediate_species)

        # 检查子反应链（5 步顺序耦合）
        sub_reactions = dc_cr.get("sub_reactions", [])
        self.assertGreaterEqual(len(sub_reactions), 5)
        self.assertIn("Axin → Axin_APC", sub_reactions)
        self.assertIn("Axin_APC → Axin_APC_GSK3b", sub_reactions)
        self.assertIn("Axin_APC_GSK3b_bcat → p_bcat", sub_reactions)
        self.assertIn("p_bcat → ub_bcat", sub_reactions)
        self.assertIn("ub_bcat → bcat_degraded", sub_reactions)

    def test_composite_reaction_node_ids(self):
        """CompositeReaction node_ids 含 Axin / APC / GSK3b / Axin_APC /
        Axin_APC_GSK3b / Axin_APC_GSK3b_bcat / p_bcat / ub_bcat / bcat_degraded。"""
        specialist = self._WntSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        cr_list = result["composite_reactions"]

        dc_cr = next(
            (cr for cr in cr_list
             if "DESTRUCTION_COMPLEX" in cr.get("id", "")),
            None,
        )
        self.assertIsNotNone(dc_cr)

        node_ids = dc_cr.get("node_ids", [])
        self.assertIn("Axin", node_ids)
        self.assertIn("APC", node_ids)
        self.assertIn("GSK3b", node_ids)
        self.assertIn("Axin_APC", node_ids)
        self.assertIn("Axin_APC_GSK3b", node_ids)
        self.assertIn("Axin_APC_GSK3b_bcat", node_ids)
        self.assertIn("p_bcat", node_ids)
        self.assertIn("ub_bcat", node_ids)
        self.assertIn("bcat_degraded", node_ids)

    def test_composite_reaction_template_destruction_complex(self):
        """CompositeReaction 标记 destruction_complex.j2 模板与 negative loop。"""
        specialist = self._WntSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        cr_list = result["composite_reactions"]

        dc_cr = next(
            (cr for cr in cr_list
             if "DESTRUCTION_COMPLEX" in cr.get("id", "")),
            None,
        )
        self.assertIsNotNone(dc_cr)
        self.assertEqual(dc_cr.get("loop_type"), "negative")
        self.assertEqual(
            dc_cr.get("template"), "destruction_complex.j2",
        )


class TestWntSharedMarker(unittest.TestCase):
    """测试 4：bCatenin shared 标记
    （与 Cell Cycle Specialist Cyclin D1 路径共享）。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.wnt_specialist import WntSpecialist

        clear_registry()
        register_specialist(WntSpecialist)
        self._WntSpecialist = WntSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_bcatenin_marked_as_shared(self):
        """species 中 bCatenin 标记 shared=True
        （与 Cell Cycle Specialist 的 Cyclin D1 路径共享）。"""
        specialist = self._WntSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        species = result["species"]

        bcat = next(
            (s for s in species if s["name"] == "bCatenin"),
            None,
        )
        self.assertIsNotNone(bcat, "Core 输出应含 bCatenin 物种")
        self.assertTrue(
            bcat.get("shared"),
            "bCatenin 应标记 shared=True（与 Cell Cycle Specialist 的 Cyclin D1 路径共享）",
        )

    def test_gsk3b_marked_as_shared(self):
        """species 中 GSK3b 标记 shared=True
        （与 PI3K-AKT Specialist 的 pAKT→GSK3β 路径共享）。"""
        specialist = self._WntSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        species = result["species"]

        gsk = next(
            (s for s in species if s["name"] == "GSK3b"),
            None,
        )
        self.assertIsNotNone(gsk, "Core 输出应含 GSK3b 物种")
        self.assertTrue(
            gsk.get("shared"),
            "GSK3b 应标记 shared=True（与 PI3K-AKT Specialist 的 pAKT→GSK3β 路径共享）",
        )


class TestWntFeedbackModule(unittest.TestCase):
    """测试 5：Feedback 模块
    （FL_BCAT_AXIN2 delay=30min + FL_AXIN2_DC_REFORMED + FL_PLRP6_DC_DISRUPTED）。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.wnt_specialist import WntSpecialist

        clear_registry()
        register_specialist(WntSpecialist)
        self._WntSpecialist = WntSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_apply_feedback_returns_at_least_3_loops(self):
        """apply_feedback() 返回 ≥3 条反馈环。"""
        specialist = self._WntSpecialist()
        loops = specialist.apply_feedback({"nodes": [], "edges": []})

        self.assertGreaterEqual(
            len(loops),
            3,
            f"期望至少 3 条反馈环，实际 {len(loops)} 条",
        )

    def test_fl_bcat_axin2_delayed_negative_feedback(self):
        """FL_BCAT_AXIN2 转录延迟负反馈（delay=30min）。"""
        specialist = self._WntSpecialist()
        loops = specialist.apply_feedback({"nodes": [], "edges": []})

        axin2_loop = next(
            (l for l in loops if "BCAT_AXIN2" in l["id"]),
            None,
        )
        self.assertIsNotNone(axin2_loop, "应含 FL_BCAT_AXIN2 反馈环")
        self.assertEqual(axin2_loop["loop_type"], "negative")
        self.assertEqual(
            axin2_loop["delay_minutes"],
            30.0,
            "FL_BCAT_AXIN2 延迟应为 30min（β-catenin→Axin2 转录延迟负反馈）",
        )
        # 节点含 bCatenin_nuclear / TCF_LEF_bcat_complex / Axin2_mRNA / Axin2
        node_ids = axin2_loop["node_ids"]
        self.assertIn("bCatenin_nuclear", node_ids)
        self.assertIn("TCF_LEF_bcat_complex", node_ids)
        self.assertIn("Axin2_mRNA", node_ids)
        self.assertIn("Axin2", node_ids)
        # 应使用 destruction_complex.j2 模板
        self.assertEqual(
            axin2_loop["template"], "destruction_complex.j2",
        )

    def test_fl_axin2_dc_reformed(self):
        """FL_AXIN2_DC_REFORMED（Axin2 重建 destruction complex, delay=0）。"""
        specialist = self._WntSpecialist()
        loops = specialist.apply_feedback({"nodes": [], "edges": []})

        reformed_loop = next(
            (l for l in loops if "AXIN2_DC_REFORMED" in l["id"]),
            None,
        )
        self.assertIsNotNone(reformed_loop, "应含 FL_AXIN2_DC_REFORMED 反馈环")
        self.assertEqual(reformed_loop["loop_type"], "negative")
        self.assertEqual(
            reformed_loop["delay_minutes"],
            0.0,
            "FL_AXIN2_DC_REFORMED delay=0（蛋白结合直接负反馈）",
        )
        # 节点含 Axin2 / destruction_complex_reformed / bCatenin
        node_ids = reformed_loop["node_ids"]
        self.assertIn("Axin2", node_ids)
        self.assertIn("destruction_complex_reformed", node_ids)
        self.assertIn("bCatenin", node_ids)

    def test_fl_plrp6_dc_disrupted_positive(self):
        """FL_PLRP6_DC_DISRUPTED（LRP6 磷酸化维持 destruction complex 解离,
        正反馈）。"""
        specialist = self._WntSpecialist()
        loops = specialist.apply_feedback({"nodes": [], "edges": []})

        plrp6_loop = next(
            (l for l in loops if "PLRP6_DC_DISRUPTED" in l["id"]),
            None,
        )
        self.assertIsNotNone(plrp6_loop, "应含 FL_PLRP6_DC_DISRUPTED 反馈环")
        self.assertEqual(plrp6_loop["loop_type"], "positive")
        self.assertEqual(
            plrp6_loop["delay_minutes"],
            0.0,
            "FL_PLRP6_DC_DISRUPTED delay=0（信号传导直接正反馈）",
        )
        # 节点含 pLRP6 / Axin_recruited / destruction_complex_disrupted / bCatenin
        node_ids = plrp6_loop["node_ids"]
        self.assertIn("pLRP6", node_ids)
        self.assertIn("Axin_recruited", node_ids)
        self.assertIn("destruction_complex_disrupted", node_ids)
        self.assertIn("bCatenin", node_ids)

    def test_feedback_loops_contain_required_fields(self):
        """每条反馈环含 id / loop_type / node_ids / delay_minutes / description。"""
        specialist = self._WntSpecialist()
        loops = specialist.apply_feedback({"nodes": [], "edges": []})

        for loop in loops:
            self.assertIn("id", loop)
            self.assertIn("loop_type", loop)
            self.assertIn("node_ids", loop)
            self.assertIn("delay_minutes", loop)
            self.assertIn("description", loop)


class TestWntCrosstalkModule(unittest.TestCase):
    """测试 6：Crosstalk 模块返回 5 条 cross-talk Reaction 片段
    （β-catenin→Cyclin D1/cMyc + pAKT→GSK3β + pERK→β-catenin + Wnt→Ca2+）。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.wnt_specialist import WntSpecialist

        clear_registry()
        register_specialist(WntSpecialist)
        self._WntSpecialist = WntSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_apply_crosstalk_returns_5_fragments(self):
        """apply_crosstalk() 返回 5 条 cross-talk Reaction 片段。"""
        specialist = self._WntSpecialist()
        fragments = specialist.apply_crosstalk(
            {"nodes": [], "edges": []}, []
        )

        self.assertEqual(
            len(fragments),
            5,
            f"期望 5 条 cross-talk 片段，实际 {len(fragments)} 条",
        )

    def test_crosstalk_contains_bcat_cyclin_d1_transcription(self):
        """含 β-catenin → Cyclin D1 transcription 片段（Wnt 促周期）。"""
        specialist = self._WntSpecialist()
        fragments = specialist.apply_crosstalk(
            {"nodes": [], "edges": []}, []
        )

        cyclin = next(
            (f for f in fragments
             if f["source"] == "bCatenin_nuclear" and f["target"] == "Cyclin_D1"),
            None,
        )
        self.assertIsNotNone(cyclin)
        self.assertEqual(cyclin["mechanism"], "transcription")
        self.assertIn("bCatenin", cyclin["shared_species"])

    def test_crosstalk_contains_bcat_cmyc_transcription(self):
        """含 β-catenin → cMyc transcription 片段（Wnt 促增殖）。"""
        specialist = self._WntSpecialist()
        fragments = specialist.apply_crosstalk(
            {"nodes": [], "edges": []}, []
        )

        cmyc = next(
            (f for f in fragments
             if f["source"] == "bCatenin_nuclear" and f["target"] == "cMyc"),
            None,
        )
        self.assertIsNotNone(cmyc)
        self.assertEqual(cmyc["mechanism"], "transcription")

    def test_crosstalk_contains_pakt_gsk3b_inhibition(self):
        """含 pAKT → GSK3b inhibition 片段（PI3K-AKT→Wnt cross-talk,
        pAKT 抑制 GSK3β 阻止 β-catenin 降解）。"""
        specialist = self._WntSpecialist()
        fragments = specialist.apply_crosstalk(
            {"nodes": [], "edges": []}, []
        )

        akt = next(
            (f for f in fragments
             if f["source"] == "pAKT" and f["target"] == "GSK3b"),
            None,
        )
        self.assertIsNotNone(akt)
        self.assertEqual(akt["mechanism"], "inhibition")
        self.assertIn("GSK3b", akt["shared_species"])

    def test_crosstalk_contains_perk_bcat_activation(self):
        """含 pERK → bCatenin activation 片段（MAPK→Wnt cross-talk,
        MAPK 磷酸化 β-catenin 增强稳定性）。"""
        specialist = self._WntSpecialist()
        fragments = specialist.apply_crosstalk(
            {"nodes": [], "edges": []}, []
        )

        erk = next(
            (f for f in fragments
             if f["source"] == "pERK" and f["target"] == "bCatenin"),
            None,
        )
        self.assertIsNotNone(erk)
        self.assertEqual(erk["mechanism"], "activation")

    def test_crosstalk_contains_wnt_ca2pkc_alternative_pathway(self):
        """含 Wnt → Ca2_PKC activation 片段（non-canonical Wnt/Ca2+ 通路）。"""
        specialist = self._WntSpecialist()
        fragments = specialist.apply_crosstalk(
            {"nodes": [], "edges": []}, []
        )

        ca2 = next(
            (f for f in fragments
             if f["source"] == "Wnt" and f["target"] == "Ca2_PKC"),
            None,
        )
        self.assertIsNotNone(ca2)
        self.assertEqual(ca2["mechanism"], "activation")

    def test_crosstalk_fragments_contain_required_fields(self):
        """每条 cross-talk 片段含 source / target / mechanism /
        shared_species / description。"""
        specialist = self._WntSpecialist()
        fragments = specialist.apply_crosstalk(
            {"nodes": [], "edges": []}, []
        )

        for f in fragments:
            self.assertIn("source", f)
            self.assertIn("target", f)
            self.assertIn("mechanism", f)
            self.assertIn("shared_species", f)
            self.assertIn("description", f)


class TestWntPerturbationModule(unittest.TestCase):
    """测试 7：Perturbation 模块返回 6 个扰动
    （ICG-001/XAV939/LGK974/Vantictumab/APC loss/CTNNB1 S45F）。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.wnt_specialist import WntSpecialist

        clear_registry()
        register_specialist(WntSpecialist)
        self._WntSpecialist = WntSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_apply_perturbation_returns_6(self):
        """apply_perturbation() 返回 6 个扰动。"""
        specialist = self._WntSpecialist()
        perturbations = specialist.apply_perturbation(
            {"nodes": [], "edges": []}, []
        )

        self.assertEqual(
            len(perturbations),
            6,
            f"期望 6 个扰动，实际 {len(perturbations)} 个",
        )

    def test_perturbation_contains_icg_001(self):
        """含 ICG-001（CBP/β-catenin interaction inhibitor）。"""
        specialist = self._WntSpecialist()
        perturbations = specialist.apply_perturbation(
            {"nodes": [], "edges": []}, []
        )

        icg = next(
            (p for p in perturbations if p.get("drug") == "ICG-001"),
            None,
        )
        self.assertIsNotNone(icg)
        self.assertEqual(icg["target"], "bCatenin")
        self.assertEqual(icg["mechanism"], "inhibition")

    def test_perturbation_contains_xav939(self):
        """含 XAV939（tankyrase inhibitor, 稳定 Axin）。"""
        specialist = self._WntSpecialist()
        perturbations = specialist.apply_perturbation(
            {"nodes": [], "edges": []}, []
        )

        xav = next(
            (p for p in perturbations if p.get("drug") == "XAV939"),
            None,
        )
        self.assertIsNotNone(xav)
        self.assertEqual(xav["target"], "Axin")
        self.assertEqual(xav["mechanism"], "inhibition")

    def test_perturbation_contains_lgk974(self):
        """含 LGK974（PORCN inhibitor, 阻止 Wnt 分泌）。"""
        specialist = self._WntSpecialist()
        perturbations = specialist.apply_perturbation(
            {"nodes": [], "edges": []}, []
        )

        lgk = next(
            (p for p in perturbations if p.get("drug") == "LGK974"),
            None,
        )
        self.assertIsNotNone(lgk)
        self.assertEqual(lgk["target"], "Wnt")
        self.assertEqual(lgk["mechanism"], "inhibition")

    def test_perturbation_contains_vantictumab(self):
        """含 Vantictumab（anti-Frizzled antibody）。"""
        specialist = self._WntSpecialist()
        perturbations = specialist.apply_perturbation(
            {"nodes": [], "edges": []}, []
        )

        van = next(
            (p for p in perturbations if p.get("drug") == "Vantictumab"),
            None,
        )
        self.assertIsNotNone(van)
        self.assertEqual(van["target"], "Frizzled")
        self.assertEqual(van["mechanism"], "inhibition")

    def test_perturbation_contains_apc_loss(self):
        """含 APC loss（loss-of-function, 结肠癌常见）。"""
        specialist = self._WntSpecialist()
        perturbations = specialist.apply_perturbation(
            {"nodes": [], "edges": []}, []
        )

        apc = next(
            (p for p in perturbations
             if p.get("ko_target") == "APC_loss"),
            None,
        )
        self.assertIsNotNone(apc)
        self.assertEqual(apc["target"], "APC")
        self.assertEqual(apc["mechanism"], "knockout")

    def test_perturbation_contains_ctnnb1_s45f(self):
        """含 CTNNB1 S45F（β-catenin mutation, 阻止磷酸化降解）。"""
        specialist = self._WntSpecialist()
        perturbations = specialist.apply_perturbation(
            {"nodes": [], "edges": []}, []
        )

        ctnnb1 = next(
            (p for p in perturbations
             if p.get("ko_target") == "CTNNB1_S45F"),
            None,
        )
        self.assertIsNotNone(ctnnb1)
        self.assertEqual(ctnnb1["target"], "bCatenin")
        self.assertEqual(ctnnb1["mechanism"], "activation")

    def test_perturbations_contain_required_fields(self):
        """每个扰动含 target / drug / mechanism / ko_target / description。"""
        specialist = self._WntSpecialist()
        perturbations = specialist.apply_perturbation(
            {"nodes": [], "edges": []}, []
        )

        for p in perturbations:
            self.assertIn("target", p)
            self.assertIn("drug", p)
            self.assertIn("mechanism", p)
            self.assertIn("ko_target", p)
            self.assertIn("description", p)


class TestWntValidationModule(unittest.TestCase):
    """测试 8：Validation 模块返回 3 条 benchmark
    （β-catenin 稳态 / Axin2 达峰 / destruction complex 完整性）。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.wnt_specialist import WntSpecialist

        clear_registry()
        register_specialist(WntSpecialist)
        self._WntSpecialist = WntSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_apply_validation_returns_3_rules(self):
        """apply_validation() 返回 3 条 Validation 规则。"""
        specialist = self._WntSpecialist()
        rules = specialist.apply_validation()

        self.assertEqual(
            len(rules),
            3,
            f"期望 3 条 Validation 规则，实际 {len(rules)} 条",
        )

    def test_bcat_steady_state_no_wnt_rule(self):
        """β-catenin 稳态（无 Wnt）<10 nM（Polakis 2002, PMID:12064617）。"""
        specialist = self._WntSpecialist()
        rules = specialist.apply_validation()

        steady = next(
            (r for r in rules
             if r["metric_name"] == "bcat_steady_state_no_wnt"),
            None,
        )
        self.assertIsNotNone(steady)
        self.assertEqual(steady["expected_max"], 10.0)
        self.assertEqual(steady["unit"], "nM")
        self.assertEqual(steady["pmid"], "PMID:12064617")

    def test_axin2_mrna_peak_time_rule(self):
        """Axin2 mRNA 达峰时间 60-120 min（Polakis 2002, PMID:12064617）。"""
        specialist = self._WntSpecialist()
        rules = specialist.apply_validation()

        peak = next(
            (r for r in rules
             if r["metric_name"] == "Axin2_mRNA_peak_time"),
            None,
        )
        self.assertIsNotNone(peak)
        self.assertEqual(peak["expected_min"], 60.0)
        self.assertEqual(peak["expected_max"], 120.0)
        self.assertEqual(peak["unit"], "minutes")
        self.assertEqual(peak["pmid"], "PMID:12064617")

    def test_destruction_complex_assembly_rule(self):
        """Destruction Complex 完整性（无 Wnt 时完整组装）。"""
        specialist = self._WntSpecialist()
        rules = specialist.apply_validation()

        assembly = next(
            (r for r in rules
             if r["metric_name"] == "destruction_complex_assembly"),
            None,
        )
        self.assertIsNotNone(assembly)
        self.assertEqual(assembly["expected"], True)
        self.assertEqual(assembly["unit"], "boolean")

    def test_validation_rules_contain_required_fields(self):
        """每条 Validation 规则含 rule_id / metric_name / expected /
        tolerance / pmid / description。"""
        specialist = self._WntSpecialist()
        rules = specialist.apply_validation()

        for r in rules:
            self.assertIn("rule_id", r)
            self.assertIn("metric_name", r)
            self.assertIn("expected", r)
            self.assertIn("tolerance", r)
            self.assertIn("pmid", r)
            self.assertIn("description", r)


class TestWntSelectTemplate(unittest.TestCase):
    """测试 9：模板选择
    （bistable → bistable_switch；
    phosphorylation → _mechanism_phosphorylation_mm）。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.wnt_specialist import WntSpecialist

        clear_registry()
        register_specialist(WntSpecialist)
        self._WntSpecialist = WntSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_select_template_bistable(self):
        """select_template('bistable') 返回 bistable_switch。"""
        specialist = self._WntSpecialist()
        self.assertEqual(
            specialist.select_template("bistable"),
            "bistable_switch",
        )

    def test_select_template_phosphorylation(self):
        """select_template('phosphorylation') 返回 _mechanism_phosphorylation_mm。"""
        specialist = self._WntSpecialist()
        self.assertEqual(
            specialist.select_template("phosphorylation"),
            "_mechanism_phosphorylation_mm",
        )

    def test_select_template_unknown_fallback(self):
        """select_template('unknown_mechanism') 走基类默认映射返回 'default'。"""
        specialist = self._WntSpecialist()
        self.assertEqual(
            specialist.select_template("unknown_mechanism"),
            "default",
        )


class TestWntFeatureFlagIsolation(unittest.TestCase):
    """测试 10：Feature Flag 隔离（flag=false 时 hook 不调用 Specialist）。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.wnt_specialist import WntSpecialist

        clear_registry()
        register_specialist(WntSpecialist)
        self._WntSpecialist = WntSpecialist

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
            "species": [{"name": "bCatenin"}],
            "reactions": [],
            "composite_reactions": [{"id": "CR_DESTRUCTION_COMPLEX"}],
        }
        specialist.apply_core.return_value = expected_core
        state = {"v4_pathway_graph": {"nodes": [{"id": "bCatenin"}], "edges": []}}

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


class TestWntMetadataAndInputValidation(unittest.TestCase):
    """测试 11：元数据 + 输入校验。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.wnt_specialist import WntSpecialist

        clear_registry()
        register_specialist(WntSpecialist)
        self._WntSpecialist = WntSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_get_metadata_returns_wnt_metadata(self):
        """get_metadata() 返回 pathway_class='WNT' + 5 模块。"""
        specialist = self._WntSpecialist()
        metadata = specialist.get_metadata()

        self.assertEqual(metadata["pathway_class"], "WNT")
        self.assertEqual(
            metadata["display_name"],
            "Wnt / β-catenin Signaling",
        )
        self.assertEqual(
            metadata["supported_modules"],
            ["core", "feedback", "crosstalk", "perturbation", "validation"],
        )
        self.assertEqual(len(metadata["supported_modules"]), 5)
        self.assertIn("version", metadata)

    def test_validate_input_empty_graph_returns_warnings(self):
        """validate_input({}) 返回非空 warning 列表。"""
        specialist = self._WntSpecialist()
        warnings = specialist.validate_input({})

        self.assertIsInstance(warnings, list)
        self.assertGreater(len(warnings), 0, "空 pathway_graph 应返回 warning")

    def test_validate_input_none_returns_warnings(self):
        """validate_input(None) 返回非空 warning 列表。"""
        specialist = self._WntSpecialist()
        warnings = specialist.validate_input(None)

        self.assertGreater(len(warnings), 0)

    def test_validate_input_valid_graph_returns_no_warnings(self):
        """含 nodes + edges 的 pathway_graph 无 warning。"""
        specialist = self._WntSpecialist()
        warnings = specialist.validate_input(
            {"nodes": [{"id": "bCatenin"}], "edges": []}
        )

        self.assertEqual(
            warnings, [],
            "含 nodes + edges 的 pathway_graph 应无 warning",
        )


if __name__ == "__main__":
    unittest.main()
