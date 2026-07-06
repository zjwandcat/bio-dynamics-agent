# BioDynamics Agent v4 - NF-κB Specialist 单元测试 (Phase 4 / Task 4.10)
#
# 测试用例（11 项，覆盖 5 模块 + flag 隔离 + 元数据 + IκBα 三步耦合
# CompositeReaction + NFkB shared + 模板选择）：
#   1. Specialist 注册：get_specialist("NF_KB") 返回 NfKappaBSpecialist 实例
#   2. Core 模块：apply_core() 返回 ≥13 条核心反应
#   3. IκBα 三步耦合 CompositeReaction：Core 输出含 CR_IKBA_DEGRADATION
#   4. NFkB shared 标记：species 中 NFkB 标记 shared=True
#      （与 Apoptosis Specialist Bcl-2 路径共享）
#   5. Feedback 模块：FL_NFKB_IKBA delay=30min + FL_NFKB_A20_IKK 双负反馈
#   6. Crosstalk 模块：apply_crosstalk() 返回 5 条 cross-talk Reaction 片段
#   7. Perturbation 模块：apply_perturbation() 返回 6 个扰动
#   8. Validation 模块：apply_validation() 返回 3 条 benchmark
#   9. 模板选择：oscillatory → oscillatory_feedback；
#      phosphorylation → _mechanism_phosphorylation_mm
#  10. Feature Flag 隔离：flag=false 时 hook 不调用 Specialist
#  11. 元数据：get_metadata() 返回 pathway_class='NF_KB'
#
# 运行：cd backend && python -m pytest tests/test_nf_kappa_b_specialist.py -v

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
class TestNfKappaBSpecialistRegistration(unittest.TestCase):
    """测试 1：Specialist 注册到 SPECIALIST_REGISTRY。"""

    def setUp(self):
        """每个测试开始前清空 registry 并重新注册 NfKappaBSpecialist。"""
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.nf_kappa_b_specialist import (
            NfKappaBSpecialist,
        )

        clear_registry()
        register_specialist(NfKappaBSpecialist)

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_get_specialist_returns_nf_kappa_b_instance(self):
        """get_specialist("NF_KB") 返回 NfKappaBSpecialist 实例。"""
        from app.pathways.pathway_registry import (
            get_specialist,
            is_specialist_available,
        )
        from app.pathways.specialists.nf_kappa_b_specialist import (
            NfKappaBSpecialist,
        )

        # 已注册
        self.assertTrue(is_specialist_available("NF_KB"))

        # get_specialist 返回 NfKappaBSpecialist 实例
        instance = get_specialist("NF_KB")
        self.assertIsNotNone(instance)
        self.assertIsInstance(instance, NfKappaBSpecialist)

        # 多次调用返回不同实例（非单例）
        instance2 = get_specialist("NF_KB")
        self.assertIsNot(instance, instance2)

    def test_pathway_class_attribute(self):
        """NfKappaBSpecialist.pathway_class == 'NF_KB'。"""
        from app.pathways.specialists.nf_kappa_b_specialist import (
            NfKappaBSpecialist,
        )

        self.assertEqual(NfKappaBSpecialist.pathway_class, "NF_KB")


class TestNfKappaBCoreModule(unittest.TestCase):
    """测试 2：Core 模块返回 ≥13 条核心反应。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.nf_kappa_b_specialist import (
            NfKappaBSpecialist,
        )

        clear_registry()
        register_specialist(NfKappaBSpecialist)
        self._NfKappaBSpecialist = NfKappaBSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_apply_core_returns_at_least_13_reactions(self):
        """apply_core() 返回 ≥13 条核心反应。"""
        specialist = self._NfKappaBSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})

        self.assertIn("species", result)
        self.assertIn("reactions", result)
        reactions = result["reactions"]
        self.assertGreaterEqual(
            len(reactions),
            13,
            f"期望至少 13 条核心反应，实际 {len(reactions)} 条",
        )

    def test_core_reaction_pairs(self):
        """13 条核心反应的 source→target 对应正确
        （TNF→TNFR→IKK→IκBα 磷酸化→泛素化→降解→NF-κB 释放→入核→转录）。"""
        specialist = self._NfKappaBSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        pairs = [(r["source"], r["target"]) for r in result["reactions"]]
        expected_pairs = [
            # TNF 信号 + IKK 激活级联
            ("TNF", "TNF_TNFR_complex"),                       # 1. TNF+TNFR 配体-受体复合物
            ("TNF_TNFR_complex", "pIKK"),                      # 2. TNF_TNFR_complex 激活 IKK
            # IκBα 三步耦合（phosphorylation→ubiquitination→proteasomal_degradation）
            ("pIKK", "pIkBa"),                                 # 3. pIKK 磷酸化 IκBα Ser32/36（step 1）
            ("pIkBa", "ubIkBa"),                               # 4. β-TrCP E3 泛素化 pIkBa（step 2）
            ("ubIkBa", "IkBa_degraded"),                       # 5. 26S 蛋白酶体降解 ubIkBa（step 3）
            # NF-κB 释放 + 入核
            ("IkBa_degraded", "NFkB"),                         # 6. IκBα 降解释放 NF-κB
            ("NFkB", "NFkB_nuclear"),                          # 7. NF-κB 入核
            # NF-κB 转录 4 条（IκBα/A20/TNF/Bcl-2）
            ("NFkB_nuclear", "IkBa_mRNA"),                     # 8. NF-κB 转录 IκBα mRNA
            ("NFkB_nuclear", "A20_mRNA"),                      # 9. NF-κB 转录 A20 mRNA
            ("NFkB_nuclear", "TNF_mRNA"),                      # 10. NF-κB 转录 TNF mRNA
            ("NFkB_nuclear", "Bcl2_mRNA"),                      # 11. NF-κB 转录 Bcl-2 mRNA
            # 翻译 2 条
            ("IkBa_mRNA", "IkBa"),                             # 12. IκBα mRNA 翻译
            ("A20_mRNA", "A20"),                               # 13. A20 mRNA 翻译
        ]
        self.assertEqual(pairs, expected_pairs)

    def test_core_reactions_contain_pathway_tag(self):
        """每条核心反应含 pathway_tag='NF_KB'。"""
        specialist = self._NfKappaBSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        for r in result["reactions"]:
            self.assertEqual(r["pathway_tag"], "NF_KB")

    def test_core_reactions_contain_required_fields(self):
        """每条核心反应含 source / target / mechanism / kinetics_type /
        substrate / product / modifier / modifier_type。"""
        specialist = self._NfKappaBSpecialist()
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

    def test_core_nf_kb_cascade_complete(self):
        """NF-κB 级联完整
        （TNF→TNF_TNFR_complex→pIKK→pIkBa→ubIkBa→IkBa_degraded→
        NFkB→NFkB_nuclear→转录）。"""
        specialist = self._NfKappaBSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        # 同时收集 source 与 target
        nodes_in_reactions = set()
        for r in result["reactions"]:
            nodes_in_reactions.add(r["source"])
            nodes_in_reactions.add(r["target"])

        # NF-κB 级联关键节点
        for node in ["TNF", "TNF_TNFR_complex", "pIKK", "pIkBa",
                     "ubIkBa", "IkBa_degraded", "NFkB", "NFkB_nuclear",
                     "IkBa_mRNA", "A20_mRNA", "TNF_mRNA", "Bcl2_mRNA"]:
            self.assertIn(
                node, nodes_in_reactions,
                f"NF-κB 级联应含 {node}",
            )

    def test_core_ikba_three_step_markers(self):
        """IκBα 三步耦合反应含 composite_step 标记（1/2/3）。"""
        specialist = self._NfKappaBSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})

        # 提取含 composite_step 标记的反应
        composite_reactions = [
            r for r in result["reactions"] if "composite_step" in r
        ]
        # 至少 3 步耦合
        self.assertGreaterEqual(
            len(composite_reactions), 3,
            "IκBα 三步耦合应含至少 3 步标记",
        )
        # 检查 1/2/3 步均存在
        steps = {r["composite_step"] for r in composite_reactions}
        self.assertIn(1, steps, "三步耦合第 1 步 phosphorylation 应存在")
        self.assertIn(2, steps, "三步耦合第 2 步 ubiquitination 应存在")
        self.assertIn(3, steps, "三步耦合第 3 步 proteasomal_degradation 应存在")
        # 检查机制对应正确
        step1 = next(r for r in composite_reactions if r["composite_step"] == 1)
        self.assertEqual(step1["mechanism"], "phosphorylation")
        step2 = next(r for r in composite_reactions if r["composite_step"] == 2)
        self.assertEqual(step2["mechanism"], "ubiquitination")
        step3 = next(r for r in composite_reactions if r["composite_step"] == 3)
        self.assertEqual(step3["mechanism"], "proteasomal_degradation")
        # 所有三步共享 composite_id
        composite_ids = {r["composite_id"] for r in composite_reactions}
        self.assertEqual(
            len(composite_ids), 1,
            "三步耦合应共享同一 composite_id",
        )
        self.assertEqual(
            composite_ids.pop(), "CR_IKBA_DEGRADATION",
        )


class TestNfKappaBCompositeReaction(unittest.TestCase):
    """测试 3：IκBα 三步耦合 CompositeReaction 输出。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.nf_kappa_b_specialist import (
            NfKappaBSpecialist,
        )

        clear_registry()
        register_specialist(NfKappaBSpecialist)
        self._NfKappaBSpecialist = NfKappaBSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_core_contains_composite_reactions(self):
        """Core 输出含 composite_reactions 字段。"""
        specialist = self._NfKappaBSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})

        self.assertIn("composite_reactions", result)
        cr_list = result["composite_reactions"]
        self.assertIsInstance(cr_list, list)
        self.assertGreater(len(cr_list), 0, "composite_reactions 不应为空")

    def test_composite_reaction_ikba_degradation(self):
        """CompositeReaction 含 IκBα 三步耦合降解（CR_IKBA_DEGRADATION）。"""
        specialist = self._NfKappaBSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        cr_list = result["composite_reactions"]

        # 查找 IκBα 三步耦合降解 CompositeReaction
        ikba_cr = next(
            (cr for cr in cr_list
             if "IKBA_DEGRADATION" in cr.get("id", "")
             or "IκBα" in cr.get("name", "")),
            None,
        )
        self.assertIsNotNone(
            ikba_cr,
            "应含 IκBα 三步耦合降解 CompositeReaction",
        )

        # 检查字段
        self.assertEqual(ikba_cr["id"], "CR_IKBA_DEGRADATION")
        self.assertEqual(ikba_cr["mechanism"], "sequential")
        self.assertEqual(ikba_cr["coupling_type"], "sequential")

    def test_composite_reaction_intermediate_species(self):
        """CompositeReaction 中间产物 pIkBa / ubIkBa / IkBa_degraded 完整。"""
        specialist = self._NfKappaBSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        cr_list = result["composite_reactions"]

        ikba_cr = next(
            (cr for cr in cr_list
             if "IKBA_DEGRADATION" in cr.get("id", "")),
            None,
        )
        self.assertIsNotNone(ikba_cr)

        # 检查中间产物
        intermediate_species = ikba_cr.get("intermediate_species", [])
        self.assertIn("pIkBa", intermediate_species)
        self.assertIn("ubIkBa", intermediate_species)
        self.assertIn("IkBa_degraded", intermediate_species)

        # 检查子反应链 pIKK → pIkBa → ubIkBa → IkBa_degraded
        sub_reactions = ikba_cr.get("sub_reactions", [])
        self.assertGreaterEqual(len(sub_reactions), 3)
        self.assertIn("pIKK → pIkBa", sub_reactions)
        self.assertIn("pIkBa → ubIkBa", sub_reactions)
        self.assertIn("ubIkBa → IkBa_degraded", sub_reactions)

    def test_composite_reaction_node_ids(self):
        """CompositeReaction node_ids 含 pIKK / pIkBa / ubIkBa / IkBa_degraded / NFkB。"""
        specialist = self._NfKappaBSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        cr_list = result["composite_reactions"]

        ikba_cr = next(
            (cr for cr in cr_list
             if "IKBA_DEGRADATION" in cr.get("id", "")),
            None,
        )
        self.assertIsNotNone(ikba_cr)

        node_ids = ikba_cr.get("node_ids", [])
        self.assertIn("pIKK", node_ids)
        self.assertIn("pIkBa", node_ids)
        self.assertIn("ubIkBa", node_ids)
        self.assertIn("IkBa_degraded", node_ids)
        self.assertIn("NFkB", node_ids)

    def test_composite_reaction_template_oscillatory(self):
        """CompositeReaction 标记 oscillatory_feedback 模板与 negative loop。"""
        specialist = self._NfKappaBSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        cr_list = result["composite_reactions"]

        ikba_cr = next(
            (cr for cr in cr_list
             if "IKBA_DEGRADATION" in cr.get("id", "")),
            None,
        )
        self.assertIsNotNone(ikba_cr)
        self.assertEqual(ikba_cr.get("loop_type"), "negative")
        self.assertEqual(
            ikba_cr.get("template"), "oscillatory_feedback.j2",
        )


class TestNfKappaBSharedMarker(unittest.TestCase):
    """测试 4：NFkB shared 标记（与 Apoptosis Specialist Bcl-2 路径共享）。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.nf_kappa_b_specialist import (
            NfKappaBSpecialist,
        )

        clear_registry()
        register_specialist(NfKappaBSpecialist)
        self._NfKappaBSpecialist = NfKappaBSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_nfkb_marked_as_shared(self):
        """species 中 NFkB 标记 shared=True
        （与 Apoptosis Specialist 的 Bcl-2 抗凋亡路径共享）。"""
        specialist = self._NfKappaBSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        species = result["species"]

        nfkb = next(
            (s for s in species if s["name"] == "NFkB"),
            None,
        )
        self.assertIsNotNone(nfkb, "Core 输出应含 NFkB 物种")
        self.assertTrue(
            nfkb.get("shared"),
            "NFkB 应标记 shared=True（与 Apoptosis Specialist 的 Bcl-2 抗凋亡路径共享）",
        )


class TestNfKappaBFeedbackModule(unittest.TestCase):
    """测试 5：Feedback 模块（FL_NFKB_IKBA delay=30min + FL_NFKB_A20_IKK 双负反馈）。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.nf_kappa_b_specialist import (
            NfKappaBSpecialist,
        )

        clear_registry()
        register_specialist(NfKappaBSpecialist)
        self._NfKappaBSpecialist = NfKappaBSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_apply_feedback_returns_at_least_3_loops(self):
        """apply_feedback() 返回 ≥3 条反馈环。"""
        specialist = self._NfKappaBSpecialist()
        loops = specialist.apply_feedback({"nodes": [], "edges": []})

        self.assertGreaterEqual(
            len(loops),
            3,
            f"期望至少 3 条反馈环，实际 {len(loops)} 条",
        )

    def test_fl_nfkb_ikba_delayed_negative_feedback(self):
        """FL_NFKB_IKBA 转录延迟负反馈（delay=30min）。"""
        specialist = self._NfKappaBSpecialist()
        loops = specialist.apply_feedback({"nodes": [], "edges": []})

        ikba_loop = next(
            (l for l in loops if "NFKB_IKBA" in l["id"]),
            None,
        )
        self.assertIsNotNone(ikba_loop, "应含 FL_NFKB_IKBA 反馈环")
        self.assertEqual(ikba_loop["loop_type"], "negative")
        self.assertEqual(
            ikba_loop["delay_minutes"],
            30.0,
            "FL_NFKB_IKBA 延迟应为 30min（NF-κB→IκBα 转录延迟负反馈振荡）",
        )
        # 节点含 NFkB_nuclear / IkBa_mRNA / IkBa / NFkB
        node_ids = ikba_loop["node_ids"]
        self.assertIn("NFkB_nuclear", node_ids)
        self.assertIn("IkBa_mRNA", node_ids)
        self.assertIn("IkBa", node_ids)
        self.assertIn("NFkB", node_ids)
        # 应使用 oscillatory_feedback 模板
        self.assertEqual(
            ikba_loop["template"], "oscillatory_feedback.j2",
        )

    def test_fl_nfkb_a20_ikk_double_negative(self):
        """FL_NFKB_A20_IKK 双负反馈（A20 抑制 IKK，delay=60min）。"""
        specialist = self._NfKappaBSpecialist()
        loops = specialist.apply_feedback({"nodes": [], "edges": []})

        a20_loop = next(
            (l for l in loops if "NFKB_A20_IKK" in l["id"]),
            None,
        )
        self.assertIsNotNone(a20_loop, "应含 FL_NFKB_A20_IKK 反馈环")
        self.assertEqual(a20_loop["loop_type"], "negative")
        self.assertEqual(
            a20_loop["delay_minutes"],
            60.0,
            "FL_NFKB_A20_IKK 延迟应为 60min（A20 转录翻译延迟）",
        )
        # 节点含 NFkB_nuclear / A20_mRNA / A20 / pIKK
        node_ids = a20_loop["node_ids"]
        self.assertIn("NFkB_nuclear", node_ids)
        self.assertIn("A20_mRNA", node_ids)
        self.assertIn("A20", node_ids)
        self.assertIn("pIKK", node_ids)

    def test_fl_ikba_nfkb_inhibition(self):
        """FL_IKBA_NFKB_INHIBITION（IκBα→NF-κB 抑制，delay=0）。"""
        specialist = self._NfKappaBSpecialist()
        loops = specialist.apply_feedback({"nodes": [], "edges": []})

        inh_loop = next(
            (l for l in loops if "IKBA_NFKB_INHIBITION" in l["id"]),
            None,
        )
        self.assertIsNotNone(inh_loop, "应含 FL_IKBA_NFKB_INHIBITION 反馈环")
        self.assertEqual(inh_loop["loop_type"], "negative")
        self.assertEqual(
            inh_loop["delay_minutes"],
            0.0,
            "IκBα→NF-κB 抑制 delay=0（蛋白结合直接负反馈）",
        )
        node_ids = inh_loop["node_ids"]
        self.assertIn("IkBa", node_ids)
        self.assertIn("NFkB", node_ids)

    def test_feedback_loops_contain_required_fields(self):
        """每条反馈环含 id / loop_type / node_ids / delay_minutes / description。"""
        specialist = self._NfKappaBSpecialist()
        loops = specialist.apply_feedback({"nodes": [], "edges": []})

        for loop in loops:
            self.assertIn("id", loop)
            self.assertIn("loop_type", loop)
            self.assertIn("node_ids", loop)
            self.assertIn("delay_minutes", loop)
            self.assertIn("description", loop)


class TestNfKappaBCrosstalkModule(unittest.TestCase):
    """测试 6：Crosstalk 模块返回 5 条 cross-talk Reaction 片段。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.nf_kappa_b_specialist import (
            NfKappaBSpecialist,
        )

        clear_registry()
        register_specialist(NfKappaBSpecialist)
        self._NfKappaBSpecialist = NfKappaBSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_apply_crosstalk_returns_5_fragments(self):
        """apply_crosstalk() 返回 5 条 cross-talk Reaction 片段。"""
        specialist = self._NfKappaBSpecialist()
        fragments = specialist.apply_crosstalk(
            {"nodes": [], "edges": []}, []
        )

        self.assertEqual(
            len(fragments),
            5,
            f"期望 5 条 cross-talk 片段，实际 {len(fragments)} 条",
        )

    def test_crosstalk_contains_nfkb_bcl2_transcription(self):
        """含 NF-κB → Bcl-2 transcription 片段（NF-κB 抗凋亡）。"""
        specialist = self._NfKappaBSpecialist()
        fragments = specialist.apply_crosstalk(
            {"nodes": [], "edges": []}, []
        )

        bcl2 = next(
            (f for f in fragments
             if f["source"] == "NFkB_nuclear" and f["target"] == "Bcl2"),
            None,
        )
        self.assertIsNotNone(bcl2)
        self.assertEqual(bcl2["mechanism"], "transcription")
        self.assertIn("NFkB", bcl2["shared_species"])

    def test_crosstalk_contains_nfkb_bclxl_transcription(self):
        """含 NF-κB → Bcl_xL transcription 片段（NF-κB 抗凋亡）。"""
        specialist = self._NfKappaBSpecialist()
        fragments = specialist.apply_crosstalk(
            {"nodes": [], "edges": []}, []
        )

        bclxl = next(
            (f for f in fragments
             if f["source"] == "NFkB_nuclear" and f["target"] == "Bcl_xL"),
            None,
        )
        self.assertIsNotNone(bclxl)
        self.assertEqual(bclxl["mechanism"], "transcription")

    def test_crosstalk_contains_nfkb_cyclin_d1_transcription(self):
        """含 NF-κB → Cyclin_D1 transcription 片段（NF-κB 促周期）。"""
        specialist = self._NfKappaBSpecialist()
        fragments = specialist.apply_crosstalk(
            {"nodes": [], "edges": []}, []
        )

        cyclin = next(
            (f for f in fragments
             if f["source"] == "NFkB_nuclear" and f["target"] == "Cyclin_D1"),
            None,
        )
        self.assertIsNotNone(cyclin)
        self.assertEqual(cyclin["mechanism"], "transcription")

    def test_crosstalk_contains_pakt_ikk_activation(self):
        """含 pAKT → IKK activation 片段（PI3K-AKT→NF-κB cross-talk）。"""
        specialist = self._NfKappaBSpecialist()
        fragments = specialist.apply_crosstalk(
            {"nodes": [], "edges": []}, []
        )

        akt = next(
            (f for f in fragments
             if f["source"] == "pAKT" and f["target"] == "IKK"),
            None,
        )
        self.assertIsNotNone(akt)
        self.assertEqual(akt["mechanism"], "activation")

    def test_crosstalk_contains_p53_nfkb_inhibition(self):
        """含 p53 → NFkB inhibition 片段（p53 拮抗 NF-κB）。"""
        specialist = self._NfKappaBSpecialist()
        fragments = specialist.apply_crosstalk(
            {"nodes": [], "edges": []}, []
        )

        p53 = next(
            (f for f in fragments
             if f["source"] == "p53" and f["target"] == "NFkB"),
            None,
        )
        self.assertIsNotNone(p53)
        self.assertEqual(p53["mechanism"], "inhibition")

    def test_crosstalk_fragments_contain_required_fields(self):
        """每条 cross-talk 片段含 source / target / mechanism /
        shared_species / description。"""
        specialist = self._NfKappaBSpecialist()
        fragments = specialist.apply_crosstalk(
            {"nodes": [], "edges": []}, []
        )

        for f in fragments:
            self.assertIn("source", f)
            self.assertIn("target", f)
            self.assertIn("mechanism", f)
            self.assertIn("shared_species", f)
            self.assertIn("description", f)


class TestNfKappaBPerturbationModule(unittest.TestCase):
    """测试 7：Perturbation 模块返回 6 个扰动。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.nf_kappa_b_specialist import (
            NfKappaBSpecialist,
        )

        clear_registry()
        register_specialist(NfKappaBSpecialist)
        self._NfKappaBSpecialist = NfKappaBSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_apply_perturbation_returns_6(self):
        """apply_perturbation() 返回 6 个扰动。"""
        specialist = self._NfKappaBSpecialist()
        perturbations = specialist.apply_perturbation(
            {"nodes": [], "edges": []}, []
        )

        self.assertEqual(
            len(perturbations),
            6,
            f"期望 6 个扰动，实际 {len(perturbations)} 个",
        )

    def test_perturbation_contains_bortezomib(self):
        """含 Bortezomib（26S 蛋白酶体抑制剂, FDA-approved）。"""
        specialist = self._NfKappaBSpecialist()
        perturbations = specialist.apply_perturbation(
            {"nodes": [], "edges": []}, []
        )

        bortezomib = next(
            (p for p in perturbations if p.get("drug") == "Bortezomib"),
            None,
        )
        self.assertIsNotNone(bortezomib)
        self.assertEqual(bortezomib["target"], "proteasome")
        self.assertEqual(bortezomib["mechanism"], "inhibition")

    def test_perturbation_contains_bay_11_7082(self):
        """含 BAY 11-7082（IKK 抑制剂）。"""
        specialist = self._NfKappaBSpecialist()
        perturbations = specialist.apply_perturbation(
            {"nodes": [], "edges": []}, []
        )

        bay = next(
            (p for p in perturbations if p.get("drug") == "BAY 11-7082"),
            None,
        )
        self.assertIsNotNone(bay)
        self.assertEqual(bay["target"], "IKK")
        self.assertEqual(bay["mechanism"], "inhibition")

    def test_perturbation_contains_ikk_16(self):
        """含 IKK-16（IKK 抑制剂）。"""
        specialist = self._NfKappaBSpecialist()
        perturbations = specialist.apply_perturbation(
            {"nodes": [], "edges": []}, []
        )

        ikk16 = next(
            (p for p in perturbations if p.get("drug") == "IKK-16"),
            None,
        )
        self.assertIsNotNone(ikk16)
        self.assertEqual(ikk16["target"], "IKK")
        self.assertEqual(ikk16["mechanism"], "inhibition")

    def test_perturbation_contains_mln120b(self):
        """含 MLN120B（IKKβ 选择性抑制剂）。"""
        specialist = self._NfKappaBSpecialist()
        perturbations = specialist.apply_perturbation(
            {"nodes": [], "edges": []}, []
        )

        mln = next(
            (p for p in perturbations if p.get("drug") == "MLN120B"),
            None,
        )
        self.assertIsNotNone(mln)
        self.assertEqual(mln["target"], "IKK")
        self.assertEqual(mln["mechanism"], "inhibition")

    def test_perturbation_contains_nfkbia_loss(self):
        """含 NFKBIA loss（loss-of-function, IκBα 失去功能）。"""
        specialist = self._NfKappaBSpecialist()
        perturbations = specialist.apply_perturbation(
            {"nodes": [], "edges": []}, []
        )

        nfkbia = next(
            (p for p in perturbations
             if p.get("ko_target") == "NFKBIA_loss"),
            None,
        )
        self.assertIsNotNone(nfkbia)
        self.assertEqual(nfkbia["target"], "IkBa")
        self.assertEqual(nfkbia["mechanism"], "knockout")

    def test_perturbation_contains_pdtc(self):
        """含 PDTC（NF-κB 抑制剂 + 抗氧化剂）。"""
        specialist = self._NfKappaBSpecialist()
        perturbations = specialist.apply_perturbation(
            {"nodes": [], "edges": []}, []
        )

        pdtc = next(
            (p for p in perturbations if p.get("drug") == "PDTC"),
            None,
        )
        self.assertIsNotNone(pdtc)
        self.assertEqual(pdtc["target"], "NFkB")
        self.assertEqual(pdtc["mechanism"], "inhibition")

    def test_perturbations_contain_required_fields(self):
        """每个扰动含 target / drug / mechanism / ko_target / description。"""
        specialist = self._NfKappaBSpecialist()
        perturbations = specialist.apply_perturbation(
            {"nodes": [], "edges": []}, []
        )

        for p in perturbations:
            self.assertIn("target", p)
            self.assertIn("drug", p)
            self.assertIn("mechanism", p)
            self.assertIn("ko_target", p)
            self.assertIn("description", p)


class TestNfKappaBValidationModule(unittest.TestCase):
    """测试 8：Validation 模块返回 3 条 benchmark。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.nf_kappa_b_specialist import (
            NfKappaBSpecialist,
        )

        clear_registry()
        register_specialist(NfKappaBSpecialist)
        self._NfKappaBSpecialist = NfKappaBSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_apply_validation_returns_3_rules(self):
        """apply_validation() 返回 3 条 Validation 规则。"""
        specialist = self._NfKappaBSpecialist()
        rules = specialist.apply_validation()

        self.assertEqual(
            len(rules),
            3,
            f"期望 3 条 Validation 规则，实际 {len(rules)} 条",
        )

    def test_nfkb_oscillation_period_rule(self):
        """NF-κB 核振荡周期 1-2 hours（Nelson 2004, PMID:14976212）。"""
        specialist = self._NfKappaBSpecialist()
        rules = specialist.apply_validation()

        period = next(
            (r for r in rules
             if r["metric_name"] == "NFkB_nuclear_oscillation_period"),
            None,
        )
        self.assertIsNotNone(period)
        self.assertEqual(period["expected_min"], 1.0)
        self.assertEqual(period["expected_max"], 2.0)
        self.assertEqual(period["unit"], "hours")
        self.assertEqual(period["pmid"], "PMID:14976212")

    def test_ikba_transcription_delay_rule(self):
        """IκBα 转录延迟 30-60 min（Nelson 2004, PMID:14976212）。"""
        specialist = self._NfKappaBSpecialist()
        rules = specialist.apply_validation()

        delay = next(
            (r for r in rules
             if r["metric_name"] == "IkBa_transcription_delay"),
            None,
        )
        self.assertIsNotNone(delay)
        self.assertEqual(delay["expected_min"], 30.0)
        self.assertEqual(delay["expected_max"], 60.0)
        self.assertEqual(delay["unit"], "minutes")
        self.assertEqual(delay["pmid"], "PMID:14976212")

    def test_nfkb_oscillation_duration_rule(self):
        """NF-κB 振荡持续时间 6-20 hours（Nelson 2004, PMID:14976212）。"""
        specialist = self._NfKappaBSpecialist()
        rules = specialist.apply_validation()

        duration = next(
            (r for r in rules
             if r["metric_name"] == "NFkB_oscillation_duration"),
            None,
        )
        self.assertIsNotNone(duration)
        self.assertEqual(duration["expected_min"], 6.0)
        self.assertEqual(duration["expected_max"], 20.0)
        self.assertEqual(duration["unit"], "hours")
        self.assertEqual(duration["pmid"], "PMID:14976212")

    def test_validation_rules_contain_required_fields(self):
        """每条 Validation 规则含 rule_id / metric_name / expected /
        tolerance / pmid / description。"""
        specialist = self._NfKappaBSpecialist()
        rules = specialist.apply_validation()

        for r in rules:
            self.assertIn("rule_id", r)
            self.assertIn("metric_name", r)
            self.assertIn("expected", r)
            self.assertIn("tolerance", r)
            self.assertIn("pmid", r)
            self.assertIn("description", r)


class TestNfKappaBSelectTemplate(unittest.TestCase):
    """测试 9：模板选择
    （oscillatory → oscillatory_feedback；
    phosphorylation → _mechanism_phosphorylation_mm）。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.nf_kappa_b_specialist import (
            NfKappaBSpecialist,
        )

        clear_registry()
        register_specialist(NfKappaBSpecialist)
        self._NfKappaBSpecialist = NfKappaBSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_select_template_oscillatory(self):
        """select_template('oscillatory') 返回 oscillatory_feedback。"""
        specialist = self._NfKappaBSpecialist()
        self.assertEqual(
            specialist.select_template("oscillatory"),
            "oscillatory_feedback",
        )

    def test_select_template_phosphorylation(self):
        """select_template('phosphorylation') 返回 _mechanism_phosphorylation_mm。"""
        specialist = self._NfKappaBSpecialist()
        self.assertEqual(
            specialist.select_template("phosphorylation"),
            "_mechanism_phosphorylation_mm",
        )

    def test_select_template_transcription(self):
        """select_template('transcription') 返回 oscillatory_feedback
        （NF-κB 转录使用 DDE 振荡模板表达延迟反馈）。"""
        specialist = self._NfKappaBSpecialist()
        self.assertEqual(
            specialist.select_template("transcription"),
            "oscillatory_feedback",
        )

    def test_select_template_bistable_fallback(self):
        """select_template('bistable') 走基类默认映射返回 bistable_switch。"""
        specialist = self._NfKappaBSpecialist()
        self.assertEqual(
            specialist.select_template("bistable"),
            "bistable_switch",
        )


class TestNfKappaBFeatureFlagIsolation(unittest.TestCase):
    """测试 10：Feature Flag 隔离（flag=false 时 hook 不调用 Specialist）。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.nf_kappa_b_specialist import (
            NfKappaBSpecialist,
        )

        clear_registry()
        register_specialist(NfKappaBSpecialist)
        self._NfKappaBSpecialist = NfKappaBSpecialist

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
            "species": [{"name": "NFkB"}],
            "reactions": [],
            "composite_reactions": [{"id": "CR_IKBA_DEGRADATION"}],
        }
        specialist.apply_core.return_value = expected_core
        state = {"v4_pathway_graph": {"nodes": [{"id": "NFkB"}], "edges": []}}

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


class TestNfKappaBMetadataAndInputValidation(unittest.TestCase):
    """测试 11：元数据 + 输入校验。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.nf_kappa_b_specialist import (
            NfKappaBSpecialist,
        )

        clear_registry()
        register_specialist(NfKappaBSpecialist)
        self._NfKappaBSpecialist = NfKappaBSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_get_metadata_returns_nf_kappa_b_metadata(self):
        """get_metadata() 返回 pathway_class='NF_KB' + 5 模块。"""
        specialist = self._NfKappaBSpecialist()
        metadata = specialist.get_metadata()

        self.assertEqual(metadata["pathway_class"], "NF_KB")
        self.assertEqual(
            metadata["display_name"],
            "NF-κB Signaling",
        )
        self.assertEqual(
            metadata["supported_modules"],
            ["core", "feedback", "crosstalk", "perturbation", "validation"],
        )
        self.assertEqual(len(metadata["supported_modules"]), 5)
        self.assertIn("version", metadata)

    def test_validate_input_empty_graph_returns_warnings(self):
        """validate_input({}) 返回非空 warning 列表。"""
        specialist = self._NfKappaBSpecialist()
        warnings = specialist.validate_input({})

        self.assertIsInstance(warnings, list)
        self.assertGreater(len(warnings), 0, "空 pathway_graph 应返回 warning")

    def test_validate_input_none_returns_warnings(self):
        """validate_input(None) 返回非空 warning 列表。"""
        specialist = self._NfKappaBSpecialist()
        warnings = specialist.validate_input(None)

        self.assertGreater(len(warnings), 0)

    def test_validate_input_valid_graph_returns_no_warnings(self):
        """含 nodes + edges 的 pathway_graph 无 warning。"""
        specialist = self._NfKappaBSpecialist()
        warnings = specialist.validate_input(
            {"nodes": [{"id": "NFkB"}], "edges": []}
        )

        self.assertEqual(
            warnings, [],
            "含 nodes + edges 的 pathway_graph 应无 warning",
        )


if __name__ == "__main__":
    unittest.main()
