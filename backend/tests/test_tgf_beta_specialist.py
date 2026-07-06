# BioDynamics Agent v4 - TGF-β Specialist 单元测试 (Phase 4 / Task 4.12)
#
# 测试用例（≥45 项，覆盖 5 模块 + flag 隔离 + 元数据 + 输入校验 + SMAD 三步耦合
# CompositeReaction + Smad2 状态机 + 模板选择 + shared_species 标记）：
#   1. Specialist 注册：get_specialist("TGF_BETA") 返回 TgfBetaSpecialist 实例
#   2. Core 模块：apply_core() 返回 11 条核心反应 + 19 物种
#   3. SMAD 复合-入核-转录三步 CompositeReaction（complex_formation→nuclear_import→transcription）
#   4. Smad2 状态机（cyto_Smad2→cyto_pSmad2→cyto_pSmad2_Smad4→nuc_pSmad2_Smad4）
#   5. Feedback 模块：FL_SMAD7 delay=30min + FL_SMURF delay=60min
#   6. Crosstalk 模块：apply_crosstalk() 返回 4 条描述性 cross-talk edge
#   7. Perturbation 模块：apply_perturbation() 返回 3 个扰动（2 药物 + 1 KO）
#   8. Validation 模块：apply_validation() 返回 3 条 benchmark
#   9. 模板选择：phosphorylation → _mechanism_phosphorylation_mm；
#      transcription → transcription_factor（规范名称, 由 ODE Renderer 决定 fallback）
#  10. Feature Flag 隔离：flag=false 时 hook 不调用 Specialist
#  11. 元数据 + 输入校验 + load_module
#
# 运行：cd backend && python -m pytest tests/test_tgf_beta_specialist.py -v

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
class TestTgfBetaSpecialistRegistration(unittest.TestCase):
    """测试 1：Specialist 注册到 SPECIALIST_REGISTRY。"""

    def setUp(self):
        """每个测试开始前清空 registry 并重新注册 TgfBetaSpecialist。"""
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.tgf_beta_specialist import (
            TgfBetaSpecialist,
        )

        clear_registry()
        register_specialist(TgfBetaSpecialist)

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_get_specialist_returns_tgf_beta_instance(self):
        """get_specialist("TGF_BETA") 返回 TgfBetaSpecialist 实例。"""
        from app.pathways.pathway_registry import (
            get_specialist,
            is_specialist_available,
        )
        from app.pathways.specialists.tgf_beta_specialist import (
            TgfBetaSpecialist,
        )

        # 已注册
        self.assertTrue(is_specialist_available("TGF_BETA"))

        # get_specialist 返回 TgfBetaSpecialist 实例
        instance = get_specialist("TGF_BETA")
        self.assertIsNotNone(instance)
        self.assertIsInstance(instance, TgfBetaSpecialist)

        # 多次调用返回不同实例（非单例）
        instance2 = get_specialist("TGF_BETA")
        self.assertIsNot(instance, instance2)

    def test_pathway_class_attribute(self):
        """TgfBetaSpecialist.pathway_class == 'TGF_BETA'。"""
        from app.pathways.specialists.tgf_beta_specialist import (
            TgfBetaSpecialist,
        )

        self.assertEqual(TgfBetaSpecialist.pathway_class, "TGF_BETA")


class TestTgfBetaCoreModule(unittest.TestCase):
    """测试 2：Core 模块返回 11 条核心反应 + 19 物种。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.tgf_beta_specialist import (
            TgfBetaSpecialist,
        )

        clear_registry()
        register_specialist(TgfBetaSpecialist)
        self._TgfBetaSpecialist = TgfBetaSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_apply_core_returns_11_reactions(self):
        """apply_core() 返回恰好 11 条核心反应。"""
        specialist = self._TgfBetaSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})

        self.assertIn("species", result)
        self.assertIn("reactions", result)
        reactions = result["reactions"]
        self.assertEqual(
            len(reactions),
            11,
            f"期望恰好 11 条核心反应，实际 {len(reactions)} 条",
        )

    def test_core_reaction_pairs(self):
        """11 条核心反应的 source→target 对应正确
        （TGF_beta→TGF_beta_TbRII→TGF_beta_TbRII_TbRI→pTbRI→pSmad2/pSmad3→
        pSmad2_Smad4/pSmad3_Smad4→pSmad2_Smad4_nuc/pSmad3_Smad4_nuc→转录）。"""
        specialist = self._TgfBetaSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        pairs = [(r["source"], r["target"]) for r in result["reactions"]]
        expected_pairs = [
            # 受体复合物形成 + TβRI 磷酸化 + R-SMAD 磷酸化
            ("TGF_beta", "TGF_beta_TbRII"),                 # 1. ligand-receptor binding
            ("TGF_beta_TbRII", "TGF_beta_TbRII_TbRI"),      # 2. receptor recruitment
            ("TGF_beta_TbRII_TbRI", "pTbRI"),               # 3. TβRII 磷酸化 TβRI
            ("pTbRI", "pSmad2"),                            # 4. pTbRI 磷酸化 Smad2
            ("pTbRI", "pSmad3"),                            # 5. pTbRI 磷酸化 Smad3
            # Co-SMAD 异源复合 + nuclear import
            ("pSmad2", "pSmad2_Smad4"),                     # 6. Co-SMAD 异源复合 (CR step 1)
            ("pSmad3", "pSmad3_Smad4"),                     # 7. Co-SMAD 异源复合
            ("pSmad2_Smad4", "pSmad2_Smad4_nuc"),          # 8. nuclear_import (CR step 2)
            ("pSmad3_Smad4", "pSmad3_Smad4_nuc"),          # 9. nuclear_import
            # 转录
            ("pSmad2_Smad4_nuc", "PAI1_mRNA"),             # 10. PAI-1 转录 (CR step 3)
            ("pSmad2_Smad4_nuc", "SMAD7_mRNA"),            # 11. SMAD7 转录（负反馈准备）
        ]
        self.assertEqual(pairs, expected_pairs)

    def test_core_reactions_contain_pathway_tag(self):
        """每条核心反应含 pathway_tag='TGF_BETA'。"""
        specialist = self._TgfBetaSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        for r in result["reactions"]:
            self.assertEqual(r["pathway_tag"], "TGF_BETA")

    def test_core_reactions_contain_required_fields(self):
        """每条核心反应含 source / target / mechanism / kinetics_type /
        substrate / product / modifier / modifier_type。"""
        specialist = self._TgfBetaSpecialist()
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

    def test_core_tgf_beta_cascade_complete(self):
        """TGF-β 级联完整（TGF_beta→TGF_beta_TbRII→TGF_beta_TbRII_TbRI→pTbRI→
        pSmad2→pSmad2_Smad4→pSmad2_Smad4_nuc→PAI1_mRNA/SMAD7_mRNA）。
        注：Smad4 作为 substrate 出现在复合反应中（不作为 source/target），
        由 test_core_species_contain_key_species 验证。"""
        specialist = self._TgfBetaSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        # 同时收集 source 与 target
        nodes_in_reactions = set()
        for r in result["reactions"]:
            nodes_in_reactions.add(r["source"])
            nodes_in_reactions.add(r["target"])

        # TGF-β 级联关键节点（source/target 出现的节点；Smad4 仅作 substrate，
        # 由 test_core_species_contain_key_species 单独验证）
        for node in ["TGF_beta", "TGF_beta_TbRII", "TGF_beta_TbRII_TbRI",
                     "pTbRI", "pSmad2", "pSmad3",
                     "pSmad2_Smad4", "pSmad3_Smad4",
                     "pSmad2_Smad4_nuc", "pSmad3_Smad4_nuc",
                     "PAI1_mRNA", "SMAD7_mRNA"]:
            self.assertIn(
                node, nodes_in_reactions,
                f"TGF-β 级联应含 {node}",
            )

    def test_core_species_count(self):
        """Core 输出含 ≥15 物种（TGF-β + TβRII + TβRI + 复合物 + SMAD 等）。"""
        specialist = self._TgfBetaSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        species = result["species"]
        self.assertGreaterEqual(
            len(species),
            15,
            f"期望至少 15 物种，实际 {len(species)} 物种",
        )

    def test_core_species_contain_key_species(self):
        """Core 输出含关键物种：TGF_beta / TbRII / TbRI / Smad2 / Smad3 / Smad4 / pSmad2 / SMAD7 / SMURF。"""
        specialist = self._TgfBetaSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        species_names = {s["name"] for s in result["species"]}

        for name in ["TGF_beta", "TbRII", "TbRI", "pTbRI",
                     "Smad2", "Smad3", "Smad4", "pSmad2", "pSmad3",
                     "pSmad2_Smad4", "pSmad3_Smad4",
                     "pSmad2_Smad4_nuc", "pSmad3_Smad4_nuc",
                     "PAI1_mRNA", "SMAD7_mRNA", "SMAD7", "SMURF"]:
            self.assertIn(
                name, species_names,
                f"Core 物种应含 {name}",
            )

    def test_core_reactions_mechanism_types(self):
        """每条核心反应 mechanism 类型正确（complex_formation / phosphorylation /
        nuclear_import / transcription）。"""
        specialist = self._TgfBetaSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        mechanisms = {r["mechanism"] for r in result["reactions"]}

        # TGF-β 通路应含这 4 种机制
        self.assertIn("complex_formation", mechanisms)
        self.assertIn("phosphorylation", mechanisms)
        self.assertIn("nuclear_import", mechanisms)
        self.assertIn("transcription", mechanisms)

    def test_core_phosphorylation_uses_michaelis_menten(self):
        """磷酸化反应使用 Michaelis_Menten 动力学（与 P3 _mechanism_phosphorylation_mm 对齐）。"""
        specialist = self._TgfBetaSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        phos_reactions = [
            r for r in result["reactions"]
            if r["mechanism"] == "phosphorylation"
        ]
        # 至少 3 条磷酸化（TβRI 磷酸化 + Smad2 磷酸化 + Smad3 磷酸化）
        self.assertGreaterEqual(len(phos_reactions), 3)
        for r in phos_reactions:
            self.assertEqual(r["kinetics_type"], "Michaelis_Menten")

    def test_core_transcription_uses_hill(self):
        """转录反应使用 Hill 动力学（pSmad2:Smad4 作转录因子）。"""
        specialist = self._TgfBetaSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        tx_reactions = [
            r for r in result["reactions"]
            if r["mechanism"] == "transcription"
        ]
        # 2 条转录（PAI-1 + SMAD7）
        self.assertEqual(len(tx_reactions), 2)
        for r in tx_reactions:
            self.assertEqual(r["kinetics_type"], "Hill")
            self.assertEqual(r["hill_coefficient"], 2)


class TestTgfBetaCompositeReaction(unittest.TestCase):
    """测试 3：SMAD 复合-入核-转录三步耦合 CompositeReaction 输出。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.tgf_beta_specialist import (
            TgfBetaSpecialist,
        )

        clear_registry()
        register_specialist(TgfBetaSpecialist)
        self._TgfBetaSpecialist = TgfBetaSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_core_contains_composite_reactions(self):
        """Core 输出含 composite_reactions 字段。"""
        specialist = self._TgfBetaSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})

        self.assertIn("composite_reactions", result)
        cr_list = result["composite_reactions"]
        self.assertIsInstance(cr_list, list)
        self.assertGreater(len(cr_list), 0, "composite_reactions 不应为空")

    def test_composite_reaction_smad_cascade(self):
        """CompositeReaction 含 SMAD 复合-入核-转录三步耦合
        （CR_SMAD_COMPLEX_NUCLEAR_TRANSCRIPTION, complex_formation→nuclear_import→transcription）。"""
        specialist = self._TgfBetaSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        cr_list = result["composite_reactions"]

        # 查找 SMAD 三步耦合 CompositeReaction
        smad_cr = next(
            (cr for cr in cr_list
             if "SMAD_COMPLEX_NUCLEAR_TRANSCRIPTION" in cr.get("id", "")
             or "SMAD" in cr.get("name", "")
             or "smad" in cr.get("name", "").lower()),
            None,
        )
        self.assertIsNotNone(
            smad_cr,
            "应含 SMAD 复合-入核-转录三步耦合 CompositeReaction",
        )

        # 检查字段
        self.assertEqual(smad_cr["id"], "CR_SMAD_COMPLEX_NUCLEAR_TRANSCRIPTION")
        self.assertEqual(smad_cr["mechanism"], "sequential")
        self.assertEqual(smad_cr["coupling_type"], "sequential")

    def test_composite_reaction_intermediate_species(self):
        """CompositeReaction 中间产物 pSmad2_Smad4 / pSmad2_Smad4_nuc /
        PAI1_mRNA 完整。"""
        specialist = self._TgfBetaSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        cr_list = result["composite_reactions"]

        smad_cr = next(
            (cr for cr in cr_list
             if "SMAD_COMPLEX_NUCLEAR_TRANSCRIPTION" in cr.get("id", "")),
            None,
        )
        self.assertIsNotNone(smad_cr)

        # 检查中间产物
        intermediate_species = smad_cr.get("intermediate_species", [])
        self.assertIn("pSmad2_Smad4", intermediate_species)
        self.assertIn("pSmad2_Smad4_nuc", intermediate_species)
        self.assertIn("PAI1_mRNA", intermediate_species)

        # 检查子反应链（3 步顺序耦合）
        sub_reactions = smad_cr.get("sub_reactions", [])
        self.assertGreaterEqual(len(sub_reactions), 3)
        self.assertIn("pSmad2 → pSmad2_Smad4", sub_reactions)
        self.assertIn("pSmad2_Smad4 → pSmad2_Smad4_nuc", sub_reactions)
        self.assertIn("pSmad2_Smad4_nuc → PAI1_mRNA", sub_reactions)

    def test_composite_reaction_node_ids(self):
        """CompositeReaction node_ids 含 pSmad2 / Smad4 / pSmad2_Smad4 /
        pSmad2_Smad4_nuc / PAI1_mRNA。"""
        specialist = self._TgfBetaSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        cr_list = result["composite_reactions"]

        smad_cr = next(
            (cr for cr in cr_list
             if "SMAD_COMPLEX_NUCLEAR_TRANSCRIPTION" in cr.get("id", "")),
            None,
        )
        self.assertIsNotNone(smad_cr)

        node_ids = smad_cr.get("node_ids", [])
        self.assertIn("pSmad2", node_ids)
        self.assertIn("Smad4", node_ids)
        self.assertIn("pSmad2_Smad4", node_ids)
        self.assertIn("pSmad2_Smad4_nuc", node_ids)
        self.assertIn("PAI1_mRNA", node_ids)

    def test_composite_reaction_three_step_markers(self):
        """SMAD 三步耦合反应含 composite_step 标记（1/2/3）。"""
        specialist = self._TgfBetaSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})

        # 提取含 composite_step 标记的反应
        composite_reactions = [
            r for r in result["reactions"] if "composite_step" in r
        ]
        # 至少 3 步耦合
        self.assertGreaterEqual(
            len(composite_reactions), 3,
            "SMAD 三步耦合应含至少 3 步标记",
        )
        # 检查 1/2/3 步均存在
        steps = {r["composite_step"] for r in composite_reactions}
        self.assertIn(1, steps, "三步耦合第 1 步 complex_formation 应存在")
        self.assertIn(2, steps, "三步耦合第 2 步 nuclear_import 应存在")
        self.assertIn(3, steps, "三步耦合第 3 步 transcription 应存在")
        # 检查机制对应正确
        step1 = next(r for r in composite_reactions if r["composite_step"] == 1)
        self.assertEqual(step1["mechanism"], "complex_formation")
        step2 = next(r for r in composite_reactions if r["composite_step"] == 2)
        self.assertEqual(step2["mechanism"], "nuclear_import")
        step3 = next(r for r in composite_reactions if r["composite_step"] == 3)
        self.assertEqual(step3["mechanism"], "transcription")
        # 所有三步共享 composite_id
        composite_ids = {r["composite_id"] for r in composite_reactions}
        self.assertEqual(
            len(composite_ids), 1,
            "三步耦合应共享同一 composite_id",
        )
        self.assertEqual(
            composite_ids.pop(), "CR_SMAD_COMPLEX_NUCLEAR_TRANSCRIPTION",
        )


class TestTgfBetaStateMachine(unittest.TestCase):
    """测试 4：Smad2 状态机（cyto_Smad2→cyto_pSmad2→cyto_pSmad2_Smad4→nuc_pSmad2_Smad4）。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.tgf_beta_specialist import (
            TgfBetaSpecialist,
        )

        clear_registry()
        register_specialist(TgfBetaSpecialist)
        self._TgfBetaSpecialist = TgfBetaSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_core_contains_state_machine(self):
        """Core 输出含 state_machine 字段（Smad2 状态机）。"""
        specialist = self._TgfBetaSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})

        self.assertIn("state_machine", result)
        sm = result["state_machine"]
        self.assertIsInstance(sm, dict)
        self.assertGreater(len(sm), 0, "state_machine 不应为空")

    def test_state_machine_smad2_four_states(self):
        """Smad2 状态机含 4 状态（cyto_Smad2→cyto_pSmad2→cyto_pSmad2_Smad4→nuc_pSmad2_Smad4）。"""
        specialist = self._TgfBetaSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        sm = result["state_machine"]

        self.assertEqual(sm["id"], "SM_SMAD2")
        self.assertEqual(sm["species"], "Smad2")
        states = sm["states"]
        self.assertEqual(len(states), 4)
        state_names = [s["name"] for s in states]
        self.assertEqual(
            state_names,
            ["cyto_Smad2", "cyto_pSmad2", "cyto_pSmad2_Smad4", "nuc_pSmad2_Smad4"],
        )
        # cyto_Smad2 为初始状态
        initial = next(s for s in states if s["name"] == "cyto_Smad2")
        self.assertTrue(initial.get("is_initial"))

    def test_state_machine_transitions(self):
        """Smad2 状态机含 3 个状态转换（phosphorylation / complex_formation / nuclear_import）。"""
        specialist = self._TgfBetaSpecialist()
        result = specialist.apply_core({"nodes": [], "edges": []})
        sm = result["state_machine"]

        transitions = sm["transitions"]
        self.assertEqual(len(transitions), 3)
        # 第 1 个转换：cyto_Smad2 → cyto_pSmad2（phosphorylation）
        t1 = transitions[0]
        self.assertEqual(t1["from_state"], "cyto_Smad2")
        self.assertEqual(t1["to_state"], "cyto_pSmad2")
        self.assertEqual(t1["trigger"], "phosphorylation")
        # 第 2 个转换：cyto_pSmad2 → cyto_pSmad2_Smad4（complex_formation）
        t2 = transitions[1]
        self.assertEqual(t2["from_state"], "cyto_pSmad2")
        self.assertEqual(t2["to_state"], "cyto_pSmad2_Smad4")
        self.assertEqual(t2["trigger"], "complex_formation")
        # 第 3 个转换：cyto_pSmad2_Smad4 → nuc_pSmad2_Smad4（nuclear_import）
        t3 = transitions[2]
        self.assertEqual(t3["from_state"], "cyto_pSmad2_Smad4")
        self.assertEqual(t3["to_state"], "nuc_pSmad2_Smad4")
        self.assertEqual(t3["trigger"], "nuclear_import")


class TestTgfBetaFeedbackModule(unittest.TestCase):
    """测试 5：Feedback 模块（FL_SMAD7 delay=30min + FL_SMURF delay=60min）。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.tgf_beta_specialist import (
            TgfBetaSpecialist,
        )

        clear_registry()
        register_specialist(TgfBetaSpecialist)
        self._TgfBetaSpecialist = TgfBetaSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_apply_feedback_returns_2_loops(self):
        """apply_feedback() 返回恰好 2 条反馈环。"""
        specialist = self._TgfBetaSpecialist()
        loops = specialist.apply_feedback({"nodes": [], "edges": []})

        self.assertEqual(
            len(loops),
            2,
            f"期望恰好 2 条反馈环，实际 {len(loops)} 条",
        )

    def test_fl_smad7_delay_30min(self):
        """FL_SMAD7 转录延迟负反馈（delay=30min）。"""
        specialist = self._TgfBetaSpecialist()
        loops = specialist.apply_feedback({"nodes": [], "edges": []})

        smad7_loop = next(
            (l for l in loops if "SMAD7" in l["id"]),
            None,
        )
        self.assertIsNotNone(smad7_loop, "应含 FL_SMAD7 反馈环")
        self.assertEqual(smad7_loop["loop_type"], "negative")
        self.assertEqual(
            smad7_loop["delay_minutes"],
            30.0,
            "FL_SMAD7 延迟应为 30min（SMAD7→TβRI 转录延迟负反馈）",
        )
        # 节点含 pSmad2_Smad4_nuc / SMAD7_mRNA / SMAD7 / TbRI
        node_ids = smad7_loop["node_ids"]
        self.assertIn("pSmad2_Smad4_nuc", node_ids)
        self.assertIn("SMAD7_mRNA", node_ids)
        self.assertIn("SMAD7", node_ids)
        self.assertIn("TbRI", node_ids)

    def test_fl_smurf_delay_60min(self):
        """FL_SMURF 泛素化负反馈（delay=60min）。"""
        specialist = self._TgfBetaSpecialist()
        loops = specialist.apply_feedback({"nodes": [], "edges": []})

        smurf_loop = next(
            (l for l in loops if "SMURF" in l["id"]),
            None,
        )
        self.assertIsNotNone(smurf_loop, "应含 FL_SMURF 反馈环")
        self.assertEqual(smurf_loop["loop_type"], "negative")
        self.assertEqual(
            smurf_loop["delay_minutes"],
            60.0,
            "FL_SMURF 延迟应为 60min（SMAD→SMURF→R-SMAD 泛素化负反馈）",
        )
        # 节点含 SMAD7 / SMURF / pSmad2 / pSmad3
        node_ids = smurf_loop["node_ids"]
        self.assertIn("SMAD7", node_ids)
        self.assertIn("SMURF", node_ids)
        self.assertIn("pSmad2", node_ids)
        self.assertIn("pSmad3", node_ids)

    def test_feedback_loops_all_negative(self):
        """所有反馈环 loop_type='negative'。"""
        specialist = self._TgfBetaSpecialist()
        loops = specialist.apply_feedback({"nodes": [], "edges": []})

        for loop in loops:
            self.assertEqual(
                loop["loop_type"], "negative",
                f"反馈环 {loop['id']} 应为 negative",
            )

    def test_fl_smad7_template(self):
        """FL_SMAD7 标记 transcription_factor.j2 模板。"""
        specialist = self._TgfBetaSpecialist()
        loops = specialist.apply_feedback({"nodes": [], "edges": []})

        smad7_loop = next(
            (l for l in loops if "SMAD7" in l["id"]),
            None,
        )
        self.assertIsNotNone(smad7_loop)
        self.assertEqual(
            smad7_loop.get("template"), "transcription_factor.j2",
        )


class TestTgfBetaCrosstalkModule(unittest.TestCase):
    """测试 6：Crosstalk 模块返回 4 条描述性 cross-talk edge。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.tgf_beta_specialist import (
            TgfBetaSpecialist,
        )

        clear_registry()
        register_specialist(TgfBetaSpecialist)
        self._TgfBetaSpecialist = TgfBetaSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_apply_crosstalk_returns_4_edges(self):
        """apply_crosstalk() 返回恰好 4 条描述性 cross-talk edge。"""
        specialist = self._TgfBetaSpecialist()
        fragments = specialist.apply_crosstalk(
            {"nodes": [], "edges": []}, []
        )

        self.assertEqual(
            len(fragments),
            4,
            f"期望恰好 4 条 cross-talk edge，实际 {len(fragments)} 条",
        )

    def test_crosstalk_contains_p15_p21(self):
        """含 TGF-β → p15/p21 片段（细胞周期抑制, shared_species=["p21","p15"]）。"""
        specialist = self._TgfBetaSpecialist()
        fragments = specialist.apply_crosstalk(
            {"nodes": [], "edges": []}, []
        )

        p15_p21 = next(
            (f for f in fragments if f["target"] == "p15_p21"),
            None,
        )
        self.assertIsNotNone(p15_p21)
        self.assertEqual(p15_p21["mechanism"], "transcription")
        self.assertIn("p21", p15_p21["shared_species"])
        self.assertIn("p15", p15_p21["shared_species"])

    def test_crosstalk_contains_bim_puma(self):
        """含 TGF-β → Bim/PUMA 片段（凋亡促进, shared_species=["Bim","PUMA"]）。"""
        specialist = self._TgfBetaSpecialist()
        fragments = specialist.apply_crosstalk(
            {"nodes": [], "edges": []}, []
        )

        bim_puma = next(
            (f for f in fragments if f["target"] == "Bim_PUMA"),
            None,
        )
        self.assertIsNotNone(bim_puma)
        self.assertEqual(bim_puma["mechanism"], "transcription")
        self.assertIn("Bim", bim_puma["shared_species"])
        self.assertIn("PUMA", bim_puma["shared_species"])

    def test_crosstalk_contains_pi3k_akt(self):
        """含 TGF-β ↔ PI3K-AKT 片段（双向 cross-talk, pAKT→Smad3 linker）。"""
        specialist = self._TgfBetaSpecialist()
        fragments = specialist.apply_crosstalk(
            {"nodes": [], "edges": []}, []
        )

        akt = next(
            (f for f in fragments
             if f["source"] == "pAKT" and f["target"] == "Smad3"),
            None,
        )
        self.assertIsNotNone(akt)
        self.assertEqual(akt["mechanism"], "phosphorylation")
        self.assertIn("AKT", akt["shared_species"])

    def test_crosstalk_contains_erk_smad(self):
        """含 ERK → Smad linker 片段（MAPK 旁路磷酸化 Smad3 linker）。"""
        specialist = self._TgfBetaSpecialist()
        fragments = specialist.apply_crosstalk(
            {"nodes": [], "edges": []}, []
        )

        erk = next(
            (f for f in fragments
             if f["source"] == "pERK" and f["target"] == "Smad3"),
            None,
        )
        self.assertIsNotNone(erk)
        self.assertEqual(erk["mechanism"], "phosphorylation")

    def test_crosstalk_shared_species_complete(self):
        """CrosstalkModuleData shared_species 含 p21/p15/Bim/PUMA。"""
        specialist = self._TgfBetaSpecialist()
        from app.pathways.pathway_modules.crosstalk.template import (
            CrosstalkModuleData,
        )
        from app.pathways.pathway_specialist_base import MODULE_CROSSTALK

        data = specialist.load_module(MODULE_CROSSTALK)
        self.assertIsInstance(data, CrosstalkModuleData)
        self.assertIn("p21", data.shared_species)
        self.assertIn("p15", data.shared_species)
        self.assertIn("Bim", data.shared_species)
        self.assertIn("PUMA", data.shared_species)

    def test_crosstalk_coordination_strategy(self):
        """CrosstalkModuleData coordination_strategy='delegate_to_coordinator'。"""
        specialist = self._TgfBetaSpecialist()
        from app.pathways.pathway_specialist_base import MODULE_CROSSTALK

        data = specialist.load_module(MODULE_CROSSTALK)
        self.assertEqual(
            data.coordination_strategy, "delegate_to_coordinator",
        )

    def test_crosstalk_fragments_contain_required_fields(self):
        """每条 cross-talk 片段含 source / target / mechanism / shared_species /
        description。"""
        specialist = self._TgfBetaSpecialist()
        fragments = specialist.apply_crosstalk(
            {"nodes": [], "edges": []}, []
        )

        for f in fragments:
            self.assertIn("source", f)
            self.assertIn("target", f)
            self.assertIn("mechanism", f)
            self.assertIn("shared_species", f)
            self.assertIn("description", f)


class TestTgfBetaPerturbationModule(unittest.TestCase):
    """测试 7：Perturbation 模块返回 3 个扰动（2 药物 + 1 KO）。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.tgf_beta_specialist import (
            TgfBetaSpecialist,
        )

        clear_registry()
        register_specialist(TgfBetaSpecialist)
        self._TgfBetaSpecialist = TgfBetaSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_apply_perturbation_returns_3(self):
        """apply_perturbation() 返回恰好 3 个扰动。"""
        specialist = self._TgfBetaSpecialist()
        perturbations = specialist.apply_perturbation(
            {"nodes": [], "edges": []}, []
        )

        self.assertEqual(
            len(perturbations),
            3,
            f"期望恰好 3 个扰动，实际 {len(perturbations)} 个",
        )

    def test_perturbation_contains_galunisertib(self):
        """含 Galunisertib（TβRI kinase inhibitor, 小分子）。"""
        specialist = self._TgfBetaSpecialist()
        perturbations = specialist.apply_perturbation(
            {"nodes": [], "edges": []}, []
        )

        gal = next(
            (p for p in perturbations if p.get("drug") == "Galunisertib"),
            None,
        )
        self.assertIsNotNone(gal)
        self.assertEqual(gal["target"], "TbRI")
        self.assertEqual(gal["mechanism"], "inhibition")

    def test_perturbation_contains_sb431542(self):
        """含 SB431542（TβRI kinase inhibitor, 小分子）。"""
        specialist = self._TgfBetaSpecialist()
        perturbations = specialist.apply_perturbation(
            {"nodes": [], "edges": []}, []
        )

        sb = next(
            (p for p in perturbations if p.get("drug") == "SB431542"),
            None,
        )
        self.assertIsNotNone(sb)
        self.assertEqual(sb["target"], "TbRI")
        self.assertEqual(sb["mechanism"], "inhibition")

    def test_perturbation_contains_smad4_loss(self):
        """含 SMAD4 loss（loss-of-function, 基因缺失）。"""
        specialist = self._TgfBetaSpecialist()
        perturbations = specialist.apply_perturbation(
            {"nodes": [], "edges": []}, []
        )

        smad4 = next(
            (p for p in perturbations if p.get("ko_target") == "SMAD4_loss"),
            None,
        )
        self.assertIsNotNone(smad4)
        self.assertEqual(smad4["target"], "Smad4")
        self.assertEqual(smad4["mechanism"], "knockout")

    def test_perturbation_drug_targets_count(self):
        """PerturbationModuleData drug_targets 含 2 个药物靶点。"""
        specialist = self._TgfBetaSpecialist()
        from app.pathways.pathway_specialist_base import MODULE_PERTURBATION

        data = specialist.load_module(MODULE_PERTURBATION)
        self.assertEqual(len(data.drug_targets), 2)
        drug_names = [p["drug"] for p in data.drug_targets]
        self.assertIn("Galunisertib", drug_names)
        self.assertIn("SB431542", drug_names)

    def test_perturbation_ko_targets_count(self):
        """PerturbationModuleData ko_targets 含 1 个 KO 靶点。"""
        specialist = self._TgfBetaSpecialist()
        from app.pathways.pathway_specialist_base import MODULE_PERTURBATION

        data = specialist.load_module(MODULE_PERTURBATION)
        self.assertEqual(len(data.ko_targets), 1)
        self.assertEqual(data.ko_targets[0]["ko_target"], "SMAD4_loss")


class TestTgfBetaValidationModule(unittest.TestCase):
    """测试 8：Validation 模块返回 3 条 benchmark。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.tgf_beta_specialist import (
            TgfBetaSpecialist,
        )

        clear_registry()
        register_specialist(TgfBetaSpecialist)
        self._TgfBetaSpecialist = TgfBetaSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_apply_validation_returns_3_rules(self):
        """apply_validation() 返回恰好 3 条 Validation 规则。"""
        specialist = self._TgfBetaSpecialist()
        rules = specialist.apply_validation()

        self.assertEqual(
            len(rules),
            3,
            f"期望恰好 3 条 Validation 规则，实际 {len(rules)} 条",
        )

    def test_validation_psmad2_peak_time(self):
        """pSmad2 达峰时间 5-15 min（Massagué 1998, PMID:9674480）。"""
        specialist = self._TgfBetaSpecialist()
        rules = specialist.apply_validation()

        rule = next(
            (r for r in rules if r["metric_name"] == "pSmad2_peak_time"),
            None,
        )
        self.assertIsNotNone(rule)
        self.assertEqual(rule["expected_min"], 5.0)
        self.assertEqual(rule["expected_max"], 15.0)
        self.assertEqual(rule["expected"], 10.0)
        self.assertEqual(rule["tolerance"], 5.0)
        self.assertEqual(rule["unit"], "minutes")
        self.assertEqual(rule["pmid"], "PMID:9674480")

    def test_validation_psmad2_smad4_nuclear(self):
        """pSmad2-Smad4 核累积 15-30 min（Schmierer 2007, PMID:17721552）。"""
        specialist = self._TgfBetaSpecialist()
        rules = specialist.apply_validation()

        rule = next(
            (r for r in rules
             if r["metric_name"] == "pSmad2_Smad4_nuclear_accumulation_time"),
            None,
        )
        self.assertIsNotNone(rule)
        self.assertEqual(rule["expected_min"], 15.0)
        self.assertEqual(rule["expected_max"], 30.0)
        self.assertEqual(rule["expected"], 22.5)
        self.assertEqual(rule["tolerance"], 7.5)
        self.assertEqual(rule["unit"], "minutes")
        self.assertEqual(rule["pmid"], "PMID:17721552")

    def test_validation_smad7_mrna_delay(self):
        """SMAD7 mRNA 延迟 30-60 min（Massagué 1998, PMID:9674480）。"""
        specialist = self._TgfBetaSpecialist()
        rules = specialist.apply_validation()

        rule = next(
            (r for r in rules if r["metric_name"] == "SMAD7_mRNA_delay"),
            None,
        )
        self.assertIsNotNone(rule)
        self.assertEqual(rule["expected_min"], 30.0)
        self.assertEqual(rule["expected_max"], 60.0)
        self.assertEqual(rule["expected"], 45.0)
        self.assertEqual(rule["tolerance"], 15.0)
        self.assertEqual(rule["unit"], "minutes")
        self.assertEqual(rule["pmid"], "PMID:9674480")

    def test_validation_tolerances(self):
        """ValidationModuleData tolerances 含 3 个 metric 的 tolerance。"""
        specialist = self._TgfBetaSpecialist()
        from app.pathways.pathway_specialist_base import MODULE_VALIDATION

        data = specialist.load_module(MODULE_VALIDATION)
        self.assertEqual(len(data.tolerances), 3)
        self.assertEqual(data.tolerances["pSmad2_peak_time"], 5.0)
        self.assertEqual(
            data.tolerances["pSmad2_Smad4_nuclear_accumulation_time"], 7.5,
        )
        self.assertEqual(data.tolerances["SMAD7_mRNA_delay"], 15.0)

    def test_validation_pmid_references_non_empty(self):
        """ValidationModuleData pmid_references 非空（含 3 条 PMID 引用）。"""
        specialist = self._TgfBetaSpecialist()
        from app.pathways.pathway_specialist_base import MODULE_VALIDATION

        data = specialist.load_module(MODULE_VALIDATION)
        self.assertEqual(len(data.pmid_references), 3)
        for ref in data.pmid_references:
            self.assertIn("pmid", ref)
            self.assertIn("citation", ref)
            self.assertEqual(ref["pathway_class"], "TGF_BETA")

    def test_validation_benchmarks_count(self):
        """ValidationModuleData benchmarks 含 3 条 benchmark。"""
        specialist = self._TgfBetaSpecialist()
        from app.pathways.pathway_specialist_base import MODULE_VALIDATION

        data = specialist.load_module(MODULE_VALIDATION)
        self.assertEqual(len(data.benchmarks), 3)
        for b in data.benchmarks:
            self.assertIn("benchmark_name", b)
            self.assertIn("value", b)
            self.assertEqual(b["condition"], "TGF-β stimulation")


class TestTgfBetaTemplateSelection(unittest.TestCase):
    """测试 9：模板选择（phosphorylation / transcription）。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.tgf_beta_specialist import (
            TgfBetaSpecialist,
        )

        clear_registry()
        register_specialist(TgfBetaSpecialist)
        self._TgfBetaSpecialist = TgfBetaSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_select_template_phosphorylation(self):
        """select_template('phosphorylation') → '_mechanism_phosphorylation_mm'。"""
        specialist = self._TgfBetaSpecialist()
        template = specialist.select_template("phosphorylation")

        self.assertEqual(template, "_mechanism_phosphorylation_mm")

    def test_select_template_transcription(self):
        """select_template('transcription') → 'transcription_factor'
        （规范名称, 由 ODE Renderer 决定 fallback 降级到 _mechanism_phosphorylation_mm）。"""
        specialist = self._TgfBetaSpecialist()
        template = specialist.select_template("transcription")

        # 返回规范名称 transcription_factor（不含 .j2 后缀, 与基类契约一致）
        # P3 transcription_factor.j2 未实现时由 ODE Renderer 降级到
        # _mechanism_phosphorylation_mm（与 JAK-STAT Specialist 处理方式一致）
        self.assertEqual(template, "transcription_factor")

    def test_select_template_default(self):
        """未匹配的 mechanism 走基类默认映射（如 'bistable' → 'bistable_switch'）。"""
        specialist = self._TgfBetaSpecialist()
        template = specialist.select_template("bistable")

        self.assertEqual(template, "bistable_switch")


class TestTgfBetaFeatureFlagIsolation(unittest.TestCase):
    """测试 10：Feature Flag 隔离（flag=false 时 hook 不调用 Specialist）。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.tgf_beta_specialist import (
            TgfBetaSpecialist,
        )

        clear_registry()
        register_specialist(TgfBetaSpecialist)
        self._TgfBetaSpecialist = TgfBetaSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_flag_false_hook_returns_empty(self):
        """flag=false 时 hook 返回空 dict 且不调用 Specialist 任何方法。"""
        specialist = MagicMock()
        specialist.apply_core.return_value = {
            "species": [],
            "reactions": [],
            "composite_reactions": [],
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
            "species": [{"name": "pSmad2"}],
            "reactions": [],
            "composite_reactions": [],
            "state_machine": {"id": "SM_SMAD2"},
        }
        specialist.apply_core.return_value = expected_core
        state = {"v4_pathway_graph": {"nodes": [{"id": "pSmad2"}], "edges": []}}

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


class TestTgfBetaMetadataAndInputValidation(unittest.TestCase):
    """测试 11：元数据 + 输入校验 + load_module。"""

    def setUp(self):
        from app.pathways.pathway_registry import (
            clear_registry,
            register_specialist,
        )
        from app.pathways.specialists.tgf_beta_specialist import (
            TgfBetaSpecialist,
        )

        clear_registry()
        register_specialist(TgfBetaSpecialist)
        self._TgfBetaSpecialist = TgfBetaSpecialist

    def tearDown(self):
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_get_metadata_returns_tgf_beta_metadata(self):
        """get_metadata() 返回 pathway_class='TGF_BETA' + 5 模块。"""
        specialist = self._TgfBetaSpecialist()
        metadata = specialist.get_metadata()

        self.assertEqual(metadata["pathway_class"], "TGF_BETA")
        self.assertEqual(
            metadata["display_name"],
            "TGF-β Signaling",
        )
        self.assertEqual(
            metadata["supported_modules"],
            ["core", "feedback", "crosstalk", "perturbation", "validation"],
        )
        self.assertEqual(len(metadata["supported_modules"]), 5)
        self.assertIn("version", metadata)

    def test_validate_input_empty_graph_returns_warnings(self):
        """validate_input({}) 返回非空 warning 列表。"""
        specialist = self._TgfBetaSpecialist()
        warnings = specialist.validate_input({})

        self.assertIsInstance(warnings, list)
        self.assertGreater(len(warnings), 0, "空 pathway_graph 应返回 warning")

    def test_validate_input_none_returns_warnings(self):
        """validate_input(None) 返回非空 warning 列表。"""
        specialist = self._TgfBetaSpecialist()
        warnings = specialist.validate_input(None)

        self.assertGreater(len(warnings), 0)

    def test_validate_input_valid_graph_returns_no_warnings(self):
        """含 nodes + edges 的 pathway_graph 无 warning。"""
        specialist = self._TgfBetaSpecialist()
        warnings = specialist.validate_input(
            {"nodes": [{"id": "pSmad2"}], "edges": []}
        )
        self.assertEqual(warnings, [])

    def test_load_module_core_returns_core_data(self):
        """load_module('core') 返回 CoreModuleData 实例。"""
        from app.pathways.pathway_modules.core.template import CoreModuleData
        from app.pathways.pathway_specialist_base import MODULE_CORE

        specialist = self._TgfBetaSpecialist()
        data = specialist.load_module(MODULE_CORE)

        self.assertIsInstance(data, CoreModuleData)
        self.assertGreater(len(data.species), 0)
        self.assertEqual(len(data.reactions), 11)

    def test_load_module_feedback_returns_feedback_data(self):
        """load_module('feedback') 返回 FeedbackModuleData 实例（delay=30min）。"""
        from app.pathways.pathway_modules.feedback.template import (
            FeedbackModuleData,
        )
        from app.pathways.pathway_specialist_base import MODULE_FEEDBACK

        specialist = self._TgfBetaSpecialist()
        data = specialist.load_module(MODULE_FEEDBACK)

        self.assertIsInstance(data, FeedbackModuleData)
        self.assertEqual(len(data.feedback_loops), 2)
        self.assertEqual(data.delay_minutes, 30.0)
        self.assertEqual(data.loop_type, "negative")

    def test_load_module_unknown_returns_none(self):
        """load_module('unknown') 返回 None。"""
        specialist = self._TgfBetaSpecialist()
        data = specialist.load_module("unknown")

        self.assertIsNone(data)


if __name__ == "__main__":
    unittest.main()
