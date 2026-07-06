# BioDynamics Agent v4 - Cross-talk Coordinator 集成测试 (Phase 4 / Task 4.13.8)
#
# 测试用例（≥6 集成测试）：
#   TestMultiPathwayE2E:
#     1. test_egfr_pi3k_shared_ras_akt: EGF+PI3K 双通路 shared Ras/AKT 同步
#     2. test_egfr_pi3k_crosstalk_edges: cross-talk edges 正确注入
#     3. test_egfr_pi3k_pathway_tag_isolation: CROSSTALK_EGFR_RTK_PI3K_AKT_mTOR 标记
#     4. test_egfr_pi3k_time_scale_alignment: max_step 取最小值
#     5. test_single_pathway_no_coordination: 单通路场景返回空
#     6. test_feature_flag_off_no_op: flag=false 时 hook 返回 {}
#
# 重要：集成测试 mock Specialist 输出（不实际调用 LLM）
# Mock 数据基于真实 EGFR / PI3K Specialist 的 apply_crosstalk / apply_core 输出格式
#
# 运行：cd backend && python -m pytest tests/test_multi_pathway_e2e.py -v

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# 确保能导入 app 包（backend/ 在 sys.path）
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


# =============================================================================
# Mock Specialist 输出工厂
# =============================================================================
# 基于 EGFRSpecialist / PI3KAKTmTORSpecialist 的 apply_crosstalk + apply_core 真实输出格式构造
# 不实际调用 LLM，仅模拟 Specialist 的静态输出

def _mock_egfr_specialist_output() -> dict:
    """Mock EGFR Specialist 输出（基于 EGFRSpecialist.apply_crosstalk + apply_core）。

    EGFR Specialist apply_crosstalk 返回 3 条 cross-talk Reaction 片段：
      1. pEGFR → PI3K（activation）
      2. pERK → ELK1 → Fos（transcription）
      3. AKT → Raf Ser259（inhibition）

    load_module(MODULE_CROSSTALK) 返回 CrosstalkModuleData:
      shared_species=["Ras"], coordination_strategy="merge"

    apply_core 返回 7 条核心反应 + 11 物种（Ras 标记 shared=True）
    """
    return {
        "pathway_class": "EGFR_RTK",
        "shared_species": ["Ras"],
        "coordination_strategy": "merge",
        "species": [
            {"name": "EGF", "species_type": "ligand", "compartment": "extracellular"},
            {"name": "EGFR", "species_type": "protein", "compartment": "membrane"},
            {"name": "pEGFR", "species_type": "protein", "compartment": "membrane"},
            {"name": "Ras", "species_type": "protein", "compartment": "membrane",
             "shared": True},
            {"name": "RasGTP", "species_type": "protein", "compartment": "membrane"},
            {"name": "Raf", "species_type": "protein", "compartment": "cytoplasm"},
        ],
        "reactions": [
            {
                "id": "R_EGF_EGFR_BIND",
                "source": "EGF",
                "target": "EGFR",
                "mechanism": "binding",
                "pathway_tag": "EGFR_RTK",
                "substrate": "EGF",
                "product": "EGFR",
                "modifier": None,
            },
            {
                "id": "R_SOS_RasGTP",
                "source": "SOS",
                "target": "RasGTP",
                "mechanism": "gtp_gdp_exchange",
                "pathway_tag": "EGFR_RTK",
                "substrate": "Ras",
                "product": "RasGTP",
                "modifier": "SOS",
            },
        ],
        "crosstalk_reactions": [
            {
                "source": "pEGFR",
                "target": "PI3K",
                "mechanism": "activation",
                "shared_species": [],
                "description": "pEGFR 直接磷酸化 PI3K（激活 PI3K-AKT-mTOR 通路）",
            },
            {
                "source": "AKT",
                "target": "Raf",
                "mechanism": "inhibition",
                "shared_species": [],
                "site": "Ser259",
                "description": "AKT 磷酸化 Raf Ser259 抑制 MAPK（PI3K→MAPK cross-talk）",
            },
        ],
        "crosstalk_edges": [
            {
                "id": "CT_EGFR_TO_PI3K",
                "source_pathway": "EGFR_RTK",
                "target_pathway": "PI3K_AKT_mTOR",
                "source_node": "PN_pEGFR",
                "target_node": "PN_PI3K",
                "mechanism": "activation",
                "shared_species": [],
                "description": "pEGFR 直接磷酸化 PI3K",
            },
        ],
        "kinetics_overrides": {"k_egfr_int": 0.01},
        "max_step": 0.1,
        "time_scale": "fast",
        "t_end": 60.0,
    }


def _mock_pi3k_specialist_output() -> dict:
    """Mock PI3K Specialist 输出（基于 PI3KAKTmTORSpecialist.apply_crosstalk + apply_core）。

    PI3K Specialist apply_crosstalk 返回 4 条 cross-talk Reaction 片段：
      1. pAKT → Raf Ser259（inhibition）
      2. pAKT → Bad Ser136（inhibition）
      3. pAKT → Mdm2 Ser166（activation）
      4. mTORC1 → HIF-1α（activation）

    load_module(MODULE_CROSSTALK) 返回 CrosstalkModuleData:
      shared_species=["AKT"], coordination_strategy="merge"

    apply_core 返回 9 条核心反应 + 16 物种（AKT 标记 shared=True，mTORC1 标记 shared=True）
    """
    return {
        "pathway_class": "PI3K_AKT_mTOR",
        "shared_species": ["AKT"],
        "coordination_strategy": "merge",
        "species": [
            {"name": "PI3K", "species_type": "protein", "compartment": "cytoplasm"},
            {"name": "PIP3", "species_type": "chemical", "compartment": "membrane"},
            {"name": "AKT", "species_type": "protein", "compartment": "cytoplasm",
             "shared": True},
            {"name": "pAKT", "species_type": "protein", "compartment": "cytoplasm"},
            {"name": "mTORC1", "species_type": "complex", "compartment": "cytoplasm",
             "shared": True},
            {"name": "TSC2", "species_type": "protein", "compartment": "cytoplasm"},
        ],
        "reactions": [
            {
                "id": "R_PI3K_PIP3",
                "source": "PI3K",
                "target": "PIP3",
                "mechanism": "activation",
                "pathway_tag": "PI3K_AKT_mTOR",
                "substrate": "PIP2",
                "product": "PIP3",
                "modifier": "PI3K",
            },
            {
                "id": "R_PIP3_pAKT",
                "source": "PIP3",
                "target": "pAKT",
                "mechanism": "phosphorylation",
                "pathway_tag": "PI3K_AKT_mTOR",
                "substrate": "AKT",
                "product": "pAKT",
                "modifier": "PIP3",
            },
        ],
        "crosstalk_reactions": [
            {
                "source": "pAKT",
                "target": "Raf",
                "mechanism": "inhibition",
                "shared_species": ["AKT"],
                "site": "Ser259",
                "description": "pAKT 磷酸化 Raf Ser259 抑制 MAPK 级联（PI3K→MAPK cross-talk）",
            },
            {
                "source": "pAKT",
                "target": "Bad",
                "mechanism": "inhibition",
                "shared_species": ["AKT"],
                "site": "Ser136",
                "description": "pAKT 磷酸化 Bad Ser136 导致其失活（抑制凋亡）",
            },
            {
                "source": "pAKT",
                "target": "Mdm2",
                "mechanism": "activation",
                "shared_species": ["AKT"],
                "site": "Ser166",
                "description": "pAKT 磷酸化 Mdm2 Ser166 激活其 E3 泛素连接酶活性",
            },
        ],
        "crosstalk_edges": [
            {
                "id": "CT_PI3K_TO_MAPK_RAF",
                "source_pathway": "PI3K_AKT_mTOR",
                "target_pathway": "MAPK_ERK",
                "source_node": "PN_pAKT",
                "target_node": "PN_Raf",
                "mechanism": "inhibition",
                "site": "Ser259",
                "description": "AKT 磷酸化 Raf Ser259 抑制 MAPK",
            },
        ],
        "kinetics_overrides": {"k_akt_phos": 0.05},
        "max_step": 1.0,
        "time_scale": "medium",
        "t_end": 120.0,
    }


def _mock_egfr_pi3k_state() -> dict:
    """构造 LangGraph state（含 v4_pathway_class + v4_specialist_outputs）。"""
    return {
        "v4_pathway_class": "MULTI:EGFR_RTK+PI3K_AKT_mTOR",
        "v4_specialist_outputs": [
            _mock_egfr_specialist_output(),
            _mock_pi3k_specialist_output(),
        ],
    }


# =============================================================================
# TestMultiPathwayE2E: EGF + PI3K 双通路集成测试
# =============================================================================
class TestMultiPathwayE2E(unittest.TestCase):
    """EGF + PI3K 双通路 Cross-talk Coordinator 端到端集成测试。

    验证 EGFR + PI3K 双通路场景下：
    - shared Ras/AKT 正确同步
    - cross-talk edges 注入正确
    - CROSSTALK_EGFR_RTK_PI3K_AKT_mTOR pathway_tag 标记
    - max_step 取最小值
    """

    def test_egfr_pi3k_shared_ras_akt(self):
        """EGFR + PI3K 双通路：验证 shared_species 包含 Ras 和 AKT。

        EGFR Specialist 标记 Ras 为 shared（species.shared=True），
        PI3K Specialist 标记 AKT 为 shared（species.shared=True），
        且 mTORC1 在 PI3K 中也标记 shared=True。
        """
        from app.crosstalk.coordinator import CrossTalkCoordinator
        coord = CrossTalkCoordinator()
        result = coord.coordinate(
            [_mock_egfr_specialist_output(), _mock_pi3k_specialist_output()],
            "MULTI:EGFR_RTK+PI3K_AKT_mTOR",
        )
        shared = result["v4_shared_species"]
        # AKT 在 PI3K species 中标记 shared=True，应被识别
        self.assertIn("AKT", shared)
        # Ras 在 EGFR species 中标记 shared=True，应被识别
        self.assertIn("Ras", shared)

    def test_egfr_pi3k_crosstalk_edges(self):
        """EGFR + PI3K 双通路：验证 cross-talk edges 正确注入。

        EGFR Specialist 提供 CT_EGFR_TO_PI3K edge，
        PI3K Specialist 提供 CT_PI3K_TO_MAPK_RAF edge，
        应全部被 Coordinator 收集并去重。
        """
        from app.crosstalk.coordinator import CrossTalkCoordinator
        coord = CrossTalkCoordinator()
        result = coord.coordinate(
            [_mock_egfr_specialist_output(), _mock_pi3k_specialist_output()],
            "MULTI:EGFR_RTK+PI3K_AKT_mTOR",
        )
        edges = result["v4_crosstalk_edges"]
        edge_ids = [e["id"] for e in edges]
        # 预识别 edges 应被收集
        self.assertIn("CT_EGFR_TO_PI3K", edge_ids)
        self.assertIn("CT_PI3K_TO_MAPK_RAF", edge_ids)
        # 每条 edge 含必填字段
        for edge in edges:
            self.assertIn("source_pathway", edge)
            self.assertIn("target_pathway", edge)
            self.assertIn("source_node", edge)
            self.assertIn("target_node", edge)
            self.assertIn("mechanism", edge)

    def test_egfr_pi3k_pathway_tag_isolation(self):
        """EGFR + PI3K 双通路：验证 CROSSTALK_EGFR_RTK_PI3K_AKT_mTOR 标记。

        cross-talk 相关参数应被标记为 CROSSTALK_<source>_<target>，
        防止 cross-pathway parameter contamination。
        """
        from app.crosstalk.coordinator import CrossTalkCoordinator
        coord = CrossTalkCoordinator()
        result = coord.coordinate(
            [_mock_egfr_specialist_output(), _mock_pi3k_specialist_output()],
            "MULTI:EGFR_RTK+PI3K_AKT_mTOR",
        )
        isolation = result["v4_pathway_tag_isolation"]
        tagged = isolation["tagged_parameters"]
        # 应有 cross-talk 参数标记（含 CROSSTALK_ 前缀）
        crosstalk_tags = [
            t["pathway_tag"] for t in tagged
            if t.get("is_crosstalk") and t["pathway_tag"].startswith("CROSSTALK_")
        ]
        self.assertTrue(len(crosstalk_tags) > 0)
        # 应包含 EGFR_RTK 或 PI3K_AKT_mTOR 之一的 cross-talk 标记
        has_egfr_crosstalk = any("EGFR_RTK" in t for t in crosstalk_tags)
        has_pi3k_crosstalk = any("PI3K_AKT_mTOR" in t for t in crosstalk_tags)
        self.assertTrue(has_egfr_crosstalk or has_pi3k_crosstalk)

    def test_egfr_pi3k_time_scale_alignment(self):
        """EGFR + PI3K 双通路：验证 max_step 取最小值。

        EGFR max_step=0.1（fast），PI3K max_step=1.0（medium），
        unified_max_step 应为 min(0.1, 1.0) = 0.1。
        """
        from app.crosstalk.coordinator import CrossTalkCoordinator
        coord = CrossTalkCoordinator()
        result = coord.coordinate(
            [_mock_egfr_specialist_output(), _mock_pi3k_specialist_output()],
            "MULTI:EGFR_RTK+PI3K_AKT_mTOR",
        )
        time_align = result["v4_time_scale_alignment"]
        self.assertEqual(time_align["unified_max_step"], 0.1)
        self.assertEqual(time_align["alignment_strategy"], "min_of_all")
        # 每通路时间尺度被记录
        scales = time_align["pathway_time_scales"]
        self.assertEqual(len(scales), 2)

    def test_single_pathway_no_coordination(self):
        """单通路场景（EGFR only）：Coordinator 返回空列表。

        pathway_class 不含 "MULTI:" 时，Coordinator 不执行合并，
        返回空 crosstalk_edges + 空 shared_species。
        """
        from app.crosstalk.coordinator import CrossTalkCoordinator
        coord = CrossTalkCoordinator()
        result = coord.coordinate(
            [_mock_egfr_specialist_output()],
            "EGFR_RTK",
        )
        self.assertEqual(result["v4_crosstalk_edges"], [])
        self.assertEqual(result["v4_shared_species"], [])
        self.assertEqual(result["v4_shared_species_sync"], {})

    def test_feature_flag_off_no_op(self):
        """Feature Flag=false 时 hook 返回 {}（不执行任何逻辑）。"""
        from app.crosstalk.coordinator import crosstalk_coordinator_hook_node
        with patch("app.config.settings") as mock_settings:
            mock_settings.V4_CROSSTALK_COORDINATOR_ENABLED = False
            state = _mock_egfr_pi3k_state()
            result = crosstalk_coordinator_hook_node(state)
            self.assertEqual(result, {})

    def test_feature_flag_on_multi_pathway_full_flow(self):
        """Feature Flag=true + 多通路：hook 完整流程执行（含所有 v4 字段）。"""
        from app.crosstalk.coordinator import crosstalk_coordinator_hook_node
        with patch("app.config.settings") as mock_settings:
            mock_settings.V4_CROSSTALK_COORDINATOR_ENABLED = True
            state = _mock_egfr_pi3k_state()
            result = crosstalk_coordinator_hook_node(state)
            # 应返回所有 v4 字段
            self.assertIn("v4_crosstalk_edges", result)
            self.assertIn("v4_shared_species", result)
            self.assertIn("v4_shared_species_sync", result)
            self.assertIn("v4_time_scale_alignment", result)
            # shared species 非空
            self.assertTrue(len(result["v4_shared_species"]) > 0)
            # crosstalk edges 非空
            self.assertTrue(len(result["v4_crosstalk_edges"]) > 0)

    def test_egfr_pi3k_shared_species_sync_strategy(self):
        """EGFR + PI3K 双通路：验证 shared species 同步策略。

        AKT 的主导通路应为 PI3K_AKT_mTOR（AKT 在 PI3K 中作为 product 被 produced）。
        """
        from app.crosstalk.coordinator import CrossTalkCoordinator
        coord = CrossTalkCoordinator()
        result = coord.coordinate(
            [_mock_egfr_specialist_output(), _mock_pi3k_specialist_output()],
            "MULTI:EGFR_RTK+PI3K_AKT_mTOR",
        )
        sync = result["v4_shared_species_sync"]
        # sync_map 应将 AKT 映射到同名 canonical 变量
        self.assertEqual(sync["sync_map"].get("AKT"), "AKT")
        # AKT 主导通路应为 PI3K_AKT_mTOR（AKT 在 PI3K 反应中作为 substrate/product）
        self.assertEqual(
            sync["pathway_assignments"].get("AKT"), "PI3K_AKT_mTOR"
        )


if __name__ == "__main__":
    unittest.main()
