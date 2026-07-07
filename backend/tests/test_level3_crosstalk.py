# BioDynamics Agent v4 - Level 3 Cross-Pathway Validation 单元测试
# (Phase 5 / Task 5.4.6)
#
# 测试 Level3CrossPathwayValidator 主类 + 3 个检查函数 + 单通路 skipped + LangGraph hook。
#
# 测试用例（28 个，>= 20 要求）：
#   - TestLevel3CrossPathwayValidator: validate() 主入口 + 异常降级
#   - TestLevel3HookNode: Feature Flag + hook 行为
#   - TestCrosstalkConsistency: cross-talk edges 一致 / 不一致 / 空
#   - TestSharedSpeciesConservation: 守恒 / 不守恒 / EGF+PI3K Ras / 空
#   - TestTimeScaleAlignment: 时间尺度对齐 / 未对齐 / 空
#   - TestSinglePathwaySkipped: 单通路 skipped pass=True / 多通路 / UNKNOWN
#
# 运行：cd backend && python -m pytest tests/test_level3_crosstalk.py -v

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# 确保能导入 app 包（backend/ 在 sys.path）
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.validation_v2.level3_crosstalk import (
    Level3CrossPathwayValidator,
    level3_hook_node,
)


# =============================================================================
# Mock 数据
# =============================================================================

# --- 多通路 pathway_class ---
MOCK_MULTI_PATHWAY = "MULTI:EGFR_RTK+PI3K_AKT_mTOR"

# --- 单通路 pathway_class ---
MOCK_SINGLE_PATHWAY = "EGFR_RTK"

# --- Mock cross-talk edges（EGFR→PI3K，pEGFR 激活 PI3K）---
MOCK_CROSSTALK_EDGES_CONSISTENT = [
    {
        "id": "CT_EGFR_PI3K_1",
        "source_pathway": "EGFR_RTK",
        "target_pathway": "PI3K_AKT_mTOR",
        "source_node": "pEGFR",
        "target_node": "PI3K",
        "mechanism": "activation",
        "shared_species": ["pEGFR", "PI3K"],
        "description": "pEGFR activates PI3K",
    },
]

# --- Mock cross-talk edges（不一致：target_node 不在 target_pathway species 中）---
MOCK_CROSSTALK_EDGES_INCONSISTENT = [
    {
        "id": "CT_EGFR_PI3K_2",
        "source_pathway": "EGFR_RTK",
        "target_pathway": "PI3K_AKT_mTOR",
        "source_node": "pEGFR",
        "target_node": "PI3K_ghost",  # 不存在
        "mechanism": "activation",
        "shared_species": ["pEGFR"],
        "description": "pEGFR activates non-existent node",
    },
]

# --- Mock specialist outputs（EGFR + PI3K，含 species + reactions）---
# EGF+PI3K shared Ras 场景：RasGTP 在 EGFR 通路产生，在 PI3K 通路消耗
MOCK_SPECIALIST_OUTPUTS_RAS_CONSERVED = [
    {
        "pathway_class": "EGFR_RTK",
        "species": [
            {"name": "EGFR"},
            {"name": "pEGFR"},
            {"name": "RasGTP"},
            {"name": "PI3K"},
        ],
        "reactions": [
            {
                "id": "r1",
                "substrate": "EGFR",
                "product": "pEGFR",
            },
            {
                "id": "r2",
                "substrate": "RasGDP",
                "product": "RasGTP",  # 产生 RasGTP，stoichiometry=1.0
            },
        ],
        "crosstalk_reactions": [
            {
                "id": "ct1",
                "source": "pEGFR",
                "target": "PI3K",  # target_node 在 target_pathway species 中
            },
        ],
    },
    {
        "pathway_class": "PI3K_AKT_mTOR",
        "species": [
            {"name": "PI3K"},
            {"name": "AKT"},
            {"name": "RasGTP"},
        ],
        "reactions": [
            {
                "id": "r3",
                "substrate": "RasGTP",  # 消耗 RasGTP，stoichiometry=1.0
                "product": "PIP3",
            },
            {
                "id": "r4",
                "substrate": "PIP3",
                "product": "AKT",
            },
        ],
    },
]

# --- Ras 不守恒：EGFR 产生 2，PI3K 消耗 1 → 误差 50% > 10% ---
MOCK_SPECIALIST_OUTPUTS_RAS_NOT_CONSERVED = [
    {
        "pathway_class": "EGFR_RTK",
        "species": [{"name": "RasGTP"}],
        "reactions": [
            {
                "id": "r1",
                "product": {"species": "RasGTP", "stoichiometry": 2.0},  # 产生 2
            },
        ],
    },
    {
        "pathway_class": "PI3K_AKT_mTOR",
        "species": [{"name": "RasGTP"}],
        "reactions": [
            {
                "id": "r2",
                "substrate": "RasGTP",  # 消耗 1
            },
        ],
    },
]

# --- Mock time_scale_alignment（有效）---
MOCK_TIME_SCALE_ALIGNED = {
    "unified_max_step": 0.1,
    "pathway_time_scales": [
        {"pathway_class": "EGFR_RTK", "max_step": 0.1, "time_scale": "fast"},
        {"pathway_class": "PI3K_AKT_mTOR", "max_step": 0.5, "time_scale": "medium"},
    ],
    "alignment_strategy": "min_of_all",
}

# --- Mock time_scale_alignment（无效：unified_max_step 缺失）---
MOCK_TIME_SCALE_NOT_ALIGNED = {
    "pathway_time_scales": [],
    "alignment_strategy": "min_of_all",
    # 缺 unified_max_step
}

# --- Mock time_scale_alignment（unified_max_step = 0）---
MOCK_TIME_SCALE_ZERO_STEP = {
    "unified_max_step": 0,
    "pathway_time_scales": [],
    "alignment_strategy": "min_of_all",
}


# =============================================================================
# TestLevel3CrossPathwayValidator: validate() 主入口
# =============================================================================
class TestLevel3CrossPathwayValidator(unittest.TestCase):
    """测试 Level3CrossPathwayValidator.validate() 主入口。"""

    def setUp(self):
        self.validator = Level3CrossPathwayValidator()

    def test_validate_multi_pathway_all_pass(self):
        """多通路场景：3 项检查全部通过 → pass=True。"""
        state = {
            "v4_pathway_class": MOCK_MULTI_PATHWAY,
            "v4_crosstalk_edges": MOCK_CROSSTALK_EDGES_CONSISTENT,
            "v4_shared_species": ["RasGTP"],
            "v4_time_scale_alignment": MOCK_TIME_SCALE_ALIGNED,
            "v4_specialist_outputs": MOCK_SPECIALIST_OUTPUTS_RAS_CONSERVED,
        }
        report = self.validator.validate(state)
        self.assertTrue(report["pass"])
        self.assertTrue(report["crosstalk_consistency"])
        self.assertLessEqual(report["shared_species_conservation"], 0.10)
        self.assertTrue(report["time_scale_alignment"])

    def test_validate_multi_pathway_conservation_fail(self):
        """多通路场景：shared species 守恒误差 > 10% → pass=False。"""
        state = {
            "v4_pathway_class": MOCK_MULTI_PATHWAY,
            "v4_crosstalk_edges": MOCK_CROSSTALK_EDGES_CONSISTENT,
            "v4_shared_species": ["RasGTP"],
            "v4_time_scale_alignment": MOCK_TIME_SCALE_ALIGNED,
            "v4_specialist_outputs": MOCK_SPECIALIST_OUTPUTS_RAS_NOT_CONSERVED,
        }
        report = self.validator.validate(state)
        self.assertFalse(report["pass"])
        self.assertGreater(report["shared_species_conservation"], 0.10)

    def test_validate_invalid_state_type(self):
        """state 不是 dict → pass=False（异常降级）。"""
        report = self.validator.validate("not_a_dict")  # type: ignore[arg-type]
        self.assertFalse(report["pass"])
        self.assertIn("reason", report)

    def test_validate_exception_degradation(self):
        """validate 内部异常 → pass=False（不抛异常）。"""
        # 模拟 _is_single_pathway 抛异常
        with patch.object(
            self.validator, "_is_single_pathway", side_effect=RuntimeError("boom")
        ):
            report = self.validator.validate({"v4_pathway_class": MOCK_MULTI_PATHWAY})
        self.assertFalse(report["pass"])

    def test_validate_report_has_required_fields(self):
        """验证报告含 spec.md 第 297 行规定的 4 个字段。"""
        state = {
            "v4_pathway_class": MOCK_MULTI_PATHWAY,
            "v4_crosstalk_edges": MOCK_CROSSTALK_EDGES_CONSISTENT,
            "v4_shared_species": ["RasGTP"],
            "v4_time_scale_alignment": MOCK_TIME_SCALE_ALIGNED,
            "v4_specialist_outputs": MOCK_SPECIALIST_OUTPUTS_RAS_CONSERVED,
        }
        report = self.validator.validate(state)
        self.assertIn("pass", report)
        self.assertIn("crosstalk_consistency", report)
        self.assertIn("shared_species_conservation", report)
        self.assertIn("time_scale_alignment", report)

    def test_validate_extracts_crosstalk_edges_from_pathway_graph(self):
        """v4_crosstalk_edges 缺失时从 v4_pathway_graph.crosstalk_edges 提取。"""
        state = {
            "v4_pathway_class": MOCK_MULTI_PATHWAY,
            "v4_pathway_graph": {
                "crosstalk_edges": MOCK_CROSSTALK_EDGES_CONSISTENT,
            },
            "v4_shared_species": ["RasGTP"],
            "v4_time_scale_alignment": MOCK_TIME_SCALE_ALIGNED,
            "v4_specialist_outputs": MOCK_SPECIALIST_OUTPUTS_RAS_CONSERVED,
        }
        report = self.validator.validate(state)
        self.assertTrue(report["crosstalk_consistency"])

    def test_validate_extracts_shared_species_from_pathway_graph(self):
        """v4_shared_species 缺失时从 v4_pathway_graph.shared_species 提取。"""
        state = {
            "v4_pathway_class": MOCK_MULTI_PATHWAY,
            "v4_crosstalk_edges": MOCK_CROSSTALK_EDGES_CONSISTENT,
            "v4_pathway_graph": {
                "shared_species": ["RasGTP"],
            },
            "v4_time_scale_alignment": MOCK_TIME_SCALE_ALIGNED,
            "v4_specialist_outputs": MOCK_SPECIALIST_OUTPUTS_RAS_CONSERVED,
        }
        report = self.validator.validate(state)
        # RasGTP 守恒（误差 <= 10%）
        self.assertLessEqual(report["shared_species_conservation"], 0.10)


# =============================================================================
# TestLevel3HookNode: Feature Flag + hook 行为
# =============================================================================
class TestLevel3HookNode(unittest.TestCase):
    """测试 level3_hook_node LangGraph 节点。"""

    def test_hook_flag_off_returns_empty(self):
        """V4_VALIDATION_PYRAMID_ENABLED=false → hook 返回 {}。"""
        with patch("app.validation_v2.level3_crosstalk.settings") as mock_settings:
            mock_settings.V4_VALIDATION_PYRAMID_ENABLED = False
            mock_settings.effective_v4_validation_pyramid_enabled.return_value = False
            result = level3_hook_node({"v4_pathway_class": MOCK_MULTI_PATHWAY})
        self.assertEqual(result, {})

    def test_hook_flag_on_writes_level3_report(self):
        """V4_VALIDATION_PYRAMID_ENABLED=true → hook 写入 v4_validation_report.level3。"""
        state = {
            "v4_pathway_class": MOCK_MULTI_PATHWAY,
            "v4_crosstalk_edges": MOCK_CROSSTALK_EDGES_CONSISTENT,
            "v4_shared_species": ["RasGTP"],
            "v4_time_scale_alignment": MOCK_TIME_SCALE_ALIGNED,
            "v4_specialist_outputs": MOCK_SPECIALIST_OUTPUTS_RAS_CONSERVED,
        }
        with patch("app.validation_v2.level3_crosstalk.settings") as mock_settings:
            mock_settings.V4_VALIDATION_PYRAMID_ENABLED = True
            result = level3_hook_node(state)
        self.assertIn("v4_validation_report", result)
        self.assertIn("level3", result["v4_validation_report"])
        level3 = result["v4_validation_report"]["level3"]
        self.assertIn("pass", level3)

    def test_hook_preserves_existing_report(self):
        """hook 不覆盖已存在的 level1 / level2 报告。"""
        state = {
            "v4_pathway_class": MOCK_SINGLE_PATHWAY,
            "v4_validation_report": {
                "level1": {"pass": True},
                "level2": {"pass": False, "track": "skipped"},
            },
        }
        with patch("app.validation_v2.level3_crosstalk.settings") as mock_settings:
            mock_settings.V4_VALIDATION_PYRAMID_ENABLED = True
            result = level3_hook_node(state)
        report = result["v4_validation_report"]
        self.assertIn("level1", report)
        self.assertIn("level2", report)
        self.assertIn("level3", report)
        # level1 / level2 保持原值
        self.assertTrue(report["level1"]["pass"])
        self.assertEqual(report["level2"]["track"], "skipped")

    def test_hook_exception_returns_empty(self):
        """hook 内部异常 → 返回 {}（不阻塞主流水线）。"""
        with patch("app.validation_v2.level3_crosstalk.settings") as mock_settings:
            mock_settings.V4_VALIDATION_PYRAMID_ENABLED = True
            with patch(
                "app.validation_v2.level3_crosstalk.Level3CrossPathwayValidator.validate",
                side_effect=RuntimeError("boom"),
            ):
                result = level3_hook_node({"v4_pathway_class": MOCK_MULTI_PATHWAY})
        self.assertEqual(result, {})

    def test_hook_single_pathway_skipped(self):
        """单通路场景 hook 写入 skipped pass=True。"""
        state = {"v4_pathway_class": MOCK_SINGLE_PATHWAY}
        with patch("app.validation_v2.level3_crosstalk.settings") as mock_settings:
            mock_settings.V4_VALIDATION_PYRAMID_ENABLED = True
            result = level3_hook_node(state)
        level3 = result["v4_validation_report"]["level3"]
        self.assertTrue(level3["pass"])
        self.assertTrue(level3.get("skipped"))


# =============================================================================
# TestCrosstalkConsistency: cross-talk consistency 检查
# =============================================================================
class TestCrosstalkConsistency(unittest.TestCase):
    """测试 _check_crosstalk_consistency。"""

    def setUp(self):
        self.validator = Level3CrossPathwayValidator()

    def test_consistent_edges(self):
        """cross-talk edges 一致（target_node 在 target_pathway species 中）。"""
        ok, violations = self.validator._check_crosstalk_consistency(
            MOCK_CROSSTALK_EDGES_CONSISTENT, MOCK_SPECIALIST_OUTPUTS_RAS_CONSERVED
        )
        self.assertTrue(ok)
        self.assertEqual(len(violations), 0)

    def test_inconsistent_edges(self):
        """cross-talk edges 不一致（target_node 不在 target_pathway species 中）。"""
        ok, violations = self.validator._check_crosstalk_consistency(
            MOCK_CROSSTALK_EDGES_INCONSISTENT, MOCK_SPECIALIST_OUTPUTS_RAS_CONSERVED
        )
        self.assertFalse(ok)
        self.assertGreater(len(violations), 0)
        self.assertEqual(violations[0]["target_pathway"], "PI3K_AKT_mTOR")
        self.assertEqual(violations[0]["target_node"], "PI3K_ghost")

    def test_empty_crosstalk_edges(self):
        """空 crosstalk_edges → 一致（无矛盾点）。"""
        ok, violations = self.validator._check_crosstalk_consistency(
            [], MOCK_SPECIALIST_OUTPUTS_RAS_CONSERVED
        )
        self.assertTrue(ok)
        self.assertEqual(len(violations), 0)

    def test_edges_without_specialist_outputs(self):
        """有 edges 但无 specialist_outputs → 一致（保守）。"""
        ok, violations = self.validator._check_crosstalk_consistency(
            MOCK_CROSSTALK_EDGES_CONSISTENT, []
        )
        self.assertTrue(ok)

    def test_violation_includes_required_fields(self):
        """violation 含 edge_id / reason / source_pathway / target_pathway。"""
        _ok, violations = self.validator._check_crosstalk_consistency(
            MOCK_CROSSTALK_EDGES_INCONSISTENT, MOCK_SPECIALIST_OUTPUTS_RAS_CONSERVED
        )
        v = violations[0]
        self.assertIn("edge_id", v)
        self.assertIn("reason", v)
        self.assertIn("source_pathway", v)
        self.assertIn("target_pathway", v)
        self.assertIn("source_node", v)
        self.assertIn("target_node", v)


# =============================================================================
# TestSharedSpeciesConservation: shared species conservation 检查
# =============================================================================
class TestSharedSpeciesConservation(unittest.TestCase):
    """测试 _check_shared_species_conservation。"""

    def setUp(self):
        self.validator = Level3CrossPathwayValidator()

    def test_conservation_within_threshold(self):
        """shared species 守恒（误差 < 10%）。"""
        # RasGTP 产生 1，消耗 1 → 误差 0%
        max_error, violations = self.validator._check_shared_species_conservation(
            ["RasGTP"], MOCK_SPECIALIST_OUTPUTS_RAS_CONSERVED
        )
        self.assertLessEqual(max_error, 0.10)
        self.assertEqual(len(violations), 0)

    def test_conservation_exceeds_threshold(self):
        """shared species 不守恒（误差 > 10%）。"""
        # RasGTP 产生 2，消耗 1 → 误差 50% > 10%
        max_error, violations = self.validator._check_shared_species_conservation(
            ["RasGTP"], MOCK_SPECIALIST_OUTPUTS_RAS_NOT_CONSERVED
        )
        self.assertGreater(max_error, 0.10)
        self.assertGreater(len(violations), 0)
        self.assertEqual(violations[0]["species"], "RasGTP")

    def test_egf_pi3k_ras_scenario(self):
        """EGF+PI3K shared Ras 场景：守恒（误差 < 10%）。"""
        max_error, _violations = self.validator._check_shared_species_conservation(
            ["RasGTP"], MOCK_SPECIALIST_OUTPUTS_RAS_CONSERVED
        )
        # spec 要求 EGF+PI3K shared Ras 守恒 < 10%
        self.assertLess(max_error, 0.10)

    def test_empty_shared_species(self):
        """空 shared_species → 守恒（误差 0.0）。"""
        max_error, violations = self.validator._check_shared_species_conservation(
            [], MOCK_SPECIALIST_OUTPUTS_RAS_CONSERVED
        )
        self.assertEqual(max_error, 0.0)
        self.assertEqual(len(violations), 0)

    def test_shared_species_not_in_reactions(self):
        """shared species 在 reactions 中未出现 → 守恒（误差 0.0）。"""
        # RasGTP 不在任何 reaction 的 substrate/product 中
        outputs = [
            {
                "pathway_class": "EGFR_RTK",
                "reactions": [{"id": "r1", "substrate": "EGF", "product": "EGFR"}],
            }
        ]
        max_error, violations = self.validator._check_shared_species_conservation(
            ["RasGTP"], outputs
        )
        self.assertEqual(max_error, 0.0)
        self.assertEqual(len(violations), 0)

    def test_stoichiometry_from_dict_format(self):
        """验证 dict 格式的 stoichiometry 提取。"""
        outputs = [
            {
                "pathway_class": "A",
                "reactions": [
                    {"id": "r1", "product": {"species": "RasGTP", "stoichiometry": 3.0}}
                ],
            },
            {
                "pathway_class": "B",
                "reactions": [
                    {"id": "r2", "substrate": {"species": "RasGTP", "stoichiometry": 3.0}}
                ],
            },
        ]
        max_error, _v = self.validator._check_shared_species_conservation(
            ["RasGTP"], outputs
        )
        # 产生 3，消耗 3 → 守恒
        self.assertLess(max_error, 0.10)

    def test_violation_includes_required_fields(self):
        """violation 含 species / produced / consumed / error / reason。"""
        _err, violations = self.validator._check_shared_species_conservation(
            ["RasGTP"], MOCK_SPECIALIST_OUTPUTS_RAS_NOT_CONSERVED
        )
        v = violations[0]
        self.assertIn("species", v)
        self.assertIn("produced", v)
        self.assertIn("consumed", v)
        self.assertIn("error", v)
        self.assertIn("reason", v)

    def test_threshold_is_10_percent(self):
        """验证阈值常量为 10%（spec.md 第 298 行）。"""
        self.assertEqual(
            Level3CrossPathwayValidator.SHARED_SPECIES_CONSERVATION_THRESHOLD, 0.10
        )


# =============================================================================
# TestTimeScaleAlignment: time-scale alignment 检查
# =============================================================================
class TestTimeScaleAlignment(unittest.TestCase):
    """测试 _check_time_scale_alignment。"""

    def setUp(self):
        self.validator = Level3CrossPathwayValidator()

    def test_aligned_time_scale(self):
        """时间尺度对齐（unified_max_step 有效）。"""
        result = self.validator._check_time_scale_alignment(MOCK_TIME_SCALE_ALIGNED)
        self.assertTrue(result)

    def test_not_aligned_missing_unified_max_step(self):
        """时间尺度未对齐（unified_max_step 缺失）。"""
        result = self.validator._check_time_scale_alignment(
            MOCK_TIME_SCALE_NOT_ALIGNED
        )
        self.assertFalse(result)

    def test_empty_time_scale_alignment(self):
        """空 time_scale_alignment → 未对齐。"""
        result = self.validator._check_time_scale_alignment({})
        self.assertFalse(result)

    def test_zero_unified_max_step(self):
        """unified_max_step = 0 → 未对齐。"""
        result = self.validator._check_time_scale_alignment(
            MOCK_TIME_SCALE_ZERO_STEP
        )
        self.assertFalse(result)

    def test_negative_unified_max_step(self):
        """unified_max_step 为负 → 未对齐。"""
        result = self.validator._check_time_scale_alignment(
            {"unified_max_step": -0.1}
        )
        self.assertFalse(result)

    def test_non_numeric_unified_max_step(self):
        """unified_max_step 为非数值 → 未对齐。"""
        result = self.validator._check_time_scale_alignment(
            {"unified_max_step": "invalid"}
        )
        self.assertFalse(result)

    def test_non_dict_time_scale_alignment(self):
        """time_scale_alignment 非 dict → 未对齐。"""
        result = self.validator._check_time_scale_alignment("not_a_dict")  # type: ignore[arg-type]
        self.assertFalse(result)


# =============================================================================
# TestSinglePathwaySkipped: 单通路 skipped
# =============================================================================
class TestSinglePathwaySkipped(unittest.TestCase):
    """测试单通路 skipped 逻辑。"""

    def setUp(self):
        self.validator = Level3CrossPathwayValidator()

    def test_single_pathway_is_single(self):
        """单通路 pathway_class 视为单通路。"""
        self.assertTrue(self.validator._is_single_pathway("EGFR_RTK"))
        self.assertTrue(self.validator._is_single_pathway("NF_KB"))

    def test_multi_pathway_not_single(self):
        """多通路 pathway_class 不视为单通路。"""
        self.assertFalse(self.validator._is_single_pathway(MOCK_MULTI_PATHWAY))
        self.assertFalse(
            self.validator._is_single_pathway("MULTI:EGFR_RTK+MAPK_ERK+PI3K_AKT_mTOR")
        )

    def test_unknown_pathway_treated_as_single(self):
        """pathway_class="UNKNOWN" 视为单通路。"""
        self.assertTrue(self.validator._is_single_pathway("UNKNOWN"))

    def test_empty_pathway_treated_as_single(self):
        """空 pathway_class 视为单通路。"""
        self.assertTrue(self.validator._is_single_pathway(""))
        self.assertTrue(self.validator._is_single_pathway(None))  # type: ignore[arg-type]

    def test_single_pathway_validate_returns_skipped(self):
        """单通路场景 validate 返回 skipped pass=True。"""
        report = self.validator.validate({"v4_pathway_class": MOCK_SINGLE_PATHWAY})
        self.assertTrue(report["pass"])
        self.assertTrue(report.get("skipped"))
        self.assertEqual(report.get("reason"), "single_pathway")

    def test_skipped_report_has_required_fields(self):
        """skipped 报告含 spec 规定的字段。"""
        report = self.validator._run_skipped("single_pathway")
        self.assertTrue(report["pass"])
        self.assertTrue(report["crosstalk_consistency"])
        self.assertEqual(report["shared_species_conservation"], 0.0)
        self.assertTrue(report["time_scale_alignment"])
        self.assertTrue(report["skipped"])
        self.assertEqual(report["reason"], "single_pathway")

    def test_skipped_pass_is_true(self):
        """单通路 skipped pass=True（与 Level 2 skipped pass=False 不同）。"""
        report = self.validator._run_skipped("single_pathway")
        self.assertTrue(report["pass"])

    def test_single_pathway_does_not_check_consistency(self):
        """单通路场景不执行 cross-talk consistency 检查。"""
        # 即使有不一致的 edges，单通路仍 skipped pass=True
        state = {
            "v4_pathway_class": MOCK_SINGLE_PATHWAY,
            "v4_crosstalk_edges": MOCK_CROSSTALK_EDGES_INCONSISTENT,
            "v4_specialist_outputs": MOCK_SPECIALIST_OUTPUTS_RAS_NOT_CONSERVED,
        }
        report = self.validator.validate(state)
        self.assertTrue(report["pass"])
        self.assertTrue(report.get("skipped"))


if __name__ == "__main__":
    unittest.main()
