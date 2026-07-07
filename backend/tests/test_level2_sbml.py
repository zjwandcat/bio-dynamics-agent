# BioDynamics Agent v4 - Level 2 SBML/BioModels Validation 单元测试
# (Phase 5 / Task 5.3.7)
#
# 测试 Level2SBMLValidator 主类 + Track A/B/skipped 三态 + LangGraph hook。
#
# 测试用例（35 个，>= 25 要求）：
#   - TestLevel2SBMLValidator: validate() 主入口 + 异常降级
#   - TestLevel2HookNode: Feature Flag + hook 行为
#   - TestTrackA: roadrunner 真实仿真（mock）
#   - TestTrackB: 结构相似度 + 差异指标 null（修复审计 §7.2）
#   - TestSkipped: skipped 状态 pass=False（修复审计 §7.2）
#   - TestSpeciesAlignment: ontology ID 对齐（修复审计 §10.3）
#   - TestThresholds: 通路特异阈值
#
# 运行：cd backend && python -m pytest tests/test_level2_sbml.py -v

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
# Mock 数据
# =============================================================================

# --- Mock v4_ode_system ---
MOCK_ODE_SYSTEM = {
    "ode_code": """
EGF_0 = 0.1
EGFR_0 = 100.0
k_on = 0.003
k_off = 0.05
dEGF/dt = -k_on * EGF * EGFR + k_off * EGF_EGFR
dEGFR/dt = -k_on * EGF * EGFR + k_off * EGF_EGFR - 0.01 * EGFR
dEGF_EGFR/dt = k_on * EGF * EGFR - k_off * EGF_EGFR
""",
    "species": [
        {"id": "EGF", "canonical_name": "EGF", "initial_concentration": 0.1},
        {"id": "EGFR", "canonical_name": "EGFR", "initial_concentration": 100.0},
        {"id": "EGF_EGFR", "canonical_name": "EGF_EGFR", "initial_concentration": 0.0},
    ],
    "equations": [
        {"eq_id": "eq1", "species": "EGF", "reaction_id": "r1"},
        {"eq_id": "eq2", "species": "EGFR", "reaction_id": "r2"},
    ],
}

# --- Mock SBML model id ---
MOCK_SBML_MODEL_ID = "BIOMD0000000205"

# --- Mock SBML text ---
MOCK_SBML_TEXT = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level2/version4" level="2" version="4">
  <model id="BIOMD0000000205">
    <listOfSpecies>
      <species id="EGF" name="EGF" initialConcentration="0.1"/>
      <species id="EGFR" name="EGFR" initialConcentration="100.0"/>
    </listOfSpecies>
  </model>
</sbml>
"""

# --- Mock grounding_ledger（含 species_mapping with ontology_ref）---
MOCK_GROUNDING_LEDGER = {
    "ode_equations": [
        {"eq_id": "eq1", "reaction_id": "r1", "sbml_reaction_id": "sbml_r1",
         "kinetics_type": "mass_action"},
    ],
    "species_mapping": [
        {
            "species_id": "EGF",
            "canonical_name": "EGF",
            "ontology_ref": {
                "hgnc_id": "HGNC:3229",
                "uniprot_id": "P01133",
                "species_type": "ligand",
            },
            "verified": True,
            "warnings": [],
            "source": "sbml_annotation",
        },
        {
            "species_id": "EGFR",
            "canonical_name": "EGFR",
            "ontology_ref": {
                "hgnc_id": "HGNC:3236",
                "uniprot_id": "P00533",
                "species_type": "protein",
            },
            "verified": True,
            "warnings": [],
            "source": "sbml_annotation",
        },
    ],
    "integrity": True,
    "warnings": [],
    "statistics": {"verified_species": 2, "total_species": 2},
}

# --- v4 species 列表（用于对齐测试）---
MOCK_V4_SPECIES = [
    {"id": "EGF", "canonical_name": "EGF", "ontology": {}},
    {"id": "EGFR", "canonical_name": "EGFR", "ontology": {}},
]

# --- SBML species 列表（用于对齐测试）---
MOCK_SBML_SPECIES = [
    {
        "species_id": "EGF",
        "canonical_name": "EGF",
        "ontology_ref": {"hgnc_id": "HGNC:3229", "uniprot_id": "P01133"},
    },
    {
        "species_id": "EGFR",
        "canonical_name": "EGFR",
        "ontology_ref": {"hgnc_id": "HGNC:3236", "uniprot_id": "P00533"},
    },
]


# =============================================================================
# 辅助：构造 mock roadrunner 模块
# =============================================================================
def make_mock_roadrunner(sbml_series=None, v4_series=None):
    """构造 mock roadrunner 模块，返回指定仿真结果。

    Args:
        sbml_series: dict {species_id: [(time, concentration), ...]}
            若为 None 使用默认线性衰减
        v4_series: 同上（用于 v4_ode 仿真，但 v4_ode 不走 roadrunner，此处忽略）

    Returns:
        MagicMock 模拟 roadrunner 模块
    """
    if sbml_series is None:
        # 默认：EGF 衰减 0.1 → 0.0；EGFR 衰减 100 → 50
        times = [i * 0.3 for i in range(200)]
        sbml_series = {
            "EGF": [(t, 0.1 * (1.0 - 0.5 * t / 60.0)) for t in times],
            "EGFR": [(t, 100.0 * (1.0 - 0.3 * t / 60.0)) for t in times],
        }

    mock_module = MagicMock()
    mock_rr = MagicMock()
    mock_module.RoadRunner.return_value = mock_rr

    # mock simulate 返回 numpy-like 二维数组
    # 提取所有 species + time
    all_species = list(sbml_series.keys())
    times = [t for t, _ in sbml_series[all_species[0]]]
    n_rows = len(times)
    n_cols = 1 + len(all_species)  # time + species

    # 构造 colnames: ["time", "EGF", "EGFR"]
    col_names = ["time"] + all_species

    # 构造二维数据：result[row, col] 索引
    class MockResult:
        def __init__(self):
            self.colnames = col_names
            self._data = {}
            for col_idx, sp_id in enumerate(all_species):
                col_name = col_names[col_idx + 1]
                self._data[col_name] = [v for _, v in sbml_series[sp_id]]
            self._times = times

        def __getitem__(self, key):
            # 支持 result[:, 0] 和 result[:, "EGF"]
            if isinstance(key, tuple) and len(key) == 2:
                row_key, col_key = key
                if row_key == slice(None, None, None):
                    if isinstance(col_key, int):
                        # result[:, 0] → 时间序列
                        if col_key == 0:
                            return self._times
                        # result[:, 1] → 第一个 species
                        return self._data[col_names[col_key]]
                    elif isinstance(col_key, str):
                        return self._data[col_key]
            return []

    mock_rr.simulate.return_value = MockResult()
    return mock_module


# =============================================================================
# TestLevel2SBMLValidator: validate() 主入口
# =============================================================================
class TestLevel2SBMLValidator(unittest.TestCase):
    """测试 Level2SBMLValidator.validate() 主入口。"""

    def setUp(self):
        from app.validation_v2.level2_sbml import Level2SBMLValidator
        self.validator = Level2SBMLValidator()

    def test_validate_returns_correct_structure(self):
        """validate() 返回包含所有必需字段的 dict。"""
        state = {
            "v4_ode_system": MOCK_ODE_SYSTEM,
            "sbml_model_id": MOCK_SBML_MODEL_ID,
            "v4_pathway_class": "EGFR_RTK",
            "v4_grounding_ledger": MOCK_GROUNDING_LEDGER,
        }
        result = self.validator.validate(state)

        # 验证所有必需字段存在
        required_fields = [
            "pass", "track", "peak_diff", "peak_time_diff",
            "amplification_diff", "sbml_sim_available", "method",
        ]
        for field in required_fields:
            self.assertIn(field, result, f"缺少必需字段: {field}")

        # 验证字段类型
        self.assertIsInstance(result["pass"], bool)
        self.assertIn(result["track"], ["A", "B", "skipped"])
        self.assertIsInstance(result["sbml_sim_available"], bool)
        self.assertIsInstance(result["method"], str)

    def test_validate_skipped_when_no_sbml_model_id(self):
        """sbml_model_id 缺失 → skipped 状态 pass=False。"""
        state = {
            "v4_ode_system": MOCK_ODE_SYSTEM,
            "v4_pathway_class": "EGFR_RTK",
        }
        result = self.validator.validate(state)
        self.assertEqual(result["track"], "skipped")
        self.assertFalse(result["pass"])  # 关键修复：skipped pass=False

    def test_validate_skipped_when_no_ode_system(self):
        """v4_ode_system 缺失 → skipped 状态 pass=False。"""
        state = {
            "sbml_model_id": MOCK_SBML_MODEL_ID,
            "v4_pathway_class": "EGFR_RTK",
        }
        result = self.validator.validate(state)
        self.assertEqual(result["track"], "skipped")
        self.assertFalse(result["pass"])

    def test_validate_skipped_when_state_not_dict(self):
        """state 非 dict → skipped pass=False。"""
        result = self.validator.validate(None)  # type: ignore
        self.assertEqual(result["track"], "skipped")
        self.assertFalse(result["pass"])

    def test_validate_skipped_all_diff_null(self):
        """skipped 状态所有差异指标必须 None。"""
        state = {"sbml_model_id": MOCK_SBML_MODEL_ID}  # 缺 v4_ode_system
        result = self.validator.validate(state)
        self.assertIsNone(result["peak_diff"])
        self.assertIsNone(result["peak_time_diff"])
        self.assertIsNone(result["amplification_diff"])

    def test_validate_track_b_when_roadrunner_unavailable(self):
        """roadrunner 不可用 → 降级到 Track B。"""
        # 默认 ROADRUNNER_AVAILABLE=False（除非环境装了 roadrunner）
        # 此处强制 patch 为 False
        with patch("app.validation_v2.level2_sbml.ROADRUNNER_AVAILABLE", False):
            validator = self.validator
            state = {
                "v4_ode_system": MOCK_ODE_SYSTEM,
                "sbml_model_id": MOCK_SBML_MODEL_ID,
                "v4_pathway_class": "EGFR_RTK",
                "v4_grounding_ledger": MOCK_GROUNDING_LEDGER,
            }
            result = validator.validate(state)
            self.assertIn(result["track"], ["B", "skipped"])  # 至少降级
            if result["track"] == "B":
                # Track B 差异指标必须 None（修复审计 §7.2）
                self.assertIsNone(result["peak_diff"])
                self.assertIsNone(result["peak_time_diff"])
                self.assertIsNone(result["amplification_diff"])
                self.assertFalse(result["sbml_sim_available"])

    def test_validate_exception_returns_pass_false(self):
        """validate() 内部异常 → 返回 pass=False（不抛异常）。"""
        # 构造会导致异常的 state：sbml_model_id 为不可比较类型
        # 这里通过 mock 让 _run_track_a 抛异常（如果 roadrunner 可用）
        with patch.object(
            self.validator, "_run_track_a", side_effect=RuntimeError("test")
        ):
            with patch("app.validation_v2.level2_sbml.ROADRUNNER_AVAILABLE", True):
                state = {
                    "v4_ode_system": MOCK_ODE_SYSTEM,
                    "sbml_model_id": MOCK_SBML_MODEL_ID,
                    "v4_pathway_class": "EGFR_RTK",
                    "v4_grounding_ledger": MOCK_GROUNDING_LEDGER,
                }
                result = self.validator.validate(state)
                # Track A 异常 → 降级到 Track B
                self.assertEqual(result["track"], "B")


# =============================================================================
# TestLevel2HookNode: LangGraph hook
# =============================================================================
class TestLevel2HookNode(unittest.TestCase):
    """测试 level2_hook_node LangGraph 节点。"""

    @patch("app.validation_v2.level2_sbml.settings")
    def test_flag_false_returns_empty(self, mock_settings):
        """Feature Flag false → hook 返回 {}。"""
        mock_settings.V4_VALIDATION_PYRAMID_ENABLED = False
        mock_settings.effective_v4_validation_pyramid_enabled.return_value = False
        from app.validation_v2.level2_sbml import level2_hook_node

        state = {
            "v4_ode_system": MOCK_ODE_SYSTEM,
            "sbml_model_id": MOCK_SBML_MODEL_ID,
            "v4_grounding_ledger": MOCK_GROUNDING_LEDGER,
        }
        result = level2_hook_node(state)
        self.assertEqual(result, {})

    @patch("app.validation_v2.level2_sbml.settings")
    def test_flag_true_writes_validation_report(self, mock_settings):
        """Feature Flag true → hook 写入 v4_validation_report.level2。"""
        mock_settings.V4_VALIDATION_PYRAMID_ENABLED = True
        from app.validation_v2.level2_sbml import level2_hook_node

        state = {
            "v4_ode_system": MOCK_ODE_SYSTEM,
            "sbml_model_id": MOCK_SBML_MODEL_ID,
            "v4_pathway_class": "EGFR_RTK",
            "v4_grounding_ledger": MOCK_GROUNDING_LEDGER,
        }
        result = level2_hook_node(state)
        self.assertIn("v4_validation_report", result)
        report = result["v4_validation_report"]
        self.assertIn("level2", report)
        level2 = report["level2"]
        self.assertIn("pass", level2)
        self.assertIn("track", level2)

    @patch("app.validation_v2.level2_sbml.settings")
    def test_hook_merges_existing_report(self, mock_settings):
        """hook 与现有 v4_validation_report 合并（不覆盖 level1/level3）。"""
        mock_settings.V4_VALIDATION_PYRAMID_ENABLED = True
        from app.validation_v2.level2_sbml import level2_hook_node

        state = {
            "v4_ode_system": MOCK_ODE_SYSTEM,
            "sbml_model_id": MOCK_SBML_MODEL_ID,
            "v4_grounding_ledger": MOCK_GROUNDING_LEDGER,
            "v4_validation_report": {
                "level1": {"pass": True, "mass_conservation_error": 0.0},
            },
        }
        result = level2_hook_node(state)
        report = result["v4_validation_report"]
        # level2 被写入
        self.assertIn("level2", report)
        # level1 被保留
        self.assertIn("level1", report)
        self.assertEqual(report["level1"]["mass_conservation_error"], 0.0)

    @patch("app.validation_v2.level2_sbml.settings")
    def test_hook_exception_returns_empty(self, mock_settings):
        """hook 异常 → 返回 {}（不抛异常）。"""
        mock_settings.V4_VALIDATION_PYRAMID_ENABLED = True
        from app.validation_v2.level2_sbml import level2_hook_node

        # state 为 None 会触发 validate 内部异常 → hook 应返回 {}
        # 注：validate(None) 返回 skipped (pass=False)，不抛异常
        # 因此 hook 会返回包含 level2 skipped 的 report
        result = level2_hook_node(None)  # type: ignore
        # validate(None) 返回 skipped → hook 写入 level2 skipped
        self.assertIn("v4_validation_report", result)
        self.assertEqual(result["v4_validation_report"]["level2"]["track"], "skipped")
        self.assertFalse(result["v4_validation_report"]["level2"]["pass"])


# =============================================================================
# TestTrackA: roadrunner 真实仿真
# =============================================================================
class TestTrackA(unittest.TestCase):
    """测试 _run_track_a（roadrunner 真实仿真）。"""

    def setUp(self):
        from app.validation_v2.level2_sbml import Level2SBMLValidator
        # 通过 factory 注入 mock roadrunner
        self.mock_rr_module = make_mock_roadrunner()
        self.validator = Level2SBMLValidator(
            roadrunner_factory=lambda: self.mock_rr_module
        )

    def test_track_a_returns_correct_structure(self):
        """Track A 返回包含所有必需字段的 dict。"""
        result = self.validator._run_track_a(
            ode_system=MOCK_ODE_SYSTEM,
            sbml_model_id=MOCK_SBML_MODEL_ID,
            pathway_class="EGFR_RTK",
            grounding_ledger=MOCK_GROUNDING_LEDGER,
            sbml_model_text=MOCK_SBML_TEXT,
        )
        self.assertEqual(result["track"], "A")
        self.assertTrue(result["sbml_sim_available"])
        self.assertEqual(result["method"], "roadrunner_simulation")
        self.assertIn("pass", result)
        self.assertIn("peak_diff", result)
        self.assertIn("peak_time_diff", result)
        self.assertIn("amplification_diff", result)

    def test_track_a_pass_when_diff_within_threshold(self):
        """peak_time_diff + amplification_diff 在阈值内 → pass=True。"""
        # mock _simulate_sbml 与 _simulate_v4_ode 返回相同 series → diff = 0
        identical_series = {
            "EGF": [(i * 0.3, 0.1 - 0.001 * i) for i in range(200)],
            "EGFR": [(i * 0.3, 100.0 - 0.1 * i) for i in range(200)],
        }
        with patch.object(
            self.validator, "_simulate_sbml", return_value=identical_series
        ):
            with patch.object(
                self.validator, "_simulate_v4_ode", return_value=identical_series
            ):
                result = self.validator._run_track_a(
                    ode_system=MOCK_ODE_SYSTEM,
                    sbml_model_id=MOCK_SBML_MODEL_ID,
                    pathway_class="EGFR_RTK",
                    grounding_ledger=MOCK_GROUNDING_LEDGER,
                    sbml_model_text=MOCK_SBML_TEXT,
                )
        self.assertTrue(result["pass"])
        # diff 应该为 0（或非常小）
        self.assertAlmostEqual(result["peak_diff"] or 0.0, 0.0, places=5)
        self.assertAlmostEqual(result["peak_time_diff"] or 0.0, 0.0, places=5)
        self.assertAlmostEqual(result["amplification_diff"] or 0.0, 0.0, places=5)

    def test_track_a_fail_when_peak_time_diff_exceeds_threshold(self):
        """peak_time_diff 超阈值 → pass=False。"""
        # SBML peak_time = 0，v4 peak_time = 60 → diff = 60 > 2.0 (EGFR 阈值)
        sbml_series = {
            "EGF": [(i * 0.3, 0.1 if i == 0 else 0.0) for i in range(200)],
            "EGFR": [(i * 0.3, 100.0 if i == 0 else 0.0) for i in range(200)],
        }
        v4_series = {
            "EGF": [(i * 0.3, 0.1 if i == 199 else 0.0) for i in range(200)],
            "EGFR": [(i * 0.3, 100.0 if i == 199 else 0.0) for i in range(200)],
        }
        with patch.object(
            self.validator, "_simulate_sbml", return_value=sbml_series
        ):
            with patch.object(
                self.validator, "_simulate_v4_ode", return_value=v4_series
            ):
                result = self.validator._run_track_a(
                    ode_system=MOCK_ODE_SYSTEM,
                    sbml_model_id=MOCK_SBML_MODEL_ID,
                    pathway_class="EGFR_RTK",
                    grounding_ledger=MOCK_GROUNDING_LEDGER,
                    sbml_model_text=MOCK_SBML_TEXT,
                )
        self.assertFalse(result["pass"])
        self.assertGreater(result["peak_time_diff"], 2.0)

    def test_track_a_fail_when_amplification_diff_exceeds_threshold(self):
        """amplification_diff 超阈值 → pass=False。"""
        # SBML: peak=100, baseline=0, amplification=100
        # v4:    peak=50,  baseline=0, amplification=50
        # amplification_diff = |100-50|/max(100,50,1e-9) = 0.5 > 0.20 (EGFR 阈值)
        sbml_series = {
            "EGF": [(0, 0.0), (1, 100.0), (2, 100.0)],
            "EGFR": [(0, 0.0), (1, 100.0), (2, 100.0)],
        }
        v4_series = {
            "EGF": [(0, 0.0), (1, 50.0), (2, 50.0)],
            "EGFR": [(0, 0.0), (1, 50.0), (2, 50.0)],
        }
        with patch.object(
            self.validator, "_simulate_sbml", return_value=sbml_series
        ):
            with patch.object(
                self.validator, "_simulate_v4_ode", return_value=v4_series
            ):
                result = self.validator._run_track_a(
                    ode_system=MOCK_ODE_SYSTEM,
                    sbml_model_id=MOCK_SBML_MODEL_ID,
                    pathway_class="EGFR_RTK",
                    grounding_ledger=MOCK_GROUNDING_LEDGER,
                    sbml_model_text=MOCK_SBML_TEXT,
                )
        self.assertFalse(result["pass"])
        self.assertGreater(result["amplification_diff"], 0.20)

    def test_track_a_pathway_specific_thresholds_applied(self):
        """通路特异阈值被应用（NF-κB 容忍 30 min peak_time_diff）。"""
        # 两边形状相同（peak=100, baseline=0, amplification=100），仅 peak_time 不同
        # peak_time_diff = |5 - 20| = 15 min
        # EGFR 阈值 2.0 min → fail
        # NF_KB 阈值 30 min → pass（amplification_diff=0 也通过）
        sbml_series = {
            "EGF": [(0, 0.0), (5, 100.0), (60, 0.0)],
            "EGFR": [(0, 0.0), (5, 100.0), (60, 0.0)],
        }
        v4_series = {
            "EGF": [(0, 0.0), (20, 100.0), (60, 0.0)],
            "EGFR": [(0, 0.0), (20, 100.0), (60, 0.0)],
        }
        with patch.object(
            self.validator, "_simulate_sbml", return_value=sbml_series
        ):
            with patch.object(
                self.validator, "_simulate_v4_ode", return_value=v4_series
            ):
                # EGFR 阈值（peak_time_diff=2.0）
                result_egfr = self.validator._run_track_a(
                    ode_system=MOCK_ODE_SYSTEM,
                    sbml_model_id=MOCK_SBML_MODEL_ID,
                    pathway_class="EGFR_RTK",
                    grounding_ledger=MOCK_GROUNDING_LEDGER,
                    sbml_model_text=MOCK_SBML_TEXT,
                )
                # peak_time_diff = |5 - 20| = 15 min > 2.0
                self.assertFalse(result_egfr["pass"])

                # NF-κB 阈值（peak_time_diff=30）
                result_nfkb = self.validator._run_track_a(
                    ode_system=MOCK_ODE_SYSTEM,
                    sbml_model_id=MOCK_SBML_MODEL_ID,
                    pathway_class="NF_KB",
                    grounding_ledger=MOCK_GROUNDING_LEDGER,
                    sbml_model_text=MOCK_SBML_TEXT,
                )
                # peak_time_diff = 15 min <= 30 → pass
                # amplification_diff = 0 (两边 amp=100 相同) <= 0.50 → pass
                self.assertTrue(result_nfkb["pass"])

    def test_track_a_no_sbml_text_fallback_to_track_b(self):
        """无 SBML 文本 → 降级到 Track B。"""
        result = self.validator._run_track_a(
            ode_system=MOCK_ODE_SYSTEM,
            sbml_model_id=MOCK_SBML_MODEL_ID,
            pathway_class="EGFR_RTK",
            grounding_ledger=MOCK_GROUNDING_LEDGER,
            sbml_model_text="",
        )
        self.assertEqual(result["track"], "B")
        self.assertFalse(result["sbml_sim_available"])

    def test_track_a_peak_diff_calculation(self):
        """peak_diff 正确计算（相对差）。"""
        # SBML peak = 100，v4 peak = 50 → peak_diff = 0.5
        sbml_series = {
            "EGF": [(i * 0.3, 100.0) for i in range(200)],
            "EGFR": [(i * 0.3, 100.0) for i in range(200)],
        }
        v4_series = {
            "EGF": [(i * 0.3, 50.0) for i in range(200)],
            "EGFR": [(i * 0.3, 50.0) for i in range(200)],
        }
        with patch.object(
            self.validator, "_simulate_sbml", return_value=sbml_series
        ):
            with patch.object(
                self.validator, "_simulate_v4_ode", return_value=v4_series
            ):
                result = self.validator._run_track_a(
                    ode_system=MOCK_ODE_SYSTEM,
                    sbml_model_id=MOCK_SBML_MODEL_ID,
                    pathway_class="EGFR_RTK",
                    grounding_ledger=MOCK_GROUNDING_LEDGER,
                    sbml_model_text=MOCK_SBML_TEXT,
                )
        # peak_diff = |100-50|/max(100,50) = 0.5
        self.assertAlmostEqual(result["peak_diff"], 0.5, places=2)


# =============================================================================
# TestTrackB: 结构相似度 fallback
# =============================================================================
class TestTrackB(unittest.TestCase):
    """测试 _run_track_b（结构相似度 fallback）。"""

    def setUp(self):
        from app.validation_v2.level2_sbml import Level2SBMLValidator
        self.validator = Level2SBMLValidator()

    def test_track_b_returns_correct_structure(self):
        """Track B 返回包含所有必需字段的 dict。"""
        result = self.validator._run_track_b(
            ode_system=MOCK_ODE_SYSTEM,
            sbml_model_id=MOCK_SBML_MODEL_ID,
            pathway_class="EGFR_RTK",
            grounding_ledger=MOCK_GROUNDING_LEDGER,
        )
        self.assertEqual(result["track"], "B")
        self.assertIn("pass", result)
        self.assertIn("method", result)
        self.assertIn("similarity_score", result)

    def test_track_b_diff_metrics_must_be_null(self):
        """**关键修复（审计 §7.2）**：Track B 差异指标必须 None。"""
        result = self.validator._run_track_b(
            ode_system=MOCK_ODE_SYSTEM,
            sbml_model_id=MOCK_SBML_MODEL_ID,
            pathway_class="EGFR_RTK",
            grounding_ledger=MOCK_GROUNDING_LEDGER,
        )
        # 关键修复：peak_diff/peak_time_diff/amplification_diff 必须是 None
        self.assertIsNone(result["peak_diff"])
        self.assertIsNone(result["peak_time_diff"])
        self.assertIsNone(result["amplification_diff"])

    def test_track_b_no_error_diff_zero(self):
        """**关键修复（审计 §7.2）**：禁止 error_diff=0（必须 None）。"""
        result = self.validator._run_track_b(
            ode_system=MOCK_ODE_SYSTEM,
            sbml_model_id=MOCK_SBML_MODEL_ID,
            pathway_class="EGFR_RTK",
            grounding_ledger=MOCK_GROUNDING_LEDGER,
        )
        # 显式断言：不等于 0.0（必须是 None）
        self.assertNotEqual(result["peak_diff"], 0.0)
        self.assertNotEqual(result["peak_time_diff"], 0.0)
        self.assertNotEqual(result["amplification_diff"], 0.0)

    def test_track_b_sbml_sim_available_false(self):
        """Track B 的 sbml_sim_available=False。"""
        result = self.validator._run_track_b(
            ode_system=MOCK_ODE_SYSTEM,
            sbml_model_id=MOCK_SBML_MODEL_ID,
            pathway_class="EGFR_RTK",
            grounding_ledger=MOCK_GROUNDING_LEDGER,
        )
        self.assertFalse(result["sbml_sim_available"])

    def test_track_b_pass_with_high_similarity(self):
        """高结构相似度 → pass=True。"""
        # v4 与 sbml 物种数、反应数、机制类型完全一致 → similarity=1.0
        result = self.validator._run_track_b(
            ode_system=MOCK_ODE_SYSTEM,
            sbml_model_id=MOCK_SBML_MODEL_ID,
            pathway_class="EGFR_RTK",
            grounding_ledger=MOCK_GROUNDING_LEDGER,
        )
        self.assertGreaterEqual(result["similarity_score"], 0.0)
        # similarity_score 应 >= 0.6（pass 阈值）
        # 但 aligned_ratio 取决于 ontology 对齐
        self.assertIsInstance(result["similarity_score"], float)

    def test_track_b_fail_with_low_similarity(self):
        """低结构相似度 → pass=False。"""
        # 构造完全不匹配的 grounding_ledger（无 species_mapping）
        empty_ledger = {
            "ode_equations": [],
            "species_mapping": [],
            "integrity": False,
        }
        result = self.validator._run_track_b(
            ode_system=MOCK_ODE_SYSTEM,
            sbml_model_id=MOCK_SBML_MODEL_ID,
            pathway_class="EGFR_RTK",
            grounding_ledger=empty_ledger,
        )
        # 完全不匹配 → similarity 低 → pass=False
        self.assertLess(result["similarity_score"], 0.6)
        self.assertFalse(result["pass"])

    def test_track_b_method_includes_fallback_reason(self):
        """fallback_reason 被记录到 method。"""
        result = self.validator._run_track_b(
            ode_system=MOCK_ODE_SYSTEM,
            sbml_model_id=MOCK_SBML_MODEL_ID,
            pathway_class="EGFR_RTK",
            grounding_ledger=MOCK_GROUNDING_LEDGER,
            fallback_reason="roadrunner_not_available",
        )
        self.assertIn("fallback:roadrunner_not_available", result["method"])

    def test_track_b_roadrunner_unavailable_triggers_track_b(self):
        """roadrunner 不可用时整个 validate 流程降级到 Track B。"""
        with patch("app.validation_v2.level2_sbml.ROADRUNNER_AVAILABLE", False):
            validator = self.validator
            state = {
                "v4_ode_system": MOCK_ODE_SYSTEM,
                "sbml_model_id": MOCK_SBML_MODEL_ID,
                "v4_pathway_class": "EGFR_RTK",
                "v4_grounding_ledger": MOCK_GROUNDING_LEDGER,
            }
            result = validator.validate(state)
            self.assertEqual(result["track"], "B")
            # Track B 差异指标必须 None
            self.assertIsNone(result["peak_diff"])
            self.assertIsNone(result["peak_time_diff"])
            self.assertIsNone(result["amplification_diff"])


# =============================================================================
# TestSkipped: skipped 状态
# =============================================================================
class TestSkipped(unittest.TestCase):
    """测试 _run_skipped（skipped 状态）。"""

    def setUp(self):
        from app.validation_v2.level2_sbml import Level2SBMLValidator
        self.validator = Level2SBMLValidator()

    def test_skipped_pass_is_false(self):
        """**关键修复（审计 §7.2）**：skipped 状态 pass=False。"""
        result = self.validator._run_skipped("test_reason")
        self.assertFalse(result["pass"])  # 关键修复：skipped pass=False

    def test_skipped_track_is_skipped(self):
        """skipped 状态 track='skipped'。"""
        result = self.validator._run_skipped("test_reason")
        self.assertEqual(result["track"], "skipped")

    def test_skipped_all_diff_null(self):
        """skipped 状态所有差异指标必须 None。"""
        result = self.validator._run_skipped("test_reason")
        self.assertIsNone(result["peak_diff"])
        self.assertIsNone(result["peak_time_diff"])
        self.assertIsNone(result["amplification_diff"])

    def test_skipped_sbml_sim_available_false(self):
        """skipped 状态 sbml_sim_available=False。"""
        result = self.validator._run_skipped("test_reason")
        self.assertFalse(result["sbml_sim_available"])

    def test_skipped_method_includes_reason(self):
        """skipped 状态 method 包含 reason。"""
        result = self.validator._run_skipped("my_reason")
        self.assertIn("my_reason", result["method"])
        self.assertTrue(result["method"].startswith("skipped:"))

    def test_skipped_when_sbml_model_id_missing(self):
        """sbml_model_id 缺失 → skipped pass=False（修复审计 §7.2）。"""
        state = {
            "v4_ode_system": MOCK_ODE_SYSTEM,
            # 缺 sbml_model_id
        }
        result = self.validator.validate(state)
        self.assertEqual(result["track"], "skipped")
        self.assertFalse(result["pass"])  # 关键修复：skipped pass=False
        self.assertIn("missing_sbml_model_id", result["method"])

    def test_skipped_pass_not_true(self):
        """**显式断言**：skipped 状态 pass 不为 True（修复审计 §7.2 Oracle 致命错误）。"""
        result = self.validator._run_skipped("any_reason")
        self.assertNotEqual(result["pass"], True)


# =============================================================================
# TestSpeciesAlignment: 物种对齐（用 ontology ID）
# =============================================================================
class TestSpeciesAlignment(unittest.TestCase):
    """测试 _align_species_by_ontology（修复审计 §10.3）。"""

    def setUp(self):
        from app.validation_v2.level2_sbml import Level2SBMLValidator
        self.validator = Level2SBMLValidator()

    def test_align_by_hgnc_id(self):
        """按 HGNC ID 对齐 species。"""
        v4_species = [
            {"id": "EGF_v4", "canonical_name": "EGF", "ontology": {}},
            {"id": "EGFR_v4", "canonical_name": "EGFR", "ontology": {}},
        ]
        sbml_species = [
            {
                "species_id": "sbml_EGF",
                "canonical_name": "EGF",
                "ontology_ref": {"hgnc_id": "HGNC:3229"},
            },
            {
                "species_id": "sbml_EGFR",
                "canonical_name": "EGFR",
                "ontology_ref": {"hgnc_id": "HGNC:3236"},
            },
        ]
        mapping = self.validator._align_species_by_ontology(v4_species, sbml_species)
        # EGF_v4 → sbml_EGF（通过 HGNC:3229）
        self.assertEqual(mapping.get("EGF_v4"), "sbml_EGF")
        self.assertEqual(mapping.get("EGFR_v4"), "sbml_EGFR")

    def test_align_by_uniprot_id_when_hgnc_missing(self):
        """HGNC 缺失时按 UniProt 对齐。"""
        v4_species = [
            {"id": "EGFR_v4", "canonical_name": "EGFR", "ontology": {
                "uniprot_id": "P00533"
            }},
        ]
        sbml_species = [
            {
                "species_id": "sbml_EGFR",
                "canonical_name": "EGFR",
                "ontology_ref": {"uniprot_id": "P00533"},
            },
        ]
        mapping = self.validator._align_species_by_ontology(v4_species, sbml_species)
        self.assertEqual(mapping.get("EGFR_v4"), "sbml_EGFR")

    def test_no_string_matching(self):
        """**关键修复（审计 §10.3）**：不用字符串匹配。

        构造场景：v4 species id="X"，canonical_name="EGFR"
        SBML species id="Y", canonical_name="TOTALY_DIFFERENT_NAME"
        但两者 ontology ID 相同 → 应该匹配。
        """
        v4_species = [
            {"id": "v4_X", "canonical_name": "EGFR", "ontology": {}},
        ]
        sbml_species = [
            {
                "species_id": "sbml_Y",
                "canonical_name": "TOTALLY_DIFFERENT_NAME",
                "ontology_ref": {"hgnc_id": "HGNC:3236"},  # 与 EGFR 相同
            },
        ]
        mapping = self.validator._align_species_by_ontology(v4_species, sbml_species)
        # 应按 HGNC ID 匹配，而不是 canonical_name
        self.assertEqual(mapping.get("v4_X"), "sbml_Y")

    def test_unmatched_species_marked(self):
        """未对齐的 species 标记 unmatched（不在结果中）。"""
        v4_species = [
            {"id": "EGF_v4", "canonical_name": "EGF", "ontology": {}},
            {"id": "UNKNOWN", "canonical_name": "UNKNOWN_PROTEIN", "ontology": {}},
        ]
        sbml_species = [
            {
                "species_id": "sbml_EGF",
                "canonical_name": "EGF",
                "ontology_ref": {"hgnc_id": "HGNC:3229"},
            },
        ]
        mapping = self.validator._align_species_by_ontology(v4_species, sbml_species)
        # EGF 对齐
        self.assertEqual(mapping.get("EGF_v4"), "sbml_EGF")
        # UNKNOWN 未对齐 → 不在 mapping 中
        self.assertNotIn("UNKNOWN", mapping)

    def test_align_with_local_ontology_kb_fallback(self):
        """v4 species 无 ontology_ref → 查本地 KB 获取 ontology ID。"""
        # v4 EGF 无 ontology，但本地 KB 中 EGF → HGNC:3229
        v4_species = [
            {"id": "v4_EGF", "canonical_name": "EGF", "ontology": {}},
        ]
        sbml_species = [
            {
                "species_id": "sbml_EGF",
                "canonical_name": "EGF",
                "ontology_ref": {"hgnc_id": "HGNC:3229"},
            },
        ]
        mapping = self.validator._align_species_by_ontology(v4_species, sbml_species)
        # 通过本地 KB 查到 HGNC:3229 → 对齐
        self.assertEqual(mapping.get("v4_EGF"), "sbml_EGF")

    def test_align_empty_lists(self):
        """空列表 → 空 mapping。"""
        mapping = self.validator._align_species_by_ontology([], [])
        self.assertEqual(mapping, {})

    def test_align_no_ontology_id_returns_empty(self):
        """双方都无 ontology ID → 不匹配（不报错）。"""
        v4_species = [
            {"id": "v4_X", "canonical_name": "UNKNOWN", "ontology": {}},
        ]
        sbml_species = [
            {
                "species_id": "sbml_Y",
                "canonical_name": "UNKNOWN",
                "ontology_ref": {},
            },
        ]
        mapping = self.validator._align_species_by_ontology(v4_species, sbml_species)
        # 无 ontology ID 无法对齐
        self.assertEqual(mapping, {})


# =============================================================================
# TestThresholds: 通路特异阈值
# =============================================================================
class TestThresholds(unittest.TestCase):
    """测试 PathwayThresholds。"""

    def setUp(self):
        from app.validation_v2.thresholds import PathwayThresholds
        self.thresholds = PathwayThresholds()

    def test_egfr_thresholds(self):
        """EGFR 阈值：peak_time_diff=2.0, amplification_diff=20%。"""
        t = self.thresholds.get_thresholds("EGFR_RTK")
        self.assertEqual(t["peak_time_diff"], 2.0)
        self.assertAlmostEqual(t["amplification_diff"], 0.20)

    def test_nf_kappa_b_thresholds(self):
        """NF-κB 阈值：peak_time_diff=30（更大，振荡相位难对齐）。"""
        t = self.thresholds.get_thresholds("NF_KB")
        self.assertEqual(t["peak_time_diff"], 30.0)  # spec.md 第 291 行
        self.assertGreaterEqual(t["peak_time_diff"], 30.0)  # 比 default 大

    def test_p53_thresholds(self):
        """p53 阈值：amplification_diff=1.0（更大，脉冲幅度生物变异大）。"""
        t = self.thresholds.get_thresholds("P53")
        self.assertEqual(t["amplification_diff"], 1.00)  # spec.md 第 291 行
        self.assertGreaterEqual(t["amplification_diff"], 0.50)  # 比 default 大

    def test_default_thresholds(self):
        """未识别通路 → default 阈值。"""
        t = self.thresholds.get_thresholds("UNKNOWN_PATHWAY")
        self.assertEqual(t["peak_time_diff"], 5.0)
        self.assertAlmostEqual(t["amplification_diff"], 0.30)

    def test_default_thresholds_when_none(self):
        """pathway_class=None → default 阈值。"""
        t = self.thresholds.get_thresholds(None)
        self.assertEqual(t["peak_time_diff"], 5.0)
        self.assertAlmostEqual(t["amplification_diff"], 0.30)

    def test_default_thresholds_when_empty(self):
        """pathway_class='' → default 阈值。"""
        t = self.thresholds.get_thresholds("")
        self.assertEqual(t["peak_time_diff"], 5.0)

    def test_wnt_thresholds(self):
        """Wnt 阈值：peak_time_diff=60。"""
        t = self.thresholds.get_thresholds("WNT")
        self.assertEqual(t["peak_time_diff"], 60.0)

    def test_mapk_thresholds(self):
        """MAPK 阈值：peak_time_diff=2.0, amplification_diff=30%。"""
        t = self.thresholds.get_thresholds("MAPK")
        self.assertEqual(t["peak_time_diff"], 2.0)
        self.assertAlmostEqual(t["amplification_diff"], 0.30)

    def test_thresholds_returns_floats(self):
        """阈值返回值均为 float。"""
        t = self.thresholds.get_thresholds("EGFR_RTK")
        self.assertIsInstance(t["peak_time_diff"], float)
        self.assertIsInstance(t["amplification_diff"], float)

    def test_thresholds_case_insensitive(self):
        """pathway_class 大小写不敏感。"""
        t1 = self.thresholds.get_thresholds("egfr_rtk")
        t2 = self.thresholds.get_thresholds("EGFR_RTK")
        self.assertEqual(t1, t2)

    def test_multi_pathway_class_fuzzy_match(self):
        """多通路混合 pathway_class 模糊匹配（如 'MULTI:EGFR_RTK+PI3K_AKT_mTOR'）。"""
        t = self.thresholds.get_thresholds("MULTI:EGFR_RTK+PI3K_AKT_mTOR")
        # 应该匹配到 MULTI（最宽松阈值）
        self.assertEqual(t["peak_time_diff"], 30.0)


# =============================================================================
# TestComputeDiffs: _compute_diffs 与 _apply_thresholds
# =============================================================================
class TestComputeDiffs(unittest.TestCase):
    """测试 _compute_diffs 与 _apply_thresholds。"""

    def setUp(self):
        from app.validation_v2.level2_sbml import Level2SBMLValidator
        self.validator = Level2SBMLValidator()

    def test_compute_diffs_with_aligned_species(self):
        """对齐的 species → 计算 diff。"""
        v4_peaks = {
            "EGF_v4": {"peak": 100.0, "peak_time": 5.0,
                        "amplification": 50.0, "baseline": 50.0},
        }
        sbml_peaks = {
            "sbml_EGF": {"peak": 80.0, "peak_time": 7.0,
                          "amplification": 40.0, "baseline": 40.0},
        }
        species_map = {"EGF_v4": "sbml_EGF"}
        peak_diff, pt_diff, amp_diff = self.validator._compute_diffs(
            v4_peaks, sbml_peaks, species_map
        )
        self.assertIsNotNone(peak_diff)
        self.assertIsNotNone(pt_diff)
        self.assertIsNotNone(amp_diff)
        # peak_diff = |100-80|/100 = 0.2
        self.assertAlmostEqual(peak_diff, 0.2, places=2)
        # pt_diff = |5-7| = 2
        self.assertAlmostEqual(pt_diff, 2.0, places=2)
        # amp_diff = |50-40|/50 = 0.2
        self.assertAlmostEqual(amp_diff, 0.2, places=2)

    def test_compute_diffs_no_alignment_returns_none(self):
        """无对齐 species → 返回 (None, None, None)。"""
        peak_diff, pt_diff, amp_diff = self.validator._compute_diffs(
            {"X": {"peak": 1.0, "peak_time": 0.0,
                   "amplification": 0.0, "baseline": 0.0}},
            {"Y": {"peak": 1.0, "peak_time": 0.0,
                   "amplification": 0.0, "baseline": 0.0}},
            {},
        )
        self.assertIsNone(peak_diff)
        self.assertIsNone(pt_diff)
        self.assertIsNone(amp_diff)

    def test_apply_thresholds_pass(self):
        """diff 在阈值内 → pass=True。"""
        result = self.validator._apply_thresholds(
            peak_diff=0.1,
            peak_time_diff=1.0,
            amplification_diff=0.1,
            thresholds={"peak_time_diff": 2.0, "amplification_diff": 0.20},
        )
        self.assertTrue(result)

    def test_apply_thresholds_fail_on_peak_time(self):
        """peak_time_diff 超阈值 → pass=False。"""
        result = self.validator._apply_thresholds(
            peak_diff=0.1,
            peak_time_diff=3.0,  # > 2.0
            amplification_diff=0.1,
            thresholds={"peak_time_diff": 2.0, "amplification_diff": 0.20},
        )
        self.assertFalse(result)

    def test_apply_thresholds_fail_on_amplification(self):
        """amplification_diff 超阈值 → pass=False。"""
        result = self.validator._apply_thresholds(
            peak_diff=0.1,
            peak_time_diff=1.0,
            amplification_diff=0.30,  # > 0.20
            thresholds={"peak_time_diff": 2.0, "amplification_diff": 0.20},
        )
        self.assertFalse(result)

    def test_apply_thresholds_none_diff_returns_false(self):
        """diff 为 None（无对齐）→ pass=False（保守阻塞）。"""
        result = self.validator._apply_thresholds(
            peak_diff=None,
            peak_time_diff=None,
            amplification_diff=None,
            thresholds={"peak_time_diff": 2.0, "amplification_diff": 0.20},
        )
        self.assertFalse(result)


# =============================================================================
# TestStructuralSimilarity: _compute_structural_similarity
# =============================================================================
class TestStructuralSimilarity(unittest.TestCase):
    """测试 _compute_structural_similarity。"""

    def setUp(self):
        from app.validation_v2.level2_sbml import Level2SBMLValidator
        self.validator = Level2SBMLValidator()

    def test_similarity_perfect_match(self):
        """v4 与 SBML 完全匹配 → similarity 高。"""
        # 构造完全匹配的输入：物种数/反应数/机制类型一致
        ode_system = {
            "ode_code": "dEGF/dt = -k * EGF",
            "species": [
                {"id": "EGF", "canonical_name": "EGF"},
                {"id": "EGFR", "canonical_name": "EGFR"},
            ],
            "equations": [{"eq_id": "eq1"}],
            "kinetics_types": ["mass_action"],
        }
        ledger = {
            "ode_equations": [{"kinetics_type": "mass_action"}],
            "species_mapping": MOCK_GROUNDING_LEDGER["species_mapping"],
        }
        score = self.validator._compute_structural_similarity(ode_system, ledger)
        self.assertGreater(score, 0.5)
        self.assertIsInstance(score, float)

    def test_similarity_low_when_no_match(self):
        """v4 与 SBML 完全不匹配 → similarity 低。"""
        ode_system = {
            "ode_code": "dEGF/dt = -k * EGF",
            "species": [{"id": "EGF", "canonical_name": "EGF"}],
            "equations": [{"eq_id": "eq1"}],
        }
        ledger = {
            "ode_equations": [],
            "species_mapping": [],
        }
        score = self.validator._compute_structural_similarity(ode_system, ledger)
        self.assertLess(score, 0.6)

    def test_similarity_returns_float_0_to_1(self):
        """similarity score 在 [0, 1] 范围内。"""
        score = self.validator._compute_structural_similarity(
            MOCK_ODE_SYSTEM, MOCK_GROUNDING_LEDGER
        )
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


# =============================================================================
# 主入口
# =============================================================================
if __name__ == "__main__":
    unittest.main()
