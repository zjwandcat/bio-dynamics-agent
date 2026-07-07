# BioDynamics Agent v4 - Sensitivity Analysis 单元测试 (Phase 5 / Task 5.8.6)
#
# 测试 SensitivityAnalyzer 主类 + LocalSensitivityAnalyzer + SobolAnalyzer
# + MorrisAnalyzer + LangGraph hook + Feature Flag 隔离 + 降级路径。
#
# 测试用例（32 个，>= 30 要求）：
#   - TestLocalSensitivityAnalyzer: local sensitivity 主类 (6)
#   - TestSobolAnalyzer: sobol 主类 + SALib 双路径 (6)
#   - TestMorrisAnalyzer: morris 主类 + SALib 双路径 (6)
#   - TestSensitivityAnalyzer: 编排器主类 (6)
#   - TestSensitivityHookNode: Feature Flag + hook 行为 (5)
#   - TestSensitivityResultDataclasses: dataclass 字段完整性 (3)
#
# 运行：cd backend && python -m pytest tests/test_sensitivity.py -v

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# 确保能导入 app 包（backend/ 在 sys.path）
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import numpy as np  # type: ignore[import-untyped]

from app.config import SALIB_AVAILABLE
from app.sensitivity.local_sensitivity import (
    LocalSensitivityAnalyzer,
    LocalSensitivityResult,
)
from app.sensitivity.morris_analyzer import MorrisAnalyzer, MorrisResult
from app.sensitivity.sensitivity_analyzer import (
    SensitivityAnalyzer,
    sensitivity_hook_node,
)
from app.sensitivity.sobol_analyzer import SobolAnalyzer, SobolResult


# =============================================================================
# Mock 数据
# =============================================================================

# --- 正常 state（含 v4_ode_system + parameters + v4_calibration_result）---
MOCK_STATE_NORMAL = {
    "sbml_model_id": "BIOMD0000000205",
    "v4_ode_system": {
        "ode_code": "d[EGFR]/dt = -k1*EGF*EGFR + k2*EGF_EGFR",
        "equations": ["d[EGFR]/dt = -k1*EGF*EGFR + k2*EGF_EGFR"],
        "parameters": {"k1": 0.1, "k2": 0.01},
    },
    "parameters": {"k1": 0.1, "k2": 0.01, "k3": 1.0},
    "v4_calibration_result": {
        "calibrated_params": {"k1": 0.12, "k2": 0.009},
        "method": "least_squares",
    },
}


def _make_mock_salib_sobol():
    """构造 mock SALib 模块（Sobol 路径）。"""
    mock_salib = MagicMock()
    # saltelli.sample 返回 (N * (2p+2), p) 的数组（p=2 参数时 2p+2=6）
    mock_salib.sample.saltelli.sample.return_value = np.array(
        [[0.05, 0.005], [0.1, 0.01], [0.15, 0.015], [0.08, 0.008]]
    )
    # sobol.analyze 返回含 S1 / ST 的 dict
    mock_salib.analyze.sobol.analyze.return_value = {
        "S1": np.array([0.5, 0.3]),
        "ST": np.array([0.6, 0.4]),
        "S1_conf": np.array([0.05, 0.03]),
        "ST_conf": np.array([0.06, 0.04]),
    }
    return mock_salib


def _make_mock_salib_morris():
    """构造 mock SALib 模块（Morris 路径）。"""
    mock_salib = MagicMock()
    # morris.sample 返回 (N * (p+1), p) 的数组
    mock_salib.sample.morris.sample.return_value = np.array(
        [[0.05, 0.005], [0.1, 0.01], [0.15, 0.015], [0.08, 0.008]]
    )
    # morris.analyze 返回含 mu / sigma / mu_star 的 dict
    mock_salib.analyze.morris.analyze.return_value = {
        "mu": np.array([0.3, 0.2]),
        "sigma": np.array([0.1, 0.05]),
        "mu_star": np.array([0.4, 0.25]),
    }
    return mock_salib


# =============================================================================
# TestLocalSensitivityAnalyzer: local sensitivity 主类
# =============================================================================
class TestLocalSensitivityAnalyzer(unittest.TestCase):
    """测试 LocalSensitivityAnalyzer 主类。"""

    def test_analyze_two_params_returns_dict(self):
        """两个参数 → 返回 dict（每个参数 LocalSensitivityResult）。"""
        analyzer = LocalSensitivityAnalyzer(delta=0.01, relative=True)
        params = {"k1": 0.1, "k2": 0.01}
        # model_func: 返回参数乘积
        model_func = lambda p: p.get("k1", 0) * p.get("k2", 0)
        results = analyzer.analyze(params, model_func)
        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), 2)
        self.assertIn("k1", results)
        self.assertIn("k2", results)
        self.assertIsInstance(results["k1"], LocalSensitivityResult)
        self.assertIsInstance(results["k2"], LocalSensitivityResult)

    def test_analyze_single_param_correct_sensitivity(self):
        """单参数 → sensitivity 正确（forward difference 相对）。"""
        analyzer = LocalSensitivityAnalyzer(delta=0.01, relative=True)
        # model_func: f(p) = 2 * p["k1"]
        model_func = lambda p: 2.0 * p.get("k1", 0)
        results = analyzer.analyze({"k1": 1.0}, model_func)
        result = results["k1"]
        # baseline = 2.0 * 1.0 = 2.0
        # perturbed = 2.0 * (1.0 * 1.01) = 2.02
        # sensitivity = (2.02 - 2.0) / 2.0 = 0.01
        self.assertAlmostEqual(result.sensitivity, 0.01, places=6)
        self.assertAlmostEqual(result.baseline, 2.0, places=6)
        self.assertAlmostEqual(result.perturbed, 2.02, places=6)
        self.assertEqual(result.method, "forward_difference_relative")

    def test_analyze_baseline_zero_no_exception(self):
        """baseline=0 → 不抛异常（用绝对差）。"""
        analyzer = LocalSensitivityAnalyzer(delta=0.01, relative=True)
        # model_func: f(p) = 0（baseline 永远为 0）
        model_func = lambda p: 0.0
        results = analyzer.analyze({"k1": 1.0}, model_func)
        result = results["k1"]
        # baseline=0 → sensitivity = perturbed - baseline = 0.0 - 0.0 = 0.0
        self.assertEqual(result.sensitivity, 0.0)
        self.assertEqual(result.baseline, 0.0)
        self.assertEqual(result.perturbed, 0.0)

    def test_analyze_exception_param_returns_zero(self):
        """异常参数（model_func 抛异常）→ sensitivity=0.0。"""
        analyzer = LocalSensitivityAnalyzer(delta=0.01, relative=True)

        def model_func(p):
            if p.get("k2") is not None:
                raise ValueError("k2 boom")
            return 1.0

        results = analyzer.analyze({"k1": 1.0, "k2": 0.5}, model_func)
        # baseline 用原参数 → 不抛异常（k2 未扰动）
        # 扰动 k2 时 model_func 抛异常 → sensitivity=0.0
        self.assertEqual(results["k2"].sensitivity, 0.0)
        # k1 正常
        self.assertIsInstance(results["k1"], LocalSensitivityResult)

    def test_analyze_nan_model_returns_zero(self):
        """NaN/Inf 模型输出 → sensitivity=0.0。"""
        analyzer = LocalSensitivityAnalyzer(delta=0.01, relative=True)
        # baseline 返回 NaN
        model_func = lambda p: float("nan")
        results = analyzer.analyze({"k1": 1.0}, model_func)
        # baseline 非有限 → 所有参数 sensitivity=0.0
        self.assertEqual(results["k1"].sensitivity, 0.0)

    def test_analyze_absolute_mode_uses_absolute_delta(self):
        """relative=False → 用绝对 delta（p + delta）。"""
        analyzer = LocalSensitivityAnalyzer(delta=0.1, relative=False)
        # model_func: f(p) = 3 * p["k1"]
        model_func = lambda p: 3.0 * p.get("k1", 0)
        results = analyzer.analyze({"k1": 2.0}, model_func)
        result = results["k1"]
        # baseline = 3 * 2.0 = 6.0
        # perturbed = 3 * (2.0 + 0.1) = 6.3
        # sensitivity = (6.3 - 6.0) / 6.0 = 0.05
        self.assertAlmostEqual(result.sensitivity, 0.05, places=6)
        self.assertEqual(result.method, "forward_difference_absolute")


# =============================================================================
# TestSobolAnalyzer: sobol 主类
# =============================================================================
class TestSobolAnalyzer(unittest.TestCase):
    """测试 SobolAnalyzer 主类。"""

    def test_analyze_salib_available_returns_sobol(self):
        """SALIB_AVAILABLE=True（mock）→ method="sobol"。"""
        mock_salib = _make_mock_salib_sobol()
        with patch.dict(sys.modules, {"SALib": mock_salib}):
            with patch("app.sensitivity.sobol_analyzer.SALIB_AVAILABLE", True):
                analyzer = SobolAnalyzer(n_samples=4, seed=42)
                result = analyzer.analyze(
                    {"k1": 0.1, "k2": 0.01}, lambda p: 1.0
                )
        self.assertEqual(result.method, "sobol")
        self.assertIn("k1", result.S1)
        self.assertIn("k2", result.S1)
        self.assertIn("k1", result.ST)
        self.assertIn("k2", result.ST)
        self.assertEqual(result.n_samples, 4)
        self.assertEqual(len(result.warnings), 0)

    def test_analyze_salib_not_available_returns_skipped(self):
        """SALIB_AVAILABLE=False（mock）→ method="skipped" + warning。"""
        with patch("app.sensitivity.sobol_analyzer.SALIB_AVAILABLE", False):
            analyzer = SobolAnalyzer()
            result = analyzer.analyze(
                {"k1": 0.1, "k2": 0.01}, lambda p: 1.0
            )
        self.assertEqual(result.method, "skipped")
        self.assertGreater(len(result.warnings), 0)
        self.assertIn("SALib", result.warnings[0])

    def test_analyze_exception_returns_skipped(self):
        """SALib 抛异常 → method="skipped" + warning。"""
        mock_salib = MagicMock()
        mock_salib.sample.saltelli.sample.side_effect = RuntimeError("saltelli boom")
        with patch.dict(sys.modules, {"SALib": mock_salib}):
            with patch("app.sensitivity.sobol_analyzer.SALIB_AVAILABLE", True):
                analyzer = SobolAnalyzer(n_samples=4)
                result = analyzer.analyze(
                    {"k1": 0.1, "k2": 0.01}, lambda p: 1.0
                )
        self.assertEqual(result.method, "skipped")
        self.assertGreater(len(result.warnings), 0)

    def test_analyze_empty_params_returns_skipped(self):
        """空 params → method="skipped"。"""
        with patch("app.sensitivity.sobol_analyzer.SALIB_AVAILABLE", True):
            analyzer = SobolAnalyzer()
            result = analyzer.analyze({}, lambda p: 1.0)
        self.assertEqual(result.method, "skipped")
        self.assertGreater(len(result.warnings), 0)

    def test_analyze_custom_problem_uses_passed_problem(self):
        """自定义 problem → 使用传入 problem。"""
        mock_salib = _make_mock_salib_sobol()
        custom_problem = {
            "num_vars": 2,
            "names": ["k1", "k2"],
            "boundaries": [[0.05, 0.15], [0.005, 0.015]],
        }
        with patch.dict(sys.modules, {"SALib": mock_salib}):
            with patch("app.sensitivity.sobol_analyzer.SALIB_AVAILABLE", True):
                analyzer = SobolAnalyzer(n_samples=4, seed=42)
                result = analyzer.analyze(
                    {"k1": 0.1, "k2": 0.01},
                    lambda p: 1.0,
                    problem=custom_problem,
                )
        self.assertEqual(result.method, "sobol")
        # 验证 sample 被调用，且 problem 是 custom_problem
        mock_salib.sample.saltelli.sample.assert_called_once()
        call_args = mock_salib.sample.saltelli.sample.call_args
        passed_problem = call_args[0][0] if call_args[0] else call_args[1].get("problem")
        self.assertIs(passed_problem, custom_problem)

    def test_sobol_result_dataclass_fields_complete(self):
        """SobolResult dataclass 字段完整。"""
        result = SobolResult(
            S1={"k1": 0.5},
            ST={"k1": 0.6},
            S2=None,
            method="sobol",
            n_samples=100,
            warnings=["test"],
        )
        self.assertEqual(result.S1, {"k1": 0.5})
        self.assertEqual(result.ST, {"k1": 0.6})
        self.assertIsNone(result.S2)
        self.assertEqual(result.method, "sobol")
        self.assertEqual(result.n_samples, 100)
        self.assertEqual(result.warnings, ["test"])


# =============================================================================
# TestMorrisAnalyzer: morris 主类
# =============================================================================
class TestMorrisAnalyzer(unittest.TestCase):
    """测试 MorrisAnalyzer 主类。"""

    def test_analyze_salib_available_returns_morris(self):
        """SALIB_AVAILABLE=True（mock）→ method="morris"。"""
        mock_salib = _make_mock_salib_morris()
        with patch.dict(sys.modules, {"SALib": mock_salib}):
            with patch("app.sensitivity.morris_analyzer.SALIB_AVAILABLE", True):
                analyzer = MorrisAnalyzer(n_trajectories=4, seed=42)
                result = analyzer.analyze(
                    {"k1": 0.1, "k2": 0.01}, lambda p: 1.0
                )
        self.assertEqual(result.method, "morris")
        self.assertIn("k1", result.mu)
        self.assertIn("k2", result.mu)
        self.assertIn("k1", result.sigma)
        self.assertIn("k2", result.sigma)
        self.assertIn("k1", result.mu_star)
        self.assertIn("k2", result.mu_star)
        self.assertEqual(result.n_trajectories, 4)
        self.assertEqual(len(result.warnings), 0)

    def test_analyze_salib_not_available_returns_skipped(self):
        """SALIB_AVAILABLE=False（mock）→ method="skipped" + warning。"""
        with patch("app.sensitivity.morris_analyzer.SALIB_AVAILABLE", False):
            analyzer = MorrisAnalyzer()
            result = analyzer.analyze(
                {"k1": 0.1, "k2": 0.01}, lambda p: 1.0
            )
        self.assertEqual(result.method, "skipped")
        self.assertGreater(len(result.warnings), 0)
        self.assertIn("SALib", result.warnings[0])

    def test_analyze_exception_returns_skipped(self):
        """SALib 抛异常 → method="skipped" + warning。"""
        mock_salib = MagicMock()
        mock_salib.sample.morris.sample.side_effect = RuntimeError("morris boom")
        with patch.dict(sys.modules, {"SALib": mock_salib}):
            with patch("app.sensitivity.morris_analyzer.SALIB_AVAILABLE", True):
                analyzer = MorrisAnalyzer(n_trajectories=4)
                result = analyzer.analyze(
                    {"k1": 0.1, "k2": 0.01}, lambda p: 1.0
                )
        self.assertEqual(result.method, "skipped")
        self.assertGreater(len(result.warnings), 0)

    def test_analyze_empty_params_returns_skipped(self):
        """空 params → method="skipped"。"""
        with patch("app.sensitivity.morris_analyzer.SALIB_AVAILABLE", True):
            analyzer = MorrisAnalyzer()
            result = analyzer.analyze({}, lambda p: 1.0)
        self.assertEqual(result.method, "skipped")
        self.assertGreater(len(result.warnings), 0)

    def test_analyze_custom_problem_uses_passed_problem(self):
        """自定义 problem → 使用传入 problem。"""
        mock_salib = _make_mock_salib_morris()
        custom_problem = {
            "num_vars": 2,
            "names": ["k1", "k2"],
            "boundaries": [[0.05, 0.15], [0.005, 0.015]],
        }
        with patch.dict(sys.modules, {"SALib": mock_salib}):
            with patch("app.sensitivity.morris_analyzer.SALIB_AVAILABLE", True):
                analyzer = MorrisAnalyzer(n_trajectories=4, seed=42)
                result = analyzer.analyze(
                    {"k1": 0.1, "k2": 0.01},
                    lambda p: 1.0,
                    problem=custom_problem,
                )
        self.assertEqual(result.method, "morris")
        mock_salib.sample.morris.sample.assert_called_once()
        call_args = mock_salib.sample.morris.sample.call_args
        passed_problem = call_args[0][0] if call_args[0] else call_args[1].get("problem")
        self.assertIs(passed_problem, custom_problem)

    def test_morris_result_dataclass_fields_complete(self):
        """MorrisResult dataclass 字段完整。"""
        result = MorrisResult(
            mu={"k1": 0.3},
            sigma={"k1": 0.1},
            mu_star={"k1": 0.4},
            method="morris",
            n_trajectories=10,
            warnings=["test"],
        )
        self.assertEqual(result.mu, {"k1": 0.3})
        self.assertEqual(result.sigma, {"k1": 0.1})
        self.assertEqual(result.mu_star, {"k1": 0.4})
        self.assertEqual(result.method, "morris")
        self.assertEqual(result.n_trajectories, 10)
        self.assertEqual(result.warnings, ["test"])


# =============================================================================
# TestSensitivityAnalyzer: 编排器主类
# =============================================================================
class TestSensitivityAnalyzer(unittest.TestCase):
    """测试 SensitivityAnalyzer 编排器主类。"""

    def test_analyze_normal_returns_full_report(self):
        """正常调用（mock 子分析器全成功）→ 返回 dict 含 local + sobol + morris + method。"""
        mock_local = MagicMock(spec=LocalSensitivityAnalyzer)
        mock_local.analyze.return_value = {
            "k1": LocalSensitivityResult(
                param_name="k1",
                sensitivity=0.5,
                baseline=1.0,
                perturbed=1.5,
                method="forward_difference_relative",
            ),
        }
        mock_sobol = MagicMock(spec=SobolAnalyzer)
        mock_sobol.analyze.return_value = SobolResult(
            S1={"k1": 0.5},
            ST={"k1": 0.6},
            method="sobol",
            n_samples=100,
        )
        mock_morris = MagicMock(spec=MorrisAnalyzer)
        mock_morris.analyze.return_value = MorrisResult(
            mu={"k1": 0.3},
            sigma={"k1": 0.1},
            mu_star={"k1": 0.4},
            method="morris",
            n_trajectories=10,
        )
        analyzer = SensitivityAnalyzer(
            local_analyzer=mock_local,
            sobol_analyzer=mock_sobol,
            morris_analyzer=mock_morris,
        )
        result = analyzer.analyze(MOCK_STATE_NORMAL)
        self.assertIn("v4_sensitivity_report", result)
        report = result["v4_sensitivity_report"]
        self.assertIn("local_sensitivity", report)
        self.assertIn("sobol", report)
        self.assertIn("morris", report)
        self.assertIn("method", report)
        self.assertEqual(report["method"], "full")
        self.assertEqual(report["local_sensitivity"], {"k1": 0.5})

    @patch("app.sensitivity.sensitivity_analyzer.SALIB_AVAILABLE", False)
    def test_analyze_salib_not_available_returns_local_only(self):
        """SALIB_AVAILABLE=False（实际环境）→ method="local_only"。"""
        # 使用默认子分析器（SALIB_AVAILABLE=False 时 sobol/morris 自动 skipped）
        analyzer = SensitivityAnalyzer()
        result = analyzer.analyze({"parameters": {"k1": 0.1, "k2": 0.01}})
        report = result["v4_sensitivity_report"]
        self.assertEqual(report["method"], "local_only")
        self.assertFalse(report["salib_available"])
        # sobol / morris 应为 skipped
        self.assertIsNotNone(report["sobol"])
        self.assertEqual(report["sobol"]["method"], "skipped")
        self.assertIsNotNone(report["morris"])
        self.assertEqual(report["morris"]["method"], "skipped")

    @patch("app.sensitivity.sensitivity_analyzer.SALIB_AVAILABLE", True)
    def test_analyze_salib_available_returns_full(self):
        """SALIB_AVAILABLE=True（mock 子分析器全成功）→ method="full" + salib_available=True。"""
        mock_local = MagicMock(spec=LocalSensitivityAnalyzer)
        mock_local.analyze.return_value = {
            "k1": LocalSensitivityResult(param_name="k1", sensitivity=0.5),
        }
        mock_sobol = MagicMock(spec=SobolAnalyzer)
        mock_sobol.analyze.return_value = SobolResult(
            S1={"k1": 0.5}, ST={"k1": 0.6}, method="sobol", n_samples=100
        )
        mock_morris = MagicMock(spec=MorrisAnalyzer)
        mock_morris.analyze.return_value = MorrisResult(
            mu={"k1": 0.3},
            sigma={"k1": 0.1},
            mu_star={"k1": 0.4},
            method="morris",
            n_trajectories=10,
        )
        analyzer = SensitivityAnalyzer(
            local_analyzer=mock_local,
            sobol_analyzer=mock_sobol,
            morris_analyzer=mock_morris,
        )
        result = analyzer.analyze({"parameters": {"k1": 0.1}})
        report = result["v4_sensitivity_report"]
        self.assertEqual(report["method"], "full")
        self.assertTrue(report["salib_available"])

    def test_analyze_no_model_func_uses_default(self):
        """无 model_func → 使用默认占位 + 不抛异常。"""
        analyzer = SensitivityAnalyzer()
        result = analyzer.analyze({"parameters": {"k1": 0.1, "k2": 0.01}})
        report = result["v4_sensitivity_report"]
        # 默认 model_func 返回参数乘积，应正常计算 local sensitivity
        self.assertIn("k1", report["local_sensitivity"])
        self.assertIn("k2", report["local_sensitivity"])
        # 不应抛异常
        self.assertNotIn("fallback", report)

    def test_analyze_empty_state_returns_fallback(self):
        """空 state → fallback 结果。"""
        analyzer = SensitivityAnalyzer()
        result = analyzer.analyze({})
        self.assertIn("v4_sensitivity_report", result)
        report = result["v4_sensitivity_report"]
        self.assertTrue(report.get("fallback"))
        self.assertEqual(report["method"], "skipped")
        self.assertEqual(report["local_sensitivity"], {})

    def test_analyze_invalid_state_type_no_exception(self):
        """state 非 dict → fallback，不抛异常。"""
        analyzer = SensitivityAnalyzer()
        result = analyzer.analyze("not_a_dict")  # type: ignore[arg-type]
        self.assertIn("v4_sensitivity_report", result)
        self.assertTrue(result["v4_sensitivity_report"].get("fallback"))


# =============================================================================
# TestSensitivityHookNode: Feature Flag + hook 行为
# =============================================================================
class TestSensitivityHookNode(unittest.TestCase):
    """测试 sensitivity_hook_node LangGraph 节点。"""

    def test_hook_flag_off_returns_empty(self):
        """V4_CALIBRATION_AGENT_ENABLED=false → hook 返回 {}。"""
        with patch("app.sensitivity.sensitivity_analyzer.settings") as mock_settings:
            mock_settings.V4_CALIBRATION_AGENT_ENABLED = False
            result = sensitivity_hook_node(MOCK_STATE_NORMAL)
        self.assertEqual(result, {})

    @patch("app.sensitivity.sensitivity_analyzer.settings")
    def test_hook_flag_on_writes_v4_field(self, mock_settings):
        """flag=true + 正常 state → 返回 {"v4_sensitivity_report": {...}}。"""
        mock_settings.V4_CALIBRATION_AGENT_ENABLED = True
        with patch.object(SensitivityAnalyzer, "analyze") as mock_analyze:
            mock_analyze.return_value = {
                "v4_sensitivity_report": {
                    "local_sensitivity": {"k1": 0.5},
                    "sobol": None,
                    "morris": None,
                    "method": "local_only",
                    "salib_available": False,
                    "warnings": [],
                }
            }
            result = sensitivity_hook_node(MOCK_STATE_NORMAL)
        self.assertIn("v4_sensitivity_report", result)

    @patch("app.sensitivity.sensitivity_analyzer.settings")
    def test_hook_exception_returns_empty(self, mock_settings):
        """flag=true + 异常 → 返回 {}，不抛异常。"""
        mock_settings.V4_CALIBRATION_AGENT_ENABLED = True
        with patch.object(
            SensitivityAnalyzer, "analyze", side_effect=RuntimeError("boom")
        ):
            result = sensitivity_hook_node(MOCK_STATE_NORMAL)
        self.assertEqual(result, {})

    def test_hook_flag_off_preserves_v3_fields(self):
        """flag=false + state 含 v3 字段 → 返回 {}，v3 字段不变（隔离验证）。"""
        v3_state = {
            "network_json": {"nodes": ["A", "B"]},
            "parameters": {"k1": 0.1},
            "ode_model": {"equations": ["dx/dt = -k1*x"]},
            "sbml_model_id": "BIOMD0000000205",
        }
        with patch("app.sensitivity.sensitivity_analyzer.settings") as mock_settings:
            mock_settings.V4_CALIBRATION_AGENT_ENABLED = False
            result = sensitivity_hook_node(v3_state)
        self.assertEqual(result, {})
        # state 字段未变（hook 返回的 dict 是新 dict，不应包含 v3 字段）
        self.assertNotIn("network_json", result)
        self.assertNotIn("parameters", result)
        self.assertNotIn("ode_model", result)

    @patch("app.sensitivity.sensitivity_analyzer.settings")
    def test_hook_flag_on_writes_correct_field_name(self, mock_settings):
        """hook 节点写入的字段名 = v4_sensitivity_report。"""
        mock_settings.V4_CALIBRATION_AGENT_ENABLED = True
        with patch.object(SensitivityAnalyzer, "analyze") as mock_analyze:
            mock_analyze.return_value = {
                "v4_sensitivity_report": {
                    "local_sensitivity": {},
                    "method": "local_only",
                    "salib_available": False,
                    "warnings": [],
                }
            }
            result = sensitivity_hook_node(MOCK_STATE_NORMAL)
        self.assertIn("v4_sensitivity_report", result)
        self.assertNotIn("v4_calibration_result", result)


# =============================================================================
# TestSensitivityResultDataclasses: dataclass 字段完整性
# =============================================================================
class TestSensitivityResultDataclasses(unittest.TestCase):
    """测试 dataclass 字段完整性。"""

    def test_local_sensitivity_result_construction(self):
        """LocalSensitivityResult 构造 + 字段访问。"""
        result = LocalSensitivityResult(
            param_name="k1",
            sensitivity=0.5,
            baseline=1.0,
            perturbed=1.5,
            method="forward_difference_relative",
        )
        self.assertEqual(result.param_name, "k1")
        self.assertEqual(result.sensitivity, 0.5)
        self.assertEqual(result.baseline, 1.0)
        self.assertEqual(result.perturbed, 1.5)
        self.assertEqual(result.method, "forward_difference_relative")

    def test_sobol_result_construction(self):
        """SobolResult 构造 + 字段访问。"""
        result = SobolResult(
            S1={"k1": 0.5, "k2": 0.3},
            ST={"k1": 0.6, "k2": 0.4},
            S2=None,
            method="sobol",
            n_samples=1024,
            warnings=[],
        )
        self.assertEqual(result.S1, {"k1": 0.5, "k2": 0.3})
        self.assertEqual(result.ST, {"k1": 0.6, "k2": 0.4})
        self.assertIsNone(result.S2)
        self.assertEqual(result.method, "sobol")
        self.assertEqual(result.n_samples, 1024)

    def test_morris_result_construction(self):
        """MorrisResult 构造 + 字段访问。"""
        result = MorrisResult(
            mu={"k1": 0.3, "k2": 0.2},
            sigma={"k1": 0.1, "k2": 0.05},
            mu_star={"k1": 0.4, "k2": 0.25},
            method="morris",
            n_trajectories=10,
            warnings=[],
        )
        self.assertEqual(result.mu, {"k1": 0.3, "k2": 0.2})
        self.assertEqual(result.sigma, {"k1": 0.1, "k2": 0.05})
        self.assertEqual(result.mu_star, {"k1": 0.4, "k2": 0.25})
        self.assertEqual(result.method, "morris")
        self.assertEqual(result.n_trajectories, 10)


if __name__ == "__main__":
    unittest.main()
