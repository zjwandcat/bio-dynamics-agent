# BioDynamics Agent v4 - Calibration Agent 单元测试 (Phase 5 / Task 5.7.6)
#
# 测试 CalibrationAgent 主类 + LeastSquaresFitter + ConfidenceIntervalEstimator
# + LangGraph hook + uncalibrated 标记 + Feature Flag 隔离。
#
# 测试用例（30 个，>= 25 要求）：
#   - TestCalibrationAgentCalibrate: calibrate() 主入口 + 异常降级 (6)
#   - TestCalibrationHookNode: Feature Flag + hook 行为 (5)
#   - TestLeastSquaresFitter: fitter 主类 + 双路径 (6)
#   - TestConfidenceIntervalEstimator: CI 主类 + 双路径 (6)
#   - TestUncalifiableMarking: uncalibrated 标记 (5)
#   - TestCalibrationModuleInit: 模块 __init__ 导出 (2)
#
# 运行：cd backend && python -m pytest tests/test_calibration_agent.py -v

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# 确保能导入 app 包（backend/ 在 sys.path）
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.calibration.calibration_agent import (
    CalibrationAgent,
    calibration_hook_node,
)
from app.calibration.confidence_interval import (
    ConfidenceInterval,
    ConfidenceIntervalEstimator,
)
from app.calibration.least_squares_fitter import FitResult, LeastSquaresFitter
from app.config import LMFIT_AVAILABLE


# =============================================================================
# Mock 数据
# =============================================================================

# --- 正常 state（含 v4_ode_system + parameters + experimental_data）---
MOCK_STATE_NORMAL = {
    "sbml_model_id": "BIOMD0000000205",
    "v4_ode_system": {
        "ode_code": "d[EGFR]/dt = -k1*EGF*EGFR + k2*EGF_EGFR",
        "equations": ["d[EGFR]/dt = -k1*EGF*EGFR + k2*EGF_EGFR"],
        "parameters": {"k1": 0.1, "k2": 0.01},
    },
    "parameters": {"k1": 0.1, "k2": 0.01, "k3": 1.0},
    "experimental_data": {
        "user_data": {
            "observations": [0.1, 0.5, 1.0, 1.5, 2.0],
            "timepoints": [0, 1, 2, 3, 4],
        },
        "observations": [0.1, 0.5, 1.0, 1.5, 2.0],
    },
    "v4_grounding_ledger": {
        "ode_equations": [
            {
                "eq_id": "eq1",
                "parameter_ids": ["k1", "k2"],
            },
            {
                "eq_id": "eq2",
                "parameter_ids": ["k3"],
            },
        ],
    },
}

# --- 成功的 FitResult mock（lmfit 路径）---
MOCK_FIT_SUCCESS_LMFIT = FitResult(
    success=True,
    params={"k1": 0.12, "k2": 0.009, "k3": 0.95},
    cost=0.001,
    nfev=15,
    message="lmfit_success",
    method="lmfit",
    residuals=[0.01, -0.02, 0.005, -0.01, 0.02],
    raw=MagicMock(),  # 模拟 lmfit.MinimizerResult
)

# --- 成功的 FitResult mock（scipy 路径）---
MOCK_FIT_SUCCESS_SCIPY = FitResult(
    success=True,
    params={"k1": 0.12, "k2": 0.009},
    cost=0.002,
    nfev=12,
    message="scipy_success",
    method="least_squares",
    residuals=[0.01, -0.02, 0.005, -0.01, 0.02],
    raw=MagicMock(),
)

# --- 失败的 FitResult mock ---
MOCK_FIT_FAILED = FitResult(
    success=False,
    params={},
    cost=float("inf"),
    nfev=0,
    message="empty_observations",
    method="lmfit",
)


# =============================================================================
# TestCalibrationAgentCalibrate: calibrate() 主入口
# =============================================================================
class TestCalibrationAgentCalibrate(unittest.TestCase):
    """测试 CalibrationAgent.calibrate() 主入口。"""

    def setUp(self):
        # 用 mock fitter + ci_estimator 隔离 lmfit/scipy 真实依赖
        self.mock_fitter = MagicMock(spec=LeastSquaresFitter)
        self.mock_ci = MagicMock(spec=ConfidenceIntervalEstimator)
        self.agent = CalibrationAgent(
            fitter=self.mock_fitter, ci_estimator=self.mock_ci
        )

    @patch("app.calibration.calibration_agent.settings")
    def test_calibrate_normal_state_returns_v4_field(self, mock_settings):
        """正常 state（v4_ode_system + sbml_model_id）→ 返回 v4_calibration_result。"""
        mock_settings.V4_CALIBRATION_AGENT_ENABLED = True
        self.mock_fitter.fit.return_value = MOCK_FIT_SUCCESS_LMFIT
        self.mock_ci.estimate.return_value = {
            "k1": {"lower": 0.1, "upper": 0.14, "std_error": 0.01, "method": "lmfit"},
            "k2": {"lower": 0.005, "upper": 0.013, "std_error": 0.002, "method": "lmfit"},
            "k3": {"lower": 0.85, "upper": 1.05, "std_error": 0.05, "method": "lmfit"},
        }
        result = self.agent.calibrate(MOCK_STATE_NORMAL)
        self.assertIn("v4_calibration_result", result)
        cal = result["v4_calibration_result"]
        self.assertIn("calibrated_params", cal)
        self.assertIn("confidence_intervals", cal)
        self.assertIn("uncalifiable", cal)
        self.assertIn("method", cal)

    @patch("app.calibration.calibration_agent.settings")
    def test_calibrate_empty_state_fallback(self, mock_settings):
        """空 state → 走 fallback 路径（v4_calibration_result 含 fallback=True）。"""
        mock_settings.V4_CALIBRATION_AGENT_ENABLED = True
        result = self.agent.calibrate({})
        self.assertIn("v4_calibration_result", result)
        cal = result["v4_calibration_result"]
        self.assertTrue(cal.get("fallback"))
        self.assertEqual(cal["method"], "none")
        self.assertEqual(cal["calibrated_params"], {})
        self.assertEqual(cal["uncalifiable"], [])

    @patch("app.calibration.calibration_agent.settings")
    def test_calibrate_invalid_state_type_no_exception(self, mock_settings):
        """state 非 dict → fallback，不抛异常。"""
        mock_settings.V4_CALIBRATION_AGENT_ENABLED = True
        # state 不是 dict，不应抛异常
        result = self.agent.calibrate("not_a_dict")  # type: ignore[arg-type]
        self.assertIn("v4_calibration_result", result)
        self.assertTrue(result["v4_calibration_result"].get("fallback"))

    @patch("app.calibration.calibration_agent.settings")
    def test_calibrate_with_experimental_data_calls_fitter(self, mock_settings):
        """有 experimental_data → 调用 fitter.fit()。"""
        mock_settings.V4_CALIBRATION_AGENT_ENABLED = True
        self.mock_fitter.fit.return_value = MOCK_FIT_SUCCESS_LMFIT
        self.mock_ci.estimate.return_value = {}
        self.agent.calibrate(MOCK_STATE_NORMAL)
        # fitter.fit 应被调用，且 target_params 非空
        self.mock_fitter.fit.assert_called_once()
        call_args = self.mock_fitter.fit.call_args
        target_params = call_args[0][0]
        self.assertIsInstance(target_params, list)
        self.assertGreater(len(target_params), 0)

    @patch("app.calibration.calibration_agent.settings")
    def test_calibrate_output_contains_method(self, mock_settings):
        """输出含 method 字段（"lmfit" 或 "least_squares"）。"""
        mock_settings.V4_CALIBRATION_AGENT_ENABLED = True
        self.mock_fitter.fit.return_value = MOCK_FIT_SUCCESS_LMFIT
        self.mock_ci.estimate.return_value = {}
        result = self.agent.calibrate(MOCK_STATE_NORMAL)
        cal = result["v4_calibration_result"]
        self.assertIn(cal["method"], ["lmfit", "least_squares"])

    @patch("app.calibration.calibration_agent.settings")
    def test_calibrate_fit_failed_returns_uncalifiable(self, mock_settings):
        """拟合失败 → uncalifiable 列表非空，warnings 非空。"""
        mock_settings.V4_CALIBRATION_AGENT_ENABLED = True
        self.mock_fitter.fit.return_value = MOCK_FIT_FAILED
        self.mock_ci.estimate.return_value = {}
        result = self.agent.calibrate(MOCK_STATE_NORMAL)
        cal = result["v4_calibration_result"]
        self.assertGreater(len(cal["uncalifiable"]), 0)
        self.assertGreater(len(cal["warnings"]), 0)


# =============================================================================
# TestCalibrationHookNode: Feature Flag + hook 行为
# =============================================================================
class TestCalibrationHookNode(unittest.TestCase):
    """测试 calibration_hook_node LangGraph 节点。"""

    def test_hook_flag_off_returns_empty(self):
        """V4_CALIBRATION_AGENT_ENABLED=false → hook 返回 {}。"""
        with patch("app.calibration.calibration_agent.settings") as mock_settings:
            mock_settings.V4_CALIBRATION_AGENT_ENABLED = False
            result = calibration_hook_node(MOCK_STATE_NORMAL)
        self.assertEqual(result, {})

    @patch("app.calibration.calibration_agent.settings")
    def test_hook_flag_on_writes_v4_field(self, mock_settings):
        """V4_CALIBRATION_AGENT_ENABLED=true → hook 写入 v4_calibration_result。"""
        mock_settings.V4_CALIBRATION_AGENT_ENABLED = True
        with patch.object(
            CalibrationAgent, "calibrate"
        ) as mock_calibrate:
            mock_calibrate.return_value = {
                "v4_calibration_result": {
                    "calibrated_params": {"k1": 0.1},
                    "confidence_intervals": {},
                    "uncalifiable": [],
                    "method": "lmfit",
                    "agent_version": "v4.0",
                    "warnings": [],
                }
            }
            result = calibration_hook_node(MOCK_STATE_NORMAL)
        self.assertIn("v4_calibration_result", result)

    @patch("app.calibration.calibration_agent.settings")
    def test_hook_exception_returns_empty(self, mock_settings):
        """hook 内部异常 → 返回 {}（不阻塞主流水线）。"""
        mock_settings.V4_CALIBRATION_AGENT_ENABLED = True
        with patch.object(
            CalibrationAgent, "calibrate", side_effect=RuntimeError("boom")
        ):
            result = calibration_hook_node(MOCK_STATE_NORMAL)
        self.assertEqual(result, {})

    def test_hook_flag_off_preserves_v3_fields(self):
        """flag=false + state 含 v3 字段 → 返回 {}，v3 字段不变（隔离验证）。"""
        v3_state = {
            "network_json": {"nodes": ["A", "B"]},
            "parameters": {"k1": 0.1},
            "ode_model": {"equations": ["dx/dt = -k1*x"]},
            "sbml_model_id": "BIOMD0000000205",
        }
        with patch("app.calibration.calibration_agent.settings") as mock_settings:
            mock_settings.V4_CALIBRATION_AGENT_ENABLED = False
            result = calibration_hook_node(v3_state)
        # 返回空 dict，不修改任何 v3 字段
        self.assertEqual(result, {})
        # state 字段未变（hook 返回的 dict 是新 dict，不应包含 v3 字段）
        self.assertNotIn("network_json", result)
        self.assertNotIn("parameters", result)
        self.assertNotIn("ode_model", result)

    @patch("app.calibration.calibration_agent.settings")
    def test_hook_flag_on_with_empty_state(self, mock_settings):
        """flag=true + 空 state → 仍返回 v4_calibration_result（fallback）。"""
        mock_settings.V4_CALIBRATION_AGENT_ENABLED = True
        result = calibration_hook_node({})
        self.assertIn("v4_calibration_result", result)
        # 空 state 应触发 fallback
        self.assertTrue(result["v4_calibration_result"].get("fallback"))


# =============================================================================
# TestLeastSquaresFitter: fitter 主类 + 双路径
# =============================================================================
class TestLeastSquaresFitter(unittest.TestCase):
    """测试 LeastSquaresFitter 主类。"""

    def test_fit_success_lmfit_path(self):
        """fit 成功（mock LMFIT_AVAILABLE=True）→ FitResult(success=True, method="lmfit")。"""
        with patch(
            "app.calibration.least_squares_fitter.LMFIT_AVAILABLE", True
        ):
            fitter = LeastSquaresFitter()
            # mock _fit_with_lmfit 直接返回成功
            with patch.object(
                fitter,
                "_fit_with_lmfit",
                return_value=MOCK_FIT_SUCCESS_LMFIT,
            ):
                result = fitter.fit(
                    ["k1", "k2"],
                    {"observations": [0.1, 0.5, 1.0, 1.5, 2.0]},
                )
        self.assertTrue(result.success)
        self.assertEqual(result.method, "lmfit")
        self.assertIn("k1", result.params)

    def test_fit_success_scipy_path(self):
        """fit 成功（mock LMFIT_AVAILABLE=False）→ FitResult(success=True, method="least_squares")。"""
        with patch(
            "app.calibration.least_squares_fitter.LMFIT_AVAILABLE", False
        ):
            fitter = LeastSquaresFitter()
            with patch.object(
                fitter,
                "_fit_with_scipy",
                return_value=MOCK_FIT_SUCCESS_SCIPY,
            ):
                result = fitter.fit(
                    ["k1", "k2"],
                    {"observations": [0.1, 0.5, 1.0, 1.5, 2.0]},
                )
        self.assertTrue(result.success)
        self.assertEqual(result.method, "least_squares")

    def test_fit_exception_returns_failure(self):
        """fit 失败（mock 抛异常）→ FitResult(success=False)。"""
        with patch(
            "app.calibration.least_squares_fitter.LMFIT_AVAILABLE", True
        ):
            fitter = LeastSquaresFitter()
            with patch.object(
                fitter,
                "_fit_with_lmfit",
                side_effect=RuntimeError("fit boom"),
            ):
                result = fitter.fit(
                    ["k1"],
                    {"observations": [0.1, 0.5, 1.0]},
                )
        self.assertFalse(result.success)
        self.assertIn("fit_exception", result.message)

    def test_fit_empty_target_params_returns_failure(self):
        """fit 空 target_params → FitResult(success=False)。"""
        fitter = LeastSquaresFitter()
        result = fitter.fit([], {"observations": [1.0, 2.0]})
        self.assertFalse(result.success)
        self.assertEqual(result.message, "empty_or_invalid_target_params")

    def test_fit_empty_observations_returns_failure(self):
        """fit 无 observations → FitResult(success=False)。"""
        fitter = LeastSquaresFitter()
        result = fitter.fit(["k1"], {})
        self.assertFalse(result.success)
        self.assertEqual(result.message, "empty_observations")

    def test_fit_default_model_func_no_exception(self):
        """fit 默认 model_func（None）→ 不抛异常。"""
        with patch(
            "app.calibration.least_squares_fitter.LMFIT_AVAILABLE", True
        ):
            fitter = LeastSquaresFitter()
            with patch.object(
                fitter,
                "_fit_with_lmfit",
                return_value=MOCK_FIT_SUCCESS_LMFIT,
            ) as mock_lmfit:
                fitter.fit(
                    ["k1", "k2"],
                    {"observations": [0.1, 0.5, 1.0, 1.5, 2.0]},
                )
            # 应调用 _fit_with_lmfit（model_func=None 时使用 _default_model）
            mock_lmfit.assert_called_once()
            # 验证 model_func 是 _default_model
            call_args = mock_lmfit.call_args[0]
            self.assertEqual(call_args[2], LeastSquaresFitter._default_model)


# =============================================================================
# TestConfidenceIntervalEstimator: CI 主类 + 双路径
# =============================================================================
class TestConfidenceIntervalEstimator(unittest.TestCase):
    """测试 ConfidenceIntervalEstimator 主类。"""

    def test_estimate_success_returns_dict(self):
        """estimate 成功 → 返回 dict（每个参数含 lower/upper/std_error）。"""
        estimator = ConfidenceIntervalEstimator(confidence_level=0.95)
        fit_result = FitResult(
            success=True,
            params={"k1": 0.1, "k2": 0.01},
            method="least_squares",  # 走 bootstrap
            residuals=[0.01, -0.02, 0.005, -0.01, 0.02],
        )
        cis = estimator.estimate(fit_result, n_samples=50)
        self.assertIsInstance(cis, dict)
        self.assertIn("k1", cis)
        self.assertIn("k2", cis)
        for name, ci in cis.items():
            self.assertIn("lower", ci)
            self.assertIn("upper", ci)
            self.assertIn("std_error", ci)
            self.assertIn("method", ci)

    def test_estimate_failed_fit_result_returns_empty(self):
        """estimate FitResult(success=False) → 返回空 dict。"""
        estimator = ConfidenceIntervalEstimator()
        fit_result = FitResult(success=False, params={}, method="none")
        cis = estimator.estimate(fit_result)
        self.assertEqual(cis, {})

    def test_estimate_lmfit_path(self):
        """estimate lmfit 路径（mock LMFIT_AVAILABLE=True + method=lmfit）→ method=lmfit。"""
        # 构造 mock lmfit.MinimizerResult
        mock_lmfit_result = MagicMock()
        mock_param_k1 = MagicMock()
        mock_param_k1.stderr = 0.01
        mock_param_k2 = MagicMock()
        mock_param_k2.stderr = 0.002
        mock_lmfit_result.params = {"k1": mock_param_k1, "k2": mock_param_k2}

        fit_result = FitResult(
            success=True,
            params={"k1": 0.1, "k2": 0.01},
            method="lmfit",
            residuals=[0.01],
            raw=mock_lmfit_result,
        )
        with patch(
            "app.calibration.confidence_interval.LMFIT_AVAILABLE", True
        ):
            estimator = ConfidenceIntervalEstimator()
            cis = estimator.estimate(fit_result, n_samples=100)
        self.assertIn("k1", cis)
        self.assertEqual(cis["k1"]["method"], "lmfit")
        self.assertGreater(cis["k1"]["std_error"], 0)

    def test_estimate_bootstrap_path(self):
        """estimate bootstrap 路径（mock LMFIT_AVAILABLE=False）→ method=bootstrap。"""
        fit_result = FitResult(
            success=True,
            params={"k1": 0.1, "k2": 0.01},
            method="least_squares",
            residuals=[0.01, -0.02, 0.005, -0.01, 0.02],
        )
        with patch(
            "app.calibration.confidence_interval.LMFIT_AVAILABLE", False
        ):
            estimator = ConfidenceIntervalEstimator()
            cis = estimator.estimate(fit_result, n_samples=50)
        self.assertIn("k1", cis)
        self.assertEqual(cis["k1"]["method"], "bootstrap")

    def test_estimate_failed_param_marked_uncalibrated(self):
        """estimate 失败参数 → uncalibrated=True。"""
        # lmfit 路径下，stderr=None 时该参数标记 uncalibrated
        mock_lmfit_result = MagicMock()
        mock_param_k1 = MagicMock()
        mock_param_k1.stderr = None  # 无法估计 stderr
        mock_param_k2 = MagicMock()
        mock_param_k2.stderr = 0.01
        mock_lmfit_result.params = {"k1": mock_param_k1, "k2": mock_param_k2}

        fit_result = FitResult(
            success=True,
            params={"k1": 0.1, "k2": 0.01},
            method="lmfit",
            residuals=[0.01],
            raw=mock_lmfit_result,
        )
        with patch(
            "app.calibration.confidence_interval.LMFIT_AVAILABLE", True
        ):
            estimator = ConfidenceIntervalEstimator()
            cis = estimator.estimate(fit_result, n_samples=100)
        # k1 stderr=None → uncalibrated=True
        self.assertTrue(cis["k1"].get("uncalibrated"))
        self.assertEqual(cis["k1"]["method"], "none")
        # k2 正常
        self.assertFalse(cis["k2"].get("uncalibrated"))
        self.assertEqual(cis["k2"]["method"], "lmfit")

    def test_estimate_exception_returns_uncalibrated_dict(self):
        """estimate 内部异常 → 所有参数标记 uncalibrated（不抛异常）。"""
        # 构造会触发异常的 fit_result（lmfit_result.params 缺 key → KeyError → stderr=None）
        bad_raw = MagicMock()
        bad_raw.params = {}  # 空 dict，访问 ["k1"] 触发 KeyError
        fit_result = FitResult(
            success=True,
            params={"k1": 0.1},
            method="lmfit",
            raw=bad_raw,
        )
        with patch(
            "app.calibration.confidence_interval.LMFIT_AVAILABLE", True
        ):
            # estimator 必须在 patch 内创建（__init__ 读取 LMFIT_AVAILABLE）
            estimator = ConfidenceIntervalEstimator()
            cis = estimator.estimate(fit_result, n_samples=10)
        # lmfit 路径：stderr=None → uncalibrated=True
        self.assertIn("k1", cis)
        self.assertTrue(cis["k1"].get("uncalibrated"))
        self.assertEqual(cis["k1"]["method"], "none")


# =============================================================================
# TestUncalifiableMarking: uncalibrated 标记
# =============================================================================
class TestUncalifiableMarking(unittest.TestCase):
    """测试 uncalibrated 参数标记。"""

    def setUp(self):
        self.mock_fitter = MagicMock(spec=LeastSquaresFitter)
        self.mock_ci = MagicMock(spec=ConfidenceIntervalEstimator)
        self.agent = CalibrationAgent(
            fitter=self.mock_fitter, ci_estimator=self.mock_ci
        )

    @patch("app.calibration.calibration_agent.settings")
    def test_fit_failed_params_added_to_uncalifiable(self, mock_settings):
        """拟合失败参数加入 uncalifiable 列表。"""
        mock_settings.V4_CALIBRATION_AGENT_ENABLED = True
        self.mock_fitter.fit.return_value = MOCK_FIT_FAILED
        self.mock_ci.estimate.return_value = {}
        result = self.agent.calibrate(MOCK_STATE_NORMAL)
        cal = result["v4_calibration_result"]
        # target_params 非空，全部失败
        self.assertGreater(len(cal["uncalifiable"]), 0)
        # k1 / k2 / k3 都应在 uncalifiable 中
        for p in ["k1", "k2", "k3"]:
            self.assertIn(p, cal["uncalifiable"])

    @patch("app.calibration.calibration_agent.settings")
    def test_failed_params_marked_in_confidence_intervals(self, mock_settings):
        """confidence_intervals 中失败参数标记 uncalibrated=True。"""
        mock_settings.V4_CALIBRATION_AGENT_ENABLED = True
        self.mock_fitter.fit.return_value = MOCK_FIT_FAILED
        self.mock_ci.estimate.return_value = {}
        result = self.agent.calibrate(MOCK_STATE_NORMAL)
        cal = result["v4_calibration_result"]
        for name, ci in cal["confidence_intervals"].items():
            self.assertTrue(ci.get("uncalibrated"))

    @patch("app.calibration.calibration_agent.settings")
    def test_uncalibrated_does_not_throw(self, mock_settings):
        """不抛异常（不阻塞）。"""
        mock_settings.V4_CALIBRATION_AGENT_ENABLED = True
        # fitter 抛异常
        self.mock_fitter.fit.side_effect = RuntimeError("fit boom")
        self.mock_ci.estimate.return_value = {}
        # 不应抛异常
        result = self.agent.calibrate(MOCK_STATE_NORMAL)
        self.assertIn("v4_calibration_result", result)
        cal = result["v4_calibration_result"]
        self.assertTrue(cal.get("fallback"))

    @patch("app.calibration.calibration_agent.settings")
    def test_warnings_record_failure_reason(self, mock_settings):
        """输出 warnings 非空（记录失败原因）。"""
        mock_settings.V4_CALIBRATION_AGENT_ENABLED = True
        self.mock_fitter.fit.return_value = MOCK_FIT_FAILED
        self.mock_ci.estimate.return_value = {}
        result = self.agent.calibrate(MOCK_STATE_NORMAL)
        cal = result["v4_calibration_result"]
        self.assertGreater(len(cal["warnings"]), 0)
        # warnings 中应包含 fit_failed 标识
        warning_text = " ".join(cal["warnings"])
        self.assertIn("fit_failed", warning_text)

    @patch("app.calibration.calibration_agent.settings")
    def test_partial_ci_failure_marked_uncalibrated(self, mock_settings):
        """部分参数 CI 失败 → 仅失败参数标记 uncalibrated。"""
        mock_settings.V4_CALIBRATION_AGENT_ENABLED = True
        # 拟合成功但 k2 CI 失败
        self.mock_fitter.fit.return_value = MOCK_FIT_SUCCESS_LMFIT
        self.mock_ci.estimate.return_value = {
            "k1": {"lower": 0.09, "upper": 0.13, "std_error": 0.01, "method": "lmfit"},
            "k2": {
                "lower": 0.0,
                "upper": 0.0,
                "std_error": 0.0,
                "method": "none",
                "uncalibrated": True,
            },
            "k3": {"lower": 0.85, "upper": 1.05, "std_error": 0.05, "method": "lmfit"},
        }
        result = self.agent.calibrate(MOCK_STATE_NORMAL)
        cal = result["v4_calibration_result"]
        self.assertIn("k2", cal["uncalifiable"])
        self.assertNotIn("k1", cal["uncalifiable"])
        self.assertNotIn("k3", cal["uncalifiable"])


# =============================================================================
# TestCalibrationModuleInit: 模块 __init__ 导出
# =============================================================================
class TestCalibrationModuleInit(unittest.TestCase):
    """测试 calibration 模块 __init__ 导出。"""

    def test_module_exports_main_symbols(self):
        """模块 __init__ 应导出 CalibrationAgent / calibration_hook_node。"""
        from app.calibration import (
            CalibrationAgent as CA,
            calibration_hook_node as CHN,
            FitResult as FR,
            LeastSquaresFitter as LSF,
            ConfidenceInterval as CI,
            ConfidenceIntervalEstimator as CIE,
        )
        self.assertIsNotNone(CA)
        self.assertIsNotNone(CHN)
        self.assertIsNotNone(FR)
        self.assertIsNotNone(LSF)
        self.assertIsNotNone(CI)
        self.assertIsNotNone(CIE)

    def test_module_all_contains_required_symbols(self):
        """__all__ 应包含 6 个主要符号。"""
        import app.calibration as cal_mod
        self.assertIn("CalibrationAgent", cal_mod.__all__)
        self.assertIn("calibration_hook_node", cal_mod.__all__)
        self.assertIn("FitResult", cal_mod.__all__)
        self.assertIn("LeastSquaresFitter", cal_mod.__all__)
        self.assertIn("ConfidenceInterval", cal_mod.__all__)
        self.assertIn("ConfidenceIntervalEstimator", cal_mod.__all__)


# =============================================================================
# 主入口
# =============================================================================
if __name__ == "__main__":
    unittest.main()
