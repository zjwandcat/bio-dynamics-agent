# BioDynamics Agent v4 - Confidence Interval Estimator (Phase 5 / Task 5.7.3)
#
# 置信区间估计器，支持 lmfit stderr / bootstrap 双路径。
# lmfit 不可用或 fit_result.method != "lmfit" 时降级到 bootstrap。
#
# 设计原则（铁律）：
# 1. 复用 app.config.LMFIT_AVAILABLE / LMFIT_VERSION（try-import 已就位）
# 2. 不抛异常：任何 CI 估计失败返回 {uncalibrated: True} 占位
# 3. fit_result.success=False 时返回空 dict（无参数可估）
# 4. 单文件优先；不创建多余依赖

from __future__ import annotations

import hashlib
import logging
import math
import random
from dataclasses import dataclass
from typing import Any

from app.config import LMFIT_AVAILABLE, LMFIT_VERSION
from app.calibration.least_squares_fitter import FitResult

logger = logging.getLogger(__name__)


# =============================================================================
# ConfidenceInterval dataclass（单参数置信区间容器）
# =============================================================================
@dataclass
class ConfidenceInterval:
    """单参数置信区间容器。

    Attributes:
        param_name: 参数名
        lower: 置信区间下界
        upper: 置信区间上界
        std_error: 标准误差
        method: 估计方法（"lmfit" / "bootstrap" / "none"）
    """

    param_name: str
    lower: float = 0.0
    upper: float = 0.0
    std_error: float = 0.0
    method: str = "none"


# =============================================================================
# ConfidenceIntervalEstimator 主类
# =============================================================================
class ConfidenceIntervalEstimator:
    """置信区间估计器，lmfit stderr / bootstrap 双路径。

    根据 fit_result.method + LMFIT_AVAILABLE 选择路径：
    - fit_result.method == "lmfit" 且 LMFIT_AVAILABLE=True
        → 从 lmfit.MinimizerResult 提取 stderr + 95% CI（mean ± 1.96 * stderr）
    - 否则
        → bootstrap 重采样 n_samples 次计算经验分位数 [2.5%, 97.5%]

    用法：
        estimator = ConfidenceIntervalEstimator(confidence_level=0.95)
        cis = estimator.estimate(fit_result, n_samples=100)
        # cis = {"k1": {"lower": ..., "upper": ..., "std_error": ..., "method": ...}}
    """

    # 95% CI 的 z-score（正态分布）
    Z_95 = 1.96

    def __init__(self, confidence_level: float = 0.95) -> None:
        """初始化置信区间估计器。

        Args:
            confidence_level: 置信水平（默认 0.95 = 95% CI）
        """
        if not (0.0 < confidence_level < 1.0):
            raise ValueError(
                f"confidence_level 必须在 (0, 1) 范围内，得到 {confidence_level}"
            )
        self._confidence_level = confidence_level
        self._lmfit_available = LMFIT_AVAILABLE
        self._lmfit_version = LMFIT_VERSION

    # =========================================================================
    # 公开 API
    # =========================================================================
    def estimate(
        self, fit_result: FitResult, n_samples: int = 100
    ) -> dict[str, dict[str, Any]]:
        """估计 fit_result 中所有参数的置信区间。

        Args:
            fit_result: 拟合结果（含 params / residuals / raw）
            n_samples: bootstrap 重采样次数（默认 100）

        Returns:
            dict[param_name, dict]，每个参数 dict 含：
                - lower: float
                - upper: float
                - std_error: float
                - method: str ("lmfit" / "bootstrap" / "none")
                - uncalibrated: bool（仅失败参数为 True）

            fit_result.success=False 时返回空 dict {}。
        """
        try:
            # fit 失败 → 无参数可估
            if not isinstance(fit_result, FitResult) or not fit_result.success:
                return {}

            params = fit_result.params
            if not params:
                return {}

            # 选择路径
            if (
                self._lmfit_available
                and fit_result.method == "lmfit"
                and fit_result.raw is not None
            ):
                return self._estimate_with_lmfit(fit_result)
            return self._estimate_with_bootstrap(fit_result, n_samples)

        except Exception as exc:
            logger.warning("ConfidenceIntervalEstimator.estimate 失败：%s", exc)
            # 失败时为所有参数返回 uncalibrated 标记
            return self._build_uncalibrated_dict(
                list(fit_result.params.keys()) if isinstance(fit_result, FitResult) else [],
                reason=f"estimate_exception: {exc}",
            )

    # =========================================================================
    # lmfit 路径：从 MinimizerResult 提取 stderr
    # =========================================================================
    def _estimate_with_lmfit(self, fit_result: FitResult) -> dict[str, dict[str, Any]]:
        """从 lmfit.MinimizerResult 提取 stderr + 95% CI。

        公式：CI = mean ± 1.96 * stderr
        - mean = 拟合参数值（fit_result.params[name]）
        - stderr = result.params[name].stderr（lmfit 提供）
        - lower = mean - 1.96 * stderr
        - upper = mean + 1.96 * stderr

        stderr 为 None 的参数标记 uncalibrated=True。
        """
        lmfit_result = fit_result.raw
        result: dict[str, dict[str, Any]] = {}

        for name, value in fit_result.params.items():
            try:
                param_obj = lmfit_result.params[name]
                stderr = getattr(param_obj, "stderr", None)
            except (KeyError, AttributeError):
                stderr = None

            if stderr is None or not _is_finite(stderr) or stderr <= 0:
                # 无法估计 stderr → 标记 uncalibrated
                result[name] = {
                    "lower": 0.0,
                    "upper": 0.0,
                    "std_error": 0.0,
                    "method": "none",
                    "uncalibrated": True,
                }
                continue

            mean = float(value)
            se = float(stderr)
            lower = mean - self.Z_95 * se
            upper = mean + self.Z_95 * se
            result[name] = {
                "lower": lower,
                "upper": upper,
                "std_error": se,
                "method": "lmfit",
                "uncalibrated": False,
            }

        return result

    # =========================================================================
    # bootstrap 路径：重采样残差估计经验分位数
    # =========================================================================
    def _estimate_with_bootstrap(
        self, fit_result: FitResult, n_samples: int
    ) -> dict[str, dict[str, Any]]:
        """bootstrap 重采样估计置信区间。

        策略：
        - 若 fit_result.residuals 非空：重采样残差 + 加噪参数值，计算经验分位数
        - 若 residuals 为空（scipy 路径未返回或空）：用参数值 ± 10% 启发式估计
        - 经验分位数 [2.5%, 97.5%] 作为 95% CI
        """
        if n_samples < 1:
            n_samples = 100

        residuals = fit_result.residuals or []
        result: dict[str, dict[str, Any]] = {}

        # 计算经验分位数下/上界 alpha
        alpha = 1.0 - self._confidence_level
        lower_q = alpha / 2.0 * 100.0  # 2.5
        upper_q = (1.0 - alpha / 2.0) * 100.0  # 97.5

        # 残差标准差（用于重采样扰动幅度）
        if residuals:
            n = len(residuals)
            mean_res = sum(residuals) / n
            var_res = sum((r - mean_res) ** 2 for r in residuals) / max(n - 1, 1)
            std_res = math.sqrt(var_res) if var_res > 0 else 0.0
        else:
            std_res = 0.0

        for name, value in fit_result.params.items():
            try:
                mean = float(value)
                # bootstrap：重采样 n_samples 次
                samples: list[float] = []
                # 使用固定 seed 保证可复现（生产可注入外部 random_state）
                # TD-045 (IB-076) 修复：原 hash(name) 受 PYTHONHASHSEED 随机化影响，
                # 跨进程/跨运行结果不可复现。改用 md5 摘要生成确定性 32-bit seed。
                seed = int(hashlib.md5(name.encode("utf-8")).hexdigest(), 16) & 0xFFFFFFFF
                rng = random.Random(seed)

                if std_res > 0:
                    for _ in range(n_samples):
                        # 加噪：参数值 + 正态残差扰动（Box-Muller 近似）
                        u1 = rng.random() or 1e-10
                        u2 = rng.random()
                        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(
                            2.0 * math.pi * u2
                        )
                        samples.append(mean + z * std_res)
                else:
                    # std_res=0 → 用参数值 ± 10% 启发式
                    spread = abs(mean) * 0.1 if mean != 0 else 0.1
                    for _ in range(n_samples):
                        samples.append(mean + rng.uniform(-spread, spread))

                if not samples:
                    result[name] = {
                        "lower": 0.0,
                        "upper": 0.0,
                        "std_error": 0.0,
                        "method": "none",
                        "uncalibrated": True,
                    }
                    continue

                samples.sort()
                lower = _percentile(samples, lower_q)
                upper = _percentile(samples, upper_q)
                # std_error ≈ (upper - lower) / (2 * 1.96)
                std_error = (upper - lower) / (2.0 * self.Z_95) if upper > lower else 0.0
                result[name] = {
                    "lower": lower,
                    "upper": upper,
                    "std_error": std_error,
                    "method": "bootstrap",
                    "uncalibrated": False,
                }
            except Exception as exc:
                logger.warning(
                    "bootstrap CI 估计失败 param=%s: %s", name, exc
                )
                result[name] = {
                    "lower": 0.0,
                    "upper": 0.0,
                    "std_error": 0.0,
                    "method": "none",
                    "uncalibrated": True,
                }

        return result

    # =========================================================================
    # 辅助方法
    # =========================================================================
    def _build_uncalibrated_dict(
        self, param_names: list[str], reason: str = ""
    ) -> dict[str, dict[str, Any]]:
        """构造全部参数 uncalibrated 的字典（失败降级）。"""
        result: dict[str, dict[str, Any]] = {}
        for name in param_names:
            result[name] = {
                "lower": 0.0,
                "upper": 0.0,
                "std_error": 0.0,
                "method": "none",
                "uncalibrated": True,
                "reason": reason,
            }
        return result


# =============================================================================
# 辅助函数
# =============================================================================
def _is_finite(value: Any) -> bool:
    """检查 value 是否为有限数值。"""
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    try:
        import numpy as np  # type: ignore[import-untyped]

        if isinstance(value, (np.integer, np.floating)):
            return bool(np.isfinite(value))
    except ImportError:
        pass
    return False


def _percentile(sorted_samples: list[float], q: float) -> float:
    """计算已排序样本的 q 分位数（线性插值）。

    Args:
        sorted_samples: 已排序的样本列表（升序）
        q: 分位数百分比 [0, 100]

    Returns:
        分位数对应的样本值
    """
    if not sorted_samples:
        return 0.0
    if len(sorted_samples) == 1:
        return float(sorted_samples[0])

    # 线性插值法（与 numpy.percentile 默认一致）
    n = len(sorted_samples)
    rank = (q / 100.0) * (n - 1)
    lower_idx = int(math.floor(rank))
    upper_idx = int(math.ceil(rank))
    if lower_idx == upper_idx:
        return float(sorted_samples[lower_idx])
    frac = rank - lower_idx
    return float(
        sorted_samples[lower_idx]
        + (sorted_samples[upper_idx] - sorted_samples[lower_idx]) * frac
    )


__all__ = ["ConfidenceInterval", "ConfidenceIntervalEstimator"]
