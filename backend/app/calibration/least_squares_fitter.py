# BioDynamics Agent v4 - Least Squares Fitter (Phase 5 / Task 5.7.2)
#
# 参数拟合引擎，支持 lmfit / scipy.optimize.least_squares 双路径。
# lmfit 不可用时自动降级到 scipy（依赖隔离策略）。
#
# 设计原则（铁律）：
# 1. 复用 app.config.LMFIT_AVAILABLE / LMFIT_VERSION（try-import 已就位）
# 2. 不抛异常：任何拟合失败返回 FitResult(success=False, ...)，由调用方降级处理
# 3. 默认 model_func：用参数积近似 forward simulation（placeholder，生产环境注入真实模型）
# 4. 单文件优先；不创建多余依赖

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable

from app.config import LMFIT_AVAILABLE, LMFIT_VERSION

logger = logging.getLogger(__name__)


# =============================================================================
# FitResult dataclass（拟合结果容器）
# =============================================================================
@dataclass
class FitResult:
    """参数拟合结果容器（lmfit / scipy 共用）。

    Attributes:
        success: 拟合是否成功
        params: 拟合后的参数 dict {param_name: value}
        cost: 残差平方和（loss 值）
        nfev: 函数评估次数
        message: 拟合器返回的消息（success / failure reason）
        method: 拟合方法标识（"lmfit" / "least_squares"）
        residuals: 残差向量（可选，bootstrap CI 用）
        raw: 原始拟合对象（lmfit.MinimizerResult / scipy.OptimizeResult，CI 提取用）
    """

    success: bool = False
    params: dict[str, float] = field(default_factory=dict)
    cost: float = float("inf")
    nfev: int = 0
    message: str = ""
    method: str = "none"
    residuals: list[float] | None = None
    raw: Any = None


# =============================================================================
# LeastSquaresFitter 主类
# =============================================================================
class LeastSquaresFitter:
    """参数拟合器，lmfit / scipy.optimize.least_squares 双路径。

    根据 LMFIT_AVAILABLE flag 选择 backend：
    - True  → lmfit.minimize（含 stderr / CI 提取）
    - False → scipy.optimize.least_squares（需 bootstrap CI）

    用法：
        fitter = LeastSquaresFitter()
        result = fitter.fit(["k1", "k2"], reference_data)
        if result.success:
            print(result.params, result.method)
    """

    # 默认参数范围（lmfit Parameters 用）
    DEFAULT_PARAM_MIN = 1e-6
    DEFAULT_PARAM_MAX = 1e3
    DEFAULT_PARAM_INIT = 1.0

    def __init__(self) -> None:
        """根据 LMFIT_AVAILABLE flag 决定 backend。"""
        self._lmfit_available = LMFIT_AVAILABLE
        self._lmfit_version = LMFIT_VERSION
        if self._lmfit_available:
            logger.info(
                "LeastSquaresFitter 使用 lmfit backend (version=%s)", self._lmfit_version
            )
        else:
            logger.warning(
                "LeastSquaresFitter 降级到 scipy.optimize.least_squares "
                "（lmfit 未安装）"
            )

    # =========================================================================
    # 公开 API
    # =========================================================================
    def fit(
        self,
        target_params: list[str],
        reference_data: dict[str, Any],
        model_func: Callable[..., list[float]] | None = None,
    ) -> FitResult:
        """拟合 target_params 到 reference_data。

        Args:
            target_params: 待拟合参数名列表（如 ["k1", "k2"]）
            reference_data: 参考数据 dict，至少含 "observations" 字段（list[float]）
                若缺失 observations，使用空列表（视为无观测数据，拟合失败）
            model_func: 模型函数（输入参数值，返回 simulated list[float]）。
                None 时使用默认 placeholder（参数积近似）。

        Returns:
            FitResult：
                - success=True 时含 params / cost / method
                - success=False 时 method 仍标识 backend（便于 CI 提取降级）
                - 失败不抛异常，返回 FitResult(success=False, ...)
        """
        try:
            # 输入校验
            if not target_params or not isinstance(target_params, list):
                return FitResult(
                    success=False,
                    message="empty_or_invalid_target_params",
                    method=self._backend_name(),
                )

            observations = self._extract_observations(reference_data)
            if not observations:
                return FitResult(
                    success=False,
                    message="empty_observations",
                    method=self._backend_name(),
                )

            # 选择 model_func（默认 placeholder）
            effective_model = model_func if model_func is not None else self._default_model

            # 双路径分发
            if self._lmfit_available:
                return self._fit_with_lmfit(
                    target_params, observations, effective_model
                )
            return self._fit_with_scipy(
                target_params, observations, effective_model
            )

        except Exception as exc:
            logger.warning("LeastSquaresFitter.fit 失败：%s", exc)
            return FitResult(
                success=False,
                message=f"fit_exception: {exc}",
                method=self._backend_name(),
            )

    # =========================================================================
    # lmfit 路径
    # =========================================================================
    def _fit_with_lmfit(
        self,
        target_params: list[str],
        observations: list[float],
        model_func: Callable[..., list[float]],
    ) -> FitResult:
        """lmfit.minimize 拟合路径。

        构造 lmfit.Parameters（每个参数 init=1.0, min=1e-6, max=1e3），
        调用 lmfit.minimize 最小化 (simulated - observed) 残差。
        """
        import lmfit  # type: ignore[import-untyped]

        # 构造 lmfit.Parameters
        lm_params = lmfit.Parameters()
        for name in target_params:
            lm_params.add(
                name,
                value=self.DEFAULT_PARAM_INIT,
                min=self.DEFAULT_PARAM_MIN,
                max=self.DEFAULT_PARAM_MAX,
            )

        def _residual(params: "lmfit.Parameters") -> list[float]:
            values = {name: params[name].value for name in target_params}
            simulated = model_func(**values)
            if not isinstance(simulated, list):
                simulated = list(simulated)
            # 对齐长度
            n = min(len(simulated), len(observations))
            return [simulated[i] - observations[i] for i in range(n)]

        result = lmfit.minimize(_residual, lm_params, method="leastsq")
        fitted = {name: float(result.params[name].value) for name in target_params}
        residuals_list: list[float] = (
            list(result.residual) if result.residual is not None else []
        )

        return FitResult(
            success=bool(result.success),
            params=fitted,
            cost=float(getattr(result, "chisqr", float("inf"))),
            nfev=int(getattr(result, "nfev", 0)),
            message=str(getattr(result, "message", "lmfit_success")),
            method="lmfit",
            residuals=residuals_list,
            raw=result,
        )

    # =========================================================================
    # scipy.optimize.least_squares 路径（lmfit 不可用降级）
    # =========================================================================
    def _fit_with_scipy(
        self,
        target_params: list[str],
        observations: list[float],
        model_func: Callable[..., list[float]],
    ) -> FitResult:
        """scipy.optimize.least_squares 拟合路径（lmfit 不可用降级）。"""
        from scipy.optimize import least_squares  # type: ignore[import-untyped]

        import numpy as np  # type: ignore[import-untyped]

        x0 = np.array([self.DEFAULT_PARAM_INIT] * len(target_params))
        bounds = (
            np.array([self.DEFAULT_PARAM_MIN] * len(target_params)),
            np.array([self.DEFAULT_PARAM_MAX] * len(target_params)),
        )

        def _residual(x: "np.ndarray") -> "np.ndarray":
            values = {name: float(x[i]) for i, name in enumerate(target_params)}
            simulated = model_func(**values)
            if not isinstance(simulated, list):
                simulated = list(simulated)
            n = min(len(simulated), len(observations))
            return np.array(
                [simulated[i] - observations[i] for i in range(n)],
                dtype=float,
            )

        result = least_squares(
            _residual,
            x0,
            bounds=bounds,
            method="trf",
            diff_step=1e-6,  # forward difference 默认
            max_nfev=200,
        )

        fitted = {
            name: float(result.x[i]) for i, name in enumerate(target_params)
        }
        residuals_list = list(result.fun) if result.fun is not None else []

        return FitResult(
            success=bool(result.success),
            params=fitted,
            cost=float(np.sum(np.square(result.fun))) if result.fun is not None else float("inf"),
            nfev=int(result.nfev),
            message=str(result.message) if result.message else "scipy_success",
            method="least_squares",
            residuals=residuals_list,
            raw=result,
        )

    # =========================================================================
    # 默认 model_func（placeholder：参数积近似 forward simulation）
    # =========================================================================
    @staticmethod
    def _default_model(**kwargs: float) -> list[float]:
        """默认 placeholder model：参数积近似 forward simulation。

        生产环境应注入真实 ODE 仿真（roadrunner / scipy.solve_ivp），
        此 placeholder 仅用于接口完整性 + 测试，不保证生物学意义。
        """
        # 参数积作为 simulated（与观测数据对齐）
        if not kwargs:
            return [0.0]
        product = 1.0
        for v in kwargs.values():
            try:
                product *= float(v)
            except (TypeError, ValueError):
                product *= 1.0
        # 返回与观测长度无关的 placeholder；调用方对齐长度
        # 用一个常量列表避免长度不匹配（residual 内部对齐）
        return [product * 0.1, product * 0.5, product * 1.0, product * 1.5, product * 2.0]

    # =========================================================================
    # 辅助方法
    # =========================================================================
    @staticmethod
    def _extract_observations(reference_data: dict[str, Any]) -> list[float]:
        """从 reference_data 提取 observations 列表。

        支持字段：
        - reference_data["observations"]: list[float]
        - reference_data["user_data"]["observations"]: list[float]
        - reference_data["values"]: list[float]（备选）
        """
        if not isinstance(reference_data, dict):
            return []

        # 优先 observations
        obs = reference_data.get("observations")
        if isinstance(obs, list) and obs:
            return [float(v) for v in obs if _is_finite(v)]

        # 备选 user_data.observations
        user_data = reference_data.get("user_data")
        if isinstance(user_data, dict):
            obs = user_data.get("observations")
            if isinstance(obs, list) and obs:
                return [float(v) for v in obs if _is_finite(v)]

        # 备选 values
        vals = reference_data.get("values")
        if isinstance(vals, list) and vals:
            return [float(v) for v in vals if _is_finite(v)]

        return []

    def _backend_name(self) -> str:
        """返回当前 backend 名称（用于失败 FitResult.method 标识）。"""
        return "lmfit" if self._lmfit_available else "least_squares"


# =============================================================================
# 辅助函数
# =============================================================================
def _is_finite(value: Any) -> bool:
    """检查 value 是否为有限数值（int / float / np.number）。"""
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    # 尝试 numpy 数值类型
    try:
        import numpy as np  # type: ignore[import-untyped]

        if isinstance(value, (np.integer, np.floating)):
            return bool(np.isfinite(value))
    except ImportError:
        pass
    return False


__all__ = ["FitResult", "LeastSquaresFitter"]
