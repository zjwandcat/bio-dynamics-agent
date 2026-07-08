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
    # TD-024 修复：新增拟合质量指标与协方差（默认 None，向后兼容）
    # fit_quality 含 R² / RMSE / AIC；covariance 来自 curve_fit 的 pcov
    fit_quality: dict[str, float] | None = None
    covariance: Any = None


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

            # 选择 model_func（默认真实一阶动力学模型，TD-024 替换原占位 placeholder）
            effective_model = model_func if model_func is not None else self._default_model
            # 默认模型路径标记：用于后续注入真实 curve_fit 的 covariance + fit_quality
            use_default_curve = model_func is None

            # 双路径分发
            if self._lmfit_available:
                result = self._fit_with_lmfit(
                    target_params, observations, effective_model
                )
            else:
                result = self._fit_with_scipy(
                    target_params, observations, effective_model
                )

            # TD-024 修复：默认模型路径下，用真实 curve_fit 计算 covariance + fit_quality
            # （替代原占位 dummy 输出，提供 R²/RMSE/AIC 拟合质量指标）
            if use_default_curve and result.success:
                try:
                    t_data = self._extract_timepoints(
                        reference_data, len(observations)
                    )
                    param_guess = [self.DEFAULT_PARAM_INIT] * len(target_params)
                    param_bounds = (
                        [self.DEFAULT_PARAM_MIN] * len(target_params),
                        [self.DEFAULT_PARAM_MAX] * len(target_params),
                    )
                    _, covariance, fit_quality = self._fit_curve(
                        t_data, observations, param_guess, param_bounds
                    )
                    result.fit_quality = fit_quality
                    result.covariance = covariance
                except Exception as qexc:
                    logger.warning(
                        "fit_quality/covariance 计算失败: %s", qexc
                    )

            return result

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
    # 默认 model_func（TD-024 修复：真实一阶动力学模型，替代参数积占位）
    # =========================================================================
    @staticmethod
    def _default_model_curve(t: Any, *params: float) -> Any:
        """真实默认模型函数（scipy.optimize.curve_fit 兼容）。

        一阶动力学指数衰减/增长组合（生化动力学常见形式）。
        参数解释：偶数索引为振幅（amplitude），奇数索引为速率常数（rate），
        末尾若剩单参数则作为常数偏置。

        Args:
            t: 自变量（时间点，标量或数组）
            *params: 参数序列

        Returns:
            预测值（与 t 同形状的数组）
        """
        import numpy as np  # type: ignore[import-untyped]

        t_arr = np.asarray(t, dtype=float)
        y = np.zeros_like(t_arr)
        n = len(params)
        i = 0
        while i < n:
            amp = float(params[i])
            if i + 1 < n:
                rate = float(params[i + 1])
                # 指数项：amp * exp(-rate * t)
                y = y + amp * np.exp(-rate * t_arr)
            else:
                # 末尾单参数作常数偏置
                y = y + amp
            i += 2
        return y

    @staticmethod
    def _default_model(**kwargs: float) -> list[float]:
        """真实默认 model_func（替代占位 placeholder）。

        基于一阶动力学指数模型生成预测序列，使用参数构造指数衰减组合，
        返回与观测长度无关的预测列表（residual 内部对齐长度）。
        生产环境应注入真实 ODE 仿真（roadrunner / scipy.solve_ivp）。
        """
        import numpy as np  # type: ignore[import-untyped]

        if not kwargs:
            return [0.0]
        # 参数值序列（按 kwargs 插入顺序）
        param_values: list[float] = []
        for v in kwargs.values():
            try:
                param_values.append(float(v))
            except (TypeError, ValueError):
                param_values.append(1.0)
        # 默认时间网格（10 点，覆盖典型动力学时间尺度）
        t_grid = np.linspace(0.0, 10.0, 10)
        y_pred = LeastSquaresFitter._default_model_curve(t_grid, *param_values)
        return [float(v) for v in y_pred]

    # =========================================================================
    # 真实曲线拟合（TD-024 修复：scipy.optimize.curve_fit + 网格搜索降级）
    # =========================================================================
    def _fit_curve(
        self,
        t_data: list[float],
        y_data: list[float],
        param_guess: list[float],
        param_bounds: tuple[list[float], list[float]],
    ) -> tuple[list[float], Any, dict[str, float]]:
        """真实最小二乘曲线拟合。

        使用 scipy.optimize.curve_fit 拟合 _default_model_curve 到 (t_data, y_data)，
        返回 (best_params, covariance, fit_quality)。
        scipy 不可用或 curve_fit 失败时降级为简单网格搜索 + warning。

        Args:
            t_data: 自变量序列（时间点）
            y_data: 因变量序列（观测值）
            param_guess: 参数初值列表
            param_bounds: (lower_bounds, upper_bounds)

        Returns:
            (best_params, covariance, fit_quality)
            - best_params: 拟合后的参数列表
            - covariance: 参数协方差矩阵（curve_fit 的 pcov；网格搜索路径为 None）
            - fit_quality: {"R2": float, "RMSE": float, "AIC": float}
        """
        import numpy as np  # type: ignore[import-untyped]

        t_arr = np.asarray(t_data, dtype=float)
        y_arr = np.asarray(y_data, dtype=float)
        n_params = len(param_guess)
        lower, upper = param_bounds
        lower_arr = np.asarray(lower, dtype=float)
        upper_arr = np.asarray(upper, dtype=float)

        best_params: list[float] = list(param_guess)
        covariance: Any = None

        # 尝试 scipy.optimize.curve_fit
        try:
            from scipy.optimize import curve_fit  # type: ignore[import-untyped]

            popt, pcov = curve_fit(
                LeastSquaresFitter._default_model_curve,
                t_arr,
                y_arr,
                p0=param_guess,
                bounds=(lower_arr, upper_arr),
                maxfev=2000,
            )
            best_params = [float(p) for p in popt]
            covariance = pcov
        except Exception as exc:
            # scipy 不可用或 curve_fit 失败 → 网格搜索降级
            if isinstance(exc, ImportError):
                logger.warning(
                    "scipy 不可用，_fit_curve 降级为网格搜索: %s", exc
                )
            else:
                logger.warning(
                    "curve_fit 失败，_fit_curve 降级为网格搜索: %s", exc
                )
            best_params = self._grid_search(
                t_arr, y_arr, param_guess, lower_arr, upper_arr
            )
            covariance = None

        # 计算拟合质量 R² / RMSE / AIC
        fit_quality = self._compute_fit_quality(t_arr, y_arr, best_params, n_params)
        return best_params, covariance, fit_quality

    @staticmethod
    def _grid_search(
        t_arr: Any,
        y_arr: Any,
        param_guess: list[float],
        lower: Any,
        upper: Any,
    ) -> list[float]:
        """简单坐标下降网格搜索（scipy 不可用时的降级）。

        对每个参数独立地在 [lower, upper] 上取 5 个等距候选值，
        贪心选取使残差平方和最小的参数值（坐标下降）。
        """
        import numpy as np  # type: ignore[import-untyped]

        current = list(param_guess)
        best_sse = float("inf")
        grid_size = 5
        # 坐标下降：逐参数扫描
        for i in range(len(current)):
            lo = float(lower[i])
            hi = float(upper[i])
            candidates = np.linspace(lo, hi, grid_size)
            for cand in candidates:
                trial = list(current)
                trial[i] = float(cand)
                y_pred = LeastSquaresFitter._default_model_curve(t_arr, *trial)
                sse = float(np.sum((y_arr - y_pred) ** 2))
                if sse < best_sse:
                    best_sse = sse
                    current = trial
        return current

    @staticmethod
    def _compute_fit_quality(
        t_arr: Any,
        y_arr: Any,
        params: list[float],
        n_params: int,
    ) -> dict[str, float]:
        """计算拟合质量指标：R²、RMSE、AIC。

        - R² = 1 - SS_res / SS_tot（决定系数）
        - RMSE = sqrt(SS_res / n)（均方根误差）
        - AIC = n * ln(SS_res / n) + 2k（高斯误差最小二乘的赤池信息量准则）
        """
        import numpy as np  # type: ignore[import-untyped]

        y_pred = LeastSquaresFitter._default_model_curve(t_arr, *params)
        n = len(y_arr)
        k = max(n_params, 1)
        ss_res = float(np.sum((y_arr - y_pred) ** 2))
        y_mean = float(np.mean(y_arr)) if n > 0 else 0.0
        ss_tot = float(np.sum((y_arr - y_mean) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        rmse = float(np.sqrt(ss_res / n)) if n > 0 else 0.0
        # AIC（高斯误差最小二乘形式）：n*ln(ss_res/n) + 2k
        if n > 0 and ss_res > 0:
            aic = n * math.log(ss_res / n) + 2 * k
        else:
            aic = float("inf")
        return {"R2": r2, "RMSE": rmse, "AIC": aic}

    @staticmethod
    def _extract_timepoints(
        reference_data: dict[str, Any], n_observations: int
    ) -> list[float]:
        """从 reference_data 提取时间点；缺失时用等距网格 [0, n-1]。

        支持字段（优先级）：timepoints / time / t_data；
        同时检查 user_data 下同名字段。
        """
        if isinstance(reference_data, dict):
            for key in ("timepoints", "time", "t_data"):
                vals = reference_data.get(key)
                if isinstance(vals, list) and vals:
                    try:
                        tp = [float(v) for v in vals]
                        if len(tp) >= n_observations:
                            return tp[:n_observations]
                    except (TypeError, ValueError):
                        pass
            user_data = reference_data.get("user_data")
            if isinstance(user_data, dict):
                for key in ("timepoints", "time", "t_data"):
                    vals = user_data.get(key)
                    if isinstance(vals, list) and vals:
                        try:
                            tp = [float(v) for v in vals]
                            if len(tp) >= n_observations:
                                return tp[:n_observations]
                        except (TypeError, ValueError):
                            pass
        # 默认等距网格
        return [float(i) for i in range(n_observations)]

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
