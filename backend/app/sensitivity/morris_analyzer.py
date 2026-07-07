# BioDynamics Agent v4 - Morris Analyzer (Phase 5 / Task 5.8.3)
#
# MorrisAnalyzer 主类 + MorrisResult dataclass。
# 职责：用 SALib 计算全局灵敏度（Morris elementary effects：μ / σ / μ*）。
#   SALib 不可用时降级到 skipped（不抛异常）。
#
# 设计原则（铁律）：
# 1. 复用 app.config.SALIB_AVAILABLE / SALIB_VERSION（try-import 已就位）
# 2. 不抛异常：任何失败返回 MorrisResult(method="skipped", warnings=[...])
# 3. 不修改 v3 任何字段；仅消费 v4_ode_system / v4_calibration_result / parameters
# 4. 失败降级：异常时返回 skipped 状态，由编排器降级到 local sensitivity
# 5. 单文件优先；不创建多余依赖
#
# 对应 spec.md Part 4 Sensitivity Analysis（第 342-346 行）
# - 输入：params dict + model_func（输入参数 dict，返回标量）
# - 输出：MorrisResult(mu, sigma, mu_star, method, n_trajectories, warnings)
# - 失败策略：SALib 不可用 → method="skipped" + warning
#
# 依赖：
# - app.config.SALIB_AVAILABLE / SALIB_VERSION（try-import 已就位）

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from app.config import SALIB_AVAILABLE, SALIB_VERSION

logger = logging.getLogger(__name__)


# =============================================================================
# MorrisResult dataclass（Morris 分析结果容器）
# =============================================================================
@dataclass
class MorrisResult:
    """Morris elementary effects 分析结果容器。

    Attributes:
        mu: Morris μ（均值，带符号）
        sigma: Morris σ（标准差，反映非线性/交互）
        mu_star: Morris μ*（绝对值均值，反映参数重要性）
        method: 方法标识（"morris" / "skipped"）
        n_trajectories: 轨迹数（传给 SALib 的 N）
        warnings: 警告信息列表
    """

    mu: dict[str, float] = field(default_factory=dict)
    sigma: dict[str, float] = field(default_factory=dict)
    mu_star: dict[str, float] = field(default_factory=dict)
    method: str = "skipped"
    n_trajectories: int = 0
    warnings: list[str] = field(default_factory=list)


# =============================================================================
# MorrisAnalyzer 主类
# =============================================================================
class MorrisAnalyzer:
    """Morris 全局灵敏度分析器（SALib 依赖）。

    SALIB_AVAILABLE=True 时：
    1. 构造 problem = {"num_vars": len(params), "names": [...], "boundaries": [...]}
    2. 调用 SALib.sample.morris.sample(problem, N, num_levels)
    3. 对每个 sample 调用 model_func 得到 Y 数组
    4. 调用 SALib.analyze.morris.analyze(problem, X, Y)
    5. 提取 mu / sigma / mu_star 字典（参数名 → 灵敏度值）

    SALIB_AVAILABLE=False 时：返回 method="skipped" + warning。

    用法：
        analyzer = MorrisAnalyzer(n_trajectories=10, num_levels=4, seed=42)
        result = analyzer.analyze({"k1": 0.1, "k2": 0.01}, model_func)
        if result.method == "morris":
            print(result.mu, result.sigma, result.mu_star)
    """

    DEFAULT_BOUNDARY_FACTOR = 0.5  # boundary = [v*0.5, v*1.5]

    def __init__(
        self,
        n_trajectories: int = 10,
        num_levels: int = 4,
        seed: int | None = 42,
    ) -> None:
        """初始化 Morris 分析器。

        Args:
            n_trajectories: 轨迹数（默认 10）
            num_levels: 网格级数（默认 4，必须偶数）
            seed: 随机种子（默认 42）
        """
        self._n_trajectories = max(int(n_trajectories), 1)
        self._num_levels = max(int(num_levels), 2)
        self._seed = seed
        self._salib_available = SALIB_AVAILABLE
        self._salib_version = SALIB_VERSION
        if self._salib_available:
            logger.info(
                "MorrisAnalyzer 使用 SALib backend (version=%s)", self._salib_version
            )
        else:
            logger.warning(
                "MorrisAnalyzer SALib 未安装，将降级到 skipped（仅 local sensitivity）"
            )

    # =========================================================================
    # 公开 API
    # =========================================================================
    def analyze(
        self,
        params: dict[str, Any],
        model_func: Callable[..., float],
        problem: dict[str, Any] | None = None,
    ) -> MorrisResult:
        """对参数执行 Morris 全局灵敏度分析。

        Args:
            params: 参数 dict {param_name: value}
            model_func: 模型函数，输入参数 dict，返回标量
            problem: SALib problem dict（可选，None 时自动构造）

        Returns:
            MorrisResult：
                - SALib 可用且成功 → method="morris"，含 mu / sigma / mu_star
                - SALib 不可用 / 异常 → method="skipped" + warnings
        """
        if not self._salib_available:
            return self._skipped_result("SALib 未安装，降级到 local sensitivity")

        if not isinstance(params, dict) or not params:
            return self._skipped_result("params 为空或非 dict")

        try:
            return self._analyze_with_salib(params, model_func, problem)
        except Exception as exc:
            logger.warning("MorrisAnalyzer.analyze 失败：%s", exc)
            return self._skipped_result(f"morris_exception: {exc}")

    # =========================================================================
    # SALib 路径
    # =========================================================================
    def _analyze_with_salib(
        self,
        params: dict[str, Any],
        model_func: Callable[..., float],
        problem: dict[str, Any] | None,
    ) -> MorrisResult:
        """使用 SALib 执行 Morris 分析。"""
        import SALib  # type: ignore[import-untyped]
        import numpy as np  # type: ignore[import-untyped]

        # 构造 problem（若未提供）
        if problem is None:
            problem = self._build_problem(params)
        else:
            if not self._validate_problem(problem, params):
                return self._skipped_result("invalid_problem_dict")

        # Morris 采样
        try:
            param_values = SALib.sample.morris.sample(
                problem,
                self._n_trajectories,
                num_levels=self._num_levels,
                seed=self._seed,
            )
        except TypeError:
            # 兼容旧版 SALib（无 seed 参数）
            param_values = SALib.sample.morris.sample(
                problem,
                self._n_trajectories,
                num_levels=self._num_levels,
            )

        # 对每个 sample 调用 model_func
        names = problem["names"]
        Y = np.zeros(param_values.shape[0])
        for i, row in enumerate(param_values):
            sample_params = {
                names[j]: float(row[j]) for j in range(len(names))
            }
            try:
                output = model_func(sample_params)
                Y[i] = _to_float_or_zero(output)
            except Exception as exc:
                logger.warning(
                    "Morris sample %d model_func 失败：%s，置 0.0", i, exc
                )
                Y[i] = 0.0

        # Morris 分析
        si = SALib.analyze.morris.analyze(problem, param_values, Y)

        # 提取 mu / sigma / mu_star
        names_list = list(names)
        mu_array = np.asarray(si["mu"], dtype=float)
        sigma_array = np.asarray(si["sigma"], dtype=float)
        mu_star_array = np.asarray(si["mu_star"], dtype=float)

        mu = {names_list[i]: float(mu_array[i]) for i in range(len(names_list))}
        sigma = {
            names_list[i]: float(sigma_array[i]) for i in range(len(names_list))
        }
        mu_star = {
            names_list[i]: float(mu_star_array[i])
            for i in range(len(names_list))
        }

        return MorrisResult(
            mu=mu,
            sigma=sigma,
            mu_star=mu_star,
            method="morris",
            n_trajectories=self._n_trajectories,
            warnings=[],
        )

    # =========================================================================
    # 辅助方法
    # =========================================================================
    def _build_problem(self, params: dict[str, Any]) -> dict[str, Any]:
        """构造 SALib problem dict。

        格式：
            {"num_vars": N, "names": [...], "boundaries": [[v*0.5, v*1.5], ...]}
        """
        names = list(params.keys())
        boundaries: list[list[float]] = []
        for value in params.values():
            v = _to_float_or_zero(value)
            if v == 0.0:
                boundaries.append([-self.DEFAULT_BOUNDARY_FACTOR, self.DEFAULT_BOUNDARY_FACTOR])
            elif v < 0:
                boundaries.append([v * (1.0 + self.DEFAULT_BOUNDARY_FACTOR), v * (1.0 - self.DEFAULT_BOUNDARY_FACTOR)])
            else:
                boundaries.append([v * self.DEFAULT_BOUNDARY_FACTOR, v * (1.0 + self.DEFAULT_BOUNDARY_FACTOR)])
        return {
            "num_vars": len(names),
            "names": names,
            "boundaries": boundaries,
        }

    @staticmethod
    def _validate_problem(
        problem: dict[str, Any], params: dict[str, Any]
    ) -> bool:
        """验证 problem dict 结构。"""
        if not isinstance(problem, dict):
            return False
        if "num_vars" not in problem or "names" not in problem:
            return False
        if "boundaries" not in problem and "bounds" not in problem:
            return False
        return True

    def _skipped_result(self, reason: str) -> MorrisResult:
        """构造 skipped 结果（降级）。"""
        return MorrisResult(
            mu={},
            sigma={},
            mu_star={},
            method="skipped",
            n_trajectories=0,
            warnings=[reason],
        )


# =============================================================================
# 辅助函数
# =============================================================================
def _to_float_or_zero(value: Any) -> float:
    """将 value 转为 float，失败返回 0.0。"""
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        import numpy as np  # type: ignore[import-untyped]

        if isinstance(value, (np.integer, np.floating)):
            return float(value)
        if isinstance(value, np.ndarray) and value.size > 0:
            return float(value.flat[0])
    except ImportError:
        pass
    if isinstance(value, dict):
        for v in value.values():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return float(v)
        return 0.0
    if isinstance(value, (list, tuple)) and value:
        return _to_float_or_zero(value[0])
    return 0.0


__all__ = ["MorrisAnalyzer", "MorrisResult"]
