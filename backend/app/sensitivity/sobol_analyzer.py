# BioDynamics Agent v4 - Sobol Analyzer (Phase 5 / Task 5.8.2)
#
# SobolAnalyzer 主类 + SobolResult dataclass。
# 职责：用 SALib 计算全局灵敏度（Sobol 一阶 / 总效应指数）。
#   SALib 不可用时降级到 skipped（不抛异常）。
#
# 设计原则（铁律）：
# 1. 复用 app.config.SALIB_AVAILABLE / SALIB_VERSION（try-import 已就位）
# 2. 不抛异常：任何失败返回 SobolResult(method="skipped", warnings=[...])
# 3. 不修改 v3 任何字段；仅消费 v4_ode_system / v4_calibration_result / parameters
# 4. 失败降级：异常时返回 skipped 状态，由编排器降级到 local sensitivity
# 5. 单文件优先；不创建多余依赖
#
# 对应 spec.md Part 4 Sensitivity Analysis（第 342-346 行）
# - 输入：params dict + model_func（输入参数 dict，返回标量）
# - 输出：SobolResult(S1, ST, S2 | None, method, n_samples, warnings)
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
# SobolResult dataclass（Sobol 分析结果容器）
# =============================================================================
@dataclass
class SobolResult:
    """Sobol 分析结果容器。

    Attributes:
        S1: 一阶灵敏度指数 {param_name: value}
        ST: 总效应灵敏度指数 {param_name: value}
        S2: 二阶交互指数（calc_second_order=True 时返回），否则 None
        method: 方法标识（"sobol" / "skipped"）
        n_samples: 采样数（实际传给 SALib 的 N）
        warnings: 警告信息列表
    """

    S1: dict[str, float] = field(default_factory=dict)
    ST: dict[str, float] = field(default_factory=dict)
    S2: dict[str, dict[str, float]] | None = None
    method: str = "skipped"
    n_samples: int = 0
    warnings: list[str] = field(default_factory=list)


# =============================================================================
# SobolAnalyzer 主类
# =============================================================================
class SobolAnalyzer:
    """Sobol 全局灵敏度分析器（SALib 依赖）。

    SALIB_AVAILABLE=True 时：
    1. 构造 problem = {"num_vars": len(params), "names": [...], "boundaries": [...]}
    2. 调用 SALib.sample.saltelli.sample(problem, N, calc_second_order=False)
    3. 对每个 sample 调用 model_func 得到 Y 数组
    4. 调用 SALib.analyze.sobol.analyze(problem, Y, calc_second_order=False)
    5. 提取 S1 / ST 字典（参数名 → 灵敏度值）

    SALIB_AVAILABLE=False 时：返回 method="skipped" + warning。

    用法：
        analyzer = SobolAnalyzer(n_samples=1024, seed=42)
        result = analyzer.analyze({"k1": 0.1, "k2": 0.01}, model_func)
        if result.method == "sobol":
            print(result.S1, result.ST)
    """

    DEFAULT_BOUNDARY_FACTOR = 0.5  # boundary = [v*0.5, v*1.5]

    def __init__(
        self, n_samples: int = 1024, seed: int | None = 42
    ) -> None:
        """初始化 Sobol 分析器。

        Args:
            n_samples: Saltelli 采样数 N（默认 1024）
            seed: 随机种子（默认 42，可复现）
        """
        self._n_samples = max(int(n_samples), 1)
        self._seed = seed
        self._salib_available = SALIB_AVAILABLE
        self._salib_version = SALIB_VERSION
        if self._salib_available:
            logger.info(
                "SobolAnalyzer 使用 SALib backend (version=%s)", self._salib_version
            )
        else:
            logger.warning(
                "SobolAnalyzer SALib 未安装，将降级到 skipped（仅 local sensitivity）"
            )

    # =========================================================================
    # 公开 API
    # =========================================================================
    def analyze(
        self,
        params: dict[str, Any],
        model_func: Callable[..., float],
        problem: dict[str, Any] | None = None,
    ) -> SobolResult:
        """对参数执行 Sobol 全局灵敏度分析。

        Args:
            params: 参数 dict {param_name: value}
            model_func: 模型函数，输入参数 dict，返回标量
            problem: SALib problem dict（可选，None 时自动构造）
                格式：{"num_vars": N, "names": [...], "boundaries": [[lo, hi], ...]}

        Returns:
            SobolResult：
                - SALib 可用且成功 → method="sobol"，含 S1 / ST
                - SALib 不可用 / 异常 → method="skipped" + warnings
        """
        if not self._salib_available:
            return self._skipped_result("SALib 未安装，降级到 local sensitivity")

        if not isinstance(params, dict) or not params:
            return self._skipped_result("params 为空或非 dict")

        try:
            return self._analyze_with_salib(params, model_func, problem)
        except Exception as exc:
            logger.warning("SobolAnalyzer.analyze 失败：%s", exc)
            return self._skipped_result(f"sobol_exception: {exc}")

    # =========================================================================
    # SALib 路径
    # =========================================================================
    def _analyze_with_salib(
        self,
        params: dict[str, Any],
        model_func: Callable[..., float],
        problem: dict[str, Any] | None,
    ) -> SobolResult:
        """使用 SALib 执行 Sobol 分析。"""
        import SALib  # type: ignore[import-untyped]
        import numpy as np  # type: ignore[import-untyped]

        # 构造 problem（若未提供）
        if problem is None:
            problem = self._build_problem(params)
        else:
            # 验证 problem 结构
            if not self._validate_problem(problem, params):
                return self._skipped_result("invalid_problem_dict")

        # Saltelli 采样
        try:
            param_values = SALib.sample.saltelli.sample(
                problem,
                self._n_samples,
                calc_second_order=False,
                seed=self._seed,
            )
        except TypeError:
            # 兼容旧版 SALib（无 seed 参数）
            param_values = SALib.sample.saltelli.sample(
                problem,
                self._n_samples,
                calc_second_order=False,
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
                    "Sobol sample %d model_func 失败：%s，置 0.0", i, exc
                )
                Y[i] = 0.0

        # Sobol 分析
        try:
            si = SALib.analyze.sobol.analyze(
                problem, Y, calc_second_order=False, seed=self._seed
            )
        except TypeError:
            si = SALib.analyze.sobol.analyze(
                problem, Y, calc_second_order=False
            )

        # 提取 S1 / ST
        s1_array = np.asarray(si["S1"], dtype=float)
        st_array = np.asarray(si["ST"], dtype=float)

        S1 = {names[i]: float(s1_array[i]) for i in range(len(names))}
        ST = {names[i]: float(st_array[i]) for i in range(len(names))}

        return SobolResult(
            S1=S1,
            ST=ST,
            S2=None,  # calc_second_order=False
            method="sobol",
            n_samples=self._n_samples,
            warnings=[],
        )

    # =========================================================================
    # 辅助方法
    # =========================================================================
    def _build_problem(self, params: dict[str, Any]) -> dict[str, Any]:
        """构造 SALib problem dict。

        格式：
            {"num_vars": N, "names": [...], "boundaries": [[v*0.5, v*1.5], ...]}

        Args:
            params: 参数 dict

        Returns:
            SALib problem dict
        """
        names = list(params.keys())
        boundaries: list[list[float]] = []
        for value in params.values():
            v = _to_float_or_zero(value)
            if v == 0.0:
                # 边界为 0 时用 ±DEFAULT_BOUNDARY_FACTOR 避免 [0, 0]
                boundaries.append([-self.DEFAULT_BOUNDARY_FACTOR, self.DEFAULT_BOUNDARY_FACTOR])
            elif v < 0:
                # 负值：保持符号，区间反转
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

    def _skipped_result(self, reason: str) -> SobolResult:
        """构造 skipped 结果（降级）。"""
        return SobolResult(
            S1={},
            ST={},
            S2=None,
            method="skipped",
            n_samples=0,
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
        # 取第一个数值
        for v in value.values():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return float(v)
        return 0.0
    if isinstance(value, (list, tuple)) and value:
        return _to_float_or_zero(value[0])
    return 0.0


__all__ = ["SobolAnalyzer", "SobolResult"]
