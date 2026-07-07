# BioDynamics Agent v4 - Local Sensitivity Analyzer (Phase 5 / Task 5.8.1)
#
# LocalSensitivityAnalyzer 主类 + LocalSensitivityResult dataclass。
# 职责：用 forward difference（前向差分）计算每个参数的局部灵敏度。
#   始终可用（无外部依赖），不依赖 SALib。
#
# 设计原则（铁律）：
# 1. 不依赖 SALib（local sensitivity 用纯 Python 实现）
# 2. 不抛异常：任何参数扰动失败返回 sensitivity=0.0 + warning
# 3. 不修改 v3 任何字段（network_json / parameters / ode_model / sbml_validator 等）
# 4. 仅消费 P1/P2/P3/P5 产出（v4_ode_system / v4_calibration_result / state.parameters）
# 5. 失败降级：异常参数返回 sensitivity=0.0，不阻塞主流水线
# 6. 单文件优先；不创建多余依赖
#
# 对应 spec.md Part 4 Sensitivity Analysis（第 342-346 行）
# - 输入：params dict + model_func（输入参数 dict，返回标量输出）
# - 输出：dict[param_name, LocalSensitivityResult]
# - 失败策略：异常参数 sensitivity=0.0 + warnings
#
# 依赖：
# - 无（纯 Python 标准库）

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)


# =============================================================================
# LocalSensitivityResult dataclass（单参数灵敏度结果容器）
# =============================================================================
@dataclass
class LocalSensitivityResult:
    """单参数 local sensitivity 结果容器。

    Attributes:
        param_name: 参数名
        sensitivity: 灵敏度值（forward difference: (f(p+Δ) - f(p)) / f(p)）
        baseline: 基线输出 f(p)
        perturbed: 扰动后输出 f(p+Δ)
        method: 计算方法标识（"forward_difference_relative" / "forward_difference_absolute"）
    """

    param_name: str
    sensitivity: float = 0.0
    baseline: float = 0.0
    perturbed: float = 0.0
    method: str = "forward_difference_relative"


# =============================================================================
# LocalSensitivityAnalyzer 主类
# =============================================================================
class LocalSensitivityAnalyzer:
    """Local sensitivity 分析器（forward difference）。

    对每个参数 p 进行扰动：
    - relative=True（默认）：perturbed_params[p] = params[p] * (1 + delta)
    - relative=False：perturbed_params[p] = params[p] + delta

    灵敏度计算：
    - baseline != 0：(perturbed - baseline) / baseline（相对灵敏度）
    - baseline == 0：perturbed - baseline（绝对差，避免除零）

    失败参数（异常 / NaN / Inf）→ sensitivity=0.0 + warnings 记录。

    用法：
        analyzer = LocalSensitivityAnalyzer(delta=0.01, relative=True)
        results = analyzer.analyze({"k1": 0.1, "k2": 0.01}, model_func)
        # results = {"k1": LocalSensitivityResult(...), "k2": ...}
    """

    def __init__(self, delta: float = 0.01, relative: bool = True) -> None:
        """初始化 local sensitivity 分析器。

        Args:
            delta: 扰动步长（默认 0.01 = 1%，forward difference 步长）
            relative: True 用相对扰动（p * (1+delta)），False 用绝对扰动（p + delta）
        """
        if delta <= 0:
            logger.warning(
                "delta 必须 > 0，得到 %s，自动重置为 0.01", delta
            )
            delta = 0.01
        self._delta = float(delta)
        self._relative = bool(relative)

    # =========================================================================
    # 公开 API
    # =========================================================================
    def analyze(
        self,
        params: dict[str, Any],
        model_func: Callable[..., float],
        output_key: str = "default",
    ) -> dict[str, LocalSensitivityResult]:
        """对所有参数执行 local sensitivity 分析。

        Args:
            params: 参数 dict {param_name: value}
            model_func: 模型函数，输入参数 dict，返回标量输出。
                签名：model_func(params: dict) -> float
                若返回 dict/list，提取 output_key 对应字段（默认取第一个数值）。
            output_key: 当 model_func 返回 dict 时提取的 key（默认 "default"）

        Returns:
            dict[param_name, LocalSensitivityResult]，每个参数对应一个结果。
            失败参数（异常 / NaN / Inf）sensitivity=0.0。
        """
        results: dict[str, LocalSensitivityResult] = {}
        warnings: list[str] = []

        if not isinstance(params, dict) or not params:
            logger.warning(
                "LocalSensitivityAnalyzer.analyze: params 为空或非 dict"
            )
            return results

        # 计算 baseline（所有参数原值）
        try:
            baseline_value = self._call_model(model_func, params, output_key)
        except Exception as exc:
            logger.warning(
                "baseline 计算失败：%s，所有参数 sensitivity=0.0", exc
            )
            for name in params.keys():
                results[name] = LocalSensitivityResult(
                    param_name=name,
                    sensitivity=0.0,
                    baseline=0.0,
                    perturbed=0.0,
                    method=self._method_name(),
                )
            return results

        if not _is_finite(baseline_value):
            logger.warning(
                "baseline 非有限数值（%s），所有参数 sensitivity=0.0",
                baseline_value,
            )
            for name in params.keys():
                results[name] = LocalSensitivityResult(
                    param_name=name,
                    sensitivity=0.0,
                    baseline=0.0,
                    perturbed=0.0,
                    method=self._method_name(),
                )
            return results

        # 对每个参数扰动
        for name, value in params.items():
            try:
                result = self._compute_forward_difference(
                    name=name,
                    value=value,
                    params=params,
                    baseline=baseline_value,
                    model_func=model_func,
                    output_key=output_key,
                )
                results[name] = result
            except Exception as exc:
                logger.warning(
                    "参数 %s 扰动失败：%s，sensitivity=0.0", name, exc
                )
                results[name] = LocalSensitivityResult(
                    param_name=name,
                    sensitivity=0.0,
                    baseline=baseline_value,
                    perturbed=0.0,
                    method=self._method_name(),
                )
                warnings.append(f"{name}: perturb_failed: {exc}")

        if warnings:
            # warnings 通过 logger 记录（不写入 dataclass，由调用方聚合）
            logger.info(
                "LocalSensitivityAnalyzer 完成：%d 参数，%d 失败降级",
                len(results),
                len(warnings),
            )
        else:
            logger.info(
                "LocalSensitivityAnalyzer 完成：%d 参数全部成功", len(results)
            )

        return results

    # =========================================================================
    # 私有方法
    # =========================================================================
    def _compute_forward_difference(
        self,
        name: str,
        value: Any,
        params: dict[str, Any],
        baseline: float,
        model_func: Callable[..., float],
        output_key: str,
    ) -> LocalSensitivityResult:
        """对单参数执行 forward difference。

        Args:
            name: 参数名
            value: 参数原值
            params: 完整参数 dict（用于拷贝扰动）
            baseline: 基线输出
            model_func: 模型函数
            output_key: 输出 key

        Returns:
            LocalSensitivityResult（含 sensitivity / baseline / perturbed / method）
        """
        # 构造扰动参数
        perturbed_params = dict(params)
        original_value = self._to_float(value)
        if original_value is None:
            # 非数值参数 → sensitivity=0.0
            return LocalSensitivityResult(
                param_name=name,
                sensitivity=0.0,
                baseline=baseline,
                perturbed=0.0,
                method=self._method_name(),
            )

        if self._relative:
            perturbed_value = original_value * (1.0 + self._delta)
        else:
            perturbed_value = original_value + self._delta

        perturbed_params[name] = perturbed_value

        # 调用模型得到扰动输出
        perturbed_output = self._call_model(
            model_func, perturbed_params, output_key
        )

        if not _is_finite(perturbed_output):
            return LocalSensitivityResult(
                param_name=name,
                sensitivity=0.0,
                baseline=baseline,
                perturbed=0.0,
                method=self._method_name(),
            )

        # 计算灵敏度
        sensitivity = self._compute_sensitivity(
            baseline=baseline, perturbed=perturbed_output
        )

        return LocalSensitivityResult(
            param_name=name,
            sensitivity=sensitivity,
            baseline=baseline,
            perturbed=perturbed_output,
            method=self._method_name(),
        )

    def _compute_sensitivity(self, baseline: float, perturbed: float) -> float:
        """计算 forward difference 灵敏度。

        - baseline != 0：(perturbed - baseline) / baseline（相对灵敏度）
        - baseline == 0：perturbed - baseline（绝对差，避免除零）
        """
        if baseline != 0:
            return (perturbed - baseline) / baseline
        return perturbed - baseline

    @staticmethod
    def _call_model(
        model_func: Callable[..., Any],
        params: dict[str, Any],
        output_key: str,
    ) -> float:
        """调用 model_func 并提取标量输出。

        - model_func 返回数值（int/float） → 直接使用
        - model_func 返回 dict → 取 output_key（或第一个数值）
        - model_func 返回 list/tuple → 取第一个数值
        """
        raw = model_func(params)
        return _extract_scalar(raw, output_key)

    @staticmethod
    def _to_float(value: Any) -> float | None:
        """将 value 转为 float，失败返回 None。"""
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        try:
            import numpy as np  # type: ignore[import-untyped]

            if isinstance(value, (np.integer, np.floating)):
                return float(value)
        except ImportError:
            pass
        return None

    def _method_name(self) -> str:
        """返回当前方法标识。"""
        return (
            "forward_difference_relative"
            if self._relative
            else "forward_difference_absolute"
        )


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


def _extract_scalar(raw: Any, output_key: str) -> float:
    """从 model_func 返回值提取标量。

    - int / float → 直接返回
    - dict → 取 output_key（缺失则取第一个数值）
    - list / tuple → 取第一个元素
    - 其他 → 0.0
    """
    if isinstance(raw, bool):
        return 0.0
    if isinstance(raw, (int, float)):
        return float(raw)
    try:
        import numpy as np  # type: ignore[import-untyped]

        if isinstance(raw, (np.integer, np.floating)):
            return float(raw)
        if isinstance(raw, np.ndarray) and raw.size > 0:
            return float(raw.flat[0])
    except ImportError:
        pass
    if isinstance(raw, dict):
        # 优先 output_key
        if output_key in raw:
            return _extract_scalar(raw[output_key], output_key)
        # 取第一个数值
        for v in raw.values():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return float(v)
        return 0.0
    if isinstance(raw, (list, tuple)) and raw:
        return _extract_scalar(raw[0], output_key)
    return 0.0


__all__ = ["LocalSensitivityAnalyzer", "LocalSensitivityResult"]
