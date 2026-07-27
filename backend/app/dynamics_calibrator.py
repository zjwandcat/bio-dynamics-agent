# BioDynamics Agent - N7 缺口 4：确定性动力学校准器
#
# 网格搜索（无 LLM 不确定性）：仿真后检查峰值时间/幅度，调整 kcat/kon 等参数
# 使仿真峰值落在期望窗口内。
#
# 设计原则（硬约束）：
#   - 纯数值优化，固定随机种子，无 LLM 调用
#   - 参数溯源 BioModels ID（不凭空造值，仅调整已有参数）
#   - 不逐案例硬编码峰值（期望窗口由 expected_dynamics 传入）
#   - 最大迭代次数限制 + C3 稳定性兜底（NaN/Inf 回滚）
#
# 集成点：sbml_validator.SBMLValidator.validate 仿真后检查峰值是否在
# expected_dynamics 窗口内；不在窗口时调用 DynamicsCalibrator.calibrate，
# 校准结果写入 validation_report.calibration_log。

from __future__ import annotations

import logging
import math
import random
from typing import Any, Callable

from app.csv_io import read_csv_robust

logger = logging.getLogger(__name__)


# 固定随机种子（确定性，硬约束）
_CALIBRATOR_SEED = 20240107


def _safe_float(value: Any, default: float = 0.0) -> float:
    """安全转 float，失败返回 default。"""
    try:
        f = float(value)
        if not math.isfinite(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def _read_simulation_series(csv_path: str) -> tuple[list[float], dict[str, list[float]]]:
    """读取仿真 CSV，返回 (times, species_series)。

    Args:
        csv_path: 仿真结果 CSV 路径（第一列 time，后续列为物种浓度）。

    Returns:
        (times, {species: [values]})。读取失败返回 ([], {})。
    """
    if not csv_path:
        return [], {}
    try:
        result = read_csv_robust(csv_path)
        return list(result.times), dict(result.species)
    except Exception as exc:
        logger.warning("DynamicsCalibrator 读取 CSV 失败 (%s): %s", csv_path, exc)
        return [], {}


def _detect_peak(
    times: list[float],
    series: list[float],
) -> tuple[float, float]:
    """用 scipy.signal.find_peaks 检测峰值时间和幅度。

    Args:
        times: 时间点列表。
        series: 物种浓度时序。

    Returns:
        (peak_time, peak_value)。无峰值或数据为空时返回 (0.0, 0.0)。
    """
    if not series or not times or len(series) != len(times):
        return 0.0, 0.0
    finite_series = [v for v in series if math.isfinite(v)]
    if not finite_series:
        return 0.0, 0.0
    peak_value = max(series)
    peak_idx = series.index(peak_value)
    peak_time = times[peak_idx] if peak_idx < len(times) else 0.0
    # 尝试用 scipy.signal.find_peaks 做更精确的峰值检测
    try:
        from scipy.signal import find_peaks  # type: ignore[import-untyped]
        peaks, _ = find_peaks(series)
        if len(peaks) > 0:
            # 取最高峰
            best_peak = max(peaks, key=lambda i: series[i])
            peak_value = series[best_peak]
            peak_time = times[best_peak] if best_peak < len(times) else peak_time
    except Exception as exc:
        logger.debug("scipy.find_peaks 不可用，回退到 max() 检测：%s", exc)
    return float(peak_time), float(peak_value)


def _compute_fold(series: list[float], baseline_mode: str = "initial") -> float:
    """计算峰值的 fold change = peak / baseline。

    Args:
        series: 物种浓度时序。
        baseline_mode: "initial" → series[0]；"min" → min(series)。

    Returns:
        fold 值。baseline 为 0 时返回 0.0。
    """
    if not series:
        return 0.0
    peak = max(series)
    if baseline_mode == "min":
        baseline = min(series)
    else:
        baseline = series[0] if series else 0.0
    if abs(baseline) < 1e-12:
        # baseline 接近 0 时，用 peak 绝对值作为 fold 的退化估计
        return float(peak)
    return float(peak) / float(abs(baseline))


def _is_finite_series(times: list[float], species: dict[str, list[float]]) -> bool:
    """C3 稳定性检查：所有时间点和浓度均为有限值（非 NaN/Inf）。"""
    for t in times:
        if not math.isfinite(t):
            return False
    for values in species.values():
        for v in values:
            if not math.isfinite(v):
                return False
    return True


class DynamicsCalibrator:
    """确定性动力学校准器：仿真后检查峰值时间/幅度，网格搜索调整 kcat/kon。

    校准流程：
      1. 读取初始仿真 CSV，检测峰值时间/幅度。
      2. 若已在期望窗口内，直接返回 calibrated=True。
      3. 否则对 adjustable_params 做坐标下降网格搜索：
         - 每个参数在其 range 内按 log_scale/linear 生成 10 个网格点。
         - 对每个网格点调用 simulate_fn 重新仿真，检测峰值。
         - 选目标函数最小的点作为该参数的新值。
      4. 达到 max_iterations 或进入窗口则停止。
      5. C3 兜底：仿真不稳定（NaN/Inf）时回滚到初始参数，标记 SimulatorFailed。
    """

    def __init__(self, max_iterations: int = 20, tolerance: float = 0.2):
        self.max_iterations = max_iterations
        self.tolerance = tolerance  # max_relative_error 容差

    def calibrate(
        self,
        simulation_csv_path: str,
        expected_dynamics: dict,
        adjustable_params: dict,
        simulate_fn: Callable[[dict[str, float]], str],
    ) -> dict:
        """校准参数使仿真峰值时间/幅度落在期望窗口内。

        Args:
            simulation_csv_path: 仿真结果 CSV 路径。
            expected_dynamics: 期望动力学窗口，支持字段：
                - "peak_time_min": [low, high] 峰值时间窗口（分钟）
                - "peak_amplitude_fold": [low, high] 峰值幅度 fold 窗口
                - "peak_amplitude_norm": [low, high] 峰值幅度归一化窗口（备选）
                - "species": 监测的物种名（可选，缺省取 fold 最大的物种）
                - "baseline_mode": "initial" | "min"（默认 initial）
            adjustable_params: 可调参数 dict：
                {"k1": {"value": 0.1, "range": [0.01, 1.0], "log_scale": True}, ...}
            simulate_fn: 接收 params dict，返回新的 simulation_csv_path。

        Returns:
            {
                "calibrated": bool,
                "iterations": int,
                "final_params": {"k1": 0.15, ...},
                "calibration_log": [
                    {"iter": 1, "params": {...}, "peak_time": 7.6,
                     "peak_amplitude": 0.067, "in_window": False}, ...
                ],
                "final_metrics": {"peak_time": 12.3, "peak_amplitude": 8.5,
                                  "in_window": True},
                "error_class": None  # 或 "MaxIterationsExceeded" / "SimulatorFailed"
            }
        """
        random.seed(_CALIBRATOR_SEED)

        # 解析 expected_dynamics 窗口
        time_window = expected_dynamics.get("peak_time_min")
        amp_window = (
            expected_dynamics.get("peak_amplitude_fold")
            or expected_dynamics.get("peak_amplitude_norm")
        )
        target_species = expected_dynamics.get("species")
        baseline_mode = str(expected_dynamics.get("baseline_mode", "initial"))

        # 无窗口信息时无法校准
        if not time_window and not amp_window:
            return self._build_result(
                calibrated=False, iterations=0,
                final_params={k: v.get("value") for k, v in adjustable_params.items()},
                calibration_log=[],
                final_metrics={"peak_time": None, "peak_amplitude": None, "in_window": False},
                error_class=None,
            )

        time_mid = (float(time_window[0]) + float(time_window[1])) / 2.0 if time_window else None
        amp_mid = (float(amp_window[0]) + float(amp_window[1])) / 2.0 if amp_window else None

        # 初始参数（深拷贝）
        initial_params: dict[str, float] = {}
        for name, spec in adjustable_params.items():
            initial_params[name] = _safe_float(spec.get("value"), 0.1)
        current_params = dict(initial_params)

        calibration_log: list[dict] = []

        def _evaluate(params: dict[str, float], csv_path: str, iter_num: int) -> dict:
            """评估一组参数的仿真指标并写入日志。"""
            times, species = _read_simulation_series(csv_path)
            # C3 稳定性检查
            if not times or not species or not _is_finite_series(times, species):
                return {
                    "iter": iter_num,
                    "params": dict(params),
                    "peak_time": None,
                    "peak_amplitude": None,
                    "in_window": False,
                    "unstable": True,
                }
            peak_time, peak_amp, _ = self._extract_target_metrics(
                times, species, target_species, baseline_mode
            )
            in_window = self._in_window(peak_time, peak_amp, time_window, amp_window)
            return {
                "iter": iter_num,
                "params": dict(params),
                "peak_time": round(peak_time, 4) if peak_time is not None else None,
                "peak_amplitude": round(peak_amp, 4) if peak_amp is not None else None,
                "in_window": in_window,
                "unstable": False,
            }

        def _objective(metrics: dict) -> float:
            """目标函数：(peak_time - time_mid)^2 + (log(amp) - log(amp_mid))^2。"""
            pt = metrics.get("peak_time")
            pa = metrics.get("peak_amplitude")
            if pt is None or pa is None:
                return float("inf")
            obj = 0.0
            if time_mid is not None:
                obj += (float(pt) - time_mid) ** 2
            if amp_mid is not None and pa > 0:
                try:
                    obj += (math.log(max(pa, 1e-12)) - math.log(max(amp_mid, 1e-12))) ** 2
                except (ValueError, OverflowError):
                    obj += float("inf")
            return obj

        # 评估初始参数
        iter_num = 0
        initial_metrics = _evaluate(current_params, simulation_csv_path, iter_num)
        calibration_log.append(initial_metrics)

        # 已在窗口内，无需校准
        if initial_metrics.get("in_window"):
            return self._build_result(
                calibrated=True, iterations=0,
                final_params=current_params,
                calibration_log=calibration_log,
                final_metrics={
                    "peak_time": initial_metrics["peak_time"],
                    "peak_amplitude": initial_metrics["peak_amplitude"],
                    "in_window": True,
                },
                error_class=None,
            )

        # 不可调参数则直接返回
        if not adjustable_params:
            return self._build_result(
                calibrated=False, iterations=0,
                final_params=current_params,
                calibration_log=calibration_log,
                final_metrics={
                    "peak_time": initial_metrics.get("peak_time"),
                    "peak_amplitude": initial_metrics.get("peak_amplitude"),
                    "in_window": False,
                },
                error_class=None,
            )

        best_params = dict(current_params)
        best_obj = _objective(initial_metrics)
        best_metrics = initial_metrics

        # 构建各参数的网格点
        param_grids: dict[str, list[float]] = {}
        for name, spec in adjustable_params.items():
            param_grids[name] = self._build_grid(spec)

        # 坐标下降网格搜索
        param_names = list(adjustable_params.keys())
        error_class = None
        calibrated = False
        round_count = 0
        max_rounds = max(1, self.max_iterations // max(1, len(param_names)))
        # C3 兜底追踪：若所有 trial 仿真均不稳定，标记 SimulatorFailed
        total_trials = 0
        unstable_trials = 0

        while iter_num < self.max_iterations and not calibrated:
            round_count += 1
            if round_count > max_rounds * len(param_names) * 2:
                # 防御性退出：避免网格点不足时死循环
                break
            improved_this_round = False
            for pname in param_names:
                if iter_num >= self.max_iterations:
                    break
                grid = param_grids.get(pname, [])
                if not grid:
                    continue
                round_best_value = best_params.get(pname)
                round_best_obj = best_obj
                round_best_metrics = best_metrics
                round_best_csv = None
                for grid_value in grid:
                    if iter_num >= self.max_iterations:
                        break
                    iter_num += 1
                    trial_params = dict(best_params)
                    trial_params[pname] = grid_value
                    total_trials += 1
                    # 调用仿真
                    try:
                        trial_csv = simulate_fn(trial_params)
                    except Exception as exc:
                        logger.warning(
                            "DynamicsCalibrator simulate_fn 失败 (iter=%d, %s=%s): %s",
                            iter_num, pname, grid_value, exc,
                        )
                        unstable_trials += 1
                        calibration_log.append({
                            "iter": iter_num,
                            "params": dict(trial_params),
                            "peak_time": None,
                            "peak_amplitude": None,
                            "in_window": False,
                            "unstable": True,
                            "error": str(exc),
                        })
                        continue
                    metrics = _evaluate(trial_params, trial_csv, iter_num)
                    calibration_log.append(metrics)
                    # C3 兜底：仿真不稳定则跳过该点（不回滚整体，仅跳过）
                    if metrics.get("unstable"):
                        unstable_trials += 1
                        continue
                    obj = _objective(metrics)
                    if obj < round_best_obj:
                        round_best_obj = obj
                        round_best_value = grid_value
                        round_best_metrics = metrics
                        round_best_csv = trial_csv
                    # 进入窗口则提前结束
                    if metrics.get("in_window"):
                        best_params = dict(trial_params)
                        best_obj = obj
                        best_metrics = metrics
                        calibrated = True
                        break
                # 应用本轮该参数的最佳值
                if not calibrated and round_best_value is not None:
                    if round_best_obj < best_obj:
                        best_params[pname] = round_best_value
                        best_obj = round_best_obj
                        best_metrics = round_best_metrics
                        improved_this_round = True
            if calibrated:
                break
            if not improved_this_round and round_count > 1:
                # 连续无改进，退出避免空转
                logger.info(
                    "DynamicsCalibrator 第 %d 轮无改进，停止搜索", round_count,
                )
                break

        # C3 兜底：最终参数仿真不稳定则回滚到初始参数
        if best_metrics.get("unstable"):
            logger.warning(
                "DynamicsCalibrator 校准后仿真不稳定，回滚到初始参数"
            )
            return self._build_result(
                calibrated=False,
                iterations=iter_num,
                final_params=initial_params,
                calibration_log=calibration_log,
                final_metrics={
                    "peak_time": initial_metrics.get("peak_time"),
                    "peak_amplitude": initial_metrics.get("peak_amplitude"),
                    "in_window": False,
                },
                error_class="SimulatorFailed",
            )

        # C3 兜底：所有 trial 仿真均不稳定 → 仿真器失效，回滚到初始参数
        if total_trials > 0 and unstable_trials == total_trials:
            logger.warning(
                "DynamicsCalibrator 所有 %d 次 trial 仿真均不稳定，回滚到初始参数",
                total_trials,
            )
            return self._build_result(
                calibrated=False,
                iterations=iter_num,
                final_params=initial_params,
                calibration_log=calibration_log,
                final_metrics={
                    "peak_time": initial_metrics.get("peak_time"),
                    "peak_amplitude": initial_metrics.get("peak_amplitude"),
                    "in_window": False,
                },
                error_class="SimulatorFailed",
            )

        if not calibrated and iter_num >= self.max_iterations:
            error_class = "MaxIterationsExceeded"

        return self._build_result(
            calibrated=calibrated,
            iterations=iter_num,
            final_params=best_params,
            calibration_log=calibration_log,
            final_metrics={
                "peak_time": best_metrics.get("peak_time"),
                "peak_amplitude": best_metrics.get("peak_amplitude"),
                "in_window": bool(best_metrics.get("in_window", False)),
            },
            error_class=error_class,
        )

    # -------------------------------------------------------------------------
    # 辅助方法
    # -------------------------------------------------------------------------
    def _build_grid(self, spec: dict) -> list[float]:
        """为单个参数构建网格点（默认 10 个点）。"""
        value = _safe_float(spec.get("value"), 0.1)
        rng = spec.get("range")
        log_scale = bool(spec.get("log_scale", False))
        n_points = int(spec.get("n_points", 10))
        if not rng or len(rng) != 2:
            return [value]
        low = _safe_float(rng[0], value * 0.1)
        high = _safe_float(rng[1], value * 10.0)
        if low > high:
            low, high = high, low
        if low <= 0 and log_scale:
            # log_scale 但下界 <=0，回退到 linear
            log_scale = False
        grid: list[float] = []
        if log_scale and low > 0:
            log_low = math.log(low)
            log_high = math.log(high)
            for i in range(n_points):
                if n_points == 1:
                    frac = 0.0
                else:
                    frac = i / (n_points - 1)
                grid.append(math.exp(log_low + frac * (log_high - log_low)))
        else:
            for i in range(n_points):
                if n_points == 1:
                    frac = 0.0
                else:
                    frac = i / (n_points - 1)
                grid.append(low + frac * (high - low))
        # 确保包含当前值（便于回退）
        if value not in grid and low <= value <= high:
            grid.append(value)
        # 去重并排序（确定性）
        grid = sorted(set(round(v, 8) for v in grid))
        return grid

    def _extract_target_metrics(
        self,
        times: list[float],
        species: dict[str, list[float]],
        target_species: str | None,
        baseline_mode: str,
    ) -> tuple[float | None, float | None, str | None]:
        """提取目标物种的峰值时间和 fold amplitude。

        Returns:
            (peak_time, peak_amplitude_fold, used_species)。
        """
        if not species or not times:
            return None, None, None
        used_species = target_species
        if used_species and used_species in species:
            series = species[used_species]
            peak_time, peak_val = _detect_peak(times, series)
            fold = _compute_fold(series, baseline_mode)
            return peak_time, fold, used_species
        # 未指定/未命中物种：取 fold 最大的物种
        best_species = None
        best_fold = -1.0
        best_peak_time = None
        for sp, series in species.items():
            if not series or len(series) != len(times):
                continue
            peak_time, _ = _detect_peak(times, series)
            fold = _compute_fold(series, baseline_mode)
            if fold > best_fold:
                best_fold = fold
                best_species = sp
                best_peak_time = peak_time
        if best_species is None:
            return None, None, None
        return best_peak_time, best_fold, best_species

    def _in_window(
        self,
        peak_time: float | None,
        peak_amp: float | None,
        time_window: list | None,
        amp_window: list | None,
    ) -> bool:
        """判断峰值时间/幅度是否都在期望窗口内。"""
        if peak_time is None or peak_amp is None:
            return False
        if time_window and len(time_window) == 2:
            if not (float(time_window[0]) <= float(peak_time) <= float(time_window[1])):
                return False
        if amp_window and len(amp_window) == 2:
            if not (float(amp_window[0]) <= float(peak_amp) <= float(amp_window[1])):
                return False
        return True

    def _build_result(
        self,
        calibrated: bool,
        iterations: int,
        final_params: dict[str, float],
        calibration_log: list[dict],
        final_metrics: dict,
        error_class: str | None,
    ) -> dict:
        """构建统一返回结构。"""
        return {
            "calibrated": calibrated,
            "iterations": iterations,
            "final_params": {k: float(v) if v is not None else None for k, v in final_params.items()},
            "calibration_log": calibration_log,
            "final_metrics": final_metrics,
            "error_class": error_class,
        }


# =============================================================================
# 模块级便捷函数（供 SBMLValidator 集成调用）
# =============================================================================
def calibrate_dynamics(
    simulation_csv_path: str,
    expected_dynamics: dict,
    adjustable_params: dict,
    simulate_fn: Callable[[dict[str, float]], str],
    max_iterations: int = 20,
    tolerance: float = 0.2,
) -> dict:
    """便捷入口：创建 DynamicsCalibrator 并执行校准。"""
    calibrator = DynamicsCalibrator(max_iterations=max_iterations, tolerance=tolerance)
    return calibrator.calibrate(
        simulation_csv_path=simulation_csv_path,
        expected_dynamics=expected_dynamics,
        adjustable_params=adjustable_params,
        simulate_fn=simulate_fn,
    )


def check_peak_in_window(
    simulation_csv_path: str,
    expected_dynamics: dict,
) -> tuple[bool, dict]:
    """检查仿真峰值是否在期望窗口内（不校准，仅判定）。

    Returns:
        (in_window, metrics_dict)。metrics_dict 含 peak_time / peak_amplitude /
        species / in_window。
    """
    times, species = _read_simulation_series(simulation_csv_path)
    if not times or not species:
        return False, {"peak_time": None, "peak_amplitude": None, "in_window": False}
    target_species = expected_dynamics.get("species")
    baseline_mode = str(expected_dynamics.get("baseline_mode", "initial"))
    calibrator = DynamicsCalibrator()
    peak_time, peak_amp, used = calibrator._extract_target_metrics(
        times, species, target_species, baseline_mode
    )
    time_window = expected_dynamics.get("peak_time_min")
    amp_window = (
        expected_dynamics.get("peak_amplitude_fold")
        or expected_dynamics.get("peak_amplitude_norm")
    )
    in_window = calibrator._in_window(peak_time, peak_amp, time_window, amp_window)
    return in_window, {
        "peak_time": round(peak_time, 4) if peak_time is not None else None,
        "peak_amplitude": round(peak_amp, 4) if peak_amp is not None else None,
        "species": used,
        "in_window": in_window,
    }


__all__ = [
    "DynamicsCalibrator",
    "calibrate_dynamics",
    "check_peak_in_window",
]
