# Feature Flag: V4_SCIENTIFIC_REVIEWER_ENABLED
#
# BioDynamics Agent - Scientific Alignment Loop: 8 项曲线指标（Curve Metrics Suite）
# （Spec: add-scientific-reviewer-and-validation-matrix, Task 2）
#
# 模块用途：
#   为每条仿真曲线计算 8 项固定指标，全部数字输出，禁止肉眼判断。Scientific
#   Reviewer Simulation Review 项 SHALL 基于此 8 项对照 Expected。
#
# 对应 Spec Requirement：
#   - 8 项曲线指标（Curve Metrics Suite）
#
# 8 项指标定义：
#   1. Peak time         — 达峰时间（min）
#   2. Peak amplitude    — 峰值幅度
#   3. Rise time         — 上升时间（从 10% 到 90% 峰值的时间）
#   4. Half decay        — 半衰期（从峰值降到 50% 峰值的时间；若不下降则 None）
#   5. Adaptation ratio  — 适应性比率（稳态 / 峰值）
#   6. Steady state      — 稳态值（最后 10% 时间窗均值）
#   7. Oscillation       — 是否振荡（bool + 振荡周期；用峰谷计数法）
#   8. AUC               — 曲线下面积（归一化，梯形积分）
#
# 禁止肉眼判断原则：
#   Reviewer 评估曲线 SHALL 仅基于 8 项数字指标，禁止"看起来下降"/"似乎振荡"
#   等模糊措辞。每项 SHALL 与 Canonical Expected 数字对照，输出
#   Expected / Simulation / Status(PASS|FAIL)。
#
# Feature Flag 守护：
#   V4_SCIENTIFIC_REVIEWER_ENABLED 默认 false。关闭时上层 Pipeline 不调用本模块
#   （由 supervisor / validation_matrix 控制）；本模块自身保持纯数值工具，不读
#   Feature Flag，便于单元测试与复用。
#
# 核心导出：
#   from app.scientific_alignment.curve_metrics import (
#       CurveMetrics, ExpectedMetric, MetricComparison,
#       compute_curve_metrics, compute_all_curves, compare_with_expected,
#   )

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# 常量定义（对应 Spec 8 项指标定义）
# =============================================================================

# 8 项指标名称（顺序固定）
CURVE_METRIC_NAMES: tuple[str, ...] = (
    "peak_time",          # 1. Peak time（达峰时间，min）
    "peak_amplitude",     # 2. Peak amplitude（峰值幅度）
    "rise_time",          # 3. Rise time（10%→90% 峰值）
    "half_decay",         # 4. Half decay（峰值→50% 峰值）
    "adaptation_ratio",   # 5. Adaptation ratio（稳态 / 峰值）
    "steady_state",       # 6. Steady state（最后 10% 时间窗均值）
    "oscillation",        # 7. Oscillation（bool + 振荡周期）
    "auc",                # 8. AUC（曲线下面积，归一化）
)

# 稳态时间窗占比（最后 10%）
_STEADY_STATE_WINDOW_FRACTION: float = 0.1

# 振荡判定：相邻峰谷最小幅度差占峰值比例（去噪）
_OSCILLATION_MIN_AMPLITUDE_FRAC: float = 0.05

# 默认时间列候选名（按优先级）
_TIME_COLUMN_CANDIDATES: tuple[str, ...] = ("time", "t", "Time", "T")


class MetricStatus:
    """曲线指标对照状态（禁止 High/Medium/Low）。"""

    PASS: str = "PASS"
    FAIL: str = "FAIL"


# =============================================================================
# 数据类（@dataclass）
# =============================================================================


@dataclass
class CurveMetrics:
    """单条曲线的 8 项指标计算结果。

    全部数字输出，禁止肉眼判断。Oscillation 项含 bool 与周期两个子字段，
    共 9 个 dataclass 字段对应 8 项指标。

    Attributes:
        peak_time: 达峰时间（min）。
        peak_amplitude: 峰值幅度。
        rise_time: 上升时间（10%→90% 峰值，min）；无法计算时为 None。
        half_decay: 半衰期（峰值→50% 峰值，min）；不下降则为 None。
        adaptation_ratio: 适应性比率（稳态 / 峰值）。
        steady_state: 稳态值（最后 10% 时间窗均值）。
        oscillation: 是否振荡（bool）。
        oscillation_period: 振荡周期（min）；不振荡则为 None。
        auc: 曲线下面积（归一化，[0, 1]）。
        trend_slope: 线性回归斜率（辅助字段，供 "increasing"/"decreasing"
            趋势判定使用）。
    """

    peak_time: float
    peak_amplitude: float
    rise_time: float | None
    half_decay: float | None
    adaptation_ratio: float
    steady_state: float
    oscillation: bool
    oscillation_period: float | None
    auc: float
    # 辅助字段：线性回归斜率，供 trend 判定使用（非 8 项指标之一）
    trend_slope: float = 0.0

    def get_metric(self, name: str) -> float | bool | None:
        """按指标名获取值。

        Args:
            name: CURVE_METRIC_NAMES 之一。

        Returns:
            对应字段值；oscillation 返回 bool；未知名称返回 None。
        """
        if name == "oscillation":
            return self.oscillation
        if name == "oscillation_period":
            return self.oscillation_period
        return getattr(self, name, None)


@dataclass
class ExpectedMetric:
    """Expected 指标定义。

    Attributes:
        metric_name: 指标名称（CURVE_METRIC_NAMES 之一，如 "peak_time"）。
        expected_range: 期望范围，支持多种格式：
            - tuple (low, high): 范围，如 (10, 20)
            - float / int: 单点值（按 tolerance 容差比较）
            - str "10-20" / "10-20 min": 范围（可含单位后缀）
            - str "<5" / "<5 min": 上界
            - str ">0.3": 下界
            - str "increasing" / "decreasing": 趋势（用 trend_slope 判定）
            - str "true" / "false": 布尔（用于 oscillation）
        tolerance: 容差（相对，0.05 表示 ±5%）。
    """

    metric_name: str
    expected_range: tuple[float, float] | float | str | bool
    tolerance: float = 0.0


@dataclass
class MetricComparison:
    """actual vs expected 对照结果。

    Attributes:
        metric_name: 指标名称。
        expected: 期望值的字符串表示（用于报告渲染）。
        actual: 实际值的字符串表示。
        status: PASS / FAIL。
        reason: FAIL 原因或 PASS 时的简要说明。
    """

    metric_name: str
    expected: str
    actual: str
    status: str = MetricStatus.FAIL
    reason: str = ""


# =============================================================================
# 向后兼容：保留骨架定义的 CurveMetric / CurveMetricsReport
# =============================================================================


@dataclass
class CurveMetric:
    """单项曲线指标结果（旧 API，保留向后兼容）。

    对应 Spec Scenario "8 项指标定义"——每项 SHALL 与 Canonical Expected 数字
    对照，输出 Expected / Simulation / Status(PASS|FAIL)。
    """

    name: str
    simulation: float | tuple[float, float] | bool
    expected: float | tuple[float, float] | str
    status: str = MetricStatus.FAIL
    unit: str = ""


@dataclass
class CurveMetricsReport:
    """单条曲线的 8 项指标完整报告（旧 API，保留向后兼容）。

    Attributes:
        molecule: 分子名称（如 "pERK"）。
        metrics: 8 项 CurveMetric 列表（顺序与 CURVE_METRIC_NAMES 一致）。
        all_pass: 是否全部 PASS（任一关键指标 FAIL → Dynamics 轴 FAIL）。
    """

    molecule: str
    metrics: list[CurveMetric] = field(default_factory=list)
    all_pass: bool = False


# =============================================================================
# 内部辅助函数：8 项指标计算
# =============================================================================


def _to_arrays(
    time_points: Sequence[float], values: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    """将输入转换为 numpy 数组并校验。"""
    time_arr = np.asarray(time_points, dtype=float)
    values_arr = np.asarray(values, dtype=float)
    if time_arr.ndim != 1 or values_arr.ndim != 1:
        raise ValueError("time_points 和 values 必须为一维序列")
    if time_arr.shape[0] != values_arr.shape[0]:
        raise ValueError(
            f"time_points 和 values 长度不一致: {time_arr.shape[0]} vs {values_arr.shape[0]}"
        )
    if time_arr.shape[0] < 2:
        raise ValueError("time_points 和 values 长度需 >= 2")
    if not np.all(np.isfinite(time_arr)) or not np.all(np.isfinite(values_arr)):
        raise ValueError("time_points / values 含 NaN 或 Inf")
    return time_arr, values_arr


def _compute_peak(
    time_arr: np.ndarray, values_arr: np.ndarray
) -> tuple[float, float, int]:
    """计算 Peak time / Peak amplitude / 峰值索引。"""
    peak_idx = int(np.argmax(values_arr))
    peak_amp = float(values_arr[peak_idx])
    peak_time = float(time_arr[peak_idx])
    return peak_time, peak_amp, peak_idx


def _compute_rise_time(
    time_arr: np.ndarray,
    values_arr: np.ndarray,
    peak_amp: float,
    peak_idx: int,
) -> float | None:
    """计算 Rise time（10%→90% 峰值，在 peak_idx 之前的上升段）。"""
    if peak_idx == 0 or peak_amp == 0:
        return None
    target_low = 0.1 * peak_amp
    target_high = 0.9 * peak_amp
    t_low: float | None = None
    t_high: float | None = None
    # 在 [0, peak_idx] 上升段查找首次跨越
    for i in range(1, peak_idx + 1):
        v_prev = float(values_arr[i - 1])
        v_curr = float(values_arr[i])
        t_prev = float(time_arr[i - 1])
        t_curr = float(time_arr[i])
        if t_low is None and v_prev <= target_low <= v_curr:
            # 线性插值
            if v_curr == v_prev:
                t_low = t_prev
            else:
                frac = (target_low - v_prev) / (v_curr - v_prev)
                t_low = t_prev + frac * (t_curr - t_prev)
        if t_low is not None and v_prev <= target_high <= v_curr:
            if v_curr == v_prev:
                t_high = t_prev
            else:
                frac = (target_high - v_prev) / (v_curr - v_prev)
                t_high = t_prev + frac * (t_curr - t_prev)
            break
    if t_low is None or t_high is None:
        return None
    return t_high - t_low


def _compute_half_decay(
    time_arr: np.ndarray,
    values_arr: np.ndarray,
    peak_amp: float,
    peak_idx: int,
) -> float | None:
    """计算 Half decay（从峰值降到 50% 峰值的时间）。

    若峰值后曲线从未下降到 50% 峰值，返回 None。
    """
    if peak_amp == 0:
        return None
    target = 0.5 * peak_amp
    # 在 [peak_idx, end] 下降段查找首次跨越
    for i in range(peak_idx + 1, len(values_arr)):
        v_prev = float(values_arr[i - 1])
        v_curr = float(values_arr[i])
        # 从上方跨越 target
        if v_prev >= target >= v_curr:
            t_prev = float(time_arr[i - 1])
            t_curr = float(time_arr[i])
            if v_curr == v_prev:
                t_cross = t_prev
            else:
                frac = (v_prev - target) / (v_prev - v_curr)
                t_cross = t_prev + frac * (t_curr - t_prev)
            return t_cross - float(time_arr[peak_idx])
    return None


def _compute_steady_state(
    time_arr: np.ndarray, values_arr: np.ndarray
) -> float:
    """计算 Steady state（最后 10% 时间窗均值，按点数计）。"""
    n = len(values_arr)
    start_idx = max(0, n - max(1, int(n * _STEADY_STATE_WINDOW_FRACTION)))
    return float(np.mean(values_arr[start_idx:]))


def _compute_adaptation_ratio(steady_state: float, peak_amp: float) -> float:
    """计算 Adaptation ratio（稳态 / 峰值）。"""
    if peak_amp == 0:
        return 0.0
    return float(steady_state / peak_amp)


def _find_local_extrema(values_arr: np.ndarray) -> tuple[list[int], list[int]]:
    """用峰谷计数法找局部极大 / 极小索引。"""
    peaks: list[int] = []
    valleys: list[int] = []
    n = len(values_arr)
    for i in range(1, n - 1):
        v_prev = float(values_arr[i - 1])
        v_curr = float(values_arr[i])
        v_next = float(values_arr[i + 1])
        if v_curr > v_prev and v_curr > v_next:
            peaks.append(i)
        elif v_curr < v_prev and v_curr < v_next:
            valleys.append(i)
    return peaks, valleys


def _filter_extrema_by_amplitude(
    time_arr: np.ndarray,
    values_arr: np.ndarray,
    peaks: list[int],
    valleys: list[int],
    peak_amp: float,
) -> tuple[list[int], list[int]]:
    """按幅度阈值过滤噪声峰谷。

    相邻峰谷幅度差需 >= 阈值（5% 峰值），否则视为噪声丢弃。
    """
    threshold = peak_amp * _OSCILLATION_MIN_AMPLITUDE_FRAC
    # 合并峰谷并按位置排序
    merged = sorted(
        [(i, "peak") for i in peaks] + [(i, "valley") for i in valleys],
        key=lambda x: x[0],
    )
    if not merged:
        return [], []
    kept_peaks: list[int] = []
    kept_valleys: list[int] = []
    for j in range(len(merged) - 1):
        idx_a, type_a = merged[j]
        idx_b, type_b = merged[j + 1]
        amp_diff = abs(float(values_arr[idx_a]) - float(values_arr[idx_b]))
        if amp_diff >= threshold:
            if type_a == "peak":
                kept_peaks.append(idx_a)
            else:
                kept_valleys.append(idx_a)
    # 处理最后一个
    last_idx, last_type = merged[-1]
    if last_type == "peak":
        kept_peaks.append(last_idx)
    else:
        kept_valleys.append(last_idx)
    return kept_peaks, kept_valleys


def _compute_oscillation(
    time_arr: np.ndarray,
    values_arr: np.ndarray,
    peak_amp: float,
) -> tuple[bool, float | None]:
    """用峰谷计数法判定是否振荡，并返回振荡周期。

    判定：至少 2 个峰 或 2 个谷（过滤噪声后）。
    周期：相邻峰（或谷）间时间距离的均值。
    """
    peaks, valleys = _find_local_extrema(values_arr)
    peaks, valleys = _filter_extrema_by_amplitude(
        time_arr, values_arr, peaks, valleys, peak_amp
    )
    is_osc = len(peaks) >= 2 or len(valleys) >= 2
    if not is_osc:
        return False, None
    # 优先用峰计算周期，否则用谷
    if len(peaks) >= 2:
        idxs = peaks
    else:
        idxs = valleys
    if len(idxs) < 2:
        return False, None
    periods = [
        float(time_arr[idxs[k + 1]]) - float(time_arr[idxs[k]])
        for k in range(len(idxs) - 1)
    ]
    period = float(np.mean(periods))
    return True, period


def _trapezoid_area(y: np.ndarray, x: np.ndarray) -> float:
    """梯形积分（兼容 numpy 1.x 与 2.x）。

    numpy 2.0 移除了 ``np.trapz``，改名为 ``np.trapezoid``；为兼容两者
    与更早版本，优先用 numpy 提供的函数，回退到手算梯形公式。
    """
    # 优先用新版 API
    trapz_fn = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
    if trapz_fn is not None:
        try:
            return float(trapz_fn(y, x))
        except Exception:  # noqa: BLE001
            pass
    # 手算：sum((y[i+1] + y[i]) / 2 * (x[i+1] - x[i]))
    if len(x) < 2:
        return 0.0
    dx = np.diff(x)
    y_avg = (y[1:] + y[:-1]) / 2.0
    return float(np.sum(y_avg * dx))


def _compute_auc(
    time_arr: np.ndarray,
    values_arr: np.ndarray,
    peak_amp: float,
) -> float:
    """计算归一化 AUC（梯形积分 / (时间范围 × 峰值)）。

    归一化后 AUC ∈ [0, 1]，表示相对"峰值方波"的面积比例。
    """
    if peak_amp == 0:
        return 0.0
    time_range = float(time_arr[-1] - time_arr[0])
    if time_range <= 0:
        return 0.0
    area = _trapezoid_area(values_arr, time_arr)
    return area / (time_range * peak_amp)


def _compute_trend_slope(
    time_arr: np.ndarray, values_arr: np.ndarray
) -> float:
    """线性回归斜率（用于 increasing / decreasing 趋势判定）。"""
    if len(time_arr) < 2:
        return 0.0
    # numpy polyfit 一次多项式 → 斜率
    try:
        slope = float(np.polyfit(time_arr, values_arr, 1)[0])
    except (np.linalg.LinAlgError, ValueError):
        slope = 0.0
    return slope


# =============================================================================
# 核心接口
# =============================================================================


def compute_curve_metrics(
    time_points: list[float] | Sequence[float],
    values: list[float] | Sequence[float],
) -> CurveMetrics:
    """计算单条曲线的 8 项指标。

    对应 Spec Requirement "8 项曲线指标（Curve Metrics Suite）"。
    全部数字输出，禁止肉眼判断。

    Args:
        time_points: 时间点序列（min）。
        values: 对应时间点的值序列。

    Returns:
        CurveMetrics，含 8 项指标。

    Raises:
        ValueError: 输入长度不一致 / 含 NaN/Inf / 长度 < 2。
    """
    time_arr, values_arr = _to_arrays(time_points, values)

    peak_time, peak_amp, peak_idx = _compute_peak(time_arr, values_arr)
    rise_time = _compute_rise_time(time_arr, values_arr, peak_amp, peak_idx)
    half_decay = _compute_half_decay(time_arr, values_arr, peak_amp, peak_idx)
    steady_state = _compute_steady_state(time_arr, values_arr)
    adaptation_ratio = _compute_adaptation_ratio(steady_state, peak_amp)
    oscillation, oscillation_period = _compute_oscillation(
        time_arr, values_arr, peak_amp
    )
    auc = _compute_auc(time_arr, values_arr, peak_amp)
    trend_slope = _compute_trend_slope(time_arr, values_arr)

    return CurveMetrics(
        peak_time=peak_time,
        peak_amplitude=peak_amp,
        rise_time=rise_time,
        half_decay=half_decay,
        adaptation_ratio=adaptation_ratio,
        steady_state=steady_state,
        oscillation=oscillation,
        oscillation_period=oscillation_period,
        auc=auc,
        trend_slope=trend_slope,
    )


def compute_all_curves(
    simulation_csv_path: str,
    key_molecules: list[str],
    *,
    time_column: str | None = None,
) -> dict[str, CurveMetrics]:
    """从 simulation.csv 计算多条关键曲线的指标。

    Args:
        simulation_csv_path: simulation.csv 文件路径。
        key_molecules: 关键分子列名列表（如
            ["pEGFR", "pMEK", "pERK", "DUSP_mRNA", "DUSP_protein"]）。
        time_column: 时间列名；None 时按 ("time", "t", "Time", "T") 自动检测。

    Returns:
        {molecule: CurveMetrics} 字典。CSV 中不存在的分子会被跳过。

    Raises:
        FileNotFoundError: CSV 文件不存在。
        ValueError: CSV 为空 / 时间列未找到 / 数值解析失败。
    """
    with open(simulation_csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"CSV 文件为空: {simulation_csv_path}")

    headers = list(rows[0].keys())

    # 自动检测时间列
    actual_time_col: str | None = None
    if time_column is not None and time_column in headers:
        actual_time_col = time_column
    else:
        for candidate in _TIME_COLUMN_CANDIDATES:
            if candidate in headers:
                actual_time_col = candidate
                break
    if actual_time_col is None:
        raise ValueError(
            f"未找到时间列（尝试 {time_column or _TIME_COLUMN_CANDIDATES}），"
            f"CSV 列: {headers}"
        )

    # 提取时间序列
    try:
        time_points = [float(row[actual_time_col]) for row in rows]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"时间列 '{actual_time_col}' 含非数值: {exc}"
        ) from exc

    result: dict[str, CurveMetrics] = {}
    for mol in key_molecules:
        if mol not in headers:
            logger.warning("compute_all_curves: 分子 '%s' 不在 CSV 列中，跳过", mol)
            continue
        try:
            values = [float(row[mol]) for row in rows]
        except (TypeError, ValueError) as exc:
            logger.warning(
                "compute_all_curves: 分子 '%s' 含非数值，跳过: %s", mol, exc
            )
            continue
        result[mol] = compute_curve_metrics(time_points, values)

    return result


def _parse_numeric_range(
    expected_range: tuple[float, float] | float | str,
) -> tuple[str, Any]:
    """解析 expected_range 为 (kind, payload)。

    kind ∈ {"single", "range_tuple", "range_str", "upper", "lower",
            "trend", "bool_str", "unknown"}。
    """
    if isinstance(expected_range, bool):
        return "bool", expected_range
    if isinstance(expected_range, (int, float)):
        return "single", float(expected_range)
    if isinstance(expected_range, tuple) and len(expected_range) == 2:
        return "range_tuple", (float(expected_range[0]), float(expected_range[1]))
    if isinstance(expected_range, str):
        s = expected_range.strip()
        # 趋势
        lower = s.lower()
        if lower in ("increasing", "decreasing"):
            return "trend", lower
        if lower in ("true", "false"):
            return "bool_str", lower == "true"
        # 移除常见单位后缀（min / s / M / u / mol 等）
        s_no_unit = re.sub(
            r"\s*(min|sec|s\b|M\b|uM|nM|µM|mol|hours?|hrs?)\s*$",
            "",
            s,
            flags=re.IGNORECASE,
        ).strip()
        # 上界 / 下界
        m_upper = re.match(r"^<\s*(-?\d+(?:\.\d+)?)$", s_no_unit)
        if m_upper:
            return "upper", float(m_upper.group(1))
        m_lower = re.match(r"^>\s*(-?\d+(?:\.\d+)?)$", s_no_unit)
        if m_lower:
            return "lower", float(m_lower.group(1))
        # 范围 "low-high" / "low–high" / "low to high"
        m_range = re.match(
            r"^(-?\d+(?:\.\d+)?)\s*(?:-|–|—|to)\s*(-?\d+(?:\.\d+)?)$",
            s_no_unit,
        )
        if m_range:
            return "range_str", (float(m_range.group(1)), float(m_range.group(2)))
        return "unknown", s
    return "unknown", expected_range


def compare_with_expected(
    actual: CurveMetrics,
    expected: ExpectedMetric,
) -> MetricComparison:
    """actual vs expected_range → PASS/FAIL。

    支持多种 expected 格式：
        - tuple (low, high) 或 "low-high" / "low-high min": 范围
        - "<5" / "<5 min": 上界
        - ">0.3": 下界
        - "increasing" / "decreasing": 趋势（用 linear regression slope 判定）
        - float / int: 单点（按 tolerance 容差比较）
        - "true" / "false" / bool: 用于 oscillation

    Args:
        actual: CurveMetrics 实例。
        expected: ExpectedMetric 实例。

    Returns:
        MetricComparison，含 status (PASS/FAIL) 与 reason。
    """
    metric_name = expected.metric_name
    expected_range = expected.expected_range
    tolerance = max(0.0, float(expected.tolerance))
    actual_value = actual.get_metric(metric_name)

    expected_str = str(expected_range)
    actual_str = str(actual_value)

    # 趋势判定（用 trend_slope）
    kind, payload = _parse_numeric_range(expected_range)
    if kind == "trend":
        slope = actual.trend_slope
        if payload == "increasing":
            ok = slope > 0
            reason = (
                f"trend_slope={slope:.4f} > 0 → increasing"
                if ok
                else f"trend_slope={slope:.4f} ≤ 0，非 increasing"
            )
        else:  # decreasing
            ok = slope < 0
            reason = (
                f"trend_slope={slope:.4f} < 0 → decreasing"
                if ok
                else f"trend_slope={slope:.4f} ≥ 0，非 decreasing"
            )
        return MetricComparison(
            metric_name=metric_name,
            expected=expected_str,
            actual=actual_str,
            status=MetricStatus.PASS if ok else MetricStatus.FAIL,
            reason=reason,
        )

    # 布尔判定（用于 oscillation）
    if kind in ("bool", "bool_str"):
        expected_bool = bool(payload)
        if not isinstance(actual_value, bool):
            return MetricComparison(
                metric_name=metric_name,
                expected=expected_str,
                actual=actual_str,
                status=MetricStatus.FAIL,
                reason=f"指标 '{metric_name}' 不是 bool 类型",
            )
        ok = actual_value == expected_bool
        return MetricComparison(
            metric_name=metric_name,
            expected=expected_str,
            actual=actual_str,
            status=MetricStatus.PASS if ok else MetricStatus.FAIL,
            reason=(
                ""
                if ok
                else f"actual={actual_value} != expected={expected_bool}"
            ),
        )

    # None 处理：actual 为 None（如 half_decay 不下降）→ FAIL
    if actual_value is None:
        return MetricComparison(
            metric_name=metric_name,
            expected=expected_str,
            actual="None",
            status=MetricStatus.FAIL,
            reason=f"actual 为 None（如 half_decay 不存在），无法对照 expected",
        )

    if isinstance(actual_value, bool):
        # 实际是 bool 但 expected 非布尔格式
        return MetricComparison(
            metric_name=metric_name,
            expected=expected_str,
            actual=actual_str,
            status=MetricStatus.FAIL,
            reason=f"指标 '{metric_name}' 为 bool，expected 需为 true/false",
        )

    actual_num = float(actual_value)

    # 数值比较
    if kind == "single":
        target = float(payload)
        lo = target * (1.0 - tolerance)
        hi = target * (1.0 + tolerance)
        ok = lo <= actual_num <= hi
        reason = (
            f"actual={actual_num:.4f} ∈ [{lo:.4f}, {hi:.4f}] "
            f"(target={target}, tol={tolerance:.2%})"
        )
    elif kind == "range_tuple":
        lo, hi = payload
        delta = (hi - lo) * tolerance
        ok = (lo - delta) <= actual_num <= (hi + delta)
        reason = (
            f"actual={actual_num:.4f} ∈ [{lo - delta:.4f}, {hi + delta:.4f}] "
            f"(range=[{lo}, {hi}], tol={tolerance:.2%})"
        )
    elif kind == "range_str":
        lo, hi = payload
        delta = (hi - lo) * tolerance
        ok = (lo - delta) <= actual_num <= (hi + delta)
        reason = (
            f"actual={actual_num:.4f} ∈ [{lo - delta:.4f}, {hi + delta:.4f}] "
            f"(range=[{lo}, {hi}], tol={tolerance:.2%})"
        )
    elif kind == "upper":
        bound = float(payload)
        ok = actual_num <= bound * (1.0 + tolerance)
        reason = (
            f"actual={actual_num:.4f} ≤ {bound * (1.0 + tolerance):.4f} "
            f"(<{bound}, tol=+{tolerance:.2%})"
        )
    elif kind == "lower":
        bound = float(payload)
        ok = actual_num >= bound * (1.0 - tolerance)
        reason = (
            f"actual={actual_num:.4f} ≥ {bound * (1.0 - tolerance):.4f} "
            f"(>{bound}, tol=-{tolerance:.2%})"
        )
    else:
        return MetricComparison(
            metric_name=metric_name,
            expected=expected_str,
            actual=actual_str,
            status=MetricStatus.FAIL,
            reason=f"无法解析 expected_range 格式: {expected_range!r}",
        )

    status = MetricStatus.PASS if ok else MetricStatus.FAIL
    if not ok:
        reason = "FAIL: " + reason
    return MetricComparison(
        metric_name=metric_name,
        expected=expected_str,
        actual=actual_str,
        status=status,
        reason=reason if not ok else "",
    )


def validate_metric_range(
    actual: float,
    expected_range: tuple[float, float] | float,
    *,
    tolerance: float = 0.0,
) -> str:
    """校验单指标实际值是否在 Expected 范围内，返回 PASS / FAIL。

    旧 API（骨架保留）。仅支持数值范围/单点/上下界，不支持趋势（趋势需
    CurveMetrics.trend_slope，请用 compare_with_expected）。

    Args:
        actual: 仿真实际值。
        expected_range: Expected 范围 (low, high) 或单点值或字符串。
        tolerance: 容差（相对，0.05 表示 ±5%）。

    Returns:
        MetricStatus.PASS 或 MetricStatus.FAIL。
    """
    # 构造占位 CurveMetrics，仅用单值字段
    placeholder = CurveMetrics(
        peak_time=float(actual),
        peak_amplitude=float(actual),
        rise_time=float(actual),
        half_decay=float(actual),
        adaptation_ratio=float(actual),
        steady_state=float(actual),
        oscillation=False,
        oscillation_period=None,
        auc=float(actual),
        trend_slope=0.0,
    )
    em = ExpectedMetric(
        metric_name="peak_time",
        expected_range=expected_range,
        tolerance=tolerance,
    )
    return compare_with_expected(placeholder, em).status


__all__ = [
    "CURVE_METRIC_NAMES",
    "MetricStatus",
    "CurveMetrics",
    "ExpectedMetric",
    "MetricComparison",
    "CurveMetric",
    "CurveMetricsReport",
    "compute_curve_metrics",
    "compute_all_curves",
    "compare_with_expected",
    "validate_metric_range",
]
