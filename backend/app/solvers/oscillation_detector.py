# BioDynamics Agent v4 - 振荡检测器
# 对应 PART C5 + 审计 §3.3 无振荡检测。
#
# 适用通路：p53 / NF-κB / TGF-β / JAK-STAT（含转录延迟负反馈）
#
# 检测方法：
#   1. 峰值检测：找出时间序列的所有局部极大值
#   2. 周期估计：相邻峰之间的时间间隔
#   3. 振幅衰减：判断是否为阻尼振荡 / 持续振荡 / 单调
#   4. 振荡判定：峰数 ≥ 2 且振幅衰减 < 50% → 持续振荡

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import find_peaks


def detect_oscillation(
    t_arr: np.ndarray,
    y_arr: np.ndarray,
    species_name: str = "",
    species_idx: int = -1,
    min_peaks: int = 2,
    amplitude_decay_threshold: float = 0.5,
) -> dict[str, Any]:
    """检测时间序列的振荡行为。

    Args:
        t_arr: 时间序列 (N_time,)
        y_arr: 状态序列 (N_time, N_species) 或 (N_time,)
        species_name: 检测的 species 名（用于报告）
        species_idx: 检测的 species 索引（-1 表示 y_arr 已是 1D）
        min_peaks: 判定振荡所需的最小峰数
        amplitude_decay_threshold: 振幅衰减阈值（第二峰/第一峰 < 此值 → 阻尼）

    Returns:
        dict: {
            "oscillatory": bool,        # 是否检测到振荡
            "n_peaks": int,             # 检测到的峰数
            "peak_times": list[float],  # 峰值时间
            "peak_values": list[float], # 峰值
            "period_minutes": float,    # 估计周期（分钟，-1 表示无法估计）
            "amplitude_first": float,   # 第一峰振幅
            "amplitude_last": float,    # 最后一峰振幅
            "decay_ratio": float,       # 振幅衰减比（last/first）
            "oscillation_type": str,    # "sustained" / "damped" / "monotonic"
            "species": str,             # 检测的 species 名
        }
    """
    # 提取 1D 时间序列
    if species_idx >= 0 and y_arr.ndim == 2:
        series = y_arr[:, species_idx]
    else:
        series = np.asarray(y_arr).flatten()

    t_arr = np.asarray(t_arr)

    # 基线（最小值）
    baseline = float(np.min(series))
    amplitude_full = float(np.max(series) - baseline)

    if amplitude_full < 1e-10:
        # 信号太平坦，无振荡
        return _no_oscillation_result(species_name, "monotonic")

    # 峰值检测
    # prominence：峰的显著度（避免噪声误判）
    prominence = amplitude_full * 0.1
    peaks, properties = find_peaks(series, prominence=prominence)

    n_peaks = len(peaks)
    peak_times = t_arr[peaks].tolist() if n_peaks > 0 else []
    peak_values = series[peaks].tolist() if n_peaks > 0 else []

    # 振幅（相对于基线）
    peak_amplitudes = [v - baseline for v in peak_values]

    if n_peaks < min_peaks:
        # 峰数不足，非振荡
        return _no_oscillation_result(species_name, "monotonic", n_peaks=n_peaks)

    # 周期估计（相邻峰间隔的平均值）
    if n_peaks >= 2:
        periods = np.diff(peak_times)
        period_minutes = float(np.mean(periods))
    else:
        period_minutes = -1.0

    # 振幅衰减比
    amp_first = peak_amplitudes[0] if peak_amplitudes else 0.0
    amp_last = peak_amplitudes[-1] if peak_amplitudes else 0.0
    decay_ratio = amp_last / amp_first if amp_first > 0 else 0.0

    # 振荡类型判定
    if decay_ratio >= amplitude_decay_threshold:
        osc_type = "sustained"  # 持续振荡
        oscillatory = True
    elif decay_ratio > 0.1:
        osc_type = "damped"  # 阻尼振荡
        oscillatory = True
    else:
        osc_type = "monotonic"  # 单调（衰减过快）
        oscillatory = False

    return {
        "oscillatory": oscillatory,
        "n_peaks": n_peaks,
        "peak_times": peak_times,
        "peak_values": peak_values,
        "period_minutes": period_minutes,
        "amplitude_first": float(amp_first),
        "amplitude_last": float(amp_last),
        "decay_ratio": float(decay_ratio),
        "oscillation_type": osc_type,
        "species": species_name,
    }


def _no_oscillation_result(
    species_name: str,
    osc_type: str,
    n_peaks: int = 0,
) -> dict[str, Any]:
    """返回非振荡结果。"""
    return {
        "oscillatory": False,
        "n_peaks": n_peaks,
        "peak_times": [],
        "peak_values": [],
        "period_minutes": -1.0,
        "amplitude_first": 0.0,
        "amplitude_last": 0.0,
        "decay_ratio": 0.0,
        "oscillation_type": osc_type,
        "species": species_name,
    }


def detect_oscillation_for_all_species(
    t_arr: np.ndarray,
    y_arr: np.ndarray,
    species_names: list[str],
) -> list[dict[str, Any]]:
    """对所有 species 执行振荡检测。

    Args:
        t_arr: 时间序列
        y_arr: 状态序列 (N_time, N_species)
        species_names: species 名列表

    Returns:
        list[dict]: 每个 species 的振荡检测结果
    """
    results: list[dict[str, Any]] = []
    for i, name in enumerate(species_names):
        if y_arr.ndim == 2 and i < y_arr.shape[1]:
            result = detect_oscillation(t_arr, y_arr, species_name=name, species_idx=i)
            results.append(result)
    return results


__all__ = [
    "detect_oscillation",
    "detect_oscillation_for_all_species",
]
