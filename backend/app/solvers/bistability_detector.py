# BioDynamics Agent v4 - 双稳态检测器
# 对应 PART C5 + 审计 §3.4 双稳态检测缺失。
#
# 适用通路：Apoptosis（Caspase3-Bid 正反馈）/ Cell_Cycle（E2F-Rb 正反馈）
#
# 检测方法：
#   1. 找到关键 species（active / Cleaved / p 开头）
#   2. 检查最终状态是否进入高态（超过最大值的 threshold_ratio）
#   3. 检测状态切换时间（首次超过 threshold 的时间）
#   4. 判定：ON（高态）/ OFF（低态）/ TRANSIENT（中间）

from __future__ import annotations

from typing import Any

import numpy as np


def detect_bistability(
    t_arr: np.ndarray,
    y_arr: np.ndarray,
    species_names: list[str],
    threshold_ratio: float = 0.5,
    key_species_hint: str = "",
) -> dict[str, Any]:
    """检测双稳态行为。

    Args:
        t_arr: 时间序列 (N_time,)
        y_arr: 状态序列 (N_time, N_species)
        species_names: species 名列表
        threshold_ratio: 高/低态判定阈值（默认 0.5，即超过最大值一半为高态）
        key_species_hint: 关键 species 名提示（如 "Caspase3_active"），
                          为空时自动检测

    Returns:
        dict: {
            "bistable": bool,        # 是否检测到双稳态（最终进入高态）
            "final_state": str,      # "ON" / "OFF" / "TRANSIENT" / "UNKNOWN"
            "switch_time": float,    # 状态切换时间（-1 表示无切换）
            "key_species": str,      # 判定所用的关键 species
            "key_species_idx": int,  # 关键 species 索引
            "max_value": float,      # 关键 species 最大值
            "final_value": float,    # 关键 species 最终值
            "initial_value": float,  # 关键 species 初始值
            "threshold": float,      # 高/低态判定阈值
        }
    """
    t_arr = np.asarray(t_arr)
    y_arr = np.asarray(y_arr)

    # 寻找关键 species
    key_idx = _find_key_species(species_names, key_species_hint)

    if key_idx < 0 or y_arr.ndim != 2 or key_idx >= y_arr.shape[1]:
        return _unknown_result()

    key_series = y_arr[:, key_idx]
    max_val = float(np.max(key_series))
    final_val = float(key_series[-1])
    initial_val = float(key_series[0])
    min_val = float(np.min(key_series))
    threshold = max_val * threshold_ratio

    # 信号变化幅度（用于判定是否发生状态转换）
    amplitude = max_val - min_val

    # 判定最终状态
    if max_val < 1e-10 or amplitude < 1e-6:
        # 信号太平坦（无状态转换），无法判定双稳态
        final_state = "UNKNOWN"
        bistable = False
    elif final_val >= threshold and (final_val - initial_val) > amplitude * 0.3:
        # ON 态：最终值 >= threshold 且发生了明显的低→高转换
        final_state = "ON"
        bistable = True
    elif final_val < threshold * 0.1:
        final_state = "OFF"
        bistable = False
    else:
        final_state = "TRANSIENT"
        bistable = False

    # 检测切换时间（首次超过 threshold 的时间）
    switch_time = -1.0
    for i, val in enumerate(key_series):
        if val >= threshold:
            switch_time = float(t_arr[i])
            break

    return {
        "bistable": bistable,
        "final_state": final_state,
        "switch_time": switch_time,
        "key_species": species_names[key_idx],
        "key_species_idx": key_idx,
        "max_value": max_val,
        "final_value": final_val,
        "initial_value": initial_val,
        "threshold": threshold,
    }


def _find_key_species(species_names: list[str], hint: str = "") -> int:
    """寻找关键 species（用于双稳态判定）。

    优先级：
      1. hint 指定的 species
      2. 以 "_active" 结尾的 species（如 Caspase3_active）
      3. 以 "Cleaved" 开头的 species（如 CleavedPARP）
      4. 以 "p" 开头且第二个字符大写的 species（如 pRb / pEGFR）
    """
    # 1. hint
    if hint:
        for i, name in enumerate(species_names):
            if name == hint:
                return i

    # 2. _active 结尾
    for i, name in enumerate(species_names):
        if name.endswith("_active"):
            return i

    # 3. Cleaved 开头
    for i, name in enumerate(species_names):
        if name.startswith("Cleaved"):
            return i

    # 4. p 开头（磷酸化）
    for i, name in enumerate(species_names):
        if name.startswith("p") and len(name) > 1 and name[1:2].isupper():
            return i

    return -1


def _unknown_result() -> dict[str, Any]:
    """返回未知结果。"""
    return {
        "bistable": False,
        "final_state": "UNKNOWN",
        "switch_time": -1.0,
        "key_species": "",
        "key_species_idx": -1,
        "max_value": 0.0,
        "final_value": 0.0,
        "initial_value": 0.0,
        "threshold": 0.0,
    }


def detect_bistability_sweep(
    t_arr: np.ndarray,
    y_arr_list: list[np.ndarray],
    species_names: list[str],
    perturbation_values: list[float],
    key_species_hint: str = "",
) -> dict[str, Any]:
    """扫描扰动强度下的双稳态行为（用于全或无决策分析）。

    Args:
        t_arr: 时间序列
        y_arr_list: 多次仿真的状态序列列表（每次对应一个扰动强度）
        species_names: species 名列表
        perturbation_values: 扰动强度列表（如药物浓度）
        key_species_hint: 关键 species 提示

    Returns:
        dict: {
            "bistable_overall": bool,   # 整体是否呈现双稳态（有 ON/OFF 跳变）
            "threshold_perturbation": float,  # 跳变阈值（扰动强度）
            "sweep_results": list[dict],  # 每次扰动的双稳态结果
        }
    """
    sweep_results: list[dict[str, Any]] = []
    states: list[str] = []

    for y_arr in y_arr_list:
        result = detect_bistability(
            t_arr, y_arr, species_names, key_species_hint=key_species_hint
        )
        sweep_results.append(result)
        states.append(result["final_state"])

    # 检测 ON/OFF 跳变
    has_on = "ON" in states
    has_off = "OFF" in states
    bistable_overall = has_on and has_off

    # 找跳变阈值（首次从 OFF → ON 的扰动强度）
    threshold_perturbation = -1.0
    for i, state in enumerate(states):
        if state == "ON" and i > 0 and states[i - 1] == "OFF":
            if i - 1 < len(perturbation_values) and i < len(perturbation_values):
                threshold_perturbation = (
                    perturbation_values[i - 1] + perturbation_values[i]
                ) / 2.0
                break

    return {
        "bistable_overall": bistable_overall,
        "threshold_perturbation": threshold_perturbation,
        "sweep_results": sweep_results,
    }


__all__ = [
    "detect_bistability",
    "detect_bistability_sweep",
]
