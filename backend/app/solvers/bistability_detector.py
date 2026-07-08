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
    jump_threshold_ratio: float = 0.3,
) -> dict[str, Any]:
    """检测双稳态行为。

    Args:
        t_arr: 时间序列 (N_time,)
        y_arr: 状态序列 (N_time, N_species)
        species_names: species 名列表
        threshold_ratio: 高/低态判定阈值（默认 0.5，即超过最大值一半为高态）
        key_species_hint: 关键 species 名提示（如 "Caspase3_active"），
                          为空时自动检测
        jump_threshold_ratio: 切换事件判定阈值（默认 0.3，即相邻时间点
                              导数幅度超过信号幅度的 30% 时记为一次切换事件）

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
            "switching_events": list[dict],  # TD-030: 状态跳变切换事件列表
        }
    """
    t_arr = np.asarray(t_arr)
    y_arr = np.asarray(y_arr)

    # 寻找关键 species
    key_idx = _find_key_species(species_names, key_species_hint)

    if key_idx < 0 or y_arr.ndim != 2 or key_idx >= y_arr.shape[1]:
        # TD-030: 未知结果同样附带空 switching_events，保持字段一致
        result = _unknown_result()
        result["switching_events"] = []
        return result

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

    # === TD-030 (IB-031) 修复：检测双稳态切换事件 ===
    # 原检测器仅判定最终 ON/OFF 状态，未捕获"从一个稳态跳变到另一个稳态"的事件。
    # 此处基于相邻时间点的差分（导数近似）扫描突变点：当 |Δy| 超过信号幅度的
    # jump_threshold_ratio 时记为一次切换事件，记录时间/前值/后值/方向，
    # 供下游报告与可视化消费。
    switching_events = _detect_switching_events(
        t_arr, key_series, amplitude, jump_threshold_ratio
    )

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
        "switching_events": switching_events,
    }


def _detect_switching_events(
    t_arr: np.ndarray,
    key_series: np.ndarray,
    amplitude: float,
    jump_threshold_ratio: float,
) -> list[dict[str, Any]]:
    """扫描时间序列，识别双稳态切换事件（突变跳变点）。

    TD-030 (IB-031) 修复：当系统从一个稳态跳变到另一个稳态时，
    相邻时间点的差分（导数近似）会显著超过信号幅度的均值水平。
    本函数以 ``amplitude * jump_threshold_ratio`` 为阈值，逐点扫描
    突变点并记录切换事件，用于补全原检测器缺失的事件检测能力。

    Args:
        t_arr: 时间序列 (N_time,)
        key_series: 关键 species 的状态序列 (N_time,)
        amplitude: 信号幅度（max - min），用于换算跳变阈值
        jump_threshold_ratio: 跳变阈值占幅度的比例（如 0.3 表示 30%）

    Returns:
        切换事件列表，每个事件 dict 含：
            - time: 跳变发生时间
            - index: 跳变点索引
            - value_before: 跳变前的值
            - value_after: 跳变后的值
            - delta: 跳变幅度（value_after - value_before）
            - direction: "up"（向上跳变）/ "down"（向下跳变）
    """
    events: list[dict[str, Any]] = []
    n = len(key_series)
    if n < 2 or amplitude < 1e-6:
        # 时间点过少或信号幅度太小，无法判定跳变
        return events

    # 跳变阈值：信号幅度的 jump_threshold_ratio 倍
    jump_threshold = amplitude * jump_threshold_ratio

    for i in range(1, n):
        # 相邻时间点的差分（导数近似，未除以 dt，仅用于突变点检测）
        delta = float(key_series[i] - key_series[i - 1])
        if abs(delta) >= jump_threshold:
            direction = "up" if delta > 0 else "down"
            events.append({
                "time": float(t_arr[i]),
                "index": int(i),
                "value_before": float(key_series[i - 1]),
                "value_after": float(key_series[i]),
                "delta": delta,
                "direction": direction,
            })
    return events


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
