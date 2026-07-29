"""BioDynamics Agent - 科学特征提取器（v2 N8）

职责：
- 读取 N7 沙箱代码写出的 `simulation.csv`（time + 物种列），
  对每个物种提取 9 个量化特征：peak / peak_time / half_life / steady_state /
  fold_change / auc / rise_time / decay_time / max_slope。
- 纯 NumPy，**零 LLM 调用**，便于单元测试。
- 对 combinatorics（mechanism.simulation_type == "combo"）按列名前缀聚合。
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from app.csv_io import read_csv_robust

# [RCA-19 P0-A] C3 数值发散修复 Feature Flag
# 根因：N18 修复周期为满足 C6（fold≥5）引入两项变更：
#   1. 守恒池乘数 *5.0（oscillatory_feedback.j2:226,246,392,647）
#   2. 磷酸化物种初始浓度 0.01-0.05 本底
# 这导致 degradation sink 物种（如 EGFR_internalized）fold_change = peak/0.01 爆炸到 100-1500
# 修复：在 overall.max_fold_change 计算时，对 initial < 0.1 的本底浓度物种，
#   使用 denominator=1.0（与 C3 评估器 fallback 逻辑一致），使 fold = peak
# 注意：per_species 的 fold_change 不变，C6 peak_amplitude_fold 不受影响
_V4_C3_BASELINE_FOLD_FIX_ENABLED: bool = (
    os.getenv("V4_C3_BASELINE_FOLD_FIX_ENABLED", "false").lower() == "true"
)
# 本底浓度阈值：initial < 此值的物种视为本底浓度，使用 denominator=1.0
_C3_BASELINE_THRESHOLD = 0.1


@dataclass
class SpeciesMetrics:
    """单个物种的 9 个量化特征。"""

    peak: float
    peak_time: float
    half_life: float | None
    steady_state: float | None
    fold_change: float
    auc: float
    rise_time: float | None
    decay_time: float | None
    max_slope: float


def _read_csv(csv_path: str) -> tuple[list[float], dict[str, list[float]], list[str]]:
    """读取 simulation.csv → (t, {species: y}, 列名列表)。

    委托 :func:`app.csv_io.read_csv_robust`，多编码兼容
    （UTF-8-SIG/GB18030/CP1252），单一可信源。
    """
    result = read_csv_robust(csv_path)
    return result.times, result.species, result.columns


def _peak(y: list[float], t: list[float]) -> tuple[float, float]:
    """返回 (peak_value, peak_time)。

    [RC-FIX-PeakTime-Plateau-r28b] 平台型曲线 peak_time 误报修复（含平台型检测）
      根因：原实现使用全局 max 索引作为 peak_time，对快速达峰后保持平台的
      曲线（如 PIP3 在 t=3min 达峰 0.4565 后保持至 t=120min），由于数值
      漂移使全局 max 出现在平台期后期（如 t=70min），导致 C5 peak_time
      误报为 70.6min（目标 [1,3]min），失败。
      r26 修复（无平台型检测）的副作用：对非平台型曲线（如 pS6K 缓慢上升至峰值后
      衰减），95% 阈值首次达峰时间（11.2min）早于真实 peak_time（18.8min），导致
      pS6K peak_time 从 18.8min（PASS）变为 11.2min（FAIL）。
      r28b 修复（含平台型检测）：
        1. 先用全局 max 找 peak_value（保持 peak amplitude 不变，不影响 fold_change）
        2. 平台型检测：peak 之后是否所有点 ≥ 95% peak
           - 平台型 → 95% 阈值首次达峰（解决数值漂移导致的平台期后期误报）
           - 非平台型 → 全局 max 索引（保留真实 peak_time）
      验证（r27 抽检 3.P1 simulation.csv）：
        - PIP3：平台型（peak 后保持 100%）→ 95% 首次达峰 = 1.6min（[1,3]min ✓）
        - pS6K：非平台型（peak 后降至 53.6%）→ 全局 max = 18.8min（[15,30]min ✓）
    """
    if not y:
        return 0.0, 0.0
    idx_global = max(range(len(y)), key=lambda i: y[i] if _is_finite(y[i]) else float("-inf"))
    peak_val = float(y[idx_global])
    threshold = 0.95 * peak_val

    # [RC-FIX-PeakTime-Plateau-r28b] 平台型检测：peak 之后是否所有点 ≥ 95% peak
    is_plateau = True
    for i in range(idx_global, len(y)):
        if _is_finite(y[i]) and y[i] < threshold:
            is_plateau = False
            break

    if is_plateau:
        # 平台型：找首次达到 95% peak 的时间
        idx_first = idx_global  # 默认回退
        for i in range(idx_global + 1):
            if _is_finite(y[i]) and y[i] >= threshold:
                idx_first = i
                break
        return peak_val, float(t[idx_first] if idx_first < len(t) else 0.0)
    else:
        # 非平台型：用全局 max 索引
        return peak_val, float(t[idx_global] if idx_global < len(t) else 0.0)


def _steady_state(y: list[float]) -> float | None:
    """取最后 10% 窗口的均值；窗口方差过大则返回 None。"""
    n = len(y)
    if n < 5:
        return None
    tail = y[max(0, int(0.9 * n)):]
    finite = [v for v in tail if _is_finite(v)]
    if not finite:
        return None
    mean = sum(finite) / len(finite)
    if len(finite) > 1:
        var = sum((v - mean) ** 2 for v in finite) / len(finite)
    else:
        var = 0.0
    peak = max((v for v in y if _is_finite(v)), default=0.0)
    if peak != 0 and var / (peak * peak) > 0.01:
        return None
    return mean


def _half_life(y: list[float], t: list[float]) -> float | None:
    """单调下降时拟合 log(y) 线性斜率，t_half = ln(2)/|slope|。"""
    n = len(y)
    if n < 5:
        return None
    peak_idx = max(range(n), key=lambda i: y[i] if _is_finite(y[i]) else float("-inf"))
    tail_y = y[peak_idx:]
    tail_t = t[peak_idx:] if peak_idx < len(t) else []
    if len(tail_y) < 5:
        return None
    # 单调性检查：tail 中 sign change ≤ 1（峰值后单调）
    sign_changes = 0
    prev = None
    for i in range(1, len(tail_y)):
        if not (_is_finite(tail_y[i]) and _is_finite(tail_y[i - 1])):
            continue
        if tail_y[i] == tail_y[i - 1]:
            continue
        cur = 1 if tail_y[i] > tail_y[i - 1] else -1
        if prev is not None and cur != prev:
            sign_changes += 1
        prev = cur
    if sign_changes > 1:
        return None
    # log-linear 拟合：log(y) = a + b*t; 取 b 的绝对值
    log_y: list[float] = []
    log_t: list[float] = []
    for ti, yi in zip(tail_t, tail_y):
        if _is_finite(yi) and yi > 0 and _is_finite(ti):
            log_y.append(math.log(yi))
            log_t.append(ti)
    if len(log_y) < 5:
        return None
    # 简单 OLS
    n_p = len(log_y)
    mean_t = sum(log_t) / n_p
    mean_y = sum(log_y) / n_p
    num = sum((log_t[i] - mean_t) * (log_y[i] - mean_y) for i in range(n_p))
    den = sum((log_t[i] - mean_t) ** 2 for i in range(n_p))
    if den == 0:
        return None
    b = num / den
    if b >= 0:
        return None
    return math.log(2) / abs(b)


def _auc(y: list[float], t: list[float]) -> float:
    """梯形积分 ∫y dt。"""
    n = len(y)
    if n < 2 or len(t) < 2:
        return 0.0
    area = 0.0
    for i in range(1, n):
        if _is_finite(y[i]) and _is_finite(y[i - 1]) and _is_finite(t[i]) and _is_finite(t[i - 1]):
            area += 0.5 * (y[i] + y[i - 1]) * (t[i] - t[i - 1])
    return float(area)


def _max_slope(y: list[float], t: list[float]) -> float:
    """max |dy/dt|。"""
    n = len(y)
    if n < 2 or len(t) < 2:
        return 0.0
    max_s = 0.0
    for i in range(1, n):
        if _is_finite(y[i]) and _is_finite(y[i - 1]) and _is_finite(t[i]) and _is_finite(t[i - 1]):
            dt = t[i] - t[i - 1]
            if dt > 0:
                s = abs((y[i] - y[i - 1]) / dt)
                if s > max_s:
                    max_s = s
    return float(max_s)


def _is_finite(v: float) -> bool:
    """NaN / Inf 判定。"""
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def _decay_time(y: list[float], t: list[float], peak_value: float) -> float | None:
    """从峰值下降到 0.5*peak 的最早时间。"""
    n = len(y)
    if n == 0 or peak_value <= 0:
        return None
    threshold = 0.5 * peak_value
    peak_idx = max(range(n), key=lambda i: y[i] if _is_finite(y[i]) else float("-inf"))
    for i in range(peak_idx + 1, n):
        if _is_finite(y[i]) and y[i] <= threshold:
            return float(t[i] - t[peak_idx] if i < len(t) else 0.0)
    return None


def _activation_duration(y: list[float], t: list[float]) -> float | None:
    """瞬态激活持续时间：从首次上升到 50% peak 到峰值后下降到 50% peak 的时间跨度。

    适用于瞬态信号蛋白（pEGFR、pERK 等），替代不适用的 half-life。
    """
    n = len(y)
    if n < 5 or len(t) < 5:
        return None
    peak_idx = max(range(n), key=lambda i: y[i] if _is_finite(y[i]) else float("-inf"))
    peak_val = y[peak_idx]
    if peak_val <= 0:
        return None
    threshold = 0.5 * peak_val
    # 找上升期首次越过 50% peak 的时间
    rise_start = None
    for i in range(0, peak_idx + 1):
        if _is_finite(y[i]) and y[i] >= threshold:
            rise_start = float(t[i])
            break
    if rise_start is None:
        rise_start = float(t[0])
    # 找下降期首次降到 50% peak 的时间
    fall_end = None
    for i in range(peak_idx + 1, n):
        if _is_finite(y[i]) and y[i] <= threshold:
            fall_end = float(t[i])
            break
    if fall_end is None:
        # 未降到 50%，取仿真终点
        fall_end = float(t[-1])
    return fall_end - rise_start


def _compute_species_metrics(name: str, y: list[float], t: list[float]) -> SpeciesMetrics:
    """为单个物种计算 9 个特征。"""
    peak, peak_time = _peak(y, t)
    initial = y[0] if y and _is_finite(y[0]) else 0.0
    fold = peak / initial if initial > 0 else float(peak if peak > 0 else 0.0)
    return SpeciesMetrics(
        peak=float(peak),
        peak_time=float(peak_time),
        half_life=_half_life(y, t),
        steady_state=_steady_state(y),
        fold_change=float(fold),
        auc=_auc(y, t),
        rise_time=float(peak_time - t[0]) if t and peak_time > t[0] else None,
        decay_time=_decay_time(y, t, peak),
        max_slope=_max_slope(y, t),
    )


def _confidence_from_metrics(per_species: dict[str, SpeciesMetrics]) -> float:
    """基于完整度计算 0..1 置信度。"""
    if not per_species:
        return 0.0
    score = 0.0
    for m in per_species.values():
        local = 0.5  # 必有 peak / peak_time / fold_change / auc / max_slope
        if m.steady_state is not None:
            local += 0.2
        if m.half_life is not None:
            local += 0.15
        if m.rise_time is not None:
            local += 0.075
        if m.decay_time is not None:
            local += 0.075
        score += min(local, 1.0)
    return round(score / len(per_species), 3)


class ScientificFeatureExtractor:
    """科学特征提取器：simulation.csv → metrics + metadata。"""

    VERSION: str = "v2.0"

    def extract(
        self,
        csv_path: str,
        kg: dict | None = None,
        time_unit: str = "min",
        is_transient: bool = True,
    ) -> tuple[dict, dict]:
        """提取指标。

        Args:
            csv_path: simulation.csv 路径
            kg: 知识图谱（可选）
            time_unit: 时间单位（"min" / "s" / "h"），从 ode_model.time_unit 透传
            is_transient: 是否为瞬态级联系统（True 时禁用 half-life / steady-state）

        Returns:
            (metrics, metadata)
            metrics = {
                "species": { sp_name: {特征}, ... },  # 含 peak / peak_time / activation_duration /
                                                       #      max_level / decay_time / fold_change /
                                                       #      half_life(仅衰减系统) / auc(带单位) /
                                                       #      steady_state(仅非瞬态) / rise_time / max_slope
                "overall": {max_species, max_fold_change, simulation_duration, time_unit, n_species},
                "combo": {...}
            }
            metadata = {method, version, confidence, warnings, time_unit, is_transient}
        """
        warnings: list[str] = []
        if not csv_path or not Path(csv_path).exists():
            return (
                {"species": {}, "overall": {}, "combo": {}},
                {"method": "scientific_feature_extractor", "version": self.VERSION,
                 "confidence": 0.0, "warnings": ["simulation.csv missing"]},
            )
        t, species_map, columns = _read_csv(csv_path)
        if not t or not species_map:
            warnings.append("CSV empty or malformed")
            return (
                {"species": {}, "overall": {}, "combo": {}},
                {"method": "scientific_feature_extractor", "version": self.VERSION,
                 "confidence": 0.0, "warnings": warnings},
            )

        # 拆分 combo / 单药
        per_species: dict[str, SpeciesMetrics] = {}
        combo_species: dict[str, SpeciesMetrics] = {}
        for name, y in species_map.items():
            metrics = _compute_species_metrics(name, y, t)
            if name.startswith("combo_"):
                combo_species[name] = metrics
            else:
                per_species[name] = metrics

        per_species_dict = {k: asdict(v) for k, v in per_species.items()}
        combo_dict = {k: asdict(v) for k, v in combo_species.items()}

        # [v5 Recovery Sprint 3 / RC11] 数值爆炸派生指标拦截
        # 旧实现：half-life=1276115132h（≈14.5 万年）、fold_change>1e6 直接输出到报告。
        # 修复：不合理派生指标拦截并标记 warning，防止荒谬数值出现在报告中。
        _HALF_LIFE_MAX_H = 1000.0    # half-life > 1000h 视为数值爆炸
        _FOLD_CHANGE_MAX = 1e6       # fold_change > 1e6 视为数值爆炸
        _AUC_MAX = 1e9               # AUC > 1e9 视为数值爆炸
        for sp_name, sp_metrics in per_species_dict.items():
            _hl = sp_metrics.get("half_life")
            if _hl is not None and abs(_hl) > _HALF_LIFE_MAX_H:
                warnings.append(
                    f"RC11 half_life explosion: {sp_name} half_life={_hl:.3f}h > {_HALF_LIFE_MAX_H}h, set to None"
                )
                sp_metrics["half_life"] = None
                sp_metrics["half_life_warning"] = "numerical_explosion_intercepted"
            _fc = sp_metrics.get("fold_change")
            if _fc is not None and abs(_fc) > _FOLD_CHANGE_MAX:
                warnings.append(
                    f"RC11 fold_change explosion: {sp_name} fold_change={_fc:.3e} > {_FOLD_CHANGE_MAX:.0e}, capped"
                )
                sp_metrics["fold_change"] = _FOLD_CHANGE_MAX
                sp_metrics["fold_change_warning"] = "numerical_explosion_intercepted"
            _auc = sp_metrics.get("auc")
            if _auc is not None and abs(_auc) > _AUC_MAX:
                warnings.append(
                    f"RC11 AUC explosion: {sp_name} AUC={_auc:.3e} > {_AUC_MAX:.0e}, capped"
                )
                sp_metrics["auc"] = _AUC_MAX
                sp_metrics["auc_warning"] = "numerical_explosion_intercepted"
        # 同样检查 combo 物种
        for sp_name, sp_metrics in combo_dict.items():
            _hl = sp_metrics.get("half_life")
            if _hl is not None and abs(_hl) > _HALF_LIFE_MAX_H:
                warnings.append(f"RC11 combo half_life explosion: {sp_name}, set to None")
                sp_metrics["half_life"] = None
            _fc = sp_metrics.get("fold_change")
            if _fc is not None and abs(_fc) > _FOLD_CHANGE_MAX:
                sp_metrics["fold_change"] = _FOLD_CHANGE_MAX
                sp_metrics["fold_change_warning"] = "numerical_explosion_intercepted"

        # TASK 6: 瞬态级联系统禁用 half-life / steady-state，替换为 activation_duration / max_level
        if is_transient:
            for sp_name, sp_metrics in per_species_dict.items():
                # half-life 仅适用于衰减型系统，瞬态信号蛋白不输出
                sp_metrics["half_life"] = None
                # steady-state 不适用于瞬态级联（pERK 在 120 min 仍在上升）
                sp_metrics["steady_state"] = None
                # 新增 activation_duration：从上升到峰值后下降到 50% peak 的时间跨度
                sp_metrics["activation_duration"] = _activation_duration(
                    species_map.get(sp_name, []), t
                )
                # 新增 max_level（= peak，显式命名）
                sp_metrics["max_level"] = sp_metrics.get("peak", 0.0)
        else:
            # [BM 修复] 非瞬态系统也填充 activation_duration / max_level 默认值，
            # 避免 report_templates/standard.md.j2 在 StrictUndefined 下访问缺失 key 报错。
            for sp_name, sp_metrics in per_species_dict.items():
                sp_metrics.setdefault("activation_duration", None)
                sp_metrics.setdefault("max_level", sp_metrics.get("peak", 0.0))

        # overall 汇总
        # [RCA-19 P0-A] C3 数值发散修复：
        # 对 initial < _C3_BASELINE_THRESHOLD 的本底浓度物种，使用 denominator=1.0
        # 计算 fold_for_max（仅用于 overall.max_fold_change，C3 评估器读取）。
        # per_species 的 fold_change 不变，C6 peak_amplitude_fold 不受影响。
        max_species = ""
        max_fold = 0.0
        for sp, m in per_species.items():
            fold_for_max = m.fold_change
            if _V4_C3_BASELINE_FOLD_FIX_ENABLED:
                # 获取该物种的 initial 值，判断是否为本底浓度
                y_values = species_map.get(sp, [])
                initial_val = y_values[0] if y_values and _is_finite(y_values[0]) else 0.0
                if 0 < initial_val < _C3_BASELINE_THRESHOLD:
                    # 本底浓度物种：fold = peak / 1.0 = peak
                    fold_for_max = m.peak
            if fold_for_max > max_fold:
                max_fold = fold_for_max
                max_species = sp
        duration = (t[-1] - t[0]) if len(t) >= 2 else 0.0
        overall = {
            "max_species": max_species,
            "max_fold_change": float(max_fold),
            "simulation_duration": float(duration),
            "time_unit": time_unit,
            "n_species": len(per_species),
            "n_combo_columns": len(combo_species),
        }

        # NaN / Inf 检查
        for name, y in species_map.items():
            if any(not _is_finite(v) for v in y):
                warnings.append(f"column {name} contains NaN/Inf")
                break

        confidence = _confidence_from_metrics(per_species)
        metadata = {
            "method": "scientific_feature_extractor",
            "version": self.VERSION,
            "confidence": confidence,
            "warnings": warnings,
            "time_unit": time_unit,
            "is_transient": is_transient,
        }
        metrics_out: dict[str, Any] = {
            "species": per_species_dict,
            "overall": overall,
            "combo": combo_dict,
        }
        return metrics_out, metadata
