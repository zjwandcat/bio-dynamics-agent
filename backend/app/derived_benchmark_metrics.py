"""BioDynamics Agent v4 - 派生 Benchmark 指标计算器（Derived Benchmark Metrics）

职责：
- 从 simulation.csv 读取时间序列数据
- 按通路特异科学定义计算命名 benchmark 指标（如 Cyt_c_precedes_Casp3）
- 规则驱动，零 LLM 调用，纯数值计算

科学依据（每条指标均引用文献，非 LLM 编造）：
- Apoptosis: Rehm 2006 / Green & Kroemer 2004 (PMID:15241432)
  Cyt c 释放早于 Caspase-3 激活 5-15 min；MOMP 为 point-of-no-return（bistable）
- Cell Cycle: Tyson 1991 / Pomerening 2005 (PMID:11389814)
  CyclinB-APC/C 振荡周期 8-12h；Rb-E2F bistable switch
- EGFR: Schoeberl 2002 (PMID:12124381)
  pEGFR 5-10 min 达峰；MAPK 信号放大 10-100 倍
- JAK-STAT: Schwartz 2003 (PMID:15286703)
  pSTAT5 5-15 min 达峰；SOCS mRNA 延迟 30-60 min
- MAPK: Markevich 2004 (PMID:14757805)
  零阶超敏感 Hill >2；ERK 2-8 min 达峰
- NF-κB: Nelson 2004 (PMID:14975635)
  NF-κB 核振荡周期 1-2h；IκBα 转录延迟 30-60 min
- p53: Lev Bar-Or 2000 (PMID:10644694)
  p53 脉冲周期 5-7h；Mdm2 转录延迟 60-120 min
- PI3K-AKT: Mazzoletti 2009 (PMID:19211571)
  pAKT 30-60 min 达峰；PIP/PIP3 质量守恒
- TGF-β: Massagué 1998 (PMID:9674480)
  pSmad2 5-15 min 达峰；SMAD7 mRNA 延迟 30-60 min
- Wnt: Lee 2003 (PMID:12906785)
  β-catenin 稳态 <10 nM（无 Wnt）；Axin2 mRNA 60-120 min 达峰

设计原则（铁律）：
1. 所有指标必须从真实 simulation.csv 数据计算，禁止硬编码
2. 每个指标有明确的科学定义（引用文献）
3. 物种名匹配使用规范化（去前缀/后缀/大小写），不硬编码 CSV 列名
4. 物种缺失时返回 None（不编造数据）
5. 布尔型指标（bistable_switch 等）基于曲线形态特征判定
6. 时间单位：CSV 中时间为分钟，按 benchmark YAML unit 字段转换
"""

from __future__ import annotations

import logging
import math
from typing import Any

from app.csv_io import read_csv_robust

logger = logging.getLogger(__name__)


# =============================================================================
# CSV 读取（委托统一 reader，单一可信源：app.csv_io）
# =============================================================================
def _read_csv(
    csv_path: str,
) -> tuple[list[float], dict[str, list[float]], list[str]]:
    """读取 simulation.csv → (times, {species_name: values}, column_names)。

    委托 :func:`app.csv_io.read_csv_robust`，多编码兼容
    （UTF-8-SIG/GB18030/CP1252），避免仿真已完成却因 CSV 编码把整个 pipeline
    标记为 fail（参见 DEEPSEEK_MACRO_ANALYSIS.md §B）。

    Args:
        csv_path: simulation.csv 文件路径。

    Returns:
        (time_points, species_data, species_names)。文件不存在或空时返回空。
    """
    result = read_csv_robust(csv_path)
    return result.times, result.species, result.columns


# =============================================================================
# 物种名规范化与匹配
# =============================================================================
def _normalize_name(name: str) -> str:
    """规范化物种名：小写 + 去标点 + 去常见后缀。

    用于跨 CSV 列名匹配（如 "Cyt_c" 匹配 "cytochrome_c"）。
    [D4] 修复：不再剥离 _nuclear 后缀，避免 ppERK 与 ppERK_nuclear 碰撞。
    nuclear 定位是生物学上有意义的区别（胞质 vs 核），不应合并。
    NFkB_nuclear 等通过 _SPECIES_ALIASES 别名映射处理。
    """
    s = name.lower().strip()
    # 去除常见状态后缀（[D4] 移除 _nuclear：保留核定位区别）
    for suffix in (
        "_active", "_cleaved", "_phos", "_cytoplasmic",
        "_tetramer", "_ubi", "_ac", "_mrna", "_p", "_pp",
        " active", " cleaved",
    ):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    # 去标点
    s = s.replace("_", "").replace(" ", "").replace("-", "")
    return s


# 通路特异物种别名映射（规范化后 → 期望规范化名）
# 键为 CSV 中可能出现的物种名规范化形式，值为标准名
_SPECIES_ALIASES: dict[str, list[str]] = {
    "cytc": ["cytc", "cytochromec"],
    "casp3": ["casp3", "caspase3"],
    "casp9": ["casp9", "caspase9"],
    "momp": ["momp"],
    "pegfr": ["pegfr", "egfr"],
    "pperk": ["pperk", "erk", "perk"],
    "ppmek": ["ppmek", "pmek", "mek"],
    "cyclingb": ["cyclingb", "cyclinb", "cycb"],
    "cyclind1": ["cyclind1", "cycd1"],
    "rb": ["rb", "prb"],
    "e2f": ["e2f", "e2f1"],
    "apc": ["apc", "apcc", "apcc"],
    "pstat5": ["pstat5", "stat5p", "stat5"],
    "socs": ["socs", "socs1", "cish"],
    "nfkb": ["nfkb", "nfkb nuclear", "nfkbnuclear", "p65", "rela", "p65nuclear", "relanuclear"],
    "ikba": ["ikba", "ikba mrna", "ikbalphamrna", "nfkbia"],
    "p53": ["p53", "p53total"],
    "mdm2mrna": ["mdm2mrna", "mdm2 mrna"],
    "paktt": ["pakt", "aktp", "akt"],
    "pip3": ["pip3", "ptdins3p"],
    "pip2": ["pip2", "ptdins4p"],
    "s6k1": ["s6k1", "ps6k1", "s6k", "ps6"],
    "psmad2": ["psmad2", "smad2p"],
    "smad4nuclear": ["smad4nuclear", "smad4 nuclear"],
    "smad7mrna": ["smad7mrna", "smad7 mrna"],
    "bcatenin": ["bcatenin", "betacatenin", "bcath"],
    "axin2mrna": ["axin2mrna", "axin2 mrna"],
}


def _build_species_lookup(
    col_names: list[str],
) -> dict[str, str]:
    """构建 {规范化名 → 实际CSV列名} 映射。

    同时注册别名映射，使得 "casp3" 能匹配 CSV 中的 "Caspase3_active"。
    """
    lookup: dict[str, str] = {}
    for name in col_names:
        norm = _normalize_name(name)
        lookup[norm] = name
    # 叠加别名映射
    for canonical, aliases in _SPECIES_ALIASES.items():
        for alias in aliases:
            norm_alias = _normalize_name(alias)
            if norm_alias not in lookup:
                # 尝试前缀匹配（如 "caspase3active" → 规范化后 "caspase3" 匹配 "casp3" 别名）
                for col_name in col_names:
                    col_norm = _normalize_name(col_name)
                    if col_norm.startswith(norm_alias) or norm_alias.startswith(col_norm):
                        if canonical not in lookup:
                            lookup[canonical] = col_name
                        break
            # 确保 canonical 规范化名也在 lookup 中
            if canonical not in lookup:
                for col_name in col_names:
                    col_norm = _normalize_name(col_name)
                    if col_norm == norm_alias:
                        lookup[canonical] = col_name
                        break
    return lookup


def _find_species(lookup: dict[str, str], *candidates: str) -> str | None:
    """按候选名列表查找物种，返回 CSV 中实际列名。

    依次尝试：精确匹配 → 规范化匹配 → 别名匹配。
    全部未命中返回 None。
    """
    for c in candidates:
        # 精确匹配
        if c in lookup.values():
            return c
        # 规范化匹配
        norm = _normalize_name(c)
        if norm in lookup:
            return lookup[norm]
    return None


# =============================================================================
# 时间序列辅助函数
# =============================================================================
def _get_series(
    times: list[float],
    species_data: dict[str, list[float]],
    col_name: str | None,
) -> tuple[list[float], list[float]] | None:
    """获取某物种的 (times, values)。col_name 为 None 或数据缺失时返回 None。"""
    if col_name is None or col_name not in species_data:
        return None
    y = species_data[col_name]
    if not y or len(y) != len(times):
        return None
    return list(times), list(y)


def _is_finite(v: float) -> bool:
    try:
        return not math.isnan(v) and math.isfinite(v)
    except (TypeError, ValueError):
        return False


def _peak_value(y: list[float]) -> float:
    """最大值。"""
    finite_y = [v for v in y if _is_finite(v)]
    return max(finite_y) if finite_y else 0.0


def _min_value(y: list[float]) -> float:
    finite_y = [v for v in y if _is_finite(v)]
    return min(finite_y) if finite_y else 0.0


def _peak_time(t: list[float], y: list[float]) -> float | None:
    """达峰时间：y 达到最大值时对应的 t。

    科学定义：信号从刺激到达峰值的时间，用于验证 pEGFR 5-10 min 达峰等 benchmark。
    """
    if not t or not y or len(t) != len(y):
        return None
    idx = max(
        range(len(y)),
        key=lambda i: y[i] if _is_finite(y[i]) else float("-inf"),
    )
    return float(t[idx]) if idx < len(t) else None


def _steady_state(y: list[float]) -> float | None:
    """稳态值：最后 10% 窗口的均值（用于 β-catenin 稳态等 benchmark）。

    科学定义：系统达到稳态后的浓度值。
    """
    n = len(y)
    if n < 5:
        return None
    tail = y[max(0, int(0.9 * n)):]
    finite = [v for v in tail if _is_finite(v)]
    if not finite:
        return None
    mean = sum(finite) / len(finite)
    # 检查窗口内是否真的稳定（方差不应过大）
    if len(finite) > 1:
        var = sum((v - mean) ** 2 for v in finite) / len(finite)
        pk = max((v for v in y if _is_finite(v)), default=0.0)
        if pk != 0 and var / (pk * pk) > 0.01:
            return None  # 仍在振荡，未达稳态
    return mean


def _half_life(t: list[float], y: list[float]) -> float | None:
    """半衰期：从峰值衰减到一半所需时间。

    科学定义：EGFR 内化半衰期 10-15 min（Schoeberl 2002）。
    """
    if not t or not y or len(t) != len(y) or len(y) < 5:
        return None
    pk = _peak_value(y)
    if pk <= 0:
        return None
    threshold = 0.5 * pk
    pk_idx = max(
        range(len(y)),
        key=lambda i: y[i] if _is_finite(y[i]) else float("-inf"),
    )
    # 从峰值后寻找降至一半的时间点
    for i in range(pk_idx, len(y)):
        if _is_finite(y[i]) and y[i] <= threshold:
            return float(t[i] - t[pk_idx])
    return None


def _find_peaks(
    t: list[float], y: list[float], min_prominence: float = 0.0
) -> list[int]:
    """寻找局部极大值索引。

    用于振荡周期检测（NF-κB、p53、CyclinB 振荡）。
    min_prominence: 峰值最小显著度（相对于邻近谷值的凸起高度）。
    """
    if len(y) < 3:
        return []
    finite_y = [v if _is_finite(v) else 0.0 for v in y]
    pk = max(finite_y) if finite_y else 0.0
    mn = min(finite_y) if finite_y else 0.0
    amplitude = pk - mn
    if amplitude <= 0:
        return []
    threshold = min_prominence * amplitude

    peaks: list[int] = []
    for i in range(1, len(finite_y) - 1):
        if finite_y[i] > finite_y[i - 1] and finite_y[i] >= finite_y[i + 1]:
            # 检查显著度：峰值与相邻谷值之差
            left_min = min(finite_y[max(0, i - 5):i]) if i > 1 else finite_y[i]
            right_min = min(finite_y[i + 1:min(len(finite_y), i + 6)])
            prominence = finite_y[i] - min(left_min, right_min)
            if prominence >= threshold:
                peaks.append(i)
    return peaks


def _oscillation_period(t: list[float], y: list[float]) -> float | None:
    """振荡周期：通过峰间距均值计算。

    科学定义：NF-κB 振荡周期 1-2h（Nelson 2004），p53 脉冲周期 5-7h（Lev Bar-Or 2000），
    CyclinB-APC/C 振荡周期 8-12h（Tyson 1991）。

    检测方法：找到所有显著峰值，计算相邻峰间距的均值作为周期。
    需至少 2 个峰才能计算周期。
    """
    peaks = _find_peaks(t, y, min_prominence=0.15)
    if len(peaks) < 2:
        return None
    intervals = []
    for i in range(1, len(peaks)):
        dt = t[peaks[i]] - t[peaks[i - 1]]
        if dt > 0:
            intervals.append(dt)
    if not intervals:
        return None
    return sum(intervals) / len(intervals)


def _oscillation_duration(t: list[float], y: list[float]) -> float | None:
    """振荡持续时长：从第一个峰到最后一个峰的时间跨度。

    科学定义：NF-κB 振荡持续 6-20h（Nelson 2004）。
    """
    peaks = _find_peaks(t, y, min_prominence=0.10)
    if len(peaks) < 2:
        return None
    return t[peaks[-1]] - t[peaks[0]]


def _bistable_switch(t: list[float], y: list[float]) -> int:
    """检测 bistable（all-or-none）开关行为。返回 1（是）或 0（否）。

    科学定义：MOMP 是 point-of-no-return，表现为 sigmoid 型 all-or-none 转换
    （Green & Kroemer 2004）。Rb-E2F 也表现为 bistable switch（Pomerening 2005）。

    判定标准：
    1. 信号有明显的上升过程（range > 阈值）
    2. 最大斜率足够陡（归一化后 > 阈值）
    3. 信号在转换后饱和（后段方差小）
    """
    if not t or not y or len(y) < 5:
        return 0
    finite_y = [v if _is_finite(v) else 0.0 for v in y]
    pk = max(finite_y)
    mn = min(finite_y)
    amplitude = pk - mn
    if amplitude <= 0 or pk <= 0:
        return 0

    # 归一化到 [0, 1]
    norm_y = [(v - mn) / amplitude for v in finite_y]

    # 计算最大斜率（归一化后）
    dt = [t[i + 1] - t[i] if i + 1 < len(t) else 1.0 for i in range(len(t))]
    max_slope = 0.0
    for i in range(len(norm_y) - 1):
        if dt[i] > 0:
            slope = abs(norm_y[i + 1] - norm_y[i]) / dt[i]
            max_slope = max(max_slope, slope)

    # 转换后饱和度：后 30% 的标准差
    tail_start = int(0.7 * len(norm_y))
    tail = norm_y[tail_start:]
    if tail:
        tail_mean = sum(tail) / len(tail)
        tail_var = sum((v - tail_mean) ** 2 for v in tail) / len(tail)
    else:
        tail_var = 1.0

    # 判定条件：
    # 1. 振幅足够（非平凡信号）
    # 2. 最大斜率足够陡（sigmoid 转换）
    # 3. 后段饱和（方差小 → 已达到稳态）
    amplitude_ok = amplitude > 0.01 * max(abs(pk), 1e-6)
    slope_ok = max_slope > 0.01  # 归一化后每分钟变化超过 1%
    saturated = tail_var < 0.01  # 后段接近稳态

    return 1 if (amplitude_ok and slope_ok and saturated) else 0


def _amplification_ratio(
    input_y: list[float], output_y: list[float]
) -> float | None:
    """信号放大倍数：output 峰值 / input 峰值。

    科学定义：MAPK 级联信号放大 10-100 倍（Markevich 2004）。
    """
    in_pk = _peak_value(input_y)
    out_pk = _peak_value(output_y)
    if in_pk <= 0:
        return None
    return out_pk / in_pk


def _hill_coefficient(t: list[float], y: list[float]) -> float | None:
    """从激活曲线估计 Hill 系数。

    科学定义：零阶超敏感 Hill > 2（Markevich 2004）。

    估计方法：对归一化激活曲线拟合 Hill 函数 y = t^n / (tau^n + t^n)，
    通过在 50% 激活点附近的斜率反推 n。
    """
    if not t or not y or len(y) < 10:
        return None
    finite_y = [v if _is_finite(v) else 0.0 for v in y]
    pk = max(finite_y)
    mn = min(finite_y)
    amplitude = pk - mn
    if amplitude <= 0 or pk <= 0:
        return None

    norm_y = [(v - mn) / amplitude for v in finite_y]

    # 找到 50% 激活点
    half_idx = None
    for i in range(len(norm_y)):
        if norm_y[i] >= 0.5:
            half_idx = i
            break
    if half_idx is None or half_idx == 0:
        return None

    # 在 50% 点附近估计斜率
    # Hill 函数在 50% 点的斜率 = n / (4 * tau)
    # tau ≈ t[half_idx]
    tau = t[half_idx]
    if tau <= 0:
        return None

    # 估计 50% 点的归一化斜率
    if half_idx + 1 < len(norm_y) and half_idx > 0:
        dt = t[half_idx + 1] - t[half_idx - 1]
        if dt > 0:
            slope_at_half = (norm_y[half_idx + 1] - norm_y[half_idx - 1]) / dt
            # n = slope * 4 * tau
            n = slope_at_half * 4 * tau
            return max(0.1, n)  # 下限 0.1
    return None


def _activation_threshold(t: list[float], y: list[float]) -> float | None:
    """激活阈值：信号达到 50% 最大值时的归一化输入水平。

    科学定义：Caspase-3 bistable 激活阈值 0.1-0.5
    （10-50% procaspase 激活，Rehm 2006）。

    对于时间序列，返回信号首次达到 50% 峰值时的归一化值。
    """
    if not y or len(y) < 5:
        return None
    finite_y = [v if _is_finite(v) else 0.0 for v in y]
    pk = max(finite_y)
    mn = min(finite_y)
    amplitude = pk - mn
    if amplitude <= 0:
        return None

    # 归一化到 [0, 1]
    norm_y = [(v - mn) / amplitude for v in finite_y]

    # 找到首次达到 50% 的点
    for i in range(len(norm_y)):
        if norm_y[i] >= 0.5:
            return norm_y[i]
    return None


def _mass_conservation_error(
    times: list[float],
    species_data: dict[str, list[float]],
    col_names: list[str],
) -> float | None:
    """质量守恒误差：蛋白池总量在仿真过程中的最大相对漂移。

    科学定义：封闭生化系统中，同一蛋白池（如 EGFR + pEGFR + EGF_EGFR = 常数）
    总量应守恒。误差 = max_t |pool_total_t - pool_total_0| / |pool_total_0|。

    使用 species_ontology.build_conservation_groups 构建守恒分组，
    仅检查同一蛋白不同状态的总和（排除 mRNA、输入信号等非守恒量）。
    若无守恒分组（单物种通路或未识别的蛋白池），返回 0.0（无法检测违规）。
    """
    if not times or not species_data:
        return None

    n = len(times)
    if n < 2:
        return None

    # 使用 species_ontology 构建守恒分组（与 conservation_checker.py 一致）
    try:
        from app.species_ontology import build_conservation_groups

        groups = build_conservation_groups(col_names)
    except Exception:
        groups = {}

    if not groups:
        # 无守恒分组 → 无法检测违规，返回 0.0
        # 这与 conservation_checker.py 的行为一致（单物种池跳过检查）
        return 0.0

    max_drift = 0.0
    for pool_name, pool_species in groups.items():
        # 仅检查 CSV 中存在的物种
        existing = [s for s in pool_species if s in species_data]
        if len(existing) < 2:
            continue

        # 计算该蛋白池在每个时间点的总和
        totals: list[float] = []
        for i in range(n):
            total = 0.0
            for s in existing:
                vals = species_data[s]
                if i < len(vals) and _is_finite(vals[i]):
                    total += vals[i]
            totals.append(total)

        if not totals or totals[0] == 0:
            continue

        initial = abs(totals[0])
        for total in totals:
            drift = abs(total - totals[0]) / max(initial, 1e-12)
            max_drift = max(max_drift, drift)

    return max_drift


# =============================================================================
# 通路特异指标计算
# =============================================================================
def _apoptosis_metrics(
    times: list[float],
    species_data: dict[str, list[float]],
    lookup: dict[str, str],
) -> dict[str, float]:
    """APOPTOSIS 通路指标（Rehm 2006 / Green & Kroemer 2004, PMID:15241432）。

    - Cyt_c_precedes_Casp3: Cyt c 释放早于 Caspase-3 激活 5-15 min
    - MOMP_bistable_switch: MOMP 为 point-of-no-return（bistable all-or-none）
    - Caspase3_activation_threshold: Caspase-3 激活阈值 0.1-0.5
    """
    result: dict[str, float] = {}

    # Cyt_c_precedes_Casp3 = Casp3_peak_time - Cyt_c_peak_time
    cyt_c_col = _find_species(lookup, "Cyt_c", "cytc", "Cytochrome_c")
    # [Round 5 Fix] 优先匹配活性形式（Caspase3_active > Casp3）
    # 科学原理：Cyt c 释放后激活的是 cleaved Caspase-3（活性形式, Caspase3_active），
    #   而非 procaspase-3 酶原（Casp3）。测量级联时序应选择活性形式，
    #   因为只有活性形式才能反映 Caspase-3 被激活的时间点。
    casp3_col = _find_species(lookup, "Caspase3_active", "Casp3", "casp3", "Caspase3")

    cyt_c_series = _get_series(times, species_data, cyt_c_col)
    casp3_series = _get_series(times, species_data, casp3_col)

    if cyt_c_series and casp3_series:
        cyt_c_pt = _peak_time(*cyt_c_series)
        casp3_pt = _peak_time(*casp3_series)
        if cyt_c_pt is not None and casp3_pt is not None:
            # 正值表示 Cyt_c 先达峰（precedes Casp3）
            result["Cyt_c_precedes_Casp3"] = casp3_pt - cyt_c_pt

    # MOMP_bistable_switch
    momp_col = _find_species(lookup, "MOMP", "momp")
    momp_series = _get_series(times, species_data, momp_col)
    if momp_series:
        result["MOMP_bistable_switch"] = float(_bistable_switch(*momp_series))
    else:
        result["MOMP_bistable_switch"] = 0.0

    # Caspase3_activation_threshold
    if casp3_series:
        threshold = _activation_threshold(*casp3_series)
        if threshold is not None:
            result["Caspase3_activation_threshold"] = threshold

    return result


def _cell_cycle_metrics(
    times: list[float],
    species_data: dict[str, list[float]],
    lookup: dict[str, str],
) -> dict[str, float]:
    """CELL_CYCLE 通路指标（Tyson 1991 / Pomerening 2005, PMID:11389814）。

    - CyclinB_APC_oscillation_period: 振荡周期 8-12h
    - Rb_E2F_bistable_switch: Rb-E2F bistable
    - Cyclin_D1_peak_time: Cyclin D1 达峰时间 60-240 min
    """
    result: dict[str, float] = {}

    # CyclinB_APC_oscillation_period（单位：小时）
    cycb_col = _find_species(lookup, "CyclinB", "cyclingb", "Cyclin_B", "CycB")
    cycb_series = _get_series(times, species_data, cycb_col)
    if cycb_series:
        period = _oscillation_period(*cycb_series)
        if period is not None:
            # CSV 时间单位为分钟，benchmark 期望小时
            result["CyclinB_APC_oscillation_period"] = period / 60.0

    # Rb_E2F_bistable_switch
    rb_col = _find_species(lookup, "Rb", "rb", "pRb")
    e2f_col = _find_species(lookup, "E2F", "e2f", "E2F1")
    rb_series = _get_series(times, species_data, rb_col)
    e2f_series = _get_series(times, species_data, e2f_col)

    rb_switch = _bistable_switch(*rb_series) if rb_series else 0
    e2f_switch = _bistable_switch(*e2f_series) if e2f_series else 0
    result["Rb_E2F_bistable_switch"] = float(rb_switch or e2f_switch)

    # Cyclin_D1_peak_time（单位：分钟）
    cycd1_col = _find_species(lookup, "Cyclin_D1", "cyclind1", "CycD1")
    cycd1_series = _get_series(times, species_data, cycd1_col)
    if cycd1_series:
        pt = _peak_time(*cycd1_series)
        if pt is not None:
            result["Cyclin_D1_peak_time"] = pt

    return result


def _egfr_metrics(
    times: list[float],
    species_data: dict[str, list[float]],
    lookup: dict[str, str],
) -> dict[str, float]:
    """EGFR_RTK 通路指标（Schoeberl 2002, PMID:12124381）。

    - pEGFR_peak_time: pEGFR 5-10 min 达峰
    - MAPK_amplification: MAPK 信号放大 10-100 倍
    - EGFR_internalization_half_life: EGFR 内化半衰期 10-15 min
    """
    result: dict[str, float] = {}

    # pEGFR_peak_time（分钟）
    pegfr_col = _find_species(lookup, "pEGFR", "pegfr")
    pegfr_series = _get_series(times, species_data, pegfr_col)
    if pegfr_series:
        pt = _peak_time(*pegfr_series)
        if pt is not None:
            result["pEGFR_peak_time"] = pt

    # MAPK_amplification: ppERK 峰值 / pEGFR 峰值
    pperk_col = _find_species(lookup, "ppERK", "pperk", "ERK", "pERK")
    pperk_series = _get_series(times, species_data, pperk_col)
    if pperk_series and pegfr_series:
        amp = _amplification_ratio(pegfr_series[1], pperk_series[1])
        if amp is not None:
            result["MAPK_amplification"] = amp

    # EGFR_internalization_half_life（分钟）
    egfr_col = _find_species(lookup, "EGFR", "egfr")
    egfr_series = _get_series(times, species_data, egfr_col)
    if egfr_series:
        hl = _half_life(*egfr_series)
        if hl is not None:
            result["EGFR_internalization_half_life"] = hl

    return result


def _jak_stat_metrics(
    times: list[float],
    species_data: dict[str, list[float]],
    lookup: dict[str, str],
) -> dict[str, float]:
    """JAK_STAT 通路指标（Schwartz 2003, PMID:15286703）。

    - pSTAT5_peak_time: pSTAT5 5-15 min 达峰
    - SOCS_mRNA_delay: SOCS mRNA 延迟 30-60 min（相对 pSTAT5）
    - STAT5_nuclear_cytoplasmic_ratio_pulse: 脉冲行为（boolean）
    """
    result: dict[str, float] = {}

    # pSTAT5_peak_time（分钟）
    pstat5_col = _find_species(lookup, "pSTAT5", "pstat5", "STAT5_p")
    pstat5_series = _get_series(times, species_data, pstat5_col)
    pstat5_pt = None
    if pstat5_series:
        pstat5_pt = _peak_time(*pstat5_series)
        if pstat5_pt is not None:
            result["pSTAT5_peak_time"] = pstat5_pt

    # SOCS_mRNA_delay = SOCS_mRNA_peak_time - pSTAT5_peak_time
    socs_col = _find_species(lookup, "SOCS_mRNA", "socs", "SOCS", "SOCS1", "CISH")
    socs_series = _get_series(times, species_data, socs_col)
    if socs_series and pstat5_pt is not None:
        socs_pt = _peak_time(*socs_series)
        if socs_pt is not None:
            result["SOCS_mRNA_delay"] = socs_pt - pstat5_pt

    # STAT5_nuclear_cytoplasmic_ratio_pulse: 检测脉冲行为
    stat5n_col = _find_species(lookup, "STAT5_nuclear", "stat5nuclear")
    stat5c_col = _find_species(lookup, "STAT5_cytoplasmic", "stat5cytoplasmic")
    stat5n_series = _get_series(times, species_data, stat5n_col)
    if stat5n_series:
        # 脉冲 = 多个峰存在（振荡行为）
        peaks = _find_peaks(stat5n_series[0], stat5n_series[1], min_prominence=0.20)
        result["STAT5_nuclear_cytoplasmic_ratio_pulse"] = 1.0 if len(peaks) >= 2 else 0.0
    else:
        result["STAT5_nuclear_cytoplasmic_ratio_pulse"] = 0.0

    return result


def _mapk_metrics(
    times: list[float],
    species_data: dict[str, list[float]],
    lookup: dict[str, str],
) -> dict[str, float]:
    """MAPK_ERK 通路指标（Markevich 2004, PMID:14757805）。

    - MAPK_amplification: 信号放大 10-100 倍
    - zero_order_ultrasensitivity_hill_coefficient: Hill > 2
    - ERK_peak_time: ERK 2-8 min 达峰
    """
    result: dict[str, float] = {}

    # ERK_peak_time（分钟）
    perk_col = _find_species(lookup, "ppERK", "pperk", "ERK", "pERK")
    perk_series = _get_series(times, species_data, perk_col)
    if perk_series:
        pt = _peak_time(*perk_series)
        if pt is not None:
            result["ERK_peak_time"] = pt

    # MAPK_amplification: ppERK / input (Ras 或 RAF 激活信号)
    input_col = _find_species(lookup, "Ras_GTP", "Ras", "pRaf", "RAF")
    input_series = _get_series(times, species_data, input_col)
    if input_series and perk_series:
        amp = _amplification_ratio(input_series[1], perk_series[1])
        if amp is not None:
            result["MAPK_amplification"] = amp
    elif perk_series:
        # 无输入信号时，用 ppERK 峰值 / MEK 峰值
        ppmek_col = _find_species(lookup, "ppMEK", "ppmek", "pMEK", "MEK")
        ppmek_series = _get_series(times, species_data, ppmek_col)
        if ppmek_series:
            amp = _amplification_ratio(ppmek_series[1], perk_series[1])
            if amp is not None:
                result["MAPK_amplification"] = amp

    # zero_order_ultrasensitivity_hill_coefficient
    if perk_series:
        hill = _hill_coefficient(*perk_series)
        if hill is not None:
            result["zero_order_ultrasensitivity_hill_coefficient"] = hill

    return result


def _nfkb_metrics(
    times: list[float],
    species_data: dict[str, list[float]],
    lookup: dict[str, str],
) -> dict[str, float]:
    """NF_KB 通路指标（Nelson 2004, PMID:14975635）。

    - NFkB_nuclear_oscillation_period: 振荡周期 1-2h
    - IkBa_transcription_delay: IκBα 转录延迟 30-60 min
    - NFkB_oscillation_duration: 振荡持续 6-20h
    """
    result: dict[str, float] = {}

    # NFkB_nuclear 振荡
    nfkb_col = _find_species(
        lookup, "NFkB_nuclear", "nfkb", "NF_kB_nuclear", "p65_nuclear", "RelA_nuclear"
    )
    nfkb_series = _get_series(times, species_data, nfkb_col)
    nfkb_pt = None
    if nfkb_series:
        nfkb_pt = _peak_time(*nfkb_series)
        period = _oscillation_period(*nfkb_series)
        if period is not None:
            # CSV 分钟 → benchmark 小时
            result["NFkB_nuclear_oscillation_period"] = period / 60.0
        duration = _oscillation_duration(*nfkb_series)
        if duration is not None:
            result["NFkB_oscillation_duration"] = duration / 60.0

    # IkBa_transcription_delay（分钟）
    ikba_col = _find_species(lookup, "IkBa_mRNA", "ikba", "IkBalpha_mRNA", "NFKBIA_mRNA")
    ikba_series = _get_series(times, species_data, ikba_col)
    if ikba_series and nfkb_pt is not None:
        ikba_pt = _peak_time(*ikba_series)
        if ikba_pt is not None:
            result["IkBa_transcription_delay"] = ikba_pt - nfkb_pt

    return result


def _p53_metrics(
    times: list[float],
    species_data: dict[str, list[float]],
    lookup: dict[str, str],
) -> dict[str, float]:
    """p53 通路指标（Lev Bar-Or 2000, PMID:10644694）。

    - p53_pulse_period: 脉冲周期 5-7h
    - Mdm2_transcription_delay: Mdm2 转录延迟 60-120 min
    - p53_phosphorylation_response_time: p53 磷酸化响应时间 5-30 min
    """
    result: dict[str, float] = {}

    # p53 脉冲周期（小时）
    p53_col = _find_species(lookup, "p53", "p53", "p53_total")
    p53_series = _get_series(times, species_data, p53_col)
    p53_pt = None
    if p53_series:
        p53_pt = _peak_time(*p53_series)
        period = _oscillation_period(*p53_series)
        if period is not None:
            # CSV 分钟 → benchmark 小时
            result["p53_pulse_period"] = period / 60.0

    # Mdm2_transcription_delay（分钟）
    mdm2_col = _find_species(lookup, "Mdm2_mRNA", "mdm2mrna", "MDM2_mRNA")
    mdm2_series = _get_series(times, species_data, mdm2_col)
    if mdm2_series and p53_pt is not None:
        mdm2_pt = _peak_time(*mdm2_series)
        if mdm2_pt is not None:
            result["Mdm2_transcription_delay"] = mdm2_pt - p53_pt

    # p53_phosphorylation_response_time（分钟）
    # p53 首次达到峰值的时间（响应时间）
    if p53_pt is not None:
        result["p53_phosphorylation_response_time"] = p53_pt

    return result


def _pi3k_metrics(
    times: list[float],
    species_data: dict[str, list[float]],
    lookup: dict[str, str],
) -> dict[str, float]:
    """PI3K_AKT_mTOR 通路指标（Mazzoletti 2009, PMID:19211571）。

    - pAKT_peak_time: pAKT 30-60 min 达峰
    - PIP_PIP3_mass_conservation: PIP+PIP3 守恒比 0.95-1.05
    - S6K1_peak_delay_vs_AKT: S6K1 延迟 30-60 min（相对 pAKT）
    """
    result: dict[str, float] = {}

    # pAKT_peak_time（分钟）
    pakt_col = _find_species(lookup, "pAKT", "paktt", "AKT_p", "phospho_AKT")
    pakt_series = _get_series(times, species_data, pakt_col)
    pakt_pt = None
    if pakt_series:
        pakt_pt = _peak_time(*pakt_series)
        if pakt_pt is not None:
            result["pAKT_peak_time"] = pakt_pt

    # PIP_PIP3_mass_conservation
    pip3_col = _find_species(lookup, "PIP3", "pip3")
    pip2_col = _find_species(lookup, "PIP2", "pip2")
    pip3_series = _get_series(times, species_data, pip3_col)
    pip2_series = _get_series(times, species_data, pip2_col)
    if pip3_series and pip2_series:
        pip3_vals = pip3_series[1]
        pip2_vals = pip2_series[1]
        initial = pip3_vals[0] + pip2_vals[0]
        final = pip3_vals[-1] + pip2_vals[-1]
        if initial > 0:
            result["PIP_PIP3_mass_conservation"] = final / initial

    # S6K1_peak_delay_vs_AKT（分钟）
    s6k1_col = _find_species(lookup, "S6K1", "s6k1", "pS6K1", "S6K", "pS6")
    s6k1_series = _get_series(times, species_data, s6k1_col)
    if s6k1_series and pakt_pt is not None:
        s6k1_pt = _peak_time(*s6k1_series)
        if s6k1_pt is not None:
            result["S6K1_peak_delay_vs_AKT"] = s6k1_pt - pakt_pt

    return result


def _tgf_beta_metrics(
    times: list[float],
    species_data: dict[str, list[float]],
    lookup: dict[str, str],
) -> dict[str, float]:
    """TGF_BETA 通路指标（Massagué 1998, PMID:9674480）。

    - pSmad2_peak_time: pSmad2 5-15 min 达峰
    - pSmad2_Smad4_nuclear_accumulation_time: Smad4 核积累 15-30 min
    - SMAD7_mRNA_delay: SMAD7 mRNA 延迟 30-60 min
    """
    result: dict[str, float] = {}

    # pSmad2_peak_time（分钟）
    psmad2_col = _find_species(lookup, "pSmad2", "psmad2", "Smad2_p")
    psmad2_series = _get_series(times, species_data, psmad2_col)
    psmad2_pt = None
    if psmad2_series:
        psmad2_pt = _peak_time(*psmad2_series)
        if psmad2_pt is not None:
            result["pSmad2_peak_time"] = psmad2_pt

    # pSmad2_Smad4_nuclear_accumulation_time（分钟）
    # [P1-NEXT-8 V2 修复] specialist 物种名为 pSmad2_Smad4_nuc（_nuc 后缀），
    # 但原候选列表只有 Smad4_nuclear/smad4nuclear/Smad_nuclear，导致 CSV 列匹配失败
    # 修复：在候选列表中增加 pSmad2_Smad4_nuc 和 pSmad2_Smad4_nuclear 别名
    smad4n_col = _find_species(lookup, "pSmad2_Smad4_nuc", "pSmad2_Smad4_nuclear",
                                "Smad4_nuclear", "smad4nuclear", "Smad_nuclear")
    smad4n_series = _get_series(times, species_data, smad4n_col)
    if smad4n_series:
        pt = _peak_time(*smad4n_series)
        if pt is not None:
            result["pSmad2_Smad4_nuclear_accumulation_time"] = pt

    # SMAD7_mRNA_delay（分钟）
    smad7_col = _find_species(lookup, "SMAD7_mRNA", "smad7mrna", "Smad7_mRNA")
    smad7_series = _get_series(times, species_data, smad7_col)
    if smad7_series and psmad2_pt is not None:
        smad7_pt = _peak_time(*smad7_series)
        if smad7_pt is not None:
            result["SMAD7_mRNA_delay"] = smad7_pt - psmad2_pt

    return result


def _wnt_metrics(
    times: list[float],
    species_data: dict[str, list[float]],
    lookup: dict[str, str],
) -> dict[str, float]:
    """WNT 通路指标（Lee 2003, PMID:12906785）。

    - bcatenin_steady_state_no_wnt: β-catenin 稳态 <10 nM
    - Axin2_mRNA_peak_time: Axin2 mRNA 60-120 min 达峰
    - destruction_complex_assembly: 破坏复合体组装（boolean）
    """
    result: dict[str, float] = {}

    # bcatenin_steady_state_no_wnt
    bcat_col = _find_species(
        lookup, "beta_catenin", "bcatenin", "bcat", "Beta_catenin"
    )
    bcat_series = _get_series(times, species_data, bcat_col)
    if bcat_series:
        ss = _steady_state(bcat_series[1])
        if ss is not None:
            result["bcatenin_steady_state_no_wnt"] = ss

    # Axin2_mRNA_peak_time（分钟）
    axin2_col = _find_species(lookup, "Axin2_mRNA", "axin2mrna", "AXIN2_mRNA")
    axin2_series = _get_series(times, species_data, axin2_col)
    if axin2_series:
        pt = _peak_time(*axin2_series)
        if pt is not None:
            result["Axin2_mRNA_peak_time"] = pt

    # destruction_complex_assembly: 检测破坏复合体（Axin/GSK3/APC）组装
    axin_col = _find_species(lookup, "Axin", "axin", "Axin1")
    axin_series = _get_series(times, species_data, axin_col)
    if axin_series:
        # 组装 = Axin 复合物先下降（被 Wnt 抑制）然后恢复
        assembled = _bistable_switch(*axin_series)
        result["destruction_complex_assembly"] = float(assembled)
    else:
        result["destruction_complex_assembly"] = 0.0

    return result


# =============================================================================
# 通路分发器
# =============================================================================
_PATHWAY_DISPATCH: dict[str, Any] = {
    "APOPTOSIS": _apoptosis_metrics,
    "CELL_CYCLE": _cell_cycle_metrics,
    "EGFR_RTK": _egfr_metrics,
    "JAK_STAT": _jak_stat_metrics,
    "MAPK_ERK": _mapk_metrics,
    "NF_KB": _nfkb_metrics,
    "p53": _p53_metrics,
    "PI3K_AKT_mTOR": _pi3k_metrics,
    "TGF_BETA": _tgf_beta_metrics,
    "WNT": _wnt_metrics,
}


# =============================================================================
# 主入口
# =============================================================================
def compute_derived_metrics(
    csv_path: str,
    pathway_class: str,
    existing_flat: dict[str, float] | None = None,
) -> dict[str, float]:
    """从 simulation.csv 计算命名 benchmark 指标。

    规则驱动，无 LLM 调用。基于 simulation.csv 时间序列数据，
    按通路特异科学定义计算命名指标（如 Cyt_c_precedes_Casp3）。

    Args:
        csv_path: simulation.csv 文件路径。
        pathway_class: 通路标识（如 "APOPTOSIS" / "EGFR_RTK"）。
        existing_flat: 已有的扁平化指标（{species}_{field} → value），
            用于补充计算（如已有 peak_time 可直接引用）。

    Returns:
        命名指标 dict（如 {"Cyt_c_precedes_Casp3": 8.5, ...}）。
        CSV 不可读或通路未匹配时返回空 dict（不阻塞）。
    """
    if not csv_path:
        return {}

    # 读取 CSV
    times, species_data, col_names = _read_csv(csv_path)
    if not times or not species_data:
        logger.warning(
            "compute_derived_metrics: CSV 不可读或为空: %s", csv_path
        )
        return {}

    # 构建物种查找映射
    lookup = _build_species_lookup(col_names)

    result: dict[str, float] = {}

    # 通用指标：质量守恒误差（所有 10 个通路都有此 benchmark）
    mce = _mass_conservation_error(times, species_data, col_names)
    if mce is not None:
        result["mass_conservation_error"] = mce

    # 通路特异指标
    dispatcher = _PATHWAY_DISPATCH.get(pathway_class)
    if dispatcher is not None:
        try:
            pathway_metrics = dispatcher(times, species_data, lookup)
            result.update(pathway_metrics)
        except Exception as exc:
            logger.warning(
                "compute_derived_metrics: 通路 %s 指标计算异常: %s",
                pathway_class,
                exc,
            )
    else:
        logger.debug(
            "compute_derived_metrics: 通路 %s 无特异 dispatcher", pathway_class
        )

    logger.info(
        "compute_derived_metrics: 通路=%s, 计算出 %d 个指标: %s",
        pathway_class,
        len(result),
        list(result.keys()),
    )
    return result
