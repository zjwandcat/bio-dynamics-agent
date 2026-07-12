# BioDynamics Agent v4 - Scientific Alignment Loop: 报告段渲染器（Task 6）
#
# 本模块是 report_renderer.py 的配套文件，负责渲染 Scientific Alignment 报告中
# 与 SA 子能力强绑定的四个 markdown 段：
#   1. BioModels 差异对照表（SubTask 6.1，Flag = SA_BIOMODELS_ORACLE）
#   2. Discussion 引导对照文本（SubTask 6.2，Flag = SA_BIOMODELS_ORACLE）
#   3. 6 维 Confidence 表（Task 25 SubTask 25.5，Flag = SA_MULTI_DIM_CONFIDENCE）
#   4. 参数 provenance 表（Task 23 SubTask 23.4，Flag = SA_PARAMETER_CONFIDENCE）
#
# 设计原则：
#   - 各段独立受 Feature Flag 守护，Flag OFF 时对应段返回 ""（不渲染）
#   - 输入为 None / 无数据时返回友好提示（如 "BioModels comparison not available."）
#   - markdown 表格列对齐，数值保留合理精度（%g 去多余零）
#   - 兼容 dataclass（BioModelsOracleReport / ParameterPrior）与 dict 两种输入
#     （便于测试与上游 dict 序列化场景）
#   - 不修改 report_renderer.py / __init__.py / config.py
#
# 依赖：
#   - app.config.settings（Feature Flag 守护）
#   - app.scientific_alignment.multi_dim_confidence.format_confidence_table
#     （6 维表渲染委托，避免重复实现）
#
# 核心导出：
#   from app.scientific_alignment.report_sections import (
#       ReportSections,
#       render_biomodels_comparison,
#       render_biomodels_discussion,
#       render_confidence_table,
#       render_parameter_table,
#       render_all_sections,
#   )

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

from app.config import settings
from app.scientific_alignment.multi_dim_confidence import format_confidence_table

logger = logging.getLogger(__name__)


# =============================================================================
# 常量
# =============================================================================

# 差异超此相对阈值（50%）标记 Within Range? = No
_OUT_OF_RANGE_REL_THRESHOLD: float = 0.5

# 数值格式化精度（%g 有效数字位数）
_NUM_PRECISION: int = 4


# =============================================================================
# 数据类
# =============================================================================
@dataclass
class ReportSections:
    """SA 报告段集合（一次性渲染所有段的结果容器）。

    Attributes:
        biomodels_table: BioModels 差异对照表 markdown（含标题）。
        biomodels_discussion: Discussion 引导对照文本。
        confidence_table: 6 维 Confidence markdown 表。
        parameter_table: 参数 provenance markdown 表。
        enabled: 任一 SA 子 Flag 开启则为 True（全 OFF 时 False）。
    """

    biomodels_table: str = ""
    biomodels_discussion: str = ""
    confidence_table: str = ""
    parameter_table: str = ""
    enabled: bool = False


# =============================================================================
# 辅助函数
# =============================================================================
def _get(obj: Any, key: str, default: Any = None) -> Any:
    """从 dataclass 或 dict 中取字段（兼容两种输入格式）。

    Args:
        obj: 数据对象（dataclass 实例或 dict）。
        key: 字段名。
        default: 字段缺失时的默认值。

    Returns:
        字段值；obj 为 None 或字段不存在时返回 default。
    """
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _fmt_num(value: Any, precision: int = _NUM_PRECISION) -> str:
    """格式化数值，去除多余零（%g 风格）。

    Args:
        value: 数值（int/float/str 均可，内部转 float）。
        precision: 有效数字位数。

    Returns:
        格式化后的字符串；None → "-"，NaN → "NaN"，非数值 → str(value)。
    """
    if value is None:
        return "-"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(v):
        return "NaN"
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    return f"{v:.{precision}g}"


def _fmt_diff(diff: Any, unit: str = "", precision: int = _NUM_PRECISION) -> str:
    """格式化差值（带正负号），追加单位。

    Args:
        diff: 差值（agent - biomodels）。
        unit: 单位字符串（如 "min"）。
        precision: 有效数字位数。

    Returns:
        形如 "+1.3 min" / "-0.07" / "0" 的字符串；非数值返回 "-"。
    """
    if diff is None:
        return "-"
    try:
        d = float(diff)
    except (TypeError, ValueError):
        return "-"
    if math.isnan(d):
        return "NaN"
    if math.isinf(d):
        return "inf" if d > 0 else "-inf"
    # %g 去多余零，正数前加 "+"
    sign = "+" if d >= 0 else "-"
    s = f"{sign}{abs(d):.{precision}g}"
    if unit:
        s += f" {unit}"
    return s


def _fmt_value_with_unit(value: Any, unit: str = "", precision: int = _NUM_PRECISION) -> str:
    """格式化数值并追加单位（无正负号）。"""
    s = _fmt_num(value, precision)
    if unit and s not in ("-", "NaN", "inf", "-inf"):
        s += f" {unit}"
    return s


def _compute_within_range(
    agent_val: Any,
    biomodels_val: Any,
    within_explicit: Any = None,
) -> bool:
    """计算指标是否在范围内（相对差异 < 50%）。

    优先使用显式 within_range 字段；缺失时按相对差异计算。

    Args:
        agent_val: Agent 仿真值。
        biomodels_val: BioModels 参考值。
        within_explicit: 显式 within_range（bool / None）。

    Returns:
        True 表示在范围内；无法计算时保守返回 True。
    """
    if within_explicit is not None:
        return bool(within_explicit)
    if agent_val is None or biomodels_val is None:
        return True
    try:
        a = float(agent_val)
        b = float(biomodels_val)
    except (TypeError, ValueError):
        return True
    if math.isnan(a) or math.isnan(b) or math.isinf(a) or math.isinf(b):
        return True
    denom = max(abs(b), 1e-9)
    rel_diff = abs(a - b) / denom
    return rel_diff < _OUT_OF_RANGE_REL_THRESHOLD


def _render_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    """生成列对齐的 markdown 表格。

    Args:
        headers: 列标题列表。
        rows: 数据行列表（每行为单元格字符串列表）。

    Returns:
        对齐的 markdown 表格字符串；headers 为空时返回 ""。
    """
    if not headers:
        return ""
    n_cols = len(headers)
    # 计算每列最大宽度（表头 + 所有数据行）
    all_lines = [headers] + [r[:n_cols] + [""] * (n_cols - len(r)) for r in rows]
    col_widths = [
        max(len(str(line[i])) for line in all_lines)
        for i in range(n_cols)
    ]
    # 表头行
    header_line = "| " + " | ".join(
        str(headers[i]).ljust(col_widths[i]) for i in range(n_cols)
    ) + " |"
    # 分隔行（dash 宽度 = 列宽 + 2，保证视觉对齐）
    sep_line = "|" + "|".join(
        "-" * (col_widths[i] + 2) for i in range(n_cols)
    ) + "|"
    # 数据行
    data_lines = []
    for row in rows:
        cells = [
            str(row[i]).ljust(col_widths[i]) if i < len(row) else "".ljust(col_widths[i])
            for i in range(n_cols)
        ]
        data_lines.append("| " + " | ".join(cells) + " |")
    return header_line + "\n" + sep_line + "\n" + "\n".join(data_lines)


# =============================================================================
# BioModels 对比数据归一化
# =============================================================================
def _normalize_comparisons(comparisons: list) -> list[tuple]:
    """将两种格式的对比数据统一为五元组列表。

    兼容格式：
      1. dict mock：{metric, agent_value, biomodels_value, unit, within_range}
      2. 真实 species_comparisons：{species, template_peak_time, sbml_peak_time,
         template_peak_value, sbml_peak_value, ...}（每个物种生成 Peak Time +
         Peak Amplitude 两行）

    Returns:
        [(metric, agent_value, biomodels_value, unit, within_range_explicit), ...]
        within_range_explicit 为 None 表示需按相对差异计算。
    """
    result: list[tuple] = []
    for cmp in comparisons or []:
        metric = _get(cmp, "metric", "")
        if metric:
            # 格式1：dict mock，直接取字段
            agent_val = _get(cmp, "agent_value", None)
            bio_val = _get(cmp, "biomodels_value", None)
            unit = _get(cmp, "unit", "") or ""
            within = _get(cmp, "within_range", None)
            result.append((str(metric), agent_val, bio_val, unit, within))
            continue
        # 格式2：真实 species_comparisons，按物种展开为 Peak Time + Peak Amplitude
        species = _get(cmp, "species", "") or _get(cmp, "sbml_species_id", "")
        if not species:
            continue
        # Peak Time 行
        tpl_pt = _get(cmp, "template_peak_time", None)
        sbml_pt = _get(cmp, "sbml_peak_time", None)
        if tpl_pt is not None and sbml_pt is not None:
            result.append((f"{species} Peak Time", tpl_pt, sbml_pt, "min", None))
        # Peak Amplitude 行
        tpl_pv = _get(cmp, "template_peak_value", None)
        sbml_pv = _get(cmp, "sbml_peak_value", None)
        if tpl_pv is not None and sbml_pv is not None:
            result.append((f"{species} Peak Amplitude", tpl_pv, sbml_pv, "", None))
    return result


def _extract_biomodel_id(biomodels_report: Any) -> str:
    """从报告中提取 BioModels ID（兼容 biomodel_id / model_id 两种字段名）。"""
    return str(
        _get(biomodels_report, "biomodel_id", "")
        or _get(biomodels_report, "model_id", "")
        or ""
    )


def _extract_comparisons(biomodels_report: Any) -> list:
    """从报告中提取对比列表（兼容 species_comparisons / comparisons 两种字段名）。"""
    return (
        _get(biomodels_report, "species_comparisons", None)
        or _get(biomodels_report, "comparisons", None)
        or []
    )


# =============================================================================
# SubTask 6.1：渲染 BioModels 差异对照表
# =============================================================================
def render_biomodels_comparison(
    biomodels_report: Any,
    pathway: str = "",
) -> str:
    """渲染 BioModels 差异对照表。Flag OFF 返回 ''。

    输入为 None 或无数据时返回 'BioModels comparison not available.'

    表格式：
        | Metric | Agent Simulation | BioModels (BIOMDxxxxxxxx) | Difference | Within Range? |

    差异超 50%（相对差异）标记 Within Range? = No，并在表后追加警告。

    Args:
        biomodels_report: BioModelsOracleReport 实例或 dict（含 model_id / comparisons）。
        pathway: 通路标识（如 "EGFR"），仅用于日志。

    Returns:
        markdown 表格字符串（含 "### BioModels Comparison" 标题）；
        Flag OFF 返回 ""；输入无数据返回友好提示。
    """
    # Feature Flag 守护
    if not settings.is_sa_feature_enabled("BIOMODELS_ORACLE"):
        logger.debug(
            "biomodels comparison table skipped: SA_BIOMODELS_ORACLE disabled "
            "(pathway=%s)", pathway,
        )
        return ""

    # 输入为 None 或无数据 → 友好提示
    if biomodels_report is None:
        return "BioModels comparison not available."

    model_id = _extract_biomodel_id(biomodels_report)
    raw_comparisons = _extract_comparisons(biomodels_report)

    if not model_id and not raw_comparisons:
        return "BioModels comparison not available."

    # 归一化为五元组列表
    normalized = _normalize_comparisons(raw_comparisons)
    if not normalized:
        # 有 model_id 但无对比数据 → 提示
        return (
            f"BioModels comparison not available (model_id={model_id}, "
            f"no comparison data)."
        )

    # 构建表格行
    headers = [
        "Metric",
        "Agent Simulation",
        f"BioModels ({model_id})",
        "Difference",
        "Within Range?",
    ]
    rows: list[list[str]] = []
    out_of_range_metrics: list[str] = []

    for metric, agent_val, bio_val, unit, within_explicit in normalized:
        within = _compute_within_range(agent_val, bio_val, within_explicit)
        agent_str = _fmt_value_with_unit(agent_val, unit)
        bio_str = _fmt_value_with_unit(bio_val, unit)
        # Difference = agent - biomodels（带正负号）
        diff_val = None
        if agent_val is not None and bio_val is not None:
            try:
                diff_val = float(agent_val) - float(bio_val)
            except (TypeError, ValueError):
                diff_val = None
        diff_str = _fmt_diff(diff_val, unit)
        within_str = "Yes" if within else "No"
        rows.append([metric, agent_str, bio_str, diff_str, within_str])
        if not within:
            out_of_range_metrics.append(metric)

    table = _render_markdown_table(headers, rows)

    # 差异超 50% 追加警告
    warning = ""
    if out_of_range_metrics:
        warning = (
            "\n\n**Warning**: The following metrics exceed 50% difference from "
            f"the BioModels reference ({model_id}): "
            f"{', '.join(out_of_range_metrics)}. Review parameter calibration "
            f"and mechanism assumptions."
        )

    return f"### BioModels Comparison\n\n{table}{warning}"


# =============================================================================
# SubTask 6.2：Discussion 引导对照文本
# =============================================================================
def render_biomodels_discussion(
    biomodels_report: Any,
    pathway: str = "",
) -> str:
    """渲染 Discussion 引导对照文本。Flag OFF 返回 ''。

    生成形如：
        "**Discussion**: Compared with BIOMDxxxxxxxx, the agent simulation shows
        ERK peak within 1.3 min of the published model, within expected biological
        variability. ..."

    所有指标均在范围内 → "within expected biological variability"；
    存在超范围指标 → 提示 divergence 并建议复核。

    Args:
        biomodels_report: BioModelsOracleReport 实例或 dict。
        pathway: 通路标识（仅用于日志）。

    Returns:
        Discussion 引导文本；Flag OFF 或输入无数据返回 ""。
    """
    if not settings.is_sa_feature_enabled("BIOMODELS_ORACLE"):
        logger.debug(
            "biomodels discussion skipped: SA_BIOMODELS_ORACLE disabled "
            "(pathway=%s)", pathway,
        )
        return ""

    if biomodels_report is None:
        return ""

    model_id = _extract_biomodel_id(biomodels_report)
    if not model_id:
        return ""

    normalized = _normalize_comparisons(_extract_comparisons(biomodels_report))

    if not normalized:
        return (
            f"**Discussion**: Compared with {model_id}, the agent simulation "
            f"results are within expected biological variability."
        )

    # 统计 within range 并取首个指标作为代表
    all_within = True
    rep_metric = ""
    rep_diff = None
    rep_unit = ""
    for metric, agent_val, bio_val, unit, within_explicit in normalized:
        within = _compute_within_range(agent_val, bio_val, within_explicit)
        if not within:
            all_within = False
        if not rep_metric:
            rep_metric = metric
            rep_unit = unit
            if agent_val is not None and bio_val is not None:
                try:
                    rep_diff = float(agent_val) - float(bio_val)
                except (TypeError, ValueError):
                    rep_diff = None

    if rep_metric and rep_diff is not None:
        diff_str = _fmt_diff(rep_diff, rep_unit)
        if all_within:
            return (
                f"**Discussion**: Compared with {model_id}, the agent simulation "
                f"shows {rep_metric} within {diff_str} of the published model, "
                f"within expected biological variability. Minor differences in "
                f"amplitude may reflect parameter estimation vs. calibrated values."
            )
        return (
            f"**Discussion**: Compared with {model_id}, the agent simulation "
            f"shows notable divergence in some metrics (e.g., {rep_metric} "
            f"{diff_str}). Review parameter calibration and mechanism assumptions."
        )

    # 无法计算代表差值时给出通用结论
    if all_within:
        return (
            f"**Discussion**: Compared with {model_id}, the agent simulation "
            f"is within expected biological variability."
        )
    return (
        f"**Discussion**: Compared with {model_id}, some metrics diverge beyond "
        f"expected biological variability. Review parameter calibration."
    )


# =============================================================================
# Task 25 SubTask 25.5：6 维 Confidence 表渲染
# =============================================================================
def render_confidence_table(multi_dim_report: Any) -> str:
    """渲染 6 维 Confidence 表。Flag OFF 返回 ''。

    直接委托给 multi_dim_confidence.format_confidence_table，避免重复实现。
    Flag OFF 时返回 ""（不渲染 skipped 提示表）。

    Args:
        multi_dim_report: MultiDimConfidenceReport 实例。

    Returns:
        6 维 Confidence markdown 表；Flag OFF 返回 ""；
        输入为 None 返回 "Multi-dimensional confidence not available."。
    """
    if not settings.is_sa_feature_enabled("MULTI_DIM_CONFIDENCE"):
        logger.debug("confidence table skipped: SA_MULTI_DIM_CONFIDENCE disabled")
        return ""

    if multi_dim_report is None:
        return "Multi-dimensional confidence not available."

    # 委托给 multi_dim_confidence.format_confidence_table
    # 该函数已处理 skipped 报告（Flag ON 但 report.skipped=True 的边界场景）
    return format_confidence_table(multi_dim_report)


# =============================================================================
# Task 23 SubTask 23.4：参数 provenance 表渲染
# =============================================================================
def _normalize_priors(priors: list) -> list[dict]:
    """将 ParameterPrior 列表或 dict 列表统一为字段字典列表。

    兼容字段名：
      - param_name / name（参数名）
      - value / confidence / source / distribution / reference / provenance_complete
    """
    result: list[dict] = []
    for p in priors or []:
        result.append({
            "name": _get(p, "param_name", "") or _get(p, "name", "") or "",
            "value": _get(p, "value", None),
            "confidence": _get(p, "confidence", "") or "",
            "source": _get(p, "source", "") or "",
            "distribution": _get(p, "distribution", "") or "",
            "reference": _get(p, "reference", "") or "",
            "provenance_complete": _get(p, "provenance_complete", False),
            "unit": _get(p, "unit", "") or "",
        })
    return result


def _has_provenance_info(priors_norm: list[dict]) -> bool:
    """判断 priors 是否携带 provenance 信息（source/distribution/reference 任一非空）。"""
    for p in priors_norm:
        if p["source"] or p["distribution"] or p["reference"] or p["confidence"]:
            return True
    return False


def render_parameter_table(parameter_report: Any) -> str:
    """渲染参数 provenance 表。Flag OFF 返回 ''。

    表格式（provenance 完整时）：
        | Parameter | Value | Confidence | Source | Distribution | Reference |

    ParameterPrior 无 provenance 字段时（全部 source/distribution/reference 为空），
    降级为仅 Parameter | Value 两列。

    Args:
        parameter_report: ParameterPriorReport 实例或 dict（含 priors 列表）。

    Returns:
        markdown 表格字符串（含 "### Parameter Provenance" 标题）；
        Flag OFF 返回 ""；输入无数据返回友好提示。
    """
    if not settings.is_sa_feature_enabled("PARAMETER_CONFIDENCE"):
        logger.debug("parameter table skipped: SA_PARAMETER_CONFIDENCE disabled")
        return ""

    if parameter_report is None:
        return "Parameter provenance not available."

    # 兼容 ParameterPriorReport.priors 与 dict["priors"]
    priors = _get(parameter_report, "priors", None)
    if priors is None:
        return "Parameter provenance not available."

    priors_norm = _normalize_priors(priors)
    if not priors_norm:
        return "Parameter provenance not available (no parameters)."

    # 判断是否降级为仅 Value 列（无任何 provenance 信息时）
    has_provenance = _has_provenance_info(priors_norm)

    if has_provenance:
        # 完整 6 列表
        headers = [
            "Parameter",
            "Value",
            "Confidence",
            "Source",
            "Distribution",
            "Reference",
        ]
        rows: list[list[str]] = []
        for p in priors_norm:
            value_str = _fmt_value_with_unit(p["value"], p["unit"])
            rows.append([
                p["name"] or "-",
                value_str,
                p["confidence"] or "-",
                p["source"] or "-",
                p["distribution"] or "-",
                p["reference"] or "-",
            ])
    else:
        # 降级：仅 Parameter | Value 两列
        headers = ["Parameter", "Value"]
        rows = []
        for p in priors_norm:
            value_str = _fmt_value_with_unit(p["value"], p["unit"])
            rows.append([p["name"] or "-", value_str])

    table = _render_markdown_table(headers, rows)
    return f"### Parameter Provenance\n\n{table}"


# =============================================================================
# 一次性渲染所有 SA 报告段
# =============================================================================
def render_all_sections(
    biomodels_report: Any = None,
    multi_dim_report: Any = None,
    parameter_report: Any = None,
    pathway: str = "",
) -> ReportSections:
    """一次性渲染所有 SA 报告段。

    各段独立受 Feature Flag 守护：Flag OFF 时对应段返回 ""。
    enabled 字段：任一 SA 子 Flag 开启则为 True。

    Args:
        biomodels_report: BioModelsOracleReport 实例或 dict（可为 None）。
        multi_dim_report: MultiDimConfidenceReport 实例（可为 None）。
        parameter_report: ParameterPriorReport 实例或 dict（可为 None）。
        pathway: 通路标识（如 "EGFR"）。

    Returns:
        ReportSections，含四个 markdown 段与 enabled 标记。
    """
    biomodels_table = render_biomodels_comparison(biomodels_report, pathway)
    biomodels_discussion = render_biomodels_discussion(biomodels_report, pathway)
    confidence_table = render_confidence_table(multi_dim_report)
    parameter_table = render_parameter_table(parameter_report)

    # enabled：任一 SA 子 Flag 开启则 True
    enabled = (
        settings.is_sa_feature_enabled("BIOMODELS_ORACLE")
        or settings.is_sa_feature_enabled("MULTI_DIM_CONFIDENCE")
        or settings.is_sa_feature_enabled("PARAMETER_CONFIDENCE")
    )

    return ReportSections(
        biomodels_table=biomodels_table,
        biomodels_discussion=biomodels_discussion,
        confidence_table=confidence_table,
        parameter_table=parameter_table,
        enabled=enabled,
    )


__all__ = [
    "ReportSections",
    "render_biomodels_comparison",
    "render_biomodels_discussion",
    "render_confidence_table",
    "render_parameter_table",
    "render_all_sections",
]
