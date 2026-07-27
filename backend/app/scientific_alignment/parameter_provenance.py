# BioDynamics Agent v4 - Scientific Alignment Sprint 5: Parameter Provenance
#
# 参数来源溯源：为每个动力学参数标注 source + confidence + origin，
# 生成 Parameter Traceability 表（Markdown），供 Report 渲染。
#
# 设计原则：
#   - 100% Rule-driven，无 LLM 调用
#   - 复用 state["parameters"] 已有的溯源四元组（value/source/confidence/origin）
#   - Feature Flag 守护：SA_SPRINT5_PROVENANCE_EXPLAINABILITY
#
# 依赖：app.config.settings；不引入新依赖。
#
# 核心导出：
#   from app.scientific_alignment.parameter_provenance import (
#       ParameterTraceabilityRow, ParameterProvenanceReport,
#       build_parameter_traceability, render_parameter_traceability_table,
#       compute_provenance_summary,
#   )

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)


# BioModels ID 正则（BIOMD0000000xxx / MODELxxxx）
_BIOMD_RE = re.compile(r"\b(BIOMD\d{10}|MODEL\d{10})\b", re.IGNORECASE)


def _extract_biomd_id_from_param(param_dict: dict[str, Any]) -> str | None:
    """从参数 dict 提取 BioModels ID。

    优先级：
      1. 显式 ``biomd_id`` 字段
      2. ``source_model`` 字段（SBML grounding 写入）
      3. ``origin`` 字符串中的 BIOMD/MODEL 模式
      4. ``source`` 字符串中的 BIOMD/MODEL 模式

    无法提取时返回 None（字段必须存在，值可为 None）。
    """
    if not isinstance(param_dict, dict):
        return None
    explicit = param_dict.get("biomd_id")
    if explicit and str(explicit).strip():
        return str(explicit).strip()
    source_model = param_dict.get("source_model")
    if source_model and str(source_model).strip():
        sm = str(source_model).strip()
        match = _BIOMD_RE.search(sm)
        if match:
            return match.group(1).upper()
        # source_model 本身可能就是 BIOMD ID
        if sm.upper().startswith(("BIOMD", "MODEL")):
            return sm.upper()
        return sm
    for field_name in ("origin", "source"):
        value = param_dict.get(field_name)
        if not value:
            continue
        match = _BIOMD_RE.search(str(value))
        if match:
            return match.group(1).upper()
    return None


# =============================================================================
# 常量
# =============================================================================
# 置信度阈值（与 nodes_v2.py _confidence_str_to_float 对齐）
_CONFIDENCE_HIGH_THRESHOLD: float = 0.85
_CONFIDENCE_MEDIUM_THRESHOLD: float = 0.5

# 来源优先级（用于排序 Traceability 表）
_SOURCE_PRIORITY: dict[str, int] = {
    "RAG": 0,
    "SBML": 1,
    "PubMed": 2,
    "KEGG": 3,
    "UniProt": 4,
    "ChEMBL": 5,
    "Inferred": 6,
}


# =============================================================================
# 数据类
# =============================================================================
@dataclass
class ParameterTraceabilityRow:
    """单条参数溯源行。

    Attributes:
        edge_key: 反应边标识（如 ``"EGF->EGFR"``）。
        param_name: 参数名（如 ``"kd"`` / ``"k_on"``）。
        value: 参数值（None 表示缺失）。
        unit: 单位（如 ``"nM"``）。
        source: 来源类型（RAG/SBML/PubMed/KEGG/UniProt/ChEMBL/Inferred）。
        origin: 具体来源标识（如 ``"PMID:12451180"`` / ``"BIOMD0000000010"``）。
        biomd_id: BioModels 模型 ID（如 ``"BIOMD0000000048"``）。强制字段，
            无法提取时为 None。用于跨模型混用检测与同源优先检索。
        confidence: 置信度（0.0-1.0）。
        confidence_label: 置信度标签（HIGH/MEDIUM/LOW）。
        is_fallback: 是否为兜底估计。
        missing: 是否为缺失参数。
    """

    edge_key: str = ""
    param_name: str = ""
    value: Any = None
    unit: str = ""
    source: str = ""
    origin: str = ""
    biomd_id: str | None = None
    confidence: float = 0.0
    confidence_label: str = ""
    is_fallback: bool = False
    missing: bool = False


@dataclass
class ParameterProvenanceReport:
    """参数溯源报告。

    Attributes:
        enabled: Feature Flag 是否启用。
        skipped: 是否跳过（Flag 关闭时为 True）。
        rows: Traceability 行列表。
        summary: 汇总统计。
        markdown_table: 渲染后的 Markdown 表格（Flag OFF 时为空）。
    """

    enabled: bool = False
    skipped: bool = False
    rows: list[ParameterTraceabilityRow] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    markdown_table: str = ""


# =============================================================================
# 辅助函数
# =============================================================================
def _confidence_to_label(confidence: float) -> str:
    """将数值置信度转为标签。

    Args:
        confidence: 置信度（0.0-1.0）。

    Returns:
        ``"HIGH"`` / ``"MEDIUM"`` / ``"LOW"``。
    """
    if confidence >= _CONFIDENCE_HIGH_THRESHOLD:
        return "HIGH"
    if confidence >= _CONFIDENCE_MEDIUM_THRESHOLD:
        return "MEDIUM"
    return "LOW"


def _format_value(value: Any) -> str:
    """格式化参数值为字符串。"""
    if value is None:
        return "N/A"
    if isinstance(value, float):
        # 保留 4 位有效数字
        if abs(value) < 0.001 or abs(value) >= 10000:
            return f"{value:.4e}"
        return f"{value:.4g}"
    return str(value)


# =============================================================================
# 主函数
# =============================================================================
def build_parameter_traceability(
    parameters: dict[str, dict] | None,
) -> list[ParameterTraceabilityRow]:
    """从 state["parameters"] 构建参数溯源行列表。

    参数字典结构（由 N5 n5_parameter_rag 生成）：
      ``{edge_key: {edge_key, param_name, value, unit, source, confidence,
                     confidence_label, origin, is_fallback, missing_parameter}}``

    Args:
        parameters: state["parameters"] 字典。

    Returns:
        ParameterTraceabilityRow 列表，按来源优先级排序。
    """
    if not parameters or not isinstance(parameters, dict):
        return []

    rows: list[ParameterTraceabilityRow] = []
    for edge_key, param_dict in parameters.items():
        if not isinstance(param_dict, dict):
            continue
        confidence = float(param_dict.get("confidence", 0.0) or 0.0)
        row = ParameterTraceabilityRow(
            edge_key=str(edge_key),
            param_name=str(param_dict.get("param_name", "")),
            value=param_dict.get("value"),
            unit=str(param_dict.get("unit", "")),
            source=str(param_dict.get("source", "Inferred")),
            origin=str(param_dict.get("origin", "")),
            biomd_id=_extract_biomd_id_from_param(param_dict),
            confidence=confidence,
            confidence_label=_confidence_to_label(confidence),
            is_fallback=bool(param_dict.get("is_fallback", False)),
            missing=bool(param_dict.get("missing_parameter", False)),
        )
        rows.append(row)

    # 按来源优先级排序
    rows.sort(
        key=lambda r: (
            _SOURCE_PRIORITY.get(r.source, 99),
            r.edge_key,
            r.param_name,
        )
    )
    return rows


def compute_provenance_summary(
    rows: list[ParameterTraceabilityRow],
) -> dict[str, Any]:
    """计算参数溯源汇总统计。

    Args:
        rows: Traceability 行列表。

    Returns:
        汇总字典，含 total / by_source / by_confidence / missing_count /
        fallback_count。
    """
    if not rows:
        return {
            "total": 0,
            "by_source": {},
            "by_confidence": {"HIGH": 0, "MEDIUM": 0, "LOW": 0},
            "missing_count": 0,
            "fallback_count": 0,
            "provenance_score": 0.0,
            "biomd_id_distribution": {},
            "cross_model_mixing": False,
        }

    by_source: dict[str, int] = {}
    by_confidence = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    missing_count = 0
    fallback_count = 0
    biomd_ids: dict[str, int] = {}

    for row in rows:
        by_source[row.source] = by_source.get(row.source, 0) + 1
        by_confidence[row.confidence_label] = (
            by_confidence.get(row.confidence_label, 0) + 1
        )
        if row.missing:
            missing_count += 1
        if row.is_fallback:
            fallback_count += 1
        if row.biomd_id:
            biomd_ids[row.biomd_id] = biomd_ids.get(row.biomd_id, 0) + 1

    # Provenance Score = 非缺失非兜底参数占比
    proven_count = sum(1 for r in rows if not r.missing and not r.is_fallback)
    provenance_score = proven_count / len(rows) if rows else 0.0

    # 跨模型混用检测：>1 个不同 biomd_id 表示混用
    cross_model_mixing = len(biomd_ids) > 1

    return {
        "total": len(rows),
        "by_source": by_source,
        "by_confidence": by_confidence,
        "missing_count": missing_count,
        "fallback_count": fallback_count,
        "provenance_score": round(provenance_score, 3),
        "biomd_id_distribution": biomd_ids,
        "cross_model_mixing": cross_model_mixing,
    }


def render_parameter_traceability_table(
    rows: list[ParameterTraceabilityRow],
) -> str:
    """渲染 Parameter Traceability Markdown 表。

    表格列：Edge | Parameter | Value | Unit | Source | Origin | Confidence | Status

    Args:
        rows: Traceability 行列表。

    Returns:
        Markdown 表格字符串。rows 为空时返回提示文本。
    """
    if not rows:
        return "> No parameters available for traceability.\n"

    header = (
        "| Edge | Parameter | Value | Unit | Source | Origin | BioModels ID | Confidence | Status |\n"
        "|------|-----------|-------|------|--------|--------|--------------|------------|--------|\n"
    )
    lines = []
    for row in rows:
        # Status 列：missing / fallback / OK
        if row.missing:
            status = "MISSING"
        elif row.is_fallback:
            status = "FALLBACK"
        else:
            status = "OK"
        biomd_display = row.biomd_id if row.biomd_id else "—"
        lines.append(
            f"| {row.edge_key} | {row.param_name} | {_format_value(row.value)} "
            f"| {row.unit} | {row.source} | {row.origin} | {biomd_display} "
            f"| {row.confidence_label} ({row.confidence:.2f}) | {status} |"
        )

    return header + "\n".join(lines) + "\n"


def generate_provenance_report(
    parameters: dict[str, dict] | None,
) -> ParameterProvenanceReport:
    """生成完整参数溯源报告（Feature Flag 守护）。

    Args:
        parameters: state["parameters"] 字典。

    Returns:
        ParameterProvenanceReport。Flag 关闭时返回 skipped 报告。
    """
    if not settings.is_sa_feature_enabled("SPRINT5_PROVENANCE_EXPLAINABILITY"):
        return ParameterProvenanceReport(
            enabled=False,
            skipped=True,
        )

    rows = build_parameter_traceability(parameters)
    summary = compute_provenance_summary(rows)
    markdown = render_parameter_traceability_table(rows)

    logger.debug(
        "[Sprint5] Parameter Provenance: total=%d missing=%d fallback=%d score=%.3f",
        summary["total"],
        summary["missing_count"],
        summary["fallback_count"],
        summary["provenance_score"],
    )

    return ParameterProvenanceReport(
        enabled=True,
        skipped=False,
        rows=rows,
        summary=summary,
        markdown_table=markdown,
    )
