# BioDynamics Agent v4 - Scientific Alignment Loop: Acceptance Criteria Gate (Task 15)
#
# Acceptance Criteria Gate：每通路强制校验 Acceptance Criteria，任一 high severity
# criterion 未满足判整体 Fail。CI 闸门调用本模块决定是否拦截（通过 exit code）。
#
# 设计目标（SubTask 15.1 / 15.2）：
#   - 从 Canonical 动态加载 expected_behavior / required_nodes / known_negative_feedback
#   - 12 类强制校验项（ERK Peak 范围 / EGFR Peak 早 / DUSP 反馈 / Nuclear ERK /
#     BioModels 对比 / PubMed 数 / Review 数 / 机制覆盖 / 实验链 / 讨论 10 问 /
#     Confidence / Validation Pyramid）
#   - 任一 high severity criterion Fail → 整体 Fail（CI 闸门拦截）
#   - medium severity criterion Fail → 仅记入 warnings，不影响整体 passed
#
# Feature Flag 守护：
#   SA_LOOP_TERMINATION 默认 OFF。关闭时返回 skipped=True 的空报告，不阻塞主流程。
#   Acceptance Gate 是 Loop Termination 的子组件。
#   铁律：V4_SCIENTIFIC_ALIGNMENT_ENABLED=false 时，子 Flag 永远不生效
#         （由 settings.is_sa_feature_enabled 强制校验）。
#
# 优雅降级：
#   - Canonical 加载失败（FileNotFoundError）时，降级为只校验非 Canonical 依赖项，
#     warnings 追加 "canonical_missing"
#   - simulation_metrics 缺失时相应 criterion 标 actual="missing"
#   - seven_axis_report / consistency_report / multi_dim_report 为 None 时
#     对应 criterion 标 actual="missing"
#
# 依赖：
#   - app.config.settings（Feature Flag 守护）
#   - app.scientific_alignment.canonical_loader.load_canonical
#   - app.scientific_alignment.mechanism_checker.normalize_node_name
#
# 核心导出：
#   from app.scientific_alignment.acceptance_gate import (
#       CriterionResult, AcceptanceReport, check_acceptance,
#   )

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

from app.config import settings
from app.scientific_alignment.canonical_loader import load_canonical
from app.scientific_alignment.mechanism_checker import normalize_node_name

logger = logging.getLogger(__name__)


# =============================================================================
# 常量
# =============================================================================

# 强制校验阈值
_MIN_PUBMED_COUNT: int = 5
_MIN_REVIEW_COUNT: int = 2
_MIN_EXPERIMENT_COUNT: int = 2
_MIN_DISCUSSION_COVERAGE: float = 0.8
_MIN_CONFIDENCE: float = 0.9
_MIN_MECHANISM_COVERAGE: float = 0.8


# =============================================================================
# 数据类
# =============================================================================
@dataclass
class CriterionResult:
    """单个 Acceptance Criterion 的校验结果。

    Attributes:
        name: criterion 标识（如 ``"erk_peak_range"`` / ``"egfr_peak_early"``）。
        passed: 是否通过。
        expected: 期望值描述（如 ``"ERK Peak 在 10-20 min"``）。
        actual: 实际值（如 ``"15.0"`` 或 ``"missing"``）。
        severity: 严重级别：
            - ``"high"``（默认）：fail 则整体 fail
            - ``"medium"``：fail 仅记入 warnings，不影响整体 passed
    """

    name: str
    passed: bool
    expected: str
    actual: str
    severity: str = "high"


@dataclass
class AcceptanceReport:
    """Acceptance Criteria 校验报告。

    Attributes:
        enabled: Feature Flag 是否开启。
        skipped: 是否跳过（Feature Flag 关闭时为 True）。
        pathway: 被校验的通路标识。
        criteria: 所有 criterion 的结果列表（CriterionResult）。
        passed: 整体是否通过（所有 high severity criterion 通过）。
        failed_criteria: 未通过的 high severity criterion 名称列表。
        warnings: 未通过的 medium severity criterion 警告列表。
        summary: 人类可读摘要。
    """

    enabled: bool
    skipped: bool = False
    pathway: str = ""
    criteria: list[CriterionResult] = field(default_factory=list)
    passed: bool = False
    failed_criteria: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    summary: str = ""


# =============================================================================
# 辅助函数：解析 expected_behavior 字符串
# =============================================================================
# 支持 "10-20 min" / "10–20 min" / "10—20 min" 等连字符变体
_PEAK_RANGE_RE: re.Pattern[str] = re.compile(
    r"(\d+(?:\.\d+)?)\s*[-\u2013\u2014]\s*(\d+(?:\.\d+)?)"
)
_PEAK_LT_RE: re.Pattern[str] = re.compile(r"<\s*(\d+(?:\.\d+)?)")
_PEAK_GT_RE: re.Pattern[str] = re.compile(r">\s*(\d+(?:\.\d+)?)")
_PEAK_SINGLE_RE: re.Pattern[str] = re.compile(r"(\d+(?:\.\d+)?)")


def _parse_peak_range(value: Any) -> Tuple[Optional[float], Optional[float]]:
    """解析 expected_behavior 中的峰值时间字符串为 (lo, hi) 范围。

    支持格式：
      - ``"10-20 min"`` / ``"10–20 min"`` / ``"10—20 min"`` → (10.0, 20.0)
      - ``"<5 min"`` → (None, 5.0)
      - ``">5 min"`` → (5.0, None)
      - ``"5 min"`` / ``"5"`` → (5.0, 5.0)
      - 数字 5 → (5.0, 5.0)
      - 其他/空 → (None, None)

    Args:
        value: expected_behavior 中的原始值。

    Returns:
        ``(lo, hi)`` 元组；lo/hi 为 None 表示无下/上界。
    """
    if value is None:
        return None, None
    if isinstance(value, (int, float)):
        v = float(value)
        return v, v
    s = str(value).strip()
    if not s:
        return None, None
    # 范围 "10-20"
    m = _PEAK_RANGE_RE.search(s)
    if m:
        return float(m.group(1)), float(m.group(2))
    # "<5"
    m = _PEAK_LT_RE.search(s)
    if m:
        return None, float(m.group(1))
    # ">5"
    m = _PEAK_GT_RE.search(s)
    if m:
        return float(m.group(1)), None
    # 单值 "5"
    m = _PEAK_SINGLE_RE.search(s)
    if m:
        v = float(m.group(1))
        return v, v
    return None, None


def _format_range(lo: Optional[float], hi: Optional[float]) -> str:
    """将 (lo, hi) 范围格式化为人类可读字符串。"""
    if lo is not None and hi is not None:
        if lo == hi:
            return str(lo)
        return f"{lo}-{hi}"
    if hi is not None:
        return f"<{hi}"
    if lo is not None:
        return f">{lo}"
    return "unknown"


# =============================================================================
# 辅助函数：节点匹配
# =============================================================================
def _normalize_nodes(nodes: list[str] | None) -> set[str]:
    """将节点列表归一化为集合（空字符串过滤）。

    Args:
        nodes: 原始节点名列表，None 视为空。

    Returns:
        归一化后的节点名集合。
    """
    if not nodes:
        return set()
    result: set[str] = set()
    for n in nodes:
        norm = normalize_node_name(n)
        if norm:
            result.add(norm)
    return result


def _contains_nuclear_erk(extracted_norm: set[str]) -> bool:
    """检查归一化节点集合中是否含 Nuclear ERK 标记。

    匹配策略（满足任一即可）：
      - 任一归一化节点同时包含 ``"NUCLEAR"`` 与 ``"ERK"`` 子串
        （如 ``"Nuclear_ERK"`` → 归一化为 ``"NUCLEARERK"``）
      - 节点集合含 ``"ERK"`` 子串的节点 且 另含 ``"NUCLEAR"`` 子串的节点

    Args:
        extracted_norm: 归一化后的节点集合。

    Returns:
        存在 Nuclear ERK 返回 True。
    """
    has_erk = any("ERK" in node for node in extracted_norm)
    has_nuclear = any("NUCLEAR" in node for node in extracted_norm)
    return has_erk and has_nuclear


# =============================================================================
# 主函数
# =============================================================================
def check_acceptance(
    pathway: str,
    simulation_metrics: dict | None = None,
    extracted_nodes: list[str] | None = None,
    seven_axis_report=None,
    consistency_report=None,
    multi_dim_report=None,
    biomodels_comparison: dict | None = None,
    cited_pmids: list[str] | None = None,
    review_count: int = 0,
    experiments: list | None = None,
    discussion_coverage: float = 0.0,
) -> AcceptanceReport:
    """对指定通路执行 Acceptance Criteria 强制校验。

    校验 12 类 criterion（见模块文档）。任一 high severity criterion Fail
    则整体 ``passed=False``。CI 闸门据此决定是否拦截。

    Canonical 加载失败时降级为只校验非 Canonical 依赖项，
    ``warnings`` 追加 ``"canonical_missing"``。

    Args:
        pathway: 通路标识（如 ``"EGFR"``）。
        simulation_metrics: 仿真指标 dict，如
            ``{"erk_peak_time": 15.0, "egfr_peak_time": 3.0}``。
        extracted_nodes: Agent 提取的机制节点列表。
        seven_axis_report: :class:`SevenAxisReport` 实例（Validation Pyramid 结果）。
        consistency_report: :class:`ConsistencyReport` 实例。
        multi_dim_report: :class:`MultiDimConfidenceReport` 实例（Confidence 校验）。
        biomodels_comparison: BioModels 对比结果 dict，如
            ``{"done": True, "model_id": "BIOMD..."}``。
        cited_pmids: 引用的 PMID 列表。
        review_count: Review 文献数量。
        experiments: 实验列表。
        discussion_coverage: 10 问覆盖率（0.0-1.0）。

    Returns:
        AcceptanceReport。Feature Flag 关闭时返回 ``skipped=True`` 的空报告。
    """
    # -------------------------------------------------------------------------
    # Feature Flag 守护：SA_LOOP_TERMINATION 默认 OFF
    # 铁律：V4_SCIENTIFIC_ALIGNMENT_ENABLED=false 时，子 Flag 永远不生效
    # -------------------------------------------------------------------------
    if not settings.is_sa_feature_enabled("LOOP_TERMINATION"):
        return AcceptanceReport(
            enabled=False,
            skipped=True,
            pathway=pathway,
        )

    report = AcceptanceReport(
        enabled=True,
        skipped=False,
        pathway=pathway,
    )

    criteria: list[CriterionResult] = []
    warnings: list[str] = []

    # -------------------------------------------------------------------------
    # 1. 加载 Canonical Reference（失败时降级为只校验非 Canonical 依赖项）
    # -------------------------------------------------------------------------
    canonical = None
    try:
        canonical = load_canonical(pathway)
    except FileNotFoundError as exc:
        # Canonical 文件不存在：降级，warnings 追加 "canonical_missing"
        warnings.append("canonical_missing")
        logger.warning(
            "Acceptance Gate: pathway %r 的 Canonical 文件不存在，"
            "降级为只校验非 Canonical 依赖项: %s",
            pathway, exc,
        )
    except Exception as exc:  # noqa: BLE001 —— Canonical 解析异常不应阻塞 Gate
        warnings.append("canonical_missing")
        logger.warning(
            "Acceptance Gate: pathway %r Canonical 加载失败，降级: %s",
            pathway, exc,
        )

    # 预处理：归一化提取节点
    extracted_norm: set[str] = _normalize_nodes(extracted_nodes)

    # 预提取 expected_behavior 字段
    expected_behavior: dict = {}
    if canonical is not None:
        expected_behavior = getattr(canonical, "expected_behavior", {}) or {}

    # -------------------------------------------------------------------------
    # 2. erk_peak_range：ERK Peak 时间在 canonical.expected_behavior.erk_peak 范围内
    # -------------------------------------------------------------------------
    if canonical is not None:
        erk_peak_raw = expected_behavior.get("erk_peak")
        lo, hi = _parse_peak_range(erk_peak_raw)
        expected_str = f"ERK Peak 在 {_format_range(lo, hi)} min"
        actual_val = None
        if simulation_metrics and "erk_peak_time" in simulation_metrics:
            actual_val = simulation_metrics["erk_peak_time"]
            actual_str = str(actual_val)
        else:
            actual_str = "missing"

        passed = False
        if actual_val is not None and (lo is not None or hi is not None):
            in_range = True
            if lo is not None and actual_val < lo:
                in_range = False
            if hi is not None and actual_val > hi:
                in_range = False
            passed = in_range

        criteria.append(CriterionResult(
            name="erk_peak_range",
            passed=passed,
            expected=expected_str,
            actual=actual_str,
            severity="high",
        ))

    # -------------------------------------------------------------------------
    # 3. egfr_peak_early：EGFR Peak 时间 < canonical.expected_behavior.egfr_peak 上限
    # -------------------------------------------------------------------------
    if canonical is not None:
        egfr_peak_raw = expected_behavior.get("egfr_peak")
        _, egfr_hi = _parse_peak_range(egfr_peak_raw)
        expected_str = f"EGFR Peak 早于 {_format_range(None, egfr_hi)} min"
        actual_val = None
        if simulation_metrics and "egfr_peak_time" in simulation_metrics:
            actual_val = simulation_metrics["egfr_peak_time"]
            actual_str = str(actual_val)
        else:
            actual_str = "missing"

        passed = False
        if actual_val is not None and egfr_hi is not None:
            passed = actual_val < egfr_hi

        criteria.append(CriterionResult(
            name="egfr_peak_early",
            passed=passed,
            expected=expected_str,
            actual=actual_str,
            severity="high",
        ))

    # -------------------------------------------------------------------------
    # 4. dusp_feedback：extracted_nodes 包含 known_negative_feedback 中任一节点
    # -------------------------------------------------------------------------
    if canonical is not None:
        feedback_nodes = getattr(canonical, "known_negative_feedback", []) or []
        feedback_norm = _normalize_nodes(feedback_nodes)
        matched = extracted_norm & feedback_norm
        expected_str = f"含负反馈节点 ({', '.join(feedback_nodes)})"
        actual_str = ", ".join(sorted(matched)) if matched else "none"
        criteria.append(CriterionResult(
            name="dusp_feedback",
            passed=bool(matched),
            expected=expected_str,
            actual=actual_str,
            severity="high",
        ))

    # -------------------------------------------------------------------------
    # 5. nuclear_erk：extracted_nodes 含 ERK 且有 nuclear 标记
    # -------------------------------------------------------------------------
    has_nuclear_erk = _contains_nuclear_erk(extracted_norm)
    criteria.append(CriterionResult(
        name="nuclear_erk",
        passed=has_nuclear_erk,
        expected="含 ERK 且有 nuclear 标记（或 Nuclear_ERK）",
        actual="present" if has_nuclear_erk else "absent",
        severity="high",
    ))

    # -------------------------------------------------------------------------
    # 6. biomodels_comparison：biomodels_comparison["done"]==True
    # -------------------------------------------------------------------------
    bm_done = bool(
        biomodels_comparison and biomodels_comparison.get("done") is True
    )
    bm_actual = "done" if bm_done else (
        "not_done" if biomodels_comparison else "missing"
    )
    criteria.append(CriterionResult(
        name="biomodels_comparison",
        passed=bm_done,
        expected="BioModels Comparison 完成",
        actual=bm_actual,
        severity="high",
    ))

    # -------------------------------------------------------------------------
    # 7. pubmed_count：len(cited_pmids)>=5
    # -------------------------------------------------------------------------
    pmid_count = len(cited_pmids) if cited_pmids else 0
    criteria.append(CriterionResult(
        name="pubmed_count",
        passed=pmid_count >= _MIN_PUBMED_COUNT,
        expected=f"PubMed 引用 >= {_MIN_PUBMED_COUNT}",
        actual=str(pmid_count),
        severity="high",
    ))

    # -------------------------------------------------------------------------
    # 8. review_count：review_count>=2
    # -------------------------------------------------------------------------
    criteria.append(CriterionResult(
        name="review_count",
        passed=review_count >= _MIN_REVIEW_COUNT,
        expected=f"Review 文献 >= {_MIN_REVIEW_COUNT}",
        actual=str(review_count),
        severity="high",
    ))

    # -------------------------------------------------------------------------
    # 9. mechanism_evidence：extracted_nodes 覆盖 required_nodes 的 80%+
    # -------------------------------------------------------------------------
    if canonical is not None:
        required_nodes = getattr(canonical, "required_nodes", []) or []
        required_norm = _normalize_nodes(required_nodes)
        if required_norm:
            matched_count = len(extracted_norm & required_norm)
            coverage = matched_count / len(required_norm)
            missing = required_norm - extracted_norm
            actual_str = (
                f"coverage={coverage:.0%} ({matched_count}/{len(required_norm)})"
                + (f", missing={sorted(missing)}" if missing else "")
            )
        else:
            coverage = 0.0
            actual_str = "no required_nodes in canonical"
        criteria.append(CriterionResult(
            name="mechanism_evidence",
            passed=coverage >= _MIN_MECHANISM_COVERAGE,
            expected=f"机制节点覆盖 >= {_MIN_MECHANISM_COVERAGE:.0%}",
            actual=actual_str,
            severity="high",
        ))

    # -------------------------------------------------------------------------
    # 10. experiment_chain：len(experiments)>=2（medium severity）
    # -------------------------------------------------------------------------
    exp_count = len(experiments) if experiments else 0
    exp_result = CriterionResult(
        name="experiment_chain",
        passed=exp_count >= _MIN_EXPERIMENT_COUNT,
        expected=f"实验数 >= {_MIN_EXPERIMENT_COUNT}",
        actual=str(exp_count),
        severity="medium",
    )
    criteria.append(exp_result)
    if not exp_result.passed:
        warnings.append(
            f"experiment_chain: 实验数 {exp_count} 低于阈值 "
            f"{_MIN_EXPERIMENT_COUNT}"
        )

    # -------------------------------------------------------------------------
    # 11. discussion_10q：discussion_coverage>=0.8（medium severity）
    # -------------------------------------------------------------------------
    disc_result = CriterionResult(
        name="discussion_10q",
        passed=discussion_coverage >= _MIN_DISCUSSION_COVERAGE,
        expected=f"10 问覆盖率 >= {_MIN_DISCUSSION_COVERAGE:.0%}",
        actual=f"{discussion_coverage:.0%}",
        severity="medium",
    )
    criteria.append(disc_result)
    if not disc_result.passed:
        warnings.append(
            f"discussion_10q: 覆盖率 {discussion_coverage:.0%} 低于阈值 "
            f"{_MIN_DISCUSSION_COVERAGE:.0%}"
        )

    # -------------------------------------------------------------------------
    # 12. confidence：multi_dim_report.final_confidence>0.9（或 seven_axis overall>0.9）
    # -------------------------------------------------------------------------
    conf_value: Optional[float] = None
    conf_source = "missing"
    if multi_dim_report is not None:
        fc = getattr(multi_dim_report, "final_confidence", None)
        if isinstance(fc, (int, float)):
            conf_value = float(fc)
            conf_source = f"multi_dim.final_confidence={conf_value:.3f}"
    if conf_value is None and seven_axis_report is not None:
        oc = getattr(seven_axis_report, "overall_confidence", None)
        if isinstance(oc, (int, float)):
            conf_value = float(oc)
            conf_source = f"seven_axis.overall_confidence={conf_value:.3f}"

    criteria.append(CriterionResult(
        name="confidence",
        passed=conf_value is not None and conf_value > _MIN_CONFIDENCE,
        expected=f"Confidence > {_MIN_CONFIDENCE}",
        actual=conf_source,
        severity="high",
    ))

    # -------------------------------------------------------------------------
    # 13. validation_pyramid：seven_axis.overall_passed==True
    #     （或 consistency_report 无 violation）
    # -------------------------------------------------------------------------
    pyramid_passed = False
    pyramid_actual = "missing"
    if seven_axis_report is not None:
        op = getattr(seven_axis_report, "overall_passed", None)
        pyramid_passed = bool(op)
        pyramid_actual = f"seven_axis.overall_passed={pyramid_passed}"
    elif consistency_report is not None:
        violations = getattr(consistency_report, "violations", []) or []
        pyramid_passed = len(violations) == 0
        pyramid_actual = (
            f"consistency.violations={len(violations)}"
        )

    criteria.append(CriterionResult(
        name="validation_pyramid",
        passed=pyramid_passed,
        expected="Validation Pyramid 通过（7 轴 overall_passed 或无 consistency 违规）",
        actual=pyramid_actual,
        severity="high",
    ))

    # -------------------------------------------------------------------------
    # 汇总：passed = 所有 high severity criterion 通过
    # -------------------------------------------------------------------------
    high_criteria = [c for c in criteria if c.severity == "high"]
    failed_high = [c.name for c in high_criteria if not c.passed]
    total = len(criteria)
    passed_count = sum(1 for c in criteria if c.passed)

    report.criteria = criteria
    report.failed_criteria = failed_high
    report.warnings = warnings
    report.passed = len(failed_high) == 0

    # 生成人类可读摘要
    if report.passed:
        report.summary = f"Acceptance: PASS ({passed_count}/{total} criteria)"
    else:
        failed_str = ", ".join(failed_high) if failed_high else ""
        report.summary = (
            f"Acceptance: FAIL ({passed_count}/{total} passed"
            f"{', failed: ' + failed_str if failed_str else ''})"
        )

    logger.info(
        "Acceptance Gate: pathway=%s, passed=%s, criteria=%d, "
        "failed_high=%s, warnings=%s",
        pathway, report.passed, total, failed_high, warnings,
    )

    return report


__all__ = [
    "CriterionResult",
    "AcceptanceReport",
    "check_acceptance",
]
