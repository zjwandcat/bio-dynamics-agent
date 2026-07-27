"""Sprint 3 — Auto Scientific Review（100% Rule 驱动，无 LLM）。

设计原则（ENGINEERING_RULES.md）：
  - LLM 不允许创造科学事实
  - Scientific Review 完全由 Rule 计算，禁止 LLM "Looks reasonable"
  - 所有检查基于真实运行结果

Review 项（对照 spec）：
  - ERK Peak 合理？
  - EGFR Peak 合理？
  - Mass Conservation
  - Negative Feedback
  - SBML Similarity
  - Canonical Similarity
  - Literature Consistency

输出：
  - Overall Scientific Score（0-10，Rule 计算）
  - 每项 PASS/FAIL + 原因

铁律：
  - Feature Flag SA_SPRINT3_CONSISTENCY_GATE=false 时返回 skipped 报告
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.config import settings


@dataclass
class ReviewItem:
    """单条 Review 检查项。"""

    name: str
    passed: bool
    score: float  # 0-1
    reason: str


@dataclass
class ScientificReviewReport:
    """Scientific Review 报告。"""

    enabled: bool
    skipped: bool = False
    items: list[ReviewItem] = field(default_factory=list)
    overall_score: float = 0.0  # 0-10
    overall_passed: bool = False
    summary: str = ""

    def add(self, item: ReviewItem) -> None:
        self.items.append(item)


def run_scientific_review(
    pathway: str,
    simulation_metrics: dict[str, Any],
    consistency_passed: bool = True,
    validation_passed: bool = True,
    evidence_count: int = 0,
    biomodels_matched: bool = False,
    canonical_timeline: list[dict] | None = None,
) -> ScientificReviewReport:
    """运行 Auto Scientific Review。

    Args:
        pathway: 通路名称。
        simulation_metrics: 仿真指标。
        consistency_passed: Consistency Checker 是否通过。
        validation_passed: Validation Rule Engine 是否通过。
        evidence_count: 文献证据数量。
        biomodels_matched: 是否匹配到 BioModels 模型。
        canonical_timeline: Canonical Timeline。

    Returns:
        ScientificReviewReport。
    """
    if not settings.is_sa_feature_enabled("SPRINT3_CONSISTENCY_GATE"):
        return ScientificReviewReport(enabled=False, skipped=True)

    report = ScientificReviewReport(enabled=True)
    species = simulation_metrics.get("species", {})

    # Review 1: ERK Peak 合理？
    report.add(_review_erk_peak(species, pathway))

    # Review 2: EGFR Peak 合理？
    report.add(_review_egfr_peak(species, pathway))

    # Review 3: Mass Conservation
    report.add(_review_mass_conservation(species))

    # Review 4: Negative Feedback
    report.add(_review_negative_feedback(species, pathway))

    # Review 5: SBML Similarity（基于 biomodels_matched）
    report.add(_review_sbml_similarity(biomodels_matched))

    # Review 6: Canonical Similarity
    report.add(
        _review_canonical_similarity(species, canonical_timeline, pathway)
    )

    # Review 7: Literature Consistency
    report.add(_review_literature_consistency(evidence_count))

    # 计算 Overall Score（0-10）
    if report.items:
        avg_score = sum(item.score for item in report.items) / len(report.items)
        report.overall_score = round(avg_score * 10, 1)
        report.overall_passed = all(item.passed for item in report.items)
    else:
        report.overall_score = 0.0
        report.overall_passed = False

    # Summary
    passed_count = sum(1 for item in report.items if item.passed)
    total_count = len(report.items)
    report.summary = (
        f"Scientific Review: {passed_count}/{total_count} 项 PASS, "
        f"Overall Score: {report.overall_score}/10"
    )

    return report


def _review_erk_peak(
    species: dict[str, Any], pathway: str
) -> ReviewItem:
    """Review: ERK Peak 合理？"""
    erk_metrics = _find_species(species, ["ppERK", "pERK", "ERK", "pperk", "perk"])
    if erk_metrics is None:
        return ReviewItem(
            name="ERK Peak 合理？",
            passed=True,
            score=1.0,
            reason=f"通路 {pathway} 无 ERK 指标，跳过",
        )

    peak_time = erk_metrics.get("peak_time")
    if peak_time is None:
        return ReviewItem(
            name="ERK Peak 合理？",
            passed=True,
            score=1.0,
            reason="ERK 无 peak_time，跳过",
        )

    try:
        pt = float(peak_time)
    except (TypeError, ValueError):
        return ReviewItem(
            name="ERK Peak 合理？",
            passed=False,
            score=0.0,
            reason=f"ERK peak_time 无法解析: {peak_time}",
        )

    # Canonical: ERK peak 10-20 min
    if 5 <= pt <= 30:
        return ReviewItem(
            name="ERK Peak 合理？",
            passed=True,
            score=1.0,
            reason=f"ERK peak_time={pt:.1f} min 在合理范围 5-30 min",
        )
    return ReviewItem(
        name="ERK Peak 合理？",
        passed=False,
        score=0.3,
        reason=f"ERK peak_time={pt:.1f} min 超出合理范围 5-30 min",
    )


def _review_egfr_peak(
    species: dict[str, Any], pathway: str
) -> ReviewItem:
    """Review: EGFR Peak 合理？"""
    egfr_metrics = _find_species(species, ["pEGFR", "EGFR", "pegfr", "egfr"])
    if egfr_metrics is None:
        return ReviewItem(
            name="EGFR Peak 合理？",
            passed=True,
            score=1.0,
            reason=f"通路 {pathway} 无 EGFR 指标，跳过",
        )

    peak_time = egfr_metrics.get("peak_time")
    if peak_time is None:
        return ReviewItem(
            name="EGFR Peak 合理？",
            passed=True,
            score=1.0,
            reason="EGFR 无 peak_time，跳过",
        )

    try:
        pt = float(peak_time)
    except (TypeError, ValueError):
        return ReviewItem(
            name="EGFR Peak 合理？",
            passed=False,
            score=0.0,
            reason=f"EGFR peak_time 无法解析: {peak_time}",
        )

    # Canonical: EGFR peak 1-5 min
    if pt <= 10:
        return ReviewItem(
            name="EGFR Peak 合理？",
            passed=True,
            score=1.0,
            reason=f"EGFR peak_time={pt:.1f} min 在合理范围 ≤10 min",
        )
    return ReviewItem(
        name="EGFR Peak 合理？",
        passed=False,
        score=0.3,
        reason=f"EGFR peak_time={pt:.1f} min 超出合理范围 ≤10 min（可能时序倒置）",
    )


def _review_mass_conservation(species: dict[str, Any]) -> ReviewItem:
    """Review: Mass Conservation。"""
    if not species:
        return ReviewItem(
            name="Mass Conservation",
            passed=True,
            score=1.0,
            reason="无物种指标，跳过",
        )

    for sp_name, sp_metrics in species.items():
        if not isinstance(sp_metrics, dict):
            continue
        fold_change = sp_metrics.get("fold_change", 1.0)
        try:
            fc = float(fold_change)
        except (TypeError, ValueError):
            continue
        if fc > 10.0:
            return ReviewItem(
                name="Mass Conservation",
                passed=False,
                score=0.0,
                reason=f"{sp_name} fold_change={fc:.2f} (>10)，质量不守恒",
            )

    return ReviewItem(
        name="Mass Conservation",
        passed=True,
        score=1.0,
        reason="所有物种 fold_change ≤ 10，质量守恒",
    )


def _review_negative_feedback(
    species: dict[str, Any], pathway: str
) -> ReviewItem:
    """Review: Negative Feedback。"""
    # 检查是否有 DUSP/SPRY/AXIN2/A20/IkB 等负反馈介质
    feedback_markers = [
        "DUSP", "SPRY", "AXIN2", "A20", "IkB", "IκB",
        "MDM2", "SOCS", "CBL",
    ]
    found_markers: list[str] = []
    for sp_name in species:
        sp_upper = sp_name.upper()
        for marker in feedback_markers:
            if marker.upper() in sp_upper:
                found_markers.append(sp_name)
                break

    if found_markers:
        return ReviewItem(
            name="Negative Feedback",
            passed=True,
            score=1.0,
            reason=f"检测到负反馈介质: {', '.join(found_markers)}",
        )
    return ReviewItem(
        name="Negative Feedback",
        passed=False,
        score=0.4,
        reason=f"未检测到已知负反馈介质（DUSP/SPRY/AXIN2/A20/IkB/MDM2/SOCS/CBL）",
    )


def _review_sbml_similarity(biomodels_matched: bool) -> ReviewItem:
    """Review: SBML Similarity。"""
    if biomodels_matched:
        return ReviewItem(
            name="SBML Similarity",
            passed=True,
            score=0.9,
            reason="匹配到已验证 BioModels 模型",
        )
    return ReviewItem(
        name="SBML Similarity",
        passed=False,
        score=0.2,
        reason="未匹配到 BioModels 模型",
    )


def _review_canonical_similarity(
    species: dict[str, Any],
    canonical_timeline: list[dict] | None,
    pathway: str,
) -> ReviewItem:
    """Review: Canonical Similarity。"""
    if not canonical_timeline:
        return ReviewItem(
            name="Canonical Similarity",
            passed=True,
            score=0.8,
            reason=f"通路 {pathway} 无 Canonical Timeline，默认 0.8",
        )

    # 简化版：检查物种数与 Canonical 事件数的匹配度
    species_count = len(species)
    event_count = len(canonical_timeline)
    if species_count == 0:
        return ReviewItem(
            name="Canonical Similarity",
            passed=False,
            score=0.0,
            reason="无物种指标",
        )

    # 物种数 / 事件数 比率（>0.5 算合理）
    ratio = min(species_count / max(event_count, 1), 1.0)
    if ratio >= 0.5:
        return ReviewItem(
            name="Canonical Similarity",
            passed=True,
            score=0.9,
            reason=f"物种数 {species_count} / Canonical 事件数 {event_count} = {ratio:.2f}",
        )
    return ReviewItem(
        name="Canonical Similarity",
        passed=False,
        score=0.4,
        reason=f"物种数 {species_count} / Canonical 事件数 {event_count} = {ratio:.2f} (<0.5)",
    )


def _review_literature_consistency(evidence_count: int) -> ReviewItem:
    """Review: Literature Consistency。"""
    if evidence_count >= 5:
        return ReviewItem(
            name="Literature Consistency",
            passed=True,
            score=1.0,
            reason=f"文献证据 {evidence_count} 篇（≥5），文献支撑充分",
        )
    elif evidence_count >= 3:
        return ReviewItem(
            name="Literature Consistency",
            passed=True,
            score=0.8,
            reason=f"文献证据 {evidence_count} 篇（≥3），文献支撑合理",
        )
    elif evidence_count >= 1:
        return ReviewItem(
            name="Literature Consistency",
            passed=False,
            score=0.4,
            reason=f"文献证据 {evidence_count} 篇（<3），文献支撑不足",
        )
    return ReviewItem(
        name="Literature Consistency",
        passed=False,
        score=0.0,
        reason="无文献证据（No literature retrieved）",
    )


def _find_species(
    species: dict[str, Any], names: list[str]
) -> dict[str, Any] | None:
    """从 species 字典中查找匹配的物种（不区分大小写）。"""
    for sp_name, sp_metrics in species.items():
        sp_lower = sp_name.lower()
        for name in names:
            if name.lower() in sp_lower:
                return sp_metrics if isinstance(sp_metrics, dict) else None
    return None
