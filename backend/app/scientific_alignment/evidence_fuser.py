# BioDynamics Agent v4 - Scientific Alignment Loop: Evidence Fuser (Task 9)
#
# 逐句五源证据融合：对每条机制断言，融合五个证据来源（PubMed / BioModels /
# Simulation / Inference / Hypothesis），输出支撑来源标注、融合置信度与地下证据检测。
#
# 五源定义：
#   [A] PubMed      — 文献证据（Review / Mechanism Paper 等）
#   [B] BioModels   — 已发表计算模型参数/动力学
#   [C] Simulation  — Agent 自己仿真结果
#   [D] Inference   — LLM 推理（无外部证据支撑）
#   [E] Hypothesis  — 假设（待验证）
#
# 依赖：app.config.settings（Feature Flag 校验）；不引入新依赖。
#
# 核心导出：
#   from app.scientific_alignment.evidence_fuser import (
#       EvidenceSource, EvidenceItem, FusedAssertion,
#       EvidenceFusionReport, fuse_evidence, evidence_docs_to_items,
#   )

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

from app.config import settings

logger = logging.getLogger(__name__)


# =============================================================================
# 常量：融合置信度加分
# =============================================================================
_BONUS_PUBMED_REVIEW: float = 0.3       # [A] Review (5分)
_BONUS_PUBMED_MECHANISM: float = 0.25   # [A] Mechanism Paper (4分)
_BONUS_PUBMED_OTHER: float = 0.15       # [A] 其他 PubMed 文献
_BONUS_BIOMODELS: float = 0.25          # [B] BioModels
_BONUS_SIMULATION: float = 0.2          # [C] Simulation
_BONUS_INFERENCE: float = 0.1           # [D] Inference
_BONUS_HYPOTHESIS: float = 0.05         # [E] Hypothesis

# confidence 阈值：用于区分 [A] Review 与 Mechanism Paper
# evidence_docs_to_items 将 EvidenceType 的 base_score 归一化到 0-1 作为 confidence
# REVIEW(5) → 1.0, MECHANISM_PAPER(4) → 0.8, BIOMODELS_SOURCE(3) → 0.6, ...
_PUBMED_REVIEW_CONF_THRESHOLD: float = 0.95
_PUBMED_MECHANISM_CONF_THRESHOLD: float = 0.75

# 地下证据置信度上限
_UNDERGROUNDED_CONF_CAP: float = 0.3
_HYPOTHESIS_ONLY_CONF_CAP: float = 0.2

# 默认 Inference 证据项 confidence（无显式推理证据时的兜底）
_DEFAULT_INFERENCE_CONFIDENCE: float = 0.5


# =============================================================================
# EvidenceSource 枚举
# =============================================================================
class EvidenceSource(str, Enum):
    """五源证据枚举，value 为来源标注字母。"""

    PUBMED = "A"          # [A] PubMed 文献
    BIOMODELS = "B"       # [B] BioModels 计算模型
    SIMULATION = "C"      # [C] Agent 仿真
    INFERENCE = "D"       # [D] LLM 推理
    HYPOTHESIS = "E"      # [E] 假设


# =============================================================================
# EvidenceItem 数据类
# =============================================================================
@dataclass
class EvidenceItem:
    """单个证据项。

    Attributes:
        source: 证据来源（A/B/C/D/E）。
        reference: 引用标识（PMID / BIOMD ID / "sim_run_001" /
            "inference" / "hypothesis"）。
        snippet: 证据片段（摘要/参数值/metric）。
        confidence: 该证据项的可信度（0.0-1.0）；[A] 来源用归一化
            base_score 编码文献类型，供 fuse_evidence 区分 Review 与
            Mechanism Paper。
    """

    source: EvidenceSource
    reference: str
    snippet: str = ""
    confidence: float = 1.0


# =============================================================================
# FusedAssertion 数据类
# =============================================================================
@dataclass
class FusedAssertion:
    """融合后的机制断言。

    Attributes:
        assertion: 断言文本。
        sources: 支撑来源列表（去重，按 A>B>C>D>E 排序）。
        evidence_items: 证据明细。
        fused_confidence: 融合后置信度（0.0-1.0）。
        undergrounded: 全 [D] 无外部支撑。
        hypothesis_only: 仅 [E] 假设。
        defect: 缺陷标签（evidence_undergrounded / hypothesis_only / 空）。
    """

    assertion: str
    sources: list[EvidenceSource]
    evidence_items: list[EvidenceItem]
    fused_confidence: float
    undergrounded: bool = False
    hypothesis_only: bool = False
    defect: str = ""


# =============================================================================
# EvidenceFusionReport 数据类
# =============================================================================
@dataclass
class EvidenceFusionReport:
    """证据融合报告。

    Attributes:
        enabled: Feature Flag 是否启用。
        assertions: 融合后的断言列表。
        total_assertions: 断言总数。
        undergrounded_count: 地下证据断言数。
        hypothesis_only_count: 仅假设断言数。
        source_coverage: 每源覆盖断言数 {"A": n, "B": n, ...}。
        skipped: 是否跳过（Flag OFF 时为 True）。
    """

    enabled: bool
    assertions: list[FusedAssertion]
    total_assertions: int
    undergrounded_count: int
    hypothesis_only_count: int
    source_coverage: dict[str, int]
    skipped: bool = False


# =============================================================================
# 辅助函数
# =============================================================================
def _get_positional_item(
    evidence_list: list[EvidenceItem] | None,
    index: int,
) -> EvidenceItem | None:
    """按索引从证据列表取项；列表为 None 或索引越界时返回 None。"""
    if not evidence_list:
        return None
    if index < 0 or index >= len(evidence_list):
        return None
    return evidence_list[index]


def _compute_fused_confidence(
    evidence_items: list[EvidenceItem],
) -> float:
    """计算融合置信度。

    算法（每源存在即加分，上限 1.0，下限 0.0）：
      - 有 [A] Review(5分) → +0.3
      - 有 [A] Mechanism Paper(4分) → +0.25
      - 有 [A] 其他 PubMed → +0.15
      - 有 [B] BioModels → +0.25
      - 有 [C] Simulation → +0.2
      - 有 [D] Inference → +0.1
      - 有 [E] Hypothesis → +0.05

    Args:
        evidence_items: 该断言的所有证据项。

    Returns:
        融合置信度（0.0-1.0）。
    """
    has_pubmed_review = False
    has_pubmed_mechanism = False
    has_pubmed_other = False
    has_biomodels = False
    has_simulation = False
    has_inference = False
    has_hypothesis = False

    for item in evidence_items:
        if item.source == EvidenceSource.PUBMED:
            if item.confidence >= _PUBMED_REVIEW_CONF_THRESHOLD:
                has_pubmed_review = True
            elif item.confidence >= _PUBMED_MECHANISM_CONF_THRESHOLD:
                has_pubmed_mechanism = True
            else:
                has_pubmed_other = True
        elif item.source == EvidenceSource.BIOMODELS:
            has_biomodels = True
        elif item.source == EvidenceSource.SIMULATION:
            has_simulation = True
        elif item.source == EvidenceSource.INFERENCE:
            has_inference = True
        elif item.source == EvidenceSource.HYPOTHESIS:
            has_hypothesis = True

    conf = 0.0
    if has_pubmed_review:
        conf += _BONUS_PUBMED_REVIEW
    if has_pubmed_mechanism:
        conf += _BONUS_PUBMED_MECHANISM
    if has_pubmed_other:
        conf += _BONUS_PUBMED_OTHER
    if has_biomodels:
        conf += _BONUS_BIOMODELS
    if has_simulation:
        conf += _BONUS_SIMULATION
    if has_inference:
        conf += _BONUS_INFERENCE
    if has_hypothesis:
        conf += _BONUS_HYPOTHESIS

    return max(0.0, min(1.0, conf))


def _sort_sources(sources: list[EvidenceSource]) -> list[EvidenceSource]:
    """按 A>B>C>D>E 排序来源列表（去重保序）。"""
    order = {
        EvidenceSource.PUBMED: 0,
        EvidenceSource.BIOMODELS: 1,
        EvidenceSource.SIMULATION: 2,
        EvidenceSource.INFERENCE: 3,
        EvidenceSource.HYPOTHESIS: 4,
    }
    unique = list(dict.fromkeys(sources))
    return sorted(unique, key=lambda s: order.get(s, 99))


# =============================================================================
# 主函数：fuse_evidence
# =============================================================================
def fuse_evidence(
    assertions: list[str],
    pubmed_evidence: list[EvidenceItem] | None = None,
    biomodels_evidence: list[EvidenceItem] | None = None,
    simulation_evidence: list[EvidenceItem] | None = None,
    inference_evidence: list[EvidenceItem] | None = None,
    hypothesis_evidence: list[EvidenceItem] | None = None,
) -> EvidenceFusionReport:
    """对每条机制断言融合五个证据来源。

    证据按断言顺序位置匹配：assertion[i] 取各源 evidence_list[i]。
    若某断言无任何显式证据，默认标注 [D] Inference（LLM 生成的断言
    至少有推理支撑）。

    Args:
        assertions: 机制断言文本列表。
        pubmed_evidence: [A] PubMed 文献证据（按断言顺序位置匹配）。
        biomodels_evidence: [B] BioModels 计算模型证据。
        simulation_evidence: [C] Agent 仿真证据。
        inference_evidence: [D] LLM 推理证据。
        hypothesis_evidence: [E] 假设证据。

    Returns:
        EvidenceFusionReport：Flag OFF 时返回 skipped 报告；
        assertions 为空时返回空报告（不算 skipped）。

    铁律：V4_SCIENTIFIC_ALIGNMENT_ENABLED=false 或 SA_EVIDENCE_FUSION=false
          时，模块返回 skipped，不执行融合。
    """
    # Feature Flag 校验：总开关或子开关任一关闭即跳过
    if not settings.is_sa_feature_enabled("EVIDENCE_FUSION"):
        return EvidenceFusionReport(
            enabled=False,
            skipped=True,
            assertions=[],
            total_assertions=0,
            undergrounded_count=0,
            hypothesis_only_count=0,
            source_coverage={},
        )

    # assertions 为空时返回空报告（不算 skipped）
    if not assertions:
        return EvidenceFusionReport(
            enabled=True,
            skipped=False,
            assertions=[],
            total_assertions=0,
            undergrounded_count=0,
            hypothesis_only_count=0,
            source_coverage={},
        )

    fused: list[FusedAssertion] = []
    undergrounded_count = 0
    hypothesis_only_count = 0
    source_coverage: dict[str, int] = {s.value: 0 for s in EvidenceSource}

    for i, assertion_text in enumerate(assertions):
        sources: list[EvidenceSource] = []
        evidence_items: list[EvidenceItem] = []

        # 位置匹配：assertion[i] ← evidence_list[i]
        pubmed_item = _get_positional_item(pubmed_evidence, i)
        if pubmed_item is not None:
            sources.append(EvidenceSource.PUBMED)
            evidence_items.append(pubmed_item)

        biomodels_item = _get_positional_item(biomodels_evidence, i)
        if biomodels_item is not None:
            sources.append(EvidenceSource.BIOMODELS)
            evidence_items.append(biomodels_item)

        sim_item = _get_positional_item(simulation_evidence, i)
        if sim_item is not None:
            sources.append(EvidenceSource.SIMULATION)
            evidence_items.append(sim_item)

        infer_item = _get_positional_item(inference_evidence, i)
        if infer_item is not None:
            sources.append(EvidenceSource.INFERENCE)
            evidence_items.append(infer_item)

        hypo_item = _get_positional_item(hypothesis_evidence, i)
        if hypo_item is not None:
            sources.append(EvidenceSource.HYPOTHESIS)
            evidence_items.append(hypo_item)

        # 无任何显式证据 → 默认 [D] Inference
        # （LLM 生成的断言至少有推理支撑，标记为地下证据）
        if not sources:
            default_item = EvidenceItem(
                source=EvidenceSource.INFERENCE,
                reference="inference",
                snippet="",
                confidence=_DEFAULT_INFERENCE_CONFIDENCE,
            )
            sources.append(EvidenceSource.INFERENCE)
            evidence_items.append(default_item)

        # 计算融合置信度
        fused_conf = _compute_fused_confidence(evidence_items)

        # 地下证据检测
        undergrounded = False
        hypothesis_only = False
        defect = ""

        external_sources = {
            EvidenceSource.PUBMED,
            EvidenceSource.BIOMODELS,
            EvidenceSource.SIMULATION,
        }
        source_set = set(sources)

        # 全 [D] Inference 而无 [A][B][C] 支撑
        if source_set == {EvidenceSource.INFERENCE}:
            undergrounded = True
            defect = "evidence_undergrounded"
            if fused_conf > _UNDERGROUNDED_CONF_CAP:
                fused_conf = _UNDERGROUNDED_CONF_CAP
            undergrounded_count += 1
        # 仅 [E] Hypothesis
        elif source_set == {EvidenceSource.HYPOTHESIS}:
            hypothesis_only = True
            defect = "hypothesis_only"
            if fused_conf > _HYPOTHESIS_ONLY_CONF_CAP:
                fused_conf = _HYPOTHESIS_ONLY_CONF_CAP
            hypothesis_only_count += 1

        # 排序来源（A>B>C>D>E）
        sorted_sources = _sort_sources(sources)

        # 更新 source_coverage（每源覆盖断言数）
        for s in set(sources):
            source_coverage[s.value] += 1

        fused.append(FusedAssertion(
            assertion=assertion_text,
            sources=sorted_sources,
            evidence_items=evidence_items,
            fused_confidence=fused_conf,
            undergrounded=undergrounded,
            hypothesis_only=hypothesis_only,
            defect=defect,
        ))

    return EvidenceFusionReport(
        enabled=True,
        skipped=False,
        assertions=fused,
        total_assertions=len(fused),
        undergrounded_count=undergrounded_count,
        hypothesis_only_count=hypothesis_only_count,
        source_coverage=source_coverage,
    )


# =============================================================================
# 便捷函数：evidence_docs_to_items
# =============================================================================
def evidence_docs_to_items(evidence_docs: list) -> list[EvidenceItem]:
    """从 EvidenceRanker 的 EvidenceDoc 列表构造 [A] EvidenceItem。

    将 EvidenceDoc.evidence_type 的 base_score 归一化到 0-1 作为 confidence，
    使 fuse_evidence 可据此区分 Review（confidence≥0.95 → +0.3）与
    Mechanism Paper（confidence≥0.75 → +0.25）。

    归一化映射：
      REVIEW(5) → 1.0
      MECHANISM_PAPER(4) → 0.8
      BIOMODELS_SOURCE(3) → 0.6
      RECENT_APPLICATION(2) → 0.4
      CASE_REPORT(1) → 0.2

    Args:
        evidence_docs: EvidenceDoc 列表（来自
            app.scientific_alignment.evidence_ranker）。

    Returns:
        EvidenceItem 列表（source=PUBMED）。
    """
    if not evidence_docs:
        return []

    items: list[EvidenceItem] = []
    for doc in evidence_docs:
        # base_score 归一化到 0-1
        base_score = getattr(doc, "base_score", 0)
        if base_score and base_score > 0:
            base = float(base_score)
        else:
            evidence_type = getattr(doc, "evidence_type", None)
            base = float(int(evidence_type)) if evidence_type else 0.0
        confidence = base / 5.0 if base > 0 else 0.0

        # snippet 组装标题 + 期刊年份
        title = getattr(doc, "title", "") or ""
        journal = getattr(doc, "journal", "") or ""
        year = getattr(doc, "year", 0) or 0
        if journal and year:
            snippet = f"{title} ({journal}, {year})"
        elif year:
            snippet = f"{title} ({year})"
        else:
            snippet = title

        items.append(EvidenceItem(
            source=EvidenceSource.PUBMED,
            reference=getattr(doc, "pmid", "") or "",
            snippet=snippet,
            confidence=confidence,
        ))

    return items


__all__ = [
    "EvidenceSource",
    "EvidenceItem",
    "FusedAssertion",
    "EvidenceFusionReport",
    "fuse_evidence",
    "evidence_docs_to_items",
]
