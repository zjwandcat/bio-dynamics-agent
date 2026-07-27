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
# 常量：source_role / source_tag 映射（依赖 EvidenceSource，故置于枚举之后）
# =============================================================================
# EvidenceSource → source_role 字符串（用于 RAG 输出标注与 EvidenceFuser 自动归类）
# 注意：[D] 在 EvidenceSource 枚举中名为 INFERENCE，但 Spec/Task 统一称 [D] Mechanism
#       （与 evidence_graph.EvidenceType.MECHANISM = "[D]Mechanism" 对齐）
_SOURCE_ROLE_MAP: dict[EvidenceSource, str] = {
    EvidenceSource.PUBMED: "PubMed",
    EvidenceSource.BIOMODELS: "BioModels",
    EvidenceSource.SIMULATION: "Simulation",
    EvidenceSource.INFERENCE: "Mechanism",
    EvidenceSource.HYPOTHESIS: "Hypothesis",
}

# EvidenceSource → source_tag 字符串（带方括号，如 "[A]"）
_SOURCE_TAG_MAP: dict[EvidenceSource, str] = {
    EvidenceSource.PUBMED: "[A]",
    EvidenceSource.BIOMODELS: "[B]",
    EvidenceSource.SIMULATION: "[C]",
    EvidenceSource.INFERENCE: "[D]",
    EvidenceSource.HYPOTHESIS: "[E]",
}


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
        source_role: 来源角色字符串（"PubMed"/"BioModels"/"Simulation"/
            "Mechanism"/"Hypothesis"），用于 RAG 输出标注与 EvidenceFuser
            自动归类证据源。未显式传入时由 `source` 派生。
        source_tag: 来源标签字符串（"[A]"/"[B]"/"[C]"/"[D]"/"[E]"），
            用于 Discussion 渲染单源标签。未显式传入时由 `source` 派生。
    """

    source: EvidenceSource
    reference: str
    snippet: str = ""
    confidence: float = 1.0
    source_role: str = ""
    source_tag: str = ""

    def __post_init__(self) -> None:
        """未显式传入 source_role / source_tag 时，从 source 派生默认值。"""
        if not self.source_role:
            self.source_role = _SOURCE_ROLE_MAP.get(self.source, "")
        if not self.source_tag:
            self.source_tag = _SOURCE_TAG_MAP.get(self.source, "")


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


# =============================================================================
# 缺口 1+2：source_role 推断 + [A]-[E] 多源结构化转换器
# =============================================================================
def _infer_source_role(evidence: dict) -> tuple[str, EvidenceSource]:
    """从证据 dict 内容推断 source_role 与 EvidenceSource 枚举。

    推断规则（按优先级，匹配即返回）：
      1. source 含 "PubMed" 或 dict 含 `pmid`        → ("PubMed",     PUBMED)      [A]
      2. source 含 "BioModels" 或 dict 含 `biomd_id` → ("BioModels",  BIOMODELS)   [B]
      3. source 含 "simulation" 或 dict 含 `sim_id`  → ("Simulation", SIMULATION)  [C]
      4. source/type 含 "canonical"/"mechanism"，
         或 dict 含 `pathway` 字段                   → ("Mechanism",  INFERENCE)   [D]
      5. 其他                                        → ("Hypothesis", HYPOTHESIS)  [E]

    Args:
        evidence: 证据 dict（可能含 source / pmid / biomd_id / sim_id /
            pathway / type 等键）。

    Returns:
        (source_role, source_enum) 元组。
    """
    source_str = str(evidence.get("source", "") or "").lower()
    type_str = str(evidence.get("type", "") or "").lower()

    # [A] PubMed
    if "pubmed" in source_str or evidence.get("pmid"):
        return ("PubMed", EvidenceSource.PUBMED)
    # [B] BioModels
    if "biomodels" in source_str or evidence.get("biomd_id"):
        return ("BioModels", EvidenceSource.BIOMODELS)
    # [C] Simulation
    if "simulation" in source_str or evidence.get("sim_id"):
        return ("Simulation", EvidenceSource.SIMULATION)
    # [D] Mechanism（canonical / mechanism 关键词，或含 pathway 字段）
    if (
        "canonical" in source_str
        or "mechanism" in source_str
        or "canonical" in type_str
        or "mechanism" in type_str
        or evidence.get("pathway")
    ):
        return ("Mechanism", EvidenceSource.INFERENCE)
    # [E] Hypothesis 兜底
    return ("Hypothesis", EvidenceSource.HYPOTHESIS)


def evidence_to_item(
    evidence: dict,
    default_role: str | None = None,
) -> EvidenceItem:
    """统一接口：将五源证据 dict 转换为 EvidenceItem。

    支持的结构化输入：
      - [A] PubMed:     {source, pmid, parameter, value, ...}
      - [B] BioModels:  {biomd_id, reaction, parameter, value, unit}
      - [C] Simulation: {sim_id, metric, value, expected, fold_change}
      - [D] Mechanism:  {pathway, reaction, type}
      - [E] Hypothesis: {text}

    推断逻辑：
      1. 若 `default_role` 显式指定（"PubMed"/"BioModels"/"Simulation"/
         "Mechanism"/"Hypothesis"），优先采用；
      2. 否则按 `_infer_source_role` 从 dict 内容推断。

    输出 EvidenceItem 字段映射：
      - source_role / source_tag: 由推断结果填充
      - reference:                各源的引用标识（PMID / BIOMD:{id} / sim:{id} /
                                  canonical:{pathway} / ""）
      - snippet:                  各源的可读文本（PubMed 用 title/parameter 组装）
      - confidence:               各源默认置信度（[D]=1.0, [E]=0.5, 其余 0.7-0.8）

    Args:
        evidence: 证据 dict。
        default_role: 可选，显式指定 source_role（覆盖自动推断）。

    Returns:
        EvidenceItem 实例（含 source_role / source_tag 字段）。
    """
    if not isinstance(evidence, dict):
        raise TypeError(f"evidence 必须为 dict，收到 {type(evidence).__name__}")

    # 1) 确定 source_role 与 EvidenceSource
    role_to_enum: dict[str, EvidenceSource] = {
        "PubMed": EvidenceSource.PUBMED,
        "BioModels": EvidenceSource.BIOMODELS,
        "Simulation": EvidenceSource.SIMULATION,
        "Mechanism": EvidenceSource.INFERENCE,
        "Hypothesis": EvidenceSource.HYPOTHESIS,
    }
    if default_role and default_role in role_to_enum:
        role = default_role
        source_enum = role_to_enum[default_role]
    else:
        role, source_enum = _infer_source_role(evidence)

    # 2) 按源构造 reference / snippet / confidence
    if source_enum == EvidenceSource.PUBMED:
        # [A] PubMed
        pmid = str(evidence.get("pmid", "") or "").strip()
        reference = f"PMID:{pmid}" if pmid else ""
        title = str(evidence.get("title", "") or evidence.get("text", "") or "").strip()
        parameter = evidence.get("parameter", "")
        value = evidence.get("value", "")
        if title:
            snippet = title
        elif parameter and value:
            snippet = f"{parameter}={value}"
        else:
            snippet = str(evidence.get("text", "") or "").strip()
        confidence = float(evidence.get("confidence", 0.8) or 0.8)

    elif source_enum == EvidenceSource.BIOMODELS:
        # [B] BioModels
        biomd_id = str(evidence.get("biomd_id", "") or "").strip()
        reference = f"BIOMD:{biomd_id}" if biomd_id else ""
        reaction = evidence.get("reaction", "")
        parameter = evidence.get("parameter", "")
        value = evidence.get("value", "")
        unit = evidence.get("unit", "")
        parts: list[str] = []
        if biomd_id:
            parts.append(f"BioModels {biomd_id}")
        if reaction:
            parts.append(f"报告 {reaction}")
        if parameter and value:
            parts.append(f"{parameter}={value}{unit}")
        snippet = " ".join(parts) if parts else str(evidence.get("text", "") or "").strip()
        confidence = float(evidence.get("confidence", 0.75) or 0.75)

    elif source_enum == EvidenceSource.SIMULATION:
        # [C] Simulation
        sim_id = str(evidence.get("sim_id", "") or "").strip()
        reference = f"sim:{sim_id}" if sim_id else ""
        metric = evidence.get("metric", "")
        value = evidence.get("value", "")
        expected = evidence.get("expected", "")
        fold_change = evidence.get("fold_change", "")
        parts_sim: list[str] = []
        if metric:
            parts_sim.append(f"仿真结果显示 {metric}={value}")
            if expected:
                parts_sim.append(f"(期望 {expected}")
                if fold_change:
                    parts_sim.append(f", fold {fold_change}")
                parts_sim.append(")")
        snippet = "".join(parts_sim) if parts_sim else str(evidence.get("text", "") or "").strip()
        confidence = float(evidence.get("confidence", 0.7) or 0.7)

    elif source_enum == EvidenceSource.INFERENCE:
        # [D] Mechanism（canonical pathway 引用）
        pathway = str(evidence.get("pathway", "") or "").strip()
        reference = f"canonical:{pathway}" if pathway else ""
        reaction = evidence.get("reaction", "")
        mtype = evidence.get("type", "")
        parts_m: list[str] = []
        if pathway:
            parts_m.append(f"Canonical {pathway} 通路")
        if reaction:
            parts_m.append(str(reaction))
        if mtype:
            parts_m.append(f"({mtype})")
        snippet = " ".join(parts_m) if parts_m else str(evidence.get("text", "") or "").strip()
        confidence = float(evidence.get("confidence", 1.0) or 1.0)

    else:
        # [E] Hypothesis（无引用支撑的假设）
        reference = ""
        snippet = str(evidence.get("text", "") or "").strip()
        confidence = float(evidence.get("confidence", 0.5) or 0.5)

    # 3) 构造 EvidenceItem（source_role / source_tag 由 __post_init__ 自动派生）
    return EvidenceItem(
        source=source_enum,
        reference=reference,
        snippet=snippet,
        confidence=confidence,
        source_role=role,
        source_tag=_SOURCE_TAG_MAP.get(source_enum, ""),
    )


def evidence_docs_to_items_multi_source(
    evidence_docs: list,
    default_role: str | None = None,
) -> list[EvidenceItem]:
    """从多源证据 dict 列表构造 EvidenceItem 列表（支持 [A]-[E] 五源）。

    与 `evidence_docs_to_items`（仅处理 [A] PubMed EvidenceDoc 对象）互补，
    本函数接受 list[dict] 输入，按 dict 内容自动归类证据源，调用
    `evidence_to_item` 转换。

    Args:
        evidence_docs: 证据 dict 列表（每个 dict 含源标识字段，详见
            `evidence_to_item` 文档）。
        default_role: 可选，显式覆盖所有项的 source_role。

    Returns:
        EvidenceItem 列表（每项含 source_role / source_tag）。
    """
    if not evidence_docs:
        return []

    items: list[EvidenceItem] = []
    for doc in evidence_docs:
        if isinstance(doc, dict):
            items.append(evidence_to_item(doc, default_role=default_role))
        else:
            # 非 dict 输入（如 EvidenceDoc 对象）兜底走旧 [A] PubMed 路径
            items.extend(evidence_docs_to_items([doc]))
    return items


__all__ = [
    "EvidenceSource",
    "EvidenceItem",
    "FusedAssertion",
    "EvidenceFusionReport",
    "fuse_evidence",
    "evidence_docs_to_items",
    "evidence_docs_to_items_multi_source",
    "evidence_to_item",
]
