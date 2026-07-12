# BioDynamics Agent v4 - Scientific Alignment Loop: RAG Evidence Boost (Task 4)
#
# 将 Evidence Ranking Score 接入 RAG Reranker，对 RAG 检索结果按
# Evidence Ranking Score + 语义分 + BM25 分融合排序。
#
# 设计目标：约束 RAG Reranker 不能让随机近期 Application Paper 排在经典 Review
# 之前。通过 EvidenceRanker 提供的 per-doc Evidence Ranking Score 计算 boost
# 系数，与既有语义/BM25 分数融合后重排。
#
# 依赖：app.config.settings（Feature Flag 校验）、
#       app.scientific_alignment.evidence_ranker（Evidence Ranking Score 来源）；
#       不引入新依赖。作为 rag_client.py 的配套文件，不修改 rag_client.py。
#
# 核心导出：
#   from app.scientific_alignment.rag_evidence_boost import (
#       RAGCandidate, BoostedCandidate, EvidenceBoostReport,
#       boost_rag_candidates, boost_from_dicts, apply_boost_to_rerank_results,
#   )

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.config import settings
from app.scientific_alignment.evidence_ranker import (
    EvidenceDoc,
    EvidenceRanker,
    EvidenceType,
)

logger = logging.getLogger(__name__)


# =============================================================================
# 常量：融合权重与归一化基数
# =============================================================================
_WEIGHT_SEMANTIC: float = 0.4      # 语义相似度权重
_WEIGHT_BM25: float = 0.3          # BM25 分数权重
_WEIGHT_BOOST: float = 0.3         # Evidence boost 权重
_MAX_EVIDENCE_SCORE: float = 5.0   # Evidence Ranking Score 上限（boost_factor 归一化基数）

# 字符串 → EvidenceType 映射（兼容多种命名风格，统一小写匹配）
_EVIDENCE_TYPE_MAP: dict[str, EvidenceType] = {
    "review": EvidenceType.REVIEW,
    "mechanism": EvidenceType.MECHANISM_PAPER,
    "mechanism_paper": EvidenceType.MECHANISM_PAPER,
    "biomodels_source": EvidenceType.BIOMODELS_SOURCE,
    "biomodels": EvidenceType.BIOMODELS_SOURCE,
    "biomodels source": EvidenceType.BIOMODELS_SOURCE,
    "application": EvidenceType.RECENT_APPLICATION,
    "recent_application": EvidenceType.RECENT_APPLICATION,
    "case_report": EvidenceType.CASE_REPORT,
}


# =============================================================================
# 数据结构
# =============================================================================
@dataclass
class RAGCandidate:
    """RAG 检索候选文档。"""

    doc_id: str               # PMID / BIOMD ID / 内部 ID
    title: str = ""
    abstract: str = ""
    semantic_score: float = 0.0   # 语义相似度 (0-1)
    bm25_score: float = 0.0       # BM25 分数（归一化到 0-1）
    evidence_type: str = "unknown"  # Review/Mechanism/BioModels_Source/Application/Case_Report/unknown
    pmid: str = ""
    source: str = ""              # pubmed / biomodels / internal


@dataclass
class BoostedCandidate:
    """boost 后的候选文档，携带 Evidence Ranking 中间态与融合分数。"""

    candidate: RAGCandidate
    evidence_rank_score: float    # Evidence Ranking Score (1-5)，unknown=0
    boost_factor: float           # boost 系数 (evidence_rank_score / 5)
    fused_score: float            # 融合后分数
    original_rank: int            # 原始排名（1-based）
    boosted_rank: int             # boost 后排名（1-based）


@dataclass
class EvidenceBoostReport:
    """Evidence Boost 报告。"""

    enabled: bool
    skipped: bool = False
    candidates: list[BoostedCandidate] = field(default_factory=list)
    original_order: list[str] = field(default_factory=list)  # 原始 doc_id 顺序
    boosted_order: list[str] = field(default_factory=list)   # boost 后顺序
    reranked: bool = False        # 排序是否变化
    review_on_top: bool = False   # Review 是否排在 Application 前


# =============================================================================
# 内部辅助函数
# =============================================================================
def _resolve_evidence_type(evidence_type_str: str) -> EvidenceType | None:
    """将字符串 evidence_type 解析为 EvidenceType 枚举。

    无法识别时返回 None（按 unknown/0 处理）。

    Args:
        evidence_type_str: 证据类型字符串（如 "Review"/"Mechanism"/"Application"）。

    Returns:
        匹配到的 EvidenceType 枚举值；无法识别时返回 None。
    """
    if not evidence_type_str:
        return None
    key = str(evidence_type_str).strip().lower()
    return _EVIDENCE_TYPE_MAP.get(key)


def _compute_evidence_rank_score(
    ranker: EvidenceRanker,
    candidate: RAGCandidate,
) -> float:
    """通过 EvidenceRanker 计算 per-doc Evidence Ranking Score。

    读取 EvidenceRanker 的中间态（score），不修改检索逻辑。
    unknown 类型返回 0.0。

    Args:
        ranker: EvidenceRanker 实例。
        candidate: RAG 检索候选文档。

    Returns:
        Evidence Ranking Score（float）。Review=5, Mechanism=4,
        BioModels_Source=3, Application=2, Case_Report=1, unknown=0。
    """
    etype = _resolve_evidence_type(candidate.evidence_type)
    if etype is None:
        # evidence_type 不识别时按 unknown(0) 处理
        return 0.0
    doc = EvidenceDoc(
        pmid=candidate.pmid or candidate.doc_id,
        title=candidate.title,
        year=0,
        evidence_type=etype,
    )
    return float(ranker.score(doc))


def _is_review_type(etype_lower: str) -> bool:
    """判断归一化后的 evidence_type 字符串是否为 Review。"""
    return etype_lower == "review"


def _is_application_type(etype_lower: str) -> bool:
    """判断归一化后的 evidence_type 字符串是否为 Application。"""
    return etype_lower in ("application", "recent_application")


def _detect_review_on_top(boosted_list: list[BoostedCandidate]) -> bool:
    """检测 Review 是否排在 Application 前。

    遍历 boosted_order，找到第一个 Review 类型和第一个 Application 类型的位置。
    若 Review 在 Application 前 → True；二者任一缺失或顺序相反 → False。

    Args:
        boosted_list: 已按 fused_score 降序排列的 BoostedCandidate 列表。

    Returns:
        Review 排在 Application 前返回 True，否则 False。
    """
    review_pos: int | None = None
    application_pos: int | None = None
    for idx, b in enumerate(boosted_list):
        etype_str = (
            b.candidate.evidence_type.strip().lower()
            if b.candidate.evidence_type
            else ""
        )
        if review_pos is None and _is_review_type(etype_str):
            review_pos = idx
        if application_pos is None and _is_application_type(etype_str):
            application_pos = idx
        if review_pos is not None and application_pos is not None:
            break
    if review_pos is None or application_pos is None:
        return False
    return review_pos < application_pos


def _build_skipped_report(
    candidates: list[RAGCandidate],
) -> EvidenceBoostReport:
    """Flag OFF 时构建 skipped 报告，candidates 保持原序。"""
    original_order = [c.doc_id for c in candidates]
    return EvidenceBoostReport(
        enabled=False,
        skipped=True,
        candidates=[
            BoostedCandidate(
                candidate=c,
                evidence_rank_score=0.0,
                boost_factor=0.0,
                fused_score=0.0,
                original_rank=idx + 1,
                boosted_rank=idx + 1,
            )
            for idx, c in enumerate(candidates)
        ],
        original_order=original_order,
        boosted_order=list(original_order),
        reranked=False,
        review_on_top=False,
    )


# =============================================================================
# 主函数
# =============================================================================
def boost_rag_candidates(
    candidates: list[RAGCandidate],
    pathway: str = "",
) -> EvidenceBoostReport:
    """对 RAG 检索候选按 Evidence Ranking Score boost 重排。

    融合公式::

        evidence_rank_score = EvidenceRanker.score(evidence_type)
        boost_factor = evidence_rank_score / 5.0
        fused_score = semantic_score * 0.4 + bm25_score * 0.3 + boost_factor * 0.3

    按 fused_score 降序排列。

    Args:
        candidates: RAG 检索候选文档列表。
        pathway: 通路标识（保留参数，供未来 Gold Standard 集成使用）。

    Returns:
        EvidenceBoostReport。Flag OFF 时 skipped=True，candidates 保持原序；
        candidates 为空时返回空报告。
    """
    # Feature Flag 校验：SA 总开关或 EVIDENCE_FUSION 子开关任一关闭即跳过
    if not settings.is_sa_feature_enabled("EVIDENCE_FUSION"):
        return _build_skipped_report(candidates)

    # candidates 为空时返回空报告（不算 skipped）
    if not candidates:
        return EvidenceBoostReport(
            enabled=True,
            skipped=False,
            candidates=[],
            original_order=[],
            boosted_order=[],
            reranked=False,
            review_on_top=False,
        )

    # 构建 EvidenceRanker（无 gold_standard，纯评分模式；evidence_type 已由调用方提供）
    ranker = EvidenceRanker(gold_standard=None)

    original_order = [c.doc_id for c in candidates]

    # SubTask 4.1/4.2：计算 per-doc evidence_rank_score（中间态）与 fused_score
    boosted_list: list[BoostedCandidate] = []
    for idx, c in enumerate(candidates):
        evidence_rank_score = _compute_evidence_rank_score(ranker, c)
        boost_factor = evidence_rank_score / _MAX_EVIDENCE_SCORE
        fused_score = (
            c.semantic_score * _WEIGHT_SEMANTIC
            + c.bm25_score * _WEIGHT_BM25
            + boost_factor * _WEIGHT_BOOST
        )
        boosted_list.append(
            BoostedCandidate(
                candidate=c,
                evidence_rank_score=evidence_rank_score,
                boost_factor=boost_factor,
                fused_score=fused_score,
                original_rank=idx + 1,
                boosted_rank=0,  # 占位，排序后填入
            )
        )

    # SubTask 4.1：按 fused_score 降序排列
    boosted_list.sort(key=lambda b: b.fused_score, reverse=True)

    # 填入 boosted_rank
    for new_idx, b in enumerate(boosted_list):
        b.boosted_rank = new_idx + 1

    boosted_order = [b.candidate.doc_id for b in boosted_list]
    reranked = boosted_order != original_order
    review_on_top = _detect_review_on_top(boosted_list)

    return EvidenceBoostReport(
        enabled=True,
        skipped=False,
        candidates=boosted_list,
        original_order=original_order,
        boosted_order=boosted_order,
        reranked=reranked,
        review_on_top=review_on_top,
    )


def boost_from_dicts(
    docs: list[dict],
    pathway: str = "",
) -> EvidenceBoostReport:
    """便捷函数：从 dict 列表构造 RAGCandidate 并 boost。

    dict 字段: doc_id / title / abstract / semantic_score / bm25_score /
    evidence_type / pmid / source（缺失字段使用默认值）。

    Args:
        docs: 候选文档 dict 列表。
        pathway: 通路标识（透传给 boost_rag_candidates）。

    Returns:
        EvidenceBoostReport。Flag OFF 时 skipped=True。
    """
    candidates = [
        RAGCandidate(
            doc_id=str(d.get("doc_id", "")),
            title=str(d.get("title", "")),
            abstract=str(d.get("abstract", "")),
            semantic_score=float(d.get("semantic_score", 0.0) or 0.0),
            bm25_score=float(d.get("bm25_score", 0.0) or 0.0),
            evidence_type=str(d.get("evidence_type", "unknown") or "unknown"),
            pmid=str(d.get("pmid", "")),
            source=str(d.get("source", "")),
        )
        for d in docs
    ]
    return boost_rag_candidates(candidates, pathway=pathway)


def apply_boost_to_rerank_results(
    rerank_results: list[dict],
    pathway: str = "",
) -> list[dict]:
    """将 boost 应用到既有 rerank 结果列表。

    输入: [{"doc_id":..., "score":..., "title":..., ...}, ...]
    输出: 重排后的同结构列表，追加 evidence_rank_score / fused_score 字段。
    输入 dict 的 "score" 字段作为 semantic_score 参与融合；
    bm25_score / evidence_type 字段缺失时分别按 0.0 / "unknown" 处理。

    Args:
        rerank_results: 既有 rerank 结果 dict 列表。
        pathway: 通路标识（透传给 boost_rag_candidates）。

    Returns:
        重排后的 dict 列表（浅拷贝，追加 evidence_rank_score / fused_score 字段）。
        Flag OFF 时原样返回（浅拷贝新列表）。
    """
    # Feature Flag 校验：OFF 时原样返回
    if not settings.is_sa_feature_enabled("EVIDENCE_FUSION"):
        return list(rerank_results)

    # 构造 RAGCandidate 列表：rerank "score" → semantic_score
    candidates = [
        RAGCandidate(
            doc_id=str(d.get("doc_id", "")),
            title=str(d.get("title", "")),
            abstract=str(d.get("abstract", "")),
            semantic_score=float(d.get("score", 0.0) or 0.0),
            bm25_score=float(d.get("bm25_score", 0.0) or 0.0),
            evidence_type=str(d.get("evidence_type", "unknown") or "unknown"),
            pmid=str(d.get("pmid", "")),
            source=str(d.get("source", "")),
        )
        for d in rerank_results
    ]

    report = boost_rag_candidates(candidates, pathway=pathway)

    # 构建 doc_id → 原始 dict 映射（保留首个匹配，应对 doc_id 重复）
    id_to_doc: dict[str, dict] = {}
    for d in rerank_results:
        did = str(d.get("doc_id", ""))
        if did not in id_to_doc:
            id_to_doc[did] = d

    # 构建 doc_id → BoostedCandidate 映射
    id_to_boost: dict[str, BoostedCandidate] = {
        b.candidate.doc_id: b for b in report.candidates
    }

    # 按 boosted_order 重排原始 dict，并追加 evidence_rank_score / fused_score 字段
    result: list[dict] = []
    for doc_id in report.boosted_order:
        original = id_to_doc.get(doc_id)
        if original is None:
            continue
        new_doc = dict(original)
        boost_info = id_to_boost.get(doc_id)
        if boost_info is not None:
            new_doc["evidence_rank_score"] = boost_info.evidence_rank_score
            new_doc["fused_score"] = boost_info.fused_score
        result.append(new_doc)
    return result


__all__ = [
    "RAGCandidate",
    "BoostedCandidate",
    "EvidenceBoostReport",
    "boost_rag_candidates",
    "boost_from_dicts",
    "apply_boost_to_rerank_results",
]
