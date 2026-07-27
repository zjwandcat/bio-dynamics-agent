# BioDynamics Agent v4 - Scientific Alignment: Canonical Literature Ranking (Task E)
#
# 用户核心诉求：
#   RAG 排序不是 BM25，而是 Canonical + Embedding + BM25 三重融合。
#   提前维护每通路 Canonical PMID 星级表（★★★★★），
#   确保 Discussion 不会引用奇怪论文。
#
# 设计：
#   1. 从 knowledge/gold_standard/literature_<pathway>.yaml 加载 Canonical PMID → stars 映射
#   2. 对 RAG 检索结果计算三重融合分数：
#        fused = canonical_score * W_canonical + embedding_score * W_embed + bm25_score * W_bm25
#   3. canonical_score = stars / 5.0（0.0-1.0）；非 Canonical 文献 canonical_score = 0.0
#   4. 按 fused 降序重排
#
# 核心导出：
#   from app.scientific_alignment.canonical_ranker import (
#       CanonicalRanker, CanonicalMatch, rerank_evidence_with_canonical,
#   )

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


# =============================================================================
# 常量
# =============================================================================
# gold_standard 根目录（与 evidence_ranker.py 对齐）
_GOLD_STANDARD_DIR: Path = (
    Path(__file__).resolve().parent.parent.parent / "knowledge" / "gold_standard"
)

# pathway 白名单正则（防路径遍历）
_PATHWAY_PATTERN: re.Pattern[str] = re.compile(r"^[a-zA-Z0-9_]+$")

# 默认三重融合权重（可被 YAML ranking_policy 覆盖）
_DEFAULT_W_CANONICAL: float = 0.4
_DEFAULT_W_EMBEDDING: float = 0.35
_DEFAULT_W_BM25: float = 0.25

# pathway 别名映射（v4_pathway_class / mechanism.pathway → gold_standard 文件名键）
_PATHWAY_ALIASES: dict[str, str] = {
    # v4_pathway_class 大写 → gold_standard 小写
    "EGFR_RTK": "egfr",
    "EGFR": "egfr",
    "MAPK_ERK": "mapk",
    "MAPK": "mapk",
    "PI3K_AKT_MTOR": "pi3k_akt_mtor",
    "TGF_BETA": "tgf_beta",
    "JAK_STAT": "jak_stat",
    "NFKB": "nf_kappa_b",
    "NF_KAPPA_B": "nf_kappa_b",
    "P53": "p53",
    "APOPTOSIS": "apoptosis",
    "CELL_CYCLE": "cell_cycle",
    "WNT": "wnt",
    "CROSS_PATHWAY": "cross_pathway",
    "CROSSPATHWAY": "cross_pathway",
    # MULTI 通路取主通路
}


def _normalize_pathway(pathway: str) -> str:
    """将 v4_pathway_class / mechanism.pathway 归一化为 gold_standard 文件名键。

    处理：
      - MULTI:EGFR_RTK+MAPK_ERK → egfr（取首个子通路，作为主通路键）
      - EGFR_RTK → egfr
      - EGFR → egfr
      - pi3k_akt_mtor → pi3k_akt_mtor（已小写直接通过）

    注意：MULTI 通路的主键仅取首个子通路用于 backward-compat，
    但 _expand_pathway_keys 会返回全部子通路 + cross_pathway。
    """
    if not pathway:
        return ""
    p = pathway.strip()

    # MULTI 通路：取首个子通路
    if p.startswith("MULTI:"):
        parts = p[6:].split("+")
        if parts:
            p = parts[0].strip()

    # 查别名表
    p_upper = p.upper()
    if p_upper in _PATHWAY_ALIASES:
        return _PATHWAY_ALIASES[p_upper]
    if p in _PATHWAY_ALIASES:
        return _PATHWAY_ALIASES[p]

    # 直接小写
    return p.lower()


def _expand_pathway_keys(pathway: str) -> list[str]:
    """展开通路标识为需要加载的 YAML 文件名键列表。

    用于 MULTI 通路场景，确保跨通路文献也被加载：
      - MULTI:EGFR_RTK+MAPK_ERK → ["egfr", "mapk", "cross_pathway"]
      - MULTI:EGFR_RTK+PI3K_AKT_MTOR → ["egfr", "pi3k_akt_mtor", "cross_pathway"]
      - CROSS_PATHWAY → ["cross_pathway"]
      - EGFR_RTK → ["egfr"]

    Returns:
        去重后的 pathway 键列表（已归一化）；主通路在最前，cross_pathway 在最后。
    """
    if not pathway:
        return []
    p = pathway.strip()

    keys: list[str] = []
    raw_subpathways: list[str] = []

    # MULTI 通路：展开所有子通路
    if p.startswith("MULTI:"):
        raw_subpathways = [s.strip() for s in p[6:].split("+") if s.strip()]
    else:
        raw_subpathways = [p]

    # 归一化每个子通路
    for sub in raw_subpathways:
        sub_upper = sub.upper()
        if sub_upper in _PATHWAY_ALIASES:
            keys.append(_PATHWAY_ALIASES[sub_upper])
        elif sub in _PATHWAY_ALIASES:
            keys.append(_PATHWAY_ALIASES[sub])
        else:
            keys.append(sub.lower())

    # MULTI 通路自动追加 cross_pathway（跨通路文献库）
    # 单通路场景不追加，避免污染单一通路检索结果
    if p.startswith("MULTI:") and "cross_pathway" not in keys:
        keys.append("cross_pathway")

    # 去重保序
    seen: set[str] = set()
    unique_keys: list[str] = []
    for k in keys:
        if k and k not in seen:
            seen.add(k)
            unique_keys.append(k)
    return unique_keys


def _normalize_pmid(pmid: Any) -> str:
    """归一化 PMID：去除 'PMID:' 前缀，strip。"""
    if not pmid:
        return ""
    return str(pmid).replace("PMID:", "").replace("pmid:", "").strip()


# =============================================================================
# 数据结构
# =============================================================================
@dataclass
class CanonicalMatch:
    """单条文献的 Canonical 匹配结果。"""

    pmid: str
    stars: int               # 1-5 星级；0 表示非 Canonical
    title: str = ""
    evidence_type: str = ""  # review / mechanism / biomodels_source
    canonical_score: float = 0.0  # stars / 5.0
    reason: str = ""  # YAML 中的 reason 字段：解释为何该文献被选为 Canonical


@dataclass
class CanonicalRankResult:
    """Canonical 排序结果。"""

    original_rank: int
    boosted_rank: int
    pmid: str
    canonical_score: float    # 0.0-1.0
    embedding_score: float    # 0.0-1.0
    bm25_score: float         # 0.0-1.0
    fused_score: float        # 三重融合分数
    stars: int                # 原始星级
    title: str
    is_canonical: bool        # 是否在 Canonical 列表中


# =============================================================================
# CanonicalRanker 排序器
# =============================================================================
class CanonicalRanker:
    """Canonical Literature Ranking 排序器。

    从 gold_standard YAML 加载 PMID → stars 映射，
    对 RAG 检索结果按 Canonical + Embedding + BM25 三重融合排序。

    Usage::

        ranker = CanonicalRanker("egfr")
        results = ranker.rerank(
            docs=[{"pmid": "11743495", "title": "...", "_semantic_score": 0.8, ...}, ...],
            query="EGFR signaling",
        )
    """

    def __init__(
        self,
        pathway: str,
        w_canonical: float = _DEFAULT_W_CANONICAL,
        w_embedding: float = _DEFAULT_W_EMBEDDING,
        w_bm25: float = _DEFAULT_W_BM25,
    ) -> None:
        """初始化排序器。

        Args:
            pathway: 通路标识（v4_pathway_class 或 mechanism.pathway）。
                     支持 MULTI:A+B 形式，会加载所有子通路 + cross_pathway YAML。
            w_canonical: Canonical stars 融合权重。
            w_embedding: Embedding 语义相似度权重。
            w_bm25: BM25 关键词匹配权重。
        """
        self._raw_pathway = pathway or ""
        self._pathway_key = _normalize_pathway(pathway)
        # 展开为所有需加载的 YAML 键（MULTI 场景包含 cross_pathway）
        self._pathway_keys = _expand_pathway_keys(pathway)
        self._w_canonical = w_canonical
        self._w_embedding = w_embedding
        self._w_bm25 = w_bm25

        # PMID → CanonicalMatch 索引
        self._canonical_index: dict[str, CanonicalMatch] = {}

        # 从 YAML 加载（可能加载多个文件）
        self._loaded = False
        self._load_gold_standard()

    @property
    def pathway_key(self) -> str:
        return self._pathway_key

    @property
    def pathway_keys(self) -> list[str]:
        """返回所有已加载 YAML 的通路键列表（MULTI 场景含多个）。"""
        return list(self._pathway_keys)

    @property
    def loaded(self) -> bool:
        """是否成功加载了 Canonical 列表。"""
        return self._loaded

    @property
    def canonical_count(self) -> int:
        """Canonical PMID 数量。"""
        return len(self._canonical_index)

    # -------------------------------------------------------------------------
    # YAML 加载
    # -------------------------------------------------------------------------
    def _load_gold_standard(self) -> None:
        """从 knowledge/gold_standard/literature_<pathway>.yaml 加载 Canonical PMID。

        当 pathway 为 MULTI:A+B 形式时，会依次加载所有子通路 YAML + cross_pathway YAML。
        多文件中同一 PMID 出现时，保留更高 stars 的记录。
        """
        if not self._pathway_keys:
            return

        # 标记 ranking_policy 是否已应用（仅首个 YAML 生效）
        self._policy_applied = False

        # 依次加载每个 pathway_key 对应的 YAML
        for pathway_key in self._pathway_keys:
            self._load_single_yaml(pathway_key)

        self._loaded = len(self._canonical_index) > 0
        if self._loaded:
            logger.info(
                "CanonicalRanker[%s]: 加载 %d 条 Canonical PMID（来自 %d 个 YAML: %s）",
                self._pathway_key,
                len(self._canonical_index),
                len(self._pathway_keys),
                ", ".join(self._pathway_keys),
            )

    def _load_single_yaml(self, pathway_key: str) -> None:
        """加载单个 pathway YAML 文件到 _canonical_index。"""
        if not pathway_key:
            return

        if not _PATHWAY_PATTERN.match(pathway_key):
            logger.warning(
                "CanonicalRanker: pathway %r 含非法字符，跳过加载",
                pathway_key,
            )
            return

        yaml_path = _GOLD_STANDARD_DIR / f"literature_{pathway_key}.yaml"
        if not yaml_path.exists():
            logger.debug(
                "CanonicalRanker: 文件不存在 %s，跳过",
                yaml_path.name,
            )
            return

        try:
            with yaml_path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
        except Exception as exc:
            logger.warning("CanonicalRanker: YAML 解析失败 %s: %s", yaml_path.name, exc)
            return

        if not isinstance(data, dict):
            return

        # 提取 ranking_policy 中的权重（仅首个 YAML 生效，避免被后续 YAML 覆盖）
        if not self._policy_applied:
            policy = data.get("ranking_policy", {}) or {}
            if isinstance(policy, dict):
                cw = policy.get("canonical_weight")
                ew = policy.get("embedding_weight")
                bw = policy.get("bm25_weight")
                if isinstance(cw, (int, float)) and cw > 0:
                    self._w_canonical = float(cw)
                if isinstance(ew, (int, float)) and ew > 0:
                    self._w_embedding = float(ew)
                if isinstance(bw, (int, float)) and bw > 0:
                    self._w_bm25 = float(bw)
            self._policy_applied = True

        # 按 evidence_type 分组提取 PMIDs + stars
        type_map = [
            ("classical_reviews", "review"),
            ("mechanism_papers", "mechanism"),
            ("biomodels_source_papers", "biomodels_source"),
            ("recent_applications", "recent_application"),
            ("case_reports", "case_report"),
        ]

        for yaml_key, etype in type_map:
            items = data.get(yaml_key, []) or []
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                pmid = _normalize_pmid(item.get("pmid", ""))
                if not pmid or pmid.upper().startswith("TODO") or pmid.upper() == "UNVERIFIED":
                    continue
                stars = int(item.get("stars", 0) or 0)
                # 同一 PMID 出现在多个分类/YAML：保留更高 stars
                existing = self._canonical_index.get(pmid)
                if existing and existing.stars >= stars:
                    continue
                title = str(item.get("title", ""))
                reason = str(item.get("reason", ""))
                self._canonical_index[pmid] = CanonicalMatch(
                    pmid=pmid,
                    stars=stars,
                    title=title,
                    evidence_type=etype,
                    canonical_score=stars / 5.0,
                    reason=reason,
                )

    # -------------------------------------------------------------------------
    # 查询
    # -------------------------------------------------------------------------
    def lookup(self, pmid: str) -> CanonicalMatch | None:
        """查询 PMID 是否在 Canonical 列表中。"""
        normalized = _normalize_pmid(pmid)
        return self._canonical_index.get(normalized)

    def canonical_score(self, pmid: str) -> float:
        """返回 PMID 的 canonical_score（0.0-1.0）。非 Canonical 返回 0.0。"""
        match = self.lookup(pmid)
        return match.canonical_score if match else 0.0

    def canonical_records(self) -> list[dict[str, Any]]:
        """Return locally governed canonical papers as structured evidence."""
        # 多 YAML 来源时，provenance 显示所有加载的 YAML 文件
        if len(self._pathway_keys) > 1:
            provenance_str = "knowledge/gold_standard/" + " | ".join(
                f"literature_{k}.yaml" for k in self._pathway_keys
            )
        else:
            provenance_str = f"knowledge/gold_standard/literature_{self._pathway_key}.yaml"
        return [
            {
                "pmid": match.pmid,
                "title": match.title,
                "source": f"PMID:{match.pmid}",
                "source_role": match.evidence_type,
                "snippet": match.title,
                "confidence": match.canonical_score,
                "stars": match.stars,
                "provenance": provenance_str,
                "retrieval_mode": "canonical_local",
                "reason": match.reason,
                "_semantic_score": 0.0,
            }
            for match in sorted(
                self._canonical_index.values(),
                key=lambda item: (-item.stars, item.pmid),
            )
        ]

    # -------------------------------------------------------------------------
    # 三重融合排序
    # -------------------------------------------------------------------------
    def rerank(
        self,
        docs: list[dict[str, Any]],
        query: str = "",
        bm25_scores: dict[str, float] | None = None,
    ) -> list[CanonicalRankResult]:
        """对 RAG 检索结果按 Canonical + Embedding + BM25 三重融合排序。

        Args:
            docs: RAG 检索结果 dict 列表。每条 dict 应包含：
                  - pmid / PMID（可选）
                  - title（可选）
                  - _semantic_score（Embedding 语义相似度，0-1）
                  - _document（文档文本，用于 BM25）
            query: 原始查询文本（用于 BM25 计算的 query）
            bm25_scores: 外部预计算的 BM25 分数（pmid → score，0-1）。
                         若为 None，则从 _document 字段在线计算。

        Returns:
            按 fused_score 降序排列的 CanonicalRankResult 列表。
        """
        if not docs:
            return []

        # 1. 提取每条文档的 pmid / embedding_score / bm25_score
        results: list[CanonicalRankResult] = []
        for idx, doc in enumerate(docs):
            pmid = _normalize_pmid(
                doc.get("pmid") or doc.get("PMID") or doc.get("_pmid") or ""
            )
            title = str(doc.get("title", "") or doc.get("_title", ""))

            # Embedding 语义分（ChromaDB _semantic_score = 1 - distance）
            emb_score = float(doc.get("_semantic_score", 0.0) or 0.0)
            # 归一化到 0-1
            emb_score = max(0.0, min(1.0, emb_score))

            # BM25 分数
            if bm25_scores and pmid in bm25_scores:
                bm25_s = float(bm25_scores[pmid])
            else:
                bm25_s = self._compute_bm25(doc, query)
            bm25_s = max(0.0, min(1.0, bm25_s))

            # Canonical 分数
            match = self.lookup(pmid)
            canon_score = match.canonical_score if match else 0.0
            stars = match.stars if match else 0

            # 三重融合
            fused = (
                canon_score * self._w_canonical
                + emb_score * self._w_embedding
                + bm25_s * self._w_bm25
            )

            results.append(CanonicalRankResult(
                original_rank=idx + 1,
                boosted_rank=0,  # 排序后填入
                pmid=pmid,
                canonical_score=canon_score,
                embedding_score=emb_score,
                bm25_score=bm25_s,
                fused_score=fused,
                stars=stars,
                title=title,
                is_canonical=match is not None,
            ))

        # 2. 按 fused_score 降序排列
        results.sort(key=lambda r: r.fused_score, reverse=True)

        # 3. 填入 boosted_rank
        for new_idx, r in enumerate(results):
            r.boosted_rank = new_idx + 1

        return results

    def _compute_bm25(self, doc: dict[str, Any], query: str) -> float:
        """简化 BM25 计算（单文档 vs query）。

        返回 0-1 归一化分数。query 词在文档中出现比例越高，分数越高。
        这不是完整 BM25，但足够用于 evidence 排序的相对比较。
        """
        if not query:
            return 0.0
        document = str(doc.get("_document", "") or doc.get("abstract", "") or doc.get("title", ""))
        if not document:
            return 0.0

        # 简单词频重叠率
        query_terms = set(
            t.lower() for t in re.findall(r"[A-Za-z0-9\-]+", query)
            if len(t) >= 2
        )
        if not query_terms:
            return 0.0

        doc_terms = set(
            t.lower() for t in re.findall(r"[A-Za-z0-9\-]+", document)
        )

        overlap = len(query_terms & doc_terms)
        return min(1.0, overlap / len(query_terms))

    # -------------------------------------------------------------------------
    # 排序报告
    # -------------------------------------------------------------------------
    def rerank_report(
        self,
        docs: list[dict[str, Any]],
        query: str = "",
    ) -> dict[str, Any]:
        """返回排序报告（含可解释性信息）。"""
        results = self.rerank(docs, query=query)
        return {
            "pathway": self._pathway_key,
            "canonical_count": self.canonical_count,
            "loaded": self._loaded,
            "weights": {
                "canonical": self._w_canonical,
                "embedding": self._w_embedding,
                "bm25": self._w_bm25,
            },
            "results": [
                {
                    "boosted_rank": r.boosted_rank,
                    "original_rank": r.original_rank,
                    "pmid": r.pmid,
                    "title": r.title,
                    "stars": r.stars,
                    "is_canonical": r.is_canonical,
                    "fused_score": round(r.fused_score, 4),
                    "canonical_score": round(r.canonical_score, 4),
                    "embedding_score": round(r.embedding_score, 4),
                    "bm25_score": round(r.bm25_score, 4),
                }
                for r in results
            ],
        }


# =============================================================================
# 便捷函数
# =============================================================================
def rerank_evidence_with_canonical(
    docs: list[dict[str, Any]],
    pathway: str,
    query: str = "",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """对 evidence RAG 检索结果应用 Canonical 三重排序。

    Args:
        docs: search_evidence 返回的 dict 列表。
        pathway: 通路标识（v4_pathway_class 或 mechanism.pathway）。
        query: 原始查询文本。

    Returns:
        (reranked_docs, report)
        - reranked_docs: 按 fused_score 降序排列的原始 dict 列表（浅拷贝，追加排序元数据）
        - report: 排序报告 dict
    """
    ranker = CanonicalRanker(pathway)
    results = ranker.rerank(docs, query=query)

    # 构建 pmid → original doc 映射
    pmid_to_doc: dict[str, dict[str, Any]] = {}
    for d in docs:
        pmid = _normalize_pmid(d.get("pmid") or d.get("PMID") or "")
        if pmid and pmid not in pmid_to_doc:
            pmid_to_doc[pmid] = d

    # 按 boosted_rank 顺序输出，追加排序元数据
    reranked: list[dict[str, Any]] = []
    for r in results:
        original = pmid_to_doc.get(r.pmid)
        if original is None:
            # pmid 为空或未匹配：用 original_rank 索引
            idx = r.original_rank - 1
            if 0 <= idx < len(docs):
                original = docs[idx]
            else:
                continue
        new_doc = dict(original)
        new_doc["_canonical_stars"] = r.stars
        new_doc["_canonical_score"] = round(r.canonical_score, 4)
        new_doc["_fused_score"] = round(r.fused_score, 4)
        new_doc["_is_canonical"] = r.is_canonical
        new_doc["_boosted_rank"] = r.boosted_rank
        reranked.append(new_doc)

    report = ranker.rerank_report(docs, query=query)
    return reranked, report


__all__ = [
    "CanonicalMatch",
    "CanonicalRankResult",
    "CanonicalRanker",
    "rerank_evidence_with_canonical",
]
