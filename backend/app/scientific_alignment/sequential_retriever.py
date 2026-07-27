# Feature Flag: V4_SEQUENTIAL_RETRIEVER
#
# BioDynamics Agent - Scientific Alignment Loop: Sequential Retriever（顺序优先级检索）
# （Spec: add-scientific-reviewer-and-validation-matrix，Task 6 实现）
#
# 模块用途：
#   将 RAG 检索从并行改为顺序优先级：canonical.yaml → BioModels → PubMed →
#   Reactome/KEGG。前一级命中足量证据时，后一级不执行（或仅补充）。禁止并行
#   检索导致随机新论文排在经典 Review 之前。
#
# 对应 Spec Requirement：
#   - Sequential Retriever（顺序优先级检索）
#
# 四级顺序检索流程：
#   1. canonical.yaml — 从 knowledge/canonical/<pathway>.yaml 加载 Canonical
#                       Reviews/Models/Mechanism（最高优先级）
#   2. BioModels      — 从 EBI BioModels 检索 Canonical BioModels
#                       （如 EGFR: BIOMD0000000010）
#   3. PubMed         — 仅当 canonical + BioModels 不足时，检索 PubMed
#                       （应用 Evidence Ranking Score：Review > Mechanism Paper >
#                       Application Paper）
#   4. Reactome/KEGG  — 仅当前三级仍缺时，补充通路拓扑
#
# 每级记录字段（写入 logs/sequential_retriever/<timestamp>.json）：
#   - stage         : canonical / biomodels / pubmed / reactome_kegg
#   - query         : 该级查询语句
#   - returned      : 该级返回结果数
#   - selected      : 该级选中结果数
#   - deduplicated  : 去重后剩余结果数
#
# 经典文献优先原则：
#   Schoeberl / Kholodenko / Murphy / Yarden 等 Canonical Reviews SHALL 排在
#   Top1-Top5；2025 SHP2 inhibitor 等近期 Application Paper SHALL NOT 排在
#   Canonical 之前。
#
# Feature Flag 守护：
#   V4_SEQUENTIAL_RETRIEVER 默认 true（独立于 V4_SCIENTIFIC_REVIEWER_ENABLED）。
#   RAG_LEGACY_PARALLEL=true 时回退到 rag_client.py 旧并行检索（Task 11 集成）。

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from app.scientific_alignment.canonical_loader import (
    CANONICAL_DIR,
    CanonicalReference,
    load_canonical,
)
from app.scientific_alignment.evidence_ranker import (
    EvidenceDoc,
    EvidenceRanker,
    EvidenceType,
)

logger = logging.getLogger(__name__)


# =============================================================================
# 常量定义（对应 Spec 顺序检索流程）
# =============================================================================

# 四级检索阶段名称（顺序固定，对应 Spec Scenario "顺序检索流程"）
RETRIEVAL_STAGES: tuple[str, ...] = (
    "canonical",        # 1. canonical.yaml（最高优先级）
    "biomodels",        # 2. BioModels
    "pubmed",           # 3. PubMed（仅当 canonical + BioModels 不足时）
    "reactome_kegg",    # 4. Reactome/KEGG（仅当前三级仍缺时）
)

# 来源优先级（数值越大优先级越高）
# 排序规则：canonical > biomodels > pubmed(review) > pubmed(mechanism_paper) >
#           pubmed(application_paper) > reactome/kegg
SOURCE_RANK: dict[str, int] = {
    "canonical": 5,
    "biomodels": 4,
    "pubmed": 3,
    "reactome": 2,
    "kegg": 2,
}

# 证据类型优先级（数值越大优先级越高）
# review > mechanism_paper > model/mechanism > application_paper > pathway
TYPE_RANK: dict[str, int] = {
    "review": 5,
    "mechanism_paper": 4,
    "model": 3,
    "mechanism": 3,
    "application_paper": 2,
    "pathway": 1,
}

# 足量判定阈值
# Rule 2: Stage 1+2 命中 ≥ 5 → 跳过 Stage 3-4
SUFFICIENCY_STAGE1_2_THRESHOLD: int = 5
# Rule 3: Stage 1+2+3 命中 ≥ 10 → 跳过 Stage 4
SUFFICIENCY_STAGE1_2_3_THRESHOLD: int = 10

# EvidenceRanker.EvidenceType → evidence_type 字符串映射
# BIOMODELS_SOURCE 归入 mechanism_paper（同为机制建模类）
# CASE_REPORT 归入 application_paper（同为非经典补充）
_ETYPE_TO_STR: dict[EvidenceType, str] = {
    EvidenceType.REVIEW: "review",
    EvidenceType.MECHANISM_PAPER: "mechanism_paper",
    EvidenceType.BIOMODELS_SOURCE: "mechanism_paper",
    EvidenceType.RECENT_APPLICATION: "application_paper",
    EvidenceType.CASE_REPORT: "application_paper",
}


# =============================================================================
# 数据结构
# =============================================================================


@dataclass
class Evidence:
    """单条证据。

    Attributes:
        source: 来源（canonical / biomodels / pubmed / reactome / kegg）
        title: 标题
        pmid: PubMed ID（可空），形如 "PMID:12451180" 或纯数字
        biomd_id: BioModels ID（可空），形如 "BIOMD0000000010"
        relevance_score: 相关性分数（canonical=1.0，PubMed 为 Evidence Ranking Score）
        evidence_type: 证据类型（review / mechanism_paper / application_paper /
            model / mechanism / pathway）
        raw: 原始数据 dict（便于上层扩展，不参与排序）
    """

    source: str
    title: str
    pmid: str | None = None
    biomd_id: str | None = None
    relevance_score: float = 0.0
    evidence_type: str = "application_paper"
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class StageLog:
    """单级检索记录（对应 Spec 每级 SHALL 记录的字段）。

    Attributes:
        stage: 阶段名称（canonical / biomodels / pubmed / reactome_kegg）
        query: 该级查询语句
        returned: 该级返回结果数
        selected: 该级选中结果数
        deduplicated: 去重后剩余结果数
    """

    stage: str
    query: str = ""
    returned: int = 0
    selected: int = 0
    deduplicated: int = 0


# 向后兼容别名（与 Task 1 骨架对齐，字段完全一致）
RetrievalStageRecord = StageLog


@dataclass
class RetrievalResult:
    """Sequential Retriever 完整检索结果。

    Attributes:
        staged_evidence: 各 stage 的 Evidence 列表（按 stage 名分组）
        final_ranked: 去重 + 排序后的最终 Evidence 列表（截断至 top_k）
        logs: 各 stage 的检索日志
        canonical_hit_rate: Top1-Top5 中 Canonical 比例（0.0-1.0）
    """

    staged_evidence: dict[str, list[Evidence]] = field(default_factory=dict)
    final_ranked: list[Evidence] = field(default_factory=list)
    logs: list[StageLog] = field(default_factory=list)
    canonical_hit_rate: float = 0.0


# =============================================================================
# SequentialRetriever：顺序优先级检索器
# =============================================================================


class SequentialRetriever:
    """顺序优先级检索器（canonical.yaml → BioModels → PubMed → Reactome/KEGG）。

    对应 Spec Requirement "Sequential Retriever（顺序优先级检索）"。

    设计原则：
        - 前一级命中足量证据时，后一级不执行（或仅补充）
        - PubMed 级应用 Evidence Ranking Score（Review > Mechanism Paper >
          Application Paper），保证经典 Review 优先
        - 每级记录 stage / query / returned / selected / deduplicated 到日志
        - 跨 stage 按 PMID/BIOMD/title 去重，保留高优先级 source 的版本
    """

    def __init__(
        self,
        canonical_dir: str | Path | None = None,
        rag_client: Any = None,
        biomodels_client: Any = None,
        *,
        log_dir: str | Path = "logs/sequential_retriever",
    ) -> None:
        """初始化 Sequential Retriever。

        Args:
            canonical_dir: Canonical YAML 目录。None 时使用 canonical_loader 默认目录。
            rag_client: PubMed 检索客户端，需提供
                ``search_pubmed(query, max_results=N)`` 方法，返回文章 dict 列表
                或 ``(articles, records, usage)`` 元组。
            biomodels_client: BioModels API 客户端，需提供
                ``download(model_id)`` 与 ``search(query, top_k=N)`` 方法。
            log_dir: 检索日志目录（默认 logs/sequential_retriever）。
        """
        self.canonical_dir: Path = (
            Path(canonical_dir) if canonical_dir else CANONICAL_DIR
        )
        self.rag_client: Any = rag_client
        self.biomodels_client: Any = biomodels_client
        self.log_dir: Path = Path(log_dir)

    # -------------------------------------------------------------------------
    # 主入口
    # -------------------------------------------------------------------------
    def retrieve(
        self,
        pathway: str,
        query: str,
        top_k: int = 10,
    ) -> RetrievalResult:
        """执行四级顺序优先级检索。

        Args:
            pathway: 通路名称（如 "egfr"），用于加载 canonical.yaml
            query: 检索查询语句（用于 PubMed stage）
            top_k: 返回 Top-K 结果（默认 10）

        Returns:
            RetrievalResult，含 staged_evidence / final_ranked / logs /
            canonical_hit_rate。
        """
        staged_evidence: dict[str, list[Evidence]] = {}
        logs: list[StageLog] = []
        # 跨 stage 累积的 Evidence（用于去重）
        accumulated: list[Evidence] = []

        # ===== Stage 1: canonical.yaml =====
        canonical_ev = self._stage_canonical(pathway)
        staged_evidence["canonical"] = list(canonical_ev)
        canonical_count = len(canonical_ev)
        # 加载 canonical 引用，获取 reviews/models 数量（用于足量判定）
        canonical_ref = self._safe_load_canonical(pathway)
        n_reviews = len(canonical_ref.canonical_reviews) if canonical_ref else 0
        n_models = len(canonical_ref.canonical_models) if canonical_ref else 0
        stage1_threshold = n_reviews + n_models
        accumulated.extend(canonical_ev)
        logs.append(StageLog(
            stage="canonical",
            query=f"pathway={pathway}",
            returned=canonical_count,
            selected=canonical_count,
            deduplicated=len(accumulated),
        ))

        # 足量判定 1：Stage 1 命中 ≥ canonical_reviews + canonical_models
        # （且 >0，防止空 canonical 误判为足量）→ 跳过 Stage 2-4
        if (
            canonical_count > 0
            and stage1_threshold > 0
            and canonical_count >= stage1_threshold
        ):
            return self._finalize(staged_evidence, accumulated, logs, top_k)

        # ===== Stage 2: BioModels =====
        canonical_models = (
            canonical_ref.canonical_models if canonical_ref else []
        )
        biomodels_ev = self._stage_biomodels(pathway, canonical_models)
        staged_evidence["biomodels"] = list(biomodels_ev)
        biomodels_count = len(biomodels_ev)
        accumulated = self._deduplicate(accumulated + biomodels_ev)
        after_dedup_12 = len(accumulated)
        logs.append(StageLog(
            stage="biomodels",
            query=f"pathway={pathway} models={canonical_models}",
            returned=biomodels_count,
            selected=biomodels_count,
            deduplicated=after_dedup_12,
        ))

        # 足量判定 2：Stage 1+2 命中 ≥ 5 → 跳过 Stage 3-4
        if after_dedup_12 >= SUFFICIENCY_STAGE1_2_THRESHOLD:
            return self._finalize(staged_evidence, accumulated, logs, top_k)

        # ===== Stage 3: PubMed =====
        needed_count = max(0, top_k - after_dedup_12)
        pubmed_ev = self._stage_pubmed(query, needed_count)
        staged_evidence["pubmed"] = list(pubmed_ev)
        pubmed_count = len(pubmed_ev)
        accumulated = self._deduplicate(accumulated + pubmed_ev)
        after_dedup_123 = len(accumulated)
        logs.append(StageLog(
            stage="pubmed",
            query=query,
            returned=pubmed_count,
            selected=pubmed_count,
            deduplicated=after_dedup_123,
        ))

        # 足量判定 3：Stage 1+2+3 命中 ≥ 10 → 跳过 Stage 4
        if after_dedup_123 >= SUFFICIENCY_STAGE1_2_3_THRESHOLD:
            return self._finalize(staged_evidence, accumulated, logs, top_k)

        # ===== Stage 4: Reactome/KEGG =====
        reactome_ev = self._stage_reactome_kegg(pathway)
        staged_evidence["reactome_kegg"] = list(reactome_ev)
        reactome_count = len(reactome_ev)
        accumulated = self._deduplicate(accumulated + reactome_ev)
        logs.append(StageLog(
            stage="reactome_kegg",
            query=f"pathway={pathway}",
            returned=reactome_count,
            selected=reactome_count,
            deduplicated=len(accumulated),
        ))

        return self._finalize(staged_evidence, accumulated, logs, top_k)

    def _finalize(
        self,
        staged_evidence: dict[str, list[Evidence]],
        accumulated: list[Evidence],
        logs: list[StageLog],
        top_k: int,
    ) -> RetrievalResult:
        """排序 + 截断 + 计算命中率 + 写日志。"""
        ranked = self._rank(accumulated)
        final_ranked = ranked[:top_k]
        # canonical_hit_rate: Top1-Top5 中 Canonical 比例
        top5 = ranked[:5]
        canonical_in_top5 = sum(1 for ev in top5 if ev.source == "canonical")
        denom = max(len(top5), 1)
        canonical_hit_rate = canonical_in_top5 / 5.0 if len(top5) >= 5 else (
            canonical_in_top5 / denom
        )
        # 写日志（失败不影响主流程）
        self._write_log(staged_evidence, final_ranked, logs, canonical_hit_rate)
        return RetrievalResult(
            staged_evidence=staged_evidence,
            final_ranked=final_ranked,
            logs=logs,
            canonical_hit_rate=canonical_hit_rate,
        )

    # -------------------------------------------------------------------------
    # Stage 实现
    # -------------------------------------------------------------------------
    def _stage_canonical(self, pathway: str) -> list[Evidence]:
        """Stage 1: 从 canonical.yaml 加载 Reviews/Models/Mechanism。

        Args:
            pathway: 通路标识（如 "egfr"）。

        Returns:
            Evidence 列表（reviews + models + mechanism）；加载失败返回空列表。
        """
        canonical = self._safe_load_canonical(pathway)
        if canonical is None:
            return []
        evidences: list[Evidence] = []
        # 1. canonical_reviews → review 类型 Evidence
        for pmid in canonical.canonical_reviews:
            evidences.append(Evidence(
                source="canonical",
                title=f"Canonical Review {pmid}",
                pmid=pmid,
                biomd_id=None,
                relevance_score=1.0,
                evidence_type="review",
                raw={
                    "pathway": canonical.pathway,
                    "category": "canonical_reviews",
                },
            ))
        # 2. canonical_models → model 类型 Evidence
        for biomd_id in canonical.canonical_models:
            evidences.append(Evidence(
                source="canonical",
                title=f"Canonical BioModel {biomd_id}",
                pmid=None,
                biomd_id=biomd_id,
                relevance_score=1.0,
                evidence_type="model",
                raw={
                    "pathway": canonical.pathway,
                    "category": "canonical_models",
                },
            ))
        # 3. canonical_mechanism → mechanism 类型 Evidence（单条汇总）
        if canonical.canonical_mechanism.required_nodes:
            evidences.append(Evidence(
                source="canonical",
                title=f"Canonical mechanism for {canonical.pathway}",
                pmid=None,
                biomd_id=None,
                relevance_score=1.0,
                evidence_type="mechanism",
                raw={
                    "pathway": canonical.pathway,
                    "category": "canonical_mechanism",
                    "required_nodes": list(
                        canonical.canonical_mechanism.required_nodes
                    ),
                    "edges": [
                        {
                            "from": e.from_node,
                            "to": e.to_node,
                            "relation": e.relation,
                        }
                        for e in canonical.canonical_mechanism.edges
                    ],
                },
            ))
        return evidences

    def _stage_biomodels(
        self,
        pathway: str,
        canonical_models: list[str],
    ) -> list[Evidence]:
        """Stage 2: 检索 Canonical BioModels + 通路关键词补充。

        Args:
            pathway: 通路标识。
            canonical_models: canonical.yaml 中的 BioModels ID 列表。

        Returns:
            Evidence 列表（每个 canonical_model 一条 + 通路关键词补充）。
        """
        evidences: list[Evidence] = []
        canonical_set = {bid.upper() for bid in canonical_models}
        # 1. 为每个 canonical_model 下载 SBML 元数据
        for biomd_id in canonical_models:
            title = f"BioModel {biomd_id}"
            sbml_loaded = False
            if self.biomodels_client is not None:
                try:
                    sbml_xml = self.biomodels_client.download(biomd_id)
                    if sbml_xml:
                        sbml_loaded = True
                        extracted = self._extract_sbml_name(sbml_xml)
                        if extracted:
                            title = extracted
                except Exception as exc:
                    logger.warning(
                        "BioModels download 失败 (%s): %s", biomd_id, exc
                    )
            evidences.append(Evidence(
                source="biomodels",
                title=title,
                pmid=None,
                biomd_id=biomd_id,
                relevance_score=0.9,
                evidence_type="model",
                raw={
                    "biomd_id": biomd_id,
                    "sbml_loaded": sbml_loaded,
                    "from": "canonical_models",
                },
            ))
        # 2. 按通路关键词搜索 BioModels 补充（仅搜索非 canonical 模型）
        if self.biomodels_client is not None:
            try:
                query = pathway.replace("_", " ")
                results = self.biomodels_client.search(query, top_k=5)
            except Exception as exc:
                logger.warning(
                    "BioModels search 失败 (%s): %s", pathway, exc
                )
                results = []
            for r in results or []:
                if not isinstance(r, dict):
                    continue
                mid = str(r.get("model_id", "") or "").strip()
                if not mid or mid.upper() in canonical_set:
                    continue
                pub_id = str(r.get("publication_id", "") or "").strip()
                evidences.append(Evidence(
                    source="biomodels",
                    title=str(r.get("name", mid) or mid),
                    pmid=pub_id or None,
                    biomd_id=mid,
                    relevance_score=0.7,
                    evidence_type="model",
                    raw=dict(r),
                ))
        return evidences

    def _stage_pubmed(self, query: str, needed_count: int) -> list[Evidence]:
        """Stage 3: PubMed 检索 + Evidence Ranking Score。

        Args:
            query: PubMed 检索查询语句。
            needed_count: 需要补充的结果数（top_k - 已有数量）。

        Returns:
            Evidence 列表，relevance_score 为 Evidence Ranking Score。
        """
        if self.rag_client is None or needed_count <= 0:
            return []
        try:
            articles = self.rag_client.search_pubmed(
                query, max_results=needed_count
            )
        except Exception as exc:
            logger.warning(
                "PubMed 检索失败 (query=%s): %s", query[:50], exc
            )
            return []
        # 兼容 MCP 客户端返回的 (articles, records, usage) 元组
        if isinstance(articles, tuple):
            articles = articles[0] if articles else []
        if not articles:
            return []
        # 复用 evidence_ranker.py 的 ranking 逻辑
        ranker = EvidenceRanker(gold_standard=None)
        evidences: list[Evidence] = []
        for art in articles:
            if not isinstance(art, dict):
                continue
            pmid = str(
                art.get("pmid", "") or art.get("uid", "") or ""
            ).strip()
            title = str(art.get("title", "") or "").strip()
            try:
                year = int(art.get("year", 0) or 0)
            except (TypeError, ValueError):
                year = 0
            pub_types = (
                art.get("pub_types") or art.get("publication_types") or []
            )
            etype = ranker.classify(pmid, {"pub_types": pub_types})
            etype_str = _ETYPE_TO_STR.get(etype, "application_paper")
            doc = EvidenceDoc(
                pmid=pmid,
                title=title,
                year=year,
                evidence_type=etype,
            )
            score = ranker.score(doc)
            evidences.append(Evidence(
                source="pubmed",
                title=title,
                pmid=pmid,
                biomd_id=None,
                relevance_score=float(score),
                evidence_type=etype_str,
                raw=dict(art),
            ))
        return evidences

    def _stage_reactome_kegg(self, pathway: str) -> list[Evidence]:
        """Stage 4: Reactome/KEGG 通路拓扑补充。

        简化实现：返回单条 Evidence 标记通路拓扑来源。实际 Reactome/KEGG
        API 集成在 Task 11 完成。

        Args:
            pathway: 通路标识。

        Returns:
            单条 pathway 类型 Evidence 列表。
        """
        return [
            Evidence(
                source="reactome",
                title=f"Reactome/KEGG pathway topology for {pathway}",
                pmid=None,
                biomd_id=None,
                relevance_score=0.3,
                evidence_type="pathway",
                raw={"pathway": pathway, "source_db": "reactome_kegg"},
            )
        ]

    # -------------------------------------------------------------------------
    # 去重 / 排序
    # -------------------------------------------------------------------------
    def _deduplicate(self, evidences: list[Evidence]) -> list[Evidence]:
        """跨 stage 按 PMID/BIOMD/title 去重，保留高优先级 source 的版本。

        去重策略：
          1. 按 source 优先级降序排序（保证高优先级先入字典）
          2. 为每条 Evidence 生成去重键（pmid / biomd_id / title）
          3. 若任一键已存在，跳过该 Evidence（保留先入的高优先级版本）

        Args:
            evidences: 待去重的 Evidence 列表。

        Returns:
            去重后的 Evidence 列表（按 source 优先级降序）。
        """
        # 按 source 优先级降序排序，保证 canonical 先入字典
        sorted_ev = sorted(
            evidences,
            key=lambda ev: -SOURCE_RANK.get(ev.source, 0),
        )
        seen: dict[str, int] = set()  # 已见去重键的 id() 集合（避免重复入字典）
        # 用 dict 存 key → id(evidence)，便于查重
        key_to_id: dict[str, int] = {}
        result: list[Evidence] = []
        for ev in sorted_ev:
            if id(ev) in seen:
                continue
            keys = self._dedup_keys(ev)
            # 检查是否已存在相同 key 的 Evidence
            if any(key in key_to_id for key in keys):
                continue
            # 新增
            for key in keys:
                key_to_id[key] = id(ev)
            seen.add(id(ev))
            result.append(ev)
        return result

    @staticmethod
    def _dedup_keys(ev: Evidence) -> list[str]:
        """生成去重键（PMID / BIOMD / title）。

        Args:
            ev: Evidence 实例。

        Returns:
            去重键列表；若 Evidence 无可用键，返回 [id:xxx] 防止误并。
        """
        keys: list[str] = []
        if ev.pmid:
            keys.append(
                f"pmid:{SequentialRetriever._normalize_pmid(ev.pmid)}"
            )
        if ev.biomd_id:
            keys.append(f"biomd:{ev.biomd_id.upper()}")
        if ev.title:
            keys.append(f"title:{ev.title.lower().strip()}")
        if not keys:
            # 无可用键时用对象 id 防止误并
            keys.append(f"id:{id(ev)}")
        return keys

    @staticmethod
    def _normalize_pmid(pmid: str) -> str:
        """归一化 PMID：去除 'PMID:' 前缀，strip 空白。"""
        if not pmid:
            return ""
        return (
            str(pmid)
            .replace("PMID:", "")
            .replace("pmid:", "")
            .strip()
        )

    def _rank(self, evidences: list[Evidence]) -> list[Evidence]:
        """按 (source_rank, type_rank, relevance_score) 降序排序。

        排序优先级（高 → 低）：
          canonical > biomodels > pubmed(review) > pubmed(mechanism_paper) >
          pubmed(application_paper) > reactome/kegg

        Args:
            evidences: 待排序的 Evidence 列表。

        Returns:
            排序后的新列表（不修改输入）。
        """
        return sorted(
            evidences,
            key=lambda ev: (
                -SOURCE_RANK.get(ev.source, 0),
                -TYPE_RANK.get(ev.evidence_type, 0),
                -ev.relevance_score,
            ),
        )

    # -------------------------------------------------------------------------
    # 日志
    # -------------------------------------------------------------------------
    def _write_log(
        self,
        staged_evidence: dict[str, list[Evidence]],
        final_ranked: list[Evidence],
        logs: list[StageLog],
        canonical_hit_rate: float,
    ) -> Path:
        """写入检索日志到 logs/sequential_retriever/<timestamp>.json。

        Args:
            staged_evidence: 各 stage 的 Evidence 列表。
            final_ranked: 排序后的最终 Evidence 列表。
            logs: 各 stage 的检索日志。
            canonical_hit_rate: Top1-Top5 中 Canonical 比例。

        Returns:
            日志文件路径；失败返回空 Path。
        """
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            log_path = self.log_dir / f"{timestamp}.json"
            payload = {
                "timestamp": timestamp,
                "logs": [
                    {
                        "stage": log.stage,
                        "query": log.query,
                        "returned": log.returned,
                        "selected": log.selected,
                        "deduplicated": log.deduplicated,
                    }
                    for log in logs
                ],
                "staged_evidence_summary": {
                    stage: len(evs)
                    for stage, evs in staged_evidence.items()
                },
                "final_ranked_count": len(final_ranked),
                "canonical_hit_rate": canonical_hit_rate,
                "final_ranked_top5": [
                    {
                        "source": ev.source,
                        "title": ev.title,
                        "pmid": ev.pmid,
                        "biomd_id": ev.biomd_id,
                        "evidence_type": ev.evidence_type,
                        "relevance_score": ev.relevance_score,
                    }
                    for ev in final_ranked[:5]
                ],
            }
            log_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return log_path
        except Exception as exc:
            logger.warning(
                "写入 sequential_retriever 日志失败: %s", exc
            )
            return Path()

    # -------------------------------------------------------------------------
    # 辅助
    # -------------------------------------------------------------------------
    def _safe_load_canonical(self, pathway: str) -> CanonicalReference | None:
        """安全加载 Canonical Reference，失败返回 None。

        Args:
            pathway: 通路标识。

        Returns:
            CanonicalReference 实例；加载失败返回 None。
        """
        try:
            return load_canonical(pathway)
        except Exception as exc:
            logger.warning(
                "Canonical Reference 加载失败 (pathway=%s): %s", pathway, exc
            )
            return None

    @staticmethod
    def _extract_sbml_name(sbml_xml: str) -> str:
        """从 SBML XML 提取 model name（best-effort，无依赖）。

        Args:
            sbml_xml: SBML XML 文本。

        Returns:
            模型名称；提取失败返回空字符串。
        """
        if not sbml_xml:
            return ""
        # 简单正则提取 <model id="..." name="...">
        match = re.search(r'<model[^>]*\bname="([^"]+)"', sbml_xml)
        if match:
            return match.group(1)
        match = re.search(r'<model[^>]*\bid="([^"]+)"', sbml_xml)
        if match:
            return match.group(1)
        return ""


# =============================================================================
# 便捷函数
# =============================================================================


def sequential_retrieve(
    pathway: str,
    query: str,
    *,
    top_k: int = 10,
    canonical_dir: str | Path | None = None,
    rag_client: Any = None,
    biomodels_client: Any = None,
) -> RetrievalResult:
    """顺序优先级检索便捷函数（等价于 SequentialRetriever().retrieve()）。

    对应 Spec Requirement "Sequential Retriever（顺序优先级检索）"。

    Args:
        pathway: 通路名称（如 "egfr"）
        query: 检索查询语句
        top_k: 返回 Top-K 结果
        canonical_dir: Canonical YAML 目录（None 用默认）
        rag_client: PubMed 检索客户端
        biomodels_client: BioModels API 客户端

    Returns:
        RetrievalResult
    """
    retriever = SequentialRetriever(
        canonical_dir=canonical_dir,
        rag_client=rag_client,
        biomodels_client=biomodels_client,
    )
    return retriever.retrieve(pathway, query, top_k=top_k)


__all__ = [
    "RETRIEVAL_STAGES",
    "SOURCE_RANK",
    "TYPE_RANK",
    "SUFFICIENCY_STAGE1_2_THRESHOLD",
    "SUFFICIENCY_STAGE1_2_3_THRESHOLD",
    "Evidence",
    "StageLog",
    "RetrievalStageRecord",
    "RetrievalResult",
    "SequentialRetriever",
    "sequential_retrieve",
]
