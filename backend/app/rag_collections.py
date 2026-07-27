# BioDynamics Agent - 四路 RAG 集合封装（v2 升级）
# 对应 biodynamics-v2-upgrade-plan.md §六：把单 collection 拆为四个语义集合
#
# 拆分目的：
# 1. Mechanism RAG   — 路径/通路知识（Node 3 检索"某 pathway 的拓扑信息"）
# 2. Parameter RAG   — 动力学参数 Kd/Km/half-life/rate（Node 5 程序注入，LLM 禁止改）
# 3. Experiment RAG  — 实验方案 Western/ELISA/qPCR/Flow（Node 9 推荐验证手段）
# 4. Evidence RAG    — 文献证据 PMID/DOI/Figure/Cell Line（Node 10 引用支撑）
#
# 复用 RagClient 的 ChromaDB 连接、混合检索（语义+BM25）、重排序能力。
# 每个 collection 拥有独立的 metadata schema 与 upsert / search 接口。
#
# 向后兼容：当 ChromaDB 不可用或对应 collection 不存在时，所有方法安全降级为返回空值。

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

from app.config import embedding_model, settings
from app.rag_client import (
    RagClient,
    normalize_param,
    _BM25,
    _tokenize,
    _SOURCE_AUTHORITY,
    _SPECIES_PRIORITY,
    _confidence_to_score,
    _extract_source_type,
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# 四路 collection 的元信息
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class CollectionSpec:
    """单个 RAG collection 的元信息。"""

    role: str             # "mechanism" | "parameter" | "experiment" | "evidence"
    name: str             # ChromaDB collection 名
    cn_label: str         # 中文短标签（前端 / 日志展示用）
    description: str      # 用途说明


# 四路 collection 注册表；缺省名来自 config.CHROMA_COLLECTION_*
COLLECTION_REGISTRY: list[CollectionSpec] = [
    CollectionSpec(
        role="mechanism",
        name=settings.CHROMA_COLLECTION_MECHANISM,
        cn_label="机制 RAG",
        description="路径/通路知识：节点、相互作用、调控关系",
    ),
    CollectionSpec(
        role="parameter",
        name=settings.CHROMA_COLLECTION_PARAMETER,
        cn_label="参数 RAG",
        description="动力学参数：Kd、Km、Vmax、half-life、rate constant",
    ),
    CollectionSpec(
        role="experiment",
        name=settings.CHROMA_COLLECTION_EXPERIMENT,
        cn_label="实验 RAG",
        description="实验方案：Western blot、Flow Cytometry、ELISA、qPCR",
    ),
    CollectionSpec(
        role="evidence",
        name=settings.CHROMA_COLLECTION_EVIDENCE,
        cn_label="证据 RAG",
        description="文献证据：PMID、DOI、Figure、Cell Line、Species",
    ),
]


def _spec_by_role(role: str) -> CollectionSpec:
    """根据角色名获取 collection 元信息；找不到则降级到 mechanism。"""
    for spec in COLLECTION_REGISTRY:
        if spec.role == role:
            return spec
    raise ValueError(f"未知 RAG 角色：{role}")


# -----------------------------------------------------------------------------
# 四路 RAG 集合封装
# -----------------------------------------------------------------------------
class RagCollections:
    """四路 RAG 集合封装。

    使用方法：
        rag = RagCollections()
        rag.upsert_parameter([{...}, {...}])
        hits = rag.search_parameter("TGF-beta inhibits CD8", top_k=5)
    """

    def __init__(self, client: chromadb.api.ClientAPI | None = None) -> None:
        self._collections: dict[str, Collection | None] = {
            spec.role: None for spec in COLLECTION_REGISTRY
        }
        self._available = False

        # 优先使用调用方传入的 client，否则自建
        if client is None:
            try:
                client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
                client.heartbeat()
            except Exception as exc:
                logger.warning("RagCollections ChromaDB 连接失败，RAG 自动降级：%s", exc)
                client = None

        self._client = client
        if client is None:
            logger.warning("RagCollections 不可用，所有 RAG 接口将返回空值")
            return

        self._available = True
        # 预热四个 collection
        for spec in COLLECTION_REGISTRY:
            self._collections[spec.role] = self._get_or_create(spec)

    @property
    def available(self) -> bool:
        return self._available

    def _get_or_create(self, spec: CollectionSpec) -> Collection | None:
        """获取或创建指定 collection。"""
        if not self._available or self._client is None:
            return None
        try:
            return self._client.get_or_create_collection(
                name=spec.name,
                metadata={"hnsw:space": "cosine", "role": spec.role},
            )
        except Exception as exc:
            logger.warning("创建 collection 失败 (%s)：%s", spec.role, exc)
            return None

    def _collection(self, role: str) -> Collection | None:
        """按角色取 collection；未初始化时延迟创建。"""
        coll = self._collections.get(role)
        if coll is not None:
            return coll
        if not self._available:
            return None
        coll = self._get_or_create(_spec_by_role(role))
        self._collections[role] = coll
        return coll

    # -------------------------------------------------------------------------
    # 元数据 schema 工具：把任意 dict 裁剪为 Chroma metadata 兼容字段
    # -------------------------------------------------------------------------
    @staticmethod
    def _to_chroma_metadata(payload: dict[str, Any]) -> dict[str, Any]:
        """把任意 dict 转换为 ChromaDB metadata（str/int/float/bool/None）。"""
        out: dict[str, Any] = {}
        for key, value in payload.items():
            if isinstance(value, (str, int, float, bool)):
                out[key] = value
            elif value is None:
                out[key] = ""
            elif isinstance(value, (list, tuple)):
                # 列表类型：序列化为逗号分隔字符串，便于 where 过滤
                out[key] = ",".join(str(v) for v in value)
            else:
                out[key] = str(value)
        return out

    @staticmethod
    def _build_search_text(*parts: Any) -> str:
        """拼接检索文本（多段拼成单字符串）。"""
        return " ".join(str(p) for p in parts if p is not None and str(p).strip())

    def _embed(self, text: str) -> list[float] | None:
        """调用全局 embedding 模型编码；失败返回 None。"""
        try:
            return embedding_model.embed_query(text)
        except Exception as exc:
            logger.warning("Embedding 失败：%s", exc)
            return None

    def _upsert(
        self,
        role: str,
        records: list[dict[str, Any]],
        text_extractor: Any,
    ) -> int:
        """通用 upsert 流程：取 collection → 编码 → 写入。"""
        coll = self._collection(role)
        if coll is None or not records:
            return 0

        ids: list[str] = []
        embeddings: list[list[float]] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for record in records:
            search_text = text_extractor(record)
            if not search_text:
                continue
            vector = self._embed(search_text)
            if vector is None:
                continue

            stable_key = json.dumps(
                self._to_chroma_metadata(record),
                ensure_ascii=True,
                sort_keys=True,
            )
            digest = hashlib.sha256(
                f"{role}\n{search_text}\n{stable_key}".encode("utf-8")
            ).hexdigest()
            ids.append(f"{role}_{digest[:32]}")
            embeddings.append(vector)
            documents.append(search_text)
            metadatas.append(self._to_chroma_metadata(record))

        if not ids:
            return 0
        try:
            coll.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            return len(ids)
        except Exception as exc:
            logger.warning("写入 collection 失败 (%s)：%s", role, exc)
            return 0

    def _search(
        self,
        role: str,
        query: str,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """通用语义检索流程。"""
        coll = self._collection(role)
        if coll is None:
            return []
        vector = self._embed(query)
        if vector is None:
            return []

        try:
            query_kwargs: dict[str, Any] = {
                "query_embeddings": [vector],
                "n_results": top_k,
                "include": ["metadatas", "documents", "distances"],
            }
            if where:
                query_kwargs["where"] = where
            results = coll.query(**query_kwargs)
        except Exception as exc:
            logger.warning("语义检索失败 (%s)：%s", role, exc)
            return []

        metadatas = results.get("metadatas", [[]])[0]
        documents = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]

        enriched: list[dict[str, Any]] = []
        for idx, meta in enumerate(metadatas):
            if not meta:
                continue
            dist = distances[idx] if idx < len(distances) else 1.0
            record = dict(meta)
            record["_document"] = documents[idx] if idx < len(documents) else ""
            record["_semantic_score"] = max(0.0, 1.0 - float(dist))
            record["_retrieval_method"] = "semantic"
            record["_collection_role"] = role
            enriched.append(record)
        return enriched

    # -------------------------------------------------------------------------
    # Mechanism RAG：路径/通路知识
    # -------------------------------------------------------------------------
    def upsert_mechanism(self, records: list[dict[str, Any]]) -> int:
        """写入机制知识。

        record schema:
        {
            "pathway": str,             # 通路名（必填）
            "entities": list[str],      # 涉及的生物实体
            "interactions": list[str],  # 主要调控关系
            "description": str,         # 机制描述
            "source": str,              # 来源（如 "Reactome" / "KEGG" / PMID）
            "pmid": str,                # PubMed ID（可选）
        }
        """
        def text_extractor(rec: dict[str, Any]) -> str:
            return self._build_search_text(
                rec.get("pathway"),
                rec.get("description"),
                " ".join(rec.get("entities", []) or []),
                " ".join(rec.get("interactions", []) or []),
                rec.get("source"),
            )
        return self._upsert("mechanism", records, text_extractor)

    def search_mechanism(
        self,
        query: str,
        top_k: int = 5,
        pathway: str | None = None,
    ) -> list[dict[str, Any]]:
        """检索机制知识（按 query 语义匹配，可选 path 过滤）。"""
        where: dict[str, Any] | None = None
        if pathway:
            where = {"pathway": pathway}
        return self._search("mechanism", query, top_k=top_k, where=where)

    # -------------------------------------------------------------------------
    # Parameter RAG：动力学参数
    # -------------------------------------------------------------------------
    def upsert_parameter(self, records: list[dict[str, Any]]) -> int:
        """写入动力学参数。

        record schema:
        {
            "param_name": str,        # 必填
            "value": float,           # 必填
            "unit": str,              # 必填
            "context": str,           # 上下文（如 "TGF-beta inhibition of CD8"）
            "species": str,
            "cell_line": str,
            "source": str,            # PMID / 内部 DB / 模型 ID
            "source_model": str,      # BIOMD 模型 ID（可选）
            "confidence": str,        # HIGH / MEDIUM / LOW
        }
        """
        def text_extractor(rec: dict[str, Any]) -> str:
            return self._build_search_text(
                rec.get("param_name"),
                rec.get("context"),
                rec.get("species"),
                rec.get("cell_line"),
                rec.get("source_model"),
            )
        # 单位归一化：复用 RagClient.normalize_param
        normalized = [normalize_param(r) for r in records]
        return self._upsert("parameter", normalized, text_extractor)

    def search_parameter(
        self,
        query: str,
        top_k: int = 5,
        species_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """检索动力学参数（优先物种过滤）。"""
        where: dict[str, Any] | None = None
        if species_filter:
            where = {"species": species_filter}
        return self._search("parameter", query, top_k=top_k, where=where)

    def search_parameter_hybrid(
        self,
        query: str,
        species_context: str = "Human",
        top_k: int = 5,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """高阶 RAG 检索：语义 + BM25 混合检索 + 重排序，返回 (结果, 洞察数据)。

        复用 rag_client 的 _BM25 / _SOURCE_AUTHORITY / _SPECIES_PRIORITY 等工具，
        操作于 parameter collection。洞察数据格式与 RagClient.search_params_hybrid 对齐。
        """
        empty_insights: dict[str, Any] = {
            "rewritten_query": query,
            "rewrites": [],
            "source_distribution": {},
            "total_candidates": 0,
            "top_selections": [],
        }

        if not self._available:
            return [], empty_insights

        # 1. 语义检索
        where: dict[str, Any] | None = None
        if species_context:
            where = {"species": species_context}
        semantic_results = self._search("parameter", query, top_k=10, where=where)

        # 2. BM25 检索（拉取全量文档构建索引）
        bm25_results: list[dict[str, Any]] = []
        coll = self._collection("parameter")
        if coll is not None:
            try:
                all_data = coll.get(include=["metadatas", "documents"])
                documents = all_data.get("documents", []) or []
                metadatas = all_data.get("metadatas", []) or []
                if documents:
                    tokenized_docs = [_tokenize(doc) for doc in documents]
                    bm25 = _BM25(tokenized_docs)
                    query_terms = _tokenize(query)
                    if query_terms:
                        hits = bm25.search(query_terms, top_k=10)
                        max_score = max((s for _, s in hits), default=1.0)
                        for doc_idx, score in hits:
                            meta = metadatas[doc_idx] if doc_idx < len(metadatas) else {}
                            if not meta:
                                continue
                            record = dict(meta)
                            record["_document"] = documents[doc_idx]
                            record["_bm25_score"] = float(score)
                            record["_semantic_score"] = (
                                float(score) / max_score if max_score > 0 else 0.0
                            )
                            record["_retrieval_method"] = "bm25"
                            bm25_results.append(record)
            except Exception as exc:
                logger.warning("Parameter BM25 检索失败：%s", exc)

        # 3. 合并去重（按 _document 内容键去重，同时命中取较高语义分）
        merged: dict[str, dict[str, Any]] = {}
        for record in semantic_results + bm25_results:
            doc_key = record.get("_document", "") or str(
                sorted({k: v for k, v in record.items() if not k.startswith("_")}.items())
            )
            if doc_key not in merged:
                merged[doc_key] = record
            else:
                existing = merged[doc_key]
                existing["_semantic_score"] = max(
                    existing.get("_semantic_score", 0.0),
                    record.get("_semantic_score", 0.0),
                )
                existing["_retrieval_method"] = "hybrid"

        candidates = list(merged.values())

        # 4. 重排序（来源权威性 0.30 + 物种特异性 0.25 + 参数完整性 0.20 + 语义相关度 0.25）
        scored: list[tuple[float, dict[str, Any]]] = []
        for record in candidates:
            source_type = _extract_source_type(record)
            authority = _SOURCE_AUTHORITY.get(source_type, 0.5)
            record_species = str(record.get("species", ""))
            species_score = _SPECIES_PRIORITY.get(record_species, 0.5)
            if record_species and species_context and record_species.lower() == species_context.lower():
                species_score = 1.0
            confidence = record.get("confidence", "MEDIUM")
            completeness = _confidence_to_score(confidence)
            value = record.get("value")
            if value is None or str(value).strip() == "":
                completeness *= 0.3
            relevance = float(record.get("_semantic_score", 0.0))
            final_score = authority * 0.30 + species_score * 0.25 + completeness * 0.20 + relevance * 0.25
            record["_rerank_score"] = final_score
            record["_source_type"] = source_type
            scored.append((final_score, record))
        scored.sort(key=lambda x: x[0], reverse=True)
        reranked = [r for _, r in scored[:top_k]]

        # 5. 构建洞察数据
        source_counter: Counter[str] = Counter()
        for c in candidates:
            source_counter[_extract_source_type(c)] += 1

        top_selections: list[dict[str, Any]] = []
        for rec in reranked:
            pmid = str(rec.get("source_pmid") or rec.get("pmid") or "").strip()
            if not pmid:
                source_str = str(rec.get("source", ""))
                pmid_match = re.search(r"PMID[:\s]*(\d+)", source_str, re.IGNORECASE)
                pmid = pmid_match.group(1) if pmid_match else ""
            top_selections.append({
                "parameter": str(rec.get("param_name", "")),
                "value": f"{rec.get('value', '')} {rec.get('unit', '')}".strip(),
                "source": str(rec.get("source", "Retrieved from RAG")),
                "pmid": pmid,
                "confidence_score": round(float(rec.get("_rerank_score", 0.0)), 2),
                "species": str(rec.get("species", "")),
                "context": str(rec.get("context", "")),
            })

        insights: dict[str, Any] = {
            "rewritten_query": query,
            "rewrites": [],
            "source_distribution": dict(source_counter),
            "total_candidates": len(candidates),
            "top_selections": top_selections,
        }
        return reranked, insights

    # -------------------------------------------------------------------------
    # Experiment RAG：实验方案
    # -------------------------------------------------------------------------
    def upsert_experiment(self, records: list[dict[str, Any]]) -> int:
        """写入实验方案。

        record schema:
        {
            "name": str,                # 方案名（如 "TGF-beta Western blot"）
            "target": str,              # 靶点
            "detection_method": str,    # Western blot / Flow / ELISA / qPCR
            "cell_line": str,
            "species": str,
            "pmid": str,                # 来源 PMID
            "description": str,
        }
        """
        def text_extractor(rec: dict[str, Any]) -> str:
            return self._build_search_text(
                rec.get("name"),
                rec.get("target"),
                rec.get("detection_method"),
                rec.get("cell_line"),
                rec.get("species"),
                rec.get("description"),
            )
        return self._upsert("experiment", records, text_extractor)

    def search_experiment(
        self,
        query: str,
        target: str | None = None,
        detection_method: str | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """检索实验方案。"""
        where: dict[str, Any] = {}
        if target:
            where["target"] = target
        if detection_method:
            where["detection_method"] = detection_method
        return self._search("experiment", query, top_k=top_k, where=where or None)

    # -------------------------------------------------------------------------
    # Evidence RAG：文献证据
    # -------------------------------------------------------------------------
    def upsert_evidence(self, records: list[dict[str, Any]]) -> int:
        """写入文献证据。

        record schema:
        {
            "pmid": str,             # PubMed ID
            "doi": str,              # DOI
            "title": str,
            "figure_ref": str,       # Fig.2A 等
            "cell_line": str,
            "species": str,
            "year": str,             # 发表年份
            "journal": str,
        }
        """
        def text_extractor(rec: dict[str, Any]) -> str:
            return self._build_search_text(
                rec.get("title"),
                rec.get("pmid"),
                rec.get("figure_ref"),
                rec.get("cell_line"),
                rec.get("species"),
                rec.get("journal"),
            )
        return self._upsert("evidence", records, text_extractor)

    def search_evidence(
        self,
        query: str,
        pmid: str | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """检索文献证据（可按 PMID 精确过滤）。"""
        where: dict[str, Any] | None = None
        if pmid:
            where = {"pmid": pmid}
        return self._search("evidence", query, top_k=top_k, where=where)

    # -------------------------------------------------------------------------
    # 集合统计
    # -------------------------------------------------------------------------
    def count(self, role: str) -> int:
        """返回指定 collection 的文档数量。"""
        coll = self._collection(role)
        if coll is None:
            return 0
        try:
            return coll.count()
        except Exception as exc:
            logger.warning("统计 collection 失败 (%s)：%s", role, exc)
            return 0

    def stats(self) -> dict[str, int]:
        """返回四路 collection 的文档数量摘要。"""
        return {spec.role: self.count(spec.role) for spec in COLLECTION_REGISTRY}


# -----------------------------------------------------------------------------
# 全局实例（懒加载，ChromaDB 不可用时安全）
# -----------------------------------------------------------------------------
_global_rag_collections: RagCollections | None = None


def get_rag_collections() -> RagCollections:
    """获取（或懒加载）全局 RagCollections 实例。

    调用方如只需要读写机制/参数/实验/证据四路，可直接用本函数。
    """
    global _global_rag_collections
    if _global_rag_collections is None:
        _global_rag_collections = RagCollections()
    return _global_rag_collections


# -----------------------------------------------------------------------------
# 向后兼容：将 RagClient 包装为 RagCollections
# -----------------------------------------------------------------------------
def wrap_rag_client_as_collections(rag_client: RagClient) -> RagCollections:
    """把已存在的 RagClient 包装为 RagCollections，复用其 ChromaDB client。

    这样 v1 的 RagClient 与 v2 的 RagCollections 共享同一份向量库存储。
    """
    if rag_client.client is None:
        return RagCollections(client=None)
    return RagCollections(client=rag_client.client)
