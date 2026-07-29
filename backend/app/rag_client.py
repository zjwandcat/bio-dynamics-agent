# BioDynamics Agent - 高阶 RAG 客户端
# 对应 1233.md 第二部分规范：查询重写 + 混合检索（语义 + BM25）+ 重排序。
# 负责 ChromaDB 本地持久化连接、动力学参数写入与语义检索，并提供单位归一化辅助函数。

import json
import logging
import math
import re
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import chromadb
import requests
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.config import embedding_model, llm, rerank_manager, settings
from app.prompts import QUERY_REWRITING_PROMPT

# Task 19 SEC-1.4: 优先使用 defusedxml 防御 XXE/实体扩展攻击
# 注意：logger 在下方定义，此处不调用 logging，避免循环引用
try:
    from defusedxml import ElementTree as _ET  # type: ignore
    _ET_FALLBACK = False
except ImportError:  # pragma: no cover - 离线场景降级
    import xml.etree.ElementTree as _ET  # type: ignore
    _ET_FALLBACK = True

logger = logging.getLogger(__name__)
if _ET_FALLBACK:
    logger.warning("defusedxml 未安装，NCBI XML 解析降级到 xml.etree（有实体扩展风险）")


# === V4 Enhancement: Sequential Retriever 延迟导入缓存 ===
# 仅在 V4_SEQUENTIAL_RETRIEVER=true 且调用方提供 pathway 时实际加载，避免
# 在 Flag=false 或无 pathway 调用时引入 scientific_alignment.sequential_retriever
# 模块的额外启动开销。
_sequential_retrieve_impl_lazy = None  # type: Any


# -----------------------------------------------------------------------------
# 单位归一化辅助函数
# -----------------------------------------------------------------------------
def _to_hours(value: float, unit: str) -> tuple[float, str]:
    """将常见时间单位统一转换为小时（h）。"""
    unit_lower = unit.lower().strip()
    if unit_lower in ("h", "hr", "hour", "hours"):
        return float(value), "h"
    if unit_lower in ("min", "minute", "minutes"):
        return float(value) / 60.0, "h"
    if unit_lower in ("s", "sec", "second", "seconds"):
        return float(value) / 3600.0, "h"
    if unit_lower in ("d", "day", "days"):
        return float(value) * 24.0, "h"
    # 无法识别时保持原值，避免丢失信息
    return float(value), unit


def _to_nM(value: float, unit: str) -> tuple[float, str]:
    """将常见浓度单位统一转换为 nM。"""
    unit_lower = unit.lower().strip()
    if unit_lower == "nm":
        return float(value), "nM"
    if unit_lower in ("um", "μm", "micromolar"):
        return float(value) * 1000.0, "nM"
    if unit_lower in ("mm", "millimolar"):
        return float(value) * 1_000_000.0, "nM"
    if unit_lower in ("pm", "picomolar"):
        return float(value) / 1000.0, "nM"
    return float(value), unit


def normalize_param(record: dict[str, Any]) -> dict[str, Any]:
    """对单条参数记录做单位归一化，时间与浓度分别转为 h / nM。"""
    normalized = dict(record)
    param_name = str(normalized.get("param_name", "")).lower()
    value = normalized.get("value")
    unit = str(normalized.get("unit", ""))

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return normalized

    # 浓度相关参数统一转为 nM
    if any(key in param_name for key in ("kd", "km", "vmax", "ec50", "ic50")):
        numeric_value, unit = _to_nM(numeric_value, unit)
    # 时间相关参数统一转为 h
    elif any(
        key in param_name
        for key in ("half-life", "half_life", "halflife", "degradation", "secretion", "rate")
    ):
        numeric_value, unit = _to_hours(numeric_value, unit)

    normalized["value"] = numeric_value
    normalized["unit"] = unit
    return normalized


def _record_to_rerank_text(record: dict[str, Any]) -> str:
    """将参数记录拼接为 rerank 模型可理解的文本片段。"""
    parts: list[str] = []
    param_name = str(record.get("param_name", "")).strip()
    if param_name:
        parts.append(param_name)
    context = str(record.get("context", "")).strip()
    if context:
        parts.append(context)
    species = str(record.get("species", "")).strip()
    if species:
        parts.append(f"Species: {species}")
    value = record.get("value")
    unit = str(record.get("unit", "")).strip()
    if value is not None and str(value).strip():
        parts.append(f"Value: {value} {unit}".strip())
    source = str(record.get("source", "")).strip()
    if source:
        parts.append(f"Source: {source}")
    return " | ".join(parts) if parts else json.dumps(record, ensure_ascii=False)


# -----------------------------------------------------------------------------
# BM25 轻量实现（纯 Python，避免额外依赖）
# -----------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"[A-Za-z0-9\-]+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    """简单分词：小写化并提取字母数字与连字符片段。"""
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


class _BM25:
    """Okapi BM25 评分器，支持对文档集合按查询词打分。"""

    def __init__(self, documents: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.documents = documents
        self.n_docs = len(documents)
        self.k1 = k1
        self.b = b
        self.avg_len = (
            sum(len(d) for d in documents) / max(self.n_docs, 1)
        )
        self.df: Counter[str] = Counter()
        for doc in documents:
            for term in set(doc):
                self.df[term] += 1
        self.idf: dict[str, float] = {}
        for term, df in self.df.items():
            # +1 平滑，保证非负
            self.idf[term] = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1)

    def _score(self, query_terms: list[str], doc_idx: int) -> float:
        doc = self.documents[doc_idx]
        doc_len = len(doc)
        if doc_len == 0:
            return 0.0
        doc_tf: Counter[str] = Counter(doc)
        score = 0.0
        for term in query_terms:
            tf = doc_tf.get(term, 0)
            if tf == 0:
                continue
            idf = self.idf.get(term, 0.0)
            denom = tf + self.k1 * (1 - self.b + self.b * doc_len / max(self.avg_len, 1))
            score += idf * (tf * (self.k1 + 1)) / denom
        return score

    def search(self, query_terms: list[str], top_k: int = 10) -> list[tuple[int, float]]:
        """返回 (doc_idx, score) 列表，按分数降序截取 top_k。"""
        scored = [(idx, self._score(query_terms, idx)) for idx in range(self.n_docs)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [item for item in scored[:top_k] if item[1] > 0]


# -----------------------------------------------------------------------------
# 查询重写结构化输出
# -----------------------------------------------------------------------------
class _RewriteItem(BaseModel):
    original: str = Field(..., description="原始词")
    standardized: str = Field(..., description="标准化后的词")
    reason: str = Field(default="", description="标准化理由")


class QueryRewriteOutput(BaseModel):
    """查询重写的结构化输出。"""

    rewritten_query: str = Field(..., description="重写后的标准化查询字符串")
    rewrites: list[_RewriteItem] = Field(default_factory=list, description="逐项重写记录")
    expanded_terms: list[str] = Field(default_factory=list, description="扩展的检索词")


# -----------------------------------------------------------------------------
# 药物候选提取结构化输出
# -----------------------------------------------------------------------------
class _DrugCandidate(BaseModel):
    """从文献中提取的单条药物候选记录。"""

    drug_name: str = Field(..., description="药物通用名或研发代号")
    target_name: str = Field(default="", description="作用靶点")
    ic50: float | None = Field(default=None, description="IC50 值（nM），未提及则留空")
    ec50: float | None = Field(default=None, description="EC50 值（nM），未提及则留空")
    clinical_dose: str = Field(default="", description="临床给药方案，如 150 mg BID")
    source: str = Field(default="", description="来源文献或数据库")
    is_clinical_candidate: bool = Field(default=False, description="是否已进入临床试验")


class _DrugExtractionOutput(BaseModel):
    """药物候选提取的结构化输出。"""

    drug_candidates: list[_DrugCandidate] = Field(
        default_factory=list, description="提取的药物候选列表"
    )


# -----------------------------------------------------------------------------
# 药物候选提取提示词
# -----------------------------------------------------------------------------
DRUG_EXTRACTION_PROMPT = """你是药物信息学专家。从给定的 PubMed 文献片段中，提取针对特定靶点的候选药物信息。

靶点：{target_name}
物种/细胞系：{species_context}

提取要求：
1. 药物名称（优先通用名、研发代号或商品名）
2. 若文献明确给出 IC50 或 EC50，提取数值并统一为 nM（uM 请乘以 1000）
3. 若文献提到临床给药方案或临床试验，标注 is_clinical_candidate=true 并提取剂量信息
4. 来源请填写 PMID、标题或期刊信息

注意：
- 仅提取与靶点 {target_name} 直接相关的抑制剂/拮抗剂/抗体药物
- 不要提取无关药物
- 若文献未提及具体药物，返回空列表
- 绝对禁止编造数据
"""


# -----------------------------------------------------------------------------
# ClinicalTrials.gov 查询辅助函数
# -----------------------------------------------------------------------------
def _query_clinical_trials(drug_name: str) -> list[dict]:
    """查询 ClinicalTrials.gov，返回临床试验列表。离线或失败返回空列表。"""
    url = "https://clinicaltrials.gov/api/v2/studies"
    params = {
        "query.term": drug_name,
        "pageSize": "5",
    }
    try:
        response = requests.get(url, params=params, timeout=8)
        response.raise_for_status()
        data = response.json()
        studies = data.get("studies", []) or []
        results: list[dict] = []
        for study in studies:
            # Task 19 SubTask 19.2: per-study isinstance 守卫
            # 防止单条非 dict study 导致整批结果丢失（被外层 try/except 静默吞掉）
            if not isinstance(study, dict):
                logger.warning("ClinicalTrials 跳过非 dict study: %r", type(study))
                continue
            try:
                protocol = study.get("protocolSection", {}) or {}
                if not isinstance(protocol, dict):
                    protocol = {}
                identification = protocol.get("identification", {}) or {}
                if not isinstance(identification, dict):
                    identification = {}
                status_module = protocol.get("statusModule", {}) or {}
                if not isinstance(status_module, dict):
                    status_module = {}
                conditions_module = protocol.get("conditionsModule", {}) or {}
                if not isinstance(conditions_module, dict):
                    conditions_module = {}
                nct_id = identification.get("nctId", "") or ""
                phase = status_module.get("phase", "Unknown") or "Unknown"
                if isinstance(phase, list):
                    phase = "/".join(str(p) for p in phase)
                status = status_module.get("overallStatus", "Unknown") or "Unknown"
                conditions = conditions_module.get("conditions", []) or []
                if not isinstance(conditions, list):
                    conditions = []
                condition = conditions[0] if conditions else ""
                results.append(
                    {
                        "nct_id": nct_id,
                        "phase": phase,
                        "condition": condition,
                        "status": status,
                    }
                )
            except Exception as exc:
                # per-study 容错：单条解析失败不影响其他 study
                logger.warning("ClinicalTrials 单条 study 解析失败，跳过: %s", exc)
                continue
        return results
    except Exception as exc:
        logger.warning("ClinicalTrials.gov 查询失败（%s）：%s", drug_name, exc)
        return []


# -----------------------------------------------------------------------------
# 来源权威性评分表
# -----------------------------------------------------------------------------
_SOURCE_AUTHORITY: dict[str, float] = {
    "PMC": 1.0,        # PubMed Central 全文，权威性最高
    "PubMed": 0.85,    # PubMed 摘要
    "Internal DB": 0.6,  # 内部知识库
    "Preprint": 0.4,   # 预印本
}

_SPECIES_PRIORITY: dict[str, float] = {
    "Human": 1.0,
    "Mouse": 0.8,
    "Rat": 0.7,
    "HeLa": 0.95,
    "HEK293": 0.9,
    "CHO": 0.75,
    "T-cell": 0.85,
}


def _confidence_to_score(confidence: str) -> float:
    """将 HIGH / MEDIUM 字符串转为数值分数。"""
    c = str(confidence).upper().strip()
    if c == "HIGH":
        return 1.0
    if c == "MEDIUM":
        return 0.6
    return 0.3


def _extract_source_type(meta: dict[str, Any]) -> str:
    """从元数据中推断来源类型（PMC / PubMed / Internal DB / Preprint）。"""
    source = str(meta.get("source", "")).upper()
    if "PMC" in source:
        return "PMC"
    if "PUBMED" in source or "PMID" in source:
        return "PubMed"
    if "BIORXIV" in source or "PREPRINT" in source:
        return "Preprint"
    return "Internal DB"


# -----------------------------------------------------------------------------
# ChromaDB 客户端封装
# -----------------------------------------------------------------------------
class RagClient:
    """ChromaDB 向量库客户端，负责动力学参数的存储、混合检索与重排序。"""

    def __init__(self) -> None:
        self.collection_name = settings.CHROMA_COLLECTION_NAME
        # [Round 5] v2 parameter collection（1607 records from 70 SBML files）
        # legacy collection（biodynamics_params, 752 records）缺少 Apoptosis 等通路参数，
        # 同时搜索 v2 collection 以覆盖全部 10 通路
        self._v2_collection_name = settings.CHROMA_COLLECTION_PARAMETER
        self.persist_dir = settings.CHROMA_PERSIST_DIR
        self._collection: Any | None = None
        self._v2_collection: Any | None = None
        self._available = False

        try:
            self.client = chromadb.PersistentClient(path=self.persist_dir)
            # 轻量探测 ChromaDB 是否可用
            self.client.heartbeat()
            self._available = True
        except Exception as exc:
            logger.warning("ChromaDB 连接失败，RAG 功能将自动降级：%s", exc)
            self.client = None  # type: ignore[assignment]
            self._available = False

    @property
    def available(self) -> bool:
        """RAG 服务是否可用。"""
        return self._available

    def _get_collection(self) -> Any:
        """获取或创建 collection；不可用则返回 None。"""
        if not self._available or self.client is None:
            return None
        if self._collection is None:
            try:
                self._collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
            except Exception as exc:
                logger.warning("创建/获取 ChromaDB collection 失败：%s", exc)
                return None
        return self._collection

    def _get_v2_collection(self) -> Any:
        """获取 v2 parameter collection（biodynamics_parameter，1607 records）。

        [Round 5] legacy collection（biodynamics_params, 752 records）仅包含 5 个
        BioModels 的参数，缺少 Apoptosis/CellCycle 等通路参数。v2 collection 包含
        70 个 SBML 文件的参数，覆盖全部 10 通路。
        """
        if not self._available or self.client is None:
            return None
        if self._v2_collection is None:
            try:
                self._v2_collection = self.client.get_or_create_collection(
                    name=self._v2_collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
            except Exception as exc:
                logger.warning("创建/获取 v2 ChromaDB collection 失败：%s", exc)
                return None
        return self._v2_collection

    def _get_evidence_collection(self) -> Any:
        """获取 biomodels_evidence collection（存放 PubMed 文献证据）。

        collection 名取自 settings.CHROMA_COLLECTION_EVIDENCE，与 RagCollections
        四路 evidence collection 保持一致。用于 search_by_pmids / build_pmid_vector_db。
        """
        if not self._available or self.client is None:
            return None
        try:
            return self.client.get_or_create_collection(
                name=settings.CHROMA_COLLECTION_EVIDENCE,
                metadata={"hnsw:space": "cosine", "role": "evidence"},
            )
        except Exception as exc:
            logger.warning("创建/获取 evidence ChromaDB collection 失败：%s", exc)
            return None

    def ensure_collection(self) -> bool:
        """确保 collection 存在。"""
        return self._get_collection() is not None

    def delete_collection(self) -> bool:
        """删除当前 collection；不存在也视为成功。"""
        if not self._available or self.client is None:
            return False
        try:
            self.client.delete_collection(name=self.collection_name)
            self._collection = None
            return True
        except Exception as exc:
            logger.warning("删除 ChromaDB collection 失败：%s", exc)
            return False

    def upsert_params(self, records: list[dict[str, Any]]) -> int:
        """批量写入参数记录，返回成功写入数量。"""
        collection = self._get_collection()
        if collection is None or not records:
            return 0

        ids: list[str] = []
        embeddings: list[list[float]] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for record in records:
            normalized = normalize_param(record)
            payload: dict[str, Any] = {}
            for key, value in normalized.items():
                # Chroma metadata 仅支持 str / int / float / bool
                if isinstance(value, (str, int, float, bool)):
                    payload[key] = value
                elif value is None:
                    payload[key] = ""
                else:
                    payload[key] = str(value)

            search_text = (
                f"{payload.get('param_name', '')} "
                f"{payload.get('context', '')} "
                f"{payload.get('species', '')} "
                f"{payload.get('cell_line', '')}"
            ).strip()
            if not search_text:
                search_text = str(payload.get("param_name", ""))

            try:
                vector = embedding_model.embed_query(search_text)
            except Exception as exc:
                logger.warning("Embedding 失败，跳过该条记录：%s", exc)
                continue

            ids.append(str(uuid.uuid4()))
            embeddings.append(vector)
            documents.append(search_text)
            metadatas.append(payload)

        if not ids:
            return 0

        try:
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            return len(ids)
        except Exception as exc:
            logger.warning("写入 ChromaDB 失败：%s", exc)
            return 0

    # -------------------------------------------------------------------------
    # 查询重写（Query Rewriting）
    # -------------------------------------------------------------------------
    def rewrite_query(
        self,
        raw_query: str,
        species_context: str = "Human",
    ) -> QueryRewriteOutput:
        """调用 LLM 对原始查询进行术语标准化与扩展。

        失败时安全降级为原始查询，不阻断主流程。
        """
        try:
            structured_llm = llm.with_structured_output(QueryRewriteOutput)
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", QUERY_REWRITING_PROMPT),
                    ("human", "请重写以下查询。"),
                ]
            )
            chain = prompt.partial(
                raw_query=raw_query,
                species_context=species_context,
            ) | structured_llm
            result: QueryRewriteOutput = chain.invoke({})
            return result
        except Exception as exc:
            logger.warning("查询重写失败，使用原始查询降级：%s", exc)
            return QueryRewriteOutput(
                rewritten_query=raw_query,
                rewrites=[],
                expanded_terms=[],
            )

    # -------------------------------------------------------------------------
    # 语义检索（Semantic Search）
    # -------------------------------------------------------------------------
    def semantic_search(
        self,
        query: str,
        top_k: int = 10,
        species_filter: str | None = None,
        type_filter: str | list[str] | None = None,
        collection: Any | None = None,
    ) -> list[dict[str, Any]]:
        """纯语义向量检索，返回带 distance 的候选记录。

        Args:
            type_filter: 参数类型过滤。可为字符串（精确匹配）或列表（任一匹配）。
                         例如 "kinetic_rate" 或 ["kinetic_rate", "binding_affinity"]。
                         设为 "exclude:initial_concentration" 可排除某类型。
            collection: [Round 5] 可选，指定搜索的 ChromaDB collection。
                        默认 None 使用 legacy collection（biodynamics_params）。
        """
        if collection is None:
            collection = self._get_collection()
        if collection is None:
            return []

        try:
            vector = embedding_model.embed_query(query)
            query_kwargs: dict[str, Any] = {
                "query_embeddings": [vector],
                "n_results": top_k,
                "include": ["metadatas", "documents", "distances"],
            }
            # 构建 where 子句：支持 species + type 联合过滤
            where_clauses: list[dict[str, Any]] = []
            if species_filter:
                where_clauses.append({"species": species_filter})
            if type_filter:
                if isinstance(type_filter, str) and type_filter.startswith("exclude:"):
                    # 排除模式：exclude:initial_concentration
                    exclude_type = type_filter[len("exclude:"):]
                    where_clauses.append({"type": {"$ne": exclude_type}})
                elif isinstance(type_filter, list):
                    # 列表模式：任一匹配
                    where_clauses.append({"type": {"$in": type_filter}})
                else:
                    # 精确匹配
                    where_clauses.append({"type": type_filter})
            if where_clauses:
                if len(where_clauses) == 1:
                    query_kwargs["where"] = where_clauses[0]
                else:
                    query_kwargs["where"] = {"$and": where_clauses}

            results = collection.query(**query_kwargs)
            metadatas = results.get("metadatas", [[]])[0]
            documents = results.get("documents", [[]])[0]
            distances = results.get("distances", [[]])[0]

            enriched: list[dict[str, Any]] = []
            for idx, meta in enumerate(metadatas):
                if not meta:
                    continue
                # cosine distance 越小越相似，转为 0-1 的相似度分数
                dist = distances[idx] if idx < len(distances) else 1.0
                similarity = max(0.0, 1.0 - float(dist))
                record = dict(meta)
                record["_document"] = documents[idx] if idx < len(documents) else ""
                record["_semantic_score"] = similarity
                record["_retrieval_method"] = "semantic"
                enriched.append(record)
            return enriched
        except Exception as exc:
            logger.warning("语义检索失败：%s", exc)
            return []

    # -------------------------------------------------------------------------
    # BM25 关键词检索
    # -------------------------------------------------------------------------
    def bm25_search(
        self,
        query: str,
        top_k: int = 10,
        type_filter: str | list[str] | None = None,
        collection: Any | None = None,
    ) -> list[dict[str, Any]]:
        """基于 BM25 的关键词检索，扫描 collection 全量文档并按词频打分。

        Args:
            type_filter: 参数类型过滤，同 semantic_search。
            collection: [Round 5] 可选，指定搜索的 ChromaDB collection。
        """
        if collection is None:
            collection = self._get_collection()
        if collection is None:
            return []

        try:
            # 拉取全量文档构建 BM25 索引（本地库规模可控）
            all_data = collection.get(include=["metadatas", "documents"])
            documents = all_data.get("documents", []) or []
            metadatas = all_data.get("metadatas", []) or []

            if not documents:
                return []

            # 按 type_filter 预过滤文档
            type_indices: list[int] = []
            if type_filter:
                for idx, meta in enumerate(metadatas):
                    if not meta:
                        continue
                    meta_type = str(meta.get("type", ""))
                    if isinstance(type_filter, str) and type_filter.startswith("exclude:"):
                        exclude_type = type_filter[len("exclude:"):]
                        if meta_type != exclude_type:
                            type_indices.append(idx)
                    elif isinstance(type_filter, list):
                        if meta_type in type_filter:
                            type_indices.append(idx)
                    else:
                        if meta_type == type_filter:
                            type_indices.append(idx)
            else:
                type_indices = list(range(len(documents)))

            if not type_indices:
                return []

            filtered_docs = [documents[i] for i in type_indices]
            filtered_metas = [metadatas[i] for i in type_indices]

            tokenized_docs = [_tokenize(doc) for doc in filtered_docs]
            bm25 = _BM25(tokenized_docs)
            query_terms = _tokenize(query)
            if not query_terms:
                return []

            hits = bm25.search(query_terms, top_k=top_k)
            max_score = max((s for _, s in hits), default=1.0)

            enriched: list[dict[str, Any]] = []
            for doc_idx, score in hits:
                meta = filtered_metas[doc_idx] if doc_idx < len(filtered_metas) else {}
                if not meta:
                    continue
                record = dict(meta)
                record["_document"] = filtered_docs[doc_idx]
                # 归一化 BM25 分数到 0-1
                record["_bm25_score"] = float(score)
                record["_semantic_score"] = (
                    float(score) / max_score if max_score > 0 else 0.0
                )
                record["_retrieval_method"] = "bm25"
                enriched.append(record)
            return enriched
        except Exception as exc:
            logger.warning("BM25 检索失败：%s", exc)
            return []

    # -------------------------------------------------------------------------
    # 混合检索（Hybrid Search）
    # -------------------------------------------------------------------------
    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        species_filter: str | None = None,
        type_filter: str | list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """混合检索：语义 + BM25 各取 top 10，按文档内容去重后合并。

        [Round 5] 同时搜索 legacy collection（biodynamics_params, 752 records）
        和 v2 collection（biodynamics_parameter, 1607 records），合并去重。
        v2 collection 包含 70 个 SBML 文件的参数，覆盖全部 10 通路（含 Apoptosis）。

        Args:
            type_filter: 参数类型过滤，同 semantic_search。默认 None（不过滤）。
        """
        # 1. Legacy collection 搜索
        semantic_results = self.semantic_search(
            query, top_k=top_k, species_filter=species_filter, type_filter=type_filter
        )
        bm25_results = self.bm25_search(query, top_k=top_k, type_filter=type_filter)

        # 2. [Round 5] v2 collection 搜索（覆盖 Apoptosis 等缺失通路）
        v2_collection = self._get_v2_collection()
        if v2_collection is not None:
            v2_semantic = self.semantic_search(
                query, top_k=top_k, species_filter=species_filter,
                type_filter=type_filter, collection=v2_collection,
            )
            v2_bm25 = self.bm25_search(
                query, top_k=top_k, type_filter=type_filter, collection=v2_collection,
            )
        else:
            v2_semantic = []
            v2_bm25 = []

        # 3. 合并去重（按 _document 内容键去重，同时命中取较高语义分）
        merged: dict[str, dict[str, Any]] = {}
        for record in semantic_results + bm25_results + v2_semantic + v2_bm25:
            doc_key = record.get("_document", "") or json.dumps(
                {k: v for k, v in record.items() if not k.startswith("_")},
                sort_keys=True,
                ensure_ascii=False,
            )
            if doc_key not in merged:
                merged[doc_key] = record
            else:
                # 同时命中两种方法时，取较高语义分数并标记 hybrid
                existing = merged[doc_key]
                existing["_semantic_score"] = max(
                    existing.get("_semantic_score", 0.0),
                    record.get("_semantic_score", 0.0),
                )
                existing["_retrieval_method"] = "hybrid"

        return list(merged.values())

    # -------------------------------------------------------------------------
    # 重排序（Re-ranking）
    # -------------------------------------------------------------------------
    def rerank_results(
        self,
        results: list[dict[str, Any]],
        species_context: str = "Human",
        top_k: int = 5,
        query: str | None = None,
    ) -> list[dict[str, Any]]:
        """按来源权威性、物种特异性、参数完整性、语义相关度综合重排序。

        当 settings.RERANK_PROVIDER 为 model/hybrid/openrouter 且传入 query 时，优先调用
        RerankManager 管理的模型（OpenRouter/SiliconFlow 等）获取模型级相关性分数；
        API 失败时自动降级到启发式规则。
        """
        if not results:
            return []

        # 收集模型 rerank 分数（若启用且可用）
        api_scores: dict[int, float] = {}
        selected_reranker: Any = None
        rerank_provider = settings.RERANK_PROVIDER.lower()
        # 兼容旧值 openrouter，统一视为 model 调用
        use_model_rerank = rerank_provider in ("openrouter", "model", "hybrid") and query is not None
        if use_model_rerank and rerank_manager is not None:
            try:
                documents = [_record_to_rerank_text(r) for r in results]
                api_results, selected_reranker = rerank_manager.rerank(
                    query=query,
                    documents=documents,
                    top_n=len(documents),
                )
                for item in api_results:
                    idx = int(item.get("index", -1))
                    score = float(item.get("relevance_score", 0.0))
                    if 0 <= idx < len(results):
                        api_scores[idx] = score
            except Exception as exc:
                logger.warning("模型 rerank 调用失败，降级到启发式重排：%s", exc)

        scored: list[tuple[float, dict[str, Any]]] = []
        for idx, record in enumerate(results):
            # 1. 来源权威性（0-1）
            source_type = _extract_source_type(record)
            authority = _SOURCE_AUTHORITY.get(source_type, 0.5)

            # 2. 物种特异性（0-1）
            record_species = str(record.get("species", ""))
            species_score = _SPECIES_PRIORITY.get(record_species, 0.5)
            if record_species and species_context and record_species.lower() == species_context.lower():
                species_score = 1.0  # 完全匹配加满

            # 3. 参数完整性（0-1）
            confidence = record.get("confidence", "MEDIUM")
            completeness = _confidence_to_score(confidence)
            value = record.get("value")
            if value is None or str(value).strip() == "":
                completeness *= 0.3  # 缺数值大幅降权

            # 4. 语义/检索相关度（0-1）
            relevance = float(record.get("_semantic_score", 0.0))

            # 5. OpenRouter rerank 模型相关性分数（0-1）
            api_relevance = api_scores.get(idx, 0.0)

            if rerank_provider in ("openrouter", "model") and api_relevance > 0:
                # 纯模型重排：以 API 分数为主，辅以少量规则修正
                final_score = api_relevance * 0.85 + authority * 0.10 + species_score * 0.05
            elif rerank_provider == "hybrid" and api_relevance > 0:
                # 混合策略：API 分数与启发式规则融合
                heuristic_score = (
                    authority * 0.30
                    + species_score * 0.25
                    + completeness * 0.20
                    + relevance * 0.25
                )
                final_score = api_relevance * 0.55 + heuristic_score * 0.45
            else:
                # 默认启发式规则
                final_score = (
                    authority * 0.30
                    + species_score * 0.25
                    + completeness * 0.20
                    + relevance * 0.25
                )

            record["_rerank_score"] = final_score
            record["_source_type"] = source_type
            record["_rerank_api_score"] = api_relevance
            if selected_reranker is not None:
                record["_reranker"] = {
                    "provider": selected_reranker.provider,
                    "model": selected_reranker.model,
                    "display_name": selected_reranker.display_name,
                }
            scored.append((final_score, record))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:top_k]]

    # -------------------------------------------------------------------------
    # 高阶检索入口（供 Node 1.5 调用）
    # -------------------------------------------------------------------------
    def search_params_hybrid(
        self,
        query: str,
        species_context: str = "Human",
        top_k: int = 5,
        type_filter: str | list[str] | None = "exclude:initial_concentration",
        pathway: str | None = None,
        prefer_biomd_id: str | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """高阶 RAG 检索入口：查询重写 → 混合检索 → 重排序。

        Args:
            type_filter: 参数类型过滤，默认排除 initial_concentration（强制 type system）。
                         可设为 None 关闭过滤，或设为 ["kinetic_rate", "binding_affinity"] 精确过滤。
            pathway: V4 Enhancement 专用——通路标识（如 "egfr"）。提供此参数且
                     V4_SEQUENTIAL_RETRIEVER=true 时优先走顺序优先级检索
                     （canonical → BioModels → PubMed → Reactome/KEGG）；
                     未提供或 Flag=false 时走原并行混合检索（不影响 v3/v4 行为）。
            prefer_biomd_id: N7 缺口 3——同源优先。提供 BioModels ID（如
                     ``"BIOMD0000000048"``）时，rerank 后对 biomd_id 匹配的候选
                     加权 ×2.0 并重排序，使同源参数优先返回，避免跨模型混用。
                     None 时关闭同源优先（保持原行为）。

        返回 (重排序后的 top_k 结果, RAG 洞察数据)。
        洞察数据包含 rewritten_query / rewrites / source_distribution / total_candidates。
        """
        # === V4 Enhancement: Feature Flag 分支（默认 OFF，不影响 v3/v4）===
        # V4_SEQUENTIAL_RETRIEVER=true 且调用方提供 pathway 时，优先走顺序检索
        # （canonical.yaml → BioModels → PubMed → Reactome/KEGG）。
        # RAG_LEGACY_PARALLEL=true 或 V4_SEQUENTIAL_RETRIEVER=false 时回退到
        # 原并行混合检索路径（保持 v3/v4 行为）。
        # 铁律：pathway=None 时本分支不触发，与 v3/v4 完全一致。
        if (
            pathway is not None
            and settings.V4_SEQUENTIAL_RETRIEVER
            and not settings.RAG_LEGACY_PARALLEL
        ):
            reranked, insights = self._search_via_sequential_retriever(
                pathway=pathway,
                query=query,
                species_context=species_context,
                top_k=top_k,
            )
            # N7 缺口 3：同源优先（顺序检索分支同样适用）
            if prefer_biomd_id and reranked:
                reranked = self._apply_same_source_boost(reranked, prefer_biomd_id)
                insights["same_source_preferred"] = prefer_biomd_id
            return reranked, insights

        if not self._available:
            return [], {
                "rewritten_query": query,
                "rewrites": [],
                "source_distribution": {},
                "total_candidates": 0,
                "top_selections": [],
            }

        # 1. 查询重写
        rewrite_output = self.rewrite_query(query, species_context)
        rewritten_query = rewrite_output.rewritten_query or query

        # 2. 混合检索（各取 top 10），按 type_filter 过滤
        candidates = self.hybrid_search(
            rewritten_query, top_k=10, type_filter=type_filter
        )

        # 3. 重排序
        reranked = self.rerank_results(
            candidates,
            species_context=species_context,
            top_k=top_k,
            query=rewritten_query,
        )

        # N7 缺口 3：同源优先——biomd_id 匹配的候选加权 ×2.0 并重排序
        if prefer_biomd_id and reranked:
            reranked = self._apply_same_source_boost(reranked, prefer_biomd_id)

        # 4. 构建洞察数据
        source_counter: Counter[str] = Counter()
        for c in candidates:
            source_counter[_extract_source_type(c)] += 1

        top_selections: list[dict[str, Any]] = []
        for rec in reranked:
            # PMID 提取：优先读 source_pmid / pmid 字段，再从 source 中正则抽取
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

        insights = {
            "rewritten_query": rewritten_query,
            "rewrites": [r.model_dump() for r in rewrite_output.rewrites],
            "expanded_terms": rewrite_output.expanded_terms,
            "source_distribution": dict(source_counter),
            "total_candidates": len(candidates),
            "top_selections": top_selections,
        }
        if prefer_biomd_id:
            insights["same_source_preferred"] = prefer_biomd_id

        return reranked, insights

    # -------------------------------------------------------------------------
    # N7 缺口 3：同源优先加权（确定性，无 LLM）
    # -------------------------------------------------------------------------
    _BIOMD_ID_RE = re.compile(r"\b(BIOMD\d{10}|MODEL\d{10})\b", re.IGNORECASE)

    @classmethod
    def _record_biomd_id(cls, record: dict[str, Any]) -> str | None:
        """从检索记录提取 BioModels ID（biomd_id / source_model / source 正则）。"""
        explicit = record.get("biomd_id")
        if explicit and str(explicit).strip():
            return str(explicit).strip().upper()
        source_model = record.get("source_model")
        if source_model and str(source_model).strip():
            sm = str(source_model).strip()
            match = cls._BIOMD_ID_RE.search(sm)
            if match:
                return match.group(1).upper()
            if sm.upper().startswith(("BIOMD", "MODEL")):
                return sm.upper()
            return sm
        for field_name in ("source", "origin"):
            value = record.get(field_name)
            if not value:
                continue
            match = cls._BIOMD_ID_RE.search(str(value))
            if match:
                return match.group(1).upper()
        return None

    def _apply_same_source_boost(
        self,
        reranked: list[dict[str, Any]],
        prefer_biomd_id: str,
    ) -> list[dict[str, Any]]:
        """对 biomd_id 匹配 prefer_biomd_id 的候选加权 ×2.0 并重排序。

        确定性操作：仅调整 _rerank_score 与顺序，不改变候选内容。
        匹配判定（大小写不敏感）：record.biomd_id / source_model / source/origin
        正则提取任一等于 prefer_biomd_id 即视为同源。
        """
        target = str(prefer_biomd_id).strip().upper()
        if not target or not reranked:
            return reranked
        for rec in reranked:
            base_score = float(rec.get("_rerank_score", 0.0))
            rec_biomd = self._record_biomd_id(rec)
            is_same_source = rec_biomd is not None and rec_biomd.upper() == target
            if is_same_source:
                # 同源加权 ×2.0，截断到 1.0
                rec["_rerank_score"] = min(1.0, base_score * 2.0)
                rec["_same_source_boosted"] = True
            else:
                rec["_same_source_boosted"] = False
        # 按 _rerank_score 降序重排（确定性，无随机）
        reranked.sort(key=lambda r: float(r.get("_rerank_score", 0.0)), reverse=True)
        return reranked

    # -------------------------------------------------------------------------
    # 4-collection 上下文富化（修复提示词1.md §5.1.1）
    # -------------------------------------------------------------------------
    def search_params_hybrid_with_context(
        self,
        query: str,
        species_context: str = "Human",
        top_k: int = 5,
        type_filter: str | list[str] | None = "exclude:initial_concentration",
        include_mechanism: bool = True,
        include_evidence: bool = True,
        rag_collections: Any = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """4-collection 高阶 RAG：参数检索 + 机制上下文 + 文献证据富化。

        对应修复提示词1.md §5.1.1：单 collection 查询改为 4 路 collection 联合查询。
        在原 search_params_hybrid 基础上，附加：
        1. mechanism collection 查询：为参数提供通路/机制上下文；
        2. evidence collection 查询：为参数提供文献证据支撑；
        3. 把机制/证据命中合并到 insights 的 mechanism_context / evidence_context 字段。

        Args:
            query: 参数查询文本（如 "EGF EGFR binding k_on"）。
            species_context: 物种上下文（默认 Human）。
            top_k: 参数 top-k 数量。
            type_filter: 参数类型过滤（同 search_params_hybrid）。
            include_mechanism: 是否富化机制上下文（默认 True）。
            include_evidence: 是否富化文献证据（默认 True）。
            rag_collections: 可选的 RagCollections 实例；None 时懒加载。

        Returns:
            (reranked 参数结果, 富化后的 insights)，insights 多出 mechanism_context
            与 evidence_context 两个字段。
        """
        # 1. 基础参数检索
        reranked, insights = self.search_params_hybrid(
            query=query,
            species_context=species_context,
            top_k=top_k,
            type_filter=type_filter,
        )

        # 2. 机制上下文富化
        mechanism_context: list[dict[str, Any]] = []
        if include_mechanism:
            try:
                if rag_collections is None:
                    from app.rag_collections import get_rag_collections
                    rag_collections = get_rag_collections()
                if rag_collections.available:
                    mechanism_context = rag_collections.search_mechanism(
                        query, top_k=3
                    )
                    # 剥离内部字段
                    mechanism_context = [
                        {k: v for k, v in r.items() if not k.startswith("_")}
                        for r in mechanism_context
                    ]
            except Exception as exc:
                logger.warning("4-collection 机制上下文富化失败：%s", exc)
                mechanism_context = []

        # 3. 文献证据富化
        evidence_context: list[dict[str, Any]] = []
        if include_evidence:
            try:
                if rag_collections is None:
                    from app.rag_collections import get_rag_collections
                    rag_collections = get_rag_collections()
                if rag_collections.available:
                    evidence_context = rag_collections.search_evidence(
                        query, top_k=3
                    )
                    # 剥离内部字段
                    evidence_context = [
                        {k: v for k, v in r.items() if not k.startswith("_")}
                        for r in evidence_context
                    ]
            except Exception as exc:
                logger.warning("4-collection 文献证据富化失败：%s", exc)
                evidence_context = []

        # 4. 合并到 insights
        insights["mechanism_context"] = mechanism_context
        insights["evidence_context"] = evidence_context
        insights["collection_coverage"] = {
            "parameter": len(reranked),
            "mechanism": len(mechanism_context),
            "evidence": len(evidence_context),
        }

        return reranked, insights

    # -------------------------------------------------------------------------
    # 兼容旧接口：纯语义检索（保持向后兼容）
    # -------------------------------------------------------------------------
    def search_params(
        self,
        query: str,
        top_k: int = 5,
        species_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """根据查询文本检索参数记录；失败时返回空列表，不影响主工作流。

        保留此方法以兼容已有调用方；内部优先走混合检索。
        """
        results, _ = self.search_params_hybrid(
            query,
            species_context=species_filter or "Human",
            top_k=top_k,
        )
        # 剥离内部字段，返回干净元数据
        clean: list[dict[str, Any]] = []
        for rec in results:
            clean.append({
                k: v for k, v in rec.items() if not k.startswith("_")
            })
        return clean

    # -------------------------------------------------------------------------
    # PubMed E-utilities: 按 PMID 拉取元数据（缺口 3）
    # 原逻辑来自 nodes_v2.py:_fetch_pubmed_evidence_sync，迁出为可复用私有方法。
    # 既能按 PMID 直接 efetch，也能按 query esearch+efetch。
    # 失败返回空 list，不抛异常。
    # -------------------------------------------------------------------------
    def _fetch_pubmed_by_pmids(
        self,
        pmids: list[str],
        *,
        query: str | None = None,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """按 PMID 列表调用 NCBI E-utilities efetch 获取文献元数据。

        Args:
            pmids: PMID 列表（已知的 PMID，直接走 efetch）。
            query: 可选——若提供且 pmids 为空，则走 esearch 检索 top_k 个 PMID。
                   兼容原 _fetch_pubmed_evidence_sync 的 query-based 调用语义。
            top_k: esearch 返回的 PMID 数量上限（默认 10）。

        Returns:
            list[dict]，每个 dict 字段: pmid / title / abstract / source /
            figure_ref / cell_line / authors / journal / pub_year / mesh_terms /
            source_role="PubMed"。失败返回空 list。
        """
        if not pmids and not query:
            return []

        # 读取 NCBI 凭据（容错，settings 缺字段时降级为空串）
        try:
            ncbi_email = getattr(settings, "NCBI_EMAIL", "") or ""
            ncbi_api_key = getattr(settings, "NCBI_API_KEY", "") or ""
        except Exception:
            ncbi_email = ""
            ncbi_api_key = ""

        # === 步骤 1: 确定 PMID 列表 ===
        id_list: list[str] = [str(p).strip() for p in pmids if p]
        if not id_list and query:
            # esearch 获取 PMID 列表
            esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            esearch_params: dict[str, Any] = {
                "db": "pubmed",
                "term": query[:200],
                "retmax": str(top_k),
                "retmode": "json",
                "sort": "relevance",
            }
            if ncbi_email:
                esearch_params["email"] = ncbi_email
                esearch_params["tool"] = "BioDynamicsAgent"
            if ncbi_api_key:
                esearch_params["api_key"] = ncbi_api_key
            try:
                resp = requests.get(esearch_url, params=esearch_params, timeout=10)
                resp.raise_for_status()
                id_list = (
                    resp.json().get("esearchresult", {}).get("idlist", []) or []
                )
                logger.info(
                    "PubMed esearch 成功: query=%s idlist=%d",
                    query[:80], len(id_list),
                )
            except Exception as exc:
                logger.warning(
                    "PubMed esearch 失败：%s (query=%s)", exc, query[:80]
                )
                return []

        if not id_list:
            return []

        # === 步骤 2: efetch 获取文献详情（最多 2 次重试，间隔 1s） ===
        efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        efetch_params: dict[str, Any] = {
            "db": "pubmed",
            "id": ",".join(id_list),
            "retmode": "xml",
        }
        if ncbi_email:
            efetch_params["email"] = ncbi_email
            efetch_params["tool"] = "BioDynamicsAgent"
        if ncbi_api_key:
            efetch_params["api_key"] = ncbi_api_key

        resp = None
        for attempt in range(3):
            try:
                resp = requests.get(efetch_url, params=efetch_params, timeout=30)
                resp.raise_for_status()
                break
            except Exception as exc:
                if attempt < 2:
                    logger.warning(
                        "PubMed efetch 第 %d 次失败（共 3 次）：%s，1s 后重试",
                        attempt + 1, exc,
                    )
                    time.sleep(1.0)
                else:
                    logger.warning("PubMed efetch 3 次重试均失败：%s", exc)
                    return []
        if resp is None:
            return []

        # === 步骤 3: 解析 XML（保留原有解析逻辑 + 扩展 authors/journal/year/mesh） ===
        articles: list[dict[str, Any]] = []
        try:
            root = _ET.fromstring(resp.text)
        except _ET.ParseError as exc:
            logger.warning("PubMed XML 解析失败：%s", exc)
            return []

        for article_elem in root.findall(".//PubmedArticle"):
            pmid_elem = article_elem.find(".//PMID")
            pmid = pmid_elem.text if pmid_elem is not None and pmid_elem.text else ""
            title_elem = article_elem.find(".//ArticleTitle")
            title = (
                "".join(title_elem.itertext()).strip()
                if title_elem is not None else ""
            )
            abstract_parts = article_elem.findall(".//AbstractText")
            abstract = " ".join(
                "".join(p.itertext()) for p in abstract_parts
            )[:2000]

            # 扩展字段：authors / journal / pub_year / mesh_terms / pub_types
            authors: list[str] = []
            for au in article_elem.findall(".//Author"):
                last = au.findtext("LastName") or ""
                fore = au.findtext("ForeName") or ""
                full = (fore + " " + last).strip() or (au.findtext("CollectiveName") or "")
                if full:
                    authors.append(full)
            journal = (
                article_elem.findtext(".//Journal/Title") or ""
            )
            pub_year = (
                article_elem.findtext(".//PubDate/Year")
                or article_elem.findtext(".//PubDate/MedlineDate") or ""
            )[:4]
            mesh_terms: list[str] = []
            for mh in article_elem.findall(".//MeshHeading/DescriptorName"):
                if mh.text:
                    mesh_terms.append(mh.text)
            pub_types = [
                "".join(item.itertext()).strip()
                for item in article_elem.findall(".//PublicationTypeList/PublicationType")
                if "".join(item.itertext()).strip()
            ]

            if pmid:
                articles.append({
                    "pmid": pmid,
                    "title": title or "",
                    "abstract": abstract,
                    "source": f"PMID:{pmid}",
                    "figure_ref": "",
                    "cell_line": "",
                    "authors": authors,
                    "journal": journal,
                    "pub_year": pub_year,
                    "mesh_terms": mesh_terms,
                    "pub_types": pub_types,
                    "source_role": "PubMed",
                })

        if articles:
            logger.info(
                "PubMed efetch 命中 %d 篇文献 (requested=%d)",
                len(articles), len(id_list),
            )
        return articles[:top_k] if query and not pmids else articles

    # -------------------------------------------------------------------------
    # 按 PMID 强制查询（缺口 1）
    # 先查 ChromaDB evidence collection，未命中再调 PubMed E-utilities 在线拉取。
    # 网络失败时返回 degraded 状态，不抛异常。
    # -------------------------------------------------------------------------
    def search_by_pmids(
        self,
        pmids: list[str],
        top_k: int = 10,
    ) -> dict[str, Any]:
        """按 PMID 列表强制查询文献证据。

        检索顺序:
          1. ChromaDB biomodels_evidence collection 按 metadata.pmid 过滤
          2. 向量库未命中的 PMID 调用 PubMed E-utilities 在线 efetch

        Args:
            pmids: PMID 列表（如 ["10959078", "11239472"]）。
            top_k: 在线 fallback 时每个 PMID 拉取的文献数上限（默认 10）。

        Returns:
            dict 结构:
              {
                "results": [
                    {"pmid", "title", "abstract", "source_role",
                     "snippet", "found_in_vector_db": bool,
                     "fetched_online": bool (optional)},
                    ...
                ],
                "degraded_stages": ["vector_db", "pubmed_online"],  # 退化的 stage
                "errors": [{"pmid": str, "error": str}, ...]
              }
            网络失败时返回 {results: [], degraded_stages: [...], errors: [...]}，
            不抛异常。
        """
        results: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        degraded_stages: list[str] = []

        if not pmids:
            return {
                "results": [],
                "degraded_stages": [],
                "errors": [],
            }

        normalized_pmids: list[str] = [str(p).strip() for p in pmids if p]

        # === Step 1: ChromaDB 向量库过滤查询 ===
        vector_hits: dict[str, dict[str, Any]] = {}
        evidence_coll = self._get_evidence_collection()
        if evidence_coll is None:
            degraded_stages.append("vector_db")
        else:
            try:
                got = evidence_coll.get(
                    where={"pmid": {"$in": normalized_pmids}},
                    include=["metadatas", "documents"],
                )
                metas = got.get("metadatas", []) or []
                docs = got.get("documents", []) or []
                for idx, meta in enumerate(metas):
                    if not meta:
                        continue
                    pmid = str(meta.get("pmid", "")).strip()
                    if not pmid:
                        continue
                    # 多条同 PMID 时保留第一条（去重）
                    if pmid in vector_hits:
                        continue
                    doc = docs[idx] if idx < len(docs) else ""
                    title = str(meta.get("title", ""))
                    abstract = str(meta.get("abstract", ""))
                    snippet = (doc or abstract or title)[:500]
                    # source_role 规范化：任务规范要求值为 PubMed / BioModels / Reactome 之一。
                    # metadata 中可能有非标准值（如 classical_reviews / mechanism_papers），
                    # 统一映射到三个标准值之一。
                    source_role_raw = str(meta.get("source_role", "")).strip()
                    source_raw = str(meta.get("source", "")).upper()
                    if "BIOMODELS" in source_raw or "BIOMD" in source_raw:
                        source_role = "BioModels"
                    elif "REACTOME" in source_raw:
                        source_role = "Reactome"
                    elif source_role_raw in ("PubMed", "BioModels", "Reactome"):
                        source_role = source_role_raw
                    else:
                        # 非标准值或空 → 默认 PubMed（PMID 来源都是 PubMed 文献）
                        source_role = "PubMed"
                    vector_hits[pmid] = {
                        "pmid": pmid,
                        "title": title,
                        "abstract": abstract,
                        "source_role": source_role,
                        "snippet": snippet,
                        "found_in_vector_db": True,
                        "fetched_online": False,
                    }
            except Exception as exc:
                logger.warning(
                    "search_by_pmids ChromaDB 过滤查询失败：%s", exc
                )
                degraded_stages.append("vector_db")

        results.extend(vector_hits.values())

        # === Step 2: 未命中的 PMID 走 PubMed E-utilities 在线拉取 ===
        missing_pmids = [p for p in normalized_pmids if p not in vector_hits]
        if missing_pmids:
            try:
                # 总超时 30s 由 efetch 内部 timeout=30 保证；这里对单 PMID
                # 不再额外切片（efetch 支持批量）。把 missing_pmids 一次性传入。
                online_articles = self._fetch_pubmed_by_pmids(
                    missing_pmids, top_k=top_k
                )
            except Exception as exc:
                logger.warning(
                    "search_by_pmids PubMed 在线拉取失败：%s", exc
                )
                online_articles = []
                degraded_stages.append("pubmed_online")
                for p in missing_pmids:
                    errors.append({"pmid": p, "error": str(exc)})

            for art in online_articles:
                pmid = str(art.get("pmid", "")).strip()
                if not pmid or pmid in vector_hits:
                    continue
                title = str(art.get("title", ""))
                abstract = str(art.get("abstract", ""))
                snippet = (abstract or title)[:500]
                results.append({
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract,
                    "source_role": "PubMed",
                    "snippet": snippet,
                    "found_in_vector_db": False,
                    "fetched_online": True,
                })

            # 在线拉取后仍缺失的 PMID 记录为 error
            online_found_pmids = {
                str(a.get("pmid", "")).strip() for a in online_articles
            }
            for p in missing_pmids:
                if p not in online_found_pmids and not any(
                    e.get("pmid") == p for e in errors
                ):
                    errors.append({
                        "pmid": p,
                        "error": "PubMed efetch returned no record",
                    })

        return {
            "results": results,
            "degraded_stages": degraded_stages,
            "errors": errors,
        }

    # -------------------------------------------------------------------------
    # 药物特定检索（Drug-specific Retriever）
    # -------------------------------------------------------------------------
    def drug_specific_retriever(
        self,
        target_name: str,
        species_context: str = "Human",
        pubmed_articles: list[dict] | None = None,
    ) -> list[dict]:
        """检索针对 target_name 的药物候选信息（IC50/EC50、临床剂量、临床试验）。

        数据来源：
        1. PubMed 文献片段（LLM 提取药物名与参数）
        2. ChromaDB 本地混合检索（补充 IC50/EC50）
        3. ClinicalTrials.gov（验证是否进入临床）

        返回结构：
        [{drug_name, ic50, ec50, clinical_dose, source, is_clinical_candidate, target_name, clinical_trial_info}]
        """
        candidates: dict[str, dict] = {}

        # 1. 从 PubMed 文章提取药物候选
        if pubmed_articles:
            structured_llm = llm.with_structured_output(_DrugExtractionOutput)
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", DRUG_EXTRACTION_PROMPT),
                    ("human", "文献片段：\n{article_text}"),
                ]
            )
            for article in pubmed_articles:
                text = self._article_to_text(article)
                if not text:
                    continue
                try:
                    chain = prompt.partial(
                        target_name=target_name,
                        species_context=species_context,
                    ) | structured_llm
                    result: _DrugExtractionOutput = chain.invoke(
                        {"article_text": text[:4000]}
                    )
                    for cand in result.drug_candidates:
                        name = cand.drug_name.strip()
                        if not name:
                            continue
                        record = cand.model_dump()
                        record["target_name"] = target_name
                        # 确保来源包含 PMID 标识，供后续提取逻辑使用
                        if not record.get("source"):
                            article_source = article.get("source", "")
                            article_pmid = article.get("pmid", "")
                            if article_pmid:
                                record["source"] = f"PMID:{article_pmid}"
                            elif article_source:
                                record["source"] = article_source
                            else:
                                record["source"] = "PubMed"
                        candidates[name] = self._merge_candidate(
                            candidates.get(name), record
                        )
                except Exception as exc:
                    logger.warning("从 PubMed 提取药物候选失败：%s", exc)

        # 2. 从 ChromaDB 检索每个候选药物的 IC50/EC50
        if self._available:
            for name in list(candidates.keys()):
                try:
                    query = f"{name} {target_name} inhibitor IC50 EC50"
                    retrieved, _ = self.search_params_hybrid(
                        query, species_context=species_context, top_k=5
                    )
                    for rec in retrieved:
                        param_name = str(rec.get("param_name", "")).lower()
                        if "ic50" not in param_name and "ec50" not in param_name:
                            continue
                        value = rec.get("value")
                        if value is None:
                            continue
                        try:
                            numeric_value = float(value)
                        except (TypeError, ValueError):
                            continue
                        existing = candidates[name]
                        if "ic50" in param_name:
                            existing["ic50"] = existing.get("ic50") or numeric_value
                        elif "ec50" in param_name:
                            existing["ec50"] = existing.get("ec50") or numeric_value
                        existing["source"] = str(
                            rec.get("source", existing.get("source", "RAG"))
                        )
                except Exception as exc:
                    logger.warning("ChromaDB 检索 %s 参数失败：%s", name, exc)

        # 3. 查询 ClinicalTrials.gov，判断临床候选状态
        final_candidates: list[dict] = []
        for name, cand in candidates.items():
            trials = _query_clinical_trials(name)
            if trials:
                cand["is_clinical_candidate"] = True
                cand["clinical_trial_info"] = trials
            else:
                cand.setdefault("is_clinical_candidate", False)
            final_candidates.append(cand)

        return final_candidates

    @staticmethod
    def _article_to_text(article: dict) -> str:
        """将 PubMed 文章字典拼接为可提取的文本。"""
        parts: list[str] = []
        title = article.get("title") or ""
        abstract = article.get("abstract") or ""
        if title:
            parts.append(f"Title: {title}")
        if abstract:
            parts.append(f"Abstract: {abstract}")
        return "\n".join(parts)

    @staticmethod
    def _merge_candidate(existing: dict | None, new: dict) -> dict:
        """合并两条药物候选记录，优先保留非空字段。"""
        if existing is None:
            return new
        merged = dict(existing)
        for key, value in new.items():
            if value is not None and value != "":
                merged[key] = value
        return merged

    def _get_vector_size(self) -> int:
        """获取当前 embedding 模型的向量维度。"""
        try:
            sample = embedding_model.embed_query("dimension test")
            return len(sample)
        except Exception as exc:
            logger.warning("获取 embedding 维度失败，使用默认值 1536：%s", exc)
            return 1536

    # -------------------------------------------------------------------------
    # V4 Enhancement: Sequential Retriever 入口（Task 11 集成）
    # -------------------------------------------------------------------------
    def sequential_retrieve(
        self,
        pathway: str,
        query: str,
        *,
        top_k: int = 10,
        canonical_dir: str | Path | None = None,
        biomodels_client: Any = None,
    ) -> Any:
        """实例方法入口：执行顺序优先级检索。

        委托给模块级 ``sequential_retrieve()``，自动传入 ``self`` 作为
        ``rag_client`` 参数（供 PubMed stage 检索使用）。

        Feature Flag:
            V4_SEQUENTIAL_RETRIEVER=true 时启用顺序检索
            （canonical.yaml → BioModels → PubMed → Reactome/KEGG）。
            RAG_LEGACY_PARALLEL=true 或 V4_SEQUENTIAL_RETRIEVER=false 时
            调用方应回退到 ``search_params_hybrid`` 旧并行检索。

        Args:
            pathway: 通路名称（如 ``"egfr"``）。
            query: 检索查询语句（用于 PubMed stage）。
            top_k: 返回 Top-K 结果（默认 10）。
            canonical_dir: Canonical YAML 目录。None 时使用默认目录。
            biomodels_client: BioModels API 客户端。None 时由底层默认创建。

        Returns:
            ``RetrievalResult``，含 ``staged_evidence`` / ``final_ranked``
            / ``logs`` / ``canonical_hit_rate`` 字段。
        """
        return sequential_retrieve(
            pathway=pathway,
            query=query,
            top_k=top_k,
            canonical_dir=canonical_dir,
            rag_client=self,
            biomodels_client=biomodels_client,
        )

    def _search_via_sequential_retriever(
        self,
        *,
        pathway: str,
        query: str,
        species_context: str,
        top_k: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """通过 sequential_retrieve 执行检索并适配为 search_params_hybrid 返回格式。

        将 ``RetrievalResult.final_ranked`` 中的 ``Evidence`` 对象转换为
        ``search_params_hybrid`` 期望的 ``list[dict]`` 格式，并构建包含
        sequential retriever 日志与命中率的 insights dict。

        失败时记录 warning 并返回空结果（保持与 ``search_params_hybrid``
        在 ``self._available=False`` 时的相同降级行为）。

        Args:
            pathway: 通路标识。
            query: 检索查询语句。
            species_context: 物种上下文（用于 top_selections 字段）。
            top_k: 返回 Top-K 结果数。

        Returns:
            ``(reranked 参数列表, insights dict)``，insights 多出
            ``sequential_retriever_logs`` 与 ``canonical_hit_rate`` 字段。
        """
        try:
            result = sequential_retrieve(
                pathway=pathway,
                query=query,
                top_k=top_k,
                rag_client=self,
            )
        except Exception as exc:  # noqa: BLE001 —— 顺序检索失败时降级为空结果
            logger.warning(
                "sequential_retrieve 调用失败，返回空结果（pathway=%s）：%s",
                pathway,
                exc,
            )
            return [], {
                "rewritten_query": query,
                "rewrites": [],
                "source_distribution": {},
                "total_candidates": 0,
                "top_selections": [],
                "sequential_retriever_error": str(exc),
            }

        # 将 Evidence 列表转换为 search_params_hybrid 兼容的 dict 列表
        reranked: list[dict[str, Any]] = []
        for ev in result.final_ranked:
            record: dict[str, Any] = dict(ev.raw) if ev.raw else {}
            record["param_name"] = ev.title
            record["source"] = ev.source
            if ev.pmid:
                record["pmid"] = ev.pmid
                record["source_pmid"] = ev.pmid
            if ev.biomd_id:
                record["biomd_id"] = ev.biomd_id
            record["_semantic_score"] = float(ev.relevance_score)
            record["_rerank_score"] = float(ev.relevance_score)
            record["_retrieval_method"] = "sequential"
            record["_evidence_type"] = ev.evidence_type
            reranked.append(record)

        # 构建 insights（含 sequential retriever 专属字段）
        source_counter: Counter[str] = Counter(
            ev.source for ev in result.final_ranked
        )
        top_selections: list[dict[str, Any]] = [
            {
                "parameter": ev.title,
                "value": "",
                "source": ev.source,
                "pmid": ev.pmid or "",
                "confidence_score": round(float(ev.relevance_score), 2),
                "species": species_context,
                "context": "",
            }
            for ev in result.final_ranked
        ]

        insights: dict[str, Any] = {
            "rewritten_query": query,
            "rewrites": [],
            "source_distribution": dict(source_counter),
            "total_candidates": len(result.final_ranked),
            "top_selections": top_selections,
            "sequential_retriever_logs": [
                {
                    "stage": log.stage,
                    "query": log.query,
                    "returned": log.returned,
                    "selected": log.selected,
                    "deduplicated": log.deduplicated,
                }
                for log in result.logs
            ],
            "canonical_hit_rate": float(result.canonical_hit_rate),
        }

        return reranked, insights


# -----------------------------------------------------------------------------
# V4 Enhancement: 模块级 Sequential Retriever 入口（Task 11 集成）
# -----------------------------------------------------------------------------
def sequential_retrieve(
    pathway: str,
    query: str,
    *,
    top_k: int = 10,
    canonical_dir: str | Path | None = None,
    rag_client: Any = None,
    biomodels_client: Any = None,
) -> Any:
    """顺序优先级检索模块级入口（委托给 scientific_alignment.sequential_retriever）。

    对应 Spec Requirement "Sequential Retriever（顺序优先级检索）"。
    四级顺序检索：canonical.yaml → BioModels → PubMed → Reactome/KEGG。
    前一级命中足量证据时，后一级不执行（或仅补充）。

    Feature Flag:
        V4_SEQUENTIAL_RETRIEVER=true 时启用顺序检索。
        RAG_LEGACY_PARALLEL=true 时由调用方决定是否回退到旧并行检索
        （本函数本身不检查 RAG_LEGACY_PARALLEL，由 search_params_hybrid
        统一控制路由）。

    Args:
        pathway: 通路名称（如 ``"egfr"``），用于加载 canonical.yaml。
        query: 检索查询语句（用于 PubMed stage）。
        top_k: 返回 Top-K 结果（默认 10）。
        canonical_dir: Canonical YAML 目录。None 时使用默认目录。
        rag_client: PubMed 检索客户端。None 时底层创建默认客户端；
            传入 ``RagClient`` 实例可复用 ChromaDB 连接。
        biomodels_client: BioModels API 客户端。None 时使用底层默认。

    Returns:
        ``RetrievalResult``，含 ``staged_evidence`` / ``final_ranked``
        / ``logs`` / ``canonical_hit_rate`` 字段。
    """
    global _sequential_retrieve_impl_lazy
    if _sequential_retrieve_impl_lazy is None:
        from app.scientific_alignment.sequential_retriever import (
            sequential_retrieve as _sequential_retrieve_impl,
        )
        _sequential_retrieve_impl_lazy = _sequential_retrieve_impl
    return _sequential_retrieve_impl_lazy(
        pathway=pathway,
        query=query,
        top_k=top_k,
        canonical_dir=canonical_dir,
        rag_client=rag_client,
        biomodels_client=biomodels_client,
    )
