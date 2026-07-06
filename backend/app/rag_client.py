# BioDynamics Agent - 高阶 RAG 客户端
# 对应 1233.md 第二部分规范：查询重写 + 混合检索（语义 + BM25）+ 重排序。
# 负责 ChromaDB 本地持久化连接、动力学参数写入与语义检索，并提供单位归一化辅助函数。

import json
import logging
import math
import re
import uuid
from collections import Counter
from typing import Any

import chromadb
import requests
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.config import embedding_model, llm, rerank_manager, settings
from app.prompts import QUERY_REWRITING_PROMPT

logger = logging.getLogger(__name__)


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
        studies = data.get("studies", [])
        results: list[dict] = []
        for study in studies:
            protocol = study.get("protocolSection", {})
            identification = protocol.get("identification", {})
            status_module = protocol.get("statusModule", {})
            conditions_module = protocol.get("conditionsModule", {})
            nct_id = identification.get("nctId", "")
            phase = status_module.get("phase", "Unknown")
            if isinstance(phase, list):
                phase = "/".join(phase)
            status = status_module.get("overallStatus", "Unknown")
            conditions = conditions_module.get("conditions", [])
            condition = conditions[0] if conditions else ""
            results.append(
                {
                    "nct_id": nct_id,
                    "phase": phase,
                    "condition": condition,
                    "status": status,
                }
            )
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
        self.persist_dir = settings.CHROMA_PERSIST_DIR
        self._collection: Any | None = None
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
    ) -> list[dict[str, Any]]:
        """纯语义向量检索，返回带 distance 的候选记录。

        Args:
            type_filter: 参数类型过滤。可为字符串（精确匹配）或列表（任一匹配）。
                         例如 "kinetic_rate" 或 ["kinetic_rate", "binding_affinity"]。
                         设为 "exclude:initial_concentration" 可排除某类型。
        """
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
    ) -> list[dict[str, Any]]:
        """基于 BM25 的关键词检索，扫描 collection 全量文档并按词频打分。

        Args:
            type_filter: 参数类型过滤，同 semantic_search。
        """
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

        Args:
            type_filter: 参数类型过滤，同 semantic_search。默认 None（不过滤）。
        """
        semantic_results = self.semantic_search(
            query, top_k=top_k, species_filter=species_filter, type_filter=type_filter
        )
        bm25_results = self.bm25_search(query, top_k=top_k, type_filter=type_filter)

        merged: dict[str, dict[str, Any]] = {}
        for record in semantic_results + bm25_results:
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
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """高阶 RAG 检索入口：查询重写 → 混合检索 → 重排序。

        Args:
            type_filter: 参数类型过滤，默认排除 initial_concentration（强制 type system）。
                         可设为 None 关闭过滤，或设为 ["kinetic_rate", "binding_affinity"] 精确过滤。

        返回 (重排序后的 top_k 结果, RAG 洞察数据)。
        洞察数据包含 rewritten_query / rewrites / source_distribution / total_candidates。
        """
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

        return reranked, insights

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
