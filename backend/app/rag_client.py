# BioDynamics Agent - ChromaDB RAG 客户端
# 负责 ChromaDB 本地持久化连接、动力学参数写入与语义检索，并提供单位归一化辅助函数。

import logging
import uuid
from typing import Any

import chromadb

from app.config import embedding_model, settings

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


# -----------------------------------------------------------------------------
# ChromaDB 客户端封装
# -----------------------------------------------------------------------------
class RagClient:
    """ChromaDB 向量库客户端，负责动力学参数的存储与检索。"""

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

    def search_params(
        self,
        query: str,
        top_k: int = 5,
        species_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """根据查询文本检索参数记录；失败时返回空列表，不影响主工作流。"""
        collection = self._get_collection()
        if collection is None:
            return []

        try:
            vector = embedding_model.embed_query(query)
            query_kwargs: dict[str, Any] = {
                "query_embeddings": [vector],
                "n_results": top_k,
                "include": ["metadatas"],
            }
            if species_filter:
                query_kwargs["where"] = {"species": species_filter}

            results = collection.query(**query_kwargs)
            metadatas = results.get("metadatas", [[]])[0]
            return [meta for meta in metadatas if meta]
        except Exception as exc:
            logger.warning("ChromaDB 检索失败：%s", exc)
            return []

    def _get_vector_size(self) -> int:
        """获取当前 embedding 模型的向量维度。"""
        try:
            sample = embedding_model.embed_query("dimension test")
            return len(sample)
        except Exception as exc:
            logger.warning("获取 embedding 维度失败，使用默认值 1536：%s", exc)
            return 1536
