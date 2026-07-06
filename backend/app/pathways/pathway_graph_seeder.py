# BioDynamics Agent v4 - Pathway Graph Seeder (Phase 4 / Task 4.14)
# 灌库 biodynamics_pathway_graph_v4 collection：遍历 10 通路，
# 调用 PathwayGraphBuilder 构建 PathwayGraph，序列化为 JSON 写入 ChromaDB。
#
# 设计原则（铁律）：
# 1. ChromaDB 不可用时降级为 no-op + warning（不阻塞导入与运行）
# 2. 不修改 v3 任何 collection（仅新增 v4 collection）
# 3. 不依赖 LLM / RAG（纯规则构建 PathwayGraph）
# 4. 脚本可独立运行，但不在测试中强制执行（避免依赖 ChromaDB 运行环境）
#
# 依赖：
# - P3 pathway_graph/builder.py PathwayGraphBuilder
# - P3 pathway_graph/initializer.py PATHWAY_INITIALIZERS + get_pathway_init_data
# - ChromaDB（try-import 降级）

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


# ChromaDB 可用性标记（try-import）
_CHROMADB_AVAILABLE: bool = False
_chroma_client: Any = None

try:
    import chromadb  # type: ignore[import-untyped]
    _CHROMADB_AVAILABLE = True
except ImportError:  # pragma: no cover
    logger.warning(
        "pathway_graph_seeder: chromadb 未安装，灌库功能降级为 no-op"
    )


# ChromaDB collection 名（P4 新增，与 v3 collection 隔离）
COLLECTION_NAME: str = "biodynamics_pathway_graph_v4"


# 10 通路类别（registry 命名，与 pathway_planner._REGISTRY_TO_INITIALIZER 对齐）
PATHWAY_CLASSES_V4: list[str] = [
    "EGFR_RTK",
    "MAPK_ERK",
    "PI3K_AKT_mTOR",
    "p53",
    "APOPTOSIS",
    "CELL_CYCLE",
    "JAK_STAT",
    "NF_KB",
    "WNT",
    "TGF_BETA",
]


def _get_chroma_client() -> Any | None:
    """获取 ChromaDB 持久化客户端（不可用时返回 None）。"""
    global _chroma_client
    if not _CHROMADB_AVAILABLE:
        return None
    if _chroma_client is None:
        try:
            from app.config import settings
            _chroma_client = chromadb.PersistentClient(
                path=settings.CHROMA_PERSIST_DIR
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "pathway_graph_seeder: ChromaDB 客户端初始化失败: %s", exc
            )
            return None
    return _chroma_client


def seed_pathway_graph_v4() -> dict[str, Any]:
    """灌库 biodynamics_pathway_graph_v4 collection。

    遍历 10 通路，调用 PathwayGraphBuilder 构建 PathwayGraph，
    序列化为 JSON 写入 ChromaDB ``biodynamics_pathway_graph_v4`` collection。

    ChromaDB 不可用时降级为 no-op + warning，返回 ``{"seeded": False, "reason": ...}``。

    Returns:
        dict 含：
        - ``seeded``: bool（是否成功灌库）
        - ``collection_name``: str（collection 名）
        - ``pathways_count``: int（成功灌库的通路数）
        - ``reason``: str（失败原因，仅 seeded=False 时存在）
    """
    client = _get_chroma_client()
    if client is None:
        return {
            "seeded": False,
            "collection_name": COLLECTION_NAME,
            "pathways_count": 0,
            "reason": "chromadb_unavailable_or_client_init_failed",
        }

    # 延迟导入 P3 模块
    try:
        from app.pathway_graph.builder import PathwayGraphBuilder
        from app.pathway_graph.initializer import PathwayInitializer
        from app.pathways.pathway_planner import _REGISTRY_TO_INITIALIZER
    except ImportError as exc:
        logger.warning(
            "pathway_graph_seeder: P3/P4 模块导入失败: %s", exc
        )
        return {
            "seeded": False,
            "collection_name": COLLECTION_NAME,
            "pathways_count": 0,
            "reason": f"import_failed: {exc}",
        }

    # 获取或创建 collection
    try:
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "v4 PathwayGraph for 10 signaling pathways"},
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "pathway_graph_seeder: 创建 collection '%s' 失败: %s",
            COLLECTION_NAME,
            exc,
        )
        return {
            "seeded": False,
            "collection_name": COLLECTION_NAME,
            "pathways_count": 0,
            "reason": f"collection_create_failed: {exc}",
        }

    builder = PathwayGraphBuilder()
    seeded_count = 0

    for pathway_class in PATHWAY_CLASSES_V4:
        # 从 registry 命名映射到 initializer 命名
        init_key = _REGISTRY_TO_INITIALIZER.get(pathway_class, pathway_class)

        # 从 initializer 获取 feedback_loops / cross_talk_edges
        try:
            init_data = PathwayInitializer.get_pathway_init_data(init_key)
            # get_pathway_init_data 返回 (core_nodes, core_edges, feedback_loops, cross_talk_edges)
            if isinstance(init_data, tuple) and len(init_data) == 4:
                _, _, feedback_loops, cross_talk_edges = init_data
            else:
                feedback_loops = []
                cross_talk_edges = []
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "pathway_graph_seeder: 获取 %s 初始化数据失败: %s，使用空值",
                pathway_class,
                exc,
            )
            feedback_loops = []
            cross_talk_edges = []

        # 构建 PathwayGraph（不传 ontology_entities / reaction_ir，使用 initializer 数据）
        try:
            graph = builder.build(
                pathway_class=pathway_class,
                ontology_entities=None,
                reaction_ir=None,
                cross_talk_edges=cross_talk_edges,
                feedback_loops=feedback_loops,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "pathway_graph_seeder: 构建 %s PathwayGraph 失败: %s，跳过",
                pathway_class,
                exc,
            )
            continue

        # 序列化为 JSON
        try:
            graph_dict = graph.to_dict() if hasattr(graph, "to_dict") else dict(graph)
            graph_json = json.dumps(graph_dict, ensure_ascii=False, default=str)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "pathway_graph_seeder: 序列化 %s PathwayGraph 失败: %s，跳过",
                pathway_class,
                exc,
            )
            continue

        # 写入 ChromaDB collection
        try:
            doc_id = f"pathway_graph_{pathway_class}"
            collection.upsert(
                ids=[doc_id],
                documents=[graph_json],
                metadatas=[{
                    "pathway_class": pathway_class,
                    "initializer_key": init_key,
                    "source": "pathway_graph_seeder_v4",
                    "version": "v4.0",
                }],
            )
            seeded_count += 1
            logger.info(
                "pathway_graph_seeder: 灌库 %s 成功（nodes=%d, edges=%d）",
                pathway_class,
                len(graph_dict.get("nodes", [])),
                len(graph_dict.get("edges", [])),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "pathway_graph_seeder: 写入 %s 到 ChromaDB 失败: %s，跳过",
                pathway_class,
                exc,
            )

    return {
        "seeded": seeded_count > 0,
        "collection_name": COLLECTION_NAME,
        "pathways_count": seeded_count,
    }


__all__ = [
    "seed_pathway_graph_v4",
    "PATHWAY_CLASSES_V4",
    "COLLECTION_NAME",
]
