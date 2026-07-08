# BioDynamics Agent v4 - 统一 Health Check（Phase 5 / Reliability Engineering）
#
# 检查所有子系统健康状态，供 /health 端点与运维监控使用。
# 每个子系统检查返回 {"healthy": bool, "latency_ms": float, "details": str}，
# 单个子系统检查异常不会影响其他子系统（内部捕获）。
#
# 设计原则：
# 1. 防御性：每个 _check_xxx 内部 try/except，绝不向上抛异常（/health 必须可用）
# 2. 延迟导入：子系统依赖（chromadb / roadrunner / rag_client）延迟到检查时导入，
#    避免模块加载时因可选依赖缺失而失败
# 3. 零写入：仅探测，不修改任何子系统状态
# 4. 标准结构：每个检查返回 healthy / latency_ms / details 三字段
#
# 用法：
#   from app.reliability.health_check import HealthChecker
#   report = HealthChecker().check_all()
#   all_healthy = all(v["healthy"] for v in report.values())

from __future__ import annotations

import logging
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)


class HealthChecker:
    """统一健康检查，检查所有子系统。

    覆盖六个子系统：
    - api        : API 服务本身（配置加载、LLM 初始化）
    - rag        : RAG 子系统（rag_client 模块与配置就绪）
    - vector_db  : 向量数据库（ChromaDB 连通性与集合）
    - solver     : ODE 求解器（roadrunner 可用性）
    - sbml       : SBML 解析依赖（lxml 或标准库 xml.etree）
    - ontology   : 本体论子系统（MCP 端点配置）
    """

    def check_all(self) -> dict[str, dict[str, Any]]:
        """检查所有子系统健康状态。

        Returns:
            dict，键为子系统名，值为
            ``{"healthy": bool, "latency_ms": float, "details": str}``
        """
        return {
            "api": self._check_api(),
            "rag": self._check_rag(),
            "vector_db": self._check_vector_db(),
            "solver": self._check_solver(),
            "sbml": self._check_sbml(),
            "ontology": self._check_ontology(),
        }

    # ------------------------------------------------------------------
    # 通用探针封装
    # ------------------------------------------------------------------
    @staticmethod
    def _probe(name: str, check_fn: Callable[[], tuple[bool, str]]) -> dict[str, Any]:
        """执行单个检查并包装为标准结果。

        Args:
            name: 子系统名（用于日志）
            check_fn: 检查函数，返回 (healthy, details)；抛异常视为不健康

        Returns:
            ``{"healthy": bool, "latency_ms": float, "details": str}``
        """
        start = time.perf_counter()
        try:
            healthy, details = check_fn()
            latency_ms = (time.perf_counter() - start) * 1000
            return {
                "healthy": bool(healthy),
                "latency_ms": round(latency_ms, 2),
                "details": str(details),
            }
        except Exception as exc:  # noqa: BLE001 - /health 必须永不抛异常
            latency_ms = (time.perf_counter() - start) * 1000
            logger.debug("HealthCheck [%s] 异常: %s", name, exc, exc_info=True)
            return {
                "healthy": False,
                "latency_ms": round(latency_ms, 2),
                "details": f"{type(exc).__name__}: {exc}",
            }

    # ------------------------------------------------------------------
    # 各子系统检查
    # ------------------------------------------------------------------
    def _check_api(self) -> dict[str, Any]:
        """检查 API 服务本身（配置加载、LLM 初始化）。"""

        def _do() -> tuple[bool, str]:
            from app.config import llm, settings

            if llm is None:
                return False, "LLM 未初始化"
            # OPENAI_API_KEY 为占位符时标记为不健康（仍可启动但无法真实调用）
            if "placeholder" in settings.OPENAI_API_KEY:
                return False, "OPENAI_API_KEY 未配置（占位符）"
            return True, f"API 就绪，模型={settings.OPENAI_MODEL}"

        return self._probe("api", _do)

    def _check_rag(self) -> dict[str, Any]:
        """检查 RAG 子系统（rag_client 模块可导入性与配置就绪）。

        注意：不在此处实例化 RagClient（会触发 ChromaDB 连接，成本较高且与
        vector_db 检查重复）；仅校验模块可导入、类存在、集合名已配置。
        """

        def _do() -> tuple[bool, str]:
            try:
                import app.rag_client as rag_mod  # type: ignore
            except Exception as exc:  # noqa: BLE001
                return False, f"rag_client 模块导入失败: {type(exc).__name__}: {exc}"
            if not hasattr(rag_mod, "RagClient"):
                return False, "rag_client 缺少 RagClient 类"
            from app.config import settings

            collections = [
                settings.CHROMA_COLLECTION_MECHANISM,
                settings.CHROMA_COLLECTION_PARAMETER,
                settings.CHROMA_COLLECTION_EXPERIMENT,
                settings.CHROMA_COLLECTION_EVIDENCE,
            ]
            return True, f"RAG 模块就绪，四路集合={collections}"

        return self._probe("rag", _do)

    def _check_vector_db(self) -> dict[str, Any]:
        """检查向量数据库（ChromaDB 连通性与集合列表）。"""

        def _do() -> tuple[bool, str]:
            from app.config import settings

            persist_dir = settings.CHROMA_PERSIST_DIR
            try:
                import chromadb  # type: ignore
            except ImportError:
                return False, "chromadb 未安装，向量库不可用"

            client = chromadb.PersistentClient(path=persist_dir)
            # heartbeat 是 ChromaDB 轻量连通性探测
            client.heartbeat()
            collections = client.list_collections()
            # 兼容 chromadb 不同版本：返回 Collection 对象或字符串
            names = [
                c.name if hasattr(c, "name") else str(c) for c in collections
            ]
            return True, f"向量库就绪，集合={names}"

        return self._probe("vector_db", _do)

    def _check_solver(self) -> dict[str, Any]:
        """检查 ODE 求解器（roadrunner 可用性）。"""

        def _do() -> tuple[bool, str]:
            from app.config import ROADRUNNER_AVAILABLE, ROADRUNNER_VERSION

            if ROADRUNNER_AVAILABLE:
                return True, f"roadrunner 可用，版本={ROADRUNNER_VERSION}"
            return (
                False,
                "roadrunner 未安装，SBML 仿真将降级到 Track B 结构相似度评分",
            )

        return self._probe("solver", _do)

    def _check_sbml(self) -> dict[str, Any]:
        """检查 SBML 解析依赖（lxml 优先，降级到标准库 xml.etree）。"""

        def _do() -> tuple[bool, str]:
            from app.config import LXML_AVAILABLE, LXML_VERSION

            if LXML_AVAILABLE:
                return True, f"lxml 可用，版本={LXML_VERSION}"
            # 降级到标准库（sbml_parser_v2 的后备路径）
            import xml.etree.ElementTree as _et  # noqa: F401

            return (
                True,
                "lxml 不可用，已降级到 xml.etree.ElementTree（标准库）",
            )

        return self._probe("sbml", _do)

    def _check_ontology(self) -> dict[str, Any]:
        """检查本体论子系统（MCP 端点配置）。"""

        def _do() -> tuple[bool, str]:
            from app.config import settings

            if not settings.MCP_ENABLED:
                return (
                    True,
                    "MCP 已禁用，本体论降级为 LLM 内部知识（不影响主流程）",
                )
            endpoints = {
                "openbiomed": settings.MCP_OPENBIOMED_URL,
                "medterm": settings.MCP_MEDTERM_URL,
                "pubmed": settings.MCP_PUBMED_URL,
                "umls": settings.MCP_UMLS_URL,
            }
            configured = [k for k, v in endpoints.items() if v]
            if not configured:
                return (
                    False,
                    "MCP 已启用但未配置任何端点 URL，本体论查询将失败",
                )
            return True, f"MCP 就绪，已配置端点={configured}"

        return self._probe("ontology", _do)


__all__ = ["HealthChecker"]
