# BioDynamics Agent v4 - Cross-talk Edge Injector (Phase 4 / Task 4.13.4)
# 将 cross-talk edges 注入到 PathwayGraph，并校验 edge schema。
#
# 设计原则（铁律）：
# 1. 不修改 v3 任何字段；不生成 ODE；不做 SBML 验证（职责边界严格）
# 2. 失败降级：任何异常返回原 pathway_graph，不阻塞主流水线
# 3. 仅新增 v4 字段（cross_talk_edges），不修改 Specialist 内部 Reaction
# 4. 按 id 去重，避免重复注入
#
# 参考：
# - spec.md Part 3 Cross-talk Coordinator Agent（第 262-272 行）
# - P3 CrossTalkEdge schema（pathway_graph/schema.py 第 228-247 行）
# - tasks.md SubTask 4.13.4

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# CrossTalkEdge 必填字段（与 P3 CrossTalkEdge schema 对齐）
# =============================================================================
# CrossTalkEdge schema 字段：
#   id / source_pathway / target_pathway / source_node / target_node / mechanism
# 其中 id 为唯一标识，mechanism 为 17 类机制之一
_REQUIRED_EDGE_FIELDS: tuple[str, ...] = (
    "id",
    "source_pathway",
    "target_pathway",
    "source_node",
    "target_node",
    "mechanism",
)


class CrossTalkEdgeInjector:
    """Cross-talk edge 注入器。

    将 cross-talk edges 注入到 PathwayGraph（dict 形式），
    并校验 edge schema、按 id 去重。

    职责边界：
    - 仅新增 ``cross_talk_edges`` 字段到 pathway_graph
    - 不修改 Specialist 内部 Reaction
    - 不生成 ODE
    - 不做 SBML 验证
    """

    def inject_edges(
        self,
        pathway_graph: dict,
        crosstalk_edges: list[dict],
    ) -> dict:
        """将 cross-talk edges 注入到 pathway_graph。

        Args:
            pathway_graph: 原 pathway_graph dict（含 nodes / edges 等字段）。
            crosstalk_edges: cross-talk edge 列表，每条含 id / source_pathway /
                target_pathway / source_node / target_node / mechanism 等字段。

        Returns:
            注入 cross-talk edges 后的 pathway_graph（新增/更新 ``cross_talk_edges``
            字段）。异常时返回原 pathway_graph（不抛异常）。
        """
        try:
            if not isinstance(pathway_graph, dict):
                logger.warning(
                    "CrossTalkEdgeInjector.inject_edges: pathway_graph 非 dict，返回空 graph"
                )
                return {}

            # 深拷贝避免修改入参
            result = dict(pathway_graph)

            # 校验 + 去重
            validated = self.validate_edges(crosstalk_edges or [])
            deduped = self.deduplicate_edges(validated)

            # 合并已存在的 cross_talk_edges（按 id 去重）
            existing = list(result.get("cross_talk_edges", []) or [])
            merged = self.deduplicate_edges(existing + deduped)

            result["cross_talk_edges"] = merged
            return result
        except Exception as exc:
            logger.warning(
                "CrossTalkEdgeInjector.inject_edges 失败，返回原 graph: %s", exc
            )
            return pathway_graph if isinstance(pathway_graph, dict) else {}

    def validate_edges(self, edges: list[dict]) -> list[dict]:
        """校验 edge schema，过滤无效 edge。

        必填字段：id / source_pathway / target_pathway / source_node /
        target_node / mechanism。缺任一字段的 edge 被过滤。

        Args:
            edges: cross-talk edge 列表。

        Returns:
            通过校验的 edge 列表（保持原顺序）。
        """
        valid: list[dict] = []
        for edge in edges or []:
            if not isinstance(edge, dict):
                continue
            # 检查必填字段（值为空字符串也算缺失）
            ok = True
            for field in _REQUIRED_EDGE_FIELDS:
                val = edge.get(field)
                if val is None or (isinstance(val, str) and not val.strip()):
                    ok = False
                    break
            if ok:
                valid.append(edge)
            else:
                logger.debug(
                    "CrossTalkEdgeInjector.validate_edges: 过滤无效 edge（缺字段）: %s",
                    edge.get("id", "<no id>"),
                )
        return valid

    def deduplicate_edges(self, edges: list[dict]) -> list[dict]:
        """按 id 去重。

        保留首次出现的 edge（按输入顺序），后续相同 id 的 edge 被丢弃。

        Args:
            edges: cross-talk edge 列表。

        Returns:
            去重后的 edge 列表（保持首次出现顺序）。
        """
        seen: set[str] = set()
        result: list[dict] = []
        for edge in edges or []:
            if not isinstance(edge, dict):
                continue
            edge_id = edge.get("id", "")
            if not edge_id:
                # 无 id 的 edge 直接保留（不参与去重）
                result.append(edge)
                continue
            if edge_id in seen:
                continue
            seen.add(edge_id)
            result.append(edge)
        return result


__all__ = ["CrossTalkEdgeInjector"]
