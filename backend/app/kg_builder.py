"""BioDynamics Agent - 知识图谱构建器（v2 N4）

职责：
- 接收 N1 输出的 `entities` + N2 输出的 `mechanism`（含 NER 派生的边），
  构建一个稳定的 KG：nodes / edges / adjacency / topology_signature / is_acyclic。
- 纯 Python，无 LLM 调用，便于单元测试。
- 检测环路；若成环，丢弃"最弱"边（heuristic：方向 = inhibition 优先丢弃）并 warning。
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any


# 简单的"边方向强度"启发式：用于环路打破时挑选"最弱"边
_INTERACTION_STRENGTH: dict[str, int] = {
    "inhibition": 1,
    "conversion": 2,
    "activation": 3,
}

import re as _re

_PLACEHOLDER_RE = _re.compile(r"^(e\d+|entity_\d+|ent_\d+|node_\d+|n\d+)$", _re.IGNORECASE)


def _is_placeholder(text: str) -> bool:
    """判断文本是否为 NER 占位符（如 e1、entity_1）。"""
    if not text:
        return True
    return bool(_PLACEHOLDER_RE.match(text.strip()))


def _choose_display_name(entity: dict) -> str:
    """为实体选择可读的显示名，优先 name，其次 alias，最后 entity_id。"""
    name = str(entity.get("name") or "").strip()
    if not _is_placeholder(name):
        return name
    aliases = entity.get("aliases") or []
    for alias in aliases:
        alias_text = str(alias).strip()
        if alias_text and not _is_placeholder(alias_text):
            return alias_text
    eid = str(entity.get("entity_id") or entity.get("id") or "").strip()
    return eid if eid else name


def _infer_edges_from_entities(entities: list[dict]) -> list[dict]:
    """从 NER 实体列表启发式生成边（无 LLM）。

    仅在 N1 给出了 *relations* 字段时使用；否则返回空列表，由调用方
    从 `network_json`（v1 遗留）或 `mechanism` 派生。
    """
    if not entities:
        return []
    return []  # 实际派生由调用方在更明确的信号下完成


def _topology_signature(nodes: list[dict], edges: list[dict]) -> str:
    """生成稳定可比较的拓扑签名（节点 id 排序 + 边 source|target 排序）。"""
    sorted_nodes = sorted({n.get("id", "") for n in nodes if n.get("id")})
    sorted_edges = sorted(
        f"{e.get('source', '')}|{e.get('target', '')}|{e.get('interaction', '')}"
        for e in edges
    )
    return "NODES[" + ",".join(sorted_nodes) + "];EDGES[" + ",".join(sorted_edges) + "]"


def _adjacency(nodes: list[dict], edges: list[dict]) -> dict[str, list[str]]:
    """构建出度邻接表。"""
    adj: dict[str, list[str]] = defaultdict(list)
    for node in nodes:
        nid = node.get("id", "")
        if nid and nid not in adj:
            adj[nid] = []
    for e in edges:
        s = e.get("source", "")
        t = e.get("target", "")
        if s and t and t not in adj[s]:
            adj[s].append(t)
    return dict(adj)


def _has_cycle(nodes: list[dict], edges: list[dict]) -> bool:
    """DFS 检测有向图环路。"""
    adj = _adjacency(nodes, edges)
    color: dict[str, int] = {nid: 0 for nid in adj}  # 0=white 1=gray 2=black

    def dfs(u: str) -> bool:
        color[u] = 1
        for v in adj.get(u, []):
            if v not in color:
                continue
            if color[v] == 1:
                return True
            if color[v] == 0 and dfs(v):
                return True
        color[u] = 2
        return False

    return any(color.get(nid, 0) == 0 and dfs(nid) for nid in list(adj.keys()))


def _break_cycle(nodes: list[dict], edges: list[dict]) -> list[dict]:
    """在环路中丢弃"最弱"边（inhibition > conversion > activation）。"""
    if not _has_cycle(nodes, edges):
        return edges
    # 按强度升序，优先丢 inhibition
    ordered = sorted(
        edges,
        key=lambda e: _INTERACTION_STRENGTH.get(str(e.get("interaction", "")).lower(), 99),
    )
    for drop in ordered:
        remaining = [e for e in edges if e is not drop]
        if not _has_cycle(nodes, remaining):
            return remaining
    return edges  # fallback


class KGBuilder:
    """知识图谱构建器：entities + relations → 标准 KG dict。"""

    def build(
        self,
        entities: list[dict] | None = None,
        relations: list[dict] | None = None,
    ) -> dict[str, Any]:
        """构建知识图谱。

        Args:
            entities: NER 输出，每个含 `entity_id`/`name`/`type`/`aliases`。
            relations: 边列表，每个含 `source`/`target`/`interaction`。

        Returns:
            标准 KG dict：nodes / edges / adjacency / topology_signature /
            is_acyclic / node_count / edge_count / dropped_edges（环路打破时填充）。
        """
        entity_list = list(entities or [])
        relation_list = list(relations or [])

        # TODO: P0-4 — 构建 entity_id → display_name 映射，用于回填 edges 的 source/target
        # N1 NER 输出的 entities 可能含 entity_id="e1" 但 name 为空或也是占位符，
        # 此时需用 aliases[0] 或 entity_id 回填。该映射确保 edges 最终使用可读实体名。
        entity_id_to_name: dict[str, str] = {}
        for e in entity_list:
            eid = str(e.get("entity_id") or "").strip()
            display_name = _choose_display_name(e)
            if eid and display_name:
                entity_id_to_name[eid] = display_name

        # 1. 规范化节点
        nodes: list[dict] = []
        seen_ids: set[str] = set()
        for e in entity_list:
            nid = str(e.get("entity_id") or e.get("id") or "").strip()
            if not nid or nid in seen_ids:
                continue
            seen_ids.add(nid)
            display_name = _choose_display_name(e)
            nodes.append(
                {
                    "id": nid,
                    "name": display_name,
                    "type": str(e.get("type") or "Unknown"),
                    "aliases": list(e.get("aliases") or []),
                }
            )

        # 2. 规范化边（TODO: P0-4 — 回填 entity_id → name，消除 e1/e2 占位符）
        # Step 2.2 结构性修复：保留 mechanism / reaction_equation 字段供
        # Signaling_Cascade_Phos 模板按机制类型生成 ODE（binding/phosphorylation/exchange/...）
        edges: list[dict] = []
        seen_edges: set[tuple[str, str, str]] = set()
        for r in relation_list:
            src = str(r.get("source") or "").strip()
            tgt = str(r.get("target") or "").strip()
            # 回填：若 source/target 是 entity_id，替换为可读 name
            src = entity_id_to_name.get(src, src)
            tgt = entity_id_to_name.get(tgt, tgt)
            interaction = str(r.get("interaction") or "activation").strip().lower()
            if not src or not tgt or src == tgt:
                continue
            key = (src, tgt, interaction)
            if key in seen_edges:
                continue
            seen_edges.add(key)
            edge_dict: dict[str, Any] = {
                "source": src,
                "target": tgt,
                "interaction": interaction,
                "directed": True,
            }
            # 保留 mechanism（binding/phosphorylation/exchange/...）和 reaction_equation
            # 这些字段由 N2 PLANNER_PROMPT 输出，是 KG→Reaction Graph→Template 流水线的关键
            mechanism = str(r.get("mechanism") or "").strip()
            if mechanism:
                edge_dict["mechanism"] = mechanism.lower()
            reaction_equation = str(r.get("reaction_equation") or "").strip()
            if reaction_equation:
                edge_dict["reaction_equation"] = reaction_equation
            edges.append(edge_dict)

        # 3. 自动补全孤立节点（如果实体被引用了但无对应 node_id）
        node_ids = {n["id"] for n in nodes}
        for r in relation_list:
            for endpoint in (r.get("source"), r.get("target")):
                if endpoint and endpoint not in node_ids:
                    display_name = entity_id_to_name.get(endpoint, endpoint)
                    nodes.append(
                        {
                            "id": str(endpoint),
                            "name": display_name,
                            "type": "Unknown",
                            "aliases": [],
                        }
                    )
                    node_ids.add(str(endpoint))

        # 4. 检测 / 打破环路
        is_acyclic = not _has_cycle(nodes, edges)
        dropped_edges: list[dict] = []
        if not is_acyclic:
            new_edges = _break_cycle(nodes, edges)
            dropped_edges = [e for e in edges if e not in new_edges]
            edges = new_edges
            is_acyclic = not _has_cycle(nodes, edges)

        return {
            "nodes": nodes,
            "edges": edges,
            "adjacency": _adjacency(nodes, edges),
            "topology_signature": _topology_signature(nodes, edges),
            "is_acyclic": is_acyclic,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "dropped_edges": dropped_edges,
        }
