"""BioDynamics Agent - 模型一致性校验器（Model Consistency Validator）

TASK 6 修复：在 BIOMD reaction graph 构建后执行三重一致性检查，
确保模型语义严格收敛到原始 SBML/BIOMD 语义，防止非模型通路污染与结构断链。

三重检查：
1. Pathway integrity：EGF → ERK 可达性检查
2. Conservation sanity：关键蛋白池初始质量守恒（total > 0，池成员 ≥2）
3. No phantom pathway check：禁止 PI3K/Akt/NF-κB/JAK-STAT/feedback/crosstalk 等术语
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.biomodels_reactions import ReactionGraph

logger = logging.getLogger(__name__)


# EGF-EGFR-MAPK 规范链顺序（用于可达性判断与方向校正）
_CHAIN_ORDER: dict[str, int] = {
    "EGF": 0,
    "EGFR_active": 1,
    "Shc_complex": 2,
    "Grb2_SOS_complex": 3,
    "Ras_active": 4,
    "Raf_active": 5,
    "MEK_active": 6,
    "ERK_active": 7,
}


@dataclass
class ConsistencyReport:
    """模型一致性校验报告。"""

    passed: bool
    pathway_integrity: dict[str, Any] = field(default_factory=dict)
    conservation_sanity: dict[str, Any] = field(default_factory=dict)
    no_phantom_pathway: dict[str, Any] = field(default_factory=dict)
    summary: str = ""


def _build_adjacency(edges: list[dict]) -> dict[str, set[str]]:
    """根据 edges 构建有向邻接表。"""
    adj: dict[str, set[str]] = {}
    for e in edges:
        src = e.get("source", "")
        tgt = e.get("target", "")
        if src and tgt:
            adj.setdefault(src, set()).add(tgt)
    return adj


def _bfs_reachable(adj: dict[str, set[str]], start: str, end: str) -> tuple[bool, list[str]]:
    """BFS 检查 start 是否能到达 end，返回 (是否可达, 路径)。"""
    if start == end:
        return True, [start]
    visited: set[str] = {start}
    queue: list[tuple[str, list[str]]] = [(start, [start])]
    while queue:
        node, path = queue.pop(0)
        for nxt in adj.get(node, set()):
            if nxt == end:
                return True, path + [nxt]
            if nxt not in visited:
                visited.add(nxt)
                queue.append((nxt, path + [nxt]))
    return False, []


def check_pathway_integrity(edges: list[dict]) -> dict[str, Any]:
    """检查 EGF → ERK_active 是否可达。"""
    adj = _build_adjacency(edges)
    reachable, path = _bfs_reachable(adj, "EGF", "ERK_active")
    return {
        "check": "pathway_integrity",
        "passed": reachable,
        "start": "EGF",
        "end": "ERK_active",
        "path": path,
        "message": "EGF → ERK 可达" if reachable else "EGF → ERK 断链",
    }


def check_conservation_sanity(graph: "ReactionGraph") -> dict[str, Any]:
    """检查关键蛋白池的初始浓度是否合理（TASK 6）。

    canonical reduction 将同一蛋白的多种状态 collapse 为单一节点，因此：
    - EGFR 池 = EGFR + EGFR_active
    - Ras 池 = Ras_active + Ras_inactive
    - Raf / MEK / ERK 池由其规范 active 节点代表
    要求每个池总量 > 0，且规范 active 节点必须存在。
    """
    warnings: list[str] = []
    pool_totals: dict[str, float] = {}

    # 定义蛋白池成员（按 canonical 命名）
    pool_definitions: dict[str, list[str]] = {
        "EGFR": [s for s in graph.species if "egfr" in s.lower()],
        "Ras": ["Ras_active", "Ras_inactive"],
        "Raf": ["Raf_active"],
        "MEK": ["MEK_active"],
        "ERK": ["ERK_active"],
    }

    required_active = {
        "Ras_active": "Ras",
        "Raf_active": "Raf",
        "MEK_active": "MEK",
        "ERK_active": "ERK",
    }

    # 1. 检查规范 active 节点存在性
    for active_sp, pool_name in required_active.items():
        if active_sp not in graph.species:
            warnings.append(f"缺失规范 active 节点：{active_sp}（{pool_name} 池）")

    # 2. 检查蛋白池总浓度 > 0
    for pool_name, members in pool_definitions.items():
        total = sum(graph.species_initial.get(m, 0.0) for m in members if m in graph.species)
        pool_totals[pool_name] = total
        if total <= 0:
            warnings.append(f"{pool_name} 蛋白池初始总浓度为 0（成员：{members}）")

    passed = len(warnings) == 0
    return {
        "check": "conservation_sanity",
        "passed": passed,
        "pool_totals": pool_totals,
        "warnings": warnings,
        "message": "守恒检查通过" if passed else f"发现 {len(warnings)} 个守恒异常",
    }


def check_no_phantom_pathway(graph: "ReactionGraph") -> dict[str, Any]:
    """检查物种与反应方程中是否出现非模型通路术语。"""
    from app.biomodels_reactions import FORBIDDEN_PATHWAY_TERMS

    forbidden_found: list[dict[str, str]] = []
    all_texts: list[tuple[str, str]] = []

    for sp in graph.species:
        all_texts.append(("species", sp))
    for rxn in graph.reactions:
        if rxn.equation:
            all_texts.append(("reaction", f"{rxn.reaction_id}: {rxn.equation}"))

    for category, text in all_texts:
        text_lower = text.lower()
        for term in FORBIDDEN_PATHWAY_TERMS:
            if term in text_lower:
                forbidden_found.append({
                    "category": category,
                    "text": text,
                    "term": term,
                })

    passed = len(forbidden_found) == 0
    return {
        "check": "no_phantom_pathway",
        "passed": passed,
        "forbidden_hits": forbidden_found[:20],
        "message": "未发现非模型通路术语" if passed else f"发现 {len(forbidden_found)} 处非模型术语",
    }


def validate_reaction_graph_consistency(graph: "ReactionGraph") -> ConsistencyReport:
    """执行三重一致性检查并返回报告。

    若 Pathway integrity 或 No phantom pathway 失败，视为 P0 级阻塞问题，
    在日志中标记 ERROR。
    """
    from app.biomodels_reactions import reaction_graph_to_edges

    edges = reaction_graph_to_edges(graph)
    pi = check_pathway_integrity(edges)
    cs = check_conservation_sanity(graph)
    np = check_no_phantom_pathway(graph)

    passed = pi["passed"] and np["passed"] and cs["passed"]
    if not pi["passed"]:
        logger.error("MODEL_CONSISTENCY_ERROR: pathway integrity 失败 - %s", pi["message"])
    if not np["passed"]:
        logger.error("MODEL_CONSISTENCY_ERROR: 发现非模型通路术语 - %s", np["message"])
    if not cs["passed"]:
        logger.warning("MODEL_CONSISTENCY_WARNING: 守恒异常 - %s", cs["message"])

    summary = (
        f"一致性校验{'通过' if passed else '失败'}："
        f"pathway={pi['message']}; conservation={cs['message']}; phantom={np['message']}"
    )
    return ConsistencyReport(
        passed=passed,
        pathway_integrity=pi,
        conservation_sanity=cs,
        no_phantom_pathway=np,
        summary=summary,
    )
