# BioDynamics Agent v4 - Pathway Planner Agent (P4)
# Phase 4 核心：通路识别与分发。从用户输入 + P1 pathway_registry 关键词规则，
# 输出 pathway_class（单通路或多通路列表）+ v4_pathway_class state 字段。
#
# 设计原则（铁律）：
# 1. Feature Flag V4_PATHWAY_PLANNER_ENABLED=false 时完全不执行，系统行为同 v3
# 2. 规则优先：先匹配 PATHWAY_REGISTRY 关键词，规则未命中才调用 LLM 兜底
# 3. LLM 失败时降级到 "UNKNOWN" 并记录 warning，不抛异常，不阻塞主流水线
# 4. 不修改 v3 任何字段（network_json / entities / mechanism 等）
# 5. 不生成 ODE / 不调用 RAG / 不做 SBML 验证（职责边界严格）
#
# 依赖（P1 + P3）：
# - app.ontology.pathway_registry.PATHWAY_REGISTRY + lookup_pathway
# - app.pathway_graph.initializer.PATHWAY_INITIALIZERS (cross_talk 数据)
# - app.config.settings / llm / strip_markdown_json

from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.config import llm, settings, strip_markdown_json
from app.ontology.pathway_registry import (
    PATHWAY_REGISTRY,
    PathwayEntry,
    lookup_pathway,
)
from app.pathway_graph.initializer import PATHWAY_INITIALIZERS

logger = logging.getLogger(__name__)


# =============================================================================
# 注册表 pathway_class ↔ PATHWAY_INITIALIZERS key 映射
# =============================================================================
# PATHWAY_REGISTRY 使用全大写键（如 "p53", "APOPTOSIS", "WNT", "TGF_BETA"）
# PATHWAY_INITIALIZERS 使用混合大小写键（如 "p53_signaling", "Apoptosis", "Wnt", "TGF_beta"）
# 此映射用于 cross-talk edge 查找时统一两端 pathway 命名。
_REGISTRY_TO_INITIALIZER: dict[str, str] = {
    "EGFR_RTK": "EGFR_RTK",
    "MAPK_ERK": "MAPK_ERK",
    "PI3K_AKT_mTOR": "PI3K_AKT_mTOR",
    "p53": "p53_signaling",
    "APOPTOSIS": "Apoptosis",
    "CELL_CYCLE": "Cell_Cycle",
    "JAK_STAT": "JAK_STAT",
    "NF_KB": "NF_kB",
    "WNT": "Wnt",
    "TGF_BETA": "TGF_beta",
}

# 反向映射：initializer key → registry pathway_class
_INITIALIZER_TO_REGISTRY: dict[str, str] = {
    v: k for k, v in _REGISTRY_TO_INITIALIZER.items()
}


# =============================================================================
# 规则优先通路识别（SubTask 4.1.2）
# =============================================================================
def _match_all_pathways(text: str) -> list[str]:
    """遍历所有 10 通路，返回命中列表（按注册表顺序）。

    与 P1 lookup_pathway 的区别：
    - lookup_pathway 返回首个命中（用于单通路场景）
    - _match_all_pathways 返回全部命中（用于多通路场景，SubTask 4.1.2）

    Args:
        text: 用户输入或机制描述文本

    Returns:
        命中的 pathway_class 列表（按注册表顺序），无命中返回空列表
    """
    if not text:
        return []
    return [
        entry.pathway_class
        for entry in PATHWAY_REGISTRY.values()
        if entry.matches(text)
    ]


def _format_multi_pathway(pathways: list[str]) -> str:
    """格式化多通路为 MULTI:A+B+C 字符串。

    Args:
        pathways: 通路类别键列表（按命中顺序）

    Returns:
        "MULTI:EGFR_RTK+PI3K_AKT_mTOR" 形式字符串
    """
    return "MULTI:" + "+".join(pathways)


def identify_pathways(user_input: str) -> list[str]:
    """规则识别所有命中通路（公开接口，供测试与外部调用）。

    Args:
        user_input: 用户原始输入文本

    Returns:
        命中的 pathway_class 列表（按注册表顺序，至少 0 个）
    """
    return _match_all_pathways(user_input)


# =============================================================================
# LLM 兜底（SubTask 4.1.3）
# =============================================================================
def _build_pathway_options_text() -> str:
    """构造 10 通路选项文本（含描述）供 LLM prompt 使用。"""
    lines: list[str] = []
    for entry in PATHWAY_REGISTRY.values():
        lines.append(f"- {entry.pathway_class}: {entry.description}")
    return "\n".join(lines)


def _llm_classify_pathway(user_input: str) -> str:
    """LLM 兜底：让 LLM 从 10 通路列表中选择最匹配的 1-3 个通路。

    触发条件：规则未命中（_match_all_pathways 返回空列表）
    降级策略：LLM 失败时返回 "UNKNOWN" 并记录 warning

    Args:
        user_input: 用户原始输入文本

    Returns:
        pathway_class 字符串：
        - 单通路如 "EGFR_RTK"
        - 多通路如 "MULTI:EGFR_RTK+PI3K_AKT_mTOR"
        - LLM 失败返回 "UNKNOWN"
    """
    options = _build_pathway_options_text()
    prompt = (
        "你是生物信号通路识别专家。请根据用户输入，从下列 10 条核心信号通路中"
        "选择最匹配的 1-3 个通路。\n\n"
        f"用户输入：{user_input}\n\n"
        f"可选通路列表：\n{options}\n\n"
        "返回 JSON 格式：{\"pathways\": [\"通路1\", \"通路2\", ...]}\n"
        "约束：\n"
        "- 至少选 1 个，最多 3 个\n"
        "- 通路名称必须严格从上面列表中选取（区分大小写）\n"
        "- 按相关性排序（最相关在前）\n"
        "- 只返回 JSON，不要其他解释"
    )

    try:
        response = llm.invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)
        cleaned = strip_markdown_json(text)

        # 提取最外层 JSON 对象
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if not match:
            logger.warning("LLM 兜底响应无 JSON：%s", cleaned[:200])
            return "UNKNOWN"

        data = json.loads(match.group(0))
        pathways_raw = data.get("pathways", [])

        # 过滤有效通路（必须在 PATHWAY_REGISTRY 中）
        valid = [p for p in pathways_raw if p in PATHWAY_REGISTRY]
        if not valid:
            logger.warning("LLM 兜底返回无效通路：%s", pathways_raw)
            return "UNKNOWN"

        # 去重保持顺序
        seen: set[str] = set()
        pathways: list[str] = []
        for p in valid:
            if p not in seen:
                pathways.append(p)
                seen.add(p)

        # 限制最多 3 个
        pathways = pathways[:3]

        if len(pathways) == 1:
            return pathways[0]
        return _format_multi_pathway(pathways)
    except Exception as exc:
        logger.warning("LLM 兜底失败，降级到 UNKNOWN: %s", exc)
        return "UNKNOWN"


# =============================================================================
# 多通路识别 + cross-talk edge 预识别（SubTask 4.1.4）
# =============================================================================
def _collect_crosstalk_edges(pathways: list[str]) -> list[dict[str, Any]]:
    """收集多通路场景下的 cross-talk edges。

    从 PATHWAY_INITIALIZERS 的 cross_talk 数据中筛选两端均位于
    identified pathway 集合内的边（按 ID 去重）。

    单通路场景（len(pathways) < 2）返回空列表。

    Args:
        pathways: 已识别的 pathway_class 列表（registry 命名，如 ["EGFR_RTK", "PI3K_AKT_mTOR"]）

    Returns:
        cross-talk edge 列表，每条含：
        - source_pathway / target_pathway / source_node / target_node
        - mechanism / shared_species / id / description
    """
    if len(pathways) < 2:
        return []

    # 构造 identified pathway 的 initializer_key 集合
    init_keys: set[str] = {
        _REGISTRY_TO_INITIALIZER.get(p, p) for p in pathways
    }

    seen_ids: set[str] = set()
    edges: list[dict[str, Any]] = []
    for _pwc, data in PATHWAY_INITIALIZERS.items():
        for ct in data.get("cross_talk", []):
            ct_id = ct.get("id", "")
            if ct_id in seen_ids:
                continue
            src = ct.get("source_pathway", "")
            tgt = ct.get("target_pathway", "")
            # 两端都必须在 identified pathway 集合内
            if src in init_keys and tgt in init_keys:
                edges.append({
                    "source_pathway": src,
                    "target_pathway": tgt,
                    "source_node": ct.get("source_node", ""),
                    "target_node": ct.get("target_node", ""),
                    "mechanism": ct.get("mechanism", ""),
                    "shared_species": ct.get("shared_species", []),
                    "id": ct_id,
                    "description": ct.get("description", ""),
                })
                seen_ids.add(ct_id)
    return edges


def _build_pathway_graph_payload(
    pathway_class: str,
    pathways: list[str],
) -> dict[str, Any]:
    """构造 v4_pathway_graph 预识别载荷（标记 primary pathway + cross-talk edges）。

    注意：此处的 v4_pathway_graph 是 Pathway Planner 输出的"预识别"载荷，
    P3 PathwayGraphBuilder 后续会构建完整 PathwayGraph 对象并覆盖此字段。

    Args:
        pathway_class: 完整 pathway_class 字符串（可能含 MULTI: 前缀）
        pathways: 已识别的 pathway_class 列表（不含前缀）

    Returns:
        dict 含 pathway_class / primary_pathway / identified_pathways /
        crosstalk_edges / source
    """
    crosstalk_edges = _collect_crosstalk_edges(pathways)
    return {
        "pathway_class": pathway_class,
        "primary_pathway": pathways[0] if pathways else None,
        "identified_pathways": list(pathways),
        "crosstalk_edges": crosstalk_edges,
        "source": "pathway_planner_v4",
        "note": "preliminary; P3 PathwayGraphBuilder 将构建完整 PathwayGraph",
    }


# =============================================================================
# 主入口：classify_pathway（规则优先 + LLM 兜底）
# =============================================================================
def classify_pathway(user_input: str) -> str:
    """通路识别主入口：规则优先 + LLM 兜底。

    匹配顺序：
    1. 规则匹配：遍历 PATHWAY_REGISTRY 10 通路关键词，收集所有命中
       - 1 个命中 → 返回单通路字符串（如 "EGFR_RTK"）
       - ≥2 个命中 → 返回多通路字符串（如 "MULTI:EGFR_RTK+PI3K_AKT_mTOR"）
    2. LLM 兜底：规则未命中时调用 LLM 选择 1-3 个通路
       - LLM 失败 → 返回 "UNKNOWN"

    Args:
        user_input: 用户原始输入文本

    Returns:
        pathway_class 字符串：
        - 单通路如 "EGFR_RTK"
        - 多通路如 "MULTI:EGFR_RTK+PI3K_AKT_mTOR"
        - 未识别 "UNKNOWN"
    """
    # 规则优先
    pathways = _match_all_pathways(user_input)
    if pathways:
        if len(pathways) == 1:
            return pathways[0]
        return _format_multi_pathway(pathways)

    # LLM 兜底
    return _llm_classify_pathway(user_input)


def parse_pathway_class(pathway_class: str) -> list[str]:
    """从 pathway_class 字符串解析出 pathway 列表（与 _format_multi_pathway 互逆）。

    Args:
        pathway_class: "EGFR_RTK" 或 "MULTI:EGFR_RTK+PI3K_AKT_mTOR" 或 "UNKNOWN"

    Returns:
        pathway 列表（"UNKNOWN" 返回空列表）
    """
    if not pathway_class or pathway_class == "UNKNOWN":
        return []
    if pathway_class.startswith("MULTI:"):
        return pathway_class[6:].split("+")
    return [pathway_class]


# =============================================================================
# LangGraph 节点 hook（feature flag 隔离）
# =============================================================================
def pathway_planner_hook_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph 节点：通路识别 hook。

    行为：
    - V4_PATHWAY_PLANNER_ENABLED=false：直接返回空 dict（不修改 state，不执行任何逻辑）
    - V4_PATHWAY_PLANNER_ENABLED=true：识别通路并写入
      state["v4_pathway_class"] + state["v4_pathway_graph"]（预识别载荷）

    严格遵守不可碰清单：
    - 不修改 v3 任何字段（network_json / entities / mechanism / parameters 等）
    - 不生成 ODE / 不调用 RAG / 不做 SBML 验证
    - 失败时降级返回空 dict，不抛异常，不阻塞主流水线

    Args:
        state: LangGraph 全局状态

    Returns:
        flag=false 时返回 {}
        flag=true 时返回 {"v4_pathway_class": str, "v4_pathway_graph": {...}}
    """
    # Feature Flag 检查：默认 false，跳过所有逻辑
    if not getattr(settings, "V4_PATHWAY_PLANNER_ENABLED", False):
        logger.debug("V4_PATHWAY_PLANNER_ENABLED=false，跳过 Pathway Planner")
        return {}

    try:
        user_input = state.get("user_input", "")
        pathway_class = classify_pathway(user_input)
        pathways = parse_pathway_class(pathway_class)
        pathway_graph = _build_pathway_graph_payload(pathway_class, pathways)

        logger.info(
            "Pathway Planner 完成：pathway_class=%s, pathways=%d, crosstalk_edges=%d",
            pathway_class,
            len(pathways),
            len(pathway_graph["crosstalk_edges"]),
        )

        # 仅写入 v4 字段，不触碰任何 v3 字段
        return {
            "v4_pathway_class": pathway_class,
            "v4_pathway_graph": pathway_graph,
        }
    except Exception as exc:
        # 任何失败都不阻塞流水线，记录 warning 并返回空更新
        logger.warning("Pathway Planner hook 失败，降级跳过: %s", exc)
        return {}


__all__ = [
    "classify_pathway",
    "identify_pathways",
    "parse_pathway_class",
    "pathway_planner_hook_node",
]
