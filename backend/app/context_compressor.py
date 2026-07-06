# BioDynamics Agent v3 - 上下文渐进式压缩模块
# 将超大结构化数据压缩为摘要，防止 Supervisor 与 Worker 的 prompt 撑爆上下文窗口。
# 原始数据保留在 state.raw_cache，仅摘要参与 LLM prompt。

from __future__ import annotations

import json
import logging
from typing import Any

from app.config import llm

logger = logging.getLogger(__name__)

# 触发压缩的 Token 阈值（按字符数/4 估算）
_COMPRESS_TOKEN_THRESHOLD = 1000


def _estimate_tokens(value: Any) -> int:
    """轻量估算给定数据的 Token 数（1 token ≈ 4 字符，中英文混合保守估计）。"""
    text = json.dumps(value, ensure_ascii=False, default=str)
    return max(1, len(text) // 4)


def _builtin_summarize(key: str, value: Any) -> str:
    """内置摘要函数：对结构化数据做零 LLM 的轻量摘要。"""
    if isinstance(value, list):
        n = len(value)
        if key in ("rag_retrieved_params", "drug_candidates", "experiment_protocols", "paper_evidence"):
            return f"[{key}] 共 {n} 条记录，已压缩。Top 3 示例：{json.dumps(value[:3], ensure_ascii=False, default=str)[:400]}"
        if key in ("mcp_term_definitions",):
            names = [str(d.get("term") or d.get("canonical_name", "?")) for d in value[:5]]
            return f"[{key}] 共 {n} 个术语：{', '.join(names)}{' ...' if n > 5 else ''}"
        return f"[{key}] 共 {n} 条记录，已压缩。"

    if isinstance(value, dict):
        if key == "knowledge_graph":
            return (
                f"[{key}] 节点 {value.get('node_count', 0)} 个，"
                f"边 {value.get('edge_count', 0)} 条，"
                f"有向无环：{value.get('is_acyclic', True)}，"
                f"拓扑签名：{value.get('topology_signature', '')}"
            )
        if key == "parameters":
            return f"[{key}] 共 {len(value)} 条边参数，来源分布：{_count_sources(value)}"
        if key in ("metrics", "feature_metadata"):
            return f"[{key}] 已压缩，原始大小约 {_estimate_tokens(value)} tokens"
        sub_keys = ", ".join(value.keys())
        return f"[{key}] 包含字段：{sub_keys}"

    text = str(value)
    if len(text) > 200:
        text = text[:200] + " ..."
    return f"[{key}] {text}"


def _count_sources(parameters: dict[str, Any]) -> dict[str, int]:
    """统计参数来源分布。"""
    sources: dict[str, int] = {}
    for edge_param in parameters.values():
        src = "ESTIMATED"
        if isinstance(edge_param, dict):
            src = edge_param.get("source") or edge_param.get("is_fallback", False) and "ESTIMATED" or "UNKNOWN"
        sources[src] = sources.get(src, 0) + 1
    return sources


def compress_value(key: str, value: Any, use_llm: bool = False) -> tuple[str, Any]:
    """压缩单个字段。

    返回 (summary, raw_value)。summary 用于传入 LLM prompt，raw_value 存入 raw_cache。
    """
    tokens = _estimate_tokens(value)
    if tokens <= _COMPRESS_TOKEN_THRESHOLD:
        # 未超限，无需压缩，summary 直接保留为可读的简短说明
        return f"[{key}] 原始数据 {tokens} tokens（未超限）", value

    if use_llm:
        try:
            prompt = (
                f"请用 2-3 句中文高度概括以下 '{key}' 数据的核心结论，"
                f"保留数量、来源、置信度等关键信息，不超过 200 字：\n"
                f"{json.dumps(value, ensure_ascii=False, default=str)[:2000]}"
            )
            response = llm.invoke([("human", prompt)])
            return str(response.content).strip(), value
        except Exception as exc:
            logger.warning("LLM 压缩 %s 失败，回退到内置摘要：%s", key, exc)

    return _builtin_summarize(key, value), value


def compress_state(
    state: dict[str, Any],
    keys_to_compress: list[str] | None = None,
    use_llm: bool = False,
) -> dict[str, Any]:
    """对 state 中的大字段进行压缩，返回更新后的 state 片段。

    调用方应将返回的 update 合并到 state，其中：
    - {key}_summary 为摘要字段（供 prompt 使用）
    - raw_cache.{key} 为原始数据（供后续节点/前端使用）
    """
    if keys_to_compress is None:
        keys_to_compress = [
            "rag_retrieved_params",
            "rag_selected_params",
            "knowledge_graph",
            "parameters",
            "drug_candidates",
            "mcp_term_definitions",
            "metrics",
            "feature_metadata",
            "experiment_protocols",
            "paper_evidence",
        ]

    summaries: dict[str, str] = {}
    raw_cache: dict[str, Any] = {}

    for key in keys_to_compress:
        value = state.get(key)
        if value is None:
            continue
        summary, raw_value = compress_value(key, value, use_llm=use_llm)
        summaries[f"{key}_summary"] = summary
        raw_cache[key] = raw_value

    update: dict[str, Any] = {"raw_cache": raw_cache}
    update.update(summaries)
    return update


def get_summary(state: dict[str, Any], key: str) -> str:
    """获取指定字段的摘要；若不存在则返回原始数据的简短说明。"""
    summary_key = f"{key}_summary"
    if summary_key in state:
        return str(state[summary_key])
    value = state.get(key)
    if value is None:
        return f"[{key}] 无数据"
    return _builtin_summarize(key, value)


def compress_worker_output(
    worker_name: str,
    output: dict[str, Any],
    use_llm: bool = False,
) -> dict[str, Any]:
    """在 Worker 节点出口对产出进行压缩，生成摘要并保留原始数据到 raw_cache。"""
    # 仅对已知可能膨胀的字段压缩
    keys: list[str] = []
    if worker_name == "worker_rag":
        keys = ["rag_retrieved_params", "rag_selected_params", "drug_candidates"]
    elif worker_name == "worker_mechanism":
        keys = ["knowledge_graph", "entities", "network_relations"]
    elif worker_name == "worker_report":
        keys = ["metrics", "feature_metadata", "experiment_protocols", "paper_evidence"]
    elif worker_name == "worker_mcp":
        keys = ["mcp_term_definitions", "mcp_tool_calls"]

    if not keys:
        return output

    merged_state = {**output}
    compressed = compress_state(merged_state, keys_to_compress=keys, use_llm=use_llm)
    merged_state.update(compressed)
    return merged_state
