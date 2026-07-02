# BioDynamics Agent - LangGraph 状态图组装
# 将机制解析、RAG 检索、方程生成、代码执行、审计纠错与报告生成节点连接为完整工作流。

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.nodes import (
    node1_5_rag_search,
    node1_parse_network,
    node2_generate_code,
    node3_execute_sandbox,
    node4_audit_and_correct,
    node5_generate_report,
)
from app.state import BioDynamicsState


def _route_after_parse(state: BioDynamicsState) -> str:
    """机制解析后，若需要人工复核则结束，否则进入 RAG 检索节点。"""
    if state.get("need_human_review", False):
        return END
    return "node1_5_rag_search"


def _route_after_audit(state: BioDynamicsState) -> str:
    """审计后根据状态决定：重试则回到生成器，失败或超过 3 次则结束，否则进入报告生成。"""
    if state.get("auditor_status") == "failed" or state.get("retry_count", 0) >= 3:
        return END
    if state.get("auditor_status") == "retry":
        return "node2_generate_code"
    return "node5_generate_report"


def build_workflow() -> StateGraph:
    """构建并返回 BioDynamics Agent 的 LangGraph 工作流。"""
    workflow = StateGraph(BioDynamicsState)

    workflow.add_node("node1_parse_network", node1_parse_network)
    workflow.add_node("node1_5_rag_search", node1_5_rag_search)
    workflow.add_node("node2_generate_code", node2_generate_code)
    workflow.add_node("node3_execute_sandbox", node3_execute_sandbox)
    workflow.add_node("node4_audit_and_correct", node4_audit_and_correct)
    workflow.add_node("node5_generate_report", node5_generate_report)

    workflow.add_edge(START, "node1_parse_network")
    workflow.add_conditional_edges(
        "node1_parse_network",
        _route_after_parse,
        {END: END, "node1_5_rag_search": "node1_5_rag_search"},
    )
    workflow.add_edge("node1_5_rag_search", "node2_generate_code")
    workflow.add_edge("node2_generate_code", "node3_execute_sandbox")
    workflow.add_edge("node3_execute_sandbox", "node4_audit_and_correct")
    workflow.add_conditional_edges(
        "node4_audit_and_correct",
        _route_after_audit,
        {
            END: END,
            "node2_generate_code": "node2_generate_code",
            "node5_generate_report": "node5_generate_report",
        },
    )
    workflow.add_edge("node5_generate_report", END)

    return workflow


compiled_workflow = build_workflow().compile(checkpointer=MemorySaver())
