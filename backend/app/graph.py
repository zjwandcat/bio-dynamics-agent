# BioDynamics Agent - LangGraph 状态图组装
# 将机制解析、RAG 检索、方程生成、代码执行、审计纠错与报告生成节点连接为完整工作流。
# v2 升级：额外提供 12 节点编译工作流（compiled_workflow_v2），由 WORKFLOW_VERSION=v2 启用。

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.nodes import (
    node0_mcp_term_lookup,
    node1_5_rag_search,
    node1_6_pkpd_inference,
    node1_parse_network,
    node2_generate_code,
    node3_execute_sandbox,
    node4_audit_and_correct,
    node5_generate_report,
)
from app.nodes_v2 import NODES_V2
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


def _route_v2_after_ner(state: BioDynamicsState) -> str:
    """v2 N1 → N2 直接继续。"""
    return "n2_mechanistic_planner"


def _route_v2_after_sandbox(state: BioDynamicsState) -> str:
    """v2 N7 沙箱：error_class=语法错误则重试一次（最多 3 次），否则进入 N8。"""
    err = state.get("error_class", "none")
    retry = int(state.get("retry_count", 0))
    if err in ("syntax_error", "runtime_error") and retry < 3:
        return "n6_ode_generator"
    return "n8_scientific_features"


def _route_v2_after_ode(state: BioDynamicsState) -> str:
    """v2 N6 → N7。"""
    return "n7_sandbox_execute"


def _route_v2_after_features(state: BioDynamicsState) -> str:
    """v2 N8 → N9。"""
    return "n9_experiment_rag"


def _route_v2_after_experiment(state: BioDynamicsState) -> str:
    """v2 N9 → N10。"""
    return "n10_evidence_rag"


def _route_v2_after_evidence(state: BioDynamicsState) -> str:
    """v2 N10 → N11。"""
    return "n11_scientific_report"


def build_workflow() -> StateGraph:
    """构建并返回 BioDynamics Agent 的 LangGraph 工作流。"""
    workflow = StateGraph(BioDynamicsState)

    workflow.add_node("node0_mcp_term_lookup", node0_mcp_term_lookup)
    workflow.add_node("node1_parse_network", node1_parse_network)
    workflow.add_node("node1_5_rag_search", node1_5_rag_search)
    workflow.add_node("node1_6_pkpd_inference", node1_6_pkpd_inference)
    workflow.add_node("node2_generate_code", node2_generate_code)
    workflow.add_node("node3_execute_sandbox", node3_execute_sandbox)
    workflow.add_node("node4_audit_and_correct", node4_audit_and_correct)
    workflow.add_node("node5_generate_report", node5_generate_report)

    # 工作流：START → node0(MCP术语查询) → node1(机制解析) → node1.5(RAG) → node1.6(PK/PD) → ...
    workflow.add_edge(START, "node0_mcp_term_lookup")
    workflow.add_edge("node0_mcp_term_lookup", "node1_parse_network")
    workflow.add_conditional_edges(
        "node1_parse_network",
        _route_after_parse,
        {END: END, "node1_5_rag_search": "node1_5_rag_search"},
    )
    workflow.add_edge("node1_5_rag_search", "node1_6_pkpd_inference")
    workflow.add_edge("node1_6_pkpd_inference", "node2_generate_code")
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


# -----------------------------------------------------------------------------
# v2 — 12 节点工作流
# -----------------------------------------------------------------------------
def build_workflow_v2() -> StateGraph:
    """构建并返回 BioDynamics Agent v2 的 12 节点 LangGraph 工作流。

    12 节点链路：
        N1(NER) → N2(Planner) → N3(Mechanism RAG) → N4(KG) → N5(Parameter RAG)
        → N6(ODE) → N7(Sandbox) → N8(Features) → N9(Experiment RAG) → N10(Evidence RAG)
        → N11(Report)
    """
    workflow = StateGraph(BioDynamicsState)

    for name, fn in NODES_V2.items():
        workflow.add_node(name, fn)

    # 主链路
    workflow.add_edge(START, "n1_ner_entity_normalize")
    workflow.add_conditional_edges(
        "n1_ner_entity_normalize",
        _route_v2_after_ner,
        {"n2_mechanistic_planner": "n2_mechanistic_planner"},
    )
    workflow.add_edge("n2_mechanistic_planner", "n3_mechanism_rag")
    workflow.add_edge("n3_mechanism_rag", "n4_kg_builder")
    workflow.add_edge("n4_kg_builder", "n5_parameter_rag")
    workflow.add_edge("n5_parameter_rag", "n6_ode_generator")
    workflow.add_conditional_edges(
        "n6_ode_generator",
        _route_v2_after_ode,
        {"n7_sandbox_execute": "n7_sandbox_execute"},
    )
    workflow.add_conditional_edges(
        "n7_sandbox_execute",
        _route_v2_after_sandbox,
        {
            "n6_ode_generator": "n6_ode_generator",
            "n8_scientific_features": "n8_scientific_features",
        },
    )
    workflow.add_conditional_edges(
        "n8_scientific_features",
        _route_v2_after_features,
        {"n9_experiment_rag": "n9_experiment_rag"},
    )
    workflow.add_conditional_edges(
        "n9_experiment_rag",
        _route_v2_after_experiment,
        {"n10_evidence_rag": "n10_evidence_rag"},
    )
    workflow.add_conditional_edges(
        "n10_evidence_rag",
        _route_v2_after_evidence,
        {"n11_scientific_report": "n11_scientific_report"},
    )
    workflow.add_edge("n11_scientific_report", END)

    return workflow


compiled_workflow = build_workflow().compile(checkpointer=MemorySaver())
compiled_workflow_v2 = build_workflow_v2().compile(checkpointer=MemorySaver())
