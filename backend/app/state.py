# BioDynamics Agent - LangGraph 状态定义
# 描述整个仿真工作流在节点之间传递的数据结构。

import operator
from typing import Annotated, TypedDict


class BioDynamicsState(TypedDict, total=False):
    """LangGraph 全局状态。所有字段均为可选，节点返回时只需提供发生变化的键值对。"""

    user_input: str
    messages: Annotated[list[str], operator.add]
    network_json: dict
    need_human_review: bool
    review_question: str
    python_code: str
    execution_status: str
    stdout_stderr: str
    image_base64: str
    retry_count: int
    auditor_status: str
    correction_suggestion: str
    failure_report: str
    final_report: str
    token_usage: dict

    # RAG 与知识注入相关字段
    species_context: str
    sbml_model_text: str
    sbml_parsed_network: dict
    rag_retrieved_params: list[dict]
    rag_selected_params: dict[str, dict]
    rag_fallback: bool
    rag_summary: str
