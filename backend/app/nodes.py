# BioDynamics Agent - LangGraph 节点实现
# 包含机制解析、RAG 检索、方程生成、代码执行、审计纠错与报告生成六个节点。

import json
import re
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.config import llm
from app.prompts import (
    NODE1_PARSER_PROMPT,
    NODE2_GENERATOR_PROMPT,
    NODE4_AUDITOR_PROMPT,
    NODE6_REPORT_PROMPT,
    RAG_DECISION_PROMPT,
)
from app.rag_client import RagClient
from app.sandbox import execute_simulation_code
from app.sbml_parser import parse_sbml_model
from app.state import BioDynamicsState
from app.token_usage import (
    UsageAccumulator,
    merge_usage,
    usage_from_accumulator,
)


# -----------------------------------------------------------------------------
# 结构化输出模型
# -----------------------------------------------------------------------------
class _NodeItem(BaseModel):
    id: str = Field(..., description="节点唯一标识符")
    name: str = Field(..., description="节点可读名称")
    type: str = Field(..., description="节点类型，如 Protein, Cell, Molecule")


class _EdgeItem(BaseModel):
    source: str = Field(..., description="源节点 id")
    target: str = Field(..., description="目标节点 id")
    interaction: str = Field(..., description="关系类型：activation / inhibition / conversion")


class NetworkOutput(BaseModel):
    need_human_review: bool = Field(default=False, description="若机制模糊设为 true")
    review_question: str = Field(default="", description="需要追问用户的问题")
    nodes: list[_NodeItem] = Field(default_factory=list, description="生物实体列表")
    edges: list[_EdgeItem] = Field(default_factory=list, description="相互作用关系列表")


class AuditorOutput(BaseModel):
    status: str = Field(..., description="决策状态：success / retry / failed")
    correction_suggestion: str = Field(default="", description="当 retry 时的修改建议")
    failure_report: str = Field(default="", description="当 failed 时的失败说明")


class ExtractedParam(BaseModel):
    """RAG 从文献中提取的单条参数记录。"""

    param_name: str = Field(..., description="参数名称，如 Kd / Km / Vmax / 半衰期")
    value: float = Field(..., description="参数数值")
    unit: str = Field(..., description="单位，已统一为 h 或 nM")
    species: str = Field(default="", description="物种，如 Human / Mouse")
    cell_line: str = Field(default="", description="细胞系，如 HeLa / T-cell")
    context: str = Field(default="", description="参数出现的生物学上下文")
    confidence: str = Field(default="MEDIUM", description="HIGH 或 MEDIUM")


class RAGExtractionOutput(BaseModel):
    """RAG 参数提取的结构化输出。"""

    params: list[ExtractedParam] = Field(default_factory=list, description="提取的参数列表")


class _SelectedParam(BaseModel):
    """RAG 决策后为单条边选定的参数。"""

    param_name: str = Field(..., description="参数名称")
    value: float = Field(..., description="参数数值")
    unit: str = Field(..., description="单位")
    source: str = Field(default="Retrieved from RAG", description="参数来源")


class RAGDecisionOutput(BaseModel):
    """RAG 参数决策的结构化输出。"""

    param_found: bool = Field(default=False, description="是否找到可用参数")
    selected_params: list[_SelectedParam] = Field(
        default_factory=list, description="选定参数列表"
    )
    reasoning: str = Field(default="", description="选择理由或缺失说明")
    fallback_to_estimation: bool = Field(
        default=True, description="是否需要后续节点估算"
    )


# -----------------------------------------------------------------------------
# 通用工具函数
# -----------------------------------------------------------------------------
def _extract_python_code(content: str) -> Optional[str]:
    """从 LLM 返回文本中提取 ```python ... ``` 代码块。"""
    match = re.search(r"```python\n(.*?)\n```", content, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\n(.*?)\n```", content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return content.strip()


def _extract_species_context(user_input: str) -> str:
    """从用户输入中轻量提取物种/细胞系上下文，未命中则默认 Human。"""
    text = user_input.lower()
    if "hela" in text:
        return "HeLa"
    if "hek293" in text or "hek-293" in text:
        return "HEK293"
    if "cho" in text:
        return "CHO"
    if "t-cell" in text or "t cell" in text:
        return "T-cell"
    if "mouse" in text or "mice" in text or "mus musculus" in text:
        return "Mouse"
    if "rat" in text:
        return "Rat"
    return "Human"


def _build_rag_params_context(rag_selected_params: dict[str, dict]) -> str:
    """将 RAG 决策结果拼接为 Node 2 可读的上下文文本。"""
    if not rag_selected_params:
        return "未检索到文献参数，全部使用估算值。"

    lines = []
    for edge_key, decision in rag_selected_params.items():
        source, target = edge_key.split("|", 1)
        if decision.get("param_found"):
            params = decision.get("selected_params", [])
            param_desc = ", ".join(
                f"{p['param_name']}={p['value']} {p['unit']} (来源：{p.get('source', 'RAG')})"
                for p in params
            )
            lines.append(f"- 边 {source} → {target}：{param_desc}。{decision.get('reasoning', '')}")
        else:
            lines.append(
                f"- 边 {source} → {target}：参数缺失，将使用估算值。"
                f"{decision.get('reasoning', '')}"
            )
    return "\n".join(lines)


def _build_sbml_context(sbml_parsed_network: dict | None) -> str:
    """将 SBML 解析结果拼接为 Node 2 可读的上下文文本。"""
    if not sbml_parsed_network or not sbml_parsed_network.get("is_reusable"):
        return "未提供可复用 SBML 模型。"

    nodes = sbml_parsed_network.get("nodes", [])
    edges = sbml_parsed_network.get("edges", [])
    reason = sbml_parsed_network.get("reuse_reason", "")
    node_names = ", ".join(n.get("id", "") for n in nodes)
    edge_desc = ", ".join(
        f"{e.get('source', '')} {e.get('interaction', '')} {e.get('target', '')}"
        for e in edges
    )
    return f"可复用 SBML 模型：{reason}\n节点：{node_names}\n关系：{edge_desc}"


# -----------------------------------------------------------------------------
# Node 1: 机制解析器
# -----------------------------------------------------------------------------
def _strip_markdown_code_blocks(text: str) -> str:
    """去除 LLM 输出的 markdown 代码块标记，保留内部 JSON/Python 内容。"""
    text = text.strip()
    if text.startswith("```"):
        # 去掉第一行的 ```json / ```python 等
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _parse_network_json(raw_text: str) -> NetworkOutput:
    """从原始文本中解析 NetworkOutput；先尝试 structured output，失败则手动清理 markdown 后解析。"""
    clean_text = _strip_markdown_code_blocks(raw_text)
    try:
        data = json.loads(clean_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"无法解析 NetworkOutput JSON：{exc}\n原始内容：{raw_text[:500]}") from exc
    return NetworkOutput(**data)


def node1_parse_network(state: BioDynamicsState) -> dict:
    """解析用户自然语言输入，输出结构化生物网络 JSON。"""
    user_input = state.get("user_input", "")
    if not user_input:
        return {
            "need_human_review": True,
            "review_question": "请输入您想建模的生物学假说或机制描述。",
            "network_json": {"nodes": [], "edges": []},
        }

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", NODE1_PARSER_PROMPT),
            ("human", "{user_input}"),
        ]
    )
    chain = prompt | llm
    usage_handler = UsageAccumulator()
    response = chain.invoke(
        {"user_input": user_input},
        config={"callbacks": [usage_handler]},
    )
    result = _parse_network_json(str(response.content))

    network_json = {
        "nodes": [node.model_dump() for node in result.nodes],
        "edges": [edge.model_dump() for edge in result.edges],
    }
    summary = (
        f"已解析出 {len(network_json['nodes'])} 个节点、"
        f"{len(network_json['edges'])} 条相互作用边。"
    )

    return {
        "network_json": network_json,
        "need_human_review": result.need_human_review,
        "review_question": result.review_question,
        "messages": [summary],
        "token_usage": merge_usage(
            state.get("token_usage"), usage_from_accumulator(usage_handler)
        ),
    }


# -----------------------------------------------------------------------------
# Node 1.5: RAG 参数检索与决策器
# -----------------------------------------------------------------------------
def node1_5_rag_search(state: BioDynamicsState) -> dict:
    """在 Node 1 和 Node 2 之间插入的 RAG 节点：检索文献参数并决策是否使用。"""
    user_input = state.get("user_input", "")
    network_json = state.get("network_json") or {"nodes": [], "edges": []}
    edges = network_json.get("edges", [])

    # Token 累加器，覆盖 SBML 解析与每条边的 RAG 决策
    usage_handler = UsageAccumulator()

    # 1. 物种上下文准备
    species_context = _extract_species_context(user_input)

    # 2. SBML 解析子步骤
    sbml_model_text = state.get("sbml_model_text", "")
    sbml_parsed_network = None
    if sbml_model_text:
        try:
            sbml_parsed_network = parse_sbml_model(
                user_input, sbml_model_text, callbacks=[usage_handler]
            )
        except Exception as exc:
            sbml_parsed_network = {
                "is_reusable": False,
                "reuse_reason": f"SBML 解析失败：{exc}",
                "nodes": [],
                "edges": [],
            }

    # 3. RAG 检索与决策
    rag_client = RagClient()
    node_name_map = {n["id"]: n.get("name", n["id"]) for n in network_json.get("nodes", [])}
    rag_selected_params: dict[str, dict] = {}
    retrieved_all: list[dict] = []

    if not rag_client.available:
        # ChromaDB 不可用时，所有边直接回退到估算，避免不必要的 LLM 调用
        for edge in edges:
            edge_key = f"{edge.get('source', '')}|{edge.get('target', '')}"
            rag_selected_params[edge_key] = {
                "param_found": False,
                "selected_params": [],
                "reasoning": "ChromaDB 向量库不可用，回退到估算。",
                "fallback_to_estimation": True,
            }
    else:
        structured_llm = llm.with_structured_output(RAGDecisionOutput)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", RAG_DECISION_PROMPT),
                ("human", "请根据检索结果做出参数决策。"),
            ]
        )

        for edge in edges:
            source_id = edge.get("source", "")
            target_id = edge.get("target", "")
            interaction = edge.get("interaction", "")
            source_name = node_name_map.get(source_id, source_id)
            target_name = node_name_map.get(target_id, target_id)
            edge_key = f"{source_id}|{target_id}"

            query = (
                f"{source_name} {interaction} {target_name} "
                f"kinetic parameter Kd Km Vmax half-life degradation secretion "
                f"species {species_context}"
            )

            try:
                retrieved = rag_client.search_params(query, top_k=5)
                retrieved_all.extend(retrieved)
            except Exception:
                retrieved = []

            try:
                chain = prompt.partial(
                    source_node=source_name,
                    target_node=target_name,
                    interaction_type=interaction,
                    species_context=species_context,
                    retrieved_params_json=json.dumps(
                        retrieved, ensure_ascii=False, indent=2
                    ),
                ) | structured_llm
                decision: RAGDecisionOutput = chain.invoke(
                    {}, config={"callbacks": [usage_handler]}
                )
                rag_selected_params[edge_key] = decision.model_dump()
            except Exception:
                # LLM 决策失败时，安全回退到估算
                rag_selected_params[edge_key] = {
                    "param_found": False,
                    "selected_params": [],
                    "reasoning": "RAG 决策异常，回退到估算。",
                    "fallback_to_estimation": True,
                }

    found_count = sum(1 for d in rag_selected_params.values() if d.get("param_found"))
    fallback = any(d.get("fallback_to_estimation") for d in rag_selected_params.values())
    total = len(edges)
    rag_summary = f"已为 {found_count}/{total} 条边检索到真实参数"
    if fallback:
        rag_summary += "，其余边将使用估算值。"
    else:
        rag_summary += "。"

    update: dict = {
        "species_context": species_context,
        "rag_retrieved_params": retrieved_all,
        "rag_selected_params": rag_selected_params,
        "rag_fallback": fallback,
        "rag_summary": rag_summary,
        "messages": [rag_summary],
        "token_usage": merge_usage(
            state.get("token_usage"), usage_from_accumulator(usage_handler)
        ),
    }
    if sbml_parsed_network is not None:
        update["sbml_parsed_network"] = sbml_parsed_network

    return update


# -----------------------------------------------------------------------------
# Node 2: 参数与方程生成器
# -----------------------------------------------------------------------------
def node2_generate_code(state: BioDynamicsState) -> dict:
    """根据结构化网络生成可执行的 SciPy ODE 仿真代码。"""
    network_json = state.get("network_json")
    if not network_json:
        raise ValueError("Node 2 缺少 network_json 状态，无法生成方程。")

    retry_count = state.get("retry_count", 0)
    if retry_count > 0:
        error_feedback = state.get("stdout_stderr", "")
    else:
        error_feedback = "无"

    rag_params_context = _build_rag_params_context(
        state.get("rag_selected_params") or {}
    )
    sbml_context = _build_sbml_context(state.get("sbml_parsed_network"))

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", NODE2_GENERATOR_PROMPT),
            ("human", "请根据上述网络拓扑生成 ODE 仿真代码。"),
        ]
    )
    chain = prompt.partial(
        network_json=json.dumps(network_json, ensure_ascii=False, indent=2),
        error_feedback=error_feedback,
        rag_params_context=rag_params_context,
        sbml_context=sbml_context,
    ) | llm
    usage_handler = UsageAccumulator()
    response = chain.invoke({}, config={"callbacks": [usage_handler]})
    content = str(response.content)
    python_code = _extract_python_code(content)

    return {
        "python_code": python_code,
        "messages": [python_code],
        "token_usage": merge_usage(
            state.get("token_usage"), usage_from_accumulator(usage_handler)
        ),
    }


# -----------------------------------------------------------------------------
# Node 3: 代码执行器（调用沙箱）
# -----------------------------------------------------------------------------
def node3_execute_sandbox(state: BioDynamicsState) -> dict:
    """在隔离沙箱中执行生成的 Python 代码。"""
    python_code = state.get("python_code", "")
    if not python_code:
        return {
            "execution_status": "pending",
            "stdout_stderr": "未收到可执行代码。",
            "image_base64": "",
        }

    result = execute_simulation_code(python_code)
    return {
        "execution_status": result["status"],
        "stdout_stderr": result["stdout_stderr"],
        "image_base64": result["image_base64"] or "",
    }


# -----------------------------------------------------------------------------
# Node 4: 审计与纠错器
# -----------------------------------------------------------------------------
_MAX_RETRY = 3


def node4_audit_and_correct(state: BioDynamicsState) -> dict:
    """审计代码执行结果，决定重试、失败或成功。"""
    execution_status = state.get("execution_status", "error")
    retry_count = state.get("retry_count", 0)

    if execution_status == "success":
        return {
            "auditor_status": "success",
            "correction_suggestion": "",
            "failure_report": "",
        }

    stdout_stderr = state.get("stdout_stderr", "")

    # 执行失败且未达最大重试次数：必须重试；LLM 仅用于生成修改建议
    if retry_count < _MAX_RETRY:
        next_retry_count = retry_count + 1
        try:
            structured_llm = llm.with_structured_output(AuditorOutput)
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", NODE4_AUDITOR_PROMPT),
                    ("human", "请分析执行结果并给出修改建议。"),
                ]
            )
            chain = prompt.partial(
                retry_count=retry_count,
                execution_status=execution_status,
                stdout_stderr=stdout_stderr,
            ) | structured_llm
            usage_handler = UsageAccumulator()
            result: AuditorOutput = chain.invoke({}, config={"callbacks": [usage_handler]})
            suggestion = result.correction_suggestion or "请检查代码逻辑并修正错误。"
            token_usage = merge_usage(
                state.get("token_usage"), usage_from_accumulator(usage_handler)
            )
        except Exception as exc:
            suggestion = f"审计器调用异常，将自动重试：{exc}"
            token_usage = state.get("token_usage")

        if next_retry_count >= _MAX_RETRY:
            return {
                "auditor_status": "failed",
                "correction_suggestion": "",
                "failure_report": (
                    f"仿真代码经过 {next_retry_count} 次重试后仍无法成功执行。"
                    f"错误日志：{stdout_stderr}"
                ),
                "retry_count": next_retry_count,
                "token_usage": token_usage,
            }

        return {
            "auditor_status": "retry",
            "correction_suggestion": suggestion,
            "failure_report": "",
            "retry_count": next_retry_count,
            "token_usage": token_usage,
        }

    failure_report = (
        f"仿真代码经过 {retry_count} 次重试后仍无法成功执行。"
        f"错误日志：{stdout_stderr}"
    )
    return {
        "auditor_status": "failed",
        "correction_suggestion": "",
        "failure_report": failure_report,
        "token_usage": state.get("token_usage"),
    }


# -----------------------------------------------------------------------------
# Node 5: 报告生成器
# -----------------------------------------------------------------------------
def node5_generate_report(state: BioDynamicsState) -> dict:
    """根据成功仿真的结果撰写生物医学分析报告。"""
    network_json = state.get("network_json") or {"nodes": [], "edges": []}
    python_code = state.get("python_code", "")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", NODE6_REPORT_PROMPT),
            ("human", "请撰写仿真预测报告。"),
        ]
    )
    chain = prompt.partial(
        network_json=json.dumps(network_json, ensure_ascii=False, indent=2),
        python_code=python_code,
    ) | llm
    usage_handler = UsageAccumulator()
    response = chain.invoke({}, config={"callbacks": [usage_handler]})

    return {
        "final_report": str(response.content).strip(),
        "token_usage": merge_usage(
            state.get("token_usage"), usage_from_accumulator(usage_handler)
        ),
    }
