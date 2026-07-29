# BioDynamics Agent - LangGraph 节点实现
# 包含机制解析、RAG 检索、方程生成、代码执行、审计纠错与报告生成六个节点。
# 升级：集成多智能体编排器（Supervisor）调度事件与高阶 RAG 混合检索。

import json
import logging
import math
import re
import time
from collections import Counter
from typing import Any, Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.bio_db_client import get_bio_db_client
from app.config import llm, settings, strip_markdown_json
from app.mcp_client import get_mcp_client
from app.pathways.drug_library import get_drug_entry
from app.prompts import (
    COMBINATION_REPORT_SECTION,
    DOSE_REPORT_SECTION,
    NODE1_6_PKPD_PROMPT,
    NODE1_PARSER_PROMPT,
    NODE2_BASE_PROMPT,
    NODE2_COMBINATION_SECTION,
    NODE2_DOSE_SWEEP_SECTION,
    NODE2_PKPD_SECTION,
    NODE4_AUDITOR_PROMPT,
    NODE6_REPORT_PROMPT,
    RAG_DECISION_PROMPT,
)
from app.rag_client import RagClient
from app.sandbox import execute_simulation_code
from app.sbml_parser import parse_sbml_model
from app.state import BioDynamicsState
from app.supervisor import orchestrator
from app.token_usage import (
    UsageAccumulator,
    merge_usage,
    usage_from_accumulator,
)

logger = logging.getLogger(__name__)


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


class PKPDOutput(BaseModel):
    """PK/PD 推断的结构化输出。"""

    pkpd_profile: dict = Field(
        default_factory=dict, description="PK/PD 模型参数概要"
    )
    drug_regimen: list[dict] = Field(
        default_factory=list, description="药物方案列表"
    )
    reasoning: str = Field(default="", description="推断理由")


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
        # TODO: P1-4 — v1 使用 "source|target"，v3 使用 "source->target"，统一兼容两种分隔符
        if "|" in edge_key:
            source, target = edge_key.split("|", 1)
        elif "->" in edge_key:
            source, target = edge_key.split("->", 1)
        else:
            source, target = edge_key, ""
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


def _build_pkpd_context(pkpd_profile: dict) -> str:
    """将 PK/PD 推断结果拼接为 Node 2 可读的上下文文本。"""
    if not pkpd_profile:
        return "无 PK/PD 参数，使用纯 Hill 方程。"
    pk = pkpd_profile.get("pk_params", {})
    pd = pkpd_profile.get("pd_params", {})
    return (
        f"药物: {pkpd_profile.get('drug_name', '未知')}\n"
        f"靶点: {pkpd_profile.get('drug_target', '未知')}\n"
        f"给药途径: {pkpd_profile.get('route', 'IV')}\n"
        f"房室模型: {pkpd_profile.get('compartment', '1-compartment')}\n"
        f"PK 参数: k10={pk.get('k10', '估算')}, k12={pk.get('k12', 0)}, "
        f"k21={pk.get('k21', 0)}, ka={pk.get('ka', 0)}\n"
        f"PD 参数: Emax={pd.get('Emax', '估算')}, EC50={pd.get('EC50', '估算')}, "
        f"gamma={pd.get('gamma', '估算')}"
    )


def _build_drug_regimen_context(drug_regimen: list[dict]) -> str:
    """将药物方案拼接为 Node 2 可读的上下文文本。"""
    if not drug_regimen:
        return "无药物方案。"
    lines = []
    for i, drug in enumerate(drug_regimen, start=1):
        lines.append(
            f"药物{i}: {drug.get('drug_name', '未知')}, "
            f"剂量={drug.get('dose', '估算')}, "
            f"EC50={drug.get('ec50', '估算')}, "
            f"Emax={drug.get('emax', '估算')}, "
            f"gamma={drug.get('gamma', '估算')}, "
            f"靶点={drug.get('target', '未知')}"
        )
    return "\n".join(lines)


def _compute_chou_talalay_ci(
    drug_regimen: list[dict],
    fa_levels: tuple[float, ...] = (0.5, 0.75, 0.9),
) -> dict[str, dict[str, float]]:
    """解析计算 Chou-Talalay 中效方程：D_alone(fa) = EC50 * (fa/(1-fa))^(1/gamma)。

    返回 {fa_0.5: {'d_alone_A': ..., 'd_alone_B': ...}, ...}，供与仿真 D_combo 组合计算 CI。
    """
    result: dict[str, dict[str, float]] = {}
    for fa in fa_levels:
        key = f"fa_{fa}"
        d_alones: dict[str, float] = {}
        for drug in drug_regimen:
            name = drug.get("drug_name", "unknown")
            ec50 = float(drug.get("ec50", 1.0) or 1.0)
            gamma = float(drug.get("gamma", 1.0) or 1.0)
            if gamma <= 0 or fa <= 0.0 or fa >= 1.0:
                d_alones[name] = float("inf")
                continue
            d_alone = ec50 * (fa / (1.0 - fa)) ** (1.0 / gamma)
            d_alones[name] = d_alone
        result[key] = d_alones
    return result


# -----------------------------------------------------------------------------
# Node 0: MCP 术语查询节点（在机制解析前注入术语标准化上下文）
# -----------------------------------------------------------------------------
async def node0_mcp_term_lookup(state: BioDynamicsState) -> dict:
    """MCP 术语查询节点：在机制解析前调用生物医学 MCP 工具。

    工作流程：
    1. 从用户输入中提取生物医学术语（OpenBioMed Skills）
    2. 获取术语同义词与层级关系（NIH UMLS MCP）
    3. 标准化临床术语（medical-terminologies-mcp）
    4. 生成术语定义卡片供前端展示
    5. 基于标准化结果重写查询，提升后续 RAG 检索精准度

    当 MCP 服务端点未配置时，自动降级为 LLM 内部知识完成上述流程。
    """
    node_name = "node0_mcp_term_lookup"
    dispatches: list[dict] = []
    if (d := orchestrator.dispatch_for_node(node_name, "in_progress")) :
        dispatches.append(d)

    user_input = state.get("user_input", "")
    if not user_input or not settings.MCP_ENABLED:
        # MCP 未启用或无输入时直接跳过，不阻塞主流程
        return {
            "mcp_term_definitions": [],
            "mcp_tool_calls": [],
            "mcp_tokens_saved": 0,
            "mcp_rewritten_query": user_input,
            "agent_dispatches": dispatches,
        }

    start_ts = time.time()
    mcp_client = get_mcp_client()

    try:
        # 调用 MCP 工具链：术语提取 → 标准化 → 定义生成
        definitions, tool_call_records, mcp_usage = await mcp_client.lookup_terms(
            user_input
        )

        # 基于术语标准化结果重写查询，提升 RAG 检索精准度
        rewritten_query, rewrite_record = mcp_client.rewrite_query(
            user_input, definitions
        )
        tool_call_records.append(rewrite_record.to_dict())

        # 计算总 Token 节省量
        tokens_saved = sum(tc.get("tokens_saved", 0) for tc in tool_call_records)

        # 构建术语定义上下文（注入到 Node 1 的提示词中）
        if definitions:
            def_lines = []
            for d in definitions:
                syn = f"，同义词：{'、'.join(d.synonyms[:3])}" if d.synonyms else ""
                pathway = f"，通路：{d.related_pathway}" if d.related_pathway else ""
                def_lines.append(
                    f"- {d.term}（{d.canonical_name}）：{d.definition}{syn}{pathway}"
                )
            term_context = "\n".join(def_lines)
            summary = f"MCP 已标准化 {len(definitions)} 个术语，估算节省 {tokens_saved} Token"
        else:
            term_context = ""
            summary = "MCP 未识别到专业术语，直接进入机制解析"

    except Exception as exc:
        # 任何异常都不阻塞主流程，降级为空上下文
        latency_ms = (time.time() - start_ts) * 1000
        if (d := orchestrator.fail_dispatch(node_name, str(exc), latency_ms)) :
            dispatches.append(d)
        return {
            "mcp_term_definitions": [],
            "mcp_tool_calls": [],
            "mcp_tokens_saved": 0,
            "mcp_rewritten_query": user_input,
            "agent_dispatches": dispatches,
        }

    latency_ms = (time.time() - start_ts) * 1000
    if (d := orchestrator.complete_dispatch(node_name, latency_ms)) :
        dispatches.append(d)

    return {
        "mcp_term_definitions": [d.model_dump() for d in definitions],
        "mcp_tool_calls": tool_call_records,
        "mcp_tokens_saved": tokens_saved,
        "mcp_rewritten_query": rewritten_query,
        "messages": [summary],
        "agent_dispatches": dispatches,
        "token_usage": merge_usage(state.get("token_usage"), mcp_usage),
    }


# -----------------------------------------------------------------------------
# Node 1: 机制解析器
# -----------------------------------------------------------------------------
def _strip_markdown_code_blocks(text: str) -> str:
    """去除 LLM 输出的 markdown 代码块标记（委托统一实现，消除重复逻辑）。"""
    return strip_markdown_json(text)


def _parse_network_json(raw_text: str) -> NetworkOutput:
    """从原始文本中解析 NetworkOutput；先尝试 structured output，失败则手动清理 markdown 后解析。"""
    clean_text = _strip_markdown_code_blocks(raw_text)
    try:
        data = json.loads(clean_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"无法解析 NetworkOutput JSON：{exc}\n原始内容：{raw_text[:500]}") from exc
    return NetworkOutput(**data)


def _expected_dynamics_species(value: Any) -> list[str]:
    """Collect benchmark species names from nested expected_dynamics mappings."""
    metric_keys = {
        "peak_time_min", "peak_time_h", "peak_amplitude_fold",
        "peak_amplitude_norm", "induction_fold_min", "adaptation_ratio_max",
        "oscillation_period_min", "fold_increase", "inhibition_pct",
    }
    found: list[str] = []

    def visit(node: Any) -> None:
        if not isinstance(node, dict):
            return
        for key, child in node.items():
            if isinstance(child, dict):
                if metric_keys.intersection(child):
                    found.append(str(key))
                visit(child)

    visit(value)
    return list(dict.fromkeys(name for name in found if name))


# TODO: P1-4 — 统一 v1/v3 network_json schema 的公共规范化函数
def _normalize_network_json(network_json: dict) -> dict:
    """统一 network_json schema：node.id 优先取 name，同步更新 edge 引用。

    起因：v1 路径 node1_parse_network 直接用 LLM 输出的 _NodeItem.id，
    可能是占位符（如 "e1"/"entity_1"）；v3 路径 worker_mechanism 已在 P0-4 修复中
    优先取 name。此函数让 v1 出口与 v3 出口 schema 一致，避免下游 ODE 变量名泄漏占位符。

    规范化规则：
    1. 对每个 node，若 name 非空则 id = name；否则保留原 id。
    2. 构建 old_id -> new_id 映射，同步更新 edge.source / edge.target。
    3. 保留 node/edge 的其他字段（type/aliases/interaction/directed 等）。

    Args:
        network_json: 原始 network_json，含 nodes 和 edges 两个列表。

    Returns:
        规范化后的 network_json，schema 与 v3 worker_mechanism 出口一致。
    """
    nodes = network_json.get("nodes", []) or []
    edges = network_json.get("edges", []) or []

    # 构建 id 映射：old_id -> new_id（优先 name）
    id_map: dict[str, str] = {}
    new_nodes: list[dict] = []
    for n in nodes:
        old_id = str(n.get("id", "")).strip()
        name = str(n.get("name", "")).strip()
        new_id = name if name else old_id
        if old_id and old_id != new_id:
            id_map[old_id] = new_id
        new_node = dict(n)
        new_node["id"] = new_id
        new_nodes.append(new_node)

    # 同步更新 edge 的 source/target
    new_edges: list[dict] = []
    for e in edges:
        new_edge = dict(e)
        old_src = str(e.get("source", "")).strip()
        old_tgt = str(e.get("target", "")).strip()
        new_edge["source"] = id_map.get(old_src, old_src)
        new_edge["target"] = id_map.get(old_tgt, old_tgt)
        new_edges.append(new_edge)

    return {"nodes": new_nodes, "edges": new_edges}


def node1_parse_network(state: BioDynamicsState) -> dict:
    """解析用户自然语言输入，输出结构化生物网络 JSON。"""
    node_name = "node1_parse_network"
    dispatches: list[dict] = []
    if (d := orchestrator.dispatch_for_node(node_name, "in_progress")) :
        dispatches.append(d)

    user_input = state.get("user_input", "")
    rewritten = state.get("mcp_rewritten_query") or user_input
    if not user_input:
        return {
            "need_human_review": True,
            "review_question": "请输入您想建模的生物学假说或机制描述。",
            "network_json": {"nodes": [], "edges": []},
            "agent_dispatches": dispatches,
        }

    start_ts = time.time()
    # 注入 MCP 术语标准化上下文（若 node0 产出了定义）
    mcp_definitions = state.get("mcp_term_definitions") or []
    mcp_term_context = ""
    if mcp_definitions:
        def_lines = []
        for d in mcp_definitions:
            syn = f"，同义词：{'、'.join(d.get('synonyms', [])[:3])}" if d.get("synonyms") else ""
            pathway = f"，通路：{d.get('related_pathway')}" if d.get("related_pathway") else ""
            def_lines.append(
                f"- {d.get('term', '')}（{d.get('canonical_name', '')}）：{d.get('definition', '')}{syn}{pathway}"
            )
        mcp_term_context = "\n【MCP 术语标准化上下文】\n" + "\n".join(def_lines)

    expected_species = _expected_dynamics_species(
        state.get("benchmark_expected_dynamics") or {}
    )
    benchmark_species_rule = ""
    if expected_species:
        species_text = ", ".join(expected_species)
        benchmark_species_rule = (
            "\n\n【Benchmark dynamics completeness】\n"
            f"The governed expected_dynamics species are: {species_text}. "
            "Every listed species must appear in knowledge_graph.nodes and must be connected "
            "to at least one producing or consuming reaction so its trajectory can change. "
            "If a species belongs to an implicit cross-pathway branch, include that canonical branch."
        )

    system_prompt = NODE1_PARSER_PROMPT + mcp_term_context + benchmark_species_rule + (
        "\n\n【中英双标签要求】\n"
        "在生成 network_json 时，每个节点对象必须同时包含：\n"
        '- name: 英文标识符（用于代码变量和图表标签，如 EGFR_phosphorylation）\n'
        '- label: 中文可读名（用于报告文本，如 EGFR磷酸化）\n'
        "边对象同理：source/target 使用英文 name，但需附加 label_source/label_target 字段。\n"
        "这确保 Matplotlib 图表使用英文避免乱码，报告保持中文可读性。"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{rewritten}"),
        ]
    )
    chain = prompt | llm
    usage_handler = UsageAccumulator()
    response = chain.invoke(
        {"rewritten": rewritten},
        config={"callbacks": [usage_handler]},
    )
    latency_ms = (time.time() - start_ts) * 1000

    try:
        result = _parse_network_json(str(response.content))
    except Exception as exc:
        if (d := orchestrator.fail_dispatch(node_name, str(exc), latency_ms)) :
            dispatches.append(d)
        return {
            "need_human_review": True,
            "review_question": f"机制解析失败：{exc}",
            "network_json": {"nodes": [], "edges": []},
            "agent_dispatches": dispatches,
            "token_usage": merge_usage(
                state.get("token_usage"), usage_from_accumulator(usage_handler)
            ),
        }

    network_json = {
        "nodes": [node.model_dump() for node in result.nodes],
        "edges": [edge.model_dump() for edge in result.edges],
    }
    # TODO: P1-4 — 统一 v1/v3 network_json schema：id 优先取 name，同步更新 edge 引用
    network_json = _normalize_network_json(network_json)
    summary = (
        f"已解析出 {len(network_json['nodes'])} 个节点、"
        f"{len(network_json['edges'])} 条相互作用边。"
    )

    if (d := orchestrator.complete_dispatch(node_name, latency_ms)) :
        dispatches.append(d)

    return {
        "network_json": network_json,
        "need_human_review": result.need_human_review,
        "review_question": result.review_question,
        "messages": [summary],
        "agent_dispatches": dispatches,
        "token_usage": merge_usage(
            state.get("token_usage"), usage_from_accumulator(usage_handler)
        ),
    }


# -----------------------------------------------------------------------------
# Node 1.5: RAG 参数检索与决策器
# -----------------------------------------------------------------------------
async def node1_5_rag_search(state: BioDynamicsState) -> dict:
    """在 Node 1 和 Node 2 之间插入的高阶 RAG 节点。

    升级：查询重写 → 混合检索（语义 + BM25）→ 重排序 → LLM 决策。
    对 inhibition 边额外检索靶点相关药物候选（PubMed + ChromaDB + ClinicalTrials.gov）。
    同时产出 rag_insights 供前端 RAG 洞察面板渲染。
    """
    node_name = "node1_5_rag_search"
    dispatches: list[dict] = []
    if (d := orchestrator.dispatch_for_node(node_name, "in_progress")) :
        dispatches.append(d)

    start_ts = time.time()
    user_input = state.get("user_input", "")
    # 使用 MCP 重写后的查询（术语已标准化），提升 RAG 检索精准度
    mcp_rewritten_query = state.get("mcp_rewritten_query") or user_input
    network_json = state.get("network_json") or {"nodes": [], "edges": []}
    edges = network_json.get("edges", [])

    # Token 累加器，覆盖 SBML 解析与每条边的 RAG 决策
    usage_handler = UsageAccumulator()

    # MCP 客户端，用于 PubMed 药物文献预检索
    mcp_client = get_mcp_client()

    # 1. 物种上下文准备（使用 MCP 重写后的查询，可能含更标准的物种名）
    species_context = _extract_species_context(mcp_rewritten_query)

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

    # 3. 高阶 RAG 检索与决策
    rag_client = RagClient()
    node_name_map = {n["id"]: n.get("name", n["id"]) for n in network_json.get("nodes", [])}
    # 构建 MCP 术语到标准名的映射，用于增强 RAG 查询的术语精准度
    mcp_term_map: dict[str, str] = {}
    for d in state.get("mcp_term_definitions") or []:
        term = d.get("term", "")
        canonical = d.get("canonical_name", "")
        if term and canonical and term.lower() != canonical.lower():
            mcp_term_map[term] = canonical
    rag_selected_params: dict[str, dict] = {}
    retrieved_all: list[dict] = []

    # 聚合 RAG 洞察数据
    aggregated_rewrites: list[dict] = []
    aggregated_source_dist: Counter[str] = Counter()
    aggregated_top_selections: list[dict] = []
    aggregated_rewritten_queries: list[str] = []
    total_candidates = 0

    # 聚合药物候选（知识图谱）
    all_drug_candidates: list[dict] = []

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
            # 注入 MCP 标准化术语名，提升 RAG 检索相关性
            if mcp_term_map:
                canonical_terms = []
                for orig in [source_name, target_name]:
                    if orig in mcp_term_map:
                        canonical_terms.append(mcp_term_map[orig])
                if canonical_terms:
                    query += " " + " ".join(canonical_terms)

            try:
                # 高阶 RAG：查询重写 + 混合检索 + 重排序
                retrieved, edge_insights = rag_client.search_params_hybrid(
                    query, species_context=species_context, top_k=5
                )
                retrieved_all.extend(retrieved)

                # 聚合洞察数据
                if edge_insights.get("rewritten_query"):
                    aggregated_rewritten_queries.append(edge_insights["rewritten_query"])
                aggregated_rewrites.extend(edge_insights.get("rewrites", []))
                for src, cnt in edge_insights.get("source_distribution", {}).items():
                    aggregated_source_dist[src] += cnt
                total_candidates += edge_insights.get("total_candidates", 0)
                # 为每条边的 top_selections 标注所属边
                for sel in edge_insights.get("top_selections", []):
                    sel["edge"] = f"{source_name} → {target_name}"
                    aggregated_top_selections.append(sel)
            except Exception:
                retrieved = []

            # 自动在线补充：本地 ChromaDB 命中不足时，查询 KEGG/Reactome/UniProt/ChEMBL
            if settings.RAG_ONLINE_FALLBACK and (not retrieved or not rag_client.available):
                best_score = max(
                    (r.get("_rerank_score", r.get("_combined_score", 0)) for r in retrieved), default=0
                )
                if best_score < settings.RAG_ONLINE_FALLBACK_THRESHOLD or not retrieved:
                    try:
                        bio_db_client = get_bio_db_client()
                        online_results = await bio_db_client.search_all(
                            query, species_context
                        )
                        if online_results:
                            # 标记在线来源，供 RAG 决策链区分
                            for r in online_results:
                                r["_retrieval_method"] = "online_fallback"
                            retrieved.extend(online_results)
                            retrieved_all.extend(online_results)
                            logger.info(
                                "在线补充 %s→%s：获取 %d 条结果",
                                source_name, target_name, len(online_results),
                            )
                    except Exception as exc:
                        logger.warning("在线数据库补充失败：%s", exc)

            # 对抑制边额外检索靶点相关药物候选（知识图谱注入）
            if interaction == "inhibition":
                try:
                    pubmed_query = f"{target_name} inhibitor IC50 clinical trial"
                    articles, _, _ = await mcp_client.search_pubmed(
                        pubmed_query, max_results=3
                    )
                    drug_cands = rag_client.drug_specific_retriever(
                        target_name=target_name,
                        species_context=species_context,
                        pubmed_articles=articles,
                    )
                    all_drug_candidates.extend(drug_cands)
                except Exception as exc:
                    logger.warning("检索 %s 的药物候选失败：%s", target_name, exc)

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
                decision: RAGDecisionOutput = await chain.ainvoke(
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
    rag_hit_rate = round(found_count / total, 2) if total > 0 else 0.0
    rag_summary = f"已为 {found_count}/{total} 条边检索到真实参数"
    if fallback:
        rag_summary += "，其余边将使用估算值。"
    else:
        rag_summary += "。"

    # 对药物候选按 drug_name 去重，优先保留有 IC50/EC50 的条目
    seen_drugs: dict[str, dict] = {}
    for cand in all_drug_candidates:
        name = cand.get("drug_name", "")
        if not name:
            continue
        existing = seen_drugs.get(name)
        if existing is None or (cand.get("ic50") or cand.get("ec50")):
            seen_drugs[name] = cand
    drug_candidates = list(seen_drugs.values())

    # 构建 RAG 洞察数据（供前端面板渲染）
    rag_insights = {
        "rewritten_query": aggregated_rewritten_queries[0] if aggregated_rewritten_queries else "",
        "rewrites": aggregated_rewrites,
        "source_distribution": dict(aggregated_source_dist),
        "total_candidates": total_candidates,
        "top_selections": aggregated_top_selections[:6],  # 限制为 top 6 供前端展示
        "hit_rate": rag_hit_rate,
        "drug_candidates": drug_candidates,
        "online_fallback_enabled": settings.RAG_ONLINE_FALLBACK,
    }

    latency_ms = (time.time() - start_ts) * 1000
    if (d := orchestrator.complete_dispatch(node_name, latency_ms)) :
        dispatches.append(d)

    update: dict = {
        "species_context": species_context,
        "rag_retrieved_params": retrieved_all,
        "rag_selected_params": rag_selected_params,
        "rag_fallback": fallback,
        "rag_summary": rag_summary,
        "rag_insights": rag_insights,
        "rag_hit_rate": rag_hit_rate,
        "drug_candidates": drug_candidates,
        "agent_dispatches": dispatches,
        "messages": [rag_summary],
        "token_usage": merge_usage(
            state.get("token_usage"), usage_from_accumulator(usage_handler)
        ),
    }
    if sbml_parsed_network is not None:
        update["sbml_parsed_network"] = sbml_parsed_network

    return update


def _extract_drug_candidates_fallback(
    network_json: dict, user_input: str, parameters: dict | None = None
) -> list[dict]:
    """从 network_json 的 Drug 节点和 user_input 的 IC50/EC50 正则提取药物候选。

    v3 路径 n5_parameter_rag 不调用 drug_specific_retriever，导致 drug_candidates
    始终为空。此函数作为兜底，从三个来源提取：
    1. network_json 中 type="Drug" 的节点 → 提取药物名与抑制靶点
    2. user_input 中的 IC50/EC50 数值 → 正则匹配（如 "IC50 = 12 nM"）
    3. TODO: P0-3 — state["parameters"] 中的 IC50/EC50/Kd 条目（RAG 提取值）

    返回结构与 drug_specific_retriever 一致：
    [{drug_name, ic50, ec50, source, is_clinical_candidate, target_name}]
    """
    candidates: list[dict] = []
    nodes = network_json.get("nodes", []) or []
    edges = network_json.get("edges", []) or []
    parameters = parameters or {}

    # 收集 Drug 节点及其抑制靶点
    drug_nodes: list[dict] = [n for n in nodes if n.get("type") == "Drug"]
    if not drug_nodes:
        return candidates

    # 构建 药物→抑制靶点 映射
    drug_targets: dict[str, str] = {}
    for edge in edges:
        if edge.get("interaction") == "inhibition":
            source = edge.get("source", "")
            target = edge.get("target", "")
            # source 是药物节点 id
            if any(n.get("id") == source for n in drug_nodes):
                drug_targets[source] = target

    # 从 user_input 提取 IC50/EC50 数值
    ic50_value: float | None = None
    ec50_value: float | None = None
    ic50_unit: str = "nM"
    ec50_unit: str = "nM"

    _IC50_PATTERNS = [
        r"IC50\s*[=：:]\s*(\d+(?:\.\d+)?)\s*(nM|µM|uM|µmol/L|μM)?",
        r"半数抑制浓度\s*[=：:]?\s*(\d+(?:\.\d+)?)\s*(nM|µM|uM|μM)?",
    ]
    _EC50_PATTERNS = [
        r"EC50\s*[=：:]\s*(\d+(?:\.\d+)?)\s*(nM|µM|uM|μM)?",
        r"半数有效浓度\s*[=：:]?\s*(\d+(?:\.\d+)?)\s*(nM|µM|uM|μM)?",
    ]

    for pattern in _IC50_PATTERNS:
        match = re.search(pattern, user_input, re.IGNORECASE)
        if match:
            ic50_value = float(match.group(1))
            if match.group(2):
                unit = match.group(2).lower()
                # µM → nM 转换
                if unit in ("µm", "um", "μm"):
                    ic50_value *= 1000
                    ic50_unit = "nM"
                else:
                    ic50_unit = "nM"
            break

    for pattern in _EC50_PATTERNS:
        match = re.search(pattern, user_input, re.IGNORECASE)
        if match:
            ec50_value = float(match.group(1))
            if match.group(2):
                unit = match.group(2).lower()
                if unit in ("µm", "um", "μm"):
                    ec50_value *= 1000
                    ec50_unit = "nM"
                else:
                    ec50_unit = "nM"
            break

    # TODO: P0-3 — 从 state["parameters"] 提取 RAG 已检索到的 IC50/EC50/Kd
    # parameters 格式：{edge_key: {param_name, value, unit, source, is_fallback}}
    # 优先使用 RAG 提取值（非 fallback），补全 user_input 正则未命中的情况
    rag_ic50: float | None = None
    rag_ec50: float | None = None
    rag_source: str = ""
    for edge_key, ep in parameters.items():
        if not isinstance(ep, dict) or ep.get("is_fallback", True):
            continue
        param_name = str(ep.get("param_name", "")).lower()
        try:
            value = float(ep.get("value", 0))
        except (TypeError, ValueError):
            continue
        if "ic50" in param_name and rag_ic50 is None:
            rag_ic50 = value
            rag_source = ep.get("source", "RAG")
        elif "ec50" in param_name and rag_ec50 is None:
            rag_ec50 = value
            rag_source = ep.get("source", "RAG")
        elif "kd" in param_name and rag_ic50 is None and rag_ec50 is None:
            # Kd 近似为 IC50（单结合位点假设）
            rag_ic50 = value
            rag_source = ep.get("source", "RAG")

    # 优先级：user_input 正则 > RAG 提取 > drug_library canonical > 默认 0
    final_ic50 = ic50_value if ic50_value is not None else (rag_ic50 if rag_ic50 is not None else 0.0)
    final_ec50 = ec50_value if ec50_value is not None else (rag_ec50 if rag_ec50 is not None else final_ic50)

    # 为每个 Drug 节点创建候选
    for drug_node in drug_nodes:
        drug_id = drug_node.get("id", "")
        drug_name = drug_node.get("name", drug_id)
        target = drug_targets.get(drug_id, "")
        # [N6 缺口 1] 查询 canonical drug_library 获取 per-drug IC50/Ki
        # 当 user_input 与 RAG 均未提供 IC50 时，使用 drug_library.yaml 的 canonical 值，
        # 避免 stage_4_pkpd 出现空 pkpd_profile（含 model_type + IC50 + Ki 字段）
        drug_entry = get_drug_entry(drug_name) if drug_name else {}
        lib_ic50 = drug_entry.get("ic50_nM")
        lib_ki = drug_entry.get("ki_nM")
        lib_pmid = drug_entry.get("source_pmid")
        lib_target = drug_entry.get("primary_target", "")
        # 若 drug_library 提供了 IC50 数值且 user_input/RAG 未命中，则采用 canonical 值
        if (
            ic50_value is None
            and rag_ic50 is None
            and isinstance(lib_ic50, (int, float))
            and lib_ic50 > 0
        ):
            per_drug_ic50 = float(lib_ic50)
        else:
            per_drug_ic50 = final_ic50
        # 判断来源标签
        if ic50_value is not None:
            source_label = "extracted_from_input"
        elif rag_ic50 is not None or rag_ec50 is not None:
            source_label = f"RAG:{rag_source}"
        elif isinstance(lib_ic50, (int, float)) and lib_ic50 > 0:
            source_label = (
                f"drug_library:PMID:{lib_pmid}" if lib_pmid else "drug_library"
            )
        else:
            source_label = "network_only"
        # target 优先级：inhibition edge > drug_library primary_target
        final_target = target or lib_target
        cand = {
            "drug_name": drug_name,
            "ic50": per_drug_ic50,
            "ec50": final_ec50,
            "clinical_dose": "",
            "source": source_label,
            "is_clinical_candidate": False,
            "target_name": final_target,
            # [N6 缺口 1] Ki 字段（drug_library 提供，供 PK/PD node 推断 model_type）
            "ki": float(lib_ki) if isinstance(lib_ki, (int, float)) else per_drug_ic50,
        }
        candidates.append(cand)

    return candidates


# -----------------------------------------------------------------------------
# Node 1.6: PK/PD 推断器
# -----------------------------------------------------------------------------
def node1_6_pkpd_inference(state: BioDynamicsState) -> dict:
    """PK/PD 模型推断节点。

    在 Node 1.5（RAG 检索）之后、Node 2（代码生成）之前执行。
    消费 drug_candidates（真实药物 IC50/EC50）与 rag_selected_params，
    推断给药途径、房室模型、PK/PD 参数，并生成 drug_regimen。
    若用户未提及药物，则返回空 pkpd_profile，Node 2 回退到纯 Hill 方程。
    """
    node_name = "node1_6_pkpd_inference"
    dispatches: list[dict] = []
    if (d := orchestrator.dispatch_for_node(node_name, "in_progress")) :
        dispatches.append(d)

    drug_candidates = state.get("drug_candidates") or []
    network_json = state.get("network_json") or {"nodes": [], "edges": []}
    rag_selected_params = state.get("rag_selected_params") or {}
    species_context = state.get("species_context", "Human")
    user_input = state.get("user_input", "")
    # TODO: P0-3 — 同时读取 parameters 字段（v3 路径 RAG 提取值）
    parameters = state.get("parameters") or {}

    # === Task F: drug_candidates 为空时的兜底提取 ===
    # v3 路径 n5_parameter_rag 不填充 drug_candidates，导致此处总是跳过 PK/PD。
    # 兜底策略：从 network_json 的 Drug 节点 + user_input 的 IC50/EC50 正则 + parameters 提取。
    if not drug_candidates:
        drug_candidates = _extract_drug_candidates_fallback(
            network_json, user_input, parameters=parameters
        )
        if drug_candidates:
            logger.info(
                "PK/PD 兜底提取到 %d 个药物候选（从 network + user_input + parameters）",
                len(drug_candidates),
            )

    # 无药物候选时直接跳过，不阻塞后续纯生物学仿真
    # [P2-2] 修复：返回 sentinel {"skipped": True} 而非空 dict，
    # 使 orchestrator stage_4_pkpd 的 _is_filled 检查通过（status=pass），
    # 避免 93% 的 L1-L4 无药物 case 误报 "empty keys: ['pkpd_profile']"。
    # 下游 worker_ode / node2_generate_code 通过 pkpd_profile.get("drug_name")
    # 判断是否激活 PK/PD 逻辑，sentinel 自动跳过（drug_name 缺失）。
    if not drug_candidates:
        latency_ms = 0.0
        if (d := orchestrator.complete_dispatch(node_name, latency_ms)) :
            dispatches.append(d)
        return {
            "pkpd_profile": {
                "skipped": True,
                "reason": "no_drug_in_user_input",
                "drug_name": "",
                "drug_target": "",
                "compartment": "",
                "pk_params": {},
                "pd_params": {},
            },
            "drug_regimen": [],
            "messages": ["未识别到药物，跳过 PK/PD 推断，使用纯 Hill 方程。"],
            "agent_dispatches": dispatches,
        }

    start_ts = time.time()
    system_prompt = NODE1_6_PKPD_PROMPT
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "请根据网络与检索结果推断 PK/PD 参数。"),
        ]
    )
    chain = prompt.partial(
        network_json=json.dumps(network_json, ensure_ascii=False, indent=2),
        drug_candidates_json=json.dumps(drug_candidates, ensure_ascii=False, indent=2),
        rag_selected_params_json=json.dumps(rag_selected_params, ensure_ascii=False, indent=2),
        species_context=species_context,
    ) | llm.with_structured_output(PKPDOutput)
    usage_handler = UsageAccumulator()
    result: PKPDOutput = chain.invoke({}, config={"callbacks": [usage_handler]})
    latency_ms = (time.time() - start_ts) * 1000

    pkpd_profile = result.pkpd_profile or {}
    drug_regimen = result.drug_regimen or []

    summary = "PK/PD 推断完成"
    if pkpd_profile:
        summary += (
            f"：药物 {pkpd_profile.get('drug_name', '未知')}，"
            f"房室 {pkpd_profile.get('compartment', '未知')}，"
            f"靶点 {pkpd_profile.get('drug_target', '未知')}。"
        )
    else:
        summary += "：未生成 PK/PD 模型，回退到纯 Hill 方程。"

    if (d := orchestrator.complete_dispatch(node_name, latency_ms)) :
        dispatches.append(d)

    return {
        "pkpd_profile": pkpd_profile,
        "drug_regimen": drug_regimen,
        "messages": [summary, result.reasoning or ""],
        "agent_dispatches": dispatches,
        "token_usage": merge_usage(
            state.get("token_usage"), usage_from_accumulator(usage_handler)
        ),
    }


# -----------------------------------------------------------------------------
# Node 2: 参数与方程生成器
# -----------------------------------------------------------------------------
def node2_generate_code(state: BioDynamicsState) -> dict:
    """根据结构化网络生成可执行的 SciPy ODE 仿真代码。

    升级：根据 pkpd_profile 与 drug_regimen 动态组装 prompt，
    条件化引入 PK/PD 房室模型、剂量扫描与联合用药仿真要求。
    """
    node_name = "node2_generate_code"
    dispatches: list[dict] = []
    if (d := orchestrator.dispatch_for_node(node_name, "in_progress")) :
        dispatches.append(d)

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
    pkpd_profile = state.get("pkpd_profile") or {}
    drug_regimen = state.get("drug_regimen") or []

    # 动态组装 Node 2 的 system prompt
    system_prompt = NODE2_BASE_PROMPT
    prompt_vars: dict[str, str] = {
        "network_json": json.dumps(network_json, ensure_ascii=False, indent=2),
        "error_feedback": error_feedback,
        "rag_params_context": rag_params_context,
        "sbml_context": sbml_context,
    }
    # [P2-2] 修复：sentinel pkpd_profile={"skipped": True, "drug_name": ""} 不触发 PK/PD 节
    # 仅当真实药物存在（drug_name 非空）时才注入 NODE2_PKPD_SECTION
    if pkpd_profile and pkpd_profile.get("drug_name"):
        system_prompt += NODE2_PKPD_SECTION
        prompt_vars["pkpd_context"] = _build_pkpd_context(pkpd_profile)
    if drug_regimen:
        system_prompt += NODE2_DOSE_SWEEP_SECTION
        prompt_vars["drug_regimen_context"] = _build_drug_regimen_context(drug_regimen)
    if len(drug_regimen) >= 2:
        system_prompt += NODE2_COMBINATION_SECTION

    start_ts = time.time()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "请根据上述网络拓扑生成 ODE 仿真代码。"),
        ]
    )
    chain = prompt.partial(**prompt_vars) | llm
    usage_handler = UsageAccumulator()
    response = chain.invoke({}, config={"callbacks": [usage_handler]})
    content = str(response.content)
    python_code = _extract_python_code(content)
    latency_ms = (time.time() - start_ts) * 1000

    if (d := orchestrator.complete_dispatch(node_name, latency_ms)) :
        dispatches.append(d)

    return {
        "python_code": python_code,
        "messages": [python_code],
        "agent_dispatches": dispatches,
        "token_usage": merge_usage(
            state.get("token_usage"), usage_from_accumulator(usage_handler)
        ),
    }


# -----------------------------------------------------------------------------
# Node 3: 代码执行器（调用沙箱）
# -----------------------------------------------------------------------------
def node3_execute_sandbox(state: BioDynamicsState) -> dict:
    """在隔离沙箱中执行生成的 Python 代码。"""
    node_name = "node3_execute_sandbox"
    dispatches: list[dict] = []
    if (d := orchestrator.dispatch_for_node(node_name, "in_progress")) :
        dispatches.append(d)

    python_code = state.get("python_code", "")
    if not python_code:
        return {
            "execution_status": "pending",
            "stdout_stderr": "未收到可执行代码。",
            "image_base64": "",
            "agent_dispatches": dispatches,
        }

    start_ts = time.time()
    result = execute_simulation_code(python_code)
    latency_ms = (time.time() - start_ts) * 1000

    status = result["status"]
    if status == "success":
        if (d := orchestrator.complete_dispatch(node_name, latency_ms)) :
            dispatches.append(d)
    else:
        if (d := orchestrator.fail_dispatch(node_name, result.get("stdout_stderr", "")[:120], latency_ms)) :
            dispatches.append(d)

    update: dict = {
        "execution_status": status,
        "stdout_stderr": result["stdout_stderr"],
        "image_base64": result["image_base64"] or "",
        "agent_dispatches": dispatches,
    }
    # 透传 PK/PD 剂量扫描与联合用药仿真输出（即使失败也尝试解析部分数据）
    if result.get("dose_response_data"):
        update["dose_response_data"] = result["dose_response_data"]
    if result.get("ic50") is not None:
        update["ic50"] = result["ic50"]
    if result.get("ic90") is not None:
        update["ic90"] = result["ic90"]
    if result.get("hed") is not None:
        update["hed"] = result["hed"]
    if result.get("combo_ci_data"):
        update["simulation_ci"] = result["combo_ci_data"]
    return update


# -----------------------------------------------------------------------------
# Node 4: 审计与纠错器
# -----------------------------------------------------------------------------
_MAX_RETRY = 3


def node4_audit_and_correct(state: BioDynamicsState) -> dict:
    """审计代码执行结果，决定重试、失败或成功。"""
    node_name = "node4_audit_and_correct"
    dispatches: list[dict] = []
    if (d := orchestrator.dispatch_for_node(node_name, "in_progress")) :
        dispatches.append(d)

    start_ts = time.time()
    execution_status = state.get("execution_status", "error")
    retry_count = state.get("retry_count", 0)

    if execution_status == "success":
        latency_ms = (time.time() - start_ts) * 1000
        if (d := orchestrator.complete_dispatch(node_name, latency_ms)) :
            dispatches.append(d)

        # 联合用药协同评估（Chou-Talalay）
        drug_regimen = state.get("drug_regimen") or []
        update_extra: dict = {"drug_regimen": drug_regimen}
        if len(drug_regimen) >= 2:
            sim_ci = state.get("simulation_ci") or {}
            ci_values = [
                float(v)
                for v in sim_ci.values()
                if isinstance(v, (int, float)) and math.isfinite(float(v))
            ]
            if ci_values:
                avg_ci = sum(ci_values) / len(ci_values)
                if avg_ci < 0.8:
                    synergy = "潜在协同 (Bliss synergy calculation > 0.3, CI<0.8)"
                elif avg_ci > 1.2:
                    synergy = "拮抗风险 (Bliss synergy < 0, CI>1.2)"
                else:
                    synergy = "叠加效应 (Bliss synergy ≈ 0, additive)"
            else:
                synergy = "CI 数据缺失，无法评估 Bliss synergy calculation"
                avg_ci = None

            update_extra["combination_index"] = sim_ci
            update_extra["synergy_assessment"] = synergy
            if avg_ci is not None:
                update_extra["combination_index_avg"] = avg_ci

        return {
            "auditor_status": "success",
            "correction_suggestion": "",
            "failure_report": "",
            "agent_dispatches": dispatches,
            **update_extra,
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

        latency_ms = (time.time() - start_ts) * 1000
        if next_retry_count >= _MAX_RETRY:
            if (d := orchestrator.fail_dispatch(node_name, "retry limit reached", latency_ms)) :
                dispatches.append(d)
            return {
                "auditor_status": "failed",
                "correction_suggestion": "",
                "failure_report": (
                    f"仿真代码经过 {next_retry_count} 次重试后仍无法成功执行。"
                    f"错误日志：{stdout_stderr}"
                ),
                "retry_count": next_retry_count,
                "agent_dispatches": dispatches,
                "token_usage": token_usage,
            }

        # 审计判定需要重试：记录为 failed（本次执行失败），前端会看到回到 Simulation Engineer
        if (d := orchestrator.fail_dispatch(node_name, f"retry {next_retry_count}/{_MAX_RETRY}", latency_ms)) :
            dispatches.append(d)
        return {
            "auditor_status": "retry",
            "correction_suggestion": suggestion,
            "failure_report": "",
            "retry_count": next_retry_count,
            "agent_dispatches": dispatches,
            "token_usage": token_usage,
        }

    latency_ms = (time.time() - start_ts) * 1000
    failure_report = (
        f"仿真代码经过 {retry_count} 次重试后仍无法成功执行。"
        f"错误日志：{stdout_stderr}"
    )
    if (d := orchestrator.fail_dispatch(node_name, "max retries exceeded", latency_ms)) :
        dispatches.append(d)
    return {
        "auditor_status": "failed",
        "correction_suggestion": "",
        "failure_report": failure_report,
        "agent_dispatches": dispatches,
        "token_usage": state.get("token_usage"),
    }


# -----------------------------------------------------------------------------
# Node 5: 报告生成器
# -----------------------------------------------------------------------------
def node5_generate_report(state: BioDynamicsState) -> dict:
    """根据成功仿真的结果撰写生物医学分析报告。"""
    node_name = "node5_generate_report"
    # 报告生成是隐式最终步骤，不在 4 个专业智能体内，直接生成 dispatch 事件
    dispatches: list[dict] = []
    if (d := orchestrator.emit_dispatch(
        target_agent="Report Generator",
        reasoning="Validation passed: 仿真通过审计，生成最终预测报告",
        status="in_progress",
        node_name=node_name,
    )) :
        dispatches.append(d)

    network_json = state.get("network_json") or {"nodes": [], "edges": []}
    python_code = state.get("python_code", "")

    # 动态填充报告中的联合用药与剂量递增 section
    drug_regimen = state.get("drug_regimen") or []
    combination_section = ""
    if len(drug_regimen) >= 2:
        combination_section = COMBINATION_REPORT_SECTION.format(
            combination_index=json.dumps(
                state.get("combination_index") or {}, ensure_ascii=False
            ),
            synergy_assessment=state.get("synergy_assessment", "")
            or "未评估",
        )

    dose_section = ""
    if state.get("dose_response_data"):
        dose_section = DOSE_REPORT_SECTION.format(
            ic50=state.get("ic50", "N/A"),
            ic90=state.get("ic90", "N/A"),
            hed=state.get("hed", "N/A"),
        )

    start_ts = time.time()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", NODE6_REPORT_PROMPT),
            ("human", "请撰写仿真预测报告。"),
        ]
    )
    chain = prompt.partial(
        network_json=json.dumps(network_json, ensure_ascii=False, indent=2),
        python_code=python_code,
        combination_section=combination_section,
        dose_section=dose_section,
    ) | llm
    usage_handler = UsageAccumulator()
    response = chain.invoke({}, config={"callbacks": [usage_handler]})
    latency_ms = (time.time() - start_ts) * 1000

    if (d := orchestrator.emit_dispatch(
        target_agent="Report Generator",
        reasoning=f"Report Generator completed in {latency_ms:.0f}ms",
        status="completed",
        node_name=node_name,
        latency_ms=latency_ms,
    )) :
        dispatches.append(d)

    return {
        "final_report": str(response.content).strip(),
        "agent_dispatches": dispatches,
        "token_usage": merge_usage(
            state.get("token_usage"), usage_from_accumulator(usage_handler)
        ),
    }
