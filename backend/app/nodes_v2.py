# BioDynamics Agent - v2 12 节点工作流节点实现
# 对应 biodynamics-v2-upgrade-plan.md 全部节点定义。
#
# 节点编号：
#   N1  NER / Entity Normalize            → entities
#   N2  Mechanistic Planner                → mechanism
#   N3  Mechanism RAG                      → mechanism.rag_evidence
#   N4  Knowledge Graph Builder (pure Py)  → knowledge_graph
#   N5  Parameter RAG                      → parameters
#   N6  ODE Generator (Template + Rule)    → ode_model + network_relations
#   N7  Sandbox Execute (AST pre-check)    → execution_result + error_class
#   N8  Scientific Feature Extraction (Py) → metrics + confidence
#   N9  Experiment RAG                     → experiment_protocols
#   N10 Evidence RAG                       → paper_evidence
#   N11 Scientific Report                  → report.markdown

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections import OrderedDict
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.config import llm, settings, strip_markdown_json
from app.bio_db_client import get_bio_db_client
from app.feature_extractor import ScientificFeatureExtractor
from app.kg_builder import KGBuilder
from app.mcp_client import get_mcp_client
from app.ode_templates import list_templates, render_template
from app.prompts import RAG_EXTRACTION_PROMPT
from app.prompts import RAG_DECISION_PROMPT
# P0-3 修复：接入 TemplateSelectorSkill 规则引擎 + Reaction IR + Domain Checker
from app.template_selector import (
    TemplateSelectorSkill,
    select_template as _select_template,
    get_simulation_time_scale,
    PHOS_CASCADE_TEMPLATES as _PHOS_TEMPLATES,
    CASCADE_TEMPLATES as _CASCADE_TEMPLATES,
)
from app.reaction_ir import build_reaction_graph, validate_reaction_graph, pre_validate_reaction_graph
from app.domain_checker import check_ode_code
from app.biomodels_client import get_biomodels_client, extract_biomodel_id
from app.prompts_v2 import (
    N1_NER_PROMPT,
    N2_PLANNER_PROMPT,
    N3_MECHANISM_RAG_PROMPT,
    N5_PARAMETER_DECISION_PROMPT,
    N6_ODE_PROMPT,
    N11_REPORT_FILL_PROMPT,
)
from app.rag_client import RagClient
from app.rag_collections import get_rag_collections
from app.report_renderer import ReportRenderer
from app.rule_engine import RuleEngine
from app.sandbox import execute_simulation_code_v2
from app.state import BioDynamicsState
from app.supervisor import orchestrator
from app.token_usage import (
    UsageAccumulator,
    merge_usage,
    usage_from_accumulator,
)

# TODO: P0-3 — 从 v1 引入 RAGDecisionOutput，统一 v1/v3 字段命名
from app.nodes import RAGDecisionOutput  # noqa: E402

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# 结构化输出 Pydantic 模型
# -----------------------------------------------------------------------------
class _Entity(BaseModel):
    entity_id: str
    name: str
    type: str
    aliases: list[str] = Field(default_factory=list)
    canonical_id: str = ""


class NEROutput(BaseModel):
    entities: list[_Entity] = Field(default_factory=list)


class _MechanismEdge(BaseModel):
    source: str
    target: str
    interaction: str


class _MechanismOutput(BaseModel):
    pathway: str
    cell: str
    simulation_type: str
    template: str
    required_outputs: list[str] = Field(default_factory=list)
    exemplars: list[str] = Field(default_factory=list)
    edges: list[_MechanismEdge] = Field(default_factory=list)


class _ParameterDecision(BaseModel):
    edge_key: str
    param_name: str
    value: float | None = None
    unit: str = ""
    source: str = ""
    confidence: str = "MEDIUM"
    is_fallback: bool = False


class _NetworkRelation(BaseModel):
    name: str
    role: str


class _EquationPattern(BaseModel):
    lhs: str
    rhs_pattern: str
    type: str


class NetworkRelationsOutput(BaseModel):
    variables: list[_NetworkRelation] = Field(default_factory=list)
    equations: list[_EquationPattern] = Field(default_factory=list)


class _ReportFillOutput(BaseModel):
    mechanism_analysis: str = ""
    simulation_interpretation: str = ""
    discussion: str = ""
    limitations: str = ""


# -----------------------------------------------------------------------------
# 工具函数：发送 agent_dispatch 事件
# -----------------------------------------------------------------------------
def _emit_in(node_name: str) -> None:
    """在节点入口发送 in_progress 调度事件。"""
    evt = orchestrator.dispatch_for_node_v2(node_name, status="in_progress")
    if evt:
        logger.debug("[v2 dispatch:in] %s", node_name)


def _emit_out(node_name: str, latency_ms: float = 0.0) -> None:
    """在节点出口发送 completed 调度事件。"""
    orchestrator.complete_dispatch_v2(node_name, latency_ms=latency_ms)


def _safe_json_parse(text: str) -> dict[str, Any]:
    """从 LLM 输出中提取 JSON 对象（容错 markdown 围栏）。

    BigModel (glm-5.1) 普通调用返回 markdown 包裹 JSON，原逐行剥离不如正则健壮；
    统一委托 config.strip_markdown_json 清洗后解析，回退正则提取最外层对象。

    TD-049 (IB-082) 修复：增强 JSON 解析鲁棒性，支持以下变体：
    - markdown 代码块（```json ... ``` / ``` ... ```）
    - 尾随逗号（trailing commas before } 或 ]）
    - 多种提取策略（直接解析 → 代码块提取 → 首尾大括号提取）
    - 记录所用策略便于调试
    """
    text = (text or "").strip()
    if not text:
        return {}

    # TD-049: 策略1 — 直接委托 strip_markdown_json 清洗后解析
    cleaned = strip_markdown_json(text)
    try:
        result = json.loads(cleaned)
        if isinstance(result, dict):
            logger.debug("TD-049 _safe_json_parse 策略1(direct) 成功")
            return result
    except Exception:
        pass

    # TD-049: 策略2 — 移除尾随逗号后重试（处理 LLM 常见的 {"a":1,} 格式）
    trailing_comma_removed = re.sub(r",\s*([}\]])", r"\1", cleaned)
    if trailing_comma_removed != cleaned:
        try:
            result = json.loads(trailing_comma_removed)
            if isinstance(result, dict):
                logger.debug("TD-049 _safe_json_parse 策略2(trailing_comma) 成功")
                return result
        except Exception:
            pass

    # TD-049: 策略3 — 显式提取 markdown 代码块内容（```json ... ``` 或 ``` ... ```）
    code_block_match = re.search(r"```(?:\w+)?\s*([\s\S]*?)\s*```", text)
    if code_block_match:
        block_content = code_block_match.group(1).strip()
        block_content = re.sub(r",\s*([}\]])", r"\1", block_content)
        try:
            result = json.loads(block_content)
            if isinstance(result, dict):
                logger.debug("TD-049 _safe_json_parse 策略3(code_block) 成功")
                return result
        except Exception:
            pass

    # TD-049: 策略4 — 正则提取最外层 { ... } 对象（含尾随逗号清理）
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        candidate = re.sub(r",\s*([}\]])", r"\1", match.group(0))
        try:
            result = json.loads(candidate)
            if isinstance(result, dict):
                logger.debug("TD-049 _safe_json_parse 策略4(regex_braces) 成功")
                return result
        except Exception:
            pass

    # TD-049: 所有策略均失败，记录警告并返回空字典（保持向后兼容）
    logger.warning(
        "TD-049 _safe_json_parse 所有策略均失败，原始文本(前200字): %s", text[:200]
    )
    return {}


def _safe_json_parse_list(text: str) -> list[dict[str, Any]]:
    """从 LLM 输出中提取 JSON 数组（容错 markdown 围栏）。

    RAG_EXTRACTION_PROMPT 要求输出参数数组，GLM 可能返回 ```json [...] ```；
    原 _safe_json_parse 仅提取 {...} 对象导致数组丢失，故新增此函数。
    """
    text = (text or "").strip()
    if not text:
        return []
    cleaned = strip_markdown_json(text)
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, list) else [parsed]
    except Exception:
        match = re.search(r"\[[\s\S]*\]", cleaned)
        if match:
            try:
                result = json.loads(match.group(0))
                return result if isinstance(result, list) else [result]
            except Exception:
                pass
    return []


# 高风险污染术语：当用户输入未提及时，过滤掉包含这些术语的 RAG 结果。
# 起因：用户查询 EGFR/Osimertinib，但 RAG 返回 TGF-β/CD8/SMAD 证据（语义相似但主题无关）。
_CONTAMINATION_TERMS: list[str] = [
    "tgf-beta", "tgf-β", "tgfb1", "tgf_beta",
    "cd8", "smad", "cd8+ t cell", "regulatory t cell",
]


def _filter_contaminated_evidence(
    evidence: list[dict], user_input: str
) -> list[dict]:
    """防御性过滤：若用户输入未提及污染术语，丢弃包含它们的结果。

    仅做负向过滤（移除明确无关的结果），不做正向匹配（不要求结果必须包含用户实体），
    避免过度过滤导致合法结果丢失。当用户输入确实提及污染术语时，跳过过滤。
    """
    if not evidence:
        return evidence
    user_lower = user_input.lower()
    user_mentions = any(term in user_lower for term in _CONTAMINATION_TERMS)
    if user_mentions:
        return evidence
    filtered: list[dict] = []
    for ev in evidence:
        ev_text = json.dumps(ev, ensure_ascii=False, default=str).lower()
        if any(term in ev_text for term in _CONTAMINATION_TERMS):
            logger.info("防御过滤：丢弃含污染术语的 RAG 结果（用户未提及）")
            continue
        filtered.append(ev)
    return filtered


# =============================================================================
# N0 — SBML Loader（用户输入含 BIOMD* / MODEL* 或通路关键词时加载 SBML）
# =============================================================================
def n0_sbml_loader(state: BioDynamicsState) -> dict:
    """N0 SBML Loader：从用户输入识别 BIOMD*/MODEL* ID 或通路关键词，
    通过 EBI BioModels REST API 加载对应 SBML 文本。

    对应 EGF-EGFR错误结论根因与后续修复计划报告.md §5.1.4 与 §5.4.1：
    通过 BioModels API 按需下载 SBML，不依赖本地文件作为默认数据源。

    Returns:
        {sbml_model_id, sbml_model_text} 或空字典（无 SBML 需求时）。
    """
    _emit_in("n0_sbml_loader")
    user_input = state.get("user_input", "")
    if not user_input:
        return {}

    # 已存在 sbml_model_text 时跳过（避免重复下载）
    existing_text = state.get("sbml_model_text", "")
    existing_id = state.get("sbml_model_id", "")
    if existing_text and existing_id:
        logger.info("N0 SBML Loader 跳过：已有 sbml_model_id=%s", existing_id)
        return {}

    # 提取 BIOMD*/MODEL* ID
    model_id = extract_biomodel_id(user_input)
    if not model_id:
        # 用户未显式提到 BIOMD*，但可能提到通路关键词
        # 仅当用户明确引用模型 ID 时才下载，避免无谓的网络请求
        logger.info("N0 SBML Loader：用户输入未含 BIOMD*/MODEL* ID，跳过")
        return {}

    logger.info("N0 SBML Loader：检测到 model_id=%s，开始下载 SBML", model_id)
    client = get_biomodels_client()
    try:
        sbml_text = client.download(model_id)
    except Exception as exc:
        logger.warning("N0 SBML Loader 下载失败 (model_id=%s): %s", model_id, exc)
        return {"sbml_model_id": model_id}

    if not sbml_text:
        logger.warning("N0 SBML Loader：SBML 文本为空，仅记录 model_id")
        return {"sbml_model_id": model_id}

    logger.info(
        "N0 SBML Loader 成功：model_id=%s, sbml_text size=%d bytes",
        model_id, len(sbml_text),
    )
    return {
        "sbml_model_id": model_id,
        "sbml_model_text": sbml_text,
    }


# =============================================================================
# N1 — NER / Entity Normalize
# =============================================================================
# TD-034 (IB-021) 修复：HGNC ID 格式校验器（离线，不调用 HGNC API）。
# HGNC ID 标准格式为 "HGNC:<数字>"（如 HGNC:3236），不符合则标记 ontology.verified=False。
_HGNC_ID_RE = re.compile(r"^HGNC:\d+$")


def _validate_gene_hgnc_ids(entities: list[dict]) -> list[dict]:
    """TD-034: 对基因/蛋白实体的 HGNC ID 进行格式校验（离线安全）。

    遍历实体列表，若实体含 hgnc_id 字段或 canonical_id 以 "HGNC:" 开头，
    则校验其格式是否匹配 "HGNC:\\d+"。不匹配时设置 ontology.verified=False 并记录警告。
    不调用任何外部 API，保持离线安全。
    """
    for ent in entities:
        if not isinstance(ent, dict):
            continue
        # 提取 HGNC ID：优先 hgnc_id 字段，其次 canonical_id 以 "HGNC:" 开头
        hgnc_id = ent.get("hgnc_id", "") or ""
        canonical_id = ent.get("canonical_id", "") or ""
        if not hgnc_id and canonical_id.upper().startswith("HGNC:"):
            hgnc_id = canonical_id
        if not hgnc_id:
            # 无 HGNC ID 的实体跳过校验（不设置 ontology 字段，保持向后兼容）
            continue
        # 格式校验
        if _HGNC_ID_RE.match(hgnc_id):
            ent.setdefault("ontology", {})["verified"] = True
        else:
            # 格式不匹配，标记未验证并记录警告
            ent.setdefault("ontology", {})["verified"] = False
            logger.warning(
                "TD-034 HGNC ID 格式校验失败（entity=%s, hgnc_id=%s），已标记 ontology.verified=False",
                ent.get("name", ent.get("entity_id", "?")), hgnc_id,
            )
    return entities


def n1_ner_entity_normalize(state: BioDynamicsState) -> dict:
    """从用户输入中提取生物实体。"""
    _emit_in("n1_ner_entity_normalize")
    user_input = state.get("user_input", "")

    try:
        # IB-029 TODO: with_structured_output
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", N1_NER_PROMPT),
                ("human", "用户输入：{user_input}"),
            ]
        )
        chain = prompt.partial(user_input=user_input) | llm
        response = chain.invoke({})
        raw_content = str(response.content)
        try:
            parsed = _safe_json_parse(raw_content)
            # IB-029: 检测 JSON 解析失败（响应非空但结果为空）
            if not parsed and raw_content.strip():
                logger.warning("N1 LLM 响应 JSON 解析失败，原始(前200字): %s", raw_content[:200])
                parsed = {"entities": [], "_parse_error": "json_parse_failed"}
            entities = parsed.get("entities", [])
        except Exception as parse_exc:
            # IB-029: 解析异常时返回结构化错误响应（不崩溃）
            logger.warning("N1 LLM 响应 JSON 解析异常：%s", parse_exc)
            entities = []
    except Exception as exc:
        logger.warning("N1 NER 失败，使用空列表降级：%s", exc)
        entities = []

    # TD-034 (IB-021): 对基因/蛋白实体的 HGNC ID 进行格式校验
    entities = _validate_gene_hgnc_ids(entities)

    return {
        "entities": entities,
        "agent_dispatches": [orchestrator.complete_dispatch(
            "n1_ner_entity_normalize", latency_ms=0.0
        )],
    }


# =============================================================================
# N2 — Mechanistic Planner
# =============================================================================
def n2_mechanistic_planner(state: BioDynamicsState) -> dict:
    """基于用户输入与实体，输出仿真方案 JSON。"""
    _emit_in("n2_mechanistic_planner")
    user_input = state.get("user_input", "")
    entities = state.get("entities", [])

    try:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", N2_PLANNER_PROMPT),
                (
                    "human",
                    "用户输入：{user_input}\n已识别实体：{entities}",
                ),
            ]
        )
        chain = prompt.partial(
            user_input=user_input,
            entities=json.dumps(entities, ensure_ascii=False),
        ) | llm
        # IB-029 TODO: with_structured_output
        response = chain.invoke({})
        raw_content = str(response.content)
        try:
            mechanism = _safe_json_parse(raw_content)
            # IB-029: 检测 JSON 解析失败（响应非空但结果为空），记录错误以便调试
            if not mechanism and raw_content.strip():
                logger.warning("N2 LLM 响应 JSON 解析失败，原始(前200字): %s", raw_content[:200])
                mechanism = {}
        except Exception as parse_exc:
            # IB-029: 解析异常时记录错误并返回结构化错误响应（不崩溃）
            logger.warning("N2 LLM JSON 解析异常：%s", parse_exc)
            mechanism = {}
    except Exception as exc:
        logger.warning("N2 Planner 失败，使用默认 simple_inhibition 降级：%s", exc)
        mechanism = {
            "pathway": "Unknown",
            "cell": "Unknown",
            "simulation_type": "simple_inhibition",
            "template": "Simple_Inhibition",
            "required_outputs": ["simulation.csv", "simulation.png", "BIO_CHECK"],
            "exemplars": [],
            "edges": [],
        }

    # 验证 template 是否在已注册模板列表内
    available = list_templates()
    if mechanism.get("template") not in available:
        logger.info("N2 模板 %s 不可用，回退到 Simple_Inhibition", mechanism.get("template"))
        mechanism["template"] = "Simple_Inhibition"
        mechanism["simulation_type"] = "simple_inhibition"

    return {
        "mechanism": mechanism,
        "agent_dispatches": [orchestrator.complete_dispatch(
            "n2_mechanistic_planner", latency_ms=0.0
        )],
    }


# =============================================================================
# N3 — Mechanism RAG
# =============================================================================
def n3_mechanism_rag(state: BioDynamicsState) -> dict:
    """用机制路径查询 RagCollections，补充 RAG 证据。"""
    _emit_in("n3_mechanism_rag")
    mechanism = dict(state.get("mechanism", {}))
    pathway = mechanism.get("pathway", "")

    rag_evidence: list[dict] = []
    rag = get_rag_collections()
    if rag.available and pathway:
        rag_evidence = rag.search_mechanism(pathway, top_k=5)
        # 仅保留 _document/_semantic_score 之外的字段
        rag_evidence = [
            {k: v for k, v in r.items() if not k.startswith("_")}
            for r in rag_evidence
        ]
        # 防御性过滤：移除与用户输入无关的污染术语结果
        rag_evidence = _filter_contaminated_evidence(
            rag_evidence, state.get("user_input", "")
        )

    # 让 LLM 把检索到的 evidence 总结为一句中文描述
    description = mechanism.get("description", "")
    if rag_evidence and not description:
        try:
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", N3_MECHANISM_RAG_PROMPT),
                    (
                        "human",
                        "用户假说：{scenario}\nRAG 命中：{chunks}",
                    ),
                ]
            )
            chain = prompt.partial(
                scenario=state.get("user_input", ""),
                chunks=json.dumps(rag_evidence[:3], ensure_ascii=False),
            ) | llm
            # IB-029 TODO: with_structured_output
            response = chain.invoke({})
            raw_content = str(response.content)
            try:
                parsed = _safe_json_parse(raw_content)
                # IB-029: 检测 JSON 解析失败（响应非空但结果为空）
                if not parsed and raw_content.strip():
                    logger.warning("N3 LLM 响应 JSON 解析失败，原始(前200字): %s", raw_content[:200])
                    parsed = {}
                description = parsed.get("mechanism_analysis", "") or parsed.get("description", "")
            except Exception as parse_exc:
                # IB-029: 解析异常时记录错误并返回空描述（不崩溃）
                logger.warning("N3 LLM JSON 解析异常：%s", parse_exc)
                description = ""
        except Exception as exc:
            logger.warning("N3 机制总结失败：%s", exc)
            description = ""

    mechanism["description"] = description
    mechanism["rag_evidence"] = rag_evidence

    return {
        "mechanism": mechanism,
        "agent_dispatches": [orchestrator.complete_dispatch(
            "n3_mechanism_rag", latency_ms=0.0
        )],
    }


# =============================================================================
# N4 — Knowledge Graph Builder（pure Python, 零 LLM）
# =============================================================================
def n4_kg_builder(state: BioDynamicsState) -> dict:
    """由 entities + relations 构建知识图谱（pure Python）。"""
    _emit_in("n4_kg_builder")
    entities = state.get("entities", [])
    # v1/v2 兼容：优先用 network_relations 边，回退到 network_json.edges
    relations = state.get("network_relations", {}).get("edges") or []
    if not relations:
        edges = state.get("network_json", {}).get("edges", []) or []
        # Step 2.2: 保留 mechanism / reaction_equation 字段（供 Signaling_Cascade_Phos 模板使用）
        relations = [
            {
                "source": e.get("source"),
                "target": e.get("target"),
                "interaction": e.get("interaction"),
                "mechanism": e.get("mechanism", ""),
                "reaction_equation": e.get("reaction_equation", ""),
            }
            for e in edges
        ]

    builder = KGBuilder()
    try:
        kg = builder.build(entities=entities, relations=relations)
    except Exception as exc:
        logger.warning("N4 KG 构建失败：%s", exc)
        kg = {
            "nodes": [],
            "edges": [],
            "adjacency": {},
            "topology_signature": "empty",
            "is_acyclic": True,
            "node_count": 0,
            "edge_count": 0,
            "dropped_edges": [],
        }

    return {
        "knowledge_graph": kg,
        "agent_dispatches": [orchestrator.complete_dispatch(
            "n4_kg_builder", latency_ms=0.0
        )],
    }


# =============================================================================
# N5 — Parameter RAG（按边查询，程序注入）
# =============================================================================

# PubMed 调用频率控制：相邻调用间至少间隔 3 秒，避免 NCBI 限流。
_pubmed_last_call_ts: float = 0.0
_PUBMED_RATE_LIMIT_SECONDS: float = 3.0
# PubMed 调用超时上限（含 esearch + efetch + LLM 提取）
_PUBMED_TIMEOUT_SECONDS: float = 30.0
# TD-044 (IB-073) 修复：将无界 dict 替换为带 TTL + LRU 的 OrderedDict 缓存。
# 最大条目数 1000，TTL 24 小时（86400 秒），避免内存无限增长。
_PUBMED_CACHE_MAX_SIZE: int = 1000
_PUBMED_CACHE_TTL_SECONDS: float = 86400.0
# 缓存结构：key → (value, timestamp)，使用 OrderedDict 实现 LRU 淘汰。
_pubmed_cache: "OrderedDict[str, tuple[list[dict], float]]" = OrderedDict()


def _get_from_pubmed_cache(key: str) -> list[dict] | None:
    """TD-044: 从 PubMed 缓存中读取值，含 TTL 过期检查与 LRU 访问顺序更新。

    若 key 不存在或已过期返回 None；命中时将 key 移至末尾（最近使用）。
    """
    if key not in _pubmed_cache:
        return None
    value, ts = _pubmed_cache[key]
    # TTL 过期检查
    if time.time() - ts > _PUBMED_CACHE_TTL_SECONDS:
        # 过期则删除并返回 None
        _pubmed_cache.pop(key, None)
        logger.debug("TD-044 PubMed 缓存过期删除：%s", key[:50])
        return None
    # LRU：移至末尾标记为最近使用
    _pubmed_cache.move_to_end(key)
    return value


def _set_in_pubmed_cache(key: str, value: list[dict]) -> None:
    """TD-044: 写入 PubMed 缓存，含 LRU 淘汰（超过 max_size 时淘汰最旧条目）。"""
    # 若 key 已存在则先删除（保证 move_to_end 语义正确）
    if key in _pubmed_cache:
        _pubmed_cache.pop(key, None)
    # LRU 淘汰：超过最大容量时删除最旧（头部）条目
    while len(_pubmed_cache) >= _PUBMED_CACHE_MAX_SIZE:
        _pubmed_cache.popitem(last=False)
    # 写入新条目（自动在末尾）
    _pubmed_cache[key] = (value, time.time())


async def _fetch_params_from_pubmed(
    query: str, species_context: str = "Human"
) -> list[dict]:
    """PubMed E-utilities 兜底：检索文献并用 LLM 提取动力学参数。

    当 ChromaDB 向量库无命中时，直连 NCBI E-utilities 检索 PubMed 文献，
    再用 RAG_EXTRACTION_PROMPT 从摘要中提取 Kd/Km/IC50 等参数。
    含速率限制（3 秒/次）、超时控制（30 秒）、内存缓存。
    离线或失败时返回空列表，不阻塞主流程。
    """
    # 缓存命中（TD-044: 使用带 TTL/LRU 的缓存读取函数）
    cache_key = f"{query}|{species_context}"
    cached = _get_from_pubmed_cache(cache_key)
    if cached is not None:
        logger.info("PubMed 缓存命中：%s", query[:50])
        return cached

    # 速率限制
    global _pubmed_last_call_ts
    elapsed = time.time() - _pubmed_last_call_ts
    if elapsed < _PUBMED_RATE_LIMIT_SECONDS:
        await asyncio.sleep(_PUBMED_RATE_LIMIT_SECONDS - elapsed)

    async def _do_fetch() -> list[dict]:
        mcp = get_mcp_client()
        articles, _, _ = await mcp.search_pubmed(query, max_results=3)
        if not articles:
            return []

        # 用 RAG_EXTRACTION_PROMPT 从摘要中提取参数
        extracted_params: list[dict] = []
        for article in articles:
            abstract = article.get("abstract", "")
            if not abstract or len(abstract) < 50:
                continue
            try:
                prompt = ChatPromptTemplate.from_messages([
                    ("system", RAG_EXTRACTION_PROMPT),
                    ("human", "{document_chunk}"),
                ])
                chain = prompt | llm
                # IB-029 TODO: with_structured_output
                response = chain.invoke({"document_chunk": abstract})
                raw_content = str(response.content)
                try:
                    params = _safe_json_parse_list(raw_content)
                    # IB-029: 检测 JSON 解析失败（响应非空但结果为空列表）
                    if not params and raw_content.strip():
                        logger.warning("PubMed LLM 响应 JSON 解析失败，原始(前200字): %s", raw_content[:200])
                        params = []
                except Exception as parse_exc:
                    # IB-029: 解析异常时记录错误并返回空列表（不崩溃）
                    logger.warning("PubMed LLM JSON 解析异常：%s", parse_exc)
                    params = []
                for p in params:
                    if isinstance(p, dict) and "value" in p:
                        p["source"] = article.get("source", f"PMID:{article.get('pmid', '')}")
                        p["species"] = p.get("species", species_context)
                        extracted_params.append(p)
            except Exception as exc:
                logger.warning("PubMed 参数提取失败：%s", exc)

        return extracted_params

    try:
        result = await asyncio.wait_for(_do_fetch(), timeout=_PUBMED_TIMEOUT_SECONDS)
        # TD-044: 使用带 TTL/LRU 的缓存写入函数
        _set_in_pubmed_cache(cache_key, result)
        _pubmed_last_call_ts = time.time()
        return result
    except asyncio.TimeoutError:
        logger.warning("PubMed 检索超时（%ss）： %s", _PUBMED_TIMEOUT_SECONDS, query[:50])
        _pubmed_last_call_ts = time.time()
        return []
    except Exception as exc:
        logger.warning("PubMed 兜底检索失败：%s", exc)
        _pubmed_last_call_ts = time.time()
        return []

async def n5_parameter_rag(state: BioDynamicsState) -> dict:
    """为 KG 中每条边查询最佳动力学参数（程序注入，LLM 禁止修改）。

    TODO: P0-1 — 重写为基于 RagClient.search_params_hybrid 的高阶检索：
    查询重写 → 混合检索 → 重排序 → RAGDecisionOutput 结构化决策。
    同时聚合 rag_insights / rag_hit_rate / drug_candidates 字段，
    与 v1 node1_5_rag_search 字段对齐，避免下游 PK/PD 节点拿不到药物候选。

    检索顺序：ChromaDB hybrid 检索 → PubMed E-utilities 兜底 → 估算。

    深度审核报告 §1.1 参数溯源强制性：
    所有参数对象必须包含溯源四元组 {value, source, confidence, origin}：
    - value: float | None（None 表示 missing_parameter）
    - source: "RAG" | "SBML" | "PubMed" | "KEGG" | "UniProt" | "ChEMBL" | "Inferred"
    - confidence: float in [0.0, 1.0]（数值化置信度）
    - origin: str（具体来源标识，如 PMID:12345 / BIOMD0000000205）
    缺失策略：confidence < 0.3 时标记 missing_parameter=True 并触发在线回退。
    """
    _emit_in("n5_parameter_rag")
    start_ts = time.time()
    kg = state.get("knowledge_graph", {}) or {}
    edges = kg.get("edges", []) or []
    species_context = state.get("species_context", "Human")
    user_input = state.get("user_input", "")

    # TODO: P0-1 — 切换到 RagClient（高阶 hybrid + rerank + insights）
    rag_client = RagClient()
    usage_handler = UsageAccumulator()
    mcp_client = get_mcp_client()

    # 构建 MCP 术语到标准名的映射，增强 RAG 查询的术语精准度
    mcp_term_map: dict[str, str] = {}
    for d in state.get("mcp_term_definitions") or []:
        term = d.get("term", "")
        canonical = d.get("canonical_name", "")
        if term and canonical and term.lower() != canonical.lower():
            mcp_term_map[term] = canonical

    parameters: dict[str, dict] = {}
    rag_selected_params: dict[str, dict] = {}  # v1 兼容字段，供 node1_6_pkpd_inference 读取
    rag_fallback = False
    # 深度审核报告 §1.1：缺失参数清单（confidence < 0.3 或 source=Inferred）
    missing_parameters: list[str] = []

    # 聚合 RAG 洞察数据
    aggregated_rewrites: list[dict] = []
    aggregated_source_dist: dict[str, int] = {}
    aggregated_top_selections: list[dict] = []
    aggregated_rewritten_queries: list[str] = []
    total_candidates = 0
    # 聚合药物候选（知识图谱，来自 inhibition 边的 drug_specific_retriever）
    all_drug_candidates: list[dict] = []

    # 动力学参数白名单：仅这些 param_name 可被采纳为 Kd/k1/k2 等动力学常数
    # initial_concentration_* 是物种初始浓度，禁止作为动力学参数
    _KINETIC_PARAM_PREFIXES = (
        "k1", "k2", "k_1", "k_2", "k_on", "k_off", "kcat", "Kd", "Ki", "Km",
        "Vmax", "V1", "V2", "V3", "hill", "n", "EC50", "IC50", "KEC50",
        "k_deg", "kdegr", "k_prod", "k_syn", "k_sec", "k_act", "k_inact",
    )

    # 深度审核报告 §1.1：溯源辅助函数
    def _extract_origin(candidate: dict | None) -> str:
        """从 RAG 候选对象提取具体来源标识（PMID/BIOMD ID/数据库 ID）。"""
        if not candidate:
            return "unknown"
        # 优先级：pmid > source_model > source > source_pmid > unknown
        pmid = candidate.get("pmid") or candidate.get("source_pmid")
        if pmid:
            return f"PMID:{pmid}"
        source_model = candidate.get("source_model")
        if source_model:
            return str(source_model)
        source = candidate.get("source", "")
        if source and source not in ("RAG", "ESTIMATED", "Inferred"):
            return str(source)
        return "unknown"

    def _confidence_str_to_float(conf_str: str) -> float:
        """将 HIGH/MEDIUM/LOW 字符串置信度转换为数值（深度审核报告 §1.1）。"""
        mapping = {"HIGH": 0.9, "MEDIUM": 0.6, "LOW": 0.2}
        return mapping.get(str(conf_str).upper(), 0.3)

    def _detect_source_type(candidate: dict | None, default: str = "RAG") -> str:
        """检测参数来源类型（RAG/SBML/PubMed/KEGG/UniProt/ChEMBL/Inferred）。"""
        if not candidate:
            return default
        retrieval_method = candidate.get("_retrieval_method", "")
        if retrieval_method == "online_fallback":
            # 在线回退来源：根据 source 字段进一步细分
            source = str(candidate.get("source", "")).lower()
            if "kegg" in source:
                return "KEGG"
            if "uniprot" in source:
                return "UniProt"
            if "chembl" in source:
                return "ChEMBL"
            if "pubmed" in source or "pmid" in source:
                return "PubMed"
            return "Inferred"
        if retrieval_method == "pubmed":
            return "PubMed"
        source_model = candidate.get("source_model", "")
        if source_model and "BIOMD" in str(source_model):
            return "SBML"
        return default

    def _top_candidate_usable(cands: list[dict]) -> dict | None:
        """当 LLM 决策缺失时，判断 top candidate 是否可直接采纳。

        关键约束：禁止采纳 initial_concentration_* 类型参数作为动力学常数。
        原因：initial_concentration 是物种的初始浓度（如 EGFR=0.3 nM），
        不是反应速率常数（k1/k2/Kd）。若错误采纳，会导致 ODE 参数完全错误。
        """
        if not cands:
            return None
        top = cands[0]
        # 拒绝 initial_concentration_* 类型参数
        param_name = str(top.get("param_name", "")).strip()
        param_name_lower = param_name.lower()
        if param_name_lower.startswith("initial_concentration"):
            logger.warning(
                "N5 拒绝 initial_concentration 作为动力学参数: %s", param_name
            )
            return None
        # 拒绝 source_type=species 的记录（同上防御）
        if str(top.get("source_type", "")).lower() == "species":
            logger.warning(
                "N5 拒绝 source_type=species 的记录作为动力学参数: %s", param_name
            )
            return None
        val = top.get("value")
        if val is None:
            return None
        try:
            float(val)
        except (TypeError, ValueError):
            return None
        # 拒绝 value=0 的参数作为 Kd（会导致 Hill 函数饱和）
        try:
            if float(val) == 0.0:
                logger.warning(
                    "N5 拒绝 value=0 的参数作为动力学参数（会导致饱和）: %s", param_name
                )
                return None
        except (TypeError, ValueError):
            return None
        score = top.get("_rerank_score", top.get("_combined_score", 0))
        # 有明确重排分数且达到阈值，或来源无分数但包含数值（如 PubMed 兜底）
        if score >= settings.RAG_ONLINE_FALLBACK_THRESHOLD:
            return top
        if score == 0 and "_rerank_score" not in top and "_combined_score" not in top:
            return top
        return None

    # ChromaDB 不可用时所有边直接走估算，避免无谓 LLM 调用
    if not rag_client.available:
        logger.warning("N5: ChromaDB 不可用，所有边将走 PubMed 兜底或估算")

    # 结构化决策链：统一用 RAGDecisionOutput，避免裸文本解析（BigModel 兼容）
    structured_llm = llm.with_structured_output(RAGDecisionOutput)
    decision_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RAG_DECISION_PROMPT),
            ("human", "请根据检索结果做出参数决策。"),
        ]
    )

    for edge_idx, edge in enumerate(edges):
        # 深度审核报告 §4.2 在线回退熔断：workflow 总时长预算控制（默认 600s = 10 分钟）
        # 超出预算时强制跳出循环，剩余边走估算降级，避免阻塞 Workflow 超过 10 分钟
        elapsed_total = time.time() - start_ts
        if elapsed_total > settings.RAG_ONLINE_TOTAL_BUDGET:
            remaining_count = len(edges) - edge_idx
            logger.warning(
                "N5 在线回退总预算耗尽（%.1fs > %.1fs），剩余 %d 条边走估算降级",
                elapsed_total, settings.RAG_ONLINE_TOTAL_BUDGET, remaining_count,
            )
            for remaining_edge in edges[edge_idx:]:
                r_src = remaining_edge.get("source", "")
                r_tgt = remaining_edge.get("target", "")
                r_key = f"{r_src}->{r_tgt}"
                if r_key not in parameters:
                    parameters[r_key] = {
                        "edge_key": r_key,
                        "param_name": "kd",
                        "value": None,
                        "unit": "nM",
                        "source": "Inferred",
                        "confidence": 0.2,
                        "confidence_label": "LOW",
                        "origin": "budget_exceeded",
                        "is_fallback": True,
                        "missing_parameter": True,
                    }
                    missing_parameters.append(f"{r_key}:kd")
            break

        source_name = edge.get("source", "")
        target_name = edge.get("target", "")
        interaction = edge.get("interaction", "")
        # Step 2.2: 读取 mechanism 字段（binding/phosphorylation/exchange/...）
        # 用于精确检索对应类型的动力学常数（k_on/k_off vs k_phos/k_dephos vs k_exchange）
        mechanism = edge.get("mechanism", "activation")
        reaction_equation = edge.get("reaction_equation", "")
        edge_key = f"{source_name}->{target_name}"

        # Step 2.2: 根据 mechanism 构建精确的 RAG 查询
        # binding → k_on/k_off (mass-action association/dissociation)
        # phosphorylation → k_phos/k_dephos (kinase catalytic rate)
        # exchange → k_exchange (GDP→GTP exchange)
        # degradation → k_deg
        mechanism_param_terms = {
            "binding":           "k_on k_off association dissociation mass-action",
            "phosphorylation":   "k_phos k_dephos phosphorylation kinase catalytic",
            "dephosphorylation": "k_dephos dephosphorylation phosphatase",
            "exchange":          "k_exchange GDP GTP nucleotide exchange SOS",
            "recruitment":       "k_on k_off recruitment binding adapter",
            "dissociation":      "k_off dissociation",
            "degradation":       "k_deg degradation half-life",
            "activation":        "k_cat k1 k2 rate constant activation",
            "inhibition":        "ki k_on k_off inhibition binding",
        }
        mech_terms = mechanism_param_terms.get(mechanism, "k1 k2 Kd rate constant")

        # 构建 RAG 查询：source + mechanism + target + reaction_equation + 参数类型关键词
        query = (
            f"{source_name} {mechanism} {target_name} "
            f"reaction: {reaction_equation} "
            f"kinetic parameter {mech_terms} "
            f"species {species_context}"
        )
        if mcp_term_map:
            canonical_terms = [
                mcp_term_map[orig] for orig in [source_name, target_name]
                if orig in mcp_term_map
            ]
            if canonical_terms:
                query += " " + " ".join(canonical_terms)

        candidates: list[dict] = []
        edge_insights: dict = {}

        # 1. ChromaDB 高阶 hybrid 检索（查询重写 + BM25 + 语义 + rerank）
        if rag_client.available:
            try:
                reranked, edge_insights = rag_client.search_params_hybrid(
                    query, species_context=species_context, top_k=5
                )
                # 保留 _rerank_score 供排序，剥离其他内部字段后做污染过滤
                candidates = _filter_contaminated_evidence(reranked, user_input)
                # 过滤掉 initial_concentration_* 类型参数（不是动力学常数）
                # 原因：initial_concentration 是物种初始浓度（如 EGFR=0.3 nM），
                # 不是反应速率常数 k1/k2/Kd，错误采纳会导致 ODE 参数完全错误
                _before_filter = len(candidates)
                candidates = [
                    c for c in candidates
                    if not str(c.get("param_name", "")).lower().startswith("initial_concentration")
                    and str(c.get("source_type", "")).lower() != "species"
                ]
                if _before_filter != len(candidates):
                    logger.info(
                        "N5 边 %s：过滤掉 %d 条 initial_concentration 候选，剩余 %d 条",
                        edge_key, _before_filter - len(candidates), len(candidates),
                    )
                # 聚合洞察数据
                if edge_insights.get("rewritten_query"):
                    aggregated_rewritten_queries.append(edge_insights["rewritten_query"])
                aggregated_rewrites.extend(edge_insights.get("rewrites", []))
                for src, cnt in edge_insights.get("source_distribution", {}).items():
                    aggregated_source_dist[src] = aggregated_source_dist.get(src, 0) + cnt
                total_candidates += edge_insights.get("total_candidates", 0)
                for sel in edge_insights.get("top_selections", []):
                    sel["edge"] = f"{source_name} → {target_name}"
                    aggregated_top_selections.append(sel)
            except Exception as exc:
                logger.warning("N5 search_params_hybrid 失败（边 %s）：%s", edge_key, exc)

        # 1.5. 在线数据库补充：本地 ChromaDB 命中不足时，查询 KEGG/Reactome/UniProt/ChEMBL
        # 深度审核报告 §4.2 熔断：单次查询超时限制为 10s（settings.RAG_ONLINE_QUERY_TIMEOUT）
        if settings.RAG_ONLINE_FALLBACK:
            best_score = max(
                (r.get("_rerank_score", r.get("_combined_score", 0)) for r in candidates),
                default=0,
            )
            if best_score < settings.RAG_ONLINE_FALLBACK_THRESHOLD or not candidates:
                try:
                    bio_db_client = get_bio_db_client()
                    # 单次查询 10s 熔断，避免单个在线源阻塞 Workflow
                    online_results = await asyncio.wait_for(
                        bio_db_client.search_all(query, species_context),
                        timeout=settings.RAG_ONLINE_QUERY_TIMEOUT,
                    )
                    if online_results:
                        online_results = _filter_contaminated_evidence(online_results, user_input)
                        for r in online_results:
                            r["_retrieval_method"] = "online_fallback"
                        candidates.extend(online_results)
                        logger.info(
                            "N5 在线补充 %s→%s：获取 %d 条结果",
                            source_name, target_name, len(online_results),
                        )
                except asyncio.TimeoutError:
                    logger.warning(
                        "N5 在线补充超时（边 %s，%.1fs 熔断）", edge_key, settings.RAG_ONLINE_QUERY_TIMEOUT,
                    )
                except Exception as exc:
                    logger.warning("N5 在线数据库补充失败（边 %s）：%s", edge_key, exc)

        # 2. PubMed E-utilities 兜底：ChromaDB 无命中时直连 NCBI 检索文献并提取参数
        # 深度审核报告 §4.2 熔断：PubMed 兜底同样受 10s 超时保护
        if not candidates:
            try:
                pubmed_params = await asyncio.wait_for(
                    _fetch_params_from_pubmed(query, species_context),
                    timeout=settings.RAG_ONLINE_QUERY_TIMEOUT,
                )
                if pubmed_params:
                    pubmed_params = _filter_contaminated_evidence(pubmed_params, user_input)
                    candidates = pubmed_params
                    logger.info("N5 PubMed 兜底命中 %d 条参数（边 %s）", len(candidates), edge_key)
            except asyncio.TimeoutError:
                logger.warning(
                    "N5 PubMed 兜底超时（边 %s，%.1fs 熔断）", edge_key, settings.RAG_ONLINE_QUERY_TIMEOUT,
                )
            except Exception as exc:
                logger.warning("N5 PubMed 兜底失败（边 %s）：%s", edge_key, exc)

        # 3. inhibition 边额外检索靶点相关药物候选（知识图谱注入）
        # 深度审核报告 §4.2 熔断：药物候选检索受 10s 超时保护
        if interaction == "inhibition":
            try:
                pubmed_query = f"{target_name} inhibitor IC50 clinical trial"
                articles, _, _ = await asyncio.wait_for(
                    mcp_client.search_pubmed(pubmed_query, max_results=3),
                    timeout=settings.RAG_ONLINE_QUERY_TIMEOUT,
                )
                drug_cands = rag_client.drug_specific_retriever(
                    target_name=target_name,
                    species_context=species_context,
                    pubmed_articles=articles,
                )
                all_drug_candidates.extend(drug_cands)
            except asyncio.TimeoutError:
                logger.warning(
                    "N5 药物候选检索超时（靶点 %s，%.1fs 熔断）", target_name, settings.RAG_ONLINE_QUERY_TIMEOUT,
                )
            except Exception as exc:
                logger.warning("N5 检索 %s 的药物候选失败：%s", target_name, exc)

        # 4. LLM 参数决策（统一用 RAGDecisionOutput 结构化输出）
        if candidates:
            try:
                chain = decision_prompt.partial(
                    source_node=source_name,
                    target_node=target_name,
                    interaction_type=interaction,
                    species_context=species_context,
                    retrieved_params_json=json.dumps(
                        candidates[:5], ensure_ascii=False, indent=2
                    ),
                ) | structured_llm
                decision: RAGDecisionOutput = await chain.ainvoke(
                    {}, config={"callbacks": [usage_handler]}
                )
                # 写入 v1 兼容字段（供 node1_6_pkpd_inference 读取）
                rag_selected_params[edge_key] = decision.model_dump()
                # 转换为 parameters[edge_key] 格式（供 n6_ode_generator 读取）
                if decision.param_found and decision.selected_params:
                    sp = decision.selected_params[0]
                    # 深度审核报告 §1.1：溯源四元组 {value, source, confidence, origin}
                    conf_str = "HIGH" if not decision.fallback_to_estimation else "MEDIUM"
                    conf_float = _confidence_str_to_float(conf_str)
                    # 提取 origin：优先从 sp.source，其次从 top candidate
                    top_for_origin = candidates[0] if candidates else None
                    origin = sp.source if sp.source and sp.source not in ("RAG", "ESTIMATED") else _extract_origin(top_for_origin)
                    source_type = _detect_source_type(top_for_origin, default="RAG")
                    parameters[edge_key] = {
                        "edge_key": edge_key,
                        "param_name": sp.param_name,
                        "value": float(sp.value),
                        "unit": sp.unit,
                        "source": source_type,
                        "confidence": conf_float,
                        "confidence_label": conf_str,  # 兼容字段
                        "origin": origin,
                        "is_fallback": False,
                        "missing_parameter": conf_float < settings.RAG_ONLINE_FALLBACK_THRESHOLD,
                    }
                    if conf_float < settings.RAG_ONLINE_FALLBACK_THRESHOLD:
                        missing_parameters.append(f"{edge_key}:{sp.param_name}")
                else:
                    # TODO: P2-7 — LLM 判定缺失但候选充足时，强制采纳 top candidate，
                    # 避免检索到 44 篇候选却 rag_hit_rate=0% 的极端情况。
                    top = _top_candidate_usable(candidates)
                    if top:
                        logger.info(
                            "N5 强制采纳 top candidate：edge=%s param=%s value=%s %s",
                            edge_key, top.get("param_name", "kd"),
                            top.get("value"), top.get("unit", "nM"),
                        )
                        # 深度审核报告 §1.1：溯源四元组
                        conf_float = 0.6  # MEDIUM
                        origin = _extract_origin(top)
                        source_type = _detect_source_type(top, default="RAG")
                        parameters[edge_key] = {
                            "edge_key": edge_key,
                            "param_name": top.get("param_name", "kd"),
                            "value": float(top.get("value")),
                            "unit": top.get("unit", "nM"),
                            "source": source_type,
                            "confidence": conf_float,
                            "confidence_label": "MEDIUM",
                            "origin": origin,
                            "is_fallback": False,
                            "missing_parameter": False,
                        }
                        rag_selected_params[edge_key] = {
                            "param_found": True,
                            "selected_params": [{
                                "param_name": top.get("param_name", "kd"),
                                "value": top.get("value"),
                                "unit": top.get("unit", "nM"),
                                "source": top.get("source", "RAG"),
                            }],
                            "reasoning": "检索到候选参数，强制采纳以避免全部回退到估算。",
                            "fallback_to_estimation": False,
                        }
                    else:
                        rag_fallback = True
                        # 深度审核报告 §1.1：估算兜底（仅 Template-only 模式允许）
                        conf_float = 0.2  # LOW
                        parameters[edge_key] = {
                            "edge_key": edge_key,
                            "param_name": "kd",
                            "value": 10.0,
                            "unit": "nM",
                            "source": "Inferred",
                            "confidence": conf_float,
                            "confidence_label": "LOW",
                            "origin": "estimated_default",
                            "is_fallback": True,
                            "missing_parameter": True,
                        }
                        missing_parameters.append(f"{edge_key}:kd")
            except Exception as exc:
                logger.warning("N5 RAGDecisionOutput 决策失败（边 %s）：%s", edge_key, exc)
                rag_fallback = True
                # 深度审核报告 §1.1：异常时溯源兜底
                top_for_origin = candidates[0] if candidates else None
                origin = _extract_origin(top_for_origin)
                source_type = _detect_source_type(top_for_origin, default="Inferred")
                parameters[edge_key] = {
                    **(candidates[0] if candidates else {}),
                    "edge_key": edge_key,
                    "source": source_type,
                    "confidence": 0.4,
                    "confidence_label": "MEDIUM",
                    "origin": origin,
                    "is_fallback": False,
                    "missing_parameter": True,
                }
                missing_parameters.append(f"{edge_key}:exception")
                rag_selected_params[edge_key] = {
                    "param_found": False,
                    "selected_params": [],
                    "reasoning": f"RAG 决策异常：{exc}",
                    "fallback_to_estimation": True,
                }
        else:
            # ChromaDB + PubMed 均无命中，标记 fallback
            rag_fallback = True
            # 深度审核报告 §1.1：完全缺失，标记 missing_parameter
            parameters[edge_key] = {
                "edge_key": edge_key,
                "param_name": "kd",
                "value": 10.0,
                "unit": "nM",
                "source": "Inferred",
                "confidence": 0.2,
                "confidence_label": "LOW",
                "origin": "estimated_default",
                "is_fallback": True,
                "missing_parameter": True,
            }
            missing_parameters.append(f"{edge_key}:kd")
            rag_selected_params[edge_key] = {
                "param_found": False,
                "selected_params": [],
                "reasoning": "ChromaDB 与 PubMed 均无命中，回退到估算。",
                "fallback_to_estimation": True,
            }

    # 计算 RAG 命中率
    found_count = sum(
        1 for d in rag_selected_params.values() if d.get("param_found")
    )
    total_edges = len(edges)
    rag_hit_rate = round(found_count / total_edges, 2) if total_edges > 0 else 0.0
    rag_summary = f"已为 {found_count}/{total_edges} 条边检索到真实参数"
    if rag_fallback:
        rag_summary += "，其余边将使用估算值。"
    else:
        rag_summary += "。"
    logger.info("N5 RAG 命中率：%s（%d/%d）", rag_hit_rate, found_count, total_edges)

    # 深度审核报告 §3.3：RAG 命中率 metrics 埋点
    try:
        from app.metrics import get_metrics
        get_metrics().record_rag_hit(rag_hit_rate, found_count, total_edges)
    except Exception as metrics_exc:
        logger.debug("RAG 命中率 metrics 埋点失败：%s", metrics_exc)

    # 深度审核报告 §4.3：多级降级模式判定
    # - Full Mode: RAG + SBML + Sandbox 全部可用（rag_hit_rate >= 0.5 且无 missing_parameter）
    # - RAG-only Mode: SBML 失败时，仅使用 RAG 参数（rag_hit_rate >= 0.3 但有部分 missing）
    # - Template-only Mode: RAG 严重缺失（rag_hit_rate < 0.3 或全部 missing）
    if rag_hit_rate >= 0.5 and not missing_parameters:
        degradation_mode = "full"
    elif rag_hit_rate >= 0.3:
        degradation_mode = "rag_only"
    else:
        degradation_mode = "template_only"
    # 记录降级模式 metrics
    try:
        from app.metrics import get_metrics
        get_metrics().record_degradation(degradation_mode)
    except Exception as metrics_exc:
        logger.debug("降级模式 metrics 埋点失败：%s", metrics_exc)
    logger.info(
        "N5 降级模式：%s（rag_hit_rate=%.2f, missing=%d）",
        degradation_mode, rag_hit_rate, len(missing_parameters),
    )

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
        "source_distribution": aggregated_source_dist,
        "total_candidates": total_candidates,
        "top_selections": aggregated_top_selections[:6],  # 限制为 top 6 供前端展示
        "hit_rate": rag_hit_rate,
        "drug_candidates": drug_candidates,
        "online_fallback_enabled": settings.RAG_ONLINE_FALLBACK,
        "missing_parameters": missing_parameters,
        "degradation_mode": degradation_mode,
    }

    latency_ms = (time.time() - start_ts) * 1000
    return {
        "parameters": parameters,
        # TODO: P0-3 — 同时产出 v1 兼容字段，避免 node1_6_pkpd_inference 拿不到 RAG 药物候选
        "rag_selected_params": rag_selected_params,
        "rag_fallback": rag_fallback,
        "rag_summary": rag_summary,
        "rag_hit_rate": rag_hit_rate,
        "rag_insights": rag_insights,
        "drug_candidates": drug_candidates,
        # 深度审核报告 §1.1 + §4.3：参数溯源 + 降级模式
        "missing_parameters": missing_parameters,
        "degradation_mode": degradation_mode,
        "agent_dispatches": [orchestrator.complete_dispatch(
            "n5_parameter_rag", latency_ms=latency_ms
        )],
        "token_usage": merge_usage(
            state.get("token_usage"), usage_from_accumulator(usage_handler)
        ),
    }


# =============================================================================
# N6 — ODE Generator（Template + Rule）
# =============================================================================
def n6_ode_generator(state: BioDynamicsState) -> dict:
    """LLM 输出 Network JSON（定性关系），Python 用 Jinja2 模板渲染 ODE 代码。

    P0-3 修复：模板选择不再依赖 LLM 单点决策，改由 TemplateSelectorSkill 规则引擎
    决定（关键词匹配 > mechanism 投票 > SBML grounding > LLM 兜底）。
    """
    _emit_in("n6_ode_generator")
    kg = state.get("knowledge_graph", {}) or {}
    parameters = state.get("parameters", {}) or {}
    mechanism = state.get("mechanism", {}) or {}
    edges = kg.get("edges", []) or []
    nodes = kg.get("nodes", []) or []
    user_input = state.get("user_input", "")
    sbml_model_id = state.get("sbml_model_id", "") or extract_biomodel_id(user_input)

    # === TASK 3: 从 BIOMD JSON 直接生成 reaction graph（禁止 template-only generation）===
    # 当用户输入含 BIOMD*/MODEL* ID 时，优先从预处理 JSON 加载真实反应网络，
    # 覆盖 KG/LLM 生成的简化 edges，确保完整反应拓扑（含复合物、双磷酸化）。
    biomd_reaction_graph = None
    if sbml_model_id:
        try:
            from app.biomodels_reactions import (
                get_reaction_graph_for_model,
                reaction_graph_to_edges,
            )
            biomd_reaction_graph = get_reaction_graph_for_model(sbml_model_id)
            if biomd_reaction_graph is not None:
                # 用 BIOMD 真实反应覆盖 KG edges
                biomd_edges = reaction_graph_to_edges(biomd_reaction_graph)
                if biomd_edges:
                    logger.info(
                        "TASK3 BIOMD 反应图覆盖 KG edges: model=%s, "
                        "BIOMD reactions=%d → edges=%d (原 KG edges=%d)",
                        sbml_model_id,
                        len(biomd_reaction_graph.reactions),
                        len(biomd_edges),
                        len(edges),
                    )
                    edges = biomd_edges
                    # 同步用 BIOMD 物种覆盖 KG nodes
                    nodes = [
                        {"name": sp, "id": sp, "type": "Protein"}
                        for sp in biomd_reaction_graph.species
                    ]
                    kg = {**kg, "edges": edges, "nodes": nodes}
                    # 同步用 BIOMD 初始浓度覆盖 parameters
                    for sp, val in biomd_reaction_graph.species_initial.items():
                        parameters.setdefault(sp, {})
                        if isinstance(parameters.get(sp), dict):
                            parameters[sp].setdefault("initial_concentration", val)
        except Exception as exc:
            logger.warning("TASK3 BIOMD 反应图加载失败，回退 KG edges: %s", exc)

    # === P0-3: TemplateSelectorSkill 规则引擎选择模板 ===
    # 不再直接使用 mechanism.template，而是经过规则引擎复核
    llm_template = mechanism.get("template", "Simple_Inhibition")
    pkpd_profile = state.get("pkpd_profile", {}) or {}
    template_selection = _select_template(
        user_input=user_input,
        edges=edges,
        llm_template=llm_template,
        sbml_model_id=sbml_model_id,
        pkpd_profile=pkpd_profile,
    )
    template_name = template_selection.template
    if template_selection.override_llm and template_name != llm_template:
        logger.info(
            "P0-3 TemplateSelector 覆盖 LLM 选择: LLM=%s → 规则=%s (置信度=%.2f, 来源=%s, 理由=%s)",
            llm_template, template_name, template_selection.confidence,
            template_selection.rule_source, template_selection.reason,
        )

    # 构建 Reaction Graph（修复提示词1.md §二.1：KG → Reaction Graph → ODE）
    reaction_graph = build_reaction_graph(kg)
    # 深度审核报告 §2.2：Reaction IR 预校验层（阻断式）
    # 1. Token Boundary Check：禁止子串匹配（防止 ERK1 vs ERK12 误判）
    # 2. Conflict Detection：检测酶-底物角色冲突
    # 3. 失败处理：触发 rule_violations 事件，阻断渲染并请求用户澄清
    pre_check = pre_validate_reaction_graph(reaction_graph)
    reaction_violations = pre_check.get("violations", [])
    reaction_warnings = pre_check.get("warnings", [])
    if reaction_warnings:
        logger.info("Reaction IR 预校验警告: %s", reaction_warnings[:3])
    if not pre_check["passed"]:
        # 阻断渲染：返回 rule_violations 事件，请求用户澄清
        logger.warning("Reaction IR 预校验阻断: %s", reaction_violations[:3])
        return {
            "ode_model": {
                "template": template_name,
                "code": "",
                "parameters_used": {},
                "rule_violations": reaction_violations,
                "template_selection": template_selection.model_dump() if hasattr(template_selection, "model_dump") else {},
                "reaction_graph": reaction_graph,
                "pre_validation": pre_check,
                "time_unit": "min",
            },
            "agent_dispatches": [_dispatch_for_v3_worker(
                "worker_ode", "failed",
                f"Reaction IR 预校验阻断：{len(reaction_violations)} 条违规",
            )],
            # 触发 rule_violations 事件供前端展示
            "raw_cache": {"rule_violations": reaction_violations},
        }
    # 后置校验（仅记录违规不阻断）
    post_violations = validate_reaction_graph(reaction_graph)
    if post_violations:
        logger.warning("Reaction IR 后置校验发现违规: %s", post_violations[:3])

    # ===== Task G: PK/PD 耦合 =====
    # 当 worker_pkpd 产出了 pkpd_profile（含药物+靶点）时，切换到 PKPD 房室模板，
    # 将药物浓度动力学（PK）与靶点抑制效应（PD Emax）显式耦合进 ODE，
    # 避免出现"仅计算效应而不作用于靶点方程"的脱耦问题。
    pkpd_profile = state.get("pkpd_profile", {}) or {}
    drug_regimen = state.get("drug_regimen", []) or []
    pkpd_active = bool(
        pkpd_profile.get("drug_name") and pkpd_profile.get("drug_target")
    )
    _pkpd_vars: dict[str, Any] = {}
    if pkpd_active:
        pk = pkpd_profile.get("pk_params", {}) or {}
        pd = pkpd_profile.get("pd_params", {}) or {}
        _drug_name = pkpd_profile.get("drug_name", "Drug")
        _drug_target = pkpd_profile.get("drug_target", "")
        _compartment = str(pkpd_profile.get("compartment", "1-compartment")).lower()
        # 剂量优先取 drug_regimen，回退默认 100 nM
        _dose = 100.0
        if drug_regimen:
            try:
                _dose = float(drug_regimen[0].get("dose", 100.0) or 100.0)
            except (TypeError, ValueError):
                _dose = 100.0
        try:
            _ec50 = float(pd.get("EC50", 10.0) or 10.0)
            _emax = float(pd.get("Emax", 1.0) or 1.0)
            _gamma = float(pd.get("gamma", 1.0) or 1.0)
            _k10 = float(pk.get("k10", 0.1) or 0.1)
            _k12 = float(pk.get("k12", 0.0) or 0.0)
            _k21 = float(pk.get("k21", 0.0) or 0.0)
        except (TypeError, ValueError):
            _ec50, _emax, _gamma, _k10, _k12, _k21 = 10.0, 1.0, 1.0, 0.1, 0.0, 0.0
        # TODO: P0-5 — EC50 单位校验（统一为 nM 量级，典型 0.1~1000 nM）
        # 读取 LLM 声明的单位字段（若有），用于判断中间区间是否需要修正
        _ec50_unit = str(pd.get("EC50_unit") or pd.get("ec50_unit") or "").lower()
        if _ec50 > 10000.0:
            # µM 误填为 nM（如 50000 表示 50000 nM = 50 µM，实际应为 50 nM）
            logger.warning("EC50=%s 疑似 µM 误填为 nM，除以 1000 修正", _ec50)
            _ec50 = _ec50 / 1000.0
        elif _ec50 < 0.001:
            # M 误填为 nM（如 0.00005 表示 0.00005 nM = 50 pM，实际应为 50 nM）
            logger.warning("EC50=%s 疑似 M 误填为 nM，乘以 1e9 修正", _ec50)
            _ec50 = _ec50 * 1e9
        elif 0.001 < _ec50 < 1.0 and _ec50_unit not in ("nm", "nM".lower()):
            # TODO: P0-5 — 中间区间启发式：LLM 常返回 µM 量级数值（如 0.05）但未换算到 nM
            # 若未显式声明单位为 nM，则视为 µM，乘以 1000 转换为 nM
            logger.warning(
                "EC50=%s 疑似 µM 量级未换算（unit=%s），乘以 1000 修正为 nM",
                _ec50, _ec50_unit or "未声明",
            )
            _ec50 = _ec50 * 1000.0
        # TODO: P1-1 — 初始药物浓度基于 IC50 量级（10× IC50 规则）
        # 起因：drug_regimen 的 dose 由 LLM 自由生成，可能与 EC50 比值失调。
        # 当 EC50=0.05 nM 而 dose=100 nM 时，比值高达 2000×，导致 Emax 饱和、
        # 曲线无剂量响应区。强制 dose >= 10*EC50，确保曲线有明显 IC50/IC90 区间。
        _min_dose = 10.0 * _ec50
        if _dose < _min_dose:
            logger.warning(
                "dose=%s 低于 10×EC50=%s，提升至 %s 以确保剂量响应曲线有效区间",
                _dose, _min_dose, _min_dose,
            )
            _dose = _min_dose
        # 根据房室模型选择模板
        if "2" in _compartment or "two" in _compartment:
            template_name = "PKPD_TwoCompartment"
        else:
            template_name = "PKPD_OneCompartment"
        _pkpd_vars = {
            "drug_name": _drug_name,
            "dose": _dose,
            "k10": _k10,
            "k12": _k12,
            "k21": _k21,
            "ec50": _ec50,
            "emax": _emax,
            "gamma": _gamma,
        }
        logger.info(
            "Task G: PK/PD 耦合激活 → 模板=%s 药物=%s 靶点=%s EC50=%s nM Emax=%s",
            template_name, _drug_name, _drug_target, _ec50, _emax,
        )

    # TODO: P2-5 — 单 inhibition 边强制 Simple_Inhibition（正向规则，防止 LLM 误选 Cascade）
    # 起因：N2 planner 可能输出 Cascade_Inhibition 等复杂模板，但 KG 只有单条 inhibition 边时，
    # 级联模板会过度复杂化。此处正向强制：非 PKPD 路径 + 单 inhibition 边 → Simple_Inhibition。
    if not pkpd_active and len(edges) == 1:
        sole_edge = edges[0]
        if sole_edge.get("interaction", "").lower() == "inhibition":
            if template_name not in ("Simple_Inhibition",):
                logger.info(
                    "P2-5: 单 inhibition 边场景，强制模板 %s → Simple_Inhibition",
                    template_name,
                )
                template_name = "Simple_Inhibition"

    # 1. LLM 输出 network_relations（不写 Python、不给数值）
    network_relations: dict[str, Any] = {
        "variables": [{"name": n.get("name", n.get("id", "")), "role": "species"} for n in nodes],
        "equations": [],
    }
    # TODO: P0-2 — 将 parameters 摘要注入 N6_ODE_PROMPT，让 LLM 基于真实 Kd 决策
    # 仅传递关键字段（edge_key/param_name/value/unit/source），避免完整对象噪声
    params_summary = [
        {
            "edge_key": k,
            "param_name": v.get("param_name", ""),
            "value": v.get("value", ""),
            "unit": v.get("unit", ""),
            "source": v.get("source", ""),
            "is_fallback": v.get("is_fallback", False),
        }
        for k, v in parameters.items()
        if isinstance(v, dict)
    ]
    try:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", N6_ODE_PROMPT),
                (
                    "human",
                    "template={template}\nedges={edges}\nnodes={nodes}\nparameters_summary={parameters_summary}",
                ),
            ]
        )
        chain = prompt.partial(
            template=template_name,
            edges=json.dumps(edges, ensure_ascii=False),
            nodes=json.dumps(nodes, ensure_ascii=False),
            parameters_summary=json.dumps(params_summary, ensure_ascii=False),
        ) | llm
        # IB-029 TODO: with_structured_output
        response = chain.invoke({})
        raw_content = str(response.content)
        try:
            network_relations = _safe_json_parse(raw_content)
            # IB-029: 检测 JSON 解析失败（响应非空但结果为空）
            if not network_relations and raw_content.strip():
                logger.warning("N6 LLM 响应 JSON 解析失败，原始(前200字): %s", raw_content[:200])
                network_relations = {}
        except Exception as parse_exc:
            # IB-029: 解析异常时记录错误并返回空字典（外层 except 会处理降级）
            logger.warning("N6 LLM JSON 解析异常：%s", parse_exc)
            network_relations = {}
    except Exception as exc:
        logger.warning("N6 ODE prompt 失败，使用 KG 直接生成：%s", exc)
        network_relations["equations"] = [
            {
                "lhs": f"d{e.get('target')}/dt",
                "rhs_pattern": "production - degradation * target",
                "type": e.get("interaction", "activation"),
            }
            for e in edges
        ]

    # 2. Rule Engine 校验（参数范围、单位、激活/抑制方向）
    rule_engine = RuleEngine()
    rule_result = rule_engine.check(network_relations, parameters)
    rule_violations = [
        {
            "rule_name": v.rule_name,
            "edge_key": v.edge_key,
            "message": v.message,
            "severity": v.severity,
        }
        for v in rule_result.violations
    ]

    # 3. 用 Jinja2 模板渲染 ODE 代码
    #    根据模板类型与边构建 species_names / y0，确保维度匹配
    simple_templates = {"Simple_Inhibition", "Simple_Activation"}
    cascade_templates = {"Cascade_Inhibition", "Cascade_Activation"}
    pkpd_templates = {"PKPD_OneCompartment", "PKPD_TwoCompartment"}
    # Step 2.2 结构性修复：新增 Signaling_Cascade_Phos 模板（基于 mass-action + 磷酸化机制）
    # 该模板要求 edges 携带 mechanism 字段，params_json 按 mechanism 分配 k_on/k_off/k_phos/k_dephos 等
    phos_cascade_templates = {"Signaling_Cascade_Phos"}

    # TODO: P1-2 — 实体名 → ODE 标识符映射（消除中文/特殊字符）
    # 优先取 entity.aliases[0]（如 "ERα"），否则清洗 name 非 ASCII 字符。
    # 起因：N6 直接取 inh_edge["target"] 导致 ODE 变量名含中文（如"雌激素受体 α"），
    # 触发 Python SyntaxError 或 CSV 列名异常。
    import re as _re
    import unicodedata as _ud
    # 构建 name → aliases 映射（来自 KG nodes）
    _name_to_alias: dict[str, str] = {}
    for n in nodes:
        n_name = n.get("name", "")
        n_aliases = n.get("aliases") or []
        if n_name and n_aliases:
            _name_to_alias[n_name] = str(n_aliases[0])

    def _to_ode_identifier(name: str) -> str:
        """将实体名转换为合法的 Python 标识符（ASCII）。

        优先级：
        1. 若 name 在 _name_to_alias 中且 alias 为 ASCII，直接返回 alias
        2. 否则对 name 做 ASCII 清洗：非字母数字字符替换为 _，确保不以数字开头
        """
        if not name:
            return "Species"
        # 1. 优先用 alias（如"雌激素受体 α" → "ERα"，但 ERα 仍含非 ASCII，继续清洗）
        alias = _name_to_alias.get(name)
        candidate = alias if alias else name
        # 2. 清洗为 ASCII 标识符
        # 2a. NFKC 规范化（全角→半角）
        cleaned = _ud.normalize("NFKC", candidate)
        # 2b. 非 ASCII 字母数字替换为 _（保留 ASCII 字母、数字、下划线）
        cleaned = _re.sub(r"[^A-Za-z0-9_]", "_", cleaned)
        # 2c. 连续 _ 合并
        cleaned = _re.sub(r"_+", "_", cleaned)
        # 2d. 去除首尾 _
        cleaned = cleaned.strip("_")
        # 2e. 若为空或以数字开头，前加 _ 或补前缀
        if not cleaned:
            return "Species"
        if cleaned[0].isdigit():
            cleaned = "_" + cleaned
        return cleaned

    # 构建原始名称（id / name / alias）到 ODE 标识符的统一映射。
    # TODO: P0-4 — 避免 e1/e2 占位符或中文名泄漏到 ODE 变量名；边里的 source/target
    # 可能与节点 name 不一致（如用 entity_id 或别名），统一映射后保证模板内一致。
    _raw_to_ode: dict[str, str] = {}
    for n in nodes:
        n_id = str(n.get("id", "")).strip()
        n_name = str(n.get("name", "")).strip()
        n_aliases = [str(a).strip() for a in (n.get("aliases") or [])]
        ode_id = _to_ode_identifier(n_name) if n_name else _to_ode_identifier(n_id)
        for raw in (n_id, n_name, *n_aliases):
            if raw:
                _raw_to_ode[raw] = ode_id

    def _raw_name_to_ode(raw_name: str) -> str:
        """将边中的原始 source/target 名转换为 ODE 标识符（优先查映射）。"""
        if not raw_name:
            return "Species"
        return _raw_to_ode.get(raw_name, _to_ode_identifier(raw_name))

    # 从边与节点中提取所有唯一物种名（保持出现顺序），并转换为 ODE 标识符
    # 修复：必须包含 reaction_equation 中出现的物种（如 Shc/MEK/MAPK/Raf/RasGDP），
    # 否则磷酸化级联中"酶+底物→酶+产物"形式的底物不会被建模，导致信号无法传递。
    # TASK 2: 接入 species_ontology 过滤 model ID / pathway name / 占位符
    from app.species_ontology import is_valid_species as _is_valid_sp
    import re as _re_mod
    def _unique_species_from_edges(edge_list: list[dict], node_list: list[dict] | None = None) -> list[str]:
        seen: list[str] = []
        # 1. 先从 KG 节点提取（保证骨架蛋白 Shc/Raf/MEK/MAPK 等都被纳入）
        if node_list:
            for n in node_list:
                sp = n.get("name") or n.get("id")
                if sp:
                    identifier = _raw_name_to_ode(sp)
                    # TASK 2: 过滤 model ID / pathway name / 占位符
                    if not _is_valid_sp(identifier):
                        logger.info("TASK2 物种过滤：'%s' 非合法物种，已剔除", identifier)
                        continue
                    if identifier not in seen:
                        seen.append(identifier)
        # 2. 再从边的 source/target 提取
        for e in edge_list:
            for sp in (e.get("source"), e.get("target")):
                if sp:
                    identifier = _raw_name_to_ode(sp)
                    if not _is_valid_sp(identifier):
                        logger.info("TASK2 物种过滤：'%s' 非合法物种，已剔除", identifier)
                        continue
                    if identifier not in seen:
                        seen.append(identifier)
            # 3. 关键修复：从 reaction_equation 中提取所有 token（含底物与复合物）
            # 例：'pEGFR + Shc → pEGFR + pShc' → 提取 Shc
            # 例：'RasGDP → RasGTP (catalyzed by SOS)' → 提取 RasGDP
            rxn = e.get("reaction_equation", "") or ""
            if rxn and "→" in rxn:
                # 去掉括号内的注释（如 "catalyzed by SOS"）
                rxn_clean = _re_mod.sub(r"\([^)]*\)", "", rxn)
                # 拆分反应物与产物
                parts = rxn_clean.split("→")
                for part in parts:
                    tokens = _re_mod.findall(r"[A-Za-z][A-Za-z0-9_\-]*", part)
                    for tok in tokens:
                        identifier = _raw_name_to_ode(tok)
                        if not _is_valid_sp(identifier):
                            continue
                        if identifier not in seen:
                            seen.append(identifier)
        return seen

    # TODO: P1-2 — 从用户输入解析初始浓度，避免 y0 全部硬编码为 10.0
    # 例："EGF=0.008 nM，EGFR=0.3 nM" → {"EGF": 0.008, "EGFR": 0.3}
    def _parse_initial_conditions(user_input: str | None) -> dict[str, float]:
        if not user_input:
            return {}
        # 匹配：物种名 = 数值 [可选单位]
        pattern = re.compile(
            r"([A-Za-z0-9_\-\u03b1-\u03c9\u0391-\u03a9]+)\s*[=＝]\s*"
            r"([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*"
            r"(nM|uM|µM|mM|M|pM|fM)?",
            re.UNICODE,
        )
        result: dict[str, float] = {}
        for name, value, unit in pattern.findall(user_input):
            try:
                val = float(value)
            except (TypeError, ValueError):
                continue
            unit = (unit or "nM").lower()
            if unit == "pm":
                val *= 0.001
            elif unit == "nm":
                pass
            elif unit in ("um", "µm"):
                val *= 1000.0
            elif unit == "mm":
                val *= 1_000_000.0
            elif unit == "m":
                val *= 1_000_000_000.0
            elif unit == "fm":
                val *= 1e-6
            result[name.strip()] = val
        return result

    _initial_conditions = _parse_initial_conditions(state.get("user_input", ""))

    if template_name in pkpd_templates and pkpd_active:
        # PK/PD 房室模板：变量 = [药物房室..., 靶点]
        _dn = _to_ode_identifier(_pkpd_vars["drug_name"])
        _dt = _to_ode_identifier(pkpd_profile.get("drug_target", "Target"))
        if template_name == "PKPD_TwoCompartment":
            species_names = [f"{_dn}_central", f"{_dn}_peripheral", _dt]
            y0 = [_pkpd_vars["dose"], 0.0, _initial_conditions.get(_dt, 10.0)]
        else:
            species_names = [_dn, _dt]
            y0 = [_pkpd_vars["dose"], _initial_conditions.get(_dt, 10.0)]
    elif template_name in simple_templates and edges:
        # Simple 模板硬编码 2 变量：取第一条匹配边
        if template_name == "Simple_Inhibition":
            edge = next((e for e in edges if e.get("interaction") == "inhibition"), edges[0])
        else:
            edge = next((e for e in edges if e.get("interaction") == "activation"), edges[0])
        # TODO: P1-2 — species_names 转换为 ASCII ODE 标识符
        species_names = [_raw_name_to_ode(edge["source"]), _raw_name_to_ode(edge["target"])]
        y0 = [_initial_conditions.get(species_names[0], 10.0), _initial_conditions.get(species_names[1], 10.0)]
    elif template_name in phos_cascade_templates and edges:
        # Step 2.3: Signaling_Cascade_Phos 模板
        # 信号级联模板：所有边涉及的物种都需建模（含 pEGFR/pShc/pMEK/pMAPK 等磷酸化中间体）
        # 关键修复：传入 nodes 以包含骨架蛋白（Shc/Raf/MEK/MAPK），并从 reaction_equation 提取底物
        species_names = _unique_species_from_edges(edges, nodes)
        # 初始条件：ligand/receptor 用用户输入或文献值，其余磷酸化中间体初始为 0
        # EGF=0.008 nM, EGFR=0.3 nM 是 BIOMD0000000205 模型的标准初始条件
        y0 = []
        for sp in species_names:
            if sp in _initial_conditions:
                y0.append(_initial_conditions[sp])
            elif sp.lower() in ("egf",):
                y0.append(0.008)
            elif sp.lower() in ("egfr",):
                y0.append(0.3)
            elif sp.lower() in ("rasgdp",):
                y0.append(1.0)  # Ras 默认 GDP 形式占主导
            elif sp.lower().startswith("p") or sp.lower() in (
                "pegfr", "pshc", "praf", "pmek", "pmapk", "rasgtp", "egf_egfr"
            ):
                y0.append(0.0)  # 磷酸化/激活形式初始为 0
            else:
                # 其余骨架蛋白（Shc/Grb2/SOS/Raf/MEK/MAPK）给一个基础浓度
                y0.append(0.1)
    elif template_name in cascade_templates and edges:
        species_names = _unique_species_from_edges(edges, nodes)
        y0 = [_initial_conditions.get(sp, 10.0) for sp in species_names]
    elif edges:
        species_names = _unique_species_from_edges(edges, nodes)
        y0 = [_initial_conditions.get(sp, 10.0) for sp in species_names]
    else:
        # 无边回退：用节点名或默认值
        species_names = [_raw_name_to_ode(n.get("name", n.get("id", "Species"))) for n in nodes] or ["Species"]
        y0 = [_initial_conditions.get(sp, 10.0) for sp in species_names]

    # 为 Simple 模板确定 inhibitor / activator / target（TODO: P1-2 — 转换为 ASCII 标识符）
    inh_edge = next((e for e in edges if e.get("interaction") == "inhibition"), None)
    act_edge = next((e for e in edges if e.get("interaction") == "activation"), None)
    inhibitor = _raw_name_to_ode(inh_edge["source"]) if inh_edge else (edges[0]["source"] if edges else "Inhibitor")
    inh_target = _raw_name_to_ode(inh_edge["target"]) if inh_edge else (edges[0]["target"] if edges else "Target")
    activator = _raw_name_to_ode(act_edge["source"]) if act_edge else (edges[0]["source"] if edges else "Activator")
    act_target = _raw_name_to_ode(act_edge["target"]) if act_edge else (edges[0]["target"] if edges else "Target")

    # TODO: P0-2 — Simple 模板从 parameters[edge_key]["value"] 读取 RAG Kd，回退到 10.0
    # 根据 Simple 模板类型选择对应边，构建 edge_key 从 parameters 提取真实 Kd
    _simple_edge = inh_edge if template_name == "Simple_Inhibition" else (act_edge if template_name == "Simple_Activation" else (inh_edge or act_edge))
    _simple_kd = 10.0
    if _simple_edge:
        _simple_edge_key = f"{_simple_edge.get('source')}->{_simple_edge.get('target')}"
        _ep = parameters.get(_simple_edge_key, {})
        if _ep and not _ep.get("is_fallback", True):
            try:
                _simple_kd = float(_ep.get("value", 10.0))
                logger.info(
                    "N6 Simple 模板 Kd 来自 RAG：edge=%s kd=%s %s",
                    _simple_edge_key, _simple_kd, _ep.get("unit", "nM"),
                )
            except (TypeError, ValueError):
                _simple_kd = 10.0

    # 为 Cascade 模板构建 params_json（按 ODE 标识符索引，与 SPECIES_NAMES 对齐）
    cascade_params: dict[str, dict] = {}
    if template_name in cascade_templates:
        for edge in edges:
            target_raw = edge.get("target", "")
            target_sp = _raw_name_to_ode(target_raw)
            edge_key = f"{edge.get('source')}->{target_raw}"
            ep = parameters.get(edge_key, {})
            cascade_params[target_sp] = {
                "kd": float(ep.get("value", 10.0)),
                "n": 2.0,
                "production": 1.0,
                "degradation": 0.1,
            }

    # Step 2.2 结构性修复：为 Signaling_Cascade_Phos 模板构建 mechanism-aware params_json
    # 每条 edge 的 mechanism 决定参数键：
    #   binding       → k_on / k_off
    #   phosphorylation → k_phos / k_dephos
    #   exchange      → k_exchange
    #   dephosphorylation → k_dephos
    #   recruitment   → k_on / k_off
    #   degradation   → k_deg
    #   activation (fallback) → k_cat / degradation
    # 参数优先级：
    #   1. RAG 真实值（parameters[edge_key]["value"]，param_name 含 k1/k2/k_on/k_off 等）
    #   2. BIOMD0000000205 文献典型值（按 mechanism 给出生化合理默认）
    phos_cascade_params: dict[str, dict] = {}
    if template_name in phos_cascade_templates:
        # BIOMD0000000205 文献典型值（Schoeberl 2002 / Ung 2008 量级）
        mechanism_defaults = {
            "binding":          {"k_on": 1.0,   "k_off": 0.01},
            "phosphorylation":  {"k_phos": 0.5, "k_dephos": 0.05},  # pEGFR 5-10 min 达峰
            "exchange":         {"k_exchange": 0.1},
            "dephosphorylation": {"k_dephos": 0.05},
            "recruitment":      {"k_on": 0.5,   "k_off": 0.02},
            "degradation":      {"k_deg": 0.01},
            "activation":       {"k_cat": 0.1,  "degradation": 0.01},
            "inhibition":       {"k_on": 0.1,   "k_off": 0.01},
        }
        for edge in edges:
            target_raw = edge.get("target", "")
            source_raw = edge.get("source", "")
            target_sp = _raw_name_to_ode(target_raw)
            mechanism = edge.get("mechanism", "activation")
            edge_key = f"{source_raw}->{target_raw}"
            ep = parameters.get(edge_key, {})

            # 从 RAG 结果中提取数值（k1→k_on/k_phos, k2→k_off/k_dephos 视 mechanism 而定）
            rag_value = None
            rag_param_name = str(ep.get("param_name", "")).lower() if ep else ""
            try:
                rag_value = float(ep.get("value")) if ep and ep.get("value") is not None else None
            except (TypeError, ValueError):
                rag_value = None

            defaults = dict(mechanism_defaults.get(mechanism, mechanism_defaults["activation"]))
            params_for_target: dict[str, float] = {}

            if mechanism == "binding":
                # k1 → k_on, k2 → k_off（若 RAG 命中）
                if rag_value is not None and not ep.get("is_fallback", True):
                    if "k_on" in rag_param_name or rag_param_name in ("k1", "kon"):
                        params_for_target["k_on"] = rag_value
                        params_for_target["k_off"] = defaults["k_off"]
                    elif "k_off" in rag_param_name or rag_param_name in ("k2", "koff"):
                        params_for_target["k_on"] = defaults["k_on"]
                        params_for_target["k_off"] = rag_value
                    else:
                        # 通用 k 值，默认分配给 k_on
                        params_for_target["k_on"] = rag_value
                        params_for_target["k_off"] = defaults["k_off"]
                else:
                    params_for_target = defaults
                params_for_target["degradation"] = 0.001  # 受体-配体复合物缓慢降解

            elif mechanism == "phosphorylation":
                # k1 → k_phos, k2 → k_dephos
                if rag_value is not None and not ep.get("is_fallback", True):
                    if rag_param_name in ("k1", "k_phos", "kphos"):
                        params_for_target["k_phos"] = rag_value
                        params_for_target["k_dephos"] = defaults["k_dephos"]
                    elif rag_param_name in ("k2", "k_dephos", "kdephos"):
                        params_for_target["k_phos"] = defaults["k_phos"]
                        params_for_target["k_dephos"] = rag_value
                    else:
                        params_for_target["k_phos"] = rag_value
                        params_for_target["k_dephos"] = defaults["k_dephos"]
                else:
                    params_for_target = defaults
                params_for_target["degradation"] = 0.002

            elif mechanism == "exchange":
                if rag_value is not None and not ep.get("is_fallback", True):
                    params_for_target["k_exchange"] = rag_value
                else:
                    params_for_target = defaults
                params_for_target["degradation"] = 0.001

            elif mechanism in ("dephosphorylation",):
                if rag_value is not None and not ep.get("is_fallback", True):
                    params_for_target["k_dephos"] = rag_value
                else:
                    params_for_target = defaults

            elif mechanism == "recruitment":
                if rag_value is not None and not ep.get("is_fallback", True):
                    params_for_target["k_on"] = rag_value
                    params_for_target["k_off"] = defaults["k_off"]
                else:
                    params_for_target = defaults

            elif mechanism == "degradation":
                if rag_value is not None and not ep.get("is_fallback", True):
                    params_for_target["k_deg"] = rag_value
                else:
                    params_for_target = defaults

            else:  # activation / inhibition (generic)
                if rag_value is not None and not ep.get("is_fallback", True):
                    params_for_target["k_cat"] = rag_value
                    params_for_target["degradation"] = 0.01
                else:
                    params_for_target = defaults

            phos_cascade_params[target_sp] = params_for_target
            logger.info(
                "N6 phos_cascade 参数映射：edge=%s mechanism=%s target=%s params=%s "
                "(RAG: param_name=%s value=%s is_fallback=%s)",
                edge_key, mechanism, target_sp, params_for_target,
                rag_param_name or "None", ep.get("value") if ep else "None",
                ep.get("is_fallback", True) if ep else True,
            )

    # Cascade 模板需要的 edges_json 必须使用 ODE 标识符，与 SPECIES_NAMES 对齐
    # Step 2.2: 保留 mechanism / reaction_equation 字段供 Signaling_Cascade_Phos 模板使用
    edges_for_template: list[dict] = [
        {
            "source": _raw_name_to_ode(e.get("source", "")),
            "target": _raw_name_to_ode(e.get("target", "")),
            "interaction": e.get("interaction", ""),
            "mechanism": e.get("mechanism", "activation"),
            "reaction_equation": e.get("reaction_equation", ""),
            "directed": e.get("directed", True),
        }
        for e in edges
    ]

    # 选择对应模板的 params_json
    if template_name in phos_cascade_templates:
        _params_json_for_template = phos_cascade_params
    elif template_name in cascade_templates:
        _params_json_for_template = cascade_params
    else:
        _params_json_for_template = {}

    # === P0-2: 仿真时长分层（修复 EGF-EGFR 错误根因 §3.3 时间尺度单位错误）===
    # 不再硬编码 48h，而是根据模板返回 (t_end, n_eval, time_unit)：
    #   - Signaling_Cascade_Phos: 120 min（pEGFR 5-10 min + MAPK 60-120 min）
    #   - Cascade_*: 60 min（默认分钟，避免 48h 错误）
    #   - PKPD_*: 48 h（房室模型小时级）
    #   - Simple_*: 48 h（药物代谢小时级）
    _t_end, _n_eval, _time_unit = get_simulation_time_scale(template_name)
    logger.info(
        "N6 仿真时长分层: template=%s → t_end=%s %s, n_eval=%s",
        template_name, _t_end, _time_unit, _n_eval,
    )

    template_vars: dict[str, Any] = {
        "species_names": species_names,
        "t_end": _t_end,
        "n_eval": _n_eval,
        "y0": y0,
        "edges": edges_for_template,
        "parameters": parameters,
        # Simple 模板变量（target 按模板类型取对应边的 target）
        "inhibitor": inhibitor,
        "target": act_target if template_name == "Simple_Activation" else inh_target,
        "activator": activator,
        # TODO: P0-2 — Kd 优先取 RAG 提取值，回退到 10.0
        "kd": _simple_kd,
        "n_hill": 2,
        "degradation": 0.1,
        "production": 1.0,
        # Cascade 模板变量
        "edges_json": edges_for_template,
        "params_json": _params_json_for_template,
        # PKPD 模板变量（Task G：非 PKPD 模板时使用默认值，模板未引用即忽略）
        "drug_name": _pkpd_vars.get("drug_name", "Drug"),
        "dose": _pkpd_vars.get("dose", 100.0),
        "k10": _pkpd_vars.get("k10", 0.1),
        "k12": _pkpd_vars.get("k12", 0.0),
        "k21": _pkpd_vars.get("k21", 0.0),
        "ec50": _pkpd_vars.get("ec50", 10.0),
        "emax": _pkpd_vars.get("emax", 1.0),
        "gamma": _pkpd_vars.get("gamma", 1.0),
    }
    try:
        code = render_template(template_name, template_vars)
    except Exception as exc:
        logger.error("N6 模板渲染失败：%s", exc)
        code = f"# 模板渲染失败：{exc}\n"

    # === P2-1: 领域常识审查（修复 EGF-EGFR 错误根因 §5.7）===
    # 对渲染后的 ODE 代码做物理 / 生物 / 化学 / 医学多维硬约束审查
    # 高严重性违规（如危险调用、负浓度）记入 rule_violations，沙箱执行前可拦截
    domain_result = check_ode_code(
        code=code,
        species_names=species_names,
        edges=edges_for_template,
        template_name=template_name,
    )
    if not domain_result.passed:
        logger.warning("N6 领域常识审查未通过：%s", domain_result.summary)
        for v in domain_result.violations:
            if v.severity == "high":
                rule_violations.append({
                    "rule": f"domain_{v.category}_{v.rule}",
                    "severity": v.severity,
                    "message": v.message,
                    "fix_suggestion": v.fix_suggestion,
                })
    else:
        logger.info("N6 领域常识审查通过：%s", domain_result.summary)

    ode_model = {
        "template": template_name,
        "code": code,
        "parameters_used": parameters,
        "rule_violations": rule_violations,
        # P0-3: 模板选择元信息（供报告与测试用）
        "template_selection": {
            "template": template_selection.template,
            "confidence": template_selection.confidence,
            "rule_source": template_selection.rule_source,
            "reason": template_selection.reason,
            "override_llm": template_selection.override_llm,
        },
        # P0-1: Reaction Graph（修复提示词1.md §二.1）
        "reaction_graph": reaction_graph,
        # P2-1: 领域常识审查摘要
        "domain_check_summary": domain_result.summary,
        "domain_violations": [
            {
                "category": v.category,
                "severity": v.severity,
                "rule": v.rule,
                "message": v.message,
            }
            for v in domain_result.violations
        ],
        # P0-2: 时间尺度分层信息
        "time_unit": _time_unit,
    }

    return {
        "ode_model": ode_model,
        "network_relations": network_relations,
        "agent_dispatches": [orchestrator.complete_dispatch(
            "n6_ode_generator", latency_ms=0.0
        )],
    }


# =============================================================================
# N7 — Sandbox Execute（AST 预检 + 错误分类）
# =============================================================================
def n7_sandbox_execute(state: BioDynamicsState) -> dict:
    """执行 ODE 代码，含 AST 预检与错误分类。"""
    _emit_in("n7_sandbox_execute")
    ode_model = state.get("ode_model", {}) or {}
    code = ode_model.get("code", "")

    # 1. AST 预检（捕获语法错误）
    error_class = "none"
    try:
        ast.parse(code)
    except SyntaxError as exc:
        logger.warning("N7 AST 预检失败：%s", exc)
        return {
            "execution_result": {
                "status": "error",
                "stdout_stderr": f"AST 预检失败：{exc}",
                "image_base64": "",
            },
            "error_class": "syntax_error",
            "agent_dispatches": [orchestrator.complete_dispatch(
                "n7_sandbox_execute", latency_ms=0.0
            )],
        }

    # 2. 沙箱执行（v2 入口：AST 已通过 + 自动错误分类 + CSV 路径）
    result = execute_simulation_code_v2(code)
    error_class = result.get("error_class", "runtime_error")

    return {
        "execution_result": result,
        "error_class": error_class,
        "agent_dispatches": [orchestrator.complete_dispatch(
            "n7_sandbox_execute", latency_ms=0.0
        )],
    }


# =============================================================================
# N8 — Scientific Feature Extraction（pure NumPy, 零 LLM）
# =============================================================================
def n8_scientific_features(state: BioDynamicsState) -> dict:
    """从 simulation.csv 提取科学指标。"""
    _emit_in("n8_scientific_features")
    execution_result = state.get("execution_result", {}) or {}
    csv_path = execution_result.get("simulation_csv_path") or state.get("simulation_csv_path", "")
    kg = state.get("knowledge_graph", {}) or {}
    # TASK 1: 从 ode_model 透传 time_unit，默认 min
    ode_model = state.get("ode_model", {}) or {}
    time_unit = ode_model.get("time_unit", "min")
    # TASK 6: 瞬态级联系统（Signaling_Cascade_Phos / Cascade_*）禁用 half-life / steady-state
    template_name = ode_model.get("template", "")
    _transient_templates = {"Signaling_Cascade_Phos", "Cascade_Activation", "Cascade_Inhibition"}
    is_transient = template_name in _transient_templates

    metrics: dict = {"species": {}, "overall": {}, "combo": {}}
    metadata: dict = {"method": "none", "version": "v2.0", "confidence": None, "warnings": []}
    confidence = None  # 区分"仿真失败"（None）与"仿真成功但低置信度"（0.0）

    if csv_path:
        try:
            extractor = ScientificFeatureExtractor()
            metrics, metadata = extractor.extract(
                csv_path, kg=kg, time_unit=time_unit, is_transient=is_transient
            )
            confidence = float(metadata.get("confidence", 0.0))
            # TASK 5: 质量守恒检查（< 5% 误差，CONSERVATION_VIOLATION 警告）
            try:
                from app.conservation_checker import (
                    check_conservation_from_csv,
                    format_conservation_warnings,
                )
                ode_model_sp = ode_model.get("species_names", [])
                conservation_report = check_conservation_from_csv(
                    csv_path, species_names=ode_model_sp or None,
                )
                if not conservation_report.passed:
                    cons_warnings = format_conservation_warnings(conservation_report)
                    metadata["warnings"].extend(cons_warnings)
                    metadata["conservation_report"] = {
                        "passed": conservation_report.passed,
                        "summary": conservation_report.summary,
                        "violation_count": len(conservation_report.violations),
                    }
                    logger.warning("TASK5 守恒检查失败：%s", conservation_report.summary)
                else:
                    logger.info("TASK5 守恒检查通过：%s", conservation_report.summary)
            except Exception as cons_exc:
                logger.warning("TASK5 守恒检查异常：%s", cons_exc)
        except Exception as exc:
            logger.warning("N8 特征提取失败：%s", exc)
            metadata["warnings"].append(str(exc))
    else:
        metadata["warnings"].append("simulation.csv 路径缺失")

    return {
        "metrics": metrics,
        "feature_metadata": metadata,
        "confidence": confidence,
        "agent_dispatches": [orchestrator.complete_dispatch(
            "n8_scientific_features", latency_ms=0.0
        )],
    }


# =============================================================================
# N9 — Experiment RAG
# =============================================================================
def n9_experiment_rag(state: BioDynamicsState) -> dict:
    """为关键靶点检索实验方案。"""
    _emit_in("n9_experiment_rag")
    kg = state.get("knowledge_graph", {}) or {}
    mechanism = state.get("mechanism", {}) or {}
    nodes = kg.get("nodes", []) or []

    rag = get_rag_collections()
    protocols: list[dict] = []
    if rag.available:
        # 选前 3 个节点（蛋白/分子）作为靶点
        targets = [
            n.get("name", n.get("id", ""))
            for n in nodes[:3]
            if n.get("type") in ("Protein", "Gene", "Molecule", "Cytokine")
        ]
        seen: set[str] = set()
        for target in targets:
            try:
                hits = rag.search_experiment(
                    f"{target} experimental protocol",
                    target=target,
                    top_k=2,
                )
                for h in hits:
                    clean = {k: v for k, v in h.items() if not k.startswith("_")}
                    key = clean.get("name", "")
                    if key and key not in seen:
                        seen.add(key)
                        protocols.append(clean)
            except Exception as exc:
                logger.warning("N9 实验 RAG 失败 (%s)：%s", target, exc)

    # 防御性过滤：移除与用户输入无关的污染术语结果
    protocols = _filter_contaminated_evidence(
        protocols, state.get("user_input", "")
    )

    return {
        "experiment_protocols": protocols,
        "agent_dispatches": [orchestrator.complete_dispatch(
            "n9_experiment_rag", latency_ms=0.0
        )],
    }


# =============================================================================
# N10 — Evidence RAG
# =============================================================================
def n10_evidence_rag(state: BioDynamicsState) -> dict:
    """检索支持性文献证据，含正向实体白名单 + 防御性过滤 + PMID 提取。"""
    _emit_in("n10_evidence_rag")
    mechanism = state.get("mechanism", {}) or {}
    query = mechanism.get("pathway") or state.get("user_input", "")

    rag = get_rag_collections()
    evidence: list[dict] = []
    if rag.available and query:
        try:
            hits = rag.search_evidence(query, top_k=5)
            evidence = [
                {k: v for k, v in h.items() if not k.startswith("_")}
                for h in hits
            ]
        except Exception as exc:
            logger.warning("N10 evidence RAG 失败：%s", exc)

    # === Task D: 防御性实体过滤（负向） ===
    # 核心场景：用户查询 EGFR/Osimertinib，但 RAG 返回 TGF-β/CD8/SMAD 证据 → 过滤掉。
    evidence = _filter_contaminated_evidence(
        evidence, state.get("user_input", "")
    )

    # === TODO: P1-3 — 正向实体白名单过滤 ===
    # 仅保留含用户输入或 KG 实体名的证据，避免语义相似但主题无关的结果。
    # 实体来源：network_json.nodes[].name + MCP 术语定义 + user_input 关键词。
    kg = state.get("knowledge_graph", {}) or {}
    kg_nodes = kg.get("nodes", []) or []
    user_input = state.get("user_input", "")
    mcp_terms = state.get("mcp_term_definitions") or []
    # 构建实体白名单（小写）
    entity_whitelist: set[str] = set()
    for n in kg_nodes:
        n_name = str(n.get("name", "")).lower()
        if n_name:
            entity_whitelist.add(n_name)
        # 加入 aliases
        for alias in (n.get("aliases") or []):
            if alias:
                entity_whitelist.add(str(alias).lower())
    for d in mcp_terms:
        term = str(d.get("term", "")).lower()
        canonical = str(d.get("canonical_name", "")).lower()
        if term:
            entity_whitelist.add(term)
        if canonical:
            entity_whitelist.add(canonical)
    # user_input 本身作为白名单（分词后取长度 >= 3 的词）
    if user_input:
        for word in re.split(r"[\s,，。.;；]+", user_input):
            if len(word) >= 3:
                entity_whitelist.add(word.lower())

    if evidence and entity_whitelist:
        filtered_evidence: list[dict] = []
        for ev in evidence:
            ev_text = json.dumps(ev, ensure_ascii=False, default=str).lower()
            # 至少匹配一个白名单实体才保留
            if any(ent in ev_text for ent in entity_whitelist):
                filtered_evidence.append(ev)
            else:
                logger.info("N10 正向过滤：证据未含 KG/MCP 实体，已丢弃")
        evidence = filtered_evidence

    # === Task H: PMID 提取与规范化 ===
    # TODO: P1-3 — PMID 正则从 \d{6,} 放宽至 \d{5,}（覆盖早期 5 位 PMID）
    _PMID_RE = re.compile(r"PMID[:\s]*(\d{5,})", re.IGNORECASE)
    # TD-036 (IB-078) 修复：PMID 格式校验器（离线，不调用 PubMed API）。
    # PMID 应为 1-8 位纯数字，不符合则标记 pmid_format_warning=True。
    _PMID_FORMAT_RE = re.compile(r"^\d{1,8}$")
    for ev in evidence:
        pmid = ev.get("pmid", "") or ev.get("source_pmid", "")
        if not pmid:
            # 从 title / context / document 等文本字段中提取 PMID
            for field_name in ("title", "context", "document", "summary", "text", "source"):
                text_val = ev.get(field_name, "")
                if text_val:
                    match = _PMID_RE.search(str(text_val))
                    if match:
                        pmid = match.group(1)
                        break
        ev["pmid"] = str(pmid) if pmid else ""
        # TD-036: PMID 格式校验 — 非空但不匹配 ^\d{1,8}$ 时标记警告
        if ev["pmid"] and not _PMID_FORMAT_RE.match(ev["pmid"]):
            ev["pmid_format_warning"] = True
            logger.warning(
                "TD-036 PMID 格式校验失败（pmid=%s），已标记 pmid_format_warning=True",
                ev["pmid"],
            )

    # TODO: P1-3 — 显式补全 figure_ref / cell_line 字段（依赖原始记录透传，缺失时填空字符串）
    for ev in evidence:
        ev.setdefault("figure_ref", "")
        ev.setdefault("cell_line", "")

    return {
        "paper_evidence": evidence,
        "agent_dispatches": [orchestrator.complete_dispatch(
            "n10_evidence_rag", latency_ms=0.0
        )],
    }


# =============================================================================
# N11 — Scientific Report（Python Markdown 模板 + LLM JSON Fill）
# =============================================================================
def n11_scientific_report(state: BioDynamicsState) -> dict:
    """LLM 输出 JSON 字段，Python 模板渲染 Markdown。"""
    _emit_in("n11_scientific_report")
    metrics = state.get("metrics", {}) or {}
    evidence = state.get("paper_evidence", []) or []
    experiments = state.get("experiment_protocols", []) or []
    kg = state.get("knowledge_graph", {}) or {}
    confidence = state.get("confidence", 0.0)
    # TASK 1: 从 ode_model 透传 time_unit 到报告模板与 LLM prompt
    ode_model = state.get("ode_model", {}) or {}
    time_unit = ode_model.get("time_unit", "min")

    # 1. LLM 输出 JSON 字段
    llm_filled: dict[str, Any] = {
        "mechanism_analysis": "",
        "simulation_interpretation": "",
        "discussion": "",
        "limitations": "",
    }
    try:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", N11_REPORT_FILL_PROMPT),
                (
                    "human",
                    "metrics={metrics}\nexperiments={experiments}\nevidence={evidence}\nconfidence={confidence}\ntime_unit={time_unit}",
                ),
            ]
        )
        chain = prompt.partial(
            metrics=json.dumps(metrics, ensure_ascii=False, default=str),
            experiments=json.dumps(experiments, ensure_ascii=False, default=str),
            evidence=json.dumps(evidence, ensure_ascii=False, default=str),
            confidence=confidence,
            time_unit=time_unit,
        ) | llm
        # IB-029 TODO: with_structured_output
        response = chain.invoke({})
        raw_content = str(response.content)
        try:
            parsed = _safe_json_parse(raw_content)
            # IB-029: 检测 JSON 解析失败（响应非空但结果为空）
            if not parsed and raw_content.strip():
                logger.warning("N11 LLM 响应 JSON 解析失败，原始(前200字): %s", raw_content[:200])
                parsed = {}
            llm_filled = {
                "mechanism_analysis": parsed.get("mechanism_analysis", ""),
                "simulation_interpretation": parsed.get("simulation_interpretation", ""),
                "discussion": parsed.get("discussion", ""),
                "limitations": parsed.get("limitations", ""),
            }
        except Exception as parse_exc:
            # IB-029: 解析异常时记录错误并保留默认空值（不崩溃）
            logger.warning("N11 LLM JSON 解析异常：%s", parse_exc)
    except Exception as exc:
        logger.warning("N11 LLM JSON Fill 失败：%s", exc)

    # 2. Python 模板渲染 Markdown
    renderer = ReportRenderer()
    sandbox_failure_reason = state.get("sandbox_failure_reason", "")
    try:
        markdown = renderer.render(
            llm_filled=llm_filled,
            metrics=metrics,
            evidence=evidence,
            experiments=experiments,
            knowledge_graph=kg,
            confidence=confidence,
            sandbox_failure_reason=sandbox_failure_reason,
            time_unit=time_unit,
        )
        forbidden_violations = renderer.check_forbidden_terms(llm_filled)
    except Exception as exc:
        logger.error("N11 报告渲染失败：%s", exc)
        markdown = f"# 报告生成失败\n\n{llm_filled}\n\n错误：{exc}"
        forbidden_violations = []

    return {
        "report": {
            "markdown": markdown,
            "llm_filled_json": llm_filled,
            "forbidden_terms_violations": forbidden_violations,
        },
        "final_report": markdown,  # 与 v1 兼容
        "agent_dispatches": [orchestrator.complete_dispatch(
            "n11_scientific_report", latency_ms=0.0
        )],
    }


# -----------------------------------------------------------------------------
# 入口：组装 12 节点工作流
# -----------------------------------------------------------------------------
NODES_V2: dict[str, Any] = {
    "n1_ner_entity_normalize": n1_ner_entity_normalize,
    "n2_mechanistic_planner": n2_mechanistic_planner,
    "n3_mechanism_rag": n3_mechanism_rag,
    "n4_kg_builder": n4_kg_builder,
    "n5_parameter_rag": n5_parameter_rag,
    "n6_ode_generator": n6_ode_generator,
    "n7_sandbox_execute": n7_sandbox_execute,
    "n8_scientific_features": n8_scientific_features,
    "n9_experiment_rag": n9_experiment_rag,
    "n10_evidence_rag": n10_evidence_rag,
    "n11_scientific_report": n11_scientific_report,
}
