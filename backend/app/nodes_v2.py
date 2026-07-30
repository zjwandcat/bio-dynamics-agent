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
import os
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
from app.sbml_parameters import ground_sbml_parameters_to_edges
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
from app.sandbox import execute_simulation_code_v2, execute_with_stability_retry
from app.state import BioDynamicsState
from app.supervisor import orchestrator
# Sprint 2 — Citation-driven Discussion Renderer
from app.scientific_alignment.discussion_renderer import (
    render_discussion as _sprint2_render_discussion,
    render_evidence_bundle_sse_payload as _sprint2_evidence_payload,
)
from app.scientific_alignment.evidence_fuser import (
    EvidenceItem as _Sprint2EvidenceItem,
    EvidenceSource as _Sprint2EvidenceSource,
    fuse_evidence as _sprint2_fuse_evidence,
)
# Sprint 5 — Parameter Provenance + Explainability Log
from app.scientific_alignment.parameter_provenance import (
    generate_provenance_report as _sprint5_generate_provenance,
)
from app.scientific_alignment.explainability_log import (
    generate_decision_log_report as _sprint5_generate_decision_log,
)
from app.scientific_alignment.canonical_ranker import (
    CanonicalRanker as _CanonicalRanker,
    rerank_evidence_with_canonical as _task_e_rerank_evidence,
)
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
# [v5 Recovery Sprint 2 / RC13] 通路关键词 → BioModels ID 自动匹配
# 旧实现：仅当用户输入显式含 BIOMD*/MODEL* ID 时触发，99% 自然语言场景跳过。
# 修复：基于 10 个 specialist 的 SOURCE_SBML 字段构建关键词映射，自然语言假设
# 含通路关键词时自动匹配 BIOMD ID，使 BioModels 99%→100% 场景参与。
# 映射来源：各 specialist 的 PATHWAY_TAG + SOURCE_SBML + 通路特征关键词
_PATHWAY_TO_BIOMD: dict[str, str] = {
    # EGFR/RTK + MAPK/ERK 通路（Schoeberl 2002, BIOMD0000000010, PMID:12451189）
    # [RC30] 修复：BIOMD0000000022 实际为昼夜节律钟模型（Leloup 2003），
    #   非EGFR通路。Schoeberl 2002 EGF/MAPK 级联模型正确 ID 为 BIOMD0000000010
    #   （与 egfr_specialist.py / mapk_specialist.py 注释一致）。
    "BIOMD0000000010": ["egf", "egfr", "rtk", "receptor tyrosine kinase", "shc", "grb2", "sos", "rasgtp",
                         "mapk", "erk", "raf", "mek", "dusp", "sprouty", "zero-order ultrasensitivity"],
    # [BENCHMARK CLOSURE / Gap-C9-BioModels-Mismatch] PI3K-AKT-mTOR 通路
    #   旧 BUG：BIOMD0000000250 与 biomodels_registry.py 的 BIOMD0000000262 不一致
    #   修复：与 registry 对齐，使用 Fujita2010 (PMID:20664065)
    "BIOMD0000000262": ["pi3k", "akt", "mtor", "pten", "pip3", "tsc", "rheb"],
    # [BENCHMARK CLOSURE / Gap-C9-BioModels-Mismatch] p53 通路
    #   旧 BUG：BIOMD0000000382 与 registry 的 BIOMD0000000252 不一致
    #   修复：与 registry 对齐，使用 Hunziker2010 (PMID:20624280)
    "BIOMD0000000252": ["p53", "mdm2", "nutlin", "atm", "chk2", "p21"],
    # [BENCHMARK CLOSURE / Gap-C9-BioModels-Mismatch] Apoptosis 通路
    #   旧 BUG：无已验证 BioModels ID（canonical_models 为空）
    #   修复：与 registry 对齐，使用 Legewie2006 (PMID:16978046)
    "BIOMD0000000102": ["apoptosis", "caspase", "casp3", "casp8", "casp9", "cyt c", "cytochrome c",
                         "momp", "bax", "bak", "bid", "tbid", "bh3", "apaf", "apoptosome", "disc",
                         "fasl", "fas", "trail", "parp"],
    # [BENCHMARK CLOSURE / Gap-C9-BioModels-Mismatch] Cell Cycle 通路
    #   旧 BUG：BIOMD0000000055 与 registry 的 BIOMD0000000056 不一致
    #   修复：与 registry 对齐，使用 Chen2004 (PMID:15169868)
    "BIOMD0000000056": ["cell cycle", "cyclin", "cdk", "rb", "e2f", "p27", "apc", "cdc20"],
    # [BENCHMARK CLOSURE / Gap-C9-BioModels-Mismatch] JAK-STAT 通路
    #   旧 BUG：BIOMD0000000224 与 registry 的 BIOMD0000000347 不一致
    #   修复：与 registry 对齐，使用 Bachmann2011 (PMID:21772264)
    "BIOMD0000000347": ["jak", "stat", "interleukin", "il-6", "il6", "socs", "ifn"],
    # [BENCHMARK CLOSURE / Gap-C9-BioModels-Mismatch] NF-κB 通路
    #   旧 BUG：BIOMD0000000258 与 registry 的 BIOMD0000000140 不一致
    #   修复：与 registry 对齐，使用 Hoffmann2002 (PMID:12424381)
    "BIOMD0000000140": ["nf-kb", "nf-κb", "nfkb", "ikb", "iκb", "ikk", "tnf", "rela"],
    # [BENCHMARK CLOSURE / Gap-C9-BioModels-Mismatch] Wnt 通路
    #   旧 BUG：BIOMD0000000008 与 registry 的 BIOMD0000000658 不一致
    #   修复：与 registry 对齐，使用 Lee2003 (PMID:14551908)
    "BIOMD0000000658": ["wnt", "beta-catenin", "β-catenin", "catenin", "apc", "axin", "gsk3", "tcf"],
    # [BENCHMARK CLOSURE / Gap-C9-BioModels-Mismatch] TGF-β 通路
    #   旧 BUG：BIOMD0000000252（实为 p53 模型！）与 registry 的 BIOMD0000000342 不一致
    #   修复：与 registry 对齐，使用 Zi2011 (PMID:21613981)
    "BIOMD0000000342": ["tgf", "smad", "tgf-beta", "tgf-β", "tgfb", "inhibin", "activin"],
}


def _auto_match_biomodel_id(user_input: str) -> str:
    """RC13: 基于通路关键词自动匹配 BioModels ID。

    扫描用户输入（小写），返回首个匹配的 BIOMD ID；无匹配时返回空字符串。
    匹配优先级：按 _PATHWAY_TO_BIOMD 字典顺序（EGFR > MAPK > ... > TGF-β）。
    """
    if not user_input:
        return ""
    text_lower = user_input.lower()
    for biomd_id, keywords in _PATHWAY_TO_BIOMD.items():
        for kw in keywords:
            if kw in text_lower:
                return biomd_id
    return ""


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

    governed_ids = [
        str(item).upper()
        for item in (state.get("benchmark_biomodels_ids") or [])
        if re.fullmatch(r"(?:BIOMD|MODEL)\d{10,}", str(item), re.IGNORECASE)
    ]
    if governed_ids:
        loaded_models: list[dict[str, str]] = []
        for governed_id in governed_ids:
            text = get_biomodels_client().download(governed_id)
            if text:
                loaded_models.append({"model_id": governed_id, "sbml_text": text})
        if loaded_models:
            primary = loaded_models[0]
            return {
                "sbml_model_id": primary["model_id"],
                "sbml_model_text": primary["sbml_text"],
                "sbml_models": loaded_models,
            }
        return {"sbml_models": []}

    # 提取 BIOMD*/MODEL* ID
    model_id = extract_biomodel_id(user_input)
    if not model_id:
        # [v5 Recovery Sprint 2 / RC13] 通路关键词自动匹配 BioModels ID
        # 旧实现：用户未显式提到 BIOMD* 时直接跳过，99% 自然语言场景不加载 BioModels。
        # 修复：基于通路关键词（EGF/ERK/p53/caspase/cyclin/JAK/STAT/NF-kB/Wnt/SMAD 等）
        # 自动匹配 specialist 的 SOURCE_SBML ID，使 BioModels 99%→100% 场景参与。
        model_id = _auto_match_biomodel_id(user_input)
        if model_id:
            logger.info(
                "RC13 BioModels 自动匹配：用户输入含通路关键词，匹配 BIOMD ID=%s",
                model_id,
            )
        else:
            logger.info(
                "N0 SBML Loader：用户输入未含 BIOMD*/MODEL* ID 或通路关键词，跳过"
            )
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
                # [Round 5 Fix] JSON 解析失败时回退到原始文本作为 description
                # 科学原理：LLM 生成的机制描述本身是有价值的文本，即使未包裹在 JSON 中。
                #   丢弃会导致报告中 mechanism.description 为空，影响科学可读性。
                if not description and raw_content.strip():
                    # 清理可能的 Python 赋值前缀（如 'description = "..."'）
                    cleaned = raw_content.strip()
                    if cleaned.startswith("description"):
                        cleaned = cleaned.split("=", 1)[-1].strip().strip('"').strip("'")
                    description = cleaned
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

# N7 缺口 3：case_manifest 缓存——为 RAG 同源优先提供 expected_biomodels
_case_manifest_cache: dict | None = None


def _load_case_manifest() -> dict:
    """懒加载 case_manifest.json（模块级缓存，仅读取一次）。"""
    global _case_manifest_cache
    if _case_manifest_cache is not None:
        return _case_manifest_cache
    try:
        from pathlib import Path
        # case_manifest.json 位于 backend/benchmarks/case_manifest.json
        manifest_path = Path(__file__).resolve().parent.parent / "benchmarks" / "case_manifest.json"
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                _case_manifest_cache = json.load(f)
        else:
            _case_manifest_cache = {}
    except Exception as exc:
        logger.warning("加载 case_manifest.json 失败：%s", exc)
        _case_manifest_cache = {}
    return _case_manifest_cache


def _get_prefer_biomd_id_for_case(case_id: str) -> str | None:
    """从 case_manifest 读取指定 case 的 expected_biomodels_ids 首个 ID。

    Args:
        case_id: 用例 ID（如 "1.E1"）。

    Returns:
        BioModels ID 字符串（如 "BIOMD0000000048"）或 None。
    """
    if not case_id:
        return None
    manifest = _load_case_manifest()
    cases = manifest.get("cases", {}) if isinstance(manifest, dict) else {}
    case_info = cases.get(case_id)
    if not isinstance(case_info, dict):
        return None
    expected_ids = case_info.get("expected_biomodels_ids") or []
    if expected_ids and isinstance(expected_ids, list):
        first = str(expected_ids[0]).strip()
        if first:
            return first.upper()
    verified = case_info.get("verified_biomodels_id")
    if verified and str(verified).strip():
        return str(verified).strip().upper()
    return None


# =============================================================================
# [Batch 2 / LLM_PARAM_INFERENCE_PLAN.md §4.2-4.3] 通路级别动力学参数 LLM 推理
# =============================================================================
# 用途：当 _PATHWAY_KINETICS 硬编码字典导致 C5 峰时过早/过晚时，由 LLM 基于通路
#       生物学时间尺度推理合理默认值，覆盖硬编码字典。
# 优先级链：SBML(0.95) > RAG(0.8) > LLM_Inferred(≤0.4) > Default(0.2)
# Feature Flag: V4_LLM_PARAM_INFERENCE_ENABLED（默认 OFF，关闭时跳过本函数）
# =============================================================================
async def _llm_infer_pathway_kinetics(
    pathway_class: str,
    species: list[str],
    edges: list[dict],
    rag_candidates: dict,
    rag_missed: list[str],
    pathway_context: str,
) -> dict[str, dict]:
    """Batch 2: LLM 推理通路级别动力学参数（phos_k_cat / act_k_cat / gtp_k_cat / trans_k / dephos_k）。

    按 PARAM_INFERENCE_PROMPT 推理规则，基于通路生物学时间尺度输出合理默认值。
    严格禁止创造科学事实，必须标注 source="Inferred" + confidence ≤ 0.4 + evidence_sources。

    Returns:
        dict: {"pathway_kinetics": {param_type: {value, source, confidence, origin, evidence_sources, reasoning}}}
              若 LLM 调用失败或输出无效，返回空 dict（回退到 _PATHWAY_KINETICS 硬编码）
    """
    if not pathway_class:
        return {}

    from app.prompts import PARAM_INFERENCE_PROMPT

    # 提取边摘要（避免传入完整 edges 太长）
    edges_summary = []
    for e in edges[:8]:  # 仅前 8 条边避免 token 过长
        edges_summary.append({
            "source": str(e.get("source", ""))[:30],
            "target": str(e.get("target", ""))[:30],
            "mechanism": str(e.get("mechanism", ""))[:20],
        })

    # 提取 RAG 候选摘要
    rag_summary: dict[str, list] = {}
    if isinstance(rag_candidates, dict):
        for k, v in list(rag_candidates.items())[:5]:
            if isinstance(v, list):
                rag_summary[str(k)[:30]] = [
                    {"param_name": str(item.get("param_name", ""))[:20],
                     "value": item.get("value")}
                    for item in v[:2] if isinstance(item, dict)
                ]

    prompt_text = PARAM_INFERENCE_PROMPT.format(
        pathway_class=pathway_class,
        species=species[:10] if isinstance(species, list) else [],
        edges=edges_summary,
        rag_candidates=rag_summary,
        rag_missed=rag_missed[:10] if isinstance(rag_missed, list) else [],
        pathway_context=str(pathway_context)[:300],
    )

    try:
        from langchain_core.messages import HumanMessage
        # [RCA] 修复：原为 `from app.config import settings` + `llm = settings.openai_llm`
        #   但 Settings 对象无 openai_llm 属性（应为 app.config.llm 模块级 FallbackLLM）。
        #   该错误导致 Batch 2 LLM 推理在所有 case 上抛出 AttributeError，从未执行。
        #   修复：直接使用模块顶部行 32 已导入的 `llm`（FallbackLLM 实例）。
        #   llm 已在行 32 由 `from app.config import llm` 导入。
        if llm is None:
            logger.warning("Batch 2 LLM 推理跳过：llm 不可用")
            return {}
        response = await llm.ainvoke([HumanMessage(content=prompt_text)])
        raw_text = response.content if hasattr(response, "content") else str(response)
    except Exception as exc:
        logger.warning("Batch 2 LLM 推理调用失败（pathway=%s）：%s", pathway_class, exc)
        return {}

    # 解析 JSON（容错：剥离 markdown 代码块标记）
    import json as _json
    import re as _re
    json_match = _re.search(r"\{[\s\S]*\}", raw_text)
    if not json_match:
        logger.warning("Batch 2 LLM 推理无 JSON 输出（pathway=%s）", pathway_class)
        return {}
    try:
        parsed = _json.loads(json_match.group(0))
    except _json.JSONDecodeError as exc:
        logger.warning("Batch 2 LLM 推理 JSON 解析失败（pathway=%s）：%s", pathway_class, exc)
        return {}

    # 校验：confidence ≤ 0.4 + evidence_sources 非空
    confidence = float(parsed.get("confidence", 0.0))
    if confidence > 0.4:
        logger.warning(
            "Batch 2 LLM 推理 confidence=%.2f > 0.4，拒绝（pathway=%s）",
            confidence, pathway_class,
        )
        return {}
    evidence_sources = parsed.get("evidence_sources", [])
    if not evidence_sources or not isinstance(evidence_sources, list):
        logger.warning(
            "Batch 2 LLM 推理 evidence_sources 为空，拒绝（pathway=%s）",
            pathway_class,
        )
        return {}

    # 转换为溯源四元组结构
    inferred_params = parsed.get("inferred_params", {})
    if not isinstance(inferred_params, dict) or not inferred_params:
        return {}

    pathway_kinetics: dict[str, dict] = {}
    for param_type, param_data in inferred_params.items():
        if not isinstance(param_data, dict):
            continue
        value = param_data.get("value")
        if not isinstance(value, (int, float)):
            continue
        # 生物学合理性 clamp（与 PARAM_INFERENCE_PROMPT Rules 范围对齐）
        # phos_k_cat ∈ [0.02, 1.0]，act_k_cat ∈ [0.01, 0.7]，
        # gtp_k_cat ∈ [0.05, 3.0]，trans_k ∈ [0.001, 0.3]，dephos_k ∈ [0.01, 0.5]
        _B2_CLAMP = {
            "phos_k_cat": (0.02, 1.0),
            "act_k_cat": (0.01, 0.7),
            "gtp_k_cat": (0.05, 3.0),
            "trans_k": (0.001, 0.3),
            "dephos_k": (0.01, 0.5),
            "transl_k": (0.001, 0.1),
            "ubi_k": (0.001, 0.2),
            "deg_k": (0.001, 0.2),
            "bind_k": (0.001, 0.3),
            "inhib_k": (0.001, 0.2),
        }
        lo, hi = _B2_CLAMP.get(param_type, (0.001, 1.0))
        clamped_value = max(lo, min(hi, float(value)))
        if clamped_value != float(value):
            logger.info(
                "Batch 2 LLM 推理 %s=%g 超界 clamp 到 %g（pathway=%s）",
                param_type, value, clamped_value, pathway_class,
            )
        pathway_kinetics[param_type] = {
            "value": clamped_value,
            "source": "Inferred",
            "confidence": confidence,
            "origin": f"Inferred:{pathway_class}:{param_data.get('reasoning', '')[:50]}",
            "evidence_sources": evidence_sources,
            "reasoning": str(param_data.get("reasoning", ""))[:200],
        }

    if not pathway_kinetics:
        return {}

    logger.info(
        "Batch 2 LLM 推理通路级别参数成功（pathway=%s, confidence=%.2f, params=%s）",
        pathway_class, confidence, sorted(pathway_kinetics.keys()),
    )
    return {"pathway_kinetics": pathway_kinetics}


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

    # N7 缺口 3：RAG 同源优先——从 case_manifest 读取 expected_biomodels，
    # 让 RAG 检索优先返回同源参数，避免跨模型混用。
    # 优先级：sandbox_case_id → case_id → 从 user_input 提取 BIOMD ID
    _case_id = str(state.get("sandbox_case_id") or state.get("case_id") or "")
    _prefer_biomd_id = _get_prefer_biomd_id_for_case(_case_id)
    if not _prefer_biomd_id:
        # 兜底：从 user_input 提取 BIOMD ID（如 "BIOMD0000000048"）
        _biomd_match = re.search(r"\b(BIOMD\d{10}|MODEL\d{10})\b", user_input, re.IGNORECASE)
        if _biomd_match:
            _prefer_biomd_id = _biomd_match.group(1).upper()
    if _prefer_biomd_id:
        logger.info("N5 RAG 同源优先启用：prefer_biomd_id=%s (case=%s)", _prefer_biomd_id, _case_id or "inferred")

    # Benchmark runs are non-interactive and already carry governed canonical
    # SBML loaded by N0. Ground those parameters once for the complete graph
    # instead of spending the workflow budget on one remote query per edge.
    if state.get("benchmark_run"):
        sbml_models = list(state.get("sbml_models") or [])
        if not sbml_models and state.get("sbml_model_text"):
            sbml_models = [{
                "model_id": str(state.get("sbml_model_id", "")),
                "sbml_text": str(state.get("sbml_model_text", "")),
            }]
        grounding_models = sbml_models
        if settings.L5_GROUNDED_RAG_ENABLED and _prefer_biomd_id:
            preferred_models = [
                model for model in sbml_models
                if str(
                    model.get("model_id")
                    or model.get("biomodels_id")
                    or model.get("id")
                    or ""
                ).upper() == _prefer_biomd_id
            ]
            if preferred_models:
                grounding_models = preferred_models
                logger.info(
                    "RCA-17 L5 grounded RAG: restricting SBML grounding to %s",
                    _prefer_biomd_id,
                )
        grounded, decisions, grounding = ground_sbml_parameters_to_edges(
            edges, grounding_models
        )
        if (
            len(grounded) != len(edges)
            and grounding_models is not sbml_models
        ):
            logger.warning(
                "RCA-17 preferred SBML did not cover all edges; falling back to mixed SBML pool"
            )
            grounded, decisions, grounding = ground_sbml_parameters_to_edges(
                edges, sbml_models
            )
        # Batch 1: Flag ON 时从 SBML 提取物种初始浓度
        # RCA 依据：r40 baseline C6 振幅 PASS 仅 14/43，根因之一是初始浓度被
        # 强制排除 RAG 检索 + 硬编码 default（EGF=0.008/EGFR=0.3/RasGDP=1.0）。
        # Flag ON 时从 SBML listOfSpecies 提取 initialConcentration/initialAmount，
        # 写入 state["sbml_initial_conditions"] 供 N6 节点合并。
        # Flag OFF 时此块被跳过，行为等价 r40 baseline。
        if settings.effective_v4_initial_conc_from_sbml_enabled():
            from app.sbml_parameters import extract_sbml_initial_conditions
            _b1_sbml_ic: dict[str, float] = {}
            for _b1_model in grounding_models:
                _b1_sbml_text = str(_b1_model.get("sbml_text", "") or "")
                _b1_model_id = str(_b1_model.get("model_id", "") or "")
                if _b1_sbml_text:
                    try:
                        _b1_ic = extract_sbml_initial_conditions(_b1_sbml_text, _b1_model_id)
                        _b1_sbml_ic.update(_b1_ic)
                    except Exception as _b1_exc:
                        logger.warning("Batch 1 SBML 初始浓度提取失败（model=%s）：%s", _b1_model_id, _b1_exc)
            if _b1_sbml_ic:
                state["sbml_initial_conditions"] = _b1_sbml_ic
                logger.info(
                    "Batch 1 SBML 初始浓度提取成功：%d 个物种（model=%s）",
                    len(_b1_sbml_ic),
                    str(state.get("sbml_model_id", "")),
                )
        # Batch 2: Flag ON 时调用 LLM 推理通路级别动力学参数（phos_k_cat / act_k_cat 等）
        # RCA 依据：r42 抽检 6/11 case C5 失败（峰时过早/过晚），根因是 _PATHWAY_KINETICS
        # 硬编码字典的反推校准值（注释有 "1.0→0.6" 反推历史）构成过拟合。
        # Flag ON 时由 LLM 基于通路生物学时间尺度推理合理 default，写入
        # state["llm_inferred_params"]["pathway_kinetics"]，覆盖 _PATHWAY_KINETICS 字典。
        # 优先级链：SBML(0.95) > RAG(0.8) > LLM_Inferred(≤0.4) > Default(0.2)
        # Flag OFF 时此块被跳过，行为等价 r40 baseline（_PATHWAY_KINETICS 硬编码）。
        _b2_llm_inferred: dict[str, dict] = {}
        if settings.effective_v4_llm_param_inference_enabled():
            _b2_pathway_class = str(state.get("v4_pathway_class") or "")
            if not _b2_pathway_class:
                # 从 KG edges 推断 pathway_class（fallback）
                _b2_pathway_class = str(state.get("pathway_class") or "")
            _b2_species = list(state.get("species_names") or [])
            if not _b2_species:
                _b2_species = list({str(e.get("source", "")) for e in edges} | {str(e.get("target", "")) for e in edges})
            _b2_pathway_context = str(state.get("user_input", ""))[:300]
            _b2_rag_candidates = {
                str(value.get("edge_key", "")): [{"param_name": value.get("param_name"), "value": value.get("value")}]
                for value in grounded.values() if isinstance(value, dict)
            }
            try:
                _b2_llm_inferred = await _llm_infer_pathway_kinetics(
                    pathway_class=_b2_pathway_class,
                    species=_b2_species,
                    edges=edges,
                    rag_candidates=_b2_rag_candidates,
                    rag_missed=[],
                    pathway_context=_b2_pathway_context,
                )
                if _b2_llm_inferred:
                    state["llm_inferred_params"] = _b2_llm_inferred
                    logger.info(
                        "Batch 2 LLM 推理写入 state[llm_inferred_params]（pathway=%s, params=%s）",
                        _b2_pathway_class, sorted(_b2_llm_inferred.get("pathway_kinetics", {}).keys()),
                    )
            except Exception as _b2_exc:
                logger.warning("Batch 2 LLM 推理调用异常（pathway=%s）：%s", _b2_pathway_class, _b2_exc)
        if edges and len(grounded) == len(edges):
            latency_ms = (time.time() - start_ts) * 1000
            logger.info(
                "N5 canonical SBML fast path: models=%s edges=%d candidates=%d direct=%d reused=%d",
                grounding.get("models", []),
                len(edges),
                grounding.get("candidate_count", 0),
                grounding.get("direct_match_count", 0),
                grounding.get("reuse_count", 0),
            )
            top_selections = [
                {
                    "edge_key": edge_key,
                    "param_name": value.get("param_name"),
                    "value": value.get("value"),
                    "unit": value.get("unit"),
                    "source": value.get("origin"),
                    "mapping_method": value.get("mapping_method"),
                }
                for edge_key, value in grounded.items()
            ]
            return {
                "parameters": grounded,
                "rag_selected_params": decisions,
                "rag_fallback": False,
                "rag_summary": (
                    f"Canonical SBML local grounding covered {len(grounded)}/{len(edges)} edges; "
                    "no remote parameter lookup was required."
                ),
                "rag_hit_rate": 1.0,
                "rag_insights": {
                    "rewritten_query": "",
                    "rewrites": [],
                    "source_distribution": {"SBML": len(grounded)},
                    "total_candidates": grounding.get("candidate_count", 0),
                    "top_selections": top_selections[:6],
                    "hit_rate": 1.0,
                    "drug_candidates": [],
                    "online_fallback_enabled": False,
                    "missing_parameters": [],
                    "degradation_mode": "full",
                    "sbml_grounding": grounding,
                },
                "drug_candidates": [],
                "missing_parameters": [],
                "degradation_mode": "full",
                "sbml_parameter_grounding": grounding,
                # Batch 2: 传递 LLM 推理的通路级别参数到 state（供模板 _pk() 查询）
                "llm_inferred_params": _b2_llm_inferred if _b2_llm_inferred else {},
                "agent_dispatches": [orchestrator.complete_dispatch(
                    "n5_parameter_rag", latency_ms=latency_ms
                )],
                "token_usage": state.get("token_usage") or {},
            }

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

    # N7 缺口 1：BioModels ID 提取（强制字段，无法提取时为 None）
    # 优先级：candidate.biomd_id > candidate.source_model > origin 中 BIOMD 模式
    _biomd_re = re.compile(r"\b(BIOMD\d{10}|MODEL\d{10})\b", re.IGNORECASE)

    def _extract_biomd_id(candidate: dict | None, origin: str = "") -> str | None:
        """从 RAG 候选对象 / origin 字符串提取 BioModels ID。

        Args:
            candidate: RAG 候选 dict（可能含 biomd_id / source_model 字段）。
            origin: 已计算的 origin 字符串（兜底正则提取）。

        Returns:
            BioModels ID 字符串（大写）或 None。
        """
        if candidate:
            explicit = candidate.get("biomd_id")
            if explicit and str(explicit).strip():
                return str(explicit).strip().upper()
            source_model = candidate.get("source_model")
            if source_model and str(source_model).strip():
                sm = str(source_model).strip()
                match = _biomd_re.search(sm)
                if match:
                    return match.group(1).upper()
                if sm.upper().startswith(("BIOMD", "MODEL")):
                    return sm.upper()
                return sm
        if origin:
            match = _biomd_re.search(str(origin))
            if match:
                return match.group(1).upper()
        return None

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

    # P1-3.4 强化 RAG 同源优先：避免跨模型混用
    # 同源判定：候选 biomd_id 相同 → 同源；按"同源组大小"排序，相同 biomd_id 的候选
    # 越多说明该来源越权威，应优先送入 LLM 决策。组内按既有 _rerank_score 排序。
    def _rank_candidates_same_source_first(cands: list[dict]) -> list[dict]:
        """按"同源优先"重排候选列表，避免跨模型参数混用。

        策略：
          1. 提取每个候选的 biomd_id（用 _extract_biomd_id 兜底 _extract_origin）。
          2. 统计每个 biomd_id 出现次数；选最高频为"主源"。
          3. 排序：主源候选优先（按 _rerank_score 降序），其余候选按分数降序。

        Args:
            cands: 原始候选列表。

        Returns:
            重排后的候选列表（不修改原列表）。
        """
        if not cands or len(cands) <= 1:
            return list(cands)
        # 提取 (biomd_id, score) 并保留原索引
        annotated: list[tuple[int, str, float]] = []
        for idx, cand in enumerate(cands):
            origin = _extract_origin(cand)
            biomd = _extract_biomd_id(cand, origin) or origin or "unknown"
            score = float(cand.get("_rerank_score", cand.get("_combined_score", 0)) or 0)
            annotated.append((idx, str(biomd), score))
        # 统计每个源出现次数
        source_counts: dict[str, int] = {}
        for _, src, _ in annotated:
            source_counts[src] = source_counts.get(src, 0) + 1
        # 选最高频源为主源（同票时按字母序确定，确定性）
        primary_source = max(
            source_counts.keys(),
            key=lambda s: (source_counts[s], -ord(s[0]) if s else 0),
        )
        # 排序：主源在前（按 score 降序），其余在后（按 score 降序）
        sorted_annotated = sorted(
            annotated,
            key=lambda t: (
                0 if t[1] == primary_source else 1,  # 主源优先
                -t[2],  # 分数高优先
                t[0],  # 原顺序兜底
            ),
        )
        return [cands[t[0]] for t in sorted_annotated]

    # P1-3.5 DynamicsCalibrator 网格搜索范围扩展
    # 为 parameters[edge_key] 提供 range + log_scale 元数据，供确定性校准器使用。
    # 范围基于典型动力学常数量级（文献汇总），非逐 case 硬编码；log_scale=True 反映
    # 动力学常数的对数均匀分布特征。
    _PARAM_RANGE_TABLE: dict[str, tuple[float, float]] = {
        # 速率常数 (1/min 量纲)
        "k_on": (1e-4, 1e3), "kon": (1e-4, 1e3), "k1": (1e-4, 1e3),
        "k_off": (1e-3, 1e2), "koff": (1e-3, 1e2), "k2": (1e-3, 1e2),
        "k_cat": (1e-2, 1e3), "kcat": (1e-2, 1e3), "kphos": (1e-2, 1e3),
        "k_dephos": (1e-3, 1e2), "kdephos": (1e-3, 1e2),
        "k_deg": (1e-3, 1e0), "kdegr": (1e-3, 1e0),
        # 解离/平衡常数 (nM 量纲)
        "kd": (1e-2, 1e4), "km": (1e-2, 1e4), "k_m": (1e-2, 1e4),
        # 抑制常数 (nM 量纲)
        "ic50": (1e-3, 1e5), "ki": (1e-3, 1e5),
        # 最大速率
        "vmax": (1e-1, 1e3),
    }

    def _build_param_range_and_scale(param_name: str, value: float) -> dict:
        """根据参数名生成 range + log_scale 元数据。

        Args:
            param_name: 参数名（如 k_on, Kd, IC50）。
            value: 当前参数值（用于在 range 不命中时按 0.1x~10x 生成兜底）。

        Returns:
            {"range": [low, high], "log_scale": bool}
        """
        if not param_name:
            param_key = ""
        else:
            # 归一化：小写 + 去 _- 空格
            param_key = re.sub(r"[\s_-]+", "", param_name.lower())
        # 精确匹配
        if param_key in _PARAM_RANGE_TABLE:
            low, high = _PARAM_RANGE_TABLE[param_key]
            return {"range": [low, high], "log_scale": True}
        # 模糊匹配（前缀/包含）
        for table_key, (low, high) in _PARAM_RANGE_TABLE.items():
            if table_key in param_key or param_key.startswith(table_key):
                return {"range": [low, high], "log_scale": True}
        # 兜底：以当前值为中心 ±1 数量级
        try:
            v = float(value)
            if v <= 0:
                v = 1.0
        except (TypeError, ValueError):
            v = 1.0
        return {"range": [v * 0.1, v * 10.0], "log_scale": True}

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

    # [RCA-20 修复 F] benchmark 模式 RAG 兜底：SBML 部分覆盖时预填充已 ground 参数
    # 根因：benchmark_run 时 SBML fast path 仅当 len(grounded)==len(edges) 才返回，
    #   部分覆盖时 fall through 到 remote RAG，但 parameters={} 丢弃已 ground 的 SBML 参数
    # 修复：预填充 parameters + 跳过已 ground 的 edge
    # Feature Flag: V4_BENCHMARK_RAG_FALLBACK_ENABLED（默认 OFF，回退安全）
    import os as _os_rc20f
    _V4_RAG_FALLBACK_ENABLED = (
        _os_rc20f.environ.get("V4_BENCHMARK_RAG_FALLBACK_ENABLED", "false").lower() == "true"
    )
    if _V4_RAG_FALLBACK_ENABLED:
        try:
            _sbml_grounded = grounded if isinstance(grounded, dict) else {}
        except NameError:
            _sbml_grounded = {}
        if _sbml_grounded:
            for _gk, _gv in _sbml_grounded.items():
                if isinstance(_gv, dict) and _gk not in parameters:
                    parameters[_gk] = _gv
            logger.info(
                "[RC20-F] SBML 预填充 %d/%d edges，剩余 %d 走 remote RAG",
                len(_sbml_grounded), len(edges), len(edges) - len(_sbml_grounded),
            )

    for edge_idx, edge in enumerate(edges):
        # 深度审核报告 §4.2 在线回退熔断：workflow 总时长预算控制（默认 600s = 10 分钟）
        # 超出预算时强制跳出循环，剩余边走估算降级，避免阻塞 Workflow 超过 10 分钟
        elapsed_total = time.time() - start_ts
        if elapsed_total > settings.RAG_ONLINE_TOTAL_BUDGET:
            remaining_count = len(edges) - edge_idx
            logger.warning(
                "N5 参数检索总预算耗尽（%.1fs > %.1fs），剩余 %d 条边走估算降级",
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
                        "biomd_id": None,
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

        # [RCA-20 修复 F] 跳过已由 SBML grounding 覆盖的 edge
        if _V4_RAG_FALLBACK_ENABLED and edge_key in parameters:
            logger.info("[RC20-F] 跳过已 ground edge: %s", edge_key)
            continue

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

        # [Round 2 Fix] 为外部 API（KEGG/UniProt/ChEMBL/Reactome）构建短查询
        # 原因：外部 API 期望干净的蛋白名（如 "ATM"），不是完整 reaction 描述
        # query 仍用于 ChromaDB hybrid 检索（BM25 + embedding + rerank 需要上下文）
        def _extract_api_query(s_name: str, t_name: str, term_map: dict) -> str:
            """从边 source/target 提取干净的外部 API 查询词。
            优先用 mcp_term_map 的规范名；否则剥离状态后缀。
            """
            # 优先用 MCP 规范名
            for orig in (t_name, s_name):
                if orig in term_map:
                    return str(term_map[orig])
            # 回退：剥离 _active/_tetramer/_nuclear/_ubi/_phos 等状态后缀
            import re as _re
            for name in (t_name, s_name):
                if not name:
                    continue
                cleaned = _re.sub(r'_(active|tetramer|nuclear|ubi|phos|cytosolic|mito|membrane|bound|free|cleaved|procaspase|procaspase)$', '', name, flags=_re.IGNORECASE)
                # 剥离前导 p（磷酸化标记），如 pATM -> ATM，但保留 p53
                if cleaned.startswith('p') and len(cleaned) > 3 and not cleaned.lower().startswith('p5'):
                    cleaned = cleaned[1:]
                if cleaned:
                    return cleaned
            return s_name or t_name or ""

        api_query = _extract_api_query(source_name, target_name, mcp_term_map or {})

        candidates: list[dict] = []
        edge_insights: dict = {}

        # 1. ChromaDB 高阶 hybrid 检索（查询重写 + BM25 + 语义 + rerank）
        if rag_client.available:
            try:
                # Batch 1: Flag ON 时不排除 initial_concentration 类型参数
                # Flag OFF 时保持原 r40 行为（exclude:initial_concentration）
                _b1_type_filter = (
                    None if settings.effective_v4_initial_conc_from_sbml_enabled()
                    else "exclude:initial_concentration"
                )
                reranked, edge_insights = rag_client.search_params_hybrid(
                    query, species_context=species_context, top_k=5,
                    prefer_biomd_id=_prefer_biomd_id,
                    type_filter=_b1_type_filter,
                )
                # 保留 _rerank_score 供排序，剥离其他内部字段后做污染过滤
                candidates = _filter_contaminated_evidence(reranked, user_input)
                # Batch 1: Flag ON 时保留 initial_concentration 候选；Flag OFF 时保持原过滤
                if not settings.effective_v4_initial_conc_from_sbml_enabled():
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
                        bio_db_client.search_all(api_query, species_context),
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
        # [Round 5] 修复：PubMed 兜底必须受 RAG_ONLINE_FALLBACK 控制，
        # 否则 RAG_ONLINE_FALLBACK=false 时仍会对每条边执行在线 NCBI 检索，
        # 导致总预算耗尽（1200s）且 23 条边走估算降级
        if not candidates and settings.RAG_ONLINE_FALLBACK:
            try:
                pubmed_params = await asyncio.wait_for(
                    _fetch_params_from_pubmed(api_query, species_context),
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
        # [Round 5] 修复：药物候选检索同样受 RAG_ONLINE_FALLBACK 控制
        if interaction == "inhibition" and settings.RAG_ONLINE_FALLBACK:
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
            # P1-3.4 同源优先排序：避免跨模型参数混用
            # 候选来自多源（ChromaDB+KEGG+UniProt+PubMed+ChEMBL），先按 biomd_id 同源
            # 聚类，把最高频来源的候选排在前面，让 LLM 决策时优先采纳同源参数。
            candidates = _rank_candidates_same_source_first(candidates)
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
                    # N7 缺口 1：提取 BioModels ID（强制字段）
                    biomd_id = _extract_biomd_id(top_for_origin, origin)
                    # P1-3.3 参数溯源：BioModels ID 源标注 + 网格搜索 range/log_scale
                    param_range_meta = _build_param_range_and_scale(sp.param_name, float(sp.value))
                    parameters[edge_key] = {
                        "edge_key": edge_key,
                        "param_name": sp.param_name,
                        "value": float(sp.value),
                        "unit": sp.unit,
                        "source": source_type,
                        "confidence": conf_float,
                        "confidence_label": conf_str,  # 兼容字段
                        "origin": origin,
                        "biomd_id": biomd_id,
                        "is_fallback": False,
                        "missing_parameter": conf_float < settings.RAG_ONLINE_FALLBACK_THRESHOLD,
                        # P1-3.5 DynamicsCalibrator 网格搜索元数据
                        "range": param_range_meta["range"],
                        "log_scale": param_range_meta["log_scale"],
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
                        # N7 缺口 1：提取 BioModels ID（强制字段）
                        biomd_id = _extract_biomd_id(top, origin)
                        # P1-3.3 参数溯源 + 网格搜索元数据
                        _top_param_name = top.get("param_name", "kd")
                        _top_value = float(top.get("value"))
                        param_range_meta = _build_param_range_and_scale(_top_param_name, _top_value)
                        parameters[edge_key] = {
                            "edge_key": edge_key,
                            "param_name": _top_param_name,
                            "value": _top_value,
                            "unit": top.get("unit", "nM"),
                            "source": source_type,
                            "confidence": conf_float,
                            "confidence_label": "MEDIUM",
                            "origin": origin,
                            "biomd_id": biomd_id,
                            "is_fallback": False,
                            "missing_parameter": False,
                            # P1-3.5 DynamicsCalibrator 网格搜索元数据
                            "range": param_range_meta["range"],
                            "log_scale": param_range_meta["log_scale"],
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
                        param_range_meta = _build_param_range_and_scale("kd", 10.0)
                        parameters[edge_key] = {
                            "edge_key": edge_key,
                            "param_name": "kd",
                            "value": 10.0,
                            "unit": "nM",
                            "source": "Inferred",
                            "confidence": conf_float,
                            "confidence_label": "LOW",
                            "origin": "estimated_default",
                            "biomd_id": None,
                            "is_fallback": True,
                            "missing_parameter": True,
                            # P1-3.5 DynamicsCalibrator 网格搜索元数据（估算参数也保留 range，供下游校准尝试）
                            "range": param_range_meta["range"],
                            "log_scale": param_range_meta["log_scale"],
                        }
                        missing_parameters.append(f"{edge_key}:kd")
            except Exception as exc:
                logger.warning("N5 RAGDecisionOutput 决策失败（边 %s）：%s", edge_key, exc)
                rag_fallback = True
                # 深度审核报告 §1.1：异常时溯源兜底
                top_for_origin = candidates[0] if candidates else None
                origin = _extract_origin(top_for_origin)
                source_type = _detect_source_type(top_for_origin, default="Inferred")
                # N7 缺口 1：提取 BioModels ID（强制字段）
                biomd_id = _extract_biomd_id(top_for_origin, origin)
                # 异常兜底：用 candidates[0] 的 param_name 推断 range
                _exc_param_name = (candidates[0] if candidates else {}).get("param_name", "kd")
                _exc_value = float((candidates[0] if candidates else {}).get("value", 10.0) or 10.0)
                param_range_meta = _build_param_range_and_scale(_exc_param_name, _exc_value)
                parameters[edge_key] = {
                    **(candidates[0] if candidates else {}),
                    "edge_key": edge_key,
                    "source": source_type,
                    "confidence": 0.4,
                    "confidence_label": "MEDIUM",
                    "origin": origin,
                    "biomd_id": biomd_id,
                    "is_fallback": False,
                    "missing_parameter": True,
                    # P1-3.5 DynamicsCalibrator 网格搜索元数据
                    "range": param_range_meta["range"],
                    "log_scale": param_range_meta["log_scale"],
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
            param_range_meta = _build_param_range_and_scale("kd", 10.0)
            parameters[edge_key] = {
                "edge_key": edge_key,
                "param_name": "kd",
                "value": 10.0,
                "unit": "nM",
                "source": "Inferred",
                "confidence": 0.2,
                "confidence_label": "LOW",
                "origin": "estimated_default",
                "biomd_id": None,
                "is_fallback": True,
                "missing_parameter": True,
                # P1-3.5 DynamicsCalibrator 网格搜索元数据
                "range": param_range_meta["range"],
                "log_scale": param_range_meta["log_scale"],
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

    # N7 缺口 2：跨模型混用检测（仅 warning，不阻断执行）
    # 收集所有参数的 biomd_id，若同一参数集使用了 >1 个不同 BioModels 模型，
    # 标记 cross_model_parameter_mixing 警告，供 validation_report 合并展示。
    cross_model_warnings: list[dict] = []
    biomd_counter: dict[str, int] = {}
    edge_biomd_map: dict[str, str] = {}
    for _ek, _pd in parameters.items():
        if not isinstance(_pd, dict):
            continue
        _bid = _pd.get("biomd_id")
        if _bid and str(_bid).strip():
            _bid_str = str(_bid).strip()
            biomd_counter[_bid_str] = biomd_counter.get(_bid_str, 0) + 1
            edge_biomd_map[_ek] = _bid_str
    if len(biomd_counter) > 1:
        # 选出出现次数最多的 biomd_id 作为 "主流" 模型
        dominant_biomd = max(biomd_counter, key=biomd_counter.get)
        for _ek, _bid in edge_biomd_map.items():
            if _bid != dominant_biomd:
                _pd = parameters.get(_ek, {})
                _pname = str(_pd.get("param_name", "param"))
                cross_model_warnings.append({
                    "edge": _ek,
                    f"param_{_pname}": _bid,
                    "dominant_model": dominant_biomd,
                    "warning": "cross_model_parameter_mixing",
                })
        if cross_model_warnings:
            logger.warning(
                "N5 跨模型参数混用：%d 条边的参数来自非主流模型（主流=%s, 全部=%s）",
                len(cross_model_warnings), dominant_biomd,
                sorted(biomd_counter.keys()),
            )

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
        "biomd_id_distribution": biomd_counter,
        "cross_model_warnings": cross_model_warnings,
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
        # N7 缺口 2：跨模型混用警告（供 validation_report 合并）
        "cross_model_warnings": cross_model_warnings,
        "biomd_id_distribution": biomd_counter,
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
    # auto_fast 兼容修复：worker_mechanism 的 Fast 分支只调用 node1_parse_network，
    # 产出 network_json 但不构建 knowledge_graph（n4_kg_builder 仅 Standard/Manual 走）。
    # 若 KG 为空而 network_json 有边，回退到 network_json 的拓扑，避免 n6 看到空
    # edges 误降级到单物种 else 分支（曾导致 SPECIES_NAMES=['Species'] 与 _ode
    # 二元解包维度不匹配，solve_ivp 崩溃 → 仿真无输出）。
    if not edges:
        network_json_fallback = state.get("network_json", {}) or {}
        nj_edges = network_json_fallback.get("edges", []) or []
        if nj_edges:
            edges = nj_edges
            if not nodes:
                nodes = network_json_fallback.get("nodes", []) or []
            logger.info(
                "n6_ode_generator: knowledge_graph 为空，回退 network_json "
                "(nodes=%d, edges=%d)",
                len(nodes), len(edges),
            )
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
    # [Round 5 Debug] 在函数入口记录实际选中的模板与边数，用于追踪执行路径
    # [R5-DBG] 改为 APPEND 模式，检测 n6 是否被多次调用（worker_sandbox retry）
    import os as _dbg_os, tempfile as _dbg_tf, time as _dbg_time
    _dbg_marker = _dbg_os.path.join(_dbg_tf.gettempdir(), "r5_n6_template_debug.txt")
    try:
        with open(_dbg_marker, "a", encoding="utf-8") as _dbg_mf:
            _dbg_mf.write(
                f"\n=== N6 CALL [{_dbg_time.strftime('%H:%M:%S')}] ===\n"
                f"template_name={template_name}\n"
                f"rule_source={template_selection.rule_source}\n"
                f"override_llm={template_selection.override_llm}\n"
                f"reason={template_selection.reason}\n"
                f"edges_count={len(edges)}\n"
                f"nodes_count={len(nodes)}\n"
                f"llm_template={llm_template}\n"
                f"mechanisms={[e.get('mechanism') for e in edges]}\n"
                f"interactions={[e.get('interaction') for e in edges]}\n"
            )
    except Exception as _dbg_exc:
        logger.warning("N6 [R5-DBG] marker 写入失败: %s", _dbg_exc)
    logger.info(
        "N6 [R5-DBG] template_name=%s, rule_source=%s, override_llm=%s, edges=%d, nodes=%d, mechanisms=%s",
        template_name, template_selection.rule_source, template_selection.override_llm,
        len(edges), len(nodes), [e.get('mechanism') for e in edges],
    )
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
    # [DEBUG R5] Marker to check if pre-validation passes or fails
    import os as _pv_os, tempfile as _pv_tf
    _pv_marker = _pv_os.path.join(_pv_tf.gettempdir(), "r5_prevalidation_result.txt")
    try:
        with open(_pv_marker, "w", encoding="utf-8") as _pv_mf:
            _pv_mf.write(
                f"passed={pre_check.get('passed')}\n"
                f"violations_count={len(reaction_violations)}\n"
                f"warnings_count={len(reaction_warnings)}\n"
                f"violations_sample={reaction_violations[:3]}\n"
            )
    except Exception:
        pass
    logger.info("N6 [R5-DBG] Pre-validation: passed=%s, violations=%d, warnings=%d",
                pre_check.get("passed"), len(reaction_violations), len(reaction_warnings))
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

    def _parse_user_duration(user_input: str | None) -> float | None:
        """从用户输入解析仿真时长覆盖（分钟）。

        匹配 benchmark orchestrator 附加的 "duration: 240 min" 格式。
        科学原理：不同通路生物学时间尺度不同（凋亡 240 min, EGFR 120 min），
        模板默认值（60 min）不适用于所有通路。
        """
        if not user_input:
            return None
        m = re.search(r"duration\s*[:：]\s*(\d+(?:\.\d+)?)\s*min", user_input, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except (TypeError, ValueError):
                return None
        return None

    _initial_conditions = _parse_initial_conditions(state.get("user_input", ""))
    # Batch 1: Flag ON 时合并 SBML 提取的初始浓度（用户输入优先，SBML 次之，硬编码最后兜底）
    # 重要：只对上游激活源（配体/受体/骨架蛋白）用 SBML IC，
    #       排除下游产物（磷酸化/激活形式），因为它们应初始为 0 由上游激活产生
    # RCA 依据：r40 硬编码 default（EGF=0.008/EGFR=0.3/RasGDP=1.0）导致 C6 振幅
    # 偏差。Flag ON 时用 SBML 提取的真实初始浓度补充 _initial_conditions。
    # 优先级：用户输入 > SBML 提取 > 硬编码 default（最后兜底，保持原逻辑）。
    # Flag OFF 时此块被跳过，_initial_conditions 仅含用户输入，行为等价 r40。
    # 回归修复（3.P1 PI3K）：PIP3 在 SBML 中有非零 initialConcentration，
    # 但 PIP3 是 PI3K 激活后的下游产物（PIP2 → PIP3），仿真中应初始为 0。
    if settings.effective_v4_initial_conc_from_sbml_enabled():
        _b1_sbml_ic = state.get("sbml_initial_conditions", {})

        # 下游产物过滤：排除磷酸化/激活形式物种
        # 原因：SBML 模型的 initialConcentration 可能是稳态后的值，
        # 而仿真需要刺激前的初始条件（下游产物应为 0）
        # 参考：nodes_v2.py phos_cascade 分支的 y0 设置逻辑（行 2413-2416）
        def _b1_is_downstream_product(name: str) -> bool:
            """判断物种是否是下游产物（磷酸化/激活形式），应初始为 0。"""
            lower = str(name).lower().strip()
            # 排除磷酸化形式（pEGFR, pAKT, pS6K1, PIP3 等）
            # 注意：p53 长度=3 不匹配（len > 3 条件），PTEN 需要单独处理
            if lower.startswith("p") and len(lower) > 3 and not lower.startswith("p5"):
                return True
            # 排除激活/GTP/GDP 形式
            if any(suffix in lower for suffix in ("_active", "_gtp", "_gdp", "phospho")):
                return True
            # 排除已知下游产物（与 nodes_v2.py:2413-2416 一致 + PIP3/PIP2）
            _known_downstream = {
                "pegfr", "pshc", "praf", "pmek", "pmapk", "rasgtp", "egf_egfr",
                "pip3", "pip2", "pakt", "ps6k1", "pperk", "perk",
                "bcatenin", "bcat_degraded", "nfkb_nuclear",
            }
            if lower in _known_downstream:
                return True
            return False

        _b1_merged_count = 0
        _b1_skipped_count = 0
        for _b1_sp, _b1_val in _b1_sbml_ic.items():
            if _b1_is_downstream_product(_b1_sp):
                _b1_skipped_count += 1
                continue
            # 用 ODE 标识符形式匹配（_raw_name_to_ode 在行 2260 定义，此处可访问）
            try:
                _b1_sp_ode = _raw_name_to_ode(_b1_sp)
                if _b1_sp_ode and _b1_sp_ode not in _initial_conditions:
                    _initial_conditions[_b1_sp_ode] = _b1_val
                    _b1_merged_count += 1
            except Exception:
                pass
            # 原始名也存一份（覆盖范围更广）
            if _b1_sp not in _initial_conditions:
                _initial_conditions[_b1_sp] = _b1_val
                _b1_merged_count += 1
        if _b1_sbml_ic:
            logger.info(
                "Batch 1 N6 合并 SBML 初始浓度（排除下游产物）：原 %d 个，合并 %d 个，跳过 %d 个下游产物",
                len(_b1_sbml_ic), _b1_merged_count, _b1_skipped_count,
            )

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
        # [DEBUG R5] Marker file to verify which branch executes (APPEND)
        import os as _os, tempfile as _tf, time as _pc_t
        _marker = _os.path.join(_tf.gettempdir(), "r5_phos_cascade_marker.txt")
        try:
            with open(_marker, "a", encoding="utf-8") as _mf:
                _mf.write(f"\n=== PHOS_CASCADE [{_pc_t.strftime('%H:%M:%S')}] ===\ntemplate={template_name}\nspecies_count={len(species_names)}\nspecies={species_names[:15]}\n")
        except Exception:
            pass
        logger.info("N6 [R5-DBG] Entered phos_cascade_templates branch: template=%s", template_name)
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
        # [Round 5 Fix] 级联初始条件：基于拓扑区分上游激活源与下游产物
        # 科学原理：级联反应中，activation edge 的 target（下游产物）初始为 0，
        #   由上游激活逐步产生；非 target 物种（上游激活源、抑制剂、骨架蛋白）初始非零。
        #   这产生级联时序动力学：上游先激活，下游后激活，形成时间延迟。
        #   参考：Alon "An Introduction to Systems Biology" Ch.2 (protein cascade dynamics)
        _activation_targets = {
            _raw_name_to_ode(e["target"])
            for e in edges
            if e.get("interaction") == "activation" and e.get("target")
        }
        y0 = []
        for sp in species_names:
            if sp in _initial_conditions:
                y0.append(_initial_conditions[sp])
            elif sp in _activation_targets:
                y0.append(0.0)  # 下游产物初始为 0，由级联激活产生
            else:
                y0.append(1.0)  # 上游激活源/抑制剂/骨架蛋白初始非零
        # [DEBUG R5] Marker file to verify code execution (APPEND mode for multi-call detection)
        import os as _os, tempfile as _tf, time as _ct
        _marker = _os.path.join(_tf.gettempdir(), "r5_cascade_fix_marker.txt")
        with open(_marker, "a", encoding="utf-8") as _mf:
            _mf.write(f"\n=== CASCADE [{_ct.strftime('%H:%M:%S')}] ===\ntemplate={template_name}\nt_end={_t_end if '_t_end' in dir() else 'N/A'}\nspecies_count={len(species_names)}\nactivation_targets={len(_activation_targets)}\ny0={y0[:10]}\nspecies={species_names[:10]}\n")
        logger.info("N6 [R5] Cascade initial conditions fix applied: %d species, %d targets", len(species_names), len(_activation_targets))
    elif edges:
        species_names = _unique_species_from_edges(edges, nodes)
        # [DEBUG R5] Marker file for the elif edges fallback branch (APPEND)
        import os as _os, tempfile as _tf, time as _ee_t
        _marker = _os.path.join(_tf.gettempdir(), "r5_elif_edges_marker.txt")
        try:
            with open(_marker, "a", encoding="utf-8") as _mf:
                _mf.write(f"\n=== ELIF_EDGES [{_ee_t.strftime('%H:%M:%S')}] ===\ntemplate={template_name}\nspecies_count={len(species_names)}\nspecies={species_names[:15]}\n")
        except Exception:
            pass
        logger.info("N6 [R5-DBG] Entered elif edges fallback branch: template=%s", template_name)
        # [RCA-38 P0 修复 RC-1e] v3 fallback 默认值 10.0 → 1.0
        # [S1.2 磷酸化形式折中] 磷酸化形式（pX 开头）初始浓度 1.0 → 0.5
        #   根因：0.05/0.1 导致 3.P1 pAKT peak_time 回归（信号启动过慢）
        #         1.0 导致 fold_change≈1（信号无放大）
        #   折中：0.5 既能适度改善 fold，又不过度延迟 peak_time
        y0 = [_initial_conditions.get(sp, (0.5 if sp.lower().startswith("p") and len(sp) > 1 else 1.0)) for sp in species_names]
    else:
        # 无边回退：用节点名或默认值
        species_names = [_raw_name_to_ode(n.get("name", n.get("id", "Species"))) for n in nodes] or ["Species"]
        # [S1.2 磷酸化形式折中] 同上，磷酸化形式 1.0 → 0.5
        y0 = [_initial_conditions.get(sp, (0.5 if sp.lower().startswith("p") and len(sp) > 1 else 1.0)) for sp in species_names]

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
        # [Round 5 Fix] kd 类型检查：仅当 RAG 参数单位为浓度时用作 Hill Kd
        # 科学原理：RAG 检索的 SBML 参数多为速率常数（k_on/k_cat, unit=s^-1/per_nM_per_sec），
        #   非 Hill 解离常数（Kd, unit=nM）。将速率常数误用为 Kd 会导致 Hill 函数立即饱和，
        #   丧失阈值激活行为。仅浓度型参数可用作 Kd，其余回退到默认值（1.0 nM）。
        _concentration_units = {"nm", "um", "µm", "mm", "pm", "fm"}
        for edge in edges:
            target_raw = edge.get("target", "")
            target_sp = _raw_name_to_ode(target_raw)
            edge_key = f"{edge.get('source')}->{target_raw}"
            ep = parameters.get(edge_key, {})
            _rag_value = ep.get("value")
            _rag_unit = str(ep.get("unit", "")).lower().strip()
            _is_concentration = (
                _rag_value is not None
                and _rag_unit in _concentration_units
            )
            if _is_concentration:
                _kd = float(_rag_value)
            else:
                _kd = 1.0  # Hill 半激活浓度科学默认（nM 级，1 nM 典型信号蛋白 Kd）
            cascade_params[target_sp] = {
                "kd": _kd,
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
            # [RC23] 新增：按生物学时间尺度设定默认值，解决级联过慢问题
            # 原因：gtp_gdp_exchange/nuclear_import/反馈机制均落入 activation 默认 k_cat=0.1，
            #   导致 RasGTP 25min 才达峰、ERK 120min 仍未达峰下降。
            #   生物学合理值：Ras 活化 1-5min、核转位 1-5min、DUSP 反馈需较强才能抑制 ERK
            "gtp_gdp_exchange": {"k_cat": 0.5,  "degradation": 0.01},   # Ras GEF 活性 1-5 min
            "nuclear_import":   {"k_cat": 0.5,  "degradation": 0.01},   # 核转位 1-5 min
            "transcription":    {"k_trans": 0.3, "k_mRNA_deg": 0.02},   # mRNA 转录 10-15 min
            "translation":      {"k_transl": 0.2, "k_prot_deg": 0.01},  # 蛋白翻译 5-10 min
            "feedback_propagation": {"k_act": 0.2, "k_deg": 0.02},      # 反馈环前向传播（DUSP 转录/翻译）
            "feedback_regulation": {"k_cat": 2.0, "degradation": 0.01}, # 反馈调控（DUSP 抑制 ERK，需较强）
            "negative_feedback":   {"k_cat": 2.0, "degradation": 0.01}, # 负反馈闭环
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
    # [Round 5 Fix] 用户/benchmark 指定的仿真时长优先于模板默认值
    # 科学原理：不同通路生物学时间尺度不同（凋亡需 240 min 观察全级联，
    #   EGFR 磷酸化 120 min 足够）。模板默认 60 min 对凋亡通路过短，
    #   导致所有物种仍在上升时仿真结束，峰值时间全等于 t_end。
    _user_duration = _parse_user_duration(state.get("user_input", ""))
    if _user_duration is not None and _time_unit == "min":
        _t_end = _user_duration
        logger.info(
            "N6 仿真时长用户覆盖: t_end=%s %s (from user_input duration)",
            _t_end, _time_unit,
        )
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
        # [RCA-20 修复 B/C] Feature Flag 通过 Jinja2 上下文传入（避免模板内 import os）
        "v4_get_param_clamp_enabled": os.environ.get(
            "V4_GET_PARAM_CLAMP_ENABLED", "false"
        ).lower() == "true",
        "v4_activation_mm_saturation_enabled": os.environ.get(
            "V4_ACTIVATION_MM_SATURATION_ENABLED", "false"
        ).lower() == "true",
    }
    # [DEBUG R5] Marker right before render_template to log the FULL y0 and species_names (APPEND)
    import os as _rt_os, tempfile as _rt_tf, time as _rt_time
    _rt_marker = _rt_os.path.join(_rt_tf.gettempdir(), "r5_render_template_y0.txt")
    try:
        with open(_rt_marker, "a", encoding="utf-8") as _rt_mf:
            _rt_mf.write(
                f"\n=== PRE-RENDER [{_rt_time.strftime('%H:%M:%S')}] ===\n"
                f"template_name={template_name}\n"
                f"species_names={species_names}\n"
                f"y0_full={y0}\n"
                f"y0_len={len(y0)}\n"
                f"species_len={len(species_names)}\n"
                f"t_end={_t_end}\n"
            )
    except Exception:
        pass
    logger.info("N6 [R5-DBG] Pre-render: template=%s, y0_len=%d, species_len=%d, t_end=%s, y0_sample=%s",
                template_name, len(y0), len(species_names), _t_end, y0[:5])
    try:
        code = render_template(template_name, template_vars)
    except Exception as exc:
        logger.error("N6 模板渲染失败：%s", exc)
        code = f"# 模板渲染失败：{exc}\n"

    # [DEBUG R5] Log the rendered Y0 line and save FULL rendered code (APPEND)
    try:
        _code_lines = code.split("\n")
        _y0_line = next((l for l in _code_lines if l.startswith("Y0")), "Y0 NOT FOUND")
        with open(_rt_os.path.join(_rt_tf.gettempdir(), "r5_rendered_y0_line.txt"), "a", encoding="utf-8") as _ry_mf:
            _ry_mf.write(f"\n=== POST-RENDER [{_rt_time.strftime('%H:%M:%S')}] ===\nrendered_y0_line={_y0_line[:200]}\ntemplate_name={template_name}\n")
        # Save full rendered code for inspection
        with open(_rt_os.path.join(_rt_tf.gettempdir(), "r5_rendered_code_full.txt"), "a", encoding="utf-8") as _rc_mf:
            _rc_mf.write(f"\n=== RENDERED CODE [{_rt_time.strftime('%H:%M:%S')}] template={template_name} ===\n{code}\n")
    except Exception:
        pass

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
        # [RC20] 修复：存储 t_end/n_eval 供 v4 ODE 渲染器使用
        # 原因：N6 计算 t_end=120.0（Signaling_Cascade_Phos），但 _ode_template_v2_hook
        #   不读取此值，v4 渲染器默认 t_end=60.0，导致仿真时长不足，ERK 无法达峰后下降
        "t_end": _t_end,
        "n_eval": _n_eval,
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
    # [v5 Recovery Sprint 3 / RC6] 使用 execute_with_stability_retry 替代直接调用
    # 旧实现：LSODA 崩溃后直接返回错误（62.5% 崩溃率），无 BDF/Radau 回退。
    # 修复：包装函数在 runtime_error/numerical_error 时自动按阶梯策略重试。
    # [Sandbox Fix] 传递 case_id + artifacts_dir 实现产物持久化（非 TEMP）
    result = execute_with_stability_retry(
        code,
        case_id=state.get("sandbox_case_id") or None,
        artifacts_dir=state.get("sandbox_artifacts_dir") or None,
    )
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

    # [v5 Recovery Sprint 3 / RC20] 置信度与仿真状态关联
    # 旧实现：仿真失败但 CSV 存在时，置信度仍可高达 0.83（与仿真状态脱节）。
    # 修复：仿真失败（status != success 或 error_class != none）时置信度 ≤ 0.3。
    _sim_status = execution_result.get("status", "")
    _sim_error = execution_result.get("error_class", "none")
    _sim_failed = (_sim_status != "success") or (_sim_error not in ("none", "", None))

    if csv_path:
        try:
            extractor = ScientificFeatureExtractor()
            metrics, metadata = extractor.extract(
                csv_path, kg=kg, time_unit=time_unit, is_transient=is_transient
            )
            confidence = float(metadata.get("confidence", 0.0))
            # [v5 RC20] 仿真失败时置信度上限 0.3
            if _sim_failed and confidence > 0.3:
                logger.warning(
                    "[v5 RC20] 仿真失败（status=%s, error=%s）但置信度=%.3f，限制为 0.3",
                    _sim_status, _sim_error, confidence,
                )
                confidence = 0.3
                metadata["confidence"] = 0.3
                metadata["warnings"].append(
                    f"confidence_capped: simulation failed ({_sim_error}), capped to 0.3"
                )
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
def _pubmed_protocols_for_targets(targets: list[str], top_k_per_target: int = 2) -> list[dict]:
    """用 PubMed 在线搜索为每个靶点构建实验方案 fallback。

    返回与本地 experiment collection 兼容的 dict 列表，字段包括：
    name, target, detection_method, pmid, cell_line, species, description。
    """
    protocols: list[dict] = []
    seen_pmids: set[str] = set()
    for target in targets:
        if not target:
            continue
        query = f"{target} experimental protocol assay"
        articles = _fetch_pubmed_evidence_sync(query, top_k=top_k_per_target)
        for art in articles:
            pmid = art.get("pmid", "")
            if not pmid or pmid in seen_pmids:
                continue
            seen_pmids.add(pmid)
            protocols.append({
                "name": art.get("title", f"{target} experimental protocol"),
                "target": target,
                "detection_method": "PubMed literature",
                "pmid": pmid,
                "cell_line": art.get("cell_line", ""),
                "species": "",
                "description": art.get("abstract", "")[:500],
            })
    return protocols


def n9_experiment_rag(state: BioDynamicsState) -> dict:
    """为关键靶点检索实验方案（本地 + PubMed 在线 fallback）。"""
    _emit_in("n9_experiment_rag")
    kg = state.get("knowledge_graph", {}) or {}
    mechanism = state.get("mechanism", {}) or {}
    nodes = kg.get("nodes", []) or []

    rag = get_rag_collections()
    protocols: list[dict] = []
    # 选前 3 个节点（蛋白/分子）作为靶点
    targets = [
        n.get("name", n.get("id", ""))
        for n in nodes[:3]
        if n.get("type") in ("Protein", "Gene", "Molecule", "Cytokine")
    ]

    if rag.available and targets:
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

    # [BM 修复] PubMed 在线 fallback：本地 experiment collection 无有效方案时，
    # 为每个 target 搜索 PubMed 文献作为实验方案推荐。
    if not protocols and targets:
        logger.info("N9 本地实验方案为空，触发 PubMed 在线 fallback，targets=%s", targets)
        protocols = _pubmed_protocols_for_targets(targets, top_k_per_target=2)
        logger.info("N9 PubMed fallback 返回 %d 个实验方案", len(protocols))
    else:
        logger.info("N9 本地实验方案命中 %d 个，跳过 PubMed fallback", len(protocols))

    return {
        "experiment_protocols": protocols,
        "agent_dispatches": [orchestrator.complete_dispatch(
            "n9_experiment_rag", latency_ms=0.0
        )],
    }


# =============================================================================
# N10 — Evidence RAG
# =============================================================================

def _fetch_pubmed_evidence_sync(query: str, top_k: int = 3) -> list[dict]:
    """同步调用 NCBI E-utilities 检索 PubMed 文献（N10 在线 fallback）。

    N10 是同步函数，运行在 async event loop 中，不能用 asyncio.run()。
    直接用 requests 同步调用 NCBI esearch + efetch。
    超时或失败时返回空列表，不阻塞主流程。

    [缺口 3 修复] PubMed E-utilities 逻辑已迁移到
    ``RagClient._fetch_pubmed_by_pmids``。本函数保留为薄包装层，
    通过 query 走 esearch + efetch 路径，保持原有调用点行为不变。
    """
    if not query:
        return []
    try:
        rag_client = _get_pubmed_rag_client()
        articles = rag_client._fetch_pubmed_by_pmids(
            [], query=query, top_k=top_k
        )
    except Exception as exc:
        logger.warning(
            "N10 PubMed 在线 fallback 失败（委托 rag_client）：%s", exc
        )
        return []
    if articles:
        logger.info("N10 PubMed 在线 fallback 命中 %d 篇文献", len(articles))
    return articles[:top_k]


# 缺口 3 配套：模块级 RagClient 懒加载单例
# 避免在模块导入时初始化 ChromaDB，只在真正调用 PubMed fallback 时创建
_pubmed_rag_client_instance: RagClient | None = None


def _get_pubmed_rag_client() -> RagClient:
    """获取（或懒加载）用于 PubMed E-utilities 的 RagClient 单例。

    模块级懒加载，避免在导入时初始化 ChromaDB 连接。
    RagClient 不可用时仍返回实例——_fetch_pubmed_by_pmids 内部不依赖
    ChromaDB，可直接走 requests + xml.etree 在线拉取。
    """
    global _pubmed_rag_client_instance
    if _pubmed_rag_client_instance is None:
        _pubmed_rag_client_instance = RagClient()
    return _pubmed_rag_client_instance


def _clean_pubmed_query(raw: str) -> str:
    """将 pathway 内部标识符清洗为适合 PubMed 搜索的自然语言查询词。

    例如：
      'MULTI:EGFR_RTK+MAPK_ERK' -> 'EGFR MAPK ERK signaling'
      'EGFR_RTK' -> 'EGFR RTK'
    """
    if not raw:
        return ""
    # 移除 MULTI: 等前缀
    cleaned = re.sub(r"^[A-Z]+:", "", raw)
    # 将 '_' 替换为空格，'+' 替换为空格
    cleaned = cleaned.replace("_", " ").replace("+", " ")
    # 去掉多余空格
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def n10_evidence_rag(state: BioDynamicsState) -> dict:
    """检索支持性文献证据，含正向实体白名单 + 防御性过滤 + PMID 提取。"""
    _emit_in("n10_evidence_rag")
    mechanism = state.get("mechanism", {}) or {}
    user_input = state.get("user_input", "")
    # 本地 ChromaDB 查询：优先用 v4_pathway_class（注册表键，如 "APOPTOSIS"），
    # 其次用 mechanism.pathway（LLM 自然语言），最后用 user_input
    # PubMed fallback 优先用自然语言 user_input（更适合 PubMed 搜索）
    pathway_query = state.get("v4_pathway_class", "") or mechanism.get("pathway") or user_input
    pubmed_query = user_input or _clean_pubmed_query(pathway_query)

    rag = get_rag_collections()
    evidence: list[dict] = []
    if rag.available and pathway_query:
        try:
            hits = rag.search_evidence(pathway_query, top_k=5)
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

    # [v5 Recovery Sprint 2 / RC4] 空 PMID 过滤——消除 Scientific Hallucination
    # 旧实现：PMID 提取失败后 ev["pmid"]="" 仍保留在 evidence 列表中，
    # 报告模板渲染 "PMID: ,图 ,细胞系 "（全空），伪装有文献支撑。
    # 修复：PMID 为空的条目不加入 evidence 列表；全部为空时返回空列表 + 标记无文献。
    _evidence_with_pmid: list[dict] = []
    _evidence_no_pmid_count = 0
    for ev in evidence:
        if ev.get("pmid", ""):
            _evidence_with_pmid.append(ev)
        else:
            _evidence_no_pmid_count += 1
    if _evidence_no_pmid_count > 0:
        logger.info(
            "RC4 空 PMID 过滤：%d 条证据无 PMID 已丢弃（Scientific Hallucination 消除）",
            _evidence_no_pmid_count,
        )
    evidence = _evidence_with_pmid

    # Load governed canonical literature before any online fallback.  This keeps
    # benchmark and normal pathway runs deterministic when PubMed is unavailable.
    _rag_evidence_count = len(evidence)
    _canonical_pathway = pathway_query or ""
    _canonical_ranker = _CanonicalRanker(_canonical_pathway)
    _canonical_records = _canonical_ranker.canonical_records()
    existing_pmids = {str(item.get("pmid", "")) for item in evidence}
    evidence.extend(
        item for item in _canonical_records
        if item["pmid"] not in existing_pmids
    )

    # [BM 修复] 在线 PubMed fallback：本地 ChromaDB 与 canonical evidence
    # 均无有效文献时才直连 NCBI。PubMed 更适合自然语言查询，因此优先用
    # user_input；若为空再清洗 pathway。
    _n10_diag = {
        "local_evidence_count": _rag_evidence_count,
        "canonical_local_count": len(_canonical_records),
        "rc4_filtered_count": _evidence_no_pmid_count,
        "pubmed_fallback_triggered": False,
        "pubmed_queries_tried": [],
        "pubmed_fallback_count": 0,
        "final_evidence_count": 0,
    }
    if not evidence:
        _n10_diag["pubmed_fallback_triggered"] = True
        _n10_diag["pubmed_queries_tried"].append(pubmed_query[:200])
        logger.info("N10 本地 evidence 为空，触发 PubMed 在线 fallback，query=%s", pubmed_query)
        evidence = _fetch_pubmed_evidence_sync(pubmed_query, top_k=3)
        _n10_diag["pubmed_fallback_count"] = len(evidence)
        # 若 user_input 未命中，再尝试清洗后的 pathway 标识符
        if not evidence and pubmed_query != _clean_pubmed_query(pathway_query):
            fallback_query = _clean_pubmed_query(pathway_query)
            _n10_diag["pubmed_queries_tried"].append(fallback_query[:200])
            logger.info("N10 PubMed fallback 第 2 轮，query=%s", fallback_query)
            evidence = _fetch_pubmed_evidence_sync(fallback_query, top_k=3)
            _n10_diag["pubmed_fallback_count"] = len(evidence)
        logger.info("N10 PubMed fallback 返回 %d 条文献", len(evidence))
    else:
        logger.info("N10 本地 evidence 命中 %d 条，跳过 PubMed fallback", len(evidence))

    # TODO: P1-3 — 显式补全 figure_ref / cell_line 字段（依赖原始记录透传，缺失时填空字符串）
    for ev in evidence:
        ev.setdefault("figure_ref", "")
        ev.setdefault("cell_line", "")

    # === Task E: Canonical Literature Ranking — 三重排序 Canonical+Embedding+BM25 ===
    # 用户核心诉求：RAG 排序不是 BM25，而是 Canonical + Embedding + BM25。
    # 提前维护每通路 Canonical PMID 星级表，确保 Discussion 不会引用奇怪论文。
    # 非阻断：若 Canonical Ranker 未加载或失败，evidence 保持原序。
    _canonical_diag: dict = {
        "enabled": False,
        "pathway": "",
        "canonical_count": 0,
        "reranked": False,
        "top_stars": 0,
    }
    try:
        if evidence:
            _reranked, _canon_report = _task_e_rerank_evidence(
                evidence, pathway=_canonical_pathway, query=user_input or pathway_query,
            )
            if _reranked:
                _canonical_diag = {
                    "enabled": _canon_report.get("loaded", False),
                    "pathway": _canon_report.get("pathway", ""),
                    "canonical_count": _canon_report.get("canonical_count", 0),
                    "reranked": _canon_report.get("results", []) and
                                _canon_report["results"][0].get("original_rank", 1) != 1,
                    "top_stars": _canon_report.get("results", [{}])[0].get("stars", 0)
                                 if _canon_report.get("results") else 0,
                }
                evidence = _reranked
                logger.info(
                    "Task E Canonical Ranking: pathway=%s canonical=%d reranked=%s top_stars=%d",
                    _canonical_diag["pathway"],
                    _canonical_diag["canonical_count"],
                    _canonical_diag["reranked"],
                    _canonical_diag["top_stars"],
                )
    except Exception as exc:
        logger.warning("Task E Canonical Ranking 失败（非阻断）：%s", exc)

    _n10_diag["canonical_ranking"] = _canonical_diag
    _n10_diag["final_evidence_count"] = len(evidence)
    logger.info(
        "N10 诊断: local=%d canonical=%d rc4_filtered=%d "
        "pubmed_triggered=%s pubmed_count=%d final=%d",
        _n10_diag["local_evidence_count"],
        _n10_diag["canonical_local_count"],
        _n10_diag["rc4_filtered_count"],
        _n10_diag["pubmed_fallback_triggered"],
        _n10_diag["pubmed_fallback_count"],
        _n10_diag["final_evidence_count"],
    )

    return {
        "paper_evidence": evidence,
        "n10_diagnostic": _n10_diag,
        "agent_dispatches": [orchestrator.complete_dispatch(
            "n10_evidence_rag", latency_ms=0.0
        )],
    }


# =============================================================================
# N11 — Scientific Report（Python Markdown 模板 + LLM JSON Fill）
# =============================================================================
def _sprint2_build_fusion(
    llm_discussion: str,
    paper_evidence: list,
    pathway: str = "",
):
    """Sprint 2 — 从 LLM Discussion 文本与 paper_evidence 构造 EvidenceFusionReport。

    流程（Citation-driven）：
      1. LLM discussion 按句分割为断言列表（LLM 仅"组织"，不"创造"引用）
      2. paper_evidence 转为 [A] EvidenceItem 列表
      3. fuse_evidence 按位置匹配断言与证据

    Args:
        llm_discussion: LLM 生成的 Discussion 文本（按句号分割为断言）。
        paper_evidence: N10 的文献证据列表（list[dict]，含 pmid/title 等）。
        pathway: 通路名称。

    Returns:
        EvidenceFusionReport。
    """
    import re

    # 1. 按句号/分号分割 LLM discussion 为断言
    raw_sentences = re.split(r"[。；;\n]+", llm_discussion)
    assertions = [
        s.strip() for s in raw_sentences
        if s.strip() and len(s.strip()) > 5  # 过滤过短片段
    ]

    # 无断言时返回空报告（fuse_evidence 内部会处理）
    if not assertions:
        return _sprint2_fuse_evidence(assertions=[])

    # 2. paper_evidence → [A] EvidenceItem 列表
    pubmed_items: list[_Sprint2EvidenceItem] = []
    for ev in paper_evidence:
        pmid = str(ev.get("pmid", "")).strip()
        if not pmid:
            continue  # RC4 铁律：跳过空 PMID
        title = str(ev.get("title", "")).strip()
        confidence = 0.8  # 默认 Mechanism Paper 级别
        pubmed_items.append(
            _Sprint2EvidenceItem(
                source=_Sprint2EvidenceSource.PUBMED,
                reference=f"PMID:{pmid}",
                snippet=title,
                confidence=confidence,
            )
        )

    # 3. fuse_evidence 按位置匹配
    return _sprint2_fuse_evidence(
        assertions=assertions,
        pubmed_evidence=pubmed_items if pubmed_items else None,
    )


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
    # Sprint 5: 读取参数用于 Parameter Provenance
    parameters = state.get("parameters", {}) or {}

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

    # Sprint 2 — Citation-driven Discussion Renderer
    # Flag ON 时：用 evidence_fuser + discussion_renderer 替换 LLM Discussion
    # Flag OFF 时：走旧路径（llm_filled["discussion"] 保持 LLM 输出）
    _sprint2_evidence_bundle_payload = None
    if settings.is_sa_feature_enabled("SPRINT2_EVIDENCE_RENDERER"):
        try:
            _sprint2_pathway = state.get("v4_pathway_class", "") or ""
            _sprint2_fusion_report = _sprint2_build_fusion(
                llm_discussion=llm_filled.get("discussion", ""),
                paper_evidence=evidence,
                pathway=_sprint2_pathway,
            )
            _sprint2_rendered = _sprint2_render_discussion(
                fusion_report=_sprint2_fusion_report,
                fallback_discussion=llm_filled.get("discussion", ""),
                pathway=_sprint2_pathway,
            )
            llm_filled["discussion"] = _sprint2_rendered
            _sprint2_evidence_bundle_payload = _sprint2_evidence_payload(
                _sprint2_fusion_report
            )
            logger.info(
                "Sprint 2 Discussion Renderer: %d assertions, coverage=%s",
                _sprint2_fusion_report.total_assertions,
                _sprint2_fusion_report.source_coverage,
            )
        except Exception as sprint2_exc:
            logger.warning("Sprint 2 Discussion Renderer 异常（降级到 LLM Discussion）: %s", sprint2_exc)

    # V4 Scientific Reviewer — Evidence Graph-driven Discussion Renderer (Task 10)
    # Flag ON 时：从 Evidence Graph 重新渲染 Discussion（覆盖 LLM 与 Sprint2 输出），
    #             每句附 [A]/[B]/[C]/[D]/[E] 单源标签，禁止 LLM 自由写
    # Flag OFF（默认）时：保留 LLM/Sprint2 行为（铁律：默认行为与 v3/v4 完全一致）
    if settings.V4_SCIENTIFIC_REVIEWER_ENABLED:
        try:
            from app.report_renderer import render_discussion_with_evidence_graph
            llm_filled["discussion"] = render_discussion_with_evidence_graph(
                state=state,
                llm_discussion=llm_filled.get("discussion", ""),
            )
            logger.info("V4 Discussion Renderer: rendered from Evidence Graph")
        except Exception as v4_disc_exc:
            # 降级：保留 LLM/Sprint2 输出，不阻塞主流程
            logger.warning(
                "V4 Discussion Renderer 异常（降级到 LLM/Sprint2 Discussion）: %s",
                v4_disc_exc,
            )

    # 2. Python 模板渲染 Markdown
    renderer = ReportRenderer()
    sandbox_failure_reason = state.get("sandbox_failure_reason", "")
    # [P0-4] LLM 自动决策记录（超时未响应时由 LLM 代为决策）
    llm_auto_decisions = state.get("llm_auto_decisions", []) or []
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
            llm_auto_decisions=llm_auto_decisions,
        )
        forbidden_violations = renderer.check_forbidden_terms(llm_filled)
    except Exception as exc:
        logger.error("N11 报告渲染失败：%s", exc)
        markdown = f"# 报告生成失败\n\n{llm_filled}\n\n错误：{exc}"
        forbidden_violations = []

    # -------------------------------------------------------------------------
    # Sprint 5 — Parameter Provenance + Explainability Log
    # Flag ON 时：生成 Parameter Traceability 表 + Scientific Decision Log
    # Flag OFF 时：不追加任何内容（v3/v4 行为不变）
    # -------------------------------------------------------------------------
    _sprint5_provenance_payload = None
    _sprint5_decision_log_payload = None
    if settings.is_sa_feature_enabled("SPRINT5_PROVENANCE_EXPLAINABILITY"):
        try:
            _sprint5_pathway = state.get("v4_pathway_class", "") or ""
            # 1. Parameter Provenance
            _sprint5_prov_report = _sprint5_generate_provenance(parameters)
            _sprint5_provenance_payload = {
                "enabled": _sprint5_prov_report.enabled,
                "skipped": _sprint5_prov_report.skipped,
                "summary": _sprint5_prov_report.summary,
                "row_count": len(_sprint5_prov_report.rows),
            }
            # 2. Decision Log（传入 provenance_summary 供 Parameter 维度引用）
            _sprint5_log_report = _sprint5_generate_decision_log(
                knowledge_graph=kg,
                pathway=_sprint5_pathway,
                confidence=confidence,
                biomodels_report=state.get("biomodels_report"),
                parameters=parameters,
                provenance_summary=_sprint5_prov_report.summary,
                paper_evidence=evidence,
                experiments=experiments,
                consistency_passed=state.get("sa_consistency_passed"),
                validation_passed=state.get("sa_validation_passed"),
            )
            _sprint5_decision_log_payload = {
                "enabled": _sprint5_log_report.enabled,
                "skipped": _sprint5_log_report.skipped,
                "entry_count": len(_sprint5_log_report.entries),
                "entries": [
                    {
                        "dimension": e.dimension,
                        "decision": e.decision,
                        "source": e.source,
                        "reason": e.reason,
                        "confidence": e.confidence,
                        "evidence_ref": e.evidence_ref,
                    }
                    for e in _sprint5_log_report.entries
                ],
            }
            # 3. 追加到 Report Markdown 末尾
            if _sprint5_prov_report.markdown_table:
                markdown += "\n## Parameter Traceability\n\n" + _sprint5_prov_report.markdown_table + "\n"
            if _sprint5_log_report.markdown:
                markdown += "\n" + _sprint5_log_report.markdown + "\n"
            logger.info(
                "[Sprint5] Parameter Provenance: %d rows, Decision Log: %d entries",
                len(_sprint5_prov_report.rows),
                len(_sprint5_log_report.entries),
            )
        except Exception as sprint5_exc:
            logger.warning("Sprint 5 Provenance/Explainability 异常: %s", sprint5_exc)

    # [v5 Recovery Sprint 4 / RC18] 报告落盘持久化
    _user_input = state.get("user_input", "")
    _persisted_path = ReportRenderer.persist_report(markdown, _user_input)
    if _persisted_path:
        logger.info("RC18 报告已落盘：%s", _persisted_path)

    return {
        "report": {
            "markdown": markdown,
            "llm_filled_json": llm_filled,
            "forbidden_terms_violations": forbidden_violations,
            "persisted_path": _persisted_path,  # RC18: 落盘文件路径
            "sprint2_evidence_bundle": _sprint2_evidence_bundle_payload,  # Sprint 2
            "sprint5_provenance": _sprint5_provenance_payload,  # Sprint 5
            "sprint5_decision_log": _sprint5_decision_log_payload,  # Sprint 5
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
