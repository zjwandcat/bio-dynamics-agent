# BioDynamics Agent v3 - Supervisor-Worker 动态编排图
# 取代 v1/v2 硬编码线性图，支持三档运行模式、人在环路、上下文压缩。

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from app.config import llm
from app.context_compressor import compress_worker_output
from app.nodes import (
    node0_mcp_term_lookup,
    node1_6_pkpd_inference,
    node1_parse_network,
    node2_generate_code,
    node4_audit_and_correct,
    _normalize_network_json,
)
from app.nodes_v2 import (
    n0_sbml_loader,
    n10_evidence_rag,
    n11_scientific_report,
    n1_ner_entity_normalize,
    n2_mechanistic_planner,
    n3_mechanism_rag,
    n4_kg_builder,
    n5_parameter_rag,
    n6_ode_generator,
    n8_scientific_features,
    n9_experiment_rag,
)
# v4 迁移 Phase 1：Ontology Agent hook（仅新增 import，不修改任何 v3 导入）
# V4_ONTOLOGY_AGENT_ENABLED=false 时 hook 节点直接返回空 dict，行为同 v3
from app.ontology.ontology_agent import ontology_hook_node
# v4 迁移 Phase 4：P4 Pathway Planner + Specialist + Cross-talk Coordinator hooks
# 三个 hook 在 worker_mechanism 后串联注入（最小侵入，不改路由）
# V4_PATHWAY_PLANNER_ENABLED / V4_PATHWAY_SPECIALIST_ENABLED /
# V4_CROSSTALK_COORDINATOR_ENABLED 均默认 false，flag=false 时 hook 返回 {}
from app.pathways.pathway_planner import pathway_planner_hook_node
from app.pathways.specialist_hook import specialist_hook_node
from app.crosstalk.coordinator import crosstalk_coordinator_hook_node
# v4 迁移 Phase 5：SBML Grounder + Validation Pyramid hooks
# _sbml_grounder_hook 在 worker_ode 后串联注入（建立五级映射链）
# _validation_pyramid_hook 在 worker_validator 后串联注入（编排 5 层验证）
# V4_SBML_GROUNDER_ENABLED / V4_VALIDATION_PYRAMID_ENABLED 均默认 false，
# flag=false 时 hook 返回 {}，行为同 v3
from app.sbml_grounder.grounder_agent import sbml_grounder_hook_node
from app.validation_v2.validation_agent import validation_pyramid_hook_node
# v4 迁移 Phase 6：Hypothesis Agent + Dynamic Router hooks（P6 / Task 6.7）
# _hypothesis_agent_hook: 在 worker_report 前注入，生成 v4_hypothesis_list
# _dynamic_router_hook: 在路由层注入，记录 13 Agent 调度（v4_agent_dispatches）
# V4_HYPOTHESIS_AGENT_ENABLED / V4_DYNAMIC_ROUTING_ENABLED 均默认 false，
# flag=false 时 hook 返回 {}，行为同 v3
from app.hypothesis.hypothesis_agent import hypothesis_agent_hook_node
from app.agent_orchestration.dynamic_router import dynamic_router_hook_node
# v4 迁移 Phase 5：Calibration Agent + Sensitivity Analyzer hooks（Task D.2 / G2）
# _calibration_hook: 在 _validation_pyramid_hook 后注入，参数校准 + 置信区间
# _sensitivity_hook: 在 _calibration_hook 后注入，local + sobol + morris 灵敏度分析
# V4_CALIBRATION_AGENT_ENABLED=false 时两个 hook 均返回 {}，行为同 v3
from app.calibration.calibration_agent import calibration_hook_node
from app.sensitivity.sensitivity_analyzer import sensitivity_hook_node
# v4 迁移 Phase 2：Reaction IR v2 + Adapter hook
# V4_REACTION_IR_ENABLED=false 时 hook 直接返回空 dict，行为同 v3
# V4_REACTION_IR_ADAPTER_ENABLED=true 时通过 Adapter 同步 network_json
from app.adapters.adapter_registry import get_adapter_registry
from app.config import settings as _v4_settings
from app.reaction_ir_v2.reaction_builder import build_from_network_json
from app.sandbox import ERR_NONE, execute_simulation_code_v2
from app.state import BioDynamicsState, normalize_v4_state, set_v4_state
from app.supervisor import orchestrator

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 人在环路全局事件：用于 clarification_node 阻塞等待前端 respond / stop
# -----------------------------------------------------------------------------
_clarification_events: dict[str, asyncio.Event] = {}
_clarification_responses: dict[str, dict[str, Any]] = {}
_clarification_stop_events: dict[str, asyncio.Event] = {}

_CLARIFICATION_TIMEOUT_SECONDS = 120  # [P0-4] 2 分钟无响应则自动选择 A（旧值 600s 太长）


def get_clarification_event(thread_id: str) -> asyncio.Event:
    """获取指定 thread_id 的 clarification 响应事件。"""
    if thread_id not in _clarification_events:
        _clarification_events[thread_id] = asyncio.Event()
    return _clarification_events[thread_id]


def get_clarification_stop_event(thread_id: str) -> asyncio.Event:
    """获取指定 thread_id 的停止事件。"""
    if thread_id not in _clarification_stop_events:
        _clarification_stop_events[thread_id] = asyncio.Event()
    return _clarification_stop_events[thread_id]


def set_clarification_response(thread_id: str, response: dict[str, Any]) -> None:
    """外部接口：回灌用户干预答案。"""
    _clarification_responses[thread_id] = response
    get_clarification_event(thread_id).set()


def set_clarification_stop(thread_id: str) -> None:
    """外部接口：请求终止当前 clarification 等待。"""
    get_clarification_stop_event(thread_id).set()


def cleanup_clarification_events(thread_id: str) -> None:
    """清理指定 thread_id 的临时事件对象。"""
    _clarification_events.pop(thread_id, None)
    _clarification_responses.pop(thread_id, None)
    _clarification_stop_events.pop(thread_id, None)


# v3 支持的 Worker 名称
WORKER_NAMES = [
    "worker_mcp",
    "worker_mechanism",
    "worker_rag",
    "worker_pkpd",
    "worker_ode",
    "worker_sandbox",
    "worker_validator",  # P0-4 新增：SBML Validator（修复提示词1.md §二.6）
    "worker_report",
]

# Manual 模式模块键到 Worker 的映射（与前端 ControlBar 约定一致）
_MODULE_TO_WORKER: dict[str, str] = {
    "terminology_mcp": "worker_mcp",
    "mechanism_graph": "worker_mechanism",
    "mechanism_parameter_rag": "worker_rag",
    "pkpd_inference": "worker_pkpd",
    "sandbox_execute": "worker_sandbox",
    "sbml_validation": "worker_validator",  # P0-4 新增
    "dose_analysis": "worker_ode",  # 剂量分析主要落在 ODE 模板渲染
    "experiment_evidence_rag": "worker_report",
    "report_generation": "worker_report",
}

# 默认完整执行计划（Auto Standard 无裁剪时的默认链路）
_FULL_PLAN = [
    "worker_mcp",
    "worker_mechanism",
    "worker_rag",
    "worker_pkpd",
    "worker_ode",
    "worker_sandbox",
    "worker_validator",  # P0-4 新增：sandbox 后做 SBML 验证
    "worker_report",
]


# -----------------------------------------------------------------------------
# 结构化输出：PreRouter 决策
# -----------------------------------------------------------------------------
class _PreRouterOutput(BaseModel):
    needs_pkpd: bool = Field(description="用户问题是否涉及药物/PKPD/剂量推断")
    needs_experiment_evidence: bool = Field(description="是否需要实验方案与文献证据检索")
    reasoning: str = Field(default="", description="裁剪决策理由")


class _ModelingAmbiguityOutput(BaseModel):
    has_ambiguity: bool = Field(default=False, description="是否存在两种合理建模方式")
    ambiguity_description: str = Field(default="", description="分歧描述")
    option_a: str = Field(default="", description="方案 A 标签")
    option_b: str = Field(default="", description="方案 B 标签")


# -----------------------------------------------------------------------------
# 通用工具函数
# -----------------------------------------------------------------------------
def _merge_node_output(state: BioDynamicsState, output: dict[str, Any]) -> BioDynamicsState:
    """将 Worker 子节点的产出合并回 state（浅合并，遵守 TypedDict 语义）。"""
    merged = dict(state)
    merged.update(output)
    return merged  # type: ignore[return-value]


def _now_ms() -> float:
    return time.time() * 1000


def _dispatch_for_v3_worker(worker_name: str, status: str = "in_progress", reasoning: str = "") -> dict:
    """为 v3 Worker 生成 agent_dispatch 事件。"""
    label_map: dict[str, str] = {
        "worker_mcp": "MCP 术语标准化",
        "worker_mechanism": "机制解析与图谱",
        "worker_rag": "知识检索 (RAG)",
        "worker_pkpd": "PK/PD 推断",
        "worker_ode": "ODE 方程生成",
        "worker_sandbox": "沙箱仿真执行",
        "worker_validator": "SBML 验证",  # P0-4 新增
        "worker_report": "预测报告生成",
    }
    return orchestrator.emit_dispatch(
        target_agent=label_map.get(worker_name, worker_name),
        reasoning=reasoning or f"Supervisor dispatched: {worker_name}",
        status=status,
        node_name=worker_name,
    )


# -----------------------------------------------------------------------------
# PreRouter：生成 execution_plan
# -----------------------------------------------------------------------------
def pre_router(state: BioDynamicsState) -> dict[str, Any]:
    """根据运行模式生成 execution_plan。"""
    mode = state.get("mode", "auto_standard")
    manual_modules = state.get("manual_modules", []) or []
    user_input = state.get("user_input", "")

    dispatches: list[dict] = []
    dispatches.append(_dispatch_for_v3_worker("pre_router", "in_progress", f"PreRouter: mode={mode}"))

    if mode == "manual":
        plan = _build_manual_plan(manual_modules)
        reasoning = "Manual 模式：按用户勾选模块生成执行计划"
    elif mode == "auto_fast":
        # TODO: P1-5 — Auto Fast plan 补 worker_rag，确保 parameters 非空
        # 起因：原 Fast plan 跳过 worker_rag，导致 ODE 只能用硬编码默认 Kd=10.0。
        # Fast 模式的 worker_rag 仅做最小参数占位（FAST_MODE_ESTIMATED），不调用在线检索，
        # 但能产出 parameters 字典供 N6 Simple 模板读取，避免下游字段缺失。
        plan = ["worker_mechanism", "worker_rag", "worker_ode", "worker_sandbox", "worker_validator", "worker_report"]
        reasoning = "Auto Fast：极简链路，含最小 RAG 占位与 SBML 验证，跳过 PKPD/证据检索"
    else:
        # auto_standard：用 LLM 判断 query 是否需要 PKPD 与实验证据
        plan, reasoning = _build_standard_plan(user_input)

    dispatches.append(_dispatch_for_v3_worker("pre_router", "completed", reasoning))

    return {
        "execution_plan": plan,
        "current_step": 0,
        "completed_workers": [],
        "pending_clarification": None,
        "clarification_response": None,
        "clarification_request": None,
        "worker_results": {},
        "agent_dispatches": dispatches,
        "messages": [f"执行计划：{' -> '.join(plan)}"],
        # 双保险：PreRouter 显式清空结构化数据字段，防止跨请求残留
        "network_json": {},
        "mcp_term_definitions": [],
        "mcp_tool_calls": [],
        "raw_cache": {},
        "drug_candidates": [],
        "simulation_csv_path": "",
        "rag_retrieved_params": [],
        "rag_selected_params": {},
        "rag_fallback": False,
        "rag_summary": "",
        "rag_hit_rate": 0.0,
        "species_context": "",
        "pkpd_profile": {},
        "drug_regimen": [],
        "dose_response_data": {},
        "parameters": {},
        "ode_model": {},
        "entities": [],
        "mechanism": {},
        "knowledge_graph": {},
        "metrics": {},
        "feature_metadata": {},
        "confidence": 0.0,
        "experiment_protocols": [],
        "paper_evidence": [],
    }


def _build_manual_plan(modules: list[str]) -> list[str]:
    """按用户勾选的模块组生成计划，并自动补全依赖。

    依赖规则：
    - report_generation -> 自动补全 sandbox_execute + mechanism_graph
    - sandbox_execute -> 自动补全 ode（沙箱需要代码）
    - TODO: P1-5 — ode -> 自动补全 mechanism_graph + rag（ODE 需要 network_json + parameters）
    - 最终按标准拓扑顺序排列：mcp -> mechanism -> rag -> pkpd -> ode -> sandbox -> report
    """
    # 收集用户显式勾选的 worker
    selected: set[str] = set()
    for m in modules:
        w = _MODULE_TO_WORKER.get(m)
        if w:
            selected.add(w)

    # 依赖补全
    if "worker_report" in selected:
        selected.add("worker_sandbox")
        selected.add("worker_validator")  # P0-4：报告依赖 SBML 验证结果
        selected.add("worker_mechanism")
    if "worker_validator" in selected:
        # SBML 验证依赖沙箱仿真 CSV
        selected.add("worker_sandbox")
    if "worker_sandbox" in selected:
        selected.add("worker_ode")
    # TODO: P1-5 — ODE 依赖 mechanism_graph（提供 network_json）和 rag（提供 parameters）
    # 起因：用户只勾选"剂量分析"(worker_ode) 时，缺少 mechanism_graph 会导致 network_json 为空，
    # N6 无法渲染 ODE；缺少 rag 会导致 parameters 为空，Kd 退化为硬编码 10.0。
    if "worker_ode" in selected:
        selected.add("worker_mechanism")
        selected.add("worker_rag")

    # 按 _FULL_PLAN 的拓扑顺序过滤，确保执行顺序正确
    plan = [w for w in _FULL_PLAN if w in selected]
    return plan if plan else ["worker_mechanism", "worker_ode", "worker_sandbox"]


def _rule_based_pkpd_check(user_input: str) -> tuple[bool, str]:
    """基于规则的轻量 PK/PD 需求检测：先于 LLM 调用，避免模型不可用时漏判。

    命中任一药物/剂量/浓度/PK-PD 关键词 → 视为需要 PK/PD；
    若同时缺少"信号通路/pathway"等机制描述词，倾向仍然需要 PK/PD；
    关键词全部为机制/通路/调控类且无任何药物痕迹 → 视为纯机制问题，跳过 PK/PD。
    """
    text = (user_input or "").lower()
    # 强药物信号：含这些词基本一定涉及 PK/PD
    pkpd_strong_keywords = [
        "pk/pd", "pkpd", "pharmacokinetic", "pharmacodynamic",
        "剂量", "dose", "dosing", "dosage", "浓度", "concentration",
        "ec50", "ic50", "ic90", "emic50", "emax", "hill coefficient",
        "给药", "administer", "administration", "给药途径", "route",
        "清除率", "clearance", "半衰期", "half-life", "auc", "cmax", "tmax",
        "抑制剂", "inhibitor", "激动剂", "agonist", "拮抗剂", "antagonist",
        "化合物", "compound", "药物", "drug", "small molecule",
    ]
    has_strong_drug = False
    for kw in pkpd_strong_keywords:
        if kw in text:
            has_strong_drug = True
            return True, f"命中强药物关键词：{kw}"

    # 纯机制信号：含这些词倾向于跳过 PK/PD
    pure_mechanism_keywords = [
        "信号通路", "signaling pathway", "pathway", "调控", "regulation",
        "激活", "activation", "activate", "activates", "activated",
        "inhibit", "抑制", "促进", "phosphorylat", "下调", "上调",
        "downstream", "upstream", "cascade", "级联",
        # 扩充：凋亡/细胞周期/信号阈值等纯机制术语（英文 BM 输入常见）
        "apoptosis", "mitochondrial", "threshold", "exceeds",
        "bax", "caspase", "bcl-2", "bcl2", "cytochrome",
        "momp", "mitochondria", "outer membrane",
        "dna damage", "p53", "mdm2",
        "nf-kb", "nf-κb", "tnf", "ikk",
        "wnt", "beta-catenin", "gsk3",
        "stat", "jak", "interleukin", "il-6", "il6",
        "egfr", "egf", "erk", "mapk", "pi3k", "akt", "mtor", "pten",
    ]
    has_mech = any(kw in text for kw in pure_mechanism_keywords)
    if has_mech and not has_strong_drug:
        return False, "纯机制问题（无药物/剂量关键词，含信号通路/调控关键词）"

    return True, "默认包含 PK/PD（规则无法判定）"


_RAG_TRIGGER_KEYWORDS = [
    "文献", "研究", "论文", "paper", "literature", "study", "studies",
    "证据", "evidence", "实验数据", "experimental", "clinical trial",
]


def _rule_based_rag_check(user_input: str) -> bool:
    """基于规则的文献检索需求判定：命中关键词时强制触发在线补充。

    与 _rule_based_pkpd_check 并列，用于在 worker_rag 中决定是否
    跳过本地命中率阈值，直接调用 bio_db_client + PubMed 在线检索。
    """
    text = (user_input or "").lower()
    return any(kw in text for kw in _RAG_TRIGGER_KEYWORDS)


def _build_standard_plan(user_input: str) -> tuple[list[str], str]:
    """Auto Standard 模式：先用规则做轻量判断，必要时再调用 LLM。"""
    # 1) 规则优先：覆盖 LLM 不可用 / JSON 解析失败 / 输出异常等所有回退场景
    rule_needs_pkpd, rule_reason = _rule_based_pkpd_check(user_input)
    needs_pkpd = rule_needs_pkpd
    needs_evidence = True
    reasoning = rule_reason

    # 2) 规则无法明确判断时再调用 LLM
    uncertain = (rule_reason == "默认包含 PK/PD（规则无法判定）")
    if uncertain:
        try:
            structured_llm = llm.with_structured_output(_PreRouterOutput)
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "你是一位生物医学建模规划助手。请判断用户问题是否需要药物动力学(PK/PD)推断、剂量递增分析，以及是否需要实验方案与文献证据检索。",
                    ),
                    ("human", "用户问题：{user_input}"),
                ]
            )
            chain = prompt.partial(user_input=user_input) | structured_llm
            result: _PreRouterOutput = chain.invoke({})
            needs_pkpd = result.needs_pkpd
            needs_evidence = result.needs_experiment_evidence
            reasoning = result.reasoning or (
                f"{'含药物/PKPD相关描述' if needs_pkpd else '纯机制问题'}；"
                f"{'需要实验证据' if needs_evidence else '跳过实验证据检索'}"
            )
        except Exception as exc:
            logger.warning("PreRouter LLM 决策失败，使用规则结果（needs_pkpd=%s）：%s", needs_pkpd, exc)
            # 异常时保留规则判定结果，不回退到完整计划
            reasoning = f"{rule_reason}（LLM 异常，保留规则判定）"

    plan = ["worker_mcp", "worker_mechanism", "worker_rag"]
    if needs_pkpd:
        plan.append("worker_pkpd")
    plan.extend(["worker_ode", "worker_sandbox", "worker_validator"])  # P0-4：sandbox 后做 SBML 验证
    if needs_evidence:
        plan.append("worker_report")
    else:
        plan.append("worker_report")  # 报告总是需要，只是可能不检索实验证据

    return plan, reasoning


# -----------------------------------------------------------------------------
# Supervisor：动态调度 + 人在环路触发
# -----------------------------------------------------------------------------
def supervisor(state: BioDynamicsState) -> dict[str, Any]:
    """Supervisor 节点：决定下一步 worker 或触发人工干预。"""
    mode = state.get("mode", "auto_standard")
    plan = state.get("execution_plan", []) or []
    current_step = state.get("current_step", 0)
    pending = state.get("pending_clarification")
    response = state.get("clarification_response")

    # 用户请求停止：直接结束
    if state.get("stop_requested"):
        return {
            "messages": ["Supervisor：收到停止请求，终止工作流。"],
        }

    # 1. 处理已回灌的人工干预答案
    if pending is not None and response is not None:
        return {
            "pending_clarification": None,
            "clarification_response": None,
            "clarification_request": None,
            "clarification_resolved": True,
            "messages": [f"已采纳用户干预：{response.get('selected_option', '')}"],
        }

    # 2. 若已有未处理的 clarification 请求，继续挂起
    if pending is not None:
        return {
            "clarification_request": pending,
        }

    # 3. 检查是否完成全部计划
    if current_step >= len(plan):
        return {
            "messages": ["Supervisor：所有计划节点已完成，结束工作流。"],
        }

    # 4. 检查当前是否需要触发人工干预
    next_worker = plan[current_step]
    clarification = _check_clarification_triggers(state, next_worker, mode)
    if clarification:
        if state.get("benchmark_run"):
            options = list(clarification.get("options") or [])
            selected = next(
                (item for item in options if "推荐" in str(item.get("label", ""))),
                options[0] if options else {"id": "A", "label": "自动继续"},
            )
            auto_response = {
                "selected_option": str(selected.get("id", "A")),
                "selected_label": str(selected.get("label", "自动继续")),
                "free_text": "",
                "context": str(clarification.get("context", "")),
                "llm_reasoning": "Non-interactive benchmark selected the recommended deterministic option.",
            }
            decision_record = {
                "context": clarification.get("context", ""),
                "question": clarification.get("question", ""),
                "selected_option": auto_response["selected_option"],
                "selected_label": auto_response["selected_label"],
                "llm_reasoning": auto_response["llm_reasoning"],
                "warning": "Deterministic non-interactive benchmark decision",
            }
            logger.info(
                "Benchmark clarification auto-resolved: context=%s option=%s",
                auto_response["context"],
                auto_response["selected_option"],
            )
            return {
                "next_worker": next_worker,
                "pending_clarification": None,
                "clarification_request": None,
                "clarification_response": auto_response,
                "clarification_resolved": True,
                "llm_auto_decisions": [decision_record],
                "messages": [f"Benchmark 自动决策：{auto_response['selected_label']}"],
            }
        # 清理该 thread 可能残留的旧的 clarification_response，避免新旧问题串扰
        thread_id = state.get("thread_id", "unknown")
        _clarification_responses.pop(thread_id, None)
        return {
            "pending_clarification": clarification,
            "clarification_request": clarification,
            "messages": [f"需要人工干预：{clarification.get('question', '')}"],
        }

    # 5. 正常调度：返回 next_worker（由条件边路由）
    return {
        "next_worker": next_worker,
        "messages": [f"Supervisor 调度：{next_worker}"],
    }


def _llm_auto_decide_clarification(state: BioDynamicsState, pending: dict) -> dict:
    """[P0-4] clarification 超时时由 LLM 自动决策最佳选项。

    基于用户假说、clarification 问题与选项，让 LLM 选择最科学合理的方案。
    保持客观真实性：LLM 仅从已有选项中选择，不创造新选项。

    Returns:
        clarification_response dict，含 selected_option / selected_label /
        free_text=None / context / llm_reasoning。
    """
    user_input = state.get("user_input", "")
    question = pending.get("question", "")
    options = pending.get("options", [])
    context = pending.get("context", "")

    # 构建选项文本
    options_text = "\n".join(
        [f"  {opt.get('id', '?')}: {opt.get('label', '')}" for opt in options]
    )

    # 不同 context 的科学偏好提示（帮助 LLM 做出更符合生物学的选择）
    context_hints = {
        "parameter_missing": "参数缺失时，优先选择能保持仿真完成的选项（通常是继续用估算值）",
        "biological_contradiction": "反馈环路在生物系统中常见且重要（如 MAPK 负反馈），优先保留反馈拓扑",
        "modeling_ambiguity": "无药物浓度数据时，线性抑制比 Emax 更稳健",
    }
    hint = context_hints.get(context, "")

    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "你是计算系统生物学专家。用户未在超时内回答建模决策问题，"
             "请基于用户假说与生物学常识选择最科学合理的选项。"
             "仅从给定选项中选择，不创造新选项。输出严格 JSON。"),
            ("human",
             "用户假说：{user_input}\n\n"
             "决策问题：{question}\n\n"
             "可选选项：\n{options}\n\n"
             "科学提示：{hint}\n\n"
             "请输出 JSON：{{\"selected_option\": \"A/B/C\", \"reasoning\": \"选择理由（1句）\"}}"),
        ])
        chain = prompt.partial(
            user_input=user_input, question=question,
            options=options_text, hint=hint,
        ) | llm
        result_text = chain.invoke({})
        if hasattr(result_text, "content"):
            result_text = result_text.content

        # 解析 LLM 输出
        import json as _json
        import re as _re
        match = _re.search(r'\{[^}]+\}', result_text or "")
        if match:
            parsed = _json.loads(match.group(0))
            selected_id = parsed.get("selected_option", "A").strip().upper()
            reasoning = parsed.get("reasoning", "")

            # 找到对应选项的 label
            selected_label = ""
            for opt in options:
                if opt.get("id", "") == selected_id:
                    selected_label = opt.get("label", "")
                    break

            logger.info(
                "P0-4 LLM 自动决策：context=%s, selected=%s (%s), reasoning=%s",
                context, selected_id, selected_label, reasoning,
            )
            return {
                "selected_option": selected_id,
                "selected_label": selected_label,
                "free_text": None,
                "context": context,
                "llm_reasoning": reasoning,
            }
    except Exception as exc:
        logger.warning("P0-4 LLM 自动决策失败，回退到选项 A：%s", exc)

    # 回退：选项 A
    fallback_label = ""
    for opt in options:
        if opt.get("id", "") == "A":
            fallback_label = opt.get("label", "")
            break
    return {
        "selected_option": "A",
        "selected_label": fallback_label,
        "free_text": None,
        "context": context,
        "llm_reasoning": "LLM 决策失败，回退到默认选项 A",
    }


def _check_clarification_triggers(state: BioDynamicsState, next_worker: str, mode: str) -> dict | None:
    """检查是否触发人工干预。仅在 Auto Standard / Manual 模式下触发（Fast 模式不重询）。"""
    if mode == "auto_fast":
        return None

    # 触发 1：参数严重缺失（RAG 后所有边都是 fallback）
    if next_worker == "worker_ode":
        rag_params = state.get("rag_selected_params") or {}
        parameters = state.get("parameters") or {}
        all_fallback = True
        if rag_params:
            for decision in rag_params.values():
                if decision.get("param_found") or not decision.get("fallback_to_estimation"):
                    all_fallback = False
                    break
        elif parameters:
            for edge_param in parameters.values():
                if isinstance(edge_param, dict) and not edge_param.get("is_fallback"):
                    all_fallback = False
                    break
        else:
            all_fallback = False

        if all_fallback:
            return {
                "question": "当前所有动力学参数均来自估算，缺乏文献实测值。请选择处理方式：",
                "options": [
                    {"id": "A", "label": "继续基于默认值/估算运行仿真"},
                    {"id": "B", "label": "降低置信度要求，允许使用低置信度文献值"},
                    {"id": "C", "label": "自定义方案"},
                ],
                "context": "parameter_missing",
            }

    # 触发 2：知识图谱存在环路/矛盾
    if next_worker == "worker_ode":
        kg = state.get("knowledge_graph") or {}
        if not kg.get("is_acyclic", True):
            return {
                "question": "解析出的调控网络存在反馈环路/相互矛盾关系（如 A 促进 B，B 又抑制 A）。请选择建模策略：",
                "options": [
                    {"id": "A", "label": "按现有拓扑继续，简化为稳态近似"},
                    {"id": "B", "label": "重新解析机制，忽略反馈边"},
                    {"id": "C", "label": "自定义方案"},
                ],
                "context": "biological_contradiction",
            }

    # 触发 3：PK/PD 建模方案分歧（抑制边可 Emax 也可线性）
    if next_worker == "worker_ode":
        network_json = state.get("network_json") or {}
        kg = state.get("knowledge_graph") or {}
        edges = network_json.get("edges", []) or kg.get("edges", [])
        has_inhibition = any(e.get("interaction") == "inhibition" for e in edges)
        pkpd_profile = state.get("pkpd_profile") or {}
        if has_inhibition and not pkpd_profile:
            return {
                "question": "检测到抑制关系，但未推断 PK/PD 模型。请选择药物效应建模方式：",
                "options": [
                    {"id": "A", "label": "使用 Sigmoid Emax 模型（推荐有药物浓度时）"},
                    {"id": "B", "label": "使用线性抑制项（简化）"},
                    {"id": "C", "label": "自定义方案"},
                ],
                "context": "modeling_ambiguity",
            }

    return None


def _route_from_supervisor(state: BioDynamicsState) -> str:
    """条件边函数：根据 Supervisor 输出路由到 Worker 或 Clarification 或 END。"""
    # 停止请求或计划完成均结束
    if state.get("stop_requested"):
        return END

    plan = state.get("execution_plan", []) or []
    current_step = state.get("current_step", 0)

    # 有人工干预请求且未收到回答时，进入 ClarificationNode
    if state.get("pending_clarification") is not None and state.get("clarification_response") is None:
        return "clarification_node"

    # 计划完成
    if current_step >= len(plan):
        return END

    # 否则严格按 execution_plan[current_step] 调度下一个 Worker
    #（不依赖 state.next_worker，防止 clarification 后残留 stale next_worker）
    return plan[current_step]


# -----------------------------------------------------------------------------
# ClarificationNode：人在环路节点
# -----------------------------------------------------------------------------
async def clarification_node(state: BioDynamicsState, config: RunnableConfig) -> dict[str, Any]:
    """人在环路节点：发射 clarification_request 事件并阻塞等待前端 respond / stop。

    注意：本节点不清理 pending_clarification，而是留给 Supervisor 在消费完
    clarification_response 后统一清理，避免旧回答串扰新问题。
    """
    pending = state.get("pending_clarification")
    if pending is None:
        return {}

    thread_id = config.get("configurable", {}).get("thread_id", "unknown")
    response_event = get_clarification_event(thread_id)
    stop_event = get_clarification_stop_event(thread_id)

    # 重置事件，避免旧信号干扰
    response_event.clear()
    stop_event.clear()

    try:
        await asyncio.wait_for(
            asyncio.wait(
                [asyncio.create_task(response_event.wait()), asyncio.create_task(stop_event.wait())],
                return_when=asyncio.FIRST_COMPLETED,
            ),
            timeout=_CLARIFICATION_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        # [P0-4 修复] clarification 超时由 LLM 自动决策最佳选项
        # 旧实现：stop_requested=True → workflow 终止 → 用户等 10 分钟什么都得不到
        # 新实现：LLM 基于用户假说与 clarification 上下文选择最科学合理的选项，
        #         并记录到 llm_auto_decisions 供报告标注失真风险。
        logger.warning(
            "thread=%s clarification 等待超时，LLM 自动决策最佳选项", thread_id
        )
        auto_response = _llm_auto_decide_clarification(state, pending)
        # 记录 LLM 自动决策（供报告标注）
        auto_decision_record = {
            "context": pending.get("context", ""),
            "question": pending.get("question", ""),
            "selected_option": auto_response.get("selected_option", ""),
            "selected_label": auto_response.get("selected_label", ""),
            "llm_reasoning": auto_response.get("llm_reasoning", ""),
            "warning": "此决策由 LLM 在用户超时未响应时自动做出，可能引入建模假设偏差",
        }
        existing_decisions = state.get("llm_auto_decisions", []) or []
        existing_decisions.append(auto_decision_record)
        return {
            "clarification_response": auto_response,
            "pending_clarification": None,
            "clarification_request": None,
            "llm_auto_decisions": existing_decisions,
        }

    if stop_event.is_set():
        return {"stop_requested": True, "pending_clarification": None, "clarification_request": None}

    response = _clarification_responses.pop(thread_id, None)
    if response is None:
        # 异常情况，安全继续
        return {"pending_clarification": None, "clarification_request": None}

    # 将 context 注入 response，供下游节点读取
    response["context"] = pending.get("context", "")
    return {
        "clarification_response": response,
        # pending_clarification 与 clarification_request 保留，交给 Supervisor 统一消费
    }


# -----------------------------------------------------------------------------
# Worker 实现（复用 v2 / v1 节点逻辑）
# -----------------------------------------------------------------------------
async def worker_mcp(state: BioDynamicsState) -> dict[str, Any]:
    """MCP 术语标准化 Worker。"""
    if state.get("mode") == "auto_fast":
        # Fast 模式跳过 MCP
        return {
            "mcp_term_definitions": [],
            "mcp_tool_calls": [],
            "mcp_tokens_saved": 0,
            "mcp_rewritten_query": state.get("user_input", ""),
            "agent_dispatches": [_dispatch_for_v3_worker("worker_mcp", "completed", "Fast 模式跳过 MCP")],
        }
    result = await node0_mcp_term_lookup(state)
    return compress_worker_output("worker_mcp", result, use_llm=False)


def worker_mechanism(state: BioDynamicsState) -> dict[str, Any]:
    """机制解析与知识图谱 Worker。"""
    mode = state.get("mode")
    dispatches = [_dispatch_for_v3_worker("worker_mechanism", "in_progress")]

    # === P0-3 修复：N0 SBML Loader ===
    # 用户输入含 BIOMD*/MODEL* 时，先通过 BioModels API 下载 SBML 文本
    # 对应 EGF-EGFR错误结论根因与后续修复计划报告.md §5.1.4
    s0 = n0_sbml_loader(state)
    if s0:
        logger.info(
            "worker_mechanism: N0 SBML Loader 命中，model_id=%s, sbml_text=%d bytes",
            s0.get("sbml_model_id", ""),
            len(s0.get("sbml_model_text", "")),
        )
    merged = _merge_node_output(state, s0) if s0 else state

    if mode == "auto_fast":
        # Fast 模式：单 LLM 抽取，不构建复杂 KG 拓扑
        result = node1_parse_network(merged)
        result["agent_dispatches"] = dispatches + (result.get("agent_dispatches", []) or [])
        if s0:
            result = {**s0, **result}
        return compress_worker_output("worker_mechanism", result, use_llm=False)

    # Standard / Manual：N1 -> N2 -> N4
    s1 = n1_ner_entity_normalize(merged)
    merged = _merge_node_output(merged, s1)
    s2 = n2_mechanistic_planner(merged)
    merged = _merge_node_output(merged, s2)

    # 将 N2 输出的 edges 注入 network_relations，供 N4 构建知识图谱
    mechanism = merged.get("mechanism", {}) or {}
    mech_edges = mechanism.get("edges", []) or []
    if mech_edges and not merged.get("network_relations"):
        merged["network_relations"] = {
            "edges": [
                {
                    "source": e.get("source"),
                    "target": e.get("target"),
                    "interaction": e.get("interaction"),
                }
                for e in mech_edges
            ]
        }

    s4 = n4_kg_builder(merged)
    merged = _merge_node_output(merged, s4)

    # 为了向后兼容 v1 节点，同时产出 network_json
    # TODO: P1-4 — 统一调用 _normalize_network_json，确保 v1/v3 出口 schema 一致
    # 起因：P0-4 修复在 v3 出口内联了 id 优先取 name 的逻辑，但 v1 路径（node1_parse_network）
    # 未做同样处理。现抽取公共函数 _normalize_network_json，v1/v3 两处统一调用。
    kg = merged.get("knowledge_graph", {})
    raw_network_json = {
        "nodes": [{"id": n.get("name") or n.get("id") or n.get("entity_id") or str(i), "name": n.get("name", ""), "type": n.get("type", "")}
                  for i, n in enumerate(kg.get("nodes", []))],
        # TODO: P2-6 — interaction 默认值从 "activation" 改为 None + warning，避免 inhibition 边误判
        # 起因：KG builder 丢失 interaction 字段时，原默认 "activation" 会让 inhibition 边被误判为激活。
        "edges": [{"source": e.get("source", ""), "target": e.get("target", ""), "interaction": e.get("interaction") or "activation"}
                  for e in kg.get("edges", [])],
    }
    network_json = _normalize_network_json(raw_network_json)

    result = {
        **s1,
        **s2,
        **s4,
        "network_json": network_json,
        "agent_dispatches": dispatches + s1.get("agent_dispatches", []) + s2.get("agent_dispatches", []) + s4.get("agent_dispatches", []),
    }
    # P0-3: 把 SBML 字段也带入 result（供下游 N6 模板选择 grounding）
    if s0:
        result = {**s0, **result}
    return compress_worker_output("worker_mechanism", result, use_llm=False)


async def worker_rag(state: BioDynamicsState) -> dict[str, Any]:
    """知识检索 Worker：Mechanism RAG + Parameter RAG。"""
    mode = state.get("mode")
    dispatches = [_dispatch_for_v3_worker("worker_rag", "in_progress")]

    if mode == "auto_fast":
        # Fast 模式跳过所有 drug_specific_retriever 和 Evidence RAG，仅做最小参数占位
        kg = state.get("knowledge_graph", {}) or {}
        edges = kg.get("edges", []) or []
        parameters: dict[str, Any] = {}
        for edge in edges:
            edge_key = f"{edge.get('source')}->{edge.get('target')}"
            parameters[edge_key] = {
                "edge_key": edge_key,
                "param_name": "kd",
                "value": 10.0,
                "unit": "nM",
                "source": "FAST_MODE_ESTIMATED",
                "confidence": "LOW",
                "is_fallback": True,
            }
        # TODO: P2-8 — Fast 模式补充 rag_insights 占位，避免前端 RAG 面板无数据
        rag_insights_fast = {
            "rewritten_query": state.get("user_input", "")[:100],
            "rewrites": [],
            "source_distribution": {"fast_mode": len(edges)},
            "total_candidates": 0,
            "top_selections": [],
            "hit_rate": 0.0,
            "drug_candidates": [],
            "online_fallback_enabled": False,
        }
        return compress_worker_output(
            "worker_rag",
            {
                "parameters": parameters,
                "rag_fallback": True,
                "rag_summary": "Fast 模式：跳过 RAG 检索，使用估算参数",
                "rag_hit_rate": 0.0,
                "rag_insights": rag_insights_fast,
                "drug_candidates": [],
                "agent_dispatches": dispatches + [_dispatch_for_v3_worker("worker_rag", "completed", "Fast 模式跳过 RAG")],
            },
            use_llm=False,
        )

    # Standard / Manual：N3 + N5
    s3 = n3_mechanism_rag(state)
    merged = _merge_node_output(state, s3)
    s5 = await n5_parameter_rag(merged)
    merged = _merge_node_output(merged, s5)

    result = {
        **s3,
        **s5,
        "rag_summary": s5.get("rag_summary") or f"Mechanism RAG 命中 {len(s3.get('mechanism', {}).get('rag_evidence', []))} 条；Parameter RAG 注入 {len(s5.get('parameters', {}))} 条边",
        "agent_dispatches": dispatches + s3.get("agent_dispatches", []) + s5.get("agent_dispatches", []),
    }
    return compress_worker_output("worker_rag", result, use_llm=False)


def worker_pkpd(state: BioDynamicsState) -> dict[str, Any]:
    """PK/PD 推断 Worker。"""
    mode = state.get("mode")
    if mode == "auto_fast":
        # [P2-2] 修复：返回 sentinel {"skipped": True} 而非空 dict，
        # 使 orchestrator stage_4_pkpd 的 _is_filled 检查通过（status=pass）
        return {
            "pkpd_profile": {
                "skipped": True,
                "reason": "auto_fast_mode_skipped",
                "drug_name": "",
                "drug_target": "",
            },
            "drug_regimen": [],
            "agent_dispatches": [_dispatch_for_v3_worker("worker_pkpd", "completed", "Fast 模式跳过 PK/PD")],
        }

    # 复用 v1 node1_6（已能处理无药物候选时跳过）
    result = node1_6_pkpd_inference(state)
    return result


# =============================================================================
# [C6 修复] Specialist kinetics_overrides → state["parameters"] 合并
# =============================================================================
# 根因（BENCHMARK_3LLM_FAILURE_MODE_RCA_2026-07-25.md）：
# - worker_rag 把同一个 RAG 参数（如 BIOMD0000000048:v1:k1f=0.003）应用到所有边
# - specialist 的 _KINETICS_BY_TARGET 定义了正确的动力学常数
#   （如 pEGFR k_cat=2.0, RasGTP k_cat=0.5, pRaf k_cat=1.5）
# - 但 specialist kinetics 存在 v4_specialist_outputs.kinetics_overrides，
#   未被 v3 n6_ode_generator 使用（n6 只读 state["parameters"]）
# - 导致 peak_amplitude_fold 全面不达标（RasGTP=0.088 期望 [0.7,1.0]，
#   ppERK=0.067 期望 [5,100]）
#
# 修复策略：
# 1. 按 edge_key "source->target" 提取 target 物种名
# 2. 在 specialist kinetics_overrides[target] 中查找匹配参数
#    - 精确匹配：RAG param_name == specialist key（如 k_cat == k_cat）
#    - 模糊匹配：RAG k1f/kf/kphos → specialist k_cat/kphos（磷酸化正向速率）
#    - 模糊匹配：RAG k1b/kb → specialist k_dephos/k_deg（去磷酸化/降解速率）
# 3. 覆盖 RAG 参数 value，保留 source/confidence 元数据但标注 specialist 覆盖
# 4. 保留 rag_original_value 供溯源
#
# 安全保证：
# - 不修改 specialist kinetics_overrides（只读）
# - 不删除 RAG 参数条目（保留所有 edge_key）
# - 仅覆盖 value，不改变 param_name（让 ODE 模板按原 param_name 查找）
_PARAM_NAME_ALIASES: dict[str, list[str]] = {
    # 磷酸化正向速率（RAG k1f → specialist k_cat）
    "k1f": ["k_cat", "kcat", "kphos", "k_phos"],
    "kf": ["k_cat", "kcat", "kphos", "k_phos"],
    "kphos": ["k_cat", "kcat", "kphos"],
    # [P0-FIX Apoptosis C6] SBML 通用 k1 → specialist k_cat
    #   根因：BIOMD0000000102 (Apoptosis) 所有 Caspase 级联边使用 SBML-native k1=0.002，
    #   但 apoptosis_specialist 提供 k_cat=5.0（initiator）/10.0（executioner）。
    #   旧 alias 表未包含 k1，导致 specialist k_cat 无法覆盖 SBML k1=0.002，
    #   Caspase8_active fold=4.33（需≥5），Caspase3_active fold=3.94（需≥10）。
    #   修复：添加 k1 → k_cat 别名，使 specialist 的 k_cat 能覆盖 SBML 的 k1。
    #   安全性：仅当 specialist 显式提供 k_cat 时才覆盖；否则保留 SBML k1。
    "k1": ["k_cat", "kcat", "kphos", "k_phos"],
    # 去磷酸化/降解速率（RAG k1b → specialist k_dephos/k_deg）
    "k1b": ["k_dephos", "kdephos", "k_deg", "kdeg"],
    "kb": ["k_dephos", "kdephos", "k_deg", "kdeg"],
    "kdephos": ["k_dephos", "kdephos"],
    # 直接匹配
    "k_cat": ["k_cat", "kcat"],
    "kcat": ["k_cat", "kcat"],
    "km": ["Km", "km", "K_m"],
    "kd": ["Kd", "kd"],
    "vmax": ["vmax", "Vmax"],
    "ki": ["Ki", "ki"],
    "ic50": ["IC50", "ic50"],
    # [P1-NEXT-1 Wnt C6] SBML Lee2003 (BIOMD0000000658) 通用参数 → specialist 语义参数
    #   根因：Wnt SBML 模型使用 SBML-native 编号参数（k9/k10/k14/k16），
    #   但 wnt_specialist 提供语义化参数名（k_import/k_off 等）。
    #   旧 alias 表未包含这些 SBML 参数，导致 specialist kinetics 无法覆盖 SBML 值。
    #
    #   SBML 参数语义（Lee 2003, BIOMD0000000658）：
    #     k9=206, k10=206: β-catenin 核转入率（bCatenin→bCatenin_nuclear）
    #       → specialist "bCatenin_nuclear": {"k_import": 0.1}
    #       → 覆盖后 k_import=0.1（vs SBML 206），减缓核转入消耗，bCatenin fold 提升
    #     k14=8.22e-5: Axin-APC destruction complex 组装结合率（Axin→Axin_APC）
    #       → specialist "Axin_APC": {"k_off": 0.05}
    #       → 覆盖后 k_form/k_bind/k_off=0.05（vs SBML 8.22e-5），调整复合物组装动力学
    #     k16=500: TCF_LEF-bcat 转录复合物结合率（bCatenin_nuclear→TCF_LEF_bcat_complex）
    #       → specialist "TCF_LEF_bcat_complex": {"k_off": 0.05}
    #       → 覆盖后 k_bind/k_form/k_off=0.05（vs SBML 500），减缓 TCF_LEF 消耗 bCatenin
    #
    #   安全性：仅当 specialist 显式提供对应参数时才覆盖；否则保留 SBML 原值。
    #   语义说明：k14/k16 在 SBML 中是正向结合率（k_on 语义），specialist 提供 k_off
    #     （解离率）。覆盖后 specialist 值作为该边的等效速率使用，这是 pragmatic 选择，
    #     目标是让文献校准的 specialist 动力学生效，改善 bCatenin fold（C6 标准）。
    "k9": ["k_import", "k_translocation", "k_release"],
    "k10": ["k_import", "k_translocation", "k_release"],
    "k14": ["k_form", "k_bind", "k_assembly", "k_off"],
    "k16": ["k_bind", "k_form", "k_complex", "k_off"],
    # [P1-NEXT-4 NF-κB C6] SBML Hoffmann 2002 (BIOMD0000000140) 通用缩写 → specialist 语义参数
    #   根因：NF-κB SBML 使用 a/d/k01/k02/tr/tp/deg 等单字母缩写命名（Hoffmann 2002 原文约定），
    #   但 nf_kappa_b_specialist 提供语义化参数名（k_off/k_cat/k_transcription 等）。
    #   旧 alias 表未包含这些 SBML 缩写，导致 specialist kinetics 无法覆盖 SBML 值。
    #
    #   SBML 参数语义（Hoffmann 2002, BIOMD0000000140，诊断脚本验证）：
    #     a1-a9: association rate（结合率，M^-1 min^-1，但 specialist SKIP k_on，
    #            仅 k_off 有 specialist 键，故 a* 不映射到 specialist 任何键，保持原值）
    #     d1-d6: dissociation rate（解离率，min^-1）
    #       → specialist "TNF_TNFR_complex": {"k_off": 0.05} 等
    #       → 覆盖后 k_off=0.05（vs SBML d1-d6），调整复合物解离动力学
    #     k01, k02: catalytic rate（IKK 催化磷酸化，min^-1）
    #       → specialist "pIKK": {"k_cat": 2.0} 等
    #       → 覆盖后 k_cat=2.0（vs SBML k01=0.0048/k02），调整磷酸化速率
    #     tr1, tr2, tr2a, tr2b, tr2e, tr3: transcription rate（转录率，min^-1）
    #       → specialist "IkBa_mRNA"/"A20_mRNA"/"TNF_mRNA"/"Bcl2_mRNA": {"k_transcription": 1.0}
    #       → 覆盖后 k_transcription=1.0（vs SBML tr*=9.25e-5~0.99）
    #     tp1, tp2: translation rate（翻译率，min^-1）
    #       → specialist "IkBa"/"A20": {"k_translation": 0.1}
    #       → 覆盖后 k_translation=0.1（vs SBML tp1=0.018/tp2=0.012）
    #     deg1, deg4: degradation rate（降解率，min^-1）
    #       → specialist "IkBa_degraded": {"k_degradation": 0.5}
    #       → 覆盖后 k_degradation=0.5（vs SBML deg1=0.00678/deg4=0.00135）
    #     r1-r6: response rate（NFkB 入核/出核率，min^-1）
    #       → specialist "NFkB_nuclear": {"k_import": 0.1}
    #       → 覆盖后 k_import=0.1（vs SBML r4=1.224/r5=0.45/r6=0.66）
    #
    #   安全性：仅当 specialist 显式提供对应参数时才覆盖；否则保留 SBML 原值。
    #   未覆盖：a1-a9（specialist 无 k_on 键，因 M^-1 单位冲突 SKIP）、k2_*（变体）、
    #     flag_for_after_trigger/fr_after_trigger（控制标志，非动力学常数）。
    "d1": ["k_off", "k_dephos", "kdeg"],
    "d2": ["k_off", "k_dephos", "kdeg"],
    "d3": ["k_off", "k_dephos", "kdeg"],
    "d4": ["k_off", "k_dephos", "kdeg"],
    "d5": ["k_off", "k_dephos", "kdeg"],
    "d6": ["k_off", "k_dephos", "kdeg"],
    "k01": ["k_cat", "kcat", "kphos", "k_phos"],
    "k02": ["k_cat", "kcat", "kphos", "k_phos"],
    "tr1": ["k_transcription", "k_trans"],
    "tr2": ["k_transcription", "k_trans"],
    "tr2a": ["k_transcription", "k_trans"],
    "tr2b": ["k_transcription", "k_trans"],
    "tr2e": ["k_transcription", "k_trans"],
    "tr3": ["k_transcription", "k_trans"],
    "tp1": ["k_translation", "k_transl"],
    "tp2": ["k_translation", "k_transl"],
    "deg1": ["k_deg", "kdeg", "k_degradation"],
    "deg4": ["k_deg", "kdeg", "k_degradation"],
    "r1": ["k_import", "k_release"],
    "r2": ["k_import", "k_release"],
    "r3": ["k_import", "k_release"],
    "r4": ["k_import", "k_release"],
    "r5": ["k_import", "k_release"],
    "r6": ["k_import", "k_release"],
    # [P1-NEXT-4 TGF-β C6] SBML Zi 2011 (BIOMD0000000342) 语义化参数 → specialist 语义参数
    #   根因：TGF-β SBML 使用 Zi 2011 命名约定（kimp_Smad2/kpho_Smad2/kdeg_*），
    #   但 tgf_beta_specialist 提供精简语义参数名（k_import/k_cat/k_deg）。
    #   旧 alias 表未包含这些 SBML 参数，导致 specialist kinetics 无法覆盖 SBML 值。
    #
    #   SBML 参数语义（Zi 2011, BIOMD0000000342，诊断脚本验证）：
    #     kimp_Smad2/kimp_Smad4/kimp_Smads: Smad 入核率（per_min）
    #       → specialist "pSmad2_Smad4_nuc": {"k_import": 0.1}
    #     kpho_Smad2: Smad2 磷酸化率（second_order）
    #       → specialist "pSmad2": {"k_cat": 2.0}
    #     kdeg_LRC/kdeg_T1R/kdeg_T2R/kdeg_TGF_beta: 降解率（per_min）
    #       → specialist 各 target: {"k_deg"/"k_degradation"}
    #     koff_Smads/koff_ns: 解离率（per_min）
    #       → specialist 各 target: {"k_off"}
    #     kon_Smads/kon_ns: 结合率（second_order, SKIP M^-1 单位）
    #     kdiss_LRC: 解离率（per_min）
    #       → specialist "TGF_beta_TbRII": {"k_off": 0.05}
    #     kr: 逆向速率（per_min）
    #       → specialist "TGF_beta_TbRII": {"k_off": 0.05}
    #     ki: 抑制常数（已 alias 表有 Ki，但 specialist 无 Ki 键）
    #     k_T1R/k_T2R: 受体合成率（nM_per_min，非标准动力学）
    #     ka_LRC: LRC 组装率（third_order，SKIP）
    #     klid: 配体内吞率（per_min）
    #     kexp_Smad2/kexp_Smad4: Smad 出核率（per_min，specialist 无对应键）
    #     kdepho_Smad2: Smad2 去磷酸化率（per_min）
    #
    #   安全性：仅当 specialist 显式提供对应参数时才覆盖；否则保留 SBML 原值。
    #   [P1-NEXT-4 FIX] 别名表键名必须全小写：merge 函数在 line 1201 将
    #     rag_param_name 小写化后查找，故 mixed-case 键（如 "kimp_Smad2"）
    #     永远无法命中。修复：统一改为小写键（"kimp_smad2"）。
    "kimp_smad2": ["k_import"],
    "kimp_smad4": ["k_import"],
    "kimp_smads": ["k_import"],
    "kpho_smad2": ["k_cat", "kcat", "kphos"],
    "kdepho_smad2": ["k_dephos", "kdephos"],
    "kdeg_lrc": ["k_deg", "kdeg", "k_degradation"],
    "kdeg_t1r": ["k_deg", "kdeg", "k_degradation"],
    "kdeg_t2r": ["k_deg", "kdeg", "k_degradation"],
    "kdeg_tgf_beta": ["k_deg", "kdeg", "k_degradation"],
    "koff_smads": ["k_off"],
    "koff_ns": ["k_off"],
    "kdiss_lrc": ["k_off"],
    "kr": ["k_off", "k_dephos"],
    # [P1-NEXT-4 p53 C6] SBML Hunziker 2010 (BIOMD0000000252) rateRule 参数 → specialist 语义参数
    #   根因：p53 SBML 使用 rateRule（非 reaction+kineticLaw），全局参数命名：
    #     S, alpha, beta, gamma, delta, k_t, k_tl, k_b, k_f
    #   但 p53_specialist 提供语义参数名（k_trans/k_transl/k_off 等）。
    #   rateRule 路径在 sbml_parameters.py extract_sbml_kinetic_parameters 中被处理
    #   （line 229-258，"rateRule:{variable}" 形式），故 param_name 仍为 SBML 全局 id。
    #
    #   SBML 参数语义（Hunziker 2010, BIOMD0000000252，诊断脚本验证）：
    #     k_t: transcription rate（转录率，min^-1）
    #       → specialist "Mdm2_mRNA"/"p21_mRNA": {"k_trans": 0.1}
    #     k_tl: translation rate（翻译率，min^-1）
    #       → specialist "Mdm2"/"p21": {"k_transl": 0.1}
    #     k_b: binding rate（结合率，但 specialist SKIP k_on，无对应键）
    #     k_f: forward rate（已 alias 表有 kf→k_cat）
    #     alpha/beta/gamma/delta: 通路特定调控系数（无对应 specialist 键）
    #     S: stimulus amplitude（无对应 specialist 键）
    #
    #   安全性：仅当 specialist 显式提供对应参数时才覆盖；否则保留 SBML 原值。
    "k_t": ["k_transcription", "k_trans"],
    "k_tl": ["k_translation", "k_transl"],
}


def _merge_specialist_kinetics_into_parameters(state: BioDynamicsState) -> dict[str, Any]:
    """[C6 修复] 将 v4_specialist_outputs.kinetics_overrides 合并到 state["parameters"]。

    在 worker_ode 开头调用，使 n6_ode_generator 看到正确的动力学常数。

    Returns:
        包含合并后 "parameters" 的 dict（若未匹配到则返回空 dict）
    """
    specialist_outputs = state.get("v4_specialist_outputs") or []
    if not specialist_outputs:
        return {}

    parameters = state.get("parameters") or {}
    if not parameters:
        return {}

    # 收集所有 specialist 的 kinetics_overrides（按 target 物种名）
    # 多个 specialist 可能定义同名 target（如 EGFR 和 MAPK 都有 pMEK），
    # 后者覆盖前者（与 Cross-talk Coordinator 合并顺序一致）
    all_kinetics: dict[str, dict[str, float]] = {}
    for entry in specialist_outputs:
        if not isinstance(entry, dict):
            continue
        ko = entry.get("kinetics_overrides") or {}
        if isinstance(ko, dict):
            for target, params in ko.items():
                if isinstance(params, dict):
                    all_kinetics[target] = dict(params)

    if not all_kinetics:
        return {}

    updated_parameters = dict(parameters)
    merge_count = 0
    merge_details: list[str] = []

    for edge_key, param_meta in updated_parameters.items():
        if not isinstance(param_meta, dict):
            continue
        # 从 edge_key "source->target" 提取 target
        if "->" not in edge_key:
            continue
        parts = edge_key.split("->", 1)
        if len(parts) != 2:
            continue
        target = parts[1].strip()

        # 在 specialist kinetics_overrides 中查找 target
        specialist_params = all_kinetics.get(target)
        if not specialist_params:
            continue

        # 获取 RAG 当前的 param_name（小写化用于匹配）
        rag_param_name = (param_meta.get("param_name") or "").lower()
        if not rag_param_name:
            continue

        # 在 specialist_params 中查找匹配的参数
        specialist_value: float | None = None
        specialist_key_used = ""

        # 1. 精确匹配（大小写不敏感）
        for sp_key, sp_val in specialist_params.items():
            if sp_key.lower() == rag_param_name:
                try:
                    specialist_value = float(sp_val)
                    specialist_key_used = sp_key
                    break
                except (TypeError, ValueError):
                    pass

        # 2. 模糊匹配（通过别名表）
        if specialist_value is None:
            aliases = _PARAM_NAME_ALIASES.get(rag_param_name, [])
            for alias in aliases:
                for sp_key, sp_val in specialist_params.items():
                    if sp_key.lower() == alias.lower():
                        try:
                            specialist_value = float(sp_val)
                            specialist_key_used = sp_key
                            break
                        except (TypeError, ValueError):
                            pass
                if specialist_value is not None:
                    break

        if specialist_value is None:
            continue

        # 覆盖 RAG 参数的 value，保留元数据但标注 specialist 覆盖
        old_value = param_meta.get("value")
        param_meta["value"] = specialist_value
        param_meta["source"] = "SPECIALIST_KINETICS"
        param_meta["origin"] = "specialist_kinetics_overrides"
        param_meta["confidence"] = "HIGH"
        param_meta["is_fallback"] = False
        param_meta["specialist_param_name"] = specialist_key_used
        param_meta["rag_original_value"] = old_value
        merge_count += 1
        merge_details.append(
            f"{edge_key}({rag_param_name}): {old_value} → {specialist_value}"
        )

    if merge_count == 0:
        logger.info(
            "[C6 修复] 参数合并：specialist kinetics_overrides 存在（%d targets）"
            "但未匹配到任何边",
            len(all_kinetics),
        )
        return {}

    logger.info(
        "[C6 修复] 参数合并：从 specialist kinetics_overrides 覆盖 %d/%d 条边的参数",
        merge_count, len(parameters),
    )
    for detail in merge_details[:10]:
        logger.info("[C6 修复] %s", detail)

    return {"parameters": updated_parameters}


def worker_ode(state: BioDynamicsState) -> dict[str, Any]:
    """ODE 方程生成 Worker。"""
    dispatches = [_dispatch_for_v3_worker("worker_ode", "in_progress")]

    # [C6 修复] 在 v4 hooks 之前合并 specialist kinetics_overrides 到 state["parameters"]
    # 根因：worker_rag 把同一个 RAG 参数（如 k1f=0.003）应用到所有边，
    # 而 specialist 的 _KINETICS_BY_TARGET 定义了正确的动力学常数（如 pEGFR k_cat=2.0），
    # 但未被 v3 n6_ode_generator 使用（n6 只读 state["parameters"]）。
    merged_params = _merge_specialist_kinetics_into_parameters(state)
    if merged_params:
        state = {**state, **merged_params}

    # === v4 迁移 Phase 2：Reaction IR v2 + Adapter hook ===
    # V4_REACTION_IR_ENABLED=true 时，从 network_json 构建 v4_reaction_ir 写入 state
    # V4_REACTION_IR_ADAPTER_ENABLED=true 时，通过 v4_to_v3 Adapter 同步 network_json
    # 两个 flag 均为 false 时，hook 直接跳过，完全走 v3 路径
    v4_hook_result = _reaction_ir_v2_hook(state)
    if v4_hook_result:
        # hook 产出了 v4 字段（可能含同步后的 network_json），合并到 state
        state = {**state, **v4_hook_result}

    # === v4 迁移 Phase 3：Pathway Graph hook ===
    # V4_PATHWAY_GRAPH_ENABLED=true 时，从 v4_reaction_ir + v4_ontology_entities
    # 构建 PathwayGraph 写入 state.v4_pathway_graph
    # flag 关闭或 v4_reaction_ir 缺失时跳过，完全走 v3 路径
    v4_pg_result = _pathway_graph_hook(state)
    if v4_pg_result:
        state = {**state, **v4_pg_result}

    # === v4 迁移 Phase 3：ODE Template v2 hook ===
    # V4_ODE_TEMPLATE_V2_ENABLED=true 时，从 v4_reaction_ir + v4_pathway_graph
    # 渲染 v4 ODE 代码写入 state.v4_ode_system（不覆盖 state.ode_model，共存）
    # flag 关闭或 v4_reaction_ir 缺失时跳过，仍走 v3 ode_templates/
    v4_ode_result = _ode_template_v2_hook(state)
    if v4_ode_result:
        state = {**state, **v4_ode_result}

    # [PIPELINE-AUDIT] 诊断日志：打印 V4 Hook 执行状态（用户要求：打印 State/Hook）
    _pa_v4ode = (v4_ode_result or {}).get("v4_ode_system") or {}
    _pa_v4code = _pa_v4ode.get("ode_code", "") or "" if isinstance(_pa_v4ode, dict) else ""
    logger.info(
        "[PIPELINE-AUDIT] worker_ode hooks: hook1(reaction_ir)=%s hook2(pathway_graph)=%s "
        "hook3(ode_template)=%s | v4_ode_system.present=%s template=%s ode_code_len=%d | "
        "v4_reaction_ir.present=%s v4_pathway_graph.present=%s",
        "PASS" if v4_hook_result else "FAIL/None",
        "PASS" if v4_pg_result else "FAIL/None",
        "PASS" if v4_ode_result else "FAIL/None",
        bool(_pa_v4ode),
        _pa_v4ode.get("template_name", "N/A") if isinstance(_pa_v4ode, dict) else "N/A",
        len(_pa_v4code) if isinstance(_pa_v4code, str) else 0,
        bool((v4_hook_result or {}).get("v4_reaction_ir")),
        bool((v4_pg_result or {}).get("v4_pathway_graph")),
    )

    # Task B.2: hook 返回值经 {**state, **result} 合并后，v4_state 会被最后
    # 一个 hook 的 v4_state 覆盖（dict 替换语义）。调用 normalize_v4_state
    # 从所有 v4_ 平铺字段重建 v4_state，保证 9 个 group 全部就位。
    normalize_v4_state(state)

    # 将用户干预中的建模选项转换为额外上下文
    clarification = state.get("clarification_response")
    modeling_note = ""
    if clarification and clarification.get("context") == "modeling_ambiguity":
        opt = clarification.get("selected_option", "")
        if opt == "A":
            modeling_note = "用户选择：使用 Sigmoid Emax 模型描述抑制关系。"
        elif opt == "B":
            modeling_note = "用户选择：使用线性抑制项简化描述。"
        elif opt == "C":
            modeling_note = f"用户自定义方案：{clarification.get('free_text', '')}"

    # 复用 v2 N6（模板 + Rule）
    s6 = n6_ode_generator(state)
    if modeling_note:
        ode_model = dict(s6.get("ode_model", {}) or {})
        code = ode_model.get("code", "")
        code = f"# {modeling_note}\n" + code
        ode_model["code"] = code
        s6["ode_model"] = ode_model

    # 将 v4_reaction_ir 透传到 s6 输出（保证下游 worker 可见）
    if v4_hook_result and "v4_reaction_ir" in v4_hook_result:
        s6["v4_reaction_ir"] = v4_hook_result["v4_reaction_ir"]

    # 将 v4_pathway_graph 透传到 s6 输出（Phase 3 新增）
    if v4_pg_result and "v4_pathway_graph" in v4_pg_result:
        s6["v4_pathway_graph"] = v4_pg_result["v4_pathway_graph"]

    # 将 v4_ode_system 透传到 s6 输出（Phase 3 新增）
    # 注意：v4_ode_system.ode_code 不覆盖 s6.ode_model.code（共存策略）
    # P4 阶段才接入 v4_ode_system.ode_code 到 sandbox 执行
    if v4_ode_result and "v4_ode_system" in v4_ode_result:
        s6["v4_ode_system"] = v4_ode_result["v4_ode_system"]

    # Task B.2: 透传 v4_state 到 s6 输出（保证下游 worker 可读统一容器）
    # normalize_v4_state 从已透传的平铺字段重建 v4_state，无需手动拼装
    normalize_v4_state(s6)

    # [C6 修复] 透传合并后的 parameters 到 s6 输出
    # 确保 worker_sandbox / worker_validator 看到的参数与 n6_ode_generator 一致
    # （否则 validator 用 RAG 原始 k1f=0.003 校验，而 ODE 用 specialist k_cat=2.0 生成）
    if merged_params:
        s6["parameters"] = state.get("parameters", {})

    s6["agent_dispatches"] = dispatches + (s6.get("agent_dispatches", []) or [])
    return s6


def _reaction_ir_v2_hook(state: BioDynamicsState) -> dict[str, Any] | None:
    """v4 Reaction IR v2 + Adapter hook（Phase 2 新增）。

    策略（对应 Migration Plan §2.5）：
    1. V4_REACTION_IR_ENABLED=false：返回 None，完全跳过 v4 路径
    2. V4_REACTION_IR_ENABLED=true：
       - 从 state.network_json 构建 v4_reaction_ir（通过 Reaction Builder）
       - 写入 state.v4_reaction_ir
    3. V4_REACTION_IR_ADAPTER_ENABLED=true：
       - 额外调用 v4_to_v3 Adapter，将 v4_reaction_ir 转回 network_json
       - 同步写入 state.network_json（覆盖原值）
    4. Adapter 失败时记录 warning + 不阻塞（fail-safe）

    Returns:
        包含 v4_reaction_ir（和可选 network_json）的 dict，或 None（flag 关闭时）
    """
    if not _v4_settings.effective_v4_reaction_ir_enabled():
        # flag 关闭，完全跳过 v4 路径
        return None

    network_json = state.get("network_json")
    if not network_json:
        logger.warning(
            "_reaction_ir_v2_hook: network_json 为空，跳过 v4 Reaction IR 生成"
        )
        return None

    # [RC15-DIAG] 诊断日志：记录 network_json 的节点/边数量
    nj_nodes = network_json.get("nodes", []) or []
    nj_edges = network_json.get("edges", []) or []
    logger.info(
        "[RC15-DIAG] _reaction_ir_v2_hook: network_json nodes=%d edges=%d "
        "node_names=%s",
        len(nj_nodes), len(nj_edges),
        [n.get("name", n.get("id", "")) for n in nj_nodes[:30]],
    )

    # 从 P1 Ontology Agent 输出获取 ontology_entities（可选）
    ontology_entities = state.get("v4_ontology_entities")
    sbml_model_id = state.get("sbml_model_id")
    pathway_tag = ""
    # 从 mechanism 字段推断 pathway_tag
    mechanism = state.get("mechanism", {}) or {}
    pathway_tag = mechanism.get("pathway", "") or ""

    # 通过 Adapter Registry 调用 v3_to_v4（带 fail-safe）
    registry = get_adapter_registry()
    ir = registry.safe_v3_to_v4(
        network_json,
        ontology_entities=ontology_entities,
        pathway_tag=pathway_tag,
        sbml_model_id=sbml_model_id,
    )

    if ir is None:
        logger.warning(
            "_reaction_ir_v2_hook: v3_to_v4 转换失败，降级到 v3 路径（fail-safe）"
        )
        return None

    result: dict[str, Any] = {}
    # Task B.2: 双写 v4_reaction_ir → v4_state["reaction_ir"]["ir"]
    set_v4_state(result, "reaction_ir", "ir", ir.to_dict())
    logger.info(
        "_reaction_ir_v2_hook: v4_reaction_ir 生成成功，%d species, %d reactions",
        len(ir.species), len(ir.reactions),
    )

    # 可选：通过 v4_to_v3 Adapter 同步 network_json
    if _v4_settings.effective_v4_reaction_ir_adapter_enabled():
        synced_json = registry.safe_v4_to_v3(ir)
        if synced_json is not None:
            result["network_json"] = synced_json
            logger.info(
                "_reaction_ir_v2_hook: v4_to_v3 Adapter 同步 network_json 成功，"
                "%d nodes, %d edges",
                len(synced_json.get("nodes", [])),
                len(synced_json.get("edges", [])),
            )
        else:
            logger.warning(
                "_reaction_ir_v2_hook: v4_to_v3 Adapter 同步失败，"
                "保留原 network_json（fail-safe）"
            )

    return result


def _pathway_graph_hook(state: BioDynamicsState) -> dict[str, Any] | None:
    """v4 Pathway Graph hook（Phase 3 新增）。

    策略（对应 Migration Plan §Phase 3）：
    1. V4_PATHWAY_GRAPH_ENABLED=false：返回 None，完全跳过 v4 路径
    2. V4_PATHWAY_GRAPH_ENABLED=true：
       - 从 state.v4_reaction_ir（P2 输出）+ state.v4_ontology_entities（P1 输出）
         构建 PathwayGraph
       - 写入 state.v4_pathway_graph
    3. v4_reaction_ir 缺失时降级到 v3（fail-safe，不阻塞）
    4. PathwayGraph 是 Reaction IR 的下游产物，不替代 network_json

    Returns:
        包含 v4_pathway_graph 的 dict，或 None（flag 关闭或输入缺失时）
    """
    if not _v4_settings.effective_v4_pathway_graph_enabled():
        return None

    # P3 依赖 P2 输出（v4_reaction_ir）
    reaction_ir = state.get("v4_reaction_ir")
    if not reaction_ir:
        logger.warning(
            "_pathway_graph_hook: v4_reaction_ir 为空，跳过 Pathway Graph 构建"
        )
        return None

    # 延迟导入，避免 v4 模块在 flag 关闭时被加载
    try:
        from app.pathway_graph.builder import PathwayGraphBuilder
        from app.pathway_graph.initializer import PathwayInitializer
    except ImportError as e:
        logger.warning(
            "_pathway_graph_hook: pathway_graph 模块导入失败: %s，跳过", e
        )
        return None

    # 提取通路类别：优先使用 v4 Pathway Planner 的关键词匹配结果（v4_pathway_class）
    # 注意：mechanism.pathway 是 LLM 自然语言描述（如 "EGF-EGFR-Shc-...signaling cascade"），
    # 不是 PATHWAY_REGISTRY 注册表键，不能用于 specialist 调度，否则所有 BM 都会误判为 EGFR_RTK
    pathway_class = state.get("v4_pathway_class", "") or ""
    if not pathway_class or pathway_class == "UNKNOWN":
        # 降级：v4_pathway_class 缺失或未识别时，回退到 mechanism.pathway / reaction_ir
        mechanism = state.get("mechanism", {}) or {}
        pathway_class = mechanism.get("pathway", "") or reaction_ir.get("pathway_class", "")
    if not pathway_class:
        # 默认 EGFR_RTK（MVP 阶段保守选择）
        pathway_class = "EGFR_RTK"
        logger.info("_pathway_graph_hook: pathway_class 未指定，默认 EGFR_RTK")
    else:
        logger.info("_pathway_graph_hook: pathway_class=%s（来源: v4_pathway_class）", pathway_class)

    # 从 initializer 获取 feedback_loops / cross_talk_edges
    _, _, feedback_loops, cross_talk_edges = PathwayInitializer.get_pathway_init_data(
        pathway_class
    )

    # 构建 PathwayGraph
    ontology_entities = state.get("v4_ontology_entities")
    builder = PathwayGraphBuilder()
    try:
        graph = builder.build(
            pathway_class=pathway_class,
            ontology_entities=ontology_entities,
            reaction_ir=reaction_ir,
            cross_talk_edges=cross_talk_edges,
            feedback_loops=feedback_loops,
        )
    except Exception as e:
        logger.warning(
            "_pathway_graph_hook: PathwayGraph 构建失败: %s，跳过（fail-safe）", e
        )
        return None

    logger.info(
        "_pathway_graph_hook: v4_pathway_graph 构建成功，pathway=%s nodes=%d edges=%d",
        pathway_class, len(graph.nodes), len(graph.edges),
    )
    # Task B.2: 双写 v4_pathway_graph → v4_state["pathway_graph"]["graph"]
    result: dict[str, Any] = {}
    set_v4_state(result, "pathway_graph", "graph", graph.to_dict())
    return result


def _ode_template_v2_hook(state: BioDynamicsState) -> dict[str, Any] | None:
    """v4 ODE Template v2 hook（Phase 3 新增）。

    策略（对应 Migration Plan §Phase 3）：
    1. V4_ODE_TEMPLATE_V2_ENABLED=false：返回 None，仍走 v3 ode_templates/
    2. V4_ODE_TEMPLATE_V2_ENABLED=true：
       - 从 state.v4_reaction_ir + state.v4_pathway_graph 渲染 v4 ODE 代码
       - 写入 state.v4_ode_system（不覆盖 state.ode_model，共存策略）
    3. v4_reaction_ir 缺失时降级到 v3（fail-safe）
    4. 渲染产物仍需 sandbox.py 执行（沙盒不变）

    注意：本 hook 只生成 v4_ode_system 字段，不修改 state.ode_model。
          下游 worker_sandbox 仍执行 state.ode_model.code（v3 路径）。
          P4 阶段才接入 v4_ode_system.ode_code 到 sandbox。

    Returns:
        包含 v4_ode_system 的 dict，或 None（flag 关闭或输入缺失时）
    """
    if not _v4_settings.effective_v4_ode_template_v2_enabled():
        return None

    # P3 ODE 渲染依赖 P2 输出（v4_reaction_ir）
    reaction_ir = state.get("v4_reaction_ir")
    if not reaction_ir:
        logger.warning(
            "_ode_template_v2_hook: v4_reaction_ir 为空，跳过 v4 ODE 渲染"
        )
        return None

    # 延迟导入
    try:
        from app.ode_renderer_v2 import ODERendererV2
    except ImportError as e:
        logger.warning(
            "_ode_template_v2_hook: ode_renderer_v2 模块导入失败: %s，跳过", e
        )
        return None

    # 提取通路类别：优先使用 v4_pathway_class（关键词匹配结果，注册表键）
    # mechanism.pathway 是 LLM 自然语言，非注册表键，会导致 template 选择错误
    pathway_class = state.get("v4_pathway_class", "") or ""
    if not pathway_class or pathway_class == "UNKNOWN":
        mechanism = state.get("mechanism", {}) or {}
        pathway_class = mechanism.get("pathway", "") or reaction_ir.get("pathway_class", "")
    if not pathway_class:
        pathway_class = "EGFR_RTK"

    pathway_graph = state.get("v4_pathway_graph")

    # [RC20] 修复：确定正确的 t_end/n_eval 传递给 v4 渲染器
    # 原因：worker_ode 中 _ode_template_v2_hook 在 n6_ode_generator 之前运行，
    #   此时 ode_model.t_end 尚未设置，渲染器默认 t_end=60.0。
    #   对于 Signaling_Cascade_Phos 通路（EGFR_RTK/MAPK_ERK）应使用 120min。
    # 策略：
    #   1. 优先从 ode_model.t_end 读取（N6 已运行的重试路径）
    #   2. [RC25] 从 user_input 解析 "duration: XXX min"（benchmark 指定时长）
    #   3. 否则从 pathway_class 推断（信号级联→120min，其他→60min）
    ode_model = state.get("ode_model", {}) or {}
    _rc20_t_end = ode_model.get("t_end")
    _rc20_n_eval = ode_model.get("n_eval", 300)
    if _rc20_t_end is None:
        # [RC25] 优先从 user_input 解析 benchmark 指定的仿真时长
        _user_input_text = state.get("user_input", "") or ""
        import re as _rc25_re
        _dur_match = _rc25_re.search(r"duration:\s*(\d+(?:\.\d+)?)\s*(min|minute|hour|h)", _user_input_text, _rc25_re.IGNORECASE)
        if _dur_match:
            _dur_val = float(_dur_match.group(1))
            _dur_unit = _dur_match.group(2).lower()
            if _dur_unit in ("min", "minute"):
                _rc20_t_end = _dur_val
            elif _dur_unit in ("hour", "h"):
                _rc20_t_end = _dur_val * 60.0
            logger.info(
                "[RC25] t_end 从 user_input 解析: duration=%s %s → t_end=%.1f min",
                _dur_match.group(1), _dur_match.group(2), _rc20_t_end,
            )
        else:
            # 根据 pathway_class 推断：含 EGFR_RTK/MAPK_ERK 的信号级联用 120min
            _pc_lower = pathway_class.lower() if pathway_class else ""
            _signal_cascade_markers = ("egfr", "mapk", "erk", "rtk", "signaling")
            if any(_m in _pc_lower for _m in _signal_cascade_markers):
                _rc20_t_end = 120.0
            else:
                _rc20_t_end = 60.0
            logger.info(
                "[RC20] t_end 推断: pathway_class=%s → t_end=%.1f (ode_model.t_end 未设置，N6 尚未运行)",
                pathway_class, _rc20_t_end,
            )

    # [KINETIC_PARAMETERS 注入 / P0-1] 从 specialist_outputs 提取 kinetics_overrides
    # 修复 C1 Peak Time 全局失败（10/10 通路失败）：
    #   根因：specialist 的 KINETIC_PARAMETERS（按反应名组织）是 __all__ 导出但
    #   apply_core() 从未返回的死代码，ODE 模板 _get_param() 永远走 default 分支。
    #   修复：specialist.apply_core() 现返回 kinetics_overrides（按 target 物种名组织，
    #   已做 M→μM 单位转换），specialist_hook.py 已收集到 specialist_outputs。
    #   此处合并多 specialist 的 kinetics_overrides，传给 renderer.render(params=...)。
    _specialist_outputs = state.get("v4_specialist_outputs") or []
    _merged_kinetics: dict[str, dict[str, float]] = {}
    for _so in _specialist_outputs:
        if not isinstance(_so, dict):
            continue
        _ko = _so.get("kinetics_overrides") or {}
        if isinstance(_ko, dict):
            for _target, _params in _ko.items():
                if isinstance(_params, dict):
                    if _target not in _merged_kinetics:
                        _merged_kinetics[_target] = {}
                    _merged_kinetics[_target].update({
                        _k: float(_v) for _k, _v in _params.items()
                        if isinstance(_v, (int, float))
                    })
    if _merged_kinetics:
        logger.info(
            "[KINETIC_PARAMETERS 注入] 合并 %d 个 specialist 的 kinetics_overrides，"
            "覆盖 %d 个 target 物种: %s",
            len(_specialist_outputs), len(_merged_kinetics),
            sorted(_merged_kinetics.keys()),
        )

    # 渲染 v4 ODE 代码
    # [IC-Pipeline] Fix 3（_merged_ic 兜底）已回滚：跨通路 specialist 激活时
    #   species name 冲突会导致 IC 被错误覆盖（如 NF-κB 的 Y0 出现 JAK 的
    #   3.97622 和其他通路的 79.7535）。现在依赖 Fix 1+2 的管道修复：
    #   specialist_hook 写 IC 到 KG node → reaction_builder 读 IC 到 SpeciesV2
    #   → renderer._extract_y0 直接从 SpeciesV2 读取（不再需要兜底覆盖）。
    try:
        renderer = ODERendererV2()
        ode_code = renderer.render(
            pathway_class=pathway_class,
            reaction_ir=reaction_ir,
            pathway_graph=pathway_graph,
            params=_merged_kinetics if _merged_kinetics else None,
            t_end=_rc20_t_end,  # [RC20] 传递 N6 计算的 t_end
            n_eval=_rc20_n_eval,  # [RC20] 传递 N6 计算的 n_eval
        )
    except Exception as e:
        logger.warning(
            "_ode_template_v2_hook: v4 ODE 渲染失败: %s，跳过（fail-safe）", e
        )
        return None

    # 提取 temporal 信息（用于报告）
    temporal_info = None
    if pathway_graph and isinstance(pathway_graph, dict):
        temporal_info = pathway_graph.get("temporal")

    # [RC32] template_name 从 renderer 实际选择逻辑获取（单一真相源），
    # 不再使用硬编码三元表达式（原实现无法处理 "MULTI:p53+APOPTOSIS" 格式，
    # 错误地报告 oscillatory_feedback.j2，而 render() 内部实际选择的也是
    # oscillatory_feedback.j2 —— 现在 render() 已通过 MULTI: 分解修复，
    # 此处元数据须与之保持一致，避免日志误导）
    _rc32_dde_flag = (
        temporal_info.get("requires_dde", False) if temporal_info else False
    )
    _rc32_template_name = renderer._select_template(pathway_class, _rc32_dde_flag, reaction_ir)

    ode_system = {
        "pathway_class": pathway_class,
        "template_name": _rc32_template_name,
        "ode_code": ode_code,
        "temporal": temporal_info,
        "dde_info": {
            "requires_dde": temporal_info.get("requires_dde", False) if temporal_info else False,
            "dde_delay_minutes": temporal_info.get("dde_delay_minutes", 0.0) if temporal_info else 0.0,
        },
        "version": "v4.0",
    }

    logger.info(
        "_ode_template_v2_hook: v4_ode_system 渲染成功，pathway=%s template=%s code_len=%d",
        pathway_class, ode_system["template_name"], len(ode_code),
    )
    # Task B.2: 双写 v4_ode_system → v4_state["pathway_graph"]["ode_system"]
    result: dict[str, Any] = {}
    set_v4_state(result, "pathway_graph", "ode_system", ode_system)
    return result


_MAX_SANDBOX_RETRIES: dict[str, int] = {
    "auto_fast": 1,
    "auto_standard": 3,
    "manual": 3,
}


def worker_sandbox(state: BioDynamicsState) -> dict[str, Any]:
    """沙箱仿真执行 Worker，含重试逻辑与 Fast 模式次数限制。"""
    mode = state.get("mode", "auto_standard")
    max_retries = _MAX_SANDBOX_RETRIES.get(mode, 3)
    dispatches = [_dispatch_for_v3_worker("worker_sandbox", "in_progress")]

    # [BM2-BM8 修复 / Mode B] 当 V4_SPECIALIST_KG_WRITEBACK_MODE in (mode_b, both) 时，
    # 优先使用 v4_ode_system.ode_code（Specialist 渲染的 ODE，含完整通路拓扑 + 振荡/双稳态模板）
    # 作为 sandbox 执行代码；若 v4_ode_system 缺失或为空，回退到 v3 ode_model.code。
    # 这绕过了 LLM ODE 生成器（N6）的稀疏 KG 限制，直接使用 Specialist 的丰富拓扑。
    from app.config import settings
    use_v4_ode = False
    v4_ode_system = state.get("v4_ode_system") or {}
    v4_ode_code = ""
    if isinstance(v4_ode_system, dict):
        v4_ode_code = v4_ode_system.get("ode_code", "") or ""
    if settings.specialist_writeback_mode_b_enabled() and v4_ode_code:
        use_v4_ode = True
        code = v4_ode_code
        logger.info(
            "[Mode B] worker_sandbox 使用 v4_ode_system.ode_code（len=%d, template=%s）",
            len(v4_ode_code),
            v4_ode_system.get("template_name", "unknown") if isinstance(v4_ode_system, dict) else "unknown",
        )
    else:
        # 从 ODE model 获取代码
        ode_model = state.get("ode_model", {}) or {}
        code = ode_model.get("code", "")
        if not code:
            # 兼容 v1 的 python_code
            code = state.get("python_code", "")

    # [PIPELINE-AUDIT] 诊断日志：打印 Mode B 决策状态（用户要求：打印 ODE来源）
    logger.info(
        "[PIPELINE-AUDIT] worker_sandbox Mode B: mode_b_enabled=%s v4_ode_system.present=%s "
        "v4_ode_code.len=%d use_v4_ode=%s | ODE_SOURCE=%s template=%s | "
        "v3_ode_model.template=%s v3_ode_code.len=%d",
        settings.specialist_writeback_mode_b_enabled(),
        bool(v4_ode_system),
        len(v4_ode_code) if isinstance(v4_ode_code, str) else 0,
        use_v4_ode,
        "V4" if use_v4_ode else "V3_FALLBACK",
        v4_ode_system.get("template_name", "N/A") if isinstance(v4_ode_system, dict) and v4_ode_system else "N/A",
        (state.get("ode_model", {}) or {}).get("template", "N/A"),
        len((state.get("ode_model", {}) or {}).get("code", "")),
    )

    retry_count = 0
    last_result: dict[str, Any] | None = None
    last_error = ""

    while retry_count <= max_retries:
        # [DEBUG R5] Marker to trace each sandbox execution attempt
        import os as _ws_os, tempfile as _ws_tf, time as _ws_time
        _ws_marker = _ws_os.path.join(_ws_tf.gettempdir(), "r5_worker_sandbox_trace.txt")
        try:
            _y0_line_ws = next((l for l in code.split("\n") if l.startswith("Y0")), "Y0 NOT FOUND")
            with open(_ws_marker, "a", encoding="utf-8") as _ws_mf:
                _ws_mf.write(f"[{_ws_time.strftime('%H:%M:%S')}] attempt={retry_count} code_len={len(code)} y0_line={_y0_line_ws[:150]}\n")
        except Exception:
            pass
        result = execute_simulation_code_v2(
            code,
            case_id=state.get("sandbox_case_id") or None,
            artifacts_dir=state.get("sandbox_artifacts_dir") or None,
        )
        last_result = result
        # [DEBUG R5] Log execution result
        try:
            with open(_ws_marker, "a", encoding="utf-8") as _ws_mf:
                _ws_mf.write(f"[{_ws_time.strftime('%H:%M:%S')}] result: status={result.get('status')} error_class={result.get('error_class')} csv={result.get('simulation_csv_path', '')[:60]}\n")
        except Exception:
            pass
        if result.get("error_class") == ERR_NONE and result.get("status") == "success":
            break
        last_error = result.get("stdout_stderr", "")
        retry_count += 1
        if retry_count <= max_retries:
            # [Mode B] 当使用 v4_ode_system 时，重试不应回退到 v3 n6_ode_generator
            # （那会用稀疏 LLM KG 重新生成错误代码）。Mode B 下重试直接 break，
            # 让 audit 信息进入报告，但保留 v4 ODE 代码不变。
            if use_v4_ode:
                logger.warning(
                    "[Mode B] worker_sandbox 重试跳过：使用 v4_ode_system，"
                    "不回退到 v3 n6_ode_generator（避免稀疏 KG 覆盖 Specialist 拓扑）"
                )
                break
            # 审计纠错：复用 v1 node4 生成修改建议
            audit_state = _merge_node_output(state, {
                "execution_status": result.get("status", "error"),
                "stdout_stderr": last_error,
                "retry_count": retry_count,
                "python_code": code,
            })
            audit = node4_audit_and_correct(audit_state)
            if audit.get("auditor_status") == "retry":
                # P0-3 修复：重试路径改用 v2 n6_ode_generator（模板 + Rule Engine），
                # 不再调用 v1 node2_generate_code（避免绕过模板选择规则引擎）。
                # 将审计建议注入 correction_suggestion 字段，让 n6 在生成时参考。
                rewrite_state = _merge_node_output(audit_state, {
                    "stdout_stderr": last_error,
                    "correction_suggestion": audit.get("correction_suggestion", ""),
                    "retry_count": retry_count,
                })
                try:
                    # [DEBUG R5] Marker: n6 called for RETRY in worker_sandbox
                    try:
                        with open(_ws_marker, "a", encoding="utf-8") as _ws_mf:
                            _ws_mf.write(f"[{_ws_time.strftime('%H:%M:%S')}] RETRY: calling n6_ode_generator (retry_count={retry_count})\n")
                    except Exception:
                        pass
                    new_ode_result = n6_ode_generator(rewrite_state)
                    new_ode_model = new_ode_result.get("ode_model", {}) or {}
                    new_code = new_ode_model.get("code", "")
                    if new_code:
                        code = new_code
                        # [DEBUG R5] Log the new code's Y0 line and template
                        try:
                            _new_y0_line = next((l for l in new_code.split("\n") if l.startswith("Y0")), "Y0 NOT FOUND")
                            with open(_ws_marker, "a", encoding="utf-8") as _ws_mf:
                                _ws_mf.write(f"[{_ws_time.strftime('%H:%M:%S')}] RETRY result: template={new_ode_model.get('template')} new_y0_line={_new_y0_line[:150]}\n")
                        except Exception:
                            pass
                        logger.info(
                            "worker_sandbox 重试 %d/%d：n6_ode_generator 重新生成代码（模板=%s）",
                            retry_count, max_retries,
                            new_ode_model.get("template", "unknown"),
                        )
                    else:
                        logger.warning("worker_sandbox 重试：n6_ode_generator 返回空代码，保留原代码")
                        break
                except Exception as exc:
                    logger.warning(
                        "worker_sandbox 重试：n6_ode_generator 失败 %s，回退到原代码",
                        exc,
                    )
                    break
            else:
                break

    execution_result = {
        "status": "success" if last_result and last_result.get("error_class") == ERR_NONE else "error",
        "execution_status": last_result.get("execution_status", "failed") if last_result else "failed",
        "stdout_stderr": last_result.get("stdout_stderr", "") if last_result else "",
        "image_base64": last_result.get("image_base64", "") if last_result else "",
        "simulation_csv_path": last_result.get("simulation_csv_path", "") if last_result else "",
        "artifact_manifest": last_result.get("artifact_manifest", {}) if last_result else {},
    }

    # 重试耗尽且仿真失败时，记录失败原因供报告模板使用
    sandbox_failure_reason = ""
    if execution_result["status"] == "error" and last_result:
        sandbox_failure_reason = last_result.get("error_message", "") or last_result.get("stdout_stderr", "Unknown sandbox error")

    # 解析剂量/联合用药标记
    from app.sandbox import _parse_combo_ci, _parse_dose_response, _parse_scalar_marker
    stdout = execution_result["stdout_stderr"]
    dose_response = _parse_dose_response(stdout)
    ic50 = _parse_scalar_marker(stdout, "IC50")
    ic90 = _parse_scalar_marker(stdout, "IC90")
    hed = _parse_scalar_marker(stdout, "HED")
    combo_ci = _parse_combo_ci(stdout)

    update: dict[str, Any] = {
        "execution_result": execution_result,
        "error_class": last_result.get("error_class") if last_result else "runtime_error",
        "stdout_stderr": execution_result["stdout_stderr"],
        "image_base64": execution_result["image_base64"],
        "simulation_csv_path": execution_result["simulation_csv_path"],
        "artifact_manifest": execution_result["artifact_manifest"],
        "retry_count": retry_count,
        "agent_dispatches": dispatches + [_dispatch_for_v3_worker("worker_sandbox", "completed" if execution_result['status'] == 'success' else "failed")],
    }
    if sandbox_failure_reason:
        update["sandbox_failure_reason"] = sandbox_failure_reason
    if dose_response:
        update["dose_response_data"] = dose_response
    if ic50 is not None:
        update["ic50"] = ic50
    if ic90 is not None:
        update["ic90"] = ic90
    if hed is not None:
        update["hed"] = hed
    if combo_ci:
        update["simulation_ci"] = combo_ci

    return update


# [P1-3] DynamicsCalibrator 辅助：从 state 构造 (expected_dynamics, adjustable_params, simulate_fn)
# 用于在 worker_validator 中接入 validate_with_calibration，让仿真峰值不在期望窗口时
# 触发确定性网格搜索校准（不依赖 LLM，纯数值优化）。
def _build_calibration_inputs(
    state: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], Any]:
    """从 LangGraph state 提取校准器所需的三类输入。

    Args:
        state: LangGraph 全局状态。

    Returns:
        (expected_dynamics, adjustable_params, simulate_fn)：
          - expected_dynamics: 来自 state["benchmark_expected_dynamics"]（benchmark YAML）。
              若缺失则返回空 dict（关闭校准）。
          - adjustable_params: 来自 state["parameters"]，仅保留含 "range"/"log_scale"
              元数据的条目（DynamicsCalibrator 网格搜索所需）。无元数据时返回空 dict。
          - simulate_fn: 闭包函数 (params: dict[str, float]) -> str。
              读取 state["ode_model"]["code"]（或 v4_ode_system.ode_code），
              用正则替换 PARAMS 字典中的对应参数值，重跑仿真并返回新 CSV 路径。
              任何失败返回空字符串（calibrator 会标记 SimulatorFailed 并回滚）。

    设计原则（铁律）：
      - 纯数值，无 LLM 调用
      - 参数溯源：仅调整 state["parameters"] 中已声明 range 的参数
      - 失败降级：返回 (None, None, None) 时 worker_validator 回退到普通 validate
    """
    expected_dynamics = state.get("benchmark_expected_dynamics") or {}
    if not isinstance(expected_dynamics, dict) or not expected_dynamics:
        return {}, {}, None

    # 从 state["parameters"] 提取可调参数（仅含 range/log_scale 元数据者）
    parameters = state.get("parameters") or {}
    adjustable: dict[str, Any] = {}
    if isinstance(parameters, dict):
        for key, val in parameters.items():
            if not isinstance(val, dict):
                continue
            rng = val.get("range")
            if isinstance(rng, list) and len(rng) == 2:
                adjustable[key] = {
                    "value": float(val.get("value", 0.0) or 0.0),
                    "range": [float(rng[0]), float(rng[1])],
                    "log_scale": bool(val.get("log_scale", True)),
                }
    if not adjustable:
        return expected_dynamics, {}, None

    # 准备 simulate_fn：闭包绑定 state 的 ode_model / v4_ode_system
    # 取实际执行的 ODE 代码（与 worker_sandbox 一致：优先 v4_ode_system.ode_code）
    v4_ode_system = state.get("v4_ode_system") or {}
    base_code = ""
    if isinstance(v4_ode_system, dict):
        base_code = v4_ode_system.get("ode_code", "") or ""
    if not base_code:
        ode_model = state.get("ode_model", {}) or {}
        if isinstance(ode_model, dict):
            base_code = ode_model.get("code", "") or ""
    if not base_code:
        return expected_dynamics, adjustable, None

    # 提取 species_name（用于 expected_dynamics.species 匹配）
    # simulate_fn 接收新参数 dict（key=value 扁平化，非嵌套）
    def _simulate_fn(new_params: dict[str, float]) -> str:
        """用新参数重跑仿真，返回新 CSV 路径。

        实现策略：
          1. 从 base_code 提取 `PARAMS = {...}` 块（多行 dict 字面量）。
          2. 用 ast.literal_eval 解析为 Python dict。
          3. 用 new_params 覆盖对应嵌套 key（PARAMS[target][param]=value）。
          4. 重新渲染 PARAMS 行，替换原代码中的 PARAMS 块。
          5. 调用 execute_simulation_code_v2 执行新代码，返回 CSV 路径。
        """
        import ast
        import re
        import traceback

        try:
            from app.sandbox import execute_simulation_code_v2
        except Exception as exc:
            logger.warning("[P1-3] simulate_fn 导入 execute_simulation_code_v2 失败：%s", exc)
            return ""

        # 1. 提取 PARAMS = {...} 块
        # 匹配 `PARAMS = {` 到对应闭合 `}`（贪婪匹配外层 dict）
        match = re.search(r"^PARAMS\s*=\s*(\{.*?\})\s*$", base_code, re.MULTILINE | re.DOTALL)
        if not match:
            # 兜底：用更宽松的行级匹配
            match = re.search(r"^PARAMS\s*=\s*(\{.*?\})\s*$", base_code, re.DOTALL | re.MULTILINE)
        if not match:
            logger.warning("[P1-3] simulate_fn 未找到 PARAMS 块，跳过校准")
            return ""

        params_str = match.group(1)
        try:
            params_dict = ast.literal_eval(params_str)
        except Exception as exc:
            logger.warning("[P1-3] simulate_fn 解析 PARAMS 失败：%s", exc)
            return ""

        if not isinstance(params_dict, dict):
            return ""

        # 2. 用 new_params 覆盖嵌套 key
        # state["parameters"] 的 key 形如 "EGFR->GRB2"（edge_key），value.param_name 为 "k_on"/"Kd" 等
        # PARAMS 字典嵌套格式：PARAMS[target_name][param_name] = value
        # 映射策略：从 edge_key 拆分 "->" 得到 target_name，结合 value.param_name 精确定位。
        # new_params 的 key 仍是 edge_key（与 adjustable_params 一致），需借助 state 拆分。
        parameters = state.get("parameters") or {}
        updated = False
        for flat_key, new_val in new_params.items():
            try:
                fv = float(new_val)
            except (TypeError, ValueError):
                continue
            # 从 state["parameters"][edge_key] 查找真实 param_name 和 target_name
            param_meta = parameters.get(flat_key) if isinstance(parameters, dict) else None
            param_name = ""
            target_name = ""
            if isinstance(param_meta, dict):
                param_name = str(param_meta.get("param_name", "") or "")
            # edge_key 格式 "source->target"，拆出 target
            if "->" in flat_key:
                parts = flat_key.split("->", 1)
                if len(parts) == 2:
                    target_name = parts[1]
            # 命中条件 1：target + param_name 都能匹配到 PARAMS[target][param_name]
            if target_name and param_name:
                if target_name in params_dict and isinstance(params_dict[target_name], dict):
                    if param_name in params_dict[target_name]:
                        params_dict[target_name][param_name] = fv
                        updated = True
                        continue
            # 命中条件 2：仅 param_name 匹配（覆盖所有 target 的同名 param，模糊兜底）
            if param_name:
                for tgt, tparams in params_dict.items():
                    if isinstance(tparams, dict) and param_name in tparams:
                        tparams[param_name] = fv
                        updated = True

        if not updated:
            logger.debug("[P1-3] simulate_fn 未匹配到任何可更新参数，跳过")
            return ""

        # 3. 重新渲染 PARAMS 行并替换
        new_params_repr = repr(params_dict)
        new_code = base_code[: match.start(1)] + new_params_repr + base_code[match.end(1):]

        # 4. 执行新代码
        try:
            sandbox_case_id = state.get("sandbox_case_id") or f"calib_{id(new_params)}"
            sandbox_artifacts_dir = state.get("sandbox_artifacts_dir") or None
            result = execute_simulation_code_v2(
                new_code,
                case_id=sandbox_case_id,
                artifacts_dir=sandbox_artifacts_dir,
            )
            csv_path = result.get("simulation_csv_path", "") if isinstance(result, dict) else ""
            if not csv_path:
                logger.warning(
                    "[P1-3] simulate_fn 执行成功但无 CSV 输出：status=%s, error=%s",
                    result.get("status") if isinstance(result, dict) else "?",
                    result.get("error_class") if isinstance(result, dict) else "?",
                )
            return csv_path
        except Exception as exc:
            logger.warning("[P1-3] simulate_fn 执行异常：%s\n%s", exc, traceback.format_exc())
            return ""

    return expected_dynamics, adjustable, _simulate_fn


def worker_validator(state: BioDynamicsState) -> dict[str, Any]:
    """P0-4 新增：SBML Validator Worker（修复提示词1.md §二.6）。

    在 worker_sandbox 完成后执行，对比模板仿真与 SBML 真实仿真，
    输出 validation_report（error_diff / peak_time_diff / amplification_diff）。

    深度审核报告 §3.2 双轨策略：
    - Track A：libroadrunner 可用 → 跑真实 SBML 仿真（CVODE）
    - Track B：libroadrunner 不可用 → 结构相似度评分（structural_confidence_score）

    深度审核报告 §1.2 三角色定位逻辑：
    - 严格判定 primary_ground_truth / calibration_reference / validation_oracle
    - 仿真后阶段仅当 SBML 已加载且 ID 匹配时才升级为 validation_oracle
    - 否则降级为 calibration_reference（避免角色混用）

    若 SBML 角色为 NONE（无 BIOMD* ID），跳过验证，pass=True。
    """
    from app.sbml_validator import get_sbml_validator
    from app.biomodels_client import detect_sbml_role, SBML_ROLE_NONE
    from app.metrics import get_metrics, time_worker

    dispatches = [_dispatch_for_v3_worker("worker_validator", "in_progress")]

    user_input = state.get("user_input", "")
    simulation_csv_path = state.get("simulation_csv_path", "")
    sbml_model_id = state.get("sbml_model_id", "")
    sbml_text = state.get("sbml_model_text", "")
    ode_model = state.get("ode_model", {}) or {}
    template_name = ode_model.get("template", "")

    # 深度审核报告 §1.2：严格三角色定位（含 sbml_model_id 和 has_loaded_sbml 参数）
    has_loaded_sbml = bool(sbml_text)
    role = detect_sbml_role(
        user_input,
        has_simulation_run=True,
        sbml_model_id=sbml_model_id,
        has_loaded_sbml=has_loaded_sbml,
    )

    update: dict[str, Any] = {
        "sbml_role": role,
        "agent_dispatches": dispatches,
    }

    # 若无 SBML 可用，直接跳过验证（不阻塞流水线）
    if role == SBML_ROLE_NONE or not simulation_csv_path:
        update["validation_report"] = {
            "error_diff": 0.0,
            "peak_time_diff": 0.0,
            "amplification_diff": 0.0,
            "sbml_sim_available": False,
            "method": "skipped",
            "role": role,
            "structural_confidence_score": 0.0,
            "status": "blocked",
            "pass": False,
            "details": {"reason": "no_sbml_or_csv"},
        }
        update["agent_dispatches"] = dispatches + [
            _dispatch_for_v3_worker("worker_validator", "completed", "跳过验证：无 SBML 或 CSV"),
        ]
        get_metrics().record_validation("skipped", True, 0.0)
        return update

    # 调用 SBMLValidator 做验证
    with time_worker("worker_validator"):
        try:
            validator = get_sbml_validator()
            # 时间尺度：根据模板判断
            from app.template_selector import get_simulation_time_scale
            t_end, _, _ = get_simulation_time_scale(template_name) if template_name else (120.0, 300, "min")

            if state.get("track_a_semantics") == "multi_model_no_single_target":
                report = validator.validate_multi_model(
                    simulation_csv_path=simulation_csv_path,
                    models=list(state.get("sbml_models") or []),
                    role=role,
                )
            else:
                # [P1-3] DynamicsCalibrator 集成：当 benchmark 提供期望动力学窗口时
                # 走 validate_with_calibration，自动检查峰值窗口 + 网格搜索校准。
                # calibration_enabled 仅当 expected_dynamics + adjustable_params 同时可用时生效。
                _expected_dyn, _adjustable_params, _simulate_fn = _build_calibration_inputs(state)
                if _expected_dyn and _adjustable_params and _simulate_fn is not None:
                    report = validator.validate_with_calibration(
                        user_input=user_input,
                        simulation_csv_path=simulation_csv_path,
                        sbml_model_id=sbml_model_id,
                        sbml_text=sbml_text,
                        template_name=template_name,
                        t_end=t_end,
                        upstream_species="pEGFR",
                        downstream_species="pMAPK",
                        expected_dynamics=_expected_dyn,
                        adjustable_params=_adjustable_params,
                        simulate_fn=_simulate_fn,
                        calibration_enabled=True,
                    )
                    logger.info(
                        "[P1-3] worker_validator 走 validate_with_calibration："
                        "calibration_status=%s, calibrated=%d, in_window=%s",
                        report.get("calibration_status", "skipped"),
                        len(report.get("calibrated_params", {}) or {}),
                        report.get("calibrated_in_window", False),
                    )
                else:
                    report = validator.validate(
                        user_input=user_input,
                        simulation_csv_path=simulation_csv_path,
                        sbml_model_id=sbml_model_id,
                        sbml_text=sbml_text,
                        template_name=template_name,
                        t_end=t_end,
                        upstream_species="pEGFR",
                        downstream_species="pMAPK",
                    )
            update["validation_report"] = report
            status = "completed" if report.get("pass", False) else "completed_with_warning"
            update["agent_dispatches"] = dispatches + [
                _dispatch_for_v3_worker(
                    "worker_validator", status,
                    f"SBML 验证：method={report.get('method','skipped')}, "
                    f"error_diff={report.get('error_diff') or 0:.3f}, "
                    f"structural_confidence={report.get('structural_confidence_score') or 0:.3f}, "
                    f"pass={report.get('pass', False)}",
                ),
            ]
        except Exception as exc:
            logger.warning("worker_validator 异常：%s", exc)
            update["validation_report"] = {
                "error_diff": 0.0,
                "peak_time_diff": 0.0,
                "amplification_diff": 0.0,
                "sbml_sim_available": False,
                "method": "skipped",
                "role": role,
                "structural_confidence_score": 0.0,
                "pass": False,  # IB-057 修复：validator 异常时不放行
                "details": {"reason": f"validator_exception: {exc}"},
            }
            update["agent_dispatches"] = dispatches + [
                _dispatch_for_v3_worker("worker_validator", "completed", f"验证异常：{exc}"),
            ]
            get_metrics().record_validation("exception", False, 0.0)

    # N7 缺口 2：将 N5 检测到的跨模型混用警告合并到 validation_report
    # 不阻断执行，仅作为 warning 记录到 provenance 报告
    _cross_model_warnings = state.get("cross_model_warnings") or []
    if _cross_model_warnings and isinstance(update.get("validation_report"), dict):
        update["validation_report"]["cross_model_warnings"] = list(_cross_model_warnings)
        update["validation_report"]["biomd_id_distribution"] = state.get("biomd_id_distribution") or {}

    return update


def worker_report(state: BioDynamicsState) -> dict[str, Any]:
    """预测报告生成 Worker：N8 + N9 + N10 + N11。"""
    mode = state.get("mode", "auto_standard")
    dispatches = [_dispatch_for_v3_worker("worker_report", "in_progress")]

    # N8 特征提取
    s8 = n8_scientific_features(state)
    merged = _merge_node_output(state, s8)

    # N9 + N10 实验与文献检索
    # 注意：auto_fast 模式此前跳过 N9/N10，导致用户通过前端（默认 auto_fast）输入假说时
    # 报告显示"暂无 PubMed 文献证据"和"暂无实验方案"。
    # PubMed 查询通常 <3s，不会显著影响 fast 模式性能，因此始终执行 N9/N10。
    s9 = n9_experiment_rag(merged)
    merged = _merge_node_output(merged, s9)
    s10 = n10_evidence_rag(merged)
    merged = _merge_node_output(merged, s10)

    # === TD-035 (IB-022) 修复：消费 fallback_used 死标志 ===
    # 读取 v4_agent_dispatches 中 fail_safe dispatcher 设置的 fallback_used 标志，
    # 当任一 dispatch 触发 v3 回退时（timeout/depth_exceeded/loop_detected），
    # 在最终报告的 metadata 中标记 fallback_was_used=True 并附加告警信息，
    # 使该标志可在输出中被查询（消除"死标志"，调用方不再需要自行解析 dispatches）。
    fallback_was_used = _check_fallback_used(state)
    report_metadata = merged.get("report_metadata") or {}
    if fallback_was_used:
        # 合并已有 metadata，附加 fallback 告警字段
        report_metadata = dict(report_metadata)
        report_metadata["fallback_was_used"] = True
        report_metadata["fallback_warning"] = (
            "本流程中部分 v4 Agent 调度触发 fail-safe 回退（timeout/depth_exceeded/"
            "loop_detected），已降级到 v3 流水线，请关注结果可靠性。"
        )
        logger.warning(
            "worker_report: 检测到 v4_agent_dispatches 含 fallback_used=True，"
            "已在报告 metadata 中标记 fallback_was_used=True"
        )
        merged["report_metadata"] = report_metadata

    # N11 报告渲染
    s11 = n11_scientific_report(merged)

    result = {
        **s8,
        **s9,
        **s10,
        **s11,
        "final_report": s11.get("report", {}).get("markdown", ""),
        "agent_dispatches": dispatches
        + s8.get("agent_dispatches", [])
        + s9.get("agent_dispatches", [])
        + s10.get("agent_dispatches", [])
        + s11.get("agent_dispatches", []),
    }
    # TD-035: 将 fallback_was_used 暴露到 result 顶层，便于调用方直接查询
    result["fallback_was_used"] = fallback_was_used
    if fallback_was_used and report_metadata:
        result["report_metadata"] = report_metadata
    return compress_worker_output("worker_report", result, use_llm=False)


def _check_fallback_used(state: BioDynamicsState) -> bool:
    """检查 state 中 v4_agent_dispatches 是否有任一 dispatch 触发了 fallback_used。

    TD-035 (IB-022) 修复：fail_safe.py 在 timeout/depth_exceeded/loop_detected
    场景下会将 dispatch.fallback_used 置为 True，但此前无任何节点消费该标志
    （死信号）。本函数集中解析 v4_agent_dispatches，供 worker_report 调用，
    使该标志可在最终报告中查询。

    Args:
        state: LangGraph 全局状态，读取 ``v4_agent_dispatches`` 列表。

    Returns:
        True 表示至少一个 dispatch 触发了 v3 回退；False 表示无回退或字段缺失。
    """
    dispatches = state.get("v4_agent_dispatches") or []
    if not isinstance(dispatches, list):
        return False
    for d in dispatches:
        if isinstance(d, dict) and d.get("fallback_used") is True:
            return True
    return False


# -----------------------------------------------------------------------------
# Worker 出口：步进 + 回 supervisor
# -----------------------------------------------------------------------------
def _advance_step(state: BioDynamicsState) -> dict[str, Any]:
    """Worker 完成后推进 current_step 并记录 completed_workers。"""
    current_step = state.get("current_step", 0)
    plan = state.get("execution_plan", []) or []
    completed = list(state.get("completed_workers", []) or [])
    if current_step < len(plan):
        completed.append(plan[current_step])
    return {
        "current_step": current_step + 1,
        "completed_workers": completed,
    }


# 为每个 Worker 包装一个“执行 + 步进”版本
async def _run_worker_mcp(state: BioDynamicsState) -> dict[str, Any]:
    out = await worker_mcp(state)
    out.update(_advance_step({**state, **out}))
    return out


def _run_worker_mechanism(state: BioDynamicsState) -> dict[str, Any]:
    out = worker_mechanism(state)
    out.update(_advance_step({**state, **out}))
    return out


async def _run_worker_rag(state: BioDynamicsState) -> dict[str, Any]:
    out = await worker_rag(state)
    out.update(_advance_step({**state, **out}))
    return out


def _run_worker_pkpd(state: BioDynamicsState) -> dict[str, Any]:
    out = worker_pkpd(state)
    out.update(_advance_step({**state, **out}))
    return out


def _run_worker_ode(state: BioDynamicsState) -> dict[str, Any]:
    out = worker_ode(state)
    out.update(_advance_step({**state, **out}))
    return out


def _run_worker_sandbox(state: BioDynamicsState) -> dict[str, Any]:
    out = worker_sandbox(state)
    out.update(_advance_step({**state, **out}))
    return out


def _run_worker_validator(state: BioDynamicsState) -> dict[str, Any]:
    """P0-4 新增：worker_validator 包装器。"""
    out = worker_validator(state)
    out.update(_advance_step({**state, **out}))
    return out


def _run_worker_report(state: BioDynamicsState) -> dict[str, Any]:
    out = worker_report(state)
    out.update(_advance_step({**state, **out}))
    return out


# -----------------------------------------------------------------------------
# 图组装
# -----------------------------------------------------------------------------
def build_workflow_v3() -> StateGraph:
    """构建 BioDynamics Agent v3 的 Supervisor-Worker 图。"""
    workflow = StateGraph(BioDynamicsState)

    workflow.add_node("pre_router", pre_router)
    workflow.add_node("supervisor", supervisor)
    workflow.add_node("clarification_node", clarification_node)
    workflow.add_node("worker_mcp", _run_worker_mcp)
    workflow.add_node("worker_mechanism", _run_worker_mechanism)
    workflow.add_node("worker_rag", _run_worker_rag)
    workflow.add_node("worker_pkpd", _run_worker_pkpd)
    workflow.add_node("worker_ode", _run_worker_ode)
    workflow.add_node("worker_sandbox", _run_worker_sandbox)
    workflow.add_node("worker_validator", _run_worker_validator)
    workflow.add_node("worker_report", _run_worker_report)

    # v4 Phase 1：在 pre_router 前插入 Ontology Agent hook 节点
    # 仅新增节点与边，不修改 pre_router 函数本身，不改变路由决策
    # V4_ONTOLOGY_AGENT_ENABLED=false 时 hook 返回空 dict，行为与 v3 完全一致
    workflow.add_node("ontology_hook", ontology_hook_node)

    # v4 Phase 4：在 worker_mechanism 后串联注入 3 个 P4 hook 节点
    # _pathway_planner_hook → _specialist_hook → _crosstalk_coordinator_hook
    # 所有 hook flag=false 时返回 {}，state 不被修改，行为与 v3 完全一致
    workflow.add_node("_pathway_planner_hook", pathway_planner_hook_node)
    workflow.add_node("_specialist_hook", specialist_hook_node)
    workflow.add_node("_crosstalk_coordinator_hook", crosstalk_coordinator_hook_node)

    # v4 Phase 5：在 worker_ode 后串联注入 2 个 P5 hook 节点
    # _sbml_grounder_hook → _validation_pyramid_hook
    # SBML Grounder 建立 ODE↔Reaction↔SBML↔Parameter↔PMID 五级映射链
    # Validation Pyramid 编排 Level 1→2→3→4→5 五层验证
    # 所有 hook flag=false 时返回 {}，state 不被修改，行为与 v3 完全一致
    # 注意：Level 4 Benchmark 需要仿真 metrics（N8 输出），metrics 未计算时
    #       ValidationAgent 自动将 Level 4 标记为 skipped pass=True（不阻塞）
    workflow.add_node("_sbml_grounder_hook", sbml_grounder_hook_node)
    workflow.add_node("_validation_pyramid_hook", validation_pyramid_hook_node)

    # v4 Phase 5 / Task D.2 (G2)：Calibration + Sensitivity hooks
    # _calibration_hook: 在 _validation_pyramid_hook 后注入，参数校准 + 置信区间
    # _sensitivity_hook: 在 _calibration_hook 后注入，local + sobol + morris 灵敏度分析
    # V4_CALIBRATION_AGENT_ENABLED=false 时两个 hook 均返回 {}（no-op pass-through），
    # state 不被修改，行为与 v3 完全一致
    workflow.add_node("_calibration_hook", calibration_hook_node)
    workflow.add_node("_sensitivity_hook", sensitivity_hook_node)

    # v4 Phase 6：P6 hook 节点
    # _dynamic_router_hook: 路由层注入，记录 13 Agent 调度（V4_DYNAMIC_ROUTING_ENABLED=false 时返回 {}）
    # _hypothesis_agent_hook: 在 worker_report 前注入，生成假设列表（V4_HYPOTHESIS_AGENT_ENABLED=false 时返回 {}）
    workflow.add_node("_dynamic_router_hook", dynamic_router_hook_node)
    workflow.add_node("_hypothesis_agent_hook", hypothesis_agent_hook_node)

    workflow.add_edge(START, "ontology_hook")
    # [P0-FIX v4_pathway_graph 时序] _dynamic_router_hook 从 ontology_hook 之后
    #   移到 worker_ode 之后。原位置导致 MechanismBuilder/ODEBuilder 在
    #   v4_pathway_graph / v4_reaction_ir 尚未生成时就被分派（chicken-and-egg），
    #   降级返回空 assignments/ode_system，使 V2 baseline 修复无法生效。
    #   修复后：worker_ode 完成后（v4_pathway_graph + v4_reaction_ir 已写入 state）
    #   才触发 DynamicRouter 分派，MechanismBuilder/ODEBuilder 可读取完整 v4 上下文。
    workflow.add_edge("ontology_hook", "pre_router")
    workflow.add_edge("pre_router", "supervisor")

    workflow.add_conditional_edges(
        "supervisor",
        _route_from_supervisor,
        {
            "clarification_node": "clarification_node",
            "worker_mcp": "worker_mcp",
            "worker_mechanism": "worker_mechanism",
            "worker_rag": "worker_rag",
            "worker_pkpd": "worker_pkpd",
            "worker_ode": "worker_ode",
            "worker_sandbox": "worker_sandbox",
            "worker_validator": "worker_validator",
            "worker_report": "_hypothesis_agent_hook",
            END: END,
        },
    )

    # v4 Phase 6: hypothesis hook → worker_report
    # V4_HYPOTHESIS_AGENT_ENABLED=false 时 hook 返回 {}，等价于直连 worker_report
    workflow.add_edge("_hypothesis_agent_hook", "worker_report")

    # 所有 Worker 完成后回到 Supervisor
    # 注意：worker_mechanism 后先经过 P4 hook 链再回 supervisor
    #       worker_validator 后先经过 P5 hook 链再回 supervisor
    # [RC17] 修复：P5 hook 链（SBML Grounder + Validation Pyramid + Calibration +
    #   Sensitivity）从 worker_ode 之后移到 worker_validator 之后。
    #   原因：Validation Pyramid Level 1/3/4 需要仿真结果（worker_sandbox 输出），
    #   放在 worker_ode 之后时仿真尚未执行，导致所有 Level 报"无仿真结果可用"，
    #   触发 validation_failed clarification，浪费 2 分钟超时等待。
    #   修复后执行顺序：worker_ode → supervisor → worker_sandbox → supervisor
    #   → worker_validator → P5 hooks → supervisor → worker_report
    # （hook flag=false 时返回 {}，state 不变，等价于 worker_validator → supervisor）
    for worker in WORKER_NAMES:
        if worker == "worker_mechanism":
            # worker_mechanism → _pathway_planner_hook → _specialist_hook
            # → _crosstalk_coordinator_hook → supervisor
            workflow.add_edge(worker, "_pathway_planner_hook")
            workflow.add_edge("_pathway_planner_hook", "_specialist_hook")
            workflow.add_edge("_specialist_hook", "_crosstalk_coordinator_hook")
            workflow.add_edge("_crosstalk_coordinator_hook", "supervisor")
        elif worker == "worker_ode":
            # [P0-FIX v4_pathway_graph 时序] worker_ode → _dynamic_router_hook → supervisor
            #   worker_ode 内部生成 v4_reaction_ir + v4_pathway_graph（完整版），
            #   完成后才触发 DynamicRouter 分派 MechanismBuilder/ODEBuilder，
            #   确保 v4 上下文可用，避免降级返回空 assignments/ode_system。
            workflow.add_edge(worker, "_dynamic_router_hook")
            workflow.add_edge("_dynamic_router_hook", "supervisor")
        elif worker == "worker_validator":
            # worker_validator → _sbml_grounder_hook → _calibration_hook
            # → _sensitivity_hook → supervisor
            # [RC17] P5 hook 链移到 worker_validator 之后，确保仿真结果可用
            # [RC26] _validation_pyramid_hook 移到 worker_report 之后，
            #   因为 Level 4 Benchmark 需要 N8 scientific_features 输出的 metrics，
            #   而 N8 在 worker_report 中调用。原位置（worker_validator 后）metrics
            #   尚未计算，导致 Level 4 永远报 metrics_not_computed_yet。
            workflow.add_edge(worker, "_sbml_grounder_hook")
            # Task D.2 (G2): calibration → sensitivity 在 sbml_grounder 后、
            # worker_report 前注入（它们依赖仿真结果，不依赖 validation）
            workflow.add_edge("_sbml_grounder_hook", "_calibration_hook")
            workflow.add_edge("_calibration_hook", "_sensitivity_hook")
            workflow.add_edge("_sensitivity_hook", "supervisor")
        elif worker == "worker_report":
            # [RC26] worker_report → _validation_pyramid_hook → supervisor
            # 原因：N8 scientific_features 在 worker_report 内部调用并写入 metrics，
            #   validation_pyramid 必须在 metrics 可用后才能执行 Level 4 Benchmark。
            #   flag=false 时 hook 返回 {}，行为同 v3（worker_report → supervisor）。
            workflow.add_edge(worker, "_validation_pyramid_hook")
            workflow.add_edge("_validation_pyramid_hook", "supervisor")
        else:
            workflow.add_edge(worker, "supervisor")

    # ClarificationNode 后也回到 Supervisor（等待 respond 回灌 clarification_response 后，Supervisor 会清除 pending 并继续）
    workflow.add_edge("clarification_node", "supervisor")

    return workflow


compiled_workflow_v3 = build_workflow_v3().compile(checkpointer=MemorySaver())
