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

_CLARIFICATION_TIMEOUT_SECONDS = 600  # 10 分钟无响应则自动取消


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
        "激活", "inhibit", "抑制", "促进", "phosphorylat", "下调", "上调",
        "downstream", "upstream", "cascade", "级联",
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
        logger.warning("thread=%s clarification 等待超时，自动取消", thread_id)
        return {"stop_requested": True, "pending_clarification": None, "clarification_request": None}

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
        return {
            "pkpd_profile": {},
            "drug_regimen": [],
            "agent_dispatches": [_dispatch_for_v3_worker("worker_pkpd", "completed", "Fast 模式跳过 PK/PD")],
        }

    # 复用 v1 node1_6（已能处理无药物候选时跳过）
    result = node1_6_pkpd_inference(state)
    return result


def worker_ode(state: BioDynamicsState) -> dict[str, Any]:
    """ODE 方程生成 Worker。"""
    dispatches = [_dispatch_for_v3_worker("worker_ode", "in_progress")]

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

    # 提取通路类别（从 mechanism.pathway 或 reaction_ir.pathway_class）
    mechanism = state.get("mechanism", {}) or {}
    pathway_class = mechanism.get("pathway", "") or reaction_ir.get("pathway_class", "")
    if not pathway_class:
        # 默认 EGFR_RTK（MVP 阶段保守选择）
        pathway_class = "EGFR_RTK"
        logger.info("_pathway_graph_hook: pathway_class 未指定，默认 EGFR_RTK")

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

    # 提取通路类别
    mechanism = state.get("mechanism", {}) or {}
    pathway_class = mechanism.get("pathway", "") or reaction_ir.get("pathway_class", "")
    if not pathway_class:
        pathway_class = "EGFR_RTK"

    pathway_graph = state.get("v4_pathway_graph")

    # 渲染 v4 ODE 代码
    try:
        renderer = ODERendererV2()
        ode_code = renderer.render(
            pathway_class=pathway_class,
            reaction_ir=reaction_ir,
            pathway_graph=pathway_graph,
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

    ode_system = {
        "pathway_class": pathway_class,
        "template_name": "oscillatory_feedback.j2" if pathway_class in {
            "p53_signaling", "NF_kB", "TGF_beta", "JAK_STAT"
        } else "bistable_switch.j2" if pathway_class in {"Apoptosis", "Cell_Cycle"} else "oscillatory_feedback.j2",
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

    # 从 ODE model 获取代码
    ode_model = state.get("ode_model", {}) or {}
    code = ode_model.get("code", "")
    if not code:
        # 兼容 v1 的 python_code
        code = state.get("python_code", "")

    retry_count = 0
    last_result: dict[str, Any] | None = None
    last_error = ""

    while retry_count <= max_retries:
        result = execute_simulation_code_v2(code)
        last_result = result
        if result.get("error_class") == ERR_NONE and result.get("status") == "success":
            break
        last_error = result.get("stdout_stderr", "")
        retry_count += 1
        if retry_count <= max_retries:
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
                    new_ode_result = n6_ode_generator(rewrite_state)
                    new_ode_model = new_ode_result.get("ode_model", {}) or {}
                    new_code = new_ode_model.get("code", "")
                    if new_code:
                        code = new_code
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
        "stdout_stderr": last_result.get("stdout_stderr", "") if last_result else "",
        "image_base64": last_result.get("image_base64", "") if last_result else "",
        "simulation_csv_path": last_result.get("simulation_csv_path", "") if last_result else "",
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
            "pass": True,
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
                    f"error_diff={report.get('error_diff',0):.3f}, "
                    f"structural_confidence={report.get('structural_confidence_score',0):.3f}, "
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

    return update


def worker_report(state: BioDynamicsState) -> dict[str, Any]:
    """预测报告生成 Worker：N8 + N9 + N10 + N11。"""
    mode = state.get("mode", "auto_standard")
    dispatches = [_dispatch_for_v3_worker("worker_report", "in_progress")]

    # N8 特征提取
    s8 = n8_scientific_features(state)
    merged = _merge_node_output(state, s8)

    if mode != "auto_fast":
        # N9 + N10 实验与文献检索
        s9 = n9_experiment_rag(merged)
        merged = _merge_node_output(merged, s9)
        s10 = n10_evidence_rag(merged)
        merged = _merge_node_output(merged, s10)
    else:
        s9 = {"experiment_protocols": []}
        s10 = {"paper_evidence": []}

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
    return compress_worker_output("worker_report", result, use_llm=False)


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
    workflow.add_edge("ontology_hook", "_dynamic_router_hook")
    workflow.add_edge("_dynamic_router_hook", "pre_router")
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
    #       worker_ode 后先经过 P5 hook 链再回 supervisor
    # （hook flag=false 时返回 {}，state 不变，等价于直连 supervisor）
    for worker in WORKER_NAMES:
        if worker == "worker_mechanism":
            # worker_mechanism → _pathway_planner_hook → _specialist_hook
            # → _crosstalk_coordinator_hook → supervisor
            workflow.add_edge(worker, "_pathway_planner_hook")
            workflow.add_edge("_pathway_planner_hook", "_specialist_hook")
            workflow.add_edge("_specialist_hook", "_crosstalk_coordinator_hook")
            workflow.add_edge("_crosstalk_coordinator_hook", "supervisor")
        elif worker == "worker_ode":
            # worker_ode → _sbml_grounder_hook → _validation_pyramid_hook
            # → _calibration_hook → _sensitivity_hook → supervisor
            # P5 hook 链：建立五级映射链 + 编排五层验证 + 参数校准 + 灵敏度分析
            # flag=false 时 hook 返回 {}，等价于 worker_ode → supervisor
            workflow.add_edge(worker, "_sbml_grounder_hook")
            workflow.add_edge("_sbml_grounder_hook", "_validation_pyramid_hook")
            # Task D.2 (G2): calibration → sensitivity 在 validation_pyramid 后、
            # hypothesis_agent_hook 前注入（hypothesis 在 worker_report 分支由
            # supervisor 条件边触发，此处 calibration/sensitivity 先回 supervisor）
            workflow.add_edge("_validation_pyramid_hook", "_calibration_hook")
            workflow.add_edge("_calibration_hook", "_sensitivity_hook")
            workflow.add_edge("_sensitivity_hook", "supervisor")
        else:
            workflow.add_edge(worker, "supervisor")

    # ClarificationNode 后也回到 Supervisor（等待 respond 回灌 clarification_response 后，Supervisor 会清除 pending 并继续）
    workflow.add_edge("clarification_node", "supervisor")

    return workflow


compiled_workflow_v3 = build_workflow_v3().compile(checkpointer=MemorySaver())
