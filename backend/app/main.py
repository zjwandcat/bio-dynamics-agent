# BioDynamics Agent v3 - FastAPI 入口
# 提供 CORS 配置、/api/chat 流式接口、人工干预 respond/stop 接口、知识库更新与记忆清除。
# v3 升级：默认 WORKFLOW_VERSION=v3，仅保留 v3 Supervisor-Worker 工作流。

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

from fastapi import BackgroundTasks, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.config import rerank_manager, settings
from app.graph_v3 import (
    compiled_workflow_v3,
    cleanup_clarification_events,
    set_clarification_response,
    set_clarification_stop,
)
from app.schemas import (
    ChatRequest,
    ClarificationResponseRequest,
    ClearMemoryRequest,
    StopRequest,
)
from app.supervisor import AGENT_REGISTRY_V2  # noqa: F401  # 保留以备后续 v2 兼容
from scripts.update_vector_db import update_vector_db
from app.bio_db_client import BioDBClient
from app.rag_client import RagClient
from app.rag_collections import get_rag_collections


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan：启动时清空 LangGraph MemorySaver，防止脏数据跨重启残留。"""
    try:
        checkpointer = compiled_workflow_v3.checkpointer
        if hasattr(checkpointer, "storage"):
            checkpointer.storage.clear()
        elif hasattr(checkpointer, "delete_thread"):
            # MemorySaver 无 storage 时尝试遍历清理
            pass
        logger.info("[Startup] LangGraph 上下文记忆已清空")
    except Exception as exc:
        logger.warning("[Startup] 上下文记忆清理异常（可忽略）：%s", exc)
    yield


app = FastAPI(
    title="BioDynamics Agent",
    description="将生物医学定性假说转化为 ODE 定量模型并执行仿真预测。",
    version="0.5.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"

# v3 Worker 到前端展示文案的映射
NODE_STATUS_MAP_V3 = {
    "pre_router": "v3：正在分析运行模式并生成执行计划...",
    "supervisor": "v3 Supervisor：正在调度下一个智能体...",
    "worker_mcp": "v3：正在进行 MCP 术语标准化...",
    "worker_mechanism": "v3：正在解析机制并构建知识图谱...",
    "worker_rag": "v3：正在进行知识检索 (RAG)...",
    "worker_pkpd": "v3：正在推断 PK/PD 模型...",
    "worker_ode": "v3：正在生成 ODE 仿真代码...",
    "worker_sandbox": "v3：正在执行沙箱仿真...",
    "worker_report": "v3：正在生成预测报告...",
    "clarification_node": "v3：等待用户人工干预...",
}

NODE_NAMES_V3 = set(NODE_STATUS_MAP_V3.keys())

# v3 单圈对应元信息：与 graph_v3._dispatch_for_v3_worker 的 label_map 保持一致
# target_agent 字段直接用中文 label，前端 agent_dispatch 与 agent_registry 按 name 匹配
V3_AGENT_DEFS: list[dict[str, str]] = [
    {
        "name": "v3 主管",
        "cn_label": "v3 主管",
        "description": "动态编排 Worker、触发人工干预、上下文压缩",
        "icon": "git-merge",
        "mapped_node": "supervisor",
    },
    {
        "name": "pre_router",
        "cn_label": "pre_router",
        "description": "根据运行模式（Auto Fast / Auto Standard / Manual）生成 execution_plan",
        "icon": "route",
        "mapped_node": "pre_router",
    },
    {
        "name": "MCP 术语标准化",
        "cn_label": "MCP 术语标准化",
        "description": "调用 MCP 工具标准化生物医学术语，注入定义上下文",
        "icon": "book-open",
        "mapped_node": "worker_mcp",
    },
    {
        "name": "机制解析与图谱",
        "cn_label": "机制解析与图谱",
        "description": "解析自然语言，提取生物实体与相互作用，输出网络 JSON 与知识图谱",
        "icon": "network",
        "mapped_node": "worker_mechanism",
    },
    {
        "name": "知识检索 (RAG)",
        "cn_label": "知识检索 (RAG)",
        "description": "高阶 RAG：查询重写 + 混合检索 + 重排序，提取动力学参数",
        "icon": "search",
        "mapped_node": "worker_rag",
    },
    {
        "name": "PK/PD 推断",
        "cn_label": "PK/PD 推断",
        "description": "推断给药途径、房室模型与 PK/PD 参数，支持联合用药协同分析",
        "icon": "syringe",
        "mapped_node": "worker_pkpd",
    },
    {
        "name": "ODE 方程生成",
        "cn_label": "ODE 方程生成",
        "description": "基于 RAG 真实参数生成 ODE 仿真代码",
        "icon": "code",
        "mapped_node": "worker_ode",
    },
    {
        "name": "沙箱仿真执行",
        "cn_label": "沙箱仿真执行",
        "description": "在沙箱中执行仿真代码并捕获结果，导出 CSV 与图像",
        "icon": "flask-conical",
        "mapped_node": "worker_sandbox",
    },
    {
        "name": "预测报告生成",
        "cn_label": "预测报告生成",
        "description": "汇总所有阶段输出，生成可读 Markdown 报告",
        "icon": "file-text",
        "mapped_node": "worker_report",
    },
]

# 拓扑顺序：圈圈按"pre_router → supervisor → plan"的顺序渲染
V3_AGENT_ORDER: list[str] = [
    "pre_router",
    "v3 主管",
    "MCP 术语标准化",
    "机制解析与图谱",
    "知识检索 (RAG)",
    "PK/PD 推断",
    "ODE 方程生成",
    "沙箱仿真执行",
    "预测报告生成",
]

# 用于过滤 v1 残留 dispatch（v3 圈圈白名单）
V3_AGENT_NAMES: set[str] = set(V3_AGENT_ORDER)

# worker 名称 → V3_AGENT_DEFS 中 name 字段的映射
_WORKER_NAME_TO_AGENT_NAME: dict[str, str] = {
    "worker_mcp": "MCP 术语标准化",
    "worker_mechanism": "机制解析与图谱",
    "worker_rag": "知识检索 (RAG)",
    "worker_pkpd": "PK/PD 推断",
    "worker_ode": "ODE 方程生成",
    "worker_sandbox": "沙箱仿真执行",
    "worker_report": "预测报告生成",
}


def _build_v3_registry_payload(plan: list[str]) -> list[dict[str, str]]:
    """根据本次 execution_plan 过滤 agent registry，仅保留本次会激活的圈。

    v3 架构下 supervisor 必然激活（每次调度都经过），
    pre_router 必然激活（生成 plan 本身），
    其余圈按 plan 中的 worker 名称映射。
    """
    active_names: set[str] = {"v3 主管", "pre_router"}
    for worker in plan or []:
        agent_name = _WORKER_NAME_TO_AGENT_NAME.get(worker)
        if agent_name:
            active_names.add(agent_name)

    by_name = {item["name"]: item for item in V3_AGENT_DEFS}
    payload = [by_name[name] for name in V3_AGENT_ORDER if name in active_names and name in by_name]
    return payload


def _sse_event(payload: Dict[str, Any]) -> str:
    """将字典封装为 SSE 数据行。"""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.get("/")
async def root() -> Dict[str, str]:
    """健康检查接口。"""
    return {"status": "ok", "service": "BioDynamics Agent", "version": "v3"}


@app.post("/api/admin/update-vector-db")
async def update_vector_db_endpoint(background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """接收前端手动触发请求，在后台更新 ChromaDB 知识库。"""

    async def _background_task() -> None:
        try:
            await asyncio.to_thread(update_vector_db, RAW_DATA_DIR)
            logger.info("知识库后台更新完成")
        except Exception as exc:
            logger.error("知识库后台更新失败：%s", exc)

    background_tasks.add_task(_background_task)
    return {
        "status": "started",
        "message": "知识库更新已启动，后台处理中...",
    }


@app.get("/api/admin/rag-status")
async def rag_status() -> Dict[str, Any]:
    """返回 RAG 知识库状态：已加载数据库、各 collection 文档数、在线补充开关。"""
    # 已注册数据库列表（内置 + 在线 API）
    databases: list[dict[str, Any]] = [
        {"name": "PubMed", "type": "online_api", "collection": "biodynamics_params"},
        {"name": "KEGG", "type": "online_api", "collection": "biodynamics_mechanism"},
        {"name": "Reactome", "type": "online_api", "collection": "biodynamics_mechanism"},
        {"name": "UniProt", "type": "online_api", "collection": "biodynamics_mechanism"},
        {"name": "ChEMBL", "type": "online_api", "collection": "biodynamics_parameter"},
        {"name": "BioModels (SBML)", "type": "local_file", "collection": "biodynamics_parameter"},
        {"name": "ClinicalTrials.gov", "type": "online_api", "collection": None},
    ]

    # 各 collection 文档数
    rag_cols = get_rag_collections()
    collections: dict[str, int] = {
        "mechanism": rag_cols.count("mechanism"),
        "parameter": rag_cols.count("parameter"),
        "experiment": rag_cols.count("experiment"),
        "evidence": rag_cols.count("evidence"),
    }
    # v1 旧 collection
    rag_client = RagClient()
    legacy_count = 0
    if rag_client.available:
        try:
            coll = rag_client._get_collection()
            if coll is not None:
                legacy_count = coll.count()
        except Exception:
            pass
    collections["legacy_params"] = legacy_count

    # 检测 data/raw/ 中用户自导入文件
    user_files: list[str] = []
    if RAW_DATA_DIR.exists():
        for f in RAW_DATA_DIR.iterdir():
            if f.is_file() and f.suffix.lower() in {".txt", ".md", ".json", ".xml", ".sbml", ".csv"}:
                user_files.append(f.name)
    if user_files:
        databases.append({"name": "其他（用户导入）", "type": "user_import", "files": user_files})

    return {
        "databases": databases,
        "collections": collections,
        "online_fallback_enabled": settings.RAG_ONLINE_FALLBACK,
        "online_fallback_threshold": settings.RAG_ONLINE_FALLBACK_THRESHOLD,
    }


def _derive_provider_name(base_url: str, default_name: str = "OpenAI-Compatible") -> str:
    """从 base_url 推断供应商名称，用于前端展示。"""
    url_lower = base_url.lower()
    if "siliconflow" in url_lower:
        return "SiliconFlow"
    if "openrouter" in url_lower:
        return "OpenRouter"
    if "bigmodel" in url_lower or "zhipu" in url_lower:
        return "智谱 BigModel"
    if "openai" in url_lower or "api.openai.com" in url_lower:
        return "OpenAI"
    return default_name


def _get_active_embedding_model() -> tuple[str, str]:
    """根据 EMBEDDING_PROVIDER 返回 (provider, model)。"""
    provider = settings.EMBEDDING_PROVIDER.lower()
    if provider == "openrouter":
        return "OpenRouter", settings.OPENROUTER_EMBEDDING_MODEL
    if provider == "siliconflow":
        return "SiliconFlow", settings.SILICONFLOW_EMBEDDING_MODEL
    if provider == "xfyun":
        return "XfyunMaas", settings.XFYUN_MAAS_EMBEDDING_MODEL
    if provider == "local":
        return "Local", settings.EMBEDDING_MODEL
    return "OpenAI", settings.EMBEDDING_MODEL


@app.get("/api/models/status")
async def models_status() -> Dict[str, Any]:
    """返回当前使用的大模型供应商与模型名，供前端展示。"""
    embedding_provider, embedding_model_name = _get_active_embedding_model()
    llm_provider = _derive_provider_name(settings.OPENAI_BASE_URL, "Primary LLM")
    backup_provider = (
        _derive_provider_name(settings.BACKUP_BASE_URL, "Backup LLM")
        if settings.BACKUP_MODEL
        else None
    )

    rerank_candidates: list[dict[str, Any]] = []
    if rerank_manager is not None:
        for cand in rerank_manager.candidates:
            rerank_candidates.append({
                "provider": cand.provider,
                "model": cand.model,
                "display_name": cand.display_name,
            })

    return {
        "llm": {
            "provider": llm_provider,
            "model": settings.OPENAI_MODEL,
            "base_url": settings.OPENAI_BASE_URL,
        },
        "backup_llm": (
            {
                "provider": backup_provider,
                "model": settings.BACKUP_MODEL,
                "base_url": settings.BACKUP_BASE_URL,
            }
            if settings.BACKUP_MODEL
            else None
        ),
        "embedding": {
            "provider": embedding_provider,
            "model": embedding_model_name,
        },
        "rerank": {
            "provider": settings.RERANK_PROVIDER,
            "selection_mode": settings.RERANK_SELECTION_MODE,
            "provider_priority": settings.RERANK_PROVIDERS,
            "candidates": rerank_candidates,
        },
    }


@app.post("/api/chat/clear-memory")
async def clear_memory(request: ClearMemoryRequest) -> Dict[str, Any]:
    """清空指定 thread_id 的内存级短期记忆。"""
    try:
        compiled_workflow_v3.checkpointer.delete_thread(request.thread_id)
    except Exception as exc:
        logger.warning("删除 thread 记忆失败（可能不存在）：%s", exc)
    cleanup_clarification_events(request.thread_id)
    return {
        "status": "ok",
        "thread_id": request.thread_id,
        "message": "短期记忆已清空",
    }


@app.post("/api/chat/respond")
async def respond_to_clarification(request: ClarificationResponseRequest) -> Dict[str, Any]:
    """接收用户在环路中的干预回答，并唤醒对应的 clarification_node。"""
    response = request.clarification_response.model_dump()
    set_clarification_response(request.thread_id, response)
    return {"status": "ok", "thread_id": request.thread_id}


@app.post("/api/chat/stop")
async def stop_generation(request: StopRequest) -> Dict[str, Any]:
    """终止指定 thread_id 的当前生成任务（在 clarification 等待时唤醒并结束）。"""
    set_clarification_stop(request.thread_id)
    return {"status": "ok", "thread_id": request.thread_id}


@app.post("/api/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    """接收用户假说，流式返回 v3 Supervisor-Worker 各节点执行状态与最终结果。"""
    return StreamingResponse(
        _v3_event_stream(request),
        media_type="text/event-stream",
    )


def _v3_event_stream(request: ChatRequest):  # type: ignore[no-untyped-def]
    """v3 工作流事件流（Supervisor-Worker 动态编排）。"""

    async def event_stream():
        thread_id = request.thread_id
        initial_state: Dict[str, Any] = {
            "user_input": request.user_input,
            "thread_id": thread_id,
            "mode": request.mode,
            "manual_modules": request.manual_modules or [],
            "retry_count": 0,
            "messages": [],
            "stop_requested": False,
            # 显式重置所有结构化数据字段，阻断跨请求数据污染
            "network_json": {},
            "mcp_term_definitions": [],
            "mcp_term_map": {},
            "mcp_tool_calls": [],
            "mcp_tokens_saved": 0,
            "mcp_rewritten_query": "",
            "raw_cache": {},
            "drug_candidates": [],
            "simulation_csv_path": "",
            "rag_retrieved_params": [],
            "rag_selected_params": {},
            "rag_fallback": False,
            "rag_summary": "",
            "rag_hit_rate": 0.0,
            "rag_insights": {},
            "species_context": "",
            "pkpd_profile": {},
            "drug_regimen": [],
            "clinical_trial_info": [],
            "combination_index": {},
            "synergy_assessment": "",
            "dose_response_data": {},
            "ic50": 0.0,
            "ic90": 0.0,
            "hed": 0.0,
            "execution_result": {},
            "error_class": "none",
            "knowledge_graph": {},
            "parameters": {},
            "ode_model": {},
            "entities": [],
            "mechanism": {},
            "metrics": {},
            "feature_metadata": {},
            "confidence": 0.0,
            "experiment_protocols": [],
            "paper_evidence": [],
            "agent_dispatches": [],
            "sandbox_failure_reason": "",
        }
        latest_token_usage: Dict[str, int] | None = None
        mcp_tokens_saved: int = 0
        clarification_emitted = False
        registry_emitted = False

        try:
            # 在流开始时下发当前使用的模型名，供前端展示真实模型而非硬编码占位
            yield _sse_event(
                {"event": "config", "data": {"model_name": settings.OPENAI_MODEL}}
            )
            async for event in compiled_workflow_v3.astream_events(
                initial_state,
                {"configurable": {"thread_id": thread_id}},
                version="v2",
            ):
                event_name = event.get("event", "")
                event_chain_name = event.get("name", "")
                metadata = event.get("metadata", {}) or {}
                node_name = metadata.get("langgraph_node")
                is_actual_node = event_chain_name in NODE_NAMES_V3

                if event_name == "on_chain_start" and is_actual_node:
                    status_text = NODE_STATUS_MAP_V3.get(node_name, f"v3：正在执行 {node_name}...")
                    yield _sse_event({"event": "node_start", "data": status_text})
                    yield _sse_event(
                        {
                            "event": "workflow_v3_state",
                            "data": {
                                "current_node": node_name,
                                "status": "running",
                                "mode": request.mode,
                            },
                        }
                    )
                    continue

                if event_name != "on_chain_end" or not is_actual_node:
                    continue

                output = event.get("data", {}).get("output", {})
                if not output:
                    output = event.get("data", {})

                # 累计 Token 使用量
                if isinstance(output, dict) and output.get("token_usage"):
                    latest_token_usage = output["token_usage"]

                # 在 pre_router 拿到 execution_plan 之后下发按 plan 过滤的 agent_registry
                # 让前端只看到本次会激活的圈；仅下发一次
                if (
                    node_name == "pre_router"
                    and not registry_emitted
                    and isinstance(output, dict)
                    and output.get("execution_plan") is not None
                ):
                    registry_payload = _build_v3_registry_payload(output.get("execution_plan") or [])
                    if registry_payload:
                        registry_emitted = True
                        yield _sse_event(
                            {
                                "event": "agent_registry",
                                "data": registry_payload,
                            }
                        )

                # 发射智能体调度事件
                # 过滤 v1 风格 dispatch（v3 worker 内部仍可能调用 v1 节点 node1_parse_network 等
                # 触发 dispatch_for_node，target_agent 为英文 "Mechanism Analysis Agent" 等，
                # 这些不是 v3 圈圈的合法名字，透传给前端会造成"圈圈乱入"）
                if isinstance(output, dict):
                    for dispatch in output.get("agent_dispatches", []) or []:
                        if not isinstance(dispatch, dict):
                            continue
                        if dispatch.get("target_agent") not in V3_AGENT_NAMES:
                            continue
                        yield _sse_event({"event": "agent_dispatch", "data": dispatch})

                # 人工干预事件：仅发射一次
                if isinstance(output, dict) and output.get("clarification_request") and not clarification_emitted:
                    clarification_emitted = True
                    yield _sse_event(
                        {
                            "event": "clarification_needed",
                            "data": output["clarification_request"],
                        }
                    )

                # 人工干预已被消费，通知前端关闭对话框
                if isinstance(output, dict) and output.get("clarification_resolved"):
                    yield _sse_event({"event": "clarification_resolved", "data": ""})
                    clarification_emitted = False

                # 累计 MCP Token 节省量
                if isinstance(output, dict) and output.get("mcp_tokens_saved"):
                    mcp_tokens_saved = max(mcp_tokens_saved, int(output["mcp_tokens_saved"]))

                # v4 Phase 6: 假设生成完成事件（前端可不订阅）
                # Hypothesis Agent hook 输出 v4_hypothesis_generated=True 时发射
                if isinstance(output, dict) and output.get("v4_hypothesis_generated"):
                    yield _sse_event({
                        "event": "v4_hypothesis_generated",
                        "data": {
                            "hypothesis_count": len(output.get("v4_hypothesis_list", [])),
                            "hypothesis_list": output.get("v4_hypothesis_list", []),
                        },
                    })

                # 各 Worker 输出映射到前端事件
                async for sse in _emit_worker_outputs(node_name, output):
                    yield sse

        except Exception as exc:
            logger.exception("v3 工作流执行异常")
            yield _sse_event({"event": "error", "data": f"工作流执行异常：{exc}"})
        finally:
            if latest_token_usage:
                payload = dict(latest_token_usage)
                if mcp_tokens_saved > 0:
                    payload["mcp_tokens_saved"] = mcp_tokens_saved
                payload["model_name"] = settings.OPENAI_MODEL
                yield _sse_event({"event": "token_usage", "data": payload})
            yield _sse_event({"event": "end", "data": ""})
            cleanup_clarification_events(thread_id)

    return event_stream()


async def _emit_worker_outputs(node_name: str, output: Dict[str, Any]):
    """将 Worker 节点的输出转换为前端 SSE 事件（异步生成器）。"""
    if not isinstance(output, dict):
        return

    def _yield(event: str, data: Any):
        return _sse_event({"event": event, "data": data})

    if node_name == "worker_mcp":
        definitions = output.get("mcp_term_definitions") or []
        if definitions:
            yield _yield(
                "mcp_term_definitions",
                {
                    "definitions": definitions,
                    "tokens_saved": output.get("mcp_tokens_saved", 0),
                    "rewritten_query": output.get("mcp_rewritten_query", ""),
                },
            )
        tool_calls = output.get("mcp_tool_calls") or []
        for tc in tool_calls:
            yield _yield("mcp_tool_call", tc)

    elif node_name == "worker_mechanism":
        kg = output.get("knowledge_graph") or {}
        if kg:
            yield _yield(
                "knowledge_graph",
                {
                    "node_count": kg.get("node_count", 0),
                    "edge_count": kg.get("edge_count", 0),
                    "is_acyclic": kg.get("is_acyclic", True),
                    "topology_signature": kg.get("topology_signature", ""),
                },
            )
        mechanism = output.get("mechanism") or {}
        if mechanism:
            yield _yield(
                "execution_log",
                (
                    f"规划：{mechanism.get('simulation_type', '?')} / "
                    f"模板 {mechanism.get('template', '?')}"
                ),
            )

    elif node_name == "worker_rag":
        rag_insights = output.get("rag_insights")
        if rag_insights:
            yield _yield("rag_insights", rag_insights)
            # 在线补充已触发时，通知前端
            if rag_insights.get("online_fallback_enabled"):
                yield _yield(
                    "rag_online_fallback",
                    {
                        "triggered": True,
                        "hit_rate": output.get("rag_hit_rate", 0.0),
                        "message": "本地 RAG 命中不足，已自动查询 KEGG/Reactome/UniProt/ChEMBL 补充",
                    },
                )
        yield _yield(
            "rag_ready",
            {
                "summary": output.get("rag_summary", ""),
                "fallback": output.get("rag_fallback", False),
                "hit_rate": output.get("rag_hit_rate", 0.0),
            },
        )

    elif node_name == "worker_pkpd":
        pkpd_profile = output.get("pkpd_profile") or {}
        if pkpd_profile:
            yield _yield("pkpd_profile", pkpd_profile)
        drug_regimen = output.get("drug_regimen") or []
        if drug_regimen:
            yield _yield("drug_regimen", drug_regimen)

    elif node_name == "worker_ode":
        ode_model = output.get("ode_model") or {}
        rule_violations = ode_model.get("rule_violations") or []
        if rule_violations:
            yield _yield("rule_violations", rule_violations)
        code = ode_model.get("code", "")
        if code:
            yield _yield("code_generated", code)

    elif node_name == "worker_sandbox":
        stdout = output.get("stdout_stderr", "")
        if stdout:
            yield _yield("execution_log", stdout)
        image_base64 = output.get("image_base64")
        if image_base64:
            yield _yield("image_ready", image_base64)
        csv_path = output.get("simulation_csv_path")
        if csv_path:
            yield _yield("simulation_csv", csv_path)
        dose_response_data = output.get("dose_response_data")
        if dose_response_data:
            yield _yield(
                "dose_response",
                {
                    **dose_response_data,
                    "ic50": output.get("ic50"),
                    "ic90": output.get("ic90"),
                    "hed": output.get("hed"),
                },
            )

    elif node_name == "worker_report":
        metrics = output.get("metrics") or {}
        if metrics:
            yield _yield("metrics", metrics)
        protocols = output.get("experiment_protocols") or []
        if protocols:
            yield _yield("experiment_protocols", protocols)
        evidence = output.get("paper_evidence") or []
        if evidence:
            yield _yield("paper_evidence", evidence)
        report = output.get("report") or {}
        if report.get("markdown"):
            yield _yield("report", report)
            yield _yield("report_ready", report.get("markdown", ""))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
