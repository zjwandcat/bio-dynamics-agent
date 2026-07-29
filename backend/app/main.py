# BioDynamics Agent v3 - FastAPI 入口
# 提供 CORS 配置、/api/chat 流式接口、人工干预 respond/stop 接口、知识库更新与记忆清除。
# v3 升级：默认 WORKFLOW_VERSION=v3，仅保留 v3 Supervisor-Worker 工作流。

import asyncio
import csv as csv_module
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

# 清除系统代理环境变量：避免 requests/httpx 尝试连接不可用的本地代理（如 Clash 127.0.0.1:7897）
# 当代理软件未运行时，所有外部 API 调用（PubMed/Rerank/在线 RAG）都会因 ProxyError 失败
# 设置 NO_PROXY=* 替代清除，兼容性更好（不影响其他可能依赖代理的场景）
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from app.config import rerank_manager, settings
from app.graph_v3 import (
    compiled_workflow_v3,
    cleanup_clarification_events,
    set_clarification_response,
    set_clarification_stop,
)
from app.logging_config import setup_logging
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
from app.v4_endpoints import router as v4_router  # v4 REST 端点（6 个端点）


logger = logging.getLogger(__name__)

# Task G.2：在导入阶段统一配置 JSON 结构化日志，确保所有后续日志（含
# uvicorn / langgraph 子 logger）均按统一格式输出。setup_logging 幂等，
# 重复调用不会叠加 handler。
setup_logging(level=settings.LOG_LEVEL, json_output=settings.LOG_JSON)


# =============================================================================
# Task 21 Step 2: Scientific Alignment 后处理（chat 流集成）
# =============================================================================
# pathway_class → canonical 名映射（与 PATHWAY_REGISTRY / benchmark_runner._PATHWAY_CLASS_TO_CANONICAL 对齐）
# P0-SA-1 修复：键名必须与 ontology/pathway_registry.py PATHWAY_REGISTRY 完全一致
_SA_PATHWAY_TO_CANONICAL: dict[str, str] = {
    "EGFR_RTK": "egfr",
    "MAPK_ERK": "mapk",
    "PI3K_AKT_mTOR": "pi3k_akt_mtor",
    "JAK_STAT": "jak_stat",
    "TGF_BETA": "tgf_beta",
    "WNT": "wnt",
    "p53": "p53",
    "NF_KB": "nf_kappa_b",
    "APOPTOSIS": "apoptosis",
    "CELL_CYCLE": "cell_cycle",
}


def _sa_flatten_metrics(metrics: dict) -> dict[str, float]:
    """将 nested metrics 扁平化为 {species}_{field} → value（复用 orchestrator 逻辑）。"""
    flat: dict[str, float] = {}
    if not isinstance(metrics, dict):
        return flat
    for species, fields in metrics.items():
        if not isinstance(fields, dict):
            try:
                flat[str(species)] = float(fields)
            except (TypeError, ValueError):
                pass
            continue
        for field_name, value in fields.items():
            try:
                flat[f"{species}_{field_name}"] = float(value)
            except (TypeError, ValueError):
                pass
    return flat


def _sa_extract_node_names(knowledge_graph: dict, entities: list | None = None) -> list[str]:
    """从 knowledge_graph 提取节点名列表（复用 orchestrator 逻辑）。"""
    nodes: list[str] = []
    if not isinstance(knowledge_graph, dict):
        return nodes
    kg_nodes = knowledge_graph.get("nodes") or []
    if isinstance(kg_nodes, list):
        for n in kg_nodes:
            if isinstance(n, dict):
                name = n.get("name") or n.get("id") or ""
                if name:
                    nodes.append(str(name))
            elif isinstance(n, str):
                nodes.append(n)
    if not nodes and entities:
        for e in entities:
            if isinstance(e, dict):
                name = e.get("name") or e.get("text") or ""
                if name:
                    nodes.append(str(name))
    return nodes


def _sa_extract_cited_pmids(paper_evidence: list) -> list[str]:
    """从 paper_evidence 提取 PMID 列表。"""
    pmids: list[str] = []
    if not isinstance(paper_evidence, list):
        return pmids
    for ev in paper_evidence:
        if not isinstance(ev, dict):
            continue
        pmid = ev.get("pmid") or ev.get("PMID") or ""
        if pmid:
            pmids.append(str(pmid))
    return pmids


async def _run_scientific_alignment_postprocess(
    pathway_class: str,
    metrics: dict,
    knowledge_graph: dict,
    paper_evidence: list,
    report_markdown: str,
    simulation_csv_path: str = "",
):
    """Scientific Alignment 后处理：Consistency Checker + Critic + Multi-dim Confidence + BioModels Calibration。

    在 worker_report 完成后运行，受 SA Feature Flags 守护。
    发射 SSE 事件：sa_consistency_report / sa_critic_report / sa_multi_dim_confidence /
                   sa_biomodels_calibration。
    任何异常被上层 try/except 捕获，不影响主流程。
    """
    canonical_name = _SA_PATHWAY_TO_CANONICAL.get(pathway_class, "")
    if not canonical_name:
        # Sprint 6 修复：MULTI:EGFR_RTK+MAPK_ERK 多通路格式解析
        # 取第一个通路作为 canonical name（主通路驱动 SA 后处理）
        if pathway_class and pathway_class.startswith("MULTI:"):
            _sub_pathways = pathway_class[6:].split("+")
            if _sub_pathways:
                canonical_name = _SA_PATHWAY_TO_CANONICAL.get(_sub_pathways[0], "")
        if not canonical_name:
            logger.info("[SA] 通路 %s 无 canonical 映射，跳过后处理", pathway_class)
            return
        logger.info(
            "[SA] MULTI 通路 %s → 使用主通路 %s 的 canonical (%s)",
            pathway_class, _sub_pathways[0], canonical_name,
        )

    metrics_flat = _sa_flatten_metrics(metrics)
    extracted_nodes = _sa_extract_node_names(knowledge_graph)
    cited_pmids = _sa_extract_cited_pmids(paper_evidence)

    logger.info(
        "[SA] 后处理启动: pathway=%s canonical=%s metrics_flat=%d nodes=%d pmids=%d",
        pathway_class, canonical_name, len(metrics_flat),
        len(extracted_nodes), len(cited_pmids),
    )

    # --- 阶段 9: Consistency Checker ---
    _consistency_passed = True  # 默认通过（Flag OFF 时）
    if settings.is_sa_feature_enabled("CONSISTENCY_CHECKER"):
        try:
            from app.scientific_alignment import check_consistency
            report = check_consistency(
                pathway=canonical_name,
                simulation_metrics=metrics_flat,
            )
            _consistency_passed = report.passed
            yield _sse_event({
                "event": "sa_consistency_report",
                "data": {
                    "passed": report.passed,
                    "rules_checked": report.rules_checked,
                    "rules_evaluated": report.rules_evaluated,
                    "violation_count": len(report.violations),
                    "violations": [
                        {
                            "rule": v.rule,
                            "assertion": v.assertion,
                            "violation_label": v.violation_label,
                            "observed_values": dict(v.observed_values) if v.observed_values else {},
                            "message": v.message,
                        }
                        for v in report.violations
                    ],
                    "pathway": canonical_name,
                },
            })
            logger.info(
                "[SA] Consistency: passed=%s violations=%d",
                report.passed, len(report.violations),
            )
        except Exception as exc:
            logger.warning("[SA] Consistency Checker 异常: %s", exc)
            _consistency_passed = False
            yield _sse_event({
                "event": "sa_consistency_error",
                "data": {"message": str(exc)},
            })

    # --- Sprint 3: Consistency 硬 Gate ---
    # Flag ON 时，Consistency 违规 → 阻断后续 SA 阶段（Critic/MultiDim/Validation/Review）
    # Flag OFF 时，Consistency 仅 Warning，继续后续阶段（当前行为）
    if settings.is_sa_feature_enabled("SPRINT3_CONSISTENCY_GATE"):
        if not _consistency_passed:
            _gate_labels = [
                v.violation_label for v in report.violations
            ] if not _consistency_passed and 'report' in dir() else ["unknown"]
            yield _sse_event({
                "event": "sa_consistency_gate_failed",
                "data": {
                    "gate": "Consistency Gate",
                    "passed": False,
                    "violation_labels": _gate_labels,
                    "message": f"Consistency Gate Failed: {'; '.join(_gate_labels)}",
                    "pathway": canonical_name,
                    "blocked_stages": ["critic", "multi_dim_confidence", "validation_rule_engine", "scientific_review"],
                },
            })
            logger.warning(
                "[SA] Sprint 3 Consistency Gate FAILED — 阻断后续 SA 阶段: %s",
                _gate_labels,
            )
            return  # 阻断后续 SA 阶段

    # --- 阶段 11: Scientific Critic Agent ---
    if settings.is_sa_feature_enabled("SCIENTIFIC_CRITIC"):
        try:
            from app.scientific_alignment import run_scientific_critic
            critic_report = run_scientific_critic(
                pathway=canonical_name,
                extracted_nodes=extracted_nodes,
                simulation_metrics=metrics_flat,
                biomodels_report=None,
                cited_pmids=cited_pmids,
            )
            yield _sse_event({
                "event": "sa_critic_report",
                "data": {
                    "overall_status": critic_report.overall_status,
                    "findings_count": len(critic_report.findings),
                    "findings": [
                        {
                            "category": f.category,
                            "severity": f.severity,
                            "finding": f.finding,
                            "evidence": f.evidence,
                            "suggestion": f.suggestion,
                        }
                        for f in critic_report.findings
                    ],
                    "pathway": canonical_name,
                },
            })
            logger.info(
                "[SA] Critic: overall_status=%s findings=%d",
                critic_report.overall_status, len(critic_report.findings),
            )
        except Exception as exc:
            logger.warning("[SA] Critic Agent 异常: %s", exc)
            yield _sse_event({
                "event": "sa_critic_error",
                "data": {"message": str(exc)},
            })

    # --- 阶段 12: Multi-dim Confidence ---
    if settings.is_sa_feature_enabled("MULTI_DIM_CONFIDENCE"):
        try:
            from app.scientific_alignment import compute_multi_dim_confidence
            md_report = compute_multi_dim_confidence(
                pathway=canonical_name,
                seven_axis_report=None,     # 7 轴报告未在 chat 流中计算
                parameter_report=None,       # 参数报告未在 chat 流中计算
                consistency_report=None,     # 已在阶段 9 计算，此处传 None 让其降级
                critic_report=None,          # Critic 报告对象类型不匹配，传 None
                cited_pmids=cited_pmids,
            )
            yield _sse_event({
                "event": "sa_multi_dim_confidence",
                "data": {
                    "overall_confidence": md_report.overall_confidence,
                    "axes": [
                        {
                            "axis_name": a.axis_name,
                            "score": a.score,
                            "status": a.status,
                            "sub_scores": dict(a.sub_scores) if a.sub_scores else {},
                        }
                        for a in md_report.axes
                    ],
                    "pathway": canonical_name,
                },
            })
            logger.info(
                "[SA] Multi-dim Confidence: overall=%s",
                md_report.overall_confidence,
            )
        except Exception as exc:
            logger.warning("[SA] Multi-dim Confidence 异常: %s", exc)
            yield _sse_event({
                "event": "sa_multi_dim_error",
                "data": {"message": str(exc)},
            })

    # --- Sprint 3: Validation Rule Engine ---
    # 100% Rule 驱动：Mass conservation / Peak time / Peak ordering / Evidence count
    _validation_passed = True
    if settings.is_sa_feature_enabled("SPRINT3_CONSISTENCY_GATE"):
        try:
            from app.scientific_alignment.validation_rule_engine import run_validation_rules
            from app.scientific_alignment.canonical_loader import load_canonical

            # P0-SA-2 修复：load_canonical() 返回 CanonicalReference dataclass，不是 dict
            # canonical_timeline 在 raw YAML dict 中；consistency_rules 用 raw 保持 list[dict] 格式
            try:
                _canonical_ref = load_canonical(canonical_name)
                _raw = _canonical_ref.raw or {}
                _canonical_timeline = _raw.get("canonical_timeline", [])
                _consistency_rules = _raw.get("consistency_rules", [])
            except Exception:
                _canonical_timeline = []
                _consistency_rules = []

            _val_report = run_validation_rules(
                pathway=canonical_name,
                simulation_metrics=metrics_flat,
                canonical_timeline=_canonical_timeline,
                consistency_rules=_consistency_rules,
                evidence_count=len(paper_evidence),
            )
            _validation_passed = _val_report.overall_passed

            yield _sse_event({
                "event": "sa_validation_rule_engine",
                "data": {
                    "enabled": _val_report.enabled,
                    "skipped": _val_report.skipped,
                    "overall_passed": _val_report.overall_passed,
                    "passed_count": _val_report.passed_count,
                    "failed_count": _val_report.failed_count,
                    "results": [
                        {
                            "rule_name": r.rule_name,
                            "passed": r.passed,
                            "message": r.message,
                            "expected": r.expected,
                            "actual": r.actual,
                        }
                        for r in _val_report.results
                    ],
                    "pathway": canonical_name,
                },
            })
            logger.info(
                "[SA] Sprint 3 Validation Rule Engine: passed=%s (%d/%d)",
                _val_report.overall_passed,
                _val_report.passed_count,
                _val_report.passed_count + _val_report.failed_count,
            )
        except Exception as exc:
            logger.warning("[SA] Sprint 3 Validation Rule Engine 异常: %s", exc)
            yield _sse_event({
                "event": "sa_validation_error",
                "data": {"message": str(exc)},
            })

    # --- Sprint 3: Scientific Review ---
    # 100% Rule 驱动：Overall Scientific Score（非 LLM "Looks reasonable"）
    if settings.is_sa_feature_enabled("SPRINT3_CONSISTENCY_GATE"):
        try:
            from app.scientific_alignment.scientific_review import run_scientific_review

            # P0-SA-2 修复：load_canonical() 返回 CanonicalReference dataclass，不是 dict
            # canonical_models 是 dataclass 字段 List[str]；canonical_timeline 在 raw YAML dict 中
            try:
                _canonical_ref = load_canonical(canonical_name)
                _raw = _canonical_ref.raw or {}
                _canonical_timeline = _raw.get("canonical_timeline", [])
                _canonical_models = _canonical_ref.canonical_models
            except Exception:
                _canonical_timeline = []
                _canonical_models = []
            _biomodels_matched = len(_canonical_models) > 0

            _review_report = run_scientific_review(
                pathway=canonical_name,
                simulation_metrics=metrics_flat,
                consistency_passed=_consistency_passed,
                validation_passed=_validation_passed,
                evidence_count=len(paper_evidence),
                biomodels_matched=_biomodels_matched,
                canonical_timeline=_canonical_timeline,
            )

            yield _sse_event({
                "event": "sa_scientific_review",
                "data": {
                    "enabled": _review_report.enabled,
                    "skipped": _review_report.skipped,
                    "overall_score": _review_report.overall_score,
                    "overall_passed": _review_report.overall_passed,
                    "summary": _review_report.summary,
                    "items": [
                        {
                            "name": item.name,
                            "passed": item.passed,
                            "score": item.score,
                            "reason": item.reason,
                        }
                        for item in _review_report.items
                    ],
                    "pathway": canonical_name,
                },
            })
            logger.info(
                "[SA] Sprint 3 Scientific Review: score=%s/10 passed=%s",
                _review_report.overall_score,
                _review_report.overall_passed,
            )
        except Exception as exc:
            logger.warning("[SA] Sprint 3 Scientific Review 异常: %s", exc)
            yield _sse_event({
                "event": "sa_scientific_review_error",
                "data": {"message": str(exc)},
            })

    # --- Task F: BioModels Parameter Calibration ---
    # Agent ODE vs RoadRunner vs BioModels 对比
    # 输出 Peak Time / Peak Value / RMSE / Correlation 差异
    # 差异超阈值 → needs_calibration=True
    if settings.is_sa_feature_enabled("BIOMODELS_CALIBRATION"):
        if not simulation_csv_path:
            logger.info("[SA] Task F BioModels Calibration 跳过：无 simulation_csv_path")
            yield _sse_event({
                "event": "sa_biomodels_calibration_skipped",
                "data": {"reason": "no_simulation_csv", "pathway": canonical_name},
            })
        else:
            try:
                from app.scientific_alignment.biomodels_calibration import BioModelsComparator

                _comparator = BioModelsComparator()
                _calib_result = _comparator.compare(
                    agent_csv_path=simulation_csv_path,
                    pathway=pathway_class,  # 传 v4_pathway_class（含 MULTI 前缀也无妨，模块内部解析）
                    duration=120.0,
                    n_points=500,
                )

                yield _sse_event({
                    "event": "sa_biomodels_calibration",
                    "data": {
                        "pathway": _calib_result.pathway,
                        "biomodel_id": _calib_result.biomodel_id,
                        "biomodel_loaded": _calib_result.biomodel_loaded,
                        "agent_species_count": _calib_result.agent_species_count,
                        "biomodel_species_count": _calib_result.biomodel_species_count,
                        "matched_count": _calib_result.matched_count,
                        "overall_rmse": round(_calib_result.overall_rmse, 4),
                        "overall_correlation": round(_calib_result.overall_correlation, 4),
                        "needs_calibration": _calib_result.needs_calibration,
                        "error": _calib_result.error,
                        # validation_report 兼容字段（对接 _c4/_c8 期望命名）
                        "rmse": round(_calib_result.rmse, 4),
                        "correlation": round(_calib_result.correlation, 4),
                        "sbml_sim_available": _calib_result.sbml_sim_available,
                        "checksum_verified": _calib_result.checksum_verified,
                        "method": _calib_result.method,
                        "pass": _calib_result.pass_,
                        "species": [
                            {
                                "agent_species": c.agent_species,
                                "biomodel_species": c.biomodel_species,
                                "matched": c.matched,
                                "peak_time_diff_min": round(c.peak_time_diff, 2),
                                "peak_value_diff_pct": round(c.peak_value_diff_pct, 4),
                                "rmse": round(c.rmse, 4),
                                "correlation": round(c.correlation, 4),
                                "needs_calibration": c.needs_calibration,
                            }
                            for c in _calib_result.species_comparisons
                        ],
                    },
                })
                logger.info(
                    "[SA] Task F BioModels Calibration: biomodel=%s loaded=%s "
                    "matched=%d rmse=%.3f corr=%.3f needs_calibration=%s",
                    _calib_result.biomodel_id,
                    _calib_result.biomodel_loaded,
                    _calib_result.matched_count,
                    _calib_result.overall_rmse,
                    _calib_result.overall_correlation,
                    _calib_result.needs_calibration,
                )
            except Exception as exc:
                logger.warning("[SA] Task F BioModels Calibration 异常: %s", exc)
                yield _sse_event({
                    "event": "sa_biomodels_calibration_error",
                    "data": {"message": str(exc), "pathway": canonical_name},
                })


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

# 挂载 v4 REST 路由（pathways / graph / simulation / benchmark / reports / sweep）
# 这些端点与 /api/chat SSE 流独立，供前端 Scientific Workspace 直接 REST 调用。
app.include_router(v4_router)


# Task G.2：全局异常处理中间件
# 捕获所有未被路由层显式处理的异常，统一返回 500 JSON 响应并记录结构化日志，
# 避免未捕获异常以 HTML 500 traceback 形式泄露给前端。
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception on %s %s: %s", request.method, request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
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


def _v3_edge_relation(edge: Dict[str, Any]) -> str:
    """将 v3 network_json 边的 relation/interaction 字段归并为 frontend 5 类枚举。

    frontend api.ts 定义 edges[].relation ∈ {activation, inhibition,
    phosphorylation, binding, catalysis}。v3 network_json 用 interaction /
    type / mechanism 等字段表达，本函数做归并。
    """
    val = (
        edge.get("relation")
        or edge.get("interaction")
        or edge.get("type")
        or edge.get("mechanism")
        or "activation"
    )
    val_lower = str(val).lower()
    if "inhibit" in val_lower or "repress" in val_lower:
        return "inhibition"
    if "phosphor" in val_lower:
        return "phosphorylation"
    if "bind" in val_lower or "complex" in val_lower or "dimer" in val_lower:
        return "binding"
    if "cataly" in val_lower or "cleav" in val_lower or "degrad" in val_lower:
        return "catalysis"
    return "activation"


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
    """返回当前使用的大模型供应商与模型名，供前端展示。

    支持三链路容灾展示：primary LLM → backup LLM → backup2 LLM。
    若用户在前端选择了某个模型（USER_SELECTED_LLM），则按选择顺序展示。
    """
    embedding_provider, embedding_model_name = _get_active_embedding_model()
    llm_provider = _derive_provider_name(settings.OPENAI_BASE_URL, "Primary LLM")
    backup_provider = (
        _derive_provider_name(settings.BACKUP_BASE_URL, "Backup LLM")
        if settings.BACKUP_MODEL
        else None
    )
    backup2_provider = (
        _derive_provider_name(settings.BACKUP2_BASE_URL, "Backup2 LLM")
        if settings.BACKUP2_MODEL
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
        "backup2_llm": (
            {
                "provider": backup2_provider,
                "model": settings.BACKUP2_MODEL,
                "base_url": settings.BACKUP2_BASE_URL,
            }
            if settings.BACKUP2_MODEL
            else None
        ),
        "user_selected_llm": settings.USER_SELECTED_LLM or None,
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


@app.get("/api/llm/models")
async def llm_models() -> Dict[str, Any]:
    """返回所有可用的 LLM 模型列表，供前端选择 UI 渲染。

    Returns:
        models: 可选模型列表，每项含 model/provider/base_url/role 字段
        current: 当前作为 primary 的模型名
        chain: 当前容灾链路 [primary, backup, backup2]
    """
    from app.config import LLM_REGISTRY, llm

    def _role_of(model_name: str) -> str:
        if model_name == settings.OPENAI_MODEL:
            return "primary"
        if model_name == settings.BACKUP_MODEL:
            return "backup"
        if model_name == settings.BACKUP2_MODEL:
            return "backup2"
        return "unknown"

    models: list[dict[str, Any]] = []
    for model_name in (
        settings.OPENAI_MODEL,
        settings.BACKUP_MODEL,
        settings.BACKUP2_MODEL,
    ):
        if not model_name or model_name not in LLM_REGISTRY:
            continue
        base_url = (
            settings.OPENAI_BASE_URL
            if model_name == settings.OPENAI_MODEL
            else settings.BACKUP_BASE_URL
            if model_name == settings.BACKUP_MODEL
            else settings.BACKUP2_BASE_URL
        )
        models.append({
            "model": model_name,
            "provider": _derive_provider_name(base_url, model_name),
            "base_url": base_url,
            "role": _role_of(model_name),
        })

    # 当前 primary（根据 USER_SELECTED_LLM 或默认 OPENAI_MODEL）
    current = settings.USER_SELECTED_LLM or settings.OPENAI_MODEL

    # 当前容灾链路顺序
    chain: list[str] = []
    if llm.primary is not None:
        chain.append(_llm_model_name(llm.primary))
    if llm.backup is not None:
        chain.append(_llm_model_name(llm.backup))
    if llm.backup2 is not None:
        chain.append(_llm_model_name(llm.backup2))

    return {
        "models": models,
        "current": current,
        "chain": chain,
    }


def _llm_model_name(llm_instance: Any) -> str:
    """从 ChatOpenAI 实例提取 model_name，兼容不同 LangChain 版本。"""
    for attr in ("model_name", "model"):
        val = getattr(llm_instance, attr, None)
        if val:
            return str(val)
    return "unknown"


@app.post("/api/llm/select")
async def llm_select(request: Request) -> Dict[str, Any]:
    """切换主 LLM 模型，重新组合三链路容灾顺序。

    请求体 JSON 格式：{"model": "poolside/laguna-s-2.1:free"}
    若 model 为空字符串或 "default"，则恢复默认链路顺序。
    """
    from app.config import LLM_REGISTRY, set_active_llm

    try:
        body = await request.json()
    except Exception:
        body = {}
    model_name = (body.get("model") or "").strip()

    # 空字符串或 "default" 恢复默认顺序（set_active_llm 内部处理）
    if not model_name or model_name.lower() == "default":
        result = set_active_llm("")
        return {
            "ok": True,
            "message": "Restored default LLM chain order",
            **result,
        }

    if model_name not in LLM_REGISTRY:
        return JSONResponse(
            status_code=400,
            content={
                "ok": False,
                "error": f"Model '{model_name}' not available",
                "available": list(LLM_REGISTRY.keys()),
            },
        )

    try:
        result = set_active_llm(model_name)
        return {"ok": True, "message": f"Primary LLM switched to {model_name}", **result}
    except Exception as exc:
        logger.error("Failed to switch LLM: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": str(exc)},
        )


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
        # 追踪当前 pathway_class：从 _pathway_planner_hook 输出中捕获，
        # 替代 v4_simulation_result / v4_pathway_graph 事件中硬编码的 "egfr" 占位
        current_pathway_class: str = ""
        # [RC30] 修复：追踪 v4_validation_report（含 level1~level5）。
        # validation_pyramid_hook_node 在 worker_report 之后执行，且不在
        # NODE_NAMES_V3 中，is_actual_node 过滤会跳过其 on_chain_end 事件，
        # 导致 level1~level5 从未发射到前端，ValidationPyramid.tsx 全部显示 "skipped"。
        latest_validation_report: dict | None = None
        worker_report_payload: dict = {}  # 保存 worker_report SSE 载荷供后续合并
        # Task 21 Step 2: SA 后处理所需数据收集
        sa_knowledge_graph: dict = {}  # 从 worker_mechanism 捕获
        sa_paper_evidence: list = []   # 从 worker_report 捕获
        sa_metrics: dict = {}          # 从 worker_report 捕获
        sa_pathway_class: str = ""     # 从 _pathway_planner_hook 捕获
        sa_simulation_csv: str = ""    # Task F: 从 worker_sandbox 捕获仿真 CSV 路径

        try:
            # 在流开始时下发当前使用的模型名，供前端展示真实模型而非硬编码占位
            yield _sse_event(
                {"event": "config", "data": {"model_name": settings.OPENAI_MODEL}}
            )
            async for event in compiled_workflow_v3.astream_events(
                initial_state,
                # [P0-1 修复] recursion_limit 默认 25 不足以走完 12 节点 workflow
                # (pre_router → supervisor → worker_mcp → ... → worker_report)，
                # 导致 RecursionError 在 n11 之前崩溃。提升到 50。
                {"configurable": {"thread_id": thread_id}, "recursion_limit": 50},
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
                    # [RC30] 修复：validation_pyramid_hook_node 不在 NODE_NAMES_V3 中，
                    # 但其 on_chain_end output 含 v4_validation_report（level1~level5）。
                    # 在 continue 之前捕获该字段，合并 worker_report 载荷后发射 SSE。
                    if event_name == "on_chain_end":
                        _hook_output = event.get("data", {}).get("output", {})
                        if isinstance(_hook_output, dict):
                            # RC30: 捕获 v4_validation_report
                            if _hook_output.get("v4_validation_report"):
                                latest_validation_report = _hook_output["v4_validation_report"]
                                logger.info(
                                    "[SSE] 捕获 v4_validation_report（来源: %s, "
                                    "overall_pass=%s, failed_levels=%s）",
                                    event_chain_name,
                                    latest_validation_report.get("overall_pass"),
                                    latest_validation_report.get("failed_levels", []),
                                )
                                # 合并 level1~level5 + overall_pass 到 worker_report 载荷，
                                # 重新发射 v4_validation_report SSE（覆盖之前无 level 的版本）
                                _vp_payload = dict(worker_report_payload)
                                for _vk in ("level1", "level2", "level3", "level4",
                                            "level5", "overall_pass", "failed_levels",
                                            "short_circuit", "agent_version"):
                                    if _vk in latest_validation_report:
                                        _vp_payload[_vk] = latest_validation_report[_vk]
                                yield _sse_event({
                                    "event": "v4_validation_report",
                                    "data": _vp_payload,
                                })
                            # Sprint 6 修复：_pathway_planner_hook / _pathway_graph_hook
                            # 不在 NODE_NAMES_V3 中，on_chain_end 事件会在此 continue，
                            # 导致 v4_pathway_class 永远不被捕获，SA 后处理永远不触发。
                            # 在 continue 之前捕获 v4_pathway_class + knowledge_graph。
                            _pc = _hook_output.get("v4_pathway_class")
                            if _pc and isinstance(_pc, str) and _pc.strip():
                                current_pathway_class = _pc.strip()
                                sa_pathway_class = current_pathway_class
                                logger.info(
                                    "[SSE] 捕获 v4_pathway_class=%s（来源节点: %s, hook 通道）",
                                    current_pathway_class, event_chain_name,
                                )
                            # 捕获 knowledge_graph（worker_mechanism 内部 hook 可能输出）
                            _kg = _hook_output.get("knowledge_graph")
                            if _kg and isinstance(_kg, dict):
                                sa_knowledge_graph = _kg
                    continue

                output = event.get("data", {}).get("output", {})
                if not output:
                    output = event.get("data", {})

                # 累计 Token 使用量
                if isinstance(output, dict) and output.get("token_usage"):
                    latest_token_usage = output["token_usage"]

                # 追踪 v4_pathway_class：从 _pathway_planner_hook 或任何包含该字段的节点输出捕获
                # 替代 v4_simulation_result / v4_pathway_graph 中硬编码 "egfr" 占位
                if isinstance(output, dict):
                    _pc = output.get("v4_pathway_class")
                    if _pc and isinstance(_pc, str) and _pc.strip():
                        current_pathway_class = _pc.strip()
                        sa_pathway_class = current_pathway_class  # Task 21 Step 2: SA 后处理复用
                        logger.info("[SSE] 捕获 v4_pathway_class=%s（来源节点: %s）", current_pathway_class, node_name)

                    # Task 21 Step 2: 捕获 SA 后处理所需数据
                    if node_name == "worker_mechanism":
                        _kg = output.get("knowledge_graph") or {}
                        if _kg:
                            sa_knowledge_graph = _kg
                    if node_name == "worker_report":
                        sa_metrics = output.get("metrics") or {}
                        sa_paper_evidence = output.get("paper_evidence") or []
                    # Task F: 捕获 worker_sandbox 的仿真 CSV 路径供 BioModels 对比
                    if node_name == "worker_sandbox":
                        _sim_csv = output.get("simulation_csv_path")
                        if _sim_csv and isinstance(_sim_csv, str):
                            sa_simulation_csv = _sim_csv

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
                # 字段名契约：统一使用 v4_hypothesis_list（与 frontend store.ts 对齐）
                if isinstance(output, dict) and output.get("v4_hypothesis_generated"):
                    hyp_list = output.get("v4_hypothesis_list", [])
                    yield _sse_event({
                        "event": "v4_hypothesis_list",
                        "data": hyp_list,
                    })
                    yield _sse_event({
                        "event": "v4_hypothesis_generated",
                        "data": {
                            "hypothesis_count": len(hyp_list),
                            "v4_hypothesis_list": hyp_list,
                        },
                    })

                # 各 Worker 输出映射到前端事件
                async for sse in _emit_worker_outputs(node_name, output, current_pathway_class):
                    yield sse

                # [RC30] 保存 worker_report 载荷，供后续 _validation_pyramid_hook
                # 的 on_chain_end 事件合并 level1~level5 后重新发射 SSE
                if node_name == "worker_report" and isinstance(output, dict):
                    _wr_metrics = output.get("metrics") or {}
                    _wr_report = output.get("report") or {}
                    worker_report_payload = {
                        "metrics": _wr_metrics,
                        "report_markdown": _wr_report.get("markdown", ""),
                        "experiment_protocols": output.get("experiment_protocols") or [],
                        "paper_evidence": output.get("paper_evidence") or [],
                        "confidence": output.get("confidence", 0.0),
                        "passed": bool(_wr_metrics and not _wr_metrics.get("has_errors", False)),
                    }

                    # Sprint 2 — Evidence Bundle SSE 事件
                    _sprint2_bundle = _wr_report.get("sprint2_evidence_bundle")
                    if _sprint2_bundle is not None:
                        yield _sse_event({
                            "event": "sa_evidence_bundle",
                            "data": _sprint2_bundle,
                        })

                    # Sprint 5 — Parameter Provenance + Decision Log SSE 事件
                    _sprint5_prov = _wr_report.get("sprint5_provenance")
                    if _sprint5_prov is not None:
                        yield _sse_event({
                            "event": "sa_parameter_provenance",
                            "data": _sprint5_prov,
                        })
                    _sprint5_log = _wr_report.get("sprint5_decision_log")
                    if _sprint5_log is not None:
                        yield _sse_event({
                            "event": "sa_decision_log",
                            "data": _sprint5_log,
                        })

            # =============================================================
            # Task 21 Step 2: Scientific Alignment 后处理
            # 在 worker_report 完成后、流结束前运行 SA 闭环。
            # 受 V4_SCIENTIFIC_ALIGNMENT_ENABLED + SA_* 子 Flag 守护，
            # Flag OFF 时完全跳过（零开销，不影响 v3 行为）。
            # =============================================================
            if settings.is_scientific_alignment_enabled() and sa_metrics and sa_pathway_class:
                try:
                    async for sa_event in _run_scientific_alignment_postprocess(
                        pathway_class=sa_pathway_class,
                        metrics=sa_metrics,
                        knowledge_graph=sa_knowledge_graph,
                        paper_evidence=sa_paper_evidence,
                        report_markdown=worker_report_payload.get("report_markdown", ""),
                        simulation_csv_path=sa_simulation_csv,
                    ):
                        yield sa_event
                except Exception as sa_exc:
                    logger.warning("[SA] 后处理异常（不影响主流程）: %s", sa_exc)
                    yield _sse_event({
                        "event": "sa_postprocess_error",
                        "data": {"message": str(sa_exc)},
                    })

        except Exception as exc:
            logger.exception("v3 工作流执行异常")
            # Task G.2：SSE 错误事件统一结构化格式 {message, code}，便于前端按 code 分支处理
            yield _sse_event({
                "event": "error",
                "data": {"message": f"工作流执行异常：{exc}", "code": "workflow_exception"},
            })
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


def _load_timeseries_from_csv(
    csv_path: str, max_points: int = 100
) -> tuple[list[float], dict[str, list[float]]]:
    """从仿真 CSV 读取时间序列供前端渲染交互式曲线 + E2E 科学断言。

    CSV 格式（由 ode_templates 生成）：首行 header `t,species1,species2,...`，
    后续行为数值。降采样到 max_points 个点以控制 SSE 事件体积（图片 base64
    已占 ~33KB，time_points×species 矩阵若全量发送会显著膨胀）。

    多编码兼容（UTF-8-SIG/GB18030/CP1252），委托 app.csv_io 统一解码边界，
    避免非 UTF-8 CSV 导致前端曲线加载失败。

    Returns:
        (time_points, species_data) — time_points 为时间列 list[float]；
        species_data 为 {species_name: list[float]} 各物种浓度时序。
        读取失败时返回 ([], {})，调用方应回退到空值。
    """
    try:
        from app.csv_io import read_csv_robust

        result = read_csv_robust(csv_path)
    except Exception as exc:
        logger.warning("读取仿真 CSV 失败 (path=%s): %s", csv_path, exc)
        return [], {}
    if result.empty or not result.columns:
        return [], {}
    n = result.row_count
    if n == 0:
        return [], {}
    # 降采样：等步长取样 + 保留末点，避免曲线失真
    step = max(1, n // max_points)
    indices = list(range(0, n, step))
    if indices and indices[-1] != n - 1:
        indices.append(n - 1)
    time_points = [result.times[i] for i in indices]
    species_data: dict[str, list[float]] = {
        name: [vals[i] for i in indices] for name, vals in result.species.items()
    }
    return time_points, species_data


async def _emit_worker_outputs(node_name: str, output: Dict[str, Any], pathway_class: str = ""):
    """将 Worker 节点的输出转换为前端 SSE 事件（异步生成器）。

    Args:
        node_name: 当前 Worker 节点名
        output: Worker 输出字典
        pathway_class: 从 _pathway_planner_hook 捕获的通路类别（替代硬编码 "egfr"）
    """
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
        # v4 SSE 事件：触发 v4_pathway_graph（与 frontend store.ts hydration 对齐）
        # worker_mechanism 输出 network_json / entities，前端可据此渲染中间图谱。
        v4_graph = output.get("v4_pathway_graph")
        if v4_graph and isinstance(v4_graph, dict):
            yield _yield("v4_pathway_graph", v4_graph)
        elif isinstance(output.get("network_json"), dict) and output["network_json"].get("nodes"):
            # 降级：从 v3 network_json 构造简易 v4_pathway_graph 载荷，保证前端有图可渲染
            nj = output["network_json"]
            yield _yield("v4_pathway_graph", {
                "pathway_class": pathway_class or "UNKNOWN",  # 从 _pathway_planner_hook 捕获
                "nodes": [
                    {
                        "id": n.get("id", n.get("name", f"N{i}")),
                        "label": n.get("label", n.get("name", "")),
                        "species": n.get("name", n.get("label", "")),
                        "node_type": "species",
                        "compartment": n.get("compartment", "cytoplasm"),
                    }
                    for i, n in enumerate(nj.get("nodes", []) or [])
                ],
                "edges": [
                    {
                        "source": e.get("source", ""),
                        "target": e.get("target", ""),
                        "relation": _v3_edge_relation(e),
                    }
                    for e in nj.get("edges", []) or []
                ],
                "modules": [],
            })

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
        # v4 SSE 事件：触发 v4_simulation_result（与 frontend store.ts hydration 对齐）
        # worker_sandbox 输出 execution_result / image_base64 / csv_path，封装为
        # SimulationResult 载荷供前端 SimulationPanel 直接渲染。
        execution_result = output.get("execution_result") or {}
        # sandbox v2 不返回 execution_result（仅返回 image_base64/csv_path/
        # stdout_stderr 等），导致 v4_simulation_result 的 time_points/species
        # 为空，前端无法画交互式曲线、E2E 无法做浓度非负断言。
        # 修复：当 execution_result 缺失 time_points/species 但 csv_path 存在时，
        # 从 CSV 读取时间序列填充（降采样到 100 点控制 SSE 体积）。
        tp = execution_result.get("time_points") if isinstance(execution_result, dict) else None
        sp = execution_result.get("species") if isinstance(execution_result, dict) else None
        if (not tp or not sp) and csv_path:
            csv_tp, csv_sp = _load_timeseries_from_csv(csv_path)
            if csv_tp and csv_sp:
                tp = tp or csv_tp
                sp = sp or csv_sp
                logger.info(
                    "[v4_sim] 从 CSV 填充 time_points(%d)/species(%d)",
                    len(tp), len(sp),
                )
        if execution_result or image_base64 or csv_path:
            yield _yield("v4_simulation_result", {
                "run_id": f"v3_{output.get('run_id', '')}",
                "pathway_class": pathway_class or "UNKNOWN",  # 从 _pathway_planner_hook 捕获
                "time_points": tp or [],
                "species": sp or {},
                "metrics": output.get("metrics", {}),
                "csv_path": csv_path,
                "image_base64": image_base64,
            })

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
        # N10 诊断事件：始终发射，便于 BM 循环排查 PubMed fallback 状态
        n10_diag = output.get("n10_diagnostic")
        if n10_diag:
            yield _yield("n10_diagnostic", n10_diag)
        report = output.get("report") or {}
        if report.get("markdown"):
            yield _yield("report", report)
            yield _yield("report_ready", report.get("markdown", ""))
        # v4 SSE 事件：触发 v4_validation_report（与 frontend store.ts hydration 对齐）
        # worker_report 输出 metrics / report / experiment_protocols，封装为
        # validationReport 载荷供前端 ValidationReportPanel 渲染。
        yield _yield("v4_validation_report", {
            "metrics": metrics,
            "report_markdown": report.get("markdown", ""),
            "experiment_protocols": protocols,
            "paper_evidence": evidence,
            "confidence": output.get("confidence", 0.0),
            "passed": bool(metrics and not metrics.get("has_errors", False)),
        })


# === V4 API ENDPOINTS ===
# BioDynamics v4 RC Sprint Task E.1: 10-pathway Official Benchmark Suite
# SSE endpoint. Versioned under /api/v4/ — does NOT modify any existing
# /api/v3/ or /api/chat endpoint. The runner is READ-ONLY and reuses
# existing P4 specialists + P5 Level4 validation; it does not invoke
# scientific code mutation paths.


@app.post("/api/v4/benchmarks/run")
async def v4_benchmarks_run() -> StreamingResponse:
    """Run the 10-pathway Official Benchmark Suite, streaming progress via SSE.

    Emits Server-Sent Events in order:
      - ``benchmark_start``: ``{pathway_class, name}`` when a pathway begins.
      - ``benchmark_progress``: ``{pathway_class, step}`` step transitions.
      - ``benchmark_result``: full single-pathway result dict (see
        ``BenchmarkRunner.run_benchmark`` schema).
      - ``benchmark_complete``: final summary ``{total, passed, failed,
        results, runtime_seconds}``.

    The endpoint is non-blocking: each pathway runs sequentially in a worker
    thread so the event loop can stream progress to the client. Failures in
    any single pathway are isolated and reported as ``status="fail"`` in the
    per-pathway result; the stream continues with the next pathway.
    """
    return StreamingResponse(
        _v4_benchmark_event_stream(),
        media_type="text/event-stream",
    )


def _v4_benchmark_event_stream():
    """SSE event generator for /api/v4/benchmarks/run."""

    async def event_stream():
        from app.benchmark_runner import BenchmarkRunner

        runner = BenchmarkRunner()
        results: list[Dict[str, Any]] = []
        start_ts = time.perf_counter()
        try:
            pathway_classes = runner.list_benchmarks()
            yield _sse_event(
                {
                    "event": "benchmark_start",
                    "data": {
                        "pathway_class": "__suite__",
                        "name": "BioDynamics v4 Official Benchmark Suite",
                        "total": len(pathway_classes),
                    },
                }
            )
            for pathway_class in pathway_classes:
                spec = runner.load_all().get(pathway_class, {})
                yield _sse_event(
                    {
                        "event": "benchmark_start",
                        "data": {
                            "pathway_class": pathway_class,
                            "name": spec.get("name", ""),
                        },
                    }
                )
                yield _sse_event(
                    {
                        "event": "benchmark_progress",
                        "data": {
                            "pathway_class": pathway_class,
                            "step": "loading_specialist",
                        },
                    }
                )
                # Run benchmark in a thread to keep the event loop responsive.
                result = await asyncio.to_thread(
                    runner.run_benchmark, pathway_class
                )
                yield _sse_event(
                    {
                        "event": "benchmark_progress",
                        "data": {
                            "pathway_class": pathway_class,
                            "step": "validation_complete",
                        },
                    }
                )
                yield _sse_event(
                    {"event": "benchmark_result", "data": result}
                )
                results.append(result)

            passed = sum(1 for r in results if r.get("status") == "pass")
            failed = len(results) - passed
            summary = {
                "total": len(results),
                "passed": passed,
                "failed": failed,
                "results": results,
                "runtime_seconds": round(time.perf_counter() - start_ts, 4),
            }
            yield _sse_event({"event": "benchmark_complete", "data": summary})
        except Exception as exc:
            logger.exception("v4 benchmark stream failed")
            # Task G.2：SSE 错误事件统一结构化格式 {message, code}
            yield _sse_event(
                {
                    "event": "error",
                    "data": {"message": f"benchmark stream error: {exc}", "code": "benchmark_stream_error"},
                }
            )
        finally:
            yield _sse_event({"event": "end", "data": ""})

    return event_stream()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
