# BioDynamics Agent v4 - Scientific Benchmark Orchestrator (Task 17)
#
# 真实端到端编排器：调用 v3 LangGraph（compiled_workflow_v3）产出真实
# simulation.csv / report.md，替代 BenchmarkRunner 的 synthetic metrics 路径。
#
# 设计原则（与 spec 对齐）：
# 1. 真实执行：通过 compiled_workflow_v3.ainvoke 跑完 Supervisor-Worker 全链，
#    产出真实 simulation.csv（sandbox 持久化）与 report.md（report_renderer 持久化）。
# 2. 12 阶段埋点：post-hoc 校验 final state 中各阶段期望 key 是否存在，
#    关键阶段缺失即 fail-fast（停止后续 SA 阶段）。
# 3. SA 字段叠加：mechanism_graph / parameter_priors / biomodels_comparison /
#    evidence_fusion / seven_axis_validation / scientific_alignment 等字段
#    受各自 SA_* Feature Flag 守护，Flag OFF 时该字段为 None（不影响核心运行）。
# 4. Feature Flag 硬约束：核心 v3 pipeline 运行不依赖任何 SA flag（SA OFF 仍能
#    产出真实 artifacts）；SA 子能力各自独立 flag 守护，关闭任一不影响其他。
# 5. 失败 fail-fast：核心阶段（mechanism/parameters/ode/sandbox/report）缺失
#    → status=fail，不再执行 SA 阶段；SA 阶段失败不阻塞核心 status。
#
# 与 BenchmarkRunner 的关系：
#   - BenchmarkRunner: READ-ONLY + synthetic metrics（快速 schema/数值检查）
#   - ScientificBenchmarkOrchestrator: 真实端到端（慢，产出真实 artifacts + SA 字段）
#   Task 18 将在 BenchmarkRunner 中增加委托逻辑（BENCHMARK_REAL_ORCHESTRATOR flag）。
#
# 依赖：
# - app.graph_v3.compiled_workflow_v3（CompiledStateGraph 单例）
# - app.graph_v3.cleanup_clarification_events（防 asyncio.Event 泄漏）
# - app.scientific_alignment.*（各 SA 子能力，按需导入避免循环依赖）

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


# =============================================================================
# 路径常量
# =============================================================================
# benchmark YAML 定义目录（与 BenchmarkRunner.BENCHMARKS_DIR 一致）
_BENCHMARKS_DIR: Path = Path(__file__).resolve().parent.parent

# 日志输出目录：每个 run 产出独立子目录，供 Loop Controller 与前端追溯
_LOGS_ROOT: Path = _BENCHMARKS_DIR.parent / "data" / "sa_logs"

# pathway_class → canonical 文件名映射（与 benchmark_runner._PATHWAY_CLASS_TO_CANONICAL 一致）
# SA 模块（check_consistency / run_seven_axis_validation / run_scientific_critic /
# check_acceptance）期望 canonical 名（如 "egfr"），而非 pathway_class（如 "EGFR_RTK"）。
# 此处局部定义避免与 app.benchmark_runner 循环导入（Task 18 会建立反向依赖）。
_PATHWAY_CLASS_TO_CANONICAL: dict[str, str] = {
    "EGFR_RTK": "egfr",
    "MAPK_ERK": "mapk",
    "PI3K_AKT_mTOR": "pi3k_akt_mtor",
    "p53": "p53",
    "APOPTOSIS": "apoptosis",
    "CELL_CYCLE": "cell_cycle",
    "JAK_STAT": "jak_stat",
    "NF_KB": "nf_kappa_b",
    "WNT": "wnt",
    "TGF_BETA": "tgf_beta",
}


# =============================================================================
# 12 管线阶段定义
# =============================================================================
@dataclass(frozen=True)
class StageSpec:
    """单阶段定义。

    Attributes:
        name: 阶段标识（如 ``stage_2_mechanism``）。
        required_keys: 该阶段期望在 final state 中出现的 key 列表；
            缺失则该阶段判 fail。空列表表示仅检测阶段执行（无强制 key）。
        critical: True 表示关键阶段——缺失即 fail-fast，停止后续阶段。
        sa_flag: 若该阶段属于 SA 子能力，填写 SA flag 名（如
            ``"CONSISTENCY_CHECKER"``）；None 表示核心阶段（不受 SA flag 守护）。
    """

    name: str
    required_keys: tuple[str, ...] = ()
    critical: bool = False
    sa_flag: str | None = None


# 12 阶段流水线（与 spec Task 17.2 "12 管线阶段埋点" 对齐）
# 阶段 0-8：核心 v3 pipeline（post-hoc 校验 final state key）
# 阶段 9-12：SA 后处理（受 SA_* flag 守护，Flag OFF 时 skipped）
PIPELINE_STAGES: tuple[StageSpec, ...] = (
    StageSpec("stage_0_pre_router", required_keys=("mode",), critical=False),
    StageSpec("stage_1_mcp", required_keys=("mcp_term_definitions",), critical=False),
    StageSpec(
        "stage_2_mechanism",
        required_keys=("mechanism", "knowledge_graph", "network_json"),
        critical=True,
    ),
    StageSpec("stage_3_rag", required_keys=("parameters",), critical=True),
    StageSpec("stage_4_pkpd", required_keys=("pkpd_profile",), critical=False),
    StageSpec("stage_5_ode", required_keys=("ode_model",), critical=True),
    StageSpec(
        "stage_6_sandbox",
        required_keys=("execution_result", "simulation_csv_path"),
        critical=True,
    ),
    StageSpec("stage_7_validator", required_keys=("validation_report",), critical=False),
    StageSpec("stage_8_report", required_keys=("report", "metrics"), critical=True),
    # --- SA 后处理阶段（受 SA_* flag 守护）---
    StageSpec("stage_9_sa_consistency", sa_flag="CONSISTENCY_CHECKER"),
    StageSpec("stage_10_sa_seven_axis", sa_flag="SEVEN_AXIS"),
    StageSpec("stage_11_sa_critic", sa_flag="SCIENTIFIC_CRITIC"),
    StageSpec("stage_12_sa_acceptance", sa_flag="LOOP_TERMINATION"),
)


@dataclass
class StageTrace:
    """单阶段执行埋点。

    Attributes:
        name: 阶段标识。
        status: ``pass`` / ``fail`` / ``skipped`` / ``not_reached``。
        reason: fail/skipped 原因（如缺失 key、flag OFF、前置阶段 fail-fast）。
        keys_observed: 实际观察到的 key 列表（核心阶段）。
        duration_seconds: 该阶段耗时（SA 阶段有意义；核心阶段为 0，因为
            post-hoc 校验无法拆分 ainvoke 内部各节点耗时）。
    """

    name: str
    status: str = "not_reached"
    reason: str = ""
    keys_observed: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0


# =============================================================================
# 编排器结果
# =============================================================================
@dataclass
class OrchestratorResult:
    """ScientificBenchmarkOrchestrator.run 返回结构（Task 17.3）。

    包含核心 pipeline artifacts 与 SA 叠加字段。SA 字段在对应 flag OFF 时为 None。
    """

    pathway_class: str
    name: str
    status: str  # "pass" | "fail"
    # --- 核心 pipeline artifacts ---
    stages: list[dict[str, Any]]  # 12 阶段埋点
    simulation_csv_path: str  # 真实 CSV 路径（空字符串表示未产出）
    report_path: str  # 真实 report.md 路径
    final_report_markdown: str  # 报告全文（供前端/下游校验）
    real_metrics: dict[str, Any]  # 从 state["metrics"] 提取的真实指标
    real_metrics_flat: dict[str, float]  # 扁平化指标（{species}_{field} → value）
    confidence: float  # v3 pipeline 自带的 confidence（0..1）
    # --- SA 叠加字段（Task 17.3 要求；flag OFF 时为 None）---
    mechanism_graph: dict[str, Any] | None = None
    parameter_priors: dict[str, Any] | None = None
    biomodels_comparison: dict[str, Any] | None = None
    evidence_fusion: dict[str, Any] | None = None
    seven_axis_validation: dict[str, Any] | None = None
    consistency_report: dict[str, Any] | None = None
    critic_report: dict[str, Any] | None = None
    multi_dim_confidence: dict[str, Any] | None = None
    acceptance_report: dict[str, Any] | None = None
    scientific_alignment: dict[str, Any] | None = None
    # --- 元信息 ---
    log_dir: str = ""
    errors: list[str] = field(default_factory=list)
    runtime_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """转换为可序列化 dict（供 BenchmarkRunner / API 层消费）。"""
        return asdict(self)


# =============================================================================
# ScientificBenchmarkOrchestrator
# =============================================================================
class ScientificBenchmarkOrchestrator:
    """真实端到端 Benchmark 编排器。

    调用 v3 LangGraph（compiled_workflow_v3）跑完 Supervisor-Worker 全链，
    产出真实 simulation.csv / report.md，并按 SA flag 叠加科学对齐字段。

    Usage::

        orch = ScientificBenchmarkOrchestrator()
        # 异步（推荐，ainvoke 原生异步）
        result = await orch.run("EGFR_RTK")
        # 同步（封装 asyncio.run，供非 async 调用方使用）
        result = orch.run_sync("EGFR_RTK")

    Feature Flag 行为：
        - 核心 v3 pipeline：总是执行（不依赖 SA flag）
        - SA 子能力（stage 9-12）：受对应 SA_* flag 守护，
          V4_SCIENTIFIC_ALIGNMENT_ENABLED=false 时全部 skipped
    """

    def __init__(self, benchmarks_dir: Path | str | None = None) -> None:
        """初始化编排器。

        Args:
            benchmarks_dir: benchmark YAML 目录覆盖（默认 ``backend/benchmarks/``）。
        """
        self.benchmarks_dir: Path = (
            Path(benchmarks_dir) if benchmarks_dir is not None else _BENCHMARKS_DIR
        )

    # =========================================================================
    # 公共入口
    # =========================================================================
    async def run(self, pathway_class: str) -> OrchestratorResult:
        """异步执行单通路真实端到端 benchmark。

        Args:
            pathway_class: 通路标识（如 ``"EGFR_RTK"``）。

        Returns:
            OrchestratorResult，含真实 artifacts 与 SA 叠加字段。
        """
        start_ts = time.perf_counter()
        # 初始化 result 结构
        result = OrchestratorResult(
            pathway_class=pathway_class,
            name="",
            status="fail",
            stages=[],
            simulation_csv_path="",
            report_path="",
            final_report_markdown="",
            real_metrics={},
            real_metrics_flat={},
            confidence=0.0,
        )

        # 1. 加载 benchmark YAML（提供 input.hypothesis 作为 user_input）
        spec = self._load_benchmark_spec(pathway_class, result)
        if spec is None:
            result.runtime_seconds = round(time.perf_counter() - start_ts, 4)
            return result
        result.name = str(spec.get("name", ""))

        # 2. 准备日志目录（每个 run 独立子目录）
        run_ts = time.strftime("%Y%m%d_%H%M%S")
        run_uid = uuid.uuid4().hex[:8]
        log_dir = _LOGS_ROOT / f"run_{run_ts}_{run_uid}"
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            result.log_dir = str(log_dir)
        except Exception as exc:
            result.errors.append(f"log_dir mkdir failed: {exc}")
            # log_dir 创建失败不阻塞，继续执行

        # 3. 调用真实 v3 LangGraph pipeline
        final_state = await self._invoke_real_pipeline(pathway_class, spec, result, log_dir)

        # 4. 12 阶段 post-hoc 校验（核心阶段 0-8）
        critical_failed = self._validate_core_stages(final_state, result)

        # 5. 提取真实 artifacts（simulation.csv / report.md / metrics）
        self._extract_real_artifacts(final_state, result)

        # 6. SA 后处理阶段（9-12）—— 仅当核心阶段全 pass 时执行
        if not critical_failed and final_state is not None:
            await self._run_sa_stages(final_state, spec, result, log_dir)
        else:
            # 核心 fail-fast：SA 阶段全部 not_reached
            for stage in PIPELINE_STAGES:
                if stage.sa_flag is not None:
                    result.stages.append(
                        asdict(
                            StageTrace(
                                name=stage.name,
                                status="not_reached",
                                reason="core pipeline fail-fast",
                            )
                        )
                    )

        # 7. 汇总 status
        result.status = "fail" if critical_failed or result.errors else "pass"
        # SA 阶段失败不改变核心 status（SA 是叠加层，非阻塞）
        # 但若核心 pass 且 SA 有 findings，仍判 pass（findings 记录在 SA 字段中）

        result.runtime_seconds = round(time.perf_counter() - start_ts, 4)

        # 8. 持久化 result.json 供前端/Loop Controller 读取
        self._persist_result(result, log_dir)

        return result

    def run_sync(self, pathway_class: str) -> OrchestratorResult:
        """同步包装：在事件循环中执行 async run。

        供 BenchmarkRunner（同步）与非 async 调用方使用。
        若当前已有运行中的事件循环（如 Jupyter），改用 ``asyncio.run`` 会报错，
        此时调用方应直接 ``await orch.run(...)``。
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            # 已在事件循环中：不能再用 asyncio.run，创建 task
            raise RuntimeError(
                "ScientificBenchmarkOrchestrator.run_sync cannot be called from a "
                "running event loop; use 'await orch.run(...)' instead"
            )
        return asyncio.run(self.run(pathway_class))

    # =========================================================================
    # 内部：benchmark YAML 加载
    # =========================================================================
    def _load_benchmark_spec(
        self,
        pathway_class: str,
        result: OrchestratorResult,
    ) -> dict[str, Any] | None:
        """加载并校验 benchmark YAML。

        Args:
            pathway_class: 通路标识。
            result: 用于追加 error。

        Returns:
            YAML dict 或 None（未找到/解析失败）。
        """
        # 扫描 benchmarks_dir 下所有 yaml，匹配 pathway_class
        if not self.benchmarks_dir.exists():
            result.errors.append(f"benchmarks_dir not found: {self.benchmarks_dir}")
            return None
        for yaml_path in sorted(self.benchmarks_dir.glob("*.yaml")):
            try:
                with yaml_path.open("r", encoding="utf-8") as fh:
                    data = yaml.safe_load(fh)
                if not isinstance(data, dict):
                    continue
                if str(data.get("pathway_class", "")) == pathway_class:
                    return data
            except yaml.YAMLError as exc:
                logger.warning(
                    "Orchestrator: %s YAML parse error: %s", yaml_path.name, exc
                )
                continue
            except Exception as exc:
                logger.warning(
                    "Orchestrator: %s unexpected error: %s", yaml_path.name, exc
                )
                continue
        result.errors.append(
            f"benchmark definition not found for pathway_class='{pathway_class}'"
        )
        return None

    # =========================================================================
    # 内部：调用真实 v3 LangGraph pipeline
    # =========================================================================
    async def _invoke_real_pipeline(
        self,
        pathway_class: str,
        spec: dict[str, Any],
        result: OrchestratorResult,
        log_dir: Path,
    ) -> dict[str, Any] | None:
        """调用 compiled_workflow_v3.ainvoke 跑完真实 pipeline。

        构建 initial_state（与 main.py._v3_event_stream 对齐），
        通过 ainvoke 非流式执行，返回 final state。

        Args:
            pathway_class: 通路标识。
            spec: benchmark YAML dict。
            result: 用于追加 error。
            log_dir: 日志目录。

        Returns:
            final state dict 或 None（pipeline 异常）。
        """
        # 1. 构建 user_input：优先用 input.hypothesis，回退到 pathway name
        input_spec = spec.get("input", {}) or {}
        user_input = str(input_spec.get("hypothesis", "")) if isinstance(input_spec, dict) else ""
        if not user_input:
            user_input = f"Simulate {pathway_class} signaling dynamics"
        # 附加 species / duration 提示，提升 pipeline 命中率
        if isinstance(input_spec, dict):
            species = input_spec.get("species", [])
            if species:
                user_input += f" | species: {', '.join(map(str, species))}"
            duration = input_spec.get("duration")
            if duration:
                user_input += f" | duration: {duration} min"

        # 2. 构建 initial_state（与 main.py:462-510 对齐，显式重置所有字段）
        thread_id = f"bench_{pathway_class}_{uuid.uuid4().hex[:8]}"
        initial_state: dict[str, Any] = {
            "user_input": user_input,
            "thread_id": thread_id,
            "mode": "auto_standard",  # 自动选择模块（非 manual）
            "manual_modules": [],
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

        # 3. 调用 compiled_workflow_v3.ainvoke（非流式，直接拿 final state）
        try:
            # 延迟导入：避免在模块加载时触发 LangGraph 初始化（影响单元测试）
            from app.graph_v3 import cleanup_clarification_events, compiled_workflow_v3

            logger.info(
                "Orchestrator: invoking real v3 pipeline for %s (thread_id=%s)",
                pathway_class,
                thread_id,
            )
            config = {
                "configurable": {"thread_id": thread_id},
                "recursion_limit": 50,  # 与 main.py 一致，避免 12 节点 RecursionError
            }
            final_state = await compiled_workflow_v3.ainvoke(initial_state, config)
            logger.info(
                "Orchestrator: v3 pipeline completed for %s, state keys=%d",
                pathway_class,
                len(final_state) if isinstance(final_state, dict) else 0,
            )
            return final_state if isinstance(final_state, dict) else {}

        except Exception as exc:
            # pipeline 异常：记录 error，返回 None（核心 fail-fast）
            import traceback
            tb = traceback.format_exc()
            result.errors.append(f"v3 pipeline exception: {exc}")
            # 写入异常日志供 RCA
            try:
                (log_dir / "pipeline_exception.log").write_text(
                    f"pathway={pathway_class}\nthread_id={thread_id}\n"
                    f"exception={exc}\n\n{tb}",
                    encoding="utf-8",
                )
            except Exception:
                pass
            logger.error(
                "Orchestrator: v3 pipeline failed for %s: %s", pathway_class, exc
            )
            return None
        finally:
            # 清理 clarification events，防 asyncio.Event 泄漏（main.py:703 同款）
            try:
                cleanup_clarification_events(thread_id)
            except Exception as exc:
                logger.debug(
                    "Orchestrator: cleanup_clarification_events failed for %s: %s",
                    thread_id,
                    exc,
                )

    # =========================================================================
    # 内部：12 阶段 post-hoc 校验
    # =========================================================================
    def _validate_core_stages(
        self,
        final_state: dict[str, Any] | None,
        result: OrchestratorResult,
    ) -> bool:
        """校验核心阶段 0-8 的期望 key 是否存在于 final state。

        Args:
            final_state: v3 pipeline 返回的 final state（None 表示 pipeline 异常）。
            result: 用于追加 stages 埋点。

        Returns:
            True 表示有任一关键阶段 fail（应 fail-fast SA 阶段）。
        """
        critical_failed = False
        state = final_state or {}

        for stage in PIPELINE_STAGES:
            # 跳过 SA 阶段（在 _run_sa_stages 中处理）
            if stage.sa_flag is not None:
                continue

            trace = StageTrace(name=stage.name)
            # 记录实际观察到的 key
            trace.keys_observed = [
                k for k in stage.required_keys if k in state
            ]

            if final_state is None:
                # pipeline 异常：所有核心阶段 fail
                trace.status = "fail"
                trace.reason = "pipeline returned None (exception)"
                if stage.critical:
                    critical_failed = True
            elif not stage.required_keys:
                # 无强制 key：默认 pass
                trace.status = "pass"
            else:
                missing = [k for k in stage.required_keys if k not in state]
                # 空 dict / 空字符串也算缺失（pipeline 未填充）
                empty_keys = [
                    k
                    for k in stage.required_keys
                    if k in state and not _is_filled(state[k])
                ]
                if not missing and not empty_keys:
                    trace.status = "pass"
                else:
                    trace.status = "fail"
                    parts = []
                    if missing:
                        parts.append(f"missing keys: {missing}")
                    if empty_keys:
                        parts.append(f"empty keys: {empty_keys}")
                    trace.reason = "; ".join(parts)
                    if stage.critical:
                        critical_failed = True

            result.stages.append(asdict(trace))

        return critical_failed

    # =========================================================================
    # 内部：提取真实 artifacts
    # =========================================================================
    def _extract_real_artifacts(
        self,
        final_state: dict[str, Any] | None,
        result: OrchestratorResult,
    ) -> None:
        """从 final state 提取 simulation.csv / report.md / metrics / confidence。

        Args:
            final_state: v3 pipeline 返回的 final state。
            result: 用于填充 artifact 字段。
        """
        if not final_state:
            return

        # 1. simulation.csv 路径（sandbox 持久化的 temp 路径）
        sim_path = final_state.get("simulation_csv_path", "")
        if isinstance(sim_path, str) and sim_path:
            result.simulation_csv_path = sim_path

        # 2. report.md 路径与全文
        report_obj = final_state.get("report", {}) or {}
        if isinstance(report_obj, dict):
            persisted = report_obj.get("persisted_path", "")
            if isinstance(persisted, str) and persisted:
                result.report_path = persisted
            markdown = report_obj.get("markdown", "")
            if isinstance(markdown, str):
                result.final_report_markdown = markdown
        # 兜底：final_report 字段
        if not result.final_report_markdown:
            fr = final_state.get("final_report", "")
            if isinstance(fr, str):
                result.final_report_markdown = fr

        # 3. 真实 metrics（worker_report 产出的 {species: {peak, peak_time, ...}, overall: {...}}）
        metrics = final_state.get("metrics", {}) or {}
        if isinstance(metrics, dict):
            result.real_metrics = metrics
            result.real_metrics_flat = _flatten_metrics(metrics)

        # 4. confidence（v3 pipeline 自带，0..1）
        conf = final_state.get("confidence", 0.0)
        try:
            result.confidence = float(conf)
        except (TypeError, ValueError):
            result.confidence = 0.0

    # =========================================================================
    # 内部：SA 后处理阶段（9-12）
    # =========================================================================
    async def _run_sa_stages(
        self,
        final_state: dict[str, Any],
        spec: dict[str, Any],
        result: OrchestratorResult,
        log_dir: Path,
    ) -> None:
        """执行 SA 后处理阶段 9-12（受 SA_* flag 守护）。

        每个阶段独立 flag 守护，Flag OFF 时 skipped；Flag ON 时执行并填充对应字段。
        SA 阶段失败不阻塞核心 status（仅记录 findings）。

        Args:
            final_state: v3 pipeline final state。
            spec: benchmark YAML。
            result: 用于填充 SA 字段。
            log_dir: 日志目录。
        """
        # 延迟导入 SA 模块（避免循环依赖 + 减少 SA OFF 时的加载开销）
        try:
            from app.config import settings
        except Exception as exc:
            result.errors.append(f"settings import failed: {exc}")
            # 标记所有 SA 阶段 skipped
            for stage in PIPELINE_STAGES:
                if stage.sa_flag is not None:
                    result.stages.append(
                        asdict(
                            StageTrace(
                                name=stage.name,
                                status="skipped",
                                reason=f"settings import failed: {exc}",
                            )
                        )
                    )
            return

        sa_enabled = settings.is_scientific_alignment_enabled()

        # pathway_class（如 "EGFR_RTK"）→ canonical 名（如 "egfr"）
        # SA 模块全部期望 canonical 名（与 knowledge/canonical/<name>.yaml 文件名一致）
        pathway_class = result.pathway_class
        canonical_name = _PATHWAY_CLASS_TO_CANONICAL.get(pathway_class, pathway_class.lower())

        # 提取 SA 阶段需要的上下文（从 final_state）
        metrics = result.real_metrics
        metrics_flat = result.real_metrics_flat
        markdown = result.final_report_markdown
        parameters = final_state.get("parameters", {}) or {}
        knowledge_graph = final_state.get("knowledge_graph", {}) or {}
        network_json = final_state.get("network_json", {}) or {}
        paper_evidence = final_state.get("paper_evidence", []) or []
        validation_report = final_state.get("validation_report", {}) or {}
        experiment_protocols = final_state.get("experiment_protocols", []) or []

        # extracted_nodes：从 knowledge_graph.nodes 提取节点名列表（供 mechanism/critic 轴）
        extracted_nodes = _extract_node_names(knowledge_graph, final_state.get("entities", []))

        # cited_pmids：从 paper_evidence 提取 PMID 列表（供 literature/critic 轴）
        cited_pmids = _extract_cited_pmids(paper_evidence)

        # 持有 report 对象（供 check_acceptance 复用，避免重复计算）
        consistency_report_obj = None
        seven_axis_report_obj = None

        # --- 阶段 9：SA Consistency Checker ---
        consistency_report_obj = await self._run_stage_consistency(
            sa_enabled, canonical_name, metrics_flat, result
        )

        # --- 阶段 10：SA Seven-Axis Validation ---
        seven_axis_report_obj = await self._run_stage_seven_axis(
            sa_enabled,
            canonical_name,
            extracted_nodes,
            metrics,
            validation_report,
            cited_pmids,
            paper_evidence,
            experiment_protocols,
            markdown,
            result,
        )

        # --- 阶段 11：SA Scientific Critic ---
        await self._run_stage_critic(
            sa_enabled,
            canonical_name,
            extracted_nodes,
            metrics,
            validation_report,
            cited_pmids,
            paper_evidence,
            experiment_protocols,
            result,
        )

        # --- 阶段 12：SA Acceptance Gate ---
        await self._run_stage_acceptance(
            sa_enabled,
            canonical_name,
            metrics_flat,
            extracted_nodes,
            consistency_report_obj,
            seven_axis_report_obj,
            cited_pmids,
            experiment_protocols,
            result,
        )

        # --- 汇总 scientific_alignment 字段 ---
        result.scientific_alignment = {
            "sa_enabled": sa_enabled,
            "canonical_name": canonical_name,
            "consistency_checked": result.consistency_report is not None,
            "seven_axis_run": result.seven_axis_validation is not None,
            "critic_run": result.critic_report is not None,
            "acceptance_run": result.acceptance_report is not None,
            "multi_dim_computed": result.multi_dim_confidence is not None,
        }

        # --- 填充 mechanism_graph / parameter_priors / biomodels_comparison / evidence_fusion ---
        # 这些是从 final state 直接提取的（非 SA 计算），始终填充（SA OFF 时也填，
        # 供下游 BenchmarkRunner 数值检查使用）
        result.mechanism_graph = {
            "knowledge_graph": knowledge_graph,
            "network_json": network_json,
            "mechanism": final_state.get("mechanism", {}) or {},
        }
        result.parameter_priors = parameters
        result.biomodels_comparison = {
            "validation_report": validation_report,
            "sbml_role": final_state.get("sbml_role", "none"),
            "sbml_model_id": final_state.get("sbml_model_id", ""),
        }
        result.evidence_fusion = {
            "paper_evidence": paper_evidence,
            "rag_insights": final_state.get("rag_insights", {}) or {},
            "rag_summary": final_state.get("rag_summary", "") or "",
        }

    # -------------------------------------------------------------------------
    # SA 阶段 9：Consistency Checker
    # -------------------------------------------------------------------------
    async def _run_stage_consistency(
        self,
        sa_enabled: bool,
        canonical_name: str,
        metrics_flat: dict[str, float],
        result: OrchestratorResult,
    ) -> Any:
        """阶段 9：Scientific Consistency Checker（受 SA_CONSISTENCY_CHECKER 守护）。

        Args:
            sa_enabled: SA 总开关状态。
            canonical_name: 通路 canonical 名（如 ``"egfr"``）。
            metrics_flat: 扁平化指标 dict（如 ``{"pEGFR_peak_time": 7.5, ...}``），
                assertion 中的变量名直接对应这些 key。
            result: 用于填充 consistency_report 字段。

        Returns:
            ConsistencyReport 对象（供 check_acceptance 复用），或 None（skipped/异常）。
        """
        stage = next(s for s in PIPELINE_STAGES if s.name == "stage_9_sa_consistency")
        trace = StageTrace(name=stage.name)

        if not sa_enabled or not _sa_flag_enabled(stage.sa_flag):
            trace.status = "skipped"
            trace.reason = "SA_CONSISTENCY_CHECKER flag OFF"
            result.stages.append(asdict(trace))
            return None

        t0 = time.perf_counter()
        try:
            from app.scientific_alignment import check_consistency

            # check_consistency(pathway, simulation_metrics) 期望 canonical 名 +
            # 扁平化指标 dict（assertion 变量名直接匹配 key）
            report = check_consistency(
                pathway=canonical_name,
                simulation_metrics=metrics_flat,
            )
            result.consistency_report = {
                "violations": [
                    {
                        "rule": v.rule,
                        "violation_label": v.violation_label,
                        "observed_values": dict(v.observed_values) if v.observed_values else {},
                        "message": v.message,
                    }
                    for v in report.violations
                ],
                "passed": report.passed,
                "rules_checked": report.rules_checked,
                "rules_evaluated": report.rules_evaluated,
                "violation_count": len(report.violations),
            }
            trace.status = "pass" if report.passed else "fail"
            trace.reason = (
                "" if report.passed else f"{len(report.violations)} consistency violations"
            )
            trace.duration_seconds = round(time.perf_counter() - t0, 4)
            result.stages.append(asdict(trace))
            return report
        except Exception as exc:
            trace.status = "fail"
            trace.reason = f"consistency checker exception: {exc}"
            trace.duration_seconds = round(time.perf_counter() - t0, 4)
            result.stages.append(asdict(trace))
            result.errors.append(f"stage_9 consistency: {exc}")
            return None

    # -------------------------------------------------------------------------
    # SA 阶段 10：Seven-Axis Validation
    # -------------------------------------------------------------------------
    async def _run_stage_seven_axis(
        self,
        sa_enabled: bool,
        canonical_name: str,
        extracted_nodes: list[str],
        metrics: dict[str, Any],
        validation_report: dict[str, Any],
        cited_pmids: list[str],
        paper_evidence: list[Any],
        experiment_protocols: list[Any],
        markdown: str,
        result: OrchestratorResult,
    ) -> Any:
        """阶段 10：7 轴 Scientific Validation（受 SA_SEVEN_AXIS 守护）。

        Returns:
            SevenAxisReport 对象（供 check_acceptance 复用），或 None（skipped/异常）。
        """
        stage = next(s for s in PIPELINE_STAGES if s.name == "stage_10_sa_seven_axis")
        trace = StageTrace(name=stage.name)

        if not sa_enabled or not _sa_flag_enabled(stage.sa_flag):
            trace.status = "skipped"
            trace.reason = "SA_SEVEN_AXIS flag OFF"
            result.stages.append(asdict(trace))
            return None

        t0 = time.perf_counter()
        try:
            from app.scientific_alignment import run_seven_axis_validation

            # biomodels_report 期望 BioModelsOracleReport 对象；此处仅有 dict，
            # 传 None 让 BioModels 轴降级（不阻塞其他轴）
            report = run_seven_axis_validation(
                pathway=canonical_name,
                extracted_nodes=extracted_nodes,
                simulation_metrics=metrics,
                biomodels_report=None,
                cited_pmids=cited_pmids,
                evidence_docs=None,  # paper_evidence 是 dict 列表，非 EvidenceDoc；传 None 让轴降级
                experiments=experiment_protocols if isinstance(experiment_protocols, list) else None,
                discussion_content=markdown,
            )
            result.seven_axis_validation = {
                "axes": [
                    {
                        "axis_name": a.axis_name,
                        "score": a.score,
                        "status": a.status,
                        "sub_scores": dict(a.sub_scores) if a.sub_scores else {},
                    }
                    for a in report.axes
                ],
                "overall_passed": report.overall_passed,
                "overall_confidence": report.overall_confidence,
                "failed_axes": list(report.failed_axes),
                "degraded_axes": list(report.degraded_axes),
                "skipped": report.skipped,
            }
            trace.status = "pass" if report.overall_passed else "fail"
            trace.reason = (
                ""
                if report.overall_passed
                else f"failed axes: {list(report.failed_axes)}"
            )
            trace.duration_seconds = round(time.perf_counter() - t0, 4)
            result.stages.append(asdict(trace))
            return report
        except Exception as exc:
            trace.status = "fail"
            trace.reason = f"seven_axis exception: {exc}"
            trace.duration_seconds = round(time.perf_counter() - t0, 4)
            result.stages.append(asdict(trace))
            result.errors.append(f"stage_10 seven_axis: {exc}")
            return None

    # -------------------------------------------------------------------------
    # SA 阶段 11：Scientific Critic
    # -------------------------------------------------------------------------
    async def _run_stage_critic(
        self,
        sa_enabled: bool,
        canonical_name: str,
        extracted_nodes: list[str],
        metrics: dict[str, Any],
        validation_report: dict[str, Any],
        cited_pmids: list[str],
        paper_evidence: list[Any],
        experiment_protocols: list[Any],
        result: OrchestratorResult,
    ) -> None:
        """阶段 11：Scientific Critic Agent（受 SA_SCIENTIFIC_CRITIC 守护）。"""
        stage = next(s for s in PIPELINE_STAGES if s.name == "stage_11_sa_critic")
        trace = StageTrace(name=stage.name)

        if not sa_enabled or not _sa_flag_enabled(stage.sa_flag):
            trace.status = "skipped"
            trace.reason = "SA_SCIENTIFIC_CRITIC flag OFF"
            result.stages.append(asdict(trace))
            return

        t0 = time.perf_counter()
        try:
            from app.scientific_alignment import run_scientific_critic

            report = run_scientific_critic(
                pathway=canonical_name,
                extracted_nodes=extracted_nodes,
                simulation_metrics=metrics,
                biomodels_report=None,  # 仅有 dict，传 None 让 BioModels 类别降级
                cited_pmids=cited_pmids,
                evidence_docs=None,
                experiments=experiment_protocols if isinstance(experiment_protocols, list) else None,
                retry_count=0,
            )
            result.critic_report = {
                "findings": [
                    {
                        "category": f.category,
                        "severity": f.severity,
                        "finding": f.finding,
                        "evidence": f.evidence,
                        "suggestion": f.suggestion,
                    }
                    for f in report.findings
                ],
                "overall_status": report.overall_status,
                "retry_required": report.retry_required,
                "retry_count": report.retry_count,
            }
            trace.status = "pass" if not report.retry_required else "fail"
            trace.reason = (
                ""
                if not report.retry_required
                else f"critic requires retry (overall_status={report.overall_status})"
            )
        except Exception as exc:
            trace.status = "fail"
            trace.reason = f"critic exception: {exc}"
            result.errors.append(f"stage_11 critic: {exc}")
        trace.duration_seconds = round(time.perf_counter() - t0, 4)
        result.stages.append(asdict(trace))

    # -------------------------------------------------------------------------
    # SA 阶段 12：Acceptance Gate
    # -------------------------------------------------------------------------
    async def _run_stage_acceptance(
        self,
        sa_enabled: bool,
        canonical_name: str,
        metrics_flat: dict[str, float],
        extracted_nodes: list[str],
        consistency_report_obj: Any,
        seven_axis_report_obj: Any,
        cited_pmids: list[str],
        experiment_protocols: list[Any],
        result: OrchestratorResult,
    ) -> None:
        """阶段 12：Acceptance Criteria Gate（受 SA_LOOP_TERMINATION 守护）。

        复用阶段 9/10 产出的 ConsistencyReport / SevenAxisReport 对象，
        避免重复计算。
        """
        stage = next(s for s in PIPELINE_STAGES if s.name == "stage_12_sa_acceptance")
        trace = StageTrace(name=stage.name)

        if not sa_enabled or not _sa_flag_enabled(stage.sa_flag):
            trace.status = "skipped"
            trace.reason = "SA_LOOP_TERMINATION flag OFF"
            result.stages.append(asdict(trace))
            return

        t0 = time.perf_counter()
        try:
            from app.scientific_alignment import check_acceptance

            report = check_acceptance(
                pathway=canonical_name,
                simulation_metrics=metrics_flat,
                extracted_nodes=extracted_nodes,
                seven_axis_report=seven_axis_report_obj,
                consistency_report=consistency_report_obj,
                multi_dim_report=None,  # Multi-dim Confidence 未在此编排器中计算
                biomodels_comparison=result.biomodels_comparison,
                cited_pmids=cited_pmids,
                review_count=0,  # Review 数量需从 evidence_docs 统计，此处简化
                experiments=experiment_protocols if isinstance(experiment_protocols, list) else None,
                discussion_coverage=0.0,  # 由 Discussion Checker 单独计算，此处简化
            )
            result.acceptance_report = {
                "enabled": report.enabled,
                "skipped": report.skipped,
                "criteria": [
                    {
                        "name": c.name,
                        "passed": c.passed,
                        "expected": c.expected,
                        "actual": c.actual,
                        "severity": c.severity,
                    }
                    for c in report.criteria
                ],
                "passed": report.passed,
                "failed_criteria": list(report.failed_criteria),
                "warnings": list(report.warnings),
                "summary": report.summary,
            }
            trace.status = "pass" if report.passed else "fail"
            trace.reason = (
                ""
                if report.passed
                else f"failed criteria: {list(report.failed_criteria)}"
            )
        except Exception as exc:
            trace.status = "fail"
            trace.reason = f"acceptance exception: {exc}"
            result.errors.append(f"stage_12 acceptance: {exc}")
        trace.duration_seconds = round(time.perf_counter() - t0, 4)
        result.stages.append(asdict(trace))

    # =========================================================================
    # 内部：持久化 result
    # =========================================================================
    def _persist_result(self, result: OrchestratorResult, log_dir: Path) -> None:
        """将 result 序列化为 result.json 写入 log_dir。

        供 Loop Controller 与前端读取。写入失败仅 warning，不阻塞。
        """
        try:
            import json

            result_path = log_dir / "orchestrator_result.json"
            result_path.write_text(
                json.dumps(result.to_dict(), ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Orchestrator: persist result failed: %s", exc)


# =============================================================================
# 模块级辅助函数
# =============================================================================
def _is_filled(value: Any) -> bool:
    """判断 state 字段是否被 pipeline 实际填充（非空/非默认）。

    空字符串、空 dict、空 list、None、0.0（confidence 默认）视为未填充。
    """
    if value is None:
        return False
    if isinstance(value, str):
        return len(value.strip()) > 0
    if isinstance(value, (dict, list)):
        return len(value) > 0
    # 数值/bool：视为已填充（具体值由下游判断）
    return True


def _flatten_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    """将 nested metrics 扁平化为 {species}_{field} → value。

    worker_report 产出结构：{species: {peak, peak_time, steady_state, ...}, overall: {...}}
    扁平化后：{"pEGFR_peak_time": 7.5, "ERK_peak_time": 15.0, "overall_amplification": 50.0, ...}
    供 BenchmarkRunner pass_criteria metric_name 匹配。

    Args:
        metrics: state["metrics"] dict。

    Returns:
        扁平化指标 dict（仅含可转为 float 的值）。
    """
    flat: dict[str, float] = {}
    if not isinstance(metrics, dict):
        return flat
    for species, fields in metrics.items():
        if not isinstance(fields, dict):
            # 标量值：直接用 species 作为 key
            try:
                flat[str(species)] = float(fields)
            except (TypeError, ValueError):
                pass
            continue
        for field_name, value in fields.items():
            try:
                flat[f"{species}_{field_name}"] = float(value)
            except (TypeError, ValueError):
                # 非数值字段（如 behavior 字符串）跳过
                continue
    return flat


def _extract_peak_times(metrics: dict[str, Any]) -> dict[str, float]:
    """从 metrics 提取 {species: peak_time}（供 Consistency Checker 备用）。

    Args:
        metrics: state["metrics"] dict。

    Returns:
        {species_name: peak_time_in_minutes}。
    """
    peak_times: dict[str, float] = {}
    if not isinstance(metrics, dict):
        return peak_times
    for species, fields in metrics.items():
        if species == "overall":
            continue
        if not isinstance(fields, dict):
            continue
        peak_time = fields.get("peak_time")
        if peak_time is None:
            continue
        try:
            peak_times[str(species)] = float(peak_time)
        except (TypeError, ValueError):
            continue
    return peak_times


def _extract_node_names(
    knowledge_graph: dict[str, Any],
    entities: list[Any] | None,
) -> list[str]:
    """从 knowledge_graph.nodes 或 entities 提取机制节点名列表。

    供 Seven-Axis Mechanism 轴与 Scientific Critic 使用。

    Args:
        knowledge_graph: state["knowledge_graph"]，期望含 ``nodes`` 列表
            （每项为 ``{"id": str, "name": str, ...}`` 或纯字符串）。
        entities: state["entities"] 备用列表（每项为 dict 或 str）。

    Returns:
        节点名列表（小写归一化前保留原名，供 normalize_node_name 匹配）。
    """
    nodes: list[str] = []
    # 1. 优先从 knowledge_graph.nodes 提取
    if isinstance(knowledge_graph, dict):
        kg_nodes = knowledge_graph.get("nodes", []) or []
        if isinstance(kg_nodes, list):
            for n in kg_nodes:
                if isinstance(n, dict):
                    name = n.get("name") or n.get("id") or n.get("label")
                    if name:
                        nodes.append(str(name))
                elif isinstance(n, str):
                    nodes.append(n)
    # 2. 回退到 entities
    if not nodes and entities:
        for e in entities or []:
            if isinstance(e, dict):
                name = e.get("name") or e.get("id") or e.get("label")
                if name:
                    nodes.append(str(name))
            elif isinstance(e, str):
                nodes.append(e)
    return nodes


def _extract_cited_pmids(paper_evidence: list[Any]) -> list[str]:
    """从 paper_evidence 提取 PMID 列表。

    paper_evidence 元素结构多样（dict 含 pmid/pmid_id/source 字段，或纯字符串），
    本函数尽力提取所有形如 PMID:xxxxxxxx 或纯数字的引用。

    Args:
        paper_evidence: state["paper_evidence"] 列表。

    Returns:
        PMID 字符串列表（去重，保留 "PMID:xxxxxxxx" 格式）。
    """
    pmids: list[str] = []
    if not isinstance(paper_evidence, list):
        return pmids
    for item in paper_evidence or []:
        if isinstance(item, dict):
            pmid = item.get("pmid") or item.get("pmid_id") or item.get("source") or ""
            if pmid:
                pmids.append(str(pmid))
        elif isinstance(item, str):
            pmids.append(item)
    # 去重
    seen: set[str] = set()
    unique: list[str] = []
    for p in pmids:
        if p and p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def _sa_flag_enabled(sa_flag: str | None) -> bool:
    """查询 SA 子 flag 是否启用（通过 settings.is_sa_feature_enabled）。

    Args:
        sa_flag: SA flag 名（如 ``"CONSISTENCY_CHECKER"``），None 视为未守护。

    Returns:
        True 表示该 SA 子能力已启用。
    """
    if sa_flag is None:
        return True
    try:
        from app.config import settings
        return bool(settings.is_sa_feature_enabled(sa_flag))
    except Exception as exc:
        logger.debug("_sa_flag_enabled(%s) failed: %s", sa_flag, exc)
        return False


__all__ = [
    "ScientificBenchmarkOrchestrator",
    "PIPELINE_STAGES",
    "StageSpec",
    "StageTrace",
    "OrchestratorResult",
]
