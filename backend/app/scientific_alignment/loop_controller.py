# BioDynamics Agent v4 - Scientific Alignment Loop: Loop Controller (Task 16)
#
# 闭环编排与 Scientific Alignment Report 生成。
#
# 设计目标（SubTask 16.1 / 16.2 / 16.3 / 16.4）：
#   - 编排完整 SA 闭环：Benchmark → Collect Trace → Evidence Audit →
#     Mechanism Audit → Simulation Audit → Grounding Audit → RCA →
#     Minimal Fix → Regression → Scientific Comparison → Repeat
#   - 生成 Scientific Alignment Report.md（含 Current Score / Root Causes /
#     Remaining Problems / Regression / Next Priority / Confidence /
#     Benchmark Results / Gold Standard Comparison）
#   - loop_status：未全 Pass 时 ongoing（禁止输出 "Done"）；全 Pass 时 aligned
#   - 禁止 "修改代码→认为应该可以→结束"，任务完成需附 benchmark_results.md
#     与 logs/run_<timestamp>/
#
# Feature Flag 守护：
#   SA_LOOP_TERMINATION 默认 OFF。关闭时返回 LoopResult(skipped=True)。
#   铁律：V4_SCIENTIFIC_ALIGNMENT_ENABLED=false 时，子 Flag 永远不生效
#         （由 settings.is_sa_feature_enabled 强制校验）。
#
# 安全设计：
#   - 不引入新依赖（仅标准库 + 已完成组件 + app.config.settings）
#   - 复用 Task 12-15 的验证器，不重新实现校验逻辑
#   - 报告写入失败时不 crash，report_path 标 "write_failed"
#   - max_iterations 防无限循环（默认 3）
#   - pathway_results 为空时返回 status=ALIGNED（空集即全对齐）
#
# 依赖：
#   - app.config.settings（Feature Flag 守护）
#   - app.scientific_alignment.seven_axis_validator.run_seven_axis_validation
#   - app.scientific_alignment.multi_dim_confidence.compute_multi_dim_confidence
#   - app.scientific_alignment.scientific_critic.run_scientific_critic
#   - app.scientific_alignment.rca.run_rca
#   - app.scientific_alignment.acceptance_gate.check_acceptance
#   - app.scientific_alignment.regression_monitor.run_regression_check
#     / RegressionMonitor
#
# 核心导出：
#   from app.scientific_alignment.loop_controller import (
#       LoopStatus, LoopIteration, LoopResult, LoopController,
#       F5_STEPS, RCA_PRIORITY, F5_LOG_DIR,
#       F5StepRecord, F5LoopReport,
#   )
#
# Task 9 追加（F5 迭代纪律）：
#   - LoopController.run_f5_iteration()：强制七步循环
#       (Commit → Run Benchmark → Measure Score → Find Root Cause →
#        Fix Root Cause → Run Again → Merge or Reject)
#   - 唯一成功指标 = Benchmark Pass Rate（基于 compare_scores 量化判定）
#   - Delta < 0 自动 Reject + Feature Flag Rollback 建议
#   - 每步留存证据到 logs/f5_loop/<timestamp>/
#   - RCA 优先级（Knowledge → Retrieval → Reaction Graph → ODE → Parameters
#     → Simulation → Validation → Discussion → Prompt）禁止下游 patch
#   - 不重新设计架构，仅新增方法；不引入新依赖（仅复用 rca.run_rca /
#     validation_matrix.compute_scientific_score / compare_scores）

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

from app.config import settings
from app.scientific_alignment.acceptance_gate import check_acceptance
from app.scientific_alignment.multi_dim_confidence import (
    MultiDimConfidenceReport,
    compute_multi_dim_confidence,
)
from app.scientific_alignment.rca import run_rca
from app.scientific_alignment.regression_monitor import (
    RegressionMonitor,
    run_regression_check,
)
from app.scientific_alignment.scientific_critic import run_scientific_critic
from app.scientific_alignment.seven_axis_validator import (
    SevenAxisReport,
    run_seven_axis_validation,
)
from app.scientific_alignment.validation_matrix import (
    ScoreCard,
    ScoreComparison,
    compare_scores as _compare_scores_fn,
    compute_scientific_score,
    render_score_card_md,
)

logger = logging.getLogger(__name__)


# =============================================================================
# 常量
# =============================================================================

# 默认校验的 10 条通路（如未显式指定 pathways 时使用）
# [BENCHMARK CLOSURE / Gap 8] 对齐 knowledge/canonical/*.yaml 文件名（小写），
# 移除不属于 10 通路集合的 Notch / Hedgehog，补回 p53 / Cell_Cycle / NF_kB。
# 旧值使用混合大小写 + Notch/Hedgehog，与 scientific_alignment/*.yaml
# （egfr/mapk/pi3k_akt_mtor/p53/apoptosis/cell_cycle/jak_stat/nf_kappa_b/wnt/tgf_beta）
# 不一致，导致 loop_controller 跳过 3 条通路 + 错误校验 2 条不存在的通路。
_DEFAULT_PATHWAYS: list[str] = [
    "egfr",
    "mapk",
    "pi3k_akt_mtor",
    "p53",
    "apoptosis",
    "cell_cycle",
    "jak_stat",
    "nf_kappa_b",
    "wnt",
    "tgf_beta",
]

# 默认日志根目录（与 multi_dim_confidence.write_breakdown_json 一致）
_DEFAULT_LOG_DIR: str = "data/sa_logs"

# 报告文件名
_REPORT_FILENAME: str = "Scientific_Alignment_Report.md"
_BENCHMARK_FILENAME: str = "benchmark_results.md"


# =============================================================================
# F5 迭代纪律相关常量（Task 9）
# =============================================================================
# F5 Loop 七步名称（顺序固定，对应 Spec Scenario "F5 Loop 步骤"）
F5_STEPS: tuple[str, ...] = (
    "commit",           # 1. Commit（暂存修改）
    "run_benchmark",    # 2. Run Benchmark（运行全量 10+10 Benchmark）
    "measure_score",    # 3. Measure Score（计算 Scientific Score）
    "find_root_cause",  # 4. Find Root Cause（RCA，按优先级定位根因层）
    "fix_root_cause",   # 5. Fix Root Cause（标记修复建议，不自动改代码）
    "run_again",        # 6. Run Again（max_iterations>1 时再次运行）
    "merge_or_reject",  # 7. Merge or Reject（Delta ≥ 0 → MERGE；Delta < 0 → REJECT）
)

# RCA 优先级（禁止下游 patch 隐藏上游问题，对应 Spec Scenario "Root Cause 优先级"）
# 顺序：Knowledge → Retrieval → Reaction Graph → ODE → Parameters →
#       Simulation → Validation → Discussion → Prompt
RCA_PRIORITY: tuple[str, ...] = (
    "knowledge",        # Knowledge 层（canonical.yaml 等）
    "retrieval",        # Retrieval 层（sequential_retriever.py / RAG）
    "reaction_graph",   # Reaction Graph 层（Node 1 mechanism parsing）
    "ode",              # ODE 层（Node 2 equation generation）
    "parameters",       # Parameters 层（Node 1.5 RAG 参数提取）
    "simulation",       # Simulation 层（Node 3 sandbox solver）
    "validation",       # Validation 层（seven_axis_validator / validation_matrix）
    "discussion",       # Discussion 层（discussion_renderer Evidence Graph）
    "prompt",           # Prompt 层（最后才优化 prompt）
)

# F5 Loop 证据日志目录（每步留存证据到 logs/f5_loop/<timestamp>/）
F5_LOG_DIR: str = "logs/f5_loop"

# 各 RCA 层对应的修复建议（规则化，禁止自动改代码）
_FIX_SUGGESTIONS: dict[str, str] = {
    "knowledge": (
        "检查 knowledge/canonical/<pathway>.yaml 是否补全 required_nodes / "
        "expected_dynamics / canonical_reviews"
    ),
    "retrieval": (
        "检查 sequential_retriever.py 四级优先级是否正确，canonical.yaml 是否命中"
    ),
    "reaction_graph": (
        "检查 Node 1 mechanism parsing 输出，对比 canonical required_nodes"
    ),
    "ode": "检查 Node 2 equation generation，对比 BioModels reference ODE",
    "parameters": (
        "检查 Node 1.5 RAG 参数提取，对比 RAG_DECISION_PROMPT 决策"
    ),
    "simulation": (
        "检查 Node 3 sandbox solver 配置（tolerance / max_step / duration）"
    ),
    "validation": (
        "检查 seven_axis_validator.py / validation_matrix.py 12 轴评估"
    ),
    "discussion": "检查 discussion_renderer.py Evidence Graph 渲染",
    "prompt": "最后才优化 prompt（治理哲学原则 5）",
}

# primary_cause 文本关键词 → RCA 层映射（按优先级顺序匹配，先命中先返回）
_RCA_KEYWORD_MAP: tuple[tuple[tuple[str, ...], str], ...] = (
    (("knowledge", "canonical", "yaml"), "knowledge"),
    (("retrieval", "rag", "pubmed"), "retrieval"),
    (("reaction", "network", "graph"), "reaction_graph"),
    (("ode", "equation"), "ode"),
    (("parameter",), "parameters"),
    (("simulation", "solver", "numerical"), "simulation"),
    (("validation", "axis"), "validation"),
    (("discussion", "report"), "discussion"),
    (("prompt",), "prompt"),
)


# =============================================================================
# 数据类
# =============================================================================

class LoopStatus(str, Enum):
    """闭环状态。

    Attributes:
        ONGOING: 未全 Pass，继续循环。
        ALIGNED: 全 Pass，对齐完成。
        SKIPPED: Flag OFF，跳过。
    """

    ONGOING = "ongoing"
    ALIGNED = "aligned"
    SKIPPED = "skipped"


@dataclass
class LoopIteration:
    """单通路单次迭代的 SA 校验结果。

    Attributes:
        iteration: 第几轮（1-based）。
        pathway: 通路标识。
        seven_axis_report: SevenAxisReport。
        multi_dim_report: MultiDimConfidenceReport。
        rca_report: RCAReport。
        acceptance_report: AcceptanceReport。
        regression_report: RegressionReport。
        critic_report: CriticReport。
        passed: acceptance 是否通过。
        score: multi_dim final_confidence。
    """

    iteration: int
    pathway: str
    seven_axis_report: object = None
    multi_dim_report: object = None
    rca_report: object = None
    acceptance_report: object = None
    regression_report: object = None
    critic_report: object = None
    passed: bool = False
    score: float = 0.0


@dataclass
class LoopResult:
    """完整 SA Loop 的执行结果。

    Attributes:
        enabled: Feature Flag 是否开启。
        skipped: 是否跳过（Flag OFF 时为 True）。
        status: 闭环状态（LoopStatus）。
        iterations: 所有 LoopIteration 列表。
        total_iterations: 总迭代次数。
        aligned_pathways: 已对齐通路列表。
        ongoing_pathways: 仍在循环的通路列表。
        report_md: Scientific Alignment Report.md 内容。
        report_path: 写入的文件路径。
        next_priority: 下一步优先修复项。
    """

    enabled: bool
    skipped: bool = False
    status: LoopStatus = LoopStatus.SKIPPED
    iterations: list[LoopIteration] = field(default_factory=list)
    total_iterations: int = 0
    aligned_pathways: list[str] = field(default_factory=list)
    ongoing_pathways: list[str] = field(default_factory=list)
    report_md: str = ""
    report_path: str = ""
    next_priority: str = ""


# =============================================================================
# F5 迭代纪律数据结构（Task 9）
# =============================================================================
@dataclass
class F5StepRecord:
    """F5 Loop 单步执行记录。

    Attributes:
        step: 步骤名称（与 ``F5_STEPS`` 元素一致，如 ``"commit"``）。
        status: 执行状态（``"success"`` / ``"failed"`` / ``"skipped"``）。
        output: 该步输出（路径 / 数值 / 摘要）。
        evidence_path: 证据文件路径（``logs/f5_loop/<timestamp>/<step>.md``）。
    """

    step: str
    status: str = "pending"
    output: str = ""
    evidence_path: str = ""


@dataclass
class F5LoopReport:
    """F5 Loop 完整执行报告（对应 Spec Requirement "F5 迭代纪律"）。

    Attributes:
        timestamp: 本次 F5 Loop 时间戳（用于 ``logs/f5_loop/<timestamp>/`` 目录）。
        commit_sha: 待评估的 Commit SHA。
        step_records: 七步执行记录列表（顺序与 ``F5_STEPS`` 一致）。
        before_score: 修改前 Score（0-100）。
        after_score: 修改后 Score（0-100）。
        delta: Score 变化（``after - before``），Delta ≥ 0 → MERGE。
        decision: ``"MERGE"`` / ``"REJECT"`` / ``"PENDING"``。
        root_cause_layer: RCA 定位的根因层（``RCA_PRIORITY`` 之一）。
        suspected_module: 嫌疑模块（精确到模块/文件/函数）。
        repair_suggestion: 修复建议（来自 ``_FIX_SUGGESTIONS``）。
        rollback_triggered: 是否触发 Feature Flag Rollback。
        report_path: 最终决策文件路径（``decision.md``）。
    """

    timestamp: str = ""
    commit_sha: str = ""
    step_records: list[F5StepRecord] = field(default_factory=list)
    before_score: float = 0.0
    after_score: float = 0.0
    delta: float = 0.0
    decision: str = "PENDING"  # MERGE / REJECT / PENDING
    root_cause_layer: str = ""  # RCA_PRIORITY 之一
    suspected_module: str = ""
    repair_suggestion: str = ""
    rollback_triggered: bool = False
    report_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        """序列化为 dict（供 JSON 导出 / 测试断言）。

        Returns:
            包含所有字段的 dict（``step_records`` 亦递归序列化为 list[dict]）。
        """
        return {
            "timestamp": self.timestamp,
            "commit_sha": self.commit_sha,
            "step_records": [
                {
                    "step": r.step,
                    "status": r.status,
                    "output": r.output,
                    "evidence_path": r.evidence_path,
                }
                for r in self.step_records
            ],
            "before_score": self.before_score,
            "after_score": self.after_score,
            "delta": self.delta,
            "decision": self.decision,
            "root_cause_layer": self.root_cause_layer,
            "suspected_module": self.suspected_module,
            "repair_suggestion": self.repair_suggestion,
            "rollback_triggered": self.rollback_triggered,
            "report_path": self.report_path,
        }


# =============================================================================
# 主类
# =============================================================================

class LoopController:
    """Scientific Alignment Loop 编排器。

    编排完整 SA 闭环：对每通路执行 7 轴验证 → 6 维置信度 → Scientific Critic
    → RCA → Acceptance Gate → Regression Monitor，收集结果并生成
    Scientific Alignment Report.md。

    使用方式::

        controller = LoopController(["EGFR"], max_iterations=3)
        result = controller.run({"EGFR": {"simulation_metrics": {...}, ...}})
        if result.status == LoopStatus.ONGOING:
            print(result.next_priority)
    """

    def __init__(
        self,
        pathways: list[str] | None = None,
        max_iterations: int = 3,
    ):
        """初始化 Loop Controller。

        Args:
            pathways: 要校验的通路列表，None 时使用默认 10 通路。
            max_iterations: 每通路最大循环次数（防无限循环，默认 3）。
        """
        self.pathways: list[str] = list(pathways) if pathways is not None else list(
            _DEFAULT_PATHWAYS
        )
        self.max_iterations: int = max(1, max_iterations)
        # 回归监控器：跨迭代记录历史，用于检测回归
        self._regression_monitor: RegressionMonitor = RegressionMonitor()

    # -------------------------------------------------------------------------
    # SubTask 16.1：单通路单次迭代编排
    # -------------------------------------------------------------------------
    def run_iteration(
        self,
        pathway: str,
        iteration: int,
        simulation_metrics: dict | None = None,
        extracted_nodes: list[str] | None = None,
        cited_pmids: list[str] | None = None,
        biomodels_report=None,
        parameter_report=None,
        pipeline_trace: dict | None = None,
        discussion_content: str = "",
        experiments: list | None = None,
        biomodels_comparison: dict | None = None,
        discussion_coverage: float = 0.0,
        review_count: int = 0,
    ) -> LoopIteration:
        """执行单通路单次迭代的完整 SA 校验链。

        校验链顺序：
            1. seven_axis_validation（7 轴验证金字塔）
            2. compute_multi_dim_confidence（6 维置信度）
            3. run_scientific_critic（独立审稿）
            4. run_rca（根因分析）
            5. check_acceptance（Acceptance Gate）
            6. run_regression_check（与前次记录比较）
            7. 记录本次结果供下次比较

        Args:
            pathway: 通路标识（如 "EGFR"）。
            iteration: 第几轮（1-based）。
            simulation_metrics: 仿真指标 dict。
            extracted_nodes: Agent 提取的机制节点列表。
            cited_pmids: 引用的 PMID 列表。
            biomodels_report: BioModelsOracleReport 实例，可为 None。
            parameter_report: ParameterPriorReport 实例，可为 None。
            pipeline_trace: 管线阶段日志（用于 RCA 链路检测）。
            discussion_content: 讨论内容文本。
            experiments: 实验列表。
            biomodels_comparison: BioModels 对比结果 dict（如 {"done": True}），
                用于 Acceptance Gate。
            discussion_coverage: 10 问覆盖率（0.0-1.0），用于 Acceptance Gate。
            review_count: Review 文献数量，用于 Acceptance Gate。

        Returns:
            LoopIteration。
        """
        logger.info(
            "LoopController.run_iteration: pathway=%s, iteration=%d",
            pathway, iteration,
        )

        # -----------------------------------------------------------------
        # 1. seven_axis_validation（7 轴验证金字塔）
        # -----------------------------------------------------------------
        seven_axis_report: SevenAxisReport = run_seven_axis_validation(
            pathway=pathway,
            extracted_nodes=extracted_nodes,
            simulation_metrics=simulation_metrics,
            biomodels_report=biomodels_report,
            cited_pmids=cited_pmids,
            experiments=experiments,
            discussion_content=discussion_content,
        )

        # -----------------------------------------------------------------
        # 2. compute_multi_dim_confidence（6 维置信度）
        #    discussion_content 为空字符串时传 None 以保持降级语义一致
        # -----------------------------------------------------------------
        disc_for_multi: str | None = (
            discussion_content if discussion_content else None
        )
        multi_dim_report: MultiDimConfidenceReport = compute_multi_dim_confidence(
            pathway=pathway,
            seven_axis_report=seven_axis_report,
            parameter_report=parameter_report,
            critic_report=None,  # critic 在下一步计算，multi_dim 先用 None
            cited_pmids=cited_pmids,
            experiments=experiments,
            discussion_content=disc_for_multi,
        )

        # -----------------------------------------------------------------
        # 3. run_scientific_critic（独立审稿）
        # -----------------------------------------------------------------
        critic_report = run_scientific_critic(
            pathway=pathway,
            extracted_nodes=extracted_nodes,
            simulation_metrics=simulation_metrics,
            biomodels_report=biomodels_report,
            cited_pmids=cited_pmids,
            experiments=experiments,
        )

        # -----------------------------------------------------------------
        # 4. run_rca（根因分析，聚合各验证器 defect + 管线链路检测）
        # -----------------------------------------------------------------
        rca_report = run_rca(
            pathway=pathway,
            seven_axis_report=seven_axis_report,
            critic_report=critic_report,
            parameter_report=parameter_report,
            pipeline_trace=pipeline_trace,
        )

        # -----------------------------------------------------------------
        # 5. check_acceptance（Acceptance Gate）
        # -----------------------------------------------------------------
        acceptance_report = check_acceptance(
            pathway=pathway,
            simulation_metrics=simulation_metrics,
            extracted_nodes=extracted_nodes,
            seven_axis_report=seven_axis_report,
            multi_dim_report=multi_dim_report,
            biomodels_comparison=biomodels_comparison,
            cited_pmids=cited_pmids,
            review_count=review_count,
            experiments=experiments,
            discussion_coverage=discussion_coverage,
        )

        # -----------------------------------------------------------------
        # 6. run_regression_check（与上次记录比较）
        # -----------------------------------------------------------------
        previous_seven_axis = self._regression_monitor._previous_reports.get(
            pathway
        )
        regression_report = run_regression_check(
            pathway=pathway,
            previous_report=previous_seven_axis,
            current_report=seven_axis_report,
        )

        # -----------------------------------------------------------------
        # 7. 记录本次结果供下次比较
        # -----------------------------------------------------------------
        self._regression_monitor.record(pathway, seven_axis_report)

        # -----------------------------------------------------------------
        # 提取 passed 与 score
        # -----------------------------------------------------------------
        passed = bool(getattr(acceptance_report, "passed", False))
        score = float(getattr(multi_dim_report, "final_confidence", 0.0))

        return LoopIteration(
            iteration=iteration,
            pathway=pathway,
            seven_axis_report=seven_axis_report,
            multi_dim_report=multi_dim_report,
            rca_report=rca_report,
            acceptance_report=acceptance_report,
            regression_report=regression_report,
            critic_report=critic_report,
            passed=passed,
            score=score,
        )

    # -------------------------------------------------------------------------
    # SubTask 16.1 / 16.3：多通路完整 SA Loop
    # -------------------------------------------------------------------------
    def run(self, pathway_results: dict[str, dict]) -> LoopResult:
        """对多通路执行完整 SA Loop。

        每通路执行一次 run_iteration，收集结果，生成 Scientific Alignment
        Report.md 并写入 logs/run_<timestamp>/。

        Args:
            pathway_results: ``{pathway: {simulation_metrics, extracted_nodes,
                cited_pmids, ...}}``。每通路的输入数据。

        Returns:
            LoopResult。Flag OFF 时返回 ``skipped=True``；pathway_results 为空
            时返回 ``status=ALIGNED``（空集即全对齐）。
        """
        # -----------------------------------------------------------------
        # 1. Feature Flag 守护：SA_LOOP_TERMINATION 默认 OFF
        # -----------------------------------------------------------------
        if not settings.is_sa_feature_enabled("LOOP_TERMINATION"):
            logger.info(
                "LoopController.run: SA_LOOP_TERMINATION disabled, skipped"
            )
            return LoopResult(
                enabled=False,
                skipped=True,
                status=LoopStatus.SKIPPED,
            )

        # -----------------------------------------------------------------
        # 2. pathway_results 为空 → 空集即全对齐
        # -----------------------------------------------------------------
        if not pathway_results:
            logger.info("LoopController.run: pathway_results 为空，空集即全对齐")
            result = LoopResult(
                enabled=True,
                skipped=False,
                status=LoopStatus.ALIGNED,
            )
            result.report_md = self.generate_report(result)
            result.report_path = self._write_report(result)
            result.next_priority = ""
            return result

        # -----------------------------------------------------------------
        # 3. 对每个通路执行 run_iteration
        # -----------------------------------------------------------------
        iterations: list[LoopIteration] = []
        for pathway, data in pathway_results.items():
            data = data or {}
            try:
                it = self.run_iteration(
                    pathway=pathway,
                    iteration=1,
                    simulation_metrics=data.get("simulation_metrics"),
                    extracted_nodes=data.get("extracted_nodes"),
                    cited_pmids=data.get("cited_pmids"),
                    biomodels_report=data.get("biomodels_report"),
                    parameter_report=data.get("parameter_report"),
                    pipeline_trace=data.get("pipeline_trace"),
                    discussion_content=data.get("discussion_content", ""),
                    experiments=data.get("experiments"),
                    biomodels_comparison=data.get("biomodels_comparison"),
                    discussion_coverage=data.get("discussion_coverage", 0.0),
                    review_count=data.get("review_count", 0),
                )
                iterations.append(it)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "LoopController.run: pathway=%s 迭代异常: %s",
                    pathway, exc,
                )
                # 异常时记录一个失败的 LoopIteration（不阻塞其他通路）
                iterations.append(LoopIteration(
                    iteration=1,
                    pathway=pathway,
                    passed=False,
                    score=0.0,
                ))

        # -----------------------------------------------------------------
        # 4. 收集 aligned / ongoing 通路
        # -----------------------------------------------------------------
        aligned_pathways: list[str] = [
            it.pathway for it in iterations if it.passed
        ]
        ongoing_pathways: list[str] = [
            it.pathway for it in iterations if not it.passed
        ]

        # -----------------------------------------------------------------
        # 5. status 判定：未全 Pass → ONGOING；全 Pass → ALIGNED
        # -----------------------------------------------------------------
        status = LoopStatus.ALIGNED if not ongoing_pathways else LoopStatus.ONGOING

        # -----------------------------------------------------------------
        # 6. 构建 LoopResult
        # -----------------------------------------------------------------
        result = LoopResult(
            enabled=True,
            skipped=False,
            status=status,
            iterations=iterations,
            total_iterations=len(iterations),
            aligned_pathways=aligned_pathways,
            ongoing_pathways=ongoing_pathways,
        )

        # -----------------------------------------------------------------
        # 7. next_priority 从 RCA primary_cause 提取（需在生成报告前设置，
        #    因为报告的 "## Next Priority" 部分引用 result.next_priority）
        # -----------------------------------------------------------------
        result.next_priority = self._extract_next_priority(iterations)

        # -----------------------------------------------------------------
        # 8. 生成 Scientific Alignment Report.md
        # -----------------------------------------------------------------
        result.report_md = self.generate_report(result)

        # -----------------------------------------------------------------
        # 9. 写入 logs/run_<timestamp>/（report 写入失败不 crash）
        # -----------------------------------------------------------------
        result.report_path = self._write_report(result)

        logger.info(
            "LoopController.run 完成: status=%s, total_iterations=%d, "
            "aligned=%d, ongoing=%d",
            result.status.value, result.total_iterations,
            len(aligned_pathways), len(ongoing_pathways),
        )

        return result

    # -------------------------------------------------------------------------
    # SubTask 16.2：生成 Scientific Alignment Report.md
    # -------------------------------------------------------------------------
    def generate_report(self, result: LoopResult) -> str:
        """生成 Scientific Alignment Report.md 内容（markdown）。

        Args:
            result: LoopResult。

        Returns:
            markdown 字符串。
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # -----------------------------------------------------------------
        # 头部
        # -----------------------------------------------------------------
        lines: list[str] = [
            "# Scientific Alignment Report",
            "",
            f"**Generated**: {timestamp}",
            f"**Loop Status**: {result.status.value}",
            f"**Total Iterations**: {result.total_iterations}",
            f"**Aligned Pathways**: {', '.join(result.aligned_pathways) or '(none)'}",
            f"**Ongoing Pathways**: {', '.join(result.ongoing_pathways) or '(none)'}",
            "",
        ]

        # -----------------------------------------------------------------
        # Summary 表
        # -----------------------------------------------------------------
        lines.append("## Summary")
        lines.append("")
        lines.append(
            "| Pathway | Iteration | Score | Acceptance | Status |"
        )
        lines.append(
            "|---------|-----------|-------|------------|--------|"
        )
        for it in result.iterations:
            accept_str = "PASS" if it.passed else "FAIL"
            status_str = "aligned" if it.passed else "ongoing"
            lines.append(
                f"| {it.pathway} | {it.iteration} | "
                f"{it.score:.2f} | {accept_str} | {status_str} |"
            )
        lines.append("")

        # -----------------------------------------------------------------
        # Root Causes (Top 5)
        # -----------------------------------------------------------------
        lines.append("## Root Causes (Top 5)")
        lines.append("")
        root_causes = self._collect_root_causes(result.iterations)
        if root_causes:
            for i, cause in enumerate(root_causes[:5], 1):
                lines.append(f"{i}. {cause}")
        else:
            lines.append("- No root causes detected")
        lines.append("")

        # -----------------------------------------------------------------
        # Remaining Problems
        # -----------------------------------------------------------------
        lines.append("## Remaining Problems")
        lines.append("")
        remaining = self._collect_remaining_problems(result.iterations)
        if remaining:
            for problem in remaining:
                lines.append(f"- {problem}")
        else:
            lines.append("- No remaining problems (all pathways aligned)")
        lines.append("")

        # -----------------------------------------------------------------
        # Regression Analysis
        # -----------------------------------------------------------------
        lines.append("## Regression Analysis")
        lines.append("")
        regression_lines = self._collect_regression_analysis(result.iterations)
        if regression_lines:
            for reg_line in regression_lines:
                lines.append(f"- {reg_line}")
        else:
            lines.append("- No regression detected (first run)")
        lines.append("")

        # -----------------------------------------------------------------
        # Next Priority
        # -----------------------------------------------------------------
        lines.append("## Next Priority")
        lines.append("")
        if result.next_priority:
            lines.append(f"1. {result.next_priority}")
        else:
            lines.append("- No next priority (all aligned or no root cause)")
        lines.append("")

        # -----------------------------------------------------------------
        # Confidence Breakdown
        # -----------------------------------------------------------------
        lines.append("## Confidence Breakdown")
        lines.append("")
        lines.append(
            "| Pathway | Mechanism | Simulation | Evidence | BioModels | "
            "Discussion | Experiment | Overall |"
        )
        lines.append(
            "|---------|-----------|------------|----------|-----------|"
            "------------|------------|---------|"
        )
        for it in result.iterations:
            dim_scores = self._extract_dim_scores(it.multi_dim_report)
            overall = it.score
            lines.append(
                f"| {it.pathway} | {dim_scores[0]:.2f} | "
                f"{dim_scores[1]:.2f} | {dim_scores[2]:.2f} | "
                f"{dim_scores[3]:.2f} | {dim_scores[4]:.2f} | "
                f"{dim_scores[5]:.2f} | {overall:.2f} |"
            )
        lines.append("")

        # -----------------------------------------------------------------
        # Benchmark Results
        # -----------------------------------------------------------------
        lines.append("## Benchmark Results")
        lines.append("")
        lines.append("(详见 benchmark_results.md)")
        lines.append("")

        # -----------------------------------------------------------------
        # Gold Standard Comparison
        # -----------------------------------------------------------------
        lines.append("## Gold Standard Comparison")
        lines.append("")
        gold_lines = self._collect_gold_standard_comparison(result.iterations)
        if gold_lines:
            for gold_line in gold_lines:
                lines.append(f"- {gold_line}")
        else:
            lines.append("- No gold standard comparison available")
        lines.append("")

        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # 私有：写入报告文件
    # -------------------------------------------------------------------------
    def _write_report(self, result: LoopResult) -> str:
        """将 Scientific Alignment Report.md 与 benchmark_results.md 写入
        ``data/sa_logs/run_<timestamp>/``。

        写入失败时返回 "write_failed"，不 crash。

        Args:
            result: LoopResult。

        Returns:
            写入的报告文件路径；失败时返回 "write_failed"。
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = os.path.join(_DEFAULT_LOG_DIR, f"run_{timestamp}")
            os.makedirs(run_dir, exist_ok=True)

            # 写入 Scientific_Alignment_Report.md
            report_path = os.path.join(run_dir, _REPORT_FILENAME)
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(result.report_md)

            # 写入 benchmark_results.md
            benchmark_path = os.path.join(run_dir, _BENCHMARK_FILENAME)
            benchmark_md = self._generate_benchmark_results(result.iterations)
            with open(benchmark_path, "w", encoding="utf-8") as f:
                f.write(benchmark_md)

            logger.info(
                "LoopController: 报告写入 %s", report_path,
            )
            return report_path
        except OSError as exc:
            logger.warning("LoopController: 报告写入失败: %s", exc)
            return "write_failed"
        except Exception as exc:  # noqa: BLE001
            logger.warning("LoopController: 报告写入异常: %s", exc)
            return "write_failed"

    # -------------------------------------------------------------------------
    # 私有：生成 benchmark_results.md
    # -------------------------------------------------------------------------
    def _generate_benchmark_results(
        self,
        iterations: list[LoopIteration],
    ) -> str:
        """生成 benchmark_results.md 内容。

        Args:
            iterations: LoopIteration 列表。

        Returns:
            markdown 字符串。
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines: list[str] = [
            "# Benchmark Results",
            "",
            f"**Generated**: {timestamp}",
            "",
            "## Acceptance Criteria Results",
            "",
            "| Pathway | Passed | Failed Criteria | Summary |",
            "|---------|--------|-----------------|---------|",
        ]

        for it in iterations:
            acceptance = it.acceptance_report
            if acceptance is None:
                lines.append(
                    f"| {it.pathway} | - | - | (no acceptance report) |"
                )
                continue
            passed = bool(getattr(acceptance, "passed", False))
            failed_criteria = getattr(acceptance, "failed_criteria", []) or []
            summary = getattr(acceptance, "summary", "")
            failed_str = ", ".join(failed_criteria) if failed_criteria else "(none)"
            # 转义管道符避免破坏表格
            failed_str = failed_str.replace("|", "\\|")
            summary = summary.replace("|", "\\|")
            lines.append(
                f"| {it.pathway} | {'PASS' if passed else 'FAIL'} | "
                f"{failed_str} | {summary} |"
            )

        lines.append("")
        lines.append("## Criteria Detail")
        lines.append("")

        for it in iterations:
            acceptance = it.acceptance_report
            if acceptance is None:
                continue
            lines.append(f"### {it.pathway}")
            lines.append("")
            lines.append(
                "| Criterion | Passed | Severity | Expected | Actual |"
            )
            lines.append(
                "|-----------|--------|----------|----------|--------|"
            )
            criteria = getattr(acceptance, "criteria", []) or []
            for criterion in criteria:
                name = getattr(criterion, "name", "")
                c_passed = bool(getattr(criterion, "passed", False))
                severity = getattr(criterion, "severity", "")
                expected = str(getattr(criterion, "expected", "")).replace("|", "\\|")
                actual = str(getattr(criterion, "actual", "")).replace("|", "\\|")
                lines.append(
                    f"| {name} | {'PASS' if c_passed else 'FAIL'} | "
                    f"{severity} | {expected} | {actual} |"
                )
            lines.append("")

        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # 私有：提取 next_priority
    # -------------------------------------------------------------------------
    def _extract_next_priority(
        self,
        iterations: list[LoopIteration],
    ) -> str:
        """从 RCA primary_cause 提取下一步优先修复项。

        优先返回第一个 ongoing 通路（未通过）的 RCA primary_cause。

        Args:
            iterations: LoopIteration 列表。

        Returns:
            primary_cause 摘要；无 ongoing 通路或无根因时返回空字符串。
        """
        for it in iterations:
            if it.passed:
                continue
            rca_report = it.rca_report
            if rca_report is None:
                continue
            primary = getattr(rca_report, "primary_cause", "")
            if primary:
                return primary
        return ""

    # -------------------------------------------------------------------------
    # 私有：收集 Root Causes（Top 5）
    # -------------------------------------------------------------------------
    def _collect_root_causes(
        self,
        iterations: list[LoopIteration],
    ) -> list[str]:
        """收集所有 ongoing 通路的 RCA 根因。

        Args:
            iterations: LoopIteration 列表。

        Returns:
            根因摘要字符串列表（每个通路最多 1 条 primary_cause）。
        """
        causes: list[str] = []
        for it in iterations:
            if it.passed:
                continue
            rca_report = it.rca_report
            if rca_report is None:
                continue
            primary = getattr(rca_report, "primary_cause", "")
            if primary:
                causes.append(f"[{it.pathway}] {primary}")
        return causes

    # -------------------------------------------------------------------------
    # 私有：收集 Remaining Problems
    # -------------------------------------------------------------------------
    def _collect_remaining_problems(
        self,
        iterations: list[LoopIteration],
    ) -> list[str]:
        """收集未对齐通路的具体问题。

        Args:
            iterations: LoopIteration 列表。

        Returns:
            问题描述字符串列表。
        """
        problems: list[str] = []
        for it in iterations:
            if it.passed:
                continue
            parts: list[str] = []

            # confidence 低于 0.9
            if it.score < 0.9:
                parts.append(
                    f"confidence {it.score:.2f} < 0.9"
                )

            # failed criteria
            acceptance = it.acceptance_report
            if acceptance is not None:
                failed_criteria = getattr(
                    acceptance, "failed_criteria", []
                ) or []
                if failed_criteria:
                    parts.append(
                        f"failed: {', '.join(failed_criteria)}"
                    )

            # RCA total_defects
            rca_report = it.rca_report
            if rca_report is not None:
                total_defects = getattr(rca_report, "total_defects", 0)
                if total_defects > 0:
                    parts.append(f"{total_defects} root causes detected")

            if not parts:
                parts.append("acceptance FAIL")
            problems.append(f"{it.pathway}: {'; '.join(parts)}")
        return problems

    # -------------------------------------------------------------------------
    # 私有：收集 Regression Analysis
    # -------------------------------------------------------------------------
    def _collect_regression_analysis(
        self,
        iterations: list[LoopIteration],
    ) -> list[str]:
        """收集回归分析信息。

        Args:
            iterations: LoopIteration 列表。

        Returns:
            回归分析描述字符串列表。
        """
        lines: list[str] = []
        for it in iterations:
            regression = it.regression_report
            if regression is None:
                continue
            # skipped 报告（Flag OFF）跳过
            if getattr(regression, "skipped", False):
                continue
            has_regression = getattr(regression, "has_regression", False)
            has_fix = getattr(regression, "has_fix", False)
            regression_axes = getattr(
                regression, "regression_axes", []
            ) or []
            fix_axes = getattr(regression, "fix_axes", []) or []
            rollback = getattr(
                regression, "rollback_suggestion", []
            ) or []
            overall_delta = float(getattr(regression, "overall_delta", 0.0))

            if has_regression:
                reg_str = (
                    f"{it.pathway}: regression on {regression_axes}, "
                    f"overall_delta={overall_delta:+.3f}"
                )
                if rollback:
                    reg_str += f", rollback suggested: {rollback}"
                lines.append(reg_str)
            elif has_fix:
                lines.append(
                    f"{it.pathway}: fix on {fix_axes}, "
                    f"overall_delta={overall_delta:+.3f}"
                )
            elif overall_delta != 0.0:
                lines.append(
                    f"{it.pathway}: no regression/fix, "
                    f"overall_delta={overall_delta:+.3f}"
                )
        return lines

    # -------------------------------------------------------------------------
    # 私有：提取 6 维得分
    # -------------------------------------------------------------------------
    def _extract_dim_scores(self, multi_dim_report: Any) -> list[float]:
        """从 MultiDimConfidenceReport 提取 6 维得分。

        维度顺序：Mechanism / Simulation / Evidence / BioModels /
        Discussion / Experiment。

        Args:
            multi_dim_report: MultiDimConfidenceReport 实例。

        Returns:
            6 维得分列表；报告为 None 或 skipped 时返回 6 个 0.0。
        """
        default_scores = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if multi_dim_report is None:
            return default_scores
        if getattr(multi_dim_report, "skipped", False):
            return default_scores

        dimensions = getattr(multi_dim_report, "dimensions", []) or []
        # 按名称映射到固定顺序
        dim_map: dict[str, float] = {}
        for dim in dimensions:
            name = getattr(dim, "name", "")
            score = float(getattr(dim, "score", 0.0))
            dim_map[name] = score

        return [
            dim_map.get("Mechanism", 0.0),
            dim_map.get("Simulation", 0.0),
            dim_map.get("Evidence", 0.0),
            dim_map.get("BioModels", 0.0),
            dim_map.get("Discussion", 0.0),
            dim_map.get("Experiment", 0.0),
        ]

    # -------------------------------------------------------------------------
    # 私有：收集 Gold Standard Comparison
    # -------------------------------------------------------------------------
    def _collect_gold_standard_comparison(
        self,
        iterations: list[LoopIteration],
    ) -> list[str]:
        """收集 Gold Standard Comparison（Acceptance Criteria 通过情况）。

        Args:
            iterations: LoopIteration 列表。

        Returns:
            每通路的 Gold Standard Comparison 描述。
        """
        lines: list[str] = []
        for it in iterations:
            acceptance = it.acceptance_report
            if acceptance is None:
                continue
            if getattr(acceptance, "skipped", False):
                continue
            criteria = getattr(acceptance, "criteria", []) or []
            if not criteria:
                continue
            total = len(criteria)
            passed_count = sum(1 for c in criteria if getattr(c, "passed", False))
            failed_names = [
                getattr(c, "name", "")
                for c in criteria
                if not getattr(c, "passed", False)
            ]
            if failed_names:
                lines.append(
                    f"{it.pathway}: {passed_count}/{total} criteria passed "
                    f"(failed: {', '.join(failed_names)})"
                )
            else:
                lines.append(
                    f"{it.pathway}: {passed_count}/{total} criteria passed "
                    f"(all passed)"
                )
        return lines

    # -------------------------------------------------------------------------
    # Task 7：Score Card 对比（F5 Loop Merge/Reject 唯一依据）
    # -------------------------------------------------------------------------
    def compare_scores(
        self,
        before: ScoreCard,
        after: ScoreCard,
    ) -> ScoreComparison:
        """对比前后 Score，输出 Delta 与 MERGE/REJECT 决策。

        对应 Spec Scenario "Score 对比"——Delta ≥ 0 → MERGE；Delta < 0 → REJECT。
        本方法作为 F5 Loop Merge/Reject 的唯一量化依据（禁止肉眼判断）。

        实现委托给 ``validation_matrix.compare_scores`` 模块级函数，保持单一
        数据源（Score Card 逻辑集中在 ``validation_matrix.py``，不在
        ``loop_controller.py`` 重复定义）。

        Args:
            before: 修改前 ScoreCard。
            after: 修改后 ScoreCard。

        Returns:
            ScoreComparison，含总分 Delta、各分项 Delta、MERGE/REJECT 决策。
        """
        return _compare_scores_fn(before, after)

    # -------------------------------------------------------------------------
    # Task 9：F5 迭代纪律（强制七步循环 + Merge/Reject 量化判定）
    # -------------------------------------------------------------------------
    def run_f5_iteration(
        self,
        *,
        commit_sha: str,
        before_score: ScoreCard | None = None,
        benchmark_runner: Callable[[], dict[str, Any]] | None = None,
        benchmark_results: dict[str, Any] | None = None,
        max_iterations: int = 1,
        rca_inputs: dict[str, Any] | None = None,
    ) -> F5LoopReport:
        """执行 F5 Loop 七步循环（强制迭代纪律）。

        七步：
            1. Commit        — 记录 commit_sha
            2. Run Benchmark — 运行全量 10+10 benchmark（或接受预计算结果）
            3. Measure Score — 计算 Scientific Score（调用 compute_scientific_score）
            4. Find Root Cause — RCA，按优先级定位根因层
            5. Fix Root Cause  — 标记修复建议（不自动修改代码）
            6. Run Again       — 再次运行 benchmark（可选，max_iterations>1 时）
            7. Merge or Reject — 对比 before/after Score，Delta ≥ 0 → MERGE

        每步留存证据到 ``logs/f5_loop/<timestamp>/``。
        Delta < 0 → 自动 Reject + 触发 Feature Flag Rollback 建议 + 输出
        ``f5_reject_report.md``。

        Args:
            commit_sha: 待评估的 Commit SHA。
            before_score: 修改前 ScoreCard（None 时跳过对比，仅 Measure）。
            benchmark_runner: 可调用对象，返回 benchmark 结果 dict
                （None 时用 benchmark_results）。
            benchmark_results: 预计算的 benchmark 结果 dict。
            max_iterations: 最大迭代次数（默认 1）。
            rca_inputs: RCA 输入数据（含 pathway / seven_axis_report /
                critic_report / parameter_report / pipeline_trace 等；
                若含 ``primary_cause`` 字符串字段则作为测试覆盖值）。

        Returns:
            F5LoopReport
        """
        # -----------------------------------------------------------------
        # 初始化报告与日志目录
        # -----------------------------------------------------------------
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(F5_LOG_DIR, timestamp)
        try:
            os.makedirs(run_dir, exist_ok=True)
        except OSError as exc:
            logger.warning("run_f5_iteration: 创建日志目录失败: %s", exc)

        report = F5LoopReport(timestamp=timestamp, commit_sha=commit_sha)
        if before_score is not None:
            report.before_score = float(before_score.total_score)

        rca_inputs = rca_inputs or {}

        # -----------------------------------------------------------------
        # Step 1: Commit — 记录 commit_sha
        # -----------------------------------------------------------------
        commit_md = (
            f"# F5 Loop Step 1: Commit\n\n"
            f"**Commit SHA**: {commit_sha}\n"
            f"**Timestamp**: {timestamp}\n"
        )
        commit_path = self._f5_write_evidence(run_dir, "commit.md", commit_md)
        report.step_records.append(F5StepRecord(
            step="commit",
            status="success",
            output=commit_sha,
            evidence_path=commit_path,
        ))

        # -----------------------------------------------------------------
        # Step 2: Run Benchmark
        # -----------------------------------------------------------------
        bench_path = self._f5_write_evidence(
            run_dir, "benchmark_results.md", "# (pending)\n"
        )
        if benchmark_runner is not None:
            try:
                benchmark_results = benchmark_runner()
                bench_md = self._render_benchmark_summary(benchmark_results)
                bench_path = self._f5_write_evidence(
                    run_dir, "benchmark_results.md", bench_md,
                )
                report.step_records.append(F5StepRecord(
                    step="run_benchmark",
                    status="success",
                    output=f"benchmark_runner called; keys={list(benchmark_results.keys())}",
                    evidence_path=bench_path,
                ))
            except Exception as exc:  # noqa: BLE001
                logger.warning("run_f5_iteration: benchmark_runner 异常: %s", exc)
                report.step_records.append(F5StepRecord(
                    step="run_benchmark",
                    status="failed",
                    output=f"benchmark_runner exception: {exc}",
                    evidence_path=bench_path,
                ))
                benchmark_results = benchmark_results or {}
        elif benchmark_results is not None:
            bench_md = self._render_benchmark_summary(benchmark_results)
            bench_path = self._f5_write_evidence(
                run_dir, "benchmark_results.md", bench_md,
            )
            report.step_records.append(F5StepRecord(
                step="run_benchmark",
                status="success",
                output=f"precomputed; keys={list(benchmark_results.keys())}",
                evidence_path=bench_path,
            ))
        else:
            report.step_records.append(F5StepRecord(
                step="run_benchmark",
                status="skipped",
                output="no benchmark runner/results provided",
                evidence_path=bench_path,
            ))
            benchmark_results = {}

        # -----------------------------------------------------------------
        # Step 3: Measure Score
        # -----------------------------------------------------------------
        score_card = self._measure_score(benchmark_results)
        score_md = render_score_card_md(score_card)
        score_path = self._f5_write_evidence(
            run_dir, "score_card.md", score_md,
        )
        report.after_score = float(score_card.total_score)
        report.step_records.append(F5StepRecord(
            step="measure_score",
            status="success",
            output=f"total_score={score_card.total_score:.2f}",
            evidence_path=score_path,
        ))

        # -----------------------------------------------------------------
        # Step 4: Find Root Cause
        # -----------------------------------------------------------------
        primary_cause, root_cause_layer, suspected_module = (
            self._run_f5_rca(rca_inputs)
        )
        if root_cause_layer not in RCA_PRIORITY:
            logger.warning(
                "run_f5_iteration: root_cause_layer=%r 不在 RCA_PRIORITY 中",
                root_cause_layer,
            )
        rca_md = (
            f"# F5 Loop Step 4: Find Root Cause\n\n"
            f"**primary_cause**: {primary_cause}\n"
            f"**root_cause_layer**: {root_cause_layer}\n"
            f"**suspected_module**: {suspected_module}\n"
            f"**RCA_PRIORITY**: {list(RCA_PRIORITY)}\n"
        )
        rca_path = self._f5_write_evidence(run_dir, "rca.md", rca_md)
        report.root_cause_layer = root_cause_layer
        report.suspected_module = suspected_module
        report.step_records.append(F5StepRecord(
            step="find_root_cause",
            status="success",
            output=f"layer={root_cause_layer}; primary={primary_cause[:80]}",
            evidence_path=rca_path,
        ))

        # -----------------------------------------------------------------
        # Step 5: Fix Root Cause（仅输出建议，不自动改代码）
        # -----------------------------------------------------------------
        suggestion = _FIX_SUGGESTIONS.get(
            root_cause_layer, _FIX_SUGGESTIONS["validation"],
        )
        fix_md = (
            f"# F5 Loop Step 5: Fix Root Cause\n\n"
            f"**root_cause_layer**: {root_cause_layer}\n"
            f"**repair_suggestion**: {suggestion}\n\n"
            f"注意：F5 Loop 不自动修改代码（这是开发者职责）。\n"
        )
        fix_path = self._f5_write_evidence(
            run_dir, "fix_suggestion.md", fix_md,
        )
        report.repair_suggestion = suggestion
        report.step_records.append(F5StepRecord(
            step="fix_root_cause",
            status="success",
            output=suggestion,
            evidence_path=fix_path,
        ))

        # -----------------------------------------------------------------
        # Step 6: Run Again（仅 max_iterations>1 且有 benchmark_runner 时）
        # -----------------------------------------------------------------
        if max_iterations > 1 and benchmark_runner is not None:
            try:
                again_results = benchmark_runner()
                again_md = self._render_benchmark_summary(again_results)
                again_path = self._f5_write_evidence(
                    run_dir, "run_again.md", again_md,
                )
                report.step_records.append(F5StepRecord(
                    step="run_again",
                    status="success",
                    output=f"re-run; total_score recompute pending",
                    evidence_path=again_path,
                ))
                # 重新计算 Score（用再次运行结果）
                score_card = self._measure_score(again_results)
                report.after_score = float(score_card.total_score)
            except Exception as exc:  # noqa: BLE001
                logger.warning("run_f5_iteration: run_again 异常: %s", exc)
                again_path = self._f5_write_evidence(
                    run_dir, "run_again.md",
                    f"# Run Again Failed\n\nException: {exc}\n",
                )
                report.step_records.append(F5StepRecord(
                    step="run_again",
                    status="failed",
                    output=f"exception: {exc}",
                    evidence_path=again_path,
                ))
        else:
            again_path = self._f5_write_evidence(
                run_dir, "run_again.md",
                "# F5 Loop Step 6: Run Again (skipped)\n\n"
                "单次迭代（max_iterations=1 或无 benchmark_runner）\n",
            )
            report.step_records.append(F5StepRecord(
                step="run_again",
                status="skipped",
                output="single iteration",
                evidence_path=again_path,
            ))

        # -----------------------------------------------------------------
        # Step 7: Merge or Reject
        # -----------------------------------------------------------------
        after_score_card = score_card
        if before_score is None:
            decision_str = "PENDING"
            report.delta = 0.0
            decision_md = (
                "# F5 Loop Step 7: Merge or Reject\n\n"
                "**decision**: PENDING\n"
                "**reason**: no before_score for comparison\n"
                f"**after_score**: {report.after_score:.2f}\n"
            )
            decision_path = self._f5_write_evidence(
                run_dir, "decision.md", decision_md,
            )
            report.decision = decision_str
            report.report_path = decision_path
            report.step_records.append(F5StepRecord(
                step="merge_or_reject",
                status="success",
                output="no before_score for comparison",
                evidence_path=decision_path,
            ))
        else:
            comparison = self.compare_scores(before_score, after_score_card)
            report.delta = float(comparison.delta_total)
            decision_str = str(comparison.decision)
            report.decision = decision_str
            if decision_str == "REJECT":
                report.rollback_triggered = True
                # 写 f5_reject_report.md
                reject_md = self._render_f5_reject_report(
                    report, comparison,
                )
                self._f5_write_evidence(
                    run_dir, "f5_reject_report.md", reject_md,
                )
                # 仅输出回滚建议（不实际修改 .env，需人工确认）
                rollback_note = (
                    "Suggested rollback: set "
                    "V4_SCIENTIFIC_REVIEWER_ENABLED=false"
                )
                decision_md = (
                    "# F5 Loop Step 7: Merge or Reject\n\n"
                    f"**decision**: REJECT\n"
                    f"**before_score**: {report.before_score:.2f}\n"
                    f"**after_score**: {report.after_score:.2f}\n"
                    f"**delta**: {report.delta:+.2f}\n"
                    f"**rollback_triggered**: True\n"
                    f"**rollback_note**: {rollback_note}\n"
                    f"**root_cause_layer**: {report.root_cause_layer}\n"
                )
                decision_path = self._f5_write_evidence(
                    run_dir, "decision.md", decision_md,
                )
                report.step_records.append(F5StepRecord(
                    step="merge_or_reject",
                    status="success",
                    output=f"REJECT; delta={report.delta:+.2f}",
                    evidence_path=decision_path,
                ))
            else:
                # MERGE：写 f5_merge_report.md
                merge_md = (
                    "# F5 Loop Merge Report\n\n"
                    f"**decision**: MERGE\n"
                    f"**before_score**: {report.before_score:.2f}\n"
                    f"**after_score**: {report.after_score:.2f}\n"
                    f"**delta**: {report.delta:+.2f}\n"
                    f"**commit_sha**: {commit_sha}\n"
                )
                self._f5_write_evidence(
                    run_dir, "f5_merge_report.md", merge_md,
                )
                decision_md = (
                    "# F5 Loop Step 7: Merge or Reject\n\n"
                    f"**decision**: MERGE\n"
                    f"**before_score**: {report.before_score:.2f}\n"
                    f"**after_score**: {report.after_score:.2f}\n"
                    f"**delta**: {report.delta:+.2f}\n"
                )
                decision_path = self._f5_write_evidence(
                    run_dir, "decision.md", decision_md,
                )
                report.step_records.append(F5StepRecord(
                    step="merge_or_reject",
                    status="success",
                    output=f"MERGE; delta={report.delta:+.2f}",
                    evidence_path=decision_path,
                ))
            report.report_path = decision_path

        logger.info(
            "run_f5_iteration 完成: commit=%s, before=%.2f, after=%.2f, "
            "delta=%+.2f, decision=%s, rca_layer=%s",
            commit_sha, report.before_score, report.after_score,
            report.delta, report.decision, report.root_cause_layer,
        )
        return report

    def should_continue_iteration(
        self,
        report: F5LoopReport,
        *,
        score_threshold: float = 90.0,
        iteration: int = 1,
        max_iterations: int = 3,
    ) -> bool:
        """判断是否应继续 F5 迭代。

        终止条件：
            - ``report.decision == "REJECT"`` → False
            - ``report.after_score >= score_threshold`` → False
            - ``iteration >= max_iterations`` → False
            - 否则 → True

        Args:
            report: 上一次 ``run_f5_iteration`` 返回的 F5LoopReport。
            score_threshold: 停止迭代的 Score 阈值（默认 90.0）。
            iteration: 当前迭代轮次（1-based）。
            max_iterations: 最大迭代次数（默认 3）。

        Returns:
            True 表示应继续迭代；False 表示应终止。
        """
        if str(report.decision) == "REJECT":
            return False
        if float(report.after_score) >= float(score_threshold):
            return False
        if int(iteration) >= int(max_iterations):
            return False
        return True

    # -------------------------------------------------------------------------
    # Task 9 私有辅助方法
    # -------------------------------------------------------------------------
    def _f5_write_evidence(
        self,
        run_dir: str,
        filename: str,
        content: str,
    ) -> str:
        """写入 F5 Loop 证据文件。

        写入失败时返回 "write_failed"，不 crash（与 ``_write_report`` 一致）。

        Args:
            run_dir: ``logs/f5_loop/<timestamp>/`` 目录。
            filename: 文件名（如 ``"commit.md"``）。
            content: 文件内容。

        Returns:
            写入的文件路径；失败时返回 "write_failed"。
        """
        try:
            os.makedirs(run_dir, exist_ok=True)
            path = os.path.join(run_dir, filename)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return path
        except OSError as exc:
            logger.warning("_f5_write_evidence: 写入失败: %s", exc)
            return "write_failed"
        except Exception as exc:  # noqa: BLE001
            logger.warning("_f5_write_evidence: 写入异常: %s", exc)
            return "write_failed"

    def _measure_score(
        self,
        benchmark_results: dict[str, Any],
    ) -> ScoreCard:
        """从 benchmark_results 提取 5 个分项并计算 ScoreCard。

        缺字段时该分项传空 dict → ``raw_score=0.0``（不 crash）。

        Args:
            benchmark_results: 含 5 个分项 key 的 dict。

        Returns:
            ScoreCard。
        """
        def _get(key: str) -> dict[str, Any]:
            val = benchmark_results.get(key)
            if isinstance(val, dict):
                return val
            return {}

        return compute_scientific_score(
            validation_matrix_result=_get("validation_matrix_result"),
            curve_metrics_result=_get("curve_metrics_result"),
            literature_result=_get("literature_result"),
            evidence_result=_get("evidence_result"),
            experiment_result=_get("experiment_result"),
            benchmark_name="f5_loop",
        )

    def _run_f5_rca(
        self,
        rca_inputs: dict[str, Any],
    ) -> tuple[str, str, str]:
        """执行 RCA 并推断根因层。

        优先级：
            1. 若 ``rca_inputs`` 含 ``primary_cause`` 字符串字段（测试覆盖），
               直接用于推断
            2. 否则调用 ``run_rca()`` 提取 ``primary_cause``
            3. 根据 ``primary_cause`` 文本关键词推断 ``root_cause_layer``

        Args:
            rca_inputs: 含 pathway / seven_axis_report / critic_report /
                parameter_report / pipeline_trace 等。

        Returns:
            ``(primary_cause, root_cause_layer, suspected_module)`` 三元组。
        """
        # 提取 run_rca 输入
        pathway = str(rca_inputs.get("pathway", "unknown"))
        seven_axis_report = rca_inputs.get("seven_axis_report")
        critic_report = rca_inputs.get("critic_report")
        parameter_report = rca_inputs.get("parameter_report")
        pipeline_trace = rca_inputs.get("pipeline_trace")

        # 测试覆盖：rca_inputs 直接含 primary_cause
        override_primary = rca_inputs.get("primary_cause")

        primary_cause = ""
        suspected_module = ""
        if override_primary and isinstance(override_primary, str) and override_primary.strip():
            primary_cause = override_primary.strip()
        else:
            try:
                rca_report = run_rca(
                    pathway=pathway,
                    seven_axis_report=seven_axis_report,
                    critic_report=critic_report,
                    parameter_report=parameter_report,
                    pipeline_trace=pipeline_trace,
                )
                primary_cause = str(getattr(rca_report, "primary_cause", "") or "")
                # 取第一条 cause 的 suspected_module
                causes = getattr(rca_report, "causes", []) or []
                if causes:
                    suspected_module = str(
                        getattr(causes[0], "suspected_module", "") or ""
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("_run_f5_rca: run_rca 异常: %s", exc)
                primary_cause = ""

        root_cause_layer = self._infer_rca_layer(primary_cause)
        # 若 suspected_module 仍为空，根据 layer 给一个默认模块名
        if not suspected_module:
            suspected_module = root_cause_layer
        return primary_cause, root_cause_layer, suspected_module

    def _infer_rca_layer(self, primary_cause: str) -> str:
        """根据 primary_cause 文本关键词推断 RCA 层。

        Args:
            primary_cause: RCA primary_cause 文本。

        Returns:
            ``RCA_PRIORITY`` 之一；无关键词命中时返回 ``"validation"``（保守）。
        """
        text = (primary_cause or "").lower()
        if not text:
            return "validation"
        for keywords, layer in _RCA_KEYWORD_MAP:
            for kw in keywords:
                if kw.lower() in text:
                    return layer
        return "validation"

    def _render_benchmark_summary(
        self,
        benchmark_results: dict[str, Any],
    ) -> str:
        """渲染 benchmark 结果摘要 Markdown。

        Args:
            benchmark_results: 含 5 个分项 key 的 dict。

        Returns:
            Markdown 字符串。
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines: list[str] = [
            "# F5 Loop Step 2: Benchmark Results",
            "",
            f"**Generated**: {timestamp}",
            "",
        ]
        if not benchmark_results:
            lines.append("(no benchmark results)")
            return "\n".join(lines)

        # 每通路 / 每分项 PASS/FAIL 摘要
        lines.append("## Per-Component Summary")
        lines.append("")
        lines.append("| Component | Status / Count |")
        lines.append("|-----------|----------------|")

        vm = benchmark_results.get("validation_matrix_result") or {}
        axes = vm.get("axes", []) if isinstance(vm, dict) else []
        overall = vm.get("overall_status", "") if isinstance(vm, dict) else ""
        if axes:
            pass_count = sum(
                1 for a in axes
                if (a.get("status", "") if isinstance(a, dict) else "")
                == "PASS"
            )
            lines.append(
                f"| validation_matrix | {pass_count}/{len(axes)} axes PASS "
                f"(overall={overall}) |"
            )
        else:
            lines.append(f"| validation_matrix | 0 axes (overall={overall}) |")

        cm = benchmark_results.get("curve_metrics_result") or {}
        metrics = cm.get("metrics", []) if isinstance(cm, dict) else []
        if metrics:
            pass_count = sum(
                1 for m in metrics
                if (m.get("passed", False) if isinstance(m, dict) else False)
            )
            lines.append(
                f"| curve_metrics | {pass_count}/{len(metrics)} metrics passed |"
            )
        else:
            lines.append("| curve_metrics | 0 metrics |")

        lit = benchmark_results.get("literature_result") or {}
        if isinstance(lit, dict):
            canonical = lit.get("canonical_pmids", []) or []
            retrieved = lit.get("retrieved_pmids", []) or []
            lines.append(
                f"| literature | canonical={len(canonical)}, "
                f"retrieved={len(retrieved)} |"
            )
        else:
            lines.append("| literature | (none) |")

        ev = benchmark_results.get("evidence_result") or {}
        if isinstance(ev, dict):
            grounded = ev.get("grounded_sentence_count", 0) or 0
            total = ev.get("total_sentence_count", 0) or 0
            lines.append(
                f"| evidence | grounded={grounded}/{total} sentences |"
            )
        else:
            lines.append("| evidence | (none) |")

        exp = benchmark_results.get("experiment_result") or {}
        if isinstance(exp, dict):
            exps = exp.get("experiments", []) or []
            passed = sum(
                1 for e in exps
                if (e.get("passed", False) if isinstance(e, dict) else False)
            )
            lines.append(
                f"| experiment | {passed}/{len(exps)} experiments passed |"
            )
        else:
            lines.append("| experiment | (none) |")

        return "\n".join(lines)

    def _render_f5_reject_report(
        self,
        report: F5LoopReport,
        comparison: ScoreComparison,
    ) -> str:
        """渲染 f5_reject_report.md。

        Args:
            report: F5LoopReport。
            comparison: ScoreComparison。

        Returns:
            Markdown 字符串。
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines: list[str] = [
            "# F5 Reject Report",
            "",
            f"**Generated**: {timestamp}",
            f"**Commit SHA**: {report.commit_sha}",
            f"**Decision**: REJECT",
            "",
            "## Score Comparison",
            "",
            f"- **Before**: {report.before_score:.2f}",
            f"- **After**: {report.after_score:.2f}",
            f"- **Delta**: {report.delta:+.2f}",
            "",
            "## Component Delta",
            "",
            "| Component | Delta |",
            "|-----------|-------|",
        ]
        for name, delta in comparison.delta_components.items():
            lines.append(f"| {name} | {delta:+.2f} |")
        lines.extend([
            "",
            "## Root Cause",
            "",
            f"- **root_cause_layer**: {report.root_cause_layer}",
            f"- **suspected_module**: {report.suspected_module}",
            f"- **repair_suggestion**: {report.repair_suggestion}",
            "",
            "## Rollback",
            "",
            "- **rollback_triggered**: True",
            "- **suggested_action**: set "
            "V4_SCIENTIFIC_REVIEWER_ENABLED=false",
            "",
        ])
        return "\n".join(lines)


__all__ = [
    "LoopStatus",
    "LoopIteration",
    "LoopResult",
    "LoopController",
    # F5 迭代纪律（Task 9）
    "F5_STEPS",
    "RCA_PRIORITY",
    "F5_LOG_DIR",
    "F5StepRecord",
    "F5LoopReport",
]
