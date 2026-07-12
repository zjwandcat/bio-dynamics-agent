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
#   )

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

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

logger = logging.getLogger(__name__)


# =============================================================================
# 常量
# =============================================================================

# 默认校验的 10 条通路（如未显式指定 pathways 时使用）
_DEFAULT_PATHWAYS: list[str] = [
    "EGFR",
    "MAPK",
    "PI3K_AKT",
    "mTOR",
    "JAK_STAT",
    "Wnt",
    "Notch",
    "TGF_beta",
    "Hedgehog",
    "Apoptosis",
]

# 默认日志根目录（与 multi_dim_confidence.write_breakdown_json 一致）
_DEFAULT_LOG_DIR: str = "data/sa_logs"

# 报告文件名
_REPORT_FILENAME: str = "Scientific_Alignment_Report.md"
_BENCHMARK_FILENAME: str = "benchmark_results.md"


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


__all__ = [
    "LoopStatus",
    "LoopIteration",
    "LoopResult",
    "LoopController",
]
