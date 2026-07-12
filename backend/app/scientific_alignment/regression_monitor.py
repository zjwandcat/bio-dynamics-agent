# BioDynamics Agent v4 - Scientific Alignment Loop: Regression Monitor (Task 14)
#
# 回归基准与自动 Rollback：比较前后两次 7 轴验证结果，检测 Regression / Fix / No-Change。
#
# 设计目标：
#   Loop Controller 每轮迭代后调用 RegressionMonitor.compare，检测本轮是否引入回归。
#   - Regression：某轴从 pass 变 fail，或分数下降 > 0.1
#   - Fix：某轴从 fail 变 pass，或分数上升 > 0.1
#   - No-Change：分数变化 ≤ 0.1（0.05~0.1 为容忍区间，也算 No-Change）
#
# 自动 Rollback（SubTask 14.2）：
#   检测到回归时输出 rollback_suggestion（建议关闭的 Feature Flag 列表）。
#   不直接修改 env，仅输出建议，由 Loop Controller / 运维人员决定是否执行。
#
# 全量回归（SubTask 14.3）：
#   任一修复（Fix）触发全量 10 通路回归（本模块仅实现比较逻辑，
#   "触发全量"由 Loop Controller 调用）。
#
# Feature Flag 守护：
#   SA_LOOP_TERMINATION 默认 OFF。关闭时返回 skipped=True 的空报告，不阻塞主流程。
#   铁律：V4_SCIENTIFIC_ALIGNMENT_ENABLED=false 时，子 Flag 永远不生效
#         （由 settings.is_sa_feature_enabled 强制校验）。
#
# 安全设计：
#   - 纯诊断/比较工具，无副作用，不修改 env 或 Feature Flag
#   - 所有外部输入做 None / 空值防御
#   - 不引入新依赖（仅标准库 + 已完成组件 + app.config.settings）
#
# 依赖：
#   - app.config.settings（Feature Flag 守护）
#   - app.scientific_alignment.seven_axis_validator.SevenAxisReport（类型引用）
#
# 核心导出：
#   from app.scientific_alignment.regression_monitor import (
#       ChangeType, AxisChange, RegressionReport,
#       RegressionMonitor, run_regression_check,
#   )

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.config import settings
from app.scientific_alignment.seven_axis_validator import SevenAxisReport

logger = logging.getLogger(__name__)


# =============================================================================
# 常量
# =============================================================================

# 分数变化阈值：
# >0.1 为 Regression/Fix，≤0.1 为 No-Change（0.05~0.1 为容忍区间）
_REGRESSION_SCORE_THRESHOLD: float = 0.1
_FIX_SCORE_THRESHOLD: float = 0.1

# overall_delta 回归阈值：整体下降 > 0.1 时标记 has_regression
_OVERALL_REGRESSION_THRESHOLD: float = 0.1

# 轴名 → Feature Flag 映射（用于 rollback 建议）
# - Mechanism / Dynamics 退步 → SA_MECHANISM_GRAPH, SA_CONSISTENCY_CHECKER
# - BioModels 退步 → SA_BIOMODELS_ORACLE
# - Evidence / Literature 退步 → SA_EVIDENCE_FUSION
# - Experiment / Discussion 退步 → 无对应 flag（建议人工检查）
_AXIS_TO_FLAGS: dict[str, list[str]] = {
    "mechanism": ["SA_MECHANISM_GRAPH", "SA_CONSISTENCY_CHECKER"],
    "dynamics": ["SA_MECHANISM_GRAPH", "SA_CONSISTENCY_CHECKER"],
    "biomodels": ["SA_BIOMODELS_ORACLE"],
    "evidence": ["SA_EVIDENCE_FUSION"],
    "literature": ["SA_EVIDENCE_FUSION"],
    # experiment / discussion 无对应 flag，建议人工检查
}


# =============================================================================
# 数据类
# =============================================================================

class ChangeType(str, Enum):
    """轴变化类型。"""

    REGRESSION = "regression"
    FIX = "fix"
    NO_CHANGE = "no_change"


@dataclass
class AxisChange:
    """单个轴的前后变化。

    Attributes:
        axis_name: 轴名称（mechanism / dynamics / biomodels / ...）。
        change_type: 变化类型（REGRESSION / FIX / NO_CHANGE）。
        previous_score: 前一次得分。
        current_score: 当前得分。
        delta: 分数变化（current - previous）。
        previous_passed: 前一次是否通过。
        current_passed: 当前是否通过。
    """

    axis_name: str
    change_type: ChangeType
    previous_score: float
    current_score: float
    delta: float
    previous_passed: bool
    current_passed: bool


@dataclass
class RegressionReport:
    """回归比较报告。

    Attributes:
        enabled: Feature Flag 是否启用。
        skipped: 是否跳过（Flag 关闭时为 True）。
        pathway: 通路标识。
        changes: 各轴的变化列表。
        has_regression: 是否存在任一回归。
        has_fix: 是否存在任一修复。
        regression_axes: 回归轴名称列表。
        fix_axes: 修复轴名称列表。
        rollback_suggestion: 建议关闭的 Feature Flag 列表。
        previous_overall: 前一次整体得分。
        current_overall: 当前整体得分。
        overall_delta: 整体得分变化（current - previous）。
    """

    enabled: bool
    skipped: bool = False
    pathway: str = ""
    changes: list[AxisChange] = field(default_factory=list)
    has_regression: bool = False
    has_fix: bool = False
    regression_axes: list[str] = field(default_factory=list)
    fix_axes: list[str] = field(default_factory=list)
    rollback_suggestion: list[str] = field(default_factory=list)
    previous_overall: float = 0.0
    current_overall: float = 0.0
    overall_delta: float = 0.0


# =============================================================================
# 主类
# =============================================================================

class RegressionMonitor:
    """比较前后两次 7 轴验证结果，检测回归与修复。

    使用方式::

        monitor = RegressionMonitor()
        monitor.record("EGFR", report1)
        # 下一轮迭代后
        report2 = run_seven_axis_validation(...)
        diff = monitor.compare("EGFR", report2)
        if diff.has_regression:
            print(diff.rollback_suggestion)
    """

    def __init__(self) -> None:
        self._previous_reports: dict[str, SevenAxisReport] = {}

    def record(self, pathway: str, report: Any) -> None:
        """记录本次 7 轴结果，供下次比较。

        Args:
            pathway: 通路标识。
            report: SevenAxisReport 实例，None 时忽略。
        """
        if report is None:
            return
        self._previous_reports[pathway] = report

    def compare(self, pathway: str, current_report: Any) -> RegressionReport:
        """比较 current 与上次记录的 report。

        若无历史则返回 No-Change 全空报告。

        Args:
            pathway: 通路标识。
            current_report: 当前 SevenAxisReport。

        Returns:
            RegressionReport。
        """
        previous_report = self._previous_reports.get(pathway)
        return run_regression_check(pathway, previous_report, current_report)

    def suggest_rollback(self, regression_axes: list[str]) -> list[str]:
        """根据退步的轴推测应关闭的 flag。

        映射规则：
            - Mechanism / Dynamics 退步 → SA_MECHANISM_GRAPH, SA_CONSISTENCY_CHECKER
            - BioModels 退步 → SA_BIOMODELS_ORACLE
            - Evidence / Literature 退步 → SA_EVIDENCE_FUSION
            - Experiment / Discussion 退步 → 无对应 flag，建议人工检查

        Args:
            regression_axes: 回归轴名称列表。

        Returns:
            建议关闭的 flag 列表（去重，保留首次出现顺序）。
        """
        return _suggest_rollback_impl(regression_axes)


# =============================================================================
# 便捷函数
# =============================================================================

def run_regression_check(
    pathway: str,
    previous_report: Any | None,
    current_report: Any | None,
) -> RegressionReport:
    """一次性比较前后两次 7 轴验证结果。

    previous 为 None 时所有轴返回 NO_CHANGE。
    Feature Flag SA_LOOP_TERMINATION 关闭时返回 skipped 报告。

    Args:
        pathway: 通路标识。
        previous_report: 前一次 SevenAxisReport，None 时全 No-Change。
        current_report: 当前 SevenAxisReport。

    Returns:
        RegressionReport。
    """
    # -------------------------------------------------------------------------
    # Feature Flag 守护：SA_LOOP_TERMINATION 默认 OFF
    # 铁律：V4_SCIENTIFIC_ALIGNMENT_ENABLED=false 时，子 Flag 永远不生效
    # -------------------------------------------------------------------------
    if not settings.is_sa_feature_enabled("LOOP_TERMINATION"):
        return RegressionReport(
            enabled=False,
            skipped=True,
            pathway=pathway,
        )

    # -------------------------------------------------------------------------
    # previous 为 None 或 skipped → 无历史，返回 No-Change 全空
    # -------------------------------------------------------------------------
    has_previous = (
        previous_report is not None
        and not getattr(previous_report, "skipped", False)
    )

    if not has_previous:
        # 无历史时：current_overall 仍记录当前得分，但 overall_delta=0.0
        current_overall = _compute_overall(current_report)
        return RegressionReport(
            enabled=True,
            skipped=False,
            pathway=pathway,
            changes=[],
            has_regression=False,
            has_fix=False,
            regression_axes=[],
            fix_axes=[],
            rollback_suggestion=[],
            previous_overall=0.0,
            current_overall=current_overall,
            overall_delta=0.0,
        )

    # -------------------------------------------------------------------------
    # 正常比较：逐轴对比
    # -------------------------------------------------------------------------
    prev_axes_by_name: dict[str, Any] = {
        axis.axis_name: axis
        for axis in getattr(previous_report, "axes", [])
    }
    curr_axes_by_name: dict[str, Any] = {
        axis.axis_name: axis
        for axis in getattr(current_report, "axes", []) if current_report is not None
    }

    # 遍历所有轴名（取并集，保留顺序：先 previous 顺序，再 current 独有）
    all_axis_names: list[str] = []
    seen_names: set[str] = set()
    for axis in getattr(previous_report, "axes", []):
        if axis.axis_name not in seen_names:
            all_axis_names.append(axis.axis_name)
            seen_names.add(axis.axis_name)
    if current_report is not None:
        for axis in getattr(current_report, "axes", []):
            if axis.axis_name not in seen_names:
                all_axis_names.append(axis.axis_name)
                seen_names.add(axis.axis_name)

    changes: list[AxisChange] = []
    regression_axes: list[str] = []
    fix_axes: list[str] = []

    for axis_name in all_axis_names:
        prev_axis = prev_axes_by_name.get(axis_name)
        curr_axis = curr_axes_by_name.get(axis_name)

        prev_score = float(getattr(prev_axis, "score", 0.0)) if prev_axis else 0.0
        curr_score = float(getattr(curr_axis, "score", 0.0)) if curr_axis else 0.0
        prev_passed = (
            getattr(prev_axis, "status", "") == "passed" if prev_axis else False
        )
        curr_passed = (
            getattr(curr_axis, "status", "") == "passed" if curr_axis else False
        )

        delta = curr_score - prev_score
        change_type = _classify_change(prev_passed, curr_passed, delta)

        changes.append(AxisChange(
            axis_name=axis_name,
            change_type=change_type,
            previous_score=prev_score,
            current_score=curr_score,
            delta=delta,
            previous_passed=prev_passed,
            current_passed=curr_passed,
        ))

        if change_type == ChangeType.REGRESSION:
            regression_axes.append(axis_name)
        elif change_type == ChangeType.FIX:
            fix_axes.append(axis_name)

    # -------------------------------------------------------------------------
    # 整体得分比较
    # -------------------------------------------------------------------------
    previous_overall = _compute_overall(previous_report)
    current_overall = _compute_overall(current_report)
    overall_delta = current_overall - previous_overall

    # -------------------------------------------------------------------------
    # has_regression / has_fix 判定
    # -------------------------------------------------------------------------
    has_regression = bool(regression_axes) or overall_delta < -_OVERALL_REGRESSION_THRESHOLD
    has_fix = bool(fix_axes)

    # -------------------------------------------------------------------------
    # rollback 建议
    # -------------------------------------------------------------------------
    rollback_suggestion = _suggest_rollback_impl(regression_axes)

    logger.info(
        "回归比较完成: pathway=%s, changes=%d, has_regression=%s, has_fix=%s, "
        "regression_axes=%s, fix_axes=%s, overall_delta=%.3f, rollback=%s",
        pathway, len(changes), has_regression, has_fix,
        regression_axes, fix_axes, overall_delta, rollback_suggestion,
    )

    return RegressionReport(
        enabled=True,
        skipped=False,
        pathway=pathway,
        changes=changes,
        has_regression=has_regression,
        has_fix=has_fix,
        regression_axes=regression_axes,
        fix_axes=fix_axes,
        rollback_suggestion=rollback_suggestion,
        previous_overall=previous_overall,
        current_overall=current_overall,
        overall_delta=overall_delta,
    )


# =============================================================================
# 私有辅助函数
# =============================================================================

def _classify_change(
    previous_passed: bool,
    current_passed: bool,
    delta: float,
) -> ChangeType:
    """根据 pass/fail 状态变化与分数变化判定 ChangeType。

    规则（优先级从高到低）：
        1. pass → fail → REGRESSION
        2. fail → pass → FIX
        3. 分数下降 > 0.1 → REGRESSION
        4. 分数上升 > 0.1 → FIX
        5. 其余（|delta| ≤ 0.1）→ NO_CHANGE（含 0.05~0.1 容忍区间）

    Args:
        previous_passed: 前一次是否通过。
        current_passed: 当前是否通过。
        delta: 分数变化（current - previous）。

    Returns:
        ChangeType。
    """
    # pass → fail：回归
    if previous_passed and not current_passed:
        return ChangeType.REGRESSION
    # fail → pass：修复
    if not previous_passed and current_passed:
        return ChangeType.FIX
    # 分数下降 > 0.1：回归
    if delta < -_REGRESSION_SCORE_THRESHOLD:
        return ChangeType.REGRESSION
    # 分数上升 > 0.1：修复
    if delta > _FIX_SCORE_THRESHOLD:
        return ChangeType.FIX
    # 其余：No-Change（含 0.05~0.1 容忍区间）
    return ChangeType.NO_CHANGE


def _compute_overall(report: Any) -> float:
    """提取报告的整体 Confidence（SevenAxisReport.overall_confidence）。

    使用 SevenAxisReport 自身的 overall_confidence（min(axes.score) * 0.9，
    低维拖累策略），与 SevenAxisValidator 的定义保持一致。

    Args:
        report: SevenAxisReport 实例，None 或 skipped 时返回 0.0。

    Returns:
        report.overall_confidence，缺失时返回 0.0。
    """
    if report is None:
        return 0.0
    if getattr(report, "skipped", False):
        return 0.0
    return float(getattr(report, "overall_confidence", 0.0))


def _suggest_rollback_impl(regression_axes: list[str]) -> list[str]:
    """根据退步的轴推测应关闭的 flag（模块级实现）。

    Args:
        regression_axes: 回归轴名称列表。

    Returns:
        建议关闭的 flag 列表（去重，保留首次出现顺序）。
    """
    flags: list[str] = []
    seen: set[str] = set()
    for axis in regression_axes:
        axis_flags = _AXIS_TO_FLAGS.get(axis.lower(), [])
        for flag in axis_flags:
            if flag not in seen:
                seen.add(flag)
                flags.append(flag)
    return flags


__all__ = [
    "ChangeType",
    "AxisChange",
    "RegressionReport",
    "RegressionMonitor",
    "run_regression_check",
]
