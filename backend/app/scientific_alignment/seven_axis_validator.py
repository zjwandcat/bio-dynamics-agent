# BioDynamics Agent v4 - Scientific Alignment Loop: Seven-Axis Validation Pyramid (Task 12)
#
# 7 轴 Scientific Validation Pyramid：每轴独立评分，任一轴 Fail 整体降 Confidence。
# 7 轴：Mechanism / Dynamics / BioModels / Literature / Experiment / Discussion / Evidence。
#
# 设计目标：
#   当前 Validation 只检查数值合理性。Task 12 升级为 7 轴验证金字塔，
#   每轴独立评分（0.0-1.0）并给出 passed/failed/degraded/skipped 状态。
#   任一轴 Fail 会拖低整体 Confidence（低维拖累策略）。
#
# 优雅降级（对未完成组件不阻塞）：
#   - Task 9  Evidence Fusion（未实现）      → Evidence 轴可降级
#   - Task 10 Scientific Discussion（未实现） → Discussion 轴可降级
#   - Task 11 Experiment Planner（未实现）    → Experiment 轴可降级
#   降级轴标记 "degraded"，不参与 passed 判定，不阻塞其他轴。
#   待对应组件就绪后，调用方提供输入即可自动升级为正式评分。
#
# Feature Flag 守护：
#   SA_SEVEN_AXIS 默认 OFF。关闭时返回 skipped=True 的空报告，不阻塞主流程。
#   铁律：V4_SCIENTIFIC_ALIGNMENT_ENABLED=false 时，子 Flag 永远不生效
#         （由 settings.is_sa_feature_enabled 强制校验）。
#
# 综合 Confidence 计算（spec SubTask 25.3 一致，低维拖累策略）：
#   overall_confidence = min(axis.score for axis in axes) * 0.9
#   （最低分轴拖低整体 Confidence，避免某轴极差却被均值掩盖）
#
# overall_passed 判定：
#   1. 非 degraded 轴全部 passed → 候选 True
#   2. degraded 轴数 > 3 → 强制 False（过多降级不可信）
#   degraded 轴不参与 passed 判定（待组件就绪后自动升级）
#
# 安全设计：
#   - 不引入新依赖（仅标准库 + 已完成组件 + app.config.settings）
#   - 每轴独立 try-except，单轴异常不阻塞其他轴（异常轴标记 degraded）
#   - 对未完成组件优雅降级，不抛异常
#   - 所有外部输入做 None / 空值防御，缺失输入对应轴标记 degraded
#
# 依赖：
#   - app.config.settings（Feature Flag 守护）
#   - app.scientific_alignment.mechanism_checker（Task 8）
#   - app.scientific_alignment.consistency_checker（Task 24）
#   - app.scientific_alignment.evidence_ranker（Task 3）
#   - app.scientific_alignment.canonical_loader（间接依赖，经 mechanism/consistency 引入）
#   注意：mechanism_checker 和 consistency_checker 都 import canonical_loader，
#         本文件同时 import 两者不会循环依赖。
#
# 核心导出：
#   from app.scientific_alignment.seven_axis_validator import (
#       AxisScore, SevenAxisReport, run_seven_axis_validation,
#   )

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

from app.config import settings
from app.scientific_alignment.consistency_checker import check_consistency
from app.scientific_alignment.evidence_ranker import (
    EvidenceDoc,
    EvidenceRanker,
    EvidenceType,
    load_literature_gold_standard,
)
from app.scientific_alignment.mechanism_checker import check_mechanism_alignment

logger = logging.getLogger(__name__)


# =============================================================================
# 常量
# =============================================================================

# 7 轴名称（固定顺序，便于调用方按索引访问）
AXIS_MECHANISM: str = "mechanism"
AXIS_DYNAMICS: str = "dynamics"
AXIS_BIOMODELS: str = "biomodels"
AXIS_LITERATURE: str = "literature"
AXIS_EXPERIMENT: str = "experiment"
AXIS_DISCUSSION: str = "discussion"
AXIS_EVIDENCE: str = "evidence"

# degraded 轴数阈值：超过此值时 overall_passed 强制为 False（过多降级不可信）
_MAX_DEGRADED_AXES: int = 3

# Mechanism 轴：缺失关键节点时的 coverage 折扣因子
# score = coverage * (1 - 0.5) = coverage * 0.5
_CRITICAL_NODE_PENALTY: float = 0.5

# Dynamics 轴：每条违规的扣分
# score = max(0.0, 1.0 - 0.3 * len(violations))
_VIOLATION_PENALTY: float = 0.3

# BioModels 轴：overall_distance 放大系数
# score = 1.0 - min(1.0, overall_distance * 5)
# 即 distance=0.2 时 score=0，distance=0.0 时 score=1.0
_BIOMODELS_DISTANCE_SCALE: float = 5.0

# Literature 轴：通过阈值（至少 2 篇 Review 且总文献 >= 5）
_LITERATURE_MIN_REVIEWS: int = 2
_LITERATURE_MIN_TOTAL: int = 5

# Experiment 轴：通过阈值（至少 2 个实验）
_EXPERIMENT_MIN_COUNT: int = 2

# Discussion 轴：通过阈值（启发式得分 >= 0.7）
_DISCUSSION_PASS_THRESHOLD: float = 0.7

# Discussion 轴：10 问关键词（启发式检查讨论是否覆盖关键科学问题）
# 注：spec 提及"10 问关键词"，此处列出 8 个核心关键词，每命中 +0.1，最高 1.0
_DISCUSSION_KEYWORDS: tuple[str, ...] = (
    "peak", "decline", "feedback", "experiment",
    "literature", "limitation", "sensitivity", "next step",
)

# Discussion 轴：每命中一个关键词的加分
_DISCUSSION_KEYWORD_BONUS: float = 0.1

# Discussion 轴：证据引用标签（[A]/[B]/[C]/[D]/[E]，对应五源证据引用）
_DISCUSSION_CITATION_TAGS: tuple[str, ...] = ("[A]", "[B]", "[C]", "[D]", "[E]")

# Evidence 轴：五源名称（pubmed / biomodels / simulation / inference / hypothesis）
_EVIDENCE_SOURCES: tuple[str, ...] = (
    "pubmed_count", "biomodels_count",
    "simulation_count", "inference_count", "hypothesis_count",
)

# Evidence 轴：通过阈值（五源覆盖率 >= 0.6，即至少 3 源有数据）
_EVIDENCE_COVERAGE_THRESHOLD: float = 0.6

# Evidence 轴：pubmed_count 加分阈值（>= 5 篇 PubMed 文献加 0.2）
_EVIDENCE_PUBMED_THRESHOLD: int = 5

# Evidence 轴：review_count 加分阈值（>= 2 篇 Review 加 0.2）
_EVIDENCE_REVIEW_THRESHOLD: int = 2

# 降级轴的默认得分（不影响 passed 判定，但参与 min 拖累）
_DEGRADED_SCORE: float = 0.5


# =============================================================================
# 数据类
# =============================================================================
@dataclass
class AxisScore:
    """单个轴的评分。

    Attributes:
        axis_name: 轴名称（mechanism / dynamics / biomodels / literature /
            experiment / discussion / evidence）。
        score: 轴得分（0.0-1.0）。
        status: 轴状态：
            - ``"passed"``：通过
            - ``"failed"``：失败（会拖低整体 Confidence）
            - ``"degraded"``：降级（未完成组件或输入缺失，不参与 passed 判定）
            - ``"skipped"``：跳过（Feature Flag 关闭，仅用于整体报告）
        sub_scores: 子项得分 dict（用于可解释性与审计）。
        failure_reasons: 失败原因列表（status="failed" 时填充）。
        warnings: 警告信息列表（非致命问题，如降级原因）。
    """

    axis_name: str
    score: float
    status: str
    sub_scores: dict[str, float] = field(default_factory=dict)
    failure_reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class SevenAxisReport:
    """7 轴验证报告。

    Attributes:
        pathway: 被验证的通路标识（如 ``"egfr"``）。
        axes: 7 个轴的评分列表（AxisScore），按固定顺序：
            mechanism / dynamics / biomodels / literature / experiment /
            discussion / evidence。
        overall_passed: 整体是否通过（非 degraded 轴全部 passed 且
            degraded 轴数 <= 3）。
        overall_confidence: 综合 Confidence（低维拖累策略：
            min(axis.score) * 0.9）。
        failed_axes: 失败轴名称列表。
        degraded_axes: 降级轴名称列表。
        skipped: 是否跳过（Feature Flag 关闭时为 True）。
        skip_reason: 跳过原因（Feature Flag 关闭时填充）。
    """

    pathway: str
    axes: list[AxisScore]
    overall_passed: bool
    overall_confidence: float
    failed_axes: list[str] = field(default_factory=list)
    degraded_axes: list[str] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str = ""


# =============================================================================
# 轴评估函数（私有，每轴独立 try-except 确保互不阻塞）
# =============================================================================

def _evaluate_mechanism_axis(
    pathway: str,
    extracted_nodes: list[str] | None,
) -> AxisScore:
    """评估 Mechanism 轴：机制节点覆盖检查。

    调用 Task 8 的 check_mechanism_alignment，校验 Agent 提取的机制节点
    是否覆盖 Canonical required_nodes。

    评分规则：
      - score = coverage * (1 - 0.5 if missing_critical_nodes else 0)
        即缺关键节点时 coverage 打五折
      - status: passed if result.passed else failed

    输入缺失（extracted_nodes 为 None）→ 降级，不阻塞。

    Args:
        pathway: 通路标识。
        extracted_nodes: Agent 提取的机制节点列表，None 时降级。

    Returns:
        AxisScore。
    """
    axis_name = AXIS_MECHANISM

    # extracted_nodes 为 None 时降级（输入缺失，非验证失败）
    if extracted_nodes is None:
        return AxisScore(
            axis_name=axis_name,
            score=_DEGRADED_SCORE,
            status="degraded",
            sub_scores={},
            warnings=["extracted_nodes 未提供，Mechanism 轴降级"],
        )

    try:
        result = check_mechanism_alignment(
            pathway=pathway,
            extracted_nodes=extracted_nodes,
            original_confidence=1.0,
        )
    except Exception as exc:
        # Canonical 加载失败等异常 → 降级（基础设施问题，非验证失败）
        logger.warning("Mechanism 轴评估异常，降级: %s", exc)
        return AxisScore(
            axis_name=axis_name,
            score=_DEGRADED_SCORE,
            status="degraded",
            sub_scores={},
            warnings=[f"机制检查异常，降级: {exc}"],
        )

    # 计算 score：缺关键节点时 coverage 打五折
    coverage = result.coverage
    has_critical_missing = bool(result.missing_critical_nodes)
    if has_critical_missing:
        score = coverage * (1.0 - _CRITICAL_NODE_PENALTY)
    else:
        score = coverage

    # 确保 score 在 [0.0, 1.0] 范围内
    score = max(0.0, min(1.0, score))

    # status
    status = "passed" if result.passed else "failed"

    # 失败原因
    failure_reasons: list[str] = []
    if not result.passed:
        if has_critical_missing:
            failure_reasons.append(
                f"缺失关键节点（负反馈/节点链首尾）: {result.missing_critical_nodes}"
            )
        if coverage < 0.8:
            failure_reasons.append(
                f"覆盖率 {coverage:.2f} 低于阈值 0.80"
            )
        if result.missing_nodes:
            failure_reasons.append(
                f"缺失 {len(result.missing_nodes)} 个必填节点: {result.missing_nodes}"
            )

    # sub_scores
    sub_scores: dict[str, float] = {
        "coverage": coverage,
        "critical_nodes_present": 0.0 if has_critical_missing else 1.0,
    }

    return AxisScore(
        axis_name=axis_name,
        score=score,
        status=status,
        sub_scores=sub_scores,
        failure_reasons=failure_reasons,
        warnings=list(result.warnings),
    )


def _evaluate_dynamics_axis(
    pathway: str,
    simulation_metrics: dict | None,
) -> AxisScore:
    """评估 Dynamics 轴：仿真结果机制级一致性校验。

    调用 Task 24 的 check_consistency，对仿真 metrics 做机制级逻辑校验
    （如 EGFR Peak 不能晚于 ERK Peak）。

    评分规则：
      - score = 1.0 if passed else max(0.0, 1.0 - 0.3 * len(violations))
      - status: passed if report.passed else failed

    输入缺失（simulation_metrics 为 None）→ 降级，不阻塞。

    Args:
        pathway: 通路标识。
        simulation_metrics: 仿真指标 dict，None 时降级。

    Returns:
        AxisScore。
    """
    axis_name = AXIS_DYNAMICS

    # simulation_metrics 为 None 时降级
    if simulation_metrics is None:
        return AxisScore(
            axis_name=axis_name,
            score=_DEGRADED_SCORE,
            status="degraded",
            sub_scores={},
            warnings=["simulation_metrics 未提供，Dynamics 轴降级"],
        )

    try:
        report = check_consistency(pathway, simulation_metrics)
    except Exception as exc:
        # Canonical 加载失败等异常 → 降级
        logger.warning("Dynamics 轴评估异常，降级: %s", exc)
        return AxisScore(
            axis_name=axis_name,
            score=_DEGRADED_SCORE,
            status="degraded",
            sub_scores={},
            warnings=[f"一致性检查异常，降级: {exc}"],
        )

    # 计算 score
    violation_count = len(report.violations)
    if report.passed:
        score = 1.0
    else:
        score = max(0.0, 1.0 - _VIOLATION_PENALTY * violation_count)

    # 确保 score 在 [0.0, 1.0] 范围内
    score = max(0.0, min(1.0, score))

    # status
    status = "passed" if report.passed else "failed"

    # 失败原因
    failure_reasons: list[str] = []
    if not report.passed:
        for v in report.violations:
            failure_reasons.append(v.message)

    # sub_scores
    rules_evaluated = report.rules_evaluated
    sub_scores: dict[str, float] = {
        "rules_evaluated": float(rules_evaluated),
        "rules_passed": float(rules_evaluated - violation_count),
        "violation_count": float(violation_count),
    }

    # 警告：已评估规则为 0 时提示（可能 Canonical 加载失败或 metrics 不匹配）
    warnings: list[str] = []
    if rules_evaluated == 0 and report.rules_checked > 0:
        warnings.append(
            f"共 {report.rules_checked} 条规则但无一条成功评估"
            "（simulation_metrics 可能缺少必要指标）"
        )
    elif rules_evaluated == 0:
        warnings.append("无规则被评估（可能 Canonical 不存在或为空）")

    return AxisScore(
        axis_name=axis_name,
        score=score,
        status=status,
        sub_scores=sub_scores,
        failure_reasons=failure_reasons,
        warnings=warnings,
    )


def _evaluate_biomodels_axis(biomodels_report: Any | None) -> AxisScore:
    """评估 BioModels 轴：BioModels Oracle 对比报告。

    评估已运行的 BioModels Oracle 报告（BioModelsOracleReport）。
    本函数不调用 run_biomodels_oracle（避免网络调用），仅评估传入的报告。

    评分规则：
      - 报告为 None → degraded, score=0.5
      - 否则：score = 1.0 - min(1.0, overall_distance * 5)
        （distance=0 → score=1.0, distance>=0.2 → score=0.0）
      - status: passed if report.status=="passed" else failed

    Args:
        biomodels_report: BioModelsOracleReport 实例，None 时降级。

    Returns:
        AxisScore。
    """
    axis_name = AXIS_BIOMODELS

    # 报告为 None 时降级
    if biomodels_report is None:
        return AxisScore(
            axis_name=axis_name,
            score=_DEGRADED_SCORE,
            status="degraded",
            sub_scores={},
            warnings=["biomodels_report 未提供，BioModels 轴降级"],
        )

    # 安全读取报告字段（biomodels_report 类型为 Any，用 getattr 防御）
    overall_distance = getattr(biomodels_report, "overall_distance", float("nan"))
    max_relative_error = getattr(biomodels_report, "max_relative_error", float("nan"))
    track = getattr(biomodels_report, "track", "")
    report_status = getattr(biomodels_report, "status", "")

    # 处理 NaN：overall_distance 为 NaN 时视为最大距离（score=0.0）
    if isinstance(overall_distance, float) and math.isnan(overall_distance):
        overall_distance = 1.0
        distance_for_score = 1.0
    else:
        distance_for_score = float(overall_distance) if overall_distance else 1.0

    # 计算 score = 1.0 - min(1.0, overall_distance * 5)
    score = 1.0 - min(1.0, distance_for_score * _BIOMODELS_DISTANCE_SCALE)
    score = max(0.0, min(1.0, score))

    # status: passed if report.status=="passed" else failed
    status = "passed" if report_status == "passed" else "failed"

    # 失败原因
    failure_reasons: list[str] = []
    if status == "failed":
        if report_status == "skipped":
            failure_reasons.append("BioModels Oracle 报告状态为 skipped（Feature Flag 关闭）")
        elif report_status == "degraded":
            failure_reasons.append("BioModels Oracle 报告状态为 degraded（Track B 降级或部分失败）")
        else:
            failure_reasons.append(
                f"BioModels Oracle 报告状态为 {report_status}"
                f"（overall_distance={distance_for_score:.4f}）"
            )

    # sub_scores（track 存为字符串，Python 不强制类型注解）
    sub_scores: dict[str, float] = {
        "overall_distance": distance_for_score,
        "max_relative_error": (
            float(max_relative_error)
            if (isinstance(max_relative_error, (int, float))
                and not (isinstance(max_relative_error, float)
                         and math.isnan(max_relative_error)))
            else 1.0
        ),
        "track": track,  # type: ignore[assignment]  # track 为 "A"/"B"/"" 字符串
    }

    return AxisScore(
        axis_name=axis_name,
        score=score,
        status=status,
        sub_scores=sub_scores,
        failure_reasons=failure_reasons,
    )


def _evaluate_literature_axis(
    pathway: str,
    cited_pmids: list[str] | None,
    evidence_docs: list[Any] | None,
) -> AxisScore:
    """评估 Literature 轴：文献覆盖与证据等级检查。

    用 EvidenceRanker（Task 3）对 evidence_docs 排序，统计 Review 数量
    （EvidenceType.REVIEW=5）与机制论文数量（MECHANISM_PAPER=4）。

    评分规则：
      - score = min(1.0, (review_count / 2.0) * 0.5 + (total_count / 5.0) * 0.5)
        （Review 数与总文献数各占 50% 权重，分别以 2 篇 / 5 篇为满分基准）
      - status: passed if review_count >= 2 and total_count >= 5 else failed

    Args:
        pathway: 通路标识（用于加载文献级 Gold Standard）。
        cited_pmids: 引用的 PMID 列表（如 ["PMID:12451180"]），用于统计总文献数。
        evidence_docs: EvidenceDoc 列表（含元数据），可为 None 或空。

    Returns:
        AxisScore。
    """
    axis_name = AXIS_LITERATURE

    # 统计总文献数：优先用 cited_pmids，其次用 evidence_docs
    if cited_pmids:
        total_count = len(cited_pmids)
    elif evidence_docs:
        total_count = len(evidence_docs)
    else:
        total_count = 0

    # 加载文献级 Gold Standard（失败时降级为空 dict，不影响轴评估）
    try:
        gold_standard = load_literature_gold_standard(pathway)
    except Exception as exc:
        logger.warning("加载文献 Gold Standard 失败，使用空 Gold Standard: %s", exc)
        gold_standard = {}

    # 创建 EvidenceRanker
    ranker = EvidenceRanker(gold_standard=gold_standard)

    # 统计 Review 与机制论文数量
    review_count = 0
    mech_count = 0

    if evidence_docs:
        # 用 EvidenceRanker 排序（确保经典文献优先，同时利用 Gold Standard 分类）
        try:
            ranked_docs = ranker.rank(evidence_docs)
        except Exception:
            # 排序失败时直接用原始列表
            ranked_docs = list(evidence_docs)

        for doc in ranked_docs:
            try:
                # 优先用 doc 自身的 evidence_type，再用 ranker.classify 补充
                doc_type = getattr(doc, "evidence_type", None)
                if doc_type is None:
                    pmid = getattr(doc, "pmid", "")
                    doc_type = ranker.classify(pmid)
                # 取 Gold Standard 分类与 doc 自身分类的较高优先级
                pmid = getattr(doc, "pmid", "")
                gold_type = ranker.classify(pmid)
                final_type = max(doc_type, gold_type)

                if final_type == EvidenceType.REVIEW:
                    review_count += 1
                elif final_type == EvidenceType.MECHANISM_PAPER:
                    mech_count += 1
            except Exception:
                continue

    # 计算 score
    score = min(
        1.0,
        (review_count / 2.0) * 0.5 + (total_count / 5.0) * 0.5,
    )
    score = max(0.0, min(1.0, score))

    # status: passed if review_count >= 2 and total_count >= 5
    passed = (review_count >= _LITERATURE_MIN_REVIEWS
              and total_count >= _LITERATURE_MIN_TOTAL)
    status = "passed" if passed else "failed"

    # 失败原因
    failure_reasons: list[str] = []
    if not passed:
        if review_count < _LITERATURE_MIN_REVIEWS:
            failure_reasons.append(
                f"Review 文献数 {review_count} 低于阈值 {_LITERATURE_MIN_REVIEWS}"
            )
        if total_count < _LITERATURE_MIN_TOTAL:
            failure_reasons.append(
                f"总文献数 {total_count} 低于阈值 {_LITERATURE_MIN_TOTAL}"
            )

    # sub_scores
    sub_scores: dict[str, float] = {
        "total_cited": float(total_count),
        "review_count": float(review_count),
        "mechanism_papers": float(mech_count),
    }

    return AxisScore(
        axis_name=axis_name,
        score=score,
        status=status,
        sub_scores=sub_scores,
        failure_reasons=failure_reasons,
    )


def _evaluate_experiment_axis(experiments: list[dict] | None) -> AxisScore:
    """评估 Experiment 轴：实验规划覆盖检查（降级轴）。

    Task 11 Experiment Planner 尚未实现，此轴对空输入降级。
    若调用方提供了实验列表，则正常评分。

    评分规则：
      - 为空（None 或 []）→ degraded, score=0.5, warning
      - 否则：score = min(1.0, len(experiments) / 2.0)
      - status: passed if len >= 2 else failed

    Args:
        experiments: 实验列表（list[dict]），每项可含 "mechanism_node" 键。

    Returns:
        AxisScore。
    """
    axis_name = AXIS_EXPERIMENT

    # 为空时降级（Task 11 未实现）
    if not experiments:
        return AxisScore(
            axis_name=axis_name,
            score=_DEGRADED_SCORE,
            status="degraded",
            sub_scores={"experiment_count": 0.0, "mechanism_linked": 0.0},
            warnings=["Experiment planner not yet integrated"],
        )

    # 计算 score
    exp_count = len(experiments)
    score = min(1.0, exp_count / float(_EXPERIMENT_MIN_COUNT))
    score = max(0.0, min(1.0, score))

    # status: passed if len >= 2
    passed = exp_count >= _EXPERIMENT_MIN_COUNT
    status = "passed" if passed else "failed"

    # 统计 mechanism_linked：含 "mechanism_node" 键的实验数
    mechanism_linked = sum(
        1 for exp in experiments
        if isinstance(exp, dict) and "mechanism_node" in exp
    )

    # 失败原因
    failure_reasons: list[str] = []
    if not passed:
        failure_reasons.append(
            f"实验数 {exp_count} 低于阈值 {_EXPERIMENT_MIN_COUNT}"
        )

    # sub_scores
    sub_scores: dict[str, float] = {
        "experiment_count": float(exp_count),
        "mechanism_linked": float(mechanism_linked),
    }

    return AxisScore(
        axis_name=axis_name,
        score=score,
        status=status,
        sub_scores=sub_scores,
        failure_reasons=failure_reasons,
    )


def _evaluate_discussion_axis(discussion_content: str) -> AxisScore:
    """评估 Discussion 轴：科学讨论完整性检查（降级轴）。

    Task 10 Scientific Discussion 尚未实现，此轴对空输入降级。
    若调用方提供了讨论内容，则做启发式关键词检查。

    评分规则：
      - 为空 → degraded, score=0.5
      - 否则做启发式检查：
        检查 10 问关键词（peak / decline / feedback / experiment / literature /
        limitation / sensitivity / next step），每个命中 +0.1，最高 1.0
      - status: passed if score >= 0.7 else failed

    Args:
        discussion_content: 讨论内容文本，空字符串时降级。

    Returns:
        AxisScore。
    """
    axis_name = AXIS_DISCUSSION

    # 为空时降级（Task 10 未实现）
    if not discussion_content:
        return AxisScore(
            axis_name=axis_name,
            score=_DEGRADED_SCORE,
            status="degraded",
            sub_scores={"questions_covered": 0.0, "evidence_cited": 0.0},
            warnings=["Scientific discussion not yet integrated"],
        )

    # 启发式：检查 10 问关键词覆盖
    content_lower = discussion_content.lower()
    questions_covered = 0
    matched_keywords: list[str] = []

    for keyword in _DISCUSSION_KEYWORDS:
        if keyword in content_lower:
            questions_covered += 1
            matched_keywords.append(keyword)

    # score = 每命中 +0.1，最高 1.0
    score = min(1.0, questions_covered * _DISCUSSION_KEYWORD_BONUS)
    score = max(0.0, min(1.0, score))

    # 统计证据引用标签 [A]/[B]/[C]/[D]/[E]
    evidence_cited = sum(
        1 for tag in _DISCUSSION_CITATION_TAGS if tag in discussion_content
    )

    # status: passed if score >= 0.7
    passed = score >= _DISCUSSION_PASS_THRESHOLD
    status = "passed" if passed else "failed"

    # 失败原因
    failure_reasons: list[str] = []
    if not passed:
        failure_reasons.append(
            f"讨论启发式得分 {score:.2f} 低于阈值 {_DISCUSSION_PASS_THRESHOLD}"
            f"（命中关键词: {matched_keywords}）"
        )

    # sub_scores
    sub_scores: dict[str, float] = {
        "questions_covered": float(questions_covered),
        "evidence_cited": float(evidence_cited),
    }

    return AxisScore(
        axis_name=axis_name,
        score=score,
        status=status,
        sub_scores=sub_scores,
        failure_reasons=failure_reasons,
    )


def _evaluate_evidence_axis(evidence_sources: dict | None) -> AxisScore:
    """评估 Evidence 轴：五源证据覆盖检查（降级轴）。

    Task 9 Evidence Fusion 尚未实现，此轴对空输入降级。
    若调用方提供了 evidence_sources dict，则正常评分。

    评分规则：
      - evidence_sources 为 None → degraded, score=0.5
      - 否则：
        五源覆盖率 = 命中源数 / 5
        （命中 = 该源 count > 0）
        score = coverage * 0.6 + (pubmed_count >= 5) * 0.2 + (review_count >= 2) * 0.2
      - status: passed if coverage >= 0.6 else failed

    五源：pubmed_count / biomodels_count / simulation_count /
          inference_count / hypothesis_count

    Args:
        evidence_sources: 证据源统计 dict，None 时降级。

    Returns:
        AxisScore。
    """
    axis_name = AXIS_EVIDENCE

    # 为 None 时降级（Task 9 未实现）
    if evidence_sources is None:
        return AxisScore(
            axis_name=axis_name,
            score=_DEGRADED_SCORE,
            status="degraded",
            sub_scores={"sources_covered": 0.0, "pubmed_count": 0.0, "review_count": 0.0},
            warnings=["Evidence fusion not yet integrated"],
        )

    # 统计五源覆盖
    sources_covered = 0
    for source_key in _EVIDENCE_SOURCES:
        count = evidence_sources.get(source_key, 0)
        if count and count > 0:
            sources_covered += 1

    coverage = sources_covered / float(len(_EVIDENCE_SOURCES))

    # 读取 pubmed_count 与 review_count
    pubmed_count = int(evidence_sources.get("pubmed_count", 0) or 0)
    # review_count 可选键（不在五源中，但调用方可额外提供）
    review_count = int(evidence_sources.get("review_count", 0) or 0)

    # 计算 score
    # coverage * 0.6 + (pubmed_count >= 5) * 0.2 + (review_count >= 2) * 0.2
    # 注：Python 中 bool * float = float（True=1.0, False=0.0）
    score = (
        coverage * 0.6
        + float(pubmed_count >= _EVIDENCE_PUBMED_THRESHOLD) * 0.2
        + float(review_count >= _EVIDENCE_REVIEW_THRESHOLD) * 0.2
    )
    score = max(0.0, min(1.0, score))

    # status: passed if coverage >= 0.6
    passed = coverage >= _EVIDENCE_COVERAGE_THRESHOLD
    status = "passed" if passed else "failed"

    # 失败原因
    failure_reasons: list[str] = []
    if not passed:
        failure_reasons.append(
            f"五源覆盖率 {coverage:.2f} 低于阈值 {_EVIDENCE_COVERAGE_THRESHOLD}"
            f"（命中 {sources_covered}/{len(_EVIDENCE_SOURCES)} 源）"
        )

    # sub_scores
    sub_scores: dict[str, float] = {
        "sources_covered": float(sources_covered),
        "pubmed_count": float(pubmed_count),
        "review_count": float(review_count),
    }

    return AxisScore(
        axis_name=axis_name,
        score=score,
        status=status,
        sub_scores=sub_scores,
        failure_reasons=failure_reasons,
    )


# =============================================================================
# 主函数
# =============================================================================
def run_seven_axis_validation(
    pathway: str,
    extracted_nodes: list[str] | None = None,
    simulation_metrics: dict | None = None,
    biomodels_report: Any | None = None,  # BioModelsOracleReport
    cited_pmids: list[str] | None = None,
    evidence_docs: list[Any] | None = None,  # list[EvidenceDoc]
    experiments: list[dict] | None = None,
    discussion_content: str = "",
    evidence_sources: dict | None = None,
) -> SevenAxisReport:
    """运行 7 轴验证。

    缺失的输入对应轴标记 degraded，不阻塞其他轴。
    Feature Flag SA_SEVEN_AXIS 关闭时返回 skipped 报告。

    Args:
        pathway: 通路标识（如 ``"egfr"``）。
        extracted_nodes: Agent 提取的机制节点列表（Mechanism 轴输入）。
        simulation_metrics: 仿真指标 dict（Dynamics 轴输入）。
        biomodels_report: BioModelsOracleReport 实例（BioModels 轴输入），
            None 时该轴降级。
        cited_pmids: 引用的 PMID 列表（Literature 轴输入）。
        evidence_docs: EvidenceDoc 列表（Literature 轴输入），可为空。
        experiments: 实验列表（Experiment 轴输入），None 或空时该轴降级。
        discussion_content: 讨论内容文本（Discussion 轴输入），空时该轴降级。
        evidence_sources: 证据源统计 dict（Evidence 轴输入），None 时该轴降级。

    Returns:
        SevenAxisReport。Feature Flag 关闭时返回 skipped=True 的空报告。
    """
    # -------------------------------------------------------------------------
    # Feature Flag 守护：默认 OFF，关闭时返回 skipped=True 不阻塞
    # 铁律：V4_SCIENTIFIC_ALIGNMENT_ENABLED=false 时，子 Flag 永远不生效
    # -------------------------------------------------------------------------
    if not settings.is_sa_feature_enabled("SEVEN_AXIS"):
        return SevenAxisReport(
            pathway=pathway,
            axes=[],
            overall_passed=True,  # 不阻塞
            overall_confidence=1.0,
            failed_axes=[],
            degraded_axes=[],
            skipped=True,
            skip_reason="SA_SEVEN_AXIS disabled",
        )

    # -------------------------------------------------------------------------
    # 逐轴评估（每轴独立 try-except，单轴异常不阻塞其他轴）
    # -------------------------------------------------------------------------
    axes: list[AxisScore] = []

    # 1. Mechanism 轴
    try:
        axes.append(_evaluate_mechanism_axis(pathway, extracted_nodes))
    except Exception as exc:
        logger.error("Mechanism 轴意外异常: %s", exc)
        axes.append(AxisScore(
            axis_name=AXIS_MECHANISM,
            score=_DEGRADED_SCORE,
            status="degraded",
            warnings=[f"轴评估意外异常: {exc}"],
        ))

    # 2. Dynamics 轴
    try:
        axes.append(_evaluate_dynamics_axis(pathway, simulation_metrics))
    except Exception as exc:
        logger.error("Dynamics 轴意外异常: %s", exc)
        axes.append(AxisScore(
            axis_name=AXIS_DYNAMICS,
            score=_DEGRADED_SCORE,
            status="degraded",
            warnings=[f"轴评估意外异常: {exc}"],
        ))

    # 3. BioModels 轴
    try:
        axes.append(_evaluate_biomodels_axis(biomodels_report))
    except Exception as exc:
        logger.error("BioModels 轴意外异常: %s", exc)
        axes.append(AxisScore(
            axis_name=AXIS_BIOMODELS,
            score=_DEGRADED_SCORE,
            status="degraded",
            warnings=[f"轴评估意外异常: {exc}"],
        ))

    # 4. Literature 轴
    try:
        axes.append(_evaluate_literature_axis(pathway, cited_pmids, evidence_docs))
    except Exception as exc:
        logger.error("Literature 轴意外异常: %s", exc)
        axes.append(AxisScore(
            axis_name=AXIS_LITERATURE,
            score=_DEGRADED_SCORE,
            status="degraded",
            warnings=[f"轴评估意外异常: {exc}"],
        ))

    # 5. Experiment 轴（降级轴）
    try:
        axes.append(_evaluate_experiment_axis(experiments))
    except Exception as exc:
        logger.error("Experiment 轴意外异常: %s", exc)
        axes.append(AxisScore(
            axis_name=AXIS_EXPERIMENT,
            score=_DEGRADED_SCORE,
            status="degraded",
            warnings=[f"轴评估意外异常: {exc}"],
        ))

    # 6. Discussion 轴（降级轴）
    try:
        axes.append(_evaluate_discussion_axis(discussion_content))
    except Exception as exc:
        logger.error("Discussion 轴意外异常: %s", exc)
        axes.append(AxisScore(
            axis_name=AXIS_DISCUSSION,
            score=_DEGRADED_SCORE,
            status="degraded",
            warnings=[f"轴评估意外异常: {exc}"],
        ))

    # 7. Evidence 轴（降级轴）
    try:
        axes.append(_evaluate_evidence_axis(evidence_sources))
    except Exception as exc:
        logger.error("Evidence 轴意外异常: %s", exc)
        axes.append(AxisScore(
            axis_name=AXIS_EVIDENCE,
            score=_DEGRADED_SCORE,
            status="degraded",
            warnings=[f"轴评估意外异常: {exc}"],
        ))

    # -------------------------------------------------------------------------
    # 综合 Confidence 计算（低维拖累策略，spec SubTask 25.3 一致）
    # overall_confidence = min(axis.score for axis in axes) * 0.9
    # -------------------------------------------------------------------------
    min_score = min(axis.score for axis in axes) if axes else 0.0
    overall_confidence = min_score * 0.9

    # -------------------------------------------------------------------------
    # overall_passed 判定
    # 1. 非 degraded 轴全部 passed → 候选 True
    # 2. degraded 轴数 > 3 → 强制 False（过多降级不可信）
    # -------------------------------------------------------------------------
    failed_axes: list[str] = [
        axis.axis_name for axis in axes if axis.status == "failed"
    ]
    degraded_axes: list[str] = [
        axis.axis_name for axis in axes if axis.status == "degraded"
    ]

    # 非 degraded 轴全部 passed
    non_degraded_axes = [
        axis for axis in axes if axis.status != "degraded"
    ]
    all_non_degraded_passed = all(
        axis.status == "passed" for axis in non_degraded_axes
    )

    # degraded 轴数过多时强制 False
    too_many_degraded = len(degraded_axes) > _MAX_DEGRADED_AXES

    overall_passed = all_non_degraded_passed and not too_many_degraded

    logger.info(
        "7 轴验证完成: pathway=%s, axes=%d, overall_passed=%s, "
        "overall_confidence=%.3f, failed=%s, degraded=%s",
        pathway, len(axes), overall_passed, overall_confidence,
        failed_axes, degraded_axes,
    )

    return SevenAxisReport(
        pathway=pathway,
        axes=axes,
        overall_passed=overall_passed,
        overall_confidence=overall_confidence,
        failed_axes=failed_axes,
        degraded_axes=degraded_axes,
        skipped=False,
        skip_reason="",
    )


__all__ = [
    "AxisScore",
    "SevenAxisReport",
    "run_seven_axis_validation",
]
