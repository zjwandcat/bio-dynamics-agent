# BioDynamics Agent v4 - Scientific Alignment Loop: Multi-dimensional Confidence (Task 25)
#
# 6 维置信度拆分：将单一 Confidence 数字升级为 6 维明细，避免某维极差被均值掩盖。
#
# 6 维（SubTask 25.1）：
#   Mechanism / Simulation / Evidence / BioModels / Discussion / Experiment
#
# 每维计算子项（SubTask 25.2）：
#   Mechanism 维 = 节点覆盖率 × 节点置信度 × Consistency 通过率
#   Simulation 维 = 数值验收通过率 × 参数置信度均值
#   Evidence 维 = 五源覆盖率 × PubMed≥5 × Review≥2
#   BioModels 维 = 对照表存在 × 差异合理
#   Discussion 维 = 10 问覆盖率 × Evidence 引用率
#   Experiment 维 = 机制链完整 × 实验数≥2
#
# 综合策略（SubTask 25.3）：
#   overall_confidence = min(6 维) × 0.9（低维拖累策略）
#   任一维 < 0.7 显式标注 "低可信" + 原因
#
# 可追溯（SubTask 25.4）：
#   write_breakdown_json(report, log_dir) 将子项得分写入
#   <log_dir>/run_<timestamp>/confidence_breakdown.json
#
# 报告渲染（SubTask 25.5）：
#   format_confidence_table(report) 返回 markdown 表格字符串（替代单一数字）
#
# Feature Flag 守护：
#   SA_MULTI_DIM_CONFIDENCE 默认 OFF。关闭时返回 skipped 报告，行为完全回退 v3
#   （单一数字 Confidence，由调用方自行处理）。
#   铁律：V4_SCIENTIFIC_ALIGNMENT_ENABLED=false 或 SA_MULTI_DIM_CONFIDENCE=false 时，
#         模块返回 skipped，不计算 6 维。
#
# 优雅降级：
#   - 输入参数为 None 时该维降级为 degraded（score=0.5，sub_items 空，reason="input_missing"）
#   - 非None输入但子项乘积为 0 时，dim score 下限 0.1（输入已提供，最低置信非零）
#   - 所有子项分数 clamp 到 [0.0, 1.0]
#
# 依赖：
#   - app.config.settings（Feature Flag 守护）
#   - app.scientific_alignment.parameter_grounder.compute_confidence_weight（Simulation 维）
#   - app.scientific_alignment.seven_axis_validator.SevenAxisReport（复用 7 轴数据）
#   - app.scientific_alignment.consistency_checker.ConsistencyReport（Mechanism 维）
#   - app.scientific_alignment.scientific_critic.CriticReport（最终调节因子）
#   注意：仅 import 类型用于注解与字段读取，不调用其构造函数，避免循环依赖。
#
# 核心导出：
#   from app.scientific_alignment.multi_dim_confidence import (
#       DimensionScore, MultiDimConfidenceReport,
#       compute_multi_dim_confidence,
#       write_breakdown_json, format_confidence_table,
#   )

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)


# =============================================================================
# 常量
# =============================================================================

# 6 维名称（固定顺序）
DIM_MECHANISM: str = "Mechanism"
DIM_SIMULATION: str = "Simulation"
DIM_EVIDENCE: str = "Evidence"
DIM_BIOMODELS: str = "BioModels"
DIM_DISCUSSION: str = "Discussion"
DIM_EXPERIMENT: str = "Experiment"

# 降级维的默认得分（输入缺失时）
_DEGRADED_SCORE: float = 0.5

# 非None输入的最低 dim score（避免乘积为 0 导致 overall 归零）
_MIN_NONZERO_SCORE: float = 0.1

# 低可信阈值：score < 此值时标注 low_confidence
_LOW_CONFIDENCE_THRESHOLD: float = 0.7

# 综合置信度折扣因子：overall = min(6维) × 0.9
_OVERALL_DISCOUNT: float = 0.9

# Evidence 五源标签（A=PubMed / B=BioModels / C=Simulation / D=Inference / E=Hypothesis）
_EVIDENCE_SOURCE_KEYS: tuple[str, ...] = ("A", "B", "C", "D", "E")

# Evidence 维阈值
_EVIDENCE_PUBMED_TARGET: int = 5   # PubMed≥5 满分基准
_EVIDENCE_REVIEW_TARGET: int = 2   # Review≥2 满分基准

# Experiment 维阈值
_EXPERIMENT_COUNT_TARGET: int = 2  # 实验数≥2 满分基准

# Discussion 10 问关键词（与 seven_axis_validator 保持一致）
_DISCUSSION_KEYWORDS: tuple[str, ...] = (
    "peak", "decline", "feedback", "experiment",
    "literature", "limitation", "sensitivity", "next step",
)

# Discussion Evidence 引用标签
_DISCUSSION_CITATION_TAGS: tuple[str, ...] = ("[A]", "[B]", "[C]", "[D]", "[E]")

# 默认日志目录
_DEFAULT_LOG_DIR: str = "data/sa_logs"


# =============================================================================
# 数据类（SubTask 25.1 + 数据结构要求）
# =============================================================================
@dataclass
class DimensionScore:
    """单个维度的评分。

    Attributes:
        name: 维度名称（"Mechanism" / "Simulation" / "Evidence" /
            "BioModels" / "Discussion" / "Experiment"）。
        score: 维度得分（0.0-1.0）。
        sub_items: 子项明细 dict，如
            ``{"node_coverage": 0.9, "node_confidence": 0.85, ...}``。
            降级维（输入缺失）时为空 dict。
        low_confidence: 是否低可信（score < 0.7）。
        reason: 低可信原因（low_confidence=False 时为空字符串）。
    """

    name: str
    score: float
    sub_items: dict[str, float] = field(default_factory=dict)
    low_confidence: bool = False
    reason: str = ""


@dataclass
class MultiDimConfidenceReport:
    """6 维置信度报告。

    Attributes:
        enabled: Feature Flag 是否开启。
        dimensions: 6 维评分列表（DimensionScore）。
        overall_confidence: 综合置信度 = min(6 维) × 0.9。
        critic_adjustment: Scientific Critic 调节因子（负数降分）。
        final_confidence: overall + critic_adjustment（下限 0.0）。
        skipped: Feature Flag OFF 时为 True。
    """

    enabled: bool
    dimensions: list[DimensionScore]
    overall_confidence: float
    critic_adjustment: float = 0.0
    final_confidence: float = 0.0
    skipped: bool = False


# =============================================================================
# 辅助函数
# =============================================================================
def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """将数值限制在 [lo, hi] 范围内。"""
    return max(lo, min(hi, value))


def _find_axis(seven_axis_report: Any, axis_name: str) -> Any | None:
    """从 SevenAxisReport 中按 axis_name 查找轴。

    Args:
        seven_axis_report: SevenAxisReport 实例（或 None / skipped 报告）。
        axis_name: 轴名称（小写，如 "mechanism" / "dynamics"）。

    Returns:
        AxisScore 实例；未找到或报告不可用时返回 None。
    """
    if seven_axis_report is None:
        return None
    # skipped 报告（axes 为空）→ 无可用轴
    if getattr(seven_axis_report, "skipped", False):
        return None
    axes = getattr(seven_axis_report, "axes", None)
    if not axes:
        return None
    for axis in axes:
        if getattr(axis, "axis_name", "") == axis_name:
            return axis
    return None


def _make_degraded_dim(name: str) -> DimensionScore:
    """构造降级维度（输入缺失，score=0.5）。"""
    return DimensionScore(
        name=name,
        score=_DEGRADED_SCORE,
        sub_items={},
        low_confidence=True,
        reason="input_missing",
    )


def _make_dim(
    name: str,
    sub_items: dict[str, float],
    product: float,
) -> DimensionScore:
    """根据子项乘积构造维度评分。

    非None输入时 dim score 下限 _MIN_NONZERO_SCORE（0.1），
    避免子项乘积为 0 导致 overall 归零。

    Args:
        name: 维度名称。
        sub_items: 子项明细。
        product: 子项乘积（已 clamp 到 [0.0, 1.0]）。

    Returns:
        DimensionScore。
    """
    score = _clamp(product)
    # 非None输入：下限 0.1（输入已提供，最低置信非零）
    if score < _MIN_NONZERO_SCORE:
        score = _MIN_NONZERO_SCORE
    low = score < _LOW_CONFIDENCE_THRESHOLD
    if low:
        # 生成低可信原因：列出低于 0.7 的子项
        low_items = [
            f"{k}={v:.2f}" for k, v in sub_items.items() if v < _LOW_CONFIDENCE_THRESHOLD
        ]
        reason = "; ".join(low_items) if low_items else f"score={score:.2f} < 0.70"
    else:
        reason = ""
    return DimensionScore(
        name=name,
        score=score,
        sub_items=sub_items,
        low_confidence=low,
        reason=reason,
    )


# =============================================================================
# 6 维计算函数（私有）
# =============================================================================

def _compute_mechanism_dim(
    seven_axis_report: Any,
    consistency_report: Any,
) -> DimensionScore:
    """Mechanism 维 = 节点覆盖率 × 节点置信度 × Consistency 通过率。

    - 节点覆盖率：7 轴 Mechanism 轴 sub_scores["coverage"]
    - 节点置信度：7 轴 Mechanism 轴 sub_scores["critical_nodes_present"]
    - Consistency 通过率：consistency_report 提供时用 (rules_evaluated - violations)
      / rules_evaluated；否则从 7 轴 Dynamics 轴 sub_scores 推导；均缺失时 0.5

    seven_axis_report 为 None 或 Mechanism 轴缺失 → 降级 0.5。
    """
    mech_axis = _find_axis(seven_axis_report, "mechanism")
    if mech_axis is None:
        return _make_degraded_dim(DIM_MECHANISM)

    sub_scores = getattr(mech_axis, "sub_scores", {}) or {}
    node_coverage = _clamp(float(sub_scores.get("coverage", 0.5)))
    node_confidence = _clamp(float(sub_scores.get("critical_nodes_present", 0.5)))

    # Consistency 通过率
    consistency_pass_rate: float
    if consistency_report is not None:
        rules_evaluated = int(getattr(consistency_report, "rules_evaluated", 0))
        violation_count = len(getattr(consistency_report, "violations", []) or [])
        if rules_evaluated > 0:
            consistency_pass_rate = _clamp(
                (rules_evaluated - violation_count) / float(rules_evaluated)
            )
        else:
            # 有报告但无规则被评估：无规则则视为通过（1.0），有规则但未评估则保守 0.5
            rules_checked = int(getattr(consistency_report, "rules_checked", 0))
            consistency_pass_rate = 1.0 if rules_checked == 0 else 0.5
    else:
        # consistency_report 未提供：从 7 轴 Dynamics 轴推导
        dyn_axis = _find_axis(seven_axis_report, "dynamics")
        if dyn_axis is not None:
            dyn_sub = getattr(dyn_axis, "sub_scores", {}) or {}
            rules_eval = float(dyn_sub.get("rules_evaluated", 0))
            rules_pass = float(dyn_sub.get("rules_passed", 0))
            if rules_eval > 0:
                consistency_pass_rate = _clamp(rules_pass / rules_eval)
            else:
                consistency_pass_rate = 1.0  # 无规则 → 视为通过
        else:
            consistency_pass_rate = 0.5  # 均缺失 → 中性

    sub_items = {
        "node_coverage": node_coverage,
        "node_confidence": node_confidence,
        "consistency_pass_rate": consistency_pass_rate,
    }
    product = node_coverage * node_confidence * consistency_pass_rate
    return _make_dim(DIM_MECHANISM, sub_items, product)


def _compute_simulation_dim(
    seven_axis_report: Any,
    parameter_report: Any,
) -> DimensionScore:
    """Simulation 维 = 数值验收通过率 × 参数置信度均值。

    - 数值验收通过率：7 轴 Dynamics 轴 score
    - 参数置信度均值：parameter_report 提供时用 compute_confidence_weight；否则 0.5

    seven_axis_report 为 None 或 Dynamics 轴缺失 → 降级 0.5。
    """
    dyn_axis = _find_axis(seven_axis_report, "dynamics")
    if dyn_axis is None:
        return _make_degraded_dim(DIM_SIMULATION)

    numerical_acceptance = _clamp(float(getattr(dyn_axis, "score", 0.5)))

    # 参数置信度均值
    if parameter_report is not None:
        try:
            from app.scientific_alignment.parameter_grounder import (
                compute_confidence_weight,
            )
            parameter_confidence_mean = _clamp(
                float(compute_confidence_weight(parameter_report))
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("compute_confidence_weight 异常，降级为 0.5: %s", exc)
            parameter_confidence_mean = 0.5
    else:
        parameter_confidence_mean = 0.5  # 未提供 → 中性

    sub_items = {
        "numerical_acceptance_rate": numerical_acceptance,
        "parameter_confidence_mean": parameter_confidence_mean,
    }
    product = numerical_acceptance * parameter_confidence_mean
    return _make_dim(DIM_SIMULATION, sub_items, product)


def _compute_evidence_dim(evidence_sources: dict | None) -> DimensionScore:
    """Evidence 维 = 五源覆盖率 × PubMed≥5 × Review≥2。

    - 五源覆盖率：A/B/C/D/E 中 count>0 的比例
    - PubMed≥5：min(1.0, A源count / 5)（部分信用）
    - Review≥2：min(1.0, review_count / 2)；review_count 键缺失时不惩罚（1.0）

    evidence_sources 为 None → 降级 0.5。
    """
    if evidence_sources is None:
        return _make_degraded_dim(DIM_EVIDENCE)

    # 五源覆盖率
    sources_hit = sum(
        1 for k in _EVIDENCE_SOURCE_KEYS
        if (evidence_sources.get(k, 0) or 0) > 0
    )
    five_source_coverage = _clamp(sources_hit / float(len(_EVIDENCE_SOURCE_KEYS)))

    # PubMed≥5（A 源 = PubMed）
    pubmed_count = int(evidence_sources.get("A", 0) or 0)
    pubmed_factor = _clamp(
        pubmed_count / float(_EVIDENCE_PUBMED_TARGET),
        0.0, 1.0,
    )

    # Review≥2（review_count 为可选键，缺失时不惩罚）
    if "review_count" in evidence_sources:
        review_count = int(evidence_sources.get("review_count", 0) or 0)
        review_factor = _clamp(
            review_count / float(_EVIDENCE_REVIEW_TARGET),
            0.0, 1.0,
        )
    else:
        review_count = 0
        review_factor = 1.0  # 未跟踪 Review 数，不惩罚

    sub_items = {
        "five_source_coverage": five_source_coverage,
        "pubmed_factor": pubmed_factor,
        "review_factor": review_factor,
    }
    product = five_source_coverage * pubmed_factor * review_factor
    return _make_dim(DIM_EVIDENCE, sub_items, product)


def _compute_biomodels_dim(seven_axis_report: Any) -> DimensionScore:
    """BioModels 维 = 对照表存在 × 差异合理。

    - 对照表存在：7 轴 BioModels 轴非 degraded → 1.0；degraded → 维度降级 0.5
    - 差异合理：7 轴 BioModels 轴 score（1.0 - distance×5）

    seven_axis_report 为 None 或 BioModels 轴 degraded → 降级 0.5。
    """
    bm_axis = _find_axis(seven_axis_report, "biomodels")
    if bm_axis is None:
        return _make_degraded_dim(DIM_BIOMODELS)

    # BioModels 轴 degraded（未提供 biomodels_report）→ 对照表不存在 → 降级
    status = getattr(bm_axis, "status", "")
    if status == "degraded":
        return _make_degraded_dim(DIM_BIOMODELS)

    comparison_table_exists = 1.0
    difference_reasonable = _clamp(float(getattr(bm_axis, "score", 0.5)))

    sub_items = {
        "comparison_table_exists": comparison_table_exists,
        "difference_reasonable": difference_reasonable,
    }
    product = comparison_table_exists * difference_reasonable
    return _make_dim(DIM_BIOMODELS, sub_items, product)


def _compute_discussion_dim(
    seven_axis_report: Any,
    discussion_content: str | None,
) -> DimensionScore:
    """Discussion 维 = 10 问覆盖率 × Evidence 引用率。

    - 10 问覆盖率：7 轮 Discussion 轴 score（关键词命中 × 0.1，最高 1.0）
    - Evidence 引用率：[A]/[B]/[C]/[D]/[E] 标签命中数 / 5

    discussion_content 为 None → 降级 0.5。
    seven_axis_report 为 None 且 discussion_content 为 None → 降级 0.5。
    discussion_content 非 None 但 7 轮不可用时从 discussion_content 直接计算。
    """
    if discussion_content is None:
        return _make_degraded_dim(DIM_DISCUSSION)

    # 优先从 7 轮 Discussion 轴读取
    disc_axis = _find_axis(seven_axis_report, "discussion")
    if disc_axis is not None:
        questions_coverage = _clamp(float(getattr(disc_axis, "score", 0.0)))
        disc_sub = getattr(disc_axis, "sub_scores", {}) or {}
        evidence_cited = float(disc_sub.get("evidence_cited", 0.0))
    else:
        # 7 轮不可用时从 discussion_content 直接计算
        content_lower = discussion_content.lower()
        questions_hit = sum(1 for kw in _DISCUSSION_KEYWORDS if kw in content_lower)
        questions_coverage = _clamp(min(1.0, questions_hit * 0.1))
        evidence_cited = float(
            sum(1 for tag in _DISCUSSION_CITATION_TAGS if tag in discussion_content)
        )

    evidence_citation_rate = _clamp(evidence_cited / float(len(_DISCUSSION_CITATION_TAGS)))

    sub_items = {
        "questions_coverage": questions_coverage,
        "evidence_citation_rate": evidence_citation_rate,
    }
    product = questions_coverage * evidence_citation_rate
    return _make_dim(DIM_DISCUSSION, sub_items, product)


def _compute_experiment_dim(
    seven_axis_report: Any,
    experiments: list | None,
) -> DimensionScore:
    """Experiment 维 = 机制链完整 × 实验数≥2。

    - 机制链完整：实验列表非空 → 1.0（已有实验规划）；空 → 0.0
    - 实验数≥2：min(1.0, len(experiments) / 2)（部分信用）

    experiments 为 None → 降级 0.5。
    """
    if experiments is None:
        return _make_degraded_dim(DIM_EXPERIMENT)

    # 机制链完整：实验列表非空即表示机制链有实验验证规划
    mechanism_chain_complete = 1.0 if len(experiments) > 0 else 0.0

    # 实验数≥2（部分信用）
    experiment_count_factor = _clamp(
        len(experiments) / float(_EXPERIMENT_COUNT_TARGET),
        0.0, 1.0,
    )

    sub_items = {
        "mechanism_chain_complete": mechanism_chain_complete,
        "experiment_count_factor": experiment_count_factor,
    }
    product = mechanism_chain_complete * experiment_count_factor
    return _make_dim(DIM_EXPERIMENT, sub_items, product)


# =============================================================================
# 主函数
# =============================================================================
def compute_multi_dim_confidence(
    pathway: str,
    seven_axis_report,
    parameter_report=None,
    consistency_report=None,
    critic_report=None,
    cited_pmids: list[str] | None = None,
    evidence_sources: dict | None = None,
    experiments: list | None = None,
    discussion_content: str | None = None,
) -> MultiDimConfidenceReport:
    """计算 6 维置信度。

    Feature Flag OFF 时返回 skipped 报告（行为完全回退 v3，单一数字）。
    Feature Flag ON 时计算 6 维，综合 = min(6 维) × 0.9，并应用 Critic 调节因子。

    Args:
        pathway: 通路标识（如 "EGFR"）。
        seven_axis_report: SevenAxisReport 实例（6 维复用其数据）。
        parameter_report: ParameterPriorReport（Simulation 维需要参数置信度均值）。
        consistency_report: ConsistencyReport（Mechanism 维需要 consistency 通过率）。
        critic_report: CriticReport（最终调节因子，负数降分）。
        cited_pmids: 引用的 PMID 列表（保留参数，当前未直接使用）。
        evidence_sources: 五源覆盖 {"A": n, "B": n, ...}。
        experiments: 实验列表。
        discussion_content: 讨论内容文本。

    Returns:
        MultiDimConfidenceReport。Flag OFF 时返回 skipped=True 的空报告。
    """
    # -------------------------------------------------------------------------
    # Feature Flag 守护：默认 OFF，关闭时返回 skipped 不阻塞
    # 铁律：V4_SCIENTIFIC_ALIGNMENT_ENABLED=false 时，子 Flag 永远不生效
    # -------------------------------------------------------------------------
    if not settings.is_sa_feature_enabled("MULTI_DIM_CONFIDENCE"):
        logger.debug(
            "multi-dim confidence skipped: SA_MULTI_DIM_CONFIDENCE disabled "
            "(pathway=%s)", pathway,
        )
        return MultiDimConfidenceReport(
            enabled=False,
            dimensions=[],
            overall_confidence=0.0,
            critic_adjustment=0.0,
            final_confidence=0.0,
            skipped=True,
        )

    # -------------------------------------------------------------------------
    # 逐维计算
    # -------------------------------------------------------------------------
    dimensions: list[DimensionScore] = []

    dimensions.append(_compute_mechanism_dim(seven_axis_report, consistency_report))
    dimensions.append(_compute_simulation_dim(seven_axis_report, parameter_report))
    dimensions.append(_compute_evidence_dim(evidence_sources))
    dimensions.append(_compute_biomodels_dim(seven_axis_report))
    dimensions.append(_compute_discussion_dim(seven_axis_report, discussion_content))
    dimensions.append(_compute_experiment_dim(seven_axis_report, experiments))

    # -------------------------------------------------------------------------
    # 综合 Confidence（低维拖累策略）：min(6 维) × 0.9
    # -------------------------------------------------------------------------
    min_score = min(d.score for d in dimensions) if dimensions else 0.0
    overall_confidence = _clamp(min_score * _OVERALL_DISCOUNT)

    # -------------------------------------------------------------------------
    # Critic 调节因子
    # -------------------------------------------------------------------------
    critic_adjustment = 0.0
    if critic_report is not None:
        critic_adjustment = float(getattr(critic_report, "confidence_adjustment", 0.0))
        # 确保 critic_adjustment 为非正数（Critic 只降分不加分）
        if critic_adjustment > 0.0:
            critic_adjustment = 0.0

    final_confidence = _clamp(overall_confidence + critic_adjustment, 0.0, 1.0)

    # -------------------------------------------------------------------------
    # 日志：低可信维度告警
    # -------------------------------------------------------------------------
    low_dims = [d.name for d in dimensions if d.low_confidence]
    if low_dims:
        logger.info(
            "multi-dim confidence: pathway=%s, overall=%.3f, final=%.3f, "
            "low_confidence_dims=%s",
            pathway, overall_confidence, final_confidence, low_dims,
        )

    logger.debug(
        "multi-dim confidence computed: pathway=%s, dims=%d, overall=%.3f, "
        "critic_adj=%.3f, final=%.3f",
        pathway, len(dimensions), overall_confidence,
        critic_adjustment, final_confidence,
    )

    return MultiDimConfidenceReport(
        enabled=True,
        dimensions=dimensions,
        overall_confidence=overall_confidence,
        critic_adjustment=critic_adjustment,
        final_confidence=final_confidence,
        skipped=False,
    )


# =============================================================================
# 可追溯：写入 JSON（SubTask 25.4）
# =============================================================================
def write_breakdown_json(
    report: MultiDimConfidenceReport,
    log_dir: str | None = None,
) -> str:
    """将子项得分写入 confidence_breakdown.json。

    写入路径：``<log_dir>/run_<timestamp>/confidence_breakdown.json``
    log_dir 默认 ``data/sa_logs``。

    Args:
        report: MultiDimConfidenceReport 实例。
        log_dir: 日志根目录，None 时使用默认 ``data/sa_logs``。

    Returns:
        写入的文件绝对/相对路径。
    """
    if log_dir is None:
        log_dir = _DEFAULT_LOG_DIR

    # Task 19 SubTask 19.4: 路径遍历防护
    # 校验 log_dir 不含 `..` 且不为绝对路径，防止任意目录写入。
    # spec.md 第 279-282 行：写入 logs/run_<timestamp>/ 时 SHALL 校验路径
    # 不包含 `..` / 绝对路径注入，文件名 SHALL 仅允许白名单字符集。
    from pathlib import Path as _Path
    _log_dir_path = _Path(log_dir)
    if _log_dir_path.is_absolute():
        logger.warning(
            "write_breakdown_json: log_dir 为绝对路径，降级到默认目录: %s", log_dir
        )
        log_dir = _DEFAULT_LOG_DIR
    elif ".." in _log_dir_path.parts:
        logger.warning(
            "write_breakdown_json: log_dir 含 '..' 路径遍历，降级到默认目录: %s", log_dir
        )
        log_dir = _DEFAULT_LOG_DIR

    # 构造时间戳子目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(log_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    filepath = os.path.join(run_dir, "confidence_breakdown.json")

    # 序列化报告
    data = {
        "enabled": report.enabled,
        "skipped": report.skipped,
        "overall_confidence": report.overall_confidence,
        "critic_adjustment": report.critic_adjustment,
        "final_confidence": report.final_confidence,
        "dimensions": [
            {
                "name": d.name,
                "score": d.score,
                "sub_items": d.sub_items,
                "low_confidence": d.low_confidence,
                "reason": d.reason,
            }
            for d in report.dimensions
        ],
    }

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.debug("confidence breakdown written to %s", filepath)
    except OSError as exc:
        logger.warning("写入 confidence_breakdown.json 失败: %s", exc)

    return filepath


# =============================================================================
# 报告渲染：6 维 markdown 表格（SubTask 25.5）
# =============================================================================
def format_confidence_table(report: MultiDimConfidenceReport) -> str:
    """返回 6 维置信度 markdown 表格字符串（替代单一数字）。

    Flag OFF（skipped）时返回简短的 skipped 提示表格。

    Args:
        report: MultiDimConfidenceReport 实例。

    Returns:
        markdown 表格字符串。
    """
    if report.skipped:
        return (
            "| Dimension | Score | Status |\n"
            "|---|---|---|\n"
            "| (skipped) | - | SA_MULTI_DIM_CONFIDENCE disabled |"
        )

    header = (
        "| Dimension | Score | Low Confidence | Sub-items | Reason |\n"
        "|---|---|---|---|---|"
    )
    rows: list[str] = []

    for d in report.dimensions:
        status = "YES" if d.low_confidence else "no"
        if d.sub_items:
            sub_str = ", ".join(f"{k}={v:.2f}" for k, v in d.sub_items.items())
        else:
            sub_str = "(degraded)"
        reason = d.reason if d.reason else "-"
        rows.append(
            f"| {d.name} | {d.score:.3f} | {status} | {sub_str} | {reason} |"
        )

    # 综合行
    rows.append(
        f"| **Overall** | **{report.overall_confidence:.3f}** | - | "
        f"min(6 dims) x 0.9 | - |"
    )

    # Critic 调节行（仅当有调节时显示）
    if report.critic_adjustment != 0.0:
        rows.append(
            f"| **Final** | **{report.final_confidence:.3f}** | - | "
            f"overall + critic ({report.critic_adjustment:+.3f}) | - |"
        )
    else:
        rows.append(
            f"| **Final** | **{report.final_confidence:.3f}** | - | "
            f"= overall (no critic) | - |"
        )

    return header + "\n" + "\n".join(rows)


__all__ = [
    "DimensionScore",
    "MultiDimConfidenceReport",
    "compute_multi_dim_confidence",
    "write_breakdown_json",
    "format_confidence_table",
]
