# BioDynamics Agent v4 - Scientific Alignment Loop: Scientific Critic Agent (Task 26)
#
# Scientific Critic Agent：在 Pipeline 末尾独立审查 Report 的科学一致性。
#
# 设计背景：
#   当前 Pipeline 为 Planner → Mechanism → ODE → Report，Report 生成后无独立审查。
#   Task 26 在 Pipeline 末尾插入 Scientific Critic Agent，独立审查报告的科学一致性。
#   Critic 不共享 Report Generator 的偏见，专门检查：
#     - Mechanism 是否符合 Canonical
#     - Evidence 是否真正支持
#     - BioModels 是否一致
#     - 有无自相矛盾（复用 Consistency Checker）
#     - 有无错误实验
#     - 有无经典论文未引用
#
# 独立性原则：
#   Critic 不读取已生成 Report 的内容做"自我审查"，而是从原始数据
#   （extracted_nodes / simulation_metrics / cited_pmids 等）独立校验，
#   避免与 Report Generator 共享偏见。
#
# 重试机制：
#   任一 fail → retry_required=True，由调用方将 concerns 注入 Prompt 触发 Report 重生成。
#   最大重试 max_retries=2 次，超限标记 unresolved=True 并额外降 Confidence。
#
# Feature Flag 守护：
#   SA_SCIENTIFIC_CRITIC 默认 OFF。关闭时返回 skipped 报告，不阻塞主流程。
#   铁律：V4_SCIENTIFIC_ALIGNMENT_ENABLED=false 时，子 Flag 永远不生效
#         （由 settings.is_sa_feature_enabled 强制校验）。
#
# 安全设计：
#   - 不引入新依赖（仅标准库 + 已完成组件 + app.config.settings）
#   - 复用 Consistency Checker / Mechanism Checker / Canonical Loader，不重新实现
#   - 每类审查独立 try-except，单类异常不阻塞其他类
#
# 依赖：
#   - app.config.settings（Feature Flag 守护）
#   - app.scientific_alignment.canonical_loader（Task 22）—— load_canonical
#   - app.scientific_alignment.consistency_checker（Task 24）—— check_consistency
#   - app.scientific_alignment.mechanism_checker（Task 8）—— check_mechanism_alignment
#
# 核心导出：
#   from app.scientific_alignment.scientific_critic import (
#       CriticFinding, CriticReport, run_scientific_critic,
#   )

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from app.config import settings
from app.scientific_alignment.canonical_loader import (
    CanonicalNotFoundError,
    load_canonical,
)
from app.scientific_alignment.consistency_checker import check_consistency
from app.scientific_alignment.mechanism_checker import check_mechanism_alignment

logger = logging.getLogger(__name__)


# =============================================================================
# 常量
# =============================================================================

# 最大重试次数：超过此值后标记 unresolved，不再触发重生成
_MAX_RETRIES: int = 2

# Confidence 调整常量
_PENALTY_PER_FAIL: float = 0.1        # 每个 fail 的 Confidence 扣分
_PENALTY_PER_CONCERN: float = 0.02    # 每个 concern 的 Confidence 扣分
_PENALTY_CAP: float = 0.5             # Confidence 扣分总上限（避免归零过多）
_PENALTY_UNRESOLVED: float = 0.2      # unresolved 时额外扣分

# Evidence 审查阈值
_EVIDENCE_MIN_TOTAL: int = 5          # 最少总文献数
_EVIDENCE_MIN_REVIEWS: int = 2        # 最少 Review 数
# 强证据类型阈值：EvidenceType >= 3 视为 [A][B][C] 支撑
#   REVIEW=5([A]) / MECHANISM_PAPER=4([A]) / BIOMODELS_SOURCE=3([B])
#   RECENT_APPLICATION=2 / CASE_REPORT=1 视为非强证据
_EVIDENCE_STRONG_TYPE_THRESHOLD: int = 3
_EVIDENCE_REVIEW_TYPE: int = 5        # EvidenceType.REVIEW 的数值

# Mechanism 覆盖率阈值
_MECHANISM_COVERAGE_FAIL: float = 0.8      # coverage < 此值 → fail
_MECHANISM_COVERAGE_CONCERN: float = 0.95  # coverage < 此值 → concern

# Experiments 审查阈值
_EXPERIMENTS_MIN_COUNT: int = 2       # 最少实验数

# References 经典论文覆盖率阈值
_REFERENCES_COVERAGE_FAIL: float = 0.5     # coverage < 此值 → fail，否则 concern

# 审查类别常量（对应 6 类 SubTask 26.2）
_CATEGORY_MECHANISM: str = "mechanism"
_CATEGORY_EVIDENCE: str = "evidence"
_CATEGORY_BIOMODELS: str = "biomodels"
_CATEGORY_CONSISTENCY: str = "consistency"
_CATEGORY_EXPERIMENTS: str = "experiments"
_CATEGORY_REFERENCES: str = "references"


# =============================================================================
# 数据类
# =============================================================================
@dataclass
class CriticFinding:
    """单条审稿发现。

    Attributes:
        category: 审查类别（mechanism / evidence / biomodels /
            consistency / experiments / references）。
        severity: 严重等级（``"fail"`` / ``"concern"`` / ``"pass"``）。
        finding: 人类可读的发现描述。
        evidence: 证据（如 violated rule label / missing PMID / missing node）。
        suggestion: 修复建议。
    """

    category: str
    severity: str
    finding: str
    evidence: str
    suggestion: str


@dataclass
class CriticReport:
    """Scientific Critic 审稿报告。

    Attributes:
        pathway: 被审查的通路标识（如 ``"egfr"``）。
        findings: 审稿发现列表（CriticFinding），每类审查一条。
        overall_status: 整体状态：
            - ``"passed"``：全部通过
            - ``"failed"``：有 fail 级发现
            - ``"concerns"``：有 concern 但无 fail
            - ``"skipped"``：Feature Flag 关闭
        retry_required: 是否需要重生成 Report（有 fail 且未超重试上限时 True）。
        retry_count: 当前重试次数（0=首次审查）。
        max_retries: 最大重试次数（默认 2）。
        confidence_adjustment: Confidence 调整值（负数=降级）。
        unresolved: 重试超限仍未解决时 True。
        summary: 人类可读总结。
        skipped: 是否跳过（Feature Flag 关闭时 True）。
        skip_reason: 跳过原因。
    """

    pathway: str
    findings: list[CriticFinding]
    overall_status: str
    retry_required: bool
    retry_count: int = 0
    max_retries: int = _MAX_RETRIES
    confidence_adjustment: float = 0.0
    unresolved: bool = False
    summary: str = ""
    skipped: bool = False
    skip_reason: str = ""


# =============================================================================
# 辅助函数
# =============================================================================
def _get_evidence_type_value(doc: Any) -> int | None:
    """安全获取 EvidenceDoc 的 evidence_type 数值。

    支持两种输入形式：
      - EvidenceDoc 对象（有 ``evidence_type`` 属性，IntEnum）
      - dict（含 ``"evidence_type"`` 键）

    Args:
        doc: EvidenceDoc 对象或 dict。

    Returns:
        evidence_type 的数值（1-5），无法获取时返回 None。
    """
    # 优先从对象属性获取
    etype = getattr(doc, "evidence_type", None)
    # 再尝试从 dict 获取
    if etype is None and isinstance(doc, dict):
        etype = doc.get("evidence_type")
    # IntEnum → int
    if etype is not None:
        try:
            return int(etype)
        except (TypeError, ValueError):
            return None
    return None


def _count_reviews(evidence_docs: list[Any]) -> int:
    """统计 Review 文献数。

    从 evidence_docs 中统计 evidence_type == REVIEW(5) 的文档数。

    Args:
        evidence_docs: Evidence 文档列表。

    Returns:
        Review 文献数。
    """
    if not evidence_docs:
        return 0
    review_count = 0
    for doc in evidence_docs:
        etype_val = _get_evidence_type_value(doc)
        if etype_val is not None and etype_val == _EVIDENCE_REVIEW_TYPE:
            review_count += 1
    return review_count


# =============================================================================
# 6 类审查函数（私有，每类独立 try-except 确保互不阻塞）
# =============================================================================

def _audit_mechanism(
    pathway: str,
    extracted_nodes: list[str] | None,
) -> CriticFinding:
    """Mechanism 审查：机制节点覆盖检查（SubTask 26.2.1）。

    调用 ``check_mechanism_alignment`` 校验 Agent 提取的机制节点是否覆盖
    Canonical required_nodes。

    判定规则（按优先级）：
      1. ``extracted_nodes`` 为 None → concern（无法审查）
      2. 检查被 Feature Flag 跳过（warnings 含 "disabled"）→ concern
      3. 缺关键节点（``missing_critical_nodes`` 非空）→ fail
      4. coverage < 0.8 → fail
      5. coverage 0.8-0.95 → concern
      6. coverage >= 0.95 → pass

    Args:
        pathway: 通路标识。
        extracted_nodes: Agent 提取的机制节点列表，None 时无法审查。

    Returns:
        CriticFinding。
    """
    # extracted_nodes 为 None 时无法审查（降级为 concern）
    if extracted_nodes is None:
        return CriticFinding(
            category=_CATEGORY_MECHANISM,
            severity="concern",
            finding="未提供机制节点，无法审查机制覆盖",
            evidence="no_extracted_nodes",
            suggestion="提供 Agent 提取的机制节点列表以启用机制审查",
        )

    try:
        result = check_mechanism_alignment(
            pathway=pathway,
            extracted_nodes=extracted_nodes,
            original_confidence=1.0,
        )
    except Exception as exc:  # noqa: BLE001 —— Canonical 加载失败等异常不阻塞
        logger.warning("Mechanism 审查异常: %s", exc)
        return CriticFinding(
            category=_CATEGORY_MECHANISM,
            severity="concern",
            finding=f"机制检查异常: {exc}",
            evidence="mechanism_check_error",
            suggestion="检查 Canonical 文件是否存在且格式正确",
        )

    coverage = result.coverage
    missing_nodes = result.missing_nodes
    missing_critical = result.missing_critical_nodes

    # 检查是否被 Feature Flag 跳过（SA_MECHANISM_GRAPH 关闭时 warnings 含 "disabled"）
    if any("disabled" in w.lower() for w in result.warnings):
        return CriticFinding(
            category=_CATEGORY_MECHANISM,
            severity="concern",
            finding="机制检查被 Feature Flag 跳过（SA_MECHANISM_GRAPH 关闭）",
            evidence="mechanism_check_skipped",
            suggestion="启用 SA_MECHANISM_GRAPH 以激活机制覆盖检查",
        )

    # 1. 缺关键节点（负反馈节点 / 节点链首尾）→ fail
    if missing_critical:
        return CriticFinding(
            category=_CATEGORY_MECHANISM,
            severity="fail",
            finding=f"缺失关键节点（负反馈/节点链首尾）: {missing_critical}",
            evidence=f"missing_critical_nodes={missing_critical}",
            suggestion="补充缺失的关键节点至机制图，确保负反馈与节点链完整",
        )

    # 2. coverage < 0.8 → fail
    if coverage < _MECHANISM_COVERAGE_FAIL:
        return CriticFinding(
            category=_CATEGORY_MECHANISM,
            severity="fail",
            finding=(
                f"机制覆盖率 {coverage:.2f} 低于阈值 "
                f"{_MECHANISM_COVERAGE_FAIL:.2f}，缺失节点: {missing_nodes}"
            ),
            evidence=f"coverage={coverage:.2f}, missing={missing_nodes}",
            suggestion="补充缺失的 Canonical 必填节点至机制图",
        )

    # 3. coverage 0.8-0.95 → concern
    if coverage < _MECHANISM_COVERAGE_CONCERN:
        return CriticFinding(
            category=_CATEGORY_MECHANISM,
            severity="concern",
            finding=(
                f"机制覆盖率 {coverage:.2f} 处于警告区间"
                f"（{_MECHANISM_COVERAGE_FAIL:.2f}-{_MECHANISM_COVERAGE_CONCERN:.2f}）"
                f"，缺失节点: {missing_nodes}"
            ),
            evidence=f"coverage={coverage:.2f}, missing={missing_nodes}",
            suggestion="考虑补充缺失节点以提高机制覆盖完整性",
        )

    # 4. coverage >= 0.95 → pass
    return CriticFinding(
        category=_CATEGORY_MECHANISM,
        severity="pass",
        finding=f"机制覆盖率 {coverage:.2f} 达标",
        evidence=f"coverage={coverage:.2f}",
        suggestion="",
    )


def _audit_evidence(
    cited_pmids: list[str] | None,
    evidence_docs: list[Any] | None,
) -> CriticFinding:
    """Evidence 审查：文献证据充分性检查（SubTask 26.2.2）。

    判定规则（按优先级）：
      1. 总文献数 < 5 → fail（``only N citations, need >=5``）
      2. 全部为 [D]Inference 无 [A][B][C] 支撑 → fail（``evidence_undergrounded``）
      3. Review 数 < 2 → concern
      4. 否则 → pass

    五源证据标签映射：
      [A]=PubMed / [B]=BioModels / [C]=Simulation / [D]=Inference / [E]=Hypothesis

    "全部为 [D]Inference 无 [A][B][C] 支撑" 判定：
      - 无 ``cited_pmids``（无 [A] PubMed 文献支持）
      - 且 ``evidence_docs`` 中无强证据类型文档
        （REVIEW=5 / MECHANISM_PAPER=4 / BIOMODELS_SOURCE=3）

    Args:
        cited_pmids: 报告引用的 PMID 列表。
        evidence_docs: Evidence 文档列表（EvidenceDoc 或 dict），可为空。

    Returns:
        CriticFinding。
    """
    # 规范化输入：None → 空列表
    cited_pmids = cited_pmids or []
    evidence_docs = evidence_docs or []

    # 统计总文献数：优先用 cited_pmids，其次用 evidence_docs
    total_count = len(cited_pmids) if cited_pmids else len(evidence_docs)

    # 1. 总文献数 < 5 → fail
    if total_count < _EVIDENCE_MIN_TOTAL:
        return CriticFinding(
            category=_CATEGORY_EVIDENCE,
            severity="fail",
            finding=f"总文献数 {total_count} 低于阈值 {_EVIDENCE_MIN_TOTAL}",
            evidence=f"only {total_count} citations, need >=5",
            suggestion=f"补充文献引用至至少 {_EVIDENCE_MIN_TOTAL} 篇",
        )

    # 2. 全部为 [D]Inference 无 [A][B][C] 支撑 → fail
    #    判定：无 cited_pmids（[A] PubMed 支持）且 evidence_docs 中无强证据类型
    has_strong_evidence = bool(cited_pmids)
    if not has_strong_evidence and evidence_docs:
        for doc in evidence_docs:
            doc_type = _get_evidence_type_value(doc)
            if doc_type is not None and doc_type >= _EVIDENCE_STRONG_TYPE_THRESHOLD:
                has_strong_evidence = True
                break

    if not has_strong_evidence:
        return CriticFinding(
            category=_CATEGORY_EVIDENCE,
            severity="fail",
            finding="全部证据为 [D]Inference，无 [A][B][C] 文献支撑",
            evidence="evidence_undergrounded",
            suggestion="补充 PubMed 文献引用或 BioModels 来源论文以提供实证支撑",
        )

    # 3. 统计 Review 数
    review_count = _count_reviews(evidence_docs)

    if review_count < _EVIDENCE_MIN_REVIEWS:
        return CriticFinding(
            category=_CATEGORY_EVIDENCE,
            severity="concern",
            finding=(
                f"Review 文献数 {review_count} 低于阈值 "
                f"{_EVIDENCE_MIN_REVIEWS}"
            ),
            evidence=f"review_count={review_count}",
            suggestion=f"补充至少 {_EVIDENCE_MIN_REVIEWS} 篇权威综述文献",
        )

    # 4. 否则 → pass
    return CriticFinding(
        category=_CATEGORY_EVIDENCE,
        severity="pass",
        finding=f"文献证据充分（总 {total_count} 篇，Review {review_count} 篇）",
        evidence=f"total={total_count}, reviews={review_count}",
        suggestion="",
    )


def _audit_biomodels(biomodels_report: Any | None) -> CriticFinding:
    """BioModels 审查：BioModels Oracle 对比报告检查（SubTask 26.2.3）。

    判定规则：
      1. ``biomodels_report`` 为 None → concern（``no_biomodels_comparison``）
      2. status="failed" → fail
      3. status="degraded" → concern
      4. status="passed" → pass
      5. 其他未知 status → concern

    Args:
        biomodels_report: BioModelsOracleReport 实例，None 时降级。

    Returns:
        CriticFinding。
    """
    # 1. 报告为 None → concern
    if biomodels_report is None:
        return CriticFinding(
            category=_CATEGORY_BIOMODELS,
            severity="concern",
            finding="未提供 BioModels Oracle 对比报告",
            evidence="no_biomodels_comparison",
            suggestion="运行 BioModels Oracle 并提供对比报告以启用 BioModels 审查",
        )

    # 安全读取报告字段（biomodels_report 类型为 Any，用 getattr 防御）
    report_status = str(getattr(biomodels_report, "status", "") or "")
    overall_distance = getattr(biomodels_report, "overall_distance", None)

    # 2. status="failed" → fail
    if report_status == "failed":
        return CriticFinding(
            category=_CATEGORY_BIOMODELS,
            severity="fail",
            finding=(
                f"BioModels Oracle 对比失败（status=failed"
                f", overall_distance={overall_distance}）"
            ),
            evidence=f"overall_distance={overall_distance}",
            suggestion="检查仿真结果与 BioModels 参考模型的距离是否在可接受范围内",
        )

    # 3. status="degraded" → concern
    if report_status == "degraded":
        return CriticFinding(
            category=_CATEGORY_BIOMODELS,
            severity="concern",
            finding="BioModels Oracle 对比降级（status=degraded）",
            evidence="status=degraded",
            suggestion="检查 BioModels Oracle 是否部分失败（如 Track B 降级）",
        )

    # 4. status="passed" → pass
    if report_status == "passed":
        return CriticFinding(
            category=_CATEGORY_BIOMODELS,
            severity="pass",
            finding=f"BioModels Oracle 对比通过（overall_distance={overall_distance}）",
            evidence=f"overall_distance={overall_distance}",
            suggestion="",
        )

    # 5. 其他未知 status → concern
    return CriticFinding(
        category=_CATEGORY_BIOMODELS,
        severity="concern",
        finding=f"BioModels Oracle 报告状态未知: {report_status!r}",
        evidence=f"unknown_status={report_status}",
        suggestion="检查 BioModels Oracle 报告的 status 字段",
    )


def _audit_consistency(
    pathway: str,
    simulation_metrics: dict | None,
) -> CriticFinding:
    """Consistency 审查：仿真结果机制级一致性校验（SubTask 26.2.4）。

    复用 ``check_consistency`` 对仿真 metrics 做机制级逻辑校验
    （如 EGFR Peak 不能晚于 ERK Peak）。

    判定规则：
      1. ``simulation_metrics`` 为 None → concern（无法审查）
      2. 有 violations → fail
      3. 无 violations 但 rules_evaluated < rules_checked → concern（部分规则未评估）
      4. 全部通过 → pass

    Args:
        pathway: 通路标识。
        simulation_metrics: 仿真指标 dict，None 时无法审查。

    Returns:
        CriticFinding。
    """
    # 1. simulation_metrics 为 None → concern
    if simulation_metrics is None:
        return CriticFinding(
            category=_CATEGORY_CONSISTENCY,
            severity="concern",
            finding="未提供仿真指标，无法审查一致性",
            evidence="no_simulation_metrics",
            suggestion="提供仿真指标 dict 以启用一致性审查",
        )

    try:
        report = check_consistency(pathway, simulation_metrics)
    except Exception as exc:  # noqa: BLE001 —— 一致性检查异常不阻塞
        logger.warning("Consistency 审查异常: %s", exc)
        return CriticFinding(
            category=_CATEGORY_CONSISTENCY,
            severity="concern",
            finding=f"一致性检查异常: {exc}",
            evidence="consistency_check_error",
            suggestion="检查 Canonical 文件与 consistency_rules 格式",
        )

    # 2. 有 violations → fail
    if report.violations:
        violation_labels = [
            v.violation_label for v in report.violations if v.violation_label
        ]
        return CriticFinding(
            category=_CATEGORY_CONSISTENCY,
            severity="fail",
            finding=(
                f"发现 {len(report.violations)} 条一致性违规: "
                f"{[v.violation_label for v in report.violations]}"
            ),
            evidence=f"violation_labels={violation_labels}",
            suggestion="修正仿真结果以满足 Canonical 自洽规则",
        )

    # 3. 无 violations 但 rules_evaluated < rules_checked → concern（部分规则未评估）
    if report.rules_evaluated < report.rules_checked:
        return CriticFinding(
            category=_CATEGORY_CONSISTENCY,
            severity="concern",
            finding=(
                f"部分规则未能评估（已评估 {report.rules_evaluated}"
                f"/{report.rules_checked}）"
            ),
            evidence=(
                f"rules_evaluated={report.rules_evaluated}, "
                f"rules_checked={report.rules_checked}"
            ),
            suggestion="检查 simulation_metrics 是否包含所有规则所需的指标",
        )

    # 4. 全部通过 → pass
    return CriticFinding(
        category=_CATEGORY_CONSISTENCY,
        severity="pass",
        finding=(
            f"一致性检查通过（{report.rules_evaluated}/{report.rules_checked} 规则）"
        ),
        evidence=(
            f"rules_evaluated={report.rules_evaluated}, "
            f"rules_checked={report.rules_checked}"
        ),
        suggestion="",
    )


def _audit_experiments(experiments: list[dict] | None) -> CriticFinding:
    """Experiments 审查：实验规划合理性检查（SubTask 26.2.5）。

    判定规则：
      1. 为空（None 或 []）→ concern（``no_experiments_proposed``）
      2. 实验数 < 2 → concern
      3. 实验未说明验证哪个机制节点（无 ``"mechanism_node"`` 键）→ fail
         （``experiment_unjustified``）
      4. 否则 → pass

    Args:
        experiments: 实验列表（list[dict]），每项可含 ``"mechanism_node"`` 键。

    Returns:
        CriticFinding。
    """
    # 1. 为空 → concern
    if not experiments:
        return CriticFinding(
            category=_CATEGORY_EXPERIMENTS,
            severity="concern",
            finding="未提出任何验证实验",
            evidence="no_experiments_proposed",
            suggestion="提出至少 2 个实验以验证机制节点的动力学预测",
        )

    # 2. 实验数 < 2 → concern
    if len(experiments) < _EXPERIMENTS_MIN_COUNT:
        return CriticFinding(
            category=_CATEGORY_EXPERIMENTS,
            severity="concern",
            finding=(
                f"实验数 {len(experiments)} 低于阈值 "
                f"{_EXPERIMENTS_MIN_COUNT}"
            ),
            evidence=f"experiment_count={len(experiments)}",
            suggestion=f"补充实验至至少 {_EXPERIMENTS_MIN_COUNT} 个",
        )

    # 3. 检查是否有实验未说明验证哪个机制节点（无 "mechanism_node" 键）
    unjustified_indices = [
        i for i, exp in enumerate(experiments)
        if not isinstance(exp, dict) or "mechanism_node" not in exp
    ]
    if unjustified_indices:
        return CriticFinding(
            category=_CATEGORY_EXPERIMENTS,
            severity="fail",
            finding=(
                f"以下实验未说明验证的机制节点（缺少 mechanism_node 键）: "
                f"索引 {unjustified_indices}"
            ),
            evidence="experiment_unjustified",
            suggestion="为每个实验明确指定其验证的机制节点（mechanism_node 字段）",
        )

    # 4. 否则 → pass
    return CriticFinding(
        category=_CATEGORY_EXPERIMENTS,
        severity="pass",
        finding=f"实验规划合理（{len(experiments)} 个实验，均关联机制节点）",
        evidence=f"experiment_count={len(experiments)}",
        suggestion="",
    )


def _audit_references(
    pathway: str,
    cited_pmids: list[str] | None,
) -> CriticFinding:
    """References 审查：经典论文覆盖检查（SubTask 26.2.6）。

    加载 Canonical 的 ``canonical_reviews``，检查 ``cited_pmids`` 是否覆盖。

    判定规则：
      1. Canonical 加载失败 → concern
      2. ``canonical_reviews`` 为空 → pass（无必引经典论文）
      3. 覆盖率 < 50% → fail（``missing canonical reviews: {missing}``）
      4. 覆盖率 50%-99% → concern
      5. 覆盖率 100% → pass

    Args:
        pathway: 通路标识。
        cited_pmids: 报告引用的 PMID 列表。

    Returns:
        CriticFinding。
    """
    # 加载 Canonical Reference
    try:
        canonical = load_canonical(pathway)
    except CanonicalNotFoundError as exc:
        logger.warning("References 审查: Canonical 文件不存在: %s", exc)
        return CriticFinding(
            category=_CATEGORY_REFERENCES,
            severity="concern",
            finding=f"Canonical 文件不存在，无法审查经典论文覆盖: {exc}",
            evidence="canonical_not_found",
            suggestion="创建通路对应的 Canonical Reference 文件",
        )
    except Exception as exc:  # noqa: BLE001 —— Canonical 解析异常不阻塞
        logger.warning("References 审查: Canonical 加载失败: %s", exc)
        return CriticFinding(
            category=_CATEGORY_REFERENCES,
            severity="concern",
            finding=f"Canonical 加载失败: {exc}",
            evidence="canonical_load_error",
            suggestion="检查 Canonical 文件格式与必填字段",
        )

    canonical_reviews = list(canonical.canonical_reviews)

    # canonical_reviews 为空 → pass（Canonical 未定义必引经典论文）
    if not canonical_reviews:
        return CriticFinding(
            category=_CATEGORY_REFERENCES,
            severity="pass",
            finding="Canonical 未定义必引经典论文",
            evidence="no_canonical_reviews",
            suggestion="",
        )

    # 计算覆盖率
    cited_set = set(cited_pmids or [])
    covered = [pmid for pmid in canonical_reviews if pmid in cited_set]
    missing = [pmid for pmid in canonical_reviews if pmid not in cited_set]
    coverage = len(covered) / len(canonical_reviews)

    # 3. 覆盖率 < 50% → fail
    if coverage < _REFERENCES_COVERAGE_FAIL:
        return CriticFinding(
            category=_CATEGORY_REFERENCES,
            severity="fail",
            finding=(
                f"经典论文覆盖率 {coverage:.2f} 低于阈值 "
                f"{_REFERENCES_COVERAGE_FAIL:.2f}，缺失: {missing}"
            ),
            evidence=f"missing canonical reviews: {missing}",
            suggestion=f"补充引用以下经典论文: {missing}",
        )

    # 4. 覆盖率 50%-99% → concern
    if coverage < 1.0:
        return CriticFinding(
            category=_CATEGORY_REFERENCES,
            severity="concern",
            finding=f"经典论文覆盖率 {coverage:.2f}，缺失: {missing}",
            evidence=f"missing={missing}, coverage={coverage:.2f}",
            suggestion=f"考虑补充引用以下经典论文: {missing}",
        )

    # 5. 覆盖率 100% → pass
    return CriticFinding(
        category=_CATEGORY_REFERENCES,
        severity="pass",
        finding=f"经典论文全覆盖（{len(covered)}/{len(canonical_reviews)}）",
        evidence=f"coverage={coverage:.2f}",
        suggestion="",
    )


# =============================================================================
# Summary 生成
# =============================================================================
def _build_summary(
    findings: list[CriticFinding],
    confidence_adjustment: float,
    retry_required: bool,
    retry_count: int,
    max_retries: int,
    unresolved: bool,
) -> str:
    """生成人类可读的审稿总结。

    格式示例::

        Scientific Critic reviewed 6 categories: 2 fail, 1 concern, 3 pass.
        Failed: mechanism (missing DUSP), consistency (egfr_peak_after_erk_peak).
        Confidence adjustment: -0.20.
        Retry required: True (retry 1/2).

    Args:
        findings: 审稿发现列表。
        confidence_adjustment: Confidence 调整值（负数=降级）。
        retry_required: 是否需要重生成。
        retry_count: 当前重试次数。
        max_retries: 最大重试次数。
        unresolved: 是否超限未解决。

    Returns:
        总结字符串。
    """
    fail_count = sum(1 for f in findings if f.severity == "fail")
    concern_count = sum(1 for f in findings if f.severity == "concern")
    pass_count = sum(1 for f in findings if f.severity == "pass")
    total = len(findings)

    lines: list[str] = [
        f"Scientific Critic reviewed {total} categories: "
        f"{fail_count} fail, {concern_count} concern, {pass_count} pass."
    ]

    # 列出 fail 项的类别与证据
    failed_findings = [f for f in findings if f.severity == "fail"]
    if failed_findings:
        fail_descs = [f"{f.category} ({f.evidence})" for f in failed_findings]
        lines.append(f"Failed: {', '.join(fail_descs)}.")

    # Confidence 调整
    lines.append(f"Confidence adjustment: {confidence_adjustment:.2f}.")

    # 重试状态
    if unresolved:
        lines.append(
            f"Retry exhausted ({retry_count}/{max_retries}), marked unresolved."
        )
    elif retry_required:
        lines.append(
            f"Retry required: True (retry {retry_count + 1}/{max_retries})."
        )
    else:
        lines.append("Retry required: False.")

    return " ".join(lines)


# =============================================================================
# 主函数
# =============================================================================
def run_scientific_critic(
    pathway: str,
    extracted_nodes: list[str] | None = None,
    simulation_metrics: dict | None = None,
    biomodels_report: Any | None = None,
    cited_pmids: list[str] | None = None,
    evidence_docs: list[Any] | None = None,
    experiments: list[dict] | None = None,
    retry_count: int = 0,
) -> CriticReport:
    """运行 Scientific Critic 独立审查。

    在 Pipeline 末尾独立审查 Report 的科学一致性。Critic 不依赖 Report Generator
    输出，而是从原始数据（extracted_nodes / simulation_metrics / cited_pmids 等）
    独立校验，避免与 Report Generator 共享偏见。

    审查 6 个类别（SubTask 26.2）：
      1. Mechanism：机制节点覆盖（复用 mechanism_checker）
      2. Evidence：文献证据充分性
      3. BioModels：BioModels Oracle 对比
      4. Consistency：仿真一致性（复用 consistency_checker）
      5. Experiments：实验规划合理性
      6. References：经典论文覆盖（复用 canonical_loader）

    任一 fail → ``retry_required=True``（由调用方将 concerns 注入 Prompt
    触发 Report 重生成）。最大重试 ``max_retries`` 次，超限标记
    ``unresolved=True`` 并额外降 Confidence。

    overall_status 判定：
      - 任一 fail → ``"failed"``, ``retry_required=True``
      - 无 fail 但有 concern → ``"concerns"``, ``retry_required=False``
      - 全部 pass → ``"passed"``, ``retry_required=False``
      - retry_count >= max_retries 且仍有 fail → ``"failed"``, ``unresolved=True``

    confidence_adjustment 计算：
      - 每个 fail: -0.1
      - 每个 concern: -0.02
      - 总调整下限 -0.5
      - unresolved=True 时额外 -0.2

    Args:
        pathway: 通路标识（如 ``"egfr"``）。
        extracted_nodes: Agent 提取的机制节点列表。
        simulation_metrics: 仿真指标 dict。
        biomodels_report: BioModels Oracle 报告（BioModelsOracleReport），
            可为 None。
        cited_pmids: 报告引用的 PMID 列表。
        evidence_docs: Evidence 文档列表（EvidenceDoc 或 dict），可为空。
        experiments: 实验列表（list[dict]），每项可含 ``"mechanism_node"`` 键。
        retry_count: 当前重试次数（0=首次审查，>=max_retries 标记 unresolved）。

    Returns:
        CriticReport。Feature Flag 关闭时返回 ``skipped=True`` 的空报告。
    """
    # -------------------------------------------------------------------------
    # Feature Flag 守护：默认 OFF，关闭时返回 skipped 不阻塞
    # 铁律：V4_SCIENTIFIC_ALIGNMENT_ENABLED=false 时，子 Flag 永远不生效
    # -------------------------------------------------------------------------
    if not settings.is_sa_feature_enabled("SCIENTIFIC_CRITIC"):
        return CriticReport(
            pathway=pathway,
            findings=[],
            overall_status="skipped",
            retry_required=False,
            skipped=True,
            skip_reason="SA_SCIENTIFIC_CRITIC disabled",
        )

    # -------------------------------------------------------------------------
    # 逐类审查（每类独立 try-except，单类异常不阻塞其他类）
    # -------------------------------------------------------------------------
    findings: list[CriticFinding] = []

    # 1. Mechanism 审查
    try:
        findings.append(_audit_mechanism(pathway, extracted_nodes))
    except Exception as exc:  # noqa: BLE001
        logger.error("Mechanism 审查意外异常: %s", exc)
        findings.append(CriticFinding(
            category=_CATEGORY_MECHANISM,
            severity="concern",
            finding=f"Mechanism 审查意外异常: {exc}",
            evidence="audit_error",
            suggestion="检查 Mechanism Checker 与 Canonical 配置",
        ))

    # 2. Evidence 审查
    try:
        findings.append(_audit_evidence(cited_pmids, evidence_docs))
    except Exception as exc:  # noqa: BLE001
        logger.error("Evidence 审查意外异常: %s", exc)
        findings.append(CriticFinding(
            category=_CATEGORY_EVIDENCE,
            severity="concern",
            finding=f"Evidence 审查意外异常: {exc}",
            evidence="audit_error",
            suggestion="检查 evidence_docs 格式",
        ))

    # 3. BioModels 审查
    try:
        findings.append(_audit_biomodels(biomodels_report))
    except Exception as exc:  # noqa: BLE001
        logger.error("BioModels 审查意外异常: %s", exc)
        findings.append(CriticFinding(
            category=_CATEGORY_BIOMODELS,
            severity="concern",
            finding=f"BioModels 审查意外异常: {exc}",
            evidence="audit_error",
            suggestion="检查 biomodels_report 格式",
        ))

    # 4. Consistency 审查
    try:
        findings.append(_audit_consistency(pathway, simulation_metrics))
    except Exception as exc:  # noqa: BLE001
        logger.error("Consistency 审查意外异常: %s", exc)
        findings.append(CriticFinding(
            category=_CATEGORY_CONSISTENCY,
            severity="concern",
            finding=f"Consistency 审查意外异常: {exc}",
            evidence="audit_error",
            suggestion="检查 Consistency Checker 与 Canonical 配置",
        ))

    # 5. Experiments 审查
    try:
        findings.append(_audit_experiments(experiments))
    except Exception as exc:  # noqa: BLE001
        logger.error("Experiments 审查意外异常: %s", exc)
        findings.append(CriticFinding(
            category=_CATEGORY_EXPERIMENTS,
            severity="concern",
            finding=f"Experiments 审查意外异常: {exc}",
            evidence="audit_error",
            suggestion="检查 experiments 格式",
        ))

    # 6. References 审查
    try:
        findings.append(_audit_references(pathway, cited_pmids))
    except Exception as exc:  # noqa: BLE001
        logger.error("References 审查意外异常: %s", exc)
        findings.append(CriticFinding(
            category=_CATEGORY_REFERENCES,
            severity="concern",
            finding=f"References 审查意外异常: {exc}",
            evidence="audit_error",
            suggestion="检查 Canonical Loader 配置",
        ))

    # -------------------------------------------------------------------------
    # 统计 fail / concern / pass
    # -------------------------------------------------------------------------
    fail_count = sum(1 for f in findings if f.severity == "fail")
    concern_count = sum(1 for f in findings if f.severity == "concern")
    has_fail = fail_count > 0

    # -------------------------------------------------------------------------
    # unresolved 判定：有 fail 且重试次数已达上限
    # -------------------------------------------------------------------------
    unresolved = has_fail and retry_count >= _MAX_RETRIES

    # -------------------------------------------------------------------------
    # overall_status 判定
    #   - 任一 fail → failed, retry_required=True
    #   - 无 fail 但有 concern → concerns, retry_required=False
    #   - 全部 pass → passed, retry_required=False
    #   - retry_count >= max_retries 且仍有 fail → failed, unresolved=True
    # -------------------------------------------------------------------------
    if unresolved:
        # 超限仍有 fail → failed + unresolved，不再重试
        overall_status = "failed"
        retry_required = False
    elif has_fail:
        # 有 fail 且未超限 → failed，需要重试
        overall_status = "failed"
        retry_required = True
    elif concern_count > 0:
        # 无 fail 但有 concern → concerns，不强制重生成
        overall_status = "concerns"
        retry_required = False
    else:
        # 全部 pass → passed
        overall_status = "passed"
        retry_required = False

    # -------------------------------------------------------------------------
    # confidence_adjustment 计算
    #   - 每个 fail: -0.1
    #   - 每个 concern: -0.02
    #   - 总调整下限 -0.5
    #   - unresolved=True 时额外 -0.2
    # -------------------------------------------------------------------------
    confidence_adjustment = -(
        fail_count * _PENALTY_PER_FAIL
        + concern_count * _PENALTY_PER_CONCERN
    )
    # 总调整下限 -0.5（避免 Confidence 被扣至过低）
    confidence_adjustment = max(confidence_adjustment, -_PENALTY_CAP)
    # unresolved 时额外 -0.2
    if unresolved:
        confidence_adjustment -= _PENALTY_UNRESOLVED

    # -------------------------------------------------------------------------
    # summary 生成
    # -------------------------------------------------------------------------
    summary = _build_summary(
        findings=findings,
        confidence_adjustment=confidence_adjustment,
        retry_required=retry_required,
        retry_count=retry_count,
        max_retries=_MAX_RETRIES,
        unresolved=unresolved,
    )

    logger.info(
        "Scientific Critic 审查完成: pathway=%s, status=%s, "
        "fail=%d, concern=%d, pass=%d, retry_required=%s, "
        "confidence_adjustment=%.2f, unresolved=%s",
        pathway, overall_status, fail_count, concern_count,
        len(findings) - fail_count - concern_count,
        retry_required, confidence_adjustment, unresolved,
    )

    return CriticReport(
        pathway=pathway,
        findings=findings,
        overall_status=overall_status,
        retry_required=retry_required,
        retry_count=retry_count,
        max_retries=_MAX_RETRIES,
        confidence_adjustment=confidence_adjustment,
        unresolved=unresolved,
        summary=summary,
        skipped=False,
        skip_reason="",
    )


__all__ = [
    "CriticFinding",
    "CriticReport",
    "run_scientific_critic",
]
