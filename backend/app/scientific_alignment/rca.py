# BioDynamics Agent v4 - Scientific Alignment Loop: Root Cause Analyzer (Task 13)
#
# 精确段定位 RCA（Root Cause Analyzer）：聚合各验证器 defect，输出结构化 RCA。
#
# 设计目标：
#   当前各验证器（Seven-Axis / Consistency / Parameter / Critic / Mechanism）
#   独立报告缺陷，但缺乏统一的根因定位。Task 13 聚合所有验证器的 defect，
#   结合 pipeline_trace 的阶段日志，精确定位缺陷发生在管线的哪个阶段，
#   并给出具体修复建议。
#
# 两条核心链路检测：
#   1. 证据丢失链：Retriever Success → Reranker Discarded → Prompt Missing
#                  → Report No Evidence
#   2. BioModels 链：Downloaded → Parser Failed → Species Mapping Failed
#                   → Validation Skipped
#
# 模糊措辞检测（SubTask 13.4）：
#   禁止 "RAG Error" / "未知错误" / "PubMed 0" / "LLM 问题" 等模糊措辞。
#   违例的 RCA 检测器自身判 defect_type=vague_rca。
#
# Feature Flag：
#   本模块是纯诊断聚合工具，无副作用，不需要 Feature Flag 守护。
#   仅当被 Loop Controller（Task 16）显式调用时执行。总是可用。
#
# 安全设计：
#   - 不引入新依赖（仅标准库）
#   - 纯函数，无副作用，不修改输入报告
#   - 所有输入做 None / 空值防御
#   - evidence 字段必须包含具体数据（数值/PMID/节点名），禁止纯文字描述
#
# 依赖：Python 标准库；不引入新依赖，不 import 其他模块（仅接收报告对象）。
#
# 核心导出：
#   from app.scientific_alignment.rca import (
#       RootCause, RCAReport, run_rca,
#   )

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# 常量
# =============================================================================

# 模糊措辞黑名单（不区分大小写检测）
_VAGUE_PHRASES: tuple[str, ...] = (
    "RAG Error",
    "未知错误",
    "PubMed 0",
    "LLM 问题",
    "unknown error",
    "undefined",
)

# 7 轴名称 → 管线阶段映射
#   mechanism  → node1_mechanism（机制提取阶段）
#   dynamics   → node2_ode（ODE 仿真阶段）
#   biomodels  → node3_sandbox（BioModels 对比阶段）
#   literature → node1.5_rag（RAG 检索阶段）
#   experiment → validation（实验验证阶段）
#   discussion → report（报告生成阶段）
#   evidence   → node1.5_rag（证据融合阶段，与 RAG 同阶段）
_AXIS_TO_STAGE: dict[str, str] = {
    "mechanism": "node1_mechanism",
    "dynamics": "node2_ode",
    "biomodels": "node3_sandbox",
    "literature": "node1.5_rag",
    "experiment": "validation",
    "discussion": "report",
    "evidence": "node1.5_rag",
}

# Critic 审查类别 → 管线阶段映射
_CRITIC_CATEGORY_TO_STAGE: dict[str, str] = {
    "mechanism": "node1_mechanism",
    "evidence": "node1.5_rag",
    "biomodels": "node3_sandbox",
    "consistency": "node2_ode",
    "experiments": "validation",
    "references": "node1.5_rag",
}

# 7 轴名称 → 推测模块文件名映射
_AXIS_TO_MODULE: dict[str, str] = {
    "mechanism": "mechanism_checker.py",
    "dynamics": "consistency_checker.py",
    "biomodels": "biomodels_oracle.py",
    "literature": "evidence_ranker.py",
    "experiment": "experiment_planner.py",
    "discussion": "report_generator.py",
    "evidence": "evidence_fusion.py",
}

# Critic 审查类别 → 推测模块文件名映射
_CRITIC_CATEGORY_TO_MODULE: dict[str, str] = {
    "mechanism": "mechanism_checker.py",
    "evidence": "evidence_ranker.py",
    "biomodels": "biomodels_oracle.py",
    "consistency": "consistency_checker.py",
    "experiments": "experiment_planner.py",
    "references": "evidence_ranker.py",
}


# =============================================================================
# 数据类
# =============================================================================
@dataclass
class RootCause:
    """单条根因记录。

    Attributes:
        failed_stage: 缺陷发生的管线阶段：
            "node0_mcp" / "node1_mechanism" / "node1.5_rag" /
            "node2_ode" / "node3_sandbox" / "report" / "validation"。
        defect_type: 缺陷类型标签（见 SubTask 13.2/13.3/13.4 的具体标签）。
        evidence: 日志片段/数据证据（具体值，非模糊描述）。
        suspected_module: 推测的模块文件名。
        repair_suggestion: 具体修复建议。
        severity: 严重等级 high / medium / low。
    """

    failed_stage: str
    defect_type: str
    evidence: str
    suspected_module: str
    repair_suggestion: str
    severity: str = "high"


@dataclass
class RCAReport:
    """RCA 报告。

    Attributes:
        causes: 根因列表（RootCause）。
        has_vague: 是否检测到模糊措辞。
        primary_cause: 最严重的一条摘要。
        total_defects: 缺陷总数。
    """

    causes: list[RootCause] = field(default_factory=list)
    has_vague: bool = False
    primary_cause: str = ""
    total_defects: int = 0


# =============================================================================
# 证据丢失链检测（SubTask 13.2）
# =============================================================================
def _check_evidence_lost_chain(pipeline_trace: dict) -> list[RootCause]:
    """检测证据丢失链路。

    链路：Retriever Success → Reranker Discarded → Prompt Missing
          → Report No Evidence

    - 检索成功但被 rerank 丢弃 → evidence_lost_at_rerank
    - 进入 prompt 但报告未引用 → evidence_lost_at_prompt
    - 报告引用但无具体 PMID → evidence_lost_at_report

    Args:
        pipeline_trace: 管线阶段日志。

    Returns:
        检测到的 RootCause 列表。
    """
    causes: list[RootCause] = []

    retriever = pipeline_trace.get("retriever", {}) or {}
    reranker = pipeline_trace.get("reranker", {}) or {}
    prompt = pipeline_trace.get("prompt", {}) or {}
    report = pipeline_trace.get("report", {}) or {}

    # 1. 检索成功但被 rerank 丢弃 → evidence_lost_at_rerank
    retriever_success = bool(retriever.get("success", False))
    discarded_count = int(reranker.get("discarded_count", 0) or 0)
    if retriever_success and discarded_count > 0:
        causes.append(RootCause(
            failed_stage="node1.5_rag",
            defect_type="evidence_lost_at_rerank",
            evidence=(
                f"retriever.success={retriever_success}, "
                f"reranker.discarded_count={discarded_count}"
            ),
            suspected_module="rag_reranker.py",
            repair_suggestion=(
                f"检查 reranker 过滤逻辑，{discarded_count} 条检索成功的证据"
                "被丢弃，调整 rerank 阈值或保留策略"
            ),
            severity="high",
        ))

    # 2. 进入 prompt 但报告未引用 → evidence_lost_at_prompt
    evidence_in_context = bool(prompt.get("evidence_in_context", False))
    pmid_cited_count = int(report.get("pmid_cited_count", 0) or 0)
    if evidence_in_context and pmid_cited_count == 0:
        causes.append(RootCause(
            failed_stage="node1.5_rag",
            defect_type="evidence_lost_at_prompt",
            evidence=(
                f"prompt.evidence_in_context={evidence_in_context}, "
                f"report.pmid_cited_count={pmid_cited_count}"
            ),
            suspected_module="prompt_builder.py",
            repair_suggestion=(
                "证据已进入 prompt 上下文但报告未引用任何 PMID，"
                "检查 Report Generator 是否提取并引用了上下文中的文献"
            ),
            severity="high",
        ))

    # 3. 报告引用但无具体 PMID → evidence_lost_at_report
    cited_pmids = report.get("cited_pmids", [])
    if pmid_cited_count > 0 and not cited_pmids:
        causes.append(RootCause(
            failed_stage="report",
            defect_type="evidence_lost_at_report",
            evidence=(
                f"report.pmid_cited_count={pmid_cited_count}, "
                f"report.cited_pmids={cited_pmids}"
            ),
            suspected_module="report_generator.py",
            repair_suggestion=(
                f"报告声称引用了 {pmid_cited_count} 篇文献但未提供具体 PMID，"
                "检查 Report Generator 是否输出了完整的 PMID 列表"
            ),
            severity="high",
        ))

    return causes


# =============================================================================
# BioModels 链检测（SubTask 13.3）
# =============================================================================
def _check_biomodels_chain(pipeline_trace: dict) -> list[RootCause]:
    """检测 BioModels 链路问题。

    链路：BioModels Downloaded → Parser Failed → Species Mapping Failed
          → Validation Skipped

    - 下载成功但解析失败 → biomodels_parser_failed
    - 解析成功但物种映射失败 → biomodels_species_mapping_failed
    - 映射成功但未做对照 → biomodels_validation_skipped

    Args:
        pipeline_trace: 管线阶段日志。

    Returns:
        检测到的 RootCause 列表。
    """
    causes: list[RootCause] = []

    biomodels = pipeline_trace.get("biomodels", {}) or {}

    downloaded = bool(biomodels.get("downloaded", False))
    parser_success = bool(biomodels.get("parser_success", False))
    species_mapping_success = bool(
        biomodels.get("species_mapping_success", False)
    )
    comparison_done = bool(biomodels.get("comparison_done", False))

    # 1. 下载成功但解析失败 → biomodels_parser_failed
    if downloaded and not parser_success:
        causes.append(RootCause(
            failed_stage="node3_sandbox",
            defect_type="biomodels_parser_failed",
            evidence=(
                f"biomodels.downloaded={downloaded}, "
                f"biomodels.parser_success={parser_success}"
            ),
            suspected_module="biomodels_parser.py",
            repair_suggestion=(
                "BioModels 模型下载成功但 SBML 解析失败，"
                "检查 SBML 文件格式与解析器兼容性"
            ),
            severity="high",
        ))

    # 2. 解析成功但物种映射失败 → biomodels_species_mapping_failed
    if parser_success and not species_mapping_success:
        causes.append(RootCause(
            failed_stage="node3_sandbox",
            defect_type="biomodels_species_mapping_failed",
            evidence=(
                f"biomodels.parser_success={parser_success}, "
                f"biomodels.species_mapping_success={species_mapping_success}"
            ),
            suspected_module="species_mapper.py",
            repair_suggestion=(
                "SBML 解析成功但物种名映射失败，"
                "检查 BioModels 物种名与项目规范名的映射表"
            ),
            severity="medium",
        ))

    # 3. 映射成功但未做对照 → biomodels_validation_skipped
    if species_mapping_success and not comparison_done:
        causes.append(RootCause(
            failed_stage="node3_sandbox",
            defect_type="biomodels_validation_skipped",
            evidence=(
                f"biomodels.species_mapping_success={species_mapping_success}, "
                f"biomodels.comparison_done={comparison_done}"
            ),
            suspected_module="biomodels_oracle.py",
            repair_suggestion=(
                "物种映射成功但未执行 BioModels 对照验证，"
                "检查 BioModels Oracle 是否被正确触发"
            ),
            severity="medium",
        ))

    return causes


# =============================================================================
# 各验证器 defect 聚合
# =============================================================================
def _aggregate_seven_axis(report: Any) -> list[RootCause]:
    """聚合 SevenAxisReport 中 failed 轴的缺陷。

    任一轴 failed → failed_stage 对应轴名，defect_type=axis_failed。

    Args:
        report: SevenAxisReport 实例。

    Returns:
        检测到的 RootCause 列表。
    """
    causes: list[RootCause] = []

    axes = getattr(report, "axes", None) or []
    for axis in axes:
        status = getattr(axis, "status", "")
        if status != "failed":
            continue

        axis_name = getattr(axis, "axis_name", "unknown")
        score = getattr(axis, "score", 0.0)
        failure_reasons = getattr(axis, "failure_reasons", []) or []

        failed_stage = _AXIS_TO_STAGE.get(axis_name, "validation")
        suspected_module = _AXIS_TO_MODULE.get(axis_name, "unknown.py")

        # evidence 包含具体数据：轴名、分数、失败原因
        evidence = (
            f"axis={axis_name}, score={score:.2f}, status={status}, "
            f"failure_reasons={failure_reasons}"
        )

        causes.append(RootCause(
            failed_stage=failed_stage,
            defect_type="axis_failed",
            evidence=evidence,
            suspected_module=suspected_module,
            repair_suggestion=(
                f"修复 {axis_name} 轴失败: "
                + (failure_reasons[0] if failure_reasons else "检查轴评估逻辑")
            ),
            severity="high",
        ))

    return causes


def _aggregate_consistency(report: Any) -> list[RootCause]:
    """聚合 ConsistencyReport 中 violations 的缺陷。

    consistency_report.violations 非空 → defect_type=consistency_violation。

    Args:
        report: ConsistencyReport 实例。

    Returns:
        检测到的 RootCause 列表。
    """
    causes: list[RootCause] = []

    violations = getattr(report, "violations", None) or []
    if not violations:
        return causes

    # 收集具体数据：违规标签与观测值
    labels = [getattr(v, "violation_label", "") for v in violations]
    # 取第一条违规的观测值作为证据
    first_violation = violations[0]
    observed_values = getattr(first_violation, "observed_values", {}) or {}

    evidence = (
        f"violations_count={len(violations)}, "
        f"violation_labels={labels}, "
        f"first_observed_values={observed_values}"
    )

    causes.append(RootCause(
        failed_stage="node2_ode",
        defect_type="consistency_violation",
        evidence=evidence,
        suspected_module="consistency_checker.py",
        repair_suggestion=(
            f"修正仿真结果以满足 {len(violations)} 条一致性规则违规"
            f"（标签: {labels}），检查仿真参数与机制因果顺序"
        ),
        severity="high",
    ))

    return causes


def _aggregate_parameter(report: Any) -> list[RootCause]:
    """聚合 ParameterPriorReport 中的参数缺陷。

    parameter_report.defect 非空 → defect_type=parameter_defect。
    检查报告级 defect 字段（防御性 getattr）与各 ParameterPrior 的 defect 字段。

    Args:
        report: ParameterPriorReport 实例。

    Returns:
        检测到的 RootCause 列表。
    """
    causes: list[RootCause] = []

    # 检查报告级 defect 字段（防御性 getattr，当前 ParameterPriorReport 无此字段）
    report_defect = getattr(report, "defect", "")
    if report_defect:
        causes.append(RootCause(
            failed_stage="node2_ode",
            defect_type="parameter_defect",
            evidence=f"report.defect={report_defect}",
            suspected_module="parameter_grounder.py",
            repair_suggestion=f"修复参数报告缺陷: {report_defect}",
            severity="medium",
        ))
        return causes

    # 检查各 ParameterPrior 的 defect 字段
    priors = getattr(report, "priors", None) or []
    defective_priors = [
        p for p in priors
        if getattr(p, "defect", "") and getattr(p, "defect", "").strip()
    ]

    if not defective_priors:
        return causes

    # 收集具体数据：缺陷参数名与缺陷标记
    param_names = [getattr(p, "param_name", "") for p in defective_priors]
    defects = [getattr(p, "defect", "") for p in defective_priors]

    evidence = (
        f"defective_param_count={len(defective_priors)}, "
        f"param_names={param_names}, "
        f"defects={defects}"
    )

    causes.append(RootCause(
        failed_stage="node2_ode",
        defect_type="parameter_defect",
        evidence=evidence,
        suspected_module="parameter_grounder.py",
        repair_suggestion=(
            f"修复 {len(defective_priors)} 个参数的 provenance 缺陷"
            f"（参数: {param_names}），补充缺失的 value/confidence/source/"
            "distribution/reference 字段"
        ),
        severity="medium",
    ))

    return causes


def _aggregate_critic(report: Any) -> list[RootCause]:
    """聚合 CriticReport 中 severity=fail 的发现。

    critic_report.findings 中 severity=fail → defect_type=critic_finding。

    Args:
        report: CriticReport 实例。

    Returns:
        检测到的 RootCause 列表。
    """
    causes: list[RootCause] = []

    findings = getattr(report, "findings", None) or []
    failed_findings = [
        f for f in findings
        if getattr(f, "severity", "") == "fail"
    ]

    if not failed_findings:
        return causes

    for finding in failed_findings:
        category = getattr(finding, "category", "unknown")
        evidence_str = getattr(finding, "evidence", "")
        finding_text = getattr(finding, "finding", "")
        suggestion = getattr(finding, "suggestion", "")

        failed_stage = _CRITIC_CATEGORY_TO_STAGE.get(category, "report")
        suspected_module = _CRITIC_CATEGORY_TO_MODULE.get(category, "unknown.py")

        evidence = (
            f"category={category}, severity=fail, "
            f"evidence={evidence_str}, finding={finding_text}"
        )

        causes.append(RootCause(
            failed_stage=failed_stage,
            defect_type="critic_finding",
            evidence=evidence,
            suspected_module=suspected_module,
            repair_suggestion=suggestion or f"修复 {category} 类别的审稿失败",
            severity="high",
        ))

    return causes


def _aggregate_mechanism(report: Any) -> list[RootCause]:
    """聚合 MechanismAlignmentResult 中缺失节点的缺陷。

    mechanism_report 缺失节点 → defect_type=mechanism_node_missing。

    Args:
        report: MechanismAlignmentResult 实例。

    Returns:
        检测到的 RootCause 列表。
    """
    causes: list[RootCause] = []

    missing_nodes = getattr(report, "missing_nodes", None) or []
    if not missing_nodes:
        return causes

    missing_critical = getattr(report, "missing_critical_nodes", []) or []
    coverage = getattr(report, "coverage", 0.0)

    # evidence 包含具体数据：缺失节点名、覆盖率
    evidence = (
        f"missing_nodes={missing_nodes}, "
        f"missing_critical_nodes={missing_critical}, "
        f"coverage={coverage:.2f}"
    )

    # 缺关键节点 → high，仅缺普通节点 → medium
    severity = "high" if missing_critical else "medium"

    suggestion = (
        f"补充缺失的机制节点: {missing_nodes}"
        + (f"（其中关键节点: {missing_critical}）" if missing_critical else "")
        + "，确保负反馈与节点链完整"
    )

    causes.append(RootCause(
        failed_stage="node1_mechanism",
        defect_type="mechanism_node_missing",
        evidence=evidence,
        suspected_module="mechanism_checker.py",
        repair_suggestion=suggestion,
        severity=severity,
    ))

    return causes


# =============================================================================
# 模糊措辞检测（SubTask 13.4）
# =============================================================================
def _detect_vague_wording(
    causes: list[RootCause],
) -> tuple[bool, list[RootCause]]:
    """检测 evidence 字段中的模糊措辞。

    检查所有 evidence 字段是否包含模糊措辞（不区分大小写）。
    命中则追加一条 RootCause(defect_type="vague_rca", ...)。

    Args:
        causes: 已收集的 RootCause 列表。

    Returns:
        (has_vague, vague_causes)：是否检测到模糊措辞 + 追加的 RootCause 列表。
    """
    vague_causes: list[RootCause] = []
    has_vague = False

    for cause in causes:
        evidence_lower = cause.evidence.lower()
        for phrase in _VAGUE_PHRASES:
            if phrase.lower() in evidence_lower:
                has_vague = True
                vague_causes.append(RootCause(
                    failed_stage="validation",
                    defect_type="vague_rca",
                    evidence=(
                        f"vague_phrase={phrase!r}, "
                        f"in_defect_type={cause.defect_type}, "
                        f"evidence_snippet={cause.evidence[:200]}"
                    ),
                    suspected_module="rca.py",
                    repair_suggestion=(
                        f"RCA evidence 中检测到模糊措辞 {phrase!r}，"
                        "请提供具体数据（数值/PMID/节点名）替代模糊描述"
                    ),
                    severity="high",
                ))
                break  # 每条 cause 仅追加一条 vague_rca

    return has_vague, vague_causes


# =============================================================================
# primary_cause 选取
# =============================================================================
def _select_primary_cause(causes: list[RootCause]) -> str:
    """选取最严重的一条根因作为 primary_cause。

    规则：severity=high 的第一条；无 high 则第一条 medium；无 medium 则第一条 low。

    Args:
        causes: RootCause 列表。

    Returns:
        primary_cause 摘要字符串；无 causes 时返回空字符串。
    """
    if not causes:
        return ""

    # 优先级：high > medium > low
    severity_order = {"high": 0, "medium": 1, "low": 2}
    sorted_causes = sorted(
        causes, key=lambda c: severity_order.get(c.severity, 3)
    )

    primary = sorted_causes[0]
    return (
        f"[{primary.severity}] {primary.failed_stage} / "
        f"{primary.defect_type}: {primary.evidence}"
    )


# =============================================================================
# 主函数
# =============================================================================
def run_rca(
    pathway: str,
    seven_axis_report: Any | None = None,
    consistency_report: Any | None = None,
    parameter_report: Any | None = None,
    critic_report: Any | None = None,
    mechanism_report: Any | None = None,
    pipeline_trace: dict | None = None,
) -> RCAReport:
    """运行根因分析，聚合各验证器 defect，输出结构化 RCA。

    聚合五类验证器报告 + 两条管线链路检测，输出统一的 RCAReport。
    所有输入参数为 None 时返回空 RCAReport（causes=[], total_defects=0）。

    Args:
        pathway: 通路标识（如 "egfr"）。
        seven_axis_report: SevenAxisReport 实例。
        consistency_report: ConsistencyReport 实例。
        parameter_report: ParameterPriorReport 实例。
        critic_report: CriticReport 实例。
        mechanism_report: MechanismAlignmentResult 实例。
        pipeline_trace: 管线阶段日志，结构如下::

            {
                "retriever": {"success": True},
                "reranker": {"discarded_count": 3},
                "prompt": {"evidence_in_context": True},
                "report": {"pmid_cited_count": 0, "cited_pmids": [...]},
                "biomodels": {
                    "downloaded": True,
                    "parser_success": False,
                    "species_mapping_success": False,
                    "comparison_done": False,
                },
            }

    Returns:
        RCAReport。所有输入为 None 时返回空报告。
    """
    # -------------------------------------------------------------------------
    # 1. 收集所有根因
    # -------------------------------------------------------------------------
    causes: list[RootCause] = []

    # 1a. 证据丢失链检测（pipeline_trace 提供）
    if pipeline_trace:
        try:
            causes.extend(_check_evidence_lost_chain(pipeline_trace))
        except Exception as exc:  # noqa: BLE001
            logger.warning("证据丢失链检测异常: %s", exc)

        # 1b. BioModels 链检测（pipeline_trace["biomodels"] 提供）
        try:
            causes.extend(_check_biomodels_chain(pipeline_trace))
        except Exception as exc:  # noqa: BLE001
            logger.warning("BioModels 链检测异常: %s", exc)

    # 1c. 各验证器 defect 聚合
    if seven_axis_report is not None:
        try:
            causes.extend(_aggregate_seven_axis(seven_axis_report))
        except Exception as exc:  # noqa: BLE001
            logger.warning("SevenAxis defect 聚合异常: %s", exc)

    if consistency_report is not None:
        try:
            causes.extend(_aggregate_consistency(consistency_report))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Consistency defect 聚合异常: %s", exc)

    if parameter_report is not None:
        try:
            causes.extend(_aggregate_parameter(parameter_report))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Parameter defect 聚合异常: %s", exc)

    if critic_report is not None:
        try:
            causes.extend(_aggregate_critic(critic_report))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Critic defect 聚合异常: %s", exc)

    if mechanism_report is not None:
        try:
            causes.extend(_aggregate_mechanism(mechanism_report))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Mechanism defect 聚合异常: %s", exc)

    # -------------------------------------------------------------------------
    # 2. 模糊措辞检测（检查所有 evidence 字段）
    # -------------------------------------------------------------------------
    try:
        has_vague, vague_causes = _detect_vague_wording(causes)
        causes.extend(vague_causes)
    except Exception as exc:  # noqa: BLE001
        has_vague = False
        logger.warning("模糊措辞检测异常: %s", exc)

    # -------------------------------------------------------------------------
    # 3. primary_cause 选取
    # -------------------------------------------------------------------------
    primary_cause = _select_primary_cause(causes)

    # -------------------------------------------------------------------------
    # 4. 构建 RCAReport
    # -------------------------------------------------------------------------
    report = RCAReport(
        causes=causes,
        has_vague=has_vague,
        primary_cause=primary_cause,
        total_defects=len(causes),
    )

    logger.info(
        "RCA 完成: pathway=%s, total_defects=%d, has_vague=%s, "
        "primary_cause=%.80s",
        pathway, report.total_defects, report.has_vague, report.primary_cause,
    )

    return report


__all__ = [
    "RootCause",
    "RCAReport",
    "run_rca",
]
