# BioDynamics Agent - Scientific Alignment Loop: 轻量科学一致性检测函数
# （Spec: add-scientific-reviewer-and-validation-matrix, Task 8 重写版）
#
# 治理哲学（最高优先级，违反即 Reject）：
#   1. Reviewer 是 **纯检测函数**，**不是 Agent**，**不注册到 supervisor.py AGENT_REGISTRY**。
#   2. Reviewer **不生成任何新科学内容**：仅做规则化检测、集合运算、数值比较；
#      禁止调用 LLM 生成审查文本。
#   3. 不重新设计架构：现有 Pipeline（Supervisor → Workers → Simulation →
#      Validation → Report）保持不变。Reviewer 作为 Validation 阶段的后处理函数。
#   4. 禁止 placeholder / hardcode benchmark 答案 / if-else 特判 benchmark。
#   5. 科学正确性铁律：Reviewer 不得创造科学事实。
#
# 模块用途：
#   提供 detect_inconsistencies() 轻量函数，聚合现有 Validation Matrix /
#   Evidence Graph / Curve Metrics / Honesty 等模块的检测结果，输出结构化
#   InconsistencyReport（dict 形式，inconsistency_report.json 兼容）。
#   Reviewer 仅做规则化集合运算与字段读取，不重新计算，不调用 LLM。
#
# 8 项规则化检测（全部无 LLM）：
#   1. Question Coverage      — 提取 question 关键词，检查 report_text 覆盖度
#   2. Mechanism 对照          — canonical required_nodes vs network_json.nodes 集合差
#   3. Simulation 指标对照     — 复用 curve_metrics_result.metrics 中 passed=False 项
#   4. BioModels 对照          — 复用 biomodels_result 中的 deviation / missing_reactions
#   5. Literature 命中率       — canonical_pmids vs retrieved_pmids 集合差
#   6. Evidence 逐句检查       — 复用 evidence_graph.detect_ungrounded / detect_undergrounded
#   7. Experiment 检查          — canonical forbidden_experiments vs experiment_plan 集合差
#   8. Honesty 检查             — 复用 scientific_honesty.review_report 结果
#
# 依赖与复用（必须复用，不得重新实现）：
#   - app.scientific_alignment.evidence_graph.detect_ungrounded / detect_undergrounded
#   - app.scientific_alignment.scientific_honesty.review_report
#   - app.scientific_alignment.curve_metrics.compare_with_expected（如可用，
#     否则直接读取 curve_metrics_result.metrics 字段）
#
# Feature Flag 守护：
#   V4_SCIENTIFIC_REVIEWER_ENABLED 默认 false。关闭时 detect_inconsistencies
#   返回 overall_status="SKIPPED" 的空报告，系统行为与 v3/v4 一致。
#
# 核心导出：
#   from app.scientific_alignment.scientific_reviewer import (
#       InconsistencyReport,
#       detect_inconsistencies,
#       ADAPTIVE_MECHANISM_KEYWORDS,
#       DECLINE_TRIGGERS,
#   )

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.config import settings
from app.scientific_alignment import (
    evidence_graph,
    scientific_honesty,
)

logger = logging.getLogger(__name__)


# =============================================================================
# 常量定义：Question Coverage 关键词库
# =============================================================================

# 负反馈 / 适配机制关键词（用于检测 "why decline" 类问题是否被回答）
# 若 question 含 DECLINE_TRIGGERS，report_text 必须含至少一个本组关键词
ADAPTIVE_MECHANISM_KEYWORDS: tuple[str, ...] = (
    "DUSP",
    "Sprouty",
    "internalization",
    "negative feedback",
    "dephosphorylation",
    "MKP",
    "SPRY",
    "PTEN",
    "适应",
    "负反馈",
    "去磷酸化",
    "内化",
    "downregulation",
)

# 问题中的 "下降 / 适配" 触发词
# 若 question 含任一关键词且 report_text 不含任一 ADAPTIVE_MECHANISM_KEYWORDS
# → question_coverage_gaps 追加 "adaptive mechanism 未解释"，overall_status=FAIL
DECLINE_TRIGGERS: tuple[str, ...] = (
    "decline",
    "decrease",
    "adaptive",
    "为什么下降",
    "为何下降",
    "为何降低",
    "为什么降低",
    "transient",
    "短暂",
    "一过性",
)


# =============================================================================
# 数据结构
# =============================================================================


@dataclass
class InconsistencyReport:
    """轻量科学一致性检测报告（纯规则，无 LLM）。

    所有字段为简单 list[str] / list[dict]，便于序列化为 inconsistency_report.json。
    Reviewer 不创造科学事实，仅做规则化聚合。

    Attributes:
        question_coverage_gaps: 用户问题未覆盖的子问题（如
            "adaptive mechanism 未解释"）。
        missing_mechanism_nodes: Canonical required_nodes 中未出现在
            network_json.nodes 的节点 ID 列表。
        failed_curve_metrics: Curve Metrics 中 passed=False 的指标项，
            每项含 name / expected / actual / delta。
        biomodels_deviations: BioModels 偏差记录列表，复用 biomodels_result
            中的 deviation / missing_reactions / parameter_mismatch 字段。
        literature_missed_canonical: Canonical PMID 中未在 retrieved_pmids
            出现的列表。
        ungrounded_sentences: Evidence Graph 中无任何证据支撑的句子列表。
        forbidden_experiments_recommended: Canonical forbidden_experiments
            中但被 experiment_plan 推荐的实验名称列表。
        honesty_violations: Scientific Honesty 违规列表，每项含
            violation_type / sentence / severity。
        overall_status: 综合状态 PASS / PARTIAL / FAIL / SKIPPED。
        pathway: 通路标识（如 "egfr"），仅用于报告标识。
        timestamp: ISO 8601 时间戳（detect_inconsistencies 调用时刻）。
    """

    question_coverage_gaps: list[str] = field(default_factory=list)
    missing_mechanism_nodes: list[str] = field(default_factory=list)
    failed_curve_metrics: list[dict] = field(default_factory=list)
    biomodels_deviations: list[dict] = field(default_factory=list)
    literature_missed_canonical: list[str] = field(default_factory=list)
    ungrounded_sentences: list[str] = field(default_factory=list)
    forbidden_experiments_recommended: list[str] = field(default_factory=list)
    honesty_violations: list[dict] = field(default_factory=list)
    overall_status: str = "PASS"
    pathway: str = ""
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        """序列化为 inconsistency_report.json 兼容 dict。

        Returns:
            包含全部 8 项检测字段 + overall_status + pathway + timestamp 的 dict。
        """
        return {
            "question_coverage_gaps": list(self.question_coverage_gaps),
            "missing_mechanism_nodes": list(self.missing_mechanism_nodes),
            "failed_curve_metrics": list(self.failed_curve_metrics),
            "biomodels_deviations": list(self.biomodels_deviations),
            "literature_missed_canonical": list(self.literature_missed_canonical),
            "ungrounded_sentences": list(self.ungrounded_sentences),
            "forbidden_experiments_recommended": list(
                self.forbidden_experiments_recommended
            ),
            "honesty_violations": list(self.honesty_violations),
            "overall_status": self.overall_status,
            "pathway": self.pathway,
            "timestamp": self.timestamp,
        }

    def to_json(self, path: str) -> str:
        """写入 inconsistency_report.json 文件，返回路径。

        Args:
            path: 目标 JSON 文件绝对路径。父目录须存在。

        Returns:
            实际写入的文件路径字符串（与 path 一致）。
        """
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=2)
        return path


# =============================================================================
# 内部辅助函数
# =============================================================================


def _text_contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    """检查 text 是否包含 keywords 中任一项（大小写不敏感）。

    Args:
        text: 待检测文本。
        keywords: 关键词元组。

    Returns:
        True 表示命中至少一个关键词。
    """
    if not text or not keywords:
        return False
    text_lower = text.lower()
    for kw in keywords:
        if not kw:
            continue
        if kw.lower() in text_lower:
            return True
    return False


def _extract_network_node_ids(network_json: dict[str, Any]) -> set[str]:
    """从 network_json.nodes 提取节点 ID 集合。

    兼容 list[dict]（取 id/name/label 字段）与 list[str] 两种格式。
    不做归一化，保持原始 ID 字符串以与 Canonical required_nodes 对照。

    Args:
        network_json: 通路网络 dict，需含 "nodes" 字段。

    Returns:
        节点 ID 字符串集合。
    """
    if not isinstance(network_json, dict):
        return set()
    nodes = network_json.get("nodes") or []
    if not isinstance(nodes, list):
        return set()
    result: set[str] = set()
    for node in nodes:
        if isinstance(node, str):
            if node.strip():
                result.add(node.strip())
        elif isinstance(node, dict):
            for key in ("id", "name", "label"):
                val = node.get(key)
                if isinstance(val, str) and val.strip():
                    result.add(val.strip())
                    break
    return result


def _extract_canonical_pmids(canonical: dict[str, Any] | None) -> list[str]:
    """从 canonical.canonical_reviews 提取 PMID 数字字符串列表。

    兼容两种 canonical_reviews 格式：
        - list[str]（如 "PMID:12345678"）→ 提取数字部分
        - list[dict]（如 {"pmid": "12345678"}）→ 提取 pmid 字段

    Args:
        canonical: Canonical Reference dict。None 时返回空列表。

    Returns:
        Canonical PMID 数字字符串列表（去重，保序）。
    """
    if not isinstance(canonical, dict):
        return []
    reviews = canonical.get("canonical_reviews") or []
    if not isinstance(reviews, list):
        return []
    pmids: list[str] = []
    seen: set[str] = set()
    for entry in reviews:
        if isinstance(entry, str):
            match = re.search(r"(\d{4,})", entry)
            if match:
                digit = match.group(1)
                if digit not in seen:
                    seen.add(digit)
                    pmids.append(digit)
        elif isinstance(entry, dict):
            for key in ("pmid", "PMID", "id", "pubmed_id"):
                val = entry.get(key)
                if val is None:
                    continue
                if isinstance(val, int):
                    digit = str(val)
                elif isinstance(val, str):
                    match = re.search(r"(\d{4,})", val)
                    if not match:
                        continue
                    digit = match.group(1)
                else:
                    continue
                if digit not in seen:
                    seen.add(digit)
                    pmids.append(digit)
                break
    return pmids


def _normalize_experiment_name(name: Any) -> str:
    """归一化实验名称为小写字符串用于子串匹配。"""
    if name is None:
        return ""
    return str(name).strip().lower()


# =============================================================================
# 8 项规则化检测（每项为独立私有函数，便于单元测试与维护）
# =============================================================================


def _check_question_coverage(
    question: str,
    report_text: str,
) -> list[str]:
    """检测 1：Question Coverage。

    规则：
        - 若 question 含 DECLINE_TRIGGERS 任一关键词，且 report_text 不含
          ADAPTIVE_MECHANISM_KEYWORDS 任一关键词 → 追加 gap。
        - 其他子问题检测可通过扩展本函数实现，当前仅做"下降/适配"检测。

    Args:
        question: 用户原始问题。
        report_text: 最终 Report 文本。

    Returns:
        未覆盖子问题列表（空列表表示全覆盖）。
    """
    gaps: list[str] = []
    if not question or not report_text:
        # question 为空时不做检测；report 为空视为完全未覆盖
        if question and not report_text:
            gaps.append("report 为空，无法覆盖问题")
        return gaps

    has_decline_trigger = _text_contains_any(question, DECLINE_TRIGGERS)
    if has_decline_trigger:
        has_adaptive_kw = _text_contains_any(report_text, ADAPTIVE_MECHANISM_KEYWORDS)
        if not has_adaptive_kw:
            gaps.append("adaptive mechanism 未解释")
    return gaps


def _check_mechanism_nodes(
    network_json: dict[str, Any],
    canonical: dict[str, Any] | None,
) -> list[str]:
    """检测 2：Mechanism 对照（集合差集）。

    规则：set(required_nodes) - set(network_json.nodes.id)

    Args:
        network_json: 通路网络 dict，需含 "nodes" 字段。
        canonical: Canonical Reference dict，需含
            canonical_mechanism.required_nodes。

    Returns:
        缺失节点 ID 列表（保序，与 Canonical required_nodes 顺序一致）。
    """
    if not isinstance(canonical, dict):
        return []
    mech = canonical.get("canonical_mechanism") or {}
    if not isinstance(mech, dict):
        return []
    required_nodes = mech.get("required_nodes") or []
    if not isinstance(required_nodes, list):
        return []

    network_ids = _extract_network_node_ids(network_json)
    missing: list[str] = []
    seen: set[str] = set()
    for node in required_nodes:
        if not isinstance(node, str):
            continue
        node_str = node.strip()
        if not node_str:
            continue
        if node_str not in network_ids and node_str not in seen:
            missing.append(node_str)
            seen.add(node_str)
    return missing


def _check_curve_metrics(
    curve_metrics_result: dict[str, Any],
    canonical: dict[str, Any] | None,
) -> list[dict]:
    """检测 3：Simulation 指标对照。

    复用 curve_metrics_result.metrics 中 passed=False 的指标项，
    输出含 name / expected / actual / delta 的 dict 列表。

    Args:
        curve_metrics_result: Curve Metrics 检测结果，含
            "metrics": list[dict]，每项含 name / passed / expected / actual。
        canonical: Canonical Reference dict（含 expected_curve_metrics，
            当前实现仅做信号保留，实际复用 curve_metrics_result 字段）。

    Returns:
        失败指标列表，每项含 name / expected / actual / delta（若可用）。
    """
    # canonical.expected_curve_metrics 仅作为信号保留，不在此重新计算
    # （治理哲学：Reviewer 不重新计算，仅复用已有结果）。
    _ = canonical  # noqa: F841  显式标记未使用，保留接口对称性

    if not isinstance(curve_metrics_result, dict):
        return []
    metrics = curve_metrics_result.get("metrics")
    if not isinstance(metrics, list):
        return []

    failed: list[dict] = []
    for metric in metrics:
        if not isinstance(metric, dict):
            continue
        passed = metric.get("passed")
        # 严格判断 passed=False（避免 None / 缺字段被误判为 FAIL）
        if passed is False:
            entry: dict[str, Any] = {
                "name": str(metric.get("name", "")),
                "expected": metric.get("expected"),
                "actual": metric.get("actual"),
            }
            if "delta" in metric:
                entry["delta"] = metric.get("delta")
            failed.append(entry)
    return failed


def _check_biomodels_deviations(
    biomodels_result: dict[str, Any],
) -> list[dict]:
    """检测 4：BioModels 对照。

    复用 biomodels_result 中的 deviation / missing_reactions /
    parameter_mismatch 字段，构造偏差记录列表。仅当字段存在且表示实际偏差
    （deviation > 0 / missing_reactions 非空 / parameter_mismatch 非空）时
    才追加记录，避免 deviation=0.0 触发误报。

    Args:
        biomodels_result: BioModels Oracle 结果 dict，含
            matched_id / deviation / missing_reactions / parameter_mismatch。

    Returns:
        偏差记录列表；无偏差时返回空列表。
    """
    if not isinstance(biomodels_result, dict):
        return []

    deviation = biomodels_result.get("deviation")
    missing_reactions = biomodels_result.get("missing_reactions")
    parameter_mismatch = biomodels_result.get("parameter_mismatch")
    matched_id = (
        biomodels_result.get("matched_id")
        or biomodels_result.get("biomodels_id")
    )

    deviations: list[dict] = []
    # deviation 仅在非零数字时记录（None / 0.0 / 0 视为无偏差）
    if deviation is not None and not isinstance(deviation, bool):
        try:
            dev_val = float(deviation)
            if dev_val > 0.0:
                deviations.append({
                    "type": "deviation",
                    "matched_id": matched_id,
                    "value": deviation,
                })
        except (TypeError, ValueError):
            # 非数值 deviation 视为异常，仍记录
            deviations.append({
                "type": "deviation",
                "matched_id": matched_id,
                "value": deviation,
            })
    # missing_reactions 仅在非空列表/字符串时记录
    if missing_reactions:
        if isinstance(missing_reactions, (list, tuple, str)) and len(missing_reactions) > 0:
            deviations.append({
                "type": "missing_reactions",
                "matched_id": matched_id,
                "value": missing_reactions,
            })
        elif not isinstance(missing_reactions, (list, tuple, str)):
            # 非空非集合类型（如 dict）也记录
            deviations.append({
                "type": "missing_reactions",
                "matched_id": matched_id,
                "value": missing_reactions,
            })
    # parameter_mismatch 仅在非空时记录
    if parameter_mismatch:
        is_empty = (
            isinstance(parameter_mismatch, (list, tuple, dict, str))
            and len(parameter_mismatch) == 0
        )
        if not is_empty:
            deviations.append({
                "type": "parameter_mismatch",
                "matched_id": matched_id,
                "value": parameter_mismatch,
            })
    return deviations


def _check_literature_coverage(
    retrieved_pmids: list[str],
    canonical: dict[str, Any] | None,
) -> list[str]:
    """检测 5：Literature 命中率（集合差集）。

    规则：set(canonical_pmids) - set(retrieved_pmids)

    Args:
        retrieved_pmids: 实际检索到的 PMID 列表（数字字符串）。
        canonical: Canonical Reference dict，含 canonical_reviews。

    Returns:
        未命中的 Canonical PMID 列表（保序，与 canonical_reviews 顺序一致）。
    """
    canonical_pmids = _extract_canonical_pmids(canonical)
    if not canonical_pmids:
        return []
    retrieved_set: set[str] = set()
    for pmid in retrieved_pmids or []:
        if isinstance(pmid, str):
            match = re.search(r"(\d{4,})", pmid)
            if match:
                retrieved_set.add(match.group(1))
        elif isinstance(pmid, int):
            retrieved_set.add(str(pmid))
    missed: list[str] = [
        pmid for pmid in canonical_pmids if pmid not in retrieved_set
    ]
    return missed


def _check_evidence_grounded(
    report_text: str,
    evidence_graph_result: dict[str, Any],
) -> list[str]:
    """检测 6：Evidence 逐句检查（复用 evidence_graph 检测结果）。

    优先复用 evidence_graph_result 中已计算的字段（如
    ungrounded_sentences / nodes[*].evidence_type）。若未提供计算结果，
    尝试用 evidence_graph.build_from_report 重新构建后调用 detect_ungrounded。

    Args:
        report_text: Report 文本（用于 fallback 重新构建）。
        evidence_graph_result: Evidence Graph 结果 dict。

    Returns:
        无证据支撑的句子文本列表。
    """
    if not isinstance(evidence_graph_result, dict):
        return []

    # 优先路径：复用 evidence_graph_result.ungrounded_sentences
    precomputed = evidence_graph_result.get("ungrounded_sentences")
    if isinstance(precomputed, list):
        return [str(s) for s in precomputed if s]

    # Fallback：从 nodes 字段提取 evidence_type == "ungrounded" 的句子
    nodes = evidence_graph_result.get("nodes")
    if isinstance(nodes, list):
        ungrounded: list[str] = []
        for node in nodes:
            if not isinstance(node, dict):
                continue
            etype = node.get("evidence_type")
            if etype == "ungrounded":
                text = node.get("text") or node.get("sentence")
                if isinstance(text, str) and text.strip():
                    ungrounded.append(text)
        return ungrounded

    # 最终 fallback：调用 evidence_graph.build_from_report + detect_ungrounded
    # 仅当 evidence_pool 字段存在时才尝试（避免无依据重建）
    if report_text and isinstance(evidence_graph_result.get("evidence_pool"), dict):
        try:
            graph = evidence_graph.build_from_report(
                report_text,
                evidence_graph_result["evidence_pool"],
            )
            ungrounded_nodes = evidence_graph.detect_ungrounded(graph)
            return [n.text for n in ungrounded_nodes if n.text]
        except Exception as exc:  # noqa: BLE001
            logger.warning("evidence_graph fallback 构建失败: %s", exc)
    return []


def _check_forbidden_experiments(
    experiment_plan: dict[str, Any],
    canonical: dict[str, Any] | None,
) -> list[str]:
    """检测 7：Experiment 检查（集合差集）。

    规则：从 experiment_plan.experiments 中找出 canonical.forbidden_experiments
    子串匹配的实验名称。

    Args:
        experiment_plan: 实验计划 dict，含 "experiments": list[dict]。
        canonical: Canonical Reference dict，含 forbidden_experiments: list[str]。

    Returns:
        被推荐但被禁止的实验名称列表。
    """
    if not isinstance(canonical, dict):
        return []
    forbidden = canonical.get("forbidden_experiments") or []
    if not isinstance(forbidden, list):
        return []
    forbidden_lower = [
        _normalize_experiment_name(f) for f in forbidden
        if isinstance(f, str) and f.strip()
    ]
    if not forbidden_lower:
        return []

    if not isinstance(experiment_plan, dict):
        return []
    experiments = experiment_plan.get("experiments")
    if not isinstance(experiments, list):
        # 兼容 list[dict] 直接传入 experiment_plan 的形态
        # （某些上层可能传 {"experiments": [...]} 或裸 list）
        return []

    recommended_forbidden: list[str] = []
    seen: set[str] = set()
    for exp in experiments:
        if not isinstance(exp, dict):
            continue
        exp_name_raw = (
            exp.get("name")
            or exp.get("experiment")
            or exp.get("id")
            or ""
        )
        exp_name_lower = _normalize_experiment_name(exp_name_raw)
        if not exp_name_lower:
            continue
        for fb_lower, fb_original in zip(forbidden_lower, forbidden):
            if not fb_lower:
                continue
            if fb_lower in exp_name_lower and fb_original not in seen:
                recommended_forbidden.append(str(fb_original))
                seen.add(fb_original)
                break
    return recommended_forbidden


def _check_honesty_violations(
    report_text: str,
    honesty_result: dict[str, Any],
) -> list[dict]:
    """检测 8：Honesty 检查（复用 scientific_honesty.review_report 结果）。

    优先复用 honesty_result 中已计算的 details 字段；若未提供则
    调用 scientific_honesty.review_report 重新检测（仍是纯规则，无 LLM）。

    Args:
        report_text: Report 文本（用于 fallback 调用 review_report）。
        honesty_result: Honesty 检测结果 dict，含
            "violations": list[str] / "details": list[dict]。

    Returns:
        违规列表，每项含 violation_type / sentence / severity。
    """
    if isinstance(honesty_result, dict):
        details = honesty_result.get("details")
        if isinstance(details, list):
            violations: list[dict] = []
            for d in details:
                if not isinstance(d, dict):
                    continue
                vtype = d.get("violation_type") or d.get("type") or ""
                sentence = d.get("sentence", "")
                # severity 推断：overstatement / citation_missing → FAIL；
                # unlabeled_claim → PARTIAL
                if vtype in ("overstatement", "citation_missing"):
                    severity = "FAIL"
                elif vtype == "unlabeled_claim":
                    severity = "PARTIAL"
                else:
                    severity = d.get("severity", "PARTIAL")
                violations.append({
                    "violation_type": str(vtype),
                    "sentence": str(sentence),
                    "severity": severity,
                })
            return violations

    # Fallback：调用 scientific_honesty.review_report 重新检测
    if report_text:
        try:
            review = scientific_honesty.review_report(report_text)
            violations = []
            for d in review.details:
                vtype = d.get("violation_type", "")
                sentence = d.get("sentence", "")
                if vtype in ("overstatement", "citation_missing"):
                    severity = "FAIL"
                elif vtype == "unlabeled_claim":
                    severity = "PARTIAL"
                else:
                    severity = "PARTIAL"
                violations.append({
                    "violation_type": str(vtype),
                    "sentence": str(sentence),
                    "severity": severity,
                })
            return violations
        except Exception as exc:  # noqa: BLE001
            logger.warning("scientific_honesty.review_report fallback 失败: %s", exc)
    return []


# =============================================================================
# overall_status 聚合
# =============================================================================


def _determine_overall_status(report: InconsistencyReport) -> str:
    """根据 8 项检测结果聚合 overall_status。

    判定规则：
        - 任一 FAIL 级别问题（missing_mechanism_nodes 非空 /
          failed_curve_metrics 非空 / honesty_violations 含 overclaim）→ FAIL
        - 仅有 PARTIAL 级别问题（question_coverage_gaps 非空但 mechanism 完整）→ PARTIAL
        - 全部通过 → PASS

    Args:
        report: 已填充 8 项字段的 InconsistencyReport。

    Returns:
        "PASS" / "PARTIAL" / "FAIL"。
    """
    # SKIPPED 状态由 Feature Flag 直接返回，此处不处理
    if report.overall_status == "SKIPPED":
        return report.overall_status

    # FAIL 级别触发条件
    fail_triggers: list[bool] = [
        bool(report.missing_mechanism_nodes),
        bool(report.failed_curve_metrics),
        bool(report.forbidden_experiments_recommended),
        any(
            v.get("severity") == "FAIL"
            for v in report.honesty_violations
            if isinstance(v, dict)
        ),
        # question_coverage_gaps 中 "adaptive mechanism 未解释" 视为 FAIL
        any("adaptive mechanism 未解释" in g for g in report.question_coverage_gaps),
    ]
    if any(fail_triggers):
        return "FAIL"

    # PARTIAL 级别触发条件
    partial_triggers: list[bool] = [
        bool(report.question_coverage_gaps),
        bool(report.biomodels_deviations),
        bool(report.literature_missed_canonical),
        bool(report.ungrounded_sentences),
        any(
            v.get("severity") == "PARTIAL"
            for v in report.honesty_violations
            if isinstance(v, dict)
        ),
    ]
    if any(partial_triggers):
        return "PARTIAL"

    return "PASS"


# =============================================================================
# 核心导出函数
# =============================================================================


def detect_inconsistencies(
    question: str,
    report_text: str,
    simulation_result: dict[str, Any],
    network_json: dict[str, Any],
    retrieved_pmids: list[str],
    biomodels_result: dict[str, Any],
    experiment_plan: dict[str, Any],
    validation_matrix_result: dict[str, Any],
    curve_metrics_result: dict[str, Any],
    evidence_graph_result: dict[str, Any],
    honesty_result: dict[str, Any],
    *,
    pathway: str | None = None,
    canonical: dict[str, Any] | None = None,
) -> InconsistencyReport:
    """轻量科学一致性检测（纯规则，无 LLM）。

    复用现有 Validation Matrix / Evidence Graph / Curve Metrics / Honesty
    检测结果，聚合为结构化不一致清单。Reviewer 不重新计算，不调用 LLM，
    不创造科学事实。

    Args:
        question: 用户原始问题文本。
        report_text: 最终 Report 文本（用于 Question Coverage 与 Honesty Fallback）。
        simulation_result: 仿真结果 dict（当前实现未直接使用，保留接口对称性）。
        network_json: 通路网络 dict，含 "nodes" 字段。
        retrieved_pmids: 实际检索到的 PMID 数字字符串列表。
        biomodels_result: BioModels Oracle 结果 dict。
        experiment_plan: 实验计划 dict，含 "experiments" 字段。
        validation_matrix_result: 12 轴 Validation Matrix 结果 dict
            （当前实现未直接使用，保留接口对称性）。
        curve_metrics_result: Curve Metrics 结果 dict，含 "metrics" 字段。
        evidence_graph_result: Evidence Graph 结果 dict。
        honesty_result: Scientific Honesty 结果 dict，含 "details" 字段。
        pathway: 通路标识（如 "egfr"），仅用于报告标识。
        canonical: Canonical Reference dict，含 canonical_mechanism /
            canonical_reviews / forbidden_experiments 等字段。

    Returns:
        InconsistencyReport 实例。Feature Flag OFF 时返回 overall_status="SKIPPED"
        的空报告，系统行为与 v3/v4 一致。
    """
    # -------------------------------------------------------------------------
    # Feature Flag 守护：Flag OFF 时返回空报告
    # -------------------------------------------------------------------------
    if not getattr(settings, "V4_SCIENTIFIC_REVIEWER_ENABLED", False):
        return InconsistencyReport(
            overall_status="SKIPPED",
            pathway=pathway or "",
        )

    # simulation_result 与 validation_matrix_result 当前未直接使用，
    # 保留参数为后续扩展（治理哲学：不破坏接口对称性）。
    _ = simulation_result  # noqa: F841
    _ = validation_matrix_result  # noqa: F841

    # -------------------------------------------------------------------------
    # 8 项规则化检测（全部无 LLM）
    # -------------------------------------------------------------------------
    report = InconsistencyReport(
        pathway=pathway or "",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    report.question_coverage_gaps = _check_question_coverage(
        question=question, report_text=report_text,
    )
    report.missing_mechanism_nodes = _check_mechanism_nodes(
        network_json=network_json, canonical=canonical,
    )
    report.failed_curve_metrics = _check_curve_metrics(
        curve_metrics_result=curve_metrics_result, canonical=canonical,
    )
    report.biomodels_deviations = _check_biomodels_deviations(
        biomodels_result=biomodels_result,
    )
    report.literature_missed_canonical = _check_literature_coverage(
        retrieved_pmids=retrieved_pmids, canonical=canonical,
    )
    report.ungrounded_sentences = _check_evidence_grounded(
        report_text=report_text,
        evidence_graph_result=evidence_graph_result,
    )
    report.forbidden_experiments_recommended = _check_forbidden_experiments(
        experiment_plan=experiment_plan, canonical=canonical,
    )
    report.honesty_violations = _check_honesty_violations(
        report_text=report_text, honesty_result=honesty_result,
    )

    # -------------------------------------------------------------------------
    # 聚合 overall_status
    # -------------------------------------------------------------------------
    report.overall_status = _determine_overall_status(report)

    logger.info(
        "detect_inconsistencies 完成: pathway=%s, overall_status=%s, "
        "gaps=%d, missing_nodes=%d, failed_metrics=%d, "
        "biomodels_deviations=%d, literature_missed=%d, "
        "ungrounded=%d, forbidden_experiments=%d, honesty_violations=%d",
        report.pathway,
        report.overall_status,
        len(report.question_coverage_gaps),
        len(report.missing_mechanism_nodes),
        len(report.failed_curve_metrics),
        len(report.biomodels_deviations),
        len(report.literature_missed_canonical),
        len(report.ungrounded_sentences),
        len(report.forbidden_experiments_recommended),
        len(report.honesty_violations),
    )

    return report


__all__ = [
    "InconsistencyReport",
    "detect_inconsistencies",
    "ADAPTIVE_MECHANISM_KEYWORDS",
    "DECLINE_TRIGGERS",
]
