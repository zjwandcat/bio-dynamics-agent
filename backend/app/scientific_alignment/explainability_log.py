# BioDynamics Agent v4 - Scientific Alignment Sprint 5: Explainability Log
#
# 科学决策日志：记录 8 个维度的决策来源与原因，在 Report 末尾渲染 Decision Log。
#
# 8 个维度：
#   1. Mechanism   — 机制分析来源（知识图谱 / Canonical）
#   2. Confidence  — 置信度评估来源（Rule Engine / Validation）
#   3. BioModels   — BioModels 模型匹配来源
#   4. Parameter   — 参数来源（RAG / SBML / Inferred）
#   5. Discussion  — 讨论文献来源（PubMed / [D] 推理）
#   6. Experiment  — 实验推荐来源（Rule Engine / YAML）
#   7. Validation  — 验证规则来源（Canonical Timeline / Consistency Rules）
#   8. Cross-talk  — 通路串扰来源（知识图谱 / 假设）
#
# 设计原则：
#   - 100% Rule-driven，无 LLM 调用
#   - 基于已有 state 数据生成，不创造新科学事实
#   - Feature Flag 守护：SA_SPRINT5_PROVENANCE_EXPLAINABILITY
#
# 依赖：app.config.settings；不引入新依赖。
#
# 核心导出：
#   from app.scientific_alignment.explainability_log import (
#       DecisionLogEntry, DecisionLogReport,
#       build_decision_log, render_decision_log,
#   )

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)


# =============================================================================
# 数据类
# =============================================================================
@dataclass
class DecisionLogEntry:
    """单条决策日志。

    Attributes:
        dimension: 维度名（Mechanism/Confidence/BioModels/Parameter/
            Discussion/Experiment/Validation/Cross-talk）。
        decision: 决策内容简述。
        source: 来源类型（KG/Canonical/RAG/SBML/PubMed/Rule/Hypothesis 等）。
        reason: 决策原因。
        confidence: 置信度（0.0-1.0，-1.0 表示不适用）。
        evidence_ref: 证据引用（如 ``"PMID:12345"`` / ``"BIOMD0000000010"``）。
    """

    dimension: str = ""
    decision: str = ""
    source: str = ""
    reason: str = ""
    confidence: float = -1.0
    evidence_ref: str = ""


@dataclass
class DecisionLogReport:
    """决策日志报告。

    Attributes:
        enabled: Feature Flag 是否启用。
        skipped: 是否跳过（Flag 关闭时为 True）。
        entries: 决策日志条目列表。
        markdown: 渲染后的 Markdown 文本（Flag OFF 时为空）。
    """

    enabled: bool = False
    skipped: bool = False
    entries: list[DecisionLogEntry] = field(default_factory=list)
    markdown: str = ""


# =============================================================================
# 8 维决策日志构建（100% Rule-driven）
# =============================================================================
def _build_mechanism_entry(
    knowledge_graph: dict, pathway: str,
) -> DecisionLogEntry:
    """维度 1: Mechanism — 机制分析来源。"""
    nodes = knowledge_graph.get("nodes", []) if isinstance(knowledge_graph, dict) else []
    edges = knowledge_graph.get("edges", []) if isinstance(knowledge_graph, dict) else []
    node_count = len(nodes) if isinstance(nodes, list) else 0
    edge_count = len(edges) if isinstance(edges, list) else 0
    return DecisionLogEntry(
        dimension="Mechanism",
        decision=f"通路 {pathway or 'Unknown'} 机制图构建（{node_count} 节点 / {edge_count} 边）",
        source="Knowledge Graph",
        reason="基于 Agent 提取的机制节点与边构建，非 LLM 创造",
        confidence=-1.0,
        evidence_ref="state.knowledge_graph",
    )


def _build_confidence_entry(confidence: float) -> DecisionLogEntry:
    """维度 2: Confidence — 置信度评估来源。"""
    conf = float(confidence or 0.0)
    label = "HIGH" if conf >= 0.7 else ("MEDIUM" if conf >= 0.4 else "LOW")
    return DecisionLogEntry(
        dimension="Confidence",
        decision=f"整体置信度 {conf:.2f} ({label})",
        source="Rule Engine",
        reason="基于参数来源、仿真收敛性、文献覆盖率的 Rule 计算结果",
        confidence=conf,
        evidence_ref="state.confidence",
    )


def _build_biomodels_entry(
    biomodels_report: dict | None, pathway: str,
) -> DecisionLogEntry:
    """维度 3: BioModels — BioModels 模型匹配来源。"""
    if not biomodels_report or not isinstance(biomodels_report, dict):
        return DecisionLogEntry(
            dimension="BioModels",
            decision="未匹配 BioModels 模型",
            source="N/A",
            reason="本次运行未触发 BioModels 匹配或无匹配结果",
            confidence=0.0,
            evidence_ref="",
        )
    matched = biomodels_report.get("matched_models", [])
    if isinstance(matched, list) and matched:
        model_ids = [m.get("model_id", str(m)) if isinstance(m, dict) else str(m) for m in matched]
        return DecisionLogEntry(
            dimension="BioModels",
            decision=f"匹配 {len(matched)} 个 BioModels 模型: {', '.join(model_ids[:3])}",
            source="BioModels Database",
            reason="基于通路名与物种从 BioModels 数据库检索",
            confidence=0.8,
            evidence_ref=", ".join(model_ids[:3]),
        )
    return DecisionLogEntry(
        dimension="BioModels",
        decision="BioModels 匹配 0 个模型",
        source="BioModels Database",
        reason="检索无结果，参数未来自已发表计算模型",
        confidence=0.2,
        evidence_ref="",
    )


def _build_parameter_entry(
    parameters: dict | None, provenance_summary: dict | None,
) -> DecisionLogEntry:
    """维度 4: Parameter — 参数来源。"""
    if not parameters:
        return DecisionLogEntry(
            dimension="Parameter",
            decision="无参数记录",
            source="N/A",
            reason="state.parameters 为空",
            confidence=0.0,
            evidence_ref="",
        )
    summary = provenance_summary or {}
    total = summary.get("total", 0)
    missing = summary.get("missing_count", 0)
    fallback = summary.get("fallback_count", 0)
    score = summary.get("provenance_score", 0.0)
    by_source = summary.get("by_source", {})

    sources_str = ", ".join(f"{k}={v}" for k, v in by_source.items()) if by_source else "unknown"
    return DecisionLogEntry(
        dimension="Parameter",
        decision=f"共 {total} 个参数（来源: {sources_str}），缺失 {missing}，兜底 {fallback}",
        source="RAG / SBML / Inferred",
        reason=f"参数溯源评分 {score:.2f}（非缺失非兜底占比）",
        confidence=float(score),
        evidence_ref=f"total={total}, missing={missing}, fallback={fallback}",
    )


def _build_discussion_entry(paper_evidence: list) -> DecisionLogEntry:
    """维度 5: Discussion — 讨论文献来源。"""
    if not paper_evidence:
        return DecisionLogEntry(
            dimension="Discussion",
            decision="No literature retrieved",
            source="[D] Inference",
            reason="Retriever 无文献结果，讨论基于通路知识推理",
            confidence=0.3,
            evidence_ref="",
        )
    pmids = []
    for ev in paper_evidence:
        if isinstance(ev, dict):
            pmid = ev.get("pmid") or ev.get("PMID") or ""
            if pmid:
                pmids.append(str(pmid))
    return DecisionLogEntry(
        dimension="Discussion",
        decision=f"基于 {len(paper_evidence)} 篇文献（{len(pmids)} 个 PMID）",
        source="[A] PubMed",
        reason="Discussion 由 EvidenceFuser 按位置匹配渲染，非 LLM 生成",
        confidence=0.7,
        evidence_ref=", ".join(pmids[:5]) if pmids else "no PMIDs",
    )


def _build_experiment_entry(
    experiments: list, pathway: str,
) -> DecisionLogEntry:
    """维度 6: Experiment — 实验推荐来源。"""
    if not experiments:
        return DecisionLogEntry(
            dimension="Experiment",
            decision="无实验推荐",
            source="N/A",
            reason="实验规划未触发或无结果",
            confidence=0.0,
            evidence_ref="",
        )
    count = len(experiments)
    return DecisionLogEntry(
        dimension="Experiment",
        decision=f"推荐 {count} 条实验（通路: {pathway or 'Unknown'}）",
        source="Rule Engine (YAML)" if settings.is_sa_feature_enabled("SPRINT4_EXPERIMENT_RULE_ENGINE") else "Rule Engine (Hardcoded)",
        reason="100% Rule 驱动，Mechanism-aware，无 LLM 创造实验方案",
        confidence=0.9,
        evidence_ref=f"knowledge/experiments/ + {count} experiments",
    )


def _build_validation_entry(
    consistency_passed: bool | None, validation_passed: bool | None,
) -> DecisionLogEntry:
    """维度 7: Validation — 验证规则来源。"""
    parts = []
    if consistency_passed is not None:
        parts.append(f"Consistency={'PASS' if consistency_passed else 'FAIL'}")
    if validation_passed is not None:
        parts.append(f"Validation={'PASS' if validation_passed else 'FAIL'}")
    decision = "; ".join(parts) if parts else "Validation 未触发"

    if consistency_passed is False:
        return DecisionLogEntry(
            dimension="Validation",
            decision=decision,
            source="Rule Engine (Canonical Timeline + Consistency Rules)",
            reason="Consistency 违规 → Hard Gate 阻断后续 SA 阶段",
            confidence=0.1,
            evidence_ref="canonical.yaml consistency_rules",
        )
    return DecisionLogEntry(
        dimension="Validation",
        decision=decision,
        source="Rule Engine (Canonical Timeline + Consistency Rules)",
        reason="基于 Canonical Timeline 与 Consistency Rules 的 100% Rule 验证",
        confidence=0.8 if consistency_passed else 0.5,
        evidence_ref="canonical.yaml canonical_timeline",
    )


def _build_crosstalk_entry(knowledge_graph: dict) -> DecisionLogEntry:
    """维度 8: Cross-talk — 通路串扰来源。"""
    edges = knowledge_graph.get("edges", []) if isinstance(knowledge_graph, dict) else []
    # 检测是否有跨通路边（简单启发式：边含已知通路名交叉）
    crosstalk_keywords = ["EGFR", "MAPK", "PI3K", "AKT", "mTOR", "JAK", "STAT",
                          "TGF", "SMAD", "WNT", "beta-catenin", "p53", "NF-kB",
                          "Apoptosis", "Caspase", "Cyclin", "CDK"]
    edge_sources = set()
    for edge in edges:
        if isinstance(edge, dict):
            src = str(edge.get("source", "") or edge.get("from", ""))
            tgt = str(edge.get("target", "") or edge.get("to", ""))
            for kw in crosstalk_keywords:
                if kw.lower() in src.lower() or kw.lower() in tgt.lower():
                    edge_sources.add(kw)

    if len(edge_sources) >= 2:
        return DecisionLogEntry(
            dimension="Cross-talk",
            decision=f"检测到 {len(edge_sources)} 个通路节点交叉: {', '.join(sorted(edge_sources)[:5])}",
            source="Knowledge Graph",
            reason="基于机制图节点名交叉检测，非 LLM 推测",
            confidence=0.6,
            evidence_ref="state.knowledge_graph edges",
        )
    return DecisionLogEntry(
        dimension="Cross-talk",
        decision="未检测到明显通路串扰",
        source="Knowledge Graph",
        reason="机制图节点名无多通路交叉",
        confidence=0.4,
        evidence_ref="state.knowledge_graph edges",
    )


# =============================================================================
# 主函数
# =============================================================================
def build_decision_log(
    knowledge_graph: dict | None = None,
    pathway: str = "",
    confidence: float = 0.0,
    biomodels_report: dict | None = None,
    parameters: dict | None = None,
    provenance_summary: dict | None = None,
    paper_evidence: list | None = None,
    experiments: list | None = None,
    consistency_passed: bool | None = None,
    validation_passed: bool | None = None,
) -> list[DecisionLogEntry]:
    """构建 8 维决策日志（100% Rule-driven）。

    Args:
        knowledge_graph: state.knowledge_graph。
        pathway: 通路标识。
        confidence: state.confidence。
        biomodels_report: BioModels 匹配报告。
        parameters: state.parameters。
        provenance_summary: 参数溯源汇总。
        paper_evidence: state.paper_evidence。
        experiments: 实验列表。
        consistency_passed: Consistency 检查是否通过。
        validation_passed: Validation 检查是否通过。

    Returns:
        DecisionLogEntry 列表（8 条，对应 8 个维度）。
    """
    kg = knowledge_graph if isinstance(knowledge_graph, dict) else {}
    entries = [
        _build_mechanism_entry(kg, pathway),
        _build_confidence_entry(confidence),
        _build_biomodels_entry(biomodels_report, pathway),
        _build_parameter_entry(parameters, provenance_summary),
        _build_discussion_entry(paper_evidence or []),
        _build_experiment_entry(experiments or [], pathway),
        _build_validation_entry(consistency_passed, validation_passed),
        _build_crosstalk_entry(kg),
    ]
    return entries


def render_decision_log(entries: list[DecisionLogEntry]) -> str:
    """渲染 Scientific Decision Log Markdown。

    Args:
        entries: 决策日志条目列表。

    Returns:
        Markdown 文本。entries 为空时返回提示文本。
    """
    if not entries:
        return "> No decision log available.\n"

    lines = ["## Scientific Decision Log\n"]
    lines.append(
        "> 以下记录各维度科学决策的来源与原因（100% Rule-driven，无 LLM 创造）。\n"
    )
    lines.append(
        "| Dimension | Decision | Source | Reason | Confidence | Evidence Ref |"
    )
    lines.append(
        "|-----------|----------|--------|--------|------------|--------------|"
    )
    for entry in entries:
        conf_str = f"{entry.confidence:.2f}" if entry.confidence >= 0 else "N/A"
        lines.append(
            f"| {entry.dimension} | {entry.decision} | {entry.source} "
            f"| {entry.reason} | {conf_str} | {entry.evidence_ref} |"
        )
    return "\n".join(lines) + "\n"


def generate_decision_log_report(
    knowledge_graph: dict | None = None,
    pathway: str = "",
    confidence: float = 0.0,
    biomodels_report: dict | None = None,
    parameters: dict | None = None,
    provenance_summary: dict | None = None,
    paper_evidence: list | None = None,
    experiments: list | None = None,
    consistency_passed: bool | None = None,
    validation_passed: bool | None = None,
) -> DecisionLogReport:
    """生成完整决策日志报告（Feature Flag 守护）。

    Args:
        同 build_decision_log。

    Returns:
        DecisionLogReport。Flag 关闭时返回 skipped 报告。
    """
    if not settings.is_sa_feature_enabled("SPRINT5_PROVENANCE_EXPLAINABILITY"):
        return DecisionLogReport(
            enabled=False,
            skipped=True,
        )

    entries = build_decision_log(
        knowledge_graph=knowledge_graph,
        pathway=pathway,
        confidence=confidence,
        biomodels_report=biomodels_report,
        parameters=parameters,
        provenance_summary=provenance_summary,
        paper_evidence=paper_evidence,
        experiments=experiments,
        consistency_passed=consistency_passed,
        validation_passed=validation_passed,
    )
    markdown = render_decision_log(entries)

    logger.debug(
        "[Sprint5] Decision Log: %d entries", len(entries),
    )

    return DecisionLogReport(
        enabled=True,
        skipped=False,
        entries=entries,
        markdown=markdown,
    )
