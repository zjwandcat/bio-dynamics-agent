# Feature Flag: V4_SCIENTIFIC_REVIEWER_ENABLED
#
# BioDynamics Agent - Scientific Alignment Loop: 逐句 Evidence Graph
# （Spec: add-scientific-reviewer-and-validation-matrix）
#
# 模块用途：
#   为 Discussion 建立 Evidence Graph，每句话必须挂载至少一个证据节点。无证据
#   节点的句子 SHALL 标记 Hallucination Risk: High。Discussion 必须由 Evidence
#   Graph 渲染生成，禁止 LLM 自由写。
#
# 对应 Spec Requirement：
#   - 逐句 Evidence Graph
#
# Evidence Graph 节点结构：
#   - sentence_id    : 句子唯一 ID（如 "S1" / "S2"）
#   - text           : 句子文本
#   - evidence_type  : [A]PubMed / [B]BioModels / [C]Simulation / [D]Mechanism /
#                      [E]Hypothesis / ungrounded（按 A>B>C>D>E 优先级取首项）
#   - evidence_ref   : PMID / BIOMD ID / simulation_metric（无则 None）
#   - confidence     : 该证据节点置信度（0.0-1.0）
#   - hallucination_risk: High / Low（无 evidence_ref → High）
#
# 逐句解析规则：
#   - 按 `.` / 换行切分 report 为句子
#   - 每句扫描 PMID / BIOMD / 仿真数字 / 机制节点词
#   - 若含 PMID → evidence_type="[A]PubMed"
#   - 若含 BIOMD → evidence_type="[B]BioModels"
#   - 若含仿真数字 → evidence_type="[C]Simulation"
#   - 若含机制节点词 → evidence_type="[D]Mechanism"（无 [A][B][C] 支撑则
#     标记 evidence_undergrounded）
#   - 其余 → evidence_type="ungrounded", hallucination_risk="High"
#
# 逐句渲染规则：
#   - 每句 SHALL 附单源标签，如 "ERK shows transient activation [C]Simulation: peak 16.1 min."
#   - 一句检测到多源时，按 A>B>C>D>E 顺序追加多个标签
#   - 全部 [D]Mechanism 而无 [A][B][C] 支撑的断言 SHALL 标记 evidence_undergrounded
#   - 无 evidence_ref 的节点 SHALL 标记 evidence_type=ungrounded + Hallucination Risk=High
#
# 禁止 LLM 自由写 Discussion：
#   Discussion SHALL 从 Evidence Graph 渲染（discussion_renderer.py），SHALL NOT
#   让 LLM 直接生成 Discussion 文本（项目硬约束：LLM 仅组织解释，不创造科学事实）。
#
# Feature Flag 守护：
#   V4_SCIENTIFIC_REVIEWER_ENABLED 默认 false。关闭时 Evidence Graph 不构建。
#
# 核心导出：
#   from app.scientific_alignment.evidence_graph import (
#       EvidenceType, EvidenceNode, EvidenceGraph,
#       build_from_report, detect_ungrounded, detect_undergrounded,
#       render_with_labels,
#   )

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# =============================================================================
# 常量定义（对应 Spec Evidence Graph 节点结构）
# =============================================================================


class EvidenceType:
    """证据类型五源 + ungrounded（对应 Spec Evidence Graph 节点 evidence_type）。

    五源标签用于 Discussion 逐句渲染：
        [A] PubMed       — 已发表文献
        [B] BioModels    — EBI BioModels 数据库
        [C] Simulation   — 仿真结果（附曲线指标）
        [D] Mechanism    — 机制断言（需 [A][B][C] 支撑，否则 evidence_undergrounded）
        [E] Hypothesis   — 待验证假设
    """

    PUBMED: str = "[A]PubMed"          # [A] PubMed
    BIOMODELS: str = "[B]BioModels"    # [B] BioModels
    SIMULATION: str = "[C]Simulation"  # [C] Simulation
    MECHANISM: str = "[D]Mechanism"    # [D] Mechanism
    HYPOTHESIS: str = "[E]Hypothesis"  # [E] Hypothesis
    UNGROUNDED: str = "ungrounded"     # 无证据（Hallucination Risk=High）


# 各证据类型置信度默认值（无外部 bonus 时的基线）
_CONFIDENCE_DEFAULTS: dict[str, float] = {
    EvidenceType.PUBMED: 0.8,
    EvidenceType.BIOMODELS: 0.75,
    EvidenceType.SIMULATION: 0.7,
    EvidenceType.MECHANISM: 0.3,
    EvidenceType.HYPOTHESIS: 0.2,
    EvidenceType.UNGROUNDED: 0.0,
}

# 证据类型优先级（A > B > C > D > E），用于多源检测时选取主类型
_TYPE_PRIORITY: dict[str, int] = {
    EvidenceType.PUBMED: 0,
    EvidenceType.BIOMODELS: 1,
    EvidenceType.SIMULATION: 2,
    EvidenceType.MECHANISM: 3,
    EvidenceType.HYPOTHESIS: 4,
}

# 需 [A][B][C] 支撑的"外部证据"类型集合
_EXTERNAL_TYPES: frozenset[str] = frozenset({
    EvidenceType.PUBMED,
    EvidenceType.BIOMODELS,
    EvidenceType.SIMULATION,
})

# 句子切分正则：按句末标点（.!?）后接空白或换行切分；保留缩写中的点（如 16.1）
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")

# PMID 检测正则：PMID: 12345678 / PMID 12345678 / PMID:12345678
_PMID_RE = re.compile(r"PMID\s*:?\s*(\d+)")

# BIOMD 检测正则：BIOMD0000000010
_BIOMD_RE = re.compile(r"(BIOMD\d+)")

# 仿真数字检测正则：peak 16.1 min / 16.1 min / peak at 16.1 min
_SIM_TIME_RE = re.compile(
    r"(?:peak\s+(?:at\s+)?|decline\s+(?:at\s+)?|rise\s+(?:at\s+)?)?"
    r"(\d+\.?\d*)\s*min",
    re.IGNORECASE,
)


# =============================================================================
# 数据结构
# =============================================================================


@dataclass
class EvidenceNode:
    """Evidence Graph 单个节点（对应一句 Discussion 文本）。

    对应 Spec Scenario "Evidence Graph 节点结构"。

    Attributes:
        sentence_id: 句子唯一 ID（如 "S1" / "S2"）。
        text: 句子文本。
        evidence_type: 主证据类型（按 A>B>C>D>E 优先级取首项，无证据则 ungrounded）。
        evidence_ref: 主证据引用（PMID 数字 / BIOMD ID / simulation_metric 字符串），
            无证据时为 None。
        confidence: 该证据节点置信度（0.0-1.0），取主证据类型默认值。
        hallucination_risk: Hallucination 风险（High / Low），无 evidence_ref → High。
        detected_tags: 该句检测到的全部 (类型, 引用) 对（按优先级排序），供
            render_with_labels 多源标签渲染使用。主 (evidence_type, evidence_ref)
            始终为 detected_tags 的首项（ungrounded 时为空列表）。
    """

    sentence_id: str
    text: str
    evidence_type: str = EvidenceType.UNGROUNDED
    evidence_ref: str | None = None
    confidence: float = 0.0
    hallucination_risk: str = "High"
    detected_tags: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class EvidenceGraph:
    """Evidence Graph 完整图（对应一段 Discussion）。

    Attributes:
        nodes: EvidenceNode 列表（顺序与 Discussion 句子顺序一致）
        ungrounded_count: ungrounded 节点数（Hallucination Risk=High）
        undergrounded_count: 纯 [D]Mechanism 无 [A][B][C] 支撑的节点数
    """

    nodes: list[EvidenceNode] = field(default_factory=list)
    ungrounded_count: int = 0
    undergrounded_count: int = 0


# =============================================================================
# 辅助函数：句子切分与证据检测
# =============================================================================


def _split_sentences(report_md: str) -> list[str]:
    """按 `.` / 换行切分 report 为句子（过滤空白与无意义残片）。

    Args:
        report_md: Report Markdown 文本。

    Returns:
        非空句子列表（保留句末标点），顺序与原文一致。
    """
    if not report_md:
        return []
    # 先按句末标点 + 空白 或 换行 切分
    raw_parts = _SENTENCE_SPLIT_RE.split(report_md.strip())
    sentences: list[str] = []
    for part in raw_parts:
        cleaned = (part or "").strip()
        if not cleaned:
            continue
        sentences.append(cleaned)
    return sentences


def _detect_pubmid(text: str) -> str | None:
    """检测句子中的 PMID 引用。

    Returns:
        首个匹配的 PMID 数字字符串（无前缀），未匹配返回 None。
    """
    match = _PMID_RE.search(text)
    if match:
        return match.group(1)
    return None


def _detect_biomd(text: str) -> str | None:
    """检测句子中的 BIOMD ID 引用。

    Returns:
        首个匹配的 BIOMD ID 字符串（含 BIOMD 前缀），未匹配返回 None。
    """
    match = _BIOMD_RE.search(text)
    if match:
        return match.group(1)
    return None


def _detect_simulation_metric(
    text: str,
    simulation_metrics: dict[str, Any],
) -> str | None:
    """检测句子中的仿真数字引用。

    匹配规则（按优先级）：
      1. evidence_pool.simulation_metrics 中数值字面量出现在句子中
         （如 "16.1" 出现且 pool 含 pERK_peak_time=16.1）→ 返回 "metric=value"
      2. 句子含 "<数字> min" 仿真时间模式（如 "peak 16.1 min"）→ 返回 "time=<数字> min"

    Args:
        text: 句子文本。
        simulation_metrics: 仿真指标字典（来自 evidence_pool）。

    Returns:
        仿真引用字符串（如 "pERK_peak_time=16.1" 或 "time=16.1 min"），未匹配返回 None。
    """
    # 规则 1：pool 中数值字面量出现在句子中
    for metric_name, value in simulation_metrics.items():
        if isinstance(value, bool):
            # bool 不是数字，跳过（避免 True/False 误匹配）
            continue
        if isinstance(value, (int, float)):
            value_str = str(value)
            # 词边界检查，避免部分匹配（如 16.1 不应匹配 16.10）
            if re.search(rf"(?<!\d){re.escape(value_str)}(?!\d)", text):
                return f"{metric_name}={value_str}"

    # 规则 2：句子含 "<数字> min" 模式
    time_match = _SIM_TIME_RE.search(text)
    if time_match:
        number = time_match.group(1)
        return f"time={number} min"

    return None


def _detect_mechanism_node(
    text: str,
    mechanism_nodes: list[str],
) -> str | None:
    """检测句子中的机制节点词。

    使用词边界匹配避免短词误匹配（如 "ERK" 不应匹配 "ERK5"）。

    Args:
        text: 句子文本。
        mechanism_nodes: 机制节点词列表（来自 evidence_pool）。

    Returns:
        首个匹配的机制节点词，未匹配返回 None。
    """
    text_lower = text.lower()
    for node in mechanism_nodes:
        node_str = str(node).strip()
        if not node_str:
            continue
        # 词边界匹配（大小写不敏感）
        pattern = rf"\b{re.escape(node_str)}\b"
        if re.search(pattern, text_lower, re.IGNORECASE):
            return node_str
    return None


def _detect_all_tags(
    text: str,
    evidence_pool: dict[str, Any],
) -> list[tuple[str, str]]:
    """对单句检测全部证据标签（按 A>B>C>D>E 优先级排序）。

    Args:
        text: 句子文本。
        evidence_pool: 证据池，含 pmids / biomodels / simulation_metrics /
            mechanism_nodes。

    Returns:
        (类型, 引用) 元组列表，按类型优先级排序；引用为空字符串表示该类型
        无具体引用标识（仅 mechanism 可能出现）。
    """
    detected: list[tuple[str, str]] = []

    pmids = evidence_pool.get("pmids") or []
    biomodels = evidence_pool.get("biomodels") or []
    simulation_metrics = evidence_pool.get("simulation_metrics") or {}
    mechanism_nodes = evidence_pool.get("mechanism_nodes") or []

    # [A] PubMed
    pubmid_ref = _detect_pubmid(text)
    if pubmid_ref is not None:
        detected.append((EvidenceType.PUBMED, pubmid_ref))

    # [B] BioModels
    biomd_ref = _detect_biomd(text)
    if biomd_ref is not None:
        detected.append((EvidenceType.BIOMODELS, biomd_ref))

    # [C] Simulation
    sim_ref = _detect_simulation_metric(text, simulation_metrics)
    if sim_ref is not None:
        detected.append((EvidenceType.SIMULATION, sim_ref))

    # [D] Mechanism（仅当未检测到 [A][B][C] 时才检测，避免重复标记）
    # Spec 要求：纯 [D]Mechanism 无 [A][B][C] 支撑 → evidence_undergrounded
    # 此处仅检测机制词；是否 undergrounded 由 detect_undergrounded 判定
    if not detected:
        mech_ref = _detect_mechanism_node(text, mechanism_nodes)
        if mech_ref is not None:
            detected.append((EvidenceType.MECHANISM, mech_ref))

    return detected


# =============================================================================
# 核心函数
# =============================================================================


def build_from_report(
    report_md: str,
    evidence_pool: dict[str, Any],
) -> EvidenceGraph:
    """从 Report Markdown 与证据池构建 Evidence Graph。

    对应 Spec Requirement "逐句 Evidence Graph"。

    Args:
        report_md: Report Markdown 文本（含 Discussion 段落）。
        evidence_pool: 证据池，含五源证据用于匹配句子，示例：
            {
                "pmids": ["12345678", "23456789"],
                "biomodels": ["BIOMD0000000010"],
                "simulation_metrics": {"pERK_peak_time": 16.1, "erk_decline": True},
                "mechanism_nodes": ["EGFR", "Ras", "ERK", "DUSP"],
            }

    Returns:
        EvidenceGraph 实例，每个句子对应一个 EvidenceNode。ungrounded_count 与
        undergrounded_count 在构建时同步统计。
    """
    evidence_pool = evidence_pool or {}
    sentences = _split_sentences(report_md)
    nodes: list[EvidenceNode] = []
    ungrounded_count = 0
    undergrounded_count = 0

    for idx, sentence_text in enumerate(sentences, start=1):
        sentence_id = f"S{idx}"
        detected_tags = _detect_all_tags(sentence_text, evidence_pool)

        if not detected_tags:
            # 无证据 → ungrounded + Hallucination Risk=High
            node = EvidenceNode(
                sentence_id=sentence_id,
                text=sentence_text,
                evidence_type=EvidenceType.UNGROUNDED,
                evidence_ref=None,
                confidence=_CONFIDENCE_DEFAULTS[EvidenceType.UNGROUNDED],
                hallucination_risk="High",
                detected_tags=[],
            )
            ungrounded_count += 1
        else:
            # 取优先级最高的类型作为主类型
            primary_type, primary_ref = detected_tags[0]
            confidence = _CONFIDENCE_DEFAULTS.get(primary_type, 0.0)
            node = EvidenceNode(
                sentence_id=sentence_id,
                text=sentence_text,
                evidence_type=primary_type,
                evidence_ref=primary_ref,
                confidence=confidence,
                hallucination_risk="Low",
                detected_tags=list(detected_tags),
            )

            # undergrounded 判定：主类型为 [D]Mechanism 且无 [A][B][C] 支撑
            detected_types = {tag for tag, _ in detected_tags}
            if (
                primary_type == EvidenceType.MECHANISM
                and not (detected_types & _EXTERNAL_TYPES)
            ):
                undergrounded_count += 1

        nodes.append(node)

    return EvidenceGraph(
        nodes=nodes,
        ungrounded_count=ungrounded_count,
        undergrounded_count=undergrounded_count,
    )


def detect_ungrounded(graph: EvidenceGraph) -> list[EvidenceNode]:
    """检测 Evidence Graph 中的 ungrounded 节点（无任何证据）。

    对应 Spec Scenario "逐句渲染"：
        - 无 evidence_ref 的节点 SHALL 标记 evidence_type=ungrounded
        - 无证据节点 SHALL 标记 Hallucination Risk=High

    Args:
        graph: EvidenceGraph 实例。

    Returns:
        evidence_type=ungrounded 的节点列表（Hallucination Risk=High）。
    """
    return [
        node for node in graph.nodes
        if node.evidence_type == EvidenceType.UNGROUNDED
    ]


def detect_undergrounded(graph: EvidenceGraph) -> list[EvidenceNode]:
    """检测 Evidence Graph 中的 undergrounded 节点（纯 [D]Mechanism 无 [A][B][C] 支撑）。

    对应 Spec Scenario "逐句渲染"：
        - 全部 [D]Mechanism 而无 [A][B][C] 支撑的断言 SHALL 标记
          evidence_undergrounded

    判定规则：
        - 节点检测到的标签集合 ⊆ { [D]Mechanism }（即只有 [D] 而无任何外部证据）

    Args:
        graph: EvidenceGraph 实例。

    Returns:
        undergrounded 节点列表（仅含 [D]Mechanism 而无 [A][B][C] 支撑）。
    """
    undergrounded: list[EvidenceNode] = []
    for node in graph.nodes:
        if not node.detected_tags:
            # ungrounded 节点不算 undergrounded（无任何证据，包括 [D]）
            continue
        detected_types = {tag for tag, _ in node.detected_tags}
        # 仅当检测到的类型全为 [D]Mechanism（无 [A][B][C]）时判定为 undergrounded
        if detected_types and not (detected_types & _EXTERNAL_TYPES):
            if EvidenceType.MECHANISM in detected_types:
                undergrounded.append(node)
    return undergrounded


def render_with_labels(graph: EvidenceGraph) -> str:
    """渲染 Evidence Graph 为带单源标签的 Discussion 文本。

    对应 Spec Scenario "逐句渲染"：
        - 每句 SHALL 附单源标签，如
          "ERK shows transient activation [C]Simulation: peak 16.1 min."
        - 一句检测到多源时，按 A>B>C>D>E 顺序追加多个标签
        - ungrounded 句子附 [ungrounded] 标签

    渲染格式：
        S1: <sentence text> [A]PubMed:12345678 [C]Simulation:pERK_peak_time=16.1
        S2: <sentence text> [D]Mechanism:DUSP
        S3: <sentence text> [ungrounded]

    Args:
        graph: EvidenceGraph 实例。

    Returns:
        渲染后的 Discussion 文本（每句一行，含 sentence_id + 原文 + 标签）。
    """
    lines: list[str] = []
    for node in graph.nodes:
        if not node.detected_tags:
            # ungrounded 节点：附 [ungrounded] 标签
            label_part = "[ungrounded]"
        else:
            # 多源标签：按 detected_tags 顺序（已按 A>B>C>D>E 排序）拼接
            tag_parts: list[str] = []
            for tag_type, tag_ref in node.detected_tags:
                if tag_ref:
                    tag_parts.append(f"{tag_type}:{tag_ref}")
                else:
                    tag_parts.append(tag_type)
            label_part = " ".join(tag_parts)
        lines.append(f"{node.sentence_id}: {node.text} {label_part}".rstrip())

    return "\n".join(lines)


__all__ = [
    "EvidenceType",
    "EvidenceNode",
    "EvidenceGraph",
    "build_from_report",
    "detect_ungrounded",
    "detect_undergrounded",
    "render_with_labels",
]
