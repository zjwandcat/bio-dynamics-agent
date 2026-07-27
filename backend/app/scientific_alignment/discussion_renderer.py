"""Sprint 2 — Citation-driven Discussion Renderer（Rule 驱动，无 LLM 创造）。

设计原则（ENGINEERING_RULES.md）：
  - LLM 不允许创造科学事实
  - Discussion 仅"组织"已有证据，不"创造"引用
  - 每句话最多含一个来源标签 [A]/[B]/[C]/[D]/[E]
  - 无文献时显式显示 "No literature retrieved"

渲染流程（Citation-driven）：
  Evidence (EvidenceFusionReport)
    → Outline (按来源分组)
      → Sentence (每句一个标签)

铁律：
  - Feature Flag SA_SPRINT2_EVIDENCE_RENDERER=false 时返回 fallback_discussion
  - 违反单源标签约束时抛出 ValueError（不降级，不静默）

Task 10 扩展（Spec: add-scientific-reviewer-and-validation-matrix）：
  新增 Evidence Graph 驱动的 Discussion 渲染接口：
    - DiscussionSection：固定段落结构（5 段）
    - render_discussion_from_evidence_graph：从 EvidenceNode 列表渲染
    - build_discussion_evidence_pool：构造五源证据池
  铁律（Spec Requirement "逐句 Evidence Graph"）：
    - Discussion 必须由 Evidence Graph 渲染，禁止 LLM 自由写
    - 每句附单源标签 [A]PubMed/[B]BioModels/[C]Simulation/[D]Mechanism/[E]Hypothesis
    - 多源节点取 detected_tags[0]（A>B>C>D>E 优先级）
    - 禁止 ungrounded 句子（无 evidence_ref 的节点不得渲染）
    - 若某段无对应类型节点，输出 "[No evidence available for this section]"
  Feature Flag 守护：
    V4_SCIENTIFIC_REVIEWER_ENABLED 默认 false。关闭时本模块新增接口不调用，
    系统回退到 Sprint 2 / LLM 直写路径（由 report_renderer.py 调度）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field as dc_field
from typing import Any, Optional

from app.scientific_alignment.evidence_fuser import (
    EvidenceFusionReport,
    EvidenceSource,
    FusedAssertion,
)
from app.scientific_alignment.evidence_graph import (
    EvidenceNode,
    EvidenceType,
)

# 来源标签 → 中文名映射
_SOURCE_LABELS: dict[EvidenceSource, str] = {
    EvidenceSource.PUBMED: "[A] 文献证据",
    EvidenceSource.BIOMODELS: "[B] BioModels 参考模型",
    EvidenceSource.SIMULATION: "[C] 仿真结果",
    EvidenceSource.INFERENCE: "[D] 机制推理",
    EvidenceSource.HYPOTHESIS: "[E] 假设",
}

# 来源排序优先级（A > B > C > D > E）
_SOURCE_PRIORITY: dict[EvidenceSource, int] = {
    EvidenceSource.PUBMED: 0,
    EvidenceSource.BIOMODELS: 1,
    EvidenceSource.SIMULATION: 2,
    EvidenceSource.INFERENCE: 3,
    EvidenceSource.HYPOTHESIS: 4,
}


class DiscussionRenderError(ValueError):
    """Discussion 渲染错误（违反单源标签约束等）。"""


def render_discussion(
    fusion_report: EvidenceFusionReport,
    fallback_discussion: str = "",
    pathway: str = "",
) -> str:
    """渲染 Citation-driven Discussion。

    Args:
        fusion_report: evidence_fuser.fuse_evidence() 的输出。
        fallback_discussion: Flag OFF 或跳过时的旧 LLM Discussion 文本。
        pathway: 通路名称（用于上下文标注）。

    Returns:
        渲染后的 Discussion Markdown 文本。

    Raises:
        DiscussionRenderError: 当断言违反单源标签约束时。
    """
    # Flag OFF 或跳过 → 走旧 LLM 路径
    if fusion_report.skipped or not fusion_report.enabled:
        return fallback_discussion

    # 无断言 → 显式空
    if not fusion_report.assertions:
        return _render_no_evidence(fusion_report, pathway)

    # 按来源优先级排序断言
    sorted_assertions = sorted(
        fusion_report.assertions,
        key=lambda a: (
            _SOURCE_PRIORITY.get(a.sources[0], 99) if a.sources else 99,
        ),
    )

    # 渲染每句（校验单源标签）
    sentences: list[str] = []
    for assertion in sorted_assertions:
        sentence = _render_assertion_sentence(assertion)
        sentences.append(sentence)

    # 组装 Discussion
    parts: list[str] = []

    # 概述行
    parts.append(_render_overview(fusion_report, pathway))

    # 逐条断言
    parts.append("")
    parts.append("**证据支撑的讨论**：")
    for i, sentence in enumerate(sentences, 1):
        parts.append(f"{i}. {sentence}")

    # 无文献提示
    if fusion_report.source_coverage.get("A", 0) == 0:
        parts.append("")
        parts.append(
            "**No literature retrieved**：本 Discussion 未经 [A] PubMed 文献交叉验证，"
            "相关结论基于 [C] 仿真结果与 [D] 机制推理，建议后续补充文献证据。"
        )

    return "\n".join(parts)


def _render_assertion_sentence(assertion: FusedAssertion) -> str:
    """将单个 FusedAssertion 渲染为一句话（单源标签）。

    铁律：每句仅含一个来源标签 [A]/[B]/[C]/[D]/[E]。
    若断言含多个来源，取最高优先级来源标注，其余来源在括号内注明。

    Raises:
        DiscussionRenderError: 断言无任何来源时。
    """
    if not assertion.sources:
        raise DiscussionRenderError(
            f"断言无来源标签: {assertion.assertion[:50]}...（违反单源标签约束）"
        )

    # 取最高优先级来源作为主标签
    primary_source = min(
        assertion.sources,
        key=lambda s: _SOURCE_PRIORITY.get(s, 99),
    )
    label = primary_source.value  # "A" / "B" / "C" / "D" / "E"

    # 构建引用标识
    refs = _format_references(assertion.evidence_items, primary_source)
    ref_str = f"（{refs}）" if refs else ""

    # 构建句子
    text = assertion.assertion.strip()
    if not text:
        text = "（无断言文本）"

    # 缺陷标注
    defect_note = ""
    if assertion.defect == "evidence_undergrounded":
        defect_note = " ⚠️ 仅 [D] 推理支撑，无外部证据"
    elif assertion.defect == "hypothesis_only":
        defect_note = " ⚠️ 仅 [E] 假设，待验证"

    # 置信度标注（仅当 < 0.5 时显式标注）
    conf_note = ""
    if assertion.fused_confidence < 0.5:
        conf_note = f"（置信度 {assertion.fused_confidence:.2f}）"

    return f"[{label}] {text}{ref_str}{conf_note}{defect_note}"


def _format_references(
    evidence_items: list, primary_source: EvidenceSource
) -> str:
    """格式化引用标识（PMID / BIOMD ID 等）。"""
    refs: list[str] = []
    for item in evidence_items:
        if item.source != primary_source:
            continue
        ref = item.reference.strip()
        if ref and ref not in refs:
            refs.append(ref)
    return "; ".join(refs)


def _render_overview(
    fusion_report: EvidenceFusionReport, pathway: str
) -> str:
    """渲染 Discussion 概述行。"""
    total = fusion_report.total_assertions
    coverage = fusion_report.source_coverage
    undergrounded = fusion_report.undergrounded_count
    hypothesis_only = fusion_report.hypothesis_only_count

    parts: list[str] = []
    if pathway:
        parts.append(f"通路：{pathway}；")
    parts.append(f"共 {total} 条断言；")
    parts.append(
        f"来源覆盖：[A]{coverage.get('A', 0)}/"
        f"[B]{coverage.get('B', 0)}/"
        f"[C]{coverage.get('C', 0)}/"
        f"[D]{coverage.get('D', 0)}/"
        f"[E]{coverage.get('E', 0)}"
    )
    if undergrounded > 0:
        parts.append(f"；⚠️ {undergrounded} 条仅 [D] 推理")
    if hypothesis_only > 0:
        parts.append(f"；⚠️ {hypothesis_only} 条仅 [E] 假设")

    return "".join(parts)


def _render_no_evidence(
    fusion_report: EvidenceFusionReport, pathway: str
) -> str:
    """无任何断言时的 Discussion 渲染。"""
    parts: list[str] = []
    parts.append("**No literature retrieved**")
    parts.append("")
    if pathway:
        parts.append(f"通路 {pathway} 的 Discussion 未经 [A] PubMed 文献交叉验证。")
    else:
        parts.append("本 Discussion 未经 [A] PubMed 文献交叉验证。")
    parts.append(
        "相关结论基于 [D] 通路知识推理与 [C] 仿真结果，"
        "建议后续通过 PubMed 检索补充文献证据。"
    )
    return "\n".join(parts)


def render_evidence_bundle_sse_payload(
    fusion_report: EvidenceFusionReport,
) -> dict[str, Any]:
    """将 EvidenceFusionReport 序列化为 SSE 事件载荷。

    用于 main.py 下发 sa_evidence_bundle 事件给前端。
    """
    if fusion_report.skipped:
        return {
            "enabled": False,
            "skipped": True,
            "reason": "SA_SPRINT2_EVIDENCE_RENDERER disabled",
        }

    return {
        "enabled": fusion_report.enabled,
        "skipped": False,
        "total_assertions": fusion_report.total_assertions,
        "undergrounded_count": fusion_report.undergrounded_count,
        "hypothesis_only_count": fusion_report.hypothesis_only_count,
        "source_coverage": fusion_report.source_coverage,
        "assertions": [
            {
                "assertion": a.assertion,
                "sources": [s.value for s in a.sources],
                "fused_confidence": round(a.fused_confidence, 3),
                "undergrounded": a.undergrounded,
                "hypothesis_only": a.hypothesis_only,
                "defect": a.defect,
                "evidence_items": [
                    {
                        "source": item.source.value,
                        "reference": item.reference,
                        "snippet": item.snippet,
                        "confidence": round(item.confidence, 3),
                    }
                    for item in a.evidence_items
                ],
            }
            for a in fusion_report.assertions
        ],
    }


# =============================================================================
# 缺口 3：表格/标题行分离逻辑（单一可信源）
# =============================================================================
# 评估器 C12 / Discussion 渲染器共用此函数，避免重复实现跳过逻辑。
# 设计原则：prose 用于 [A]-[E] 标签覆盖率检查（C12），non_prose 不参与句子计数。
_NON_PROSE_PREFIXES: tuple[str, ...] = ("#", "|", "```", ">")
# 纯列表标记行：`---` / `***` / `1.` / `-` / `*`（无实际内容）
_PURE_LIST_MARKER_RE = re.compile(r"^\s*(?:[-*]+|\d+\.)\s*$")


def split_discussion_prose_vs_tables(
    discussion_text: str,
) -> tuple[str, list[str]]:
    """分离 Discussion 文本中的叙述句与表格/标题/代码块/引用行。

    用于 C12 [A]-[E] 标签覆盖率检查：prose 部分每句必须有标签，
    non_prose（表格/标题/代码块/引用/纯列表标记/空行）不参与句子计数。

    分离规则（行级）：
      - 以 `#` / `|` / ` ``` ` / `>` 开头 → non_prose
      - 空行（或仅空白） → non_prose
      - 纯列表标记行（`---` / `***` / `1.` / `-` / `*`，无内容） → non_prose
      - 其余行 → prose

    Args:
        discussion_text: 完整 Discussion 文本（Markdown）。

    Returns:
        (prose_text, non_prose_lines) 元组：
          - prose_text: prose 行按原始顺序以 `\n` 连接的字符串（无尾随换行）。
          - non_prose_lines: non_prose 行列表（已 strip，保留原始顺序）。
    """
    if not discussion_text:
        return ("", [])

    prose_lines: list[str] = []
    non_prose_lines: list[str] = []

    for raw_line in discussion_text.splitlines():
        line = raw_line.strip()
        # 空行 → non_prose
        if not line:
            non_prose_lines.append(line)
            continue
        # 表格/标题/代码块/引用 → non_prose
        if line.startswith(_NON_PROSE_PREFIXES):
            non_prose_lines.append(line)
            continue
        # 纯列表标记行（如 `---` / `***` / `1.`） → non_prose
        if _PURE_LIST_MARKER_RE.match(line):
            non_prose_lines.append(line)
            continue
        # 其余 → prose
        prose_lines.append(line)

    prose_text = "\n".join(prose_lines)
    return (prose_text, non_prose_lines)


# =============================================================================
# Task 10 — Evidence Graph 驱动的 Discussion Renderer
# （Spec: add-scientific-reviewer-and-validation-matrix）
# =============================================================================
# 设计原则（PEP 20: 显式优于隐式）：
#   - Discussion SHALL 从 Evidence Graph 渲染，SHALL NOT 让 LLM 自由写
#   - 每句 SHALL 附单源标签，多源节点取 detected_tags[0]（A>B>C>D>E 优先级）
#   - 禁止 ungrounded 句子（无 evidence_ref 的节点不得渲染）
#   - 5 个固定段落（标题固定），无证据的段落输出占位文本
# =============================================================================

# 5 个固定段落标题（对应 Spec Task 10）
_SECTION_QUESTION_COVERAGE: str = "Question Coverage"
_SECTION_MECHANISM_INTERPRETATION: str = "Mechanism Interpretation"
_SECTION_DYNAMICS_ANALYSIS: str = "Dynamics Analysis"
_SECTION_BIOMODELS_COMPARISON: str = "BioModels Comparison"
_SECTION_LIMITATIONS_HYPOTHESES: str = "Scientific Limitations & Hypotheses"

# 段落无证据时的占位文本
_NO_EVIDENCE_PLACEHOLDER: str = "[No evidence available for this section]"


@dataclass
class DiscussionSection:
    """Discussion 段落结构（对应 Spec Task 10 的 5 个固定段落之一）。

    每个段落持有若干 EvidenceNode（来自 Evidence Graph），并负责将其渲染为
    带 [A]/[B]/[C]/[D]/[E] 单源标签的 Markdown 文本。

    Attributes:
        title: 段落标题（如 "Mechanism Interpretation"，固定 5 选 1）。
        sentences: 该段所有 EvidenceNode（已按主类型筛选，可能含 ungrounded
            节点——to_markdown 会自动过滤）。
        canonical_reference: 可选的 Canonical Mechanism 引用字符串（仅
            Mechanism Interpretation 段使用，渲染为 "**Canonical Mechanism**: ..."）。
    """

    title: str
    sentences: list[EvidenceNode] = dc_field(default_factory=list)
    canonical_reference: Optional[str] = None

    def to_markdown(self) -> str:
        """渲染段落为 Markdown 文本。

        渲染规则（Spec Task 10）：
          1. 段落以 "## <title>" 开头
          2. 若设置 canonical_reference，紧接一行 "**Canonical Mechanism**: <ref>"
          3. 每个非 ungrounded 节点渲染为 "<text> [X]Type:ref" 一行
          4. 多源节点取 detected_tags[0] 作为单源标签
          5. ungrounded 节点（evidence_type=ungrounded 或 evidence_ref 为空）跳过
          6. 若渲染后无任何句子，输出占位文本 "[No evidence available for this section]"

        Returns:
            段落 Markdown 文本（含标题、可选 canonical 引用、句子列表或占位文本）。
        """
        lines: list[str] = [f"## {self.title}", ""]

        # 可选：渲染 Canonical Mechanism 引用（仅 Mechanism Interpretation 段使用）
        if self.canonical_reference:
            lines.append(f"**Canonical Mechanism**: {self.canonical_reference}")
            lines.append("")

        rendered_sentences: list[str] = []
        for node in self.sentences:
            # 过滤 ungrounded 节点（无 evidence_ref 不得渲染）
            if node.evidence_type == EvidenceType.UNGROUNDED:
                continue
            if not node.evidence_ref:
                continue
            label = _format_single_source_label(node)
            if not label:
                # 无有效标签（理论不应发生，因为已过滤 ungrounded），防御性跳过
                continue
            rendered_sentences.append(f"{node.text} {label}")

        if rendered_sentences:
            lines.extend(rendered_sentences)
        else:
            lines.append(_NO_EVIDENCE_PLACEHOLDER)

        return "\n".join(lines)


def _format_single_source_label(node: EvidenceNode) -> str:
    """格式化节点的单源标签：返回 "[X]Type" 或 "[X]Type:ref"。

    多源节点取 detected_tags[0]（按 A>B>C>D>E 优先级，由 evidence_graph
    保证）；若 detected_tags 为空（理论不应发生，因为已过滤 ungrounded），
    退回到 evidence_type + evidence_ref。

    Args:
        node: EvidenceNode 实例。

    Returns:
        单源标签字符串（如 "[A]PubMed:12345678"），无有效类型时返回空字符串。
    """
    if node.detected_tags:
        tag_type, tag_ref = node.detected_tags[0]
    else:
        tag_type = node.evidence_type
        tag_ref = node.evidence_ref or ""

    # 校验 tag_type 为五源之一（提取字母标签 [A]-[E]）
    letter = _extract_letter(tag_type)
    if not letter:
        return ""

    if tag_ref:
        return f"{tag_type}:{tag_ref}"
    return tag_type


def _extract_letter(type_str: str) -> Optional[str]:
    """从证据类型字符串提取单字母标签。

    输入 "[A]PubMed" → 返回 "A"；输入 "ungrounded" → 返回 None。

    Args:
        type_str: EvidenceType 常量字符串（如 "[A]PubMed" / "ungrounded"）。

    Returns:
        单字母标签 "A"/"B"/"C"/"D"/"E"，或 None（非五源类型）。
    """
    if not type_str or len(type_str) < 3:
        return None
    if type_str[0] == "[" and type_str[2] == "]":
        letter = type_str[1]
        if letter in ("A", "B", "C", "D", "E"):
            return letter
    return None


def _extract_canonical_required_nodes(canonical_mechanism: Any) -> list[str]:
    """从 canonical_mechanism 提取 required_nodes 列表（兼容 dict / 对象）。

    canonical_mechanism 在 state 中可能以两种形式存在：
      - dict：{"required_nodes": ["EGFR", "Grb2", ...]}
      - CanonicalMechanism 对象：含 .required_nodes 属性

    Args:
        canonical_mechanism: canonical_mechanism 字段（dict 或对象）。

    Returns:
        required_nodes 字符串列表；无该字段时返回空列表。
    """
    if canonical_mechanism is None:
        return []
    # 对象形式：优先 .required_nodes 属性
    if hasattr(canonical_mechanism, "required_nodes"):
        try:
            return [str(n) for n in canonical_mechanism.required_nodes if n]
        except Exception:
            return []
    # dict 形式
    if isinstance(canonical_mechanism, dict):
        nodes = canonical_mechanism.get("required_nodes") or []
        return [str(n) for n in nodes if n]
    return []


def render_discussion_from_evidence_graph(
    question: str,
    evidence_graph: list,
    simulation_metrics: Optional[dict] = None,
    biomodels_diff: Optional[dict] = None,
    canonical_mechanism: Optional[dict] = None,
) -> str:
    """从 Evidence Graph 渲染 Discussion（非 LLM 直写）。

    对应 Spec Task 10 / Requirement "逐句 Evidence Graph"。

    渲染规则：
      1. Discussion 必须包含 5 个固定段落（每段标题固定）：
         - "Question Coverage" — 用 [D]Mechanism 节点回答用户问题
         - "Mechanism Interpretation" — 用 [A]PubMed + [B]BioModels + [D]Mechanism
           节点解释机制（文献支撑机制解释，符合 Spec "禁止 LLM 自由写" 铁律：
           机制断言必须有文献或 BioModels 支撑）
         - "Dynamics Analysis" — 用 [C]Simulation 节点描述动力学（含数值）
         - "BioModels Comparison" — 用 [B]BioModels 节点对比仿真与 BioModels 差异
         - "Scientific Limitations & Hypotheses" — 用 [E]Hypothesis 节点列出局限与假设
      2. 每句附单源标签（取 EvidenceNode.evidence_type）；多源节点取 detected_tags[0]
      3. 禁止 LLM 自由创作；所有句子必须来自 evidence_graph 节点
      4. 若 evidence_graph 中无某段所需类型的节点，该段输出占位文本
      5. 禁止 ungrounded 句子（无 evidence_ref 的节点不得渲染）

    Args:
        question: 用户原始问题（用于 Question Coverage 段上下文，当前未直接渲染
            到输出，保留参数以供未来 LLM 辅助 Question Coverage 段使用）。
        evidence_graph: list[EvidenceNode]，由 evidence_graph.build_from_report
            构造或外部传入；也兼容 EvidenceGraph 对象（含 .nodes 属性）。
        simulation_metrics: 仿真指标（用于 [C]Simulation 标签的事实来源，可选）。
            当前未直接渲染，保留参数以供 Dynamics Analysis 段补充数值上下文。
        biomodels_diff: BioModels 对照差异（[B] 标签的事实来源，可选）。
            当前未直接渲染，保留参数以供 BioModels Comparison 段补充差异上下文。
        canonical_mechanism: 来自 canonical_yaml 的 canonical_mechanism（含
            required_nodes），渲染到 Mechanism Interpretation 段头部作为引用。

    Returns:
        Discussion Markdown 文本，含 5 个固定段落，每句附单源标签。
    """
    # 兼容 EvidenceGraph 对象（含 .nodes）与 list[EvidenceNode] 两种输入
    if evidence_graph is None:
        nodes: list[EvidenceNode] = []
    elif hasattr(evidence_graph, "nodes"):
        nodes = list(evidence_graph.nodes)
    else:
        nodes = list(evidence_graph or [])

    # 过滤 ungrounded 节点（无 evidence_ref 不得渲染，全段通用）
    # Spec 铁律：禁止 ungrounded 句子
    grounded_nodes: list[EvidenceNode] = [
        n for n in nodes
        if n.evidence_type != EvidenceType.UNGROUNDED and n.evidence_ref
    ]

    # 按段落类型筛选节点（一个节点可能出现在多个段中，如 [B] 同时出现在
    # Mechanism Interpretation 与 BioModels Comparison）
    section_question = DiscussionSection(
        title=_SECTION_QUESTION_COVERAGE,
        sentences=[
            n for n in grounded_nodes
            if n.evidence_type == EvidenceType.MECHANISM
        ],
    )
    section_mechanism = DiscussionSection(
        title=_SECTION_MECHANISM_INTERPRETATION,
        sentences=[
            n for n in grounded_nodes
            if n.evidence_type in (
                EvidenceType.PUBMED,      # [A] 文献支撑机制解释
                EvidenceType.BIOMODELS,   # [B] BioModels 参考模型
                EvidenceType.MECHANISM,   # [D] 机制断言
            )
        ],
    )
    section_dynamics = DiscussionSection(
        title=_SECTION_DYNAMICS_ANALYSIS,
        sentences=[
            n for n in grounded_nodes
            if n.evidence_type == EvidenceType.SIMULATION
        ],
    )
    section_biomodels = DiscussionSection(
        title=_SECTION_BIOMODELS_COMPARISON,
        sentences=[
            n for n in grounded_nodes
            if n.evidence_type == EvidenceType.BIOMODELS
        ],
    )
    section_hypotheses = DiscussionSection(
        title=_SECTION_LIMITATIONS_HYPOTHESES,
        sentences=[
            n for n in grounded_nodes
            if n.evidence_type == EvidenceType.HYPOTHESIS
        ],
    )

    # 渲染 Canonical Mechanism 引用（若有），仅注入 Mechanism Interpretation 段
    required_nodes = _extract_canonical_required_nodes(canonical_mechanism)
    if required_nodes:
        section_mechanism.canonical_reference = " → ".join(required_nodes)

    # 组装最终 Markdown（5 段顺序固定）
    parts: list[str] = []
    for section in (
        section_question,
        section_mechanism,
        section_dynamics,
        section_biomodels,
        section_hypotheses,
    ):
        parts.append(section.to_markdown())
        parts.append("")  # 段落间空行分隔

    return "\n".join(parts).rstrip() + "\n"


def build_discussion_evidence_pool(
    retrieved_papers: list[dict],
    biomodels_matches: list[dict],
    simulation_metrics: dict,
    mechanism_graph: dict,
    hypotheses: list[str],
) -> list[dict]:
    """构造 Evidence Pool（供 evidence_graph.build_from_report 使用）。

    将 BioDynamics state 中的五源原始数据归一化为 list[dict] 格式，每个 dict
    含 source_type / source_ref / content / confidence 四字段。

    五源映射：
      - [A] PubMed       — retrieved_papers（每篇含 pmid / title）
      - [B] BioModels     — biomodels_matches（每个含 biomd_id / name）
      - [C] Simulation    — simulation_metrics（每个数值指标一项）
      - [D] Mechanism     — mechanism_graph.nodes（每个节点名一项）
      - [E] Hypothesis    — hypotheses（每个假设一项）

    Args:
        retrieved_papers: 检索到的 PubMed 论文列表（list[dict]，含 pmid / title）。
        biomodels_matches: BioModels 匹配列表（list[dict]，含 biomd_id 或 id / name）。
        simulation_metrics: 仿真指标字典（如 {"pERK_peak_time": 16.1}）。
        mechanism_graph: 机制图（network_json / knowledge_graph，含 nodes 列表）。
        hypotheses: 假设列表（list[str] 或 list[dict]，dict 含 id / text）。

    Returns:
        list of dict，每个 dict 含：
          - source_type: "A" / "B" / "C" / "D" / "E"
          - source_ref: PMID / BIOMD / metric_name=value / mechanism_node / hypothesis_id
          - content: 该证据的文本描述
          - confidence: 0.0-1.0
    """
    pool: list[dict] = []

    # [A] PubMed 文献证据
    for paper in retrieved_papers or []:
        if not isinstance(paper, dict):
            continue
        pmid = str(paper.get("pmid", "")).strip()
        if not pmid:
            # 无 PMID 的论文跳过（铁律：不创造引用）
            continue
        title = str(paper.get("title", "") or "").strip()
        pool.append({
            "source_type": "A",
            "source_ref": f"PMID:{pmid}",
            "content": title or f"PubMed PMID:{pmid}",
            "confidence": float(paper.get("confidence", 0.8)),
        })

    # [B] BioModels 参考模型
    for bm in biomodels_matches or []:
        if not isinstance(bm, dict):
            continue
        biomd_id = str(bm.get("biomd_id") or bm.get("id") or "").strip()
        if not biomd_id:
            continue
        name = str(bm.get("name") or bm.get("title") or "").strip()
        pool.append({
            "source_type": "B",
            "source_ref": biomd_id,
            "content": name or f"BioModels {biomd_id}",
            "confidence": float(bm.get("confidence", 0.75)),
        })

    # [C] Simulation 仿真结果（每个数值指标一项）
    for metric_name, value in (simulation_metrics or {}).items():
        # 跳过 bool（True/False 不是数值指标）
        if isinstance(value, bool):
            continue
        if not isinstance(value, (int, float)):
            continue
        pool.append({
            "source_type": "C",
            "source_ref": f"{metric_name}={value}",
            "content": f"{metric_name} = {value}",
            "confidence": 0.7,
        })

    # [D] Mechanism 机制节点（从 mechanism_graph 提取节点名）
    mechanism_nodes = _extract_mechanism_node_names(mechanism_graph)
    for node_name in mechanism_nodes:
        pool.append({
            "source_type": "D",
            "source_ref": str(node_name),
            "content": f"Mechanism node: {node_name}",
            "confidence": 0.3,
        })

    # [E] Hypothesis 假设
    for idx, hyp in enumerate(hypotheses or [], start=1):
        if isinstance(hyp, dict):
            hyp_id = str(hyp.get("id") or f"H{idx}")
            hyp_text = str(
                hyp.get("text") or hyp.get("statement") or hyp.get("content") or ""
            ).strip()
        else:
            hyp_id = f"H{idx}"
            hyp_text = str(hyp).strip()
        if not hyp_text:
            continue
        pool.append({
            "source_type": "E",
            "source_ref": hyp_id,
            "content": hyp_text,
            "confidence": 0.2,
        })

    return pool


def _extract_mechanism_node_names(mechanism_graph: Any) -> list[str]:
    """从机制图提取节点名列表（兼容 network_json / knowledge_graph 格式）。

    支持的输入格式：
      - {"nodes": [{"id": "EGFR"}, {"name": "DUSP"}, ...], "edges": [...]}
      - {"nodes": ["EGFR", "DUSP", ...], ...}
      - 任何含 nodes 字段的 dict

    Args:
        mechanism_graph: 机制图 dict（network_json / knowledge_graph）。

    Returns:
        节点名列表（str），无 nodes 字段时返回空列表。
    """
    if not isinstance(mechanism_graph, dict):
        return []
    nodes_field = mechanism_graph.get("nodes")
    if not nodes_field:
        return []
    names: list[str] = []
    for n in nodes_field:
        if isinstance(n, dict):
            # 优先 id，其次 name / label
            name = n.get("id") or n.get("name") or n.get("label")
            if name:
                names.append(str(name))
        elif isinstance(n, str):
            names.append(n)
        # 其他类型跳过
    return names


# Task 10 新增导出
__all___task10 = [
    "DiscussionSection",
    "render_discussion_from_evidence_graph",
    "build_discussion_evidence_pool",
    "split_discussion_prose_vs_tables",
]
