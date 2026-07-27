# BioDynamics Agent v4 - Scientific Alignment Loop: Scientific Discussion Prompt (Task 10)
#
# Scientific Discussion 优化：强制 LLM 在 Discussion 中回答 10 个核心科学问题，
# 每个回答必须标注 Evidence 来源 [A]/[B]/[C]/[D]/[E]，
# 禁止"与经典文献一致"等模板措辞（必须有 PMID + 量化对比）。
#
# 设计背景：
#   当前 Discussion 生成依赖 LLM 自由发挥，常出现模板化措辞、缺乏量化对比、
#   无证据引用等问题。Task 10 通过强制 10 问 + Evidence 标注 + 模板检测器，
#   提升 Discussion 的科学严谨性。
#
# SubTask 10.1：强制必答 10 问
# SubTask 10.2：每个回答必引 Evidence（[A]PubMed / [B]BioModels / [C]Simulation /
#               [D]Inference / [E]Hypothesis）
# SubTask 10.3：Discussion 模板检测器（含"与经典文献一致"但无 PMID 与量化对比
#               → 标记 discussion_template_violation，violation_type="template_phrase"）
#
# Feature Flag 守护：
#   SA_SEVEN_AXIS 默认 OFF（Discussion 优化属于 7 轴 Validation 的 Discussion 轴输入）。
#   关闭时 get_scientific_discussion_prompt 返回空字符串（不覆盖 v3 prompt），
#   check_discussion 返回 skipped=True 的空报告。
#   铁律：V4_SCIENTIFIC_ALIGNMENT_ENABLED=false 时，子 Flag 永远不生效
#         （由 settings.is_sa_feature_enabled 强制校验）。
#
# 依赖：
#   - app.config.settings（Feature Flag 守护）
#
# 核心导出：
#   from app.scientific_alignment.discussion_prompt import (
#       DiscussionViolation, DiscussionCheckReport,
#       get_scientific_discussion_prompt, check_discussion,
#       REQUIRED_QUESTIONS, TEMPLATE_VIOLATION_PATTERNS,
#   )

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from app.config import settings

logger = logging.getLogger(__name__)


# =============================================================================
# 常量
# =============================================================================

# 必答 10 问（固定顺序，question_index 1-10 对应此列表索引 0-9）
REQUIRED_QUESTIONS: list[str] = [
    "Why does the peak occur at this time point?",
    "Why does the signal decline?",
    "Why doesn't the signal persistently increase under sustained stimulation?",
    "What is the negative feedback regulator?",
    "Which experiments support this mechanism?",
    "What are the similarities and differences with literature (quantified)?",
    "What are the model limitations?",
    "What is the parameter sensitivity?",
    "What are the next experiments?",
    "How does it compare with classical computational models (BioModels)?",
]

# 模板违规短语：出现这些短语但同段无 PMID 和数值 → template_phrase 违规
TEMPLATE_VIOLATION_PATTERNS: list[str] = [
    "与经典文献一致",
    "consistent with classical literature",
    "与已知机制相符",
    "in line with known mechanisms",
    "广泛报道",
    "widely reported",
]

# Evidence 来源标签（五源证据引用）
_EVIDENCE_TAGS: tuple[str, ...] = ("[A]", "[B]", "[C]", "[D]", "[E]")

# Evidence 标签正则（匹配 [A] / [B] / [C] / [D] / [E]）
_EVIDENCE_TAG_RE = re.compile(r"\[([ABCDE])\]")

# PMID 正则（匹配 PMID:123456 或 PMID: 123456，不区分大小写）
_PMID_RE = re.compile(r"PMID[:\s]*\d+", re.IGNORECASE)

# 数值正则（匹配整数或小数，用于量化对比检测，如 15、14.8、0.2）
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")

# 每个问题的关键词匹配规则：列表 of 规则，每条规则为关键词元组（AND 关系）
# 任一规则全部命中即视为该问题被回答（OR of ANDs）
# 关键词均按小写匹配
_QUESTION_KEYWORD_RULES: list[list[tuple[str, ...]]] = [
    # Q1: "peak" + ("time" / "occurs" / "because")
    [("peak", "time"), ("peak", "occurs"), ("peak", "because")],
    # Q2: "decline" / "decrease" / "down" / "feedback"
    [("decline",), ("decrease",), ("down",), ("feedback",)],
    # Q3: "persist" / "sustain" / "transient" / "not continue"
    [("persist",), ("sustain",), ("transient",), ("not continue",)],
    # Q4: "feedback" / "DUSP" / "SPRY" / "negative"
    [("feedback",), ("dusp",), ("spry",), ("negative",)],
    # Q5: "experiment" / "WB" / "qPCR" / "inhibitor"
    [("experiment",), ("wb",), ("qpcr",), ("inhibitor",)],
    # Q6: "literature" / "compare" / "PMID" / "consistent"
    [("literature",), ("compare",), ("pmid",), ("consistent",)],
    # Q7: "limitation" / "caveat" / "assumption"
    [("limitation",), ("caveat",), ("assumption",)],
    # Q8: "sensitivity" / "parameter" / "robust"
    [("sensitivity",), ("parameter",), ("robust",)],
    # Q9: "next" / "future" / "suggest" / "validate"
    [("next",), ("future",), ("suggest",), ("validate",)],
    # Q10: "BioModels" / "BIOMD" / ("model" + "compare")
    [("biomodels",), ("biomd",), ("model", "compare")],
]

# 通过阈值
_COVERAGE_PASS_THRESHOLD: float = 0.8
_EVIDENCE_PASS_THRESHOLD: float = 0.6


# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class DiscussionViolation:
    """单条 Discussion 违规记录。"""

    question_index: int          # 1-10
    question: str
    violation_type: str          # "missing" / "no_evidence" / "template_phrase" / "no_quantification"
    detail: str


@dataclass
class DiscussionCheckReport:
    """Discussion 检测报告。"""

    enabled: bool
    skipped: bool = False
    questions_covered: list[bool] = field(default_factory=list)  # 长度 10
    coverage_rate: float = 0.0     # 0-1
    evidence_citation_rate: float = 0.0  # 有 [A][B][C][D][E] 标注的回答占比
    violations: list[DiscussionViolation] = field(default_factory=list)
    passed: bool = False           # coverage>=0.8 且 evidence_rate>=0.6


# =============================================================================
# Prompt 生成
# =============================================================================

def get_scientific_discussion_prompt(
    pathway: str, simulation_metrics: dict | None = None
) -> str:
    """返回 SA 增强版 Discussion system prompt（含 10 问强制要求）。

    Flag OFF 时返回空字符串（表示不覆盖 v3 prompt）。

    Args:
        pathway: 通路名称（如 "EGFR"）。
        simulation_metrics: 仿真指标字典（可选，用于在 prompt 中注入量化上下文）。

    Returns:
        SA 增强版 Discussion system prompt；Flag OFF 时返回 ""。
    """
    # Feature Flag 守护：SA_SEVEN_AXIS 关闭时不覆盖 v3 prompt
    if not settings.is_sa_feature_enabled("SEVEN_AXIS"):
        return ""

    # 构造仿真指标上下文（如有）
    metrics_hint = ""
    if simulation_metrics:
        try:
            key_metrics = []
            for key in ("peak_time", "peak_value", "steady_state", "decline_rate"):
                if key in simulation_metrics:
                    val = simulation_metrics[key]
                    key_metrics.append(f"  - {key}: {val}")
            if key_metrics:
                metrics_hint = (
                    "\n\n仿真指标参考（必须在 Discussion 中量化对比）：\n"
                    + "\n".join(key_metrics)
                )
        except Exception:
            logger.debug("构造仿真指标上下文失败，忽略", exc_info=True)

    # 构造 10 问列表（带编号）
    questions_block = "\n".join(
        f"  Q{i + 1}. {q}" for i, q in enumerate(REQUIRED_QUESTIONS)
    )

    # 构造模板违规禁令
    forbidden_phrases = "\n".join(
        f"  - \"{p}\"" for p in TEMPLATE_VIOLATION_PATTERNS
    )

    prompt = f"""You are the Scientific Discussion Author for pathway: {pathway}.

# REQUIRED: 10 Mandatory Questions
You MUST answer ALL 10 questions below. Each answer must be a dedicated paragraph labeled "Q1" ... "Q10".
{questions_block}{metrics_hint}

# REQUIRED: Evidence Citation
Every answer MUST cite at least one Evidence source using the bracket tags:
  [A] PubMed literature (include PMID:xxxxxx)
  [B] BioModels reference (include BIOMDxxxxxx)
  [C] Simulation result (from this run)
  [D] Inference (clearly state the reasoning chain)
  [E] Hypothesis (clearly label as testable hypothesis)

Answers WITHOUT an evidence tag will be flagged as "no_evidence" violations.

# FORBIDDEN: Template Phrases
The following phrases are FORBIDDEN because they signal template/boilerplate text:
{forbidden_phrases}

If you reference literature, you MUST:
  1. Include a concrete PMID (e.g., PMID:12451180)
  2. Provide a QUANTIFIED comparison (e.g., "ERK Peak 15min vs BioModels 14.8min, diff +0.2min")

# REQUIRED: Quantified Comparison
For Q6 (literature comparison) and Q10 (BioModels comparison), you MUST provide
numeric quantification. Example format:
  "ERK Peak 15min vs PMID:10712587 reported 14min, diff +1min [A]"
  "ERK Peak 15min vs BioModels BIOMD0000000010 14.8min, diff +0.2min [B]"

# Output Format
Produce exactly 10 labeled paragraphs (Q1-Q10). Do not add filler sentences.
Each paragraph must be self-contained and cite its evidence source.
"""
    return prompt


# =============================================================================
# Discussion 检测器
# =============================================================================

def _split_segments(content: str) -> list[str]:
    """将 Discussion 内容按句/段分割。

    分割策略：
    1. 先按换行分割（段落优先）
    2. 再按句号 + 空白分割（句子级）

    保留含小数的数字（如 14.8min）不被截断：仅在句号后跟空白或位于行尾时分割。
    """
    if not content:
        return []
    # 按换行分割
    raw_lines = re.split(r"\n+", content)
    segments: list[str] = []
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        # 按句号 + 空白分割（避免截断小数如 0.2min）
        # 正则：句号后跟至少一个空白字符，或句号位于行尾
        sentences = re.split(r"\.\s+|\.$", line)
        for sent in sentences:
            sent = sent.strip()
            if sent:
                segments.append(sent)
    return segments


def _segment_matches_question(
    segment_lower: str, rules: list[tuple[str, ...]]
) -> bool:
    """检测段落是否匹配某问题的关键词规则（OR of ANDs）。

    Args:
        segment_lower: 已转小写的段落文本。
        rules: 关键词规则列表，每条规则为关键词元组（AND 关系）。
            任一规则全部命中即视为匹配。

    Returns:
        任一规则全部命中 → True；否则 False。
    """
    for rule in rules:
        if all(kw in segment_lower for kw in rule):
            return True
    return False


def _segment_has_evidence(segment: str) -> bool:
    """检测段落是否包含 Evidence 标签 [A]/[B]/[C]/[D]/[E]。"""
    return bool(_EVIDENCE_TAG_RE.search(segment))


def _segment_has_pmid(segment: str) -> bool:
    """检测段落是否包含 PMID 引用。"""
    return bool(_PMID_RE.search(segment))


def _segment_has_quantification(segment: str) -> bool:
    """检测段落是否包含量化数值（至少一个数字）。"""
    return bool(_NUMBER_RE.search(segment))


def _segment_template_phrases(segment_lower: str) -> list[str]:
    """检测段落是否包含模板违规短语，返回命中的短语列表（原始大小写）。"""
    return [
        p for p in TEMPLATE_VIOLATION_PATTERNS
        if p.lower() in segment_lower
    ]


def check_discussion(discussion_content: str) -> DiscussionCheckReport:
    """检测 Discussion 是否覆盖 10 问且每问有 Evidence 标注。

    Flag OFF 时返回 skipped=True 的报告。
    空内容时 coverage_rate=0，passed=False。

    Args:
        discussion_content: Discussion 文本内容。

    Returns:
        DiscussionCheckReport 检测报告。
    """
    # Feature Flag 守护
    enabled = settings.is_sa_feature_enabled("SEVEN_AXIS")
    if not enabled:
        return DiscussionCheckReport(
            enabled=False,
            skipped=True,
            questions_covered=[False] * 10,
            coverage_rate=0.0,
            evidence_citation_rate=0.0,
            violations=[],
            passed=False,
        )

    # 空内容防御
    if not discussion_content or not discussion_content.strip():
        violations = [
            DiscussionViolation(
                question_index=i + 1,
                question=REQUIRED_QUESTIONS[i],
                violation_type="missing",
                detail="Discussion 内容为空，该问题未被回答",
            )
            for i in range(10)
        ]
        return DiscussionCheckReport(
            enabled=True,
            skipped=False,
            questions_covered=[False] * 10,
            coverage_rate=0.0,
            evidence_citation_rate=0.0,
            violations=violations,
            passed=False,
        )

    # 分割段落
    segments = _split_segments(discussion_content)
    segments_lower = [s.lower() for s in segments]

    questions_covered: list[bool] = []
    violations: list[DiscussionViolation] = []
    answered_count = 0
    answered_with_evidence = 0

    for q_idx in range(10):
        question = REQUIRED_QUESTIONS[q_idx]
        rules = _QUESTION_KEYWORD_RULES[q_idx]

        # 找到匹配该问题的段落索引
        matching_segments: list[int] = []
        for seg_idx, seg_lower in enumerate(segments_lower):
            if _segment_matches_question(seg_lower, rules):
                matching_segments.append(seg_idx)

        if not matching_segments:
            # 问题未被回答
            questions_covered.append(False)
            violations.append(DiscussionViolation(
                question_index=q_idx + 1,
                question=question,
                violation_type="missing",
                detail=f"未检测到对 Q{q_idx + 1} 的回答（关键词未命中）",
            ))
            continue

        # 问题被回答
        questions_covered.append(True)
        answered_count += 1

        # 检查是否有 Evidence 标签（在任一匹配段落中）
        has_evidence = any(
            _segment_has_evidence(segments[seg_idx])
            for seg_idx in matching_segments
        )
        if has_evidence:
            answered_with_evidence += 1
        else:
            violations.append(DiscussionViolation(
                question_index=q_idx + 1,
                question=question,
                violation_type="no_evidence",
                detail=(
                    f"Q{q_idx + 1} 已回答但未标注 Evidence 来源 "
                    f"[A]/[B]/[C]/[D]/[E]"
                ),
            ))

        # 检查模板违规短语（在匹配段落中）
        for seg_idx in matching_segments:
            seg_lower = segments_lower[seg_idx]
            seg_orig = segments[seg_idx]
            matched_phrases = _segment_template_phrases(seg_lower)
            if matched_phrases:
                has_pmid = _segment_has_pmid(seg_orig)
                has_num = _segment_has_quantification(seg_orig)
                # 含模板短语但同段无 PMID 或无数值 → template_phrase 违规
                if not has_pmid or not has_num:
                    violations.append(DiscussionViolation(
                        question_index=q_idx + 1,
                        question=question,
                        violation_type="template_phrase",
                        detail=(
                            f"Q{q_idx + 1} 段落含模板短语 {matched_phrases} "
                            f"但缺少 PMID 或量化数值"
                        ),
                    ))

        # Q6 特别检查：文献对比必须有量化数值
        if q_idx == 5:
            has_num = any(
                _segment_has_quantification(segments[seg_idx])
                for seg_idx in matching_segments
            )
            if not has_num:
                violations.append(DiscussionViolation(
                    question_index=q_idx + 1,
                    question=question,
                    violation_type="no_quantification",
                    detail="Q6 文献对比要求量化数值，但未检测到数字",
                ))

    # 计算覆盖率与证据引用率
    coverage_rate = answered_count / 10.0
    evidence_citation_rate = (
        answered_with_evidence / answered_count if answered_count > 0 else 0.0
    )

    passed = (
        coverage_rate >= _COVERAGE_PASS_THRESHOLD
        and evidence_citation_rate >= _EVIDENCE_PASS_THRESHOLD
    )

    return DiscussionCheckReport(
        enabled=True,
        skipped=False,
        questions_covered=questions_covered,
        coverage_rate=coverage_rate,
        evidence_citation_rate=evidence_citation_rate,
        violations=violations,
        passed=passed,
    )
