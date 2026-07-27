# Feature Flag: V4_SCIENTIFIC_REVIEWER_ENABLED
#
# BioDynamics Agent - Scientific Alignment Loop: Scientific Honesty 四类标签
# （Spec: add-scientific-reviewer-and-validation-matrix）
#
# 模块用途：
#   强制区分四类科学声明：Prediction / Hypothesis / Simulation / Literature。
#   禁止"模型已经证明"等过度声明。
#
# 对应 Spec Requirement：
#   - Scientific Honesty 四类标签
#
# 四类标签定义：
#   - Prediction  — 基于机制的预测（尚未仿真）
#                   关键词：predict / prediction / would / expected to
#   - Hypothesis  — 待验证的假设
#                   关键词：hypothesis / hypothesize / speculate / conjecture /
#                           possibly / might / could
#   - Simulation  — 仿真结果（附曲线指标）
#                   关键词：simulation / simulate / simulated / model shows /
#                           model predicts（须附 metric）
#   - Literature  — 已发表文献结论（附 PMID）
#                   必须含 PMID 引用
#
# 三类违规检测：
#   1. honesty_violation=overstatement     — "模型已经证明"/"simulation proves"/
#                                             "definitely" 等过度声明措辞
#   2. honesty_violation=citation_missing  — [Literature] 标签缺 PMID
#   3. honesty_violation=unlabeled_claim   — 未标注任何四类标签的声明
#
# 过度声明替换规则：
#   - "模型已经证明"  → "模型提示"
#   - "simulation proves" → "Simulation suggests"
#   - "definitely" / "absolutely" → 删除或替换为谦逊措辞
#   - "Hypothesis: ..." 保留
#
# 与 Validation Matrix 关系：
#   Scientific Writing Validation 轴（第 10 轴）SHALL 调用本模块检测；任一违规
#   → Scientific Writing 轴 FAIL；[Literature] 缺 PMID → Evidence Attribution 轴 FAIL。
#
# Feature Flag 守护：
#   V4_SCIENTIFIC_REVIEWER_ENABLED 默认 false。关闭时 Honesty 检测不执行。
#
# 核心导出：
#   from app.scientific_alignment.scientific_honesty import (
#       ClaimLabel, HonestyReview, HonestyLabel, HonestyViolationType,
#       classify_claim, detect_overstatement, detect_unlabeled_claim,
#       check_citation_missing, review_report,
#       # 兼容旧 API
#       HonestyViolation, HonestyReport,
#       check_scientific_honesty, replace_overclaim,
#   )

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable


# =============================================================================
# 常量定义（对应 Spec 四类标签定义）
# =============================================================================


class HonestyLabel:
    """四类科学声明标签（对应 Spec Scenario "四类标签定义"）。

    同时暴露 UNLABELED 用于"未标注"状态。
    """

    PREDICTION: str = "Prediction"    # 基于机制的预测（尚未仿真）
    HYPOTHESIS: str = "Hypothesis"    # 待验证的假设
    SIMULATION: str = "Simulation"    # 仿真结果（附曲线指标）
    LITERATURE: str = "Literature"    # 已发表文献结论（附 PMID）
    UNLABELED: str = "unlabeled"      # 未标注任何四类标签


class HonestyViolationType:
    """三类 Honesty 违规（对应 Spec Scenario "禁止过度声明" 等）。

    注：OVERCLAIM 与 OVERSTATEMENT 等价，均映射到 "overstatement"。
    """

    OVERSTATEMENT: str = "overstatement"
    CITATION_MISSING: str = "citation_missing"
    UNLABELED_CLAIM: str = "unlabeled_claim"
    # 兼容旧名（保留为同义字符串，不破坏既有调用方）
    OVERCLAIM: str = "overstatement"


# -----------------------------------------------------------------------------
# 四类标签关键词库（对应 Spec "四类标签定义"）
# -----------------------------------------------------------------------------

# Prediction 关键词（基于机制的预测，尚未仿真）
PREDICTION_KEYWORDS: tuple[str, ...] = (
    "predict",
    "prediction",
    "would",
    "expected to",
)

# Hypothesis 关键词（待验证的假设）
HYPOTHESIS_KEYWORDS: tuple[str, ...] = (
    "hypothesis",
    "hypothesize",
    "hypothesise",
    "speculate",
    "conjecture",
    "possibly",
    "might",
    "could",
)

# Simulation 关键词（仿真结果，须附曲线指标）
SIMULATION_KEYWORDS: tuple[str, ...] = (
    "simulation",
    "simulate",
    "simulated",
    "model shows",
    "model predicts",
    "model suggests",
    "the model",
    "in silico",
)

# Literature 显式标签/关键词（用于 citation_missing 判定；分类仍以 PMID 为准）
LITERATURE_TAG_KEYWORDS: tuple[str, ...] = (
    "literature",
    "[literature]",
)

# -----------------------------------------------------------------------------
# 过度声明措辞库（中英文，对应 Spec Scenario "禁止过度声明"）
# -----------------------------------------------------------------------------

# 英文过度声明措辞
OVERSTATEMENT_PATTERNS_EN: tuple[str, ...] = (
    "proves",
    "definitely",
    "conclusively",
    "model proves",
    "simulation proves",
    "definitive",
    "certainly",
    "undoubtedly",
)

# 中文过度声明措辞
OVERSTATEMENT_PATTERNS_ZH: tuple[str, ...] = (
    "模型已经证明",
    "仿真证明",
    "确定",
    "毫无疑问",
    "完全证实",
)

# 合并所有过度声明措辞（保留旧名 OVERCLAIM_PATTERNS 为别名）
OVERSTATEMENT_PATTERNS: tuple[str, ...] = (
    OVERSTATEMENT_PATTERNS_EN + OVERSTATEMENT_PATTERNS_ZH
)
OVERCLAIM_PATTERNS: tuple[str, ...] = OVERSTATEMENT_PATTERNS

# 过度声明 → 谦逊措辞替换映射（建议替换，对应 Spec Scenario "禁止过度声明"）
OVERSTATEMENT_REPLACEMENTS: dict[str, str] = {
    "模型已经证明": "模型提示",
    "仿真证明": "Simulation suggests",
    "simulation proves": "Simulation suggests",
    "model proves": "Simulation suggests",
    "proves": "suggests",
    "definitely": "",
    "conclusively": "",
    "definitive": "",
    "certainly": "",
    "undoubtedly": "",
    "确定": "",
    "毫无疑问": "",
    "完全证实": "提示",
}
# 兼容旧名
OVERCLAIM_REPLACEMENTS: dict[str, str] = OVERSTATEMENT_REPLACEMENTS


# =============================================================================
# 数据结构
# =============================================================================


@dataclass
class ClaimLabel:
    """单条声明的分类标签。

    对应 Spec "四类标签定义" 的分类输出。

    Attributes:
        label: Prediction / Hypothesis / Simulation / Literature / unlabeled
        evidence: 命中的关键词或 PMID（如 "predict" / "PMID:12345678"）
        confidence: 置信度 0.0-1.0
    """

    label: str
    evidence: str
    confidence: float


@dataclass
class HonestyViolation:
    """单条 Honesty 违规记录（兼容旧 API）。

    Attributes:
        violation_type: HonestyViolationType 之一
        sentence: 违规所在的句子文本
        label: 相关标签（如 "Literature"），unlabeled_claim 时为空
        reason: 违规原因（如 "[Literature] 标签缺 PMID"）
    """

    violation_type: str
    sentence: str
    label: str = ""
    reason: str = ""


@dataclass
class HonestyReview:
    """Scientific Honesty 完整审查报告。

    对应 Spec Requirement "Scientific Honesty 四类标签" 的整份报告审查输出。

    Attributes:
        status: PASS / PARTIAL / FAIL
        violations: 违规类型列表（如 ["overstatement", "citation_missing"]）
        details: 每条违规的 {sentence, violation_type, reason}
    """

    status: str
    violations: list[str] = field(default_factory=list)
    details: list[dict] = field(default_factory=list)


@dataclass
class HonestyReport:
    """Scientific Honesty 完整审查报告（兼容旧 API）。

    Attributes:
        violations: 违规列表
        status: PASS / FAIL
        labeled_claims: 已标注声明数
        unlabeled_claims: 未标注声明数
    """

    violations: list[HonestyViolation] = field(default_factory=list)
    status: str = "PASS"
    labeled_claims: int = 0
    unlabeled_claims: int = 0


# =============================================================================
# 正则表达式
# =============================================================================

# PMID 引用：PMID: 12345678 / PMID:12345678 / PMID 12345678 / pmid: 12345678
# 至少 4 位数字以避免误匹配普通数字。
_PMID_PATTERN: re.Pattern[str] = re.compile(
    r"PMID[:\s]*(\d{4,})", re.IGNORECASE
)

# 曲线指标（数字 + 单位）：16.1 min / 20 nM / 0.5 / 45.8 min / 2 fold / 80 %
_METRIC_PATTERN: re.Pattern[str] = re.compile(
    r"\d+\.?\d*\s*(?:min|nM|uM|μM|mM|M|%|a\.u\.|fold|hr|hour)",
    re.IGNORECASE,
)

# 句子分隔符：英文句号/感叹号/问号 + 空白，或换行；兼容中文标点。
_SENTENCE_SPLIT_PATTERN: re.Pattern[str] = re.compile(
    r"(?<=[.!?。！？])\s+|\n+"
)


# =============================================================================
# 内部辅助函数
# =============================================================================


def _find_first(text_lower: str, keywords: Iterable[str]) -> str | None:
    """在已小写化的文本中查找第一个命中的关键词。

    Args:
        text_lower: 已转小写的文本
        keywords: 候选关键词列表

    Returns:
        命中的原始关键词（保持原大小写形式）或 None
    """
    for kw in keywords:
        if kw.lower() in text_lower:
            return kw
    return None


def _split_sentences(text: str) -> list[str]:
    """将文本切分为句子列表（保留非空句子）。

    Args:
        text: 原始文本

    Returns:
        非空句子列表（已 strip）
    """
    if not text:
        return []
    parts = _SENTENCE_SPLIT_PATTERN.split(text)
    return [p.strip() for p in parts if p and p.strip()]


# =============================================================================
# 核心接口（对应任务说明 5 个公开函数）
# =============================================================================


def classify_claim(text: str) -> ClaimLabel:
    """分类单条声明为四类标签之一或 unlabeled。

    优先级顺序（与 Spec "四类标签定义" 对齐）：
        1. Literature  — 含 PMID 引用（最强信号）
        2. Simulation  — 含仿真关键词（含曲线指标时置信度更高）
        3. Prediction  — 含预测关键词
        4. Hypothesis  — 含假设关键词
        5. unlabeled   — 无任何标签线索

    Args:
        text: 单条声明文本

    Returns:
        ClaimLabel 实例，label/evidence/confidence 已填充
    """
    if not text or not text.strip():
        return ClaimLabel(
            label=HonestyLabel.UNLABELED, evidence="", confidence=1.0
        )

    text_lower = text.lower()

    # 1. Literature — 含 PMID 引用
    pmid_match = _PMID_PATTERN.search(text)
    if pmid_match:
        pmid = pmid_match.group(1)
        return ClaimLabel(
            label=HonestyLabel.LITERATURE,
            evidence=f"PMID:{pmid}",
            confidence=0.95,
        )

    # 2. Simulation — 含仿真关键词（含数字指标时置信度更高）
    sim_kw = _find_first(text_lower, SIMULATION_KEYWORDS)
    if sim_kw:
        has_metric = bool(_METRIC_PATTERN.search(text))
        confidence = 0.9 if has_metric else 0.7
        return ClaimLabel(
            label=HonestyLabel.SIMULATION,
            evidence=sim_kw,
            confidence=confidence,
        )

    # 3. Prediction — 含预测关键词
    pred_kw = _find_first(text_lower, PREDICTION_KEYWORDS)
    if pred_kw:
        return ClaimLabel(
            label=HonestyLabel.PREDICTION,
            evidence=pred_kw,
            confidence=0.85,
        )

    # 4. Hypothesis — 含假设关键词
    hyp_kw = _find_first(text_lower, HYPOTHESIS_KEYWORDS)
    if hyp_kw:
        return ClaimLabel(
            label=HonestyLabel.HYPOTHESIS,
            evidence=hyp_kw,
            confidence=0.8,
        )

    # 5. unlabeled — 无任何标签线索
    return ClaimLabel(label=HonestyLabel.UNLABELED, evidence="", confidence=1.0)


def detect_overstatement(text: str) -> bool:
    """检测过度声明措辞。

    检查文本是否包含中英文过度声明措辞，如 "proves" / "definitely" /
    "模型已经证明" / "仿真证明" / "毫无疑问" 等。

    Args:
        text: 待检测文本

    Returns:
        True 表示命中过度声明措辞，False 表示未命中
    """
    if not text:
        return False
    text_lower = text.lower()
    for pattern in OVERSTATEMENT_PATTERNS:
        if pattern.lower() in text_lower:
            return True
    return False


def detect_unlabeled_claim(text: str, evidence_pool: dict | None = None) -> bool:
    """检测未标注声明（无任何四类标签线索）。

    对应 Spec Scenario "四类标签定义" — 未标注的声明 SHALL 标记
    `honesty_violation=unlabeled_claim`。

    Args:
        text: 待检测文本
        evidence_pool: 预留参数，未来可用于校验 evidence 是否在证据池中；
                       本实现仅做标签线索检测，不依赖此参数

    Returns:
        True 表示该声明未标注任何四类标签，False 表示已标注
    """
    if not text or not text.strip():
        return False
    label = classify_claim(text)
    return label.label == HonestyLabel.UNLABELED


def check_citation_missing(text: str) -> bool:
    """检测 [Literature] 标签或 Literature 关键词缺 PMID。

    对应 Spec Scenario "Literature 标签必须附 PMID"。

    判定规则：
        - 若文本含 [Literature] 显式标签或 "literature" 关键词，但无 PMID 引用
          → 返回 True（缺引用）
        - 若文本含 PMID → 返回 False
        - 若文本不含 Literature 标签线索 → 返回 False（不视为 citation 缺失）

    Args:
        text: 待检测文本

    Returns:
        True 表示存在 [Literature] 标签但缺 PMID
    """
    if not text:
        return False
    text_lower = text.lower()
    has_literature_tag = any(
        kw.lower() in text_lower for kw in LITERATURE_TAG_KEYWORDS
    )
    has_pmid = bool(_PMID_PATTERN.search(text))
    return has_literature_tag and not has_pmid


def review_report(report_md: str) -> HonestyReview:
    """审查整份 Report，输出 HonestyReview。

    对应 Spec Requirement "Scientific Honesty 四类标签"。

    检测三类违规：
        1. overstatement     — 过度声明措辞
        2. citation_missing  — [Literature] 缺 PMID
        3. unlabeled_claim   — 未标注的声明

    Status 判定：
        - 无违规 → PASS
        - 仅有 unlabeled_claim → PARTIAL
        - 有 overstatement 或 citation_missing → FAIL

    Args:
        report_md: Report Markdown 文本

    Returns:
        HonestyReview，含违规类型列表与每条违规详情
    """
    violations: list[str] = []
    details: list[dict] = []

    for sentence in _split_sentences(report_md):
        # 1. overstatement（最严重，优先判定；同句不再判其他违规）
        if detect_overstatement(sentence):
            if "overstatement" not in violations:
                violations.append("overstatement")
            details.append({
                "sentence": sentence,
                "violation_type": "overstatement",
                "reason": (
                    "含过度声明措辞，建议替换为 Simulation suggests / "
                    "Hypothesis: ... / 模型提示"
                ),
            })
            continue

        # 2. citation_missing
        if check_citation_missing(sentence):
            if "citation_missing" not in violations:
                violations.append("citation_missing")
            details.append({
                "sentence": sentence,
                "violation_type": "citation_missing",
                "reason": "[Literature] 标签缺 PMID 引用",
            })
            continue

        # 3. unlabeled_claim
        if detect_unlabeled_claim(sentence):
            if "unlabeled_claim" not in violations:
                violations.append("unlabeled_claim")
            details.append({
                "sentence": sentence,
                "violation_type": "unlabeled_claim",
                "reason": (
                    "声明未标注任何四类标签 "
                    "(Prediction/Hypothesis/Simulation/Literature)"
                ),
            })

    # Status 判定
    if not violations:
        status = "PASS"
    elif "overstatement" in violations or "citation_missing" in violations:
        status = "FAIL"
    else:
        status = "PARTIAL"

    return HonestyReview(
        status=status,
        violations=violations,
        details=details,
    )


# =============================================================================
# 兼容旧 API（保留以避免破坏既有调用方；新代码请使用上方接口）
# =============================================================================


def check_scientific_honesty(report_md: str) -> HonestyReport:
    """检测 Report 中的 Scientific Honesty 违规（兼容旧 API）。

    本函数为旧 API 兼容入口，内部委托 review_report 实现并转换为
    HonestyReport。

    Args:
        report_md: Report Markdown 文本

    Returns:
        HonestyReport
    """
    review = review_report(report_md)

    violations: list[HonestyViolation] = []
    labeled = 0
    unlabeled = 0
    for d in review.details:
        v = HonestyViolation(
            violation_type=d.get("violation_type", ""),
            sentence=d.get("sentence", ""),
            label="",
            reason=d.get("reason", ""),
        )
        violations.append(v)
        if v.violation_type == HonestyViolationType.UNLABELED_CLAIM:
            unlabeled += 1
        else:
            labeled += 1

    return HonestyReport(
        violations=violations,
        status="FAIL" if review.status == "FAIL" else "PASS",
        labeled_claims=labeled,
        unlabeled_claims=unlabeled,
    )


def replace_overclaim(report_md: str) -> str:
    """将过度声明措辞替换为谦逊措辞。

    对应 Spec Scenario "禁止过度声明" — SHALL 替换为 "Simulation suggests" /
    "Hypothesis: ..." / "模型提示" 等谦逊措辞。

    Args:
        report_md: 原始 Report Markdown 文本

    Returns:
        替换后的 Report Markdown 文本
    """
    result = report_md
    for old, new in OVERSTATEMENT_REPLACEMENTS.items():
        # 大小写不敏感替换
        result = re.sub(re.escape(old), new, result, flags=re.IGNORECASE)
    # 清理替换后产生的多余空格
    result = re.sub(r"\s{2,}", " ", result)
    return result
