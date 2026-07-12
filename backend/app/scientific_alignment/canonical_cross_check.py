# BioDynamics Agent v4 - Scientific Alignment Loop: Canonical Cross-Check (Task 22.3)
#
# 交叉一致性校验：加载三处标准源并校验 Canonical 与 Gold Standard / Literature 的一致性。
#
# 三处标准源：
#   1. knowledge/canonical/<pathway>.yaml           — Canonical Reference（本通路唯一权威）
#   2. benchmarks/scientific_alignment/<pathway>.yaml — Gold Standard（机器可读标准答案）
#   3. knowledge/gold_standard/literature_<pathway>.yaml — Literature Gold Standard（文献级标准）
#
# 校验项：
#   a. canonical_models ⊆ Gold Standard required_biomodels（冲突级别 [CONFLICT]）
#   b. canonical_reviews ⊆ literature classical_reviews 的 PMID 集合（差异级别 [WARNING]）
#   c. canonical_mechanism.required_nodes ⊇ Gold Standard required_mechanisms 关键节点
#      （大小写不敏感匹配；差异级别 [WARNING]）
#
# 返回值：问题描述列表（空列表 = 完全一致）。每条以 [CONFLICT] 或 [WARNING] 前缀。
#
# 依赖：canonical_loader.py / gold_standard_schema.py / evidence_ranker.py（均已存在）

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Set

from app.scientific_alignment.canonical_loader import (
    CanonicalReference,
    load_canonical,
)
from app.scientific_alignment.evidence_ranker import (
    EvidenceRanker,
    load_literature_gold_standard,
)
from app.scientific_alignment.gold_standard_schema import (
    GoldStandard,
    load_gold_standard,
)

logger = logging.getLogger(__name__)


# =============================================================================
# 常量
# =============================================================================
# 非蛋白关键词（从 Gold Standard mechanism ID 中分词后需排除的通用词）
_NON_PROTEIN_TOKENS: Set[str] = {
    "activation", "inhibition", "phosphorylation", "binding", "degradation",
    "transcription", "translocation", "nuclear", "cascade", "feedback",
    "negative", "positive", "complex", "receptor", "destruction", "stabilization",
    "dimerization", "dimer", "amplification", "sensing", "arrest", "commitment",
    "switch", "checkpoint", "initiator", "effector", "resynthesis", "export",
    "cleavage", "recruitment", "heterodimer", "inhibitory", "signal",
}

# PMID 归一化正则：提取纯数字
_PMID_DIGIT_RE: re.Pattern[str] = re.compile(r"\d+")


# =============================================================================
# 归一化辅助
# =============================================================================
def _normalize_pmid(pmid: Any) -> str:
    """归一化 PMID：提取纯数字部分，跳过 TODO 占位符。

    Args:
        pmid: 原始 PMID（如 ``"PMID:7657691"`` 或 ``"TODO: verify PMID"``）。

    Returns:
        纯数字字符串（如 ``"7657691"``）；TODO 或无效值返回空字符串。
    """
    if not pmid:
        return ""
    text = str(pmid)
    if "TODO" in text.upper():
        return ""
    match = _PMID_DIGIT_RE.search(text)
    return match.group(0) if match else ""


def _normalize_biomodel(bm: Any) -> str:
    """归一化 BioModels ID：去除空白与注释，统一大写。

    Args:
        bm: 原始 BioModels ID（如 ``"BIOMD0000000010"``）。

    Returns:
        归一化后的 ID 字符串；无效值返回空字符串。
    """
    if not bm:
        return ""
    return str(bm).strip().upper()


def _normalize_node(node: Any) -> str:
    """归一化机制节点名：去除空白，转大写，便于大小写不敏感匹配。

    Args:
        node: 原始节点名（如 ``"egfr"`` 或 ``"EGFR"``）。

    Returns:
        大写节点名字符串。
    """
    if not node:
        return ""
    return str(node).strip().upper()


# =============================================================================
# 交叉校验核心
# =============================================================================
def _check_models_subset(
    canonical_models: List[str],
    gold_biomodels: List[str],
    pathway: str,
) -> List[str]:
    """校验 canonical_models ⊆ Gold Standard required_biomodels。

    Args:
        canonical_models: Canonical 的权威 BioModels ID 列表。
        gold_biomodels: Gold Standard 的 required_biomodels 列表。
        pathway: 通路标识（用于错误描述）。

    Returns:
        冲突描述列表（空 = 一致）。
    """
    issues: List[str] = []
    gold_set: Set[str] = {
        _normalize_biomodel(bm) for bm in gold_biomodels if _normalize_biomodel(bm)
    }
    for cm in canonical_models:
        normalized = _normalize_biomodel(cm)
        if not normalized:
            continue
        if normalized not in gold_set:
            issues.append(
                f"[CONFLICT] {pathway}: canonical_models 含 {normalized}，"
                f"但 Gold Standard required_biomodels 中不存在（{sorted(gold_set)}）"
            )
    return issues


def _check_reviews_subset(
    canonical_reviews: List[str],
    literature_data: Dict[str, Any],
    pathway: str,
) -> List[str]:
    """校验 canonical_reviews ⊆ literature classical_reviews 的 PMID 集合。

    注意：canonical_reviews 中的 PMID 若在 literature classical_reviews 中找不到，
    报 [WARNING]（非冲突，可能是 Canonical 将 mechanism paper 也列为权威综述）。

    Args:
        canonical_reviews: Canonical 的权威综述 PMID 列表。
        literature_data: literature_<pathway>.yaml 解析后的 dict。
        pathway: 通路标识。

    Returns:
        差异描述列表（空 = 一致）。
    """
    issues: List[str] = []

    # 提取 literature classical_reviews 的所有有效 PMID
    lit_pmids: Set[str] = set()
    classical = literature_data.get("classical_reviews") or []
    if isinstance(classical, list):
        for item in classical:
            if isinstance(item, dict):
                lit_pmids.add(_normalize_pmid(item.get("pmid", "")))
    lit_pmids.discard("")

    for cr in canonical_reviews:
        normalized = _normalize_pmid(cr)
        if not normalized:
            continue
        if normalized not in lit_pmids:
            issues.append(
                f"[WARNING] {pathway}: canonical_reviews 含 PMID:{normalized}，"
                f"但 literature classical_reviews 中未找到"
                f"（可能在 mechanism_papers 中，非严格冲突）"
            )
    return issues


def _check_mechanism_nodes(
    canonical_nodes: List[str],
    gold_mechanisms: List[str],
    pathway: str,
) -> List[str]:
    """校验 canonical_mechanism.required_nodes ⊇ Gold Standard required_mechanisms 关键节点。

    匹配策略（大小写不敏感）：
    - Gold Standard 的 mechanism ID（如 ``"EGF_EGFR_binding"``）按下划线分词
    - 排除非蛋白通用词（activation/binding/cascade 等）
    - 剩余 token 视为蛋白名，检查是否能在 Canonical required_nodes 中找到对应
    - 若某 mechanism ID 的所有蛋白 token 均未匹配到 Canonical 节点，报 [WARNING]

    Args:
        canonical_nodes: Canonical 的机制节点列表（大写蛋白名）。
        gold_mechanisms: Gold Standard 的 required_mechanisms 列表（机制边 ID）。
        pathway: 通路标识。

    Returns:
        差异描述列表（空 = 一致）。
    """
    issues: List[str] = []

    # 构建 Canonical 节点大写集合
    canonical_set: Set[str] = {
        _normalize_node(n) for n in canonical_nodes if _normalize_node(n)
    }

    for mech in gold_mechanisms:
        if not isinstance(mech, str) or not mech:
            continue
        # 按下划线分词，过滤非蛋白词
        tokens = [
            t for t in mech.split("_")
            if t and t.lower() not in _NON_PROTEIN_TOKENS
        ]
        if not tokens:
            continue
        # 检查是否有至少一个 token 匹配 Canonical 节点（大小写不敏感）
        matched = any(_normalize_node(t) in canonical_set for t in tokens)
        if not matched:
            issues.append(
                f"[WARNING] {pathway}: Gold Standard mechanism '{mech}' "
                f"的蛋白 token {tokens} 在 Canonical required_nodes "
                f"（{sorted(canonical_set)}）中均未找到对应"
            )
    return issues


# =============================================================================
# 公共 API
# =============================================================================
def cross_check_consistency(pathway: str) -> List[str]:
    """加载三处标准源并校验 Canonical 与 Gold Standard / Literature 的一致性。

    校验项：
      a. canonical_models ⊆ Gold Standard required_biomodels → [CONFLICT]
      b. canonical_reviews ⊆ literature classical_reviews PMID 集合 → [WARNING]
      c. canonical_mechanism.required_nodes ⊇ Gold Standard required_mechanisms
         关键节点（大小写不敏感）→ [WARNING]

    Args:
        pathway: 通路标识（如 ``"egfr"``），仅允许 [a-zA-Z0-9_]。

    Returns:
        问题描述列表（空列表 = 完全一致）。每条以 [CONFLICT] 或 [WARNING] 前缀。
        [CONFLICT] 为硬冲突（Canonical 与 Gold Standard 矛盾）；
        [WARNING] 为软差异（非严格冲突但需人工确认）。
    """
    issues: List[str] = []

    # 1. 加载 Canonical Reference
    try:
        cr: CanonicalReference = load_canonical(pathway)
    except Exception as exc:
        issues.append(
            f"[CONFLICT] {pathway}: 无法加载 Canonical Reference: {exc}"
        )
        return issues

    # 2. 加载 Gold Standard（benchmarks/scientific_alignment/<pathway>.yaml）
    try:
        gs: GoldStandard = load_gold_standard(pathway)
    except Exception as exc:
        issues.append(
            f"[WARNING] {pathway}: 无法加载 Gold Standard: {exc}"
        )
        gs = None

    # 3. 加载 Literature Gold Standard（knowledge/gold_standard/literature_<pathway>.yaml）
    try:
        lit_data: Dict[str, Any] = load_literature_gold_standard(pathway)
    except Exception as exc:
        issues.append(
            f"[WARNING] {pathway}: 无法加载 Literature Gold Standard: {exc}"
        )
        lit_data = {}

    # a. canonical_models ⊆ Gold Standard required_biomodels
    if gs is not None:
        issues.extend(
            _check_models_subset(
                cr.canonical_models, gs.required_biomodels, pathway
            )
        )

    # b. canonical_reviews ⊆ literature classical_reviews PMID 集合
    if lit_data:
        issues.extend(
            _check_reviews_subset(
                cr.canonical_reviews, lit_data, pathway
            )
        )

    # c. canonical_mechanism.required_nodes ⊇ Gold Standard required_mechanisms 关键节点
    if gs is not None:
        issues.extend(
            _check_mechanism_nodes(
                cr.required_nodes, gs.required_mechanisms, pathway
            )
        )

    return issues


__all__ = [
    "cross_check_consistency",
]
