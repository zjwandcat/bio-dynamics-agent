# BioDynamics Agent v4 - Scientific Alignment Loop: Mechanism Alignment Graph (Task 8)
#
# 机制图对齐检查：校验 Agent 提取的机制节点是否覆盖 Canonical required_nodes。
#
# 设计目标：
#   机制分析必须覆盖通路的关键节点链。若 Agent 输出的机制图缺失关键节点
#   （如 EGFR 通路缺 DUSP 负反馈），Confidence 必须降级。
#
# 检查流程：
#   1. 从 canonical_loader.load_canonical(pathway) 加载 required_nodes
#   2. 将 Agent 提取的节点与 Canonical 节点做归一化匹配（大小写不敏感 + 同义词）
#   3. 计算 coverage = matched / required
#   4. 按缺失节点类型（普通 / 关键）计算 Confidence penalty
#   5. 返回 MechanismAlignmentResult，含 passed / coverage / adjusted_confidence 等
#
# Confidence 扣分规则：
#   - 缺失普通节点：penalty += 0.05
#   - 缺失关键节点（负反馈节点 或 节点链首尾）：penalty += 0.15
#   - penalty 上限 0.5（避免归零）
#   - adjusted_confidence = max(0.0, original_confidence - penalty)
#
# passed 判定：
#   passed = (coverage >= coverage_threshold) and (not missing_critical_nodes)
#   注意：缺关键节点即使 coverage 达标也算 fail（机制不完整）。
#
# 关键节点判定：
#   - 节点出现在 canonical.known_negative_feedback 中（直接匹配或复合条目组成部分）
#   - 节点为 canonical_mechanism.required_nodes 的第一个或最后一个（节点链首尾）
#
# Feature Flag 守护：
#   SA_MECHANISM_GRAPH 默认 OFF。关闭时返回 passed=True 不阻塞，仅记录 warning。
#   铁律：V4_SCIENTIFIC_ALIGNMENT_ENABLED=false 时，子 Flag 永远不生效
#         （由 settings.is_sa_feature_enabled 强制校验）。
#
# 安全设计：
#   - 所有 required_nodes 来自 Canonical YAML，禁止硬编码通路特定节点逻辑
#   - 只读 Canonical YAML，不修改
#   - 不引入新依赖（仅标准库 + canonical_loader + app.config.settings）
#
# 依赖：canonical_loader.py（已存在）、app.config.settings；不引入新依赖。
#
# 核心导出：
#   from app.scientific_alignment.mechanism_checker import (
#       MechanismAlignmentResult,
#       check_mechanism_alignment,
#       normalize_node_name,
#   )

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Set

from app.config import settings
from app.scientific_alignment.canonical_loader import load_canonical

logger = logging.getLogger(__name__)


# =============================================================================
# 常量
# =============================================================================

# Confidence 扣分规则常量
_PENALTY_PER_NORMAL_NODE: float = 0.05    # 缺失普通节点扣分
_PENALTY_PER_CRITICAL_NODE: float = 0.15  # 缺失关键节点扣分
_PENALTY_CAP: float = 0.50                # penalty 总上限（避免归零）

# 非字母数字字符正则：用于节点名归一化（去除 - _ / 空格 . 等所有分隔符）
# 例如 "p-ERK" → "PERK"、"beta_catenin" → "BETACATENIN"、"CyclinD-CDK4/6" → "CYCLINDDKD46"
_NON_ALNUM_RE: re.Pattern[str] = re.compile(r"[^A-Za-z0-9]+")


# -----------------------------------------------------------------------------
# 节点同义词表
# -----------------------------------------------------------------------------
# 设计说明：
#   - key  ：Canonical 归一化键（大写 + 去分隔符），代表唯一蛋白实体
#   - value：该实体的所有同义词集合（含 Canonical 键本身）
#   - 匹配时输入名与 Canonical 节点名都经 normalize_node_name 归一化后比较
#   - 表覆盖 EGFR / MAPK / p53 / Apoptosis / Cell Cycle / JAK-STAT /
#     NF-κB / Wnt / TGF-β / PI3K-AKT-mTOR 通路常见蛋白
#
# 同义词来源：HUGO Gene Names / UniProt / 常见文献别名 / 修饰态变体
# 表可扩展：新增通路节点时在此追加同义词组即可，无需改动主逻辑。
_NODE_SYNONYMS: Dict[str, Set[str]] = {
    # === p53 通路 ===
    "P53": {"P53", "TP53", "TRP53", "P53PROTEIN", "TUMORPROTEINP53"},
    "MDM2": {"MDM2", "HDM2", "MDM2PROTEIN"},
    "ATM": {"ATM", "ATAXIATELANGIECTASIAMUTATED"},
    "P21": {"P21", "CDKN1A", "WAF1", "CIP1"},

    # === EGFR / MAPK 通路 ===
    "EGF": {"EGF", "EPIDERMALGROWTHFACTOR"},
    "EGFR": {"EGFR", "ERBB1", "HER1"},
    "PEGFR": {"PEGFR", "EGFRP", "EGFR_P", "P_EGFR", "PHOSPHOEGFR"},
    "SHC": {"SHC", "SHCA", "SHC1"},
    "GRB2": {"GRB2", "GROWTHFACTORRECEPTORBOUNDPROTEIN2"},
    "SOS": {"SOS", "SOS1", "SONOFSEVENLESS"},
    "RAS": {"RAS", "KRAS", "HRAS", "NRAS"},
    "RASGTP": {"RASGTP", "RAS_GTP", "RASACTIVE", "RAS_ACTIVE", "RASGTPBOUND"},
    "RAF": {"RAF", "RAF1", "BRAF", "ARAF", "CRAF"},
    "MEK": {"MEK", "MEK1", "MEK2", "MAP2K1", "MAP2K2", "MKK1", "MKK2"},
    "ERK": {"ERK", "ERK1", "ERK2", "MAPK1", "MAPK3"},
    "PPERK": {"PPERK", "ERKP", "ERK_P", "P_ERK", "PERK", "ERKACTIVE", "PHOSPHOERK"},
    "DUSP": {"DUSP", "DUSP1", "DUSP2", "DUSP4", "MKP", "MKP1", "MKP2"},
    "SPRY": {"SPRY", "SPRY1", "SPRY2", "SPROUTY", "SPROUTY1", "SPROUTY2"},

    # === PI3K-AKT-mTOR 通路 ===
    "PI3K": {"PI3K", "PI3KINASE", "PIK3CA", "PIK3CB", "PIK3CD"},
    "PIP3": {"PIP3", "PHOSPHATIDYLINOSITOLTRIPHOSPHATE", "PIP3LIPID"},
    "PDK1": {"PDK1", "PDPK1"},
    "AKT": {"AKT", "AKT1", "AKT2", "AKT3", "PKB", "PROTEINKINASEB"},
    "TSC2": {"TSC2", "TUBERIN"},
    "MTORC1": {"MTORC1", "MTOR", "MTORCOMPLEX1"},
    "S6K": {"S6K", "S6K1", "RPS6KB1", "P70S6K", "S6KINASE"},

    # === NF-κB 通路 ===
    "IKK": {"IKK", "IKKALPHA", "IKKBETA", "IKKGAMMA", "IKK1", "IKK2", "NEMO", "IKKCOMPLEX"},
    "IKBALPHA": {"IKBALPHA", "IKBA", "IKB", "NFKBIA"},
    "NFKBP65": {"NFKBP65", "RELA", "P65", "NFKB", "NFKAPPAB"},
    "A20": {"A20", "TNFAIP3"},

    # === Wnt 通路 ===
    "WNT": {"WNT", "WNT1", "WNT3A", "WNT5A"},
    "FRIZZLED": {"FRIZZLED", "FZD", "FZD1", "FZD2", "FZD7", "FZRECEPTOR"},
    "LRP6": {"LRP6", "LRP5", "LRP"},
    "AXIN": {"AXIN", "AXIN1"},
    "AXIN2": {"AXIN2", "AXIL", "CONDUCTIN"},
    "APC": {"APC", "ADENOMATOUSPOLYPOSISCOLI"},
    "GSK3": {"GSK3", "GSK3ALPHA", "GSK3BETA", "GSK3B", "GLYCOGENSYNTHASEKINASE3"},
    "BETACATENIN": {"BETACATENIN", "CATENIN", "CTNNB1", "BCATENIN"},
    "TCFLEF": {"TCFLEF", "TCF", "LEF", "TCF1", "LEF1", "TCF4"},

    # === TGF-β 通路 ===
    "TGFBETA": {"TGFBETA", "TGFB", "TGFB1", "TGFB2", "TGFB3", "TRANSFORMINGGROWTHFACTORBETA"},
    "TGFBR1": {"TGFBR1", "ALK5", "TGFRI", "TGFBETARECEPTOR1"},
    "TGFBR2": {"TGFBR2", "TGFRII", "TGFBETARECEPTOR2"},
    "SMAD2": {"SMAD2"},
    "SMAD3": {"SMAD3"},
    "SMAD4": {"SMAD4", "DPC4", "MADH4"},
    "SMAD7": {"SMAD7"},
    "SNON": {"SNON", "SKIL"},

    # === JAK-STAT 通路 ===
    "CYTOKINERECEPTOR": {"CYTOKINERECEPTOR", "INTERFERONRECEPTOR", "IFNRECEPTOR"},
    "JAK": {"JAK", "JAK1", "JAK2", "JAK3", "TYK2", "JANUSKINASE"},
    "STAT5": {"STAT5", "STAT5A", "STAT5B"},
    "SOCS3": {"SOCS3", "SOC3"},
    "CIS": {"CIS", "CISH", "CIS1", "CYTOKINEINDUCIBLESH2"},

    # === Cell Cycle 通路 ===
    "CYCLINDDKD46": {"CYCLINDDKD46", "CYCLINDCDK46"},
    "CYCLINECDK2": {"CYCLINECDK2"},
    "CYCLINBCDK1": {"CYCLINBCDK1"},
    "RB": {"RB", "RB1", "RETINOBLASTOMA", "RETINOBLASTOMAPROTEIN"},
    "E2F": {"E2F", "E2F1", "E2F2", "E2F3"},
    "APCC": {"APCC", "APCCOMPLEX", "ANAPHASEPROMOTINGCOMPLEX"},
    "CDC20": {"CDC20", "CELLDIVISIONCYCLE20"},

    # === Apoptosis 通路 ===
    "FAS": {"FAS", "FASRECEPTOR", "CD95", "APO1"},
    "CASPASE8": {"CASPASE8", "CASP8", "FLICE", "MACH"},
    "CASPASE3": {"CASPASE3", "CASP3", "CPP32", "YAMA"},
    "BAX": {"BAX"},
    "BAK": {"BAK", "BAK1"},
    "TBID": {"TBID", "TRUNCATEDBID"},
    "XIAP": {"XIAP", "BIRC4"},
}


# =============================================================================
# 节点名归一化
# =============================================================================
def _basic_normalize(name: str) -> str:
    """基础归一化：转大写 + 去除所有非字母数字字符。

    去除 ``-`` ``_`` ``/`` 空格 ``.`` 等分隔符，例如：
      - ``"p-ERK"``        → ``"PERK"``
      - ``"beta_catenin"`` → ``"BETACATENIN"``
      - ``"CyclinD-CDK4/6"`` → ``"CYCLINDDKD46"``
      - ``"IkB_alpha"``    → ``"IKBALPHA"``

    Args:
        name: 原始节点名。

    Returns:
        大写、无分隔符的字符串；空输入返回空字符串。
    """
    if not name:
        return ""
    text = str(name).strip().upper()
    return _NON_ALNUM_RE.sub("", text)


def _build_synonym_lookup() -> Dict[str, str]:
    """构建同义词反向查找表：basic-normalized 同义词 → Canonical 归一化键。

    遍历 ``_NODE_SYNONYMS``，将每个同义词（经基础归一化）映射到其 Canonical 键
    （同样经基础归一化）。这样 ``normalize_node_name`` 即可通过一次字典查询
    完成同义词归并。

    Returns:
        dict，键为同义词的基础归一化形式，值为 Canonical 归一化键。
    """
    lookup: Dict[str, str] = {}
    for canonical_key, synonyms in _NODE_SYNONYMS.items():
        norm_key = _basic_normalize(canonical_key)
        for syn in synonyms:
            norm_syn = _basic_normalize(syn)
            if norm_syn:
                lookup[norm_syn] = norm_key
        # 确保 Canonical 键自身可查（即使未显式列入 synonyms 集合）
        if norm_key and norm_key not in lookup:
            lookup[norm_key] = norm_key
    return lookup


# 模块加载时构建一次，后续 normalize_node_name 直接查询
_SYNONYM_LOOKUP: Dict[str, str] = _build_synonym_lookup()


def normalize_node_name(name: str) -> str:
    """归一化节点名：基础归一化 + 同义词映射。

    处理步骤：
      1. 基础归一化（大写 + 去所有非字母数字分隔符）
      2. 若结果在同义词查找表中，映射到 Canonical 归一化键
      3. 否则返回基础归一化结果

    匹配时把输入名与 Canonical 节点名都经此函数归一化后比较，
    即可实现大小写不敏感 + 同义词匹配。例如：
      - ``"mkp1"`` → ``"DUSP"``（同义词映射）
      - ``"TP53"`` → ``"P53"``（同义词映射）
      - ``"p-ERK"`` → ``"PERK"``（基础归一化，无同义词映射）
      - ``"EGF"``  → ``"EGF"``（基础归一化，无同义词映射）

    Args:
        name: 原始节点名（如 ``"mkp1"``、``"TP53"``、``"p-ERK"``）。

    Returns:
        归一化后的节点名（如 ``"DUSP"``、``"P53"``、``"PERK"``）；
        空输入返回空字符串。
    """
    norm = _basic_normalize(name)
    if not norm:
        return ""
    return _SYNONYM_LOOKUP.get(norm, norm)


# =============================================================================
# 数据类
# =============================================================================
@dataclass
class MechanismAlignmentResult:
    """机制图对齐检查结果。

    Attributes:
        pathway: 通路标识（如 ``"egfr"``）。
        required_nodes: Canonical 要求的节点列表（原始名）。
        extracted_nodes: Agent 提取的节点列表（原始名）。
        missing_nodes: 缺失的 Canonical 节点列表（原始名）。
        extra_nodes: 多余的 Agent 节点列表（不在 required_nodes 中，非违规，
            仅信息性）。
        matched_nodes: 成功匹配的 Canonical 节点列表（原始名）。
        coverage: 覆盖率 matched / required（0.0-1.0）。
        confidence_penalty: 缺失节点导致的 Confidence 扣分。
        adjusted_confidence: 原始 Confidence 扣减 penalty 后的值。
        passed: 是否通过（coverage >= 阈值 且无关键节点缺失）。
        missing_critical_nodes: 缺失的关键节点列表（负反馈节点或节点链首尾）。
        warnings: 警告信息列表。
    """

    pathway: str = ""
    required_nodes: List[str] = field(default_factory=list)
    extracted_nodes: List[str] = field(default_factory=list)
    missing_nodes: List[str] = field(default_factory=list)
    extra_nodes: List[str] = field(default_factory=list)
    matched_nodes: List[str] = field(default_factory=list)
    coverage: float = 1.0
    confidence_penalty: float = 0.0
    adjusted_confidence: float = 1.0
    passed: bool = True
    missing_critical_nodes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# =============================================================================
# 关键节点判定
# =============================================================================
def _is_critical_node(
    node: str,
    required_nodes: List[str],
    known_negative_feedback: List[str],
) -> bool:
    """判定节点是否为关键节点。

    关键节点定义（满足任一即为关键）：
      1. 节点为 ``required_nodes`` 的第一个或最后一个（节点链首尾）。
      2. 节点出现在 ``known_negative_feedback`` 列表中——直接匹配，或作为
         复合负反馈条目的组成部分。

    复合负反馈条目（如 ``"MDM2-p53"``、``"APC_C-CyclinB"``、``"S6K-IRS1"``）
    按 ``"-"`` 拆分后逐部分匹配，任一部分命中即视为该节点参与负反馈。

    注意：按 ``"-"`` 而非所有非字母数字字符拆分，因为节点名本身可含 ``_`` /
    等字符（如 ``"APC_C"``、``"IkB_alpha"``），需保留完整。

    Args:
        node: 待判定的 Canonical 节点名（原始名）。
        required_nodes: Canonical required_nodes 列表。
        known_negative_feedback: Canonical known_negative_feedback 列表。

    Returns:
        True 表示为关键节点。
    """
    if not node:
        return False
    norm_node = normalize_node_name(node)
    if not norm_node:
        return False

    # 规则 1：节点链首尾（第一个或最后一个 required_node）
    if required_nodes:
        first_norm = normalize_node_name(required_nodes[0])
        last_norm = normalize_node_name(required_nodes[-1])
        if norm_node in (first_norm, last_norm):
            return True

    # 规则 2：出现在 known_negative_feedback 中（直接匹配或复合条目组成部分）
    for fb_entry in known_negative_feedback:
        if not fb_entry:
            continue
        # 复合负反馈条目按 "-" 拆分（如 "MDM2-p53" → ["MDM2", "p53"]）
        # 注意：不用非字母数字拆分，因为节点名本身可含 _ / 等字符
        parts = str(fb_entry).split("-")
        for part in parts:
            if part and normalize_node_name(part) == norm_node:
                return True

    return False


# =============================================================================
# 主函数
# =============================================================================
def check_mechanism_alignment(
    pathway: str,
    extracted_nodes: list[str],
    original_confidence: float = 1.0,
    coverage_threshold: float = 0.8,
) -> MechanismAlignmentResult:
    """检查 Agent 提取的机制节点是否覆盖 Canonical required_nodes。

    Args:
        pathway: 通路标识（如 ``"egfr"``），仅允许 ``[a-zA-Z0-9_]``。
        extracted_nodes: Agent 提取的机制节点列表（如
            ``["EGF", "EGFR", "RAS", "ERK"]``）。
        original_confidence: 原始 Confidence（0.0-1.0），将被扣减 penalty。
        coverage_threshold: 通过阈值，默认 0.8。

    Returns:
        MechanismAlignmentResult，含 coverage / passed / adjusted_confidence 等。

    Raises:
        CanonicalNotFoundError: Canonical 文件不存在（由 load_canonical 抛出）。
        CanonicalMissingError: Canonical 缺失必填字段（由 load_canonical 抛出）。
        CanonicalPathwayNameError: pathway 含非法字符（由 load_canonical 抛出）。
    """
    # -------------------------------------------------------------------------
    # Feature Flag 守护：默认 OFF，关闭时返回 passed=True 不阻塞
    # 铁律：V4_SCIENTIFIC_ALIGNMENT_ENABLED=false 时，子 Flag 永远不生效
    # -------------------------------------------------------------------------
    if not settings.is_sa_feature_enabled("MECHANISM_GRAPH"):
        return MechanismAlignmentResult(
            pathway=pathway,
            required_nodes=[],
            extracted_nodes=list(extracted_nodes),
            missing_nodes=[],
            extra_nodes=[],
            matched_nodes=[],
            coverage=1.0,
            confidence_penalty=0.0,
            adjusted_confidence=original_confidence,
            passed=True,  # Flag 关闭时不阻塞
            missing_critical_nodes=[],
            warnings=["SA_MECHANISM_GRAPH disabled, check skipped"],
        )

    # -------------------------------------------------------------------------
    # 1. 加载 Canonical Reference（失败时抛出异常，交由上层处理）
    #    所有 required_nodes 来自 Canonical YAML，禁止硬编码通路特定节点
    # -------------------------------------------------------------------------
    canonical = load_canonical(pathway)
    required_nodes: List[str] = list(canonical.required_nodes)
    known_negative_feedback: List[str] = list(canonical.known_negative_feedback)

    # -------------------------------------------------------------------------
    # 2. 归一化 Agent 提取节点，构建匹配集合
    #    extracted_norm_set      ：所有提取节点的归一化形式集合
    #    extracted_norm_to_orig  ：归一化形式 → 原始名（用于 extra_nodes 报告）
    # -------------------------------------------------------------------------
    extracted_norm_set: Set[str] = set()
    for enode in extracted_nodes:
        norm = normalize_node_name(enode)
        if norm:
            extracted_norm_set.add(norm)

    # -------------------------------------------------------------------------
    # 3. 逐个比对 Canonical required_nodes，确定 matched / missing
    #    匹配策略：归一化后集合成员判定（大小写不敏感 + 同义词）
    # -------------------------------------------------------------------------
    matched_nodes: List[str] = []
    missing_nodes: List[str] = []
    required_norm_set: Set[str] = set()

    for rnode in required_nodes:
        rnorm = normalize_node_name(rnode)
        if rnorm:
            required_norm_set.add(rnorm)
        if rnorm and rnorm in extracted_norm_set:
            matched_nodes.append(rnode)
        else:
            missing_nodes.append(rnode)

    # -------------------------------------------------------------------------
    # 4. 确定 extra_nodes（Agent 提取但不在 required_nodes 中的节点，仅信息性）
    # -------------------------------------------------------------------------
    extra_nodes: List[str] = []
    for enode in extracted_nodes:
        enorm = normalize_node_name(enode)
        if enorm and enorm not in required_norm_set:
            extra_nodes.append(str(enode))

    # -------------------------------------------------------------------------
    # 5. 计算覆盖率 coverage = matched / required
    # -------------------------------------------------------------------------
    required_count = len(required_nodes)
    matched_count = len(matched_nodes)
    if required_count > 0:
        coverage = matched_count / required_count
    else:
        # 无要求节点视为完全覆盖（Canonical 校验保证非空，此处防御性处理）
        coverage = 1.0

    # -------------------------------------------------------------------------
    # 6. 识别缺失的关键节点（负反馈节点 或 节点链首尾）
    # -------------------------------------------------------------------------
    missing_critical_nodes: List[str] = []
    for mnode in missing_nodes:
        if _is_critical_node(mnode, required_nodes, known_negative_feedback):
            missing_critical_nodes.append(mnode)

    # -------------------------------------------------------------------------
    # 7. 计算 Confidence penalty
    #    - 普通缺失节点：每个 +0.05
    #    - 关键缺失节点：每个 +0.15
    #    - penalty 上限 0.5（避免归零）
    #    - adjusted_confidence = max(0.0, original_confidence - penalty)
    # -------------------------------------------------------------------------
    normal_missing_count = len(missing_nodes) - len(missing_critical_nodes)
    penalty = (
        normal_missing_count * _PENALTY_PER_NORMAL_NODE
        + len(missing_critical_nodes) * _PENALTY_PER_CRITICAL_NODE
    )
    # penalty 上限保护（避免归零）
    penalty = min(penalty, _PENALTY_CAP)
    adjusted_confidence = max(0.0, original_confidence - penalty)

    # -------------------------------------------------------------------------
    # 8. passed 判定
    #    coverage 达标 且 无关键节点缺失 才算通过。
    #    注意：缺关键节点即使 coverage 达标也算 fail（机制不完整）。
    # -------------------------------------------------------------------------
    passed = (coverage >= coverage_threshold) and (not missing_critical_nodes)

    # -------------------------------------------------------------------------
    # 9. 生成 warnings（按严重性排序：关键缺失 > 普通缺失 > 低覆盖 > 多余节点）
    # -------------------------------------------------------------------------
    warnings: List[str] = []
    if missing_critical_nodes:
        warnings.append(
            f"missing critical nodes (negative feedback / chain ends): "
            f"{missing_critical_nodes}"
        )
    if missing_nodes:
        warnings.append(
            f"missing {len(missing_nodes)} required node(s): {missing_nodes}"
        )
    if coverage < coverage_threshold:
        warnings.append(
            f"coverage {coverage:.2f} below threshold {coverage_threshold:.2f}"
        )
    if extra_nodes:
        warnings.append(
            f"{len(extra_nodes)} extra node(s) not in canonical "
            f"(informational only): {extra_nodes}"
        )

    logger.debug(
        "mechanism alignment check: pathway=%s coverage=%.2f missing=%s "
        "critical=%s penalty=%.2f adjusted_conf=%.2f passed=%s",
        pathway, coverage, missing_nodes, missing_critical_nodes,
        penalty, adjusted_confidence, passed,
    )

    return MechanismAlignmentResult(
        pathway=pathway,
        required_nodes=required_nodes,
        extracted_nodes=list(extracted_nodes),
        missing_nodes=missing_nodes,
        extra_nodes=extra_nodes,
        matched_nodes=matched_nodes,
        coverage=coverage,
        confidence_penalty=penalty,
        adjusted_confidence=adjusted_confidence,
        passed=passed,
        missing_critical_nodes=missing_critical_nodes,
        warnings=warnings,
    )


__all__ = [
    "MechanismAlignmentResult",
    "check_mechanism_alignment",
    "normalize_node_name",
]
