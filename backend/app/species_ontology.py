"""BioDynamics Agent - 物种本体层（Species Ontology Layer）

TASK 2 修复：建立物种语义分类与过滤规则，解决 SBML 语义错误。

职责：
- 将 LLM/KG 输出的物种名映射到 canonical species type
- 过滤非物种 token（model ID、pathway name、空字符串）
- 区分 free / bound / phosphorylated 状态，禁止混合建模
- 提供 total protein 守恒分组（用于 TASK 5 守恒检查）

Canonical Species Types:
- ligand: EGF
- receptor: EGFR
- complex: EGF_EGFR, EGF_EGFR_2
- phosphorylated_complex: EGF_pEGFR_2
- adaptor: Shc, Grb2, SOS
- signaling_protein: Ras, Raf, MEK, ERK
- gtpase: RasGDP, RasGTP
- kinase: MEK, ERK, Raf
- phosphatase: SHP, Pase, PP2A, MKP3
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# =============================================================================
# 物种类型枚举
# =============================================================================
SPECIES_TYPES = {
    "ligand",
    "receptor",
    "complex",
    "phosphorylated_complex",
    "adaptor",
    "signaling_protein",
    "gtpase",
    "kinase",
    "phosphatase",
    "double_phosphorylated",
    "unknown",
}


# =============================================================================
# 非物种 token 黑名单（model ID / pathway name / 占位符）
# =============================================================================
_BIOMD_PATTERN = re.compile(r"^BIOMD\d+$", re.IGNORECASE)
_MODEL_PATTERN = re.compile(r"^MODEL\d+$", re.IGNORECASE)
_PATHWAY_KEYWORDS = {
    "pathway", "signaling", "cascade", "egfr_mapk", "mapk_signaling",
    "egf_egfr_signaling_pathway", "signal_transduction",
}
_PLACEHOLDER_NAMES = {
    "", "species", "target", "inhibitor", "activator", "drug", "ligand",
    "receptor", "enzyme", "substrate", "product", "complex",
}


def is_valid_species(name: str) -> bool:
    """判断 token 是否为合法物种名（非 model ID / pathway name / 占位符）。

    Args:
        name: 待判断的物种名

    Returns:
        True 若为合法物种
    """
    if not name or not name.strip():
        return False
    name_clean = name.strip()
    # 过滤 BIOMD* / MODEL* ID
    if _BIOMD_PATTERN.match(name_clean) or _MODEL_PATTERN.match(name_clean):
        return False
    # 过滤 pathway name
    name_lower = name_clean.lower()
    if name_lower in _PATHWAY_KEYWORDS:
        return False
    # 过滤通配占位符（但保留具体名称如 "EGF_EGFR_complex"）
    if name_lower in _PLACEHOLDER_NAMES:
        return False
    # 过滤纯数字
    if name_clean.isdigit():
        return False
    return True


# =============================================================================
# 物种名 → canonical type 推断
# =============================================================================
def infer_species_type(name: str) -> str:
    """根据物种名推断 canonical type。

    规则（按优先级）：
    1. 名称含 "pp" 前缀且为激酶 → double_phosphorylated
    2. 名称含 "p" 前缀且为激酶 → phosphorylated kinase
    3. 名称含 "_" 或 "-" 连接多个蛋白 → complex
    4. 名称匹配已知配体/受体/适配蛋白/激酶/磷酸酶
    """
    if not name:
        return "unknown"
    name_lower = name.lower().replace("-", "_")

    # 双磷酸化激酶（ppMEK, ppERK, ppRaf）
    if name_lower.startswith("pp") and any(
        k in name_lower for k in ("mek", "erk", "raf")
    ):
        return "double_phosphorylated"

    # 磷酸化复合物（EGF_pEGFR_2）
    if "pegfr" in name_lower or "pshc" in name_lower or "praf" in name_lower:
        if "_" in name_lower or "-" in name_lower:
            return "phosphorylated_complex"

    # 复合物（含多个蛋白名连接）
    parts = re.split(r"[_\-]", name_lower)
    protein_count = sum(1 for p in parts if len(p) > 2 and p not in ("2", "the", "and"))
    if protein_count >= 2 and ("complex" in name_lower or any(
        k in name_lower for k in ("egf_egfr", "shc_grb2", "grb2_sos", "raf_ras",
                                    "praf_mek", "ppmek_erk")
    )):
        return "complex"

    # 配体
    if name_lower in ("egf", "epidermal_growth_factor", "ligand"):
        return "ligand"

    # 受体
    if name_lower in ("egfr", "erbb1", "erbb2", "her2", "receptor"):
        return "receptor"

    # GTPase
    if name_lower in ("rasgdp", "rasgtp", "ras_gdp", "ras_gtp", "rho_gdp", "rhogtp"):
        return "gtpase"

    # 适配蛋白
    if name_lower in ("shc", "shc1", "grb2", "sos", "sos1"):
        return "adaptor"

    # 激酶
    if name_lower in ("raf", "raf1", "mek", "map2k1", "erk", "mapk1", "mapk"):
        return "kinase"

    # 磷酸化激酶（pRaf, pMEK, pERK）
    if name_lower in ("praf", "pmek", "perk", "phosphorylated_raf",
                       "phosphorylated_mek", "phosphorylated_erk",
                       "phosphorylated_egfr"):
        return "kinase"  # 磷酸化形式仍属于激酶类，但状态不同

    # 磷酸酶
    if name_lower in ("shp", "shp2", "pase", "pp2a", "mkp3", "phosphatase"):
        return "phosphatase"

    return "unknown"


# =============================================================================
# 物种状态分类（free / bound / phosphorylated）
# =============================================================================
def infer_species_state(name: str) -> str:
    """推断物种的状态：free / bound / phosphorylated / double_phosphorylated。

    用于守恒检查：total = free + bound + phosphorylated
    """
    if not name:
        return "free"
    name_lower = name.lower().replace("-", "_")

    # 双磷酸化
    if name_lower.startswith("pp") and any(
        k in name_lower for k in ("mek", "erk", "raf")
    ):
        return "double_phosphorylated"

    # 单磷酸化
    if name_lower.startswith("p") and any(
        k in name_lower for k in ("egfr", "shc", "raf", "mek", "erk")
    ):
        return "phosphorylated"

    # 复合物（bound 状态）
    if "_" in name_lower or "-" in name_lower:
        if any(k in name_lower for k in ("egf_egfr", "shc_grb2", "grb2_sos",
                                          "raf_ras", "praf_mek", "ppmek_erk")):
            return "bound"

    return "free"


# =============================================================================
# 守恒分组：哪些物种属于同一个 "total pool"
# =============================================================================
def build_conservation_groups(species_names: list[str]) -> dict[str, list[str]]:
    """构建守恒分组：total(protein) = sum(all states of protein)。

    例：EGFR pool = [EGFR_free, EGF_EGFR, EGF_EGFR_2, EGF_pEGFR_2, ...]
        MEK pool = [MEK, pMEK, ppMEK, pRaf_MEK, pRaf_pMEK, ...]

    Returns:
        {pool_name: [species_indices or names]}
    """
    groups: dict[str, list[str]] = {}
    for name in species_names:
        if not is_valid_species(name):
            continue
        pool = _get_pool_name(name)
        if pool:
            groups.setdefault(pool, []).append(name)
    # 仅保留 ≥2 个成员的组（单成员无守恒意义）
    return {k: v for k, v in groups.items() if len(v) >= 2}


def _get_pool_name(name: str) -> str:
    """提取物种所属的蛋白池名。"""
    name_lower = name.lower().replace("-", "_")
    # [CAL-06b] 排除聚合器节点（如 "MAPK cascade"）：这些是概念性聚合节点，
    # 不是真实分子物种，不应参与质量守恒检查。它们从上游接收质量但不回流，
    # 导致 ERK pool 出现虚假 drift（+12.6% from MAPK cascade gaining 0.294
    # mass from pRaf cross-pool transfer）。
    if any(kw in name_lower for kw in ("cascade", "signaling", "pathway")):
        return ""
    # EGFR 池
    if "egfr" in name_lower or "erbb1" in name_lower:
        return "EGFR"
    # Shc 池
    if "shc" in name_lower:
        return "Shc"
    # Grb2 池
    if "grb2" in name_lower:
        return "Grb2"
    # SOS 池
    if "sos" in name_lower:
        return "SOS"
    # Ras 池
    if "ras" in name_lower:
        return "Ras"
    # Raf 池
    if "raf" in name_lower:
        return "Raf"
    # MEK 池
    if "mek" in name_lower or "map2k" in name_lower:
        return "MEK"
    # ERK 池
    if "erk" in name_lower or "mapk" in name_lower:
        return "ERK"
    return ""


# =============================================================================
# 物种过滤与分类主入口
# =============================================================================
@dataclass
class SpeciesClassification:
    """物种分类结果。"""
    valid_species: list[str] = field(default_factory=list)
    filtered_tokens: list[str] = field(default_factory=list)
    type_map: dict[str, str] = field(default_factory=dict)
    state_map: dict[str, str] = field(default_factory=dict)
    conservation_groups: dict[str, list[str]] = field(default_factory=dict)


def classify_species(raw_names: list[str]) -> SpeciesClassification:
    """对原始物种名列表进行分类、过滤、分组。

    Args:
        raw_names: 来自 KG/LLM 输出的物种名列表

    Returns:
        SpeciesClassification 包含合法物种、被过滤 token、类型映射、状态映射、守恒分组
    """
    valid: list[str] = []
    filtered: list[str] = []
    type_map: dict[str, str] = {}
    state_map: dict[str, str] = {}

    seen: set[str] = set()
    for name in raw_names:
        name_clean = (name or "").strip()
        if not name_clean:
            continue
        if name_clean in seen:
            continue
        seen.add(name_clean)

        if not is_valid_species(name_clean):
            filtered.append(name_clean)
            continue

        valid.append(name_clean)
        type_map[name_clean] = infer_species_type(name_clean)
        state_map[name_clean] = infer_species_state(name_clean)

    conservation_groups = build_conservation_groups(valid)

    return SpeciesClassification(
        valid_species=valid,
        filtered_tokens=filtered,
        type_map=type_map,
        state_map=state_map,
        conservation_groups=conservation_groups,
    )
