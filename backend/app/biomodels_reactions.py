"""BioDynamics Agent - BIOMD 反应图生成器（Reaction Graph Generator）

TASK 3 修复：从 BIOMD0000000205 JSON 直接生成 reactions / species / stoichiometry。

禁止 template-only reaction generation；禁止 generic "activation" 替代 reaction。
仅支持 mass-action kinetics（TASK 4 联动）。

数据源：backend/data/processed/{model_id}.json
该 JSON 由 EBI BioModels SBML 预处理生成，包含：
- initial_concentration 条目（type=initial_concentration）
- kinetic_rate 条目（type=kinetic_rate，含 reaction_id / reaction_equation / reactants / products）
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.species_ontology import is_valid_species, infer_species_type

logger = logging.getLogger(__name__)

# BIOMD JSON 数据目录
_BIOMD_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


# =============================================================================
# TASK 1 / TASK 4 / TASK 6：模型语义收敛硬约束
# =============================================================================
# 唯一允许通路：EGF → EGFR → Shc → Grb2 → SOS → Ras → Raf → MEK → ERK
ALLOWED_PATHWAY_SET: frozenset[str] = frozenset({"EGF_EGFR_MAPK"})

# 严格禁止的非模型术语（PI3K/Akt/NF-κB/JAK-STAT/feedback/crosstalk 等）
FORBIDDEN_PATHWAY_TERMS: tuple[str, ...] = (
    "pi3k", "akt", "mtor", "nf-kappa", "nf-kb", "nfkb", "nf-κb",
    "jak", "stat", "stat3", "stat5",
    "feedback", "crosstalk", "cross-talk", "emergent",
)

# 核心物种集合（含配体、受体、适配蛋白、激酶、GTPase 及其活性/复合形式）
# 只有涉及这些物种的反应才被保留，防止模型外通路污染。
CORE_SPECIES_SET: frozenset[str] = frozenset({
    # 配体 / 受体
    "EGF", "EGFR",
    # 受体活性 / 复合形式（canonical）
    "EGFR_active",
    # 适配蛋白
    "Shc", "pShc", "Shc_complex",
    "Grb2", "Grb2_SOS_complex",
    "SOS",
    # GTPase
    "Ras", "RasGDP", "RasGTP", "Ras_active", "Ras_inactive",
    # MAPK 级联
    "Raf", "pRaf", "Raf_active",
    "MEK", "pMEK", "ppMEK", "MEK_active",
    "ERK", "pERK", "ppERK", "MAPK", "pMAPK", "ppMAPK", "ERK_active",
    # 磷酸酶（仅作为去磷酸化反应的催化剂，保留但合并）
    "SHP", "Pase", "PP2A", "MKP3",
})

# Canonical reduction：长复合物 / 活性状态 → 规范节点（TASK 2）
# 目标：消除 EGF_pEGFR_2_pShc_Grb2_SOS_RasGDP / Raf_RasGTP / Grb2_SOS 等曲线爆炸源。
CANONICAL_REDUCTION_MAP: dict[str, str] = {
    # --- EGF-EGFR 受体复合物 → EGFR_active ---
    "EGF-EGFR": "EGFR_active",
    "EGF_EGFR": "EGFR_active",
    "EGF-EGFR-2": "EGFR_active",
    "EGF_EGFR_2": "EGFR_active",
    "EGF-pEGFR-2": "EGFR_active",
    "EGF_pEGFR_2": "EGFR_active",
    "pEGFR": "EGFR_active",

    # --- Shc 相关复合物 → Shc_complex ---
    "EGF-pEGFR-2-Shc": "Shc_complex",
    "EGF_pEGFR_2_Shc": "Shc_complex",
    "EGF-pEGFR-2-pShc": "Shc_complex",
    "EGF_pEGFR_2_pShc": "Shc_complex",
    "pShc": "Shc_complex",
    "pShc-SHP": "Shc_complex",
    "pShc_SHP": "Shc_complex",
    "Shc": "Shc_complex",

    # --- Grb2-SOS 复合物 → Grb2_SOS_complex ---
    "Grb2-SOS": "Grb2_SOS_complex",
    "Grb2_SOS": "Grb2_SOS_complex",
    "EGF-pEGFR-2-Grb2": "Grb2_SOS_complex",
    "EGF_pEGFR_2_Grb2": "Grb2_SOS_complex",
    "EGF-pEGFR-2-Grb2-SOS": "Grb2_SOS_complex",
    "EGF_pEGFR_2_Grb2_SOS": "Grb2_SOS_complex",
    "EGF-pEGFR-2-pShc-Grb2": "Grb2_SOS_complex",
    "EGF_pEGFR_2_pShc_Grb2": "Grb2_SOS_complex",
    "EGF-pEGFR-2-pShc-Grb2-SOS": "Grb2_SOS_complex",
    "EGF_pEGFR_2_pShc_Grb2_SOS": "Grb2_SOS_complex",
    "Grb2": "Grb2_SOS_complex",
    "SOS": "Grb2_SOS_complex",

    # --- Ras 活性形式 → Ras_active / Ras_inactive ---
    "RasGTP": "Ras_active",
    "Ras_GTP": "Ras_active",
    "RasGDP": "Ras_inactive",
    "Ras_GDP": "Ras_inactive",
    "EGF-pEGFR-2-pShc-Grb2-SOS-RasGDP": "Ras_active",
    "EGF_pEGFR_2_pShc_Grb2_SOS_RasGDP": "Ras_active",
    "EGF-pEGFR-2-Grb2-SOS-RasGDP": "Ras_active",
    "EGF_pEGFR_2_Grb2_SOS_RasGDP": "Ras_active",

    # --- Raf 活性形式 → Raf_active ---
    "Raf-RasGTP": "Raf_active",
    "Raf_RasGTP": "Raf_active",
    "pRaf": "Raf_active",
    "Raf": "Raf_active",

    # --- MEK 活性形式 → MEK_active ---
    "pRaf-MEK": "MEK_active",
    "pRaf_MEK": "MEK_active",
    "pRaf-pMEK": "MEK_active",
    "pRaf_pMEK": "MEK_active",
    "pMEK": "MEK_active",
    "ppMEK": "MEK_active",
    "MEK": "MEK_active",

    # --- ERK 活性形式 → ERK_active ---
    "ppMEK-ERK": "ERK_active",
    "ppMEK_ERK": "ERK_active",
    "ppMEK-pERK": "ERK_active",
    "ppMEK_pERK": "ERK_active",
    "pERK": "ERK_active",
    "ppERK": "ERK_active",
    "pMAPK": "ERK_active",
    "ppMAPK": "ERK_active",
    "MAPK": "ERK_active",
    "ERK": "ERK_active",
}

# 可视化白名单（TASK 5）：只允许绘制 ≤ 8 个 canonical species
PLOT_CANONICAL_SET: tuple[str, ...] = (
    "EGFR_active",
    "Shc_complex",
    "Grb2_SOS_complex",
    "Ras_active",
    "Raf_active",
    "MEK_active",
    "ERK_active",
)

# 必须存在的 MAPK 链边（TASK 3）。若 BIOMD 反应图缺失，将强制注入。
_REQUIRED_MAPK_CHAIN: list[tuple[str, str, str]] = [
    ("Raf_active", "MEK_active", "phosphorylation"),
    ("MEK_active", "ERK_active", "phosphorylation"),
]

# 规范 EGF-EGFR-MAPK 链顺序（用于 edge 方向校正，确保信号流单向传递）
_CHAIN_ORDER: dict[str, int] = {
    "EGF": 0,
    "EGFR_active": 1,
    "Shc_complex": 2,
    "Grb2_SOS_complex": 3,
    "Ras_active": 4,
    "Raf_active": 5,
    "MEK_active": 6,
    "ERK_active": 7,
}


def _canonicalize_equation(equation: str) -> str:
    """将反应方程中的所有物种名 collapse 为 canonical 节点。

    例：EGF-pEGFR-2 + Shc → EGF-pEGFR-2-Shc
        → EGFR_active + Shc_complex → Shc_complex
    """
    if not equation or "→" not in equation:
        return equation
    lhs, rhs = equation.split("→", 1)

    def _canonicalize_side(side: str) -> str:
        tokens = [t.strip() for t in side.replace("+", " ").split() if t.strip()]
        canonical_tokens: list[str] = []
        for tok in tokens:
            sp, coef = _parse_stoichiometry_token(tok)
            can = collapse_species(sp)
            if coef != 1.0:
                canonical_tokens.append(f"{coef} {can}")
            else:
                canonical_tokens.append(can)
        return " + ".join(canonical_tokens)

    return f"{_canonicalize_side(lhs)} → {_canonicalize_side(rhs)}"


def collapse_species(species_name: str) -> str:
    """将长复合物名 / 磷酸化状态 collapse 为 canonical 节点（TASK 2）。

    例：EGF_pEGFR_2_pShc_Grb2_SOS_RasGDP → Ras_active
        Raf_RasGTP → Raf_active
        Grb2_SOS → Grb2_SOS_complex
    """
    if not species_name:
        return species_name
    # 先精确匹配；若未命中，尝试将 '-' 替换为 '_' 再匹配
    canonical = CANONICAL_REDUCTION_MAP.get(species_name)
    if canonical is not None:
        return canonical
    normalized = species_name.replace("-", "_")
    canonical = CANONICAL_REDUCTION_MAP.get(normalized)
    if canonical is not None:
        return canonical
    return species_name


def _contains_forbidden_term(text: str) -> bool:
    """检查文本是否含禁止的非模型通路术语（TASK 4）。"""
    if not text:
        return False
    text_lower = text.lower()
    return any(term in text_lower for term in FORBIDDEN_PATHWAY_TERMS)


def _is_core_species(species_name: str) -> bool:
    """判断物种是否属于 EGF_EGFR_MAPK 核心集合。"""
    if not species_name:
        return False
    canonical = collapse_species(species_name)
    return canonical in CORE_SPECIES_SET or species_name in CORE_SPECIES_SET


def _reaction_in_allowed_pathway(reactants: list[str], products: list[str], equation: str) -> bool:
    """判断反应是否属于 ALLOWED_PATHWAY_SET（TASK 1 / TASK 4）。

    规则：
    1. 反应方程或物种名含 FORBIDDEN_PATHWAY_TERMS → reject
    2. 所有 reactants / products 必须都是 core species（或 collapse 后是 core species）
    3. 仅当反应在 EGF → ERK 主链上才保留
    """
    # 1. 禁止术语检查
    if _contains_forbidden_term(equation):
        return False

    all_tokens = reactants + products
    # 2. 核心物种检查
    for token in all_tokens:
        sp_name, _ = _parse_stoichiometry_token(token)
        if not _is_core_species(sp_name):
            return False

    # 3. 必须至少有一个 reactant 或 product 是 canonical active/complex 形式，
    #    避免无关的降解 / 合成反应混入。
    canonical_tokens = {collapse_species(_parse_stoichiometry_token(t)[0]) for t in all_tokens}
    chain_canonicals = {
        "EGFR_active", "Shc_complex", "Grb2_SOS_complex", "Ras_active",
        "Ras_inactive", "Raf_active", "MEK_active", "ERK_active",
    }
    if not (canonical_tokens & chain_canonicals):
        # 允许 EGF + EGFR → EGFR_active 这种启动反应
        if not ({"EGF", "EGFR"} & canonical_tokens):
            return False

    return True


def _ensure_mapk_chain(edges: list[dict]) -> list[dict]:
    """强制确保 Raf_active → MEK_active → ERK_active 链存在（TASK 3）。

    若现有 edges 中缺失关键磷酸化边，则注入 mass-action phosphorylation 边。
    """
    edge_set: set[tuple[str, str]] = {
        (e.get("source", ""), e.get("target", "")) for e in edges
    }
    for src, tgt, mech in _REQUIRED_MAPK_CHAIN:
        if (src, tgt) not in edge_set and (tgt, src) not in edge_set:
            logger.warning("MAPK 断链修复：强制注入 %s → %s (%s)", src, tgt, mech)
            edges.append({
                "source": src,
                "target": tgt,
                "interaction": mech,
                "mechanism": mech,
                "reaction_equation": f"{src} + {tgt.replace('_active', '')} → {src} + {tgt}",
                "reaction_id": f"injected_{src}_{tgt}",
                "k_forward": 0.5,
                "k_reverse": 0.0,
                "is_reversible": False,
                "source_model": "BIOMD0000000205_canonical",
                "injected": True,
            })
    return edges


# =============================================================================
# 数据结构
# =============================================================================
@dataclass
class Reaction:
    """单个反应的 mass-action 表示。"""
    reaction_id: str
    reaction_name: str
    equation: str  # 原始反应方程式
    reactants: list[str]  # 反应物（含化学计量数前缀，如 "2 EGF-EGFR"）
    products: list[str]
    # mass-action 参数
    k_forward: float = 0.0  # 正向速率 k1
    k_reverse: float = 0.0  # 逆向速率 k2（可逆反应）
    unit_forward: str = ""  # 正向速率单位
    unit_reverse: str = ""
    is_reversible: bool = False


@dataclass
class ReactionGraph:
    """完整反应图。"""
    species: list[str]  # 唯一物种列表（已过滤 model ID）
    species_initial: dict[str, float]  # 物种初始浓度（nM）
    reactions: list[Reaction]  # 反应列表
    stoichiometry: dict[str, dict[str, float]]  # {reaction_id: {species: stoich}}
    conservation_groups: dict[str, list[str]] = field(default_factory=dict)
    source_model: str = ""
    consistency_report: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# JSON 加载
# =============================================================================
def load_biomd_json(model_id: str) -> list[dict[str, Any]] | None:
    """加载 BIOMD JSON 数据文件。

    Args:
        model_id: 如 "BIOMD0000000205"

    Returns:
        参数条目列表，或 None（文件不存在）
    """
    json_path = _BIOMD_DATA_DIR / f"{model_id}.json"
    if not json_path.exists():
        logger.warning("BIOMD JSON 不存在：%s", json_path)
        return None
    try:
        with json_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.error("加载 BIOMD JSON 失败：%s", exc)
        return None


# =============================================================================
# 反应图构建
# =============================================================================
def _parse_stoichiometry_token(token: str) -> tuple[str, float]:
    """解析带化学计量数的 token，如 "2 EGF-EGFR" → ("EGF-EGFR", 2.0)。"""
    token = token.strip()
    match = re.match(r"^(\d+(?:\.\d+)?)\s+(.+)$", token)
    if match:
        return match.group(2).strip(), float(match.group(1))
    return token, 1.0


def build_reaction_graph(model_id: str) -> ReactionGraph | None:
    """从 BIOMD JSON 构建完整反应图。

    流程：
    1. 加载 JSON
    2. 提取 initial_concentration → species_initial
    3. 提取 kinetic_rate → 按 reaction_id 聚合，构建 Reaction 列表
    4. 从 reactants/products 提取唯一 species（过滤 model ID）
    5. 构建 stoichiometry matrix

    Args:
        model_id: 如 "BIOMD0000000205"

    Returns:
        ReactionGraph 或 None（加载失败）
    """
    data = load_biomd_json(model_id)
    if data is None:
        return None

    # 1. 提取初始浓度，并对物种名进行 canonical collapse
    raw_initial: dict[str, float] = {}
    for entry in data:
        if entry.get("type") == "initial_concentration":
            sp = entry.get("species", "").strip()
            val = entry.get("value", 0.0)
            if sp and is_valid_species(sp):
                raw_initial[sp] = float(val)

    # TASK 2：将初始浓度聚合到 canonical species（质量守恒池合并）
    species_initial: dict[str, float] = {}
    for sp, val in raw_initial.items():
        canonical = collapse_species(sp)
        species_initial[canonical] = species_initial.get(canonical, 0.0) + float(val)

    # 2. 按 reaction_id 聚合动力学参数，并应用模型语义收敛过滤
    reactions_by_id: dict[str, Reaction] = {}
    rejected_reactions: list[str] = []
    for entry in data:
        if entry.get("type") != "kinetic_rate":
            continue
        rxn_id = entry.get("reaction_id", "")
        if not rxn_id:
            continue
        if rxn_id not in reactions_by_id:
            reactions_by_id[rxn_id] = Reaction(
                reaction_id=rxn_id,
                reaction_name=entry.get("reaction_name", rxn_id),
                equation=entry.get("reaction_equation", ""),
                reactants=entry.get("reactants", []),
                products=entry.get("products", []),
            )
        rxn = reactions_by_id[rxn_id]
        param_name = entry.get("param_name", "").lower()
        value = float(entry.get("value", 0.0))
        unit = entry.get("unit", "")
        if param_name in ("k1", "k_on", "kon", "kf", "k_forward"):
            rxn.k_forward = value
            rxn.unit_forward = unit
        elif param_name in ("k2", "k_off", "koff", "kr", "k_reverse"):
            rxn.k_reverse = value
            rxn.unit_reverse = unit
            rxn.is_reversible = True

    # TASK 1 / TASK 4：过滤非 ALLOWED_PATHWAY_SET 的反应
    filtered_reactions: list[Reaction] = []
    for rxn in reactions_by_id.values():
        if _reaction_in_allowed_pathway(rxn.reactants, rxn.products, rxn.equation):
            filtered_reactions.append(rxn)
        else:
            rejected_reactions.append(rxn.reaction_id)
    reactions = filtered_reactions

    if rejected_reactions:
        logger.info(
            "BIOMD 语义收敛：已拒绝 %d 条非 EGF_EGFR_MAPK 反应（示例：%s）",
            len(rejected_reactions), rejected_reactions[:5]
        )

    # TASK 2：对保留的反应执行 canonical reduction（collapse 长复合物名）
    for rxn in reactions:
        rxn.reactants = [collapse_species(r) for r in rxn.reactants]
        rxn.products = [collapse_species(p) for p in rxn.products]
        # collapse 后可能出现自环（如 Ras_active → Ras_active），后续 edges 构建会去重

    # 3. 提取唯一 species（从 reactants/products），已 collapse
    species_set: list[str] = []
    species_seen: set[str] = set()
    for rxn in reactions:
        for token_list in (rxn.reactants, rxn.products):
            for token in token_list:
                sp_name, _ = _parse_stoichiometry_token(token)
                sp_name = collapse_species(sp_name)
                if is_valid_species(sp_name) and sp_name not in species_seen:
                    species_seen.add(sp_name)
                    species_set.append(sp_name)

    # 4. 过滤初始浓度：仅保留仍在模型中的 canonical species（TASK 1/4）
    species_initial = {
        sp: val for sp, val in species_initial.items()
        if sp in species_seen and is_valid_species(sp)
    }

    # 5. 构建化学计量矩阵（使用 collapse 后的物种名）
    stoichiometry: dict[str, dict[str, float]] = {}
    for rxn in reactions:
        stoich: dict[str, float] = {}
        for token in rxn.reactants:
            sp, coef = _parse_stoichiometry_token(token)
            sp = collapse_species(sp)
            if sp in species_seen:
                stoich[sp] = stoich.get(sp, 0.0) - abs(coef)  # 反应物消耗
        for token in rxn.products:
            sp, coef = _parse_stoichiometry_token(token)
            sp = collapse_species(sp)
            if sp in species_seen:
                stoich[sp] = stoich.get(sp, 0.0) + abs(coef)  # 产物生成
        stoichiometry[rxn.reaction_id] = stoich

    # 5. 守恒分组
    from app.species_ontology import build_conservation_groups
    conservation_groups = build_conservation_groups(species_set)

    logger.info(
        "BIOMD 反应图构建完成：model=%s, species=%d, reactions=%d, conservation_groups=%d",
        model_id, len(species_set), len(reactions), len(conservation_groups),
    )

    graph = ReactionGraph(
        species=species_set,
        species_initial=species_initial,
        reactions=reactions,
        stoichiometry=stoichiometry,
        conservation_groups=conservation_groups,
        source_model=model_id,
    )

    # TASK 6：执行三重一致性校验
    from app.model_consistency_validator import validate_reaction_graph_consistency
    consistency_report = validate_reaction_graph_consistency(graph)
    graph.consistency_report = {
        "passed": consistency_report.passed,
        "pathway_integrity": consistency_report.pathway_integrity,
        "conservation_sanity": consistency_report.conservation_sanity,
        "no_phantom_pathway": consistency_report.no_phantom_pathway,
        "summary": consistency_report.summary,
    }
    if not consistency_report.passed:
        logger.error(
            "BIOMD 一致性校验未通过：%s", consistency_report.summary
        )

    return graph


# =============================================================================
# 转换为 ODE 模板可用的 edges
# =============================================================================
def reaction_graph_to_edges(graph: ReactionGraph) -> list[dict]:
    """将 ReactionGraph 转换为 ODE 生成器可用的 edges 列表。

    每个 reaction 转换为一条 edge，mechanism 根据反应物/产物数量推断：
    - 2 reactants → 1 product: binding (mass-action)
    - 1 reactant → 1 product: conversion / phosphorylation (mass-action)
    - 1 reactant → 2 products: dissociation (mass-action)

    TASK 1/2/3/4：
    - 再次过滤非 EGF_EGFR_MAPK 反应（防御性）
    - 对 species 做 canonical reduction（source / target / reaction_equation）
    - 对 canonical 链方向进行校正（防止 collapse 后方向颠倒）
    - 跳过 collapse 导致的自环 / 自释放边
    - 去重边
    - 强制注入 Raf_active → MEK_active → ERK_active
    """
    edges: list[dict] = []
    edge_keys_seen: set[tuple[str, str]] = set()

    for rxn in graph.reactions:
        # 防御性：再次应用允许通路过滤
        if not _reaction_in_allowed_pathway(rxn.reactants, rxn.products, rxn.equation):
            continue

        # canonical 化反应方程，确保 ODE 模板能找到匹配物种
        canonical_equation = _canonicalize_equation(rxn.equation)

        # 推断 mechanism（基于 canonical 后的反应物/产物数量）
        n_reactants = len(rxn.reactants)
        n_products = len(rxn.products)
        if n_reactants >= 2 and n_products == 1:
            mechanism = "binding"
        elif n_reactants == 1 and n_products >= 2:
            mechanism = "dissociation"
        elif n_reactants == 1 and n_products == 1:
            eq_lower = canonical_equation.lower()
            if any(k in eq_lower for k in ("pegfr", "pshc", "praf", "pmek", "perk", "pmapk", "ppmek", "pperf")):
                mechanism = "phosphorylation"
            else:
                mechanism = "conversion"
        else:
            mechanism = "binding"  # 默认 mass-action

        # source / target 推断（canonical）
        # 对酶催化的 binding 反应（一个 reactant 同时是 product），
        # 选择“非酶” reactant 作为 source，确保信号流方向正确。
        reactant_cans = [collapse_species(_parse_stoichiometry_token(t)[0]) for t in rxn.reactants]
        product_cans = [collapse_species(_parse_stoichiometry_token(t)[0]) for t in rxn.products]

        if mechanism == "binding" and len(product_cans) == 1 and product_cans[0] in reactant_cans:
            # 酶 = product，底物/输入 = 另一个 reactant
            enzyme = product_cans[0]
            candidates = [r for r in reactant_cans if r != enzyme]
            if not candidates:
                continue
            # 若在规范链上，优先选上游作为 source
            src_sp = min(
                candidates,
                key=lambda s: _CHAIN_ORDER.get(s, 999),
            )
            tgt_sp = enzyme
        else:
            src = rxn.reactants[0] if rxn.reactants else ""
            tgt = rxn.products[0] if rxn.products else ""
            src_sp, _ = _parse_stoichiometry_token(src)
            tgt_sp, _ = _parse_stoichiometry_token(tgt)
            src_sp = collapse_species(src_sp)
            tgt_sp = collapse_species(tgt_sp)

        # 跳过缺失物种
        if not src_sp or not tgt_sp:
            continue

        # 跳过 collapse 后出现的自环
        if src_sp == tgt_sp:
            continue

        # TASK 3：规范链方向校正（确保信号流从上游流向下游）
        if src_sp in _CHAIN_ORDER and tgt_sp in _CHAIN_ORDER:
            if _CHAIN_ORDER[tgt_sp] < _CHAIN_ORDER[src_sp]:
                src_sp, tgt_sp = tgt_sp, src_sp

        # 跳过 collapse 导致的“自释放”解离边
        # 例：Ras_active → Grb2_SOS_complex + Ras_active（reactant 与 product 重叠）
        if mechanism == "dissociation":
            canonical_reactants = {collapse_species(_parse_stoichiometry_token(t)[0]) for t in rxn.reactants}
            canonical_products = {collapse_species(_parse_stoichiometry_token(t)[0]) for t in rxn.products}
            if canonical_reactants & canonical_products:
                continue

        # 去重（有向边）
        edge_key = (src_sp, tgt_sp)
        if edge_key in edge_keys_seen:
            continue
        edge_keys_seen.add(edge_key)

        edges.append({
            "source": src_sp,
            "target": tgt_sp,
            "interaction": mechanism,
            "mechanism": mechanism,
            "reaction_equation": canonical_equation,
            "reaction_id": rxn.reaction_id,
            "k_forward": rxn.k_forward,
            "k_reverse": rxn.k_reverse,
            "unit_forward": rxn.unit_forward,
            "unit_reverse": rxn.unit_reverse,
            "is_reversible": rxn.is_reversible,
            "source_model": graph.source_model,
        })

    # TASK 3：强制确保 MAPK 链完整
    edges = _ensure_mapk_chain(edges)

    logger.info(
        "BIOMD reaction_graph_to_edges: %d reactions → %d canonical edges",
        len(graph.reactions), len(edges)
    )
    return edges


def get_reaction_graph_for_model(model_id: str) -> ReactionGraph | None:
    """公开入口：根据 model_id 获取反应图。"""
    return build_reaction_graph(model_id)
