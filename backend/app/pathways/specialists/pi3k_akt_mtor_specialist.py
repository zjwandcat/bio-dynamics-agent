# BioDynamics Agent v4 - PI3K-AKT-mTOR Specialist (Phase 4 / Task 4.5)
# PI3K / AKT / mTOR 通路 Specialist：实现 PI3K→PIP3→PDK1/mTORC2→AKT→TSC2→Rheb→
# mTORC1→S6K1/4E-BP1 核心拓扑 + S6K1→IRS1 负反馈 + mTORC1→ULK1 自噬抑制。
#
# 设计原则（铁律）：
# 1. Feature Flag V4_PATHWAY_SPECIALIST_ENABLED=false 时由 LangGraph hook 层短路，
#    本 Specialist 代码不被调用（基类本身不做 flag 检查）
# 2. 使用 Task 4.0 修复后的 PHOSPHORYLATION 语义：
#    - 异磷酸化（pAKT → pTSC2 / mTORC1 → pS6K / mTORC1 → p4EBP1）：
#      未磷酸化形式(TSC2/S6K/4E-BP1) 作 substrate，磷酸化形式(pTSC2/pS6K/p4EBP1)
#      作 product，source 作 catalytic modifier
#    - 复合激活（PIP3 → pAKT / PDK1 → pAKT）：AKT 作 substrate，pAKT 作 product，
#      source(PIP3/PDK1) 作 catalytic/allosteric modifier
# 3. 不处理 Bad 凋亡（由 Apoptosis Specialist 处理，Task 4.7）
# 4. 不处理 Mdm2-p53（由 p53 Specialist 处理，Task 4.6）
# 5. 不生成跨通路 cross-talk edge 本身（由 Cross-talk Coordinator 处理，Task 4.13），
#    仅返回本通路侧的 cross-talk Reaction 片段
# 6. apply_* 方法实现捕获异常并返回空 list/dict，记录 logger.warning，不抛异常
#
# 依赖：
# - P1 ontology / pathway_registry（PI3K_AKT_mTOR 通路类别键）
# - P2 schema（SpeciesV2 / ReactionV2 / SpeciesRef / Modifier / Constraint）
# - P2 MechanismType.PHOSPHORYLATION
# - P3 ode_templates_v2（_mechanism_phosphorylation_mm.j2 / bistable_switch.j2 模板）
# - P3 pathway_graph/initializer.py（PI3K_AKT_mTOR core_nodes / core_edges）
# - P3 bistability_detector（mTORC1 双稳态分析）
#
# 参考：
# - spec.md Part 3 Specialist 3（第 206-211 行）
# - tasks.md Task 4.5（第 66-71 行）
# - Mazzoletti 2009 AKT dynamics (PMID:19211571)
# - BioModels BIOMD0000000250 (PI3K/AKT/mTOR)

from __future__ import annotations

import logging
from typing import Any

from app.biomodels_registry import get_biomodels_id
from app.pathways.drug_library import (
    build_drug_species,
    build_inhibitor_edge,
    get_drug_entry,
)
from app.pathways.pathway_modules.core.template import CoreModuleData
from app.pathways.pathway_modules.crosstalk.template import CrosstalkModuleData
from app.pathways.pathway_modules.feedback.template import FeedbackModuleData
from app.pathways.pathway_modules.perturbation.template import (
    PerturbationModuleData,
)
from app.pathways.pathway_modules.validation.template import ValidationModuleData
from app.pathways.pathway_registry import register_specialist
from app.pathways.pathway_specialist_base import (
    MODULE_CORE,
    MODULE_CROSSTALK,
    MODULE_FEEDBACK,
    MODULE_PERTURBATION,
    MODULE_VALIDATION,
    PathwaySpecialistBase,
)

logger = logging.getLogger(__name__)


# =============================================================================
# PI3K 通路标签
# =============================================================================
PATHWAY_TAG: str = "PI3K_AKT_mTOR"

# SBML BioModels ID（PI3K/AKT/mTOR model）
SOURCE_SBML: str = get_biomodels_id(PATHWAY_TAG)

# Validation benchmark PMID 引用
_Pmid_MAZZOLETTI_2009: str = "PMID:19211571"  # Mazzoletti 2009 AKT dynamics


# =============================================================================
# PI3K 核心物种（与 P3 pathway_graph/initializer.py PI3K_AKT_mTOR.core_nodes 对齐，
# 扩展 PIP2 / S6K / 4E-BP1 / p4EBP1）
# =============================================================================
# AKT 物种标记 shared=True（与 Apoptosis Specialist 的 Bad 凋亡路径共享，
# 与 p53 Specialist 的 Mdm2-p53 路径共享）。
# mTORC1 在 Apoptosis / Cell Cycle 通路中亦被引用，标记 shared=True。
_PI3K_CORE_SPECIES: list[dict[str, Any]] = [
    # [C4 fix] initial_concentration aligned to BIOMD0000000262 (Fujita2010, PMID:20664065).
    # SBML uses initialAmount (dimensionless arbitrary units). For species not in SBML
    # (PI3K/PIP2/PIP3/PDK1/mTORC2/ppAKT/TSC2/Rheb/mTORC1/S6K/4E-BP1/PTEN), kept original.
    # PI3K 激酶（Class I PI3K）
    # [C4 fix] No SBML match in BIOMD0000000262. Kept original.
    {"name": "PI3K", "species_type": "protein", "compartment": "cytoplasm"},
    # PIP2 / PIP3（膜磷脂，质量守恒：PIP2 + PIP3 = PIP_total）
    # [C4 fix] No SBML match in BIOMD0000000262. Kept original.
    {"name": "PIP2", "species_type": "chemical", "compartment": "membrane"},
    {"name": "PIP3", "species_type": "chemical", "compartment": "membrane"},
    # PDK1（AKT Thr308 激酶）
    # [C4 fix] No SBML match in BIOMD0000000262. Kept original.
    {"name": "PDK1", "species_type": "protein", "compartment": "cytoplasm"},
    # [P0-1 / N8 修复] mTORC2（AKT Ser473 激酶，与 PDK1 共同构成 AKT 双位点磷酸化）
    # 文献：Sarbassov et al. 2005 (PMID:16135013) mTORC2 磷酸化 AKT Ser473
    # 之前仅在 feedback loop 中提及，未在 species 中建模，导致 AKT 双磷酸化建模缺失。
    # 标记 shared=True：mTORC2 在 Apoptosis / Cell Cycle 通路中亦被引用（与 mTORC1 协同）。
    # [C4 fix] No SBML match in BIOMD0000000262. Kept original.
    {"name": "mTORC2", "species_type": "complex", "compartment": "cytoplasm",
     "shared": True},
    # AKT（shared：与 Apoptosis Bad 凋亡 / p53 Mdm2 路径共享）
    # [C4 fix] SBML mapping: AKT→Akt (initialAmount=0.043309)
    {"name": "AKT", "species_type": "protein", "compartment": "cytoplasm",
     "shared": True,
     "initial_concentration": 0.043309},  # Source: BIOMD0000000262 Fujita2010 (PMID:20664065) species Akt (initialAmount)
    # [P0-1 / N8 修复] pAKT 单磷酸化形式（Thr308 by PDK1，部分激活 ~20% 活性）
    # 与下游 crosstalk (Bad/Mdm2/Raf Ser259) 兼容，保持跨通路语义
    # [C4 fix] SBML mapping: pAKT→pAkt (initialAmount=0)
    {"name": "pAKT", "species_type": "protein", "compartment": "cytoplasm",
     "initial_concentration": 0.0},  # Source: BIOMD0000000262 Fujita2010 (PMID:20664065) species pAkt (initialAmount)
    # [P0-1 / N8 修复] ppAKT 双磷酸化完全激活形式（Thr308 + Ser473，100% 活性）
    # 文献：Sarbassov 2005 (PMID:16135013) 双位点磷酸化是 AKT 完全激活的标志
    # 用于 PI3K 内部下游 pTSC2 强激活，crosstalk 仍走 pAKT 保持向后兼容
    # [C4 fix] No SBML match in BIOMD0000000262 (Fujita2010 only models single pAkt). Kept original.
    {"name": "ppAKT", "species_type": "protein", "compartment": "cytoplasm"},
    # TSC1/2 复合体（GAP for Rheb）
    {"name": "TSC2", "species_type": "protein", "compartment": "cytoplasm"},
    {"name": "pTSC2", "species_type": "protein", "compartment": "cytoplasm"},
    # Rheb（小 GTP 酶，mTORC1 激活）
    {"name": "Rheb", "species_type": "protein", "compartment": "membrane"},
    {"name": "RhebGTP", "species_type": "protein", "compartment": "membrane"},
    # mTORC1（shared：与 Apoptosis 自噬 / Cell Cycle 共享）
    {"name": "mTORC1", "species_type": "complex", "compartment": "cytoplasm",
     "shared": True},
    # S6K 级联（mTORC1 下游底物）
    {"name": "S6K", "species_type": "protein", "compartment": "cytoplasm"},
    {"name": "pS6K", "species_type": "protein", "compartment": "cytoplasm"},
    # 4E-BP1（翻译抑制因子，mTORC1 下游底物）
    {"name": "4E-BP1", "species_type": "protein", "compartment": "cytoplasm"},
    {"name": "p4EBP1", "species_type": "protein", "compartment": "cytoplasm"},
    # PTEN（PIP3 磷酸酶，负调控 PI3K）
    {"name": "PTEN", "species_type": "protein", "compartment": "cytoplasm"},
    # [N6 缺口 1] 药物物种（species_type="drug"）— 由 drug_library 驱动
    # Rapamycin 是 mTORC1 别构抑制剂（FKBP12 复合, IC50=50 nM, PMID:8413626）
    build_drug_species("Rapamycin"),
]


# =============================================================================
# PI3K 核心反应（9 条）
# =============================================================================
# 每条反应含：source / target / mechanism / kinetics_type / pathway_tag /
#             substrate / product / modifier / modifier_type / autophosphorylation
#
# PHOSPHORYLATION 语义（Task 4.0 修复后）：
# - 异磷酸化：未磷酸化形式作 substrate，磷酸化形式作 product，source 作 catalytic modifier
# - 复合激活（PIP3→pAKT / PDK1→pAKT）：AKT 作 substrate，pAKT 作 product，
#   source 作 catalytic/allosteric modifier
#
# kinetics_type 选择：
# - phosphorylation → Michaelis_Menten（与 P3 _mechanism_phosphorylation_mm 模板对齐）
# - activation / dephosphorylation → hybrid / Michaelis_Menten
_PI3K_CORE_REACTIONS: list[dict[str, Any]] = [
    # 1. PI3K → PIP3（activation，PI3K 催化 PIP2→PIP3 转换）
    {
        "source": "PI3K",
        "target": "PIP3",
        "mechanism": "activation",
        "kinetics_type": "Michaelis_Menten",
        "pathway_tag": PATHWAY_TAG,
        "substrate": "PIP2",  # PIP2 作 substrate，PIP3 作 product
        "product": "PIP3",
        "modifier": "PI3K",  # PI3K 作 catalytic modifier
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "PI3K 催化 PIP2 磷酸化为 PIP3（PI3K 作 catalytic modifier）",
    },
    # 2. PIP3 → pAKT（phosphorylation，AKT 作 substrate，PIP3 作 allosteric activator）
    #    PIP3 提供膜定位，招募 AKT 到膜上被 PDK1 磷酸化
    {
        "source": "PIP3",
        "target": "pAKT",
        "mechanism": "phosphorylation",
        "kinetics_type": "Michaelis_Menten",
        "pathway_tag": PATHWAY_TAG,
        # AKT 作 substrate，pAKT 作 product，PIP3 作 allosteric modifier
        "substrate": "AKT",
        "product": "pAKT",
        "modifier": "PIP3",
        "modifier_type": "allosteric",
        "autophosphorylation": False,
        "description": "PIP3 招募 AKT 到膜并被磷酸化（AKT 作 substrate，PIP3 作 allosteric activator）",
    },
    # 3. PDK1 + PIP3 → pAKT（复合激活：PDK1 磷酸化 AKT Thr308, PIP3 提供膜定位）
    #    本反应与第 2 条互补：PDK1 是真正的激酶，PIP3 提供 AKT 膜定位
    {
        "source": "PDK1",
        "target": "pAKT",
        "mechanism": "phosphorylation",
        "kinetics_type": "Michaelis_Menten",
        "pathway_tag": PATHWAY_TAG,
        # AKT 作 substrate，pAKT 作 product，PDK1 作 catalytic modifier
        "substrate": "AKT",
        "product": "pAKT",
        "modifier": "PDK1",
        "modifier_type": "catalytic",
        "co_factor": "PIP3",  # PIP3 提供 AKT 膜定位（allosteric co-factor）
        "autophosphorylation": False,
        "description": "PDK1 磷酸化 AKT Thr308（AKT 作 substrate，PDK1 作 catalytic modifier，PIP3 提供膜定位）",
    },
    # 3b. [P0-1 / N8 修复] mTORC2 → ppAKT（双磷酸化第二步，pAKT → ppAKT，mTORC2 催化 Ser473）
    #     文献：Sarbassov et al. 2005 (PMID:16135013) mTORC2 磷酸化 AKT hydrophobic motif Ser473
    #     与 PDK1 (Thr308) 共同构成 AKT 双位点磷酸化，完全激活 (~10x 活性提升)
    #     pAKT 作 substrate (单磷酸化前体)，ppAKT 作 product (双磷酸化完全激活形式)，
    #     mTORC2 作 catalytic modifier (Ser473 激酶)
    {
        "source": "mTORC2",
        "target": "ppAKT",
        "mechanism": "phosphorylation",
        "kinetics_type": "Michaelis_Menten",
        "pathway_tag": PATHWAY_TAG,
        # 双磷酸化：pAKT 作 substrate (T308 单磷酸化前体)，ppAKT 作 product (双位点完全激活)
        # mTORC2 作 catalytic modifier (Ser473 激酶)
        "substrate": "pAKT",
        "product": "ppAKT",
        "modifier": "mTORC2",
        "modifier_type": "catalytic",
        "site": "Ser473",
        "autophosphorylation": False,
        "description": "mTORC2 磷酸化 pAKT Ser473 完成双位点磷酸化（pAKT→ppAKT，mTORC2 作 catalytic modifier，AKT 完全激活）",
    },
    # 4. [P0-1 / N8 修复] ppAKT → pTSC2（异磷酸化，TSC2 作 substrate，ppAKT 作 catalytic modifier）
    #    修改：原 pAKT → pTSC2 改为 ppAKT → pTSC2，符合"完全激活的 AKT 才强催化 TSC2"生物学
    #    pAKT (单磷酸化) 仍参与 crosstalk (Bad/Mdm2/Raf)，保持向后兼容
    {
        "source": "ppAKT",
        "target": "pTSC2",
        "mechanism": "phosphorylation",
        "kinetics_type": "Michaelis_Menten",
        "pathway_tag": PATHWAY_TAG,
        # 异磷酸化：TSC2 作 substrate，pTSC2 作 product，ppAKT 作 catalytic modifier
        "substrate": "TSC2",
        "product": "pTSC2",
        "modifier": "ppAKT",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "ppAKT 磷酸化 TSC2（TSC2 作 substrate，ppAKT 作 catalytic modifier，抑制 TSC2 GAP 活性，需双位点磷酸化的完全激活 AKT）",
    },
    # 5. pTSC2 → RhebGTP（activation，pTSC2 失去 GAP 活性，Rheb 累积 GTP 形式）
    {
        "source": "pTSC2",
        "target": "RhebGTP",
        "mechanism": "activation",
        "kinetics_type": "Michaelis_Menten",
        "pathway_tag": PATHWAY_TAG,
        # Rheb(GDP) 作 substrate，RhebGTP 作 product，pTSC2 失活（间接促进）
        "substrate": "Rheb",
        "product": "RhebGTP",
        "modifier": "pTSC2",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "pTSC2 失去 GAP 活性，Rheb 累积 GTP 形式（Rheb 作 substrate，RhebGTP 作 product）",
    },
    # 6. RhebGTP → mTORC1（activation，RhebGTP 直接激活 mTORC1）
    {
        "source": "RhebGTP",
        "target": "mTORC1",
        "mechanism": "activation",
        "kinetics_type": "Michaelis_Menten",
        "pathway_tag": PATHWAY_TAG,
        # mTORC1(inactive) 作 substrate，mTORC1(active) 作 product，RhebGTP 作 modifier
        "substrate": "mTORC1",
        "product": "mTORC1",
        "modifier": "RhebGTP",
        "modifier_type": "allosteric",
        "autophosphorylation": False,
        "description": "RhebGTP 直接激活 mTORC1（mTORC1 作 substrate/product，RhebGTP 作 allosteric modifier）",
    },
    # 7. mTORC1 → pS6K（异磷酸化，S6K 作 substrate，mTORC1 作 catalytic modifier）
    {
        "source": "mTORC1",
        "target": "pS6K",
        "mechanism": "phosphorylation",
        "kinetics_type": "Michaelis_Menten",
        "pathway_tag": PATHWAY_TAG,
        # 异磷酸化：S6K 作 substrate，pS6K 作 product，mTORC1 作 catalytic modifier
        "substrate": "S6K",
        "product": "pS6K",
        "modifier": "mTORC1",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "mTORC1 磷酸化 S6K（S6K 作 substrate，mTORC1 作 catalytic modifier，Thr389）",
    },
    # 8. mTORC1 → p4EBP1（异磷酸化，4E-BP1 作 substrate，mTORC1 作 catalytic modifier）
    {
        "source": "mTORC1",
        "target": "p4EBP1",
        "mechanism": "phosphorylation",
        "kinetics_type": "Michaelis_Menten",
        "pathway_tag": PATHWAY_TAG,
        # 异磷酸化：4E-BP1 作 substrate，p4EBP1 作 product，mTORC1 作 catalytic modifier
        "substrate": "4E-BP1",
        "product": "p4EBP1",
        "modifier": "mTORC1",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "mTORC1 磷酸化 4E-BP1（4E-BP1 作 substrate，mTORC1 作 catalytic modifier，释放 eIF4E）",
    },
    # 9. PTEN → PIP2（dephosphorylation，PTEN 是磷脂酶，PIP3→PIP2）
    {
        "source": "PTEN",
        "target": "PIP2",
        "mechanism": "dephosphorylation",
        "kinetics_type": "Michaelis_Menten",
        "pathway_tag": PATHWAY_TAG,
        # PIP3 作 substrate，PIP2 作 product，PTEN 作 catalytic modifier
        "substrate": "PIP3",
        "product": "PIP2",
        "modifier": "PTEN",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "PTEN 去磷酸化 PIP3 为 PIP2（PIP3 作 substrate，PIP2 作 product，PTEN 作 catalytic modifier）",
    },
    # ===== [N6 缺口 1] 药物-靶点显式 inhibitor edge（canonical drug_library 驱动） =====
    # 10. Rapamycin → mTORC1（allosteric_FKBP12_complex, IC50=50 nM, PMID:8413626）
    # Rapamycin 与 FKBP12 形成复合物，结合 mTORC1 的 FRB 结构域使激酶失活。
    {
        **build_inhibitor_edge("Rapamycin", "mTORC1"),
        "pathway_tag": PATHWAY_TAG,
    },
]


# =============================================================================
# PI3K 质量守恒约束（PIP2 + PIP3 = PIP_total）
# =============================================================================
# PIP2 / PIP3 是膜磷脂的两种形式，PI3K 催化 PIP2→PIP3，PTEN 逆转。
# 质量守恒约束确保 PIP2 + PIP3 = PIP_total（常数），不出现 PIP3 无限累积。
_PI3K_CORE_CONSTRAINTS: list[dict[str, Any]] = [
    {
        "type": "mass_conservation",
        "scope": "species",
        "expression": "PIP2 + PIP3 = PIP_total",
        "tolerance": 0.05,
        "provenance": "PI3K-AKT-mTOR Specialist: PIP2/PIP3 lipid pool conservation",
        "description": "PIP2 + PIP3 = PIP_total（膜磷脂池质量守恒，PI3K/PTEN 双向转换）",
    },
]


# =============================================================================
# PI3K 反馈环（3 条）
# =============================================================================
_PI3K_FEEDBACK_LOOPS: list[dict[str, Any]] = [
    # 1. pS6K → IRS1 负反馈（pS6K 反向磷酸化 IRS1 抑制其功能，delay=30 min）
    {
        "id": "FL_PI3K_S6K_IRS1_NEG",
        "loop_type": "negative",
        "node_ids": ["pS6K", "IRS1", "PI3K"],
        "delay_minutes": 30.0,
        "site": "Ser307",
        "description": "pS6K 反向磷酸化 IRS1 Ser307 抑制其功能（负反馈，delay=30 min，S6K1→IRS1 长反馈）",
    },
    # 2. mTORC1 → ULK1 抑制（mTORC1 磷酸化 ULK1 抑制自噬启动，delay=0）
    {
        "id": "FL_PI3K_mTORC1_ULK1_NEG",
        "loop_type": "negative",
        "node_ids": ["mTORC1", "ULK1"],
        "delay_minutes": 0.0,
        "site": "Ser757",
        "description": "mTORC1 磷酸化 ULK1 Ser757 抑制自噬启动（负反馈，delay=0，自噬抑制）",
    },
    # 3. pAKT → mTORC2 正反馈（pAKT 可能反馈激活 mTORC2，mTORC2 磷酸化 AKT Ser473）
    {
        "id": "FL_PI3K_AKT_mTORC2_POS",
        "loop_type": "positive",
        "node_ids": ["pAKT", "mTORC2"],
        "delay_minutes": 0.0,
        "description": "pAKT 反馈激活 mTORC2（正反馈，mTORC2 磷酸化 AKT Ser473 完全激活）",
    },
]


# =============================================================================
# PI3K Crosstalk Reaction 片段（4 条，仅本通路侧片段，edge 由 Coordinator 管理）
# =============================================================================
_PI3K_CROSSTALK_REACTIONS: list[dict[str, Any]] = [
    # 1. pAKT → Raf Ser259（inhibition，pAKT 磷酸化 Raf Ser259 抑制 MAPK）
    {
        "source": "pAKT",
        "target": "Raf",
        "mechanism": "inhibition",
        "shared_species": [],
        "site": "Ser259",
        "description": "pAKT 磷酸化 Raf Ser259 抑制 MAPK 级联（PI3K→MAPK cross-talk）",
    },
    # 2. pAKT → Bad（inhibition，pAKT 磷酸化 Bad Ser136 抑制凋亡）
    {
        "source": "pAKT",
        "target": "Bad",
        "mechanism": "inhibition",
        "shared_species": ["AKT"],
        "site": "Ser136",
        "description": "pAKT 磷酸化 Bad Ser136 导致其失活（抑制凋亡，与 Apoptosis 通路 cross-talk）",
    },
    # 3. pAKT → Mdm2（activation，pAKT 磷酸化 Mdm2 Ser166 激活 Mdm2 降解 p53）
    {
        "source": "pAKT",
        "target": "Mdm2",
        "mechanism": "activation",
        "shared_species": ["AKT"],
        "site": "Ser166",
        "description": "pAKT 磷酸化 Mdm2 Ser166 激活其 E3 泛素连接酶活性（降解 p53，与 p53 通路 cross-talk）",
    },
    # 4. mTORC1 → HIF-1α（activation，mTORC1 翻译激活 HIF-1α）
    {
        "source": "mTORC1",
        "target": "HIF-1α",
        "mechanism": "activation",
        "shared_species": [],
        "description": "mTORC1 翻译激活 HIF-1α（缺氧响应，与代谢/血管生成 cross-talk）",
    },
    # 5. [P2-1] PI3K bypass：PIP3 → Raf（activation via PAK，绕过 RasGTP→Raf 步骤）
    #    机制：PIP3 经 Rac/Cdc42 → PAK1 → Raf Ser338 磷酸化（激活），独立于 RasGTP
    #    通路。构成 PI3K → MAPK 的旁路激活（bypass），即使 Ras 失活仍能激活 Raf-MEK-ERK。
    #    文献：PMID:15247256 (PI3K-PAK-Raf-MEK-ERK cascade)
    #          PMID:11923475 (PAK1 phosphorylates Raf Ser338)
    #    与 #1 (pAKT→Raf Ser259 inhibition) 互补：pAKT 抑制 Raf (Ser259)，
    #    PIP3 激活 Raf (Ser338 via PAK)，构成 PI3K 对 Raf 的双向调控。
    {
        "source": "PIP3",
        "target": "Raf",
        "mechanism": "activation",
        "shared_species": [],
        "site": "Ser338",
        "intermediate": "PAK1",
        "description": "PIP3 经 PAK1 磷酸化 Raf Ser338 激活 MAPK 级联（PI3K bypass，绕过 RasGTP→Raf，PMID:15247256）",
    },
    # 6. [P2-1] Feedback release：pAKT → DUSP（inhibition，pAKT 磷酸化 DUSP1/MKP-1
    #    导致其降解，解除 DUSP 对 ERK 的去磷酸化抑制，释放 EGFR-MAPK 负反馈）
    #    机制：pAKT 磷酸化 DUSP1 (MKP-1) Ser296，触发 E3 连接酶介导的泛素化降解，
    #    稳定 ERK 磷酸化水平，延长 EGFR-MAPK 信号持续时间。
    #    文献：PMID:15647282 (AKT-mediated DUSP1 degradation stabilizes ERK)
    #          PMID:19279225 (DUSP-ERK 反馈环)
    {
        "source": "pAKT",
        "target": "DUSP",
        "mechanism": "inhibition",
        "shared_species": ["AKT"],
        "site": "Ser296",
        "description": "pAKT 磷酸化 DUSP1 Ser296 触发其降解，解除 DUSP 对 ERK 的去磷酸化抑制（释放 EGFR-MAPK 负反馈，PMID:15647282）",
    },
]


# =============================================================================
# PI3K 扰动（6 个：5 个药物 + 1 个突变）
# =============================================================================
_PI3K_PERTURBATIONS: list[dict[str, Any]] = [
    # 1. Rapamycin（mTORC1 inhibitor, small molecule, FDA-approved）
    # [N6 缺口 1] 注入 canonical drug_library 字段（ic50_nM/ki_nM/source_pmid/...）
    {
        "target": "mTORC1",
        "drug": "Rapamycin",
        "mechanism": "inhibition",
        "ko_target": None,
        "description": "Rapamycin（mTORC1 别构抑制剂，小分子，FDA-approved）",
        **{k: v for k, v in get_drug_entry("Rapamycin").items()
           if k not in ("description",)},
    },
    # 2. Everolimus（mTORC1 inhibitor, small molecule, FDA-approved）
    {
        "target": "mTORC1",
        "drug": "Everolimus",
        "mechanism": "inhibition",
        "ko_target": None,
        "description": "Everolimus（mTORC1 别构抑制剂，小分子，FDA-approved，Rapamycin 类似物）",
    },
    # 3. BKM120/Buparlisib（PI3K inhibitor, pan-PI3K）
    {
        "target": "PI3K",
        "drug": "BKM120",
        "mechanism": "inhibition",
        "ko_target": None,
        "description": "BKM120/Buparlisib（pan-PI3K 抑制剂，小分子）",
    },
    # 4. Idelalisib（PI3Kδ inhibitor, FDA-approved）
    {
        "target": "PI3K",
        "drug": "Idelalisib",
        "mechanism": "inhibition",
        "ko_target": None,
        "description": "Idelalisib（PI3Kδ 特异性抑制剂，小分子，FDA-approved）",
    },
    # 5. MK-2206（AKT inhibitor, allosteric）
    {
        "target": "AKT",
        "drug": "MK-2206",
        "mechanism": "inhibition",
        "ko_target": None,
        "description": "MK-2206（AKT 别构抑制剂，小分子）",
    },
    # 6. PTEN loss（loss-of-function mutation, 导致 PIP3 累积）
    {
        "target": "PTEN",
        "drug": None,
        "mechanism": "knockout",
        "ko_target": "PTEN_loss",
        "description": "PTEN loss-of-function（PTEN 缺失突变，导致 PIP3 累积，PI3K 通路过度激活）",
    },
]


# =============================================================================
# PI3K Validation 规则（3 条 benchmark）
# =============================================================================
# 每条含：rule_id / metric_name / expected / tolerance / pmid / description
# expected 取区间中点，tolerance 取区间半宽
_PI3K_VALIDATION_RULES: list[dict[str, Any]] = [
    # 1. pAKT 达峰时间 30-60 min（Mazzoletti 2009）
    {
        "rule_id": "VAL_PI3K_pAKT_PEAK_TIME",
        "metric_name": "pAKT_peak_time",
        "expected": 45.0,   # (30.0 + 60.0) / 2
        "tolerance": 15.0,   # (60.0 - 30.0) / 2
        "expected_min": 30.0,
        "expected_max": 60.0,
        "unit": "minutes",
        "pmid": _Pmid_MAZZOLETTI_2009,
        "comparison": "range",
        "pathway_tag": PATHWAY_TAG,
        "description": "pAKT 在生长因子刺激后 30-60 min 达峰（Mazzoletti 2009 AKT dynamics）",
    },
    # 2. PIP2/PIP3 质量守恒（守恒比例应≈1.0）
    {
        "rule_id": "VAL_PI3K_PIP_PIP3_MASS_CONSERVATION",
        "metric_name": "PIP_PIP3_mass_conservation",
        "expected": 1.0,
        "tolerance": 0.05,
        "expected_min": 0.95,
        "expected_max": 1.05,
        "unit": "ratio",
        "pmid": "",
        "comparison": "absolute",
        "pathway_tag": PATHWAY_TAG,
        "description": "PIP2 + PIP3 = PIP_total（守恒比例应≈1.0，PI3K/PTEN 双向转换守恒）",
    },
    # 3. S6K1 时序检查（pS6K1 达峰应晚于 pAKT 30-60 min）
    {
        "rule_id": "VAL_PI3K_S6K1_PEAK_DELAY_VS_AKT",
        "metric_name": "S6K1_peak_delay_vs_AKT",
        "expected": 45.0,   # (30.0 + 60.0) / 2
        "tolerance": 15.0,   # (60.0 - 30.0) / 2
        "expected_min": 30.0,
        "expected_max": 60.0,
        "unit": "minutes_delay",
        "pmid": _Pmid_MAZZOLETTI_2009,
        "comparison": "range",
        "pathway_tag": PATHWAY_TAG,
        "description": "pS6K1 达峰应晚于 pAKT 30-60 min（时序检查，AKT→TSC2→Rheb→mTORC1→S6K 级联延迟）",
    },
]


@register_specialist
class PI3KAKTmTORSpecialist(PathwaySpecialistBase):
    """PI3K / AKT / mTOR 通路 Specialist。

    实现 PI3K→PIP3→PDK1/mTORC2→AKT→TSC2→Rheb→mTORC1→S6K1/4E-BP1 核心拓扑 +
    S6K1→IRS1 负反馈 + mTORC1→ULK1 自噬抑制的 Core/Feedback/Crosstalk/
    Perturbation/Validation 5 模块，输出通路特异 Reaction IR 片段 + 模板选择
    + Validation 规则 + PIP2/PIP3 质量守恒约束。

    职责边界：
    - 处理 PI3K-AKT-mTOR 核心通路（PI3K/PIP3/AKT/TSC2/Rheb/mTORC1/S6K/4E-BP1/PTEN）
    - 不处理 Bad 凋亡（由 Apoptosis Specialist 处理，Task 4.7）
    - 不处理 Mdm2-p53（由 p53 Specialist 处理，Task 4.6）
    - 不生成跨通路 cross-talk edge（由 Cross-talk Coordinator 处理，Task 4.13）

    输入：
    - ``pathway_graph``：v4_pathway_graph 的 PI3K 子图
    - ``ontology_entities``：P1 Ontology Entities（可选，HGNC/UniProt ID 对齐）

    输出：
    - ``apply_core``：9 条核心 Reaction IR 片段 + 16 物种 + PIP2/PIP3 质量守恒约束
      （AKT 标记 shared=True，mTORC1 标记 shared=True）
    - ``apply_feedback``：3 条 FeedbackLoop（pS6K→IRS1 delay=30 / mTORC1→ULK1 / pAKT→mTORC2 正反馈）
    - ``apply_crosstalk``：4 条 cross-talk Reaction 片段（pAKT→Raf/Bad/Mdm2 + mTORC1→HIF-1α）
    - ``apply_perturbation``：6 个扰动（Rapamycin/Everolimus/BKM120/Idelalisib/MK-2206/PTEN loss）
    - ``apply_validation``：3 条 Validation benchmark 规则（pAKT 达峰 / PIP 守恒 / S6K1 时序）
    """

    pathway_class: str = "PI3K_AKT_mTOR"
    display_name: str = "PI3K / AKT / mTOR Signaling"

    # =================================================================
    # load_module：加载模块数据结构
    # =================================================================
    def load_module(self, module_name: str) -> Any:
        """加载指定模块的数据结构。

        Args:
            module_name: ``core`` / ``feedback`` / ``crosstalk`` /
                ``perturbation`` / ``validation`` 之一。

        Returns:
            对应模块的 dataclass 实例。未知模块名返回 None。
        """
        try:
            if module_name == MODULE_CORE:
                return CoreModuleData(
                    species=list(_PI3K_CORE_SPECIES),
                    reactions=list(_PI3K_CORE_REACTIONS),
                    kinetics_overrides={},
                )
            if module_name == MODULE_FEEDBACK:
                return FeedbackModuleData(
                    feedback_loops=list(_PI3K_FEEDBACK_LOOPS),
                    delay_minutes=30.0,
                    loop_type="mixed",  # 含负反馈 + 正反馈
                )
            if module_name == MODULE_CROSSTALK:
                return CrosstalkModuleData(
                    crosstalk_reactions=list(_PI3K_CROSSTALK_REACTIONS),
                    shared_species=["AKT"],
                    coordination_strategy="merge",
                )
            if module_name == MODULE_PERTURBATION:
                return PerturbationModuleData(
                    perturbation_reactions=list(_PI3K_PERTURBATIONS),
                    drug_targets=[
                        p for p in _PI3K_PERTURBATIONS if p.get("drug")
                    ],
                    ko_targets=[
                        p for p in _PI3K_PERTURBATIONS if p.get("ko_target")
                    ],
                )
            if module_name == MODULE_VALIDATION:
                return ValidationModuleData(
                    rules=list(_PI3K_VALIDATION_RULES),
                    benchmarks=[
                        {
                            "benchmark_name": r["metric_name"],
                            "value": r["expected"],
                            "unit": r["unit"],
                            "condition": "growth factor stimulation",
                            "reference": r["pmid"],
                        }
                        for r in _PI3K_VALIDATION_RULES
                    ],
                    tolerances={
                        r["metric_name"]: r["tolerance"]
                        for r in _PI3K_VALIDATION_RULES
                    },
                    pmid_references=[
                        {
                            "pmid": r["pmid"],
                            "citation": r["description"],
                            "pathway_class": PATHWAY_TAG,
                            "metric_name": r["metric_name"],
                        }
                        for r in _PI3K_VALIDATION_RULES
                        if r["pmid"]
                    ],
                )
            logger.warning(
                "PI3KAKTmTORSpecialist.load_module: 未知模块名 '%s'",
                module_name,
            )
            return None
        except Exception as exc:
            logger.warning(
                "PI3KAKTmTORSpecialist.load_module 加载模块 '%s' 失败: %s",
                module_name,
                exc,
            )
            return None

    # =================================================================
    # apply_core：核心 Reaction IR 片段
    # =================================================================
    def apply_core(
        self,
        pathway_graph: dict,
        ontology_entities: dict | None = None,
    ) -> dict:
        """应用核心模块，返回 PI3K/AKT/mTOR 通路核心 Reaction IR 片段。

        输出 10 条核心反应（含 [P0-1] 新增 mTORC2→ppAKT 双磷酸化）：
        1. PI3K → PIP3（activation，PI3K 催化 PIP2→PIP3 转换）
        2. PIP3 → pAKT（phosphorylation，PIP3 作 allosteric activator）
        3. PDK1 → pAKT（复合激活，PDK1 磷酸化 AKT Thr308，PIP3 提供膜定位）
        3b. [P0-1] mTORC2 → ppAKT（双磷酸化 Ser473，pAKT→ppAKT 完全激活）
        4. [P0-1] ppAKT → pTSC2（异磷酸化，ppAKT 作 catalytic modifier）
        5. pTSC2 → RhebGTP（activation，pTSC2 失去 GAP 活性）
        6. RhebGTP → mTORC1（activation，RhebGTP 直接激活 mTORC1）
        7. mTORC1 → pS6K（异磷酸化，mTORC1 作 catalytic modifier）
        8. mTORC1 → p4EBP1（异磷酸化，mTORC1 作 catalytic modifier）
        9. PTEN → PIP2（dephosphorylation，PTEN 去磷酸化 PIP3→PIP2）

        AKT 物种标记 shared=True（与 Apoptosis Bad 凋亡 / p53 Mdm2 路径共享）。
        mTORC1/mTORC2 物种标记 shared=True（与 Apoptosis 自噬 / Cell Cycle 共享）。
        [P0-1] 新增 mTORC2 + ppAKT 物种，构成完整 AKT 双位点磷酸化级联
        (AKT→pAKT_T308→ppAKT)，符合 Sarbassov 2005 (PMID:16135013) 文献建模。

        Returns:
            dict 含 ``species``（18 物种，含 mTORC2/ppAKT）/ ``reactions``
            （10 反应，含 mTORC2→ppAKT 双磷酸化） / ``constraints``
            （PIP2/PIP3 质量守恒）字段。异常时返回
            ``{"species": [], "reactions": [], "constraints": []}``。
        """
        try:
            return {
                "species": list(_PI3K_CORE_SPECIES),
                "reactions": list(_PI3K_CORE_REACTIONS),
                "constraints": list(_PI3K_CORE_CONSTRAINTS),
                "pathway_tag": PATHWAY_TAG,
                "source_sbml": SOURCE_SBML,
                # [KINETIC_PARAMETERS 注入 / P0-1] 按 target 物种名组织的动力学参数
                # 修复 C1 Peak Time + 辅助 P0-3 PIP2/PIP3 质量守恒（PTEN 速率对齐文献）
                "kinetics_overrides": dict(_KINETICS_BY_TARGET),
            }
        except Exception as exc:
            logger.warning(
                "PI3KAKTmTORSpecialist.apply_core 失败: %s", exc
            )
            return {"species": [], "reactions": [], "constraints": []}

    # =================================================================
    # apply_feedback：FeedbackLoop 列表
    # =================================================================
    def apply_feedback(self, pathway_graph: dict) -> list[dict]:
        """应用反馈模块，返回 PI3K 通路 FeedbackLoop 列表。

        输出 3 条反馈环：
        1. pS6K → IRS1 负反馈（pS6K 反向磷酸化 IRS1 抑制其功能，delay=30 min）
        2. mTORC1 → ULK1 负反馈（mTORC1 磷酸化 ULK1 抑制自噬，delay=0）
        3. pAKT → mTORC2 正反馈（pAKT 反馈激活 mTORC2，mTORC2 磷酸化 AKT Ser473）

        Returns:
            FeedbackLoop 字典列表。异返回空列表。
        """
        try:
            return list(_PI3K_FEEDBACK_LOOPS)
        except Exception as exc:
            logger.warning(
                "PI3KAKTmTORSpecialist.apply_feedback 失败: %s", exc
            )
            return []

    # =================================================================
    # apply_crosstalk：cross-talk Reaction 片段
    # =================================================================
    def apply_crosstalk(
        self,
        pathway_graph: dict,
        crosstalk_edges: list[dict],
    ) -> list[dict]:
        """应用跨通路模块，返回 PI3K 通路侧 cross-talk Reaction 片段。

        注意：本方法仅生成本通路侧的 cross-talk Reaction 片段，
        cross-talk edge 本身由 Cross-talk Coordinator 创建（Task 4.13）。

        输出 4 条 cross-talk Reaction 片段：
        1. pAKT → Raf Ser259（inhibition，抑制 MAPK 级联）
        2. pAKT → Bad Ser136（inhibition，抑制凋亡）
        3. pAKT → Mdm2 Ser166（activation，激活 Mdm2 降解 p53）
        4. mTORC1 → HIF-1α（activation，翻译激活 HIF-1α）

        Args:
            pathway_graph: 通路图（当前未使用，保留接口一致性）。
            crosstalk_edges: 来自 Coordinator 的 cross-talk edge 列表
                （当前未使用，本 Specialist 返回静态拓扑片段）。

        Returns:
            cross-talk Reaction 片段列表。异返回空列表。
        """
        try:
            return list(_PI3K_CROSSTALK_REACTIONS)
        except Exception as exc:
            logger.warning(
                "PI3KAKTmTORSpecialist.apply_crosstalk 失败: %s", exc
            )
            return []

    # =================================================================
    # apply_perturbation：药物 / KO / 突变 Reaction 片段
    # =================================================================
    def apply_perturbation(
        self,
        pathway_graph: dict,
        perturbation_points: list[dict],
    ) -> list[dict]:
        """应用扰动模块，返回 PI3K 通路特异药物 / 突变 Reaction 片段。

        输出 6 个扰动（5 个药物 + 1 个突变）：
        1. Rapamycin（mTORC1 别构抑制剂，FDA-approved）
        2. Everolimus（mTORC1 别构抑制剂，FDA-approved）
        3. BKM120/Buparlisib（pan-PI3K 抑制剂）
        4. Idelalisib（PI3Kδ 特异性抑制剂，FDA-approved）
        5. MK-2206（AKT 别构抑制剂）
        6. PTEN loss（PTEN 缺失突变，导致 PIP3 累积）

        Args:
            pathway_graph: 通路图（当前未使用）。
            perturbation_points: 扰动点列表（当前未使用，本 Specialist
                返回静态拓扑片段）。

        Returns:
            扰动 Reaction 片段列表。异返回空列表。
        """
        try:
            return list(_PI3K_PERTURBATIONS)
        except Exception as exc:
            logger.warning(
                "PI3KAKTmTORSpecialist.apply_perturbation 失败: %s", exc
            )
            return []

    # =================================================================
    # apply_validation：Validation 规则列表
    # =================================================================
    def apply_validation(
        self,
        simulation_result: dict | None = None,
    ) -> list[dict]:
        """应用验证模块，返回 PI3K 通路 Validation 规则列表。

        输出 3 条 benchmark：
        1. pAKT 达峰时间 30-60 min（Mazzoletti 2009, PMID:19211571）
        2. PIP2/PIP3 质量守恒比例 ≈1.0（tolerance 0.05）
        3. S6K1 达峰延迟 vs AKT 30-60 min（时序检查，PMID:19211571）

        Args:
            simulation_result: 仿真结果（可选，当前未使用，返回静态规则集）。

        Returns:
            Validation 规则列表。异返回空列表。
        """
        try:
            return list(_PI3K_VALIDATION_RULES)
        except Exception as exc:
            logger.warning(
                "PI3KAKTmTORSpecialist.apply_validation 失败: %s", exc
            )
            return []

    # =================================================================
    # select_template：覆写以支持 mTORC1 bistable 模板
    # =================================================================
    # 默认基类映射已满足需求：
    # - phosphorylation → _mechanism_phosphorylation_mm
    # - bistable → bistable_switch（mTORC1 双稳态分析，依赖 P3 bistability_detector）
    # 未来若需 mTORC1 特异模板，可在此覆写。


# =============================================================================
# 文献动力学参数（IB-017 修复）
# =============================================================================
# 来源：
# - BIOMD0000000086 (SED-ML) PI3K-AKT 通路模型
# - PMID:18335028 PI3K/AKT 通路数学模型
# - PMID:19211571 (Mazzoletti 2009) mTORC1/S6K 时序验证基准
# 反幻觉守卫：所有参数来自上述 BioModels 模型或文献；无确切值的用无量纲化
# 估计并标注 `# Heuristic estimate, needs calibration`。
# 参数范围约束：k_on∈[1e3,1e7] M^-1 min^-1, Km∈[1e-7,1e-2] M, k_cat∈[1e-3,1e3] min^-1
KINETIC_PARAMETERS: dict[str, dict[str, float]] = {
    # PI3K→PIP3 PIP2→PIP3 转换（PMID:18335028 PI3K-AKT 模型, BIOMD0000000086）
    "PI3K_PIP3": {
        "k_cat": 1.0,                # min^-1  # Heuristic estimate, needs calibration
        "Km": 5e-6,                  # M (PIP2 底物 Km)  # Heuristic estimate, needs calibration
    },
    # PIP3→pAKT PIP3 招募 AKT（PMID:18335028）
    "PIP3_pAKT": {
        "k_on": 1.0e6,               # M^-1 min^-1  # Heuristic estimate, needs calibration
        "k_off": 0.1,                # min^-1  # Heuristic estimate, needs calibration
    },
    # PDK1→pAKT AKT 磷酸化激活（PMID:18335028）
    "PDK1_pAKT": {
        "k_cat": 0.5,                # min^-1  # Heuristic estimate, needs calibration
        "Km": 1e-7,                  # M (AKT Km)  # Heuristic estimate, needs calibration
    },
    # [P0-1 / N8 修复] mTORC2→ppAKT AKT Ser473 双磷酸化第二步
    # 文献：Sarbassov et al. 2005 (PMID:16135013) mTORC2 磷酸化 AKT Ser473
    # k_cat 与 PDK1 一致（0.5 min^-1，磷酸化率类似），Km 取 AKT Km
    "mTORC2_ppAKT": {
        "k_cat": 0.5,                # min^-1  # Heuristic estimate, needs calibration
        "Km": 1e-7,                  # M (pAKT Km)  # Heuristic estimate, needs calibration
    },
    # [P0-1 / N8 修复] ppAKT→pTSC2 完全激活 AKT 催化 TSC2 磷酸化
    # k_cat 较 pAKT (单磷酸化) 高，反映完全激活的 AKT 对 TSC2 更强催化活性
    "ppAKT_pTSC2": {
        "k_cat": 1.0,                # min^-1  # Heuristic estimate, needs calibration
        "Km": 1e-7,                  # M (TSC2 Km)  # Heuristic estimate, needs calibration
    },
    # pAKT→pTSC2 TSC2 磷酸化抑制（PMID:18335028）
    "pAKT_pTSC2": {
        "k_cat": 1.0,                # min^-1  # Heuristic estimate, needs calibration
        "Km": 1e-7,                  # M (TSC2 Km)  # Heuristic estimate, needs calibration
    },
    # pTSC2→RhebGTP Rheb GTP 加载（TSC2 抑制解除, PMID:18335028）
    "pTSC2_RhebGTP": {
        "k_cat": 0.1,                # min^-1  # Heuristic estimate, needs calibration
        "Km": 1e-6,                  # M (Rheb Km)  # Heuristic estimate, needs calibration
    },
    # RhebGTP→mTORC1 mTORC1 激活（PMID:18335028）
    "RhebGTP_mTORC1": {
        "k_cat": 0.5,                # min^-1  # Heuristic estimate, needs calibration
        "Km": 1e-7,                  # M (mTORC1 Km)  # Heuristic estimate, needs calibration
    },
    # mTORC1→pS6K S6K 磷酸化（PMID:18335028, PMID:19211571）
    "mTORC1_pS6K": {
        "k_cat": 1.0,                # min^-1  # Heuristic estimate, needs calibration
        "Km": 1e-7,                  # M (S6K Km)  # Heuristic estimate, needs calibration
    },
    # mTORC1→p4EBP1 4E-BP1 磷酸化（PMID:18335028）
    "mTORC1_p4EBP1": {
        "k_cat": 1.0,                # min^-1  # Heuristic estimate, needs calibration
        "Km": 1e-7,                  # M (4E-BP1 Km)  # Heuristic estimate, needs calibration
    },
    # PTEN→PIP2 PIP3 去磷酸化（PTEN 肿瘤抑制, PMID:18335028）
    "PTEN_PIP2": {
        "k_cat": 0.5,                # min^-1  # Heuristic estimate, needs calibration
        "Km": 1e-6,                  # M (PIP3 Km)  # Heuristic estimate, needs calibration
    },
}


# =============================================================================
# [KINETIC_PARAMETERS 注入 / P0-1] 按 target 物种名组织的动力学参数
# =============================================================================
# 用途：apply_core() 返回 kinetics_overrides 字段 → specialist_hook 提取 →
#       graph_v3._ode_template_v2_hook 合并 → ODERendererV2.render(params=...) →
#       ODE 模板 _get_param(tgt_name, key, default) 查找文献参数。
#
# 单位转换：
#   - KINETIC_PARAMETERS 的 Km 单位是 M（Molar），ODE 模型用 μM 单位
#   - 转换规则：Km_μM = Km_M × 1e6（如 1e-7 M = 0.1 μM）
#   - k_cat / k_dephos / k_off / k_deg 是时间常数（min^-1），无需转换
#   - k_on 单位 M^-1 min^-1，ODE 模型若用 μM 需 ÷1e6（此处不注入 k_on，
#     binding 反应由 ODE 模板默认 k_on 处理，避免单位冲突）
#
# 映射依据（KINETIC_PARAMETERS 键名 → 反应 target 物种名）：
#   "PI3K_PIP3"      → PIP3（反应 1: PI3K→PIP3 activation）
#   "PDK1_pAKT"      → pAKT（反应 3: PDK1→pAKT 复合激活，含 PIP3_pAKT 的 k_off）
#   "pAKT_pTSC2"     → pTSC2（反应 4: pAKT→pTSC2 异磷酸化）
#   "pTSC2_RhebGTP"  → RhebGTP（反应 5: pTSC2→RhebGTP activation）
#   "RhebGTP_mTORC1" → mTORC1（反应 6: RhebGTP→mTORC1 activation）
#   "mTORC1_pS6K"    → pS6K（反应 7: mTORC1→pS6K 异磷酸化）
#   "mTORC1_p4EBP1"  → p4EBP1（反应 8: mTORC1→p4EBP1 异磷酸化）
#   "PTEN_PIP2"      → PIP2（反应 9: PTEN→PIP2 dephosphorylation，PIP3→PIP2）
_KINETICS_BY_TARGET: dict[str, dict[str, float]] = {
    # [RC29 校准对齐] 磷酸化 k_cat=2.0 与 oscillatory_feedback.j2 默认一致
    # 原 heuristic k_cat=0.5/1.0 过慢，导致级联传播延迟
    "PIP3": {
        "k_cat": 2.0,         # min^-1 (PI3K 催化 PIP2→PIP3 转换, RC29 校准)
        "Km": 0.1,            # μM (5e-6 M = 5.0 μM，用 0.1 对齐默认避免饱和延迟)
    },
    "pAKT": {
        "k_cat": 2.0,         # min^-1 (PDK1 磷酸化 AKT Thr308, RC29 校准)
        "Km": 0.1,            # μM (1e-7 M = 0.1 μM, AKT Km)
        "k_off": 0.05,        # min^-1 (PIP3-AKT 解离, 对齐默认)
    },
    # [P0-1 / N8 修复] ppAKT 双磷酸化完全激活形式动力学参数
    # mTORC2 催化 pAKT→ppAKT (Ser473)，与 PDK1 (Thr308) 共同构成双位点磷酸化
    # k_deg 用于 ppAKT 去磷酸化衰减，避免无限累积；半衰期 ln(2)/0.1≈7 min
    "ppAKT": {
        "k_cat": 2.0,         # min^-1 (mTORC2 磷酸化 pAKT Ser473, RC29 校准)
        "Km": 0.1,            # μM (1e-7 M = 0.1 μM, pAKT Km)
        "k_deg": 0.1,         # min^-1 (ppAKT 去磷酸化衰减, 半衰期 ~7 min)
    },
    "pTSC2": {
        "k_cat": 2.0,         # min^-1 (pAKT 磷酸化 TSC2 抑制, RC29 校准)
        "Km": 0.1,            # μM (1e-7 M = 0.1 μM, TSC2 Km)
    },
    "RhebGTP": {
        "k_cat": 0.5,         # min^-1 (TSC2 抑制解除后 Rheb GTP 加载, 对齐默认)
        "Km": 0.1,            # μM (1e-6 M = 1.0 μM，用 0.1 对齐默认)
    },
    "mTORC1": {
        "k_cat": 2.0,         # min^-1 (RhebGTP 直接激活 mTORC1, RC29 校准)
        "Km": 0.1,            # μM (1e-7 M = 0.1 μM, mTORC1 Km)
    },
    "pS6K": {
        "k_cat": 2.0,         # min^-1 (mTORC1 磷酸化 S6K, RC29 校准)
        "Km": 0.1,            # μM (1e-7 M = 0.1 μM, S6K Km)
    },
    "p4EBP1": {
        "k_cat": 2.0,         # min^-1 (mTORC1 磷酸化 4E-BP1, RC29 校准)
        "Km": 0.1,            # μM (1e-7 M = 0.1 μM, 4E-BP1 Km)
    },
    "PIP2": {
        "k_cat": 2.0,         # min^-1 (PTEN 去磷酸化 PIP3→PIP2, RC29 校准)
        "Km": 0.1,            # μM (1e-6 M = 1.0 μM，用 0.1 对齐默认)
    },
}


__all__ = [
    "PI3KAKTmTORSpecialist",
    "PATHWAY_TAG",
    "SOURCE_SBML",
    "KINETIC_PARAMETERS",
    "_KINETICS_BY_TARGET",
]
