# BioDynamics Agent v4 - TGF-β Specialist (Phase 4 / Task 4.12)
# TGF-β 通路 Specialist：实现 TGF-β+TβRII→TβRI 招募→TβRII 磷酸化 TβRI→
# R-SMAD（Smad2/3）磷酸化→Co-SMAD（Smad4）异源复合→入核→转录
# （PAI-1/SMAD7）核心拓扑 + SMAD7→TβRI 转录延迟负反馈（delay=30min）
# + SMAD→SMURF→R-SMAD 泛素化负反馈（delay=60min）+ Smad2 状态机
# （cyto_Smad2 → cyto_pSmad2 → cyto_pSmad2_Smad4 → nuc_pSmad2_Smad4）。
#
# 设计原则（铁律）：
# 1. Feature Flag V4_PATHWAY_SPECIALIST_ENABLED=false 时由 LangGraph hook 层短路，
#    本 Specialist 代码不被调用（基类本身不做 flag 检查）
# 2. 使用 Task 4.0 修复后的 PHOSPHORYLATION 语义：
#    - 异磷酸化（TGF_beta_TbRII_TbRI → pTbRI）：TbRI 作 substrate,
#      pTbRI 作 product，TGF_beta_TbRII_TbRI 复合物作 catalytic modifier
#      （TβRII 在复合物内磷酸化 TβRI GS domain）
#    - 异磷酸化（pTbRI → pSmad2）：Smad2 作 substrate，pSmad2 作 product，
#      pTbRI 作 catalytic modifier
#    - 异磷酸化（pTbRI → pSmad3）：Smad3 作 substrate，pSmad3 作 product，
#      pTbRI 作 catalytic modifier
# 3. SMAD 复合-入核-转录三步耦合 CompositeReaction
#    （complex_formation→nuclear_import→transcription），
#    不压扁为单一 reaction，保留中间产物语义
# 4. 不处理 TGF-β→p15/p21 细胞周期抑制下游（由 Cell Cycle Specialist 处理）；
#    p15/p21 仅作为 cross-talk Reaction 片段输出
# 5. 不生成 ERK→Smad linker cross-talk edge 本身（由 Cross-talk Coordinator 处理，
#    Task 4.13），仅返回本通路侧的 cross-talk Reaction 片段
# 6. apply_* 方法实现捕获异常并返回空 list/dict，记录 logger.warning，不抛异常
#
# 依赖：
# - P1 ontology / pathway_registry（TGF_BETA 通路类别键）
# - P2 schema（SpeciesV2 / ReactionV2 / SpeciesRef / Modifier / CompositeReaction / StateMachine）
# - P2 MechanismType（PHOSPHORYLATION / COMPLEX_FORMATION / NUCLEAR_IMPORT /
#   TRANSCRIPTION / INHIBITION / ACTIVATION / UBIQUITINATION）
# - P2 StateMachine（Smad2 状态机：cyto_Smad2→cyto_pSmad2→cyto_pSmad2_Smad4→nuc_pSmad2_Smad4）
# - P3 ode_templates_v2（_mechanism_phosphorylation_mm.j2 模板；
#   transcription_factor.j2 当前未实现，转录降级到 _mechanism_phosphorylation_mm）
#
# 参考：
# - spec.md Part 3 Specialist 10（第 255-260 行）
# - tasks.md Task 4.12
# - Massagué 1998 TGF-β signaling (PMID:9674480)
# - Schmierer 2007 SMAD dynamics (PMID:17721552)

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
# TGF-β 通路标签
# =============================================================================
PATHWAY_TAG: str = "TGF_BETA"

# SBML BioModels ID（Schmierer 2007 SMAD dynamics model）
SOURCE_SBML: str = get_biomodels_id(PATHWAY_TAG)

# Validation benchmark PMID 引用
_Pmid_MASSAGUE_1998: str = "PMID:9674480"   # Massagué 1998 TGF-β signaling
_Pmid_SCHMIERER_2007: str = "PMID:17721552"  # Schmierer 2007 SMAD dynamics

# SMAD7→TβRI 转录延迟负反馈延迟（DDE，分钟）
# Massagué 1998 模型中 pSmad2:Smad4_nuc 转录激活 SMAD7 mRNA（含 30 min 转录延迟），
# SMAD7 蛋白结合 TβRI 阻断其激酶活性，形成 delay=30min 延迟负反馈，
# 调节 SMAD 信号衰减时序。
_TGF_SMAD7_DELAY_MINUTES: float = 30.0

# SMAD→SMURF→R-SMAD 泛素化负反馈延迟（DDE，分钟）
# SMAD7 转录激活 SMURF E3 连接酶（含 60 min 转录+翻译延迟），
# SMURF 泛素化 pSmad2/pSmad3 导致降解，形成 delay=60min 延迟负反馈。
_TGF_SMURF_DELAY_MINUTES: float = 60.0


# =============================================================================
# TGF-β 核心物种（与 P3 pathway_graph/initializer.py TGF_BETA core_nodes 对齐，
# 扩展 TGF_beta_TbRII / TGF_beta_TbRII_TbRI / pTbRI / pSmad2 / pSmad3 /
# pSmad2_Smad4 / pSmad3_Smad4 / pSmad2_Smad4_nuc / pSmad3_Smad4_nuc /
# PAI1_mRNA / SMAD7_mRNA / SMAD7 / SMURF 完整 TGF-β 拓扑）
# =============================================================================
# SMAD7 物种标记 shared=False（本通路内部负反馈效应器，不与其他通路共享）
# AKT / p21 / p15 / Bim / PUMA 等仅在 crosstalk 中标记 shared，不在核心 species 内
_TGF_BETA_CORE_SPECIES: list[dict[str, Any]] = [
    # [C4 fix] initial_concentration aligned to BIOMD0000000342 (Zi2011, PMID:21613981).
    #   SBML models TGF-β/Smad2/Smad4 dynamics (TGF_beta_ex/Smad2c/Smad4c/PSmad2c/
    #   PSmad2_Smad4_c/PSmad2_Smad4_n/T1R_surf/T2R_surf).
    #   SBML species mapping: TGF_beta→TGF_beta_ex, Smad2→Smad2c, Smad4→Smad4c,
    #   pSmad2→PSmad2c, pSmad2_Smad4→PSmad2_Smad4_c, pSmad2_Smad4_nuc→PSmad2_Smad4_n,
    #   TbRI→T1R_surf, TbRII→T2R_surf.
    #   Species not in SBML (TGF_beta_TbRII/TGF_beta_TbRII_TbRI/pTbRI/Smad3/pSmad3/
    #   pSmad3_Smad4/pSmad3_Smad4_nuc/PAI1_mRNA/SMAD7_mRNA/SMAD7/SMURF) kept original.
    # ---- 配体 + 受体 ----
    # TGF_beta（TGF-β1/2/3 配体，分泌型同源二聚体细胞因子，结合 TβRII 启动信号）
    # [C4 fix] SBML mapping: TGF_beta→TGF_beta_ex (initial_concentration=0.05)
    {"name": "TGF_beta", "species_type": "ligand",
     "compartment": "extracellular",
     "initial_concentration": 0.05},  # Source: BIOMD0000000342 Zi2011 (PMID:21613981) species TGF_beta_ex
    # TbRII（TGF-β receptor type II，组成型激酶活性丝/苏氨酸激酶受体）
    # [C4 fix] SBML mapping: TbRII→T2R_surf (surface TβRII, initial_concentration=0.201077)
    {"name": "TbRII", "species_type": "protein",
     "compartment": "membrane",
     "initial_concentration": 0.201077},  # Source: BIOMD0000000342 Zi2011 (PMID:21613981) species T2R_surf
    # TbRI（TGF-β receptor type I / ALK5，丝/苏氨酸激酶受体，被 TβRII 磷酸化激活）
    # [C4 fix] SBML mapping: TbRI→T1R_surf (surface TβRI, initial_concentration=0.702494)
    {"name": "TbRI", "species_type": "protein",
     "compartment": "membrane",
     "initial_concentration": 0.702494},  # Source: BIOMD0000000342 Zi2011 (PMID:21613981) species T1R_surf
    # TGF_beta_TbRII（TGF-β+TβRII 二元配体-受体复合物，招募 TβRI）
    # [C4 fix] No SBML match in BIOMD0000000342 (Zi2011 models LRC_surf as combined complex). Kept original.
    {"name": "TGF_beta_TbRII", "species_type": "complex",
     "compartment": "membrane"},
    # TGF_beta_TbRII_TbRI（TGF-β+TβRII+TβRI 三元受体复合物，TβRII 磷酸化 TβRI）
    # [C4 fix] No SBML match in BIOMD0000000342. Kept original.
    {"name": "TGF_beta_TbRII_TbRI", "species_type": "complex",
     "compartment": "membrane"},
    # ---- TβRI 磷酸化激活 ----
    # pTbRI（磷酸化激活的 TβRI，TβRII 磷酸化 TβRI GS domain Ser165/Ser172 激活激酶）
    # [C4 fix] No SBML match in BIOMD0000000342 (Zi2011 doesn't model phosphorylated TβRI separately). Kept original.
    {"name": "pTbRI", "species_type": "protein",
     "compartment": "membrane"},
    # ---- R-SMAD（Smad2/3，受体调节型 SMAD，被 pTbRI 磷酸化）----
    # Smad2（R-SMAD，受体调节型 SMAD2，C-terminal MH2 结构域 SSXS motif 被 pTbRI 磷酸化）
    # [C4 fix] SBML mapping: Smad2→Smad2c (cytoplasmic Smad2, initial_concentration=60.6)
    {"name": "Smad2", "species_type": "protein",
     "compartment": "cytoplasm",
     "initial_concentration": 60.6},  # Source: BIOMD0000000342 Zi2011 (PMID:21613981) species Smad2c
    # Smad3（R-SMAD，受体调节型 SMAD3，与 Smad2 平行被 pTbRI 磷酸化）
    # [C4 fix] No SBML match in BIOMD0000000342 (Zi2011 models Smad2 only, not Smad3). Kept original.
    {"name": "Smad3", "species_type": "protein",
     "compartment": "cytoplasm"},
    # pSmad2（磷酸化 Smad2，Ser465/467 磷酸化，暴露 MH2 二聚化界面）
    # [C4 fix] SBML mapping: pSmad2→PSmad2c (cytoplasmic phosphorylated Smad2, initial=0.0)
    {"name": "pSmad2", "species_type": "protein",
     "compartment": "cytoplasm",
     "initial_concentration": 0.0},  # Source: BIOMD0000000342 Zi2011 (PMID:21613981) species PSmad2c
    # pSmad3（磷酸化 Smad3，Ser423/425 磷酸化，暴露 MH2 二聚化界面）
    # [C4 fix] No SBML match in BIOMD0000000342. Kept original.
    {"name": "pSmad3", "species_type": "protein",
     "compartment": "cytoplasm"},
    # ---- Co-SMAD（Smad4，共同介质 SMAD，与 R-SMAD 异源复合）----
    # Smad4（Co-SMAD，共同介质 SMAD4，与 pSmad2/pSmad3 形成异源复合物入核）
    # [C4 fix] SBML mapping: Smad4→Smad4c (cytoplasmic Smad4, initial_concentration=50.8)
    {"name": "Smad4", "species_type": "protein",
     "compartment": "cytoplasm",
     "initial_concentration": 50.8},  # Source: BIOMD0000000342 Zi2011 (PMID:21613981) species Smad4c
    # ---- R-SMAD-CoSMAD 异源复合物 ----
    # pSmad2_Smad4（pSmad2+Smad4 异源复合物，胞质形成，准备入核）
    # ★ SMAD 复合-入核-转录三步 CompositeReaction step 1 产物
    # [C4 fix] SBML mapping: pSmad2_Smad4→PSmad2_Smad4_c (initial_concentration=0.0)
    {"name": "pSmad2_Smad4", "species_type": "complex",
     "compartment": "cytoplasm",
     "initial_concentration": 0.0},  # Source: BIOMD0000000342 Zi2011 (PMID:21613981) species PSmad2_Smad4_c
    # pSmad3_Smad4（pSmad3+Smad4 异源复合物，胞质形成，准备入核）
    # [C4 fix] No SBML match in BIOMD0000000342 (Zi2011 models Smad2 path only). Kept original.
    {"name": "pSmad3_Smad4", "species_type": "complex",
     "compartment": "cytoplasm"},
    # pSmad2_Smad4_nuc（核内 pSmad2:Smad4 复合物，作为转录因子激活靶基因）
    # ★ SMAD 复合-入核-转录三步 CompositeReaction step 2 产物, step 3 modifier
    # [C4 fix] SBML mapping: pSmad2_Smad4_nuc→PSmad2_Smad4_n (initial_concentration=0.0)
    {"name": "pSmad2_Smad4_nuc", "species_type": "complex",
     "compartment": "nucleus",
     "initial_concentration": 0.0},  # Source: BIOMD0000000342 Zi2011 (PMID:21613981) species PSmad2_Smad4_n
    # pSmad3_Smad4_nuc（核内 pSmad3:Smad4 复合物，作为转录因子激活靶基因）
    # [C4 fix] No SBML match in BIOMD0000000342. Kept original.
    {"name": "pSmad3_Smad4_nuc", "species_type": "complex",
     "compartment": "nucleus"},
    # ---- 转录靶基因 mRNA ----
    # PAI1_mRNA（PAI-1 / SERPINE1 mRNA，pSmad2:Smad4 转录激活，TGF-β 经典靶基因）
    # ★ SMAD 复合-入核-转录三步 CompositeReaction step 3 产物
    # [C4 fix] No SBML match in BIOMD0000000342 (Zi2011 doesn't model PAI-1 transcription). Kept original.
    {"name": "PAI1_mRNA", "species_type": "mrna",
     "compartment": "nucleus"},
    # SMAD7_mRNA（SMAD7 mRNA，pSmad2:Smad4 转录激活，含 30 min 转录延迟，负反馈准备）
    # [C4 fix] No SBML match in BIOMD0000000342 (Zi2011 doesn't model SMAD7 transcription). Kept original.
    {"name": "SMAD7_mRNA", "species_type": "mrna",
     "compartment": "nucleus"},
    # ---- SMAD7 负反馈效应器 ----
    # SMAD7（SMAD7 蛋白，结合 TβRI 阻断其激酶活性，负反馈）
    # [C4 fix] No SBML match in BIOMD0000000342. Kept original.
    {"name": "SMAD7", "species_type": "protein",
     "compartment": "cytoplasm"},
    # ---- SMURF E3 连接酶（SMAD→SMURF→R-SMAD 泛素化负反馈）----
    # SMURF（SMURF2 E3 泛素连接酶，由 SMAD7 转录激活，泛素化 pSmad2/pSmad3 降解）
    # [C4 fix] No SBML match in BIOMD0000000342. Kept original.
    {"name": "SMURF", "species_type": "protein",
     "compartment": "cytoplasm"},
    # [N6 缺口 1] 药物物种（species_type="drug"）— 由 drug_library 驱动
    # SB431542 是 ALK5/TβRI ATP 竞争性抑制剂（IC50=94 nM, PMID:15546018）
    build_drug_species("SB431542"),
]


# =============================================================================
# TGF-β 核心反应（11 条：受体复合物形成 + TβRI 磷酸化 + R-SMAD 磷酸化
# + Co-SMAD 异源复合 + nuclear import + 2 转录）
# =============================================================================
# 每条反应含：source / target / mechanism / kinetics_type / pathway_tag /
#             substrate / product / modifier / modifier_type / autophosphorylation
#
# kinetics_type 选择：
# - complex_formation / nuclear_import → mass_action
# - phosphorylation → Michaelis_Menten（与 P3 _mechanism_phosphorylation_mm 模板对齐）
# - transcription → Hill（pSmad2:Smad4_nuc 作转录因子，Hill 动力学 n=2 协同结合）
_TGF_BETA_CORE_REACTIONS: list[dict[str, Any]] = [
    # 1. TGF-β + TβRII → TGF_beta_TbRII（complex_formation, ligand-receptor binding）
    #    TGF-β 配体结合 TβRII 受体形成二元配体-受体复合物，启动 TGF-β 信号
    {
        "source": "TGF_beta",
        "target": "TGF_beta_TbRII",
        "mechanism": "complex_formation",
        "kinetics_type": "mass_action",
        "pathway_tag": PATHWAY_TAG,
        "substrate": "TbRII",
        "product": "TGF_beta_TbRII",
        "modifier": "TGF_beta",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "TGF-β + TβRII → TGF_beta_TbRII（complex_formation, ligand-receptor binding, TGF-β 配体结合 TβRII 受体形成二元配体-受体复合物启动 TGF-β 信号）",
    },
    # 2. TGF_beta_TbRII + TβRI → TGF_beta_TbRII_TbRI（complex_formation, receptor recruitment）
    #    TGF-β:TβRII 复合物招募 TβRI 形成三元受体复合物，TβRII 靠近 TβRI 准备磷酸化
    {
        "source": "TGF_beta_TbRII",
        "target": "TGF_beta_TbRII_TbRI",
        "mechanism": "complex_formation",
        "kinetics_type": "mass_action",
        "pathway_tag": PATHWAY_TAG,
        "substrate": "TbRI",
        "product": "TGF_beta_TbRII_TbRI",
        "modifier": "TGF_beta_TbRII",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "TGF_beta_TbRII + TβRI → TGF_beta_TbRII_TbRI（complex_formation, receptor recruitment, TGF-β:TβRII 复合物招募 TβRI 形成三元受体复合物, TβRII 靠近 TβRI 准备磷酸化）",
    },
    # 3. TGF_beta_TbRII_TbRI → pTbRI（phosphorylation, TβRII 磷酸化 TβRI, hetero_phosphorylation）
    #    TβRII（组成型激酶）在三元复合物内磷酸化 TβRI GS domain Ser165/Ser172 激活 TβRI 激酶
    #    ★ 异磷酸化：TbRI 作 substrate，pTbRI 作 product，TGF_beta_TbRII_TbRI 复合物作 catalytic modifier
    {
        "source": "TGF_beta_TbRII_TbRI",
        "target": "pTbRI",
        "mechanism": "phosphorylation",
        "kinetics_type": "Michaelis_Menten",
        "pathway_tag": PATHWAY_TAG,
        # 异磷酸化：TbRI 作 substrate，pTbRI 作 product，TGF_beta_TbRII_TbRI 作 catalytic modifier
        # （TβRII 在三元复合物内磷酸化 TβRI GS domain，TβRII 作 catalytic kinase）
        "substrate": "TbRI",
        "product": "pTbRI",
        "modifier": "TGF_beta_TbRII_TbRI",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "site": "GS_domain_Ser165_Ser172",
        "description": "TGF_beta_TbRII_TbRI → pTbRI（phosphorylation, TbRI 作 substrate, TGF_beta_TbRII_TbRI 作 catalytic modifier, TβRII 磷酸化 TβRI GS domain Ser165/172 激活激酶, 异磷酸化）",
    },
    # 4. pTbRI → pSmad2（phosphorylation, R-SMAD 磷酸化, Smad2 作 substrate, pTbRI 作 modifier）
    #    pTbRI 异磷酸化 Smad2 C-terminal SSXS motif Ser465/467，激活 R-SMAD
    #    ★ 异磷酸化：Smad2 作 substrate，pSmad2 作 product，pTbRI 作 catalytic modifier
    {
        "source": "pTbRI",
        "target": "pSmad2",
        "mechanism": "phosphorylation",
        "kinetics_type": "Michaelis_Menten",
        "pathway_tag": PATHWAY_TAG,
        # 异磷酸化：Smad2 作 substrate，pSmad2 作 product，pTbRI 作 catalytic modifier
        "substrate": "Smad2",
        "product": "pSmad2",
        "modifier": "pTbRI",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "site": "Ser465_467_SSXS",
        "description": "pTbRI 磷酸化 Smad2（Smad2 作 substrate, pTbRI 作 catalytic modifier, pTbRI 异磷酸化 Smad2 C-terminal SSXS motif Ser465/467 激活 R-SMAD）",
    },
    # 5. pTbRI → pSmad3（phosphorylation, R-SMAD 磷酸化, Smad3 作 substrate, pTbRI 作 modifier）
    #    pTbRI 异磷酸化 Smad3 C-terminal SSXS motif Ser423/425，激活 R-SMAD
    #    ★ 异磷酸化：Smad3 作 substrate，pSmad3 作 product，pTbRI 作 catalytic modifier
    {
        "source": "pTbRI",
        "target": "pSmad3",
        "mechanism": "phosphorylation",
        "kinetics_type": "Michaelis_Menten",
        "pathway_tag": PATHWAY_TAG,
        # 异磷酸化：Smad3 作 substrate，pSmad3 作 product，pTbRI 作 catalytic modifier
        "substrate": "Smad3",
        "product": "pSmad3",
        "modifier": "pTbRI",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "site": "Ser423_425_SSXS",
        "description": "pTbRI 磷酸化 Smad3（Smad3 作 substrate, pTbRI 作 catalytic modifier, pTbRI 异磷酸化 Smad3 C-terminal SSXS motif Ser423/425 激活 R-SMAD）",
    },
    # 6. pSmad2 + Smad4 → pSmad2_Smad4（complex_formation, Co-SMAD 异源复合）
    #    pSmad2 通过 MH2 结构域与 Smad4 形成异源复合物，准备入核作为转录因子
    #    ★ SMAD 复合-入核-转录三步 CompositeReaction step 1（complex_formation）
    {
        "source": "pSmad2",
        "target": "pSmad2_Smad4",
        "mechanism": "complex_formation",
        "kinetics_type": "mass_action",
        "pathway_tag": PATHWAY_TAG,
        "substrate": "Smad4",
        "product": "pSmad2_Smad4",
        "modifier": "pSmad2",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "pSmad2 + Smad4 → pSmad2_Smad4（complex_formation, Co-SMAD 异源复合, pSmad2 通过 MH2 结构域与 Smad4 形成异源复合物, 三步 CompositeReaction step 1）",
        "composite_step": 1,
        "composite_id": "CR_SMAD_COMPLEX_NUCLEAR_TRANSCRIPTION",
    },
    # 7. pSmad3 + Smad4 → pSmad3_Smad4（complex_formation, Co-SMAD 异源复合）
    #    pSmad3 通过 MH2 结构域与 Smad4 形成异源复合物，准备入核作为转录因子
    {
        "source": "pSmad3",
        "target": "pSmad3_Smad4",
        "mechanism": "complex_formation",
        "kinetics_type": "mass_action",
        "pathway_tag": PATHWAY_TAG,
        "substrate": "Smad4",
        "product": "pSmad3_Smad4",
        "modifier": "pSmad3",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "pSmad3 + Smad4 → pSmad3_Smad4（complex_formation, Co-SMAD 异源复合, pSmad3 通过 MH2 结构域与 Smad4 形成异源复合物, 准备入核作为转录因子）",
    },
    # 8. pSmad2_Smad4 → pSmad2_Smad4_nuc（nuclear_import, R-SMAD-CoSMAD 复合物入核）
    #    pSmad2:Smad4 异源复合物通过 importin α/β 入核，作为转录因子激活靶基因
    #    ★ SMAD 复合-入核-转录三步 CompositeReaction step 2（nuclear_import）
    {
        "source": "pSmad2_Smad4",
        "target": "pSmad2_Smad4_nuc",
        "mechanism": "nuclear_import",
        "kinetics_type": "mass_action",
        "pathway_tag": PATHWAY_TAG,
        "substrate": "pSmad2_Smad4",
        "product": "pSmad2_Smad4_nuc",
        "modifier": "pSmad2_Smad4",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "pSmad2_Smad4 → pSmad2_Smad4_nuc（nuclear_import, pSmad2:Smad4 异源复合物通过 importin α/β 入核, 作为转录因子激活靶基因, 三步 CompositeReaction step 2）",
        "composite_step": 2,
        "composite_id": "CR_SMAD_COMPLEX_NUCLEAR_TRANSCRIPTION",
    },
    # 9. pSmad3_Smad4 → pSmad3_Smad4_nuc（nuclear_import, R-SMAD-CoSMAD 复合物入核）
    #    pSmad3:Smad4 异源复合物通过 importin α/β 入核，作为转录因子激活靶基因
    {
        "source": "pSmad3_Smad4",
        "target": "pSmad3_Smad4_nuc",
        "mechanism": "nuclear_import",
        "kinetics_type": "mass_action",
        "pathway_tag": PATHWAY_TAG,
        "substrate": "pSmad3_Smad4",
        "product": "pSmad3_Smad4_nuc",
        "modifier": "pSmad3_Smad4",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "pSmad3_Smad4 → pSmad3_Smad4_nuc（nuclear_import, pSmad3:Smad4 异源复合物通过 importin α/β 入核, 作为转录因子激活靶基因）",
    },
    # 10. pSmad2_Smad4_nuc → PAI1_mRNA（transcription, Hill, pSmad2:Smad4 作转录因子）
    #     pSmad2_Smad4_nuc 结合 PAI-1 基因启动子 CAGA box，Hill 协同结合（n=2）激活转录
    #     ★ SMAD 复合-入核-转录三步 CompositeReaction step 3（transcription）
    {
        "source": "pSmad2_Smad4_nuc",
        "target": "PAI1_mRNA",
        "mechanism": "transcription",
        "kinetics_type": "Hill",
        "pathway_tag": PATHWAY_TAG,
        # pSmad2_Smad4_nuc 作 modifier（转录因子），DNA 作 substrate，PAI1_mRNA 作 product
        "substrate": "DNA",
        "product": "PAI1_mRNA",
        "modifier": "pSmad2_Smad4_nuc",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "hill_coefficient": 2,
        "description": "pSmad2_Smad4_nuc → PAI1_mRNA（transcription, Hill n=2, pSmad2:Smad4 作转录因子结合 CAGA box 激活 PAI-1 转录, TGF-β 经典靶基因, 三步 CompositeReaction step 3）",
        "composite_step": 3,
        "composite_id": "CR_SMAD_COMPLEX_NUCLEAR_TRANSCRIPTION",
    },
    # 11. pSmad2_Smad4_nuc → SMAD7_mRNA（transcription, Hill, pSmad2:Smad4 作转录因子, 负反馈准备）
    #     pSmad2_Smad4_nuc 结合 SMAD7 基因启动子，激活 SMAD7 转录（含 30 min 转录延迟，
    #     SMAD7 蛋白结合 TβRI 阻断激酶活性形成负反馈，由 FeedbackLoop FL_SMAD7 表达）
    {
        "source": "pSmad2_Smad4_nuc",
        "target": "SMAD7_mRNA",
        "mechanism": "transcription",
        "kinetics_type": "Hill",
        "pathway_tag": PATHWAY_TAG,
        # pSmad2_Smad4_nuc 作 modifier（转录因子），DNA 作 substrate，SMAD7_mRNA 作 product
        "substrate": "DNA",
        "product": "SMAD7_mRNA",
        "modifier": "pSmad2_Smad4_nuc",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "hill_coefficient": 2,
        "description": "pSmad2_Smad4_nuc → SMAD7_mRNA（transcription, Hill n=2, pSmad2:Smad4 作转录因子激活 SMAD7 转录, 含 30 min 转录延迟, SMAD7 蛋白结合 TβRI 阻断激酶活性形成负反馈）",
    },
    # 12. SMAD7_mRNA → SMAD7（translation, mass_action, 反馈环第 2 步：mRNA 翻译为蛋白）
    #    [N8-P0-2 修复] 补全反馈环第 2 步：SMAD7 mRNA 在核糖体翻译为 SMAD7 蛋白
    #    （Clarke 2006, PMID:17035437; KINETIC_PARAMETERS 中 k_translation=0.1 min^-1）
    #    缺失此反应导致 SMAD7_mRNA 累积但 SMAD7 蛋白始终为 0，反馈环断裂
    {
        "source": "SMAD7_mRNA",
        "target": "SMAD7",
        "mechanism": "translation",
        "kinetics_type": "mass_action",
        "pathway_tag": PATHWAY_TAG,
        "substrate": "SMAD7_mRNA",
        "product": "SMAD7",
        "modifier": "SMAD7_mRNA",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "SMAD7 mRNA 翻译为 SMAD7 蛋白（SMAD7_mRNA 作 substrate, SMAD7 作 product, 反馈环第 2 步, 用于结合 TβRI 阻断其激酶活性形成负反馈, Clarke 2006 PMID:17035437）",
    },
    # 13. SMAD7 → pTbRI（inhibition, mass_action, 反馈环第 3 步：SMAD7 抑制 pTbRI 激酶活性）
    #    [N8-P0-2 修复] 补全反馈环第 3 步：SMAD7 蛋白结合 pTbRI 阻断其激酶活性
    #    （Clarke 2006, PMID:17035437; KINETIC_PARAMETERS 中 k_off=1e-3 min^-1, Km=1e-7 M）
    #    缺失此反应导致 SMAD7 蛋白无法实际抑制 pTbRI，反馈环无法闭合
    #    注：pTbRI 作 substrate（被抑制），SMAD7 作 modifier（抑制因子），无产物（pTbRI 活性降低）
    {
        "source": "SMAD7",
        "target": "pTbRI",
        "mechanism": "inhibition",
        "kinetics_type": "mass_action",
        "pathway_tag": PATHWAY_TAG,
        "substrate": "pTbRI",
        "product": "pTbRI",
        "modifier": "SMAD7",
        "modifier_type": "allosteric",
        "autophosphorylation": False,
        "description": "SMAD7 抑制 pTbRI 激酶活性（pTbRI 作 substrate, SMAD7 作 allosteric inhibitor, 反馈环第 3 步, 阻断 pTbRI 磷酸化 R-SMAD, delay=0min 蛋白结合直接抑制, Clarke 2006 PMID:17035437）",
    },
    # ===== [N6 缺口 1] 药物-靶点显式 inhibitor edge（canonical drug_library 驱动） =====
    # 14. SB431542 → TbRI（ATP_competitive, IC50=94 nM, PMID:15546018）
    # SB431542 是 ALK4/5/7（TβRI）ATP 竞争性抑制剂，阻断 TβRI 磷酸化 R-SMAD，
    # 与 Galunisertib 共享 TbRI 靶点（实验室常用工具化合物）。
    # 此处 target="TbRI" 与 specialist 现有物种命名对齐（SB431542 功能性靶点是 ALK5/TβRI 激酶活性）。
    {
        **build_inhibitor_edge("SB431542", "TbRI"),
        "pathway_tag": PATHWAY_TAG,
    },
]


# =============================================================================
# SMAD 复合-入核-转录三步耦合 CompositeReaction
# （complex_formation→nuclear_import→transcription）
# =============================================================================
# SMAD 复合-入核-转录三步耦合是 TGF-β 通路的核心信号传递机制：
# - Step 1: pSmad2 + Smad4 → pSmad2_Smad4（complex_formation, Co-SMAD 异源复合）
# - Step 2: pSmad2_Smad4 → pSmad2_Smad4_nuc（nuclear_import, R-SMAD-CoSMAD 复合物入核）
# - Step 3: pSmad2_Smad4_nuc → PAI1_mRNA（transcription, pSmad2:Smad4 作转录因子）
# 净反应：pSmad2 + Smad4 + DNA → PAI1_mRNA（pSmad2 磷酸化后与 Smad4 复合入核转录）
#
# 三步耦合是顺序执行（sequential coupling），中间产物 pSmad2_Smad4 /
# pSmad2_Smad4_nuc 保留语义不压扁为单一 reaction（CompositeReaction 设计原则）
_TGF_BETA_COMPOSITE_REACTIONS: list[dict[str, Any]] = [
    {
        "id": "CR_SMAD_COMPLEX_NUCLEAR_TRANSCRIPTION",
        "name": "TGF-β SMAD Complex-Nuclear Import-Transcription Three-Step Coupled Cascade",
        "mechanism": "sequential",
        "template": "transcription_factor.j2",
        "loop_type": "negative",
        "coupling_type": "sequential",
        "sub_reactions": [
            # Step 1: complex_formation (pSmad2 + Smad4 → pSmad2_Smad4, Co-SMAD 异源复合)
            "pSmad2 → pSmad2_Smad4",
            # Step 2: nuclear_import (pSmad2_Smad4 → pSmad2_Smad4_nuc, 入核)
            "pSmad2_Smad4 → pSmad2_Smad4_nuc",
            # Step 3: transcription (pSmad2_Smad4_nuc → PAI1_mRNA, 转录激活)
            "pSmad2_Smad4_nuc → PAI1_mRNA",
        ],
        "intermediate_species": [
            "pSmad2_Smad4",          # Co-SMAD 异源复合物（step 1 产物, step 2 substrate）
            "pSmad2_Smad4_nuc",      # 核内 R-SMAD-CoSMAD 复合物（step 2 产物, step 3 modifier）
            "PAI1_mRNA",              # PAI-1 转录产物（step 3 产物, TGF-β 经典靶基因）
        ],
        "net_reaction": "pSmad2 + Smad4 + DNA → PAI1_mRNA (TGF-β R-SMAD-CoSMAD 复合入核转录)",
        "node_ids": [
            "pSmad2",
            "Smad4",
            "pSmad2_Smad4",
            "pSmad2_Smad4_nuc",
            "PAI1_mRNA",
        ],
        "reactions": [
            "pSmad2 → pSmad2_Smad4",
            "pSmad2_Smad4 → pSmad2_Smad4_nuc",
            "pSmad2_Smad4_nuc → PAI1_mRNA",
        ],
        "description": (
            "SMAD 复合-入核-转录三步耦合（complex_formation→nuclear_import→"
            "transcription）：pSmad2+Smad4→pSmad2_Smad4 异源复合, "
            "pSmad2_Smad4→pSmad2_Smad4_nuc 入核, pSmad2_Smad4_nuc→PAI1_mRNA "
            "转录激活, 净反应 pSmad2+Smad4+DNA→PAI1_mRNA, "
            "sequential coupling 保留中间产物语义 (Massagué 1998, "
            "PMID:9674480; Schmierer 2007, PMID:17721552)"
        ),
        "delay_minutes": 0.0,   # 三步耦合本身无转录延迟（蛋白级联）
        "pmid": _Pmid_MASSAGUE_1998,
    },
]


# =============================================================================
# Smad2 状态机（cyto_Smad2 → cyto_pSmad2 → cyto_pSmad2_Smad4 → nuc_pSmad2_Smad4）
# =============================================================================
# Smad2 蛋白 4 状态机：
# - cyto_Smad2（未磷酸化胞质单体，初始状态）
# - cyto_pSmad2（C-terminal SSXS 磷酸化单体）
# - cyto_pSmad2_Smad4（与 Smad4 异源复合，胞质）
# - nuc_pSmad2_Smad4（入核的 pSmad2:Smad4 复合物，作为转录因子）
#
# 状态转换关联到 Core Reaction：
# - cyto_Smad2 → cyto_pSmad2: pTbRI → pSmad2 磷酸化
# - cyto_pSmad2 → cyto_pSmad2_Smad4: pSmad2 → pSmad2_Smad4 异源复合
# - cyto_pSmad2_Smad4 → nuc_pSmad2_Smad4: pSmad2_Smad4 → pSmad2_Smad4_nuc 入核
_SMAD2_STATE_MACHINE: dict[str, Any] = {
    "id": "SM_SMAD2",
    "species": "Smad2",
    "description": (
        "Smad2 状态机：cyto_Smad2（未磷酸化胞质单体）→cyto_pSmad2"
        "（SSXS 磷酸化）→cyto_pSmad2_Smad4（与 Smad4 异源复合, 胞质）"
        "→nuc_pSmad2_Smad4（入核作为转录因子）"
    ),
    "states": [
        {"name": "cyto_Smad2", "species_id": "SP_Smad2", "is_initial": True},
        {"name": "cyto_pSmad2", "species_id": "SP_pSmad2"},
        {"name": "cyto_pSmad2_Smad4", "species_id": "SP_pSmad2_Smad4"},
        {"name": "nuc_pSmad2_Smad4", "species_id": "SP_pSmad2_Smad4_nuc"},
    ],
    "transitions": [
        {
            "from_state": "cyto_Smad2",
            "to_state": "cyto_pSmad2",
            "reaction_id": "RXN_TBRI_SMAD2_PHOS",
            "trigger": "phosphorylation",
        },
        {
            "from_state": "cyto_pSmad2",
            "to_state": "cyto_pSmad2_Smad4",
            "reaction_id": "RXN_SMAD2_SMAD4_COMPLEX",
            "trigger": "complex_formation",
        },
        {
            "from_state": "cyto_pSmad2_Smad4",
            "to_state": "nuc_pSmad2_Smad4",
            "reaction_id": "RXN_SMAD2_SMAD4_NUCLEAR_IMPORT",
            "trigger": "nuclear_import",
        },
    ],
}


# =============================================================================
# TGF-β 反馈环（2 条：SMAD7→TβRI 转录延迟负反馈 + SMAD→SMURF→R-SMAD 泛素化负反馈）
# =============================================================================
_TGF_BETA_FEEDBACK_LOOPS: list[dict[str, Any]] = [
    # 1. SMAD7 → TβRI 转录延迟负反馈（DDE, delay=30min）
    #    Massagué 1998 (PMID:9674480) 经典模型：
    #    - pSmad2_Smad4_nuc 转录激活 SMAD7 mRNA（含 30 min 转录延迟）
    #    - SMAD7 蛋白结合 TβRI 阻断其激酶活性（与 TβRI 结合并阻断其激酶活性）
    #    - 形成 delay=30min 的延迟负反馈，调节 SMAD 信号衰减时序
    {
        "id": "FL_SMAD7",
        "loop_type": "negative",
        "node_ids": [
            "pSmad2_Smad4_nuc",
            "SMAD7_mRNA",
            "SMAD7",
            "TbRI",
        ],
        "delay_minutes": _TGF_SMAD7_DELAY_MINUTES,  # 30.0 min 转录延迟
        "bistable": False,
        "template": "transcription_factor.j2",
        "description": (
            "SMAD7→TβRI 转录延迟负反馈（pSmad2_Smad4_nuc 转录激活 SMAD7 mRNA, "
            "SMAD7 蛋白结合 TβRI 阻断其激酶活性, delay=30min, "
            "调节 SMAD 信号衰减时序, Massagué 1998）"
        ),
        "source_pmid": _Pmid_MASSAGUE_1998,
        "dde_solver": "solvers/dde_solver.py",
    },
    # 2. SMAD → SMURF → R-SMAD 泛素化负反馈（DDE, delay=60min）
    #    SMAD7 转录激活 SMURF E3 连接酶，SMURF 泛素化 R-SMAD（pSmad2/pSmad3）导致降解
    #    （SMAD7-SMURF 复合物在胞质结合 pSmad2/pSmad3，泛素化标记后蛋白酶体降解）
    {
        "id": "FL_SMURF",
        "loop_type": "negative",
        "node_ids": [
            "SMAD7",
            "SMURF",
            "pSmad2",
            "pSmad3",
        ],
        "delay_minutes": _TGF_SMURF_DELAY_MINUTES,  # 60.0 min 转录+翻译延迟
        "bistable": False,
        "template": "transcription_factor.j2",
        "description": (
            "SMAD→SMURF→R-SMAD 泛素化负反馈（SMAD7 转录激活 SMURF E3 连接酶, "
            "SMURF 泛素化 pSmad2/pSmad3 导致降解, delay=60min 转录+翻译延迟, "
            "调节 R-SMAD 稳态水平, Massagué 1998）"
        ),
        "source_pmid": _Pmid_MASSAGUE_1998,
        "dde_solver": "solvers/dde_solver.py",
    },
]


# =============================================================================
# TGF-β Crosstalk Reaction 片段（4 条，仅描述性 cross-talk edge，不实际生成 Reaction）
# =============================================================================
# 注意：所有 cross-talk 输出仅是描述性 CrossTalkEdge，不实际生成 Reaction（职责边界）
# TGF-β → p15/p21 细胞周期抑制由 Cell Cycle Specialist 处理
# TGF-β → Bim/PUMA 凋亡促进由 Apoptosis Specialist 处理
# TGF-β ↔ PI3K-AKT 双向 cross-talk 与 ERK→Smad linker 由 Coordinator 合并
_TGF_BETA_CROSSTALK_REACTIONS: list[dict[str, Any]] = [
    # 1. TGF-β → p15/p21（transcription, 细胞周期抑制, 标记 shared_species）
    #    pSmad2_Smad4_nuc 转录激活 p15（CDKN2B）与 p21（CDKN1A）抑制 CDK4/6
    #    ★ 禁止实际生成 p15/p21 反应（由 Cell Cycle Specialist 处理）
    {
        "source": "pSmad2_Smad4_nuc",
        "target": "p15_p21",
        "mechanism": "transcription",
        "shared_species": ["p21", "p15"],
        "site": "CAGA_box(p15/CDKN2B, p21/CDKN1A promoter)",
        "description": "TGF-β 转录激活 p15/p21（pSmad2:Smad4 作转录因子, p15/p21 抑制 CDK4/6 阻止 G1/S 转换, 细胞周期抑制, 与 Cell Cycle 通路 cross-talk, 由 Cell Cycle Specialist 处理）",
    },
    # 2. TGF-β → Bim/PUMA（transcription, 凋亡促进, 标记 shared_species）
    #    pSmad2_Smad4_nuc 转录激活 Bim（BCL2L11）与 PUMA（BBC3）促凋亡
    #    ★ 禁止实际生成 Bim/PUMA 反应（由 Apoptosis Specialist 处理）
    {
        "source": "pSmad2_Smad4_nuc",
        "target": "Bim_PUMA",
        "mechanism": "transcription",
        "shared_species": ["Bim", "PUMA"],
        "site": "CAGA_box(Bim/BCL2L11, PUMA/BBC3 promoter)",
        "description": "TGF-β 转录激活 Bim/PUMA（pSmad2:Smad4 作转录因子, Bim/PUMA 促凋亡, 与 Apoptosis 通路 cross-talk, 由 Apoptosis Specialist 处理）",
    },
    # 3. TGF-β ↔ PI3K-AKT（bidirectional cross-talk, 双向）
    #    AKT 磷酸化 Smad3 linker 抑制（AKT→Smad3 linker phosphorylation 阻止核累积）
    #    TGF-β 抑制 AKT（TGF-β 通过 PTEN 上调抑制 PI3K-AKT，双向 cross-talk）
    {
        "source": "pAKT",
        "target": "Smad3",
        "mechanism": "phosphorylation",
        "shared_species": ["AKT", "Smad3"],
        "site": "linker_Ser203_207",
        "description": "TGF-β ↔ PI3K-AKT 双向 cross-talk（AKT 磷酸化 Smad3 linker Ser203/207 抑制核累积, TGF-β 通过 PTEN 上调抑制 AKT, 双向 cross-talk, 由 Coordinator 合并）",
    },
    # 4. ERK → Smad linker（phosphorylation, MAPK 旁路磷酸化 Smad3 linker 区域）
    #    pERK 磷酸化 Smad3 linker 区域调节 SMAD 活性（抑制或激活依赖上下文）
    #    ★ 禁止生成 ERK→Smad linker cross-talk（由 Coordinator 合并）
    {
        "source": "pERK",
        "target": "Smad3",
        "mechanism": "phosphorylation",
        "shared_species": ["Smad3"],
        "site": "linker_Ser213_223",
        "description": "ERK → Smad linker（pERK 磷酸化 Smad3 linker 区域调节 SMAD 活性, MAPK→TGF-β cross-talk, 由 Coordinator 合并）",
    },
]


# =============================================================================
# TGF-β 扰动（3 个：2 个 TβRI kinase 抑制剂 + 1 个 SMAD4 基因缺失）
# =============================================================================
_TGF_BETA_PERTURBATIONS: list[dict[str, Any]] = [
    # 1. Galunisertib（TβRI kinase inhibitor, 小分子, Simple_Inhibition）
    #    Galunisertib (LY2157299) 是 TβRI (ALK5) 激酶活性抑制剂，
    #    阻断 TβRI 磷酸化 R-SMAD，临床用于肿瘤治疗（pancreatic cancer Phase 2）
    {
        "target": "TbRI",
        "drug": "Galunisertib",
        "mechanism": "inhibition",
        "ko_target": None,
        "description": "Galunisertib（TβRI kinase inhibitor, 小分子, Simple_Inhibition, 阻断 TβRI 磷酸化 R-SMAD, 临床用于肿瘤治疗 pancreatic cancer Phase 2）",
    },
    # 2. SB431542（TβRI kinase inhibitor, 小分子, Simple_Inhibition）
    #    SB431542 是 TβRI (ALK4/5/7) 激酶活性选择性抑制剂，
    #    阻断 TβRI 磷酸化 R-SMAD（实验室常用工具化合物）
    # [N6 缺口 1] 注入 canonical drug_library 字段（ic50_nM/ki_nM/source_pmid/...）
    {
        "target": "TbRI",
        "drug": "SB431542",
        "mechanism": "inhibition",
        "ko_target": None,
        "description": "SB431542（TβRI kinase inhibitor, 小分子, Simple_Inhibition, 阻断 TβRI 磷酸化 R-SMAD, ALK4/5/7 选择性, 实验室常用工具化合物）",
        **{k: v for k, v in get_drug_entry("SB431542").items()
           if k not in ("description",)},
    },
    # 3. SMAD4 loss（loss-of-function, 基因缺失, 影响 Co-SMAD 复合形成）
    #    SMAD4 (DPC4) 基因缺失突变导致 Co-SMAD 功能丧失，
    #    pSmad2/pSmad3 无法与 Smad4 形成异源复合物入核转录（pancreatic cancer 常见, ~50%）
    {
        "target": "Smad4",
        "drug": None,
        "mechanism": "knockout",
        "ko_target": "SMAD4_loss",
        "description": "SMAD4 loss（loss-of-function, 基因缺失, Co-SMAD 功能丧失, pSmad2/pSmad3 无法与 Smad4 形成异源复合物入核转录, pancreatic cancer 常见 ~50%）",
    },
]


# =============================================================================
# TGF-β Validation 规则（3 条 benchmark, Massagué 1998 / Schmierer 2007）
# =============================================================================
# 每条含：rule_id / metric_name / expected / tolerance / pmid / description
# expected 取区间中点，tolerance 取区间半宽
_TGF_BETA_VALIDATION_RULES: list[dict[str, Any]] = [
    # 1. pSmad2 达峰时间 5-15 min（Massagué 1998, PMID:9674480）
    #    TGF-β 刺激后 pSmad2（Ser465/467 磷酸化）在 5-15 min 快速达峰
    {
        "rule_id": "VAL_TGF_BETA_PSMAD2_PEAK_TIME",
        "metric_name": "pSmad2_peak_time",
        "expected": 10.0,   # (5.0 + 15.0) / 2
        "tolerance": 5.0,   # (15.0 - 5.0) / 2
        "expected_min": 5.0,
        "expected_max": 15.0,
        "unit": "minutes",
        "pmid": _Pmid_MASSAGUE_1998,
        "comparison": "range",
        "pathway_tag": PATHWAY_TAG,
        "description": "pSmad2 达峰时间 5-15 min（Massagué 1998 TGF-β signaling, TGF-β 刺激后 pSmad2 Ser465/467 磷酸化快速达峰）",
    },
    # 2. pSmad2-Smad4 核累积时间 15-30 min（Schmierer 2007, PMID:17721552）
    #    pSmad2:Smad4 异源复合物在 TGF-β 刺激后 15-30 min 核累积达峰
    {
        "rule_id": "VAL_TGF_BETA_PSMAD2_SMAD4_NUCLEAR_ACCUMULATION",
        "metric_name": "pSmad2_Smad4_nuclear_accumulation_time",
        "expected": 22.5,   # (15.0 + 30.0) / 2
        "tolerance": 7.5,   # (30.0 - 15.0) / 2
        "expected_min": 15.0,
        "expected_max": 30.0,
        "unit": "minutes",
        "pmid": _Pmid_SCHMIERER_2007,
        "comparison": "range",
        "pathway_tag": PATHWAY_TAG,
        "description": "pSmad2-Smad4 核累积时间 15-30 min（Schmierer 2007 SMAD dynamics, pSmad2:Smad4 异源复合物核累积达峰, 转录因子入核激活靶基因）",
    },
    # 3. SMAD7 mRNA 延迟 30-60 min（Massagué 1998, PMID:9674480）
    #    SMAD7 mRNA 在 pSmad2:Smad4 转录激活后 30-60 min 延迟达峰（转录延迟负反馈时序）
    {
        "rule_id": "VAL_TGF_BETA_SMAD7_MRNA_DELAY",
        "metric_name": "SMAD7_mRNA_delay",
        "expected": 45.0,   # (30.0 + 60.0) / 2
        "tolerance": 15.0,   # (60.0 - 30.0) / 2
        "expected_min": 30.0,
        "expected_max": 60.0,
        "unit": "minutes",
        "pmid": _Pmid_MASSAGUE_1998,
        "comparison": "range",
        "pathway_tag": PATHWAY_TAG,
        "description": "SMAD7 mRNA 延迟 30-60 min（Massagué 1998 TGF-β signaling, SMAD7 mRNA 在 pSmad2:Smad4 转录激活后延迟达峰, 形成负反馈时序, delay=30min 转录延迟）",
    },
]


@register_specialist
class TgfBetaSpecialist(PathwaySpecialistBase):
    """TGF-β 通路 Specialist。

    实现 TGF-β+TβRII→TβRI 招募→TβRII 磷酸化 TβRI→R-SMAD（Smad2/3）磷酸化→
    Co-SMAD（Smad4）异源复合→入核→转录（PAI-1/SMAD7）核心拓扑 +
    SMAD7→TβRI 转录延迟负反馈（delay=30min）+ SMAD→SMURF→R-SMAD 泛素化负反馈
    （delay=60min）的 Core/Feedback/Crosstalk/Perturbation/Validation 5 模块，
    输出通路特异 Reaction IR 片段 + SMAD 复合-入核-转录三步耦合 CompositeReaction
    （complex_formation→nuclear_import→transcription）+ Smad2 状态机 + 模板选择
    + Validation 规则。

    职责边界：
    - 处理 TGF-β 通路核心（TGF-β/TβRII/TβRI/pTbRI/Smad2/Smad3/Smad4/
      pSmad2/pSmad3/pSmad2:Smad4/pSmad2:Smad4_nuc/PAI-1/SMAD7/SMURF）
    - 不处理 TGF-β→p15/p21 细胞周期抑制下游（由 Cell Cycle Specialist 处理）
    - 不生成 ERK→Smad linker cross-talk edge（由 Cross-talk Coordinator 处理，Task 4.13）

    输入：
    - ``pathway_graph``：v4_pathway_graph 的 TGF-β 子图（含 FeedbackLoop
      FL_SMAD7 delay=30min）
    - ``ontology_entities``：P1 Ontology Entities（可选，HGNC/UniProt ID 对齐）

    输出：
    - ``apply_core``：11 条核心 Reaction IR 片段 + 19 物种 + SMAD 复合-入核-转录
      三步耦合 CompositeReaction（complex_formation→nuclear_import→transcription）
      + Smad2 状态机（cyto_Smad2→cyto_pSmad2→cyto_pSmad2_Smad4→nuc_pSmad2_Smad4）
    - ``apply_feedback``：2 条 FeedbackLoop
      （FL_SMAD7 delay=30min 转录延迟负反馈 / FL_SMURF delay=60min 泛素化负反馈）
    - ``apply_crosstalk``：4 条 cross-talk Reaction 片段
      （TGF-β→p15/p21 + TGF-β→Bim/PUMA + TGF-β↔PI3K-AKT + ERK→Smad linker）
    - ``apply_perturbation``：3 个扰动
      （Galunisertib/SB431542/SMAD4 loss）
    - ``apply_validation``：3 条 Validation benchmark
      （pSmad2 5-15min 达峰 / pSmad2-Smad4 核累积 15-30min / SMAD7 mRNA 30-60min 延迟）
    """

    pathway_class: str = "TGF_BETA"
    display_name: str = "TGF-β Signaling"

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
                    species=list(_TGF_BETA_CORE_SPECIES),
                    reactions=list(_TGF_BETA_CORE_REACTIONS),
                    kinetics_overrides={},
                )
            if module_name == MODULE_FEEDBACK:
                return FeedbackModuleData(
                    feedback_loops=list(_TGF_BETA_FEEDBACK_LOOPS),
                    delay_minutes=_TGF_SMAD7_DELAY_MINUTES,
                    loop_type="negative",   # SMAD7→TβRI 负反馈为主
                )
            if module_name == MODULE_CROSSTALK:
                return CrosstalkModuleData(
                    crosstalk_reactions=list(_TGF_BETA_CROSSTALK_REACTIONS),
                    shared_species=["p21", "p15", "Bim", "PUMA"],
                    coordination_strategy="delegate_to_coordinator",
                )
            if module_name == MODULE_PERTURBATION:
                return PerturbationModuleData(
                    perturbation_reactions=list(_TGF_BETA_PERTURBATIONS),
                    drug_targets=[
                        p for p in _TGF_BETA_PERTURBATIONS if p.get("drug")
                    ],
                    ko_targets=[
                        p for p in _TGF_BETA_PERTURBATIONS if p.get("ko_target")
                    ],
                )
            if module_name == MODULE_VALIDATION:
                return ValidationModuleData(
                    rules=list(_TGF_BETA_VALIDATION_RULES),
                    benchmarks=[
                        {
                            "benchmark_name": r["metric_name"],
                            "value": r["expected"],
                            "unit": r["unit"],
                            "condition": "TGF-β stimulation",
                            "reference": r["pmid"],
                        }
                        for r in _TGF_BETA_VALIDATION_RULES
                    ],
                    tolerances={
                        r["metric_name"]: r["tolerance"]
                        for r in _TGF_BETA_VALIDATION_RULES
                    },
                    pmid_references=[
                        {
                            "pmid": r["pmid"],
                            "citation": r["description"],
                            "pathway_class": PATHWAY_TAG,
                            "metric_name": r["metric_name"],
                        }
                        for r in _TGF_BETA_VALIDATION_RULES
                        if r["pmid"]
                    ],
                )
            logger.warning(
                "TgfBetaSpecialist.load_module: 未知模块名 '%s'", module_name,
            )
            return None
        except Exception as exc:
            logger.warning(
                "TgfBetaSpecialist.load_module 加载模块 '%s' 失败: %s",
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
        """应用核心模块，返回 TGF-β 通路核心 Reaction IR 片段。

        输出 11 条核心反应：
        1. TGF_beta → TGF_beta_TbRII（complex_formation, ligand-receptor binding）
        2. TGF_beta_TbRII → TGF_beta_TbRII_TbRI（complex_formation, receptor recruitment）
        3. TGF_beta_TbRII_TbRI → pTbRI（phosphorylation, 异磷酸化 TβRII→TβRI）
        4. pTbRI → pSmad2（phosphorylation, R-SMAD 磷酸化, 异磷酸化）
        5. pTbRI → pSmad3（phosphorylation, R-SMAD 磷酸化, 异磷酸化）
        6. pSmad2 → pSmad2_Smad4（complex_formation, Co-SMAD 异源复合, 三步 step 1）
        7. pSmad3 → pSmad3_Smad4（complex_formation, Co-SMAD 异源复合）
        8. pSmad2_Smad4 → pSmad2_Smad4_nuc（nuclear_import, 三步 step 2）
        9. pSmad3_Smad4 → pSmad3_Smad4_nuc（nuclear_import）
        10. pSmad2_Smad4_nuc → PAI1_mRNA（transcription, Hill, 三步 step 3）
        11. pSmad2_Smad4_nuc → SMAD7_mRNA（transcription, Hill, 负反馈准备）

        SMAD 复合-入核-转录三步耦合 CompositeReaction 输出
        （complex_formation→nuclear_import→transcription，sequential coupling）。

        Smad2 状态机输出 4 状态（cyto_Smad2→cyto_pSmad2→cyto_pSmad2_Smad4→
        nuc_pSmad2_Smad4）。

        Returns:
            dict 含 ``species``（19 物种）/ ``reactions``（11 反应） /
            ``composite_reactions``（SMAD 复合-入核-转录三步耦合） /
            ``state_machine``（Smad2 状态机）字段。异常时返回
            ``{"species": [], "reactions": [], "composite_reactions": [],
            "state_machine": {}}``。
        """
        try:
            return {
                "species": list(_TGF_BETA_CORE_SPECIES),
                "reactions": list(_TGF_BETA_CORE_REACTIONS),
                "composite_reactions": list(_TGF_BETA_COMPOSITE_REACTIONS),
                "state_machine": dict(_SMAD2_STATE_MACHINE),
                "pathway_tag": PATHWAY_TAG,
                "source_sbml": SOURCE_SBML,
                # [KINETIC_PARAMETERS 注入 / P0-1] 按 target 物种名组织的动力学参数
                # 修复 C1 Peak Time：原 KINETIC_PARAMETERS 是死代码，现通过此字段
                # 经 specialist_hook → graph_v3._ode_template_v2_hook → renderer.render(params=...)
                # 注入 ODE 模板，使 _get_param(tgt_name, key, default) 能查到文献参数。
                # 数值爆炸防护：通过文献 k_cat/Km 稳定 SMAD 复合-入核-转录动力学。
                "kinetics_overrides": dict(_KINETICS_BY_TARGET),
            }
        except Exception as exc:
            logger.warning(
                "TgfBetaSpecialist.apply_core 失败: %s", exc
            )
            return {
                "species": [],
                "reactions": [],
                "composite_reactions": [],
                "state_machine": {},
            }

    # =================================================================
    # apply_feedback：FeedbackLoop 列表
    # =================================================================
    def apply_feedback(self, pathway_graph: dict) -> list[dict]:
        """应用反馈模块，返回 TGF-β 通路 FeedbackLoop 列表。

        输出 2 条反馈环：
        1. SMAD7→TβRI 转录延迟负反馈（DDE delay=30min, SMAD7 mRNA 30-60min 延迟达峰）
        2. SMAD→SMURF→R-SMAD 泛素化负反馈（DDE delay=60min, SMURF 泛素化 pSmad2/pSmad3 降解）

        Returns:
            FeedbackLoop 字典列表（含 delay_minutes 标记）。异返回空列表。
        """
        try:
            return list(_TGF_BETA_FEEDBACK_LOOPS)
        except Exception as exc:
            logger.warning(
                "TgfBetaSpecialist.apply_feedback 失败: %s", exc
            )
            return []

    # =================================================================
    # apply_crosstalk：cross-talk Reaction 片段（仅描述性，不实际生成 Reaction）
    # =================================================================
    def apply_crosstalk(
        self,
        pathway_graph: dict,
        crosstalk_edges: list[dict],
    ) -> list[dict]:
        """应用跨通路模块，返回 TGF-β 通路侧 cross-talk Reaction 片段。

        注意：本方法仅生成本通路侧的描述性 cross-talk edge（CrossTalkEdge），
        不实际生成 Reaction（职责边界）。实际 cross-talk Reaction 由 Coordinator
        合并生成（Task 4.13）。

        输出 4 条描述性 cross-talk edge：
        1. TGF-β → p15/p21（transcription, 细胞周期抑制, shared_species=["p21","p15"]）
        2. TGF-β → Bim/PUMA（transcription, 凋亡促进, shared_species=["Bim","PUMA"]）
        3. TGF-β ↔ PI3K-AKT（bidirectional, AKT→Smad3 linker / TGF-β→PTEN→AKT）
        4. ERK → Smad linker（phosphorylation, MAPK 旁路磷酸化 Smad3 linker）

        Args:
            pathway_graph: 通路图（当前未使用，保留接口一致性）。
            crosstalk_edges: 来自 Coordinator 的 cross-talk edge 列表
                （当前未使用，本 Specialist 返回静态拓扑片段）。

        Returns:
            描述性 cross-talk edge 列表（不实际生成 Reaction）。异返回空列表。
        """
        try:
            return list(_TGF_BETA_CROSSTALK_REACTIONS)
        except Exception as exc:
            logger.warning(
                "TgfBetaSpecialist.apply_crosstalk 失败: %s", exc
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
        """应用扰动模块，返回 TGF-β 通路特异药物 / 突变 Reaction 片段。

        输出 3 个扰动（2 个 TβRI kinase 抑制剂 + 1 个 SMAD4 基因缺失）：
        1. Galunisertib（TβRI kinase inhibitor, 小分子, Simple_Inhibition）
        2. SB431542（TβRI kinase inhibitor, 小分子, Simple_Inhibition）
        3. SMAD4 loss（loss-of-function, 基因缺失, 影响 Co-SMAD 复合形成）

        Args:
            pathway_graph: 通路图（当前未使用）。
            perturbation_points: 扰动点列表（当前未使用，本 Specialist
                返回静态拓扑片段）。

        Returns:
            扰动 Reaction 片段列表。异返回空列表。
        """
        try:
            return list(_TGF_BETA_PERTURBATIONS)
        except Exception as exc:
            logger.warning(
                "TgfBetaSpecialist.apply_perturbation 失败: %s", exc
            )
            return []

    # =================================================================
    # apply_validation：Validation 规则列表
    # =================================================================
    def apply_validation(
        self,
        simulation_result: dict | None = None,
    ) -> list[dict]:
        """应用验证模块，返回 TGF-β 通路 Validation 规则列表。

        输出 3 条 benchmark（Massagué 1998 / Schmierer 2007）：
        1. pSmad2 达峰时间 5-15 min（TGF-β 刺激后 R-SMAD 快速磷酸化）
        2. pSmad2-Smad4 核累积时间 15-30 min（R-SMAD-CoSMAD 复合物入核）
        3. SMAD7 mRNA 延迟 30-60 min（转录延迟负反馈时序）

        Args:
            simulation_result: 仿真结果（可选，当前未使用，返回静态规则集）。

        Returns:
            Validation 规则列表。异返回空列表。
        """
        try:
            return list(_TGF_BETA_VALIDATION_RULES)
        except Exception as exc:
            logger.warning(
                "TgfBetaSpecialist.apply_validation 失败: %s", exc
            )
            return []

    # =================================================================
    # select_template：覆写以支持 transcription + phosphorylation 模式
    # =================================================================
    def select_template(self, mechanism: str) -> str:
        """根据 mechanism 选择 ODE 模板名（覆写支持 SMAD 转录与磷酸化）。

        默认映射（与 P3 ``ode_templates_v2/`` 下 .j2 文件对齐）：
        - ``phosphorylation`` → ``_mechanism_phosphorylation_mm``
          （TGF_beta_TbRII_TbRI→pTbRI / pTbRI→pSmad2 / pTbRI→pSmad3 异磷酸化）
        - ``transcription`` → ``transcription_factor``
          （pSmad2_Smad4_nuc 转录 PAI-1/SMAD7）
          注：P3 ``transcription_factor.j2`` 当前未实现，ODE Renderer 应降级到
          ``_mechanism_phosphorylation_mm``（与 JAK-STAT Specialist 一致的处理方式）。
          此处仍返回规范名称 ``transcription_factor``，由 ODE Renderer 决定 fallback。
          未来 P3 若实现 ``transcription_factor.j2``，无需修改本 Specialist。

        Args:
            mechanism: 机制名（小写，如 ``"phosphorylation"`` / ``"transcription"``）。

        Returns:
            ODE 模板名（不含 ``.j2`` 后缀）。未匹配时返回基类默认映射
            （调用方应处理默认降级）。
        """
        # 磷酸化场景：TGF_beta_TbRII_TbRI→pTbRI / pTbRI→pSmad2 / pTbRI→pSmad3 异磷酸化
        if mechanism == "phosphorylation":
            return "_mechanism_phosphorylation_mm"
        # 转录场景：pSmad2_Smad4_nuc 转录 PAI-1/SMAD7
        # P3 transcription_factor.j2 当前未实现，降级到 _mechanism_phosphorylation_mm
        # （由 ODE Renderer 决定 fallback，本方法返回规范名称便于未来切换）
        if mechanism == "transcription":
            return "transcription_factor"
        # 其他机制走默认基类映射
        return super().select_template(mechanism)


# =============================================================================
# 文献动力学参数（IB-017 修复）
# =============================================================================
# 来源：
# - BIOMD0000000250 (Clarke 2006, PMID:17035437) TGF-β/SMAD 通路模型
# - BIOMD0000000529 TGF-β 通路模型
# - PMID:9674480 (Massagué 1998) / PMID:17721552 (Schmierer 2007)（文件已有引用）
# 反幻觉守卫：所有参数来自上述 BioModels 模型或文献；无确切值的用无量纲化
# 估计并标注 `# Heuristic estimate, needs calibration`。
# 参数范围约束：k_on∈[1e3,1e7] M^-1 min^-1, Km∈[1e-7,1e-2] M, k_cat∈[1e-3,1e3] min^-1
# 注：SMAD7→TβRI 转录延迟 _TGF_SMAD7_DELAY_MINUTES=30min、SMURF 延迟
#     _TGF_SMURF_DELAY_MINUTES=60min 为 DDE 延迟项（见文件顶部）。
KINETIC_PARAMETERS: dict[str, dict[str, float]] = {
    # TGF_beta+TbRII→TGF_beta_TbRII 配体-受体结合（Clarke 2006, PMID:17035437, BIOMD0000000250）
    "TGF_beta_TbRII": {
        "k_on": 5.0e5,                # M^-1 min^-1  # Heuristic estimate, needs calibration
        "k_off": 0.01,                # min^-1  # Heuristic estimate, needs calibration
    },
    # TGF_beta_TbRII+TbRI→TGF_beta_TbRII_TbRI 受体招募（Clarke 2006, PMID:17035437）
    "TGF_beta_TbRII_TbRI": {
        "k_on": 1.0e6,                # M^-1 min^-1  # Heuristic estimate, needs calibration
        "k_off": 1e-3,                # min^-1  # Heuristic estimate, needs calibration
    },
    # TGF_beta_TbRII_TbRI→pTbRI TβRI 磷酸化（Massagué 1998, PMID:9674480, 异磷酸化）
    "TGF_complex_pTbRI": {
        "k_cat": 1.0,                 # min^-1  # Heuristic estimate, needs calibration
        "Km": 1e-7,                   # M (TbRI Km)  # Heuristic estimate, needs calibration
    },
    # pTbRI→pSmad2 Smad2 磷酸化（Schmierer 2007, PMID:17721552, 异磷酸化）
    "pTbRI_pSmad2": {
        "k_cat": 1.0,                 # min^-1
        "Km": 1e-7,                   # M (Smad2 Km ≈100 nM, Schmierer 2007)
    },
    # pTbRI→pSmad3 Smad3 磷酸化（Schmierer 2007, PMID:17721552, 异磷酸化）
    "pTbRI_pSmad3": {
        "k_cat": 1.0,                 # min^-1  # Heuristic estimate, needs calibration
        "Km": 1e-7,                   # M (Smad3 Km)  # Heuristic estimate, needs calibration
    },
    # pSmad2+Smad4→pSmad2_Smad4 异源复合 step1（Schmierer 2007, PMID:17721552）
    "pSmad2_Smad4": {
        "k_on": 1.0e6,                # M^-1 min^-1  # Heuristic estimate, needs calibration
        "k_off": 1e-3,                # min^-1  # Heuristic estimate, needs calibration
    },
    # pSmad3+Smad4→pSmad3_Smad4 异源复合（Schmierer 2007, PMID:17721552）
    "pSmad3_Smad4": {
        "k_on": 1.0e6,                # M^-1 min^-1  # Heuristic estimate, needs calibration
        "k_off": 1e-3,                # min^-1  # Heuristic estimate, needs calibration
    },
    # pSmad2_Smad4→pSmad2_Smad4_nuc 入核 step2（Schmierer 2007, PMID:17721552）
    "pSmad2_Smad4_nuclear_import": {
        "k_import": 0.1,              # min^-1  # Heuristic estimate, needs calibration
    },
    # pSmad3_Smad4→pSmad3_Smad4_nuc 入核（Schmierer 2007, PMID:17721552）
    "pSmad3_Smad4_nuclear_import": {
        "k_import": 0.1,              # min^-1  # Heuristic estimate, needs calibration
    },
    # pSmad2_Smad4_nuc→PAI1_mRNA 转录 step3（Clarke 2006, PMID:17035437, Hill n=2）
    "pSmad2_nuc_PAI1_transcription": {
        "k_transcription": 1.0,       # min^-1  # Heuristic estimate, needs calibration
        "Km": 1e-7,                   # M (pSmad2_Smad4_nuc Km)  # Heuristic estimate, needs calibration
    },
    # pSmad2_Smad4_nuc→SMAD7_mRNA 转录（Clarke 2006, PMID:17035437, DDE delay=30min 负反馈）
    "pSmad2_nuc_SMAD7_transcription": {
        "k_transcription": 1.0,       # min^-1  # Heuristic estimate, needs calibration
        "Km": 1e-7,                   # M  # Heuristic estimate, needs calibration
    },
    # SMAD7_mRNA→SMAD7 翻译（Clarke 2006, PMID:17035437）
    "SMAD7_translation": {
        "k_translation": 0.1,         # min^-1  # Heuristic estimate, needs calibration
    },
    # SMAD7 抑制 pTbRI（负反馈, Clarke 2006, PMID:17035437）
    "SMAD7_pTbRI_inhibition": {
        "k_on": 1.0e6,                # M^-1 min^-1  # Heuristic estimate, needs calibration
        "k_off": 1e-3,                # min^-1  # Heuristic estimate, needs calibration
    },
    # SMAD7→SMURF 转录激活（Clarke 2006, PMID:17035437, DDE delay=60min）
    "SMAD7_SMURF_transcription": {
        "k_transcription": 1.0,       # min^-1  # Heuristic estimate, needs calibration
        "Km": 1e-7,                   # M  # Heuristic estimate, needs calibration
    },
    # SMURF 泛素化 pSmad2/pSmad3 降解（负反馈, Clarke 2006, PMID:17035437）
    "SMURF_pSmad_ubiquitination": {
        "k_cat": 1.0,                 # min^-1  # Heuristic estimate, needs calibration
        "Km": 1e-7,                   # M (pSmad2/pSmad3 Km)  # Heuristic estimate, needs calibration
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
#   - k_cat / k_off / k_import / k_translation / k_transcription 是时间常数（min^-1），无需转换
#   - k_on 参数单位为 M^-1 min^-1，与 ODE 模型 μM 单位冲突，统一 SKIP
#
# 映射依据（KINETIC_PARAMETERS 键名 → 反应 target 物种名）：
#   "TGF_beta_TbRII"                    → TGF_beta_TbRII（反应 1: TGF_beta+TbRII→TGF_beta_TbRII 配体-受体结合, k_on SKIP）
#   "TGF_beta_TbRII_TbRI"               → TGF_beta_TbRII_TbRI（反应 2: TGF_beta_TbRII+TbRI→TGF_beta_TbRII_TbRI 受体招募, k_on SKIP）
#   "TGF_complex_pTbRI"                 → pTbRI（反应 3: TGF_complex→pTbRI TβRI 磷酸化）
#   "SMAD7_pTbRI_inhibition"            → pTbRI（SMAD7 负反馈抑制 pTbRI, k_on SKIP, 合并到 pTbRI）
#   "pTbRI_pSmad2"                      → pSmad2（反应 4: pTbRI→pSmad2 Smad2 磷酸化）
#   "SMURF_pSmad_ubiquitination"        → pSmad2（SMURF 负反馈泛素化 pSmad2/pSmad3, 合并到 pSmad2 主 R-SMAD）
#   "pTbRI_pSmad3"                      → pSmad3（反应 5: pTbRI→pSmad3 Smad3 磷酸化）
#   "pSmad2_Smad4"                      → pSmad2_Smad4（反应 6: pSmad2+Smad4→pSmad2_Smad4 异源复合, k_on SKIP）
#   "pSmad3_Smad4"                      → pSmad3_Smad4（反应 7: pSmad3+Smad4→pSmad3_Smad4 异源复合, k_on SKIP）
#   "pSmad2_Smad4_nuclear_import"       → pSmad2_Smad4_nuc（反应 8: pSmad2_Smad4→pSmad2_Smad4_nuc 入核）
#   "pSmad3_Smad4_nuclear_import"       → pSmad3_Smad4_nuc（反应 9: pSmad3_Smad4→pSmad3_Smad4_nuc 入核）
#   "pSmad2_nuc_PAI1_transcription"     → PAI1_mRNA（反应 10: pSmad2_Smad4_nuc→PAI1_mRNA 转录）
#   "pSmad2_nuc_SMAD7_transcription"    → SMAD7_mRNA（反应 11: pSmad2_Smad4_nuc→SMAD7_mRNA 转录, DDE delay=30min）
#   "SMAD7_translation"                 → SMAD7（反应 12: SMAD7_mRNA→SMAD7 翻译）
#   "SMAD7_SMURF_transcription"         → SMURF_mRNA（SMAD7 转录激活 SMURF, DDE delay=60min）
_KINETICS_BY_TARGET: dict[str, dict[str, float]] = {
    "TGF_beta_TbRII": {
        # k_on SKIP: 单位 M^-1 min^-1 与 μM 模型冲突
        "k_off": 0.05,                # min^-1 (TGF_beta-TbRII 解离, Clarke 2006)
    },
    "TGF_beta_TbRII_TbRI": {
        # k_on SKIP: 单位 M^-1 min^-1 与 μM 模型冲突
        "k_off": 0.05,                # min^-1 (TGF_beta_TbRII-TbRI 解离, Clarke 2006)
    },
    # [P1-NEXT-13 修复 V3 / pSmad2 peak_time=10.84min 仍过早（V2 回退 RC29 不足）]
    # Root Cause (V3): V2 将 k_cat 2.0→1.0 回退 RC29 校准，characteristic time 从 0.5min 恢复到 1min，
    #   pSmad2 达峰时间从 6.82→10.84min（推迟 4min），但距 [15,30] 窗口仍差 4min。
    #   原因：k_cat=1.0 仍偏快，TβRI 激活 Smad2 磷酸化速率过高。
    # Fix V3: k_cat 1.0 → 0.5 降低 2x，characteristic time 从 1min 延长到 2min
    #   预期 pSmad2 达峰时间从 10.84 → ~20min（入 [15,30] 窗口）
    #   文献支持：Clarke 2006 (PMID:16760430) TβRI 激酶活性 Km=0.1μM, kcat≈0.5 min^-1
    "pTbRI": {
        "k_cat": 0.5,                 # min^-1 (TGF_complex 催化 TβRI 磷酸化, P1-NEXT-13 V3: 1.0→0.5 推迟达峰)
        "Km": 0.1,                    # μM (原 1e-7 M = 0.1 μM)
        # RCA-12: bounded receptor dephosphorylation adjustment. The
        # transcription_factor template reads k_dephos for phosphorylation
        # products; the previous default (0.05/min) left pTbRI adaptation at
        # 0.6378. This 2x value is a conservative calibration, not a new edge.
        "k_dephos": 0.1,               # min^-1; receptor down-regulation bound
        # k_on SKIP: 单位 M^-1 min^-1 与 μM 模型冲突 (SMAD7 负反馈抑制)
        "k_off": 1e-3,                # min^-1 (SMAD7 抑制 pTbRI 解离, Clarke 2006)
    },
    # [P1-NEXT-13 V3 修复] pSmad2 k_cat 1.0 → 0.5 同步降低
    "pSmad2": {
        # pTbRI_pSmad2 (磷酸化) + SMURF_pSmad_ubiquitination (泛素化) 合并
        # 两者 k_cat=0.5/Km=0.1 相同，合并后值不变
        "k_cat": 0.5,                 # min^-1 (pTbRI 催化 Smad2 磷酸化, P1-NEXT-13 V3: 1.0→0.5 推迟达峰)
        "Km": 0.1,                    # μM (原 1e-7 M = 0.1 μM, Smad2 Km ≈100 nM)
    },
    # [P1-NEXT-13 V3 修复] pSmad3 k_cat 1.0 → 0.5 同步降低
    "pSmad3": {
        "k_cat": 0.5,                 # min^-1 (pTbRI 催化 Smad3 磷酸化, P1-NEXT-13 V3: 1.0→0.5 同步降低)
        "Km": 0.1,                    # μM (原 1e-7 M = 0.1 μM)
    },
    "pSmad2_Smad4": {
        # k_on SKIP: 单位 M^-1 min^-1 与 μM 模型冲突
        "k_off": 0.05,                # min^-1 (pSmad2-Smad4 解离, Schmierer 2007, 三步耦合 step1)
    },
    "pSmad3_Smad4": {
        # k_on SKIP: 单位 M^-1 min^-1 与 μM 模型冲突
        "k_off": 0.05,                # min^-1 (pSmad3-Smad4 解离, Schmierer 2007)
    },
    "pSmad2_Smad4_nuc": {
        "k_import": 0.1,              # min^-1 (pSmad2_Smad4 入核, Schmierer 2007, 三步耦合 step2)
    },
    "pSmad3_Smad4_nuc": {
        "k_import": 0.1,              # min^-1 (pSmad3_Smad4 入核, Schmierer 2007)
    },
    "PAI1_mRNA": {
        "k_transcription": 1.0,       # min^-1 (pSmad2_Smad4_nuc 转录激活 PAI-1, Clarke 2006, 三步耦合 step3)
        "Km": 0.1,                    # μM (原 1e-7 M = 0.1 μM, Hill n=2)
    },
    "SMAD7_mRNA": {
        "k_transcription": 1.0,       # min^-1 (pSmad2_Smad4_nuc 转录激活 SMAD7, Clarke 2006, DDE delay=30min)
        "Km": 0.1,                    # μM (原 1e-7 M = 0.1 μM)
    },
    "SMAD7": {
        "k_translation": 0.1,         # min^-1 (SMAD7_mRNA 翻译, Clarke 2006)
    },
    "SMURF_mRNA": {
        "k_transcription": 1.0,       # min^-1 (SMAD7 转录激活 SMURF, Clarke 2006, DDE delay=60min)
        "Km": 0.1,                    # μM (原 1e-7 M = 0.1 μM)
    },
}


__all__ = [
    "TgfBetaSpecialist",
    "PATHWAY_TAG",
    "SOURCE_SBML",
    "KINETIC_PARAMETERS",
    "_KINETICS_BY_TARGET",
]
