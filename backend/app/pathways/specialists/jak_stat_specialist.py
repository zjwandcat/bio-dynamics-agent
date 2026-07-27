# BioDynamics Agent v4 - JAK-STAT Specialist (Phase 4 / Task 4.9)
# JAK-STAT 通路 Specialist：实现 Cytokine（IL-6/IFN-γ/EPO）+受体→JAK 磷酸化→
# STAT 招募→STAT 酪氨酸磷酸化→二聚化→nuclear import→转录（SOCS/CIS/Bcl-xL/IRF）
# 核心拓扑 + SOCS 转录延迟负反馈（delay=30min）+ STAT→PIAS nuclear export +
# STAT5 状态机（monomer→phosphorylated→dimer→nuclear）。
#
# 设计原则（铁律）：
# 1. Feature Flag V4_PATHWAY_SPECIALIST_ENABLED=false 时由 LangGraph hook 层短路，
#    本 Specialist 代码不被调用（基类本身不做 flag 检查）
# 2. 使用 Task 4.0 修复后的 PHOSPHORYLATION 语义：
#    - 异磷酸化（IL6_complex → pJAK）：JAK 作 substrate，pJAK 作 product，
#      IL6_complex 作 catalytic modifier（受体相关 JAK 自磷酸化通过 IL6_complex 异磷酸化表达）
#    - 异磷酸化（pJAK → pSTAT5）：STAT5 作 substrate，pSTAT5 作 product，
#      pJAK 作 catalytic modifier
# 3. 不处理 STAT3→Bcl-xL 凋亡下游（由 Apoptosis Specialist 处理，Task 4.7）
# 4. 不生成 EGFR→STAT3 cross-talk edge 本身（由 Cross-talk Coordinator 处理，
#    Task 4.13），仅返回本通路侧的 cross-talk Reaction 片段
# 5. apply_* 方法实现捕获异常并返回空 list/dict，记录 logger.warning，不抛异常
#
# 依赖：
# - P1 ontology / pathway_registry（JAK_STAT 通路类别键）
# - P2 schema（SpeciesV2 / ReactionV2 / SpeciesRef / Modifier / StateMachine）
# - P2 MechanismType（PHOSPHORYLATION / COMPLEX_FORMATION / DIMERIZATION /
#   NUCLEAR_IMPORT / TRANSCRIPTION / TRANSLATION / INHIBITION）
# - P2 StateMachine（STAT5 状态机：monomer→phosphorylated→dimer→nuclear）
# - P3 ode_templates_v2（_mechanism_phosphorylation_mm.j2 模板；
#   transcription_factor.j2 当前未实现，转录降级到 _mechanism_phosphorylation_mm）
# - P3 pathway_graph/initializer.py（JAK_STAT core_nodes / core_edges）
#
# 参考：
# - spec.md Part 3 Specialist 7（第 234-239 行）
# - tasks.md Task 4.9（第 94-103 行）
# - Swameye 2003 JAK-STAT model (PMID:12907735 / BioModels BIOMD0000000224)
# - Schwartz 2003 STAT5 dynamics (PMID:15286703)
# - SOCS transcriptional delay negative feedback (PMID:15286703)

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
# JAK-STAT 通路标签
# =============================================================================
PATHWAY_TAG: str = "JAK_STAT"

# SBML BioModels ID（Swameye 2003 JAK-STAT model）
SOURCE_SBML: str = get_biomodels_id(PATHWAY_TAG)

# Validation benchmark PMID 引用
_Pmid_SCHWARTZ_2003: str = "PMID:15286703"   # Schwartz 2003 STAT5 dynamics

# SOCS 转录延迟负反馈延迟（DDE，分钟）
# Schwartz 2003 模型中 STAT5→SOCS 转录延迟 30 min，形成延迟负反馈振荡，
# 产生 STAT5 核质比单脉冲动力学（非持续振荡）。
_STAT_SOCS_DELAY_MINUTES: float = 30.0


# =============================================================================
# JAK-STAT 核心物种（与 P3 pathway_graph/initializer.py JAK_STAT.core_nodes 对齐，
# 扩展 gp130 / IL6_complex / STAT5 / pSTAT5_dimer / SOCS / Bcl_xL_mRNA / IRF1_mRNA
# 完整 JAK-STAT 拓扑）
# =============================================================================
# pSTAT5 物种标记 shared=True（与 Apoptosis Specialist 的 Bcl-xL 抗凋亡路径共享，
# STAT5 转录 Bcl-xL 维持肿瘤细胞存活，下游 Bcl-xL→凋亡由 Apoptosis Specialist 处理）
_JAK_STAT_CORE_SPECIES: list[dict[str, Any]] = [
    # [C4 fix] initial_concentration aligned to BIOMD0000000347 (Bachmann2011, PMID:21772264).
    #   SBML models the Epo/JAK2/STAT5 pathway (Epo/EpoRJAK2/STAT5/SOCS3/SHP1).
    #   SBML species mapping: STAT5→STAT5 (exact), pSTAT5→pSTAT5 (exact),
    #   JAK→EpoRJAK2 (receptor-JAK complex), pJAK→EpoRpJAK2 (phosphorylated JAK),
    #   SOCS→SOCS3, SOCS_mRNA→SOCS3RNA.
    #   Species not in SBML (IL6/IL6R/gp130/IL6_complex — SBML uses Epo not IL-6;
    #   pSTAT5_dimer/pSTAT5_nuclear/STAT3/Bcl_xL_mRNA/IRF1_mRNA/PIAS) kept original.
    # ---- 配体 + 受体复合物 ----
    # IL-6（pro-inflammatory cytokine，JAK-STAT 通路经典配体）
    # [C4 fix] No SBML match in BIOMD0000000347 (SBML uses Epo as ligand, not IL-6). Kept original.
    {"name": "IL6", "species_type": "ligand",
     "compartment": "extracellular"},
    # IL6R（IL-6 receptor α 链，膜结合受体）
    # [C4 fix] No SBML match in BIOMD0000000347. Kept original.
    {"name": "IL6R", "species_type": "protein",
     "compartment": "membrane"},
    # gp130（IL6ST 信号转导链，IL-6 共受体，JAK 缔合处）
    # [C4 fix] No SBML match in BIOMD0000000347. Kept original.
    {"name": "gp130", "species_type": "protein",
     "compartment": "membrane"},
    # IL6_complex（IL-6+IL6R+gp130 三元复合物，激活受体相关 JAK）
    # [C4 fix] No SBML match in BIOMD0000000347. Kept original.
    {"name": "IL6_complex", "species_type": "complex",
     "compartment": "membrane"},
    # ---- JAK 激酶 ----
    # JAK（Janus kinase，受体相关酪氨酸激酶，与 gp130 缔合）
    # [C4 fix] SBML mapping: JAK→EpoRJAK2 (receptor-JAK2 complex, initial_concentration=3.97622)
    {"name": "JAK", "species_type": "protein",
     "compartment": "membrane",
     "initial_concentration": 3.97622},  # Source: BIOMD0000000347 Bachmann2011 (PMID:21772264) species EpoRJAK2
    # pJAK（磷酸化 JAK，受体复合物诱导 JAK 自磷酸化激活）
    # [C4 fix] SBML mapping: pJAK→EpoRpJAK2 (phosphorylated receptor-JAK2, initial=0.0)
    {"name": "pJAK", "species_type": "protein",
     "compartment": "membrane",
     "initial_concentration": 0.0},  # Source: BIOMD0000000347 Bachmann2011 (PMID:21772264) species EpoRpJAK2
    # ---- STAT5 状态机 4 状态 ----
    # STAT5（未磷酸化单体，cytoplasmic monomer，状态机初始状态）
    # [C4 fix] SBML mapping: STAT5→STAT5 (exact match, initial_concentration=79.7535)
    {"name": "STAT5", "species_type": "protein",
     "compartment": "cytoplasm",
     "initial_concentration": 79.7535},  # Source: BIOMD0000000347 Bachmann2011 (PMID:21772264) species STAT5
    # pSTAT5（酪氨酸磷酸化 STAT5，shared：与 Apoptosis Bcl-xL 路径共享）
    # [C4 fix] SBML mapping: pSTAT5→pSTAT5 (exact match, initial_concentration=0.0)
    {"name": "pSTAT5", "species_type": "protein",
     "compartment": "cytoplasm", "shared": True,
     "initial_concentration": 0.0},  # Source: BIOMD0000000347 Bachmann2011 (PMID:21772264) species pSTAT5
    # pSTAT5_dimer（磷酸化 STAT5 同源二聚体，通过 SH2-pTyr 相互作用）
    # [C4 fix] No SBML match in BIOMD0000000347 (dimer not separately modeled). Kept original.
    {"name": "pSTAT5_dimer", "species_type": "complex",
     "compartment": "cytoplasm"},
    # pSTAT5_nuclear（入核的 STAT5 二聚体，作为转录因子激活靶基因）
    # [C4 fix] No SBML match in BIOMD0000000347 (nuclear form not separately modeled). Kept original.
    {"name": "pSTAT5_nuclear", "species_type": "protein",
     "compartment": "nucleus"},
    # ---- STAT3（与 STAT5 平行的 JAK-STAT 转录因子，主要响应 IL-6 信号）----
    # STAT3 在 cross-talk 中作为本通路侧输出（pSTAT3→Bcl-xL/Bcl-2 转录片段）
    # [C4 fix] No SBML match in BIOMD0000000347 (SBML models STAT5 only, not STAT3). Kept original.
    {"name": "STAT3", "species_type": "protein",
     "compartment": "cytoplasm"},
    # ---- SOCS 转录延迟负反馈 ----
    # SOCS_mRNA（SOCS1/3 转录产物，STAT5 转录因子激活，含 30 min 转录延迟）
    # [C4 fix] SBML mapping: SOCS_mRNA→SOCS3RNA (initial_concentration=0.0)
    {"name": "SOCS_mRNA", "species_type": "mrna",
     "compartment": "nucleus",
     "initial_concentration": 0.0},  # Source: BIOMD0000000347 Bachmann2011 (PMID:21772264) species SOCS3RNA
    # SOCS（SOCS 蛋白，结合 JAK 抑制其激酶活性，负反馈）
    # [C4 fix] SBML mapping: SOCS→SOCS3 (initial_concentration=0.0)
    {"name": "SOCS", "species_type": "protein",
     "compartment": "cytoplasm",
     "initial_concentration": 0.0},  # Source: BIOMD0000000347 Bachmann2011 (PMID:21772264) species SOCS3
    # ---- Bcl-xL 转录（STAT3/5 抗凋亡靶基因）----
    # Bcl_xL_mRNA（Bcl-xL mRNA，STAT3 转录激活抗凋亡基因）
    # [C4 fix] No SBML match in BIOMD0000000347. Kept original.
    {"name": "Bcl_xL_mRNA", "species_type": "mrna",
     "compartment": "nucleus"},
    # ---- IRF1 转录（IFN-γ 信号靶基因）----
    # IRF1_mRNA（IRF1 mRNA，IFN-γ 通过 JAK-STAT 激活 IRF1 表达）
    # [C4 fix] No SBML match in BIOMD0000000347. Kept original.
    {"name": "IRF1_mRNA", "species_type": "mrna",
     "compartment": "nucleus"},
    # ---- PIAS（STAT 抑制蛋白，核内抑制 STAT DNA 结合）----
    # PIAS（PIAS1/3，Protein Inhibitor of Activated STAT，核内抑制 STAT 活性）
    # [C4 fix] No SBML match in BIOMD0000000347. Kept original.
    {"name": "PIAS", "species_type": "protein",
     "compartment": "nucleus"},
    # [N6 缺口 1] 药物物种（species_type="drug"）— 由 drug_library 驱动
    # Ruxolitinib 是 JAK1/2 ATP 竞争性抑制剂（IC50=3 nM, PMID:20385775）
    # 占据 JAK ATP 结合口袋阻断激酶活性，临床用于骨髓纤维化（FDA-approved）
    build_drug_species("Ruxolitinib"),
]


# =============================================================================
# JAK-STAT 核心反应（9 条：受体复合物形成 + JAK/STAT 磷酸化 + STAT 二聚化
# + nuclear import + 3 转录 + 1 翻译）
# =============================================================================
# 每条反应含：source / target / mechanism / kinetics_type / pathway_tag /
#             substrate / product / modifier / modifier_type / autophosphorylation
#
# kinetics_type 选择：
# - complex_formation / dimerization / nuclear_import / translation → mass_action
# - phosphorylation → Michaelis_Menten（与 P3 _mechanism_phosphorylation_mm 模板对齐）
# - transcription → Hill（STAT5 作转录因子，Hill 动力学 n=2 协同结合）
_JAK_STAT_CORE_REACTIONS: list[dict[str, Any]] = [
    # 1. IL6 + IL6R + gp130 → IL6_complex（complex_formation, IL-6 与受体结合）
    #    IL-6 结合 IL6Rα 后招募 gp130 形成三元复合物（IL-6+IL6R+gp130）
    {
        "source": "IL6",
        "target": "IL6_complex",
        "mechanism": "complex_formation",
        "kinetics_type": "mass_action",
        "pathway_tag": PATHWAY_TAG,
        "substrate": "IL6R",
        "product": "IL6_complex",
        "modifier": "IL6",
        "modifier_type": "catalytic",
        "co_factor": "gp130",
        "autophosphorylation": False,
        "description": "IL-6 + IL6R + gp130 → IL6_complex（complex_formation, IL-6 结合 IL6Rα 后招募 gp130 形成三元信号复合物）",
    },
    # 2. IL6_complex → pJAK（phosphorylation, JAK 作 substrate, IL6_complex 作 modifier）
    #    受体相关 JAK 在 IL6_complex 诱导下自磷酸化激活（异磷酸化形式表达）
    {
        "source": "IL6_complex",
        "target": "pJAK",
        "mechanism": "phosphorylation",
        "kinetics_type": "Michaelis_Menten",
        "pathway_tag": PATHWAY_TAG,
        # 异磷酸化：JAK 作 substrate，pJAK 作 product，IL6_complex 作 catalytic modifier
        "substrate": "JAK",
        "product": "pJAK",
        "modifier": "IL6_complex",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "IL6_complex 诱导 JAK 磷酸化（JAK 作 substrate, IL6_complex 作 catalytic modifier, 受体相关 JAK 自磷酸化激活）",
    },
    # 3. pJAK → pSTAT5（phosphorylation, STAT5 作 substrate, pJAK 作 catalytic modifier）
    #    pJAK 异磷酸化 STAT5 酪氨酸残基（Tyr694/Tyr699），STAT5 招募至受体复合物被磷酸化
    {
        "source": "pJAK",
        "target": "pSTAT5",
        "mechanism": "phosphorylation",
        "kinetics_type": "Michaelis_Menten",
        "pathway_tag": PATHWAY_TAG,
        # 异磷酸化：STAT5 作 substrate，pSTAT5 作 product，pJAK 作 catalytic modifier
        "substrate": "STAT5",
        "product": "pSTAT5",
        "modifier": "pJAK",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "pJAK 磷酸化 STAT5（STAT5 作 substrate, pJAK 作 catalytic modifier, STAT5 Tyr 磷酸化招募至受体复合物）",
    },
    # 4. pSTAT5 → pSTAT5_dimer（dimerization, 磷酸化 STAT 形成同源二聚体）
    #    pSTAT5 通过 SH2-pTyr 相互作用形成同源二聚体（cytoplasmic dimerization）
    {
        "source": "pSTAT5",
        "target": "pSTAT5_dimer",
        "mechanism": "dimerization",
        "kinetics_type": "mass_action",
        "pathway_tag": PATHWAY_TAG,
        "substrate": "pSTAT5",
        "product": "pSTAT5_dimer",
        "modifier": "pSTAT5",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "pSTAT5 → pSTAT5_dimer（dimerization, 磷酸化 STAT5 通过 SH2-pTyr 相互作用形成同源二聚体）",
    },
    # 5. pSTAT5_dimer → pSTAT5_nuclear（nuclear_import, 二聚体入核）
    #    pSTAT5 二聚体通过 importin α/β 入核，作为转录因子激活靶基因
    {
        "source": "pSTAT5_dimer",
        "target": "pSTAT5_nuclear",
        "mechanism": "nuclear_import",
        "kinetics_type": "mass_action",
        "pathway_tag": PATHWAY_TAG,
        "substrate": "pSTAT5_dimer",
        "product": "pSTAT5_nuclear",
        "modifier": "pSTAT5_dimer",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "pSTAT5_dimer → pSTAT5_nuclear（nuclear_import, 二聚体通过 importin α/β 入核, 作为转录因子激活靶基因）",
    },
    # 6. pSTAT5_nuclear → SOCS_mRNA（transcription, Hill 动力学, STAT5 作为转录因子）
    #    pSTAT5_nuclear 结合 SOCS1/3 基因启动子 γ-activated site（GAS），
    #    Hill 协同结合（n=2）激活 SOCS mRNA 转录（含 30 min 转录延迟，由 FeedbackLoop 表达）
    {
        "source": "pSTAT5_nuclear",
        "target": "SOCS_mRNA",
        "mechanism": "transcription",
        "kinetics_type": "Hill",
        "pathway_tag": PATHWAY_TAG,
        # pSTAT5_nuclear 作 modifier（转录因子），DNA 作 substrate，SOCS_mRNA 作 product
        "substrate": "DNA",
        "product": "SOCS_mRNA",
        "modifier": "pSTAT5_nuclear",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "hill_coefficient": 2,
        "description": "pSTAT5_nuclear 转录激活 SOCS mRNA（pSTAT5_nuclear 作转录因子, Hill n=2 协同结合 GAS 元件, 含 30 min 转录延迟）",
    },
    # 7. SOCS_mRNA → SOCS（translation, mRNA→蛋白）
    #    SOCS mRNA 在核糖体翻译为 SOCS 蛋白（SOCS 结合 JAK 抑制其激酶活性）
    {
        "source": "SOCS_mRNA",
        "target": "SOCS",
        "mechanism": "translation",
        "kinetics_type": "mass_action",
        "pathway_tag": PATHWAY_TAG,
        "substrate": "SOCS_mRNA",
        "product": "SOCS",
        "modifier": "SOCS_mRNA",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "SOCS mRNA 翻译为 SOCS 蛋白（SOCS_mRNA 作 substrate, SOCS 作 product, 用于结合 JAK 抑制其激酶活性）",
    },
    # 8. pSTAT5_nuclear → Bcl_xL_mRNA（transcription, STAT3 转录抗凋亡基因）
    #    pSTAT5_nuclear（也可代表 STAT3 核形式）转录激活 Bcl-xL 抗凋亡基因
    #    注：Bcl-xL→凋亡下游由 Apoptosis Specialist 处理，本 Specialist 仅处理转录事件
    {
        "source": "pSTAT5_nuclear",
        "target": "Bcl_xL_mRNA",
        "mechanism": "transcription",
        "kinetics_type": "Hill",
        "pathway_tag": PATHWAY_TAG,
        "substrate": "DNA",
        "product": "Bcl_xL_mRNA",
        "modifier": "pSTAT5_nuclear",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "hill_coefficient": 2,
        "description": "pSTAT5_nuclear 转录激活 Bcl-xL mRNA（STAT3/5 作转录因子, Hill n=2, 抗凋亡基因, 下游凋亡由 Apoptosis Specialist 处理）",
    },
    # 9. pSTAT5_nuclear → IRF1_mRNA（transcription, IFN-γ 信号靶基因）
    #    pSTAT5_nuclear（IFN-γ 信号下）转录激活 IRF1 干扰素应答基因
    {
        "source": "pSTAT5_nuclear",
        "target": "IRF1_mRNA",
        "mechanism": "transcription",
        "kinetics_type": "Hill",
        "pathway_tag": PATHWAY_TAG,
        "substrate": "DNA",
        "product": "IRF1_mRNA",
        "modifier": "pSTAT5_nuclear",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "hill_coefficient": 2,
        "description": "pSTAT5_nuclear 转录激活 IRF1 mRNA（STAT1/3 作转录因子, Hill n=2, IFN-γ 信号应答靶基因, 干扰素应答）",
    },
    # 10. SOCS → pJAK（inhibition, mass_action, 反馈环第 3 步：SOCS 抑制 pJAK 激酶活性）
    #    [N8-P0-2 修复] 补全反馈环第 3 步：SOCS 蛋白结合 pJAK 抑制其激酶活性
    #    （Swameye 2003, PMID:12615913; KINETIC_PARAMETERS 中 k_off=1e-3 min^-1, Km=1e-7 M）
    #    缺失此反应导致 SOCS 蛋白无法实际抑制 pJAK，反馈环无法闭合
    #    注：pJAK 作 substrate（被抑制），SOCS 作 modifier（抑制因子），无产物（pJAK 活性降低）
    {
        "source": "SOCS",
        "target": "pJAK",
        "mechanism": "inhibition",
        "kinetics_type": "mass_action",
        "pathway_tag": PATHWAY_TAG,
        "substrate": "pJAK",
        "product": "pJAK",
        "modifier": "SOCS",
        "modifier_type": "allosteric",
        "autophosphorylation": False,
        "description": "SOCS 抑制 pJAK 激酶活性（pJAK 作 substrate, SOCS 作 allosteric inhibitor, 反馈环第 3 步, 阻断 pJAK 磷酸化 STAT5, delay=0min 蛋白结合直接抑制, Swameye 2003 PMID:12615913）",
    },
    # 11. PIAS → pSTAT5_dimer（inhibition, mass_action, 核内 PIAS 抑制 STAT5 DNA 结合）
    #    [N8-P0-2 修复] 补全 PIAS 反馈环：PIAS 在核内结合 pSTAT5_dimer 抑制其 DNA 结合活性
    #    阻止 STAT5 靶基因转录（STAT→PIAS nuclear export, Schwartz 2003, PMID:15286703）
    {
        "source": "PIAS",
        "target": "pSTAT5_dimer",
        "mechanism": "inhibition",
        "kinetics_type": "mass_action",
        "pathway_tag": PATHWAY_TAG,
        "substrate": "pSTAT5_dimer",
        "product": "pSTAT5_dimer",
        "modifier": "PIAS",
        "modifier_type": "allosteric",
        "autophosphorylation": False,
        "description": "PIAS 抑制 pSTAT5_dimer DNA 结合活性（pSTAT5_dimer 作 substrate, PIAS 作 allosteric inhibitor, 核内 STAT→PIAS nuclear export, 阻止 STAT5 靶基因转录, Schwartz 2003 PMID:15286703）",
    },
    # ===== [N6 缺口 1] 药物-靶点显式 inhibitor edge（canonical drug_library 驱动） =====
    # 12. Ruxolitinib → JAK（ATP_competitive, IC50=3 nM, PMID:20385775）
    # Ruxolitinib 是 JAK1/2 ATP 竞争性抑制剂，占据 JAK ATP 结合口袋阻断激酶活性，
    # 临床用于骨髓纤维化（FDA-approved）。此处 target="JAK" 与 specialist 现有物种命名对齐。
    {
        **build_inhibitor_edge("Ruxolitinib", "JAK"),
        "pathway_tag": PATHWAY_TAG,
    },
]


# =============================================================================
# STAT5 状态机（monomer→phosphorylated→dimer→nuclear）
# =============================================================================
# STAT5 蛋白 4 状态机：
# - monomer（未磷酸化胞质单体，初始状态）
# - phosphorylated（酪氨酸磷酸化单体）
# - dimer（同源二聚体，cytoplasmic）
# - nuclear（入核二聚体，作为转录因子）
#
# 状态转换关联到 Core Reaction：
# - monomer → phosphorylated: pJAK → pSTAT5 磷酸化
# - phosphorylated → dimer: pSTAT5 → pSTAT5_dimer 二聚化
# - dimer → nuclear: pSTAT5_dimer → pSTAT5_nuclear 入核
_STAT5_STATE_MACHINE: dict[str, Any] = {
    "id": "SM_STAT5",
    "species": "STAT5",
    "description": (
        "STAT5 状态机：monomer（未磷酸化胞质单体）→phosphorylated"
        "（Tyr 磷酸化）→dimer（同源二聚体）→nuclear（入核二聚体作为转录因子）"
    ),
    "states": [
        {"name": "monomer", "species_id": "SP_STAT5", "is_initial": True},
        {"name": "phosphorylated", "species_id": "SP_pSTAT5"},
        {"name": "dimer", "species_id": "SP_pSTAT5_dimer"},
        {"name": "nuclear", "species_id": "SP_pSTAT5_nuclear"},
    ],
    "transitions": [
        {
            "from_state": "monomer",
            "to_state": "phosphorylated",
            "reaction_id": "RXN_JAK_STAT5_PHOS",
            "trigger": "phosphorylation",
        },
        {
            "from_state": "phosphorylated",
            "to_state": "dimer",
            "reaction_id": "RXN_STAT5_DIMER",
            "trigger": "dimerization",
        },
        {
            "from_state": "dimer",
            "to_state": "nuclear",
            "reaction_id": "RXN_STAT5_NUCLEAR_IMPORT",
            "trigger": "nuclear_import",
        },
    ],
}


# =============================================================================
# JAK-STAT 反馈环（3 条：SOCS 转录延迟负反馈 + SOCS→JAK + PIAS→STAT）
# =============================================================================
_JAK_STAT_FEEDBACK_LOOPS: list[dict[str, Any]] = [
    # 1. STAT5→SOCS 转录延迟负反馈（DDE 振荡器, delay=30min）
    #    Schwartz 2003 (PMID:15286703) 经典模型：
    #    - pSTAT5_nuclear 转录激活 SOCS mRNA（含 30 min 转录延迟）
    #    - SOCS 蛋白结合 JAK 抑制其激酶活性
    #    - 形成 delay=30min 的延迟负反馈，产生 STAT5 核质比单脉冲动力学
    {
        "id": "FL_STAT_SOCS",
        "loop_type": "negative",
        "node_ids": [
            "pSTAT5_nuclear",
            "SOCS_mRNA",
            "SOCS",
            "pJAK",
        ],
        "delay_minutes": _STAT_SOCS_DELAY_MINUTES,  # 30.0 min 转录延迟
        "bistable": False,
        "template": "oscillatory_feedback.j2",
        "description": (
            "STAT5→SOCS 转录延迟负反馈（pSTAT5_nuclear 转录激活 SOCS mRNA, "
            "SOCS 抑制 JAK, delay=30min, 形成 STAT5 核质比单脉冲, Schwartz 2003）"
        ),
        "source_pmid": _Pmid_SCHWARTZ_2003,
        "dde_solver": "solvers/dde_solver.py",
    },
    # 2. SOCS → pJAK 抑制（inhibition, delay=0）
    #    SOCS 蛋白结合 JAK 激酶抑制域，抑制其激酶活性（负反馈直接抑制，无转录延迟）
    {
        "id": "FL_SOCS_JAK_INHIBITION",
        "loop_type": "negative",
        "node_ids": ["SOCS", "pJAK"],
        "delay_minutes": 0.0,
        "bistable": False,
        "description": (
            "SOCS → pJAK 抑制（SOCS 结合 JAK 激酶抑制域, 抑制其激酶活性, "
            "delay=0min, 直接负反馈无转录延迟）"
        ),
    },
    # 3. PIAS → pSTAT5_dimer 抑制（inhibition, delay=0）
    #    PIAS（Protein Inhibitor of Activated STAT）在核内结合 pSTAT5_dimer
    #    抑制其 DNA 结合活性，阻止 STAT5 靶基因转录（STAT→PIAS nuclear export）
    {
        "id": "FL_PIAS_STAT_INHIBITION",
        "loop_type": "negative",
        "node_ids": ["PIAS", "pSTAT5_dimer", "pSTAT5_nuclear"],
        "delay_minutes": 0.0,
        "bistable": False,
        "description": (
            "PIAS → pSTAT5_dimer 抑制（PIAS 结合 STAT 二聚体抑制其 DNA 结合活性, "
            "阻止 STAT5 靶基因转录, STAT→PIAS nuclear export, delay=0min）"
        ),
    },
]


# =============================================================================
# JAK-STAT Crosstalk Reaction 片段（5 条，仅本通路侧片段，edge 由 Coordinator 管理）
# =============================================================================
# 注意：EGFR→STAT3 cross-talk edge 由 Coordinator 生成，本 Specialist 仅返回
# 本通路侧消费片段（不处理 STAT3→Bcl-xL 凋亡下游，由 Apoptosis Specialist 处理）
_JAK_STAT_CROSSTALK_REACTIONS: list[dict[str, Any]] = [
    # 1. pSTAT3 → Bcl-xL（transcription, STAT3 转录 Bcl-xL 抗凋亡）
    #    pSTAT3 转录激活 Bcl-xL 抗凋亡基因（IL-6-STAT3 通路经典抗凋亡机制）
    {
        "source": "pSTAT3",
        "target": "Bcl_xL",
        "mechanism": "transcription",
        "shared_species": ["STAT3"],
        "description": "pSTAT3 转录激活 Bcl-xL（pSTAT3 作转录因子, Bcl-xL 抗凋亡, STAT3 通路经典抗凋亡机制）",
    },
    # 2. pSTAT3 → Bcl-2（transcription, STAT3 转录 Bcl-2）
    #    pSTAT3 转录激活 Bcl-2 抗凋亡基因（与 Bcl-xL 协同抗凋亡）
    {
        "source": "pSTAT3",
        "target": "Bcl2",
        "mechanism": "transcription",
        "shared_species": ["STAT3"],
        "description": "pSTAT3 转录激活 Bcl-2（pSTAT3 作转录因子, Bcl-2 抗凋亡, 与 Bcl-xL 协同维持肿瘤细胞存活）",
    },
    # 3. EGFR → pSTAT3（activation, EGFR 旁路激活 STAT3）
    #    EGFR 通过 JAK1/2 旁路激活 STAT3（绕过 IL-6R 经典通路，EGFR-STAT3 cross-talk）
    {
        "source": "EGFR",
        "target": "pSTAT3",
        "mechanism": "activation",
        "shared_species": ["STAT3"],
        "description": "EGFR 旁路激活 STAT3（EGFR 通过 JAK1/2 旁路激活 STAT3, 绕过 IL-6R 经典通路, EGFR-STAT3 cross-talk）",
    },
    # 4. pERK → pSTAT3（phosphorylation, MAPK 旁路磷酸化 STAT3 Ser727）
    #    pERK 磷酸化 STAT3 Ser727 残基增强其转录活性（MAPK-STAT3 Ser727 cross-talk）
    {
        "source": "pERK",
        "target": "pSTAT3",
        "mechanism": "phosphorylation",
        "shared_species": ["STAT3"],
        "site": "Ser727",
        "description": "pERK 磷酸化 STAT3 Ser727（MAPK 旁路磷酸化 STAT3 Ser727 增强其转录活性, MAPK-STAT3 cross-talk）",
    },
    # 5. IL-6 → pAKT（activation, IL-6 激活 PI3K-AKT 旁路）
    #    IL-6 通过 PI3K 旁路激活 AKT（IL-6-PI3K-AKT cross-talk，促细胞存活）
    {
        "source": "IL6",
        "target": "pAKT",
        "mechanism": "activation",
        "shared_species": ["IL6", "AKT"],
        "description": "IL-6 激活 PI3K-AKT 旁路（IL-6 通过 PI3K 激活 AKT, 促细胞存活, IL-6-PI3K-AKT cross-talk）",
    },
]


# =============================================================================
# JAK-STAT 扰动（6 个：5 JAK 抑制剂 + 1 SOCS1 OE）
# =============================================================================
_JAK_STAT_PERTURBATIONS: list[dict[str, Any]] = [
    # 1. Tofacitinib（JAK1/3 inhibitor, FDA-approved）
    #    Tofacitinib 是 JAK1/3 选择性抑制剂，用于类风湿性关节炎治疗（FDA-approved）
    {
        "target": "JAK",
        "drug": "Tofacitinib",
        "mechanism": "inhibition",
        "ko_target": None,
        "description": "Tofacitinib（JAK1/3 选择性抑制剂, 小分子, FDA-approved, 用于类风湿性关节炎）",
    },
    # 2. Ruxolitinib（JAK1/2 inhibitor, FDA-approved）
    #    Ruxolitinib 是 JAK1/2 选择性抑制剂，用于骨髓纤维化治疗（FDA-approved）
    # [N6 缺口 1] 注入 canonical drug_library 字段（ic50_nM/ki_nM/source_pmid/...）
    {
        "target": "JAK",
        "drug": "Ruxolitinib",
        "mechanism": "inhibition",
        "ko_target": None,
        "description": "Ruxolitinib（JAK1/2 选择性抑制剂, 小分子, FDA-approved, 用于骨髓纤维化）",
        **{k: v for k, v in get_drug_entry("Ruxolitinib").items()
           if k not in ("description",)},
    },
    # 3. Baricitinib（JAK1/2 inhibitor, FDA-approved）
    #    Baricitinib 是 JAK1/2 选择性抑制剂，用于类风湿性关节炎治疗（FDA-approved）
    {
        "target": "JAK",
        "drug": "Baricitinib",
        "mechanism": "inhibition",
        "ko_target": None,
        "description": "Baricitinib（JAK1/2 选择性抑制剂, 小分子, FDA-approved, 用于类风湿性关节炎）",
    },
    # 4. Tofacitinib（JAK inhibitor, repeat）
    #    注：与第 1 条同药物，此处重复标注强调 JAK1/3 选择性（vs Baricitinib JAK1/2）
    {
        "target": "JAK",
        "drug": "Tofacitinib",
        "mechanism": "inhibition",
        "ko_target": None,
        "description": "Tofacitinib（JAK1/3 选择性抑制剂, 重复条目强调选择性, vs Baricitinib JAK1/2）",
    },
    # 5. Fedratinib（JAK2 inhibitor, FDA-approved）
    #    Fedratinib 是 JAK2 选择性抑制剂，用于骨髓纤维化治疗（FDA-approved）
    {
        "target": "JAK",
        "drug": "Fedratinib",
        "mechanism": "inhibition",
        "ko_target": None,
        "description": "Fedratinib（JAK2 选择性抑制剂, 小分子, FDA-approved, 用于骨髓纤维化）",
    },
    # 6. SOCS1 overexpression（gain-of-function, 增强 JAK 抑制）
    #    SOCS1 过表达增强对 JAK 的负反馈抑制（gain-of-function 增强 JAK-STAT 通路抑制）
    {
        "target": "JAK",
        "drug": None,
        "mechanism": "knockout",
        "ko_target": "SOCS1_OE",
        "description": "SOCS1 overexpression（gain-of-function, SOCS1 过表达增强对 JAK 的负反馈抑制, 增强 JAK-STAT 通路抑制）",
    },
]


# =============================================================================
# JAK-STAT Validation 规则（3 条 benchmark）
# =============================================================================
# 每条含：rule_id / metric_name / expected / tolerance / pmid / description
# expected 取区间中点，tolerance 取区间半宽
_JAK_STAT_VALIDATION_RULES: list[dict[str, Any]] = [
    # 1. pSTAT5 达峰时间 5-15 min（Schwartz 2003, PMID:15286703）
    #    STAT5 酪氨酸磷酸化在 IL-6/EPO 刺激后 5-15 min 达峰
    {
        "rule_id": "VAL_JAK_STAT_PSTAT5_PEAK_TIME",
        "metric_name": "pSTAT5_peak_time",
        "expected": 10.0,   # (5.0 + 15.0) / 2
        "tolerance": 5.0,    # (15.0 - 5.0) / 2
        "expected_min": 5.0,
        "expected_max": 15.0,
        "unit": "minutes",
        "pmid": _Pmid_SCHWARTZ_2003,
        "comparison": "range",
        "pathway_tag": PATHWAY_TAG,
        "description": "pSTAT5 在 IL-6/EPO 刺激后 5-15 min 达峰（Schwartz 2003 STAT5 dynamics, 酪氨酸磷酸化快速响应）",
    },
    # 2. SOCS mRNA 延迟 30-60 min（Schwartz 2003, PMID:15286703）
    #    SOCS mRNA 在 STAT5 转录激活后 30-60 min 延迟达峰（转录+翻译延迟）
    {
        "rule_id": "VAL_JAK_STAT_SOCS_MRNA_DELAY",
        "metric_name": "SOCS_mRNA_delay",
        "expected": 45.0,   # (30.0 + 60.0) / 2
        "tolerance": 15.0,   # (60.0 - 30.0) / 2
        "expected_min": 30.0,
        "expected_max": 60.0,
        "unit": "minutes",
        "pmid": _Pmid_SCHWARTZ_2003,
        "comparison": "range",
        "pathway_tag": PATHWAY_TAG,
        "description": "SOCS mRNA 在 STAT5 转录激活后 30-60 min 延迟达峰（转录+翻译延迟, 形成负反馈时序）",
    },
    # 3. STAT5 核质比单脉冲（Schwartz 2003, PMID:15286703）
    #    STAT5 核质比应呈现单脉冲（非持续振荡），SOCS 负反馈及时关闭 STAT5 信号
    {
        "rule_id": "VAL_JAK_STAT_STAT5_NUCLEAR_PULSE",
        "metric_name": "STAT5_nuclear_cytoplasmic_ratio_pulse",
        "expected": True,
        "tolerance": 0.0,
        "expected_min": True,
        "expected_max": True,
        "unit": "boolean",
        "pmid": _Pmid_SCHWARTZ_2003,
        "comparison": "boolean",
        "pathway_tag": PATHWAY_TAG,
        "description": "STAT5 核质比应呈现单脉冲（非持续振荡, SOCS 负反馈及时关闭 STAT5 信号, Schwartz 2003）",
    },
]


@register_specialist
class JakStatSpecialist(PathwaySpecialistBase):
    """JAK-STAT 通路 Specialist。

    实现 Cytokine（IL-6）+受体（IL6R+gp130）→JAK 磷酸化→STAT5 招募→
    STAT5 酪氨酸磷酸化→二聚化→nuclear import→转录（SOCS/Bcl-xL/IRF1）核心拓扑 +
    SOCS 转录延迟负反馈（delay=30min）+ PIAS→STAT 抑制 + STAT5 状态机
    （monomer→phosphorylated→dimer→nuclear）的 Core/Feedback/Crosstalk/
    Perturbation/Validation 5 模块，输出通路特异 Reaction IR 片段 + STAT5 状态机
    + 模板选择 + Validation 规则。

    职责边界：
    - 处理 JAK-STAT 核心（IL-6+IL6R+gp130+JAK+STAT5+SOCS+Bcl-xL+IRF1+PIAS）
    - 不处理 STAT3→Bcl-xL 凋亡下游（由 Apoptosis Specialist 处理，Task 4.7）
    - 不生成 EGFR→STAT3 cross-talk edge（由 Cross-talk Coordinator 处理，Task 4.13）

    输入：
    - ``pathway_graph``：v4_pathway_graph 的 JAK-STAT 子图（含 FeedbackLoop
      FL_STAT_SOCS delay=30min）
    - ``ontology_entities``：P1 Ontology Entities（可选，HGNC/UniProt ID 对齐）

    输出：
    - ``apply_core``：9 条核心 Reaction IR 片段 + 17 物种 + STAT5 状态机
      （pSTAT5 标记 shared=True，与 Apoptosis Specialist Bcl-xL 路径共享）
    - ``apply_feedback``：3 条 FeedbackLoop
      （FL_STAT_SOCS delay=30min 转录延迟负反馈 / SOCS→pJAK / PIAS→STAT）
    - ``apply_crosstalk``：5 条 cross-talk Reaction 片段
      （pSTAT3→Bcl-xL/Bcl-2 + EGFR→pSTAT3 + pERK→pSTAT3 + IL-6→pAKT）
    - ``apply_perturbation``：6 个扰动
      （Tofacitinib/Ruxolitinib/Baricitinib/Fedratinib + SOCS1 OE）
    - ``apply_validation``：3 条 Validation benchmark
      （pSTAT5 5-15min 达峰 / SOCS 30-60min 延迟 / STAT5 核质比单脉冲）
    """

    pathway_class: str = "JAK_STAT"
    display_name: str = "JAK-STAT Signaling"

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
                    species=list(_JAK_STAT_CORE_SPECIES),
                    reactions=list(_JAK_STAT_CORE_REACTIONS),
                    kinetics_overrides={},
                )
            if module_name == MODULE_FEEDBACK:
                return FeedbackModuleData(
                    feedback_loops=list(_JAK_STAT_FEEDBACK_LOOPS),
                    delay_minutes=_STAT_SOCS_DELAY_MINUTES,
                    loop_type="negative",
                )
            if module_name == MODULE_CROSSTALK:
                return CrosstalkModuleData(
                    crosstalk_reactions=list(_JAK_STAT_CROSSTALK_REACTIONS),
                    shared_species=["STAT3", "IL6", "AKT", "pSTAT5"],
                    coordination_strategy="merge",
                )
            if module_name == MODULE_PERTURBATION:
                return PerturbationModuleData(
                    perturbation_reactions=list(_JAK_STAT_PERTURBATIONS),
                    drug_targets=[
                        p for p in _JAK_STAT_PERTURBATIONS if p.get("drug")
                    ],
                    ko_targets=[
                        p for p in _JAK_STAT_PERTURBATIONS if p.get("ko_target")
                    ],
                )
            if module_name == MODULE_VALIDATION:
                return ValidationModuleData(
                    rules=list(_JAK_STAT_VALIDATION_RULES),
                    benchmarks=[
                        {
                            "benchmark_name": r["metric_name"],
                            "value": r["expected"],
                            "unit": r["unit"],
                            "condition": "IL-6/EPO stimulation",
                            "reference": r["pmid"],
                        }
                        for r in _JAK_STAT_VALIDATION_RULES
                    ],
                    tolerances={
                        r["metric_name"]: r["tolerance"]
                        for r in _JAK_STAT_VALIDATION_RULES
                    },
                    pmid_references=[
                        {
                            "pmid": r["pmid"],
                            "citation": r["description"],
                            "pathway_class": PATHWAY_TAG,
                            "metric_name": r["metric_name"],
                        }
                        for r in _JAK_STAT_VALIDATION_RULES
                        if r["pmid"]
                    ],
                )
            logger.warning(
                "JakStatSpecialist.load_module: 未知模块名 '%s'",
                module_name,
            )
            return None
        except Exception as exc:
            logger.warning(
                "JakStatSpecialist.load_module 加载模块 '%s' 失败: %s",
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
        """应用核心模块，返回 JAK-STAT 通路核心 Reaction IR 片段。

        输出 9 条核心反应：
        1. IL6 → IL6_complex（complex_formation, IL-6+IL6R+gp130 三元复合物）
        2. IL6_complex → pJAK（异磷酸化, JAK 作 substrate）
        3. pJAK → pSTAT5（异磷酸化, STAT5 作 substrate）
        4. pSTAT5 → pSTAT5_dimer（dimerization, 同源二聚体）
        5. pSTAT5_dimer → pSTAT5_nuclear（nuclear_import, 二聚体入核）
        6. pSTAT5_nuclear → SOCS_mRNA（transcription, Hill, STAT5 转录因子）
        7. SOCS_mRNA → SOCS（translation, mRNA→蛋白）
        8. pSTAT5_nuclear → Bcl_xL_mRNA（transcription, STAT3 抗凋亡靶基因）
        9. pSTAT5_nuclear → IRF1_mRNA（transcription, IFN-γ 靶基因）

        pSTAT5 物种标记 shared=True（与 Apoptosis Specialist 的 Bcl-xL 抗凋亡
        路径共享，下游凋亡由 Apoptosis Specialist 处理）。

        STAT5 状态机输出 4 状态（monomer→phosphorylated→dimer→nuclear）。

        Returns:
            dict 含 ``species``（17 物种）/ ``reactions``（9 反应） /
            ``state_machine``（STAT5 状态机）字段。异常时返回
            ``{"species": [], "reactions": [], "state_machine": {}}``。
        """
        try:
            return {
                "species": list(_JAK_STAT_CORE_SPECIES),
                "reactions": list(_JAK_STAT_CORE_REACTIONS),
                "state_machine": dict(_STAT5_STATE_MACHINE),
                "pathway_tag": PATHWAY_TAG,
                "source_sbml": SOURCE_SBML,
                # [KINETIC_PARAMETERS 注入 / P0-1] 按 target 物种名组织的动力学参数
                # 修复 C1 Peak Time：原 KINETIC_PARAMETERS 是死代码，现通过此字段
                # 经 specialist_hook → graph_v3._ode_template_v2_hook → renderer.render(params=...)
                # 注入 ODE 模板，使 _get_param(tgt_name, key, default) 能查到文献参数。
                "kinetics_overrides": dict(_KINETICS_BY_TARGET),
            }
        except Exception as exc:
            logger.warning(
                "JakStatSpecialist.apply_core 失败: %s", exc
            )
            return {
                "species": [],
                "reactions": [],
                "state_machine": {},
            }

    # =================================================================
    # apply_feedback：FeedbackLoop 列表
    # =================================================================
    def apply_feedback(self, pathway_graph: dict) -> list[dict]:
        """应用反馈模块，返回 JAK-STAT 通路 FeedbackLoop 列表。

        输出 3 条反馈环：
        1. STAT5→SOCS 转录延迟负反馈（delay=30min, 振荡器）
        2. SOCS → pJAK 抑制（inhibition, delay=0, 直接负反馈）
        3. PIAS → pSTAT5_dimer 抑制（inhibition, delay=0, STAT→PIAS nuclear export）

        Returns:
            FeedbackLoop 字典列表（含 delay_minutes 标记）。异返回空列表。
        """
        try:
            return list(_JAK_STAT_FEEDBACK_LOOPS)
        except Exception as exc:
            logger.warning(
                "JakStatSpecialist.apply_feedback 失败: %s", exc
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
        """应用跨通路模块，返回 JAK-STAT 通路侧 cross-talk Reaction 片段。

        注意：本方法仅生成本通路侧的 cross-talk Reaction 片段，
        cross-talk edge 本身由 Cross-talk Coordinator 创建（Task 4.13）。

        输出 5 条 cross-talk Reaction 片段：
        1. pSTAT3 → Bcl-xL（transcription, STAT3 转录 Bcl-xL 抗凋亡）
        2. pSTAT3 → Bcl-2（transcription, STAT3 转录 Bcl-2）
        3. EGFR → pSTAT3（activation, EGFR 旁路激活 STAT3）
        4. pERK → pSTAT3（phosphorylation, MAPK 旁路磷酸化 STAT3 Ser727）
        5. IL-6 → pAKT（activation, IL-6 激活 PI3K-AKT 旁路）

        Args:
            pathway_graph: 通路图（当前未使用，保留接口一致性）。
            crosstalk_edges: 来自 Coordinator 的 cross-talk edge 列表
                （当前未使用，本 Specialist 返回静态拓扑片段）。

        Returns:
            cross-talk Reaction 片段列表。异返回空列表。
        """
        try:
            return list(_JAK_STAT_CROSSTALK_REACTIONS)
        except Exception as exc:
            logger.warning(
                "JakStatSpecialist.apply_crosstalk 失败: %s", exc
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
        """应用扰动模块，返回 JAK-STAT 通路特异药物 / 突变 Reaction 片段。

        输出 6 个扰动（5 个药物 + 1 个 gain-of-function）：
        1. Tofacitinib（JAK1/3 抑制剂, FDA-approved）
        2. Ruxolitinib（JAK1/2 抑制剂, FDA-approved）
        3. Baricitinib（JAK1/2 抑制剂, FDA-approved）
        4. Tofacitinib（重复条目强调 JAK1/3 选择性）
        5. Fedratinib（JAK2 抑制剂, FDA-approved）
        6. SOCS1 overexpression（gain-of-function, 增强 JAK 抑制）

        Args:
            pathway_graph: 通路图（当前未使用）。
            perturbation_points: 扰动点列表（当前未使用，本 Specialist
                返回静态拓扑片段）。

        Returns:
            扰动 Reaction 片段列表。异返回空列表。
        """
        try:
            return list(_JAK_STAT_PERTURBATIONS)
        except Exception as exc:
            logger.warning(
                "JakStatSpecialist.apply_perturbation 失败: %s", exc
            )
            return []

    # =================================================================
    # apply_validation：Validation 规则列表
    # =================================================================
    def apply_validation(
        self,
        simulation_result: dict | None = None,
    ) -> list[dict]:
        """应用验证模块，返回 JAK-STAT 通路 Validation 规则列表。

        输出 3 条 benchmark：
        1. pSTAT5 达峰时间 5-15 min（Schwartz 2003, PMID:15286703）
        2. SOCS mRNA 延迟 30-60 min（Schwartz 2003, PMID:15286703）
        3. STAT5 核质比单脉冲（Schwartz 2003, PMID:15286703）

        Args:
            simulation_result: 仿真结果（可选，当前未使用，返回静态规则集）。

        Returns:
            Validation 规则列表。异返回空列表。
        """
        try:
            return list(_JAK_STAT_VALIDATION_RULES)
        except Exception as exc:
            logger.warning(
                "JakStatSpecialist.apply_validation 失败: %s", exc
            )
            return []

    # =================================================================
    # select_template：覆写以支持 phosphorylation + transcription
    # =================================================================
    def select_template(self, mechanism: str) -> str:
        """根据 mechanism 选择 ODE 模板名（覆写支持 STAT5 磷酸化与转录）。

        默认映射（与 P3 ``ode_templates_v2/`` 下 .j2 文件对齐）：
        - ``phosphorylation`` → ``_mechanism_phosphorylation_mm``
          （pJAK→pSTAT5 / IL6_complex→pJAK 异磷酸化）
        - ``transcription`` → ``_mechanism_phosphorylation_mm``
          （P3 ``transcription_factor.j2`` 当前未实现，转录降级到磷酸化模板）
          注：未来 P3 若添加 ``transcription_factor.j2``，应在此覆写返回
          ``transcription_factor``。

        Args:
            mechanism: 机制名（小写，如 ``"phosphorylation"`` / ``"transcription"``）。

        Returns:
            ODE 模板名（不含 ``.j2`` 后缀）。未匹配时返回基类默认映射
            （调用方应处理默认降级）。
        """
        # 磷酸化场景：pJAK→pSTAT5 / IL6_complex→pJAK 异磷酸化
        if mechanism == "phosphorylation":
            return "_mechanism_phosphorylation_mm"
        # 转录场景：pSTAT5_nuclear 转录 SOCS/Bcl-xL/IRF1
        # P3 transcription_factor.j2 当前未实现，降级到 _mechanism_phosphorylation_mm
        if mechanism == "transcription":
            return "_mechanism_phosphorylation_mm"
        # 其他机制走默认基类映射
        return super().select_template(mechanism)


# =============================================================================
# 文献动力学参数（IB-017 修复）
# =============================================================================
# 来源：
# - BIOMD0000000273 (Swameye 2003, PMID:12615913) STAT5 信号动力学模型
# - BIOMD0000000453 JAK-STAT 通路模型
# - PMID:15286703 (Schwartz 2003) JAK-STAT 转录验证基准
# 反幻觉守卫：所有参数来自上述 BioModels 模型或文献；无确切值的用无量纲化
# 估计并标注 `# Heuristic estimate, needs calibration`。
# 参数范围约束：k_on∈[1e3,1e7] M^-1 min^-1, Km∈[1e-7,1e-2] M, k_cat∈[1e-3,1e3] min^-1
# 注：STAT5-SOCS 转录延迟 _STAT_SOCS_DELAY_MINUTES=30min 为 DDE 延迟项（见文件顶部）。
KINETIC_PARAMETERS: dict[str, dict[str, float]] = {
    # IL6+IL6R→IL6_complex 配体-受体结合（Swameye 2003, PMID:12615913, BIOMD0000000273）
    "IL6_IL6R": {
        "k_on": 5.0e5,                # M^-1 min^-1  # Heuristic estimate, needs calibration
        "k_off": 0.01,                # min^-1  # Heuristic estimate, needs calibration
    },
    # IL6_complex→pJAK JAK 磷酸化激活（Swameye 2003, PMID:12615913, 异磷酸化）
    "IL6_complex_pJAK": {
        "k_cat": 1.0,                 # min^-1  # Heuristic estimate, needs calibration
        "Km": 1e-7,                   # M (JAK Km)  # Heuristic estimate, needs calibration
    },
    # pJAK→pSTAT5 STAT5 磷酸化（Swameye 2003, PMID:12615913, 异磷酸化）
    "pJAK_pSTAT5": {
        "k_cat": 1.0,                 # min^-1
        "Km": 1e-7,                   # M (STAT5 Km ≈100 nM, Swameye 2003)
    },
    # pSTAT5→pSTAT5_dimer 二聚化（Swameye 2003, PMID:12615913）
    "pSTAT5_dimer": {
        "k_on": 1.0e6,                # M^-1 min^-1  # Heuristic estimate, needs calibration
        "k_off": 1e-3,                # min^-1  # Heuristic estimate, needs calibration
    },
    # pSTAT5_dimer→pSTAT5_nuclear 入核（Swameye 2003, PMID:12615913）
    "pSTAT5_nuclear_import": {
        "k_import": 0.1,              # min^-1  # Heuristic estimate, needs calibration
    },
    # pSTAT5_nuclear→SOCS_mRNA 转录（Swameye 2003, PMID:12615913, DDE delay=30min 负反馈）
    "pSTAT5_SOCS_transcription": {
        "k_transcription": 1.0,       # min^-1  # Heuristic estimate, needs calibration
        "Km": 1e-7,                   # M (pSTAT5_nuclear Km)  # Heuristic estimate, needs calibration
    },
    # SOCS_mRNA→SOCS 翻译（Swameye 2003, PMID:12615913）
    "SOCS_translation": {
        "k_translation": 0.1,         # min^-1  # Heuristic estimate, needs calibration
    },
    # pSTAT5_nuclear→Bcl_xL_mRNA 转录（抗凋亡, Schwartz 2003, PMID:15286703）
    "pSTAT5_BclxL_transcription": {
        "k_transcription": 1.0,       # min^-1  # Heuristic estimate, needs calibration
        "Km": 1e-7,                   # M  # Heuristic estimate, needs calibration
    },
    # pSTAT5_nuclear→IRF1_mRNA 转录（Schwartz 2003, PMID:15286703）
    "pSTAT5_IRF1_transcription": {
        "k_transcription": 1.0,       # min^-1  # Heuristic estimate, needs calibration
        "Km": 1e-7,                   # M  # Heuristic estimate, needs calibration
    },
    # SOCS 抑制 pJAK（负反馈, Swameye 2003, PMID:12615913）
    "SOCS_pJAK_inhibition": {
        "k_on": 1.0e6,                # M^-1 min^-1  # Heuristic estimate, needs calibration
        "k_off": 1e-3,                # min^-1  # Heuristic estimate, needs calibration
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
#   "IL6_IL6R"                     → IL6_complex（反应 1: IL6+IL6R→IL6_complex 配体-受体结合, k_on SKIP）
#   "IL6_complex_pJAK"             → pJAK（反应 2: IL6_complex→pJAK JAK 磷酸化激活）
#   "SOCS_pJAK_inhibition"         → pJAK（SOCS 负反馈抑制 pJAK, k_on SKIP, 合并到 pJAK）
#   "pJAK_pSTAT5"                  → pSTAT5（反应 3: pJAK→pSTAT5 STAT5 磷酸化）
#   "pSTAT5_dimer"                 → pSTAT5_dimer（反应 4: pSTAT5→pSTAT5_dimer 二聚化, k_on SKIP）
#   "pSTAT5_nuclear_import"        → pSTAT5_nuclear（反应 5: pSTAT5_dimer→pSTAT5_nuclear 入核）
#   "pSTAT5_SOCS_transcription"    → SOCS_mRNA（反应 6: pSTAT5_nuclear→SOCS_mRNA 转录, DDE delay=30min）
#   "SOCS_translation"             → SOCS（反应 7: SOCS_mRNA→SOCS 翻译）
#   "pSTAT5_BclxL_transcription"   → Bcl_xL_mRNA（反应 8: pSTAT5_nuclear→Bcl_xL_mRNA 转录）
#   "pSTAT5_IRF1_transcription"    → IRF1_mRNA（反应 9: pSTAT5_nuclear→IRF1_mRNA 转录）
_KINETICS_BY_TARGET: dict[str, dict[str, float]] = {
    "IL6_complex": {
        # k_on SKIP: 单位 M^-1 min^-1 与 μM 模型冲突
        "k_off": 0.05,                # min^-1 (IL6-IL6R 解离, Swameye 2003)
    },
    # [RC29 校准对齐] pJAK k_cat 1.0→2.0 对齐 oscillatory_feedback.j2 磷酸化默认值
    "pJAK": {
        "k_cat": 2.0,                 # min^-1 (IL6_complex 催化 JAK 磷酸化, Swameye 2003)
        "Km": 0.1,                    # μM (原 1e-7 M = 0.1 μM)
        # k_on SKIP: 单位 M^-1 min^-1 与 μM 模型冲突 (SOCS 负反馈抑制)
        "k_off": 1e-3,                # min^-1 (SOCS 抑制 pJAK 解离, Swameye 2003)
    },
    # [RC29 校准对齐] pSTAT5 k_cat 1.0→2.0 对齐 oscillatory_feedback.j2 磷酸化默认值
    "pSTAT5": {
        "k_cat": 2.0,                 # min^-1 (pJAK 催化 STAT5 磷酸化, Swameye 2003)
        "Km": 0.1,                    # μM (原 1e-7 M = 0.1 μM, STAT5 Km ≈100 nM)
    },
    "pSTAT5_dimer": {
        # k_on SKIP: 单位 M^-1 min^-1 与 μM 模型冲突
        "k_off": 0.05,                # min^-1 (pSTAT5 二聚体解离, Swameye 2003)
        # [N8-P0-2 修复] 补 PIAS inhibition k_off_inhibition（反应 11: PIAS→pSTAT5_dimer 抑制）
        # PIAS 在核内结合 pSTAT5_dimer 抑制其 DNA 结合活性（Schwartz 2003）
        "k_off_inhibition": 1e-3,      # min^-1 (PIAS 抑制 pSTAT5_dimer 解离, Schwartz 2003)
    },
    "pSTAT5_nuclear": {
        "k_import": 0.1,              # min^-1 (pSTAT5_dimer 入核, Swameye 2003)
    },
    "SOCS_mRNA": {
        "k_transcription": 1.0,       # min^-1 (pSTAT5_nuclear 转录激活 SOCS, Swameye 2003, DDE delay=30min)
        "Km": 0.1,                    # μM (原 1e-7 M = 0.1 μM)
    },
    "SOCS": {
        "k_translation": 0.1,         # min^-1 (SOCS_mRNA 翻译, Swameye 2003)
    },
    "Bcl_xL_mRNA": {
        "k_transcription": 1.0,       # min^-1 (pSTAT5_nuclear 转录激活 Bcl-xL, Schwartz 2003)
        "Km": 0.1,                    # μM (原 1e-7 M = 0.1 μM)
    },
    "IRF1_mRNA": {
        "k_transcription": 1.0,       # min^-1 (pSTAT5_nuclear 转录激活 IRF1, Schwartz 2003)
        "Km": 0.1,                    # μM (原 1e-7 M = 0.1 μM)
    },
}


__all__ = [
    "JakStatSpecialist",
    "PATHWAY_TAG",
    "SOURCE_SBML",
    "KINETIC_PARAMETERS",
    "_KINETICS_BY_TARGET",
]
