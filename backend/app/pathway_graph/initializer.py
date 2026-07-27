# BioDynamics Agent v4 - Pathway Graph Initializer
# 对应 PART C4：10 Pathway Initialization Strategy。
#
# 为每条通路定义：
#   * core modules（核心拓扑）
#   * feedback loops（通路内反馈）
#   * cross-talk nodes（跨通路交叉点）
#   * perturbation points（药物 / KO / 突变影响点）
#
# 设计原则：
# 1. 每条通路的拓扑来自 pathway_registry（P1）+ 文献共识（不调用 LLM）
# 2. feedback loops / cross-talk / perturbation points 显式声明，不靠检测
# 3. SBML 对齐：每条通路标注 source SBML BioModels ID（用于 P1 解析）
# 4. 10 条通路覆盖 v4 Scientific Architecture Part 3 全部通路
# 5. 不破坏 P1/P2：本文件只产出初始化数据，不修改 P1/P2 代码

from __future__ import annotations

from typing import Any

from app.biomodels_registry import get_biomodels_id

from .schema import (
    CrossTalkEdge,
    FeedbackLoop,
    PathwayModule,
    PathwayNode,
    PathwayState,
    TimeScale,
)


# =============================================================================
# 10 条通路初始化数据
# =============================================================================
# 每条通路包含：
#   - pathway_class: 通路类别（与 pathway_registry 对齐）
#   - display_name: 显示名
#   - source_sbml: SBML BioModels ID（用于 P1 SBML Parser）
#   - source_kegg: KEGG pathway ID
#   - core_nodes: 核心节点列表 [(name, species_type, compartment, time_scale)]
#   - core_edges: 核心边列表 [(source, target, mechanism, kinetics_type)]
#   - feedback_loops: 反馈环列表
#   - cross_talk: 跨通路交叉点列表
#   - perturbation_points: 药物/KO 影响点列表
# =============================================================================

PATHWAY_INITIALIZERS: dict[str, dict[str, Any]] = {
    # =========================================================================
    # 1. EGFR RTK Signaling
    # =========================================================================
    "EGFR_RTK": {
        "display_name": "EGFR Receptor Tyrosine Kinase Signaling",
        "source_sbml": get_biomodels_id("EGFR_RTK"),
        "source_kegg": "hsa04012",
        "core_nodes": [
            ("EGF", "ligand", "extracellular", TimeScale.FAST),
            ("EGFR", "protein", "membrane", TimeScale.FAST),
            ("pEGFR", "protein", "membrane", TimeScale.FAST),
            ("Shc", "protein", "cytoplasm", TimeScale.FAST),
            ("pShc", "protein", "cytoplasm", TimeScale.FAST),
            ("Grb2", "protein", "cytoplasm", TimeScale.FAST),
            ("SOS", "protein", "cytoplasm", TimeScale.FAST),
            ("Ras", "protein", "membrane", TimeScale.FAST),  # shared with MAPK
            ("RasGTP", "protein", "membrane", TimeScale.FAST),
            # [BENCHMARK CLOSURE / Gap 1D] 补全 specialist 已声明但 initializer 缺失的物种，
            # 避免 reaction_ir.species 由 initializer 派生时 EGFR_internalized 不在 SP_IDX
            # 导致 degradation 边被静默丢弃（见 logs/f5_loop/20260720_004811/rca.md）。
            ("RasGDP", "protein", "membrane", TimeScale.FAST),
            ("RasGAP", "protein", "cytoplasm", TimeScale.FAST),
            ("Raf", "protein", "cytoplasm", TimeScale.FAST),
            ("pRaf", "protein", "cytoplasm", TimeScale.FAST),
            ("MEK", "protein", "cytoplasm", TimeScale.FAST),
            ("pMEK", "protein", "cytoplasm", TimeScale.FAST),
            ("ppMEK", "protein", "cytoplasm", TimeScale.FAST),
            ("ERK", "protein", "cytoplasm", TimeScale.FAST),
            ("pERK", "protein", "cytoplasm", TimeScale.FAST),
            ("ppERK", "protein", "cytoplasm", TimeScale.FAST),
            ("ppERK_nuclear", "protein", "nucleus", TimeScale.FAST),
            ("DUSP_mRNA", "mrna", "nucleus", TimeScale.MEDIUM),
            ("DUSP", "protein", "cytoplasm", TimeScale.MEDIUM),
            # EGFR 内吞后受体池（降解产物 sink，由 degradation 边写入）：
            ("EGFR_internalized", "protein", "endosome", TimeScale.MEDIUM),
        ],
        "core_edges": [
            ("EGF", "EGFR", "binding", "mass_action"),
            ("EGFR", "pEGFR", "phosphorylation", "Michaelis_Menten"),
            ("pEGFR", "pShc", "phosphorylation", "Michaelis_Menten"),
            ("pEGFR", "Grb2", "binding", "mass_action"),
            ("Grb2", "SOS", "binding", "mass_action"),
            ("SOS", "RasGTP", "gtp_gdp_exchange", "mass_action"),
            ("RasGTP", "pRaf", "phosphorylation", "Michaelis_Menten"),
        ],
        "feedback_loops": [
            {
                "id": "FL_EGFR_INTERNALIZATION",
                "loop_type": "negative",
                "node_ids": ["PN_pEGFR", "PN_EGFR"],
                "delay_minutes": 15.0,
                "description": "pEGFR 触发受体内吞与降解（负反馈）",
            },
        ],
        "cross_talk": [
            {
                "id": "CT_EGFR_TO_PI3K",
                "source_pathway": "EGFR_RTK",
                "target_pathway": "PI3K_AKT_mTOR",
                "source_node": "PN_pEGFR",
                "target_node": "PN_PI3K",
                "mechanism": "activation",
                "shared_species": [],
                "description": "pEGFR 直接磷酸化 PI3K",
            },
            {
                "id": "CT_EGFR_TO_MAPK_RAS",
                "source_pathway": "EGFR_RTK",
                "target_pathway": "MAPK_ERK",
                "source_node": "PN_RasGTP",
                "target_node": "PN_Raf",
                "mechanism": "activation",
                "shared_species": ["Ras"],
                "description": "RasGTP 共享激活 MAPK 级联",
            },
        ],
        "perturbation_points": [
            {"target": "EGFR", "drug": "Gefitinib", "mechanism": "inhibition"},
            {"target": "EGFR", "drug": "Erlotinib", "mechanism": "inhibition"},
            {"target": "Ras", "drug": "Tipifarnib", "mechanism": "inhibition"},
        ],
    },

    # =========================================================================
    # 2. MAPK / ERK Cascade
    # =========================================================================
    "MAPK_ERK": {
        "display_name": "MAPK / ERK Signaling Cascade",
        "source_sbml": get_biomodels_id("MAPK_ERK"),
        "source_kegg": "hsa04010",
        "core_nodes": [
            ("RasGTP", "protein", "membrane", TimeScale.FAST),  # shared
            ("Raf", "protein", "cytoplasm", TimeScale.FAST),     # shared
            ("pRaf", "protein", "cytoplasm", TimeScale.FAST),
            ("MEK", "protein", "cytoplasm", TimeScale.FAST),    # shared
            ("pMEK", "protein", "cytoplasm", TimeScale.FAST),
            ("ERK", "protein", "cytoplasm", TimeScale.FAST),    # shared
            ("pERK", "protein", "cytoplasm", TimeScale.FAST),
        ],
        "core_edges": [
            ("pRaf", "pMEK", "phosphorylation", "Michaelis_Menten"),
            ("pMEK", "pERK", "phosphorylation", "Michaelis_Menten"),
            ("pERK", "pERK", "phosphorylation", "Michaelis_Menten"),  # 自磷酸化
        ],
        "feedback_loops": [
            {
                "id": "FL_MAPK_SOS_NEG",
                "loop_type": "negative",
                "node_ids": ["PN_pERK", "PN_SOS", "PN_RasGTP"],
                "delay_minutes": 0.0,
                "description": "pERK 磷酸化 SOS 导致其失活（负反馈，无转录延迟）",
            },
        ],
        "cross_talk": [
            {
                "id": "CT_PI3K_TO_MAPK_RAF",
                "source_pathway": "PI3K_AKT_mTOR",
                "target_pathway": "MAPK_ERK",
                "source_node": "PN_AKT",
                "target_node": "PN_Raf",
                "mechanism": "inhibition",
                "shared_species": [],
                "site": "Ser259",
                "description": "AKT 磷酸化 Raf Ser259 抑制 MAPK 级联",
            },
            {
                "id": "CT_MAPK_TO_CELL_CYCLE",
                "source_pathway": "MAPK_ERK",
                "target_pathway": "Cell_Cycle",
                "source_node": "PN_pERK",
                "target_node": "PN_Myc",
                "mechanism": "activation",
                "shared_species": ["ERK"],
                "description": "pERK 激活 Myc 表达驱动细胞周期",
            },
        ],
        "perturbation_points": [
            {"target": "MEK", "drug": "Trametinib", "mechanism": "inhibition"},
            {"target": "BRAF", "drug": "Vemurafenib", "mechanism": "inhibition"},
        ],
    },

    # =========================================================================
    # 3. PI3K / AKT / mTOR
    # =========================================================================
    "PI3K_AKT_mTOR": {
        "display_name": "PI3K / AKT / mTOR Signaling",
        "source_sbml": get_biomodels_id("PI3K_AKT_mTOR"),
        "source_kegg": "hsa04151",
        "core_nodes": [
            ("PI3K", "protein", "cytoplasm", TimeScale.FAST),
            ("PIP3", "chemical", "membrane", TimeScale.FAST),
            ("PDK1", "protein", "cytoplasm", TimeScale.FAST),
            ("AKT", "protein", "cytoplasm", TimeScale.FAST),     # shared
            ("pAKT", "protein", "cytoplasm", TimeScale.FAST),
            ("TSC2", "protein", "cytoplasm", TimeScale.MEDIUM),
            ("pTSC2", "protein", "cytoplasm", TimeScale.MEDIUM),
            ("Rheb", "protein", "membrane", TimeScale.MEDIUM),
            ("RhebGTP", "protein", "membrane", TimeScale.MEDIUM),
            ("mTORC1", "complex", "cytoplasm", TimeScale.MEDIUM),  # shared
            ("pS6K", "protein", "cytoplasm", TimeScale.MEDIUM),
            ("PTEN", "protein", "cytoplasm", TimeScale.MEDIUM),
        ],
        "core_edges": [
            ("PI3K", "PIP3", "activation", "hybrid"),
            ("PIP3", "pAKT", "phosphorylation", "Michaelis_Menten"),
            ("pAKT", "pTSC2", "phosphorylation", "Michaelis_Menten"),
            ("pTSC2", "RhebGTP", "activation", "hybrid"),
            ("RhebGTP", "mTORC1", "activation", "hybrid"),
            ("mTORC1", "pS6K", "phosphorylation", "Michaelis_Menten"),
            ("PTEN", "PIP3", "dephosphorylation", "Michaelis_Menten"),
        ],
        "feedback_loops": [
            {
                "id": "FL_mTORC1_S6K_NEG",
                "loop_type": "negative",
                "node_ids": ["PN_pS6K", "PN_PI3K"],
                "delay_minutes": 30.0,
                "description": "pS6K 反向抑制 IRS1/PI3K（负反馈）",
            },
        ],
        "cross_talk": [
            {
                "id": "CT_PI3K_TO_MAPK_RAF",
                "source_pathway": "PI3K_AKT_mTOR",
                "target_pathway": "MAPK_ERK",
                "source_node": "PN_pAKT",
                "target_node": "PN_Raf",
                "mechanism": "inhibition",
                "site": "Ser259",
                "description": "AKT 磷酸化 Raf Ser259 抑制 MAPK",
            },
            {
                "id": "CT_PI3K_TO_APOPTOSIS",
                "source_pathway": "PI3K_AKT_mTOR",
                "target_pathway": "Apoptosis",
                "source_node": "PN_pAKT",
                "target_node": "PN_Bad",
                "mechanism": "inhibition",
                "shared_species": ["AKT"],
                "description": "pAKT 磷酸化 Bad 抑制凋亡",
            },
        ],
        "perturbation_points": [
            {"target": "PI3K", "drug": "Buparlisib", "mechanism": "inhibition"},
            {"target": "AKT", "drug": "MK-2206", "mechanism": "inhibition"},
            {"target": "mTOR", "drug": "Rapamycin", "mechanism": "inhibition"},
        ],
    },

    # =========================================================================
    # 4. p53 Signaling
    # =========================================================================
    "p53_signaling": {
        "display_name": "p53 Tumor Suppressor Signaling",
        "source_sbml": get_biomodels_id("p53"),
        "source_kegg": "hsa04115",
        "core_nodes": [
            ("p53", "protein", "nucleus", TimeScale.MEDIUM),     # shared
            ("p53_ubi", "protein", "nucleus", TimeScale.MEDIUM),
            ("Mdm2", "protein", "nucleus", TimeScale.MEDIUM),
            ("Mdm2_mRNA", "mrna", "nucleus", TimeScale.MEDIUM),
            ("ATM", "protein", "nucleus", TimeScale.FAST),
            ("pATM", "protein", "nucleus", TimeScale.FAST),
            ("p21", "protein", "nucleus", TimeScale.MEDIUM),     # shared with Cell_Cycle
            ("p21_mRNA", "mrna", "nucleus", TimeScale.MEDIUM),
        ],
        "core_edges": [
            ("pATM", "p53", "phosphorylation", "Michaelis_Menten"),
            ("p53", "Mdm2_mRNA", "transcription", "Hill"),
            ("Mdm2_mRNA", "Mdm2", "translation", "mass_action"),
            ("Mdm2", "p53_ubi", "ubiquitination", "Michaelis_Menten"),
            ("p53_ubi", "p53", "proteasomal_degradation", "mass_action"),
            ("p53", "p21_mRNA", "transcription", "Hill"),
            ("p21_mRNA", "p21", "translation", "mass_action"),
        ],
        "feedback_loops": [
            {
                "id": "FL_p53_MDM2_NEG",
                "loop_type": "negative",
                "node_ids": ["PN_p53", "PN_Mdm2_mRNA", "PN_Mdm2"],
                "delay_minutes": 60.0,  # 转录延迟
                "description": "p53 转录激活 Mdm2，Mdm2 降解 p53（负反馈振荡）",
                "source_pmid": "PMID:12717450",
            },
        ],
        "cross_talk": [
            {
                "id": "CT_P53_TO_APOPTOSIS",
                "source_pathway": "p53_signaling",
                "target_pathway": "Apoptosis",
                "source_node": "PN_p53",
                "target_node": "PN_Bax",
                "mechanism": "activation",
                "shared_species": ["p53"],
                "description": "p53 转录激活 Bax 启动凋亡",
            },
            {
                "id": "CT_P53_TO_CELL_CYCLE",
                "source_pathway": "p53_signaling",
                "target_pathway": "Cell_Cycle",
                "source_node": "PN_p21",
                "target_node": "PN_Cdk2",
                "mechanism": "inhibition",
                "shared_species": ["p53"],
                "description": "p21 抑制 Cdk2 阻滞细胞周期",
            },
        ],
        "perturbation_points": [
            {"target": "Mdm2", "drug": "Nutlin-3", "mechanism": "inhibition"},
            {"target": "p53", "drug": "PRIMA-1", "mechanism": "activation"},
        ],
    },

    # =========================================================================
    # 5. Apoptosis (Intrinsic + Extrinsic)
    # =========================================================================
    "Apoptosis": {
        "display_name": "Apoptosis (Intrinsic + Extrinsic)",
        "source_sbml": get_biomodels_id("APOPTOSIS"),
        "source_kegg": "hsa04210",
        "core_nodes": [
            ("Bax", "protein", "mitochondria", TimeScale.MEDIUM),
            ("Bcl2", "protein", "mitochondria", TimeScale.MEDIUM),
            ("Bad", "protein", "cytoplasm", TimeScale.MEDIUM),
            ("pBad", "protein", "cytoplasm", TimeScale.MEDIUM),
            ("Caspase9", "protein", "cytoplasm", TimeScale.MEDIUM),
            ("Caspase3", "protein", "cytoplasm", TimeScale.MEDIUM),  # shared
            ("Caspase3_active", "protein", "cytoplasm", TimeScale.MEDIUM),
            ("PARP", "protein", "nucleus", TimeScale.MEDIUM),
            ("CleavedPARP", "protein", "nucleus", TimeScale.MEDIUM),
        ],
        "core_edges": [
            ("Bax", "Caspase9", "activation", "hybrid"),
            ("Bcl2", "Bax", "inhibition", "hybrid"),
            ("Caspase9", "Caspase3_active", "cleavage", "mass_action"),
            ("Caspase3_active", "CleavedPARP", "cleavage", "mass_action"),
            ("pBad", "Bad", "dephosphorylation", "Michaelis_Menten"),
        ],
        "feedback_loops": [
            {
                "id": "FL_CASPASE3_BID_POS",
                "loop_type": "positive",
                "node_ids": ["PN_Caspase3_active", "PN_Bax"],
                "delay_minutes": 0.0,
                "description": "Caspase3 切割 Bid → tBid → Bax（正反馈放大）",
            },
        ],
        "cross_talk": [
            {
                "id": "CT_PI3K_TO_APOPTOSIS_BAD",
                "source_pathway": "PI3K_AKT_mTOR",
                "target_pathway": "Apoptosis",
                "source_node": "PN_pAKT",
                "target_node": "PN_Bad",
                "mechanism": "phosphorylation",
                "shared_species": ["AKT"],
                "description": "pAKT 磷酸化 Bad 抑制凋亡",
            },
            {
                "id": "CT_P53_TO_APOPTOSIS_BAX",
                "source_pathway": "p53_signaling",
                "target_pathway": "Apoptosis",
                "source_node": "PN_p53",
                "target_node": "PN_Bax",
                "mechanism": "transcription",
                "shared_species": ["p53"],
                "description": "p53 转录激活 Bax",
            },
        ],
        "perturbation_points": [
            {"target": "Bcl2", "drug": "Venetoclax", "mechanism": "inhibition"},
            {"target": "Bcl2", "drug": "Navitoclax", "mechanism": "inhibition"},
            {"target": "Caspase3", "drug": "Z-VAD-FMK", "mechanism": "inhibition"},
        ],
    },

    # =========================================================================
    # 6. Cell Cycle
    # =========================================================================
    "Cell_Cycle": {
        "display_name": "Cell Cycle Regulation",
        "source_sbml": get_biomodels_id("CELL_CYCLE"),
        "source_kegg": "hsa04110",
        "core_nodes": [
            ("CyclinD", "protein", "nucleus", TimeScale.SLOW),
            ("Cdk4", "protein", "nucleus", TimeScale.SLOW),
            ("CyclinD_Cdk4", "complex", "nucleus", TimeScale.SLOW),
            ("CyclinE", "protein", "nucleus", TimeScale.SLOW),
            ("Cdk2", "protein", "nucleus", TimeScale.SLOW),
            ("CyclinE_Cdk2", "complex", "nucleus", TimeScale.SLOW),
            ("Rb", "protein", "nucleus", TimeScale.SLOW),     # shared
            ("pRb", "protein", "nucleus", TimeScale.SLOW),
            ("E2F", "protein", "nucleus", TimeScale.SLOW),    # shared
            ("Myc", "protein", "nucleus", TimeScale.MEDIUM),  # shared
        ],
        "core_edges": [
            ("Myc", "CyclinD", "transcription", "Hill"),
            ("CyclinD", "CyclinD_Cdk4", "complex_formation", "mass_action"),
            ("CyclinD_Cdk4", "pRb", "phosphorylation", "Michaelis_Menten"),
            ("CyclinE_Cdk2", "pRb", "phosphorylation", "Michaelis_Menten"),
            ("pRb", "E2F", "dissociation", "mass_action"),
            ("E2F", "CyclinE", "transcription", "Hill"),
            ("CyclinE", "CyclinE_Cdk2", "complex_formation", "mass_action"),
        ],
        "feedback_loops": [
            {
                "id": "FL_CELL_CYCLE_E2F_RB_NEG",
                "loop_type": "positive",
                "node_ids": ["PN_E2F", "PN_CyclinE", "PN_CyclinE_Cdk2", "PN_pRb"],
                "delay_minutes": 0.0,
                "description": "E2F → CyclinE → Cdk2 → pRb → 释放更多 E2F（正反馈，G1/S 转换）",
            },
            {
                "id": "FL_P21_CDK2_NEG",
                "loop_type": "negative",
                "node_ids": ["PN_p21", "PN_CyclinE_Cdk2"],
                "delay_minutes": 0.0,
                "description": "p21 抑制 CyclinE_Cdk2（p53 通路交叉）",
            },
        ],
        "cross_talk": [
            {
                "id": "CT_P53_TO_CELL_CYCLE_P21",
                "source_pathway": "p53_signaling",
                "target_pathway": "Cell_Cycle",
                "source_node": "PN_p21",
                "target_node": "PN_CyclinE_Cdk2",
                "mechanism": "inhibition",
                "shared_species": ["p53"],
                "description": "p21 抑制 CyclinE_Cdk2 阻滞 G1/S",
            },
        ],
        "perturbation_points": [
            {"target": "Cdk4", "drug": "Palbociclib", "mechanism": "inhibition"},
            {"target": "Cdk2", "drug": "Dinaciclib", "mechanism": "inhibition"},
        ],
    },

    # =========================================================================
    # 7. JAK-STAT
    # =========================================================================
    "JAK_STAT": {
        "display_name": "JAK-STAT Signaling",
        "source_sbml": get_biomodels_id("JAK_STAT"),
        "source_kegg": "hsa04630",
        "core_nodes": [
            ("IL6", "ligand", "extracellular", TimeScale.FAST),
            ("IL6R", "protein", "membrane", TimeScale.FAST),
            ("JAK", "protein", "membrane", TimeScale.FAST),
            ("pJAK", "protein", "membrane", TimeScale.FAST),
            ("STAT3", "protein", "cytoplasm", TimeScale.FAST),   # shared
            ("pSTAT3", "protein", "cytoplasm", TimeScale.FAST),
            ("pSTAT3_nuclear", "protein", "nucleus", TimeScale.MEDIUM),
            ("SOCS3", "protein", "cytoplasm", TimeScale.MEDIUM),
            ("SOCS3_mRNA", "mrna", "nucleus", TimeScale.MEDIUM),
        ],
        "core_edges": [
            ("IL6", "IL6R", "binding", "mass_action"),
            ("IL6R", "pJAK", "phosphorylation", "Michaelis_Menten"),
            ("pJAK", "pSTAT3", "phosphorylation", "Michaelis_Menten"),
            ("pSTAT3", "pSTAT3_nuclear", "nuclear_import", "mass_action"),
            ("pSTAT3_nuclear", "SOCS3_mRNA", "transcription", "Hill"),
            ("SOCS3_mRNA", "SOCS3", "translation", "mass_action"),
            ("SOCS3", "pJAK", "inhibition", "hybrid"),
        ],
        "feedback_loops": [
            {
                "id": "FL_JAK_STAT_SOCS3_NEG",
                "loop_type": "negative",
                "node_ids": ["PN_pSTAT3_nuclear", "PN_SOCS3_mRNA", "PN_SOCS3", "PN_pJAK"],
                "delay_minutes": 30.0,  # 转录延迟
                "description": "pSTAT3 转录激活 SOCS3，SOCS3 抑制 JAK（负反馈）",
                "source_pmid": "PMID:14532115",
            },
        ],
        "cross_talk": [
            {
                "id": "CT_STAT_TO_APOPTOSIS",
                "source_pathway": "JAK_STAT",
                "target_pathway": "Apoptosis",
                "source_node": "PN_pSTAT3_nuclear",
                "target_node": "PN_Bcl2",
                "mechanism": "transcription",
                "shared_species": ["STAT"],
                "description": "pSTAT3 转录激活 Bcl2 抗凋亡",
            },
        ],
        "perturbation_points": [
            {"target": "JAK", "drug": "Ruxolitinib", "mechanism": "inhibition"},
            {"target": "JAK", "drug": "Tofacitinib", "mechanism": "inhibition"},
        ],
    },

    # =========================================================================
    # 8. NF-κB
    # =========================================================================
    "NF_kB": {
        "display_name": "NF-κB Signaling",
        "source_sbml": get_biomodels_id("NF_KB"),
        "source_kegg": "hsa04064",
        "core_nodes": [
            ("TNFa", "ligand", "extracellular", TimeScale.FAST),
            ("TNFR", "protein", "membrane", TimeScale.FAST),
            ("IKK", "protein", "cytoplasm", TimeScale.FAST),
            ("IKK_active", "protein", "cytoplasm", TimeScale.FAST),
            ("IkBa", "protein", "cytoplasm", TimeScale.MEDIUM),
            ("IkBa_mRNA", "mrna", "nucleus", TimeScale.MEDIUM),
            ("NFkB", "protein", "cytoplasm", TimeScale.MEDIUM),    # shared
            ("NFkB_nuclear", "protein", "nucleus", TimeScale.MEDIUM),
        ],
        "core_edges": [
            ("TNFa", "TNFR", "binding", "mass_action"),
            ("TNFR", "IKK_active", "activation", "hybrid"),
            ("IKK_active", "IkBa", "phosphorylation", "Michaelis_Menten"),
            ("IkBa", "NFkB", "sequestration", "mass_action"),
            ("NFkB", "NFkB_nuclear", "nuclear_import", "mass_action"),
            ("NFkB_nuclear", "IkBa_mRNA", "transcription", "Hill"),
            ("IkBa_mRNA", "IkBa", "translation", "mass_action"),
        ],
        "feedback_loops": [
            {
                "id": "FL_NFKB_IKBA_NEG",
                "loop_type": "negative",
                "node_ids": ["PN_NFkB_nuclear", "PN_IkBa_mRNA", "PN_IkBa", "PN_NFkB"],
                "delay_minutes": 30.0,  # 转录延迟 → 振荡
                "description": "NF-κB 转录激活 IκBα，IκBα 扣留 NF-κB（负反馈振荡）",
                "source_pmid": "PMID:19351952",
            },
        ],
        "cross_talk": [
            {
                "id": "CT_NFKB_TO_APOPTOSIS",
                "source_pathway": "NF_kB",
                "target_pathway": "Apoptosis",
                "source_node": "PN_NFkB_nuclear",
                "target_node": "PN_Bcl2",
                "mechanism": "transcription",
                "shared_species": ["NFkB"],
                "description": "NF-κB 转录激活 Bcl2 抗凋亡",
            },
        ],
        "perturbation_points": [
            {"target": "IKK", "drug": "BMS-345541", "mechanism": "inhibition"},
            {"target": "NFkB", "drug": "Bortezomib", "mechanism": "inhibition"},
        ],
    },

    # =========================================================================
    # 9. Wnt / β-catenin
    # =========================================================================
    "Wnt": {
        "display_name": "Wnt / β-catenin Signaling",
        "source_sbml": get_biomodels_id("WNT"),
        "source_kegg": "hsa04310",
        "core_nodes": [
            ("Wnt", "ligand", "extracellular", TimeScale.FAST),
            ("Frizzled", "protein", "membrane", TimeScale.FAST),
            ("Dishevelled", "protein", "cytoplasm", TimeScale.FAST),
            ("Axin", "protein", "cytoplasm", TimeScale.MEDIUM),
            ("APC", "protein", "cytoplasm", TimeScale.MEDIUM),
            ("GSK3B", "protein", "cytoplasm", TimeScale.MEDIUM),    # shared
            ("bCatenin", "protein", "cytoplasm", TimeScale.MEDIUM),
            ("bCatenin_nuclear", "protein", "nucleus", TimeScale.MEDIUM),
            ("TCF_LEF", "protein", "nucleus", TimeScale.MEDIUM),
            ("Myc_Wnt", "protein", "nucleus", TimeScale.MEDIUM),    # shared (Myc)
        ],
        "core_edges": [
            ("Wnt", "Frizzled", "binding", "mass_action"),
            ("Frizzled", "Dishevelled", "activation", "hybrid"),
            ("Dishevelled", "Axin", "inhibition", "hybrid"),
            ("GSK3B", "bCatenin", "phosphorylation", "Michaelis_Menten"),  # 降解
            ("bCatenin", "bCatenin_nuclear", "nuclear_import", "mass_action"),
            ("bCatenin_nuclear", "TCF_LEF", "binding", "mass_action"),
            ("TCF_LEF", "Myc_Wnt", "transcription", "Hill"),
        ],
        "feedback_loops": [
            {
                "id": "FL_WNT_AXIN_NEG",
                "loop_type": "negative",
                "node_ids": ["PN_bCatenin_nuclear", "PN_Axin"],
                "delay_minutes": 60.0,
                "description": "β-catenin 转录激活 Axin2（负反馈，调节通路活性）",
            },
        ],
        "cross_talk": [
            {
                "id": "CT_PI3K_TO_WNT_GSK3B",
                "source_pathway": "PI3K_AKT_mTOR",
                "target_pathway": "Wnt",
                "source_node": "PN_pAKT",
                "target_node": "PN_GSK3B",
                "mechanism": "inhibition",
                "shared_species": ["GSK3B"],
                "site": "Ser9",
                "description": "pAKT 磷酸化 GSK3B Ser9 抑制其活性 → 稳定 β-catenin",
            },
        ],
        "perturbation_points": [
            {"target": "bCatenin", "drug": "ICG-001", "mechanism": "inhibition"},
            {"target": "Wnt", "drug": "LGK974", "mechanism": "inhibition"},
        ],
    },

    # =========================================================================
    # 10. TGF-β / SMAD
    # =========================================================================
    "TGF_beta": {
        "display_name": "TGF-β / SMAD Signaling",
        "source_sbml": get_biomodels_id("TGF_BETA"),
        "source_kegg": "hsa04350",
        "core_nodes": [
            ("TGFB", "ligand", "extracellular", TimeScale.FAST),
            ("TGFBR", "protein", "membrane", TimeScale.FAST),
            ("TGFBR_active", "protein", "membrane", TimeScale.FAST),
            ("SMAD2", "protein", "cytoplasm", TimeScale.MEDIUM),    # shared
            ("pSMAD2", "protein", "cytoplasm", TimeScale.MEDIUM),
            ("SMAD4", "protein", "cytoplasm", TimeScale.MEDIUM),
            ("SMAD_complex", "complex", "cytoplasm", TimeScale.MEDIUM),
            ("SMAD_complex_nuclear", "complex", "nucleus", TimeScale.MEDIUM),
            ("SMAD7", "protein", "cytoplasm", TimeScale.MEDIUM),
            ("SMAD7_mRNA", "mrna", "nucleus", TimeScale.MEDIUM),
        ],
        "core_edges": [
            ("TGFB", "TGFBR", "binding", "mass_action"),
            ("TGFBR", "TGFBR_active", "phosphorylation", "Michaelis_Menten"),
            ("TGFBR_active", "pSMAD2", "phosphorylation", "Michaelis_Menten"),
            ("pSMAD2", "SMAD_complex", "complex_formation", "mass_action"),
            ("SMAD_complex", "SMAD_complex_nuclear", "nuclear_import", "mass_action"),
            ("SMAD_complex_nuclear", "SMAD7_mRNA", "transcription", "Hill"),
            ("SMAD7_mRNA", "SMAD7", "translation", "mass_action"),
            ("SMAD7", "TGFBR_active", "inhibition", "hybrid"),
        ],
        "feedback_loops": [
            {
                "id": "FL_TGFB_SMAD7_NEG",
                "loop_type": "negative",
                "node_ids": ["PN_SMAD_complex_nuclear", "PN_SMAD7_mRNA", "PN_SMAD7", "PN_TGFBR_active"],
                "delay_minutes": 60.0,
                "description": "SMAD 复合体转录激活 SMAD7，SMAD7 抑制 TGFBR（负反馈）",
                "source_pmid": "PMID:11294857",
            },
        ],
        "cross_talk": [
            {
                "id": "CT_SMAD_TO_CELL_CYCLE",
                "source_pathway": "TGF_beta",
                "target_pathway": "Cell_Cycle",
                "source_node": "PN_SMAD_complex_nuclear",
                "target_node": "PN_CyclinD",
                "mechanism": "inhibition",
                "shared_species": ["SMAD"],
                "description": "SMAD 复合体抑制 Cyclin D 转录（G1 阻滞）",
            },
        ],
        "perturbation_points": [
            {"target": "TGFBR", "drug": "Galunisertib", "mechanism": "inhibition"},
            {"target": "SMAD", "drug": "Trabedersen", "mechanism": "inhibition"},
        ],
    },
}


class PathwayInitializer:
    """10 通路初始化器。

    对应 PART C4。从 PATHWAY_INITIALIZERS 字典读取通路拓扑，
    转换为 PathwayNode / PathwayEdge / FeedbackLoop / CrossTalkEdge 对象列表，
    供 PathwayGraphBuilder 使用。

    使用方式::

        initializer = PathwayInitializer()
        nodes, edges, feedbacks, crosstalks = initializer.get_pathway_init_data("EGFR_RTK")
        builder = PathwayGraphBuilder()
        graph = builder.build(
            pathway_class="EGFR_RTK",
            ontology_entities=ontology,
            reaction_ir=reaction_ir,
            feedback_loops=feedbacks,
            cross_talk_edges=crosstalks,
        )
    """

    @staticmethod
    def list_pathways() -> list[str]:
        """返回所有可用通路类别。"""
        return list(PATHWAY_INITIALIZERS.keys())

    @staticmethod
    def get_pathway_metadata(pathway_class: str) -> dict[str, Any]:
        """返回通路元信息（display_name / source_sbml / source_kegg）。"""
        data = PATHWAY_INITIALIZERS.get(pathway_class, {})
        return {
            "pathway_class": pathway_class,
            "display_name": data.get("display_name", pathway_class),
            "source_sbml": data.get("source_sbml"),
            "source_kegg": data.get("source_kegg"),
        }

    @staticmethod
    def get_pathway_init_data(
        pathway_class: str,
    ) -> tuple[list[PathwayNode], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        """返回通路初始化数据（nodes + edges + feedback_loops + cross_talk_edges）。

        Returns:
            (nodes, edges_raw, feedbacks_raw, crosstalks_raw)
            nodes: PathwayNode 对象列表（已含 states / compartment / time_scale）
            edges_raw: 边的 dict 列表（PathwayGraphBuilder 会从 reaction_ir 提取，
                       此处仅作 reference，主要用 feedback / crosstalk）
            feedbacks_raw: feedback loop dict 列表
            crosstalks_raw: cross-talk edge dict 列表
        """
        data = PATHWAY_INITIALIZERS.get(pathway_class)
        if not data:
            return [], [], [], []

        # 构建 PathwayNode 对象（含状态标记）
        nodes: list[PathwayNode] = []
        for name, species_type, compartment, time_scale in data.get("core_nodes", []):
            # 磷酸化状态标记（如 pEGFR / pAKT 等以 p 开头的物种）
            states: list[PathwayState] = []
            if name.startswith("p") and name[1:2].isupper():
                states.append(PathwayState(
                    name="phosphorylated",
                    state_type="phosphorylation",
                    is_initial=False,
                ))
            elif name.endswith("_nuclear"):
                states.append(PathwayState(
                    name="nuclear",
                    state_type="localization",
                    is_initial=False,
                ))
            elif name.endswith("_active"):
                states.append(PathwayState(
                    name="active",
                    state_type="conformational",
                    is_initial=False,
                ))

            nodes.append(PathwayNode(
                id=f"PN_{name}",
                canonical_name=name,
                display_name=name,
                species_type=species_type,
                pathway_class=pathway_class,
                module=PathwayModule.CORE,
                compartment=compartment,
                time_scale=time_scale,
                states=states,
            ))

        # edges_raw（仅作 reference，实际边由 PathwayGraphBuilder 从 reaction_ir 提取）
        edges_raw: list[dict[str, Any]] = []
        for src, tgt, mechanism, kinetics in data.get("core_edges", []):
            edges_raw.append({
                "source": src,
                "target": tgt,
                "mechanism": mechanism,
                "kinetics_type": kinetics,
                "pathway_tag": pathway_class,
            })

        feedbacks_raw = data.get("feedback_loops", [])
        crosstalks_raw = data.get("cross_talk", [])

        return nodes, edges_raw, feedbacks_raw, crosstalks_raw

    @staticmethod
    def get_perturbation_points(pathway_class: str) -> list[dict[str, Any]]:
        """返回通路扰动点（药物 / KO / 突变影响点）。"""
        data = PATHWAY_INITIALIZERS.get(pathway_class, {})
        return data.get("perturbation_points", [])

    @staticmethod
    def get_all_cross_talk_edges() -> list[dict[str, Any]]:
        """返回所有通路的 cross-talk edges（用于 multi-pathway 集成）。"""
        all_ct: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for pwc, data in PATHWAY_INITIALIZERS.items():
            for ct in data.get("cross_talk", []):
                if ct.get("id") not in seen_ids:
                    all_ct.append(ct)
                    seen_ids.add(ct.get("id", ""))
        return all_ct


__all__ = ["PATHWAY_INITIALIZERS", "PathwayInitializer"]
