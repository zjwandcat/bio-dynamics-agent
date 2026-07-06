# BioDynamics Agent v4 - Wnt Specialist (Phase 4 / Task 4.11)
# Wnt / β-catenin 通路 Specialist：实现 Off 状态 destruction complex
# （Axin+APC+GSK3β+CK1→β-catenin 磷酸化→泛素化→降解）+ On 状态
# （Wnt+Frizzled+LRP5/6→Dvl→LRP6 磷酸化→Axin 招募→destruction complex 解离→
# β-catenin 累积→入核→TCF/LEF 转录）核心拓扑 + β-catenin→Axin2 转录延迟
# 负反馈（delay=30min）+ Axin2 重建 destruction complex 负反馈 + LRP6 磷酸化
# 维持 destruction complex 解离正反馈。
#
# 设计原则（铁律）：
# 1. Feature Flag V4_PATHWAY_SPECIALIST_ENABLED=false 时由 LangGraph hook 层短路，
#    本 Specialist 代码不被调用（基类本身不做 flag 检查）
# 2. 使用 Task 4.0 修复后的 PHOSPHORYLATION 语义：
#    - 异磷酸化（Axin_APC_GSK3b_bcat → p_bcat）：β-catenin 作 substrate,
#      p_bcat 作 product，Axin_APC_GSK3b destruction complex 作 catalytic modifier
#      （GSK3β+CK1 在 destruction complex 内磷酸化 β-catenin Ser33/37/Thr41）
#    - 异磷酸化（pDvl → pLRP6）：LRP6 作 substrate，pLRP6 作 product，
#      pDvl 作 catalytic modifier
# 3. Destruction Complex 五步耦合 CompositeReaction
#    （binding→binding→phosphorylation→ubiquitination→proteasomal_degradation），
#    不压扁为单一 reaction，保留中间产物语义
# 4. 不处理 β-catenin→Cyclin D1 细胞周期下游（由 Cell Cycle Specialist 处理，
#    Task 4.6）；Cyclin D1 / cMyc 转录仅作为 cross-talk Reaction 片段输出
# 5. 不生成 PI3K-AKT→GSK3β cross-talk edge 本身（由 Cross-talk Coordinator 处理，
#    Task 4.13），仅返回本通路侧的 cross-talk Reaction 片段
# 6. apply_* 方法实现捕获异常并返回空 list/dict，记录 logger.warning，不抛异常
#
# 依赖：
# - P1 ontology / pathway_registry（Wnt 通路类别键）
# - P2 schema（SpeciesV2 / ReactionV2 / SpeciesRef / Modifier / CompositeReaction）
# - P2 MechanismType（COMPLEX_FORMATION / BINDING / PHOSPHORYLATION /
#   UBIQUITINATION / PROTEASOMAL_DEGRADATION / DISSOCIATION / NUCLEAR_IMPORT /
#   TRANSCRIPTION / TRANSLATION / INHIBITION / ACTIVATION / TRANSLOCATION）
# - P3 ode_templates_v2（bistable_switch.j2 / _mechanism_phosphorylation_mm.j2 模板）
# - P3 pathway_graph/initializer.py（Wnt core_nodes / core_edges）
# - P2 composite_reaction.py（CR_WNT_DESTRUCTION 五步通路参考）
#
# 参考：
# - spec.md Part 3 Specialist 9（第 248-253 行）
# - tasks.md Task 4.11（第 116-125 行）
# - Polakis 2002 Wnt signaling (PMID:12064617)
# - Lee 2003 Wnt model (BioModels BIOMD0000000008)

from __future__ import annotations

import logging
from typing import Any

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
# Wnt 通路标签
# =============================================================================
PATHWAY_TAG: str = "WNT"

# SBML BioModels ID（Lee 2003 Wnt model）
SOURCE_SBML: str = "BIOMD0000000008"

# Validation benchmark PMID 引用（Polakis 2002 Wnt signaling）
_PMID_POLAKIS_2002: str = "PMID:12064617"

# β-catenin→Axin2 转录延迟负反馈延迟（DDE，分钟）
# Polakis 2002 模型中 β-catenin 入核转录激活 Axin2 mRNA（含 30 min 转录延迟），
# Axin2 蛋白重新合成促进 destruction complex 形成，形成 delay=30min 延迟负反馈，
# 调节 β-catenin 稳态水平与 Axin2 mRNA 1-2h 达峰。
_WNT_BCAT_AXIN2_DELAY_MINUTES: float = 30.0


# =============================================================================
# Wnt 核心物种（与 P3 pathway_graph/initializer.py Wnt.core_nodes 对齐，
# 扩展 Wnt_Fz_LRP_complex / pDvl / pLRP6 / Axin_APC / Axin_APC_GSK3b /
# Axin_APC_GSK3b_bcat / p_bcat / ub_bcat / bcat_degraded / destruction_complex_disrupted /
# destruction_complex_reformed / Axin2_mRNA / Cyclin_D1_mRNA / cMyc_mRNA 完整 Wnt 拓扑）
# =============================================================================
# bCatenin 物种标记 shared=True（与 Cell Cycle Specialist 的 Cyclin D1 路径共享，
# β-catenin 转录 Cyclin D1 推动细胞周期，下游 Cyclin D1→Cdk4 由 Cell Cycle Specialist 处理）
# GSK3b 物种标记 shared=True（与 PI3K-AKT Specialist 的 pAKT→GSK3β 路径共享，
# pAKT 抑制 GSK3β 阻止 β-catenin 降解，cross-talk edge 由 Coordinator 管理）
_WNT_CORE_SPECIES: list[dict[str, Any]] = [
    # ---- 配体 + 受体 ----
    # Wnt（Wnt 配体，分泌型糖蛋白，结合 Frizzled 受体启动 Wnt 信号）
    {"name": "Wnt", "species_type": "ligand",
     "compartment": "extracellular"},
    # Frizzled（Fz，Wnt 七次跨膜受体，与 LRP5/6 形成复合物）
    {"name": "Frizzled", "species_type": "protein",
     "compartment": "membrane"},
    # LRP5_6（LDL receptor-related protein 5/6，Wnt 共受体，被 Dvl 磷酸化）
    {"name": "LRP5_6", "species_type": "protein",
     "compartment": "membrane"},
    # Wnt_Fz_LRP_complex（Wnt+Frizzled+LRP5/6 三元复合物，受体激活启动下游）
    {"name": "Wnt_Fz_LRP_complex", "species_type": "complex",
     "compartment": "membrane"},
    # ---- Dvl 信号 ----
    # Dvl（Dishevelled，胞质信号转导蛋白，被 Wnt_Fz_LRP_complex 磷酸化激活）
    {"name": "Dvl", "species_type": "protein",
     "compartment": "cytoplasm"},
    # pDvl（磷酸化激活的 Dvl，催化 LRP6 磷酸化）
    {"name": "pDvl", "species_type": "protein",
     "compartment": "cytoplasm"},
    # pLRP6（磷酸化 LRP6，pDvl 异磷酸化 LRP6 PPPSP motifs，招募 Axin）
    {"name": "pLRP6", "species_type": "protein",
     "compartment": "membrane"},
    # ---- Destruction Complex 组分（Off 状态） ----
    # Axin（scaffold 蛋白，destruction complex 核心 scaffold，与 APC/GSK3β/CK1 结合）
    {"name": "Axin", "species_type": "protein",
     "compartment": "cytoplasm"},
    # Axin_recruited（被招募到膜的 Axin，从 destruction complex 解离）
    {"name": "Axin_recruited", "species_type": "protein",
     "compartment": "membrane"},
    # APC（Adenomatous Polyposis Coli，tumor suppressor，destruction complex 组分）
    {"name": "APC", "species_type": "protein",
     "compartment": "cytoplasm"},
    # GSK3b（GSK3β，serine/threonine kinase，destruction complex 组分，shared）
    # pAKT 抑制 GSK3β Ser9 阻止 β-catenin 降解（PI3K-AKT→Wnt cross-talk）
    {"name": "GSK3b", "species_type": "protein",
     "compartment": "cytoplasm", "shared": True},
    # CK1（Casein Kinase 1，destruction complex 组分，β-catenin Thr41 priming）
    {"name": "CK1", "species_type": "protein",
     "compartment": "cytoplasm"},
    # ---- Destruction Complex 中间产物 ----
    # Axin_APC（Axin-APC 二元复合物，destruction complex 第 1 步组装产物）
    {"name": "Axin_APC", "species_type": "complex",
     "compartment": "cytoplasm"},
    # Axin_APC_GSK3b（Axin-APC-GSK3β 三元复合物，destruction complex 第 2 步组装产物）
    {"name": "Axin_APC_GSK3b", "species_type": "complex",
     "compartment": "cytoplasm"},
    # Axin_APC_GSK3b_bcat（destruction complex-β-catenin 复合物，招募 β-catenin）
    {"name": "Axin_APC_GSK3b_bcat", "species_type": "complex",
     "compartment": "cytoplasm"},
    # ---- β-catenin 五步降解中间产物 ----
    # bCatenin（β-catenin，cytoplasmic pool，shared：与 Cell Cycle Specialist
    # Cyclin D1 路径共享，β-catenin 转录 Cyclin D1 推动细胞周期）
    {"name": "bCatenin", "species_type": "protein",
     "compartment": "cytoplasm", "shared": True},
    # p_bcat（磷酸化 β-catenin，Ser33/37/Thr41，五步降解第 3 步产物）
    {"name": "p_bcat", "species_type": "protein",
     "compartment": "cytoplasm"},
    # ub_bcat（泛素化 β-catenin，五步降解第 4 步产物，被 β-TrCP E3 ligase 标记）
    {"name": "ub_bcat", "species_type": "protein",
     "compartment": "cytoplasm"},
    # bcat_degraded（蛋白酶体降解后的 β-catenin 残余，五步降解第 5 步产物）
    {"name": "bcat_degraded", "species_type": "protein",
     "compartment": "cytoplasm"},
    # ---- Destruction Complex 状态 ----
    # destruction_complex_disrupted（解离状态的 destruction complex，Wnt On 时由 Axin 招募触发）
    {"name": "destruction_complex_disrupted", "species_type": "complex",
     "compartment": "cytoplasm"},
    # destruction_complex_reformed（重建状态的 destruction complex，Axin2 重建执行负反馈）
    {"name": "destruction_complex_reformed", "species_type": "complex",
     "compartment": "cytoplasm"},
    # ---- β-catenin 入核 + 转录 ----
    # bCatenin_nuclear（核内 β-catenin，累积后与 TCF/LEF 形成转录复合物）
    {"name": "bCatenin_nuclear", "species_type": "protein",
     "compartment": "nucleus"},
    # TCF_LEF（T-cell factor/Lymphoid enhancer factor，转录因子，与 β-catenin 形成复合物）
    {"name": "TCF_LEF", "species_type": "protein",
     "compartment": "nucleus"},
    # TCF_LEF_bcat_complex（β-catenin-TCF/LEF 转录激活复合物，激活下游靶基因）
    {"name": "TCF_LEF_bcat_complex", "species_type": "complex",
     "compartment": "nucleus"},
    # ---- Axin2 负反馈 ----
    # Axin2_mRNA（Axin2 mRNA，β-catenin 转录激活，含 30 min 转录延迟，负反馈）
    {"name": "Axin2_mRNA", "species_type": "mrna",
     "compartment": "nucleus"},
    # Axin2（Axin2 蛋白，重新合成促进 destruction complex 形成降解 β-catenin）
    {"name": "Axin2", "species_type": "protein",
     "compartment": "cytoplasm"},
    # ---- Cyclin D1 / cMyc 转录（cross-talk 到 Cell Cycle，下游由 Cell Cycle 处理）----
    # Cyclin_D1_mRNA（Cyclin D1 mRNA，β-catenin 转录激活推动细胞周期）
    {"name": "Cyclin_D1_mRNA", "species_type": "mrna",
     "compartment": "nucleus"},
    # cMyc_mRNA（c-Myc mRNA，β-catenin 转录激活促增殖）
    {"name": "cMyc_mRNA", "species_type": "mrna",
     "compartment": "nucleus"},
]


# =============================================================================
# Wnt 核心反应（17 条：Off 状态 6 + On 状态 11）
# =============================================================================
# 每条反应含：source / target / mechanism / kinetics_type / pathway_tag /
#             substrate / product / modifier / modifier_type / autophosphorylation
#
# kinetics_type 选择：
# - complex_formation / nuclear_import / translation / dissociation → mass_action
# - phosphorylation → Michaelis_Menten（与 P3 _mechanism_phosphorylation_mm 模板对齐）
# - ubiquitination → Michaelis_Menten（E3 ligase 催化，MM 动力学）
# - proteasomal_degradation → mass_action（蛋白酶体降解，一级动力学）
# - transcription → Hill（β-catenin-TCF/LEF 作转录因子，Hill 动力学 n=2 协同结合）
# - activation / translocation → mass_action（信号传导，一级动力学）
_WNT_CORE_REACTIONS: list[dict[str, Any]] = [
    # ============== Off 状态（无 Wnt 信号，β-catenin 降解，6 条） ==============
    # 1. Axin + APC → Axin_APC（complex_formation, destruction complex 第 1 步 binding）
    #    Axin scaffold 蛋白结合 APC tumor suppressor 形成 destruction complex 骨架
    #    ★ Destruction Complex 五步 CompositeReaction step 1（binding）
    {
        "source": "Axin",
        "target": "Axin_APC",
        "mechanism": "complex_formation",
        "kinetics_type": "mass_action",
        "pathway_tag": PATHWAY_TAG,
        "substrate": "APC",
        "product": "Axin_APC",
        "modifier": "Axin",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "Axin + APC → Axin_APC（complex_formation, destruction complex 第 1 步 binding, Axin scaffold 结合 APC tumor suppressor 形成骨架）",
        "composite_step": 1,
        "composite_id": "CR_DESTRUCTION_COMPLEX",
    },
    # 2. Axin_APC + GSK3β → Axin_APC_GSK3b（complex_formation, 第 2 步 binding）
    #    Axin_APC 复合物招募 GSK3β kinase，形成三元 destruction complex
    #    ★ Destruction Complex 五步 CompositeReaction step 2（binding）
    {
        "source": "Axin_APC",
        "target": "Axin_APC_GSK3b",
        "mechanism": "complex_formation",
        "kinetics_type": "mass_action",
        "pathway_tag": PATHWAY_TAG,
        "substrate": "GSK3b",
        "product": "Axin_APC_GSK3b",
        "modifier": "Axin_APC",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "Axin_APC + GSK3β → Axin_APC_GSK3b（complex_formation, destruction complex 第 2 步 binding, 招募 GSK3β kinase 形成三元复合物）",
        "composite_step": 2,
        "composite_id": "CR_DESTRUCTION_COMPLEX",
    },
    # 3. Axin_APC_GSK3b + β-catenin → Axin_APC_GSK3b_bcat（complex_formation, 招募 β-catenin）
    #    Destruction complex 招募 β-catenin 进入降解机器（不属于五步 CompositeReaction，
    #    为 β-catenin 招募步骤，在磷酸化前发生）
    {
        "source": "Axin_APC_GSK3b",
        "target": "Axin_APC_GSK3b_bcat",
        "mechanism": "complex_formation",
        "kinetics_type": "mass_action",
        "pathway_tag": PATHWAY_TAG,
        "substrate": "bCatenin",
        "product": "Axin_APC_GSK3b_bcat",
        "modifier": "Axin_APC_GSK3b",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "Axin_APC_GSK3b + β-catenin → Axin_APC_GSK3b_bcat（complex_formation, destruction complex 招募 β-catenin 进入降解机器, 不属于五步 CompositeReaction, 为磷酸化前招募步骤）",
    },
    # 4. Axin_APC_GSK3b_bcat → p_bcat（phosphorylation, β-catenin 作 substrate,
    #    destruction complex 作 catalytic modifier, GSK3β+CK1 磷酸化 Ser33/37/Thr41）
    #    ★ Destruction Complex 五步 CompositeReaction step 3（phosphorylation）
    {
        "source": "Axin_APC_GSK3b_bcat",
        "target": "p_bcat",
        "mechanism": "phosphorylation",
        "kinetics_type": "Michaelis_Menten",
        "pathway_tag": PATHWAY_TAG,
        # 异磷酸化：bCatenin 作 substrate，p_bcat 作 product，
        # Axin_APC_GSK3b destruction complex 作 catalytic modifier（GSK3β+CK1 在复合物内催化）
        "substrate": "bCatenin",
        "product": "p_bcat",
        "modifier": "Axin_APC_GSK3b",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "site": "Ser33/37/Thr41",
        "description": "Axin_APC_GSK3b_bcat → p_bcat（phosphorylation, bCatenin 作 substrate, Axin_APC_GSK3b 作 catalytic modifier, GSK3β+CK1 在 destruction complex 内磷酸化 β-catenin Ser33/37/Thr41, 五步 CompositeReaction step 3）",
        "composite_step": 3,
        "composite_id": "CR_DESTRUCTION_COMPLEX",
    },
    # 5. p_bcat → ub_bcat（ubiquitination, p_bcat 作 substrate, β-TrCP E3 ligase 作 modifier）
    #    β-TrCP E3 泛素连接酶识别磷酸化 β-catenin，多泛素化标记
    #    ★ Destruction Complex 五步 CompositeReaction step 4（ubiquitination）
    {
        "source": "p_bcat",
        "target": "ub_bcat",
        "mechanism": "ubiquitination",
        "kinetics_type": "Michaelis_Menten",
        "pathway_tag": PATHWAY_TAG,
        # p_bcat 作 substrate，ub_bcat 作 product，β-TrCP E3 ligase 作 catalytic modifier
        "substrate": "p_bcat",
        "product": "ub_bcat",
        "modifier": "BTRCP_E3_ligase",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "p_bcat → ub_bcat（ubiquitination, p_bcat 作 substrate, BTRCP_E3_ligase 作 catalytic modifier, β-TrCP E3 泛素连接酶识别磷酸化 β-catenin 多泛素化标记, 五步 CompositeReaction step 4）",
        "composite_step": 4,
        "composite_id": "CR_DESTRUCTION_COMPLEX",
    },
    # 6. ub_bcat → bcat_degraded（proteasomal_degradation, 26S 蛋白酶体降解泛素化 β-catenin）
    #    26S 蛋白酶体识别多泛素链降解 β-catenin（五步 CompositeReaction step 5）
    #    ★ Destruction Complex 五步 CompositeReaction step 5（proteasomal_degradation）
    {
        "source": "ub_bcat",
        "target": "bcat_degraded",
        "mechanism": "proteasomal_degradation",
        "kinetics_type": "mass_action",
        "pathway_tag": PATHWAY_TAG,
        # ub_bcat 作 substrate（被降解），bcat_degraded 作 product（残余）
        "substrate": "ub_bcat",
        "product": "bcat_degraded",
        "modifier": "proteasome",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "ub_bcat → bcat_degraded（proteasomal_degradation, ub_bcat 作 substrate, proteasome 作 catalytic modifier, 26S 蛋白酶体识别多泛素链降解 β-catenin, 五步 CompositeReaction step 5）",
        "composite_step": 5,
        "composite_id": "CR_DESTRUCTION_COMPLEX",
    },
    # ============== On 状态（Wnt 信号，β-catenin 累积，11 条） ==============
    # 7. Wnt + Frizzled + LRP5/6 → Wnt_Fz_LRP_complex（complex_formation, 受体激活）
    #    Wnt 配体结合 Frizzled + LRP5/6 形成三元受体复合物，启动 Wnt 信号
    {
        "source": "Wnt",
        "target": "Wnt_Fz_LRP_complex",
        "mechanism": "complex_formation",
        "kinetics_type": "mass_action",
        "pathway_tag": PATHWAY_TAG,
        "substrate": "Frizzled",
        "product": "Wnt_Fz_LRP_complex",
        "modifier": "Wnt",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "Wnt + Frizzled + LRP5/6 → Wnt_Fz_LRP_complex（complex_formation, Wnt 配体结合 Frizzled+LRP5/6 形成三元受体复合物启动 Wnt 信号）",
    },
    # 8. Wnt_Fz_LRP_complex → pDvl（activation, Dvl 磷酸化激活）
    #    Wnt_Fz_LRP_complex 招募并磷酸化激活 Dvl（Dishevelled），传递 Wnt 信号
    {
        "source": "Wnt_Fz_LRP_complex",
        "target": "pDvl",
        "mechanism": "activation",
        "kinetics_type": "mass_action",
        "pathway_tag": PATHWAY_TAG,
        # Dvl 作 substrate，pDvl 作 product，Wnt_Fz_LRP_complex 作 catalytic modifier
        "substrate": "Dvl",
        "product": "pDvl",
        "modifier": "Wnt_Fz_LRP_complex",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "Wnt_Fz_LRP_complex → pDvl（activation, Dvl 作 substrate, Wnt_Fz_LRP_complex 作 catalytic modifier, Wnt_Fz_LRP_complex 招募并磷酸化激活 Dvl 传递 Wnt 信号）",
    },
    # 9. pDvl → pLRP6（phosphorylation, LRP6 作 substrate, pDvl 作 catalytic modifier）
    #    pDvl 异磷酸化 LRP6 PPPSP motifs，pLRP6 招募 Axin 到膜
    #    ★ 异磷酸化（LRP6 作 substrate，pLRP6 作 product，pDvl 作 catalytic modifier）
    {
        "source": "pDvl",
        "target": "pLRP6",
        "mechanism": "phosphorylation",
        "kinetics_type": "Michaelis_Menten",
        "pathway_tag": PATHWAY_TAG,
        # 异磷酸化：LRP6 作 substrate，pLRP6 作 product，pDvl 作 catalytic modifier
        "substrate": "LRP5_6",
        "product": "pLRP6",
        "modifier": "pDvl",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "site": "PPPSP_motifs",
        "description": "pDvl → pLRP6（phosphorylation, LRP5_6 作 substrate, pLRP6 作 product, pDvl 作 catalytic modifier, pDvl 异磷酸化 LRP6 PPPSP motifs, pLRP6 招募 Axin 到膜）",
    },
    # 10. pLRP6 → Axin_recruited（translocation, Axin 被招募到膜）
    #     pLRP6 招募 Axin 从 destruction complex 解离并转位到膜上
    {
        "source": "pLRP6",
        "target": "Axin_recruited",
        "mechanism": "translocation",
        "kinetics_type": "mass_action",
        "pathway_tag": PATHWAY_TAG,
        "substrate": "Axin",
        "product": "Axin_recruited",
        "modifier": "pLRP6",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "pLRP6 → Axin_recruited（translocation, Axin 作 substrate, Axin_recruited 作 product, pLRP6 作 catalytic modifier, pLRP6 招募 Axin 从 destruction complex 解离并转位到膜上）",
    },
    # 11. Axin_recruited → destruction_complex_disrupted（dissociation, destruction complex 解离）
    #     Axin 被招募到膜后 destruction complex 解离，β-catenin 逃逸降解
    {
        "source": "Axin_recruited",
        "target": "destruction_complex_disrupted",
        "mechanism": "dissociation",
        "kinetics_type": "mass_action",
        "pathway_tag": PATHWAY_TAG,
        "substrate": "Axin_APC_GSK3b",
        "product": "destruction_complex_disrupted",
        "modifier": "Axin_recruited",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "Axin_recruited → destruction_complex_disrupted（dissociation, Axin_APC_GSK3b 作 substrate, destruction_complex_disrupted 作 product, Axin_recruited 作 catalytic modifier, Axin 被招募到膜后 destruction complex 解离 β-catenin 逃逸降解）",
    },
    # 12. bCatenin → bCatenin_nuclear（nuclear_import, β-catenin 累积后入核）
    #     Destruction complex 解离后 β-catenin 累积，通过 importin 入核作为转录因子
    {
        "source": "bCatenin",
        "target": "bCatenin_nuclear",
        "mechanism": "nuclear_import",
        "kinetics_type": "mass_action",
        "pathway_tag": PATHWAY_TAG,
        "substrate": "bCatenin",
        "product": "bCatenin_nuclear",
        "modifier": "bCatenin",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "bCatenin → bCatenin_nuclear（nuclear_import, bCatenin 作 substrate, bCatenin_nuclear 作 product, destruction complex 解离后 β-catenin 累积通过 importin 入核作为转录因子）",
    },
    # 13. bCatenin_nuclear + TCF/LEF → TCF_LEF_bcat_complex（complex_formation, 转录激活复合物）
    #     核内 β-catenin 结合 TCF/LEF 转录因子，形成转录激活复合物
    {
        "source": "bCatenin_nuclear",
        "target": "TCF_LEF_bcat_complex",
        "mechanism": "complex_formation",
        "kinetics_type": "mass_action",
        "pathway_tag": PATHWAY_TAG,
        "substrate": "TCF_LEF",
        "product": "TCF_LEF_bcat_complex",
        "modifier": "bCatenin_nuclear",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "bCatenin_nuclear + TCF/LEF → TCF_LEF_bcat_complex（complex_formation, TCF_LEF 作 substrate, TCF_LEF_bcat_complex 作 product, bCatenin_nuclear 作 catalytic modifier, 核内 β-catenin 结合 TCF/LEF 形成转录激活复合物）",
    },
    # 14. TCF_LEF_bcat_complex → Axin2_mRNA（transcription, Hill 动力学, β-catenin 转录 Axin2 负反馈）
    #     TCF_LEF_bcat_complex 结合 Axin2 基因启动子，Hill 协同结合（n=2）激活转录
    #     （含 30 min 转录延迟，由 FeedbackLoop FL_BCAT_AXIN2 表达，形成负反馈）
    {
        "source": "TCF_LEF_bcat_complex",
        "target": "Axin2_mRNA",
        "mechanism": "transcription",
        "kinetics_type": "Hill",
        "pathway_tag": PATHWAY_TAG,
        # TCF_LEF_bcat_complex 作 modifier（转录因子），DNA 作 substrate，Axin2_mRNA 作 product
        "substrate": "DNA",
        "product": "Axin2_mRNA",
        "modifier": "TCF_LEF_bcat_complex",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "hill_coefficient": 2,
        "description": "TCF_LEF_bcat_complex → Axin2_mRNA（transcription, Hill n=2, TCF_LEF_bcat_complex 作转录因子, 含 30 min 转录延迟负反馈, Axin2 重建 destruction complex 降解 β-catenin）",
    },
    # 15. Axin2_mRNA → Axin2（translation, mRNA→蛋白）
    #     Axin2 mRNA 翻译为 Axin2 蛋白，促进 destruction complex 形成负反馈
    {
        "source": "Axin2_mRNA",
        "target": "Axin2",
        "mechanism": "translation",
        "kinetics_type": "mass_action",
        "pathway_tag": PATHWAY_TAG,
        "substrate": "Axin2_mRNA",
        "product": "Axin2",
        "modifier": "Axin2_mRNA",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "Axin2_mRNA → Axin2（translation, Axin2_mRNA 作 substrate, Axin2 作 product, Axin2 蛋白促进 destruction complex 形成负反馈）",
    },
    # 16. TCF_LEF_bcat_complex → Cyclin_D1_mRNA（transcription, β-catenin 转录 Cyclin D1）
    #     β-catenin 转录激活 Cyclin D1（CCND1）推动细胞周期，下游由 Cell Cycle Specialist 处理
    {
        "source": "TCF_LEF_bcat_complex",
        "target": "Cyclin_D1_mRNA",
        "mechanism": "transcription",
        "kinetics_type": "Hill",
        "pathway_tag": PATHWAY_TAG,
        "substrate": "DNA",
        "product": "Cyclin_D1_mRNA",
        "modifier": "TCF_LEF_bcat_complex",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "hill_coefficient": 2,
        "description": "TCF_LEF_bcat_complex → Cyclin_D1_mRNA（transcription, Hill n=2, β-catenin 转录激活 Cyclin D1 推动细胞周期, 下游 Cyclin D1→Cdk4 由 Cell Cycle Specialist 处理）",
    },
    # 17. TCF_LEF_bcat_complex → cMyc_mRNA（transcription, β-catenin 转录 cMyc）
    #     β-catenin 转录激活 c-Myc 促增殖，下游由 Cell Cycle Specialist 处理
    {
        "source": "TCF_LEF_bcat_complex",
        "target": "cMyc_mRNA",
        "mechanism": "transcription",
        "kinetics_type": "Hill",
        "pathway_tag": PATHWAY_TAG,
        "substrate": "DNA",
        "product": "cMyc_mRNA",
        "modifier": "TCF_LEF_bcat_complex",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "hill_coefficient": 2,
        "description": "TCF_LEF_bcat_complex → cMyc_mRNA（transcription, Hill n=2, β-catenin 转录激活 c-Myc 促增殖, 下游 c-Myc→Cyclin D 由 Cell Cycle Specialist 处理）",
    },
]


# =============================================================================
# Destruction Complex 五步耦合 CompositeReaction
# （binding→binding→phosphorylation→ubiquitination→proteasomal_degradation）
# =============================================================================
# Destruction Complex 五步耦合降解是 Wnt 通路的核心调控机制：
# - Step 1: Axin + APC → Axin_APC（complex_formation/binding, destruction complex 第 1 步组装）
# - Step 2: Axin_APC + GSK3β → Axin_APC_GSK3b（complex_formation/binding, 第 2 步组装）
# - Step 3: Axin_APC_GSK3b_bcat → p_bcat（phosphorylation, GSK3β+CK1 磷酸化 β-catenin）
# - Step 4: p_bcat → ub_bcat（ubiquitination, β-TrCP E3 ligase 泛素化）
# - Step 5: ub_bcat → bcat_degraded（proteasomal_degradation, 26S 蛋白酶体降解）
# 净反应：β-catenin → ∅（无 Wnt 信号时 destruction complex 降解 β-catenin 维持稳态 <10 nM）
#
# 五步耦合是顺序执行（sequential coupling），中间产物 Axin_APC / Axin_APC_GSK3b /
# p_bcat / ub_bcat 保留语义不压扁为单一 reaction（CompositeReaction 设计原则）
_WNT_COMPOSITE_REACTIONS: list[dict[str, Any]] = [
    {
        "id": "CR_DESTRUCTION_COMPLEX",
        "name": "Wnt Destruction Complex Five-Step Coupled Degradation",
        "mechanism": "sequential",
        "template": "destruction_complex.j2",
        "loop_type": "negative",
        "coupling_type": "sequential",
        "sub_reactions": [
            # Step 1: binding (Axin + APC → Axin_APC, complex_formation)
            "Axin → Axin_APC",
            # Step 2: binding (Axin_APC + GSK3β → Axin_APC_GSK3b, complex_formation)
            "Axin_APC → Axin_APC_GSK3b",
            # Step 3: phosphorylation (Axin_APC_GSK3b_bcat → p_bcat, GSK3β+CK1)
            "Axin_APC_GSK3b_bcat → p_bcat",
            # Step 4: ubiquitination (p_bcat → ub_bcat, β-TrCP E3 ligase)
            "p_bcat → ub_bcat",
            # Step 5: proteasomal_degradation (ub_bcat → bcat_degraded, 26S proteasome)
            "ub_bcat → bcat_degraded",
        ],
        "intermediate_species": [
            "Axin_APC",              # destruction complex 第 1 步组装产物（step 1 产物, step 2 substrate）
            "Axin_APC_GSK3b",        # destruction complex 第 2 步组装产物（step 2 产物, 招募 β-catenin）
            "Axin_APC_GSK3b_bcat",   # destruction complex-β-catenin 复合物（招募产物, step 3 substrate）
            "p_bcat",                # 磷酸化 β-catenin（step 3 产物, step 4 substrate）
            "ub_bcat",               # 泛素化 β-catenin（step 4 产物, step 5 substrate）
            "bcat_degraded",         # 降解残余（step 5 产物, β-catenin 降解完成）
        ],
        "net_reaction": "β-catenin → ∅ (Wnt off state, destruction complex 五步降解)",
        "node_ids": [
            "Axin",
            "APC",
            "GSK3b",
            "Axin_APC",
            "Axin_APC_GSK3b",
            "Axin_APC_GSK3b_bcat",
            "p_bcat",
            "ub_bcat",
            "bcat_degraded",
        ],
        "reactions": [
            "Axin → Axin_APC",
            "Axin_APC → Axin_APC_GSK3b",
            "Axin_APC_GSK3b_bcat → p_bcat",
            "p_bcat → ub_bcat",
            "ub_bcat → bcat_degraded",
        ],
        "description": (
            "Destruction Complex 五步耦合降解（binding→binding→phosphorylation→"
            "ubiquitination→proteasomal_degradation）：Axin+APC→Axin_APC, "
            "Axin_APC+GSK3β→Axin_APC_GSK3b, destruction complex 磷酸化 β-catenin "
            "Ser33/37/Thr41, β-TrCP E3 泛素化 p_bcat, 26S 蛋白酶体降解 ub_bcat, "
            "净反应 β-catenin→∅ 维持稳态 <10 nM (Wnt off state), "
            "sequential coupling 保留中间产物语义 (Polakis 2002, PMID:12064617)"
        ),
        "delay_minutes": 0.0,   # 五步耦合本身无转录延迟（蛋白级联）
        "pmid": _PMID_POLAKIS_2002,
    },
]


# =============================================================================
# Wnt 反馈环（3 条：β-catenin→Axin2 延迟负反馈 + Axin2 重建 destruction complex
# + LRP6 磷酸化维持 destruction complex 解离正反馈）
# =============================================================================
_WNT_FEEDBACK_LOOPS: list[dict[str, Any]] = [
    # 1. β-catenin→Axin2 转录延迟负反馈（DDE, delay=30min）
    #    Polakis 2002 (PMID:12064617) 经典模型：
    #    - β-catenin 入核转录激活 Axin2 mRNA（含 30 min 转录延迟）
    #    - Axin2 蛋白重新合成促进 destruction complex 形成，降解 β-catenin
    #    - 形成 delay=30min 的延迟负反馈，调节 β-catenin 稳态与 Axin2 mRNA 1-2h 达峰
    {
        "id": "FL_BCAT_AXIN2",
        "loop_type": "negative",
        "node_ids": [
            "bCatenin_nuclear",
            "TCF_LEF_bcat_complex",
            "Axin2_mRNA",
            "Axin2",
        ],
        "delay_minutes": _WNT_BCAT_AXIN2_DELAY_MINUTES,  # 30.0 min 转录延迟
        "bistable": False,
        "template": "destruction_complex.j2",
        "description": (
            "β-catenin→Axin2 转录延迟负反馈（bCatenin_nuclear 转录激活 Axin2 mRNA, "
            "Axin2 蛋白重新合成促进 destruction complex 形成降解 β-catenin, "
            "delay=30min, Axin2 mRNA 1-2h 达峰, Polakis 2002）"
        ),
        "source_pmid": _PMID_POLAKIS_2002,
        "dde_solver": "solvers/dde_solver.py",
    },
    # 2. Axin2 → destruction_complex_reformed（negative feedback, Axin2 重建 destruction complex）
    #    Axin2 蛋白作为 scaffold 重建 destruction complex，降解 β-catenin 执行负反馈
    #    （Axin2 与 Axin1 功能同源，但受 β-catenin 转录调控，是负反馈效应器）
    {
        "id": "FL_AXIN2_DC_REFORMED",
        "loop_type": "negative",
        "node_ids": [
            "Axin2",
            "destruction_complex_reformed",
            "bCatenin",
        ],
        "delay_minutes": 0.0,   # 蛋白结合重建，无转录延迟
        "bistable": False,
        "description": (
            "Axin2 → destruction_complex_reformed（Axin2 蛋白作为 scaffold 重建 "
            "destruction complex 降解 β-catenin 执行负反馈, Axin2 与 Axin1 功能同源, "
            "delay=0min 蛋白结合直接负反馈）"
        ),
        "source_pmid": _PMID_POLAKIS_2002,
    },
    # 3. LRP6 phosphorylation → destruction_complex_disrupted（positive feedback, Wnt 信号维持）
    #    pLRP6 持续招募 Axin 维持 destruction complex 解离，正反馈维持 Wnt 信号
    #    （Wnt 信号自身的放大环路，确保 β-catenin 持续累积）
    {
        "id": "FL_PLRP6_DC_DISRUPTED",
        "loop_type": "positive",
        "node_ids": [
            "pLRP6",
            "Axin_recruited",
            "destruction_complex_disrupted",
            "bCatenin",
        ],
        "delay_minutes": 0.0,   # 信号传导，无转录延迟
        "bistable": False,
        "description": (
            "LRP6 phosphorylation → destruction_complex_disrupted（pLRP6 持续招募 "
            "Axin 维持 destruction complex 解离, 正反馈维持 Wnt 信号, "
            "确保 β-catenin 持续累积, delay=0min 信号传导直接正反馈）"
        ),
        "source_pmid": _PMID_POLAKIS_2002,
    },
]


# =============================================================================
# Wnt Crosstalk Reaction 片段（5 条，仅本通路侧片段，edge 由 Coordinator 管理）
# =============================================================================
# β-catenin 作为转录因子激活下游促增殖基因（Cyclin D1/cMyc）
# pAKT→GSK3β / pERK→β-catenin 由其他 Specialist 生成相反方向片段，
# 本 Specialist 仅返回本通路侧消费片段
_WNT_CROSSTALK_REACTIONS: list[dict[str, Any]] = [
    # 1. β-catenin → Cyclin D1（transcription, Wnt 激活 Cyclin D1 推动细胞周期）
    #    bCatenin_nuclear-TCF/LEF 转录激活 Cyclin D1（CCND1），推动 G1/S 转换
    {
        "source": "bCatenin_nuclear",
        "target": "Cyclin_D1",
        "mechanism": "transcription",
        "shared_species": ["bCatenin"],
        "site": "TCF/LEF site(Cyclin D1/CCND1 promoter)",
        "description": "β-catenin 转录激活 Cyclin D1（bCatenin_nuclear-TCF/LEF 作转录因子, Cyclin D1 推动 G1/S 转换, Wnt 促增殖机制, 与 Cell Cycle 通路 cross-talk）",
    },
    # 2. β-catenin → cMyc（transcription, Wnt 激活 cMyc）
    #    bCatenin_nuclear-TCF/LEF 转录激活 c-Myc，促增殖与代谢重编程
    {
        "source": "bCatenin_nuclear",
        "target": "cMyc",
        "mechanism": "transcription",
        "shared_species": ["bCatenin"],
        "site": "TCF/LEF site(c-Myc/MYC promoter)",
        "description": "β-catenin 转录激活 cMyc（bCatenin_nuclear-TCF/LEF 作转录因子, c-Myc 促增殖与代谢重编程, Wnt 经典靶基因, 与 Cell Cycle 通路 cross-talk）",
    },
    # 3. pAKT → GSK3β（inhibition, pAKT 抑制 GSK3β 阻止 β-catenin 降解）
    #    pAKT 磷酸化 GSK3β Ser9 抑制其活性，阻止 destruction complex 磷酸化 β-catenin
    #    （PI3K-AKT→Wnt cross-talk, pAKT 稳定 β-catenin 增强 Wnt 信号）
    {
        "source": "pAKT",
        "target": "GSK3b",
        "mechanism": "inhibition",
        "shared_species": ["GSK3b"],
        "site": "Ser9",
        "description": "pAKT 抑制 GSK3β（pAKT 磷酸化 GSK3β Ser9 抑制其活性, 阻止 destruction complex 磷酸化 β-catenin, 稳定 β-catenin 增强 Wnt 信号, PI3K-AKT→Wnt cross-talk）",
    },
    # 4. pERK → β-catenin（activation, MAPK 磷酸化 β-catenin 增强稳定性）
    #    pERK 磷酸化 β-catenin Ser612 增强其稳定性与转录活性
    #    （MAPK→Wnt cross-talk, MAPK 增强 Wnt 信号）
    {
        "source": "pERK",
        "target": "bCatenin",
        "mechanism": "activation",
        "shared_species": ["bCatenin"],
        "site": "Ser612",
        "description": "pERK 激活 β-catenin（pERK 磷酸化 β-catenin Ser612 增强其稳定性与转录活性, MAPK 增强 Wnt 信号, MAPK→Wnt cross-talk）",
    },
    # 5. Wnt → Ca2+/PKC（alternative pathway, non-canonical Wnt）
    #    Wnt 通过 Frizzled 激活异三聚体 G 蛋白 → PLC → DAG/IP3 → Ca2+/PKC
    #    （non-canonical Wnt/Ca2+ 通路，不依赖 β-catenin）
    {
        "source": "Wnt",
        "target": "Ca2_PKC",
        "mechanism": "activation",
        "shared_species": [],
        "description": "Wnt → Ca2+/PKC（non-canonical Wnt/Ca2+ 通路, Wnt 通过 Frizzled 激活异三聚体 G 蛋白→PLC→DAG/IP3→Ca2+/PKC, 不依赖 β-catenin, alternative pathway）",
    },
]


# =============================================================================
# Wnt 扰动（6 个：4 个药物 + 2 个突变）
# =============================================================================
_WNT_PERTURBATIONS: list[dict[str, Any]] = [
    # 1. ICG-001（CBP/β-catenin interaction inhibitor）
    #    ICG-001 是 CBP/β-catenin 相互作用抑制剂，阻断 β-catenin-CBP 转录激活
    #    （选择性抑制 Wnt 转录活性，不破坏 β-catenin 累积）
    {
        "target": "bCatenin",
        "drug": "ICG-001",
        "mechanism": "inhibition",
        "ko_target": None,
        "description": "ICG-001（CBP/β-catenin 相互作用抑制剂, 小分子, 阻断 β-catenin-CBP 转录激活, 选择性抑制 Wnt 转录活性）",
    },
    # 2. XAV939（tankyrase inhibitor, 稳定 Axin 促进 β-catenin 降解）
    #    XAV939 是 tankyrase (TNKS) 抑制剂，稳定 Axin1/2 蛋白，
    #    增强 destruction complex 活性，促进 β-catenin 降解
    {
        "target": "Axin",
        "drug": "XAV939",
        "mechanism": "inhibition",
        "ko_target": None,
        "description": "XAV939（tankyrase TNKS 抑制剂, 小分子, 稳定 Axin1/2 蛋白, 增强 destruction complex 活性, 促进 β-catenin 降解）",
    },
    # 3. LGK974（PORCN inhibitor, 阻止 Wnt 分泌）
    #    LGK974 是 porcupine (PORCN) 抑制剂，阻止 Wnt 配体棕榈酰化与分泌
    #    （阻断所有 Wnt 配体分泌，全面抑制 Wnt 通路）
    {
        "target": "Wnt",
        "drug": "LGK974",
        "mechanism": "inhibition",
        "ko_target": None,
        "description": "LGK974（porcupine PORCN 抑制剂, 小分子, 阻止 Wnt 配体棕榈酰化与分泌, 阻断所有 Wnt 配体分泌, 全面抑制 Wnt 通路）",
    },
    # 4. Vantictumab（anti-Frizzled antibody, 阻断受体）
    #    Vantictumab 是抗 Frizzled 受体单克隆抗体，阻断 Wnt-Frizzled 结合
    #    （阻断多个 Frizzled 亚型，广谱 Wnt 抑制剂）
    {
        "target": "Frizzled",
        "drug": "Vantictumab",
        "mechanism": "inhibition",
        "ko_target": None,
        "description": "Vantictumab（抗 Frizzled 单克隆抗体, 阻断 Wnt-Frizzled 结合, 阻断多个 Frizzled 亚型, 广谱 Wnt 抑制剂）",
    },
    # 5. APC loss（loss-of-function mutation, 结肠癌常见, β-catenin 累积）
    #    APC 基因 loss-of-function 突变导致 destruction complex 失活，
    #    β-catenin 持续累积入核转录（结肠癌常见突变，>80% 散发性结肠癌）
    {
        "target": "APC",
        "drug": None,
        "mechanism": "knockout",
        "ko_target": "APC_loss",
        "description": "APC loss（loss-of-function mutation, destruction complex 失活, β-catenin 持续累积入核转录, 结肠癌常见突变, >80% 散发性结肠癌）",
    },
    # 6. CTNNB1 S45F（β-catenin mutation, 阻止磷酸化降解）
    #    CTNNB1 (β-catenin) S45F 突变丢失 CK1 α Thr41 priming 位点，
    #    阻止 GSK3β 磷酸化 Ser33/37/Thr41，β-catenin 逃逸降解（肝癌常见）
    {
        "target": "bCatenin",
        "drug": None,
        "mechanism": "activation",
        "ko_target": "CTNNB1_S45F",
        "description": "CTNNB1 S45F（β-catenin 突变, 丢失 CK1 α Thr41 priming 位点, 阻止 GSK3β 磷酸化 Ser33/37/Thr41, β-catenin 逃逸降解, 肝癌常见突变）",
    },
]


# =============================================================================
# Wnt Validation 规则（3 条 benchmark, Polakis 2002, PMID:12064617）
# =============================================================================
# 每条含：rule_id / metric_name / expected / tolerance / pmid / description
# expected 取区间中点，tolerance 取区间半宽
_WNT_VALIDATION_RULES: list[dict[str, Any]] = [
    # 1. β-catenin 稳态（无 Wnt）<10 nM（Polakis 2002, PMID:12064617）
    #    无 Wnt 信号时 destruction complex 持续降解 β-catenin 维持稳态 <10 nM
    {
        "rule_id": "VAL_WNT_BCAT_STEADY_STATE_NO_WNT",
        "metric_name": "bcat_steady_state_no_wnt",
        "expected": 5.0,    # (0.0 + 10.0) / 2
        "tolerance": 5.0,   # (10.0 - 0.0) / 2
        "expected_min": 0.0,
        "expected_max": 10.0,
        "unit": "nM",
        "pmid": _PMID_POLAKIS_2002,
        "comparison": "range",
        "pathway_tag": PATHWAY_TAG,
        "description": "β-catenin 稳态（无 Wnt）<10 nM（Polakis 2002 Wnt signaling, 无 Wnt 信号时 destruction complex 持续降解 β-catenin 维持低稳态）",
    },
    # 2. Axin2 mRNA 达峰时间 60-120 minutes（Polakis 2002, PMID:12064617）
    #    β-catenin 转录激活 Axin2 mRNA 后 1-2h 达峰（含 30 min 转录延迟）
    {
        "rule_id": "VAL_WNT_AXIN2_MRNA_PEAK_TIME",
        "metric_name": "Axin2_mRNA_peak_time",
        "expected": 90.0,    # (60.0 + 120.0) / 2
        "tolerance": 30.0,   # (120.0 - 60.0) / 2
        "expected_min": 60.0,
        "expected_max": 120.0,
        "unit": "minutes",
        "pmid": _PMID_POLAKIS_2002,
        "comparison": "range",
        "pathway_tag": PATHWAY_TAG,
        "description": "Axin2 mRNA 达峰时间 60-120 min（Polakis 2002 Wnt signaling, β-catenin 转录激活 Axin2 mRNA 后 1-2h 达峰, 含 30 min 转录延迟）",
    },
    # 3. Destruction Complex 完整性（无 Wnt 时完整组装）
    #    无 Wnt 信号时 destruction complex 应完整组装
    #    （Axin+APC+GSK3β+CK1+β-catenin 全部组分存在）
    {
        "rule_id": "VAL_WNT_DESTRUCTION_COMPLEX_ASSEMBLY",
        "metric_name": "destruction_complex_assembly",
        "expected": True,
        "tolerance": 0.0,   # 布尔值无容忍度
        "expected_min": None,
        "expected_max": None,
        "unit": "boolean",
        "pmid": _PMID_POLAKIS_2002,
        "comparison": "exact",
        "pathway_tag": PATHWAY_TAG,
        "description": "Destruction Complex 完整性（无 Wnt 时 destruction complex 应完整组装 Axin+APC+GSK3β+CK1+β-catenin, Polakis 2002 Wnt signaling）",
    },
]


@register_specialist
class WntSpecialist(PathwaySpecialistBase):
    """Wnt / β-catenin 通路 Specialist。

    实现 Off 状态 destruction complex（Axin+APC+GSK3β+CK1→β-catenin 磷酸化→
    泛素化→降解）+ On 状态（Wnt+Frizzled+LRP5/6→Dvl→LRP6 磷酸化→Axin 招募→
    destruction complex 解离→β-catenin 累积→入核→TCF/LEF 转录）核心拓扑 +
    β-catenin→Axin2 转录延迟负反馈（delay=30min）的 Core/Feedback/Crosstalk/
    Perturbation/Validation 5 模块，输出通路特异 Reaction IR 片段 + Destruction
    Complex 五步耦合 CompositeReaction（binding→binding→phosphorylation→
    ubiquitination→proteasomal_degradation）+ 模板选择 + Validation 规则。

    职责边界：
    - 处理 Wnt 通路核心（Wnt/Frizzled/LRP5_6/Dvl/Axin/APC/GSK3β/CK1/β-catenin/TCF_LEF/Axin2）
    - 不处理 β-catenin→Cyclin D1 细胞周期下游（由 Cell Cycle Specialist 处理，Task 4.6）
    - 不生成 PI3K-AKT→GSK3β cross-talk edge（由 Cross-talk Coordinator 处理，Task 4.13）

    输入：
    - ``pathway_graph``：v4_pathway_graph 的 Wnt 子图（含 CompositeReaction
      CR_DESTRUCTION_COMPLEX）
    - ``ontology_entities``：P1 Ontology Entities（可选，HGNC/UniProt ID 对齐）

    输出：
    - ``apply_core``：17 条核心 Reaction IR 片段 + 28 物种 + Destruction Complex
      五步耦合 CompositeReaction（binding→binding→phosphorylation→ubiquitination→
      proteasomal_degradation）（bCatenin 与 GSK3b 物种标记 shared=True，
      分别与 Cell Cycle Specialist 的 Cyclin D1 路径与 PI3K-AKT Specialist
      的 pAKT→GSK3β 路径共享）
    - ``apply_feedback``：3 条 FeedbackLoop
      （FL_BCAT_AXIN2 delay=30min 转录延迟负反馈 / FL_AXIN2_DC_REFORMED
      Axin2 重建 destruction complex / FL_PLRP6_DC_DISRUPTED LRP6 磷酸化
      维持 destruction complex 解离正反馈）
    - ``apply_crosstalk``：5 条 cross-talk Reaction 片段
      （β-catenin→Cyclin D1/cMyc + pAKT→GSK3β + pERK→β-catenin + Wnt→Ca2+/PKC）
    - ``apply_perturbation``：6 个扰动
      （ICG-001/XAV939/LGK974/Vantictumab/APC loss/CTNNB1 S45F）
    - ``apply_validation``：3 条 Validation benchmark
      （β-catenin 稳态 <10 nM / Axin2 mRNA 1-2h 达峰 / destruction complex 完整性,
      Polakis 2002, PMID:12064617）
    """

    pathway_class: str = "WNT"
    display_name: str = "Wnt / β-catenin Signaling"

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
                    species=list(_WNT_CORE_SPECIES),
                    reactions=list(_WNT_CORE_REACTIONS),
                    kinetics_overrides={},
                )
            if module_name == MODULE_FEEDBACK:
                return FeedbackModuleData(
                    feedback_loops=list(_WNT_FEEDBACK_LOOPS),
                    delay_minutes=_WNT_BCAT_AXIN2_DELAY_MINUTES,
                    loop_type="negative",   # β-catenin→Axin2 负反馈为主
                )
            if module_name == MODULE_CROSSTALK:
                return CrosstalkModuleData(
                    crosstalk_reactions=list(_WNT_CROSSTALK_REACTIONS),
                    shared_species=["bCatenin", "GSK3b"],
                    coordination_strategy="merge",
                )
            if module_name == MODULE_PERTURBATION:
                return PerturbationModuleData(
                    perturbation_reactions=list(_WNT_PERTURBATIONS),
                    drug_targets=[
                        p for p in _WNT_PERTURBATIONS if p.get("drug")
                    ],
                    ko_targets=[
                        p for p in _WNT_PERTURBATIONS if p.get("ko_target")
                    ],
                )
            if module_name == MODULE_VALIDATION:
                return ValidationModuleData(
                    rules=list(_WNT_VALIDATION_RULES),
                    benchmarks=[
                        {
                            "benchmark_name": r["metric_name"],
                            "value": r["expected"],
                            "unit": r["unit"],
                            "condition": "Wnt stimulation",
                            "reference": r["pmid"],
                        }
                        for r in _WNT_VALIDATION_RULES
                    ],
                    tolerances={
                        r["metric_name"]: r["tolerance"]
                        for r in _WNT_VALIDATION_RULES
                    },
                    pmid_references=[
                        {
                            "pmid": r["pmid"],
                            "citation": r["description"],
                            "pathway_class": PATHWAY_TAG,
                            "metric_name": r["metric_name"],
                        }
                        for r in _WNT_VALIDATION_RULES
                        if r["pmid"]
                    ],
                )
            logger.warning(
                "WntSpecialist.load_module: 未知模块名 '%s'", module_name,
            )
            return None
        except Exception as exc:
            logger.warning(
                "WntSpecialist.load_module 加载模块 '%s' 失败: %s",
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
        """应用核心模块，返回 Wnt 通路核心 Reaction IR 片段。

        输出 17 条核心反应：
        1. Axin → Axin_APC（complex_formation, destruction complex 第 1 步 binding）
        2. Axin_APC → Axin_APC_GSK3b（complex_formation, 第 2 步 binding）
        3. Axin_APC_GSK3b → Axin_APC_GSK3b_bcat（complex_formation, 招募 β-catenin）
        4. Axin_APC_GSK3b_bcat → p_bcat（phosphorylation, 五步 step 3）
        5. p_bcat → ub_bcat（ubiquitination, 五步 step 4）
        6. ub_bcat → bcat_degraded（proteasomal_degradation, 五步 step 5）
        7. Wnt → Wnt_Fz_LRP_complex（complex_formation, 受体激活）
        8. Wnt_Fz_LRP_complex → pDvl（activation, Dvl 磷酸化激活）
        9. pDvl → pLRP6（phosphorylation, LRP6 磷酸化, 异磷酸化）
        10. pLRP6 → Axin_recruited（translocation, Axin 被招募到膜）
        11. Axin_recruited → destruction_complex_disrupted（dissociation）
        12. bCatenin → bCatenin_nuclear（nuclear_import, β-catenin 入核）
        13. bCatenin_nuclear → TCF_LEF_bcat_complex（complex_formation, 转录复合物）
        14. TCF_LEF_bcat_complex → Axin2_mRNA（transcription, Hill, Axin2 负反馈）
        15. Axin2_mRNA → Axin2（translation, Axin2 翻译）
        16. TCF_LEF_bcat_complex → Cyclin_D1_mRNA（transcription, cross-talk 到 Cell Cycle）
        17. TCF_LEF_bcat_complex → cMyc_mRNA（transcription, cross-talk 到 Cell Cycle）

        bCatenin 物种标记 shared=True（与 Cell Cycle Specialist 的 Cyclin D1 路径共享，
        下游 Cyclin D1→Cdk4 由 Cell Cycle Specialist 处理）。
        GSK3b 物种标记 shared=True（与 PI3K-AKT Specialist 的 pAKT→GSK3β 路径共享）。

        Destruction Complex 五步耦合 CompositeReaction 输出
        （binding→binding→phosphorylation→ubiquitination→proteasomal_degradation，
        sequential coupling）。

        Returns:
            dict 含 ``species``（28 物种）/ ``reactions``（17 反应） /
            ``composite_reactions``（Destruction Complex 五步耦合）字段。异常时返回
            ``{"species": [], "reactions": [], "composite_reactions": []}``。
        """
        try:
            return {
                "species": list(_WNT_CORE_SPECIES),
                "reactions": list(_WNT_CORE_REACTIONS),
                "composite_reactions": list(_WNT_COMPOSITE_REACTIONS),
                "pathway_tag": PATHWAY_TAG,
                "source_sbml": SOURCE_SBML,
            }
        except Exception as exc:
            logger.warning(
                "WntSpecialist.apply_core 失败: %s", exc
            )
            return {
                "species": [],
                "reactions": [],
                "composite_reactions": [],
            }

    # =================================================================
    # apply_feedback：FeedbackLoop 列表
    # =================================================================
    def apply_feedback(self, pathway_graph: dict) -> list[dict]:
        """应用反馈模块，返回 Wnt 通路 FeedbackLoop 列表。

        输出 3 条反馈环：
        1. β-catenin→Axin2 转录延迟负反馈（DDE delay=30min, Axin2 mRNA 1-2h 达峰）
        2. Axin2 → destruction_complex_reformed（negative feedback,
           Axin2 重建 destruction complex 降解 β-catenin）
        3. LRP6 phosphorylation → destruction_complex_disrupted
           （positive feedback, Wnt 信号维持）

        Returns:
            FeedbackLoop 字典列表（含 delay_minutes 标记）。异返回空列表。
        """
        try:
            return list(_WNT_FEEDBACK_LOOPS)
        except Exception as exc:
            logger.warning(
                "WntSpecialist.apply_feedback 失败: %s", exc
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
        """应用跨通路模块，返回 Wnt 通路侧 cross-talk Reaction 片段。

        注意：本方法仅生成本通路侧的 cross-talk Reaction 片段，
        cross-talk edge 本身由 Cross-talk Coordinator 创建（Task 4.13）。

        输出 5 条 cross-talk Reaction 片段：
        1. β-catenin → Cyclin D1（transcription, Wnt 激活 Cyclin D1 推动细胞周期）
        2. β-catenin → cMyc（transcription, Wnt 激活 cMyc）
        3. pAKT → GSK3β（inhibition, pAKT 抑制 GSK3β 阻止 β-catenin 降解）
        4. pERK → β-catenin（activation, MAPK 磷酸化 β-catenin 增强稳定性）
        5. Wnt → Ca2+/PKC（activation, non-canonical Wnt/Ca2+ 通路）

        Args:
            pathway_graph: 通路图（当前未使用，保留接口一致性）。
            crosstalk_edges: 来自 Coordinator 的 cross-talk edge 列表
                （当前未使用，本 Specialist 返回静态拓扑片段）。

        Returns:
            cross-talk Reaction 片段列表。异返回空列表。
        """
        try:
            return list(_WNT_CROSSTALK_REACTIONS)
        except Exception as exc:
            logger.warning(
                "WntSpecialist.apply_crosstalk 失败: %s", exc
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
        """应用扰动模块，返回 Wnt 通路特异药物 / 突变 Reaction 片段。

        输出 6 个扰动（4 个药物 + 2 个突变）：
        1. ICG-001（CBP/β-catenin 相互作用抑制剂, 阻断转录激活）
        2. XAV939（tankyrase 抑制剂, 稳定 Axin 促进 β-catenin 降解）
        3. LGK974（PORCN 抑制剂, 阻止 Wnt 分泌）
        4. Vantictumab（抗 Frizzled 单克隆抗体, 阻断受体）
        5. APC loss（loss-of-function, destruction complex 失活, 结肠癌常见）
        6. CTNNB1 S45F（β-catenin 突变, 阻止磷酸化降解, 肝癌常见）

        Args:
            pathway_graph: 通路图（当前未使用）。
            perturbation_points: 扰动点列表（当前未使用，本 Specialist
                返回静态拓扑片段）。

        Returns:
            扰动 Reaction 片段列表。异返回空列表。
        """
        try:
            return list(_WNT_PERTURBATIONS)
        except Exception as exc:
            logger.warning(
                "WntSpecialist.apply_perturbation 失败: %s", exc
            )
            return []

    # =================================================================
    # apply_validation：Validation 规则列表
    # =================================================================
    def apply_validation(
        self,
        simulation_result: dict | None = None,
    ) -> list[dict]:
        """应用验证模块，返回 Wnt 通路 Validation 规则列表。

        输出 3 条 benchmark（Polakis 2002, PMID:12064617）：
        1. β-catenin 稳态（无 Wnt）<10 nM（destruction complex 持续降解）
        2. Axin2 mRNA 达峰时间 60-120 min（β-catenin 转录激活, 含 30 min 延迟）
        3. Destruction Complex 完整性（无 Wnt 时完整组装 Axin+APC+GSK3β+CK1+β-catenin）

        Args:
            simulation_result: 仿真结果（可选，当前未使用，返回静态规则集）。

        Returns:
            Validation 规则列表。异返回空列表。
        """
        try:
            return list(_WNT_VALIDATION_RULES)
        except Exception as exc:
            logger.warning(
                "WntSpecialist.apply_validation 失败: %s", exc
            )
            return []

    # =================================================================
    # select_template：覆写以支持 bistable + phosphorylation 模式
    # =================================================================
    def select_template(self, mechanism: str) -> str:
        """根据 mechanism 选择 ODE 模板名（覆写支持双稳态与磷酸化）。

        默认映射（与 P3 ``ode_templates_v2/`` 下 .j2 文件对齐）：
        - ``bistable`` → ``bistable_switch``（β-catenin 累积 toggle）
        - ``phosphorylation`` → ``_mechanism_phosphorylation_mm``
          （Axin_APC_GSK3b_bcat→p_bcat / pDvl→pLRP6 异磷酸化）

        Args:
            mechanism: 机制名（小写，如 ``"bistable"`` / ``"phosphorylation"``）。

        Returns:
            ODE 模板名（不含 ``.j2`` 后缀）。未匹配时返回 ``"default"``
            （调用方应处理默认降级）。
        """
        # 双稳态模式：β-catenin 累积 toggle（Wnt on/off 切换）
        if mechanism == "bistable":
            return "bistable_switch"
        # 磷酸化场景：Axin_APC_GSK3b_bcat→p_bcat / pDvl→pLRP6 异磷酸化
        if mechanism == "phosphorylation":
            return "_mechanism_phosphorylation_mm"
        # 其他机制走默认基类映射
        return super().select_template(mechanism)


__all__ = [
    "WntSpecialist",
    "PATHWAY_TAG",
    "SOURCE_SBML",
]
