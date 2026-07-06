# BioDynamics Agent v4 - Cell Cycle Specialist (Phase 4 / Task 4.8)
# Cell Cycle 通路 Specialist：实现 Cyclin D/E/A/B-CDK 级联（G1→G1/S→S→G2→M）+
# APC/C-Cdc20 降解 Cyclin B / Securin + Rb-E2F G1/S toggle（bistable）+
# CyclinB-APC/C 延迟负反馈振荡器（delay=30min）+ p53→p21 抑制 CDK2/4。
#
# 设计原则（铁律）：
# 1. Feature Flag V4_PATHWAY_SPECIALIST_ENABLED=false 时由 LangGraph hook 层短路，
#    本 Specialist 代码不被调用（基类本身不做 flag 检查）
# 2. 使用 Task 4.0 修复后的 PHOSPHORYLATION 语义：
#    - 异磷酸化（CyclinD_CDK4 → pRb_phosphorylated）：未磷酸化 Rb 作 substrate，
#      磷酸化 pRb_phosphorylated 作 product，CyclinD_CDK4 作 catalytic modifier
#    - 异磷酸化（CyclinE_CDK2 → E2F_active）：E2F 作 substrate，
#      E2F_active 作 product，CyclinE_CDK2 作 catalytic modifier
# 3. 不处理 p53→p21 上游（由 p53 Specialist 处理，Task 4.6）
# 4. 不生成 ERK→Cyclin D cross-talk edge 本身（由 Cross-talk Coordinator 处理，
#    Task 4.13），仅返回本通路侧的 cross-talk Reaction 片段
# 5. apply_* 方法实现捕获异常并返回空 list/dict，记录 logger.warning，不抛异常
#
# 依赖：
# - P1 ontology / pathway_registry（Cell_Cycle 通路类别键）
# - P2 schema（SpeciesV2 / ReactionV2 / SpeciesRef / Modifier / CompositeReaction）
# - P2 MechanismType（PHOSPHORYLATION / COMPLEX_FORMATION / DISSOCIATION /
#   TRANSCRIPTION / TRANSLATION / ACTIVATION / PROTEASOMAL_DEGRADATION）
# - P3 ode_templates_v2（oscillatory_feedback.j2 / bistable_switch.j2 模板）
# - P3 bistability_detector.py（Rb-E2F toggle bistable 检测）
# - P3 pathway_graph/initializer.py（Cell_Cycle core_nodes / core_edges）
#
# 参考：
# - spec.md Part 3 Specialist 6（第 227-232 行）
# - tasks.md Task 4.8（第 100-109 行）
# - Pomerening 2005 Cdk1 oscillation (PMID:11389814)
# - Yao 2008 bistable Rb-E2F switch (PMID:12064617)
# - BioModels BIOMD0000000055 (Tyson1991 cell cycle)

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
# Cell Cycle 通路标签
# =============================================================================
PATHWAY_TAG: str = "CELL_CYCLE"

# SBML BioModels ID（Tyson1991 cell cycle model）
SOURCE_SBML: str = "BIOMD0000000055"

# Validation benchmark PMID 引用
_PMID_POMERENING_2005: str = "PMID:11389814"   # Pomerening 2005 Cdk1 oscillation
_PMID_YAO_2008: str = "PMID:12064617"          # Yao 2008 bistable Rb-E2F switch

# CyclinB-APC/C 延迟负反馈振荡器延迟（DDE，分钟）
# Pomerening 2005 模型中 CyclinB-CDK1 激活 APC/C-Cdc20，APC/C 降解 Cyclin B，
# 形成 delay=30min 的延迟负反馈振荡，产生 8-12 小时周期细胞周期振荡。
_CYCLINB_APC_DELAY_MINUTES: float = 30.0


# =============================================================================
# Cell Cycle 核心物种（与 P3 pathway_graph/initializer.py Cell_Cycle.core_nodes
# 对齐，扩展 Cyclin_A/B + CDK1 + APC/C-Cdc20 + Securin 完整细胞周期拓扑）
# =============================================================================
# p21 物种标记 shared=True（与 p53 Specialist 的 p53→p21 转录路径共享，
# p21 抑制 CyclinE-CDK2 / CyclinD-CDK4 阻滞 G1/S 转换）
_CELL_CYCLE_CORE_SPECIES: list[dict[str, Any]] = [
    # ---- G1 phase 物种 ----
    # Cyclin D（G1 早期 D 型 cyclin，外源信号诱导）
    {"name": "Cyclin_D", "species_type": "protein",
     "compartment": "nucleus"},
    # CDK4/6（G1 期 CDK，与 Cyclin D 形成 CyclinD-CDK4/6 复合物）
    # 注：以 CDK4 代表 CDK4/6 家族（Palbociclib 等 CDK4/6 抑制剂作用靶点）
    {"name": "CDK4", "species_type": "protein",
     "compartment": "nucleus"},
    # CyclinD-CDK4 复合物（G1 期活性形式，磷酸化 Rb）
    {"name": "CyclinD_CDK4", "species_type": "complex",
     "compartment": "nucleus"},
    # Rb（视网膜母细胞瘤蛋白，未磷酸化为抑制形式，结合抑制 E2F）
    {"name": "Rb", "species_type": "protein",
     "compartment": "nucleus"},
    # pRb_phosphorylated（磷酸化 Rb，释放 E2F，G1/S 转换）
    {"name": "pRb_phosphorylated", "species_type": "protein",
     "compartment": "nucleus"},
    # E2F（转录因子，未磷酸化 Rb 结合时为抑制形式）
    {"name": "E2F", "species_type": "protein",
     "compartment": "nucleus"},
    # E2F_free（Rb 磷酸化后释放的游离 E2F，激活 Cyclin E 转录）
    {"name": "E2F_free", "species_type": "protein",
     "compartment": "nucleus"},
    # ---- G1/S transition 物种 ----
    # Cyclin E（G1/S 转换 cyclin，E2F 转录靶基因）
    {"name": "Cyclin_E", "species_type": "protein",
     "compartment": "nucleus"},
    # Cyclin_E_mRNA（E2F 转录激活产物）
    {"name": "Cyclin_E_mRNA", "species_type": "mrna",
     "compartment": "nucleus"},
    # CDK2（S 期 CDK，与 Cyclin E / Cyclin A 形成复合物）
    {"name": "CDK2", "species_type": "protein",
     "compartment": "nucleus"},
    # CyclinE-CDK2 复合物（G1/S 转换活性形式，正反馈磷酸化 Rb）
    {"name": "CyclinE_CDK2", "species_type": "complex",
     "compartment": "nucleus"},
    # E2F_active（CyclinE-CDK2 进一步磷酸化激活 E2F，正反馈）
    {"name": "E2F_active", "species_type": "protein",
     "compartment": "nucleus"},
    # ---- S/G2 phase 物种 ----
    # Cyclin A（S 期 + G2 期 cyclin）
    {"name": "Cyclin_A", "species_type": "protein",
     "compartment": "nucleus"},
    # CyclinA-CDK2 复合物（S 期活性形式，驱动 DNA 复制）
    {"name": "CyclinA_CDK2", "species_type": "complex",
     "compartment": "nucleus"},
    # CDK1（M 期 CDK，与 Cyclin A / Cyclin B 形成复合物）
    {"name": "CDK1", "species_type": "protein",
     "compartment": "nucleus"},
    # CyclinA-CDK1 复合物（G2 期活性形式，驱动 G2/M 转换）
    {"name": "CyclinA_CDK1", "species_type": "complex",
     "compartment": "nucleus"},
    # ---- M phase 物种 ----
    # Cyclin B（M 期 cyclin，mitotic cyclin）
    {"name": "Cyclin_B", "species_type": "protein",
     "compartment": "nucleus"},
    # CyclinB-CDK1 复合物（M 期活性形式，MPF，驱动有丝分裂进入）
    {"name": "CyclinB_CDK1", "species_type": "complex",
     "compartment": "nucleus"},
    # APC/C-Cdc20 active（后期促进复合物 APC/C 与 Cdc20 形成活性 E3 泛素连接酶）
    {"name": "APC_C_Cdc20_active", "species_type": "complex",
     "compartment": "nucleus"},
    # CyclinB_degraded（APC/C 降解 Cyclin B 的产物，后期退出有丝分裂）
    {"name": "CyclinB_degraded", "species_type": "protein",
     "compartment": "nucleus"},
    # Securin_degraded（APC/C 降解 Securin 释放 Separase，启动姊妹染色单体分离）
    {"name": "Securin_degraded", "species_type": "protein",
     "compartment": "nucleus"},
    # ---- 共享物种 ----
    # p21（CDK 抑制剂，shared：与 p53 Specialist 的 p53→p21 转录路径共享）
    # p21 抑制 CyclinE-CDK2 / CyclinD-CDK4 阻滞 G1/S 转换
    {"name": "p21", "species_type": "protein",
     "compartment": "nucleus", "shared": True},
]


# =============================================================================
# Cell Cycle 核心反应（14 条：Cyclin-CDK 级联 11 + Rb-E2F toggle 3）
# =============================================================================
# 每条反应含：source / target / mechanism / kinetics_type / pathway_tag /
#             substrate / product / modifier / modifier_type / autophosphorylation
#
# kinetics_type 选择：
# - complex_formation / dissociation / translation / proteasomal_degradation → mass_action
# - phosphorylation → Michaelis_Menten（与 P3 _mechanism_phosphorylation_mm 模板对齐）
# - transcription → Hill（E2F 作为转录因子，Hill 动力学 n=2 协同结合）
# - activation → hybrid（变构调控 / 复合调控，CyclinB-CDK1 激活 APC/C-Cdc20）
_CELL_CYCLE_CORE_REACTIONS: list[dict[str, Any]] = [
    # ===== Cyclin-CDK 级联（11 条：4 phase complex + Rb 磷酸化 + E2F 释放 + APC/C）=====
    # 1. Cyclin_D + CDK4 → CyclinD_CDK4（complex_formation, G1 phase）
    #    Cyclin D 与 CDK4/6 结合形成 G1 期活性复合物
    {
        "source": "Cyclin_D",
        "target": "CyclinD_CDK4",
        "mechanism": "complex_formation",
        "kinetics_type": "mass_action",
        "pathway_tag": PATHWAY_TAG,
        "substrate": "CDK4",
        "product": "CyclinD_CDK4",
        "modifier": "Cyclin_D",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "Cyclin D + CDK4/6 → CyclinD_CDK4（complex_formation, G1 phase, D 型 cyclin 与 CDK4/6 结合）",
    },
    # 2. CyclinD_CDK4 → pRb_phosphorylated（异磷酸化, Rb 作 substrate, CyclinD_CDK4 作 modifier）
    #    CyclinD-CDK4 异磷酸化 Rb（初始磷酸化，启动 Rb 释放 E2F）
    {
        "source": "CyclinD_CDK4",
        "target": "pRb_phosphorylated",
        "mechanism": "phosphorylation",
        "kinetics_type": "Michaelis_Menten",
        "pathway_tag": PATHWAY_TAG,
        # 异磷酸化：Rb 作 substrate，pRb_phosphorylated 作 product，CyclinD_CDK4 作 catalytic modifier
        "substrate": "Rb",
        "product": "pRb_phosphorylated",
        "modifier": "CyclinD_CDK4",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "CyclinD-CDK4 异磷酸化 Rb（Rb 作 substrate, CyclinD_CDK4 作 catalytic modifier, G1 期初始磷酸化释放 E2F）",
    },
    # 3. pRb_phosphorylated → E2F_free（dissociation, Rb 释放 E2F）
    #    Rb 磷酸化后构象改变，释放游离 E2F 转录因子（无 modifier，自发解离）
    {
        "source": "pRb_phosphorylated",
        "target": "E2F_free",
        "mechanism": "dissociation",
        "kinetics_type": "mass_action",
        "pathway_tag": PATHWAY_TAG,
        "substrate": "pRb_phosphorylated",
        "product": "E2F_free",
        "modifier": None,
        "modifier_type": None,   # 自发解离，无催化 modifier
        "autophosphorylation": False,
        "description": "pRb_phosphorylated 释放 E2F（dissociation, Rb 磷酸化后构象改变释放游离 E2F 转录因子, 无 modifier 自发解离）",
    },
    # 4. Cyclin_E + CDK2 → CyclinE_CDK2（complex_formation, G1/S transition）
    #    Cyclin E 与 CDK2 结合形成 G1/S 转换活性复合物
    {
        "source": "Cyclin_E",
        "target": "CyclinE_CDK2",
        "mechanism": "complex_formation",
        "kinetics_type": "mass_action",
        "pathway_tag": PATHWAY_TAG,
        "substrate": "CDK2",
        "product": "CyclinE_CDK2",
        "modifier": "Cyclin_E",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "Cyclin E + CDK2 → CyclinE_CDK2（complex_formation, G1/S transition, E 型 cyclin 与 CDK2 结合）",
    },
    # 5. CyclinE_CDK2 → E2F_active（异磷酸化, E2F 正反馈激活）
    #    CyclinE-CDK2 进一步磷酸化 E2F 激活其转录活性（正反馈）
    {
        "source": "CyclinE_CDK2",
        "target": "E2F_active",
        "mechanism": "phosphorylation",
        "kinetics_type": "Michaelis_Menten",
        "pathway_tag": PATHWAY_TAG,
        # 异磷酸化：E2F 作 substrate，E2F_active 作 product，CyclinE_CDK2 作 catalytic modifier
        "substrate": "E2F_free",
        "product": "E2F_active",
        "modifier": "CyclinE_CDK2",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "CyclinE-CDK2 异磷酸化激活 E2F（E2F_free 作 substrate, E2F_active 作 product, CyclinE_CDK2 作 modifier, 正反馈激活）",
    },
    # 6. Cyclin_A + CDK2 → CyclinA_CDK2（complex_formation, S phase）
    #    Cyclin A 与 CDK2 结合形成 S 期活性复合物（驱动 DNA 复制）
    {
        "source": "Cyclin_A",
        "target": "CyclinA_CDK2",
        "mechanism": "complex_formation",
        "kinetics_type": "mass_action",
        "pathway_tag": PATHWAY_TAG,
        "substrate": "CDK2",
        "product": "CyclinA_CDK2",
        "modifier": "Cyclin_A",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "Cyclin A + CDK2 → CyclinA_CDK2（complex_formation, S phase, A 型 cyclin 与 CDK2 结合驱动 DNA 复制）",
    },
    # 7. Cyclin_A + CDK1 → CyclinA_CDK1（complex_formation, G2 phase）
    #    Cyclin A 与 CDK1 结合形成 G2 期活性复合物（驱动 G2/M 转换准备）
    {
        "source": "Cyclin_A",
        "target": "CyclinA_CDK1",
        "mechanism": "complex_formation",
        "kinetics_type": "mass_action",
        "pathway_tag": PATHWAY_TAG,
        "substrate": "CDK1",
        "product": "CyclinA_CDK1",
        "modifier": "Cyclin_A",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "Cyclin A + CDK1 → CyclinA_CDK1（complex_formation, G2 phase, A 型 cyclin 与 CDK1 结合驱动 G2/M 转换）",
    },
    # 8. Cyclin_B + CDK1 → CyclinB_CDK1（complex_formation, M phase entry）
    #    Cyclin B 与 CDK1 结合形成 M 期活性复合物 MPF（M-phase promoting factor）
    {
        "source": "Cyclin_B",
        "target": "CyclinB_CDK1",
        "mechanism": "complex_formation",
        "kinetics_type": "mass_action",
        "pathway_tag": PATHWAY_TAG,
        "substrate": "CDK1",
        "product": "CyclinB_CDK1",
        "modifier": "Cyclin_B",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "Cyclin B + CDK1 → CyclinB_CDK1（complex_formation, M phase entry, MPF 形成驱动有丝分裂进入）",
    },
    # 9. CyclinB_CDK1 → APC_C_Cdc20_active（activation, CyclinB-CDK1 激活 APC/C-Cdc20）
    #    CyclinB-CDK1 (MPF) 磷酸化激活 APC/C-Cdc20（后期促进复合物）
    {
        "source": "CyclinB_CDK1",
        "target": "APC_C_Cdc20_active",
        "mechanism": "activation",
        "kinetics_type": "hybrid",
        "pathway_tag": PATHWAY_TAG,
        "substrate": "APC_C_Cdc20",
        "product": "APC_C_Cdc20_active",
        "modifier": "CyclinB_CDK1",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "CyclinB-CDK1 激活 APC/C-Cdc20（activation, MPF 磷酸化激活后期促进复合物, 启动后期）",
    },
    # 10. APC_C_Cdc20_active → CyclinB_degraded（proteasomal_degradation, APC/C 降解 Cyclin B）
    #     APC/C-Cdc20 多泛素化 Cyclin B 标记蛋白酶体降解（后期退出有丝分裂）
    {
        "source": "APC_C_Cdc20_active",
        "target": "CyclinB_degraded",
        "mechanism": "proteasomal_degradation",
        "kinetics_type": "mass_action",
        "pathway_tag": PATHWAY_TAG,
        "substrate": "Cyclin_B",
        "product": "CyclinB_degraded",
        "modifier": "APC_C_Cdc20_active",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "APC/C-Cdc20 降解 Cyclin B（proteasomal_degradation, 多泛素化标记蛋白酶体降解, 后期退出有丝分裂）",
    },
    # 11. APC_C_Cdc20_active → Securin_degraded（proteasomal_degradation, APC/C 降解 Securin 释放 Separase）
    #     APC/C-Cdc20 降解 Securin 释放 Separase，启动姊妹染色单体分离
    {
        "source": "APC_C_Cdc20_active",
        "target": "Securin_degraded",
        "mechanism": "proteasomal_degradation",
        "kinetics_type": "mass_action",
        "pathway_tag": PATHWAY_TAG,
        "substrate": "Securin",
        "product": "Securin_degraded",
        "modifier": "APC_C_Cdc20_active",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "APC/C-Cdc20 降解 Securin（proteasomal_degradation, 释放 Separase 启动姊妹染色单体分离）",
    },
    # ===== Rb-E2F toggle（3 条：E2F 转录 Cyclin E + 翻译 + 正反馈磷酸化 Rb）=====
    # 12. E2F_free → Cyclin_E_mRNA（transcription, Hill 动力学, E2F 激活 Cyclin E 转录正反馈）
    #     E2F 结合 Cyclin E 基因启动子，Hill 协同结合（n=2）激活转录
    {
        "source": "E2F_free",
        "target": "Cyclin_E_mRNA",
        "mechanism": "transcription",
        "kinetics_type": "Hill",
        "pathway_tag": PATHWAY_TAG,
        # E2F_free 作 modifier（转录因子），DNA 作 substrate，Cyclin_E_mRNA 作 product
        "substrate": "DNA",
        "product": "Cyclin_E_mRNA",
        "modifier": "E2F_free",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "hill_coefficient": 2,
        "description": "E2F_free 转录激活 Cyclin E mRNA（E2F_free 作转录因子, Hill n=2 协同结合, G1/S 转换正反馈）",
    },
    # 13. Cyclin_E_mRNA → Cyclin_E（translation, mRNA→protein）
    #     Cyclin E mRNA 在核糖体翻译为 Cyclin E 蛋白（用于形成 CyclinE-CDK2）
    {
        "source": "Cyclin_E_mRNA",
        "target": "Cyclin_E",
        "mechanism": "translation",
        "kinetics_type": "mass_action",
        "pathway_tag": PATHWAY_TAG,
        "substrate": "Cyclin_E_mRNA",
        "product": "Cyclin_E",
        "modifier": "Cyclin_E_mRNA",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "Cyclin E mRNA 翻译为 Cyclin E 蛋白（Cyclin_E_mRNA 作 substrate, Cyclin_E 作 product, 用于形成 CyclinE-CDK2）",
    },
    # 14. CyclinE_CDK2 → pRb_phosphorylated（positive feedback, CyclinE-CDK2 进一步磷酸化 Rb）
    #     CyclinE-CDK2 进一步磷酸化 Rb（正反馈，维持 E2F 释放，bistable toggle）
    {
        "source": "CyclinE_CDK2",
        "target": "pRb_phosphorylated",
        "mechanism": "phosphorylation",
        "kinetics_type": "Michaelis_Menten",
        "pathway_tag": PATHWAY_TAG,
        # 异磷酸化：Rb 作 substrate，pRb_phosphorylated 作 product，CyclinE_CDK2 作 catalytic modifier
        "substrate": "Rb",
        "product": "pRb_phosphorylated",
        "modifier": "CyclinE_CDK2",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "CyclinE-CDK2 进一步磷酸化 Rb（positive feedback, Rb 作 substrate, CyclinE_CDK2 作 modifier, 维持 E2F 释放 bistable toggle）",
    },
]


# =============================================================================
# Cell Cycle CompositeReaction（Rb-E2F G1/S toggle, bistable）
# =============================================================================
# Rb-E2F G1/S toggle 是细胞周期的 bistable 开关：
# - CyclinD-CDK4 初始磷酸化 Rb 释放 E2F
# - E2F 转录激活 Cyclin E → CyclinE-CDK2
# - CyclinE-CDK2 进一步磷酸化 Rb（正反馈），维持 E2F 释放
# - 形成 Rb(low)↔E2F(high) 的 bistable toggle，G1/S 转换 point-of-no-return
# Yao 2008 (PMID:12064617) 证明 Rb-E2F 系统呈现 bistable G1/S 转换开关行为
_CELL_CYCLE_COMPOSITE_REACTIONS: list[dict[str, Any]] = [
    {
        "id": "CR_RB_E2F_TOGGLE",
        "name": "Rb-E2F G1/S Bistable Toggle",
        "mechanism": "bistable",
        "template": "bistable_switch.j2",
        "loop_type": "positive",
        "point_of_no_return": True,
        "node_ids": [
            "CyclinD_CDK4",
            "pRb_phosphorylated",
            "E2F_free",
            "Cyclin_E_mRNA",
            "Cyclin_E",
            "CyclinE_CDK2",
        ],
        "reactions": [
            # CyclinD-CDK4 → pRb（初始磷酸化释放 E2F）
            "CyclinD_CDK4 → pRb_phosphorylated",
            # pRb → E2F_free（释放 E2F）
            "pRb_phosphorylated → E2F_free",
            # E2F → Cyclin_E_mRNA（E2F 转录激活 Cyclin E）
            "E2F_free → Cyclin_E_mRNA",
            # Cyclin_E_mRNA → Cyclin_E（翻译）
            "Cyclin_E_mRNA → Cyclin_E",
            # Cyclin_E → CyclinE_CDK2（复合物形成）
            "Cyclin_E → CyclinE_CDK2",
            # CyclinE-CDK2 → pRb（正反馈磷酸化 Rb，闭合 toggle）
            "CyclinE_CDK2 → pRb_phosphorylated",
        ],
        "description": (
            "Rb-E2F G1/S bistable toggle（CyclinD-CDK4→pRb→E2F→CyclinE→"
            "CyclinE-CDK2→pRb 正反馈环）：Rb 磷酸化后 E2F 释放，E2F 激活 "
            "CyclinE-CDK2 进一步磷酸化 Rb，正反馈维持 G1/S 转换，bistable "
            "point-of-no-return（Yao 2008, PMID:12064617）"
        ),
        "bistable_threshold": 0.3,   # Rb 磷酸化 > 30% 触发 G1/S 转换
        "delay_minutes": 0.0,        # Rb-E2F toggle 无转录延迟（蛋白级联）
        "pmid": _PMID_YAO_2008,
    },
]


# =============================================================================
# Cell Cycle 反馈环（4 条：CyclinB-APC/C 振荡 + p53→p21 + Rb-E2F bistable + APC/C-Cdh1）
# =============================================================================
_CELL_CYCLE_FEEDBACK_LOOPS: list[dict[str, Any]] = [
    # 1. CyclinB-APC/C 延迟负反馈（DDE 振荡器, delay=30min）
    #    Pomerening 2005 (PMID:11389814) 经典模型：
    #    - CyclinB-CDK1 激活 APC/C-Cdc20
    #    - APC/C-Cdc20 降解 Cyclin B
    #    - 形成 delay=30min 的延迟负反馈振荡，产生 8-12 小时细胞周期振荡
    {
        "id": "FL_CYCLINB_APC",
        "loop_type": "negative",
        "node_ids": [
            "CyclinB_CDK1",
            "APC_C_Cdc20_active",
            "CyclinB_degraded",
            "Cyclin_B",
        ],
        "delay_minutes": _CYCLINB_APC_DELAY_MINUTES,  # 30.0 min
        "bistable": False,
        "template": "oscillatory_feedback.j2",
        "description": (
            "CyclinB-APC/C 延迟负反馈振荡器（CyclinB-CDK1 激活 APC/C-Cdc20, "
            "APC/C 降解 Cyclin B, delay=30min, 8-12h 周期振荡, Pomerening 2005）"
        ),
        "source_pmid": _PMID_POMERENING_2005,
        "dde_solver": "solvers/dde_solver.py",
    },
    # 2. p53 → p21 抑制 CDK2/4（来自 p53 Specialist 的 cross-talk）
    #    p53 转录激活 p21，p21 抑制 CyclinE-CDK2 / CyclinD-CDK4 阻滞 G1/S
    {
        "id": "FL_P53_P21_CDK_NEG",
        "loop_type": "negative",
        "node_ids": ["p21", "CyclinE_CDK2", "CyclinD_CDK4"],
        "delay_minutes": 0.0,
        "bistable": False,
        "description": (
            "p53→p21 抑制 CDK2/4（p21 抑制 CyclinE-CDK2 / CyclinD-CDK4, "
            "阻滞 G1/S 转换, 来自 p53 Specialist 的 cross-talk）"
        ),
    },
    # 3. Rb-E2F toggle bistable（G1/S point-of-no-return）
    #    Rb 磷酸化后 E2F 释放，E2F 激活 CyclinE-CDK2 进一步磷酸化 Rb，
    #    正反馈 toggle 维持 G1/S 转换（Yao 2008 bistable）
    {
        "id": "FL_RB_E2F_TOGGLE_BISTABLE",
        "loop_type": "positive",
        "node_ids": [
            "CyclinD_CDK4",
            "pRb_phosphorylated",
            "E2F_free",
            "Cyclin_E_mRNA",
            "Cyclin_E",
            "CyclinE_CDK2",
        ],
        "delay_minutes": 0.0,
        "bistable": True,
        "point_of_no_return": True,
        "template": "bistable_switch.j2",
        "description": (
            "Rb-E2F toggle bistable（Rb 磷酸化释放 E2F, E2F 激活 CyclinE-CDK2 "
            "进一步磷酸化 Rb, 正反馈 toggle 维持 G1/S 转换, point-of-no-return）"
        ),
        "source_pmid": _PMID_YAO_2008,
    },
    # 4. APC/C-Cdh1（G1 早期激活, 降解 Cyclin A/B 维持 G1 静止）
    #    APC/C-Cdh1 在 G1 早期激活，降解 Cyclin A / Cyclin B 维持 G1 期静止状态
    {
        "id": "FL_APC_CDH1_G1_MAINTENANCE",
        "loop_type": "negative",
        "node_ids": ["APC_C_Cdh1", "Cyclin_A", "Cyclin_B"],
        "delay_minutes": 0.0,
        "bistable": False,
        "description": (
            "APC/C-Cdh1 G1 早期维持（APC/C-Cdh1 在 G1 早期激活, 降解 "
            "Cyclin A / Cyclin B 维持 G1 期静止状态, 防止 premature S/M 入口）"
        ),
    },
]


# =============================================================================
# Cell Cycle Crosstalk Reaction 片段（5 条，仅本通路侧片段，edge 由 Coordinator 管理）
# =============================================================================
# 接收 pERK / p53 / pAKT / Myc 的 cross-talk 调控本通路 Cyclin D / p21
# 注意：ERK→Cyclin D cross-talk edge 由 Coordinator 生成，本 Specialist 仅返回
# 本通路侧消费片段
_CELL_CYCLE_CROSSTALK_REACTIONS: list[dict[str, Any]] = [
    # 1. pERK → Cyclin_D（transcription, MAPK 激活 Cyclin D 转录）
    #    pERK 激活转录因子（如 Fos/Myc）激活 Cyclin D1 转录（早期 G1 反应）
    {
        "source": "pERK",
        "target": "Cyclin_D",
        "mechanism": "transcription",
        "shared_species": ["ERK"],
        "description": "pERK 激活 Cyclin D 转录（pERK 激活转录因子 Fos/Myc 诱导 Cyclin D1 表达, MAPK→Cell Cycle cross-talk）",
    },
    # 2. p53 → p21（transcription, 来自 p53 Specialist）
    #    p53 转录激活 p21（CDK 抑制剂），p21 抑制 CyclinE-CDK2 / CyclinD-CDK4 阻滞 G1/S
    {
        "source": "p53",
        "target": "p21",
        "mechanism": "transcription",
        "shared_species": ["p53", "p21"],
        "site": "p53RE(p21/CDKN1A promoter)",
        "description": "p53 转录激活 p21（p53 作转录因子, p21 抑制 CyclinE-CDK2 / CyclinD-CDK4 阻滞 G1/S, 与 p53 通路 cross-talk）",
    },
    # 3. pAKT → p21（inhibition, pAKT 抑制 p21）
    #    pAKT 磷酸化 p21 抑制其胞质稳定性，降低 p21 对 CDK 的抑制（促细胞周期进入）
    {
        "source": "pAKT",
        "target": "p21",
        "mechanism": "inhibition",
        "shared_species": ["AKT", "p21"],
        "description": "pAKT 抑制 p21（pAKT 磷酸化 p21 降低其胞质稳定性, 释放对 CDK2/4 的抑制, 促细胞周期进入, 与 PI3K 通路 cross-talk）",
    },
    # 4. Myc → Cyclin_D/E（transcription, Myc 激活 Cyclin D/E 转录）
    #    Myc 转录因子激活 Cyclin D1 / Cyclin E1 转录（促 G1 进展）
    {
        "source": "Myc",
        "target": "Cyclin_D",
        "mechanism": "transcription",
        "shared_species": ["Myc"],
        "site": "E-box(Cyclin D1/E1 promoter)",
        "description": "Myc 激活 Cyclin D / E 转录（Myc 作转录因子结合 E-box, 激活 Cyclin D1 / Cyclin E1 表达促 G1 进展）",
    },
    # 5. pAKT → GSK3β → Cyclin_D（inhibition, pAKT 抑制 GSK3β 阻止 Cyclin D 降解）
    #    pAKT 磷酸化 GSK3β Ser9 抑制其活性，阻止 GSK3β 磷酸化 Cyclin D1 Thr286
    #    导致 Cyclin D1 不被蛋白酶体降解，累积促 G1 进展
    {
        "source": "pAKT",
        "target": "Cyclin_D",
        "mechanism": "inhibition",
        "shared_species": ["AKT"],
        "intermediate": "GSK3β",
        "site": "GSK3β_Ser9",
        "description": "pAKT 抑制 GSK3β 阻止 Cyclin D 降解（pAKT 磷酸化 GSK3β Ser9 抑制其活性, 阻止 GSK3β 磷酸化 Cyclin D1 Thr286 防止降解, 累积促 G1 进展）",
    },
]


# =============================================================================
# Cell Cycle 扰动（6 个：5 个药物 + 1 个突变）
# =============================================================================
_CELL_CYCLE_PERTURBATIONS: list[dict[str, Any]] = [
    # 1. Palbociclib / PD-0332991（CDK4/6 inhibitor, FDA-approved）
    #    Palbociclib 是 CDK4/6 选择性抑制剂，用于 HR+ 乳腺癌治疗（FDA-approved）
    {
        "target": "CDK4",
        "drug": "Palbociclib",
        "mechanism": "inhibition",
        "ko_target": None,
        "description": "Palbociclib / PD-0332991（CDK4/6 选择性抑制剂, 小分子, FDA-approved, 用于 HR+ 乳腺癌）",
    },
    # 2. Ribociclib / LEE011（CDK4/6 inhibitor, FDA-approved）
    #    Ribociclib 是 CDK4/6 选择性抑制剂，用于 HR+ 乳腺癌治疗（FDA-approved）
    {
        "target": "CDK4",
        "drug": "Ribociclib",
        "mechanism": "inhibition",
        "ko_target": None,
        "description": "Ribociclib / LEE011（CDK4/6 选择性抑制剂, 小分子, FDA-approved, 用于 HR+ 乳腺癌）",
    },
    # 3. Abemaciclib / Verzenio（CDK4/6 inhibitor, FDA-approved）
    #    Abemaciclib 是 CDK4/6 选择性抑制剂，用于 HR+ 乳腺癌治疗（FDA-approved）
    {
        "target": "CDK4",
        "drug": "Abemaciclib",
        "mechanism": "inhibition",
        "ko_target": None,
        "description": "Abemaciclib / Verzenio（CDK4/6 选择性抑制剂, 小分子, FDA-approved, 用于 HR+ 乳腺癌）",
    },
    # 4. Roscovitine / Seliciclib（CDK2 inhibitor）
    #    Roscovitine 是 CDK2 / CDK1 / CDK7 / CDK9 抑制剂（实验工具药, 进入临床 II 期）
    {
        "target": "CDK2",
        "drug": "Roscovitine",
        "mechanism": "inhibition",
        "ko_target": None,
        "description": "Roscovitine / Seliciclib（CDK2 / CDK1 / CDK7 / CDK9 抑制剂, 小分子, 实验工具药）",
    },
    # 5. Flavopiridol / Alvocidib（pan-CDK inhibitor）
    #    Flavopiridol 是广谱 CDK 抑制剂（CDK1/2/4/6/7/9），临床试用于 AML / CLL
    {
        "target": "CDK2",
        "drug": "Flavopiridol",
        "mechanism": "inhibition",
        "ko_target": None,
        "description": "Flavopiridol / Alvocidib（pan-CDK 广谱抑制剂, CDK1/2/4/6/7/9, 小分子, 临床试用于 AML / CLL）",
    },
    # 6. CDKN2A / p16 loss（loss-of-function mutation, CDK4/6 失去抑制）
    #    CDKN2A 编码 p16INK4a（CDK4/6 内源性抑制剂），loss 导致 CDK4/6 失去抑制
    #    （常见于胰腺癌 / 黑色素瘤 / 胶质瘤等）
    {
        "target": "CDK4",
        "drug": None,
        "mechanism": "knockout",
        "ko_target": "CDKN2A_p16_loss",
        "description": "CDKN2A / p16 loss（loss-of-function, CDK4/6 内源性抑制剂 p16INK4a 失活, CDK4/6 失去抑制, 常见于胰腺癌 / 黑色素瘤 / 胶质瘤）",
    },
]


# =============================================================================
# Cell Cycle Validation 规则（3 条 benchmark）
# =============================================================================
# 每条含：rule_id / metric_name / expected / tolerance / pmid / description
# expected 取区间中点，tolerance 取区间半宽
_CELL_CYCLE_VALIDATION_RULES: list[dict[str, Any]] = [
    # 1. CyclinB-APC/C 振荡周期 8-12 hours（Pomerening 2005, in vitro）
    #    Cdk1-Cyclin B-APC/C 延迟负反馈产生 8-12 小时周期振荡
    {
        "rule_id": "VAL_CELL_CYCLE_CYCLINB_APC_OSCILLATION",
        "metric_name": "CyclinB_APC_oscillation_period",
        "expected": 10.0,   # (8.0 + 12.0) / 2
        "tolerance": 2.0,    # (12.0 - 8.0) / 2
        "expected_min": 8.0,
        "expected_max": 12.0,
        "unit": "hours",
        "pmid": _PMID_POMERENING_2005,
        "comparison": "range",
        "pathway_tag": PATHWAY_TAG,
        "description": "CyclinB-APC/C 延迟负反馈振荡周期 8-12 小时（Pomerening 2005 Cdk1 oscillation, in vitro 靶细胞周期振荡）",
    },
    # 2. Rb-E2F bistable G1/S switch（Yao 2008, PMID:12064617）
    #    Rb-E2F 系统应呈现 bistable G1/S 转换开关行为
    {
        "rule_id": "VAL_CELL_CYCLE_RB_E2F_BISTABLE_SWITCH",
        "metric_name": "Rb_E2F_bistable_switch",
        "expected": True,
        "tolerance": 0.0,
        "expected_min": True,
        "expected_max": True,
        "unit": "boolean",
        "pmid": _PMID_YAO_2008,
        "comparison": "boolean",
        "pathway_tag": PATHWAY_TAG,
        "description": "Rb-E2F 应呈现 bistable G1/S 转换开关（Yao 2008 bistable Rb-E2F switch, 一旦 Rb 磷酸化超过阈值 E2F 永久释放）",
    },
    # 3. Cyclin D1 达峰时间 60-240 minutes（G1 期 Cyclin D1 表达）
    #    G1 期 Cyclin D1 在 60-240 min 内达峰（受生长因子诱导表达）
    {
        "rule_id": "VAL_CELL_CYCLE_CYCLIN_D1_PEAK_TIME",
        "metric_name": "Cyclin_D1_peak_time",
        "expected": 150.0,   # (60.0 + 240.0) / 2
        "tolerance": 90.0,   # (240.0 - 60.0) / 2
        "expected_min": 60.0,
        "expected_max": 240.0,
        "unit": "minutes",
        "pmid": _PMID_POMERENING_2005,
        "comparison": "range",
        "pathway_tag": PATHWAY_TAG,
        "description": "G1 期 Cyclin D1 达峰时间 60-240 min（受生长因子诱导表达, G1 早期响应）",
    },
]


@register_specialist
class CellCycleSpecialist(PathwaySpecialistBase):
    """Cell Cycle 通路 Specialist。

    实现 Cyclin D-CDK4/6（G1）→Cyclin E-CDK2（G1/S）→Cyclin A-CDK2（S）
    →Cyclin A-CDK1（G2）→Cyclin B-CDK1（M）→APC/C-Cdc20 核心拓扑 +
    Rb-E2F G1/S toggle（bistable）+ CyclinB-APC/C 延迟负反馈振荡器（delay=30min）
    + p53→p21 抑制 CDK2/4 的 Core/Feedback/Crosstalk/Perturbation/Validation
    5 模块，输出通路特异 Reaction IR 片段 + Rb-E2F toggle CompositeReaction
    + 模板选择 + Validation 规则。

    职责边界：
    - 处理 Cell Cycle 核心（Cyclin D/E/A/B + CDK1/2/4 + Rb/E2F + APC/C-Cdc20）
    - 不处理 p53→p21 上游（由 p53 Specialist 处理，Task 4.6）
    - 不生成 ERK→Cyclin D cross-talk edge（由 Cross-talk Coordinator 处理，Task 4.13）
    - 不处理 AKT 上游（由 PI3K Specialist 处理，Task 4.5）

    输入：
    - ``pathway_graph``：v4_pathway_graph 的 Cell Cycle 子图（含 FeedbackLoop
      FL_CYCLINB_APC delay=30min）
    - ``ontology_entities``：P1 Ontology Entities（可选，HGNC/UniProt ID 对齐）

    输出：
    - ``apply_core``：14 条核心 Reaction IR 片段 + 23 物种 + Rb-E2F toggle
      CompositeReaction（bistable point-of-no-return）
      （p21 标记 shared=True，与 p53 Specialist 共享）
    - ``apply_feedback``：4 条 FeedbackLoop（CyclinB-APC/C delay=30min 振荡 /
      p53→p21 CDK 抑制 / Rb-E2F bistable toggle / APC/C-Cdh1 G1 维持）
    - ``apply_crosstalk``：5 条 cross-talk Reaction 片段
      （pERK→CyclinD + p53→p21 + pAKT→p21 + Myc→CyclinD/E + pAKT→GSK3β→CyclinD）
    - ``apply_perturbation``：6 个扰动
      （Palbociclib/Ribociclib/Abemaciclib/Roscovitine/Flavopiridol/CDKN2A loss）
    - ``apply_validation``：3 条 Validation benchmark
      （CyclinB-APC/C 振荡 8-12h / Rb-E2F bistable / Cyclin D1 达峰 60-240min）
    """

    pathway_class: str = "CELL_CYCLE"
    display_name: str = "Cell Cycle Regulation"

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
                    species=list(_CELL_CYCLE_CORE_SPECIES),
                    reactions=list(_CELL_CYCLE_CORE_REACTIONS),
                    kinetics_overrides={},
                )
            if module_name == MODULE_FEEDBACK:
                return FeedbackModuleData(
                    feedback_loops=list(_CELL_CYCLE_FEEDBACK_LOOPS),
                    delay_minutes=_CYCLINB_APC_DELAY_MINUTES,
                    loop_type="mixed",  # 含负反馈振荡 + 正反馈 bistable
                )
            if module_name == MODULE_CROSSTALK:
                return CrosstalkModuleData(
                    crosstalk_reactions=list(_CELL_CYCLE_CROSSTALK_REACTIONS),
                    shared_species=["p21", "ERK", "p53", "AKT", "Myc"],
                    coordination_strategy="merge",
                )
            if module_name == MODULE_PERTURBATION:
                return PerturbationModuleData(
                    perturbation_reactions=list(_CELL_CYCLE_PERTURBATIONS),
                    drug_targets=[
                        p for p in _CELL_CYCLE_PERTURBATIONS if p.get("drug")
                    ],
                    ko_targets=[
                        p for p in _CELL_CYCLE_PERTURBATIONS if p.get("ko_target")
                    ],
                )
            if module_name == MODULE_VALIDATION:
                return ValidationModuleData(
                    rules=list(_CELL_CYCLE_VALIDATION_RULES),
                    benchmarks=[
                        {
                            "benchmark_name": r["metric_name"],
                            "value": r["expected"],
                            "unit": r["unit"],
                            "condition": "cell cycle progression",
                            "reference": r["pmid"],
                        }
                        for r in _CELL_CYCLE_VALIDATION_RULES
                    ],
                    tolerances={
                        r["metric_name"]: r["tolerance"]
                        for r in _CELL_CYCLE_VALIDATION_RULES
                    },
                    pmid_references=[
                        {
                            "pmid": r["pmid"],
                            "citation": r["description"],
                            "pathway_class": PATHWAY_TAG,
                            "metric_name": r["metric_name"],
                        }
                        for r in _CELL_CYCLE_VALIDATION_RULES
                        if r["pmid"]
                    ],
                )
            logger.warning(
                "CellCycleSpecialist.load_module: 未知模块名 '%s'",
                module_name,
            )
            return None
        except Exception as exc:
            logger.warning(
                "CellCycleSpecialist.load_module 加载模块 '%s' 失败: %s",
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
        """应用核心模块，返回 Cell Cycle 通路核心 Reaction IR 片段。

        输出 14 条核心反应（Cyclin-CDK 级联 11 + Rb-E2F toggle 3）：
        1. Cyclin_D → CyclinD_CDK4（complex_formation, G1 phase）
        2. CyclinD_CDK4 → pRb_phosphorylated（异磷酸化, Rb 作 substrate）
        3. pRb_phosphorylated → E2F_free（dissociation, Rb 释放 E2F）
        4. Cyclin_E → CyclinE_CDK2（complex_formation, G1/S transition）
        5. CyclinE_CDK2 → E2F_active（异磷酸化, E2F 正反馈激活）
        6. Cyclin_A → CyclinA_CDK2（complex_formation, S phase）
        7. Cyclin_A → CyclinA_CDK1（complex_formation, G2 phase）
        8. Cyclin_B → CyclinB_CDK1（complex_formation, M phase entry）
        9. CyclinB_CDK1 → APC_C_Cdc20_active（activation, APC/C-Cdc20 激活）
        10. APC_C_Cdc20_active → CyclinB_degraded（proteasomal_degradation）
        11. APC_C_Cdc20_active → Securin_degraded（proteasomal_degradation）
        12. E2F_free → Cyclin_E_mRNA（transcription, Hill, 正反馈）
        13. Cyclin_E_mRNA → Cyclin_E（translation）
        14. CyclinE_CDK2 → pRb_phosphorylated（positive feedback, 进一步磷酸化 Rb）

        p21 物种标记 shared=True（与 p53 Specialist 的 p53→p21 转录路径共享）。

        CompositeReaction 输出 Rb-E2F G1/S toggle（bistable, point-of-no-return）。

        Returns:
            dict 含 ``species``（23 物种）/ ``reactions``（14 反应） /
            ``composite_reactions``（Rb-E2F toggle）字段。异常时返回
            ``{"species": [], "reactions": [], "composite_reactions": []}``。
        """
        try:
            return {
                "species": list(_CELL_CYCLE_CORE_SPECIES),
                "reactions": list(_CELL_CYCLE_CORE_REACTIONS),
                "composite_reactions": list(_CELL_CYCLE_COMPOSITE_REACTIONS),
                "pathway_tag": PATHWAY_TAG,
                "source_sbml": SOURCE_SBML,
            }
        except Exception as exc:
            logger.warning(
                "CellCycleSpecialist.apply_core 失败: %s", exc
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
        """应用反馈模块，返回 Cell Cycle 通路 FeedbackLoop 列表。

        输出 4 条反馈环：
        1. CyclinB-APC/C 延迟负反馈振荡器（delay=30min, 8-12h 周期振荡）
        2. p53 → p21 抑制 CDK2/4（负反馈, 来自 p53 Specialist cross-talk）
        3. Rb-E2F toggle bistable（正反馈, G1/S point-of-no-return）
        4. APC/C-Cdh1 G1 早期维持（负反馈, 降解 Cyclin A/B 维持 G1 静止）

        Returns:
            FeedbackLoop 字典列表（含 bistable / delay_minutes 标记）。
            异常返回空列表。
        """
        try:
            return list(_CELL_CYCLE_FEEDBACK_LOOPS)
        except Exception as exc:
            logger.warning(
                "CellCycleSpecialist.apply_feedback 失败: %s", exc
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
        """应用跨通路模块，返回 Cell Cycle 通路侧 cross-talk Reaction 片段。

        注意：本方法仅生成本通路侧的 cross-talk Reaction 片段，
        cross-talk edge 本身由 Cross-talk Coordinator 创建（Task 4.13）。

        输出 5 条 cross-talk Reaction 片段：
        1. pERK → Cyclin_D（transcription, MAPK 激活 Cyclin D 转录）
        2. p53 → p21（transcription, p53 激活 p21 抑制 CDK2/4）
        3. pAKT → p21（inhibition, pAKT 抑制 p21）
        4. Myc → Cyclin_D（transcription, Myc 激活 Cyclin D/E 转录）
        5. pAKT → Cyclin_D（inhibition, pAKT 抑制 GSK3β 阻止 Cyclin D 降解）

        Args:
            pathway_graph: 通路图（当前未使用，保留接口一致性）。
            crosstalk_edges: 来自 Coordinator 的 cross-talk edge 列表
                （当前未使用，本 Specialist 返回静态拓扑片段）。

        Returns:
            cross-talk Reaction 片段列表。异返回空列表。
        """
        try:
            return list(_CELL_CYCLE_CROSSTALK_REACTIONS)
        except Exception as exc:
            logger.warning(
                "CellCycleSpecialist.apply_crosstalk 失败: %s", exc
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
        """应用扰动模块，返回 Cell Cycle 通路特异药物 / 突变 Reaction 片段。

        输出 6 个扰动（5 个药物 + 1 个突变）：
        1. Palbociclib（CDK4/6 抑制剂, FDA-approved）
        2. Ribociclib（CDK4/6 抑制剂, FDA-approved）
        3. Abemaciclib（CDK4/6 抑制剂, FDA-approved）
        4. Roscovitine / Seliciclib（CDK2 抑制剂）
        5. Flavopiridol / Alvocidib（pan-CDK 抑制剂）
        6. CDKN2A / p16 loss（loss-of-function, CDK4/6 失去抑制）

        Args:
            pathway_graph: 通路图（当前未使用）。
            perturbation_points: 扰动点列表（当前未使用，本 Specialist
                返回静态拓扑片段）。

        Returns:
            扰动 Reaction 片段列表。异返回空列表。
        """
        try:
            return list(_CELL_CYCLE_PERTURBATIONS)
        except Exception as exc:
            logger.warning(
                "CellCycleSpecialist.apply_perturbation 失败: %s", exc
            )
            return []

    # =================================================================
    # apply_validation：Validation 规则列表
    # =================================================================
    def apply_validation(
        self,
        simulation_result: dict | None = None,
    ) -> list[dict]:
        """应用验证模块，返回 Cell Cycle 通路 Validation 规则列表。

        输出 3 条 benchmark：
        1. CyclinB-APC/C 振荡周期 8-12 hours（Pomerening 2005, PMID:11389814）
        2. Rb-E2F bistable G1/S switch（Yao 2008, PMID:12064617）
        3. Cyclin D1 达峰时间 60-240 min（G1 期响应）

        Args:
            simulation_result: 仿真结果（可选，当前未使用，返回静态规则集）。

        Returns:
            Validation 规则列表。异返回空列表。
        """
        try:
            return list(_CELL_CYCLE_VALIDATION_RULES)
        except Exception as exc:
            logger.warning(
                "CellCycleSpecialist.apply_validation 失败: %s", exc
            )
            return []

    # =================================================================
    # select_template：覆写以支持 oscillatory / bistable 模式
    # =================================================================
    def select_template(self, mechanism: str) -> str:
        """根据 mechanism 选择 ODE 模板名（覆写支持振荡与 bistable）。

        默认映射（与 P3 ``ode_templates_v2/`` 下 .j2 文件对齐）：
        - ``oscillatory`` → ``oscillatory_feedback``（CyclinB-APC/C 延迟振荡）
        - ``bistable`` → ``bistable_switch``（Rb-E2F G1/S toggle）
        - ``phosphorylation`` → ``_mechanism_phosphorylation_mm``
          （CyclinD-CDK4→Rb / CyclinE-CDK2→E2F / CyclinE-CDK2→Rb 异磷酸化）
        - ``transcription`` → ``oscillatory_feedback``（E2F 转录 Cyclin E 周期振荡）

        Args:
            mechanism: 机制名（小写，如 ``"oscillatory"`` / ``"bistable"``）。

        Returns:
            ODE 模板名（不含 ``.j2`` 后缀）。未匹配时返回 ``"default"``
            （调用方应处理默认降级）。
        """
        # bistable 模式：Rb-E2F G1/S toggle point-of-no-return
        if mechanism == "bistable":
            return "bistable_switch"
        # oscillatory 模式：CyclinB-APC/C 延迟负反馈振荡（8-12h 周期）
        if mechanism == "oscillatory":
            return "oscillatory_feedback"
        # 转录场景：E2F 转录 Cyclin E，使用 oscillatory_feedback 模板
        if mechanism == "transcription":
            return "oscillatory_feedback"
        # 磷酸化场景：Cyclin-CDK 异磷酸化 Rb/E2F
        if mechanism == "phosphorylation":
            return "_mechanism_phosphorylation_mm"
        # 其他机制走默认基类映射
        return super().select_template(mechanism)


__all__ = [
    "CellCycleSpecialist",
    "PATHWAY_TAG",
    "SOURCE_SBML",
]
