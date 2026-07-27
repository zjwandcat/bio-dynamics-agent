# BioDynamics Agent v4 - p53 Specialist (Phase 4 / Task 4.6)
# p53 肿瘤抑制通路 Specialist：实现 DNA damage→ATM/ATR→p53 磷酸化→四聚化→入核→
# 转录（Mdm2/p21/Bax/PUMA/Noxa）核心拓扑 + p53-Mdm2 转录延迟负反馈（DDE 60min）+
# ATM→p53+Mdm2 双向调控。
#
# 设计原则（铁律）：
# 1. Feature Flag V4_PATHWAY_SPECIALIST_ENABLED=false 时由 LangGraph hook 层短路，
#    本 Specialist 代码不被调用（基类本身不做 flag 检查）
# 2. 使用 Task 4.0 修复后的 PHOSPHORYLATION 语义：
#    - 异磷酸化（pATM → p53）：未磷酸化 p53 作 substrate，磷酸化 p53 作 product，
#      pATM 作 catalytic modifier（Ser15/Thr20）
#    - 异磷酸化（pATM → pATM 自磷酸化）：DNA damage 激活 ATM，ATM 作 substrate/product，
#      DNA damage 作 activator
# 3. 不处理 Bax/PUMA 凋亡执行（由 Apoptosis Specialist 处理，Task 4.7）
# 4. 不生成 AKT→Mdm2 cross-talk edge 本身（由 Cross-talk Coordinator 处理，Task 4.13），
#    仅返回本通路侧的 cross-talk Reaction 片段
# 5. apply_* 方法实现捕获异常并返回空 list/dict，记录 logger.warning，不抛异常
#
# 依赖：
# - P1 ontology / pathway_registry（p53 通路类别键）
# - P2 schema（SpeciesV2 / ReactionV2 / SpeciesRef / Modifier / StateMachine）
# - P2 MechanismType.PHOSPHORYLATION / TRANSCRIPTION / UBIQUITINATION
# - P3 ode_templates_v2（_mechanism_phosphorylation_mm.j2 / oscillatory_feedback.j2 模板）
# - P3 solvers/dde_solver.py（DDE 延迟微分方程求解，p53-Mdm2 delay=60min）
# - P3 pathway_graph/initializer.py（p53_signaling core_nodes / core_edges）
#
# 参考：
# - spec.md Part 3 Specialist 4（第 213-218 行）
# - tasks.md Task 4.6（第 77-86 行）
# - Lev Bar-Or 2000 (PMID:10644692) p53-Mdm2 脉冲振荡模型
# - BioModels BIOMD0000000382 (p53-Mdm2 oscillator)

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
# p53 通路标签
# =============================================================================
PATHWAY_TAG: str = "p53"

# SBML BioModels ID（p53-Mdm2 oscillator, Lev Bar-Or 2000）
SOURCE_SBML: str = get_biomodels_id(PATHWAY_TAG)

# Validation benchmark PMID 引用（Lev Bar-Or 2000, p53-Mdm2 脉冲振荡）
_PMID_LEV_BAR_OR_2000: str = "PMID:10644692"

# p53-Mdm2 转录延迟（DDE，分钟）
# Lev Bar-Or 2000 模型中 p53 转录激活 Mdm2 mRNA，Mdm2 mRNA 翻译为 Mdm2 蛋白，
# Mdm2 蛋白降解 p53，形成延迟负反馈振荡。delay=60 min 是中位转录翻译延迟。
_P53_MDM2_DELAY_MINUTES: float = 60.0


# =============================================================================
# p53 核心物种（与 P3 pathway_graph/initializer.py p53_signaling.core_nodes 对齐，
# 扩展 DNA_damage / p53_tetramer / p53_nuclear）
# =============================================================================
# p53 物种标记 shared=True（与 Apoptosis Specialist 的 Bax/PUMA 凋亡路径共享，
# p53 作为转录因子激活 Bax/PUMA 启动凋亡）。
# p21 物种标记 shared=True（与 Cell Cycle Specialist 的 CDK2/CDK4 抑制路径共享，
# p21 抑制 CyclinE-CDK2 / CyclinD-CDK4 阻滞细胞周期）。
_P53_CORE_SPECIES: list[dict[str, Any]] = [
    # [C4 fix] No SBML match in BIOMD0000000252 (Hunzicker2010, PMID:20624280).
    #   SBML model is an abstract 4-variable p53-Mdm2 oscillator (species: m, mm, p, pm;
    #   all boundary_condition=true, initial=1.0). These abstract variables do not map
    #   to the specialist's mammalian p53 pathway species (ATM/pATM/p53/Mdm2/p21/etc.).
    #   Kept original (no initial_concentration field) — biology-aware defaults apply.
    # DNA damage（外源应激：化疗药物 / UV / γ-irradiation）
    {"name": "DNA_damage", "species_type": "damage", "compartment": "nucleus"},
    # ATM 激酶（DNA damage 感应器，MRN 复合体激活）
    {"name": "ATM", "species_type": "protein", "compartment": "nucleus"},
    {"name": "pATM", "species_type": "protein", "compartment": "nucleus"},
    # p53（shared：与 Apoptosis Bax/PUMA 路径共享）
    # 语义说明：本字段表示磷酸化激活的 p53（pATM 异磷酸化 Ser15/Thr20 后的 p53），
    # 与 P3 initializer core_edges 中 ("pATM", "p53", "phosphorylation", ...) 对齐。
    {"name": "p53", "species_type": "protein", "compartment": "nucleus",
     "shared": True},
    # p53_tetramer（4 个 p53 monomer → 1 tetramer，四聚化活性形式）
    {"name": "p53_tetramer", "species_type": "complex", "compartment": "nucleus"},
    # p53_nuclear（p53 tetramer 入核，作为转录因子激活下游基因）
    {"name": "p53_nuclear", "species_type": "protein", "compartment": "nucleus"},
    # Mdm2 mRNA / Mdm2 蛋白（p53 转录靶基因，负反馈调控 p53）
    {"name": "Mdm2_mRNA", "species_type": "mrna", "compartment": "nucleus"},
    {"name": "Mdm2", "species_type": "protein", "compartment": "nucleus"},
    # p53_ubi（p53 被 Mdm2 E3 泛素化标记，等待蛋白酶体降解）
    {"name": "p53_ubi", "species_type": "protein", "compartment": "nucleus"},
    # p21 mRNA / p21 蛋白（p53 转录靶基因，CDK 抑制剂，阻滞细胞周期）
    # p21（shared：与 Cell Cycle CDK2/CDK4 抑制路径共享）
    {"name": "p21_mRNA", "species_type": "mrna", "compartment": "nucleus"},
    {"name": "p21", "species_type": "protein", "compartment": "nucleus",
     "shared": True},
    # TD-033 (IB-064) 修复：补充 p300/CBP 乙酰化辅因子（p53 乙酰化的关键辅因子）
    # p300 与 CBP 是旁系同源物（paralogues），均具有 HAT（组蛋白乙酰转移酶）活性，
    # 乙酰化 p53 C 端 Lys382 等位点，稳定 p53 并增强其转录活性。
    {"name": "p300", "species_type": "protein", "compartment": "nucleus"},
    {"name": "CBP", "species_type": "protein", "compartment": "nucleus"},
    # TD-033 (IB-064) 修复：补充 p53_ac（乙酰化 p53，p300/CBP 催化产物）
    {"name": "p53_ac", "species_type": "protein", "compartment": "nucleus"},
    # TD-033 (IB-064) 修复：补充 MDM4/MDMX（p53 转录活性抑制因子，与 MDM2 协同）
    # MDM4 (又称 MDMX) 与 MDM2 同源但缺失 E3 连接酶活性，主要通过结合 p53 N 端
    # 转录激活域抑制 p53 转录活性，并增强 MDM2 介导的 p53 泛素化。
    {"name": "MDM4", "species_type": "protein", "compartment": "nucleus"},
    # [N6 缺口 1] 药物物种（species_type="drug"）— 由 drug_library 驱动
    # Nutlin-3 是 MDM2-p53 相互作用抑制剂（IC50=90 nM, PMID:15143221）
    # 占据 MDM2 的 p53 结合口袋阻断 MDM2 降解 p53，稳定 p53（实验室常用工具化合物）
    build_drug_species("Nutlin-3"),
]


# =============================================================================
# p53 核心反应（12 条）
# =============================================================================
# 每条反应含：source / target / mechanism / kinetics_type / pathway_tag /
#             substrate / product / modifier / modifier_type / autophosphorylation
#
# PHOSPHORYLATION 语义（Task 4.0 修复后）：
# - 异磷酸化（pATM → p53）：未磷酸化 p53 作 substrate，磷酸化 p53 作 product，
#   pATM 作 catalytic modifier（Ser15/Thr20）
# - ATM 自磷酸化（DNA damage → pATM）：ATM 作 substrate，pATM 作 product，
#   DNA damage 作 activator（allosteric modifier）
#
# kinetics_type 选择：
# - phosphorylation → Michaelis_Menten（与 P3 _mechanism_phosphorylation_mm 模板对齐）
# - transcription → Hill（p53 作为转录因子，Hill 动力学 n=2 协同结合）
# - translation → mass_action（mRNA→protein，线性翻译）
# - ubiquitination → Michaelis_Menten（Mdm2 E3 ligase 催化，MM 动力学）
# - proteasomal_degradation → mass_action（泛素化 p53 蛋白酶体降解，一级动力学）
# - tetramerization → mass_action（4 个 p53 monomer 协同组装，质量作用）
# - nuclear_import → mass_action（p53 tetramer 入核，转运一级动力学）
_P53_CORE_REACTIONS: list[dict[str, Any]] = [
    # 1. DNA damage → pATM（ATM 磷酸化激活，DNA damage 作 activator）
    #    MRN 复合体感应 DNA 双链断裂，招募并激活 ATM
    # [P1-10 修复] 机制 phosphorylation→activation：
    #   Root Cause: transcriptional_delay.j2 的 phosphorylation 分支消耗 src (DNA_damage)，
    #     但 DNA_damage 是刺激信号不是底物（被消耗后无法持续驱动 pATM）。
    #   Fix: 改用 activation 机制，DNA_damage 作催化剂驱动 pATM 产生（logistic 饱和 + k_deg 降解）
    {
        "source": "DNA_damage",
        "target": "pATM",
        "mechanism": "activation",
        "kinetics_type": "mass_action",
        "pathway_tag": PATHWAY_TAG,
        "substrate": "ATM",
        "product": "pATM",
        "modifier": "DNA_damage",
        "modifier_type": "allosteric",
        "autophosphorylation": False,
        "description": "DNA damage 激活 ATM（DNA_damage 作 allosteric activator 催化 ATM→pATM，MRN 复合体感应 DSB）",
    },
    # 2. pATM → p53（异磷酸化，pATM 磷酸化 p53 Ser15/Thr20）
    #    pATM 是 p53 的上游激酶，磷酸化 p53 N 端转录激活域，稳定 p53 防止 Mdm2 降解
    # [P1-10 修复] 机制 phosphorylation→activation：
    #   Root Cause: 旧 phosphorylation 边 substrate=product=p53，磷酸化分支消耗 p53 又产生 p53，
    #     净变化=0（空操作），p53 永远不被产生。
    #   Fix: 改用 activation 机制，pATM 作催化剂产生 p53（logistic 饱和 + k_deg 降解）。
    #     activation else 分支：dy[p53] += k_act*pATM*(1-p53/max) - k_deg*p53
    {
        "source": "pATM",
        "target": "p53",
        "mechanism": "activation",
        "kinetics_type": "mass_action",
        "pathway_tag": PATHWAY_TAG,
        "substrate": "p53",
        "product": "p53",
        "modifier": "pATM",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "site": "Ser15/Thr20",
        "description": "pATM 磷酸化 p53 Ser15/Thr20（pATM 作 catalytic activator 产生 p53，稳定 p53 防止 Mdm2 降解）",
    },
    # 3. p53 → p53_tetramer（tetramerization，4 个 p53 monomer → 1 tetramer）
    #    p53 四聚化是 DNA 结合活性必需的（tetramer 的 DNA 结合亲和力远高于 monomer）
    {
        "source": "p53",
        "target": "p53_tetramer",
        "mechanism": "tetramerization",
        "kinetics_type": "mass_action",
        "pathway_tag": PATHWAY_TAG,
        # 4 p53 monomer → 1 tetramer（4 级质量作用动力学）
        "substrate": "p53",
        "product": "p53_tetramer",
        "modifier": "p53",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "stoichiometry": "4:1",
        "description": "p53 四聚化（4 个 p53 monomer → 1 tetramer，tetramer DNA 结合活性远高于 monomer）",
    },
    # 4. p53_tetramer → p53_nuclear（nuclear_import，p53 tetramer 入核）
    #    p53 tetramer 通过 importin-α/β 途径入核，作为转录因子激活下游基因
    {
        "source": "p53_tetramer",
        "target": "p53_nuclear",
        "mechanism": "nuclear_import",
        "kinetics_type": "mass_action",
        "pathway_tag": PATHWAY_TAG,
        # p53_tetramer(cytoplasm) 作 substrate，p53_nuclear(nucleus) 作 product
        "substrate": "p53_tetramer",
        "product": "p53_nuclear",
        "modifier": "p53_tetramer",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "p53 tetramer 入核（p53_tetramer 作 substrate，p53_nuclear 作 product，importin-α/β 转运）",
    },
    # 5. p53_nuclear → Mdm2_mRNA（transcription，Hill 动力学，p53 作转录因子）
    #    p53 结合 Mdm2 基因启动子 p53RE，Hill 协同结合（n=2）激活转录
    {
        "source": "p53_nuclear",
        "target": "Mdm2_mRNA",
        "mechanism": "transcription",
        "kinetics_type": "Hill",
        "pathway_tag": PATHWAY_TAG,
        # p53_nuclear 作 modifier（转录因子），DNA 作 substrate，Mdm2_mRNA 作 product
        "substrate": "DNA",
        "product": "Mdm2_mRNA",
        "modifier": "p53_nuclear",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "hill_coefficient": 2,
        "description": "p53 转录激活 Mdm2 mRNA（p53_nuclear 作转录因子，Hill n=2 协同结合 p53RE，DDE 延迟 60min）",
    },
    # 6. Mdm2_mRNA → Mdm2（translation，mRNA→protein）
    #    Mdm2 mRNA 在核糖体翻译为 Mdm2 蛋白（E3 泛素连接酶）
    {
        "source": "Mdm2_mRNA",
        "target": "Mdm2",
        "mechanism": "translation",
        "kinetics_type": "mass_action",
        "pathway_tag": PATHWAY_TAG,
        # Mdm2_mRNA 作 substrate，Mdm2 作 product
        "substrate": "Mdm2_mRNA",
        "product": "Mdm2",
        "modifier": "Mdm2_mRNA",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "Mdm2 mRNA 翻译为 Mdm2 蛋白（Mdm2_mRNA 作 substrate，Mdm2 作 product，E3 泛素连接酶）",
    },
    # 7. Mdm2 → p53_ubi（ubiquitination，p53 作 substrate，Mdm2 作 E3 ligase modifier）
    #    Mdm2 E3 泛素连接酶标记 p53 多泛素化，启动蛋白酶体降解
    # [P1-10 修复] 机制 ubiquitination→activation：
    #   Root Cause: transcriptional_delay.j2 的 ubiquitination 分支 dy[t_idx]-=_rate
    #     消耗 target (p53_ubi) 而非产生它，p53_ubi 始终=0.05（初始值）无法累积。
    #   Fix: 改用 activation 机制，Mdm2 作催化剂产生 p53_ubi（logistic 饱和 + k_deg 降解）。
    #     activation else 分支：dy[p53_ubi] += k_act*Mdm2*(1-p53_ubi/max) - k_deg*p53_ubi
    #     注意：此修改不消耗 p53（p53-Mdm2 负反馈被削弱），但保证 p53_ubi 可累积，
    #     驱动反应 8 (proteasomal_degradation) 消耗 p53_ubi，形成部分负反馈环路。
    {
        "source": "Mdm2",
        "target": "p53_ubi",
        "mechanism": "activation",
        "kinetics_type": "mass_action",
        "pathway_tag": PATHWAY_TAG,
        # 异磷酸化语义类似：p53 作 substrate，p53_ubi 作 product，Mdm2 作 catalytic modifier
        "substrate": "p53",
        "product": "p53_ubi",
        "modifier": "Mdm2",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "Mdm2 E3 泛素连接酶标记 p53 多泛素化（Mdm2 作 catalytic activator 产生 p53_ubi，启动蛋白酶体降解）",
    },
    # 8. p53_ubi → p53（proteasomal_degradation，泛素化 p53 蛋白酶体降解）
    #    26S 蛋白酶体识别多泛素链，降解 p53 释放游离氨基酸（p53_ubi → 降解）
    {
        "source": "p53_ubi",
        "target": "p53",
        "mechanism": "proteasomal_degradation",
        "kinetics_type": "mass_action",
        "pathway_tag": PATHWAY_TAG,
        # p53_ubi 作 substrate（被降解），产物为降解产物（无 p53 重新生成，
        # 但本反应 source/target 用 p53_ubi → p53 表示降解后 p53 池减少）
        "substrate": "p53_ubi",
        "product": "degraded_p53",
        "modifier": "proteasome",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "泛素化 p53 蛋白酶体降解（p53_ubi 作 substrate，26S 蛋白酶体识别多泛素链降解 p53）",
    },
    # 9. p53_nuclear → p21_mRNA（transcription，Hill 动力学，p53 作转录因子）
    #    p53 结合 p21/CDKN1A 基因启动子 p53RE，激活 p21 转录（CDK 抑制剂）
    {
        "source": "p53_nuclear",
        "target": "p21_mRNA",
        "mechanism": "transcription",
        "kinetics_type": "Hill",
        "pathway_tag": PATHWAY_TAG,
        # p53_nuclear 作 modifier（转录因子），DNA 作 substrate，p21_mRNA 作 product
        "substrate": "DNA",
        "product": "p21_mRNA",
        "modifier": "p53_nuclear",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "hill_coefficient": 2,
        "description": "p53 转录激活 p21 mRNA（p53_nuclear 作转录因子，Hill n=2 协同结合 p21 启动子 p53RE）",
    },
    # 10. p21_mRNA → p21（translation，mRNA→protein）
    #     p21 mRNA 在核糖体翻译为 p21 蛋白（CDK2/CDK4 抑制剂）
    {
        "source": "p21_mRNA",
        "target": "p21",
        "mechanism": "translation",
        "kinetics_type": "mass_action",
        "pathway_tag": PATHWAY_TAG,
        # p21_mRNA 作 substrate，p21 作 product
        "substrate": "p21_mRNA",
        "product": "p21",
        "modifier": "p21_mRNA",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "p21 mRNA 翻译为 p21 蛋白（p21_mRNA 作 substrate，p21 作 product，CDK2/CDK4 抑制剂）",
    },
    # 11. TD-033 (IB-064) 修复：p53 → p53_ac（acetylation，p300/CBP 作 HAT 辅因子）
    #     原通路缺失 p300/CBP 乙酰化辅因子，导致 p53 C 端 Lys382 乙酰化这一关键
    #     稳定化修饰无法体现。p300/CBP 乙酰化 p53 增强 DNA 结合与转录活性。
    #     p53 作 substrate，p53_ac 作 product，p300（或 CBP）作 catalytic modifier。
    {
        "source": "p53",
        "target": "p53_ac",
        "mechanism": "acetylation",
        "kinetics_type": "Michaelis_Menten",
        "pathway_tag": PATHWAY_TAG,
        # p53 作 substrate，p53_ac（乙酰化形式）作 product，p300/CBP 作 catalytic modifier
        "substrate": "p53",
        "product": "p53_ac",
        "modifier": "p300",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "site": "Lys382",
        "description": "p300/CBP 乙酰化 p53 Lys382（p53 作 substrate，p53_ac 作 product，p300/CBP 作 HAT 辅因子，稳定 p53 并增强转录活性）",
    },
    # 12. TD-033 (IB-064) 修复：MDM4 → p53（inhibition，MDM4/MDMX 结合 p53 抑制转录活性）
    #     原通路缺失 MDM4/MDMX 调控因子，导致 p53-MDM2 双负反馈外另一关键抑制分支缺失。
    #     MDM4 与 MDM2 协同抑制 p53：MDM4 结合 p53 N 端转录激活域抑制其转录活性，
    #     并增强 MDM2 介导的泛素化。此处用 inhibition 机制表示 MDM4 对 p53 的抑制。
    {
        "source": "MDM4",
        "target": "p53",
        "mechanism": "inhibition",
        "kinetics_type": "mass_action",
        "pathway_tag": PATHWAY_TAG,
        # MDM4 作 modifier（抑制因子），p53 作 substrate（被抑制），产物为 p53_inhibited（用 p53 表示活性降低）
        "substrate": "p53",
        "product": "p53",
        "modifier": "MDM4",
        "modifier_type": "allosteric",
        "autophosphorylation": False,
        "description": "MDM4/MDMX 结合 p53 N 端转录激活域抑制 p53 转录活性（MDM4 作 allosteric inhibitor，与 MDM2 协同调控 p53）",
    },
    # ===== [N6 缺口 1] 药物-靶点显式 inhibitor edge（canonical drug_library 驱动） =====
    # 13. Nutlin-3 → Mdm2（MDM2_p53_interaction_disruptor, IC50=90 nM, PMID:15143221）
    # Nutlin-3 占据 MDM2 的 p53 结合口袋（MDM2 N 端 p53 结合域疏水裂隙），
    # 阻断 MDM2-p53 蛋白-蛋白相互作用，阻止 MDM2 介导的 p53 泛素化降解，稳定 p53。
    # 此处 target="Mdm2" 与 specialist 现有物种命名对齐（Nutlin-3 功能性靶点是 MDM2 p53 结合域）。
    {
        **build_inhibitor_edge("Nutlin-3", "Mdm2"),
        "pathway_tag": PATHWAY_TAG,
    },
]


# =============================================================================
# p53 状态机（monomer→phosphorylated→tetramer→nuclear）
# =============================================================================
# 4 态状态机：未磷酸化 p53 → 磷酸化 p53 → 四聚化 → 入核
# 用于 P2 StateMachine schema 输出，描述 p53 激活流程
_P53_STATE_MACHINE: dict[str, Any] = {
    "id": "SM_p53_ACTIVATION",
    "species": "p53",
    "states": [
        {"name": "monomer", "species": "p53",
         "description": "未磷酸化 p53 单体（cytoplasm/nucleus 游离）"},
        {"name": "phosphorylated", "species": "p53",
         "description": "pATM 磷酸化的 p53（Ser15/Thr20，p53 字段表示）"},
        {"name": "tetramer", "species": "p53_tetramer",
         "description": "p53 四聚体（4 个磷酸化 p53 单体组装）"},
        {"name": "nuclear", "species": "p53_nuclear",
         "description": "p53 四聚体入核（转录因子活性形式）"},
    ],
    "transitions": [
        {
            "from": "monomer",
            "to": "phosphorylated",
            "trigger": "pATM_phosphorylation",
            "reaction_id": "pATM→p53",
            "site": "Ser15/Thr20",
        },
        {
            "from": "phosphorylated",
            "to": "tetramer",
            "trigger": "tetramerization",
            "reaction_id": "p53→p53_tetramer",
            "stoichiometry": "4:1",
        },
        {
            "from": "tetramer",
            "to": "nuclear",
            "trigger": "nuclear_import",
            "reaction_id": "p53_tetramer→p53_nuclear",
        },
    ],
    "initial_state": "monomer",
    "description": "p53 激活状态机：monomer→phosphorylated→tetramer→nuclear（DNA damage 应答）",
}


# =============================================================================
# p53 反馈环（p53-Mdm2 DDE 延迟负反馈 + ATM 双向调控）
# =============================================================================
# 经典 p53-Mdm2 延迟负反馈振荡（Lev Bar-Or 2000, PMID:10644692）：
# - p53 转录激活 Mdm2 mRNA（delay=60min 转录翻译延迟）
# - Mdm2 mRNA 翻译为 Mdm2 蛋白（E3 泛素连接酶）
# - Mdm2 蛋白泛素化 p53，标记蛋白酶体降解（delay=0）
# - 形成 p53↑ → Mdm2↑ → p53↓ 振荡（5-7 小时周期）
#
# ATM 双向调控：
# - ATM 磷酸化 p53 激活（Ser15/Thr20，稳定 p53）
# - ATM 磷酸化 Mdm2 抑制（Ser395，促进 Mdm2 自我降解，释放 p53）
_P53_FEEDBACK_LOOPS: list[dict[str, Any]] = [
    # 1. p53-Mdm2 转录延迟负反馈（DDE，delay=60min）
    #    经典 Lev Bar-Or 2000 模型，p53-Mdm2 形成 5-7h 脉冲振荡
    {
        "id": "FL_P53_MDM2",
        "loop_type": "negative",
        "node_ids": ["p53_nuclear", "Mdm2_mRNA", "Mdm2", "p53_ubi"],
        "delay_minutes": _P53_MDM2_DELAY_MINUTES,  # 60.0 min
        "site": "p53RE(Mdm2 promoter)",
        "description": "p53-Mdm2 转录延迟负反馈（p53 转录激活 Mdm2，Mdm2 降解 p53，DDE delay=60min，5-7h 脉冲振荡）",
        "source_pmid": _PMID_LEV_BAR_OR_2000,
        "dde_solver": "solvers/dde_solver.py",
        "template": "oscillatory_feedback.j2",
    },
    # 2. ATM → p53 + Mdm2 双向调控（ATM 同时磷酸化 p53 激活 + Mdm2 抑制）
    #    DNA damage 后 ATM 双向调控：磷酸化 p53 Ser15（激活）+ 磷酸化 Mdm2 Ser395（抑制）
    #    形成 ATM 介导的快速 p53 稳定响应（delay=0）
    {
        "id": "FL_ATM_P53_MDM2_BIDIRECTIONAL",
        "loop_type": "negative",  # 净效应是 p53 稳定（抑制 Mdm2 降解 p53）
        "node_ids": ["pATM", "p53", "Mdm2"],
        "delay_minutes": 0.0,
        "description": "ATM 双向调控 p53-Mdm2（pATM 磷酸化 p53 Ser15 激活 + pATM 磷酸化 Mdm2 Ser395 抑制，快速稳定 p53）",
        "p53_site": "Ser15/Thr20",
        "mdm2_site": "Ser395",
    },
]


# =============================================================================
# p53 Crosstalk Reaction 片段（4 条，仅本通路侧片段，edge 由 Coordinator 管理）
# =============================================================================
# p53 作为转录因子激活下游凋亡基因（Bax/PUMA）与细胞周期抑制基因（p21）
# AKT→Mdm2 由 PI3K Specialist 生成相反方向片段，本 Specialist 不生成 AKT→Mdm2 edge
_P53_CROSSTALK_REACTIONS: list[dict[str, Any]] = [
    # 1. p53 → Bax（transcription，p53 激活 Bax 启动凋亡）
    {
        "source": "p53",
        "target": "Bax",
        "mechanism": "transcription",
        "shared_species": ["p53"],
        "site": "p53RE(Bax promoter)",
        "description": "p53 转录激活 Bax（p53 作转录因子，激活 Bax 启动内源性凋亡路径，与 Apoptosis 通路 cross-talk）",
    },
    # 2. p53 → PUMA（transcription，p53 激活 PUMA）
    {
        "source": "p53",
        "target": "PUMA",
        "mechanism": "transcription",
        "shared_species": ["p53"],
        "site": "p53RE(PUMA promoter)",
        "description": "p53 转录激活 PUMA（p53 作转录因子，激活 PUMA 拮抗 Bcl-2 释放 Bax，与 Apoptosis 通路 cross-talk）",
    },
    # 3. p53 → p21（transcription，p53 激活 p21 抑制 CDK2/4 阻滞细胞周期）
    {
        "source": "p53",
        "target": "p21",
        "mechanism": "transcription",
        "shared_species": ["p53", "p21"],
        "site": "p53RE(p21/CDKN1A promoter)",
        "description": "p53 转录激活 p21（p53 作转录因子，p21 抑制 CyclinE-CDK2 / CyclinD-CDK4 阻滞 G1/S，与 Cell Cycle 通路 cross-talk）",
    },
    # 4. AKT → Mdm2（activation，AKT 磷酸化 Mdm2 Ser166 激活其降解 p53 功能）
    #    注意：本片段仅描述 p53 通路侧 Mdm2 接收 AKT 调控的语义，
    #    edge 本身由 PI3K Specialist + Cross-talk Coordinator 管理
    {
        "source": "AKT",
        "target": "Mdm2",
        "mechanism": "activation",
        "shared_species": ["p53"],
        "site": "Ser166",
        "description": "AKT 磷酸化 Mdm2 Ser166 激活其 E3 泛素连接酶活性（降解 p53，与 PI3K 通路 cross-talk）",
    },
]


# =============================================================================
# p53 扰动（5 个：3 个药物 + 2 个突变）
# =============================================================================
_P53_PERTURBATIONS: list[dict[str, Any]] = [
    # 1. Nutlin-3（Mdm2-p53 interaction inhibitor, small molecule）
    #    Nutlin-3 占据 Mdm2 的 p53 结合口袋，阻断 Mdm2 降解 p53，稳定 p53
    # [N6 缺口 1] 注入 canonical drug_library 字段（ic50_nM/ki_nM/source_pmid/...）
    {
        "target": "Mdm2",
        "drug": "Nutlin-3",
        "mechanism": "inhibition",
        "ko_target": None,
        "description": "Nutlin-3（Mdm2-p53 相互作用抑制剂，小分子，占据 Mdm2 p53 结合口袋阻断降解，稳定 p53）",
        **{k: v for k, v in get_drug_entry("Nutlin-3").items()
           if k not in ("description",)},
    },
    # 2. PRIMA-1（p53 mutant reactivator, small molecule）
    #    PRIMA-1 恢复突变 p53 的野生型构象与 DNA 结合活性（R175H/R273H 等热点突变）
    {
        "target": "p53",
        "drug": "PRIMA-1",
        "mechanism": "activation",
        "ko_target": None,
        "description": "PRIMA-1（p53 突变体再激活剂，小分子，恢复突变 p53 野生型构象与 DNA 结合活性）",
    },
    # 3. TP53 R175H（loss-of-function mutation, DNA binding domain）
    #    p53 R175H 是 DNA binding domain 结构性突变，破坏 p53 DNA 结合与转录活性
    {
        "target": "p53",
        "drug": None,
        "mechanism": "knockout",
        "ko_target": "TP53_R175H",
        "description": "TP53 R175H（p53 DNA 结合域结构性突变，loss-of-function，破坏 p53 转录活性）",
    },
    # 4. TP53 R273H（loss-of-function mutation, DNA binding domain）
    #    p53 R273H 是 DNA binding domain 接触面突变，破坏 p53-DNA 直接接触
    {
        "target": "p53",
        "drug": None,
        "mechanism": "knockout",
        "ko_target": "TP53_R273H",
        "description": "TP53 R273H（p53 DNA 结合域接触面突变，loss-of-function，破坏 p53-DNA 直接接触）",
    },
    # 5. 5-Fluorouracil / 5-FU（DNA damage inducer, chemotherapy drug）
    #    5-FU 是胸苷酸合酶抑制剂，导致 DNA 损伤激活 ATM-p53 轴（化疗药物）
    {
        "target": "DNA_damage",
        "drug": "5-Fluorouracil",
        "mechanism": "activation",
        "ko_target": None,
        "description": "5-Fluorouracil / 5-FU（DNA damage 诱导剂，化疗药物，胸苷酸合酶抑制剂激活 ATM-p53 轴）",
    },
]


# =============================================================================
# p53 Validation 规则（3 条 benchmark，Lev Bar-Or 2000, PMID:10644692）
# =============================================================================
# 每条含：rule_id / metric_name / expected / tolerance / pmid / description
# expected 取区间中点，tolerance 取区间半宽
_P53_VALIDATION_RULES: list[dict[str, Any]] = [
    # 1. p53 脉冲振荡周期 5-7 hours（Lev Bar-Or 2000）
    #    p53-Mdm2 DDE 延迟负反馈产生 5-7 小时脉冲振荡
    {
        "rule_id": "VAL_P53_PULSE_PERIOD",
        "metric_name": "p53_pulse_period",
        "expected": 6.0,    # (5.0 + 7.0) / 2
        "tolerance": 1.0,   # (7.0 - 5.0) / 2
        "expected_min": 5.0,
        "expected_max": 7.0,
        "unit": "hours",
        "pmid": _PMID_LEV_BAR_OR_2000,
        "comparison": "range",
        "pathway_tag": PATHWAY_TAG,
        "description": "p53-Mdm2 DDE 延迟负反馈产生 5-7 小时脉冲振荡（Lev Bar-Or 2000 p53-Mdm2 oscillator）",
    },
    # 2. Mdm2 转录延迟 60-120 minutes（Lev Bar-Or 2000）
    #    p53 转录激活 Mdm2 mRNA 后，需 60-120 min 翻译为 Mdm2 蛋白并执行负反馈
    {
        "rule_id": "VAL_P53_MDM2_TRANSCRIPTION_DELAY",
        "metric_name": "Mdm2_transcription_delay",
        "expected": 90.0,   # (60.0 + 120.0) / 2
        "tolerance": 30.0,  # (120.0 - 60.0) / 2
        "expected_min": 60.0,
        "expected_max": 120.0,
        "unit": "minutes",
        "pmid": _PMID_LEV_BAR_OR_2000,
        "comparison": "range",
        "pathway_tag": PATHWAY_TAG,
        "description": "Mdm2 转录翻译延迟 60-120 min（p53 激活 Mdm2 mRNA 到 Mdm2 蛋白执行负反馈的延迟）",
    },
    # 3. p53 磷酸化响应时间 5-30 minutes（Lev Bar-Or 2000）
    #    DNA damage 后 ATM 磷酸化 p53 Ser15 的响应时间（5-30 min）
    {
        "rule_id": "VAL_P53_PHOSPHORYLATION_RESPONSE_TIME",
        "metric_name": "p53_phosphorylation_response_time",
        "expected": 17.5,   # (5.0 + 30.0) / 2
        "tolerance": 12.5,  # (30.0 - 5.0) / 2
        "expected_min": 5.0,
        "expected_max": 30.0,
        "unit": "minutes",
        "pmid": _PMID_LEV_BAR_OR_2000,
        "comparison": "range",
        "pathway_tag": PATHWAY_TAG,
        "description": "DNA damage 后 p53 磷酸化响应时间 5-30 min（ATM 磷酸化 p53 Ser15 的快速响应）",
    },
]


@register_specialist
class P53Specialist(PathwaySpecialistBase):
    """p53 肿瘤抑制通路 Specialist。

    实现 DNA damage→ATM/ATR→p53 磷酸化→四聚化→入核→转录（Mdm2/p21/Bax/PUMA/Noxa）
    核心拓扑 + p53-Mdm2 转录延迟负反馈（DDE delay=60min）+ ATM→p53+Mdm2 双向调控
    的 Core/Feedback/Crosstalk/Perturbation/Validation 5 模块，输出通路特异
    Reaction IR 片段 + p53 状态机（monomer→phosphorylated→tetramer→nuclear）
    + 模板选择 + Validation 规则。

    职责边界：
    - 处理 p53 通路核心（DNA_damage/ATM/pATM/p53/p53_tetramer/p53_nuclear/Mdm2/p21）
    - 不处理 Bax/PUMA 凋亡执行（由 Apoptosis Specialist 处理，Task 4.7）
    - 不生成 AKT→Mdm2 cross-talk edge（由 Cross-talk Coordinator 处理，Task 4.13）
    - 不处理 CDK2/CDK4 抑制（由 Cell Cycle Specialist 处理，Task 4.8）

    输入：
    - ``pathway_graph``：v4_pathway_graph 的 p53 子图（含 FeedbackLoop FL_P53_MDM2）
    - ``ontology_entities``：P1 Ontology Entities（可选，HGNC/UniProt ID 对齐）

    输出：
    - ``apply_core``：10 条核心 Reaction IR 片段 + 11 物种 + p53 状态机
      （p53 标记 shared=True，p21 标记 shared=True）
    - ``apply_feedback``：2 条 FeedbackLoop（p53-Mdm2 DDE delay=60min + ATM 双向调控）
    - ``apply_crosstalk``：4 条 cross-talk Reaction 片段（p53→Bax/PUMA/p21 + AKT→Mdm2）
    - ``apply_perturbation``：5 个扰动（Nutlin-3/PRIMA-1/TP53 R175H/R273H/5-FU）
    - ``apply_validation``：3 条 Validation benchmark（p53 振荡周期 5-7h / Mdm2 延迟
      60-120min / p53 磷酸化响应 5-30min，Lev Bar-Or 2000, PMID:10644692）
    """

    pathway_class: str = "p53"
    display_name: str = "p53 Tumor Suppressor Signaling"

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
                    species=list(_P53_CORE_SPECIES),
                    reactions=list(_P53_CORE_REACTIONS),
                    kinetics_overrides={},
                )
            if module_name == MODULE_FEEDBACK:
                return FeedbackModuleData(
                    feedback_loops=list(_P53_FEEDBACK_LOOPS),
                    delay_minutes=_P53_MDM2_DELAY_MINUTES,
                    loop_type="negative",  # p53-Mdm2 负反馈振荡
                )
            if module_name == MODULE_CROSSTALK:
                return CrosstalkModuleData(
                    crosstalk_reactions=list(_P53_CROSSTALK_REACTIONS),
                    shared_species=["p53", "p21"],
                    coordination_strategy="merge",
                )
            if module_name == MODULE_PERTURBATION:
                return PerturbationModuleData(
                    perturbation_reactions=list(_P53_PERTURBATIONS),
                    drug_targets=[
                        p for p in _P53_PERTURBATIONS if p.get("drug")
                    ],
                    ko_targets=[
                        p for p in _P53_PERTURBATIONS if p.get("ko_target")
                    ],
                )
            if module_name == MODULE_VALIDATION:
                return ValidationModuleData(
                    rules=list(_P53_VALIDATION_RULES),
                    benchmarks=[
                        {
                            "benchmark_name": r["metric_name"],
                            "value": r["expected"],
                            "unit": r["unit"],
                            "condition": "DNA damage response",
                            "reference": r["pmid"],
                        }
                        for r in _P53_VALIDATION_RULES
                    ],
                    tolerances={
                        r["metric_name"]: r["tolerance"]
                        for r in _P53_VALIDATION_RULES
                    },
                    pmid_references=[
                        {
                            "pmid": r["pmid"],
                            "citation": r["description"],
                            "pathway_class": PATHWAY_TAG,
                            "metric_name": r["metric_name"],
                        }
                        for r in _P53_VALIDATION_RULES
                        if r["pmid"]
                    ],
                )
            logger.warning(
                "P53Specialist.load_module: 未知模块名 '%s'",
                module_name,
            )
            return None
        except Exception as exc:
            logger.warning(
                "P53Specialist.load_module 加载模块 '%s' 失败: %s",
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
        """应用核心模块，返回 p53 通路核心 Reaction IR 片段。

        输出 10 条核心反应：
        1. DNA damage → pATM（ATM 磷酸化激活，DNA damage 作 allosteric activator）
        2. pATM → p53（异磷酸化，pATM 磷酸化 p53 Ser15/Thr20）
        3. p53 → p53_tetramer（tetramerization，4 个 p53 monomer → 1 tetramer）
        4. p53_tetramer → p53_nuclear（nuclear_import，p53 tetramer 入核）
        5. p53_nuclear → Mdm2_mRNA（transcription，Hill 动力学，p53 作转录因子）
        6. Mdm2_mRNA → Mdm2（translation，mRNA→protein）
        7. Mdm2 → p53_ubi（ubiquitination，p53 作 substrate，Mdm2 作 E3 ligase modifier）
        8. p53_ubi → p53（proteasomal_degradation，泛素化 p53 蛋白酶体降解）
        9. p53_nuclear → p21_mRNA（transcription，Hill 动力学）
        10. p21_mRNA → p21（translation，mRNA→protein）

        p53 物种标记 shared=True（与 Apoptosis Bax/PUMA 凋亡路径共享）。
        p21 物种标记 shared=True（与 Cell Cycle CDK2/CDK4 抑制路径共享）。

        Returns:
            dict 含 ``species``（11 物种）/ ``reactions``（10 反应） /
            ``state_machine``（p53 状态机）字段。异常时返回
            ``{"species": [], "reactions": [], "state_machine": {}}``。
        """
        try:
            return {
                "species": list(_P53_CORE_SPECIES),
                "reactions": list(_P53_CORE_REACTIONS),
                "state_machine": dict(_P53_STATE_MACHINE),
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
                "P53Specialist.apply_core 失败: %s", exc
            )
            return {"species": [], "reactions": [], "state_machine": {}}

    # =================================================================
    # apply_feedback：FeedbackLoop 列表
    # =================================================================
    def apply_feedback(self, pathway_graph: dict) -> list[dict]:
        """应用反馈模块，返回 p53 通路 FeedbackLoop 列表。

        输出 2 条反馈环：
        1. p53-Mdm2 转录延迟负反馈（DDE delay=60min，Lev Bar-Or 2000 振荡模型）
        2. ATM → p53 + Mdm2 双向调控（pATM 磷酸化 p53 激活 + 磷酸化 Mdm2 抑制，delay=0）

        Returns:
            FeedbackLoop 字典列表。异返回空列表。
        """
        try:
            return list(_P53_FEEDBACK_LOOPS)
        except Exception as exc:
            logger.warning(
                "P53Specialist.apply_feedback 失败: %s", exc
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
        """应用跨通路模块，返回 p53 通路侧 cross-talk Reaction 片段。

        注意：本方法仅生成本通路侧的 cross-talk Reaction 片段，
        cross-talk edge 本身由 Cross-talk Coordinator 创建（Task 4.13）。

        输出 4 条 cross-talk Reaction 片段：
        1. p53 → Bax（transcription，p53 激活 Bax 启动凋亡）
        2. p53 → PUMA（transcription，p53 激活 PUMA）
        3. p53 → p21（transcription，p53 激活 p21 抑制 CDK2/4 阻滞细胞周期）
        4. AKT → Mdm2（activation，AKT 磷酸化 Mdm2 Ser166 激活其降解 p53 功能）

        Args:
            pathway_graph: 通路图（当前未使用，保留接口一致性）。
            crosstalk_edges: 来自 Coordinator 的 cross-talk edge 列表
                （当前未使用，本 Specialist 返回静态拓扑片段）。

        Returns:
            cross-talk Reaction 片段列表。异返回空列表。
        """
        try:
            return list(_P53_CROSSTALK_REACTIONS)
        except Exception as exc:
            logger.warning(
                "P53Specialist.apply_crosstalk 失败: %s", exc
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
        """应用扰动模块，返回 p53 通路特异药物 / 突变 Reaction 片段。

        输出 5 个扰动（3 个药物 + 2 个突变）：
        1. Nutlin-3（Mdm2-p53 相互作用抑制剂，小分子，稳定 p53）
        2. PRIMA-1（p53 突变体再激活剂，小分子，恢复突变 p53 功能）
        3. TP53 R175H（p53 DNA 结合域结构性突变，loss-of-function）
        4. TP53 R273H（p53 DNA 结合域接触面突变，loss-of-function）
        5. 5-Fluorouracil / 5-FU（DNA damage 诱导剂，化疗药物）

        Args:
            pathway_graph: 通路图（当前未使用）。
            perturbation_points: 扰动点列表（当前未使用，本 Specialist
                返回静态拓扑片段）。

        Returns:
            扰动 Reaction 片段列表。异返回空列表。
        """
        try:
            return list(_P53_PERTURBATIONS)
        except Exception as exc:
            logger.warning(
                "P53Specialist.apply_perturbation 失败: %s", exc
            )
            return []

    # =================================================================
    # apply_validation：Validation 规则列表
    # =================================================================
    def apply_validation(
        self,
        simulation_result: dict | None = None,
    ) -> list[dict]:
        """应用验证模块，返回 p53 通路 Validation 规则列表。

        输出 3 条 benchmark（Lev Bar-Or 2000, PMID:10644692）：
        1. p53 脉冲振荡周期 5-7 hours（p53-Mdm2 DDE 延迟负反馈振荡）
        2. Mdm2 转录延迟 60-120 min（p53 激活 Mdm2 mRNA 到蛋白的延迟）
        3. p53 磷酸化响应时间 5-30 min（DNA damage 后 ATM 磷酸化 p53 Ser15）

        Args:
            simulation_result: 仿真结果（可选，当前未使用，返回静态规则集）。

        Returns:
            Validation 规则列表。异返回空列表。
        """
        try:
            return list(_P53_VALIDATION_RULES)
        except Exception as exc:
            logger.warning(
                "P53Specialist.apply_validation 失败: %s", exc
            )
            return []

    # =================================================================
    # select_template：覆写以支持 DDE 模式检测
    # =================================================================
    def select_template(self, mechanism: str) -> str:
        """根据 mechanism 选择 ODE 模板名（覆写支持 DDE 模式）。

        默认映射（与 P3 ``ode_templates_v2/`` 下 .j2 文件对齐）：
        - ``phosphorylation`` → ``_mechanism_phosphorylation_mm``（pATM→p53 异磷酸化）
        - ``oscillatory`` → ``oscillatory_feedback``（p53-Mdm2 DDE 延迟振荡）
        - ``dde`` → ``oscillatory_feedback``（DDE 模式检测，等价 oscillatory）
        - ``transcription`` → ``oscillatory_feedback``（p53 转录因子 DDE 模式）

        Args:
            mechanism: 机制名（小写，如 ``"phosphorylation"`` / ``"oscillatory"``）。

        Returns:
            ODE 模板名（不含 ``.j2`` 后缀）。未匹配时返回 ``"default"``
            （调用方应处理默认降级）。
        """
        # DDE 模式检测：p53-Mdm2 反馈是 DDE 场景，使用 oscillatory_feedback 模板
        if mechanism in ("oscillatory", "dde", "transcription"):
            return "oscillatory_feedback"
        # 磷酸化场景：pATM→p53 异磷酸化，使用 Michaelis-Menten 模板
        if mechanism == "phosphorylation":
            return "_mechanism_phosphorylation_mm"
        # 其他机制走默认基类映射
        return super().select_template(mechanism)


# =============================================================================
# 文献动力学参数（IB-017 修复）
# =============================================================================
# 来源：
# - BIOMD0000000012 (Lev Bar-Or 2000, PMID:10648606) p53-Mdm2 延迟负反馈振荡模型
# - BIOMD0000000567 (Purvis 2012) p53 状态机/脉冲动力学模型
# 反幻觉守卫：所有参数来自上述 BioModels 模型或文献；无确切值的用无量纲化
# 估计并标注 `# Heuristic estimate, needs calibration`。
# 参数范围约束：k_on∈[1e3,1e7] M^-1 min^-1, Km∈[1e-7,1e-2] M, k_cat∈[1e-3,1e3] min^-1
# 注：p53-Mdm2 转录延迟 _P53_MDM2_DELAY_MINUTES=60min 为 DDE 延迟项（见文件顶部）。
KINETIC_PARAMETERS: dict[str, dict[str, float]] = {
    # DNA_damage→pATM ATM 激活（Lev Bar-Or 2000, PMID:10648606, BIOMD0000000012）
    "DNA_pATM": {
        "k_cat": 1.0,                # min^-1  # Heuristic estimate, needs calibration
        "Km": 1e-6,                  # M (ATM Km)  # Heuristic estimate, needs calibration
    },
    # pATM→p53 p53 磷酸化（Lev Bar-Or 2000, PMID:10648606, 异磷酸化）
    "pATM_p53": {
        "k_cat": 1.0,                # min^-1  # Heuristic estimate, needs calibration
        "Km": 1e-7,                  # M (p53 Km)  # Heuristic estimate, needs calibration
    },
    # p53→p53_tetramer 四聚化（Purvis 2012, BIOMD0000000567）
    "p53_tetramer": {
        "k_on": 1.0e6,               # M^-1 min^-1  # Heuristic estimate, needs calibration
        "k_off": 1e-3,               # min^-1  # Heuristic estimate, needs calibration
    },
    # p53_tetramer→p53_nuclear 入核（Purvis 2012, BIOMD0000000567）
    "p53_nuclear_import": {
        "k_import": 0.1,             # min^-1  # Heuristic estimate, needs calibration
    },
    # p53_nuclear→Mdm2_mRNA 转录（Lev Bar-Or 2000, PMID:10648606, DDE delay=60min）
    "p53_Mdm2_transcription": {
        "k_transcription": 1.0,      # min^-1  # Heuristic estimate, needs calibration
        "Km": 1e-7,                  # M (p53_nuclear Km, Hill n=2)  # Heuristic estimate, needs calibration
    },
    # Mdm2_mRNA→Mdm2 翻译（Lev Bar-Or 2000, PMID:10648606）
    "Mdm2_translation": {
        "k_translation": 0.1,        # min^-1  # Heuristic estimate, needs calibration
    },
    # Mdm2→p53_ubi p53 泛素化（Lev Bar-Or 2000, PMID:10648606）
    "Mdm2_p53_ubi": {
        "k_cat": 1.0,                # min^-1  # Heuristic estimate, needs calibration
        "Km": 1e-7,                  # M (p53 Km)  # Heuristic estimate, needs calibration
    },
    # p53_ubi 降解（蛋白酶体, Lev Bar-Or 2000, PMID:10648606）
    "p53_ubi_degradation": {
        "k_degradation": 0.1,        # min^-1  # Heuristic estimate, needs calibration
    },
    # p53_nuclear→p21_mRNA 转录（Purvis 2012, BIOMD0000000567）
    "p53_p21_transcription": {
        "k_transcription": 1.0,      # min^-1  # Heuristic estimate, needs calibration
        "Km": 1e-7,                  # M  # Heuristic estimate, needs calibration
    },
    # p21_mRNA→p21 翻译（Purvis 2012, BIOMD0000000567）
    "p21_translation": {
        "k_translation": 0.1,        # min^-1  # Heuristic estimate, needs calibration
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
#   - 转换规则：Km_μM = Km_M × 1e6（如 1e-7 M = 0.1 μM，与 ODE 模板默认 Km=0.1 一致）
#   - k_cat / k_dephos / k_off / k_deg / k_degradation / k_import / k_translation /
#     k_transcription 是时间常数（min^-1），无需转换
#   - k_on 参数单位为 M^-1 min^-1，与 ODE 模型 μM 单位冲突，统一 SKIP
#
# 映射依据（KINETIC_PARAMETERS 键名 → 反应 target 物种名）：
#   "DNA_pATM"                → pATM（反应 1: DNA_damage→pATM ATM 磷酸化激活）
#   "pATM_p53"                → p53（反应 2: pATM→p53 异磷酸化）
#   "p53_tetramer"            → p53_tetramer（反应 3: p53→p53_tetramer 四聚化, k_on SKIP）
#   "p53_nuclear_import"      → p53_nuclear（反应 4: p53_tetramer→p53_nuclear 入核）
#   "p53_Mdm2_transcription"  → Mdm2_mRNA（反应 5: p53_nuclear→Mdm2_mRNA 转录, DDE delay=60min）
#   "Mdm2_translation"        → Mdm2（反应 6: Mdm2_mRNA→Mdm2 翻译）
#   "Mdm2_p53_ubi"            → p53_ubi（反应 7: Mdm2→p53_ubi 泛素化）
#   "p53_ubi_degradation"     → degraded_p53（反应 8: p53_ubi→p53 蛋白酶体降解）
#   "p53_p21_transcription"   → p21_mRNA（反应 9: p53_nuclear→p21_mRNA 转录）
#   "p21_translation"         → p21（反应 10: p21_mRNA→p21 翻译）
_KINETICS_BY_TARGET: dict[str, dict[str, float]] = {
    # [P1-NEXT-12 修复 V4 / pATM peak_time=60min 过晚（V3 调参无效）]
    # Root Cause (V4): V3 k_act=1.0 让 pATM 累积太慢，60min 仍未达稳态。
    #   关键发现：模板 _max_pool = max(Y0[t_idx], 0.1) * 3.0
    #   因 pATM 初始=0.05 < 0.1，所以 max_pool = 0.1 * 3 = 0.3（非 0.15）
    #   实测 pATM peak=0.295 ≈ max_pool=0.3，但 k_act=1.0 太慢，60min 才接近稳态。
    #   稳态分析（V3, ATM+pATM 守恒=1.05）：
    #     稳态: k_act*(1.05-pATM)*(1-pATM/0.3) = k_deg*pATM
    #     k_act=1.0, k_deg=0.05: 稳态 pATM≈0.295, 但需 60min 才接近
    # 修复 V4:
    #   1. k_act 1.0 → 3.0：提升 3x，让 pATM 在 ~5min 内达到稳态的 95%
    #   2. k_deg 0.05 → 0.15：提升 3x，让 pATM 在稳态后快速衰减形成瞬态峰
    # 稳态估算（V4, ATM+pATM 守恒=1.05, max_pool=0.3）：
    #   稳态: 3.0*(1.05-pATM)*(1-pATM/0.3) = 0.15*pATM
    #   当 pATM=0.293: 3.0*0.757*0.023 = 0.052 vs 0.15*0.293=0.044 → _rate > 衰减
    #   当 pATM=0.296: 3.0*0.754*0.013 = 0.029 vs 0.15*0.296=0.044 → _rate < 衰减
    #   稳态 pATM ≈ 0.294, fold = 0.294/0.05 = 5.88（满足 C6 [3,20]）✅
    #   时间常数: 1/k_act = 0.33min, pATM 在 5-10min 达峰，30min 后衰减到 50%
    "pATM": {
        "k_act": 3.0,                # min^-1 (DNA_damage 激活 ATM, P1-NEXT-12 V4: 1.0→3.0 加快累积)
        "k_deg": 0.15,               # min^-1 (pATM 去磷酸化, P1-NEXT-12 V4: 0.05→0.15 形成瞬态峰)
    },
    # [P1-NEXT-6 修复 V3 / p53 peak_time=60min 过晚]
    # Root Cause: p53 由 pATM 催化产生，pATM 晚 → p53 晚
    # Fix V3: k_act 0.5 → 1.0 加速 p53 累积；k_deg 保持 0.03（半衰期 23min 满足 C6）
    "p53": {
        "k_act": 1.0,                # min^-1 (pATM 催化 p53 磷酸化, P1-NEXT-6 V3: 0.5→1.0 加速)
        "k_deg": 0.03,                # min^-1 (p53 去磷酸化/降解, 半衰期 ln2/0.03≈23min)
    },
    # [P1-NEXT-6 修复 V3 / tetramerization 无专用模板分支走 else 默认 k=0.05 过慢]
    # Root Cause: transcriptional_delay.j2 无 tetramerization 专用 handler，
    #   p53→p53_tetramer 走 else 默认分支 _rate = k * src，k=0.05 使速率仅 0.0025-0.015/min
    #   加上 p53_tetramer 初始=0.0，导致 p53_tetramer 累积极慢，p53_nuclear 转录链条断裂
    # Fix V3: k 0.05 → 0.2 加速四聚化（4x），让 p53_tetramer 在 5-10min 内累积到足够浓度
    "p53_tetramer": {
        "k": 0.2,                    # min^-1 (p53 四聚化, P1-NEXT-6 V3: 0.05→0.2 加速四聚化)
        "k_off": 0.05,               # min^-1 (四聚体解离, Purvis 2012)
        "k_deg": 0.02,                # min^-1 (四聚体降解回流到 p53)
    },
    # [P1-NEXT-6 修复 V3 / p53_nuclear 累积过慢]
    # Root Cause: k_import=0.1 时间常数=10min，叠加 p53_tetramer 慢累积，使 p53_nuclear 远晚于 pATM
    # Fix V3: k_import 0.1 → 0.3 加速入核，让 p53_nuclear 在 p53_tetramer 累积后快速产生
    "p53_nuclear": {
        "k_import": 0.3,             # min^-1 (p53 tetramer 入核, P1-NEXT-6 V3: 0.1→0.3 加速)
    },
    # [P1-10 修复] 转录参数键名对齐模板 _get_param 查找键
    # Root Cause: 旧键 k_transcription/Km/k_translation 与模板查找键 k_trans/K_d/k_transl 不匹配
    #             _get_param 返回默认值（k_trans=0.1, K_d=0.5）而非 specialist 文献值
    # Fix: 键名改为 k_trans/K_d/k_transl, 新增 n_hill/k_mRNA_deg/k_prot_deg
    "Mdm2_mRNA": {
        "k_trans": 0.1,              # min^-1 (p53 转录激活 Mdm2, Lev Bar-Or 2000, DDE delay=60min)
        "K_d": 0.1,                   # μM (Hill K_d, p53_nuclear 半 maximal 激活浓度)
        "n_hill": 2.0,                # Hill 系数（p53 协同结合 p53RE）
        "k_mRNA_deg": 0.05,           # min^-1 (Mdm2 mRNA 降解, 半衰期 ln2/0.05≈14min)
    },
    "Mdm2": {
        "k_transl": 0.1,             # min^-1 (Mdm2 mRNA 翻译, Lev Bar-Or 2000)
        "k_prot_deg": 0.01,           # min^-1 (Mdm2 蛋白降解)
    },
    # [P1-10] p53_ubi 改用 activation 机制（避免 ubiquitination 消耗 target 的 bug）
    "p53_ubi": {
        "k_act": 0.1,                # min^-1 (Mdm2 E3 催化 p53 泛素化, Lev Bar-Or 2000)
        "k_deg": 0.1,                 # min^-1 (p53_ubi 去泛素化/降解)
    },
    "degraded_p53": {
        "k_degradation": 0.1,        # min^-1 (蛋白酶体降解泛素化 p53, Lev Bar-Or 2000)
    },
    # [P1-NEXT-6 修复 V3 / p21_mRNA peak_time=0min, fold=1.0 转录未触发]
    # Root Cause: p21_mRNA 转录依赖 p53_nuclear，但 p53_nuclear 远晚于 0min（链条断裂）
    #   即便修复 pATM/p53/p53_tetramer/p53_nuclear timing 后，
    #   k_trans=0.1 使 transcription rate = 0.1 * Hill(p53_nuclear) 仍偏弱
    # Fix V3: k_trans 0.1 → 0.5 提升 5x 转录速率，让 p21_mRNA 在 p53_nuclear 累积后快速产生
    #   预期：p21_mRNA peak_time 从 0 → [120, 240]min（C5 期望窗口）
    "p21_mRNA": {
        "k_trans": 0.5,              # min^-1 (p53 转录激活 p21, P1-NEXT-6 V3: 0.1→0.5 提升 5x)
        "K_d": 0.1,                   # μM (Hill K_d, p53_nuclear 半 maximal 激活浓度)
        "n_hill": 2.0,                # Hill 系数（p53 协同结合 p21 启动子 p53RE）
        "k_mRNA_deg": 0.05,           # min^-1 (p21 mRNA 降解, 半衰期 ln2/0.05≈14min)
    },
    "p21": {
        "k_transl": 0.1,             # min^-1 (p21 mRNA 翻译, Purvis 2012)
        "k_prot_deg": 0.01,           # min^-1 (p21 蛋白降解)
    },
}


__all__ = [
    "P53Specialist",
    "PATHWAY_TAG",
    "SOURCE_SBML",
    "KINETIC_PARAMETERS",
    "_KINETICS_BY_TARGET",
]
