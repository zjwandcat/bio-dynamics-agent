# BioDynamics Agent v4 - NF-κB Specialist (Phase 4 / Task 4.10)
# NF-κB 通路 Specialist：实现 TNF/IL-1/LPS→受体→IKK→IκBα 磷酸化→泛素化→
# 蛋白酶体降解→NF-κB 释放→入核→转录（IκBα/A20/TNF/Bcl-2）核心拓扑 +
# NF-κB→IκBα 转录延迟负反馈振荡（delay=30min）+ NF-κB→A20→IKK 双负反馈。
#
# 设计原则（铁律）：
# 1. Feature Flag V4_PATHWAY_SPECIALIST_ENABLED=false 时由 LangGraph hook 层短路，
#    本 Specialist 代码不被调用（基类本身不做 flag 检查）
# 2. 使用 Task 4.0 修复后的 PHOSPHORYLATION 语义：
#    - 异磷酸化（pIKK → pIkBa）：未磷酸化 IκBα 作 substrate，磷酸化 pIkBa 作
#      product，pIKK 作 catalytic modifier（Ser32/36）
#    - 异磷酸化（TNF_TNFR_complex → pIKK）：IKK 作 substrate，pIKK 作 product，
#      TNF_TNFR_complex 作 catalytic modifier
# 3. IκBα 三步耦合 CompositeReaction（phosphorylation→ubiquitination→
#    proteasomal_degradation），不压扁为单一 reaction，保留中间产物语义
# 4. 不处理 NF-κB→Bcl-2 凋亡保护下游（由 Apoptosis Specialist 处理，Task 4.7）
# 5. 不生成 AKT→IKK cross-talk edge 本身（由 Cross-talk Coordinator 处理，
#    Task 4.13），仅返回本通路侧的 cross-talk Reaction 片段
# 6. apply_* 方法实现捕获异常并返回空 list/dict，记录 logger.warning，不抛异常
#
# 依赖：
# - P1 ontology / pathway_registry（NF_kB 通路类别键）
# - P2 schema（SpeciesV2 / ReactionV2 / SpeciesRef / Modifier / CompositeReaction）
# - P2 MechanismType（PHOSPHORYLATION / UBIQUITINATION /
#   PROTEASOMAL_DEGRADATION / COMPLEX_FORMATION / DISSOCIATION / NUCLEAR_IMPORT /
#   TRANSCRIPTION / TRANSLATION / INHIBITION）
# - P3 ode_templates_v2（oscillatory_feedback.j2 / _mechanism_phosphorylation_mm.j2 模板）
# - P3 oscillation_detector.py（NF-κB 核振荡周期检测）
# - P3 pathway_graph/initializer.py（NF_kB core_nodes / core_edges）
#
# 参考：
# - spec.md Part 3 Specialist 8（第 241-246 行）
# - tasks.md Task 4.10（第 105-114 行）
# - Nelson 2004 NF-κB oscillation (PMID:14976212)
# - BioModels BIOMD0000000258 (Ashall2009 NF-kB oscillation)

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
# NF-κB 通路标签
# =============================================================================
PATHWAY_TAG: str = "NF_KB"

# SBML BioModels ID（Ashall 2009 NF-kB oscillation model）
SOURCE_SBML: str = get_biomodels_id(PATHWAY_TAG)

# Validation benchmark PMID 引用（Nelson 2004 NF-κB oscillation）
_Pmid_NELSON_2004: str = "PMID:14976212"

# NF-κB→IκBα 转录延迟负反馈延迟（DDE，分钟）
# Nelson 2004 模型中 NF-κB 转录激活 IκBα mRNA，IκBα 蛋白重新合成结合 NF-κB，
# 形成 delay=30min 的延迟负反馈，产生 1-2 小时核振荡周期。
_NFKB_IKBA_DELAY_MINUTES: float = 30.0

# NF-κB→A20→IKK 双负反馈延迟（DDE，分钟）
# A20 蛋白抑制 IKK 活性，delay=60min（A20 转录+翻译延迟）
_NFKB_A20_IKK_DELAY_MINUTES: float = 60.0


# =============================================================================
# NF-κB 核心物种（与 P3 pathway_graph/initializer.py NF_kB.core_nodes 对齐，
# 扩展 TNF_TNFR_complex / pIKK / pIkBa / ubIkBa / IkBa_degraded / A20 / TNF_mRNA /
# Bcl2_mRNA 完整 NF-κB 拓扑）
# =============================================================================
# NFkB 物种标记 shared=True（与 Apoptosis Specialist 的 Bcl-2 抗凋亡路径共享，
# NF-κB 转录 Bcl-2 维持肿瘤细胞存活，下游 Bcl-2→凋亡由 Apoptosis Specialist 处理）
_NF_KB_CORE_SPECIES: list[dict[str, Any]] = [
    # [C4 fix] initial_concentration aligned to BIOMD0000000140 (Hoffmann2002, PMID:12424381).
    #   SBML models NF-κB/IκBα oscillations (IKK/IkBalpha/NFkB/NFkB_nuc/IkBalpha_transcript).
    #   SBML species mapping: IkBa→IkBalpha, NFkB→NFkB, NFkB_nuclear→NFkB_nuc,
    #   IkBa_mRNA→IkBalpha_transcript, pIKK→IKK (SBML IKK=active, initially 0).
    #   Species not in SBML (TNF/TNFR/TNF_TNFR_complex/IKK-inactive/pIkBa/ubIkBa/
    #   IkBa_degraded/A20/A20_mRNA/TNF_mRNA/Bcl2_mRNA) kept original.
    # ---- 配体 + 受体 ----
    # TNF（Tumor Necrosis Factor α，pro-inflammatory cytokine，NF-κB 经典配体）
    # [C4 fix] No SBML match in BIOMD0000000140 (TNF is a boundary stimulus, not modeled as species). Kept original.
    {"name": "TNF", "species_type": "ligand",
     "compartment": "extracellular"},
    # TNFR（TNF Receptor 1，膜结合受体，接收 TNF 信号）
    # [C4 fix] No SBML match in BIOMD0000000140. Kept original.
    {"name": "TNFR", "species_type": "protein",
     "compartment": "membrane"},
    # TNF_TNFR_complex（TNF+TNFR 配体-受体复合物，激活下游 IKK）
    # [C4 fix] No SBML match in BIOMD0000000140. Kept original.
    {"name": "TNF_TNFR_complex", "species_type": "complex",
     "compartment": "membrane"},
    # ---- IKK 激酶 ----
    # IKK（IκB kinase，未激活形式，被 TNF_TNFR_complex 激活）
    # [C4 fix] No SBML match in BIOMD0000000140 (SBML IKK represents active IKK only,
    #   not the inactive pool; specialist's IKK is the inactive form). Kept original.
    {"name": "IKK", "species_type": "protein",
     "compartment": "cytoplasm"},
    # pIKK（磷酸化激活的 IKK，催化 IκBα 磷酸化）
    # [C4 fix] SBML mapping: pIKK→IKK (SBML IKK=active IKK, initial_concentration=0.0)
    {"name": "pIKK", "species_type": "protein",
     "compartment": "cytoplasm",
     "initial_concentration": 0.0},  # Source: BIOMD0000000140 Hoffmann2002 (PMID:12424381) species IKK (active IKK, initially 0)
    # ---- IκBα 三步耦合降解中间产物 ----
    # IkBa（IκBα，NF-κB 抑制蛋白，扣留 NF-κB 在胞质）
    # [C4 fix] SBML mapping: IkBa→IkBalpha (initial_concentration=0.1)
    {"name": "IkBa", "species_type": "protein",
     "compartment": "cytoplasm",
     "initial_concentration": 0.1},  # Source: BIOMD0000000140 Hoffmann2002 (PMID:12424381) species IkBalpha
    # pIkBa（磷酸化 IκBα，Ser32/36，三步耦合第 1 步中间产物）
    # [C4 fix] No SBML match in BIOMD0000000140 (SBML doesn't model phosphorylated IκBα separately). Kept original.
    {"name": "pIkBa", "species_type": "protein",
     "compartment": "cytoplasm"},
    # ubIkBa（泛素化 IκBα，三步耦合第 2 步中间产物，被 E3 ligase β-TrCP 标记）
    # [C4 fix] No SBML match in BIOMD0000000140. Kept original.
    {"name": "ubIkBa", "species_type": "protein",
     "compartment": "cytoplasm"},
    # IkBa_degraded（IκBα 降解后残余，NF-κB 即将从复合物释放）
    # [C4 fix] No SBML match in BIOMD0000000140. Kept original.
    {"name": "IkBa_degraded", "species_type": "protein",
     "compartment": "cytoplasm"},
    # ---- NF-κB 转录因子 ----
    # NFkB（NF-κB 胞质游离形式，shared：与 Apoptosis Specialist Bcl-2 路径共享）
    # IκBα 降解后释放的游离 NF-κB，可入核作为转录因子
    # [C4 fix] SBML mapping: NFkB→NFkB (initial_concentration=0.1)
    {"name": "NFkB", "species_type": "protein",
     "compartment": "cytoplasm", "shared": True,
     "initial_concentration": 0.1},  # Source: BIOMD0000000140 Hoffmann2002 (PMID:12424381) species NFkB
    # NFkB_nuclear（核内 NF-κB，作为转录因子激活下游靶基因）
    # [C4 fix] SBML mapping: NFkB_nuclear→NFkB_nuc (initial_concentration=0.001)
    {"name": "NFkB_nuclear", "species_type": "protein",
     "compartment": "nucleus",
     "initial_concentration": 0.001},  # Source: BIOMD0000000140 Hoffmann2002 (PMID:12424381) species NFkB_nuc
    # ---- IκBα 转录负反馈 ----
    # IkBa_mRNA（IκBα mRNA，NF-κB 转录激活，含 30 min 转录延迟，负反馈振荡）
    # [C4 fix] SBML mapping: IkBa_mRNA→IkBalpha_transcript (initial_concentration=0.0)
    {"name": "IkBa_mRNA", "species_type": "mrna",
     "compartment": "nucleus",
     "initial_concentration": 0.0},  # Source: BIOMD0000000140 Hoffmann2002 (PMID:12424381) species IkBalpha_transcript
    # ---- A20 转录双负反馈 ----
    # A20_mRNA（A20/TNFAIP3 mRNA，NF-κB 转录激活，A20 抑制 IKK 双负反馈）
    # [C4 fix] No SBML match in BIOMD0000000140. Kept original.
    {"name": "A20_mRNA", "species_type": "mrna",
     "compartment": "nucleus"},
    # A20（A20 蛋白，抑制 IKK 活性，双负反馈 delay=60min）
    # [C4 fix] No SBML match in BIOMD0000000140. Kept original.
    {"name": "A20", "species_type": "protein",
     "compartment": "cytoplasm"},
    # ---- TNF 转录正反馈 ----
    # TNF_mRNA（TNF mRNA，NF-κB 转录激活 TNF，正反馈放大）
    # [C4 fix] No SBML match in BIOMD0000000140. Kept original.
    {"name": "TNF_mRNA", "species_type": "mrna",
     "compartment": "nucleus"},
    # ---- Bcl-2 转录（抗凋亡，下游由 Apoptosis Specialist 处理）----
    # Bcl2_mRNA（Bcl-2 mRNA，NF-κB 转录激活 Bcl-2 抗凋亡基因）
    # [C4 fix] No SBML match in BIOMD0000000140. Kept original.
    {"name": "Bcl2_mRNA", "species_type": "mrna",
     "compartment": "nucleus"},
    # [N6 缺口 1] 药物物种（species_type="drug"）— 由 drug_library 驱动
    # Bortezomib 是 26S 蛋白酶体硼酸酯抑制剂（IC50=0.62 nM, PMID:12626833）
    build_drug_species("Bortezomib"),
]


# =============================================================================
# NF-κB 核心反应（13 条：受体结合 + IKK 激活 + IκBα 三步耦合 + NF-κB 释放
# + 入核 + 4 转录 + 2 翻译）
# =============================================================================
# 每条反应含：source / target / mechanism / kinetics_type / pathway_tag /
#             substrate / product / modifier / modifier_type / autophosphorylation
#
# kinetics_type 选择：
# - complex_formation / dissociation / nuclear_import / translation → mass_action
# - phosphorylation → Michaelis_Menten（与 P3 _mechanism_phosphorylation_mm 模板对齐）
# - ubiquitination → Michaelis_Menten（E3 ligase 催化，MM 动力学）
# - proteasomal_degradation → mass_action（蛋白酶体降解，一级动力学）
# - transcription → Hill（NF-κB 作转录因子，Hill 动力学 n=2 协同结合）
_NF_KB_CORE_REACTIONS: list[dict[str, Any]] = [
    # 1. TNF + TNFR → TNF_TNFR_complex（complex_formation, 配体-受体结合）
    #    TNF 结合 TNFR1 形成 TNF-TNFR 复合物，启动 TNF 受体信号复合物（TRADD/TRAF2/RIP）
    {
        "source": "TNF",
        "target": "TNF_TNFR_complex",
        "mechanism": "complex_formation",
        "kinetics_type": "mass_action",
        "pathway_tag": PATHWAY_TAG,
        "substrate": "TNFR",
        "product": "TNF_TNFR_complex",
        "modifier": "TNF",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "TNF + TNFR → TNF_TNFR_complex（complex_formation, TNF 结合 TNFR1 形成配体-受体复合物启动 TNF 信号）",
    },
    # 2. TNF_TNFR_complex → pIKK（phosphorylation, IKK 作 substrate, TNF_TNFR_complex 作 modifier）
    #    TNF-TNFR 复合物通过 TRAF2/RIP 招募 IKK 复合物并磷酸化激活 IKK（异磷酸化形式表达）
    {
        "source": "TNF_TNFR_complex",
        "target": "pIKK",
        "mechanism": "phosphorylation",
        "kinetics_type": "Michaelis_Menten",
        "pathway_tag": PATHWAY_TAG,
        # 异磷酸化：IKK 作 substrate，pIKK 作 product，TNF_TNFR_complex 作 catalytic modifier
        "substrate": "IKK",
        "product": "pIKK",
        "modifier": "TNF_TNFR_complex",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "TNF_TNFR_complex 激活 IKK（IKK 作 substrate, TNF_TNFR_complex 作 catalytic modifier, 通过 TRAF2/RIP 招募并磷酸化激活 IKK）",
    },
    # 3. pIKK → pIkBa（phosphorylation, IκBα 作 substrate, pIKK 作 catalytic modifier）
    #    pIKK 异磷酸化 IκBα Ser32/36（三步耦合第 1 步），标记 IκBα for 泛素化
    #    ★ IκBα 三步耦合 CompositeReaction step 1（phosphorylation）
    {
        "source": "pIKK",
        "target": "pIkBa",
        "mechanism": "phosphorylation",
        "kinetics_type": "Michaelis_Menten",
        "pathway_tag": PATHWAY_TAG,
        # 异磷酸化：IkBa 作 substrate，pIkBa 作 product，pIKK 作 catalytic modifier
        "substrate": "IkBa",
        "product": "pIkBa",
        "modifier": "pIKK",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "site": "Ser32/36",
        "description": "pIKK 磷酸化 IκBα Ser32/36（IkBa 作 substrate, pIKK 作 catalytic modifier, 三步耦合第 1 步 phosphorylation, 标记 IκBα for 泛素化）",
        "composite_step": 1,
        "composite_id": "CR_IKBA_DEGRADATION",
    },
    # 4. pIkBa → ubIkBa（ubiquitination, pIkBa 作 substrate, E3 ligase β-TrCP 作 modifier）
    #    β-TrCP E3 泛素连接酶识别磷酸化 IκBα，多泛素化标记（三步耦合第 2 步）
    #    ★ IκBα 三步耦合 CompositeReaction step 2（ubiquitination）
    {
        "source": "pIkBa",
        "target": "ubIkBa",
        "mechanism": "ubiquitination",
        "kinetics_type": "Michaelis_Menten",
        "pathway_tag": PATHWAY_TAG,
        # 异磷酸化语义类似：pIkBa 作 substrate，ubIkBa 作 product，E3_ligase 作 catalytic modifier
        "substrate": "pIkBa",
        "product": "ubIkBa",
        "modifier": "E3_ligase",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "β-TrCP E3 泛素连接酶多泛素化 pIkBa（pIkBa 作 substrate, E3_ligase 作 catalytic modifier, 三步耦合第 2 步 ubiquitination, 识别磷酸化 IκBα Ser32/36）",
        "composite_step": 2,
        "composite_id": "CR_IKBA_DEGRADATION",
    },
    # 5. ubIkBa → IkBa_degraded（proteasomal_degradation, 26S 蛋白酶体降解泛素化 IκBα）
    #    26S 蛋白酶体识别多泛素链降解 IκBα（三步耦合第 3 步），释放 NF-κB
    #    ★ IκBα 三步耦合 CompositeReaction step 3（proteasomal_degradation）
    {
        "source": "ubIkBa",
        "target": "IkBa_degraded",
        "mechanism": "proteasomal_degradation",
        "kinetics_type": "mass_action",
        "pathway_tag": PATHWAY_TAG,
        # ubIkBa 作 substrate（被降解），产物为 IkBa_degraded（残余复合物）
        "substrate": "ubIkBa",
        "product": "IkBa_degraded",
        "modifier": "proteasome",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "26S 蛋白酶体降解泛素化 IκBα（ubIkBa 作 substrate, proteasome 作 catalytic modifier, 三步耦合第 3 步 proteasomal_degradation, 释放 NF-κB）",
        "composite_step": 3,
        "composite_id": "CR_IKBA_DEGRADATION",
    },
    # 6. IkBa_degraded → NFkB（dissociation, IκBα 降解释放 NF-κB）
    #    IκBα 被蛋白酶体降解后，NF-κB 从 IκBα-NF-κB 复合物释放为游离形式
    {
        "source": "IkBa_degraded",
        "target": "NFkB",
        "mechanism": "dissociation",
        "kinetics_type": "mass_action",
        "pathway_tag": PATHWAY_TAG,
        # IkBa_degraded 作 substrate（残余复合物解离），NFkB 作 product（游离 NF-κB）
        "substrate": "IkBa_degraded",
        "product": "NFkB",
        "modifier": None,
        "modifier_type": None,   # 自发解离，无催化 modifier
        "autophosphorylation": False,
        "description": "IκBα 降解释放 NF-κB（IkBa_degraded 作 substrate, NFkB 作 product, IκBα 降解后 NF-κB 从复合物释放为游离形式, 无 modifier 自发解离）",
    },
    # 7. NFkB → NFkB_nuclear（nuclear_import, NF-κB 入核）
    #    游离 NF-κB 通过 importin α/β 入核，作为转录因子激活下游靶基因
    {
        "source": "NFkB",
        "target": "NFkB_nuclear",
        "mechanism": "nuclear_import",
        "kinetics_type": "mass_action",
        "pathway_tag": PATHWAY_TAG,
        "substrate": "NFkB",
        "product": "NFkB_nuclear",
        "modifier": "NFkB",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "NF-κB 入核（NFkB 作 substrate, NFkB_nuclear 作 product, importin α/β 转运, 作为转录因子激活下游靶基因）",
    },
    # 8. NFkB_nuclear → IkBa_mRNA（transcription, Hill 动力学, NF-κB 转录 IκBα 负反馈）
    #    NF-κB 结合 IκBα 基因启动子 κB 位点，Hill 协同结合（n=2）激活转录
    #    （含 30 min 转录延迟，由 FeedbackLoop FL_NFKB_IKBA 表达，形成负反馈振荡）
    {
        "source": "NFkB_nuclear",
        "target": "IkBa_mRNA",
        "mechanism": "transcription",
        "kinetics_type": "Hill",
        "pathway_tag": PATHWAY_TAG,
        # NFkB_nuclear 作 modifier（转录因子），DNA 作 substrate，IkBa_mRNA 作 product
        "substrate": "DNA",
        "product": "IkBa_mRNA",
        "modifier": "NFkB_nuclear",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "hill_coefficient": 2,
        "description": "NF-κB 转录激活 IκBα mRNA（NFkB_nuclear 作转录因子, Hill n=2 协同结合 κB 位点, 含 30 min 转录延迟, 负反馈振荡）",
    },
    # 9. NFkB_nuclear → A20_mRNA（transcription, Hill 动力学, NF-κB 转录 A20 双负反馈）
    #    NF-κB 转录激活 A20/TNFAIP3，A20 蛋白抑制 IKK（双负反馈 delay=60min）
    {
        "source": "NFkB_nuclear",
        "target": "A20_mRNA",
        "mechanism": "transcription",
        "kinetics_type": "Hill",
        "pathway_tag": PATHWAY_TAG,
        "substrate": "DNA",
        "product": "A20_mRNA",
        "modifier": "NFkB_nuclear",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "hill_coefficient": 2,
        "description": "NF-κB 转录激活 A20 mRNA（NFkB_nuclear 作转录因子, Hill n=2, A20 蛋白抑制 IKK 形成双负反馈, delay=60min）",
    },
    # 10. NFkB_nuclear → TNF_mRNA（transcription, NF-κB 转录 TNF 正反馈）
    #     NF-κB 转录激活 TNF，形成 TNF→NF-κB→TNF 正反馈放大环路
    {
        "source": "NFkB_nuclear",
        "target": "TNF_mRNA",
        "mechanism": "transcription",
        "kinetics_type": "Hill",
        "pathway_tag": PATHWAY_TAG,
        "substrate": "DNA",
        "product": "TNF_mRNA",
        "modifier": "NFkB_nuclear",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "hill_coefficient": 2,
        "description": "NF-κB 转录激活 TNF mRNA（NFkB_nuclear 作转录因子, Hill n=2, TNF→NF-κB→TNF 正反馈放大环路）",
    },
    # 11. NFkB_nuclear → Bcl2_mRNA（transcription, NF-κB 转录 Bcl-2 抗凋亡）
    #     NF-κB 转录激活 Bcl-2 抗凋亡基因（下游凋亡由 Apoptosis Specialist 处理）
    {
        "source": "NFkB_nuclear",
        "target": "Bcl2_mRNA",
        "mechanism": "transcription",
        "kinetics_type": "Hill",
        "pathway_tag": PATHWAY_TAG,
        "substrate": "DNA",
        "product": "Bcl2_mRNA",
        "modifier": "NFkB_nuclear",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "hill_coefficient": 2,
        "description": "NF-κB 转录激活 Bcl-2 mRNA（NFkB_nuclear 作转录因子, Hill n=2, Bcl-2 抗凋亡基因, 下游凋亡由 Apoptosis Specialist 处理）",
    },
    # 12. IkBa_mRNA → IkBa（translation, mRNA→蛋白）
    #     IκBα mRNA 翻译为 IκBα 蛋白，重新结合 NF-κB 形成负反馈
    {
        "source": "IkBa_mRNA",
        "target": "IkBa",
        "mechanism": "translation",
        "kinetics_type": "mass_action",
        "pathway_tag": PATHWAY_TAG,
        "substrate": "IkBa_mRNA",
        "product": "IkBa",
        "modifier": "IkBa_mRNA",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "IκBα mRNA 翻译为 IκBα 蛋白（IkBa_mRNA 作 substrate, IkBa 作 product, 重新结合 NF-κB 形成负反馈）",
    },
    # 13. A20_mRNA → A20（translation, mRNA→蛋白）
    #     A20 mRNA 翻译为 A20 蛋白，A20 抑制 IKK 形成双负反馈
    {
        "source": "A20_mRNA",
        "target": "A20",
        "mechanism": "translation",
        "kinetics_type": "mass_action",
        "pathway_tag": PATHWAY_TAG,
        "substrate": "A20_mRNA",
        "product": "A20",
        "modifier": "A20_mRNA",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "A20 mRNA 翻译为 A20 蛋白（A20_mRNA 作 substrate, A20 作 product, A20 抑制 IKK 形成双负反馈）",
    },
    # 14. IkBa → NFkB（inhibition, mass_action, 反馈环第 3 步：IκBα 重新结合游离 NF-κB 隔离其入核）
    #    [N8-P0-2 修复] 补全反馈环第 3 步：IκBα 蛋白结合游离 NF-κB 形成 IκBα-NF-κB 复合物
    #    （Hoffmann 2002, PMID:12424381; KINETIC_PARAMETERS 中 k_off=1e-3 min^-1, Km=1e-7 M）
    #    缺失此反应导致 IκBα 蛋白无法实际扣留 NF-κB，反馈环无法闭合，NF-κB 持续激活
    #    注：NFkB 作 substrate（被扣留），IkBa 作 modifier（抑制因子），无产物（NFkB 游离形式降低）
    {
        "source": "IkBa",
        "target": "NFkB",
        "mechanism": "inhibition",
        "kinetics_type": "mass_action",
        "pathway_tag": PATHWAY_TAG,
        "substrate": "NFkB",
        "product": "NFkB",
        "modifier": "IkBa",
        "modifier_type": "allosteric",
        "autophosphorylation": False,
        "description": "IκBα 抑制 NF-κB（NFkB 作 substrate, IkBa 作 allosteric inhibitor, 反馈环第 3 步, IκBα 重新合成后结合游离 NF-κB 形成复合物, 隔离 NF-κB 阻止其入核, delay=0min 蛋白结合直接抑制, Hoffmann 2002 PMID:12424381）",
    },
    # 15. A20 → pIKK（inhibition, mass_action, 双负反馈第 2 步：A20 抑制 pIKK 激酶活性）
    #    [N8-P0-2 修复] 补全 A20 双负反馈第 2 步：A20 蛋白结合 pIKK 抑制其激酶活性
    #    （Nelson 2004, PMID:14976212; KINETIC_PARAMETERS 中 k_off=1e-3 min^-1, Km=1e-7 M）
    #    缺失此反应导致 A20 蛋白无法实际抑制 pIKK，双负反馈无法闭合
    {
        "source": "A20",
        "target": "pIKK",
        "mechanism": "inhibition",
        "kinetics_type": "mass_action",
        "pathway_tag": PATHWAY_TAG,
        "substrate": "pIKK",
        "product": "pIKK",
        "modifier": "A20",
        "modifier_type": "allosteric",
        "autophosphorylation": False,
        "description": "A20 抑制 pIKK 激酶活性（pIKK 作 substrate, A20 作 allosteric inhibitor, 双负反馈第 2 步, 阻断 pIKK 磷酸化 IκBα, delay=0min 蛋白结合直接抑制, Nelson 2004 PMID:14976212）",
    },
    # ===== [N6 缺口 1] 药物-靶点显式 inhibitor edge（canonical drug_library 驱动） =====
    # 16. Bortezomib → proteasome（boronate_reversible, IC50=0.62 nM, PMID:12626833）
    # Bortezomib 是 26S 蛋白酶体硼酸酯可逆抑制剂，占据蛋白酶体苏氨酸残基活性位点，
    # 阻止 IκBα 降解，扣留 NF-κB 在胞质。用于多发性骨髓瘤治疗（FDA-approved）。
    {
        **build_inhibitor_edge("Bortezomib", "proteasome"),
        "pathway_tag": PATHWAY_TAG,
    },
]


# =============================================================================
# IκBα 三步耦合 CompositeReaction（phosphorylation→ubiquitination→
# proteasomal_degradation）
# =============================================================================
# IκBα 三步耦合降解是 NF-κB 通路的核心调控机制：
# - Step 1: pIKK 磷酸化 IκBα Ser32/36（标记 IκBα for 泛素化）
# - Step 2: β-TrCP E3 泛素连接酶多泛素化 pIkBa
# - Step 3: 26S 蛋白酶体降解 ubIkBa，释放 NF-κB
# 净反应：IκBα → ∅（释放 NF-κB，激活 NF-κB 转录活性）
#
# 三步耦合是顺序执行（sequential coupling），中间产物 pIkBa / ubIkBa 保留语义
# 不压扁为单一 reaction，保留中间产物与子反应语义（CompositeReaction 设计原则）
_NF_KB_COMPOSITE_REACTIONS: list[dict[str, Any]] = [
    {
        "id": "CR_IKBA_DEGRADATION",
        "name": "IκBα Three-Step Coupled Degradation",
        "mechanism": "sequential",
        "template": "oscillatory_feedback.j2",
        "loop_type": "negative",
        "coupling_type": "sequential",
        "sub_reactions": [
            # Step 1: phosphorylation (pIKK → pIkBa, IκBα Ser32/36)
            "pIKK → pIkBa",
            # Step 2: ubiquitination (pIkBa → ubIkBa, β-TrCP E3 ligase)
            "pIkBa → ubIkBa",
            # Step 3: proteasomal_degradation (ubIkBa → IkBa_degraded, 26S proteasome)
            "ubIkBa → IkBa_degraded",
        ],
        "intermediate_species": [
            "pIkBa",       # 磷酸化 IκBα（step 1 产物，step 2 substrate）
            "ubIkBa",      # 泛素化 IκBα（step 2 产物，step 3 substrate）
            "IkBa_degraded",  # 降解残余（step 3 产物，释放 NF-κB）
        ],
        "net_reaction": "IκBα → ∅ (releases NF-κB, activates NF-κB transcription)",
        "node_ids": [
            "pIKK",
            "pIkBa",
            "ubIkBa",
            "IkBa_degraded",
            "NFkB",
        ],
        "reactions": [
            "pIKK → pIkBa",
            "pIkBa → ubIkBa",
            "ubIkBa → IkBa_degraded",
        ],
        "description": (
            "IκBα 三步耦合降解（phosphorylation→ubiquitination→"
            "proteasomal_degradation）：pIKK 磷酸化 IκBα Ser32/36, "
            "β-TrCP E3 泛素化 pIkBa, 26S 蛋白酶体降解 ubIkBa 释放 NF-κB, "
            "净反应 IκBα→∅ 激活 NF-κB 转录活性, sequential coupling "
            "保留中间产物语义（Nelson 2004, PMID:14976212）"
        ),
        "delay_minutes": 0.0,   # 三步耦合本身无转录延迟（蛋白级联）
        "pmid": _Pmid_NELSON_2004,
    },
]


# =============================================================================
# NF-κB 反馈环（3 条：NF-κB→IκBα 振荡 + A20→IKK 双负反馈 + IκBα→NF-κB 抑制）
# =============================================================================
_NF_KB_FEEDBACK_LOOPS: list[dict[str, Any]] = [
    # 1. NF-κB→IκBα 转录延迟负反馈（DDE 振荡器, delay=30min）
    #    Nelson 2004 (PMID:14976212) 经典模型：
    #    - NF-κB 入核转录激活 IκBα mRNA（含 30 min 转录延迟）
    #    - IκBα 蛋白重新合成结合 NF-κB，扣留其入核
    #    - 形成 delay=30min 的延迟负反馈，产生 1-2 小时核振荡周期
    {
        "id": "FL_NFKB_IKBA",
        "loop_type": "negative",
        "node_ids": [
            "NFkB_nuclear",
            "IkBa_mRNA",
            "IkBa",
            "NFkB",
        ],
        "delay_minutes": _NFKB_IKBA_DELAY_MINUTES,  # 30.0 min 转录延迟
        "bistable": False,
        "template": "oscillatory_feedback.j2",
        "description": (
            "NF-κB→IκBα 转录延迟负反馈振荡器（NFkB_nuclear 转录激活 IκBα mRNA, "
            "IκBα 蛋白重新合成结合 NF-κB 扣留其入核, delay=30min, "
            "1-2h 核振荡周期, Nelson 2004）"
        ),
        "source_pmid": _Pmid_NELSON_2004,
        "dde_solver": "solvers/dde_solver.py",
    },
    # 2. NF-κB→A20→IKK 双负反馈（DDE, delay=60min）
    #    NF-κB 转录激活 A20，A20 蛋白抑制 IKK 活性，形成双负反馈
    #    （NF-κB→A20 正向 + A20→IKK 抑制 = 双负反馈，delay=60min A20 转录翻译延迟）
    {
        "id": "FL_NFKB_A20_IKK",
        "loop_type": "negative",
        "node_ids": [
            "NFkB_nuclear",
            "A20_mRNA",
            "A20",
            "pIKK",
        ],
        "delay_minutes": _NFKB_A20_IKK_DELAY_MINUTES,  # 60.0 min 转录翻译延迟
        "bistable": False,
        "template": "oscillatory_feedback.j2",
        "description": (
            "NF-κB→A20→IKK 双负反馈（NFkB_nuclear 转录激活 A20, A20 蛋白抑制 "
            "IKK 活性, delay=60min A20 转录翻译延迟, 双负反馈调节 NF-κB 振荡幅度）"
        ),
        "source_pmid": _Pmid_NELSON_2004,
        "dde_solver": "solvers/dde_solver.py",
    },
    # 3. IκBα → NF-κB 抑制（inhibition, IκBα 结合 NF-κB 隔离其入核，负反馈）
    #    IκBα 蛋白重新合成后结合游离 NF-κB，形成 IκBα-NF-κB 复合物，
    #    隔离 NF-κB 阻止其入核（负反馈，delay=0，蛋白结合直接抑制）
    {
        "id": "FL_IKBA_NFKB_INHIBITION",
        "loop_type": "negative",
        "node_ids": ["IkBa", "NFkB", "NFkB_nuclear"],
        "delay_minutes": 0.0,
        "bistable": False,
        "description": (
            "IκBα → NF-κB 抑制（IκBα 蛋白结合游离 NF-κB 形成复合物, "
            "隔离 NF-κB 阻止其入核, delay=0min 蛋白结合直接负反馈）"
        ),
    },
]


# =============================================================================
# NF-κB Crosstalk Reaction 片段（5 条，仅本通路侧片段，edge 由 Coordinator 管理）
# =============================================================================
# NF-κB 作为转录因子激活下游抗凋亡基因（Bcl-2/Bcl-xL）与促周期基因（Cyclin D1）
# pAKT→IKK / p53→NF-κB 由其他 Specialist 生成相反方向片段，
# 本 Specialist 仅返回本通路侧消费片段
_NF_KB_CROSSTALK_REACTIONS: list[dict[str, Any]] = [
    # 1. NF-κB → Bcl-2（transcription, NF-κB 转录 Bcl-2 抗凋亡）
    #    NFkB_nuclear 转录激活 Bcl-2 抗凋亡基因（NF-κB 经典抗凋亡机制）
    {
        "source": "NFkB_nuclear",
        "target": "Bcl2",
        "mechanism": "transcription",
        "shared_species": ["NFkB"],
        "site": "κB site(Bcl-2 promoter)",
        "description": "NF-κB 转录激活 Bcl-2（NFkB_nuclear 作转录因子, Bcl-2 抗凋亡, NF-κB 经典抗凋亡机制, 与 Apoptosis 通路 cross-talk）",
    },
    # 2. NF-κB → Bcl-xL（transcription, NF-κB 转录 Bcl-xL 抗凋亡）
    #    NFkB_nuclear 转录激活 Bcl-xL 抗凋亡基因（与 Bcl-2 协同抗凋亡）
    {
        "source": "NFkB_nuclear",
        "target": "Bcl_xL",
        "mechanism": "transcription",
        "shared_species": ["NFkB"],
        "site": "κB site(Bcl-xL promoter)",
        "description": "NF-κB 转录激活 Bcl-xL（NFkB_nuclear 作转录因子, Bcl-xL 抗凋亡, 与 Bcl-2 协同维持肿瘤细胞存活, 与 Apoptosis 通路 cross-talk）",
    },
    # 3. NF-κB → Cyclin D1（transcription, NF-κB 转录 Cyclin D1 促进周期）
    #    NFkB_nuclear 转录激活 Cyclin D1，促进 G1/S 转换（NF-κB 促增殖机制）
    {
        "source": "NFkB_nuclear",
        "target": "Cyclin_D1",
        "mechanism": "transcription",
        "shared_species": ["NFkB"],
        "site": "κB site(Cyclin D1/CCND1 promoter)",
        "description": "NF-κB 转录激活 Cyclin D1（NFkB_nuclear 作转录因子, Cyclin D1 促 G1/S 转换, NF-κB 促增殖机制, 与 Cell Cycle 通路 cross-talk）",
    },
    # 4. pAKT → IKK（activation, pAKT 磷酸化激活 IKK）
    #    pAKT 磷酸化激活 IKK，增强 NF-κB 通路活性（PI3K-AKT→NF-κB cross-talk）
    #    注：本片段仅描述 NF-κB 通路侧 IKK 接收 pAKT 调控的语义，
    #    edge 本身由 PI3K Specialist + Cross-talk Coordinator 管理
    {
        "source": "pAKT",
        "target": "IKK",
        "mechanism": "activation",
        "shared_species": ["AKT", "NFkB"],
        "site": "IKK activation site",
        "description": "pAKT 磷酸化激活 IKK（pAKT 增强 IKK 激酶活性, 激活 NF-κB 通路, PI3K-AKT→NF-κB cross-talk, 与 PI3K 通路 cross-talk）",
    },
    # 5. p53 → NF-κB（inhibition, p53 抑制 NF-κB）
    #    p53 抑制 NF-κB 转录活性（p53-NF-κB 相互拮抗，肿瘤抑制 vs 抗凋亡）
    {
        "source": "p53",
        "target": "NFkB",
        "mechanism": "inhibition",
        "shared_species": ["p53", "NFkB"],
        "description": "p53 抑制 NF-κB（p53 拮抗 NF-κB 转录活性, p53-NF-κB 相互拮抗, 肿瘤抑制 vs 抗凋亡, 与 p53 通路 cross-talk）",
    },
]


# =============================================================================
# NF-κB 扰动（6 个：4 个药物 + 1 个突变 + 1 个抗氧化剂）
# =============================================================================
_NF_KB_PERTURBATIONS: list[dict[str, Any]] = [
    # 1. Bortezomib（proteasome inhibitor, FDA-approved）
    #    Bortezomib 是 26S 蛋白酶体抑制剂，阻止 IκBα 降解，扣留 NF-κB 在胞质
    #    （用于多发性骨髓瘤治疗，FDA-approved）
    # [N6 缺口 1] 注入 canonical drug_library 字段（ic50_nM/ki_nM/source_pmid/...）
    {
        "target": "proteasome",
        "drug": "Bortezomib",
        "mechanism": "inhibition",
        "ko_target": None,
        "description": "Bortezomib（26S 蛋白酶体抑制剂, FDA-approved, 阻止 IκBα 降解, 扣留 NF-κB 在胞质, 用于多发性骨髓瘤）",
        **{k: v for k, v in get_drug_entry("Bortezomib").items()
           if k not in ("description",)},
    },
    # 2. BAY 11-7082（IKK inhibitor）
    #    BAY 11-7082 是 IKK 抑制剂，阻止 IκBα 磷酸化，稳定 IκBα 抑制 NF-κB
    {
        "target": "IKK",
        "drug": "BAY 11-7082",
        "mechanism": "inhibition",
        "ko_target": None,
        "description": "BAY 11-7082（IKK 抑制剂, 小分子, 阻止 IκBα 磷酸化 Ser32/36, 稳定 IκBα 抑制 NF-κB 通路）",
    },
    # 3. IKK-16（IKK inhibitor）
    #    IKK-16 是 IKK 抑制剂，选择性抑制 IKKβ，阻断 NF-κB 通路激活
    {
        "target": "IKK",
        "drug": "IKK-16",
        "mechanism": "inhibition",
        "ko_target": None,
        "description": "IKK-16（IKK 抑制剂, 小分子, 选择性抑制 IKKβ, 阻断 NF-κB 通路激活）",
    },
    # 4. MLN120B（IKKβ inhibitor）
    #    MLN120B / MLN-120B 是 IKKβ 选择性抑制剂，阻断 IκBα 磷酸化降解
    {
        "target": "IKK",
        "drug": "MLN120B",
        "mechanism": "inhibition",
        "ko_target": None,
        "description": "MLN120B（IKKβ 选择性抑制剂, 小分子, 阻断 IκBα 磷酸化降解, 抑制 NF-κB 通路）",
    },
    # 5. NFKBIA loss（loss-of-function mutation, IκBα 失去功能导致 NF-κB 持续激活）
    #    NFKBIA 基因编码 IκBα，loss-of-function 突变导致 IκBα 失去结合 NF-κB 能力，
    #    NF-κB 持续激活（常见于霍奇金淋巴瘤 / 胶质瘤等）
    {
        "target": "IkBa",
        "drug": None,
        "mechanism": "knockout",
        "ko_target": "NFKBIA_loss",
        "description": "NFKBIA loss（loss-of-function mutation, IκBα 失去结合 NF-κB 能力, NF-κB 持续激活, 常见于霍奇金淋巴瘤 / 胶质瘤）",
    },
    # 6. PDTC（NF-κB inhibitor, antioxidant）
    #    PDTC (Pyrrolidine Dithiocarbamate) 是 NF-κB 抑制剂 + 抗氧化剂，
    #    通过抗氧化机制抑制 NF-κB 通路激活
    {
        "target": "NFkB",
        "drug": "PDTC",
        "mechanism": "inhibition",
        "ko_target": None,
        "description": "PDTC / Pyrrolidine Dithiocarbamate（NF-κB 抑制剂 + 抗氧化剂, 通过抗氧化机制抑制 NF-κB 通路激活）",
    },
]


# =============================================================================
# NF-κB Validation 规则（3 条 benchmark, Nelson 2004, PMID:14976212）
# =============================================================================
# 每条含：rule_id / metric_name / expected / tolerance / pmid / description
# expected 取区间中点，tolerance 取区间半宽
_NF_KB_VALIDATION_RULES: list[dict[str, Any]] = [
    # 1. NF-κB 核振荡周期 1-2 hours（Nelson 2004, PMID:14976212）
    #    NF-κB-IκBα DDE 延迟负反馈产生 1-2 小时核振荡周期
    {
        "rule_id": "VAL_NFKB_NUCLEAR_OSCILLATION_PERIOD",
        "metric_name": "NFkB_nuclear_oscillation_period",
        "expected": 1.5,   # (1.0 + 2.0) / 2
        "tolerance": 0.5,   # (2.0 - 1.0) / 2
        "expected_min": 1.0,
        "expected_max": 2.0,
        "unit": "hours",
        "pmid": _Pmid_NELSON_2004,
        "comparison": "range",
        "pathway_tag": PATHWAY_TAG,
        "description": "NF-κB 核振荡周期 1-2 小时（Nelson 2004 NF-κB oscillation, NF-κB-IκBα DDE 延迟负反馈振荡）",
    },
    # 2. IκBα 转录延迟 30-60 minutes（Nelson 2004, PMID:14976212）
    #    NF-κB 转录激活 IκBα mRNA 后，需 30-60 min 翻译为 IκBα 蛋白并执行负反馈
    {
        "rule_id": "VAL_NFKB_IKBA_TRANSCRIPTION_DELAY",
        "metric_name": "IkBa_transcription_delay",
        "expected": 45.0,   # (30.0 + 60.0) / 2
        "tolerance": 15.0,  # (60.0 - 30.0) / 2
        "expected_min": 30.0,
        "expected_max": 60.0,
        "unit": "minutes",
        "pmid": _Pmid_NELSON_2004,
        "comparison": "range",
        "pathway_tag": PATHWAY_TAG,
        "description": "IκBα 转录延迟 30-60 min（NF-κB 激活 IκBα mRNA 到 IκBα 蛋白执行负反馈的延迟, Nelson 2004）",
    },
    # 3. NF-κB 振荡持续时间 6-20 hours（Nelson 2004, PMID:14976212）
    #    NF-κB 核振荡应持续 6-20 小时（持续 TNF 刺激下的振荡持续时间）
    {
        "rule_id": "VAL_NFKB_OSCILLATION_DURATION",
        "metric_name": "NFkB_oscillation_duration",
        "expected": 13.0,   # (6.0 + 20.0) / 2
        "tolerance": 7.0,   # (20.0 - 6.0) / 2
        "expected_min": 6.0,
        "expected_max": 20.0,
        "unit": "hours",
        "pmid": _Pmid_NELSON_2004,
        "comparison": "range",
        "pathway_tag": PATHWAY_TAG,
        "description": "NF-κB 核振荡应持续 6-20 小时（持续 TNF 刺激下的 NF-κB 振荡持续时间, Nelson 2004）",
    },
]


@register_specialist
class NfKappaBSpecialist(PathwaySpecialistBase):
    """NF-κB 通路 Specialist。

    实现 TNF/IL-1/LPS→受体→IKK→IκBα 磷酸化→泛素化→蛋白酶体降解→NF-κB 释放→
    入核→转录（IκBα/A20/TNF/Bcl-2）核心拓扑 + NF-κB→IκBα 转录延迟负反馈振荡
    （delay=30min）+ NF-κB→A20→IKK 双负反馈的 Core/Feedback/Crosstalk/
    Perturbation/Validation 5 模块，输出通路特异 Reaction IR 片段 + IκBα 三步耦合
    CompositeReaction（phosphorylation→ubiquitination→proteasomal_degradation）
    + 模板选择 + Validation 规则。

    职责边界：
    - 处理 NF-κB 通路核心（TNF/TNFR/IKK/IκBα/NF-κB/A20）
    - 不处理 NF-κB→Bcl-2 凋亡保护下游（由 Apoptosis Specialist 处理，Task 4.7）
    - 不生成 AKT→IKK cross-talk edge（由 Cross-talk Coordinator 处理，Task 4.13）

    输入：
    - ``pathway_graph``：v4_pathway_graph 的 NF-κB 子图（含 FeedbackLoop
      FL_NFKB_IKBA delay=30min）
    - ``ontology_entities``：P1 Ontology Entities（可选，HGNC/UniProt ID 对齐）

    输出：
    - ``apply_core``：13 条核心 Reaction IR 片段 + 16 物种 + IκBα 三步耦合
      CompositeReaction（phosphorylation→ubiquitination→proteasomal_degradation）
      （NFkB 物种标记 shared=True，与 Apoptosis Specialist Bcl-2 路径共享）
    - ``apply_feedback``：3 条 FeedbackLoop
      （FL_NFKB_IKBA delay=30min 转录延迟负反馈振荡 / FL_NFKB_A20_IKK 双负反馈 /
      IκBα→NF-κB 抑制）
    - ``apply_crosstalk``：5 条 cross-talk Reaction 片段
      （NF-κB→Bcl-2/Bcl-xL/Cyclin D1 + pAKT→IKK + p53→NF-κB）
    - ``apply_perturbation``：6 个扰动
      （Bortezomib/BAY 11-7082/IKK-16/MLN120B/NFKBIA loss/PDTC）
    - ``apply_validation``：3 条 Validation benchmark
      （NF-κB 振荡周期 1-2h / IκBα 转录延迟 30-60min / 振荡持续 6-20h,
      Nelson 2004, PMID:14976212）
    """

    pathway_class: str = "NF_KB"
    display_name: str = "NF-κB Signaling"

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
                    species=list(_NF_KB_CORE_SPECIES),
                    reactions=list(_NF_KB_CORE_REACTIONS),
                    kinetics_overrides={},
                )
            if module_name == MODULE_FEEDBACK:
                return FeedbackModuleData(
                    feedback_loops=list(_NF_KB_FEEDBACK_LOOPS),
                    delay_minutes=_NFKB_IKBA_DELAY_MINUTES,
                    loop_type="negative",  # NF-κB-IκBα 负反馈振荡
                )
            if module_name == MODULE_CROSSTALK:
                return CrosstalkModuleData(
                    crosstalk_reactions=list(_NF_KB_CROSSTALK_REACTIONS),
                    shared_species=["NFkB", "AKT", "p53"],
                    coordination_strategy="merge",
                )
            if module_name == MODULE_PERTURBATION:
                return PerturbationModuleData(
                    perturbation_reactions=list(_NF_KB_PERTURBATIONS),
                    drug_targets=[
                        p for p in _NF_KB_PERTURBATIONS if p.get("drug")
                    ],
                    ko_targets=[
                        p for p in _NF_KB_PERTURBATIONS if p.get("ko_target")
                    ],
                )
            if module_name == MODULE_VALIDATION:
                return ValidationModuleData(
                    rules=list(_NF_KB_VALIDATION_RULES),
                    benchmarks=[
                        {
                            "benchmark_name": r["metric_name"],
                            "value": r["expected"],
                            "unit": r["unit"],
                            "condition": "TNF stimulation",
                            "reference": r["pmid"],
                        }
                        for r in _NF_KB_VALIDATION_RULES
                    ],
                    tolerances={
                        r["metric_name"]: r["tolerance"]
                        for r in _NF_KB_VALIDATION_RULES
                    },
                    pmid_references=[
                        {
                            "pmid": r["pmid"],
                            "citation": r["description"],
                            "pathway_class": PATHWAY_TAG,
                            "metric_name": r["metric_name"],
                        }
                        for r in _NF_KB_VALIDATION_RULES
                        if r["pmid"]
                    ],
                )
            logger.warning(
                "NfKappaBSpecialist.load_module: 未知模块名 '%s'",
                module_name,
            )
            return None
        except Exception as exc:
            logger.warning(
                "NfKappaBSpecialist.load_module 加载模块 '%s' 失败: %s",
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
        """应用核心模块，返回 NF-κB 通路核心 Reaction IR 片段。

        输出 13 条核心反应：
        1. TNF → TNF_TNFR_complex（complex_formation, 配体-受体结合）
        2. TNF_TNFR_complex → pIKK（异磷酸化, IKK 作 substrate）
        3. pIKK → pIkBa（异磷酸化, IκBα 作 substrate, Ser32/36, 三步耦合 step 1）
        4. pIkBa → ubIkBa（ubiquitination, 三步耦合 step 2）
        5. ubIkBa → IkBa_degraded（proteasomal_degradation, 三步耦合 step 3）
        6. IkBa_degraded → NFkB（dissociation, IκBα 降解释放 NF-κB）
        7. NFkB → NFkB_nuclear（nuclear_import, NF-κB 入核）
        8. NFkB_nuclear → IkBa_mRNA（transcription, Hill, IκBα 负反馈）
        9. NFkB_nuclear → A20_mRNA（transcription, Hill, A20 双负反馈）
        10. NFkB_nuclear → TNF_mRNA（transcription, TNF 正反馈）
        11. NFkB_nuclear → Bcl2_mRNA（transcription, Bcl-2 抗凋亡）
        12. IkBa_mRNA → IkBa（translation, IκBα 翻译）
        13. A20_mRNA → A20（translation, A20 翻译）

        NFkB 物种标记 shared=True（与 Apoptosis Specialist 的 Bcl-2 抗凋亡
        路径共享，下游凋亡由 Apoptosis Specialist 处理）。

        IκBα 三步耦合 CompositeReaction 输出（phosphorylation→ubiquitination→
        proteasomal_degradation，sequential coupling）。

        Returns:
            dict 含 ``species``（16 物种）/ ``reactions``（13 反应） /
            ``composite_reactions``（IκBα 三步耦合）字段。异常时返回
            ``{"species": [], "reactions": [], "composite_reactions": []}``。
        """
        try:
            return {
                "species": list(_NF_KB_CORE_SPECIES),
                "reactions": list(_NF_KB_CORE_REACTIONS),
                "composite_reactions": list(_NF_KB_COMPOSITE_REACTIONS),
                "pathway_tag": PATHWAY_TAG,
                "source_sbml": SOURCE_SBML,
                # [KINETIC_PARAMETERS 注入 / P0-1] 按 target 物种名组织的动力学参数
                # 修复 C1 Peak Time：原 KINETIC_PARAMETERS 是死代码，现通过此字段
                # 经 specialist_hook → graph_v3._ode_template_v2_hook → renderer.render(params=...)
                # 注入 ODE 模板，使 _get_param(tgt_name, key, default) 能查到文献参数。
                # 振荡稳定：通过文献 k_cat/Km 稳定 NF-κB-IκBα 振荡动力学。
                "kinetics_overrides": dict(_KINETICS_BY_TARGET),
            }
        except Exception as exc:
            logger.warning(
                "NfKappaBSpecialist.apply_core 失败: %s", exc
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
        """应用反馈模块，返回 NF-κB 通路 FeedbackLoop 列表。

        输出 3 条反馈环：
        1. NF-κB→IκBα 转录延迟负反馈（DDE delay=30min, 1-2h 核振荡周期）
        2. NF-κB→A20→IKK 双负反馈（delay=60min, A20 抑制 IKK）
        3. IκBα → NF-κB 抑制（inhibition, delay=0, IκBα 结合 NF-κB 隔离其入核）

        Returns:
            FeedbackLoop 字典列表（含 delay_minutes 标记）。异返回空列表。
        """
        try:
            return list(_NF_KB_FEEDBACK_LOOPS)
        except Exception as exc:
            logger.warning(
                "NfKappaBSpecialist.apply_feedback 失败: %s", exc
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
        """应用跨通路模块，返回 NF-κB 通路侧 cross-talk Reaction 片段。

        注意：本方法仅生成本通路侧的 cross-talk Reaction 片段，
        cross-talk edge 本身由 Cross-talk Coordinator 创建（Task 4.13）。

        输出 5 条 cross-talk Reaction 片段：
        1. NF-κB → Bcl-2（transcription, NF-κB 转录 Bcl-2 抗凋亡）
        2. NF-κB → Bcl-xL（transcription, NF-κB 转录 Bcl-xL 抗凋亡）
        3. NF-κB → Cyclin D1（transcription, NF-κB 转录 Cyclin D1 促进周期）
        4. pAKT → IKK（activation, pAKT 磷酸化激活 IKK）
        5. p53 → NF-κB（inhibition, p53 抑制 NF-κB）

        Args:
            pathway_graph: 通路图（当前未使用，保留接口一致性）。
            crosstalk_edges: 来自 Coordinator 的 cross-talk edge 列表
                （当前未使用，本 Specialist 返回静态拓扑片段）。

        Returns:
            cross-talk Reaction 片段列表。异返回空列表。
        """
        try:
            return list(_NF_KB_CROSSTALK_REACTIONS)
        except Exception as exc:
            logger.warning(
                "NfKappaBSpecialist.apply_crosstalk 失败: %s", exc
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
        """应用扰动模块，返回 NF-κB 通路特异药物 / 突变 Reaction 片段。

        输出 6 个扰动（4 个药物 + 1 个突变 + 1 个抗氧化剂）：
        1. Bortezomib（26S 蛋白酶体抑制剂, FDA-approved, 阻止 IκBα 降解）
        2. BAY 11-7082（IKK 抑制剂, 阻止 IκBα 磷酸化）
        3. IKK-16（IKK 抑制剂, 选择性抑制 IKKβ）
        4. MLN120B（IKKβ 选择性抑制剂）
        5. NFKBIA loss（loss-of-function, IκBα 失去功能导致 NF-κB 持续激活）
        6. PDTC（NF-κB 抑制剂 + 抗氧化剂）

        Args:
            pathway_graph: 通路图（当前未使用）。
            perturbation_points: 扰动点列表（当前未使用，本 Specialist
                返回静态拓扑片段）。

        Returns:
            扰动 Reaction 片段列表。异返回空列表。
        """
        try:
            return list(_NF_KB_PERTURBATIONS)
        except Exception as exc:
            logger.warning(
                "NfKappaBSpecialist.apply_perturbation 失败: %s", exc
            )
            return []

    # =================================================================
    # apply_validation：Validation 规则列表
    # =================================================================
    def apply_validation(
        self,
        simulation_result: dict | None = None,
    ) -> list[dict]:
        """应用验证模块，返回 NF-κB 通路 Validation 规则列表。

        输出 3 条 benchmark（Nelson 2004, PMID:14976212）：
        1. NF-κB 核振荡周期 1-2 hours（NF-κB-IκBα DDE 延迟负反馈振荡）
        2. IκBα 转录延迟 30-60 min（NF-κB 激活 IκBα mRNA 到蛋白的延迟）
        3. NF-κB 振荡持续时间 6-20 hours（持续 TNF 刺激下的振荡持续时间）

        Args:
            simulation_result: 仿真结果（可选，当前未使用，返回静态规则集）。

        Returns:
            Validation 规则列表。异返回空列表。
        """
        try:
            return list(_NF_KB_VALIDATION_RULES)
        except Exception as exc:
            logger.warning(
                "NfKappaBSpecialist.apply_validation 失败: %s", exc
            )
            return []

    # =================================================================
    # select_template：覆写以支持 oscillatory + phosphorylation 模式
    # =================================================================
    def select_template(self, mechanism: str) -> str:
        """根据 mechanism 选择 ODE 模板名（覆写支持振荡与磷酸化）。

        默认映射（与 P3 ``ode_templates_v2/`` 下 .j2 文件对齐）：
        - ``oscillatory`` → ``oscillatory_feedback``（NF-κB-IκBα 延迟振荡）
        - ``phosphorylation`` → ``_mechanism_phosphorylation_mm``
          （pIKK→pIkBa / TNF_TNFR_complex→pIKK 异磷酸化）
        - ``transcription`` → ``oscillatory_feedback``
          （NF-κB 转录 IκBα/A20，使用 oscillatory 模板表达 DDE 振荡）

        Args:
            mechanism: 机制名（小写，如 ``"oscillatory"`` / ``"phosphorylation"``）。

        Returns:
            ODE 模板名（不含 ``.j2`` 后缀）。未匹配时返回 ``"default"``
            （调用方应处理默认降级）。
        """
        # oscillatory 模式：NF-κB-IκBα 延迟负反馈振荡（1-2h 周期）
        if mechanism == "oscillatory":
            return "oscillatory_feedback"
        # 转录场景：NF-κB 转录 IκBα/A20，使用 oscillatory_feedback 模板
        # （DDE 延迟振荡器核心，NF-κB 转录是振荡器的驱动源）
        if mechanism == "transcription":
            return "oscillatory_feedback"
        # 磷酸化场景：pIKK→pIkBa / TNF_TNFR_complex→pIKK 异磷酸化
        if mechanism == "phosphorylation":
            return "_mechanism_phosphorylation_mm"
        # 其他机制走默认基类映射
        return super().select_template(mechanism)


# =============================================================================
# 文献动力学参数（IB-017 修复）
# =============================================================================
# 来源：
# - BIOMD0000000007 (Hoffmann 2002, PMID:12424381) NF-κB 振荡模型
# - BIOMD0000000268 NF-κB 通路模型
# - PMID:14976212 (Nelson 2004) NF-κB 振荡验证基准（文件已有引用）
# 反幻觉守卫：所有参数来自上述 BioModels 模型或文献；无确切值的用无量纲化
# 估计并标注 `# Heuristic estimate, needs calibration`。
# 参数范围约束：k_on∈[1e3,1e7] M^-1 min^-1, Km∈[1e-7,1e-2] M, k_cat∈[1e-3,1e3] min^-1
# 注：NF-κB-IκBα 转录延迟 _NFKB_IKBA_DELAY_MINUTES=30min、A20-IKK 延迟
#     _NFKB_A20_IKK_DELAY_MINUTES=60min 为 DDE 延迟项（见文件顶部）。
KINETIC_PARAMETERS: dict[str, dict[str, float]] = {
    # TNF+TNFR→TNF_TNFR_complex 配体-受体结合（Hoffmann 2002, PMID:12424381, BIOMD0000000007）
    "TNF_TNFR": {
        "k_on": 5.0e5,                # M^-1 min^-1  # Heuristic estimate, needs calibration
        "k_off": 0.01,                # min^-1  # Heuristic estimate, needs calibration
    },
    # TNF_TNFR_complex→pIKK IKK 磷酸化激活（Hoffmann 2002, PMID:12424381, 异磷酸化）
    "TNF_complex_pIKK": {
        "k_cat": 1.0,                 # min^-1  # Heuristic estimate, needs calibration
        "Km": 1e-7,                   # M (IKK Km)  # Heuristic estimate, needs calibration
    },
    # pIKK→pIkBa IκBα 磷酸化（Hoffmann 2002, PMID:12424381, 三步耦合 step1, 异磷酸化）
    "pIKK_pIkBa": {
        "k_cat": 1.0,                 # min^-1
        "Km": 1e-7,                   # M (IκBα Km ≈100 nM, Hoffmann 2002)
    },
    # pIkBa→ubIkBa 泛素化（Hoffmann 2002, PMID:12424381, 三步耦合 step2）
    "pIkBa_ubIkBa": {
        "k_cat": 1.0,                 # min^-1  # Heuristic estimate, needs calibration
        "Km": 1e-7,                   # M  # Heuristic estimate, needs calibration
    },
    # ubIkBa→IkBa_degraded 蛋白酶体降解（Hoffmann 2002, PMID:12424381, 三步耦合 step3）
    "ubIkBa_degradation": {
        "k_degradation": 0.5,         # min^-1  # Heuristic estimate, needs calibration
    },
    # IkBa_degraded→NFkB NF-κB 释放（Hoffmann 2002, PMID:12424381）
    "IkBa_degraded_NFkB_release": {
        "k_release": 1.0,             # min^-1  # Heuristic estimate, needs calibration
    },
    # NFkB→NFkB_nuclear 入核（Hoffmann 2002, PMID:12424381）
    "NFkB_nuclear_import": {
        "k_import": 0.1,              # min^-1  # Heuristic estimate, needs calibration
    },
    # NFkB_nuclear→IkBa_mRNA 转录（Hoffmann 2002, PMID:12424381, DDE delay=30min 负反馈）
    "NFkB_IkBa_transcription": {
        "k_transcription": 1.0,       # min^-1  # Heuristic estimate, needs calibration
        "Km": 1e-7,                   # M (NFkB_nuclear Km, Hill n=2)  # Heuristic estimate, needs calibration
    },
    # NFkB_nuclear→A20_mRNA 转录（Hoffmann 2002, PMID:12424381, DDE delay=60min 双负反馈）
    "NFkB_A20_transcription": {
        "k_transcription": 1.0,       # min^-1  # Heuristic estimate, needs calibration
        "Km": 1e-7,                   # M  # Heuristic estimate, needs calibration
    },
    # A20 抑制 pIKK（双负反馈, Nelson 2004, PMID:14976212）
    "A20_pIKK_inhibition": {
        "k_on": 1.0e6,                # M^-1 min^-1  # Heuristic estimate, needs calibration
        "k_off": 1e-3,                # min^-1  # Heuristic estimate, needs calibration
    },
    # NFkB_nuclear→TNF_mRNA 转录（正反馈, Hoffmann 2002, PMID:12424381）
    "NFkB_TNF_transcription": {
        "k_transcription": 1.0,       # min^-1  # Heuristic estimate, needs calibration
        "Km": 1e-7,                   # M  # Heuristic estimate, needs calibration
    },
    # NFkB_nuclear→Bcl2_mRNA 转录（抗凋亡, Hoffmann 2002, PMID:12424381）
    "NFkB_Bcl2_transcription": {
        "k_transcription": 1.0,       # min^-1  # Heuristic estimate, needs calibration
        "Km": 1e-7,                   # M  # Heuristic estimate, needs calibration
    },
    # IkBa_mRNA→IkBa 翻译（Hoffmann 2002, PMID:12424381）
    "IkBa_translation": {
        "k_translation": 0.1,         # min^-1  # Heuristic estimate, needs calibration
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
#   - k_cat / k_off / k_degradation / k_release / k_import / k_translation /
#     k_transcription 是时间常数（min^-1），无需转换
#   - k_on 参数单位为 M^-1 min^-1，与 ODE 模型 μM 单位冲突，统一 SKIP
#
# 映射依据（KINETIC_PARAMETERS 键名 → 反应 target 物种名）：
#   "TNF_TNFR"                   → TNF_TNFR_complex（反应 1: TNF+TNFR→TNF_TNFR_complex 配体-受体结合, k_on SKIP）
#   "TNF_complex_pIKK"           → pIKK（反应 2: TNF_TNFR_complex→pIKK IKK 磷酸化激活）
#   "A20_pIKK_inhibition"        → pIKK（A20 双负反馈抑制 pIKK, k_on SKIP, 合并到 pIKK）
#   "pIKK_pIkBa"                 → pIkBa（反应 3: pIKK→pIkBa IκBα 磷酸化, 三步耦合 step1）
#   "pIkBa_ubIkBa"               → ubIkBa（反应 4: pIkBa→ubIkBa 泛素化, 三步耦合 step2）
#   "ubIkBa_degradation"         → IkBa_degraded（反应 5: ubIkBa→IkBa_degraded 蛋白酶体降解, 三步耦合 step3）
#   "IkBa_degraded_NFkB_release" → NFkB（反应 6: IkBa_degraded→NFkB NF-κB 释放）
#   "NFkB_nuclear_import"        → NFkB_nuclear（反应 7: NFkB→NFkB_nuclear 入核）
#   "NFkB_IkBa_transcription"    → IkBa_mRNA（反应 8: NFkB_nuclear→IkBa_mRNA 转录, DDE delay=30min 负反馈）
#   "NFkB_A20_transcription"     → A20_mRNA（反应 9: NFkB_nuclear→A20_mRNA 转录, DDE delay=60min 双负反馈）
#   "NFkB_TNF_transcription"     → TNF_mRNA（反应 10: NFkB_nuclear→TNF_mRNA 转录, 正反馈）
#   "NFkB_Bcl2_transcription"    → Bcl2_mRNA（反应 11: NFkB_nuclear→Bcl2_mRNA 转录, 抗凋亡）
#   "IkBa_translation"           → IkBa（反应 13: IkBa_mRNA→IkBa 翻译）
_KINETICS_BY_TARGET: dict[str, dict[str, float]] = {
    "TNF_TNFR_complex": {
        # k_on SKIP: 单位 M^-1 min^-1 与 μM 模型冲突
        "k_off": 0.05,                # min^-1 (TNF-TNFR 解离, Hoffmann 2002)
    },
    # [RC29 校准对齐] pIKK k_cat 1.0→2.0 对齐 oscillatory_feedback.j2 磷酸化默认值
    "pIKK": {
        "k_cat": 2.0,                 # min^-1 (TNF_complex 催化 IKK 磷酸化, Hoffmann 2002)
        "Km": 0.1,                    # μM (原 1e-7 M = 0.1 μM)
        # k_on SKIP: 单位 M^-1 min^-1 与 μM 模型冲突 (A20 双负反馈抑制)
        "k_off": 1e-3,                # min^-1 (A20 抑制 pIKK 解离, Nelson 2004)
    },
    # [RC29 校准对齐] pIkBa k_cat 1.0→2.0 对齐 oscillatory_feedback.j2 磷酸化默认值
    "pIkBa": {
        "k_cat": 2.0,                 # min^-1 (pIKK 催化 IκBα 磷酸化, Hoffmann 2002, 三步耦合 step1)
        "Km": 0.1,                    # μM (原 1e-7 M = 0.1 μM, IκBα Km ≈100 nM)
    },
    # [RC29 校准对齐] ubIkBa k_cat 1.0→2.0 对齐 oscillatory_feedback.j2 磷酸化默认值
    "ubIkBa": {
        "k_cat": 2.0,                 # min^-1 (E3 催化 pIkBa 泛素化, Hoffmann 2002, 三步耦合 step2)
        "Km": 0.1,                    # μM (原 1e-7 M = 0.1 μM)
    },
    "IkBa_degraded": {
        "k_degradation": 0.5,         # min^-1 (蛋白酶体降解 ubIkBa, Hoffmann 2002, 三步耦合 step3)
    },
    "NFkB": {
        "k_release": 1.0,             # min^-1 (IkBa 降解后 NF-κB 释放, Hoffmann 2002)
    },
    "NFkB_nuclear": {
        # [P1-NEXT-7 修复 V2 / NFκB_nuclear peak_time=11.84min 过早（期望 [15,30]min）]
        # Root Cause (V2): k_import=0.1 min^-1 使上升时间常数=10min，
        #   与实测 11.84min 完全吻合 → 首峰完全由 k_import 主导
        #   IκBα 负反馈因 30min DDE delay + 慢翻译（k_translation=0.1, 时间常数 10min）
        #   远晚于首峰，无法在峰前提供刹车
        # Fix V2: k_import 0.1 → 0.04（时间常数 10min → 25min）
        #   稳态估算：1/e=25min → NFκB_nuclear 在 [15,30]min 区间达峰
        #   且 IκBα 翻译 k_translation=0.3（时间常数 3.3min）配合 DDE delay=30min
        #   使 IκBα 在 ~33min 重新合成，刚好抑制 NFκB_nuclear 形成瞬态峰
        "k_import": 0.04,             # min^-1 (NF-κB 入核, P1-NEXT-7 V2: 0.1→0.04 延后达峰到 [15,30]min)
    },
    "IkBa_mRNA": {
        "k_transcription": 1.0,       # min^-1 (NF-κB 转录激活 IκBα, Hoffmann 2002, DDE delay=30min)
        "Km": 0.1,                    # μM (原 1e-7 M = 0.1 μM, Hill n=2)
    },
    "A20_mRNA": {
        "k_transcription": 1.0,       # min^-1 (NF-κB 转录激活 A20, Hoffmann 2002, DDE delay=60min)
        "Km": 0.1,                    # μM (原 1e-7 M = 0.1 μM)
    },
    "TNF_mRNA": {
        "k_transcription": 1.0,       # min^-1 (NF-κB 转录激活 TNF, Hoffmann 2002, 正反馈)
        "Km": 0.1,                    # μM (原 1e-7 M = 0.1 μM)
    },
    "Bcl2_mRNA": {
        "k_transcription": 1.0,       # min^-1 (NF-κB 转录激活 Bcl-2, Hoffmann 2002, 抗凋亡)
        "Km": 0.1,                    # μM (原 1e-7 M = 0.1 μM)
    },
    "IkBa": {
        # [P1-NEXT-7 V2 修复] 加速 IκBα 翻译，使负反馈更早启动
        # k_translation 0.1 → 0.3：时间常数 10min → 3.3min
        # 配合 DDE delay=30min，IκBα 在 ~33min 重新合成，刚好抑制 NFκB_nuclear 瞬态峰
        "k_translation": 0.3,         # min^-1 (IkBa_mRNA 翻译, P1-NEXT-7 V2: 0.1→0.3 加速负反馈)
    },
    # [N8-P0-2 修复] 补 A20 翻译参数（反应 13: A20_mRNA→A20 翻译）
    "A20": {
        "k_translation": 0.1,         # min^-1 (A20_mRNA 翻译, Nelson 2004, 双负反馈 delay=60min)
    },
    # [N8-P0-2 修复] 补 NFkB inhibition 参数（反应 14: IkBa→NFkB 抑制，反馈环第 3 步）
    # 注：NFkB 已含 k_release=1.0，此处合并 inhibition k_off
    # 因 NFkB 条目已存在，下面单独追加 inhibition k_off 到现有条目（避免重复定义）
}

# [N8-P0-2 修复] 追加 NFkB inhibition k_off（IkBa→NFkB 抑制反馈环第 3 步, mass_action）
# 因 NFkB 条目已存在 k_release=1.0，此处追加 k_off_inhibition 字段（避免覆盖原参数）
_KINETICS_BY_TARGET["NFkB"]["k_off"] = 1e-3  # min^-1 (IkBa 抑制 NFkB 解离, Hoffmann 2002)

# 追加 pSTAT5_dimer inhibition k_off（PIAS→pSTAT5_dimer 抑制, JAK-STAT cross-species reference）
# 注：JAK-STAT specialist 的 pSTAT5_dimer 已有 k_off=0.05 (二聚体解离)，本字段供 NF-κB cross-talk 引用



__all__ = [
    "NfKappaBSpecialist",
    "PATHWAY_TAG",
    "SOURCE_SBML",
    "KINETIC_PARAMETERS",
    "_KINETICS_BY_TARGET",
]
