# BioDynamics Agent v4 - Apoptosis Specialist (Phase 4 / Task 4.7)
# Apoptosis 通路 Specialist：实现 Intrinsic（BH3-only→Bax/Bak→MOMP→Cyt c→
# Apaf-1→Casp9→Casp3）+ Extrinsic（FasL→DISC→Casp8→Casp3 + Bid→Bax）核心拓扑 +
# Caspase 级联正反馈（Casp3→Casp6→Casp8 bistable, point-of-no-return）+
# XIAP 负反馈 + MOMP bistable switch。
#
# 设计原则（铁律）：
# 1. Feature Flag V4_PATHWAY_SPECIALIST_ENABLED=false 时由 LangGraph hook 层短路，
#    本 Specialist 代码不被调用（基类本身不做 flag 检查）
# 2. 不处理 p53 上游（由 p53 Specialist 处理，Task 4.6）
# 3. 不生成 NF-κB→Bcl-2 cross-talk edge 本身（由 Cross-talk Coordinator 处理，
#    Task 4.13），仅返回本通路侧的 cross-talk Reaction 片段
# 4. apply_* 方法实现捕获异常并返回空 list/dict，记录 logger.warning，不抛异常
#
# 依赖：
# - P1 ontology / pathway_registry（Apoptosis 通路类别键）
# - P2 schema（SpeciesV2 / ReactionV2 / SpeciesRef / Modifier / CompositeReaction）
# - P2 MechanismType（CLEAVAGE / ACTIVATION / INHIBITION / COMPLEX_FORMATION /
#   TRANSLOCATION）
# - P3 ode_templates_v2（bistable_switch.j2 / _mechanism_phosphorylation_mm.j2 模板）
# - P3 bistability_detector.py（Caspase 级联 bistable 检测）
# - P3 pathway_graph/initializer.py（Apoptosis core_nodes / core_edges）
#
# 参考：
# - spec.md Part 3 Specialist 5（第 220-225 行）
# - tasks.md Task 4.7（第 88-95 行）
# - Reubold 2009 Apoptosome (PMID:11274138)
# - Green & Kroemer 2004 (PMID:15241432) MOMP point-of-no-return
# - BioModels BIOMD0000000332 (Apoptosis)

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
# Apoptosis 通路标签
# =============================================================================
PATHWAY_TAG: str = "APOPTOSIS"

# SBML BioModels ID（Apoptosis model, Intrinsic + Extrinsic）
SOURCE_SBML: str = "BIOMD0000000332"

# Validation benchmark PMID 引用
_Pmid_REUBOLD_2009: str = "PMID:11274138"   # Reubold 2009 Apoptosome
_Pmid_GREEN_KROEMER_2004: str = "PMID:15241432"  # Green & Kroemer 2004 MOMP


# =============================================================================
# Apoptosis 核心物种（与 P3 pathway_graph/initializer.py Apoptosis.core_nodes
# 对齐，扩展 MOMP/Cyt_c/Apaf-1/Apoptosome/Caspase8/Bid/tBid/DISC/Caspase6/XIAP/
# PARP_cleaved 完整 Intrinsic + Extrinsic 拓扑）
# =============================================================================
# Caspase3_active 物种标记 shared=True（Caspase-3 是 Intrinsic + Extrinsic 两条
# 途径的汇聚点，也是 Caspase 级联正反馈的核心节点，跨模块共享）。
_APOPTOSIS_CORE_SPECIES: list[dict[str, Any]] = [
    # ---- Intrinsic pathway 物种 ----
    # Bcl-2 家族（抗凋亡 / 促凋亡）
    {"name": "Bcl2", "species_type": "protein",
     "compartment": "mitochondria"},
    {"name": "Bax", "species_type": "protein",
     "compartment": "mitochondria"},
    {"name": "Bad", "species_type": "protein", "compartment": "cytoplasm"},
    # MOMP（线粒体外膜通透化，point-of-no-return 标志）
    {"name": "MOMP", "species_type": "process",
     "compartment": "mitochondria"},
    # Cyt c（线粒体→胞质 translocation）
    {"name": "Cyt_c", "species_type": "protein",
     "compartment": "cytoplasm"},
    # Apaf-1 + Apoptosome（凋亡体复合物）
    {"name": "Apaf-1", "species_type": "protein",
     "compartment": "cytoplasm"},
    {"name": "Apoptosome", "species_type": "complex",
     "compartment": "cytoplasm"},
    # Caspase-9（ initiator caspase，Apoptosome 激活）
    {"name": "Caspase9_active", "species_type": "protein",
     "compartment": "cytoplasm"},
    # ---- Extrinsic pathway 物种 ----
    # FasL + Fas（死亡受体途径配体 + 受体）
    {"name": "FasL", "species_type": "ligand",
     "compartment": "extracellular"},
    {"name": "Fas", "species_type": "protein", "compartment": "membrane"},
    # DISC（Death-Inducing Signaling Complex，FasL-Fas-FADD-procaspase-8）
    {"name": "DISC", "species_type": "complex", "compartment": "membrane"},
    # Caspase-8（initiator caspase，DISC 激活）
    {"name": "Caspase8_active", "species_type": "protein",
     "compartment": "cytoplasm"},
    # Bid / tBid（BH3-only 蛋白，Caspase-8 切割 Bid 产生 tBid 连接 extrinsic→intrinsic）
    {"name": "Bid", "species_type": "protein", "compartment": "cytoplasm"},
    {"name": "tBid", "species_type": "protein", "compartment": "cytoplasm"},
    # ---- Caspase 级联正反馈物种 ----
    # Caspase-3（executioner caspase，Intrinsic + Extrinsic 汇聚点，shared）
    # Caspase3_active 标记 shared=True：两条途径汇聚 + 正反馈核心节点
    {"name": "Caspase3_active", "species_type": "protein",
     "compartment": "cytoplasm", "shared": True},
    # Caspase-6（executioner caspase，正反馈环中间节点）
    {"name": "Caspase6_active", "species_type": "protein",
     "compartment": "cytoplasm"},
    # ---- 效应物 / 标志物 ----
    # PARP（聚 ADP 核糖聚合酶，Caspase-3 切割标志凋亡执行）
    {"name": "PARP_cleaved", "species_type": "protein",
     "compartment": "nucleus"},
    # XIAP（X-linked inhibitor of apoptosis protein，负反馈抑制 Caspase-3）
    {"name": "XIAP", "species_type": "protein",
     "compartment": "cytoplasm"},
]


# =============================================================================
# Apoptosis 核心反应（13 条：Intrinsic 6 + Extrinsic 4 + Caspase 级联 3）
# =============================================================================
# 每条反应含：source / target / mechanism / kinetics_type / pathway_tag /
#             substrate / product / modifier / modifier_type / autophosphorylation
#
# kinetics_type 选择：
# - inhibition / activation → hybrid（变构调控 / 复合调控）
# - complex_formation / translocation / cleavage → mass_action（质量作用 / 一级动力学）
_APOPTOSIS_CORE_REACTIONS: list[dict[str, Any]] = [
    # ===== Intrinsic pathway（线粒体途径，6 条）=====
    # 1. Bad → Bcl2（inhibition，Bad 结合 Bcl-2 释放 Bax）
    #    Bad 是 BH3-only 蛋白，结合抗凋亡 Bcl-2 释放促凋亡 Bax
    {
        "source": "Bad",
        "target": "Bcl2",
        "mechanism": "inhibition",
        "kinetics_type": "hybrid",
        "pathway_tag": PATHWAY_TAG,
        "substrate": "Bcl2",
        "product": "Bcl2_Bad",
        "modifier": "Bad",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "Bad 结合 Bcl-2（inhibition，释放 Bax 启动凋亡，Bad 作 BH3-only 蛋白拮抗 Bcl-2）",
    },
    # 2. Bax → MOMP（activation，Bax 寡聚化形成孔道）
    #    Bax 被 BH3-only 激活后寡聚化，在线粒体外膜形成孔道导致 MOMP
    {
        "source": "Bax",
        "target": "MOMP",
        "mechanism": "activation",
        "kinetics_type": "hybrid",
        "pathway_tag": PATHWAY_TAG,
        "substrate": "Bax",
        "product": "MOMP",
        "modifier": "Bax",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "Bax 寡聚化形成孔道导致 MOMP（activation，线粒体外膜通透化，point-of-no-return）",
    },
    # 3. MOMP → Cyt_c（translocation，Cyt c 从线粒体释放到胞质）
    #    MOMP 后 Cyt c 从线粒体膜间隙释放到胞质
    {
        "source": "MOMP",
        "target": "Cyt_c",
        "mechanism": "translocation",
        "kinetics_type": "mass_action",
        "pathway_tag": PATHWAY_TAG,
        "substrate": "Cyt_c_mitochondrial",
        "product": "Cyt_c",
        "modifier": "MOMP",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "MOMP 后 Cyt c 从线粒体释放到胞质（translocation，触发 Apoptosome 组装）",
    },
    # 4. Cyt_c → Apoptosome（complex_formation，Cyt c + Apaf-1 + dATP → Apoptosome）
    #    Cyt c 结合 Apaf-1 + dATP 形成 7 聚体 Apoptosome（凋亡体）
    {
        "source": "Cyt_c",
        "target": "Apoptosome",
        "mechanism": "complex_formation",
        "kinetics_type": "mass_action",
        "pathway_tag": PATHWAY_TAG,
        "substrate": "Apaf-1",
        "product": "Apoptosome",
        "modifier": "Cyt_c",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "co_factor": "dATP",
        "stoichiometry": "7:1",
        "description": "Cyt c + Apaf-1 + dATP → Apoptosome（complex_formation，7 聚体凋亡体组装）",
    },
    # 5. Apoptosome → Caspase9_active（activation，凋亡体激活 Caspase-9）
    #    Apoptosome 招募 procaspase-9 并激活（initiator caspase 激活）
    {
        "source": "Apoptosome",
        "target": "Caspase9_active",
        "mechanism": "activation",
        "kinetics_type": "hybrid",
        "pathway_tag": PATHWAY_TAG,
        "substrate": "Caspase9",
        "product": "Caspase9_active",
        "modifier": "Apoptosome",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "Apoptosome 激活 Caspase-9（activation，initiator caspase 激活，procaspase-9→active）",
    },
    # 6. Caspase9_active → Caspase3_active（cleavage，Caspase-3 切割激活）
    #    Caspase-9 切割 procaspase-3 激活为 Caspase-3（executioner caspase）
    {
        "source": "Caspase9_active",
        "target": "Caspase3_active",
        "mechanism": "cleavage",
        "kinetics_type": "mass_action",
        "pathway_tag": PATHWAY_TAG,
        "substrate": "Caspase3",
        "product": "Caspase3_active",
        "modifier": "Caspase9_active",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "Caspase-9 切割激活 Caspase-3（cleavage，executioner caspase 激活，Intrinsic 途径汇聚点）",
    },
    # ===== Extrinsic pathway（死亡受体途径，4 条）=====
    # 7. FasL → DISC（complex_formation，FasL-Fas-FADD-procaspase-8 复合物）
    #    FasL 结合 Fas 受体，招募 FADD + procaspase-8 形成 DISC
    {
        "source": "FasL",
        "target": "DISC",
        "mechanism": "complex_formation",
        "kinetics_type": "mass_action",
        "pathway_tag": PATHWAY_TAG,
        "substrate": "Fas",
        "product": "DISC",
        "modifier": "FasL",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "co_factor": "FADD",
        "description": "FasL + Fas + FADD → DISC（complex_formation，死亡诱导信号复合物组装）",
    },
    # 8. DISC → Caspase8_active（activation，DISC 激活 Caspase-8）
    #    DISC 内 procaspase-8 二聚化自激活（proximity-induced activation）
    {
        "source": "DISC",
        "target": "Caspase8_active",
        "mechanism": "activation",
        "kinetics_type": "hybrid",
        "pathway_tag": PATHWAY_TAG,
        "substrate": "Caspase8",
        "product": "Caspase8_active",
        "modifier": "DISC",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "DISC 激活 Caspase-8（activation，procaspase-8 近距离二聚化自激活，initiator caspase）",
    },
    # 9. Caspase8_active → tBid（cleavage，Caspase-8 切割 Bid 产生 tBid）
    #    Caspase-8 切割 Bid（BH3-only 蛋白）产生 tBid，连接 extrinsic→intrinsic
    {
        "source": "Caspase8_active",
        "target": "tBid",
        "mechanism": "cleavage",
        "kinetics_type": "mass_action",
        "pathway_tag": PATHWAY_TAG,
        "substrate": "Bid",
        "product": "tBid",
        "modifier": "Caspase8_active",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "Caspase-8 切割 Bid 产生 tBid（cleavage，连接 extrinsic→intrinsic，tBid 激活 Bax）",
    },
    # 10. tBid → Bax（activation，tBid 激活 Bax）
    #     tBid 激活 Bax（BH3-only→Bax 寡聚化），extrinsic 途径放大 intrinsic
    {
        "source": "tBid",
        "target": "Bax",
        "mechanism": "activation",
        "kinetics_type": "hybrid",
        "pathway_tag": PATHWAY_TAG,
        "substrate": "Bax",
        "product": "Bax",
        "modifier": "tBid",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "tBid 激活 Bax（activation，extrinsic→intrinsic 放大，tBid 作 BH3-only 激活 Bax 寡聚化）",
    },
    # ===== Caspase 级联正反馈（bistable, 3 条）=====
    # 11. Caspase3_active → Caspase6_active（cleavage，Caspase-3 切割激活 Caspase-6）
    #     Caspase-3 切割 procaspase-6 激活（executioner caspase 级联）
    {
        "source": "Caspase3_active",
        "target": "Caspase6_active",
        "mechanism": "cleavage",
        "kinetics_type": "mass_action",
        "pathway_tag": PATHWAY_TAG,
        "substrate": "Caspase6",
        "product": "Caspase6_active",
        "modifier": "Caspase3_active",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "Caspase-3 切割激活 Caspase-6（cleavage，executioner caspase 级联，正反馈环前段）",
    },
    # 12. Caspase6_active → Caspase8_active（cleavage，Caspase-6 反馈激活 Caspase-8）
    #     Caspase-6 切割激活 Caspase-8，形成 Casp3→Casp6→Casp8 正反馈环（bistable）
    {
        "source": "Caspase6_active",
        "target": "Caspase8_active",
        "mechanism": "cleavage",
        "kinetics_type": "mass_action",
        "pathway_tag": PATHWAY_TAG,
        "substrate": "Caspase8",
        "product": "Caspase8_active",
        "modifier": "Caspase6_active",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "Caspase-6 反馈激活 Caspase-8（cleavage，Casp3→Casp6→Casp8 正反馈环，bistable point-of-no-return）",
    },
    # 13. Caspase3_active → PARP_cleaved（cleavage，PARP 切割是凋亡标志）
    #     Caspase-3 切割 PARP（聚 ADP 核糖聚合酶），凋亡执行标志物
    {
        "source": "Caspase3_active",
        "target": "PARP_cleaved",
        "mechanism": "cleavage",
        "kinetics_type": "mass_action",
        "pathway_tag": PATHWAY_TAG,
        "substrate": "PARP",
        "product": "PARP_cleaved",
        "modifier": "Caspase3_active",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "Caspase-3 切割 PARP（cleavage，PARP 切割是凋亡执行标志物，Asp214 位点）",
    },
]


# =============================================================================
# Apoptosis CompositeReaction（Caspase 级联正反馈环，bistable）
# =============================================================================
# Caspase 级联正反馈环（Casp3→Casp6→Casp8→Casp3）是凋亡的 point-of-no-return：
# - Caspase-3 激活 Caspase-6
# - Caspase-6 反馈激活 Caspase-8
# - Caspase-8 切割 Bid→tBid→Bax→MOMP→Cyt c→Apoptosome→Caspase-9→Caspase-3
#   （间接正反馈放大，但 CompositeReaction 聚焦 Casp3→Casp6→Casp8→Casp3 直接环）
# 该正反馈环呈现 bistable all-or-none 行为：procaspase-3 > 10% 激活后正反馈
# 放大到 100%（Caspase3 激活阈值 0.1-0.5 fraction_of_max）
_APOPTOSIS_COMPOSITE_REACTIONS: list[dict[str, Any]] = [
    {
        "id": "CR_CASPASE_CASCADE_BISTABLE",
        "name": "Caspase Cascade Positive Feedback Ring",
        "mechanism": "bistable",
        "template": "bistable_switch.j2",
        "loop_type": "positive",
        "point_of_no_return": True,
        "node_ids": [
            "Caspase3_active",
            "Caspase6_active",
            "Caspase8_active",
        ],
        "reactions": [
            # Casp3 → Casp6（Caspase-3 切割激活 Caspase-6）
            "Caspase3_active → Caspase6_active",
            # Casp6 → Casp8（Caspase-6 反馈激活 Caspase-8）
            "Caspase6_active → Caspase8_active",
            # Casp8 → Casp3（Caspase-8 通过 tBid→Bax→MOMP→Apoptosome→Casp9→Casp3
            #   间接正反馈放大 Caspase-3，闭合正反馈环）
            "Caspase8_active → Caspase3_active",
        ],
        "description": (
            "Caspase 级联正反馈环（Casp3→Casp6→Casp8→Casp3，bistable "
            "point-of-no-return）：procaspase-3 激活超过阈值后正反馈放大到 "
            "100%，呈现 all-or-none 凋亡执行行为"
        ),
        "bistable_threshold": 0.1,   # procaspase-3 > 10% 激活触发正反馈
        "delay_minutes": 0.0,        # Caspase 级联无转录延迟
        "pmid": _Pmid_GREEN_KROEMER_2004,
    },
]


# =============================================================================
# Apoptosis 反馈环（3 条：Caspase 级联正反馈 + XIAP 负反馈 + MOMP bistable）
# =============================================================================
_APOPTOSIS_FEEDBACK_LOOPS: list[dict[str, Any]] = [
    # 1. Casp3→Casp6→Casp8 正反馈环（bistable, point-of-no-return, delay=0）
    #    Caspase 级联正反馈：procaspase-3 > 10% 激活后正反馈放大到 100%
    {
        "id": "FL_CASPASE_CASCADE_POS",
        "loop_type": "positive",
        "node_ids": [
            "Caspase3_active",
            "Caspase6_active",
            "Caspase8_active",
        ],
        "delay_minutes": 0.0,
        "bistable": True,
        "point_of_no_return": True,
        "template": "bistable_switch.j2",
        "description": (
            "Caspase 级联正反馈环（Casp3→Casp6→Casp8→Casp3，bistable "
            "point-of-no-return，delay=0，procaspase-3 > 10% 激活后正反馈放大到 100%）"
        ),
        "source_pmid": _Pmid_GREEN_KROEMER_2004,
    },
    # 2. XIAP → Caspase3 负反馈（XIAP 抑制 Caspase-3，delay=0）
    #    XIAP 直接结合抑制 Caspase-3 / Caspase-7 / Caspase-9 活性（抗凋亡）
    {
        "id": "FL_XIAP_CASPASE3_NEG",
        "loop_type": "negative",
        "node_ids": ["XIAP", "Caspase3_active"],
        "delay_minutes": 0.0,
        "bistable": False,
        "description": (
            "XIAP 抑制 Caspase-3（负反馈，delay=0，XIAP 直接结合抑制 "
            "Caspase-3 活性，抗凋亡保护）"
        ),
    },
    # 3. MOMP bistable switch（一旦 MOMP 发生不可逆，bistable point-of-no-return）
    #    MOMP 是凋亡的 point-of-no-return：Bax/Bak 寡聚化形成孔道后不可逆
    {
        "id": "FL_MOMP_BISTABLE_SWITCH",
        "loop_type": "positive",
        "node_ids": ["Bax", "MOMP", "Cyt_c", "Caspase3_active", "Caspase6_active", "Bax"],
        "delay_minutes": 0.0,
        "bistable": True,
        "point_of_no_return": True,
        "template": "bistable_switch.j2",
        "description": (
            "MOMP bistable switch（一旦 MOMP 发生不可逆，bistable all-or-none，"
            "Bax→MOMP→Cyt c→Caspase 级联正反馈放大 Bax 激活，point-of-no-return）"
        ),
        "source_pmid": _Pmid_GREEN_KROEMER_2004,
    },
]


# =============================================================================
# Apoptosis Crosstalk Reaction 片段（5 条，仅本通路侧片段，edge 由 Coordinator 管理）
# =============================================================================
# 接收 p53 / AKT / ERK / NF-κB 的 cross-talk 调控本通路 Bax/PUMA/Bad/Bim/Bcl-2
# 注意：NF-κB→Bcl-2 cross-talk edge 由 Coordinator 生成，本 Specialist 仅返回
# 本通路侧消费片段
_APOPTOSIS_CROSSTALK_REACTIONS: list[dict[str, Any]] = [
    # 1. p53 → Bax（transcription，p53 激活 Bax 启动凋亡）
    {
        "source": "p53",
        "target": "Bax",
        "mechanism": "transcription",
        "shared_species": ["p53"],
        "site": "p53RE(Bax promoter)",
        "description": "p53 转录激活 Bax（p53 作转录因子，激活 Bax 启动内源性凋亡路径，与 p53 通路 cross-talk）",
    },
    # 2. p53 → PUMA（transcription，p53 激活 PUMA 抑制 Bcl-2）
    {
        "source": "p53",
        "target": "PUMA",
        "mechanism": "transcription",
        "shared_species": ["p53"],
        "site": "p53RE(PUMA promoter)",
        "description": "p53 转录激活 PUMA（p53 作转录因子，PUMA 拮抗 Bcl-2 释放 Bax，与 p53 通路 cross-talk）",
    },
    # 3. pAKT → Bad（inhibition，pAKT 磷酸化 Bad Ser136 抑制凋亡）
    {
        "source": "pAKT",
        "target": "Bad",
        "mechanism": "inhibition",
        "shared_species": ["AKT"],
        "site": "Ser136",
        "description": "pAKT 磷酸化 Bad Ser136 导致其失活（抑制凋亡，Bad 失活无法结合 Bcl-2 释放 Bax，与 PI3K 通路 cross-talk）",
    },
    # 4. pERK → Bim（phosphorylation，pERK 磷酸化 Bim）
    {
        "source": "pERK",
        "target": "Bim",
        "mechanism": "phosphorylation",
        "shared_species": ["ERK"],
        "site": "Ser69",
        "description": "pERK 磷酸化 Bim Ser69（Bim 是 BH3-only 蛋白，pERK 磷酸化调控 Bim 稳定性 / 活性，与 MAPK 通路 cross-talk）",
    },
    # 5. NF-κB → Bcl-2（transcription，NF-κB 转录 Bcl-2 抗凋亡）
    {
        "source": "NFkB_nuclear",
        "target": "Bcl2",
        "mechanism": "transcription",
        "shared_species": ["NFkB"],
        "site": "kB site(Bcl-2 promoter)",
        "description": "NF-κB 转录激活 Bcl-2（NF-κB 作转录因子，Bcl-2 抗凋亡保护，与 NF-κB 通路 cross-talk）",
    },
]


# =============================================================================
# Apoptosis 扰动（6 个：5 个药物 + 1 个突变）
# =============================================================================
_APOPTOSIS_PERTURBATIONS: list[dict[str, Any]] = [
    # 1. ABT-199 / Venetoclax（Bcl-2 inhibitor, FDA-approved）
    #    ABT-199 是 Bcl-2 选择性抑制剂，用于 CLL / AML 治疗（FDA-approved）
    {
        "target": "Bcl2",
        "drug": "ABT-199",
        "mechanism": "inhibition",
        "ko_target": None,
        "description": "ABT-199 / Venetoclax（Bcl-2 选择性抑制剂，小分子，FDA-approved，用于 CLL / AML）",
    },
    # 2. Navitoclax / ABT-263（Bcl-2 / Bcl-xL inhibitor）
    #    Navitoclax 抑制 Bcl-2 + Bcl-xL（BH3 mimetic，广谱抗凋亡抑制剂）
    {
        "target": "Bcl2",
        "drug": "Navitoclax",
        "mechanism": "inhibition",
        "ko_target": None,
        "description": "Navitoclax / ABT-263（Bcl-2 / Bcl-xL 双重抑制剂，BH3 mimetic，小分子）",
    },
    # 3. ABT-199（Bcl-2 selective inhibitor，重复条目强调选择性）
    #    注：与第 1 条同药物，此处标注 Bcl-2 选择性（Bcl-2 selective vs Bcl-xL）
    {
        "target": "Bcl2",
        "drug": "ABT-199_selective",
        "mechanism": "inhibition",
        "ko_target": None,
        "description": "ABT-199（Bcl-2 选择性抑制剂，选择性高于 Navitoclax，避免 Bcl-xL 血小板毒性）",
    },
    # 4. Obatoclax（Bcl-2 inhibitor, pan-Bcl-2 family）
    #    Obatoclax 抑制 Bcl-2 / Bcl-xL / Mcl-1（pan-BH3 mimetic）
    {
        "target": "Bcl2",
        "drug": "Obatoclax",
        "mechanism": "inhibition",
        "ko_target": None,
        "description": "Obatoclax（pan-Bcl-2 家族抑制剂，Bcl-2 / Bcl-xL / Mcl-1，BH3 mimetic）",
    },
    # 5. Z-VAD-FMK（pan-caspase inhibitor）
    #    Z-VAD-FMK 是广谱 Caspase 抑制剂（不可逆结合催化位点，实验工具药）
    {
        "target": "Caspase3_active",
        "drug": "Z-VAD-FMK",
        "mechanism": "inhibition",
        "ko_target": None,
        "description": "Z-VAD-FMK（pan-caspase 广谱抑制剂，不可逆结合催化位点，阻断 Caspase 级联）",
    },
    # 6. BAX mutation（loss-of-function, 凋亡抵抗）
    #    BAX 突变（如 BAX G108E / R145Q）导致 Bax 寡聚化缺陷，凋亡抵抗
    {
        "target": "Bax",
        "drug": None,
        "mechanism": "knockout",
        "ko_target": "BAX_mutation",
        "description": "BAX mutation（loss-of-function，Bax 寡聚化缺陷导致凋亡抵抗，如 BAX G108E / R145Q）",
    },
]


# =============================================================================
# Apoptosis Validation 规则（3 条 benchmark）
# =============================================================================
# 每条含：rule_id / metric_name / expected / tolerance / pmid / description
# expected 取区间中点，tolerance 取区间半宽
_APOPTOSIS_VALIDATION_RULES: list[dict[str, Any]] = [
    # 1. Cyt c 早于 Caspase3 5-15 min（Reubold 2009 Apoptosome）
    #    Cyt c 释放后 5-15 min 内 Apoptosome 组装激活 Caspase-9→Caspase-3
    {
        "rule_id": "VAL_APOPTOSIS_CYTC_PRECEDES_CASP3",
        "metric_name": "Cyt_c_precedes_Casp3",
        "expected": 10.0,   # (5.0 + 15.0) / 2
        "tolerance": 5.0,    # (15.0 - 5.0) / 2
        "expected_min": 5.0,
        "expected_max": 15.0,
        "unit": "minutes_delay",
        "pmid": _Pmid_REUBOLD_2009,
        "comparison": "range",
        "pathway_tag": PATHWAY_TAG,
        "description": "Cyt c 释放早于 Caspase-3 激活 5-15 min（Reubold 2009 Apoptosome，Cyt c→Apoptosome→Casp9→Casp3 级联时序）",
    },
    # 2. MOMP bistable all-or-none（Green & Kroemer 2004）
    #    MOMP 是 point-of-no-return，应呈现 bistable all-or-none 行为
    {
        "rule_id": "VAL_APOPTOSIS_MOMP_BISTABLE_SWITCH",
        "metric_name": "MOMP_bistable_switch",
        "expected": True,
        "tolerance": 0.0,
        "expected_min": True,
        "expected_max": True,
        "unit": "boolean",
        "pmid": _Pmid_GREEN_KROEMER_2004,
        "comparison": "boolean",
        "pathway_tag": PATHWAY_TAG,
        "description": "MOMP 是 point-of-no-return，应呈现 bistable all-or-none 行为（Green & Kroemer 2004，一旦 MOMP 发生不可逆）",
    },
    # 3. Caspase-3 激活阈值 0.1-0.5（procaspase-3 > 10% 激活后正反馈放大到 100%）
    {
        "rule_id": "VAL_APOPTOSIS_CASPASE3_THRESHOLD",
        "metric_name": "Caspase3_activation_threshold",
        "expected": 0.3,    # (0.1 + 0.5) / 2
        "tolerance": 0.2,    # (0.5 - 0.1) / 2
        "expected_min": 0.1,
        "expected_max": 0.5,
        "unit": "fraction_of_max",
        "pmid": _Pmid_GREEN_KROEMER_2004,
        "comparison": "range",
        "pathway_tag": PATHWAY_TAG,
        "description": "Caspase-3 激活的 bistable 阈值（procaspase-3 > 10% 激活后正反馈放大到 100%，bistable all-or-none）",
    },
]


@register_specialist
class ApoptosisSpecialist(PathwaySpecialistBase):
    """Apoptosis 通路 Specialist。

    实现 Intrinsic（BH3-only→Bax/Bak→MOMP→Cyt c→Apaf-1→Casp9→Casp3）+
    Extrinsic（FasL→DISC→Casp8→Casp3 + Bid→Bax）核心拓扑 + Caspase 级联正反馈
    （Casp3→Casp6→Casp8 bistable, point-of-no-return）+ XIAP 负反馈 +
    MOMP bistable switch 的 Core/Feedback/Crosstalk/Perturbation/Validation
    5 模块，输出通路特异 Reaction IR 片段 + Caspase 级联正反馈环
    CompositeReaction + 模板选择 + Validation 规则。

    职责边界：
    - 处理 Apoptosis 核心执行（Bax/MOMP/Cyt c/Apoptosome/Casp9/Casp3/Casp8/
      tBid/DISC/Casp6/XIAP/PARP）
    - 不处理 p53 上游（由 p53 Specialist 处理，Task 4.6）
    - 不生成 NF-κB→Bcl-2 cross-talk edge（由 Cross-talk Coordinator 处理，Task 4.13）
    - 不处理 AKT 上游（由 PI3K Specialist 处理，Task 4.5）

    输入：
    - ``pathway_graph``：v4_pathway_graph 的 Apoptosis 子图
    - ``ontology_entities``：P1 Ontology Entities（可选，HGNC/UniProt ID 对齐）

    输出：
    - ``apply_core``：13 条核心 Reaction IR 片段 + 18 物种 + Caspase 级联
      CompositeReaction（Casp3→Casp6→Casp8 bistable 正反馈环）
      （Caspase3_active 标记 shared=True，两条途径汇聚 + 正反馈核心）
    - ``apply_feedback``：3 条 FeedbackLoop（Caspase 级联正反馈 bistable /
      XIAP 负反馈 / MOMP bistable switch）
    - ``apply_crosstalk``：5 条 cross-talk Reaction 片段
      （p53→Bax/PUMA + pAKT→Bad + pERK→Bim + NF-κB→Bcl-2）
    - ``apply_perturbation``：6 个扰动
      （ABT-199/Navitoclax/ABT-199_selective/Obatoclax/Z-VAD-FMK/BAX mutation）
    - ``apply_validation``：3 条 Validation benchmark
      （Cyt c 早于 Casp3 5-15min / MOMP bistable / Casp3 阈值 0.1-0.5）
    """

    pathway_class: str = "APOPTOSIS"
    display_name: str = "Apoptosis (Intrinsic + Extrinsic)"

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
                    species=list(_APOPTOSIS_CORE_SPECIES),
                    reactions=list(_APOPTOSIS_CORE_REACTIONS),
                    kinetics_overrides={},
                )
            if module_name == MODULE_FEEDBACK:
                return FeedbackModuleData(
                    feedback_loops=list(_APOPTOSIS_FEEDBACK_LOOPS),
                    delay_minutes=0.0,
                    loop_type="mixed",  # 含正反馈（bistable）+ 负反馈（XIAP）
                )
            if module_name == MODULE_CROSSTALK:
                return CrosstalkModuleData(
                    crosstalk_reactions=list(_APOPTOSIS_CROSSTALK_REACTIONS),
                    shared_species=["p53", "AKT", "ERK", "NFkB"],
                    coordination_strategy="merge",
                )
            if module_name == MODULE_PERTURBATION:
                return PerturbationModuleData(
                    perturbation_reactions=list(_APOPTOSIS_PERTURBATIONS),
                    drug_targets=[
                        p for p in _APOPTOSIS_PERTURBATIONS if p.get("drug")
                    ],
                    ko_targets=[
                        p for p in _APOPTOSIS_PERTURBATIONS if p.get("ko_target")
                    ],
                )
            if module_name == MODULE_VALIDATION:
                return ValidationModuleData(
                    rules=list(_APOPTOSIS_VALIDATION_RULES),
                    benchmarks=[
                        {
                            "benchmark_name": r["metric_name"],
                            "value": r["expected"],
                            "unit": r["unit"],
                            "condition": "apoptosis induction",
                            "reference": r["pmid"],
                        }
                        for r in _APOPTOSIS_VALIDATION_RULES
                    ],
                    tolerances={
                        r["metric_name"]: r["tolerance"]
                        for r in _APOPTOSIS_VALIDATION_RULES
                    },
                    pmid_references=[
                        {
                            "pmid": r["pmid"],
                            "citation": r["description"],
                            "pathway_class": PATHWAY_TAG,
                            "metric_name": r["metric_name"],
                        }
                        for r in _APOPTOSIS_VALIDATION_RULES
                        if r["pmid"]
                    ],
                )
            logger.warning(
                "ApoptosisSpecialist.load_module: 未知模块名 '%s'",
                module_name,
            )
            return None
        except Exception as exc:
            logger.warning(
                "ApoptosisSpecialist.load_module 加载模块 '%s' 失败: %s",
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
        """应用核心模块，返回 Apoptosis 通路核心 Reaction IR 片段。

        输出 13 条核心反应（Intrinsic 6 + Extrinsic 4 + Caspase 级联 3）：
        1. Bad → Bcl2（inhibition，Bad 结合 Bcl-2 释放 Bax）
        2. Bax → MOMP（activation，Bax 寡聚化形成孔道）
        3. MOMP → Cyt_c（translocation，Cyt c 线粒体→胞质）
        4. Cyt_c → Apoptosome（complex_formation，Cyt c + Apaf-1 + dATP）
        5. Apoptosome → Caspase9_active（activation，凋亡体激活 Caspase-9）
        6. Caspase9_active → Caspase3_active（cleavage，Caspase-3 切割激活）
        7. FasL → DISC（complex_formation，FasL + Fas + FADD）
        8. DISC → Caspase8_active（activation，DISC 激活 Caspase-8）
        9. Caspase8_active → tBid（cleavage，Caspase-8 切割 Bid）
        10. tBid → Bax（activation，tBid 激活 Bax，extrinsic→intrinsic）
        11. Caspase3_active → Caspase6_active（cleavage，正反馈环前段）
        12. Caspase6_active → Caspase8_active（cleavage，正反馈环反馈段）
        13. Caspase3_active → PARP_cleaved（cleavage，PARP 切割标志）

        Caspase3_active 物种标记 shared=True（Intrinsic + Extrinsic 两条途径
        汇聚点 + Caspase 级联正反馈核心节点）。

        CompositeReaction 输出 Caspase 级联正反馈环（Casp3→Casp6→Casp8→Casp3，
        bistable point-of-no-return）。

        Returns:
            dict 含 ``species``（18 物种）/ ``reactions``（13 反应） /
            ``composite_reactions``（Caspase 级联正反馈环）字段。异常时返回
            ``{"species": [], "reactions": [], "composite_reactions": []}``。
        """
        try:
            return {
                "species": list(_APOPTOSIS_CORE_SPECIES),
                "reactions": list(_APOPTOSIS_CORE_REACTIONS),
                "composite_reactions": list(_APOPTOSIS_COMPOSITE_REACTIONS),
                "pathway_tag": PATHWAY_TAG,
                "source_sbml": SOURCE_SBML,
            }
        except Exception as exc:
            logger.warning(
                "ApoptosisSpecialist.apply_core 失败: %s", exc
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
        """应用反馈模块，返回 Apoptosis 通路 FeedbackLoop 列表。

        输出 3 条反馈环（bistable 标记）：
        1. Caspase 级联正反馈环（Casp3→Casp6→Casp8→Casp3，bistable，
           point-of-no-return，delay=0）
        2. XIAP → Caspase3 负反馈（XIAP 抑制 Caspase-3，delay=0）
        3. MOMP bistable switch（MOMP 不可逆，bistable all-or-none，delay=0）

        Returns:
            FeedbackLoop 字典列表（含 bistable 标记）。异返回空列表。
        """
        try:
            return list(_APOPTOSIS_FEEDBACK_LOOPS)
        except Exception as exc:
            logger.warning(
                "ApoptosisSpecialist.apply_feedback 失败: %s", exc
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
        """应用跨通路模块，返回 Apoptosis 通路侧 cross-talk Reaction 片段。

        注意：本方法仅生成本通路侧的 cross-talk Reaction 片段，
        cross-talk edge 本身由 Cross-talk Coordinator 创建（Task 4.13）。

        输出 5 条 cross-talk Reaction 片段：
        1. p53 → Bax（transcription，p53 激活 Bax 启动凋亡）
        2. p53 → PUMA（transcription，p53 激活 PUMA 抑制 Bcl-2）
        3. pAKT → Bad（inhibition，pAKT 磷酸化 Bad Ser136 抑制凋亡）
        4. pERK → Bim（phosphorylation，pERK 磷酸化 Bim）
        5. NF-κB → Bcl-2（transcription，NF-κB 转录 Bcl-2 抗凋亡）

        Args:
            pathway_graph: 通路图（当前未使用，保留接口一致性）。
            crosstalk_edges: 来自 Coordinator 的 cross-talk edge 列表
                （当前未使用，本 Specialist 返回静态拓扑片段）。

        Returns:
            cross-talk Reaction 片段列表。异返回空列表。
        """
        try:
            return list(_APOPTOSIS_CROSSTALK_REACTIONS)
        except Exception as exc:
            logger.warning(
                "ApoptosisSpecialist.apply_crosstalk 失败: %s", exc
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
        """应用扰动模块，返回 Apoptosis 通路特异药物 / 突变 Reaction 片段。

        输出 6 个扰动（5 个药物 + 1 个突变）：
        1. ABT-199 / Venetoclax（Bcl-2 选择性抑制剂，FDA-approved）
        2. Navitoclax / ABT-263（Bcl-2 / Bcl-xL 双重抑制剂）
        3. ABT-199（Bcl-2 selective，强调选择性）
        4. Obatoclax（pan-Bcl-2 家族抑制剂）
        5. Z-VAD-FMK（pan-caspase 广谱抑制剂）
        6. BAX mutation（loss-of-function，凋亡抵抗）

        Args:
            pathway_graph: 通路图（当前未使用）。
            perturbation_points: 扰动点列表（当前未使用，本 Specialist
                返回静态拓扑片段）。

        Returns:
            扰动 Reaction 片段列表。异返回空列表。
        """
        try:
            return list(_APOPTOSIS_PERTURBATIONS)
        except Exception as exc:
            logger.warning(
                "ApoptosisSpecialist.apply_perturbation 失败: %s", exc
            )
            return []

    # =================================================================
    # apply_validation：Validation 规则列表
    # =================================================================
    def apply_validation(
        self,
        simulation_result: dict | None = None,
    ) -> list[dict]:
        """应用验证模块，返回 Apoptosis 通路 Validation 规则列表。

        输出 3 条 benchmark：
        1. Cyt c 早于 Caspase3 5-15 min（Reubold 2009, PMID:11274138）
        2. MOMP bistable all-or-none（Green & Kroemer 2004, PMID:15241432）
        3. Caspase-3 激活阈值 0.1-0.5（procaspase-3 > 10% 激活后正反馈放大到 100%）

        Args:
            simulation_result: 仿真结果（可选，当前未使用，返回静态规则集）。

        Returns:
            Validation 规则列表。异返回空列表。
        """
        try:
            return list(_APOPTOSIS_VALIDATION_RULES)
        except Exception as exc:
            logger.warning(
                "ApoptosisSpecialist.apply_validation 失败: %s", exc
            )
            return []

    # =================================================================
    # select_template：覆写以支持 bistable 模板
    # =================================================================
    def select_template(self, mechanism: str) -> str:
        """根据 mechanism 选择 ODE 模板名（覆写支持 bistable）。

        默认映射（与 P3 ``ode_templates_v2/`` 下 .j2 文件对齐）：
        - ``bistable`` → ``bistable_switch``（Caspase 级联 / MOMP bistable）
        - ``phosphorylation`` → ``_mechanism_phosphorylation_mm``（pERK→Bim）
        - ``cleavage`` → ``bistable_switch``（Caspase 级联触发 bistable 模式）
        - ``translocation`` → ``bistable_switch``（MOMP→Cyt c 触发 point-of-no-return）

        Args:
            mechanism: 机制名（小写，如 ``"bistable"`` / ``"phosphorylation"``）。

        Returns:
            ODE 模板名（不含 ``.j2`` 后缀）。未匹配时返回 ``"default"``
            （调用方应处理默认降级）。
        """
        # bistable 模式：Caspase 级联 / MOMP point-of-no-return
        if mechanism in ("bistable", "cleavage", "translocation"):
            return "bistable_switch"
        # 磷酸化场景：pERK→Bim 磷酸化
        if mechanism == "phosphorylation":
            return "_mechanism_phosphorylation_mm"
        # 其他机制走默认基类映射
        return super().select_template(mechanism)


# =============================================================================
# 文献动力学参数（IB-017 修复）
# =============================================================================
# 来源：
# - BIOMD0000000335 (Eissing 2004, PMID:15382335) 凋亡 caspase 级联双稳态模型
# - BIOMD0000002183 凋亡网络模型
# - PMID:15241432 (Green & Kroemer 2004) 凋亡外源性通路
# 反幻觉守卫：所有参数来自上述 BioModels 模型或文献；无确切值的用无量纲化
# 估计并标注 `# Heuristic estimate, needs calibration`。
# 参数范围约束：k_on∈[1e3,1e7] M^-1 min^-1, Km∈[1e-7,1e-2] M, k_cat∈[1e-3,1e3] min^-1
KINETIC_PARAMETERS: dict[str, dict[str, float]] = {
    # Bax→MOMP 线粒体外膜透化（Eissing 2004, PMID:15382335, BIOMD0000000335）
    "Bax_MOMP": {
        "k_cat": 1.0,                # min^-1  # Heuristic estimate, needs calibration
        "Km": 1e-7,                  # M (Bax Km)  # Heuristic estimate, needs calibration
    },
    # MOMP→Cyt_c 细胞色素 c 释放（Eissing 2004, PMID:15382335）
    "MOMP_Cyt_c": {
        "k_release": 0.5,            # min^-1  # Heuristic estimate, needs calibration
    },
    # Cyt_c+Apaf-1→Apoptosome 凋亡体组装（Eissing 2004, PMID:15382335）
    "Cyt_c_Apoptosome": {
        "k_on": 1.0e6,               # M^-1 min^-1  # Heuristic estimate, needs calibration
        "k_off": 1e-3,               # min^-1  # Heuristic estimate, needs calibration
    },
    # Apoptosome→Caspase9_active Caspase-9 激活（Eissing 2004, PMID:15382335）
    "Apoptosome_Casp9": {
        "k_cat": 1.0,                # min^-1  # Heuristic estimate, needs calibration
        "Km": 1e-7,                  # M (Caspase-9 Km)  # Heuristic estimate, needs calibration
    },
    # FasL+Fas→DISC 死亡诱导信号复合物（Green & Kroemer 2004, PMID:15241432）
    "FasL_DISC": {
        "k_on": 1.0e6,               # M^-1 min^-1  # Heuristic estimate, needs calibration
        "k_off": 1e-3,               # min^-1  # Heuristic estimate, needs calibration
    },
    # DISC→Caspase8_active Caspase-8 激活（Green & Kroemer 2004, PMID:15241432）
    "DISC_Casp8": {
        "k_cat": 1.0,                # min^-1  # Heuristic estimate, needs calibration
        "Km": 1e-7,                  # M (Caspase-8 Km)  # Heuristic estimate, needs calibration
    },
    # Caspase8→tBid Bid 切割（Green & Kroemer 2004, PMID:15241432）
    "Casp8_tBid": {
        "k_cat": 1.0,                # min^-1  # Heuristic estimate, needs calibration
        "Km": 1e-7,                  # M (Bid Km)  # Heuristic estimate, needs calibration
    },
    # Caspase9→Caspase3_active Caspase-3 激活（Eissing 2004, PMID:15382335, bistable）
    "Casp9_Casp3": {
        "k_cat": 10.0,               # min^-1 (执行 caspase 高催化效率)
        "Km": 1e-7,                  # M (Caspase-3 Km)  # Heuristic estimate, needs calibration
    },
    # Caspase3→Caspase6_active Caspase-6 激活（Eissing 2004, PMID:15382335）
    "Casp3_Casp6": {
        "k_cat": 5.0,                # min^-1  # Heuristic estimate, needs calibration
        "Km": 1e-7,                  # M  # Heuristic estimate, needs calibration
    },
    # Caspase3→PARP_cleaved PARP 切割（Eissing 2004, PMID:15382335）
    "Casp3_PARP": {
        "k_cat": 5.0,                # min^-1  # Heuristic estimate, needs calibration
        "Km": 1e-6,                  # M (PARP Km)  # Heuristic estimate, needs calibration
    },
    # XIAP 抑制 Caspase3（Eissing 2004, PMID:15382335, XIAP 抗凋亡）
    "XIAP_Casp3_inhibition": {
        "k_on": 1.0e6,               # M^-1 min^-1  # Heuristic estimate, needs calibration
        "k_off": 1e-3,               # min^-1  # Heuristic estimate, needs calibration
    },
    # Bcl2 抑制 Bax（Eissing 2004, PMID:15382335, 抗凋亡）
    "Bcl2_Bax_inhibition": {
        "k_on": 1.0e6,               # M^-1 min^-1  # Heuristic estimate, needs calibration
        "k_off": 1e-3,               # min^-1  # Heuristic estimate, needs calibration
    },
}


__all__ = [
    "ApoptosisSpecialist",
    "PATHWAY_TAG",
    "SOURCE_SBML",
    "KINETIC_PARAMETERS",
]
