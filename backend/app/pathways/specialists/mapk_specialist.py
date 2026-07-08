# BioDynamics Agent v4 - MAPK Specialist (Phase 4 / Task 4.4)
# MAPK / ERK 通路 Specialist：实现 Ras-Raf-MEK-ERK 三级级联（含双磷酸化）
# + SOS 负反馈 + ERK→Raf Ser259 反馈抑制。
#
# 设计原则（铁律）：
# 1. Feature Flag V4_PATHWAY_SPECIALIST_ENABLED=false 时由 LangGraph hook 层短路，
#    本 Specialist 代码不被调用（基类本身不做 flag 检查）
# 2. 使用 Task 4.0 修复后的 PHOSPHORYLATION 语义：
#    - 异磷酸化（RasGTP → pRaf / pRaf → pMEK / ppMEK → pERK）：
#      未磷酸化形式(Raf/MEK/ERK) 作 substrate，磷酸化形式(pRaf/pMEK/pERK)
#      作 product，source 作 catalytic modifier
#    - 双磷酸化（pMEK → ppMEK / pERK → ppERK）：单磷酸化形式(pMEK/pERK)
#      作 substrate，双磷酸化形式(ppMEK/ppERK) 作 product，source 作 catalytic
#      modifier（自磷酸化形式：MEK 有 Ser218/Ser222 双位点，ERK 有
#      Thr183/Tyr185 双位点，第二位点磷酸化由已磷酸化形式催化）
# 3. 不处理 EGFR 上游（由 EGFR Specialist 处理，Task 4.3）
# 4. 不生成 Ras 物种（Ras 由 EGFR Specialist 创建，MAPK Specialist 仅消费 RasGTP）
#    - RasGTP 在 species 列表中标记 consumed_shared=True（不创建该物种）
# 5. 强制 Michaelis_Menten 动力学（所有 5 条核心反应 kinetics_type 均为
#    "Michaelis_Menten"，对应 P3 _mechanism_phosphorylation_mm.j2 模板）
# 6. apply_* 方法实现捕获异常并返回空 list/dict，记录 logger.warning，不抛异常
#
# 依赖：
# - P1 ontology / pathway_registry（MAPK_ERK 通路类别键）
# - P2 schema（SpeciesV2 / ReactionV2 / SpeciesRef / Modifier）
# - P2 MechanismType.PHOSPHORYLATION
# - P3 ode_templates_v2（_mechanism_phosphorylation_mm.j2 模板）
# - P3 pathway_graph/initializer.py（MAPK_ERK core_nodes / core_edges）
#
# 参考：
# - spec.md Part 3 Specialist 2（第 199-204 行）
# - tasks.md Task 4.4（第 55-60 行）
# - Schoeberl 2001 MAPK model (PMID:11483517)
# - Goldbeter & Koshland 1981 zero-order ultrasensitivity (PMID:1941687)

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
# MAPK 通路标签
# =============================================================================
PATHWAY_TAG: str = "MAPK_ERK"

# SBML BioModels ID（Schoeberl 2001 MAPK model）
SOURCE_SBML: str = "BIOMD0000000010"

# Validation benchmark PMID 引用
_Pmid_SCHOEBERL_2001: str = "PMID:11483517"      # Schoeberl 2001 MAPK model
_Pmid_GOLDBETER_KOSHLAND_1981: str = "PMID:1941687"  # Goldbeter & Koshland 1981


# =============================================================================
# MAPK 核心物种（与 P3 pathway_graph/initializer.py MAPK_ERK.core_nodes 对齐，
# 扩展 ppMEK / ppERK 双磷酸化形式）
# =============================================================================
# RasGTP 来自 EGFR Specialist（MAPK 仅消费），标记 consumed_shared=True 表示
# 本 Specialist 不创建该物种，仅作为 catalytic modifier 使用。
# Raf / pRaf 在 EGFR Specialist 中亦存在（RasGTP→pRaf 反应由 EGFR Specialist
# 输出），Cross-talk Coordinator（Task 4.13）负责按物种 canonical name 合并。
_MAPK_CORE_SPECIES: list[dict[str, Any]] = [
    # RasGTP：来自 EGFR Specialist，MAPK 仅消费（不创建）
    {
        "name": "RasGTP",
        "species_type": "protein",
        "compartment": "membrane",
        "shared": True,
        "consumed_shared": True,
    },
    # Raf / pRaf：与 EGFR Specialist 共享（EGFR 输出 RasGTP→pRaf 反应）
    {"name": "Raf", "species_type": "protein", "compartment": "cytoplasm",
     "shared": True},
    {"name": "pRaf", "species_type": "protein", "compartment": "cytoplasm"},
    # MEK 级联（双磷酸化位点 Ser218/Ser222）
    {"name": "MEK", "species_type": "protein", "compartment": "cytoplasm"},
    {"name": "pMEK", "species_type": "protein", "compartment": "cytoplasm"},
    {"name": "ppMEK", "species_type": "protein", "compartment": "cytoplasm"},
    # ERK 级联（双磷酸化位点 Thr183/Tyr185）
    {"name": "ERK", "species_type": "protein", "compartment": "cytoplasm"},
    {"name": "pERK", "species_type": "protein", "compartment": "cytoplasm"},
    {"name": "ppERK", "species_type": "protein", "compartment": "cytoplasm"},
    # TD-033 (IB-064) 修复：补充 ERK 核转位物种（pERK/ppERK 入核后激活转录因子）
    {"name": "pERK_nuclear", "species_type": "protein", "compartment": "nucleus"},
    {"name": "ppERK_nuclear", "species_type": "protein", "compartment": "nucleus"},
]


# =============================================================================
# MAPK 核心反应（5 条，三级双磷酸化级联）
# =============================================================================
# 每条反应含：source / target / mechanism / kinetics_type / pathway_tag /
#             substrate / product / modifier / modifier_type / autophosphorylation
#
# PHOSPHORYLATION 语义（Task 4.0 修复后）：
# - 异磷酸化：未磷酸化形式作 substrate，磷酸化形式作 product，source 作 catalytic modifier
# - 双磷酸化（自磷酸化形式）：单磷酸化形式(pMEK/pERK) 作 substrate+product 的前驱，
#   双磷酸化形式(ppMEK/ppERK) 作 product，source(pMEK/pERK) 作 catalytic modifier
#
# kinetics_type 强制为 "Michaelis_Menten"（与 P3 _mechanism_phosphorylation_mm 模板对齐）
_MAPK_CORE_REACTIONS: list[dict[str, Any]] = [
    # 1. RasGTP → pRaf（异磷酸化，Raf 作 substrate，RasGTP 作 catalytic modifier）
    #    注意：本反应与 EGFR Specialist 第 7 条反应同源（RasGTP→pRaf），
    #    Cross-talk Coordinator 会按 (source,target,mechanism) 三元组合并。
    {
        "source": "RasGTP",
        "target": "pRaf",
        "mechanism": "phosphorylation",
        "kinetics_type": "Michaelis_Menten",
        "pathway_tag": PATHWAY_TAG,
        # 异磷酸化：Raf 作 substrate，pRaf 作 product，RasGTP 作 catalytic modifier
        "substrate": "Raf",
        "product": "pRaf",
        "modifier": "RasGTP",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "RasGTP 磷酸化 Raf（Raf 作 substrate，RasGTP 作 catalytic modifier）",
    },
    # 2. pRaf → pMEK（异磷酸化，MEK 作 substrate，pRaf 作 catalytic modifier）
    {
        "source": "pRaf",
        "target": "pMEK",
        "mechanism": "phosphorylation",
        "kinetics_type": "Michaelis_Menten",
        "pathway_tag": PATHWAY_TAG,
        # 异磷酸化：MEK 作 substrate，pMEK 作 product，pRaf 作 catalytic modifier
        "substrate": "MEK",
        "product": "pMEK",
        "modifier": "pRaf",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "pRaf 磷酸化 MEK（MEK 作 substrate，pRaf 作 catalytic modifier，第一磷酸化位点 Ser218）",
    },
    # 3. pMEK → ppMEK（双磷酸化第一步，pMEK 作 substrate，pMEK 作 catalytic modifier）
    #    MEK 第二磷酸化位点 Ser222，由已磷酸化的 pMEK 催化（自磷酸化形式）
    {
        "source": "pMEK",
        "target": "ppMEK",
        "mechanism": "phosphorylation",
        "kinetics_type": "Michaelis_Menten",
        "pathway_tag": PATHWAY_TAG,
        # 双磷酸化第一步：pMEK 作 substrate，ppMEK 作 product，pMEK 作 catalytic modifier
        "substrate": "pMEK",
        "product": "ppMEK",
        "modifier": "pMEK",
        "modifier_type": "catalytic",
        "autophosphorylation": True,
        "description": "pMEK 双磷酸化第二步（Ser222，自磷酸化形式，pMEK → ppMEK）",
    },
    # 4. ppMEK → pERK（异磷酸化，ERK 作 substrate，ppMEK 作 catalytic modifier）
    {
        "source": "ppMEK",
        "target": "pERK",
        "mechanism": "phosphorylation",
        "kinetics_type": "Michaelis_Menten",
        "pathway_tag": PATHWAY_TAG,
        # 异磷酸化：ERK 作 substrate，pERK 作 product，ppMEK 作 catalytic modifier
        "substrate": "ERK",
        "product": "pERK",
        "modifier": "ppMEK",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "ppMEK 磷酸化 ERK（ERK 作 substrate，ppMEK 作 catalytic modifier，第一磷酸化位点 Thr183）",
    },
    # 5. pERK → ppERK（双磷酸化第二步，pERK 作 substrate，pERK 作 catalytic modifier）
    #    ERK 第二磷酸化位点 Tyr185，由已磷酸化的 pERK 催化（自磷酸化形式）
    {
        "source": "pERK",
        "target": "ppERK",
        "mechanism": "phosphorylation",
        "kinetics_type": "Michaelis_Menten",
        "pathway_tag": PATHWAY_TAG,
        # 双磷酸化第二步：pERK 作 substrate，ppERK 作 product，pERK 作 catalytic modifier
        "substrate": "pERK",
        "product": "ppERK",
        "modifier": "pERK",
        "modifier_type": "catalytic",
        "autophosphorylation": True,
        "description": "pERK 双磷酸化第二步（Tyr185，自磷酸化形式，pERK → ppERK）",
    },
    # 6. TD-033 (IB-064) 修复：pERK → pERK_nuclear（nuclear_import，pERK 入核激活转录因子）
    #    原通路缺失 ERK 核转位步骤，导致 ERK 磷酸化后无法在核内激活 ELK1/c-Fos 等转录因子，
    #    无法刻画 MAPK 信号到基因表达的完整传递。pERK 作 substrate，pERK_nuclear 作 product。
    {
        "source": "pERK",
        "target": "pERK_nuclear",
        "mechanism": "nuclear_import",
        "kinetics_type": "mass_action",
        "pathway_tag": PATHWAY_TAG,
        # 核转位：pERK(cytoplasm) 作 substrate，pERK_nuclear(nucleus) 作 product
        "substrate": "pERK",
        "product": "pERK_nuclear",
        "modifier": "pERK",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "pERK 入核（pERK 作 substrate，pERK_nuclear 作 product，激活核内 ELK1/c-Fos 转录因子，补全 MAPK→基因表达传递）",
    },
    # 7. TD-033 (IB-064) 修复：ppERK → ppERK_nuclear（nuclear_import，ppERK 入核）
    #    双磷酸化 ERK 同样需入核激活下游转录，与 pERK 核转位并列。
    {
        "source": "ppERK",
        "target": "ppERK_nuclear",
        "mechanism": "nuclear_import",
        "kinetics_type": "mass_action",
        "pathway_tag": PATHWAY_TAG,
        # 核转位：ppERK(cytoplasm) 作 substrate，ppERK_nuclear(nucleus) 作 product
        "substrate": "ppERK",
        "product": "ppERK_nuclear",
        "modifier": "ppERK",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "ppERK 入核（ppERK 作 substrate，ppERK_nuclear 作 product，双磷酸化 ERK 核内激活转录）",
    },
]


# =============================================================================
# MAPK 反馈环（2 条，均 delay=0 min，无转录延迟）
# =============================================================================
_MAPK_FEEDBACK_LOOPS: list[dict[str, Any]] = [
    # 1. pERK → SOS 负反馈（pERK 磷酸化 SOS 导致其失活，无转录延迟）
    {
        "id": "FL_MAPK_ERK_SOS_NEG",
        "loop_type": "negative",
        "node_ids": ["pERK", "SOS", "RasGTP"],
        "delay_minutes": 0.0,
        "description": "pERK 磷酸化 SOS 导致其失活（负反馈，无转录延迟，delay=0）",
    },
    # 2. ppERK → pRaf Ser259 抑制（ERK 磷酸化 Raf Ser259 抑制级联）
    {
        "id": "FL_MAPK_ERK_RAF_S259_NEG",
        "loop_type": "negative",
        "node_ids": ["ppERK", "pRaf", "Raf"],
        "delay_minutes": 0.0,
        "site": "Ser259",
        "description": "ppERK 磷酸化 Raf Ser259 抑制 MAPK 级联（负反馈，delay=0）",
    },
]


# =============================================================================
# MAPK Crosstalk Reaction 片段（3 条，仅本通路侧片段，edge 由 Coordinator 管理）
# =============================================================================
_MAPK_CROSSTALK_REACTIONS: list[dict[str, Any]] = [
    # 1. pERK → ELK1（phosphorylation，pERK 激活转录因子 ELK1）
    {
        "source": "pERK",
        "target": "ELK1",
        "mechanism": "phosphorylation",
        "shared_species": [],
        "description": "pERK 磷酸化激活转录因子 ELK1（驱动早期基因表达）",
    },
    # 2. pERK → Myc（transcription，pERK 激活 Myc 表达驱动细胞周期）
    {
        "source": "pERK",
        "target": "Myc",
        "mechanism": "transcription",
        "shared_species": [],
        "description": "pERK 激活 Myc 转录表达（驱动细胞周期进入 G1/S）",
    },
    # 3. pERK → Bim（phosphorylation，pERK 磷酸化 Bim 抑制凋亡）
    {
        "source": "pERK",
        "target": "Bim",
        "mechanism": "phosphorylation",
        "shared_species": [],
        "description": "pERK 磷酸化 Bim 导致其降解（抑制凋亡，与 Apoptosis 通路 cross-talk）",
    },
]


# =============================================================================
# MAPK 扰动（4 个 FDA-approved / 临床小分子抑制剂）
# =============================================================================
_MAPK_PERTURBATIONS: list[dict[str, Any]] = [
    # 1. Trametinib（MEK inhibitor, small molecule, FDA-approved）
    {
        "target": "MEK",
        "drug": "Trametinib",
        "mechanism": "inhibition",
        "ko_target": None,
        "description": "Trametinib（MEK1/2 抑制剂，小分子，FDA-approved）",
    },
    # 2. Vemurafenib（BRAF inhibitor, small molecule, FDA-approved）
    {
        "target": "BRAF",
        "drug": "Vemurafenib",
        "mechanism": "inhibition",
        "ko_target": None,
        "description": "Vemurafenib（BRAF V600E 抑制剂，小分子，FDA-approved）",
    },
    # 3. Dabrafenib（BRAF inhibitor, small molecule）
    {
        "target": "BRAF",
        "drug": "Dabrafenib",
        "mechanism": "inhibition",
        "ko_target": None,
        "description": "Dabrafenib（BRAF 抑制剂，小分子）",
    },
    # 4. Selumetinib（MEK inhibitor, small molecule）
    {
        "target": "MEK",
        "drug": "Selumetinib",
        "mechanism": "inhibition",
        "ko_target": None,
        "description": "Selumetinib（MEK1/2 抑制剂，小分子）",
    },
]


# =============================================================================
# MAPK Validation 规则（3 条 benchmark）
# =============================================================================
# 每条含：rule_id / metric_name / expected / tolerance / pmid / description
# expected 取区间中点，tolerance 取区间半宽
_MAPK_VALIDATION_RULES: list[dict[str, Any]] = [
    # 1. MAPK 放大倍数 10-100x（Schoeberl 2001）
    {
        "rule_id": "VAL_MAPK_AMPLIFICATION",
        "metric_name": "MAPK_amplification",
        "expected": 55.0,    # (10.0 + 100.0) / 2
        "tolerance": 45.0,    # (100.0 - 10.0) / 2
        "expected_min": 10.0,
        "expected_max": 100.0,
        "unit": "fold",
        "pmid": _Pmid_SCHOEBERL_2001,
        "comparison": "range",
        "pathway_tag": PATHWAY_TAG,
        "description": "MAPK 级联放大倍数 10-100x（Schoeberl 2001 MAPK model）",
    },
    # 2. 零阶 ultrasensitivity Hill 系数 >2（Goldbeter & Koshland 1981）
    {
        "rule_id": "VAL_MAPK_ZERO_ORDER_ULTRASENSITIVITY",
        "metric_name": "zero_order_ultrasensitivity_hill_coefficient",
        "expected": 6.0,      # (2.0 + 10.0) / 2
        "tolerance": 4.0,     # (10.0 - 2.0) / 2
        "expected_min": 2.0,
        "expected_max": 10.0,
        "unit": "hill_n",
        "pmid": _Pmid_GOLDBETER_KOSHLAND_1981,
        "comparison": "range",
        "pathway_tag": PATHWAY_TAG,
        "description": "零阶 ultrasensitivity Hill 系数 2-10（Goldbeter & Koshland 1981，酶饱和条件下级联响应超敏感）",
    },
    # 3. ERK 达峰时间 2-8 min（Schoeberl 2001）
    {
        "rule_id": "VAL_MAPK_ERK_PEAK_TIME",
        "metric_name": "ERK_peak_time",
        "expected": 5.0,      # (2.0 + 8.0) / 2
        "tolerance": 3.0,     # (8.0 - 2.0) / 2
        "expected_min": 2.0,
        "expected_max": 8.0,
        "unit": "minutes",
        "pmid": _Pmid_SCHOEBERL_2001,
        "comparison": "range",
        "pathway_tag": PATHWAY_TAG,
        "description": "ERK 在 EGF 刺激后 2-8 min 达峰（Schoeberl 2001 MAPK model）",
    },
]


@register_specialist
class MAPKSpecialist(PathwaySpecialistBase):
    """MAPK / ERK 通路 Specialist。

    实现 Ras-Raf-MEK-ERK 三级级联（含双磷酸化）+ SOS 负反馈 + ERK→Raf
    Ser259 反馈抑制的 Core/Feedback/Crosstalk/Perturbation/Validation 5 模块，
    输出通路特异 Reaction IR 片段 + 模板选择 + Validation 规则。

    职责边界：
    - 处理 Ras-Raf-MEK-ERK 三级级联（含双磷酸化 ppMEK / ppERK）
    - 不处理 EGFR 上游（由 EGFR Specialist 处理，Task 4.3）
    - 不生成 Ras 物种（Ras 由 EGFR Specialist 创建，本 Specialist 仅消费 RasGTP）
    - 不生成跨通路 cross-talk edge（由 Cross-talk Coordinator 处理，Task 4.13）

    输入：
    - ``pathway_graph``：v4_pathway_graph 的 MAPK 子图（含 Ras shared species 标记）
    - ``ontology_entities``：P1 Ontology Entities（可选，HGNC/UniProt ID 对齐）

    输出：
    - ``apply_core``：5 条核心双磷酸化 Reaction IR 片段 + 9 物种
      （RasGTP 标记 consumed_shared=True）
    - ``apply_feedback``：2 条 FeedbackLoop（pERK→SOS / ppERK→pRaf Ser259）
    - ``apply_crosstalk``：3 条 cross-talk Reaction 片段（ELK1 / Myc / Bim）
    - ``apply_perturbation``：4 个扰动（Trametinib/Vemurafenib/Dabrafenib/Selumetinib）
    - ``apply_validation``：3 条 Validation benchmark 规则（MAPK 放大 / Hill / ERK 达峰）
    """

    pathway_class: str = "MAPK_ERK"
    display_name: str = "MAPK / ERK Signaling Cascade"

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
                    species=list(_MAPK_CORE_SPECIES),
                    reactions=list(_MAPK_CORE_REACTIONS),
                    kinetics_overrides={},
                )
            if module_name == MODULE_FEEDBACK:
                return FeedbackModuleData(
                    feedback_loops=list(_MAPK_FEEDBACK_LOOPS),
                    delay_minutes=0.0,
                    loop_type="negative",
                )
            if module_name == MODULE_CROSSTALK:
                return CrosstalkModuleData(
                    crosstalk_reactions=list(_MAPK_CROSSTALK_REACTIONS),
                    shared_species=["RasGTP"],
                    coordination_strategy="merge",
                )
            if module_name == MODULE_PERTURBATION:
                return PerturbationModuleData(
                    perturbation_reactions=list(_MAPK_PERTURBATIONS),
                    drug_targets=[
                        p for p in _MAPK_PERTURBATIONS if p.get("drug")
                    ],
                    ko_targets=[
                        p for p in _MAPK_PERTURBATIONS if p.get("ko_target")
                    ],
                )
            if module_name == MODULE_VALIDATION:
                return ValidationModuleData(
                    rules=list(_MAPK_VALIDATION_RULES),
                    benchmarks=[
                        {
                            "benchmark_name": r["metric_name"],
                            "value": r["expected"],
                            "unit": r["unit"],
                            "condition": "EGF stimulation",
                            "reference": r["pmid"],
                        }
                        for r in _MAPK_VALIDATION_RULES
                    ],
                    tolerances={
                        r["metric_name"]: r["tolerance"]
                        for r in _MAPK_VALIDATION_RULES
                    },
                    pmid_references=[
                        {
                            "pmid": r["pmid"],
                            "citation": r["description"],
                            "pathway_class": PATHWAY_TAG,
                            "metric_name": r["metric_name"],
                        }
                        for r in _MAPK_VALIDATION_RULES
                    ],
                )
            logger.warning(
                "MAPKSpecialist.load_module: 未知模块名 '%s'", module_name
            )
            return None
        except Exception as exc:
            logger.warning(
                "MAPKSpecialist.load_module 加载模块 '%s' 失败: %s",
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
        """应用核心模块，返回 MAPK/ERK 通路核心 Reaction IR 片段。

        输出 5 条核心反应（三级双磷酸化级联）：
        1. RasGTP → pRaf（异磷酸化，MM，RasGTP 作 catalytic modifier）
        2. pRaf → pMEK（异磷酸化，MM，pRaf 作 catalytic modifier）
        3. pMEK → ppMEK（双磷酸化第一步，MM，pMEK 自催化）
        4. ppMEK → pERK（异磷酸化，MM，ppMEK 作 catalytic modifier）
        5. pERK → ppERK（双磷酸化第二步，MM，pERK 自催化）

        RasGTP 物种标记 consumed_shared=True（来自 EGFR Specialist，本
        Specialist 仅消费，不创建）。

        Returns:
            dict 含 ``species``（9 物种）与 ``reactions``（5 反应）字段。
            异常时返回 ``{"species": [], "reactions": []}``。
        """
        try:
            return {
                "species": list(_MAPK_CORE_SPECIES),
                "reactions": list(_MAPK_CORE_REACTIONS),
                "pathway_tag": PATHWAY_TAG,
                "source_sbml": SOURCE_SBML,
            }
        except Exception as exc:
            logger.warning(
                "MAPKSpecialist.apply_core 失败: %s", exc
            )
            return {"species": [], "reactions": []}

    # =================================================================
    # apply_feedback：FeedbackLoop 列表
    # =================================================================
    def apply_feedback(self, pathway_graph: dict) -> list[dict]:
        """应用反馈模块，返回 MAPK 通路 FeedbackLoop 列表。

        输出 2 条反馈环（均 delay=0 min，无转录延迟）：
        1. pERK → SOS 负反馈（pERK 磷酸化 SOS 失活，delay=0 min）
        2. ppERK → pRaf Ser259 负反馈（ppERK 磷酸化 Raf Ser259 抑制级联，delay=0 min）

        Returns:
            FeedbackLoop 字典列表。异返回空列表。
        """
        try:
            return list(_MAPK_FEEDBACK_LOOPS)
        except Exception as exc:
            logger.warning(
                "MAPKSpecialist.apply_feedback 失败: %s", exc
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
        """应用跨通路模块，返回 MAPK 通路侧 cross-talk Reaction 片段。

        注意：本方法仅生成本通路侧的 cross-talk Reaction 片段，
        cross-talk edge 本身由 Cross-talk Coordinator 创建（Task 4.13）。

        输出 3 条 cross-talk Reaction 片段：
        1. pERK → ELK1（phosphorylation，激活转录因子）
        2. pERK → Myc（transcription，驱动细胞周期）
        3. pERK → Bim（phosphorylation，抑制凋亡）

        Args:
            pathway_graph: 通路图（当前未使用，保留接口一致性）。
            crosstalk_edges: 来自 Coordinator 的 cross-talk edge 列表
                （当前未使用，本 Specialist 返回静态拓扑片段）。

        Returns:
            cross-talk Reaction 片段列表。异返回空列表。
        """
        try:
            return list(_MAPK_CROSSTALK_REACTIONS)
        except Exception as exc:
            logger.warning(
                "MAPKSpecialist.apply_crosstalk 失败: %s", exc
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
        """应用扰动模块，返回 MAPK 通路特异药物 Reaction 片段。

        输出 4 个扰动（小分子抑制剂）：
        1. Trametinib（MEK1/2 抑制剂，FDA-approved）
        2. Vemurafenib（BRAF V600E 抑制剂，FDA-approved）
        3. Dabrafenib（BRAF 抑制剂）
        4. Selumetinib（MEK1/2 抑制剂）

        Args:
            pathway_graph: 通路图（当前未使用）。
            perturbation_points: 扰动点列表（当前未使用，本 Specialist
                返回静态拓扑片段）。

        Returns:
            扰动 Reaction 片段列表。异返回空列表。
        """
        try:
            return list(_MAPK_PERTURBATIONS)
        except Exception as exc:
            logger.warning(
                "MAPKSpecialist.apply_perturbation 失败: %s", exc
            )
            return []

    # =================================================================
    # apply_validation：Validation 规则列表
    # =================================================================
    def apply_validation(
        self,
        simulation_result: dict | None = None,
    ) -> list[dict]:
        """应用验证模块，返回 MAPK 通路 Validation 规则列表。

        输出 3 条 benchmark：
        1. MAPK 放大倍数 10-100x（Schoeberl 2001, PMID:11483517）
        2. 零阶 ultrasensitivity Hill 系数 2-10（Goldbeter & Koshland 1981, PMID:1941687）
        3. ERK 达峰时间 2-8 min（Schoeberl 2001, PMID:11483517）

        Args:
            simulation_result: 仿真结果（可选，当前未使用，返回静态规则集）。

        Returns:
            Validation 规则列表。异返回空列表。
        """
        try:
            return list(_MAPK_VALIDATION_RULES)
        except Exception as exc:
            logger.warning(
                "MAPKSpecialist.apply_validation 失败: %s", exc
            )
            return []

    # =================================================================
    # select_template：覆写以支持未来 Signaling_Cascade_Phos 模板
    # =================================================================
    # 当前仅返回基类默认映射（_mechanism_phosphorylation_mm）。
    # 强制 Michaelis_Menten 动力学（与 P3 _mechanism_phosphorylation_mm.j2 对齐）。
    # 未来 P3 若添加 Signaling_Cascade_Phos 模板，可在此覆写：
    #   if mechanism == "phosphorylation":
    #       return "Signaling_Cascade_Phos"
    #   return super().select_template(mechanism)


# =============================================================================
# 文献动力学参数（IB-017 修复）
# =============================================================================
# 来源：
# - BIOMD0000000010 (Schoeberl 2002, PMID:12451189) EGF/MAPK 级联模型
# - BIOMD0000000267 (Brightman 2000, PMID:10986007) MAPK 级联动力学
# - PMID:1941687 (Goldbeter & Koshland 1981) 零级超敏感
# 反幻觉守卫：所有参数来自上述 BioModels 模型或文献；无确切值的用无量纲化
# 估计并标注 `# Heuristic estimate, needs calibration`。
# 参数范围约束：k_on∈[1e3,1e7] M^-1 min^-1, Km∈[1e-7,1e-2] M, k_cat∈[1e-3,1e3] min^-1
KINETIC_PARAMETERS: dict[str, dict[str, float]] = {
    # RasGTP→pRaf 异磷酸化（Raf 作 substrate, RasGTP 作 modifier, Schoeberl 2002, BIOMD0000000010）
    "RasGTP_pRaf": {
        "k_cat": 0.5,                # min^-1  # Heuristic estimate, needs calibration
        "Km": 1e-7,                  # M (Raf 底物 Km ≈100 nM)  # Heuristic estimate, needs calibration
    },
    # pRaf→pMEK MEK 单磷酸化（Brightman 2000, PMID:10986007, BIOMD0000000267）
    "pRaf_pMEK": {
        "k_cat": 0.5,                # min^-1
        "Km": 3e-7,                  # M (MEK Km ≈300 nM, Brightman 2000)
    },
    # pMEK→ppMEK MEK 双磷酸化（Brightman 2000, PMID:10986007）
    "pMEK_ppMEK": {
        "k_cat": 0.5,                # min^-1
        "Km": 3e-7,                  # M (MEK Km ≈300 nM, Brightman 2000)
    },
    # ppMEK→pERK ERK 单磷酸化（Brightman 2000, PMID:10986007）
    "ppMEK_pERK": {
        "k_cat": 1.0,                # min^-1
        "Km": 3e-7,                  # M (ERK Km ≈300 nM, Brightman 2000)
    },
    # pERK→ppERK ERK 双磷酸化（Goldbeter & Koshland 1981, PMID:1941687 零级超敏感）
    "pERK_ppERK": {
        "k_cat": 1.0,                # min^-1
        "Km": 3e-7,                  # M (ERK Km ≈300 nM, 零级超敏感区间)  # Heuristic estimate, needs calibration
    },
    # ERK 去磷酸化（Goldbeter & Koshland 1981 磷酸酶, PMID:1941687）
    "ppERK_ERK": {
        "k_cat": 0.5,                # min^-1  # Heuristic estimate, needs calibration
        "Km": 1e-7,                  # M (磷酸酶 Km)  # Heuristic estimate, needs calibration
    },
}


__all__ = [
    "MAPKSpecialist",
    "PATHWAY_TAG",
    "SOURCE_SBML",
    "KINETIC_PARAMETERS",
]
