# BioDynamics Agent v4 - EGFR Specialist (Phase 4 / Task 4.3)
# EGFR RTK 通路 Specialist：实现 Core/Feedback/Crosstalk/Perturbation/Validation 5 模块。
#
# 设计原则（铁律）：
# 1. Feature Flag V4_PATHWAY_SPECIALIST_ENABLED=false 时由 LangGraph hook 层短路，
#    本 Specialist 代码不被调用（基类本身不做 flag 检查）
# 2. 使用 Task 4.0 修复后的 PHOSPHORYLATION 语义：
#    - 自磷酸化（EGFR → pEGFR，target 是 p+source 形式）：source 作 substrate+product，无 modifier
#    - 异磷酸化（pEGFR → pShc）：target 的未磷酸化形式(Shc) 作 substrate，
#      target 磷酸化形式(pShc) 作 product，source(pEGFR) 作 catalytic modifier
# 3. 不处理 MAPK 级联（由 MAPK Specialist 处理，Task 4.4）
# 4. 不生成跨通路 cross-talk edge 本身（由 Cross-talk Coordinator 处理，Task 4.13），
#    仅返回本通路侧的 cross-talk Reaction 片段
# 5. apply_* 方法实现捕获异常并返回空 list/dict，记录 logger.warning，不抛异常
#
# 依赖：
# - P1 ontology / pathway_registry（EGFR_RTK 通路类别键）
# - P2 schema（SpeciesV2 / ReactionV2 / SpeciesRef / Modifier）
# - P3 ode_templates_v2（_mechanism_phosphorylation_mm.j2 模板）
# - P3 pathway_graph/initializer.py（EGFR_RTK core_nodes / core_edges）
#
# 参考：
# - spec.md Part 3 Specialist 1（第 192-197 行）
# - tasks.md Task 4.3（第 44-50 行）

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
# EGFR 通路标签
# =============================================================================
PATHWAY_TAG: str = "EGFR_RTK"

# SBML BioModels ID（Levchenko 2000 EGFR model）
SOURCE_SBML: str = "BIOMD0000000022"
# Validation benchmark PMID 引用
_PMID_LEVCHENKO_2000: str = "PMID:11923475"   # Levchenko 2000 EGFR model
_PMID_SCHOEBERL_2001: str = "PMID:11483517"  # Schoeberl 2001 MAPK


# =============================================================================
# EGFR 核心物种（与 P3 pathway_graph/initializer.py EGFR_RTK.core_nodes 对齐）
# =============================================================================
_EGFR_CORE_SPECIES: list[dict[str, Any]] = [
    {"name": "EGF", "species_type": "ligand", "compartment": "extracellular"},
    {"name": "EGFR", "species_type": "protein", "compartment": "membrane"},
    {"name": "pEGFR", "species_type": "protein", "compartment": "membrane"},
    {"name": "Shc", "species_type": "protein", "compartment": "cytoplasm"},
    {"name": "pShc", "species_type": "protein", "compartment": "cytoplasm"},
    {"name": "Grb2", "species_type": "protein", "compartment": "cytoplasm"},
    {"name": "SOS", "species_type": "protein", "compartment": "cytoplasm"},
    # Ras 与 MAPK Specialist 共享（shared_species 标记）
    {"name": "Ras", "species_type": "protein", "compartment": "membrane",
     "shared": True},
    {"name": "RasGTP", "species_type": "protein", "compartment": "membrane"},
    {"name": "Raf", "species_type": "protein", "compartment": "cytoplasm"},
    {"name": "pRaf", "species_type": "protein", "compartment": "cytoplasm"},
]


# =============================================================================
# EGFR 核心反应（7 条，与 P3 initializer EGFR_RTK.core_edges 对齐）
# =============================================================================
# 每条反应含：source / target / mechanism / kinetics_type / pathway_tag /
#             substrate / product / modifier / modifier_type / autophosphorylation
#
# PHOSPHORYLATION 语义（Task 4.0 修复后）：
# - 自磷酸化（EGFR → pEGFR）：source(EGFR) 作 substrate，target(pEGFR) 作 product，
#   无 modifier（autophosphorylation=True）
# - 异磷酸化（pEGFR → pShc）：未磷酸化形式(Shc) 作 substrate，
#   磷酸化形式(pShc) 作 product，source(pEGFR) 作 catalytic modifier
_EGFR_CORE_REACTIONS: list[dict[str, Any]] = [
    # 1. EGF + EGFR → EGF-EGFR complex（配体-受体结合）
    {
        "source": "EGF",
        "target": "EGFR",
        "mechanism": "binding",
        "kinetics_type": "mass_action",
        "pathway_tag": PATHWAY_TAG,
        "substrate": "EGF",
        "product": "EGFR",  # 结合反应：source→target，无 modifier
        "modifier": None,
        "autophosphorylation": False,
        "description": "EGF 配体结合 EGFR 受体（mass_action）",
    },
    # 2. EGFR → pEGFR（自磷酸化，受体二聚化后 Tyr 自磷酸化）
    {
        "source": "EGFR",
        "target": "pEGFR",
        "mechanism": "phosphorylation",
        "kinetics_type": "Michaelis_Menten",
        "pathway_tag": PATHWAY_TAG,
        # 自磷酸化：source 作 substrate+product，无 modifier
        "substrate": "EGFR",
        "product": "pEGFR",
        "modifier": None,
        "autophosphorylation": True,
        "description": "EGFR 二聚化后自磷酸化（Tyr 残基，MM 动力学）",
    },
    # 3. pEGFR → pShc（异磷酸化，Shc 作 substrate，pEGFR 作 catalytic modifier）
    {
        "source": "pEGFR",
        "target": "pShc",
        "mechanism": "phosphorylation",
        "kinetics_type": "Michaelis_Menten",
        "pathway_tag": PATHWAY_TAG,
        # 异磷酸化：Shc 作 substrate，pShc 作 product，pEGFR 作 catalytic modifier
        "substrate": "Shc",
        "product": "pShc",
        "modifier": "pEGFR",
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "pEGFR 磷酸化 Shc（Shc 作 substrate，pEGFR 作 catalytic modifier）",
    },
    # 4. pEGFR → Grb2（ adaptor 蛋白结合）
    {
        "source": "pEGFR",
        "target": "Grb2",
        "mechanism": "binding",
        "kinetics_type": "mass_action",
        "pathway_tag": PATHWAY_TAG,
        "substrate": "pEGFR",
        "product": "Grb2",
        "modifier": None,
        "autophosphorylation": False,
        "description": "pEGFR 结合 Grb2 adaptor 蛋白",
    },
    # 5. Grb2 → SOS（Grb2-SOS 复合物形成）
    {
        "source": "Grb2",
        "target": "SOS",
        "mechanism": "binding",
        "kinetics_type": "mass_action",
        "pathway_tag": PATHWAY_TAG,
        "substrate": "Grb2",
        "product": "SOS",
        "modifier": None,
        "autophosphorylation": False,
        "description": "Grb2 结合 SOS（GEF 蛋白）",
    },
    # 6. SOS → RasGTP（GTP/GDP 交换，Ras 作 substrate，SOS 作 GEF modifier）
    {
        "source": "SOS",
        "target": "RasGTP",
        "mechanism": "gtp_gdp_exchange",
        "kinetics_type": "mass_action",
        "pathway_tag": PATHWAY_TAG,
        "substrate": "Ras",  # Ras GDP → Ras GTP，Ras 作 substrate
        "product": "RasGTP",
        "modifier": "SOS",  # SOS 作 GEF (catalytic) modifier
        "modifier_type": "catalytic",
        "autophosphorylation": False,
        "description": "SOS 催化 Ras GDP→GTP 交换（GEF 活性）",
    },
    # 7. RasGTP → pRaf（异磷酸化，Raf 作 substrate，RasGTP 作 catalytic modifier）
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
]


# =============================================================================
# EGFR 反馈环（2 条）
# =============================================================================
_EGFR_FEEDBACK_LOOPS: list[dict[str, Any]] = [
    # 1. ERK → SOS 负反馈（pERK 磷酸化 SOS 导致其失活，无转录延迟）
    {
        "id": "FL_EGFR_ERK_SOS_NEG",
        "loop_type": "negative",
        "node_ids": ["pERK", "SOS", "RasGTP"],
        "delay_minutes": 0.0,
        "description": "pERK 磷酸化 SOS 导致其失活（负反馈，无转录延迟，delay=0）",
    },
    # 2. EGFR 内吞降解（pEGFR 触发受体内吞与降解，delay=15 min）
    {
        "id": "FL_EGFR_INTERNALIZATION",
        "loop_type": "negative",
        "node_ids": ["pEGFR", "EGFR"],
        "delay_minutes": 15.0,
        "description": "pEGFR 触发受体内吞与降解（负反馈，delay=15 min）",
    },
]


# =============================================================================
# EGFR Crosstalk Reaction 片段（3 条，仅本通路侧片段，edge 由 Coordinator 管理）
# =============================================================================
_EGFR_CROSSTALK_REACTIONS: list[dict[str, Any]] = [
    # 1. pEGFR → PI3K（activation，pEGFR 直接磷酸化 PI3K）
    {
        "source": "pEGFR",
        "target": "PI3K",
        "mechanism": "activation",
        "shared_species": [],
        "description": "pEGFR 直接磷酸化 PI3K（激活 PI3K-AKT-mTOR 通路）",
    },
    # 2. pERK → ELK1 → Fos（transcription，pERK 激活转录因子）
    {
        "source": "pERK",
        "target": "Fos",
        "mechanism": "transcription",
        "shared_species": [],
        "intermediate": "ELK1",
        "description": "pERK 激活转录因子 ELK1 → Fos 转录（早期基因表达）",
    },
    # 3. AKT → Raf Ser259（inhibition，AKT 磷酸化 Raf Ser259 抑制 MAPK）
    {
        "source": "AKT",
        "target": "Raf",
        "mechanism": "inhibition",
        "shared_species": [],
        "site": "Ser259",
        "description": "AKT 磷酸化 Raf Ser259 抑制 MAPK 级联（PI3K→MAPK cross-talk）",
    },
]


# =============================================================================
# EGFR 扰动（4 个：3 药物 + 1 突变）
# =============================================================================
_EGFR_PERTURBATIONS: list[dict[str, Any]] = [
    # 1. Gefitinib（EGFR inhibitor, small molecule）
    {
        "target": "EGFR",
        "drug": "Gefitinib",
        "mechanism": "inhibition",
        "ko_target": None,
        "description": "Gefitinib（EGFR 酪氨酸激酶抑制剂，小分子）",
    },
    # 2. Erlotinib（EGFR inhibitor, small molecule）
    {
        "target": "EGFR",
        "drug": "Erlotinib",
        "mechanism": "inhibition",
        "ko_target": None,
        "description": "Erlotinib（EGFR 酪氨酸激酶抑制剂，小分子）",
    },
    # 3. Cetuximab（EGFR antibody, monoclonal）
    {
        "target": "EGFR",
        "drug": "Cetuximab",
        "mechanism": "inhibition",
        "ko_target": None,
        "description": "Cetuximab（EGFR 单克隆抗体，阻断配体结合）",
    },
    # 4. EGFR vIII（constitutively active mutant, deletion of exon 2-7）
    {
        "target": "EGFR",
        "drug": None,
        "mechanism": "activation",
        "ko_target": "EGFR_vIII",
        "description": "EGFR vIII（组成性激活突变，外显子 2-7 缺失）",
    },
]


# =============================================================================
# EGFR Validation 规则（3 条 benchmark）
# =============================================================================
# 每条含：rule_id / metric_name / expected / tolerance / pmid / description
# expected 取区间中点，tolerance 取区间半宽
_EGFR_VALIDATION_RULES: list[dict[str, Any]] = [
    # 1. pEGFR 达峰时间 5-10 min（Levchenko 2000）
    {
        "rule_id": "VAL_EGFR_PEGFR_PEAK_TIME",
        "metric_name": "pEGFR_peak_time",
        "expected": 7.5,   # (5.0 + 10.0) / 2
        "tolerance": 2.5,   # (10.0 - 5.0) / 2
        "expected_min": 5.0,
        "expected_max": 10.0,
        "unit": "minutes",
        "pmid": _PMID_LEVCHENKO_2000,
        "comparison": "range",
        "pathway_tag": PATHWAY_TAG,
        "description": "pEGFR 在 EGF 刺激后 5-10 min 达峰（Levchenko 2000 EGFR model）",
    },
    # 2. MAPK 放大倍数 >10x（Schoeberl 2001）
    {
        "rule_id": "VAL_EGFR_MAPK_AMPLIFICATION",
        "metric_name": "MAPK_amplification",
        "expected": 55.0,  # (10.0 + 100.0) / 2
        "tolerance": 45.0,  # (100.0 - 10.0) / 2
        "expected_min": 10.0,
        "expected_max": 100.0,
        "unit": "fold",
        "pmid": _PMID_SCHOEBERL_2001,
        "comparison": "range",
        "pathway_tag": PATHWAY_TAG,
        "description": "MAPK 级联放大倍数 10-100x（Schoeberl 2001 MAPK model）",
    },
    # 3. EGFR 内吞半衰期 10-15 min（Levchenko 2000）
    {
        "rule_id": "VAL_EGFR_INTERNALIZATION_HALF_LIFE",
        "metric_name": "EGFR_internalization_half_life",
        "expected": 12.5,  # (10.0 + 15.0) / 2
        "tolerance": 2.5,  # (15.0 - 10.0) / 2
        "expected_min": 10.0,
        "expected_max": 15.0,
        "unit": "minutes",
        "pmid": _PMID_LEVCHENKO_2000,
        "comparison": "range",
        "pathway_tag": PATHWAY_TAG,
        "description": "EGFR 受体内吞半衰期 10-15 min（Levchenko 2000 EGFR model）",
    },
]


@register_specialist
class EGFRSpecialist(PathwaySpecialistBase):
    """EGFR RTK 通路 Specialist。

    实现 EGFR RTK 通路 Core/Feedback/Crosstalk/Perturbation/Validation 5 模块，
    输出通路特异 Reaction IR 片段 + 模板选择 + Validation 规则。

    职责边界：
    - 处理 EGFR RTK 上游（EGF/EGFR/pEGFR/Shc/Grb2/SOS/Ras/Raf）
    - 不处理 MAPK 级联（由 MAPK Specialist 处理，Task 4.4）
    - 不生成跨通路 cross-talk edge（由 Cross-talk Coordinator 处理，Task 4.13）

    输入：
    - ``pathway_graph``：v4_pathway_graph 的 EGFR 子图
    - ``ontology_entities``：P1 Ontology Entities（可选，HGNC/UniProt ID 对齐）

    输出：
    - ``apply_core``：7 条核心 Reaction IR 片段 + 11 物种
    - ``apply_feedback``：2 条 FeedbackLoop（ERK→SOS / EGFR 内吞）
    - ``apply_crosstalk``：3 条 cross-talk Reaction 片段（PI3K / Fos / Raf Ser259）
    - ``apply_perturbation``：4 个扰动（Gefitinib/Erlotinib/Cetuximab/EGFR vIII）
    - ``apply_validation``：3 条 Validation benchmark 规则
    """

    pathway_class: str = "EGFR_RTK"
    display_name: str = "EGFR RTK Signaling"

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
                    species=list(_EGFR_CORE_SPECIES),
                    reactions=list(_EGFR_CORE_REACTIONS),
                    kinetics_overrides={},
                )
            if module_name == MODULE_FEEDBACK:
                return FeedbackModuleData(
                    feedback_loops=list(_EGFR_FEEDBACK_LOOPS),
                    delay_minutes=15.0,
                    loop_type="negative",
                )
            if module_name == MODULE_CROSSTALK:
                return CrosstalkModuleData(
                    crosstalk_reactions=list(_EGFR_CROSSTALK_REACTIONS),
                    shared_species=["Ras"],
                    coordination_strategy="merge",
                )
            if module_name == MODULE_PERTURBATION:
                return PerturbationModuleData(
                    perturbation_reactions=list(_EGFR_PERTURBATIONS),
                    drug_targets=[
                        p for p in _EGFR_PERTURBATIONS if p.get("drug")
                    ],
                    ko_targets=[
                        p for p in _EGFR_PERTURBATIONS if p.get("ko_target")
                    ],
                )
            if module_name == MODULE_VALIDATION:
                return ValidationModuleData(
                    rules=list(_EGFR_VALIDATION_RULES),
                    benchmarks=[
                        {
                            "benchmark_name": r["metric_name"],
                            "value": r["expected"],
                            "unit": r["unit"],
                            "condition": "EGF stimulation",
                            "reference": r["pmid"],
                        }
                        for r in _EGFR_VALIDATION_RULES
                    ],
                    tolerances={
                        r["metric_name"]: r["tolerance"]
                        for r in _EGFR_VALIDATION_RULES
                    },
                    pmid_references=[
                        {
                            "pmid": r["pmid"],
                            "citation": r["description"],
                            "pathway_class": PATHWAY_TAG,
                            "metric_name": r["metric_name"],
                        }
                        for r in _EGFR_VALIDATION_RULES
                    ],
                )
            logger.warning(
                "EGFRSpecialist.load_module: 未知模块名 '%s'", module_name
            )
            return None
        except Exception as exc:
            logger.warning(
                "EGFRSpecialist.load_module 加载模块 '%s' 失败: %s",
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
        """应用核心模块，返回 EGFR RTK 通路核心 Reaction IR 片段。

        输出 7 条核心反应：
        1. EGF + EGFR → binding（mass_action）
        2. EGFR → pEGFR（自磷酸化，MM）
        3. pEGFR → pShc（异磷酸化，MM，pEGFR 作 catalytic modifier）
        4. pEGFR → Grb2（binding）
        5. Grb2 → SOS（binding）
        6. SOS → RasGTP（gtp_gdp_exchange，SOS 作 GEF modifier）
        7. RasGTP → pRaf（异磷酸化，MM，RasGTP 作 catalytic modifier）

        Returns:
            dict 含 ``species``（11 物种）与 ``reactions``（7 反应）字段。
            异常时返回 ``{"species": [], "reactions": []}``。
        """
        try:
            return {
                "species": list(_EGFR_CORE_SPECIES),
                "reactions": list(_EGFR_CORE_REACTIONS),
                "pathway_tag": PATHWAY_TAG,
                "source_sbml": SOURCE_SBML,
            }
        except Exception as exc:
            logger.warning(
                "EGFRSpecialist.apply_core 失败: %s", exc
            )
            return {"species": [], "reactions": []}

    # =================================================================
    # apply_feedback：FeedbackLoop 列表
    # =================================================================
    def apply_feedback(self, pathway_graph: dict) -> list[dict]:
        """应用反馈模块，返回 EGFR 通路 FeedbackLoop 列表。

        输出 2 条反馈环：
        1. ERK → SOS 负反馈（pERK 磷酸化 SOS 失活，delay=0 min）
        2. EGFR 内吞降解（pEGFR 触发受体内吞，delay=15 min）

        Returns:
            FeedbackLoop 字典列表。异返回空列表。
        """
        try:
            return list(_EGFR_FEEDBACK_LOOPS)
        except Exception as exc:
            logger.warning(
                "EGFRSpecialist.apply_feedback 失败: %s", exc
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
        """应用跨通路模块，返回 EGFR 通路侧 cross-talk Reaction 片段。

        注意：本方法仅生成本通路侧的 cross-talk Reaction 片段，
        cross-talk edge 本身由 Cross-talk Coordinator 创建（Task 4.13）。

        输出 3 条 cross-talk Reaction 片段：
        1. pEGFR → PI3K（activation）
        2. pERK → ELK1 → Fos（transcription）
        3. AKT → Raf Ser259（inhibition）

        Args:
            pathway_graph: 通路图（当前未使用，保留接口一致性）。
            crosstalk_edges: 来自 Coordinator 的 cross-talk edge 列表
                （当前未使用，本 Specialist 返回静态拓扑片段）。

        Returns:
            cross-talk Reaction 片段列表。异返回空列表。
        """
        try:
            return list(_EGFR_CROSSTALK_REACTIONS)
        except Exception as exc:
            logger.warning(
                "EGFRSpecialist.apply_crosstalk 失败: %s", exc
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
        """应用扰动模块，返回 EGFR 通路特异药物 / 突变 Reaction 片段。

        输出 4 个扰动：
        1. Gefitinib（EGFR 抑制剂，小分子）
        2. Erlotinib（EGFR 抑制剂，小分子）
        3. Cetuximab（EGFR 单克隆抗体）
        4. EGFR vIII（组成性激活突变，外显子 2-7 缺失）

        Args:
            pathway_graph: 通路图（当前未使用）。
            perturbation_points: 扰动点列表（当前未使用，本 Specialist
                返回静态拓扑片段）。

        Returns:
            扰动 Reaction 片段列表。异返回空列表。
        """
        try:
            return list(_EGFR_PERTURBATIONS)
        except Exception as exc:
            logger.warning(
                "EGFRSpecialist.apply_perturbation 失败: %s", exc
            )
            return []

    # =================================================================
    # apply_validation：Validation 规则列表
    # =================================================================
    def apply_validation(
        self,
        simulation_result: dict | None = None,
    ) -> list[dict]:
        """应用验证模块，返回 EGFR 通路 Validation 规则列表。

        输出 3 条 benchmark：
        1. pEGFR 达峰时间 5-10 min（Levchenko 2000, PMID:11923475）
        2. MAPK 放大倍数 10-100x（Schoeberl 2001, PMID:11483517）
        3. EGFR 内吞半衰期 10-15 min（Levchenko 2000, PMID:11923475）

        Args:
            simulation_result: 仿真结果（可选，当前未使用，返回静态规则集）。

        Returns:
            Validation 规则列表。异返回空列表。
        """
        try:
            return list(_EGFR_VALIDATION_RULES)
        except Exception as exc:
            logger.warning(
                "EGFRSpecialist.apply_validation 失败: %s", exc
            )
            return []

    # =================================================================
    # select_template：覆写以支持未来 Signaling_Cascade_Phos 模板
    # =================================================================
    # 当前仅返回基类默认映射（_mechanism_phosphorylation_mm）。
    # 未来 P3 若添加 Signaling_Cascade_Phos 模板，可在此覆写：
    #   if mechanism == "phosphorylation":
    #       return "Signaling_Cascade_Phos"
    #   return super().select_template(mechanism)


__all__ = [
    "EGFRSpecialist",
    "PATHWAY_TAG",
    "SOURCE_SBML",
]
