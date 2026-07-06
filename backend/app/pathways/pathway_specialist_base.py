# BioDynamics Agent v4 - Pathway Specialist Base (Phase 4 / Task 4.2)
# Specialist 抽象基类：定义 5 模块（core/feedback/crosstalk/perturbation/validation）
# 的统一接口，由 10 个具体 Specialist 子类实现（Task 4.3-4.12）。
#
# 设计原则（铁律）：
# 1. Feature Flag V4_PATHWAY_SPECIALIST_ENABLED=false 时 Specialist 不执行
#    （由 LangGraph hook 层负责短路，基类本身不做 flag 检查）
# 2. 6 个 apply_* / load_module 接口为 abstract method，子类必须实现，否则 TypeError
# 3. 3 个辅助方法（select_template / get_metadata / validate_input）为具体实现，
#    子类可直接继承或覆写
# 4. apply_* 方法的子类实现应捕获异常并返回空 list/dict，记录 logger.warning，
#    不阻塞流水线（错误处理契约，见各方法 docstring）
# 5. 不修改 v3 任何字段；不生成 ODE；不调用 RAG；不做 SBML 验证（职责边界严格）
#
# 依赖（P3）：
# - app.ode_templates_v2（4 个 .j2 模板：_mechanism_phosphorylation_mm /
#   bistable_switch / oscillatory_feedback / _dde_helpers）
# - app.pathways.pathway_modules（5 子目录 dataclass 数据结构模板）

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# 5 模块名称常量
# =============================================================================
MODULE_CORE: str = "core"
MODULE_FEEDBACK: str = "feedback"
MODULE_CROSSTALK: str = "crosstalk"
MODULE_PERTURBATION: str = "perturbation"
MODULE_VALIDATION: str = "validation"

# 默认全部 5 模块支持的列表（供 supported_modules 类属性默认值引用）
ALL_MODULES: list[str] = [
    MODULE_CORE,
    MODULE_FEEDBACK,
    MODULE_CROSSTALK,
    MODULE_PERTURBATION,
    MODULE_VALIDATION,
]


# =============================================================================
# mechanism → ODE 模板名映射（select_template 默认实现使用）
# =============================================================================
# 与 P3 ode_templates_v2/ 下 4 个 .j2 文件对齐：
# - _mechanism_phosphorylation_mm.j2：磷酸化 Michaelis-Menten 动力学
# - bistable_switch.j2：双稳态开关（Apoptosis / Cell Cycle）
# - oscillatory_feedback.j2：转录延迟反馈振荡（p53 / NF-κB / TGF-β / JAK-STAT）
# - _dde_helpers.j2：DDE 求解器辅助（被其他模板 include，不直接选择）
_MECHANISM_TEMPLATE_MAP: dict[str, str] = {
    "phosphorylation": "_mechanism_phosphorylation_mm",
    "bistable": "bistable_switch",
    "oscillatory": "oscillatory_feedback",
}


class PathwaySpecialistBase(ABC):
    """Pathway Specialist 抽象基类。

    每个 Specialist 子类代表一条信号通路（如 EGFR / MAPK / p53 等），通过
    5 个模块（core / feedback / crosstalk / perturbation / validation）输出
    通路特异的 Reaction IR 片段与 Validation 规则。

    子类契约：
    1. 必须覆写类属性：``pathway_class`` / ``display_name``
    2. ``supported_modules`` 默认 5 模块全支持，子类可裁剪（如某通路无反馈）
    3. 必须实现 6 个 abstract method：
       - ``load_module(module_name)``：加载模块数据结构
       - ``apply_core(pathway_graph, ontology_entities)``：核心 Reaction IR 片段
       - ``apply_feedback(pathway_graph)``：FeedbackLoop 列表
       - ``apply_crosstalk(pathway_graph, crosstalk_edges)``：cross-talk Reaction 片段
       - ``apply_perturbation(pathway_graph, perturbation_points)``：扰动 Reaction 片段
       - ``apply_validation(simulation_result)``：Validation 规则列表
    4. apply_* 方法的实现应捕获异常并返回空 list/dict，记录 logger.warning，
       不阻塞流水线（错误处理契约）

    不可碰清单：
    - 不修改 v3 任何字段（network_json / entities / mechanism 等）
    - 不生成 ODE / 不调用 RAG / 不做 SBML 验证（职责边界严格）
    """

    # 子类必须覆写：通路类别键（如 "EGFR_RTK"），与 P1 pathway_registry 对齐
    pathway_class: str = ""

    # 子类必须覆写：人类可读通路名（如 "EGFR RTK Signaling"）
    display_name: str = ""

    # 默认 5 模块全支持；子类可裁剪（如某通路无 feedback 模块）
    # 使用 list 字面量 + 类属性复制，避免可变默认值共享问题
    supported_modules: list[str] = list(ALL_MODULES)

    # =================================================================
    # 六接口（abstract method，子类必须实现）
    # =================================================================
    @abstractmethod
    def load_module(self, module_name: str) -> Any:
        """加载指定模块的数据结构。

        Args:
            module_name: 模块名，取值为 ``core`` / ``feedback`` / ``crosstalk``
                / ``perturbation`` / ``validation`` 之一。

        Returns:
            对应模块的数据结构实例（如 ``CoreModuleData`` / ``FeedbackModuleData``
            等 dataclass），具体类型由子类决定。子类可返回 None 表示该模块
            未实现（应与 ``supported_modules`` 一致）。
        """

    @abstractmethod
    def apply_core(
        self,
        pathway_graph: dict,
        ontology_entities: dict | None = None,
    ) -> dict:
        """应用核心模块，返回 Reaction IR 片段。

        Args:
            pathway_graph: 通路图（v4_pathway_graph 或其子图）。
            ontology_entities: P1 Ontology Entities（可选，用于 HGNC/UniProt
                ID 对齐与 SBO term 查找）。

        Returns:
            Reaction IR 片段 dict，至少包含 ``species`` 与 ``reactions`` 字段。
            子类实现应在异常时返回 ``{"species": [], "reactions": []}``，
            记录 logger.warning，不抛异常。
        """

    @abstractmethod
    def apply_feedback(self, pathway_graph: dict) -> list[dict]:
        """应用反馈模块，返回 FeedbackLoop 列表。

        Args:
            pathway_graph: 通路图。

        Returns:
            FeedbackLoop 字典列表，每条含 ``loop_id`` / ``source`` /
            ``target`` / ``sign``（"positive" | "negative"）/ ``delay``
            （分钟，0 表示无延迟）等字段。子类实现应在异常时返回空列表，
            记录 logger.warning，不抛异常。
        """

    @abstractmethod
    def apply_crosstalk(
        self,
        pathway_graph: dict,
        crosstalk_edges: list[dict],
    ) -> list[dict]:
        """应用跨通路模块，返回 cross-talk Reaction 片段。

        注意：本方法仅生成本通路侧的 cross-talk Reaction IR 片段，
        cross-talk edge 本身由 Cross-talk Coordinator 创建（Task 4.13）。

        Args:
            pathway_graph: 通路图。
            crosstalk_edges: cross-talk edge 列表（来自 Coordinator），
                每条含 ``source_pathway`` / ``target_pathway`` /
                ``source_node`` / ``target_node`` / ``mechanism`` 等字段。

        Returns:
            cross-talk Reaction IR 片段列表。子类实现应在异常时返回空列表，
            记录 logger.warning，不抛异常。
        """

    @abstractmethod
    def apply_perturbation(
        self,
        pathway_graph: dict,
        perturbation_points: list[dict],
    ) -> list[dict]:
        """应用扰动模块，返回药物 / KO / 突变 Reaction 片段。

        Args:
            pathway_graph: 通路图。
            perturbation_points: 扰动点列表，每条含 ``target`` / ``type``
                （"drug" | "knockout" | "mutation"）/ ``concentration`` /
                ``time`` 等字段。

        Returns:
            扰动 Reaction IR 片段列表。子类实现应在异常时返回空列表，
            记录 logger.warning，不抛异常。
        """

    @abstractmethod
    def apply_validation(
        self,
        simulation_result: dict | None = None,
    ) -> list[dict]:
        """应用验证模块，返回 Validation 规则列表。

        Args:
            simulation_result: 仿真结果（可选，用于动态校验时序与峰值）。
                为 None 时返回静态规则集（基于文献 benchmark）。

        Returns:
            Validation 规则列表，每条 dict 至少含：
            - ``metric_name``: 指标名（如 "pEGFR_peak_time_min"）
            - ``expected``: 期望值（如 7.5，单位 min）
            - ``tolerance``: 容忍度（如 2.5，表示 5-10 min 区间）
            - ``pmid``: 文献 PMID（如 "Lev Bar-Or 2000"）
            - 其他可选字段：``comparison`` / ``unit`` / ``pathway_tag``
            子类实现应在异常时返回空列表，记录 logger.warning，不抛异常。
        """

    # =================================================================
    # 辅助方法（具体实现，子类可继承或覆写）
    # =================================================================
    def select_template(self, mechanism: str) -> str:
        """根据 mechanism 选择 ODE 模板名。

        默认映射（与 P3 ``ode_templates_v2/`` 下 .j2 文件对齐）：
        - ``phosphorylation`` → ``_mechanism_phosphorylation_mm``
        - ``bistable`` → ``bistable_switch``
        - ``oscillatory`` → ``oscillatory_feedback``

        子类可覆写以支持更多 mechanism（如 ``transcription`` →
        ``transcription_factor``、``destruction_complex`` 等，待对应模板
        在 P3 后续 Task 中补充）。

        Args:
            mechanism: 机制名（小写，如 ``"phosphorylation"``）。

        Returns:
            ODE 模板名（不含 ``.j2`` 后缀）。未匹配时返回 ``"default"``
            （调用方应处理默认降级）。
        """
        return _MECHANISM_TEMPLATE_MAP.get(mechanism, "default")

    def get_metadata(self) -> dict:
        """返回 Specialist 元数据。

        Returns:
            dict 含：
            - ``pathway_class``: 通路类别键
            - ``display_name``: 人类可读通路名
            - ``supported_modules``: 支持的模块列表（副本，避免外部修改）
            - ``version``: Specialist 实现版本（如 ``"v4.2"``）
        """
        return {
            "pathway_class": self.pathway_class,
            "display_name": self.display_name,
            "supported_modules": list(self.supported_modules),
            "version": "v4.2",
        }

    def validate_input(self, pathway_graph: dict) -> list[str]:
        """输入校验，返回 warning 列表（不抛异常）。

        检查 ``pathway_graph`` 是否含必要字段（``nodes`` / ``edges``），
        缺失时追加 warning 字符串。本方法仅做轻量结构校验，不做语义校验。

        Args:
            pathway_graph: 通路图 dict。

        Returns:
            warning 字符串列表。空列表表示无 warning（输入结构 OK）。
        """
        warnings: list[str] = []
        if not pathway_graph:
            warnings.append("pathway_graph 为空或 None")
            return warnings
        if not isinstance(pathway_graph, dict):
            warnings.append(
                f"pathway_graph 类型错误：期望 dict，实际 "
                f"{type(pathway_graph).__name__}"
            )
            return warnings
        if "nodes" not in pathway_graph:
            warnings.append("pathway_graph 缺少 'nodes' 字段")
        if "edges" not in pathway_graph:
            warnings.append("pathway_graph 缺少 'edges' 字段")
        return warnings


__all__ = [
    "PathwaySpecialistBase",
    "ALL_MODULES",
    "MODULE_CORE",
    "MODULE_FEEDBACK",
    "MODULE_CROSSTALK",
    "MODULE_PERTURBATION",
    "MODULE_VALIDATION",
]
