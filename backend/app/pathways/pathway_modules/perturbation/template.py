# BioDynamics Agent v4 - Pathway Specialist Perturbation 模块数据结构模板 (Task 4.2.3)
# 定义 PerturbationModuleData dataclass，作为 PathwaySpecialistBase.apply_perturbation()
# 的返回数据结构骨架。具体 Specialist 子类（Task 4.3-4.12）可实例化并填充。

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PerturbationModuleData:
    """扰动模块数据结构：药物 / KO / 突变 Reaction 片段。

    由 ``PathwaySpecialistBase.apply_perturbation()`` 返回，包含通路特异的
    药物抑制、基因敲除、突变等扰动 Reaction 片段。
    """

    # 扰动 Reaction IR 片段列表
    # 每条含 reactants / products / modifiers / kinetic_law / perturbation_type
    perturbation_reactions: list[dict] = field(default_factory=list)

    # 药物靶点列表（drug_targets）
    # 每条含 target / drug_name / IC50 / Ki / concentration / time
    drug_targets: list[dict] = field(default_factory=list)

    # 基因敲除靶点列表（knockout_targets）
    # 每条含 target_gene / ko_type（"complete" | "partial"）/ residual_activity
    ko_targets: list[dict] = field(default_factory=list)


__all__ = ["PerturbationModuleData"]
