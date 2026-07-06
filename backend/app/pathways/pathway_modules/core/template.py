# BioDynamics Agent v4 - Pathway Specialist Core 模块数据结构模板 (Task 4.2.3)
# 定义 CoreModuleData dataclass，作为 PathwaySpecialistBase.apply_core() 的
# 返回数据结构骨架。具体 Specialist 子类（Task 4.3-4.12）可实例化并填充。

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CoreModuleData:
    """核心模块数据结构：通路核心拓扑的 Reaction IR 片段。

    由 ``PathwaySpecialistBase.apply_core()`` 返回，包含通路核心的物种与反应。
    每个 Specialist 子类应填充对应通路的实际 species / reactions。
    """

    # 通路核心的物种列表（SpeciesV2.model_dump() dict）
    # 每条含 name / compartment / species_type / initial_concentration 等
    species: list[dict] = field(default_factory=list)

    # 通路核心的反应列表（ReactionV2.model_dump() dict）
    # 每条含 reactants / products / modifiers / kinetic_law / mechanism 等
    reactions: list[dict] = field(default_factory=list)

    # 动力学参数覆盖（key: 参数名，value: 数值或 dict）
    # 用于 Specialist 强制通路特异参数（如 EGFR 内吞 k_int = 0.01 /s）
    kinetics_overrides: dict[str, object] = field(default_factory=dict)


__all__ = ["CoreModuleData"]
