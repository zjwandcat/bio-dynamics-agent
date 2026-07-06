# BioDynamics Agent v4 - Pathway Specialist Crosstalk 模块数据结构模板 (Task 4.2.3)
# 定义 CrosstalkModuleData dataclass，作为 PathwaySpecialistBase.apply_crosstalk() 的
# 返回数据结构骨架。具体 Specialist 子类（Task 4.3-4.12）可实例化并填充。

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CrosstalkModuleData:
    """跨通路模块数据结构：本通路侧的 cross-talk Reaction 片段。

    由 ``PathwaySpecialistBase.apply_crosstalk()`` 返回，包含本通路消费的
    cross-talk Reaction 片段与共享物种信息。

    注意：cross-talk edge 本身由 Cross-talk Coordinator（Task 4.13）创建，
    本数据结构仅承载 Specialist 生成的本通路侧 Reaction 片段。
    """

    # cross-talk Reaction IR 片段列表
    # 每条含 reactants / products / modifiers / kinetic_law / pathway_tag
    # pathway_tag 必须形如 "CROSSTALK_A_B" 以强制隔离（防参数污染）
    crosstalk_reactions: list[dict] = field(default_factory=list)

    # 共享物种列表（如 Ras / AKT / MEK 跨通路标记）
    # Coordinator 后续会强制这些物种在不同通路间使用同一 ODE 变量
    shared_species: list[str] = field(default_factory=list)

    # 共享物种同步策略："merge"（合并为同一变量）/ "alias"（别名引用）
    # / "separate"（保持独立，Coordinator 决定合并策略）
    coordination_strategy: str = "merge"


__all__ = ["CrosstalkModuleData"]
