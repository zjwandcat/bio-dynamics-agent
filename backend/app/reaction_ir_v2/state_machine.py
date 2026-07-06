# BioDynamics Agent v4 - 蛋白质状态机辅助逻辑
# 对应 v4 Scientific Architecture Part 4 §4.2.4 + §4.7（EGFR 状态机示例）。
#
# 设计原则：
# 1. 状态机是 v4 Reaction IR 的一等公民，替代 v3 的空 state_transitions 扩展点
# 2. 状态机由 Reaction Builder 根据 Pathway Graph 自动构建，也可由用户显式注入
# 3. 状态转换必须关联到存在的 Reaction（Validation Rule 5）
# 4. 状态机用于表达多状态蛋白（如 EGFR 单体→二聚体→磷酸化→招募 Grb2）

from __future__ import annotations

from typing import Any

from app.reaction_ir_v2.schema import (
    ReactionIRv2,
    State,
    StateMachine,
    Transition,
)


# =============================================================================
# 状态机构建器
# =============================================================================
class StateMachineBuilder:
    """蛋白质状态机构建辅助器。

    用法：
        builder = StateMachineBuilder("EGFR_STATE_MACHINE", "EGFR")
        builder.add_state("monomer", "SP_EGFR", is_initial=True)
        builder.add_state("egf_bound", "SP_EGFR_EGF")
        builder.add_transition("monomer", "egf_bound", "RXN_001", "ligand_binding")
        sm = builder.build()
    """

    def __init__(self, sm_id: str, species_name: str) -> None:
        self.sm_id = sm_id
        self.species_name = species_name
        self._states: list[State] = []
        self._transitions: list[Transition] = []
        self._state_names: set[str] = set()

    def add_state(
        self,
        name: str,
        species_id: str,
        is_initial: bool = False,
    ) -> "StateMachineBuilder":
        """添加状态。重复添加同名状态会被忽略。"""
        if name in self._state_names:
            return self
        self._states.append(State(
            name=name,
            species_id=species_id,
            is_initial=is_initial,
        ))
        self._state_names.add(name)
        return self

    def add_transition(
        self,
        from_state: str,
        to_state: str,
        reaction_id: str,
        trigger: str = "ligand_binding",
    ) -> "StateMachineBuilder":
        """添加状态转换。from_state / to_state 应已通过 add_state 添加。"""
        self._transitions.append(Transition(
            from_state=from_state,
            to_state=to_state,
            reaction_id=reaction_id,
            trigger=trigger,
        ))
        return self

    def build(self) -> StateMachine:
        """构建 StateMachine 对象。"""
        return StateMachine(
            id=self.sm_id,
            species=self.species_name,
            states=list(self._states),
            transitions=list(self._transitions),
        )


# =============================================================================
# EGFR 状态机示例（架构 §4.2.4）
# =============================================================================
def build_egfr_state_machine(
    egfr_species_id: str = "SP_EGFR",
    egf_bound_id: str = "SP_EGFR_EGF",
    dimer_id: str = "SP_EGFR_DIMER",
    phosphorylated_id: str = "SP_pEGFR",
    grb2_bound_id: str = "SP_pEGFR_GRB2",
    internalized_id: str = "SP_EGFR_INT",
    degraded_id: str = "SP_EGFR_DEG",
) -> StateMachine:
    """构建 EGFR 状态机（架构 §4.2.4 示例）。

    状态转换链：
      monomer → EGF-bound → dimer → phosphorylated → Grb2-bound → internalized → degraded

    每个状态转换关联一个 Reaction ID（实际 ID 由 Reaction Builder 分配）。
    """
    builder = StateMachineBuilder("EGFR_STATE_MACHINE", "EGFR")
    # 状态
    builder.add_state("monomer", egfr_species_id, is_initial=True)
    builder.add_state("egf_bound", egf_bound_id)
    builder.add_state("dimer", dimer_id)
    builder.add_state("phosphorylated_dimer", phosphorylated_id)
    builder.add_state("grb2_bound", grb2_bound_id)
    builder.add_state("internalized", internalized_id)
    builder.add_state("degraded", degraded_id)
    # 转换
    builder.add_transition("monomer", "egf_bound", "RXN_EGFR_BINDING", "ligand_binding")
    builder.add_transition("egf_bound", "dimer", "RXN_EGFR_DIMERIZATION", "dimerization")
    builder.add_transition(
        "dimer", "phosphorylated_dimer",
        "RXN_EGFR_AUTOPHOS", "phosphorylation",
    )
    builder.add_transition(
        "phosphorylated_dimer", "grb2_bound",
        "RXN_EGFR_GRB2_RECRUIT", "recruitment",
    )
    builder.add_transition(
        "grb2_bound", "internalized",
        "RXN_EGFR_ENDOCYTOSIS", "internalization",
    )
    builder.add_transition(
        "internalized", "degraded",
        "RXN_EGFR_DEGRADATION", "degradation",
    )
    return builder.build()


# =============================================================================
# 状态机查询辅助
# =============================================================================
def get_initial_state(sm: StateMachine) -> State | None:
    """获取状态机的初始状态。"""
    for s in sm.states:
        if s.is_initial:
            return s
    return None


def get_transitions_from(sm: StateMachine, state_name: str) -> list[Transition]:
    """获取从指定状态出发的所有转换。"""
    return [t for t in sm.transitions if t.from_state == state_name]


def get_transitions_to(sm: StateMachine, state_name: str) -> list[Transition]:
    """获取到达指定状态的所有转换。"""
    return [t for t in sm.transitions if t.to_state == state_name]


def validate_state_machine(sm: StateMachine, ir: ReactionIRv2) -> list[str]:
    """校验单个状态机（Validation Rule 5 的单机版本）。

    检查：
    1. 至少一个初始状态
    2. transition.from_state / to_state 都在 states 列表
    3. transition.reaction_id 引用的反应存在（非空时）
    """
    violations: list[str] = []
    state_names = {s.name for s in sm.states}
    # 1. 初始状态
    initial_count = sum(1 for s in sm.states if s.is_initial)
    if initial_count == 0:
        violations.append(
            f"StateMachine {sm.id}: 缺少初始状态（is_initial=True）"
        )
    elif initial_count > 1:
        violations.append(
            f"StateMachine {sm.id}: 存在 {initial_count} 个初始状态（应仅有 1 个）"
        )
    # 2. transition 状态存在性
    for trans in sm.transitions:
        if trans.from_state and trans.from_state not in state_names:
            violations.append(
                f"StateMachine {sm.id}: from_state '{trans.from_state}' 不在 states 列表"
            )
        if trans.to_state and trans.to_state not in state_names:
            violations.append(
                f"StateMachine {sm.id}: to_state '{trans.to_state}' 不在 states 列表"
            )
    # 3. reaction 存在性
    reaction_ids = {rxn.id for rxn in ir.reactions}
    for trans in sm.transitions:
        if trans.reaction_id and trans.reaction_id not in reaction_ids:
            violations.append(
                f"StateMachine {sm.id}: transition {trans.from_state}→{trans.to_state} "
                f"引用了不存在的反应 {trans.reaction_id}"
            )
    return violations


def state_machine_to_dict(sm: StateMachine) -> dict[str, Any]:
    """状态机序列化为 dict（供日志 / 调试用）。"""
    return sm.model_dump()


__all__ = [
    "StateMachineBuilder",
    "build_egfr_state_machine",
    "get_initial_state",
    "get_transitions_from",
    "get_transitions_to",
    "validate_state_machine",
    "state_machine_to_dict",
]
