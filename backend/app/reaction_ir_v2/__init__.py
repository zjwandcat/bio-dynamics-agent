# BioDynamics Agent v4 - Reaction IR v2（反应中间表示层）
# Phase 2 新增模块：实现 17 类机制 + CompositeReaction + State Machine + Compartment + Constraint。
# 对应 v4 Scientific Architecture Part 4。
#
# 设计原则（铁律）：
# 1. Feature Flag V4_REACTION_IR_ENABLED=false 时完全不执行，系统行为同 v3
# 2. v4 Reaction IR 与 v3 network_json 通过 Adapter 双向兼容
# 3. 新代码写入 v4_reaction_ir 字段时，可通过 Adapter 同步到 network_json
# 4. 旧代码写入 network_json 时，不同步到 v4_reaction_ir（v4 字段保持 None）
# 5. 所有模块零 LLM 依赖，纯规则与 Pydantic 模型

from app.reaction_ir_v2.schema import (
    Compartment,
    CompositeReaction,
    Constraint,
    Modifier,
    OntologyRef,
    Provenance,
    ReactionIRv2,
    ReactionV2,
    SpeciesRef,
    SpeciesV2,
    State,
    StateMachine,
    Transition,
)
from app.reaction_ir_v2.mechanism_types import (
    MechanismType,
    get_mechanism_category,
    is_degradation_mechanism,
    is_enzymatic_mechanism,
    is_transport_mechanism,
    v3_interaction_to_mechanism,
)
from app.reaction_ir_v2.reaction_builder import (
    build_from_network_json,
    build_from_pathway_graph,
)
from app.reaction_ir_v2.state_machine import (
    StateMachineBuilder,
    build_egfr_state_machine,
)
from app.reaction_ir_v2.composite_reaction import (
    CompositeReactionBuilder,
    build_wnt_destruction_complex_reactions,
)
from app.reaction_ir_v2.constraints import check_all_constraints
from app.reaction_ir_v2.validation_rules import validate_all

__all__ = [
    # Schema
    "ReactionIRv2", "SpeciesV2", "ReactionV2", "CompositeReaction",
    "StateMachine", "State", "Transition", "Compartment", "Constraint",
    "SpeciesRef", "Modifier", "OntologyRef", "Provenance",
    # Mechanism types
    "MechanismType", "v3_interaction_to_mechanism",
    "get_mechanism_category", "is_enzymatic_mechanism",
    "is_transport_mechanism", "is_degradation_mechanism",
    # Builder
    "build_from_network_json", "build_from_pathway_graph",
    "StateMachineBuilder", "build_egfr_state_machine",
    "CompositeReactionBuilder", "build_wnt_destruction_complex_reactions",
    # Validation
    "check_all_constraints", "validate_all",
]
