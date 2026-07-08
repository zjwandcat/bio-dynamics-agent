# BioDynamics Agent v4 - Reaction IR v2 Schema
# 对应 v4 Scientific Architecture Part 4 的 6 个核心组件：
#   Species / Reaction / CompositeReaction / StateMachine / Compartment / Constraint
#
# 设计原则（铁律）：
# 1. 所有模型使用 Pydantic v2，保证 JSON 序列化与运行时校验
# 2. 17 类机制 + SBO term 通过 mechanism_types.py 引用 P1 已定义的常量
# 3. v4 Reaction IR 与 v3 network_json 通过 Adapter 双向兼容
# 4. State Machine / CompositeReaction 为可选字段，降级模式（v3→v4）可为空
# 5. 所有 provenance 字段强制可空，保证 fail-safe 时不阻塞

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

# 模块级 logger，供 TD-041 降级警告使用
_logger = logging.getLogger(__name__)


# =============================================================================
# 1. Species 相关 Schema
# =============================================================================

class OntologyRef(BaseModel):
    """物种的标准本体引用（对应架构 §4.2.1 的 ontology 字段）。

    所有 ID 强制可空：v3→v4 降级模式下无法填入本体 ID，此时 verified=False。
    """

    hgnc_id: str | None = None          # HGNC 基因 ID，如 "HGNC:3236"
    uniprot_id: str | None = None       # UniProt 蛋白 accession，如 "P00533"
    chebi_id: str | None = None         # ChEBI 化学实体 ID（配体/药物用）
    go_terms: list[str] = Field(default_factory=list)  # GO 功能术语列表
    sbo_term: str | None = None         # SBO 生物学本体，如 "SBO:0000252" for protein
    verified: bool = False              # 是否已通过 API 验证（False 表示降级模式）


class SpeciesV2(BaseModel):
    """v4 Species：结构化物种定义（替代 v3 network_json 中扁平的 node）。

    对应架构 §4.2.1。每个 Species 携带本体引用、区室、初始浓度与溯源链。
    """

    id: str                             # 全局唯一 ID，如 "SP_001"
    canonical_name: str                 # 规范名，如 "EGFR"
    display_name: str = ""              # 显示名，如 "EGF Receptor"
    ontology: OntologyRef = Field(default_factory=OntologyRef)
    species_type: str = "protein"       # 11 类之一：ligand/receptor/kinase/...
    state_machine: str | None = None    # 关联的状态机 ID
    compartment: str = "cytoplasm"      # extracellular/membrane/cytoplasm/nucleus/mitochondria
    initial_concentration: float = 0.0  # 初始浓度（nM）
    concentration_unit: str = "nM"      # nM / molecule_per_cell
    # TD-031 (IB-038): 单位转换因子，用于浓度单位换算（如 nM → molecule/cell），默认 1.0（向后兼容）
    unit_conversion_factor: float = 1.0
    # 溯源链：SBML model ID / PMID / UniProt entry
    source_sbml: str | None = None
    source_pmid: str | None = None
    source_uniprot: str | None = None

    @field_validator("compartment")
    @classmethod
    def _validate_compartment(cls, v: str) -> str:
        """区室必须是 5 个合法值之一，非法值降级为 cytoplasm。"""
        allowed = {"extracellular", "membrane", "cytoplasm", "nucleus", "mitochondria"}
        if v not in allowed:
            return "cytoplasm"
        return v


class SpeciesRef(BaseModel):
    """反应中物种引用（含化学计量与角色）。"""

    species_id: str
    stoichiometry: int = 1
    role: str = "substrate"  # substrate/product/enzyme/cofactor

    @field_validator("role")
    @classmethod
    def _validate_role(cls, v: str) -> str:
        allowed = {"substrate", "product", "enzyme", "cofactor"}
        if v not in allowed:
            return "substrate"
        return v


class Modifier(BaseModel):
    """反应调控因子（酶 / 催化剂 / 变构调节子）。"""

    species_id: str
    modifier_type: str = "catalytic"  # catalytic/allosteric/inhibitory/activating
    # TD-028 (IB-011): site 改为 list[str] 支持多位点，向后兼容单 string（validator 自动转换）
    site: list[str] = Field(default_factory=list)  # 修饰位点列表，如 ["Ser259", "Tyr1068"]
    # TD-027 (IB-010): 调控因子动力学参数（全部可选，默认 None，向后兼容）
    ki: float | None = None             # 抑制常数 Ki（inhibitory 调控子用）
    kact: float | None = None           # 激活常数 Kact（activating 调控子用）
    n_hill: float | None = None         # Hill 系数（协同调控用）
    inhibition_type: str | None = None  # 抑制类型：competitive/uncompetitive/noncompetitive/mixed
    alpha: float | None = None          # 变构耦合因子（allosteric 调控子用）

    @field_validator("modifier_type")
    @classmethod
    def _validate_modifier_type(cls, v: str) -> str:
        allowed = {"catalytic", "allosteric", "inhibitory", "activating"}
        if v not in allowed:
            return "catalytic"
        return v

    @field_validator("site", mode="before")
    @classmethod
    def _validate_site(cls, v):
        """TD-028: 向后兼容——单 string 自动转为 list[str]，None 转为空 list。

        使用 mode="before" 在 Pydantic list[str] 类型强转之前拦截原始输入，
        避免裸字符串被拒绝（Pydantic v2 默认不会把 str 拆成 list）。
        """
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return v

    @field_validator("inhibition_type")
    @classmethod
    def _validate_inhibition_type(cls, v: str | None) -> str | None:
        """TD-027: inhibition_type 必须为合法值之一或 None。"""
        if v is None:
            return None
        allowed = {"competitive", "uncompetitive", "noncompetitive", "mixed"}
        if v not in allowed:
            raise ValueError(
                f"inhibition_type 非法值 '{v}'，合法值: {sorted(allowed)}"
            )
        return v


# =============================================================================
# 2. Reaction 相关 Schema
# =============================================================================

class Provenance(BaseModel):
    """反应溯源信息（对应架构 §4.2.2 的 provenance 字段）。"""

    source_sbml_reaction: str | None = None  # SBML reaction ID
    source_pmid: str | None = None
    source_kegg: str | None = None           # KEGG reaction ID


class Constraint(BaseModel):
    """约束定义（对应架构 §4.2.6）。"""

    type: str                          # mass_conservation/steady_state/non_negative/enzymatic/thermodynamic
    scope: str = "species"             # species/reaction/pathway/global
    expression: str                    # 约束表达式，如 "EGFR + pEGFR + EGF-EGFR = EGFR_total"
    # TD-040 (IB-036): 结构化约束表达式，与 expression 字符串并存（向后兼容，可选）
    # 形如 {"lhs": [{"species": "A", "coeff": 1}], "rhs": [{"species": "B_total", "coeff": 1}], "operator": "="}
    expression_structured: dict[str, Any] | None = None
    tolerance: float = 0.05            # 容差（0.05 表示 5%）
    provenance: str = ""               # 约束来源，如 "Schoeberl 2002"
    # TD-041 (IB-037): strict 模式控制非法 type 的处理方式
    # strict=True（默认）：非法 type 抛 ValueError；strict=False：降级为 non_negative 并记录警告（向后兼容遗留数据）
    strict: bool = True

    # 合法 type 集合（供 model_validator 校验）
    _CONSTRAINT_ALLOWED_TYPES = frozenset({
        "mass_conservation", "steady_state", "non_negative",
        "enzymatic", "thermodynamic",
    })

    @model_validator(mode="after")
    def _validate_type_strict(self) -> "Constraint":
        """TD-041: type 非法时根据 strict 决定抛错或降级（替代旧的静默降级）。"""
        if self.type in self._CONSTRAINT_ALLOWED_TYPES:
            return self
        # 非法 type
        if self.strict:
            raise ValueError(
                f"Constraint.type 非法值 '{self.type}'，"
                f"合法值: {sorted(self._CONSTRAINT_ALLOWED_TYPES)}"
            )
        # strict=False：降级为 non_negative 并记录警告（向后兼容遗留数据）
        _logger.warning(
            "Constraint.type 非法值 '%s'，strict=False 降级为 'non_negative'", self.type
        )
        self.type = "non_negative"
        return self


class ReactionV2(BaseModel):
    """v4 Reaction：结构化反应定义（替代 v3 network_json 中扁平的 edge）。

    对应架构 §4.2.2。每条 Reaction 携带机制类型、动力学类型、反应物/产物/调控因子、
    区室、参数上下文、通路标签与溯源链。
    """

    id: str                                # 全局唯一 ID，如 "RXN_001"
    reaction_type: str                     # 17 类机制之一（见 mechanism_types.py）
    kinetics_type: str = "mass_action"     # mass_action/Michaelis_Menten/Hill/Boolean/hybrid
    reactants: list[SpeciesRef] = Field(default_factory=list)
    products: list[SpeciesRef] = Field(default_factory=list)
    modifiers: list[Modifier] = Field(default_factory=list)
    compartments: list[str] = Field(default_factory=list)  # 涉及的 compartment
    parameter_context: str = ""            # 参数上下文，如 "EGF-EGFR binding kon/koff"
    pathway_tag: str = ""                  # 通路标签，如 "EGFR_RTK"
    provenance: Provenance = Field(default_factory=Provenance)
    constraints: list[Constraint] = Field(default_factory=list)

    @field_validator("kinetics_type")
    @classmethod
    def _validate_kinetics(cls, v: str) -> str:
        allowed = {"mass_action", "Michaelis_Menten", "Hill", "Boolean", "hybrid"}
        if v not in allowed:
            return "mass_action"
        return v


# =============================================================================
# 3. CompositeReaction Schema（组合反应）
# =============================================================================

class CompositeReaction(BaseModel):
    """组合反应：多个子反应的有序集合（对应架构 §4.2.3）。

    典型示例：Wnt destruction complex = binding + phosphorylation + ubiquitination 三步耦合。
    """

    id: str
    name: str                                # 如 "Wnt destruction complex"
    sub_reactions: list[str] = Field(default_factory=list)  # 子反应 ID 有序列表
    coupling_type: str = "sequential"        # sequential/branched/cyclic
    intermediate_species: list[str] = Field(default_factory=list)  # 中间产物 species_id
    net_reaction: str = ""                   # 净反应方程，如 "β-catenin → ∅"

    @field_validator("coupling_type")
    @classmethod
    def _validate_coupling(cls, v: str) -> str:
        allowed = {"sequential", "branched", "cyclic"}
        if v not in allowed:
            return "sequential"
        return v


# =============================================================================
# 4. State Machine Schema（蛋白质状态转换）
# =============================================================================

class State(BaseModel):
    """状态机中的单个状态。"""

    name: str                # 如 "monomer" / "dimer" / "phosphorylated_dimer"
    species_id: str          # 对应的 Species ID
    is_initial: bool = False


class Transition(BaseModel):
    """状态机中的状态转换。"""

    from_state: str
    to_state: str
    reaction_id: str         # 关联的 Reaction ID（触发状态转换的反应）
    trigger: str = "ligand_binding"  # ligand_binding/phosphorylation/internalization/...


class StateMachine(BaseModel):
    """蛋白质状态机（对应架构 §4.2.4）。

    典型示例：EGFR 单体 → EGF-bound → 二聚体 → 磷酸化二聚体 → Grb2-bound → 内吞 → 降解。
    """

    id: str                  # 如 "EGFR_STATE_MACHINE"
    species: str             # 关联的蛋白名，如 "EGFR"
    states: list[State] = Field(default_factory=list)
    transitions: list[Transition] = Field(default_factory=list)


# =============================================================================
# 5. Compartment Schema
# =============================================================================

class Compartment(BaseModel):
    """细胞区室（对应架构 §4.2.5）。"""

    name: str                # extracellular/membrane/cytoplasm/nucleus/mitochondria
    size: float = 1.0        # 体积比，如 cytoplasm=0.5, nucleus=0.1
    transport_reactions: list[str] = Field(default_factory=list)  # 跨该区室的运输反应 ID

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        allowed = {
            "extracellular", "membrane", "cytoplasm", "nucleus", "mitochondria",
        }
        if v not in allowed:
            return "cytoplasm"
        return v


# =============================================================================
# 6. ReactionIRv2 顶层容器
# =============================================================================

class ReactionIRv2(BaseModel):
    """v4 Reaction IR 顶层容器（对应架构 §4.2 全部 6 个组件）。

    所有 v4 Reaction IR 操作的入口与出口。Adapter 的 v3_to_v4 输出此对象，
    v4_to_v3 输入此对象。State 字段 v4_reaction_ir 存储其 .model_dump() 结果。
    """

    species: list[SpeciesV2] = Field(default_factory=list)
    reactions: list[ReactionV2] = Field(default_factory=list)
    composite_reactions: list[CompositeReaction] = Field(default_factory=list)
    state_machines: list[StateMachine] = Field(default_factory=list)
    compartments: list[Compartment] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list)
    # 元信息
    version: str = "v4.0"
    source: str = "v4_native"  # v4_native / v3_downgraded / sbml_parsed
    warnings: list[str] = Field(default_factory=list)

    def species_by_id(self, species_id: str) -> SpeciesV2 | None:
        """按 ID 查找物种。"""
        for sp in self.species:
            if sp.id == species_id:
                return sp
        return None

    def species_by_name(self, name: str) -> SpeciesV2 | None:
        """按规范名查找物种（首个匹配）。"""
        for sp in self.species:
            if sp.canonical_name == name:
                return sp
        return None

    def reaction_by_id(self, reaction_id: str) -> ReactionV2 | None:
        """按 ID 查找反应。"""
        for rxn in self.reactions:
            if rxn.id == reaction_id:
                return rxn
        return None

    def reactions_for_species(self, species_id: str) -> list[ReactionV2]:
        """返回涉及指定物种的所有反应。"""
        result: list[ReactionV2] = []
        for rxn in self.reactions:
            involved = (
                any(ref.species_id == species_id for ref in rxn.reactants)
                or any(ref.species_id == species_id for ref in rxn.products)
                or any(mod.species_id == species_id for mod in rxn.modifiers)
            )
            if involved:
                result.append(rxn)
        return result

    def to_dict(self) -> dict[str, Any]:
        """序列化为 dict（供 state 存储 / Adapter 消费）。"""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReactionIRv2":
        """从 dict 反序列化（容错：忽略未知字段）。"""
        return cls.model_validate(data)
