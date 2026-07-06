# BioDynamics Agent v4 - Pathway Graph Schema
# 对应 v4 Scientific Architecture Part 2 Layer 2 + Part 3 §3.1 通路模块化设计。
#
# 核心定义（PART C1）：
#   Pathway Graph =
#     not a list of reactions
#     but:
#     hierarchical signaling network with:
#       * nodes (protein / gene / complex)
#       * edges (mechanistic reactions)
#       * states (phosphorylation / localization)
#       * feedback loops
#       * cross-talk edges
#
# 设计原则：
# 1. Pydantic v2 模型，保证 JSON 序列化与运行时校验
# 2. PathwayNode 携带 Layer 1 OntologyRef（HGNC/UniProt/ChEBI/GO/SBO）
# 3. PathwayEdge 携带 mechanism_type（17 类之一）+ pathway_tag + provenance
# 4. PathwayState 表达蛋白的磷酸化/定位/复合状态（不压扁为多个 species）
# 5. FeedbackLoop / CrossTalkEdge 为一等公民，显式标记（不靠环路检测事后识别）
# 6. TemporalAnnotation 强制多时间尺度（fast/medium/slow），回应审计 §3.6 max_step 过粗

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# 枚举：通路模块类别（对应架构 Part 3 §3.1 的 5 模块化设计）
# =============================================================================
class PathwayModule(str, Enum):
    """通路 5 模块类别（每条通路都由这 5 个模块组成）。

    对应架构 §3.1：
      - CORE: 通路核心拓扑与机制
      - FEEDBACK: 通路内负/正反馈
      - CROSSTALK: 与其他通路的交叉点
      - PERTURBATION: 药物 / KO / 突变影响
      - VALIDATION: 通路特异验证规则与 benchmark
    """

    CORE = "core"
    FEEDBACK = "feedback"
    CROSSTALK = "crosstalk"
    PERTURBATION = "perturbation"
    VALIDATION = "validation"


# =============================================================================
# 时间尺度枚举（回应审计 §3.6 无多时间尺度分层）
# =============================================================================
class TimeScale(str, Enum):
    """三层时间尺度（对应架构 §6.2）。"""

    FAST = "fast"        # 0-30 min：磷酸化、结合、cleavage
    MEDIUM = "medium"    # 30 min - 6h：转录、翻译、ubiquitination
    SLOW = "slow"        # 6h - 48h：细胞周期、降解、稳态


# =============================================================================
# PathwayNode Schema
# =============================================================================
class PathwayNode(BaseModel):
    """Pathway Graph 节点：蛋白 / 基因 / 复合物 / 化学实体。

    对应 PART C2。每个 node 携带 Layer 1 OntologyRef、所属通路、模块类别、
    时间尺度与状态列表。状态用 PathwayState 表达，避免压扁为多个 species。
    """

    id: str                              # 全局唯一 ID，如 "PN_EGFR"
    canonical_name: str                  # 规范名，如 "EGFR"
    display_name: str = ""               # 显示名
    species_type: str = "protein"        # protein/gene/complex/ligand/drug/rna
    pathway_class: str = ""              # 所属通路类别（如 "EGFR_RTK"）
    module: PathwayModule = PathwayModule.CORE  # 所属模块（5 模块之一）
    # Ontology 引用（来自 P1 Ontology Agent）
    hgnc_id: str | None = None
    uniprot_id: str | None = None
    chebi_id: str | None = None
    go_terms: list[str] = Field(default_factory=list)
    sbo_term: str | None = None
    ontology_verified: bool = False      # P1 Ontology Agent 是否已验证
    # 区室（与 SpeciesV2.compartment 对齐）
    compartment: str = "cytoplasm"       # extracellular/membrane/cytoplasm/nucleus/mitochondria
    # 时间尺度（用于 ODE 求解器 max_step 配置）
    time_scale: TimeScale = TimeScale.FAST
    # 状态列表（磷酸化/定位/复合状态，避免压扁为多个 species）
    states: list["PathwayState"] = Field(default_factory=list)
    # 是否为 shared species（跨通路共享，如 Ras/AKT/MEK）
    is_shared: bool = False
    shared_with: list[str] = Field(default_factory=list)  # 共享的其他通路类别
    # 溯源
    source_sbml: str | None = None
    source_pmid: str | None = None

    @field_validator("compartment")
    @classmethod
    def _validate_compartment(cls, v: str) -> str:
        allowed = {"extracellular", "membrane", "cytoplasm", "nucleus", "mitochondria"}
        return v if v in allowed else "cytoplasm"

    @field_validator("species_type")
    @classmethod
    def _validate_species_type(cls, v: str) -> str:
        allowed = {"protein", "gene", "complex", "ligand", "drug", "chemical", "mrna", "rna"}
        return v if v in allowed else "protein"


# =============================================================================
# PathwayState Schema（节点状态，避免压扁）
# =============================================================================
class PathwayState(BaseModel):
    """蛋白节点的状态（磷酸化/定位/复合）。

    对应 PART C2 State representation。一个 EGFR 节点可有多个状态：
      - monomer / dimer / phosphorylated / internalized
    状态转换由 PathwayEdge 触发，不再为每个状态创建独立 species。
    """

    name: str                            # 状态名，如 "phosphorylated" / "nuclear"
    state_type: str = "phosphorylation"  # phosphorylation/localization/binding/conformational
    is_initial: bool = False             # 是否为初始状态
    # 状态修饰位点（如 EGFR pY1068）
    site: str | None = None              # 如 "Tyr1068" / "Ser259"

    @field_validator("state_type")
    @classmethod
    def _validate_state_type(cls, v: str) -> str:
        allowed = {
            "phosphorylation", "localization", "binding",
            "conformational", "ubiquitination", "cleavage",
        }
        return v if v in allowed else "phosphorylation"


# =============================================================================
# PathwayEdge Schema
# =============================================================================
class PathwayEdge(BaseModel):
    """Pathway Graph 边：机制化反应（不是简单的 activation/inhibition）。

    对应 PART C2。每条 edge 携带 17 类机制之一、通路标签、模块类别、
    时间尺度与溯源链。Cross-talk edge 用 is_crosstalk=True 标记。
    """

    id: str                              # 全局唯一 ID，如 "PE_EGF_EGFR_BIND"
    source: str                          # source node id
    target: str                          # target node id
    mechanism: str = "activation"        # 17 类机制之一（与 ReactionV2.reaction_type 对齐）
    pathway_tag: str = ""                # 通路标签，如 "EGFR_RTK"
    module: PathwayModule = PathwayModule.CORE
    # 是否为 cross-talk edge（跨通路）
    is_crosstalk: bool = False
    cross_talk_to: str = ""              # 跨向的通路类别（is_crosstalk=True 时必填）
    # 反馈环标记（回应审计 §2.5 环路打破丢 inhibition）
    is_feedback: bool = False
    feedback_type: str = ""              # "negative" / "positive" / ""
    # 时间尺度
    time_scale: TimeScale = TimeScale.FAST
    # SBO term（来自 P1 sbo_terms）
    sbo_term: str | None = None
    # 默认动力学（由 Mechanism Layer 决定，P3 阶段先用默认值）
    kinetics_type: str = "mass_action"   # mass_action/Michaelis_Menten/Hill/Boolean/hybrid
    # 溯源
    source_sbml_reaction: str | None = None
    source_pmid: str | None = None
    source_kegg: str | None = None
    # 修饰位点（如 AKT → Raf Ser259 抑制）
    site: str | None = None

    @field_validator("mechanism")
    @classmethod
    def _validate_mechanism(cls, v: str) -> str:
        # 17 类机制白名单（与 mechanism_types.py 对齐）
        allowed = {
            "phosphorylation", "dephosphorylation", "ubiquitination",
            "binding", "dissociation", "dimerization", "complex_formation", "sequestration",
            "cleavage", "gtp_gdp_exchange",
            "transcription", "translation",
            "nuclear_import", "nuclear_export", "cytoplasm_translocation",
            "degradation", "proteasomal_degradation",
            "inhibition", "activation",
        }
        return v if v in allowed else "activation"

    @field_validator("kinetics_type")
    @classmethod
    def _validate_kinetics(cls, v: str) -> str:
        allowed = {"mass_action", "Michaelis_Menten", "Hill", "Boolean", "hybrid"}
        return v if v in allowed else "mass_action"


# =============================================================================
# FeedbackLoop Schema（显式标记反馈环）
# =============================================================================
class FeedbackLoop(BaseModel):
    """反馈环：显式标记的反馈环路（不靠环路检测事后识别）。

    回应审计 §2.5 环路打破丢 inhibition。v4 不再做环路打破，
    而是显式标记反馈环并要求 Validation Layer 验证其稳定性。
    """

    id: str                              # 如 "FL_MAPK_SOS_NEG"
    loop_type: str = "negative"          # negative / positive
    pathway_class: str = ""
    # 环路涉及的边（有序，构成环路）
    edge_ids: list[str] = Field(default_factory=list)
    # 环路涉及的节点
    node_ids: list[str] = Field(default_factory=list)
    # 延迟（用于 DDE 求解器，回应审计 §3.2 无 DDE）
    delay_minutes: float = 0.0           # 转录延迟约 30-120 min
    # 来源
    source_pmid: str | None = None
    description: str = ""

    @field_validator("loop_type")
    @classmethod
    def _validate_loop_type(cls, v: str) -> str:
        return v if v in {"negative", "positive"} else "negative"


# =============================================================================
# CrossTalkEdge Schema（跨通路交叉点）
# =============================================================================
class CrossTalkEdge(BaseModel):
    """Cross-talk edge：跨通路交叉点（回应审计 §2.2 cross-talk 被禁止）。

    v4 显式支持 cross-talk。每条 cross-talk edge 携带两个 pathway_tag，
    标记 shared species 与调控方向。
    """

    id: str                              # 如 "CT_PI3K_AKT_TO_MAPK_RAF"
    source_pathway: str                  # 源通路类别，如 "PI3K_AKT_mTOR"
    target_pathway: str                  # 目标通路类别，如 "MAPK_ERK"
    source_node: str                     # 源通路中的 node id
    target_node: str                     # 目标通路中的 node id
    mechanism: str = "inhibition"        # 17 类机制之一
    # shared species 标记（如 Ras / AKT / MEK 跨通路共享）
    shared_species: list[str] = Field(default_factory=list)
    # 调控位点（如 AKT → Raf Ser259）
    site: str | None = None
    sbo_term: str | None = None
    source_pmid: str | None = None
    description: str = ""


# =============================================================================
# TemporalAnnotation Schema（多时间尺度标注）
# =============================================================================
class TemporalAnnotation(BaseModel):
    """通路级时间尺度标注（回应审计 §3.6 max_step 过粗）。

    每条通路根据其主导过程分配时间尺度，ODE 求解器据此配置 max_step。
    """

    pathway_class: str
    primary_scale: TimeScale = TimeScale.FAST
    # 求解器最大步长（分钟）
    max_step_minutes: float = 0.1        # FAST 默认 0.1 min
    # 仿真总时长（分钟）
    t_end_minutes: float = 60.0
    # 是否需要 DDE（转录延迟反馈）
    requires_dde: bool = False
    # DDE 延迟（分钟，requires_dde=True 时使用）
    dde_delay_minutes: float = 0.0


# =============================================================================
# PathwayGraph 顶层容器
# =============================================================================
class PathwayGraph(BaseModel):
    """v4 Pathway Graph 顶层容器。

    对应 PART C2 Required Data Model。包含 nodes / edges / feedback_loops /
    cross_talk_edges / temporal_annotation。可序列化为 dict 存入 state.v4_pathway_graph。
    """

    pathway_class: str = ""              # 主通路类别（多通路时为 "MULTI"）
    nodes: list[PathwayNode] = Field(default_factory=list)
    edges: list[PathwayEdge] = Field(default_factory=list)
    feedback_loops: list[FeedbackLoop] = Field(default_factory=list)
    cross_talk_edges: list[CrossTalkEdge] = Field(default_factory=list)
    temporal: TemporalAnnotation | None = None
    # 元信息
    version: str = "v4.0"
    source: str = "v4_native"            # v4_native / v3_downgraded / registry_init
    warnings: list[str] = Field(default_factory=list)

    def node_by_id(self, node_id: str) -> PathwayNode | None:
        """按 ID 查找节点。"""
        for n in self.nodes:
            if n.id == node_id:
                return n
        return None

    def node_by_name(self, name: str) -> PathwayNode | None:
        """按规范名查找节点（首个匹配）。"""
        for n in self.nodes:
            if n.canonical_name == name:
                return n
        return None

    def edges_for_node(self, node_id: str) -> list[PathwayEdge]:
        """返回涉及指定节点的所有边。"""
        return [
            e for e in self.edges
            if e.source == node_id or e.target == node_id
        ]

    def shared_species_nodes(self) -> list[PathwayNode]:
        """返回所有 shared species 节点（跨通路共享）。"""
        return [n for n in self.nodes if n.is_shared]

    def to_dict(self) -> dict[str, Any]:
        """序列化为 dict（供 state 存储）。"""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PathwayGraph":
        """从 dict 反序列化（容错：忽略未知字段）。"""
        return cls.model_validate(data)


# 解决 Pydantic 前向引用
PathwayNode.model_rebuild()


__all__ = [
    "PathwayModule",
    "TimeScale",
    "PathwayNode",
    "PathwayState",
    "PathwayEdge",
    "FeedbackLoop",
    "CrossTalkEdge",
    "TemporalAnnotation",
    "PathwayGraph",
]
