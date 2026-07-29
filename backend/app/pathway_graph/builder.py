# BioDynamics Agent v4 - Pathway Graph Builder
# 对应 PART C3：Pathway Graph Builder。
#
# 数据流（P1 → P2 → P3）：
#   Ontology Agent (P1)  ──→  PathwayNode (canonical_name, hgnc_id, uniprot_id, ...)
#                                    │
#   Reaction IR v2 (P2)  ──→  PathwayEdge (mechanism, kinetics_type, ...)
#                                    │
#   pathway_registry (P1) ──→  pathway_class / module / shared_species 标记
#                                    │
#   SBML BioModels (P1)  ──→  source_sbml / source_sbml_reaction 溯源
#                                    ▼
#                          PathwayGraph (P3)
#
# 设计原则：
# 1. 纯规则构建，不调用 LLM（避免幻觉）
# 2. 输入：ontology_entities (dict) + reaction_ir (dict) + pathway_class (str)
# 3. 输出：PathwayGraph 对象（可序列化为 state.v4_pathway_graph）
# 4. 容错：ontology 或 reaction_ir 缺失时返回 warnings，不抛异常
# 5. 不破坏 P1/P2 不可碰清单：只读取它们的输出 dict，不修改它们

from __future__ import annotations

import logging
from typing import Any

from .schema import (
    CrossTalkEdge,
    FeedbackLoop,
    PathwayEdge,
    PathwayGraph,
    PathwayModule,
    PathwayNode,
    PathwayState,
    TemporalAnnotation,
    TimeScale,
)
# v4 迁移 P2→P3 接口适配：复用 P2 的 SBO term 查询（避免硬编码）
try:
    from app.reaction_ir_v2.mechanism_types import MechanismType as _V4MechanismType
    _V4_MECHANISM_AVAILABLE = True
except ImportError:  # pragma: no cover
    _V4_MECHANISM_AVAILABLE = False

logger = logging.getLogger(__name__)


def _lookup_sbo_term(mechanism: str) -> str | None:
    """从 P2 MechanismType 查询 SBO term，P2 不可碰时降级返回 None。"""
    if not _V4_MECHANISM_AVAILABLE or not mechanism:
        return None
    try:
        return _V4MechanismType(mechanism).sbo_term
    except (ValueError, KeyError):
        return None


# =============================================================================
# 机制类型 → 默认动力学映射（P3 阶段先用规则默认值，P4 Mechanism Layer 接管）
# =============================================================================
MECHANISM_DEFAULT_KINETICS: dict[str, str] = {
    # 酶催化类（默认 Michaelis-Menten，回应审计 §3.1 致命错误）
    "phosphorylation": "Michaelis_Menten",
    "dephosphorylation": "Michaelis_Menten",
    "ubiquitination": "Michaelis_Menten",
    # 结合/解离类（mass-action，可逆）
    "binding": "mass_action",
    "dissociation": "mass_action",
    "dimerization": "mass_action",
    "complex_formation": "mass_action",
    "sequestration": "mass_action",
    # 转换类（mass-action 单分子）
    "cleavage": "mass_action",
    "gtp_gdp_exchange": "mass_action",
    # 基因表达类（Hill，转录因子饱和）
    "transcription": "Hill",
    "translation": "mass_action",
    # 转运类（mass-action）
    "nuclear_import": "mass_action",
    "nuclear_export": "mass_action",
    "cytoplasm_translocation": "mass_action",
    # 降解类（mass-action 一级）
    "degradation": "mass_action",
    "proteasomal_degradation": "mass_action",
    # 抽象调控类（hybrid，下游由 Specialist 决定）
    "inhibition": "hybrid",
    "activation": "hybrid",
}

# 机制类型 → 时间尺度映射（回应审计 §3.6 无多时间尺度）
MECHANISM_TIME_SCALE: dict[str, TimeScale] = {
    "phosphorylation": TimeScale.FAST,
    "dephosphorylation": TimeScale.FAST,
    "binding": TimeScale.FAST,
    "dissociation": TimeScale.FAST,
    "dimerization": TimeScale.FAST,
    "complex_formation": TimeScale.FAST,
    "sequestration": TimeScale.FAST,
    "cleavage": TimeScale.FAST,
    "gtp_gdp_exchange": TimeScale.FAST,
    "transcription": TimeScale.MEDIUM,
    "translation": TimeScale.MEDIUM,
    "ubiquitination": TimeScale.MEDIUM,
    "nuclear_import": TimeScale.MEDIUM,
    "nuclear_export": TimeScale.MEDIUM,
    "cytoplasm_translocation": TimeScale.MEDIUM,
    "degradation": TimeScale.SLOW,
    "proteasomal_degradation": TimeScale.SLOW,
    "inhibition": TimeScale.FAST,
    "activation": TimeScale.FAST,
}

# 跨通路共享 species 白名单（来自 v4 Scientific Architecture §3.1 shared species）
SHARED_SPECIES: dict[str, list[str]] = {
    "Ras": ["EGFR_RTK", "MAPK_ERK"],
    "AKT": ["PI3K_AKT_mTOR", "Apoptosis", "Cell_Cycle"],
    "MEK": ["MAPK_ERK", "EGFR_RTK"],
    "ERK": ["MAPK_ERK", "EGFR_RTK", "Cell_Cycle"],
    "p53": ["p53_signaling", "Apoptosis", "Cell_Cycle"],
    "NFkB": ["NF_kB", "Apoptosis", "Inflammation"],
    "SMAD": ["TGF_beta", "Cell_Cycle"],
    "STAT": ["JAK_STAT", "Apoptosis"],
    "GSK3B": ["Wnt", "PI3K_AKT_mTOR", "Apoptosis"],
    "Myc": ["MAPK_ERK", "PI3K_AKT_mTOR", "Cell_Cycle", "Apoptosis"],
    "Rb": ["Cell_Cycle", "Apoptosis"],
    "E2F": ["Cell_Cycle", "Apoptosis"],
    "mTOR": ["PI3K_AKT_mTOR", "Cell_Cycle"],
    "HIF1A": ["PI3K_AKT_mTOR", "Apoptosis"],
    "Caspase3": ["Apoptosis", "Cell_Cycle"],
}


class PathwayGraphBuilder:
    """Pathway Graph 构建器。

    纯规则构建，从 ontology + reaction_ir + pathway_registry 派生 PathwayGraph。
    不调用 LLM，不修改输入数据。

    使用方式::

        builder = PathwayGraphBuilder()
        graph = builder.build(
            pathway_class="EGFR_RTK",
            ontology_entities={...},  # P1 Ontology Agent 输出
            reaction_ir={...},         # P2 Reaction IR v2 输出
        )
        state["v4_pathway_graph"] = graph.to_dict()
    """

    def __init__(self) -> None:
        self.warnings: list[str] = []

    # -------------------------------------------------------------------------
    # 主构建入口
    # -------------------------------------------------------------------------
    def build(
        self,
        pathway_class: str,
        ontology_entities: dict[str, Any] | None = None,
        reaction_ir: dict[str, Any] | None = None,
        cross_talk_edges: list[dict[str, Any]] | None = None,
        feedback_loops: list[dict[str, Any]] | None = None,
    ) -> PathwayGraph:
        """构建 PathwayGraph。

        Args:
            pathway_class: 通路类别，如 "EGFR_RTK"
            ontology_entities: P1 Ontology Agent 输出（dict 形式）
            reaction_ir: P2 Reaction IR v2 输出（dict 形式）
            cross_talk_edges: 跨通路交叉点（可选，由 initializer 提供）
            feedback_loops: 反馈环（可选，由 initializer 提供）

        Returns:
            PathwayGraph 对象
        """
        self.warnings.clear()
        ontology_entities = ontology_entities or {}
        reaction_ir = reaction_ir or {}

        # 0. 构造 species_id → canonical_name 反查表（供 edges 解析 species_id 用）
        #    P2 ReactionIRv2.model_dump() 的 SpeciesRef 仅含 species_id，需用此表反查名称。
        id_to_name = self._build_species_id_index(reaction_ir)

        # 1. 从 reaction_ir 提取 nodes（species）+ edges（reactions）
        ir_nodes = self._extract_nodes_from_ir(reaction_ir, pathway_class)
        ir_edges = self._extract_edges_from_ir(reaction_ir, pathway_class, id_to_name)

        # 2. 用 ontology_entities 增强 nodes（HGNC/UniProt/ChEBI/GO/SBO + verified）
        enriched_nodes = self._enrich_nodes_with_ontology(ir_nodes, ontology_entities)

        # 3. 标记 shared species（跨通路共享）
        self._mark_shared_species(enriched_nodes)

        # 4. 构建 feedback_loops 与 cross_talk_edges
        fb_loops = self._build_feedback_loops(feedback_loops, pathway_class)
        ct_edges = self._build_cross_talk_edges(cross_talk_edges)

        # 5. 时间尺度标注
        temporal = self._build_temporal_annotation(
            pathway_class, enriched_nodes, ir_edges, fb_loops
        )

        graph = PathwayGraph(
            pathway_class=pathway_class,
            nodes=enriched_nodes,
            edges=ir_edges,
            feedback_loops=fb_loops,
            cross_talk_edges=ct_edges,
            temporal=temporal,
            source="v4_native",
            warnings=list(self.warnings),
        )

        logger.info(
            "PathwayGraph built: pathway=%s nodes=%d edges=%d feedback=%d crosstalk=%d",
            pathway_class, len(graph.nodes), len(graph.edges),
            len(graph.feedback_loops), len(graph.cross_talk_edges),
        )
        return graph

    # -------------------------------------------------------------------------
    # 从 Reaction IR v2 提取 nodes
    # -------------------------------------------------------------------------
    def _extract_nodes_from_ir(
        self,
        reaction_ir: dict[str, Any],
        pathway_class: str,
    ) -> list[PathwayNode]:
        """从 Reaction IR v2 的 species 列表提取 PathwayNode。

        Reaction IR v2 schema（P2）：
          {
            "species": [{"id", "name", "compartment", "initial_concentration", ...}],
            "reactions": [...]
          }
        """
        nodes: list[PathwayNode] = []
        species_list = reaction_ir.get("species", [])
        if not species_list:
            self.warnings.append("reaction_ir.species 为空，无法提取 nodes")
            return nodes

        seen_ids: set[str] = set()
        for sp in species_list:
            sp_id = sp.get("id") or sp.get("canonical_name") or sp.get("name", "")
            if not sp_id or sp_id in seen_ids:
                continue
            seen_ids.add(sp_id)

            # P2 schema 优先用 canonical_name（架构 §4.2.1），fallback 到 name/id
            name = sp.get("canonical_name") or sp.get("name") or sp_id
            compartment = sp.get("compartment", "cytoplasm")
            species_type = sp.get("species_type", "protein")

            # 状态提取（P2 SpeciesV2.states，避免压扁为多个 species）
            states: list[PathwayState] = []
            for st in sp.get("states", []):
                if isinstance(st, dict):
                    states.append(PathwayState(
                        name=st.get("name", "active"),
                        state_type=st.get("state_type", "phosphorylation"),
                        is_initial=st.get("is_initial", False),
                        site=st.get("site"),
                    ))

            node = PathwayNode(
                id=f"PN_{name}",
                canonical_name=name,
                display_name=sp.get("display_name", name),
                species_type=species_type,
                pathway_class=pathway_class,
                module=PathwayModule.CORE,
                compartment=compartment,
                time_scale=TimeScale.FAST,
                states=states,
                source_sbml=sp.get("source_sbml"),
                source_pmid=sp.get("source_pmid"),
            )
            nodes.append(node)

        return nodes

    # -------------------------------------------------------------------------
    # 从 Reaction IR v2 提取 edges
    # -------------------------------------------------------------------------
    def _extract_edges_from_ir(
        self,
        reaction_ir: dict[str, Any],
        pathway_class: str,
        id_to_name: dict[str, str] | None = None,
    ) -> list[PathwayEdge]:
        """从 Reaction IR v2 的 reactions 列表提取 PathwayEdge。

        Reaction IR v2 schema（P2）：
          {
            "reactions": [{
              "id", "reaction_type", "reactants": [{"species_id", "stoichiometry", "role"}],
              "products": [{"species_id", ...}],
              "modifiers": [{"species_id", "modifier_type", "site"}],
              "provenance": {"source_sbml_reaction", "source_pmid", "source_kegg"}
            }]
          }

        字段路径适配：
        - SpeciesRef 用 species_id，需通过 id_to_name 反查为 canonical_name
        - provenance 嵌套读取（source_sbml_reaction / source_pmid / source_kegg）
        - sbo_term 从 reaction_type + MECHANISM_SBO_MAP 推断（P2 ReactionV2 无 sbo_term 字段）
        """
        edges: list[PathwayEdge] = []
        reactions = reaction_ir.get("reactions", [])
        if not reactions:
            self.warnings.append("reaction_ir.reactions 为空，无法提取 edges")
            return edges
        id_to_name = id_to_name or {}

        for i, rxn in enumerate(reactions):
            rxn_id = rxn.get("id") or f"R{i}"
            reaction_type = rxn.get("reaction_type") or rxn.get("mechanism", "activation")

            # source / target 提取（兼容 P2 schema 的多种字段名）
            source_name = self._get_first_reactant(rxn, id_to_name)
            target_name = self._get_first_product(rxn, id_to_name)
            if not source_name or not target_name:
                self.warnings.append(
                    f"reaction '{rxn_id}' 无法解析 source/target，已跳过"
                )
                continue

            kinetics = MECHANISM_DEFAULT_KINETICS.get(reaction_type, "mass_action")
            time_scale = MECHANISM_TIME_SCALE.get(reaction_type, TimeScale.FAST)

            # provenance 嵌套字段路径适配（P2 Provenance 嵌套在 reaction 下）
            provenance = rxn.get("provenance") or {}
            source_sbml_reaction = (
                provenance.get("source_sbml_reaction")
                if isinstance(provenance, dict)
                else None
            ) or rxn.get("source_sbml_reaction")
            source_pmid = (
                provenance.get("source_pmid")
                if isinstance(provenance, dict)
                else None
            ) or rxn.get("source_pmid")
            source_kegg = (
                provenance.get("source_kegg")
                if isinstance(provenance, dict)
                else None
            ) or rxn.get("source_kegg")

            # sbo_term：P2 ReactionV2 无此字段，从 P2 MechanismType 反查
            sbo_term = rxn.get("sbo_term") or _lookup_sbo_term(reaction_type)

            edge = PathwayEdge(
                id=f"PE_{rxn_id}",
                source=f"PN_{source_name}",
                target=f"PN_{target_name}",
                mechanism=reaction_type,
                pathway_tag=pathway_class,
                module=PathwayModule.CORE,
                time_scale=time_scale,
                sbo_term=sbo_term,
                kinetics_type=kinetics,
                source_sbml_reaction=source_sbml_reaction,
                source_pmid=source_pmid,
                source_kegg=source_kegg,
                site=rxn.get("site"),
            )
            edges.append(edge)

        return edges

    # -------------------------------------------------------------------------
    # 构造 species_id → canonical_name 反查表
    # -------------------------------------------------------------------------
    @staticmethod
    def _build_species_id_index(reaction_ir: dict[str, Any]) -> dict[str, str]:
        """从 reaction_ir.species 构造 id → canonical_name 反查表。

        P2 SpeciesRef 仅含 species_id（无 name 字段），edges 解析时需此表反查名称。
        """
        index: dict[str, str] = {}
        for sp in reaction_ir.get("species", []) or []:
            sp_id = sp.get("id")
            if not sp_id:
                continue
            name = sp.get("canonical_name") or sp.get("name") or sp_id
            index[sp_id] = name
        return index

    @staticmethod
    def _get_first_reactant(rxn: dict[str, Any], id_to_name: dict[str, str] | None = None) -> str | None:
        """从 reaction 提取第一个 reactant 名（兼容多种 schema，含 P2 SpeciesRef）。

        返回 None 的场景：SpeciesRef 的 species_id 不在反查表且 dict 中无 name/species/id 字段。
        这种情况视为未引用的 species（dangling reference），由调用方触发 warning 并跳过。
        """
        id_to_name = id_to_name or {}
        for key in ("source", "reactants", "substrates", "left"):
            val = rxn.get(key)
            if isinstance(val, str):
                return val
            if isinstance(val, list) and val:
                first = val[0]
                if isinstance(first, str):
                    return first
                if isinstance(first, dict):
                    # P2 SpeciesRef 优先用 species_id → id_to_name 反查
                    species_id = first.get("species_id")
                    if species_id and id_to_name:
                        mapped = id_to_name.get(species_id)
                        if mapped:
                            return mapped
                        # species_id 存在但反查表无此 id → 视为未引用 species，返回 None
                        if "species_id" in first:
                            return None
                    return (
                        first.get("name")
                        or first.get("species")
                        or first.get("id")
                    )
        return None

    @staticmethod
    def _get_first_product(rxn: dict[str, Any], id_to_name: dict[str, str] | None = None) -> str | None:
        """从 reaction 提取第一个 product 名（兼容多种 schema，含 P2 SpeciesRef）。"""
        id_to_name = id_to_name or {}
        for key in ("target", "products", "right"):
            val = rxn.get(key)
            if isinstance(val, str):
                return val
            if isinstance(val, list) and val:
                first = val[0]
                if isinstance(first, str):
                    return first
                if isinstance(first, dict):
                    species_id = first.get("species_id")
                    if species_id and id_to_name:
                        mapped = id_to_name.get(species_id)
                        if mapped:
                            return mapped
                        if "species_id" in first:
                            return None
                    return (
                        first.get("name")
                        or first.get("species")
                        or first.get("id")
                    )
        return None

    # -------------------------------------------------------------------------
    # 用 Ontology 增强 nodes
    # -------------------------------------------------------------------------
    def _enrich_nodes_with_ontology(
        self,
        nodes: list[PathwayNode],
        ontology_entities: dict[str, Any],
    ) -> list[PathwayNode]:
        """用 P1 Ontology Agent 输出增强 PathwayNode 的 ontology 引用。

        ontology_entities schema（P1）：
          {
            "EGFR": {
              "hgnc_id": "HGNC:3236",
              "uniprot_id": "P00533",
              "chebi_id": None,
              "go_terms": ["GO:0007169"],
              "sbo_term": "SBO:0000250",
              "verified": True,
              "compartment": "membrane",
              ...
            },
            ...
          }
        """
        if not ontology_entities:
            self.warnings.append("ontology_entities 为空，nodes 缺失 ontology 引用")
            return nodes

        for node in nodes:
            ent = ontology_entities.get(node.canonical_name)
            if not ent:
                # 尝试大小写不敏感匹配
                ent = next(
                    (v for k, v in ontology_entities.items()
                     if k.lower() == node.canonical_name.lower()),
                    None,
                )
            if not ent:
                self.warnings.append(
                    f"node '{node.canonical_name}' 未在 ontology_entities 中找到，"
                    f"ontology_verified=False"
                )
                continue

            node.hgnc_id = ent.get("hgnc_id")
            node.uniprot_id = ent.get("uniprot_id")
            node.chebi_id = ent.get("chebi_id")
            node.go_terms = ent.get("go_terms", []) or []
            node.sbo_term = ent.get("sbo_term")
            node.ontology_verified = bool(ent.get("verified", False))
            # ontology 提供的 compartment 优先
            if ent.get("compartment"):
                node.compartment = ent["compartment"]

        return nodes

    # -------------------------------------------------------------------------
    # 标记 shared species
    # -------------------------------------------------------------------------
    def _mark_shared_species(self, nodes: list[PathwayNode]) -> None:
        """标记跨通路共享 species（如 Ras/AKT/MEK）。"""
        for node in nodes:
            shared_with = SHARED_SPECIES.get(node.canonical_name)
            if shared_with:
                node.is_shared = True
                node.shared_with = list(shared_with)

    # -------------------------------------------------------------------------
    # 构建 feedback loops
    # -------------------------------------------------------------------------
    def _build_feedback_loops(
        self,
        feedback_loops: list[dict[str, Any]] | None,
        pathway_class: str,
    ) -> list[FeedbackLoop]:
        """从 initializer 提供的 feedback_loops 构建 FeedbackLoop 对象。"""
        if not feedback_loops:
            return []

        result: list[FeedbackLoop] = []
        for fl in feedback_loops:
            if not isinstance(fl, dict):
                continue
            try:
                result.append(FeedbackLoop(
                    id=fl.get("id", f"FL_{pathway_class}_{len(result)}"),
                    loop_type=fl.get("loop_type", "negative"),
                    pathway_class=fl.get("pathway_class", pathway_class),
                    edge_ids=fl.get("edge_ids", []),
                    node_ids=fl.get("node_ids", []),
                    delay_minutes=fl.get("delay_minutes", 0.0),
                    source_pmid=fl.get("source_pmid"),
                    description=fl.get("description", ""),
                ))
            except Exception as e:
                self.warnings.append(f"feedback_loop 构建失败: {e}")
        return result

    # -------------------------------------------------------------------------
    # 构建 cross-talk edges
    # -------------------------------------------------------------------------
    def _build_cross_talk_edges(
        self,
        cross_talk_edges: list[dict[str, Any]] | None,
    ) -> list[CrossTalkEdge]:
        """从 initializer 提供的 cross_talk_edges 构建 CrossTalkEdge 对象。"""
        if not cross_talk_edges:
            return []

        result: list[CrossTalkEdge] = []
        for ct in cross_talk_edges:
            if not isinstance(ct, dict):
                continue
            try:
                result.append(CrossTalkEdge(
                    id=ct.get("id", f"CT_{len(result)}"),
                    source_pathway=ct.get("source_pathway", ""),
                    target_pathway=ct.get("target_pathway", ""),
                    source_node=ct.get("source_node", ""),
                    target_node=ct.get("target_node", ""),
                    mechanism=ct.get("mechanism", "inhibition"),
                    shared_species=ct.get("shared_species", []),
                    site=ct.get("site"),
                    sbo_term=ct.get("sbo_term"),
                    source_pmid=ct.get("source_pmid"),
                    description=ct.get("description", ""),
                ))
            except Exception as e:
                self.warnings.append(f"cross_talk_edge 构建失败: {e}")
        return result

    # -------------------------------------------------------------------------
    # 时间尺度标注
    # -------------------------------------------------------------------------
    def _build_temporal_annotation(
        self,
        pathway_class: str,
        nodes: list[PathwayNode],
        edges: list[PathwayEdge],
        feedback_loops: list[FeedbackLoop] | None = None,
    ) -> TemporalAnnotation:
        """根据通路主导过程构建时间尺度标注。

        规则：
          - 含正延迟反馈 → requires_dde=True，延迟取反馈环声明值
          - 含细胞周期（slow species）→ t_end=2880 min (48h)
          - 默认 fast signaling → t_end=60 min, max_step=0.1 min
        """
        # 检测是否含 slow species（degradation / cell cycle）
        has_slow = any(e.time_scale == TimeScale.SLOW for e in edges)
        # 检测是否含 transcription（MEDIUM，可能需要 DDE）
        has_transcription = any(e.mechanism == "transcription" for e in edges)
        positive_delays = [
            float(loop.delay_minutes)
            for loop in (feedback_loops or [])
            if float(loop.delay_minutes or 0.0) > 0.0
        ]
        # 保留历史通路默认，同时让 Wnt/其他 specialist 声明的延迟真正进入运行时。
        dde_pathways = {"p53_signaling", "NF_kB", "TGF_beta"}
        requires_dde = bool(positive_delays) or pathway_class in dde_pathways

        if has_slow:
            primary_scale = TimeScale.SLOW
            t_end = 2880.0  # 48h
            max_step = 1.0
        elif has_transcription:
            primary_scale = TimeScale.MEDIUM
            t_end = 360.0  # 6h
            max_step = 0.5
        else:
            primary_scale = TimeScale.FAST
            t_end = 60.0
            max_step = 0.1

        # Specialist feedback metadata is authoritative; legacy pathways retain
        # the 60-minute default when no explicit delay was supplied.
        dde_delay = max(positive_delays) if positive_delays else (60.0 if requires_dde else 0.0)

        return TemporalAnnotation(
            pathway_class=pathway_class,
            primary_scale=primary_scale,
            max_step_minutes=max_step,
            t_end_minutes=t_end,
            requires_dde=requires_dde,
            dde_delay_minutes=dde_delay,
        )


__all__ = ["PathwayGraphBuilder", "MECHANISM_DEFAULT_KINETICS", "MECHANISM_TIME_SCALE"]
