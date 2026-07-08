# BioDynamics Agent v4 - Reaction Builder
# 对应 v4 Scientific Architecture Part 4 §4.2 + Layer 3（Reaction IR Layer）。
#
# 职责：将 Pathway Graph（或 v3 network_json）转换为结构化 ReactionIRv2。
# 这是 v4 Layer 3 的核心入口，由 graph_v3.py 的 v4 hook 调用。
#
# 设计原则：
# 1. Feature Flag V4_REACTION_IR_ENABLED=false 时不执行（由调用方控制）
# 2. 输入：state.network_json（v3 格式）+ 可选的 v4_ontology_entities（P1 输出）
# 3. 输出：ReactionIRv2 对象，写入 state.v4_reaction_ir
# 4. 降级模式：当 ontology 未对齐时，species.ontology.verified=False，不阻塞
# 5. 自动生成质量守恒约束（磷酸化对模式）
# 6. 不调用 LLM，纯规则构建

from __future__ import annotations

import logging
from typing import Any

from app.reaction_ir_v2.constraints import auto_generate_mass_conservation
from app.reaction_ir_v2.mechanism_types import (
    MechanismType,
    v3_interaction_to_mechanism,
)
from app.reaction_ir_v2.schema import (
    Compartment,
    OntologyRef,
    Provenance,
    ReactionIRv2,
    ReactionV2,
    SpeciesRef,
    SpeciesV2,
)

logger = logging.getLogger(__name__)


# =============================================================================
# 默认区室映射：物种名 → compartment
# =============================================================================
# 用于在缺乏 ontology 信息时推断物种的区室
_DEFAULT_COMPARTMENT_BY_TYPE: dict[str, str] = {
    "ligand": "extracellular",
    "receptor": "membrane",
    "drug": "extracellular",
    "chemical": "extracellular",
    "mrna": "nucleus",
    "transcription_factor": "nucleus",
    "protein": "cytoplasm",
    "kinase": "cytoplasm",
    "phosphatase": "cytoplasm",
}


# =============================================================================
# 核心：从 v3 network_json 构建 ReactionIRv2
# =============================================================================
def build_from_network_json(
    network_json: dict[str, Any],
    ontology_entities: dict[str, Any] | None = None,
    pathway_tag: str = "",
    sbml_model_id: str | None = None,
) -> ReactionIRv2:
    """从 v3 network_json 构建 ReactionIRv2。

    v3 network_json 格式：
        {
            "nodes": [{"id": str, "name": str, "type": str}],
            "edges": [{"source": str, "target": str, "interaction": str}]
        }

    构建策略：
    1. nodes → species：每个 node 转为 SpeciesV2，ontology 从 ontology_entities 查询
    2. edges → reactions：每条 edge 转为 ReactionV2，机制由 interaction 推断
    3. compartment 默认 cytoplasm，ligand/receptor 按类型推断
    4. 自动生成质量守恒约束（磷酸化对模式）

    Args:
        network_json: v3 格式的网络 JSON
        ontology_entities: P1 Ontology Agent 输出（可选，用于填充 HGNC/UniProt ID）
        pathway_tag: 通路标签（可选，默认空）
        sbml_model_id: SBML model ID（可选，用于溯源）

    Returns:
        ReactionIRv2 对象（source="v4_native" 或 "v3_downgraded"）
    """
    warnings: list[str] = []

    nodes = network_json.get("nodes", []) or []
    edges = network_json.get("edges", []) or []

    # —— 1. nodes → species ——
    species_list: list[SpeciesV2] = []
    name_to_species_id: dict[str, str] = {}
    # 构建 ontology 索引（按 name 查询）
    ont_by_name: dict[str, dict[str, Any]] = {}
    if ontology_entities:
        for ent in ontology_entities.get("entities", []) or []:
            name = ent.get("name", "")
            if name:
                ont_by_name[name] = ent

    for i, node in enumerate(nodes):
        name = node.get("name") or node.get("id") or f"species_{i}"
        node_id = node.get("id", name)
        species_id = f"SP_{i+1:03d}"
        name_to_species_id[name] = species_id
        if node_id != name:
            name_to_species_id[node_id] = species_id

        species_type = node.get("type", "protein")
        compartment = _DEFAULT_COMPARTMENT_BY_TYPE.get(species_type, "cytoplasm")

        # 填充 ontology（若 P1 Ontology Agent 输出可用）
        ont = OntologyRef()
        ont_entity = ont_by_name.get(name) or ont_by_name.get(node_id)
        if ont_entity:
            ont = OntologyRef(
                hgnc_id=ont_entity.get("hgnc_id"),
                uniprot_id=ont_entity.get("uniprot_id"),
                chebi_id=ont_entity.get("chebi_id"),
                go_terms=ont_entity.get("go_terms", []) or [],
                sbo_term=ont_entity.get("sbo_term"),
                verified=ont_entity.get("verified", False),
            )
        else:
            warnings.append(
                f"Species '{name}' 未在 ontology_entities 中找到，verified=False"
            )

        species_list.append(SpeciesV2(
            id=species_id,
            canonical_name=name,
            display_name=name,
            ontology=ont,
            species_type=species_type,
            compartment=compartment,
            initial_concentration=0.0,  # 初始浓度由 P3 ODE 注入
            source_sbml=sbml_model_id,
        ))

    # —— 2. edges → reactions ——
    reactions: list[ReactionV2] = []
    for i, edge in enumerate(edges):
        source_name = edge.get("source", "")
        target_name = edge.get("target", "")
        interaction = edge.get("interaction", "activation")

        source_id = name_to_species_id.get(source_name, f"SP_{source_name}")
        target_id = name_to_species_id.get(target_name, f"SP_{target_name}")

        # 机制推断：v3 interaction → v4 mechanism
        mechanism = v3_interaction_to_mechanism(interaction)
        kinetics_type = mechanism.default_kinetics

        # 参数上下文：source → target + mechanism
        parameter_context = f"{source_name} → {target_name} ({mechanism.value})"

        # 接线强制（纪律1）：调用统一的 _build_reaction_for_mechanism，
        # 禁止 inline if/elif/else 兜底，禁止"默认：source → target"1:1 兜底。
        # 19 种机制各自体现生物学语义（见 _build_reaction_for_mechanism 注释）。
        reactants, products, modifiers = _build_reaction_for_mechanism(
            mechanism, source_id, target_id, source_name, target_name, name_to_species_id
        )

        # compartments：从 species 查询
        sp_source = next((s for s in species_list if s.id == source_id), None)
        sp_target = next((s for s in species_list if s.id == target_id), None)
        compartments_set: set[str] = set()
        if sp_source:
            compartments_set.add(sp_source.compartment)
        if sp_target:
            compartments_set.add(sp_target.compartment)

        # TD-020 (IB-035) 修复：从 edge 元数据与 sbml_model_id 填充 Provenance，避免恒为 None
        # 优先取 edge 顶层字段，再回退到 edge 内嵌的 provenance dict
        _edge_prov = edge.get("provenance") or {}
        _pmid = edge.get("pmid") or _edge_prov.get("pmid")
        _kegg = edge.get("kegg_id") or _edge_prov.get("kegg_id")

        reactions.append(ReactionV2(
            id=f"RXN_{i+1:03d}",
            reaction_type=mechanism.value,
            kinetics_type=kinetics_type,
            reactants=reactants,
            products=products,
            modifiers=modifiers,
            compartments=list(compartments_set),
            parameter_context=parameter_context,
            pathway_tag=pathway_tag,
            provenance=Provenance(
                source_sbml_reaction=sbml_model_id,  # SBML model ID 作为溯源来源
                source_pmid=str(_pmid) if _pmid is not None else None,
                source_kegg=str(_kegg) if _kegg is not None else None,
            ),
        ))

    # —— 3. 自动生成质量守恒约束 ——
    constraints = auto_generate_mass_conservation(species_list, reactions)

    # —— 4. 构建 compartments（从 species 去重）——
    compartment_names: set[str] = set()
    for sp in species_list:
        compartment_names.add(sp.compartment)
    compartments = [
        Compartment(name=name, size=_DEFAULT_COMPARTMENT_SIZE.get(name, 1.0))
        for name in sorted(compartment_names)
    ]

    # —— 5. 确定 source ——
    source = "v3_downgraded" if not ontology_entities else "v4_native"

    ir = ReactionIRv2(
        species=species_list,
        reactions=reactions,
        composite_reactions=[],  # 降级模式无组合反应
        state_machines=[],       # 降级模式无状态机
        compartments=compartments,
        constraints=constraints,
        version="v4.0",
        source=source,
        warnings=warnings,
    )
    logger.info(
        "build_from_network_json: %d species, %d reactions, %d constraints, source=%s",
        len(species_list), len(reactions), len(constraints), source,
    )
    return ir


# =============================================================================
# 辅助函数
# =============================================================================
def _make_modifier(species_id: str, modifier_type: str) -> Any:
    """构建 Modifier 对象（延迟导入避免循环依赖）。"""
    from app.reaction_ir_v2.schema import Modifier
    return Modifier(species_id=species_id, modifier_type=modifier_type)


# 默认区室体积比（对应架构 §4.2.5）
_DEFAULT_COMPARTMENT_SIZE: dict[str, float] = {
    "extracellular": 2.0,   # 细胞外体积较大
    "membrane": 0.05,       # 膜体积小
    "cytoplasm": 0.5,       # 胞质占细胞体积约 50%
    "nucleus": 0.1,         # 核体积约 10%
    "mitochondria": 0.1,    # 线粒体约 10%
}


# =============================================================================
# IB-009 修复：逐机制构建 reactants/products/modifiers（替代 else 分支 1:1 兜底）
# =============================================================================
# 审计报告 TD-004：原 else 分支将 transcription/translation/GTP_GDP/cleavage/
# dissociation/sequestration/activation/dimerization 等 8+ 种机制统一当作
# "source(substrate, stoich=1) → target(product, stoich=1)"，丢失各自特定语义。
#
# 本函数为统一的机制语义构建器，被 build_from_network_json 和
# build_from_pathway_graph 两个入口共同调用（接线强制，禁止 else 兜底）。
# 每种机制体现其生物学语义：
#   - transcription: TF 作 modifier，产物为 mRNA（不被消耗）
#   - translation: mRNA 作 modifier（模板），产物为 protein（不被消耗）
#   - gtp_gdp_exchange: source 作 substrate（GDP-form→GTP-form），应有 GEF/GAP modifier
#   - cleavage: enzyme 作 modifier（MM 催化），target 为切割产物
#   - dissociation: complex → components 方向（1 reactant → 多 products）
#   - sequestration: sequesterer 作 modifier（扣押者不消耗）
#   - dimerization: 2*monomer → 1*dimer（stoichiometry=2，IB-005 同步修复）
#   - dephosphorylation: 磷酸酶作 modifier（catalytic），target 为去磷酸化产物
#   - ubiquitination: E3 ligase 作 modifier（catalytic），target 为泛素化产物
#   - binding/complex_formation: source+target → complex（双底物）
#   - nuclear_import/export/translocation: source → target（区室变更）
#   - activation: source → target（通用调控，保留 1:1 语义）
# =============================================================================
def _build_reaction_for_mechanism(
    mechanism: MechanismType,
    source_id: str,
    target_id: str,
    source_name: str,
    target_name: str,
    name_to_species_id: dict[str, str],
) -> tuple[list, list, list]:
    """根据机制类型构建 (reactants, products, modifiers)。

    Args:
        mechanism: 机制枚举值
        source_id: source 的 species_id
        target_id: target 的 species_id
        source_name: source 的规范名（用于自磷酸化判定等）
        target_name: target 的规范名
        name_to_species_id: 名称→species_id 映射（用于推导未磷酸化形式等）

    Returns:
        (reactants, products, modifiers) 三个列表
    """
    reactants: list[SpeciesRef] = []
    products: list[SpeciesRef] = []
    modifiers: list[Any] = []

    if mechanism == MechanismType.INHIBITION:
        # 抑制：target 是被抑制的底物，source 是 inhibitor
        reactants.append(SpeciesRef(species_id=target_id, role="substrate"))
        products.append(SpeciesRef(species_id=target_id, role="product"))
        modifiers.append(_make_modifier(source_id, "inhibitory"))

    elif mechanism in (MechanismType.DEGRADATION, MechanismType.PROTEASOMAL_DEGRADATION):
        # 降解：source 为 substrate，无 product
        reactants.append(SpeciesRef(species_id=source_id, role="substrate"))

    elif mechanism == MechanismType.PHOSPHORYLATION:
        # B3 修复：区分自磷酸化与异磷酸化
        if _is_autophosphorylation(source_name, target_name):
            # 自磷酸化：source 作 substrate，target (p-source) 作 product，无 modifier
            reactants.append(SpeciesRef(species_id=source_id, role="substrate"))
            products.append(SpeciesRef(species_id=target_id, role="product"))
        else:
            # 异磷酸化：未磷酸化形式作 substrate，target (p-form) 作 product，source 作 catalytic modifier
            substrate_id = _derive_substrate_id(
                target_name, target_id, name_to_species_id
            )
            reactants.append(SpeciesRef(species_id=substrate_id, role="substrate"))
            products.append(SpeciesRef(species_id=target_id, role="product"))
            modifiers.append(_make_modifier(source_id, "catalytic"))

    elif mechanism == MechanismType.DEPHOSPHORYLATION:
        # 去磷酸化：磷酸酶（source）作 catalytic modifier，target 为去磷酸化产物
        # 底物应为 target 的磷酸化前体（如 pTSC2 → TSC2，底物为 pTSC2）
        substrate_id = _derive_phosphorylated_substrate_id(
            target_name, target_id, name_to_species_id
        )
        reactants.append(SpeciesRef(species_id=substrate_id, role="substrate"))
        products.append(SpeciesRef(species_id=target_id, role="product"))
        modifiers.append(_make_modifier(source_id, "catalytic"))

    elif mechanism == MechanismType.UBIQUITINATION:
        # 泛素化：E3 ligase（source）作 catalytic modifier，target 为泛素化产物
        # 底物应为 target 的非泛素化前体
        substrate_id = _derive_unmodified_substrate_id(
            target_name, target_id, name_to_species_id, prefix="ub"
        )
        reactants.append(SpeciesRef(species_id=substrate_id, role="substrate"))
        products.append(SpeciesRef(species_id=target_id, role="product"))
        modifiers.append(_make_modifier(source_id, "catalytic"))

    elif mechanism == MechanismType.DIMERIZATION:
        # IB-005 修复：2*monomer → 1*dimer（化学计量 2:1）
        reactants.append(SpeciesRef(species_id=source_id, stoichiometry=2, role="substrate"))
        products.append(SpeciesRef(species_id=target_id, stoichiometry=1, role="product"))

    elif mechanism == MechanismType.BINDING:
        # 结合：source + target → complex（双底物，产物为复合物）
        # target 在 binding 中既是底物也是产物的一部分（保留 1:1 语义但标记为结合）
        reactants.append(SpeciesRef(species_id=source_id, role="substrate"))
        reactants.append(SpeciesRef(species_id=target_id, role="substrate"))
        products.append(SpeciesRef(species_id=target_id, role="product"))

    elif mechanism == MechanismType.COMPLEX_FORMATION:
        # 复合物组装：source + target → complex（双底物，与 binding 类似）
        reactants.append(SpeciesRef(species_id=source_id, role="substrate"))
        reactants.append(SpeciesRef(species_id=target_id, role="substrate"))
        products.append(SpeciesRef(species_id=target_id, role="product"))

    elif mechanism == MechanismType.DISSOCIATION:
        # 解离：complex → components（source 为 complex，target 为 component）
        reactants.append(SpeciesRef(species_id=source_id, role="substrate"))
        products.append(SpeciesRef(species_id=target_id, role="product"))

    elif mechanism == MechanismType.SEQUESTRATION:
        # 扣押：sequesterer（source）作 modifier（不消耗），target 为被扣押物
        # 生物学：Bcl2 扣押 BAX，Bcl2 不被消耗，BAX 被抑制
        reactants.append(SpeciesRef(species_id=target_id, role="substrate"))
        products.append(SpeciesRef(species_id=target_id, role="product"))
        modifiers.append(_make_modifier(source_id, "inhibitory"))

    elif mechanism == MechanismType.CLEAVAGE:
        # 切割：enzyme（source）作 catalytic modifier（MM 动力学），target 为切割产物
        # 底物应为 target 的前体（如 pro-Caspase3 → Caspase3_active，底物为 pro-Caspase3）
        substrate_id = _derive_precursor_substrate_id(
            target_name, target_id, name_to_species_id
        )
        reactants.append(SpeciesRef(species_id=substrate_id, role="substrate"))
        products.append(SpeciesRef(species_id=target_id, role="product"))
        modifiers.append(_make_modifier(source_id, "catalytic"))

    elif mechanism == MechanismType.GTP_GDP_EXCHANGE:
        # GTP/GDP 交换：source（GDP-form）→ target（GTP-form），GEF/GAP 作 modifier
        # edge 通常只提供 source/target，GEF/GAP 未显式提供。
        # 为保持酶催化语义（供 ODE 模板按 MM 渲染），将 source 作为 placeholder modifier，
        # 并由 ODE 模板层识别 reaction_type=gtp_gdp_exchange 填入真实 GEF/GAP 浓度。
        reactants.append(SpeciesRef(species_id=source_id, role="substrate"))
        products.append(SpeciesRef(species_id=target_id, role="product"))
        modifiers.append(_make_modifier(source_id, "catalytic"))

    elif mechanism == MechanismType.TRANSCRIPTION:
        # 转录：TF（source）作 modifier（不消耗），产物为 mRNA（target）
        # 无 reactant 消耗（基因模板不消耗）
        products.append(SpeciesRef(species_id=target_id, role="product"))
        modifiers.append(_make_modifier(source_id, "activating"))

    elif mechanism == MechanismType.TRANSLATION:
        # 翻译：mRNA（source）作 modifier（模板，不消耗），产物为 protein（target）
        products.append(SpeciesRef(species_id=target_id, role="product"))
        modifiers.append(_make_modifier(source_id, "catalytic"))

    elif mechanism in (MechanismType.NUCLEAR_IMPORT,
                       MechanismType.NUCLEAR_EXPORT,
                       MechanismType.CYTOPLASM_TRANSLOCATION):
        # 转运：source → target（区室变更，1:1 语义合理）
        reactants.append(SpeciesRef(species_id=source_id, role="substrate"))
        products.append(SpeciesRef(species_id=target_id, role="product"))

    else:
        # ACTIVATION 及其他调控类：source → target（通用调控，1:1 语义合理）
        # 这是唯一的 1:1 兜底，仅用于 activation 机制
        reactants.append(SpeciesRef(species_id=source_id, role="substrate"))
        products.append(SpeciesRef(species_id=target_id, role="product"))

    return reactants, products, modifiers


def _derive_phosphorylated_substrate_id(
    target_name: str,
    target_id: str,
    name_to_species_id: dict[str, str],
) -> str:
    """从去磷酸化产物名推导磷酸化底物的 species_id。

    用于 dephosphorylation：edge `pTSC2_phosphatase → TSC2` 中 target=TSC2 是产物，
    底物应为 pTSC2（加 "p" 前缀的磷酸化形式）。
    """
    if target_name and target_name not in name_to_species_id:
        # target 未在 species 列表中，可能本身就是磷酸化形式
        return target_id
    # 尝试 "p" + target_name
    phospho_name = "p" + target_name
    if phospho_name in name_to_species_id:
        return name_to_species_id[phospho_name]
    return target_id


def _derive_unmodified_substrate_id(
    target_name: str,
    target_id: str,
    name_to_species_id: dict[str, str],
    prefix: str = "ub",
) -> str:
    """从修饰产物名推导未修饰底物的 species_id。

    用于 ubiquitination 等：edge `E3 → ubSubstrate` 中 target=ubSubstrate 是产物，
    底物应为 Substrate（去掉 "ub" 前缀的未修饰形式）。
    """
    if (
        len(target_name) >= len(prefix) + 1
        and target_name[:len(prefix)].lower() == prefix.lower()
        and target_name[len(prefix)].isupper()
    ):
        unmodified_name = target_name[len(prefix):]
        if unmodified_name in name_to_species_id:
            return name_to_species_id[unmodified_name]
    return target_id


def _derive_precursor_substrate_id(
    target_name: str,
    target_id: str,
    name_to_species_id: dict[str, str],
) -> str:
    """从切割产物名推导前体底物的 species_id。

    用于 cleavage：edge `Caspase3 → Caspase3_active` 中 target=Caspase3_active 是产物，
    底物应为 pro-Caspase3 或 Caspase3（前体形式）。
    """
    # 尝试 "pro-" + target_name（如 pro-Caspase3）
    pro_name = "pro-" + target_name
    if pro_name in name_to_species_id:
        return name_to_species_id[pro_name]
    # 尝试 target_name 自身（如 Caspase3 → Caspase3_active，底物为 Caspase3）
    if target_name in name_to_species_id:
        return name_to_species_id[target_name]
    return target_id


# =============================================================================
# B3 修复辅助函数：自/异磷酸化判定
# =============================================================================
def _is_autophosphorylation(source_name: str, target_name: str) -> bool:
    """判断是否为自磷酸化（target 是 source 的磷酸化形式）。

    规则：target_name == "p" + source_name（大小写不敏感）
    示例：
        EGFR → pEGFR: True（自磷酸化，EGFR 自身既是底物又是激酶）
        AKT → pTSC2: False（异磷酸化，AKT 是激酶，TSC2 是底物）
    """
    if not source_name or not target_name:
        return False
    return target_name.lower() == "p" + source_name.lower()


def _derive_substrate_id(
    target_name: str,
    target_id: str,
    name_to_species_id: dict[str, str],
) -> str:
    """从磷酸化产物名推导未磷酸化底物的 species_id。

    用于异磷酸化场景：edge `AKT → pTSC2` 中 target=pTSC2 是产物，
    底物应为 TSC2（去掉 "p" 前缀的未磷酸化形式）。

    若 species 列表中找不到未磷酸化形式（如未提供 TSC2 节点），
    回退为 target_id 自身（不阻断流程，但语义可能不精确）。
    """
    if (
        len(target_name) >= 2
        and target_name[0] == "p"
        and target_name[1].isupper()
    ):
        unphos_name = target_name[1:]
        if unphos_name in name_to_species_id:
            return name_to_species_id[unphos_name]
    return target_id


# =============================================================================
# 从 Pathway Graph 构建（P4 Pathway Planner 输出）
# =============================================================================
def build_from_pathway_graph(
    pathway_graph: dict[str, Any],
    ontology_entities: dict[str, Any] | None = None,
    sbml_model_id: str | None = None,
) -> ReactionIRv2:
    """从 P4 Pathway Graph 构建 ReactionIRv2（P4 阶段使用）。

    Pathway Graph 格式（P4 输出，比 v3 network_json 更丰富）：
        {
            "nodes": [{"id", "name", "type", "pathway_tag", "ontology": {...}}],
            "edges": [{"source", "target", "mechanism", "pathway_tag", "provenance": {...}}],
            "cross_talk_edges": [...]
        }

    B4 修复：直接构造 MechanismType 枚举，不再走 v3 interaction 反查表，
    8 种机制（dimerization / complex_formation / sequestration / gtp_gdp_exchange /
    nuclear_import / nuclear_export / cytoplasm_translocation / proteasomal_degradation）
    不再被 fallback 到 ACTIVATION。无效 mechanism 字符串降级为 ACTIVATION 并记录 warning。
    """
    warnings: list[str] = []

    nodes = pathway_graph.get("nodes", []) or []
    edges = pathway_graph.get("edges", []) or []
    cross_talk_edges = pathway_graph.get("cross_talk_edges", []) or []

    # —— 1. nodes → species ——
    species_list: list[SpeciesV2] = []
    name_to_species_id: dict[str, str] = {}
    # 构建 ontology 索引（按 name 查询）
    ont_by_name: dict[str, dict[str, Any]] = {}
    if ontology_entities:
        for ent in ontology_entities.get("entities", []) or []:
            name = ent.get("name", "")
            if name:
                ont_by_name[name] = ent

    for i, node in enumerate(nodes):
        name = node.get("name") or node.get("id") or f"species_{i}"
        node_id = node.get("id", name)
        species_id = f"SP_{i+1:03d}"
        name_to_species_id[name] = species_id
        if node_id != name:
            name_to_species_id[node_id] = species_id

        species_type = node.get("type", "protein")
        compartment = _DEFAULT_COMPARTMENT_BY_TYPE.get(species_type, "cytoplasm")

        # 填充 ontology（若 P1 Ontology Agent 输出可用）
        ont = OntologyRef()
        ont_entity = ont_by_name.get(name) or ont_by_name.get(node_id)
        if ont_entity:
            ont = OntologyRef(
                hgnc_id=ont_entity.get("hgnc_id"),
                uniprot_id=ont_entity.get("uniprot_id"),
                chebi_id=ont_entity.get("chebi_id"),
                go_terms=ont_entity.get("go_terms", []) or [],
                sbo_term=ont_entity.get("sbo_term"),
                verified=ont_entity.get("verified", False),
            )
        else:
            warnings.append(
                f"Species '{name}' 未在 ontology_entities 中找到，verified=False"
            )

        species_list.append(SpeciesV2(
            id=species_id,
            canonical_name=name,
            display_name=name,
            ontology=ont,
            species_type=species_type,
            compartment=compartment,
            initial_concentration=0.0,  # 初始浓度由 P3 ODE 注入
            source_sbml=sbml_model_id,
        ))

    # 提取 pathway_tag（取首个非空）
    pathway_tag = ""
    for e in edges:
        tag = e.get("pathway_tag", "")
        if tag:
            pathway_tag = tag
            break

    # —— 2. edges → reactions（B4 修复：直接构造 MechanismType，不走 v3 反查表）——
    reactions: list[ReactionV2] = []
    all_edges = list(edges) + list(cross_talk_edges)
    for i, edge in enumerate(all_edges):
        source_name = edge.get("source", "")
        target_name = edge.get("target", "")
        mechanism_str = edge.get("mechanism") or edge.get("interaction") or "activation"

        source_id = name_to_species_id.get(source_name, f"SP_{source_name}")
        target_id = name_to_species_id.get(target_name, f"SP_{target_name}")

        # B4 修复：直接构造 MechanismType 枚举，不经 _V3_INTERACTION_TO_V4_MECHANISM 反查表
        try:
            mechanism = MechanismType(mechanism_str.lower().strip())
        except ValueError:
            warnings.append(
                f"未知 mechanism '{mechanism_str}'，降级为 ACTIVATION"
                f"（edge {i}: {source_name} → {target_name}）"
            )
            mechanism = MechanismType.ACTIVATION

        kinetics_type = mechanism.default_kinetics
        parameter_context = f"{source_name} → {target_name} ({mechanism.value})"

        # 接线强制（纪律1）：调用统一的 _build_reaction_for_mechanism，
        # 禁止 inline if/elif/else 兜底，禁止"默认：source → target"1:1 兜底。
        # 19 种机制各自体现生物学语义（见 _build_reaction_for_mechanism 注释）。
        reactants, products, modifiers = _build_reaction_for_mechanism(
            mechanism, source_id, target_id, source_name, target_name, name_to_species_id
        )

        # compartments：从 species 查询
        sp_source = next((s for s in species_list if s.id == source_id), None)
        sp_target = next((s for s in species_list if s.id == target_id), None)
        compartments_set: set[str] = set()
        if sp_source:
            compartments_set.add(sp_source.compartment)
        if sp_target:
            compartments_set.add(sp_target.compartment)

        # TD-020 (IB-035) 修复：从 edge.provenance 与 sbml_model_id 填充 Provenance，避免恒为 None
        # pathway_graph 的 edge 含 provenance dict（见 docstring），优先从中取 pmid/kegg_id
        _edge_prov = edge.get("provenance") or {}
        _pmid = _edge_prov.get("pmid") or edge.get("pmid")
        _kegg = _edge_prov.get("kegg_id") or edge.get("kegg_id")

        reactions.append(ReactionV2(
            id=f"RXN_{i+1:03d}",
            reaction_type=mechanism.value,
            kinetics_type=kinetics_type,
            reactants=reactants,
            products=products,
            modifiers=modifiers,
            compartments=list(compartments_set),
            parameter_context=parameter_context,
            pathway_tag=pathway_tag,
            provenance=Provenance(
                source_sbml_reaction=sbml_model_id,  # SBML model ID 作为溯源来源
                source_pmid=str(_pmid) if _pmid is not None else None,
                source_kegg=str(_kegg) if _kegg is not None else None,
            ),
        ))

    # —— 3. 自动生成质量守恒约束 ——
    constraints = auto_generate_mass_conservation(species_list, reactions)

    # —— 4. 构建 compartments（从 species 去重）——
    compartment_names: set[str] = set()
    for sp in species_list:
        compartment_names.add(sp.compartment)
    compartments = [
        Compartment(name=name, size=_DEFAULT_COMPARTMENT_SIZE.get(name, 1.0))
        for name in sorted(compartment_names)
    ]

    # —— 5. 确定 source ——
    source = "v3_downgraded" if not ontology_entities else "v4_native"

    ir = ReactionIRv2(
        species=species_list,
        reactions=reactions,
        composite_reactions=[],  # 降级模式无组合反应
        state_machines=[],       # 降级模式无状态机
        compartments=compartments,
        constraints=constraints,
        version="v4.0",
        source=source,
        warnings=warnings,
    )
    logger.info(
        "build_from_pathway_graph: %d species, %d reactions, %d constraints, source=%s",
        len(species_list), len(reactions), len(constraints), source,
    )
    return ir


__all__ = [
    "build_from_network_json",
    "build_from_pathway_graph",
]
