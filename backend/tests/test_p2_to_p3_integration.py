# BioDynamics Agent v4 - P2→P3 集成测试
# 覆盖真实 ReactionIRv2.model_dump() 输出驱动 PathwayGraphBuilder
# 修复审计报告 §6.2 / §7.1 的 B1/B2/S2/S3 P0 阻断：
#   B1: species.canonical_name vs name
#   B2: reactants[].species_id vs name/species/id
#   S2: ReactionV2 缺 sbo_term
#   S3: provenance 嵌套 vs 平铺
#
# 运行方式：
#   cd backend
#   python -m pytest tests/test_p2_to_p3_integration.py -v

from __future__ import annotations

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_DIR))


def _build_real_reaction_ir():
    """构造真实的 ReactionIRv2.model_dump() 输出（含 SpeciesRef/Provenance）。"""
    from app.reaction_ir_v2.schema import (
        ReactionIRv2,
        SpeciesV2,
        ReactionV2,
        SpeciesRef,
        Modifier,
        Provenance,
        Constraint,
        Compartment,
    )

    # 真实 P2 序列化产物：species 用 canonical_name，SpeciesRef 用 species_id，
    # Provenance 嵌套在 reaction 下，ReactionV2 无 sbo_term 字段。
    species_list = [
        SpeciesV2(
            id="SP_EGF",
            canonical_name="EGF",
            display_name="Epidermal Growth Factor",
            species_type="ligand",
            compartment="extracellular",
            initial_concentration=10.0,
        ),
        SpeciesV2(
            id="SP_EGFR",
            canonical_name="EGFR",
            display_name="EGF Receptor",
            species_type="receptor",
            compartment="membrane",
            initial_concentration=100.0,
        ),
        SpeciesV2(
            id="SP_pEGFR",
            canonical_name="pEGFR",
            display_name="Phosphorylated EGFR",
            species_type="receptor",
            compartment="membrane",
            initial_concentration=0.0,
        ),
    ]

    reactions = [
        ReactionV2(
            id="RXN_001",
            reaction_type="binding",
            kinetics_type="mass_action",
            reactants=[SpeciesRef(species_id="SP_EGF", role="substrate")],
            products=[SpeciesRef(species_id="SP_EGFR", role="product")],
            modifiers=[],
            compartments=["extracellular", "membrane"],
            parameter_context="EGF binding activates EGFR",
            pathway_tag="EGFR_RTK",
            provenance=Provenance(
                source_sbml_reaction="BIOMD0000000022_R1",
                source_pmid="12345678",
                source_kegg="hsa04012_R1",
            ),
        ),
        ReactionV2(
            id="RXN_002",
            reaction_type="phosphorylation",
            kinetics_type="Michaelis_Menten",
            reactants=[SpeciesRef(species_id="SP_EGFR", role="substrate")],
            products=[SpeciesRef(species_id="SP_pEGFR", role="product")],
            modifiers=[Modifier(species_id="SP_EGFR", modifier_type="catalytic", site="Tyr1068")],
            compartments=["membrane"],
            parameter_context="EGFR autophosphorylation",
            pathway_tag="EGFR_RTK",
            provenance=Provenance(
                source_sbml_reaction="BIOMD0000000022_R2",
                source_pmid="12345678",
            ),
        ),
    ]

    ir = ReactionIRv2(
        species=species_list,
        reactions=reactions,
        composite_reactions=[],
        state_machines=[],
        compartments=[
            Compartment(name="extracellular", size=1.0),
            Compartment(name="membrane", size=0.1),
        ],
        constraints=[],
        version="v4.0",
        source="v4_native",
    )
    return ir.model_dump()


# =============================================================================
# 1. 端到端集成测试：真实 P2 序列化 → P3 PathwayGraph
# =============================================================================
def test_p2_to_p3_integration_end_to_end():
    """真实 ReactionIRv2.model_dump() 输出驱动 PathwayGraphBuilder。

    修复前断言（会失败）：
    - len(edges) == 0  ← B2 阻断，所有 edges 被跳过
    - nodes[0].canonical_name == "SP_EGF"  ← B1 阻断，fallback 到 species.id

    修复后断言：
    - len(edges) > 0
    - nodes 有正确的 canonical_name（不是 species.id）
    - edge.source/target 指向 PN_<canonical_name>
    """
    from app.pathway_graph.builder import PathwayGraphBuilder

    reaction_ir = _build_real_reaction_ir()
    builder = PathwayGraphBuilder()
    graph = builder.build(
        pathway_class="EGFR_RTK",
        ontology_entities=None,
        reaction_ir=reaction_ir,
    )

    # —— B1 修复验证：nodes 使用 canonical_name 而非 species.id ——
    node_names = {n.canonical_name for n in graph.nodes}
    assert "EGF" in node_names, f"节点缺少 canonical_name='EGF'，实际={node_names}"
    assert "EGFR" in node_names, f"节点缺少 canonical_name='EGFR'，实际={node_names}"
    assert "pEGFR" in node_names, f"节点缺少 canonical_name='pEGFR'，实际={node_names}"
    # 不能 fallback 到 species.id
    assert "SP_EGF" not in node_names, f"节点错误 fallback 到 species.id，实际={node_names}"

    # —— B2 修复验证：edges 非空，且能正确解析 species_id → canonical_name ——
    assert len(graph.edges) >= 2, (
        f"P2→P3 edges 为空（B2 未修复），实际 edges={len(graph.edges)}"
    )

    # 验证 edge 指向 PN_<canonical_name>
    edge_endpoints = {(e.source, e.target) for e in graph.edges}
    assert ("PN_EGFR", "PN_pEGFR") in edge_endpoints, (
        f"phosphorylation edge 未正确解析 species_id→canonical_name，"
        f"实际 endpoints={edge_endpoints}"
    )


# =============================================================================
# 2. B1 单测：species 字段名适配
# =============================================================================
def test_b1_canonical_name_priority():
    """B1：P2 用 canonical_name，P3 优先读取（修复前用 name 取不到）。"""
    from app.pathway_graph.builder import PathwayGraphBuilder

    ir = {
        "species": [
            {"id": "SP_X", "canonical_name": "ProteinX", "compartment": "cytoplasm", "species_type": "protein"},
        ],
        "reactions": [],
    }
    builder = PathwayGraphBuilder()
    graph = builder.build(pathway_class="TEST", reaction_ir=ir)
    assert len(graph.nodes) == 1
    assert graph.nodes[0].canonical_name == "ProteinX", (
        f"B1 未修复：canonical_name 取错，实际={graph.nodes[0].canonical_name}"
    )


def test_b1_canonical_name_fallback_to_name():
    """B1 fallback：P2 退化产物若只有 name，P3 仍能正确读取。"""
    from app.pathway_graph.builder import PathwayGraphBuilder

    ir = {
        "species": [
            {"id": "SP_X", "name": "ProteinX", "compartment": "cytoplasm", "species_type": "protein"},
        ],
        "reactions": [],
    }
    builder = PathwayGraphBuilder()
    graph = builder.build(pathway_class="TEST", reaction_ir=ir)
    assert graph.nodes[0].canonical_name == "ProteinX"


def test_b1_canonical_name_fallback_to_id():
    """B1 终极 fallback：仅 id 时，用 id 当 name（不阻断流程）。"""
    from app.pathway_graph.builder import PathwayGraphBuilder

    ir = {
        "species": [
            {"id": "EGFR", "compartment": "membrane", "species_type": "receptor"},
        ],
        "reactions": [],
    }
    builder = PathwayGraphBuilder()
    graph = builder.build(pathway_class="TEST", reaction_ir=ir)
    assert graph.nodes[0].canonical_name == "EGFR"


# =============================================================================
# 3. B2 单测：SpeciesRef.species_id 解析
# =============================================================================
def test_b2_species_id_resolved_via_index():
    """B2：P2 SpeciesRef 用 species_id，P3 通过 id→name 反查表解析。"""
    from app.pathway_graph.builder import PathwayGraphBuilder

    ir = {
        "species": [
            {"id": "SP_EGFR", "canonical_name": "EGFR", "compartment": "membrane", "species_type": "receptor"},
            {"id": "SP_pEGFR", "canonical_name": "pEGFR", "compartment": "membrane", "species_type": "receptor"},
        ],
        "reactions": [
            {
                "id": "RXN_001",
                "reaction_type": "phosphorylation",
                "reactants": [{"species_id": "SP_EGFR", "role": "substrate"}],
                "products": [{"species_id": "SP_pEGFR", "role": "product"}],
                "provenance": {},
            },
        ],
    }
    builder = PathwayGraphBuilder()
    graph = builder.build(pathway_class="EGFR_RTK", reaction_ir=ir)

    assert len(graph.edges) == 1, f"B2 未修复：edges 应为 1，实际={len(graph.edges)}"
    assert graph.edges[0].source == "PN_EGFR"
    assert graph.edges[0].target == "PN_pEGFR"


def test_b2_species_id_fallback_to_dict_keys():
    """B2 fallback：SpeciesRef 含 name/species/id 字段时也能解析（兼容 v3 简化格式）。"""
    from app.pathway_graph.builder import PathwayGraphBuilder

    ir = {
        "species": [],
        "reactions": [
            {
                "id": "RXN_001",
                "reaction_type": "activation",
                "reactants": [{"name": "EGFR", "role": "substrate"}],
                "products": [{"name": "pEGFR", "role": "product"}],
            },
        ],
    }
    builder = PathwayGraphBuilder()
    graph = builder.build(pathway_class="TEST", reaction_ir=ir)
    assert len(graph.edges) == 1
    assert graph.edges[0].source == "PN_EGFR"
    assert graph.edges[0].target == "PN_pEGFR"


def test_b2_species_id_string_list_compat():
    """B2 兼容：reactants/products 为字符串列表（v3 network_json 风格）。"""
    from app.pathway_graph.builder import PathwayGraphBuilder

    ir = {
        "species": [],
        "reactions": [
            {
                "id": "RXN_001",
                "reaction_type": "activation",
                "reactants": ["EGFR"],
                "products": ["pEGFR"],
            },
        ],
    }
    builder = PathwayGraphBuilder()
    graph = builder.build(pathway_class="TEST", reaction_ir=ir)
    assert len(graph.edges) == 1
    assert graph.edges[0].source == "PN_EGFR"
    assert graph.edges[0].target == "PN_pEGFR"


# =============================================================================
# 4. S2 单测：sbo_term 从 P2 MechanismType 反查
# =============================================================================
def test_s2_sbo_term_inferred_from_mechanism():
    """S2：P2 ReactionV2 无 sbo_term，P3 从 MechanismType 反查。"""
    from app.pathway_graph.builder import PathwayGraphBuilder

    ir = {
        "species": [
            {"id": "SP_EGFR", "canonical_name": "EGFR", "compartment": "membrane", "species_type": "receptor"},
            {"id": "SP_pEGFR", "canonical_name": "pEGFR", "compartment": "membrane", "species_type": "receptor"},
        ],
        "reactions": [
            {
                "id": "RXN_001",
                "reaction_type": "phosphorylation",
                "reactants": [{"species_id": "SP_EGFR"}],
                "products": [{"species_id": "SP_pEGFR"}],
            },
        ],
    }
    builder = PathwayGraphBuilder()
    graph = builder.build(pathway_class="EGFR_RTK", reaction_ir=ir)
    sbo = graph.edges[0].sbo_term
    assert sbo is not None, "sbo_term 仍为 None（S2 未修复）"
    assert sbo.startswith("SBO:"), f"sbo_term 格式错：{sbo}"


# =============================================================================
# 5. S3 单测：provenance 嵌套字段路径
# =============================================================================
def test_s3_provenance_nested_path():
    """S3：P2 Provenance 嵌套在 reaction.provenance 下，P3 正确读取。"""
    from app.pathway_graph.builder import PathwayGraphBuilder

    ir = {
        "species": [
            {"id": "SP_EGFR", "canonical_name": "EGFR", "compartment": "membrane", "species_type": "receptor"},
            {"id": "SP_pEGFR", "canonical_name": "pEGFR", "compartment": "membrane", "species_type": "receptor"},
        ],
        "reactions": [
            {
                "id": "RXN_001",
                "reaction_type": "phosphorylation",
                "reactants": [{"species_id": "SP_EGFR"}],
                "products": [{"species_id": "SP_pEGFR"}],
                "provenance": {
                    "source_sbml_reaction": "BIOMD0000000022_R2",
                    "source_pmid": "12345678",
                    "source_kegg": "hsa04012_R2",
                },
            },
        ],
    }
    builder = PathwayGraphBuilder()
    graph = builder.build(pathway_class="EGFR_RTK", reaction_ir=ir)
    edge = graph.edges[0]
    assert edge.source_sbml_reaction == "BIOMD0000000022_R2", (
        f"provenance.source_sbml_reaction 未读出：{edge.source_sbml_reaction}"
    )
    assert edge.source_pmid == "12345678", (
        f"provenance.source_pmid 未读出：{edge.source_pmid}"
    )
    assert edge.source_kegg == "hsa04012_R2", (
        f"provenance.source_kegg 未读出：{edge.source_kegg}"
    )


def test_s3_provenance_flat_path_fallback():
    """S3 fallback：provenance 平铺在 reaction 顶层也能读（兼容简化格式）。"""
    from app.pathway_graph.builder import PathwayGraphBuilder

    ir = {
        "species": [
            {"id": "SP_EGFR", "canonical_name": "EGFR", "compartment": "membrane", "species_type": "receptor"},
            {"id": "SP_pEGFR", "canonical_name": "pEGFR", "compartment": "membrane", "species_type": "receptor"},
        ],
        "reactions": [
            {
                "id": "RXN_001",
                "reaction_type": "phosphorylation",
                "reactants": [{"species_id": "SP_EGFR"}],
                "products": [{"species_id": "SP_pEGFR"}],
                "source_sbml_reaction": "BIOMD0000000022_R2",
                "source_pmid": "99999999",
            },
        ],
    }
    builder = PathwayGraphBuilder()
    graph = builder.build(pathway_class="EGFR_RTK", reaction_ir=ir)
    edge = graph.edges[0]
    assert edge.source_sbml_reaction == "BIOMD0000000022_R2"
    assert edge.source_pmid == "99999999"


# =============================================================================
# 6. 警告处理：无法解析的边不阻断
# =============================================================================
def test_unresolvable_edge_warns_but_does_not_block():
    """未引用的 species_id 应触发 warning 并跳过，不阻断整个图。"""
    from app.pathway_graph.builder import PathwayGraphBuilder

    ir = {
        "species": [
            {"id": "SP_EGFR", "canonical_name": "EGFR", "compartment": "membrane", "species_type": "receptor"},
            {"id": "SP_pEGFR", "canonical_name": "pEGFR", "compartment": "membrane", "species_type": "receptor"},
        ],
        "reactions": [
            {
                "id": "RXN_BAD",
                "reaction_type": "phosphorylation",
                "reactants": [{"species_id": "SP_NONEXIST"}],
                "products": [{"species_id": "SP_pEGFR"}],
            },
            {
                "id": "RXN_GOOD",
                "reaction_type": "activation",
                "reactants": [{"species_id": "SP_EGFR"}],
                "products": [{"species_id": "SP_pEGFR"}],
            },
        ],
    }
    builder = PathwayGraphBuilder()
    graph = builder.build(pathway_class="EGFR_RTK", reaction_ir=ir)

    # RXN_BAD 被跳过，RXN_GOOD 保留
    edge_ids = {e.id for e in graph.edges}
    assert "PE_RXN_BAD" not in edge_ids
    assert "PE_RXN_GOOD" in edge_ids
    # 应有 warning 记录
    assert any("RXN_BAD" in w for w in graph.warnings), (
        f"未解析的 edge 应触发 warning，实际 warnings={graph.warnings}"
    )
