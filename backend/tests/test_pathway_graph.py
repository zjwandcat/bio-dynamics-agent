# BioDynamics Agent v4 - Phase 3 单元测试
# 对应 PART C6：Minimum Viable P3 Design
#
# 测试范围：
#   1. PathwayGraph schema 校验
#   2. PathwayGraphBuilder 从 reaction_ir 构建
#   3. PathwayInitializer 10 通路覆盖
#   4. ODERendererV2 模板选择与渲染
#   5. oscillation_detector / bistability_detector
#   6. Feature flag 关闭时行为同 v3（smoke test）
#
# 运行方式：
#   cd backend
#   python -m pytest tests/test_pathway_graph.py -v

from __future__ import annotations

import os
import sys
from pathlib import Path

# 确保 backend/app 可导入
BACKEND_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND_DIR))


# =============================================================================
# 1. PathwayGraph schema 校验
# =============================================================================
def test_pathway_graph_schema_basic():
    """测试 PathwayGraph schema 基本校验。"""
    from app.pathway_graph.schema import (
        PathwayGraph, PathwayNode, PathwayEdge, PathwayModule, TimeScale,
    )

    node = PathwayNode(
        id="PN_EGFR",
        canonical_name="EGFR",
        species_type="protein",
        pathway_class="EGFR_RTK",
        compartment="membrane",
        time_scale=TimeScale.FAST,
    )
    assert node.canonical_name == "EGFR"
    assert node.compartment == "membrane"
    assert node.module == PathwayModule.CORE

    edge = PathwayEdge(
        id="PE_EGF_EGFR",
        source="PN_EGF",
        target="PN_EGFR",
        mechanism="binding",
        pathway_tag="EGFR_RTK",
    )
    assert edge.mechanism == "binding"
    assert edge.kinetics_type == "mass_action"  # 默认值

    graph = PathwayGraph(
        pathway_class="EGFR_RTK",
        nodes=[node],
        edges=[edge],
    )
    assert graph.pathway_class == "EGFR_RTK"
    assert len(graph.nodes) == 1
    assert len(graph.edges) == 1
    assert graph.node_by_id("PN_EGFR") is not None
    assert graph.node_by_name("EGFR") is not None


def test_pathway_graph_schema_invalid_compartment_fallback():
    """测试无效 compartment 回退到 cytoplasm。"""
    from app.pathway_graph.schema import PathwayNode

    node = PathwayNode(
        id="PN_TEST",
        canonical_name="TEST",
        compartment="invalid_compartment",  # 无效值
    )
    assert node.compartment == "cytoplasm"  # 回退默认


def test_pathway_graph_schema_invalid_mechanism_fallback():
    """测试无效 mechanism 回退到 activation。"""
    from app.pathway_graph.schema import PathwayEdge

    edge = PathwayEdge(
        id="PE_TEST",
        source="PN_A",
        target="PN_B",
        mechanism="invalid_mechanism",  # 无效值
    )
    assert edge.mechanism == "activation"  # 回退默认


# =============================================================================
# 2. PathwayGraphBuilder 从 reaction_ir 构建
# =============================================================================
def test_pathway_graph_builder_basic():
    """测试 PathwayGraphBuilder 基本构建。"""
    from app.pathway_graph.builder import PathwayGraphBuilder

    reaction_ir = {
        "species": [
            {"name": "EGF", "compartment": "extracellular", "species_type": "ligand"},
            {"name": "EGFR", "compartment": "membrane", "species_type": "protein"},
            {"name": "pEGFR", "compartment": "membrane", "species_type": "protein"},
        ],
        "reactions": [
            {
                "id": "R1",
                "source": "EGF",
                "target": "EGFR",
                "reaction_type": "binding",
            },
            {
                "id": "R2",
                "source": "EGFR",
                "target": "pEGFR",
                "reaction_type": "phosphorylation",
            },
        ],
    }

    ontology_entities = {
        "EGFR": {
            "hgnc_id": "HGNC:3236",
            "uniprot_id": "P00533",
            "verified": True,
            "compartment": "membrane",
        },
    }

    builder = PathwayGraphBuilder()
    graph = builder.build(
        pathway_class="EGFR_RTK",
        ontology_entities=ontology_entities,
        reaction_ir=reaction_ir,
    )

    assert graph.pathway_class == "EGFR_RTK"
    assert len(graph.nodes) == 3
    assert len(graph.edges) == 2

    # 检查 ontology 增强
    egfr_node = graph.node_by_name("EGFR")
    assert egfr_node is not None
    assert egfr_node.hgnc_id == "HGNC:3236"
    assert egfr_node.uniprot_id == "P00533"
    assert egfr_node.ontology_verified is True

    # 检查 mechanism → kinetics 映射
    binding_edge = next(e for e in graph.edges if e.mechanism == "binding")
    assert binding_edge.kinetics_type == "mass_action"

    phos_edge = next(e for e in graph.edges if e.mechanism == "phosphorylation")
    assert phos_edge.kinetics_type == "Michaelis_Menten"  # v4 恢复 MM


def test_pathway_graph_builder_empty_inputs():
    """测试 PathwayGraphBuilder 容错（空输入）。"""
    from app.pathway_graph.builder import PathwayGraphBuilder

    builder = PathwayGraphBuilder()
    graph = builder.build(pathway_class="TEST")

    assert graph.pathway_class == "TEST"
    assert len(graph.nodes) == 0
    assert len(graph.edges) == 0
    assert len(graph.warnings) > 0  # 应有警告


# =============================================================================
# 3. PathwayInitializer 10 通路覆盖
# =============================================================================
def test_pathway_initializer_10_pathways():
    """测试 PathwayInitializer 覆盖全部 10 通路。"""
    from app.pathway_graph.initializer import PathwayInitializer

    pathways = PathwayInitializer.list_pathways()
    assert len(pathways) == 10

    expected = {
        "EGFR_RTK", "MAPK_ERK", "PI3K_AKT_mTOR", "p53_signaling",
        "Apoptosis", "Cell_Cycle", "JAK_STAT", "NF_kB", "Wnt", "TGF_beta",
    }
    assert set(pathways) == expected


def test_pathway_initializer_egfr_data():
    """测试 EGFR_RTK 通路初始化数据。"""
    from app.pathway_graph.initializer import PathwayInitializer

    nodes, edges, feedbacks, crosstalks = PathwayInitializer.get_pathway_init_data("EGFR_RTK")

    assert len(nodes) > 0
    # EGFR 应在 nodes 中
    egfr_node = next((n for n in nodes if n.canonical_name == "EGFR"), None)
    assert egfr_node is not None
    assert egfr_node.compartment == "membrane"

    # 应有 feedback_loops
    assert len(feedbacks) > 0
    # 应有 cross_talk
    assert len(crosstalks) > 0


def test_pathway_initializer_metadata():
    """测试通路元信息。"""
    from app.pathway_graph.initializer import PathwayInitializer

    meta = PathwayInitializer.get_pathway_metadata("EGFR_RTK")
    assert meta["pathway_class"] == "EGFR_RTK"
    assert meta["source_sbml"] == "BIOMD0000000022"
    assert meta["source_kegg"] == "hsa04012"


# =============================================================================
# 4. ODERendererV2 模板选择与渲染
# =============================================================================
def test_ode_renderer_v2_template_selection():
    """测试 ODERendererV2 模板选择逻辑。"""
    from app.ode_renderer_v2 import ODERendererV2

    renderer = ODERendererV2()

    # 振荡通路 → oscillatory_feedback.j2
    assert renderer._select_template("p53_signaling", False) == "oscillatory_feedback.j2"
    assert renderer._select_template("NF_kB", False) == "oscillatory_feedback.j2"
    assert renderer._select_template("TGF_beta", False) == "oscillatory_feedback.j2"
    assert renderer._select_template("JAK_STAT", False) == "oscillatory_feedback.j2"

    # 双稳态通路 → bistable_switch.j2
    assert renderer._select_template("Apoptosis", False) == "bistable_switch.j2"
    assert renderer._select_template("Cell_Cycle", False) == "bistable_switch.j2"

    # DDE 需求 → oscillatory_feedback.j2
    assert renderer._select_template("EGFR_RTK", True) == "oscillatory_feedback.j2"


def test_ode_renderer_v2_render_basic():
    """测试 ODERendererV2 基本渲染。"""
    from app.ode_renderer_v2 import ODERendererV2

    reaction_ir = {
        "species": [
            {"name": "p53", "initial_concentration": 0.1},
            {"name": "Mdm2_mRNA", "initial_concentration": 0.0},
            {"name": "Mdm2", "initial_concentration": 0.0},
        ],
        "reactions": [
            {"source": "p53", "target": "Mdm2_mRNA", "reaction_type": "transcription"},
            {"source": "Mdm2_mRNA", "target": "Mdm2", "reaction_type": "translation"},
        ],
    }

    renderer = ODERendererV2()
    ode_code = renderer.render(
        pathway_class="p53_signaling",
        reaction_ir=reaction_ir,
        t_end=360.0,
    )

    assert isinstance(ode_code, str)
    assert len(ode_code) > 0
    assert "SPECIES_NAMES" in ode_code
    assert "p53" in ode_code
    assert "Mdm2_mRNA" in ode_code
    assert "DDE_DELAY" in ode_code  # p53_signaling 需要 DDE


# =============================================================================
# 5. 检测器测试
# =============================================================================
def test_oscillation_detector_monotonic():
    """测试振荡检测器（单调信号）。"""
    import numpy as np
    from app.solvers.oscillation_detector import detect_oscillation

    # 单调递增信号
    t = np.linspace(0, 100, 100)
    y = t.copy()  # 线性递增

    result = detect_oscillation(t, y, species_name="test")
    assert result["oscillatory"] is False
    assert result["oscillation_type"] == "monotonic"


def test_oscillation_detector_sustained():
    """测试振荡检测器（持续振荡信号）。"""
    import numpy as np
    from app.solvers.oscillation_detector import detect_oscillation

    # 正弦振荡信号
    t = np.linspace(0, 100, 1000)
    y = np.sin(t * 0.5) + 1.0  # 振幅 1，偏置 1

    result = detect_oscillation(t, y, species_name="test")
    assert result["n_peaks"] >= 2
    assert result["period_minutes"] > 0


def test_bistability_detector_on_state():
    """测试双稳态检测器（高态）。"""
    import numpy as np
    from app.solvers.bistability_detector import detect_bistability

    # 信号最终进入高态
    t = np.linspace(0, 100, 100)
    y = np.zeros((100, 2))
    y[:, 0] = np.linspace(0, 1, 100)  # Caspase3_active 从 0 到 1
    y[:, 1] = 0.5  # 其他 species

    result = detect_bistability(t, y, ["Caspase3_active", "other"])
    assert result["final_state"] == "ON"
    assert result["bistable"] is True


def test_bistability_detector_off_state():
    """测试双稳态检测器（低态：信号先短暂升高后回落）。"""
    import numpy as np
    from app.solvers.bistability_detector import detect_bistability

    # 信号短暂升高后回落到低态（瞬态，非持续高态）
    t = np.linspace(0, 100, 100)
    y = np.zeros((100, 2))
    # 前 30 步从 0 升到 0.8，后 70 步降回 0.05
    y[:30, 0] = np.linspace(0, 0.8, 30)
    y[30:, 0] = np.linspace(0.8, 0.05, 70)
    y[:, 1] = 0.5

    result = detect_bistability(t, y, ["Caspase3_active", "other"])
    # 最终值 0.05 远低于 max 0.8，判为 OFF 或 TRANSIENT（非 ON）
    assert result["final_state"] in ("OFF", "TRANSIENT")
    assert result["bistable"] is False


# =============================================================================
# 6. Feature flag 关闭时行为同 v3（smoke test）
# =============================================================================
def test_feature_flags_default_false():
    """测试 P3 feature flags 默认为 false。"""
    from app.config import settings

    assert settings.V4_PATHWAY_GRAPH_ENABLED is False
    assert settings.V4_ODE_TEMPLATE_V2_ENABLED is False


def test_pathway_graph_hook_disabled_when_flag_false():
    """测试 V4_PATHWAY_GRAPH_ENABLED=false 时 hook 返回 None。"""
    from app.graph_v3 import _pathway_graph_hook

    state = {"v4_reaction_ir": {"species": [], "reactions": []}}
    result = _pathway_graph_hook(state)
    assert result is None  # flag 关闭，返回 None


def test_ode_template_v2_hook_disabled_when_flag_false():
    """测试 V4_ODE_TEMPLATE_V2_ENABLED=false 时 hook 返回 None。"""
    from app.graph_v3 import _ode_template_v2_hook

    state = {"v4_reaction_ir": {"species": [], "reactions": []}}
    result = _ode_template_v2_hook(state)
    assert result is None  # flag 关闭，返回 None


def test_pathway_graph_hook_skipped_when_reaction_ir_missing():
    """测试 v4_reaction_ir 缺失时 hook 返回 None（即使 flag 开启）。"""
    import os
    os.environ["V4_PATHWAY_GRAPH_ENABLED"] = "true"
    # 重新加载 settings
    import importlib
    from app import config as _config_mod
    importlib.reload(_config_mod)
    from app.graph_v3 import _pathway_graph_hook

    state = {}  # 无 v4_reaction_ir
    result = _pathway_graph_hook(state)
    assert result is None  # 输入缺失，返回 None

    # 恢复 flag
    os.environ.pop("V4_PATHWAY_GRAPH_ENABLED", None)
    importlib.reload(_config_mod)


# =============================================================================
# 主程序
# =============================================================================
if __name__ == "__main__":
    # 简单运行所有测试
    test_funcs = [
        test_pathway_graph_schema_basic,
        test_pathway_graph_schema_invalid_compartment_fallback,
        test_pathway_graph_schema_invalid_mechanism_fallback,
        test_pathway_graph_builder_basic,
        test_pathway_graph_builder_empty_inputs,
        test_pathway_initializer_10_pathways,
        test_pathway_initializer_egfr_data,
        test_pathway_initializer_metadata,
        test_ode_renderer_v2_template_selection,
        test_ode_renderer_v2_render_basic,
        test_oscillation_detector_monotonic,
        test_oscillation_detector_sustained,
        test_bistability_detector_on_state,
        test_bistability_detector_off_state,
        test_feature_flags_default_false,
        test_pathway_graph_hook_disabled_when_flag_false,
        test_ode_template_v2_hook_disabled_when_flag_false,
    ]
    passed = 0
    failed = 0
    for func in test_funcs:
        try:
            func()
            print(f"PASS: {func.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAIL: {func.__name__}: {e}")
            failed += 1
    print(f"\n=== {passed} passed, {failed} failed ===")
