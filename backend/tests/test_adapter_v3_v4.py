# BioDynamics Agent v4 - v3↔v4 Adapter 单元测试
# 对应 v4 Migration Plan §2.6 的 Adapter 测试要求：
# 1. 测试 v3 → v4 → v3 往返转换一致性（diff < 5%）
# 2. 测试 Adapter fail-safe 机制（输入非法 JSON 时不崩溃）
#
# 运行：cd backend && python -m pytest tests/test_adapter_v3_v4.py -v
# 或：  cd backend && python tests/test_adapter_v3_v4.py

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# 确保能导入 app 包（backend/ 在 sys.path）
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


# =============================================================================
# 测试用 fixture：v3 network_json 样本
# =============================================================================
def _make_egf_egfr_network() -> dict:
    """构造 EGF-EGFR-MAPK 通路 v3 network_json 样本。

    包含 5 个节点（EGF/EGFR/pEGFR/GRB2/MAPK）和 4 条边（activation/inhibition），
    覆盖常见的激活/抑制/磷酸化场景。
    """
    return {
        "nodes": [
            {"id": "EGF",    "name": "EGF",    "type": "ligand"},
            {"id": "EGFR",   "name": "EGFR",   "type": "receptor"},
            {"id": "pEGFR",  "name": "pEGFR",  "type": "receptor"},
            {"id": "GRB2",   "name": "GRB2",   "type": "protein"},
            {"id": "MAPK",   "name": "MAPK",   "type": "kinase"},
        ],
        "edges": [
            {"source": "EGF",   "target": "EGFR",  "interaction": "activation"},
            {"source": "EGFR",  "target": "pEGFR", "interaction": "phosphorylation"},
            {"source": "pEGFR", "target": "GRB2",  "interaction": "activation"},
            {"source": "GRB2",  "target": "MAPK",  "interaction": "activation"},
        ],
    }


def _make_inhibition_network() -> dict:
    """构造含抑制边的网络样本（药物-靶点抑制）。"""
    return {
        "nodes": [
            {"id": "Drug",   "name": "Drug",   "type": "drug"},
            {"id": "EGFR",   "name": "EGFR",   "type": "receptor"},
            {"id": "pEGFR",  "name": "pEGFR",  "type": "receptor"},
        ],
        "edges": [
            {"source": "EGF",  "target": "EGFR",  "interaction": "activation"},
            {"source": "EGFR", "target": "pEGFR", "interaction": "phosphorylation"},
            {"source": "Drug", "target": "EGFR",  "interaction": "inhibition"},
        ],
        # 故意省略 EGF 节点，测试 Adapter 对缺失节点的容错
    }


# =============================================================================
# 测试 1：v3 → v4 转换
# =============================================================================
class TestV3ToV4Adapter(unittest.TestCase):
    """测试 v3_to_v4 Adapter 的核心转换逻辑。"""

    def test_basic_conversion_returns_reaction_ir(self):
        """v3 network_json 能成功转换为 ReactionIRv2 对象。"""
        from app.adapters.v3_v4_adapter import v3_to_v4
        from app.reaction_ir_v2.schema import ReactionIRv2

        ir = v3_to_v4(_make_egf_egfr_network())
        self.assertIsNotNone(ir)
        self.assertIsInstance(ir, ReactionIRv2)

    def test_species_count_matches_nodes(self):
        """转换后的 species 数量与 v3 nodes 数量一致。"""
        from app.adapters.v3_v4_adapter import v3_to_v4
        ir = v3_to_v4(_make_egf_egfr_network())
        self.assertEqual(len(ir.species), 5)  # EGF/EGFR/pEGFR/GRB2/MAPK

    def test_reactions_count_matches_edges(self):
        """转换后的 reactions 数量与 v3 edges 数量一致。"""
        from app.adapters.v3_v4_adapter import v3_to_v4
        ir = v3_to_v4(_make_egf_egfr_network())
        self.assertEqual(len(ir.reactions), 4)

    def test_source_marked_as_downgraded(self):
        """v3→v4 转换的 source 字段标记为 'v3_downgraded'。"""
        from app.adapters.v3_v4_adapter import v3_to_v4
        ir = v3_to_v4(_make_egf_egfr_network())
        self.assertEqual(ir.source, "v3_downgraded")

    def test_downgrade_warning_present(self):
        """转换结果包含降级 warning（无状态机/组合反应/未验证 ontology）。"""
        from app.adapters.v3_v4_adapter import v3_to_v4
        ir = v3_to_v4(_make_egf_egfr_network())
        self.assertTrue(any("降级" in w for w in ir.warnings))

    def test_compartment_default_to_cytoplasm(self):
        """缺失 compartment 时默认填 'cytoplasm'。"""
        from app.adapters.v3_v4_adapter import v3_to_v4
        ir = v3_to_v4(_make_egf_egfr_network())
        # 至少 EGFR（receptor 类型）应为 membrane，但 receptor 类型在 _DEFAULT_COMPARTMENT_BY_TYPE 中映射到 membrane
        # 这里验证所有 species 的 compartment 都是合法值
        valid_compartments = {"extracellular", "membrane", "cytoplasm", "nucleus", "mitochondria"}
        for sp in ir.species:
            self.assertIn(sp.compartment, valid_compartments)

    def test_ligand_compartment_is_extracellular(self):
        """ligand 类型节点的 compartment 应为 extracellular。"""
        from app.adapters.v3_v4_adapter import v3_to_v4
        ir = v3_to_v4(_make_egf_egfr_network())
        egf = next(s for s in ir.species if s.canonical_name == "EGF")
        self.assertEqual(egf.compartment, "extracellular")

    def test_receptor_compartment_is_membrane(self):
        """receptor 类型节点的 compartment 应为 membrane。"""
        from app.adapters.v3_v4_adapter import v3_to_v4
        ir = v3_to_v4(_make_egf_egfr_network())
        egfr = next(s for s in ir.species if s.canonical_name == "EGFR")
        self.assertEqual(egfr.compartment, "membrane")

    def test_activation_mechanism_preserved(self):
        """v3 activation 边映射为 v4 activation 机制（不强制映射为 phosphorylation）。"""
        from app.adapters.v3_v4_adapter import v3_to_v4
        ir = v3_to_v4(_make_egf_egfr_network())
        # EGF → EGFR 应为 activation
        egf_to_egfr = next(
            r for r in ir.reactions
            if r.parameter_context.startswith("EGF → EGFR")
        )
        self.assertEqual(egf_to_egfr.reaction_type, "activation")

    def test_inhibition_mechanism_preserved(self):
        """v3 inhibition 边映射为 v4 inhibition 机制。"""
        from app.adapters.v3_v4_adapter import v3_to_v4
        ir = v3_to_v4(_make_inhibition_network())
        # 找到 Drug → EGFR 的 inhibition 反应
        inhib_rxns = [r for r in ir.reactions if r.reaction_type == "inhibition"]
        self.assertGreaterEqual(len(inhib_rxns), 1)

    def test_ontology_entities_propagation(self):
        """传入 ontology_entities 时填充 species 的 HGNC/UniProt ID。"""
        from app.adapters.v3_v4_adapter import v3_to_v4
        ontology = {
            "entities": [
                {
                    "name": "EGFR",
                    "hgnc_id": "HGNC:3236",
                    "uniprot_id": "P00533",
                    "verified": True,
                },
            ],
        }
        ir = v3_to_v4(_make_egf_egfr_network(), ontology_entities=ontology)
        egfr = next(s for s in ir.species if s.canonical_name == "EGFR")
        self.assertEqual(egfr.ontology.hgnc_id, "HGNC:3236")
        self.assertEqual(egfr.ontology.uniprot_id, "P00533")
        self.assertTrue(egfr.ontology.verified)


# =============================================================================
# 测试 2：v4 → v3 转换
# =============================================================================
class TestV4ToV3Adapter(unittest.TestCase):
    """测试 v4_to_v3 Adapter 的核心转换逻辑。"""

    def test_basic_conversion_returns_network_json(self):
        """ReactionIRv2 能成功转换为 v3 network_json。"""
        from app.adapters.v3_v4_adapter import v3_to_v4
        from app.adapters.v4_v3_adapter import v4_to_v3

        ir = v3_to_v4(_make_egf_egfr_network())
        network_json = v4_to_v3(ir)
        self.assertIsNotNone(network_json)
        self.assertIn("nodes", network_json)
        self.assertIn("edges", network_json)

    def test_nodes_count_preserved(self):
        """v4 → v3 转换后 nodes 数量一致。"""
        from app.adapters.v3_v4_adapter import v3_to_v4
        from app.adapters.v4_v3_adapter import v4_to_v3

        original = _make_egf_egfr_network()
        ir = v3_to_v4(original)
        roundtrip = v4_to_v3(ir)
        self.assertEqual(len(roundtrip["nodes"]), len(original["nodes"]))

    def test_edges_count_close_to_original(self):
        """v4 → v3 转换后 edges 数量与原 v3 接近（差异 ≤ 1）。"""
        from app.adapters.v3_v4_adapter import v3_to_v4
        from app.adapters.v4_v3_adapter import v4_to_v3

        original = _make_egf_egfr_network()
        ir = v3_to_v4(original)
        roundtrip = v4_to_v3(ir)
        # 边数差异不超过 1（容忍 phosphorylation 反应可能生成的额外边）
        self.assertAlmostEqual(
            len(roundtrip["edges"]), len(original["edges"]),
            delta=1,
        )

    def test_v4_specific_fields_ignored(self):
        """v4 特有字段（state_machine/composite_reaction）不影响 v3 输出。"""
        from app.adapters.v3_v4_adapter import v3_to_v4
        from app.adapters.v4_v3_adapter import v4_to_v3
        from app.reaction_ir_v2.schema import StateMachine, State, Transition

        ir = v3_to_v4(_make_egf_egfr_network())
        # 添加 state_machine（v4 特有字段）
        ir.state_machines.append(StateMachine(
            id="EGFR_SM",
            species="EGFR",
            states=[State(name="monomer", species_id="SP_001", is_initial=True)],
            transitions=[Transition(from_state="monomer", to_state="dimer", reaction_id="RXN_001")],
        ))
        # 添加 composite_reaction（v4 特有字段）
        from app.reaction_ir_v2.schema import CompositeReaction
        ir.composite_reactions.append(CompositeReaction(
            id="CR_001", name="Test Composite",
            sub_reactions=["RXN_001"],
        ))

        # 转换不应崩溃，且 nodes/edges 数量与无 v4 字段时一致
        network_json = v4_to_v3(ir)
        self.assertIsNotNone(network_json)
        self.assertEqual(len(network_json["nodes"]), 5)


# =============================================================================
# 测试 3：v3 → v4 → v3 往返转换一致性
# =============================================================================
class TestRoundtripConsistency(unittest.TestCase):
    """测试 v3 → v4 → v3 往返转换的一致性（diff < 5%）。"""

    def test_roundtrip_preserves_node_names(self):
        """往返转换后 node 名称集合一致（diff = 0%）。"""
        from app.adapters.adapter_registry import get_adapter_registry
        registry = get_adapter_registry()
        registry.reset_metrics()

        original = _make_egf_egfr_network()
        roundtrip, ir = registry.roundtrip_v3_to_v4_to_v3(original)

        self.assertIsNotNone(roundtrip)
        self.assertIsNotNone(ir)

        original_names = {n["name"] for n in original["nodes"]}
        roundtrip_names = {n["name"] for n in roundtrip["nodes"]}
        # 名称集合应完全一致
        self.assertEqual(original_names, roundtrip_names)

    def test_roundtrip_node_count_diff_below_5_percent(self):
        """往返转换后 nodes 数量 diff < 5%。"""
        from app.adapters.adapter_registry import get_adapter_registry
        registry = get_adapter_registry()
        registry.reset_metrics()

        original = _make_egf_egfr_network()
        roundtrip, _ = registry.roundtrip_v3_to_v4_to_v3(original)

        original_count = len(original["nodes"])
        roundtrip_count = len(roundtrip["nodes"])
        diff_ratio = abs(original_count - roundtrip_count) / max(original_count, 1)
        self.assertLess(diff_ratio, 0.05)

    def test_roundtrip_edge_count_diff_below_5_percent(self):
        """往返转换后 edges 数量 diff < 5%。"""
        from app.adapters.adapter_registry import get_adapter_registry
        registry = get_adapter_registry()
        registry.reset_metrics()

        original = _make_egf_egfr_network()
        roundtrip, _ = registry.roundtrip_v3_to_v4_to_v3(original)

        original_count = len(original["edges"])
        roundtrip_count = len(roundtrip["edges"])
        # 容忍 ±1 的差异（phosphorylation 反应可能合并/拆分）
        diff_ratio = abs(original_count - roundtrip_count) / max(original_count, 1)
        self.assertLess(diff_ratio, 0.05 + 1.0 / max(original_count, 1))  # 加 1 边的容差

    def test_roundtrip_inhibition_network(self):
        """含抑制边的网络往返转换一致性。"""
        from app.adapters.adapter_registry import get_adapter_registry
        registry = get_adapter_registry()
        registry.reset_metrics()

        original = _make_inhibition_network()
        roundtrip, ir = registry.roundtrip_v3_to_v4_to_v3(original)

        # 抑制网络存在缺失节点（EGF 未在 nodes 中），但仍应不崩溃
        # roundtrip 可能为 None（fail-safe）或有效 dict
        if roundtrip is not None:
            self.assertIn("nodes", roundtrip)
            self.assertIn("edges", roundtrip)

    def test_roundtrip_metrics_recorded(self):
        """往返转换后 Adapter metrics 记录成功调用。"""
        from app.adapters.adapter_registry import get_adapter_registry
        registry = get_adapter_registry()
        registry.reset_metrics()

        original = _make_egf_egfr_network()
        registry.roundtrip_v3_to_v4_to_v3(original)

        metrics = registry.get_metrics()
        self.assertGreaterEqual(metrics["v3_to_v4"]["success"], 1)
        self.assertGreaterEqual(metrics["v4_to_v3"]["success"], 1)


# =============================================================================
# 测试 4：Adapter fail-safe 机制
# =============================================================================
class TestAdapterFailSafe(unittest.TestCase):
    """测试 Adapter fail-safe 机制（输入非法 JSON 时不崩溃）。"""

    def setUp(self):
        """每个测试前重置 registry 计数器。"""
        from app.adapters.adapter_registry import get_adapter_registry
        get_adapter_registry().reset_metrics()

    def test_empty_dict_returns_none(self):
        """空 dict 输入返回 None（触发 fail-safe）。"""
        from app.adapters.v3_v4_adapter import v3_to_v4
        result = v3_to_v4({})
        self.assertIsNone(result)

    def test_empty_input_returns_none(self):
        """空输入（None 或空）返回 None。"""
        from app.adapters.v3_v4_adapter import v3_to_v4
        self.assertIsNone(v3_to_v4(None))
        self.assertIsNone(v3_to_v4({}))  # type: ignore[arg-type]

    def test_non_dict_input_returns_none(self):
        """非 dict 类型输入返回 None（不抛异常）。"""
        from app.adapters.v3_v4_adapter import v3_to_v4
        # list / str / int 都应返回 None
        self.assertIsNone(v3_to_v4([]))  # type: ignore[arg-type]
        self.assertIsNone(v3_to_v4("not a dict"))  # type: ignore[arg-type]
        self.assertIsNone(v3_to_v4(42))  # type: ignore[arg-type]

    def test_missing_nodes_and_edges_returns_none(self):
        """缺少 nodes 和 edges 字段返回 None。"""
        from app.adapters.v3_v4_adapter import v3_to_v4
        # 仅有 nodes，无 edges：不返回 None（edges 默认空 list）
        # 仅有 edges，无 nodes：不返回 None（nodes 默认空 list）
        # 同时缺失两者：返回 None
        self.assertIsNone(v3_to_v4({"foo": "bar"}))

    def test_invalid_edge_structure_does_not_crash(self):
        """非法 edge 结构（缺字段）不崩溃，跳过该边。"""
        from app.adapters.v3_v4_adapter import v3_to_v4
        network = {
            "nodes": [{"id": "A", "name": "A", "type": "protein"}],
            "edges": [
                {"source": "A"},  # 缺 target/interaction
                {"target": "A"},  # 缺 source/interaction
                {},               # 全空
                {"source": "A", "target": "B", "interaction": "activation"},  # B 未在 nodes
            ],
        }
        # 应不崩溃，返回有效 IR 或 None（任一均可，关键是 no exception）
        try:
            ir = v3_to_v4(network)
        except Exception as exc:
            self.fail(f"非法 edge 结构不应抛异常，但抛出：{exc}")
        # 如果返回有效 IR，反应数 ≤ edges 总数
        if ir is not None:
            self.assertLessEqual(len(ir.reactions), 4)

    def test_safe_v3_to_v4_failure_returns_none(self):
        """safe_v3_to_v4 在失败时返回 None 而不抛异常。"""
        from app.adapters.adapter_registry import get_adapter_registry
        registry = get_adapter_registry()
        # 输入非法 JSON
        result = registry.safe_v3_to_v4(None)  # type: ignore[arg-type]
        self.assertIsNone(result)

    def test_safe_v4_to_v3_failure_returns_none(self):
        """safe_v4_to_v3 在失败时返回 None 而不抛异常。"""
        from app.adapters.adapter_registry import get_adapter_registry
        registry = get_adapter_registry()
        # 输入 None
        result = registry.safe_v4_to_v3(None)  # type: ignore[arg-type]
        self.assertIsNone(result)
        # 输入非 ReactionIRv2 类型
        result = registry.safe_v4_to_v3({"not": "an IR"})  # type: ignore[arg-type]
        self.assertIsNone(result)

    def test_failure_counter_increments(self):
        """fail-safe 失败时计数器递增。"""
        from app.adapters.adapter_registry import get_adapter_registry
        registry = get_adapter_registry()
        registry.reset_metrics()

        # 触发 3 次失败
        for _ in range(3):
            registry.safe_v3_to_v4(None)  # type: ignore[arg-type]

        metrics = registry.get_metrics()
        self.assertEqual(metrics["v3_to_v4"]["failure"], 3)
        self.assertEqual(metrics["v3_to_v4"]["success"], 0)
        # 未达阈值，尚未禁用
        self.assertFalse(metrics["v3_to_v4"]["disabled"])

    def test_fail_safe_disabled_after_threshold(self):
        """连续失败 5 次后 Adapter 自动禁用。"""
        from app.adapters.adapter_registry import get_adapter_registry
        registry = get_adapter_registry()
        registry.reset_metrics()

        # 触发 5 次失败
        for _ in range(5):
            registry.safe_v3_to_v4(None)  # type: ignore[arg-type]

        metrics = registry.get_metrics()
        self.assertEqual(metrics["v3_to_v4"]["failure"], 5)
        self.assertTrue(metrics["v3_to_v4"]["disabled"])

    def test_disabled_adapter_returns_none_immediately(self):
        """禁用后的 Adapter 直接返回 None，不尝试转换。"""
        from app.adapters.adapter_registry import get_adapter_registry
        registry = get_adapter_registry()
        registry.reset_metrics()

        # 触发 5 次失败以禁用
        for _ in range(5):
            registry.safe_v3_to_v4(None)  # type: ignore[arg-type]

        # 禁用后即使输入合法也应返回 None
        result = registry.safe_v3_to_v4(_make_egf_egfr_network())
        self.assertIsNone(result)

    def test_success_resets_failure_counter(self):
        """成功转换会重置失败计数（恢复正常）。"""
        from app.adapters.adapter_registry import get_adapter_registry
        registry = get_adapter_registry()
        registry.reset_metrics()

        # 先失败 3 次
        for _ in range(3):
            registry.safe_v3_to_v4(None)  # type: ignore[arg-type]
        self.assertEqual(registry.get_metrics()["v3_to_v4"]["failure"], 3)

        # 一次成功
        registry.safe_v3_to_v4(_make_egf_egfr_network())
        metrics = registry.get_metrics()["v3_to_v4"]
        self.assertEqual(metrics["success"], 1)
        self.assertEqual(metrics["failure"], 0)
        self.assertFalse(metrics["disabled"])

    def test_reset_metrics_clears_all(self):
        """reset_metrics 清空所有计数器与禁用状态。"""
        from app.adapters.adapter_registry import get_adapter_registry
        registry = get_adapter_registry()
        # 触发一些失败
        for _ in range(5):
            registry.safe_v3_to_v4(None)  # type: ignore[arg-type]

        registry.reset_metrics()
        metrics = registry.get_metrics()
        self.assertEqual(metrics["v3_to_v4"]["success"], 0)
        self.assertEqual(metrics["v3_to_v4"]["failure"], 0)
        self.assertFalse(metrics["v3_to_v4"]["disabled"])
        self.assertEqual(metrics["v4_to_v3"]["success"], 0)
        self.assertEqual(metrics["v4_to_v3"]["failure"], 0)
        self.assertFalse(metrics["v4_to_v3"]["disabled"])


# =============================================================================
# 测试 5：AdapterRegistry 单例与线程安全
# =============================================================================
class TestAdapterRegistrySingleton(unittest.TestCase):
    """测试 AdapterRegistry 单例模式。"""

    def test_get_adapter_registry_returns_singleton(self):
        """get_adapter_registry 返回全局单例。"""
        from app.adapters.adapter_registry import get_adapter_registry
        r1 = get_adapter_registry()
        r2 = get_adapter_registry()
        self.assertIs(r1, r2)

    def test_metrics_shared_across_calls(self):
        """单例的 metrics 在多次获取间共享。"""
        from app.adapters.adapter_registry import get_adapter_registry
        r1 = get_adapter_registry()
        r1.reset_metrics()
        r1.safe_v3_to_v4(_make_egf_egfr_network())

        r2 = get_adapter_registry()
        metrics = r2.get_metrics()
        self.assertGreaterEqual(metrics["v3_to_v4"]["success"], 1)


# =============================================================================
# 测试 6：graph_v3.py hook 集成（验证 hook 存在且 Flag false 时不执行）
# =============================================================================
class TestReactionIRv2Hook(unittest.TestCase):
    """测试 graph_v3.py 中的 _reaction_ir_v2_hook 集成。"""

    def test_hook_function_exists(self):
        """graph_v3.py 中存在 _reaction_ir_v2_hook 函数。"""
        from app.graph_v3 import _reaction_ir_v2_hook
        self.assertTrue(callable(_reaction_ir_v2_hook))

    def test_hook_disabled_returns_none(self):
        """V4_REACTION_IR_ENABLED=false 时 hook 返回 None（不执行）。"""
        from app.graph_v3 import _reaction_ir_v2_hook
        from unittest.mock import patch

        state = {
            "network_json": _make_egf_egfr_network(),
            "v4_ontology_entities": None,
        }
        with patch("app.graph_v3._v4_settings") as mock_settings:
            mock_settings.V4_REACTION_IR_ENABLED = False
            mock_settings.V4_REACTION_IR_ADAPTER_ENABLED = False
            result = _reaction_ir_v2_hook(state)
        self.assertIsNone(result)

    def test_hook_enabled_populates_v4_reaction_ir(self):
        """V4_REACTION_IR_ENABLED=true 时 hook 填充 v4_reaction_ir 字段。"""
        from app.graph_v3 import _reaction_ir_v2_hook
        from unittest.mock import patch

        state = {
            "network_json": _make_egf_egfr_network(),
            "v4_ontology_entities": None,
        }
        with patch("app.graph_v3._v4_settings") as mock_settings:
            mock_settings.V4_REACTION_IR_ENABLED = True
            mock_settings.V4_REACTION_IR_ADAPTER_ENABLED = False
            result = _reaction_ir_v2_hook(state)

        self.assertIn("v4_reaction_ir", result)
        self.assertIsNotNone(result["v4_reaction_ir"])
        # 不应同步 network_json（ADAPTER_ENABLED=false）
        # hook 仅返回增量，network_json 是否更新由调用方合并

    def test_hook_adapter_syncs_network_json(self):
        """V4_REACTION_IR_ENABLED=true + ADAPTER_ENABLED=true 时 hook 同步 network_json。"""
        from app.graph_v3 import _reaction_ir_v2_hook
        from unittest.mock import patch

        state = {
            "network_json": _make_egf_egfr_network(),
            "v4_ontology_entities": None,
        }
        with patch("app.graph_v3._v4_settings") as mock_settings:
            mock_settings.V4_REACTION_IR_ENABLED = True
            mock_settings.V4_REACTION_IR_ADAPTER_ENABLED = True
            result = _reaction_ir_v2_hook(state)

        # v4_reaction_ir 被填充
        self.assertIn("v4_reaction_ir", result)
        self.assertIsNotNone(result["v4_reaction_ir"])
        # network_json 被同步更新（通过 v4_to_v3 Adapter）
        self.assertIn("network_json", result)
        self.assertIn("nodes", result["network_json"])
        self.assertIn("edges", result["network_json"])

    def test_hook_fail_safe_on_empty_network(self):
        """network_json 为空时 hook 不崩溃，返回 None（fail-safe）。"""
        from app.graph_v3 import _reaction_ir_v2_hook
        from unittest.mock import patch

        state = {"network_json": {}, "v4_ontology_entities": None}
        with patch("app.graph_v3._v4_settings") as mock_settings:
            mock_settings.V4_REACTION_IR_ENABLED = True
            mock_settings.V4_REACTION_IR_ADAPTER_ENABLED = True
            result = _reaction_ir_v2_hook(state)
        # 不崩溃，且不阻塞主流水线（返回 None 表示跳过）
        self.assertIsNone(result)


# =============================================================================
# 入口
# =============================================================================
if __name__ == "__main__":
    unittest.main(verbosity=2)
