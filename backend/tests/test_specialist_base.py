# BioDynamics Agent v4 - Pathway Specialist Base 单元测试 (Phase 4 / Task 4.2)
#
# 测试用例：
#   1. 基类接口完整：PathwaySpecialistBase 含 6 个 abstract method + 3 个辅助方法
#   2. 子类必须实现所有 abstract method：未实现所有 method 时应抛 TypeError
#   3. plugin registry 注册/查询：register_specialist 装饰器 + get_specialist/list/is_available
#   4. registry 初始为空：未注册任何 Specialist 时 list_specialists 返回空列表
#   5. select_template 映射：phosphorylation/bistable/oscillatory 三种机制映射正确
#   6. 5 模块 dataclass 字段完整：5 个 ModuleData 可实例化且字段可访问
#   7. Feature Flag 隔离：V4_PATHWAY_SPECIALIST_ENABLED=false 时 hook 不调用任何 Specialist
#   8. config flag 默认 false：settings.V4_PATHWAY_SPECIALIST_ENABLED 默认为 False
#
# 运行：cd backend && python -m pytest tests/test_specialist_base.py -v

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# 确保能导入 app 包（backend/ 在 sys.path）
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


# =============================================================================
# 测试用 mock Specialist 子类
# =============================================================================
class _CompleteMockSpecialist:  # 占位，下方实际导入后 patch 为基类子类
    pass


def _make_complete_mock_specialist():
    """构造完整的 mock Specialist 子类（实现所有 6 接口）。

    使用工厂函数而非模块级定义，避免被 register_specialist 装饰器副作用污染。
    """
    from app.pathways.pathway_specialist_base import PathwaySpecialistBase

    class _CompleteSpecialist(PathwaySpecialistBase):
        """完整的 mock Specialist 子类，实现所有 6 个 abstract method。"""

        pathway_class = "MOCK_PATHWAY"
        display_name = "Mock Pathway for Testing"

        def load_module(self, module_name: str):
            return {"module": module_name, "loaded": True}

        def apply_core(self, pathway_graph, ontology_entities=None):
            return {"species": [{"name": "MOCK_SPECIES"}], "reactions": []}

        def apply_feedback(self, pathway_graph):
            return [{"loop_id": "FL_MOCK", "sign": "negative"}]

        def apply_crosstalk(self, pathway_graph, crosstalk_edges):
            return []

        def apply_perturbation(self, pathway_graph, perturbation_points):
            return []

        def apply_validation(self, simulation_result=None):
            return [
                {
                    "metric_name": "mock_peak_time_min",
                    "expected": 7.5,
                    "tolerance": 2.5,
                    "pmid": "MOCK_2000",
                }
            ]

    return _CompleteSpecialist


def _make_incomplete_mock_specialist():
    """构造不完整的 mock Specialist 子类（仅实现 1 个接口）。

    用于验证未实现所有 abstract method 时实例化抛 TypeError。
    """
    from app.pathways.pathway_specialist_base import PathwaySpecialistBase

    class _IncompleteSpecialist(PathwaySpecialistBase):
        """不完整的 mock Specialist 子类，仅实现 load_module。"""

        pathway_class = "INCOMPLETE_MOCK"

        def load_module(self, module_name: str):
            return None
        # 缺少 apply_core / apply_feedback / apply_crosstalk /
        # apply_perturbation / apply_validation

    return _IncompleteSpecialist


def _specialist_hook_mock(state, specialist_instance):
    """模拟 LangGraph specialist hook（测试用，验证 feature flag 隔离）。

    行为：
    - V4_PATHWAY_SPECIALIST_ENABLED=false：直接返回 {}，不调用 specialist
    - V4_PATHWAY_SPECIALIST_ENABLED=true：调用 specialist.apply_core 并返回结果

    真实 hook 在 Task 4.14 graph_v3.py 中实现，本函数仅用于隔离测试。
    """
    from app.config import settings

    if not getattr(settings, "V4_PATHWAY_SPECIALIST_ENABLED", False):
        return {}

    pathway_graph = state.get("v4_pathway_graph", {})
    return {"v4_specialist_core": specialist_instance.apply_core(pathway_graph)}


# =============================================================================
# 测试类
# =============================================================================
class TestSpecialistBaseInterface(unittest.TestCase):
    """测试 1：基类接口完整（6 abstract method + 3 辅助方法）。"""

    def test_base_is_abstract_class(self):
        """PathwaySpecialistBase 是 ABC，不能直接实例化。"""
        from app.pathways.pathway_specialist_base import PathwaySpecialistBase

        with self.assertRaises(TypeError):
            PathwaySpecialistBase()

    def test_six_abstract_methods_defined(self):
        """基类定义 6 个 abstract method。"""
        from app.pathways.pathway_specialist_base import PathwaySpecialistBase

        expected_abstract = frozenset(
            {
                "load_module",
                "apply_core",
                "apply_feedback",
                "apply_crosstalk",
                "apply_perturbation",
                "apply_validation",
            }
        )
        self.assertEqual(
            PathwaySpecialistBase.__abstractmethods__,
            expected_abstract,
        )

    def test_three_helper_methods_exist_and_concrete(self):
        """基类有 3 个辅助方法，且非 abstract。"""
        from app.pathways.pathway_specialist_base import PathwaySpecialistBase

        for helper in ("select_template", "get_metadata", "validate_input"):
            self.assertTrue(
                hasattr(PathwaySpecialistBase, helper),
                f"基类缺少辅助方法: {helper}",
            )
            self.assertNotIn(
                helper,
                PathwaySpecialistBase.__abstractmethods__,
                f"辅助方法 {helper} 不应是 abstract",
            )


class TestSubclassMustImplementAllAbstract(unittest.TestCase):
    """测试 2：子类必须实现所有 abstract method，否则实例化抛 TypeError。"""

    def test_incomplete_subclass_raises_typeerror(self):
        """不完整子类（仅实现 1 个接口）实例化时抛 TypeError。"""
        incomplete_cls = _make_incomplete_mock_specialist()
        with self.assertRaises(TypeError) as ctx:
            incomplete_cls()
        # 验证错误信息提及缺失的 abstract method
        error_msg = str(ctx.exception)
        for missing in (
            "apply_core",
            "apply_feedback",
            "apply_crosstalk",
            "apply_perturbation",
            "apply_validation",
        ):
            self.assertIn(missing, error_msg)

    def test_complete_subclass_instantiable(self):
        """完整子类（实现全部 6 接口）可正常实例化。"""
        complete_cls = _make_complete_mock_specialist()
        instance = complete_cls()
        self.assertEqual(instance.pathway_class, "MOCK_PATHWAY")
        self.assertEqual(instance.display_name, "Mock Pathway for Testing")


class TestPluginRegistry(unittest.TestCase):
    """测试 3 + 4：plugin registry 注册/查询 + registry 初始为空。"""

    def setUp(self):
        """每个测试开始前清空 registry，避免相互污染。"""
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def tearDown(self):
        """每个测试结束后清空 registry，避免污染后续测试。"""
        from app.pathways.pathway_registry import clear_registry

        clear_registry()

    def test_registry_initially_empty(self):
        """测试 4：未注册任何 Specialist 时，list_specialists 返回空列表。"""
        from app.pathways.pathway_registry import (
            get_specialist,
            is_specialist_available,
            list_specialists,
        )

        self.assertEqual(list_specialists(), [])
        self.assertFalse(is_specialist_available("EGFR_RTK"))
        self.assertIsNone(get_specialist("EGFR_RTK"))

    def test_register_and_query_specialist(self):
        """测试 3：用 register_specialist 注册 mock Specialist，验证查询接口。"""
        from app.pathways.pathway_registry import (
            get_specialist,
            is_specialist_available,
            list_specialists,
            register_specialist,
        )

        complete_cls = _make_complete_mock_specialist()

        # 注册前：不存在
        self.assertNotIn("MOCK_PATHWAY", list_specialists())
        self.assertFalse(is_specialist_available("MOCK_PATHWAY"))

        # 通过 register_specialist 装饰器注册
        registered = register_specialist(complete_cls)
        self.assertIs(registered, complete_cls, "装饰器应返回原始类")

        # 注册后：可查询
        self.assertIn("MOCK_PATHWAY", list_specialists())
        self.assertTrue(is_specialist_available("MOCK_PATHWAY"))

        # get_specialist 返回实例（每次新实例）
        instance = get_specialist("MOCK_PATHWAY")
        self.assertIsNotNone(instance)
        self.assertIsInstance(instance, complete_cls)

        # 多次调用返回不同实例（非单例）
        instance2 = get_specialist("MOCK_PATHWAY")
        self.assertIsNot(instance, instance2, "get_specialist 应返回新实例")

    def test_get_specialist_returns_none_for_unregistered(self):
        """未注册的 pathway_class 查询返回 None，不抛异常。"""
        from app.pathways.pathway_registry import get_specialist

        result = get_specialist("NON_EXISTENT_PATHWAY")
        self.assertIsNone(result)

    def test_register_specialist_skips_empty_pathway_class(self):
        """pathway_class 为空的子类注册时跳过（记录 warning，不抛异常）。"""
        from app.pathways.pathway_registry import (
            list_specialists,
            register_specialist,
        )
        from app.pathways.pathway_specialist_base import PathwaySpecialistBase

        class _EmptyPathwayClass(PathwaySpecialistBase):
            """pathway_class 未覆写，应被跳过。"""

            def load_module(self, module_name):
                return None

            def apply_core(self, pathway_graph, ontology_entities=None):
                return {"species": [], "reactions": []}

            def apply_feedback(self, pathway_graph):
                return []

            def apply_crosstalk(self, pathway_graph, crosstalk_edges):
                return []

            def apply_perturbation(self, pathway_graph, perturbation_points):
                return []

            def apply_validation(self, simulation_result=None):
                return []

        # pathway_class="" 默认值，注册应跳过
        register_specialist(_EmptyPathwayClass)
        self.assertEqual(list_specialists(), [])


class TestSelectTemplateMapping(unittest.TestCase):
    """测试 5：select_template 机制→模板名映射。"""

    def test_phosphorylation_mapping(self):
        """phosphorylation → _mechanism_phosphorylation_mm。"""
        complete_cls = _make_complete_mock_specialist()
        instance = complete_cls()
        self.assertEqual(
            instance.select_template("phosphorylation"),
            "_mechanism_phosphorylation_mm",
        )

    def test_bistable_mapping(self):
        """bistable → bistable_switch。"""
        complete_cls = _make_complete_mock_specialist()
        instance = complete_cls()
        self.assertEqual(instance.select_template("bistable"), "bistable_switch")

    def test_oscillatory_mapping(self):
        """oscillatory → oscillatory_feedback。"""
        complete_cls = _make_complete_mock_specialist()
        instance = complete_cls()
        self.assertEqual(
            instance.select_template("oscillatory"), "oscillatory_feedback"
        )

    def test_unknown_mechanism_returns_default(self):
        """未匹配的 mechanism 返回 'default'（降级处理）。"""
        complete_cls = _make_complete_mock_specialist()
        instance = complete_cls()
        self.assertEqual(instance.select_template("unknown_mechanism"), "default")


class TestModuleDataclasses(unittest.TestCase):
    """测试 6：5 模块 dataclass 字段完整 + 可实例化。"""

    def test_core_module_data_fields(self):
        """CoreModuleData 含 species / reactions / kinetics_overrides 字段。"""
        from app.pathways.pathway_modules.core.template import CoreModuleData

        data = CoreModuleData()
        self.assertEqual(data.species, [])
        self.assertEqual(data.reactions, [])
        self.assertEqual(data.kinetics_overrides, {})

        # 可填充字段
        data.species.append({"name": "EGF"})
        data.reactions.append({"id": "R1"})
        data.kinetics_overrides["k1"] = 0.01
        self.assertEqual(data.species[0]["name"], "EGF")
        self.assertEqual(data.reactions[0]["id"], "R1")
        self.assertEqual(data.kinetics_overrides["k1"], 0.01)

    def test_feedback_module_data_fields(self):
        """FeedbackModuleData 含 feedback_loops / delay_minutes / loop_type。"""
        from app.pathways.pathway_modules.feedback.template import FeedbackModuleData

        data = FeedbackModuleData()
        self.assertEqual(data.feedback_loops, [])
        self.assertEqual(data.delay_minutes, 0.0)
        self.assertEqual(data.loop_type, "negative")

        # 可填充字段
        data.feedback_loops.append({"loop_id": "FL1"})
        data.delay_minutes = 60.0
        data.loop_type = "positive"
        self.assertEqual(data.feedback_loops[0]["loop_id"], "FL1")
        self.assertEqual(data.delay_minutes, 60.0)
        self.assertEqual(data.loop_type, "positive")

    def test_crosstalk_module_data_fields(self):
        """CrosstalkModuleData 含 crosstalk_reactions / shared_species / coordination_strategy。"""
        from app.pathways.pathway_modules.crosstalk.template import CrosstalkModuleData

        data = CrosstalkModuleData()
        self.assertEqual(data.crosstalk_reactions, [])
        self.assertEqual(data.shared_species, [])
        self.assertEqual(data.coordination_strategy, "merge")

        # 可填充字段
        data.crosstalk_reactions.append({"id": "CT1"})
        data.shared_species.append("Ras")
        data.coordination_strategy = "alias"
        self.assertEqual(data.crosstalk_reactions[0]["id"], "CT1")
        self.assertEqual(data.shared_species[0], "Ras")
        self.assertEqual(data.coordination_strategy, "alias")

    def test_perturbation_module_data_fields(self):
        """PerturbationModuleData 含 perturbation_reactions / drug_targets / ko_targets。"""
        from app.pathways.pathway_modules.perturbation.template import (
            PerturbationModuleData,
        )

        data = PerturbationModuleData()
        self.assertEqual(data.perturbation_reactions, [])
        self.assertEqual(data.drug_targets, [])
        self.assertEqual(data.ko_targets, [])

        # 可填充字段
        data.perturbation_reactions.append({"id": "P1"})
        data.drug_targets.append({"target": "EGFR", "drug_name": "Gefitinib"})
        data.ko_targets.append({"target_gene": "KRAS"})
        self.assertEqual(data.perturbation_reactions[0]["id"], "P1")
        self.assertEqual(data.drug_targets[0]["drug_name"], "Gefitinib")
        self.assertEqual(data.ko_targets[0]["target_gene"], "KRAS")

    def test_validation_module_data_fields(self):
        """ValidationModuleData 含 rules / benchmarks / tolerances / pmid_references。"""
        from app.pathways.pathway_modules.validation.template import ValidationModuleData

        data = ValidationModuleData()
        self.assertEqual(data.rules, [])
        self.assertEqual(data.benchmarks, [])
        self.assertEqual(data.tolerances, {})
        self.assertEqual(data.pmid_references, [])

        # 可填充字段
        data.rules.append({"metric_name": "pEGFR_peak_time_min"})
        data.benchmarks.append({"benchmark_name": "pEGFR_5-10min"})
        data.tolerances["pEGFR_peak_time_min"] = 2.5
        data.pmid_references.append({"pmid": "12345678"})
        self.assertEqual(data.rules[0]["metric_name"], "pEGFR_peak_time_min")
        self.assertEqual(data.benchmarks[0]["benchmark_name"], "pEGFR_5-10min")
        self.assertEqual(data.tolerances["pEGFR_peak_time_min"], 2.5)
        self.assertEqual(data.pmid_references[0]["pmid"], "12345678")

    def test_module_data_independent_instances(self):
        """每个 dataclass 实例的字段独立（default_factory 不共享可变默认值）。"""
        from app.pathways.pathway_modules.core.template import CoreModuleData

        data1 = CoreModuleData()
        data2 = CoreModuleData()
        data1.species.append({"name": "X"})
        self.assertEqual(data1.species, [{"name": "X"}])
        self.assertEqual(data2.species, [], "实例间不应共享可变默认值")


class TestFeatureFlagIsolation(unittest.TestCase):
    """测试 7：Feature Flag 隔离（V4_PATHWAY_SPECIALIST_ENABLED=false 时 hook 不调用 Specialist）。"""

    def test_flag_false_hook_returns_empty_and_does_not_call_specialist(self):
        """flag=false 时 hook 返回空 dict 且不调用 Specialist 任何方法。"""
        mock_specialist = MagicMock()
        mock_specialist.apply_core.return_value = {"species": [], "reactions": []}
        state = {"v4_pathway_graph": {"nodes": [], "edges": []}}

        with patch("app.config.settings.V4_PATHWAY_SPECIALIST_ENABLED", False):
            result = _specialist_hook_mock(state, mock_specialist)

        self.assertEqual(result, {})
        mock_specialist.apply_core.assert_not_called()

    def test_flag_true_hook_calls_specialist(self):
        """flag=true 时 hook 调用 Specialist.apply_core 并返回结果。"""
        mock_specialist = MagicMock()
        expected_core = {"species": [{"name": "EGF"}], "reactions": []}
        mock_specialist.apply_core.return_value = expected_core
        state = {"v4_pathway_graph": {"nodes": [{"id": "EGF"}], "edges": []}}

        with patch("app.config.settings.V4_PATHWAY_SPECIALIST_ENABLED", True):
            result = _specialist_hook_mock(state, mock_specialist)

        self.assertEqual(result, {"v4_specialist_core": expected_core})
        mock_specialist.apply_core.assert_called_once_with(
            {"nodes": [{"id": "EGF"}], "edges": []}
        )


class TestConfigFlagDefault(unittest.TestCase):
    """测试 8：config flag 默认 false。"""

    def test_flag_default_false(self):
        """settings.V4_PATHWAY_SPECIALIST_ENABLED 默认为 False。

        前提假设：测试环境未设置 V4_PATHWAY_SPECIALIST_ENABLED=true 环境变量。
        若 CI 环境显式设置该变量为 true，本测试应跳过。
        """
        from app.config import settings

        self.assertFalse(
            settings.V4_PATHWAY_SPECIALIST_ENABLED,
            "V4_PATHWAY_SPECIALIST_ENABLED 应默认为 False（铁律：flag=false 时 v3 行为不变）",
        )

    def test_flag_field_exists_in_settings(self):
        """settings 对象含 V4_PATHWAY_SPECIALIST_ENABLED 字段。"""
        from app.config import settings

        self.assertTrue(
            hasattr(settings, "V4_PATHWAY_SPECIALIST_ENABLED"),
            "settings 必须含 V4_PATHWAY_SPECIALIST_ENABLED 字段",
        )


class TestGetMetadataAndValidateInput(unittest.TestCase):
    """补充测试：get_metadata / validate_input 辅助方法行为。"""

    def test_get_metadata_returns_complete_dict(self):
        """get_metadata 返回含 4 个字段的 dict。"""
        complete_cls = _make_complete_mock_specialist()
        instance = complete_cls()
        metadata = instance.get_metadata()

        self.assertEqual(metadata["pathway_class"], "MOCK_PATHWAY")
        self.assertEqual(metadata["display_name"], "Mock Pathway for Testing")
        self.assertEqual(
            metadata["supported_modules"],
            ["core", "feedback", "crosstalk", "perturbation", "validation"],
        )
        self.assertEqual(metadata["version"], "v4.2")

    def test_validate_input_empty_graph_returns_warning(self):
        """空 pathway_graph 返回 warning。"""
        complete_cls = _make_complete_mock_specialist()
        instance = complete_cls()
        warnings = instance.validate_input({})
        self.assertEqual(len(warnings), 1)
        self.assertIn("pathway_graph", warnings[0])

    def test_validate_input_none_graph_returns_warning(self):
        """None pathway_graph 返回 warning。"""
        complete_cls = _make_complete_mock_specialist()
        instance = complete_cls()
        warnings = instance.validate_input(None)
        self.assertEqual(len(warnings), 1)

    def test_validate_input_valid_graph_returns_no_warnings(self):
        """含 nodes + edges 的 pathway_graph 无 warning。"""
        complete_cls = _make_complete_mock_specialist()
        instance = complete_cls()
        warnings = instance.validate_input(
            {"nodes": [{"id": "EGF"}], "edges": []}
        )
        self.assertEqual(warnings, [])

    def test_validate_input_missing_nodes_returns_warning(self):
        """缺少 nodes 字段返回 warning。"""
        complete_cls = _make_complete_mock_specialist()
        instance = complete_cls()
        warnings = instance.validate_input({"edges": []})
        self.assertEqual(len(warnings), 1)
        self.assertIn("nodes", warnings[0])

    def test_supported_modules_default_all_five(self):
        """基类 supported_modules 默认含全部 5 模块。"""
        complete_cls = _make_complete_mock_specialist()
        instance = complete_cls()
        self.assertEqual(
            instance.supported_modules,
            ["core", "feedback", "crosstalk", "perturbation", "validation"],
        )


if __name__ == "__main__":
    unittest.main()
