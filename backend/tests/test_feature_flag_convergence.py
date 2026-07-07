# BioDynamics Agent v4 - Feature Flag Convergence Tests (Task B.3)
#
# 验证 13 个细粒度 V4 flag 收敛为 3 个粗粒度 flag 的聚合逻辑：
# 1. 三个粗粒度 flag 默认 OFF
# 2. V4_SCIENTIFIC_LAYER_ENABLED=true 启用 P1-P4 所有 effective_* 方法
# 3. V4_VALIDATION_ENABLED=true 启用 P5 所有 effective_* 方法
# 4. V4_HYPOTHESIS_ENABLED=true 启用 P6 所有 effective_* 方法
# 5. 三个粗粒度全 OFF → 所有 effective_* 返回 False（v3 等价行为，无 hook 触发）
# 6. 细粒度 env override 优先于粗粒度（coarse=ON + fine env=OFF → effective=OFF）
# 7. 粗粒度 OFF + 细粒度属性 ON → effective=ON（向后兼容 @patch 测试）
#
# 注意：Settings 类的类属性使用 os.getenv() 作为默认值，在模块导入时求值。
# 因此测试通过 patch.object 设置实例属性 + patch.dict 设置 env 变量来模拟
# 不同配置场景，而非依赖 Settings() 重新读取 env。
#
# 运行：cd backend && python -m pytest tests/test_feature_flag_convergence.py -v

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# 确保能导入 app 包（backend/ 在 sys.path）
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


# =============================================================================
# 测试常量：所有 v4 相关 env 变量名（3 粗 + 13 细）
# =============================================================================
ALL_V4_ENV_KEYS = [
    # 粗粒度
    "V4_SCIENTIFIC_LAYER_ENABLED",
    "V4_VALIDATION_ENABLED",
    "V4_HYPOTHESIS_ENABLED",
    # 细粒度 P1-P4（8 个）
    "V4_ONTOLOGY_AGENT_ENABLED",
    "V4_PATHWAY_GRAPH_ENABLED",
    "V4_REACTION_IR_ENABLED",
    "V4_REACTION_IR_ADAPTER_ENABLED",
    "V4_ODE_TEMPLATE_V2_ENABLED",
    "V4_PATHWAY_PLANNER_ENABLED",
    "V4_PATHWAY_SPECIALIST_ENABLED",
    "V4_CROSSTALK_COORDINATOR_ENABLED",
    # 细粒度 P5（3 个）
    "V4_SBML_GROUNDER_ENABLED",
    "V4_VALIDATION_PYRAMID_ENABLED",
    "V4_CALIBRATION_AGENT_ENABLED",
    # 细粒度 P6（2 个）
    "V4_HYPOTHESIS_AGENT_ENABLED",
    "V4_DYNAMIC_ROUTING_ENABLED",
]


def _clean_v4_env() -> dict[str, str | None]:
    """清除所有 v4 env 变量，返回被清除的原值（用于恢复）。

    Returns:
        {env_key: original_value_or_None} 映射，调用方可在 tearDown 中恢复。
    """
    saved: dict[str, str | None] = {}
    for key in ALL_V4_ENV_KEYS:
        if key in os.environ:
            saved[key] = os.environ.pop(key)
        else:
            saved[key] = None
    return saved


def _restore_v4_env(saved: dict[str, str | None]) -> None:
    """恢复被 _clean_v4_env 清除的 env 变量。"""
    for key, val in saved.items():
        if val is not None:
            os.environ[key] = val
        elif key in os.environ:
            del os.environ[key]


class TestCoarseFlagsDefaultOff(unittest.TestCase):
    """验证 1：三个粗粒度 flag 默认 OFF。"""

    def setUp(self):
        self._saved = _clean_v4_env()

    def tearDown(self):
        _restore_v4_env(self._saved)

    def test_all_coarse_flags_default_false(self):
        """无 env 设置时，三个粗粒度 flag 默认值均为 False。"""
        from app.config import Settings

        s = Settings()
        self.assertFalse(s.V4_SCIENTIFIC_LAYER_ENABLED)
        self.assertFalse(s.V4_VALIDATION_ENABLED)
        self.assertFalse(s.V4_HYPOTHESIS_ENABLED)


class TestAllOffIsV3Behavior(unittest.TestCase):
    """验证 5：三个粗粒度全 OFF → 所有 effective_* 返回 False（v3 等价）。"""

    def setUp(self):
        self._saved = _clean_v4_env()
        from app.config import Settings

        self.settings = Settings()

    def tearDown(self):
        _restore_v4_env(self._saved)

    def test_all_effective_methods_false_when_all_off(self):
        """粗粒度全 OFF + 细粒度全 OFF → 所有 13 个 effective_* 返回 False。"""
        s = self.settings
        # P1-P4 科学层（8 个）
        self.assertFalse(s.effective_v4_ontology_enabled())
        self.assertFalse(s.effective_v4_pathway_graph_enabled())
        self.assertFalse(s.effective_v4_reaction_ir_enabled())
        self.assertFalse(s.effective_v4_reaction_ir_adapter_enabled())
        self.assertFalse(s.effective_v4_ode_template_v2_enabled())
        self.assertFalse(s.effective_v4_pathway_planner_enabled())
        self.assertFalse(s.effective_v4_pathway_specialist_enabled())
        self.assertFalse(s.effective_v4_crosstalk_coordinator_enabled())
        # P5 验证层（3 个）
        self.assertFalse(s.effective_v4_sbml_grounder_enabled())
        self.assertFalse(s.effective_v4_validation_pyramid_enabled())
        self.assertFalse(s.effective_v4_calibration_agent_enabled())
        # P6 假设层（2 个）
        self.assertFalse(s.effective_v4_hypothesis_enabled())
        self.assertFalse(s.effective_v4_dynamic_routing_enabled())

    def test_v3_hooks_return_empty_when_all_off(self):
        """粗粒度全 OFF 时关键 hook 返回 {}（v3 等价行为，无 v4 副作用）。

        覆盖 P1-P6 各层代表性 hook，验证 flag=false 短路返回 {}。
        """
        # P1-P4 hooks
        from app.ontology.ontology_agent import ontology_hook_node
        from app.pathways.pathway_planner import pathway_planner_hook_node
        from app.pathways.specialist_hook import specialist_hook_node
        from app.crosstalk.coordinator import crosstalk_coordinator_hook_node
        # P5 hooks
        from app.sbml_grounder.grounder_agent import sbml_grounder_hook_node
        from app.validation_v2.validation_agent import validation_pyramid_hook_node
        from app.calibration.calibration_agent import calibration_hook_node
        # P6 hooks
        from app.agent_orchestration.dynamic_router import dynamic_router_hook_node
        from app.hypothesis.hypothesis_agent import hypothesis_agent_hook_node

        state = {
            "user_input": "EGF binds EGFR",
            "network_json": {"nodes": [], "edges": []},
            "entities": [],
            "mechanism": {},
        }

        # 用 patch 把全局 settings 的粗粒度 flag 全部置 False
        with patch(
            "app.config.settings.V4_SCIENTIFIC_LAYER_ENABLED", False
        ), patch(
            "app.config.settings.V4_VALIDATION_ENABLED", False
        ), patch(
            "app.config.settings.V4_HYPOTHESIS_ENABLED", False
        ):
            # P1-P4 hooks
            self.assertEqual(ontology_hook_node(state), {})
            self.assertEqual(pathway_planner_hook_node(state), {})
            self.assertEqual(specialist_hook_node(state), {})
            self.assertEqual(crosstalk_coordinator_hook_node(state), {})
            # P5 hooks
            self.assertEqual(sbml_grounder_hook_node(state), {})
            self.assertEqual(validation_pyramid_hook_node(state), {})
            self.assertEqual(calibration_hook_node(state), {})
            # P6 hooks
            self.assertEqual(dynamic_router_hook_node(state), {})
            self.assertEqual(hypothesis_agent_hook_node(state), {})

    def test_v3_validation_level_hooks_return_empty_when_all_off(self):
        """粗粒度全 OFF 时 Validation Pyramid 5 个 Level hook 均返回 {}。"""
        from app.validation_v2.level1_internal import level1_hook_node
        from app.validation_v2.level2_sbml import level2_hook_node
        from app.validation_v2.level3_crosstalk import level3_hook_node
        from app.validation_v2.level4_benchmark import level4_hook_node
        from app.validation_v2.level5_hypothesis import level5_hook_node

        state = {
            "user_input": "EGF binds EGFR",
            "network_json": {"nodes": [], "edges": []},
        }

        with patch(
            "app.config.settings.V4_VALIDATION_ENABLED", False
        ), patch(
            "app.config.settings.V4_HYPOTHESIS_ENABLED", False
        ):
            self.assertEqual(level1_hook_node(state), {})
            self.assertEqual(level2_hook_node(state), {})
            self.assertEqual(level3_hook_node(state), {})
            self.assertEqual(level4_hook_node(state), {})
            self.assertEqual(level5_hook_node(state), {})

    def test_v3_agents_v4_stubs_return_empty_when_all_off(self):
        """粗粒度全 OFF 时 agents_v4 4 个 stub agent 均返回 {}。"""
        from app.agents_v4.ode_builder import ODEBuilderAgent
        from app.agents_v4.mechanism_builder import MechanismBuilderAgent
        from app.agents_v4.simulation_planner import SimulationPlannerAgent
        from app.agents_v4.parameter_agent import ParameterAgent

        state = {
            "user_input": "EGF binds EGFR",
            "network_json": {"nodes": [], "edges": []},
        }

        with patch(
            "app.config.settings.V4_HYPOTHESIS_ENABLED", False
        ):
            self.assertEqual(ODEBuilderAgent().build(state), {})
            self.assertEqual(MechanismBuilderAgent().build(state), {})
            self.assertEqual(SimulationPlannerAgent().plan(state), {})
            self.assertEqual(ParameterAgent().manage(state), {})


class TestScientificLayerCoarseOn(unittest.TestCase):
    """验证 2：V4_SCIENTIFIC_LAYER_ENABLED=true 启用 P1-P4 所有 effective_*。"""

    def setUp(self):
        self._saved = _clean_v4_env()
        from app.config import Settings

        self.settings = Settings()

    def tearDown(self):
        _restore_v4_env(self._saved)

    def test_p1_p4_effective_methods_all_true(self):
        """粗粒度 SCIENTIFIC_LAYER=ON → P1-P4 的 8 个 effective_* 全部 True。"""
        s = self.settings
        with patch.object(s, "V4_SCIENTIFIC_LAYER_ENABLED", True):
            self.assertTrue(s.effective_v4_ontology_enabled())
            self.assertTrue(s.effective_v4_pathway_graph_enabled())
            self.assertTrue(s.effective_v4_reaction_ir_enabled())
            self.assertTrue(s.effective_v4_reaction_ir_adapter_enabled())
            self.assertTrue(s.effective_v4_ode_template_v2_enabled())
            self.assertTrue(s.effective_v4_pathway_planner_enabled())
            self.assertTrue(s.effective_v4_pathway_specialist_enabled())
            self.assertTrue(s.effective_v4_crosstalk_coordinator_enabled())

    def test_p5_p6_still_off_when_only_scientific_on(self):
        """粗粒度 SCIENTIFIC_LAYER=ON 但 VALIDATION/HYPOTHESIS=OFF → P5/P6 仍 False。"""
        s = self.settings
        with patch.object(s, "V4_SCIENTIFIC_LAYER_ENABLED", True):
            self.assertFalse(s.effective_v4_sbml_grounder_enabled())
            self.assertFalse(s.effective_v4_validation_pyramid_enabled())
            self.assertFalse(s.effective_v4_calibration_agent_enabled())
            self.assertFalse(s.effective_v4_hypothesis_enabled())
            self.assertFalse(s.effective_v4_dynamic_routing_enabled())


class TestValidationCoarseOn(unittest.TestCase):
    """验证 3：V4_VALIDATION_ENABLED=true 启用 P5 所有 effective_*。"""

    def setUp(self):
        self._saved = _clean_v4_env()
        from app.config import Settings

        self.settings = Settings()

    def tearDown(self):
        _restore_v4_env(self._saved)

    def test_p5_effective_methods_all_true(self):
        """粗粒度 VALIDATION=ON → P5 的 3 个 effective_* 全部 True。"""
        s = self.settings
        with patch.object(s, "V4_VALIDATION_ENABLED", True):
            self.assertTrue(s.effective_v4_sbml_grounder_enabled())
            self.assertTrue(s.effective_v4_validation_pyramid_enabled())
            self.assertTrue(s.effective_v4_calibration_agent_enabled())

    def test_p1_p4_p6_still_off_when_only_validation_on(self):
        """粗粒度 VALIDATION=ON 但 SCIENTIFIC/HYPOTHESIS=OFF → P1-P4/P6 仍 False。"""
        s = self.settings
        with patch.object(s, "V4_VALIDATION_ENABLED", True):
            self.assertFalse(s.effective_v4_ontology_enabled())
            self.assertFalse(s.effective_v4_pathway_planner_enabled())
            self.assertFalse(s.effective_v4_crosstalk_coordinator_enabled())
            self.assertFalse(s.effective_v4_hypothesis_enabled())
            self.assertFalse(s.effective_v4_dynamic_routing_enabled())


class TestHypothesisCoarseOn(unittest.TestCase):
    """验证 4：V4_HYPOTHESIS_ENABLED=true 启用 P6 所有 effective_*。"""

    def setUp(self):
        self._saved = _clean_v4_env()
        from app.config import Settings

        self.settings = Settings()

    def tearDown(self):
        _restore_v4_env(self._saved)

    def test_p6_effective_methods_all_true(self):
        """粗粒度 HYPOTHESIS=ON → P6 的 2 个 effective_* 全部 True。"""
        s = self.settings
        with patch.object(s, "V4_HYPOTHESIS_ENABLED", True):
            self.assertTrue(s.effective_v4_hypothesis_enabled())
            self.assertTrue(s.effective_v4_dynamic_routing_enabled())

    def test_p1_p5_still_off_when_only_hypothesis_on(self):
        """粗粒度 HYPOTHESIS=ON 但 SCIENTIFIC/VALIDATION=OFF → P1-P4/P5 仍 False。"""
        s = self.settings
        with patch.object(s, "V4_HYPOTHESIS_ENABLED", True):
            self.assertFalse(s.effective_v4_ontology_enabled())
            self.assertFalse(s.effective_v4_pathway_planner_enabled())
            self.assertFalse(s.effective_v4_validation_pyramid_enabled())
            self.assertFalse(s.effective_v4_sbml_grounder_enabled())


class TestAllCoarseOn(unittest.TestCase):
    """补充验证：三个粗粒度全 ON → 所有 13 个 effective_* 全部 True。"""

    def setUp(self):
        self._saved = _clean_v4_env()
        from app.config import Settings

        self.settings = Settings()

    def tearDown(self):
        _restore_v4_env(self._saved)

    def test_all_13_effective_methods_true(self):
        """三个粗粒度全 ON → 所有 13 个 effective_* 全部 True。"""
        s = self.settings
        with patch.object(s, "V4_SCIENTIFIC_LAYER_ENABLED", True), \
             patch.object(s, "V4_VALIDATION_ENABLED", True), \
             patch.object(s, "V4_HYPOTHESIS_ENABLED", True):
            # P1-P4（8 个）
            self.assertTrue(s.effective_v4_ontology_enabled())
            self.assertTrue(s.effective_v4_pathway_graph_enabled())
            self.assertTrue(s.effective_v4_reaction_ir_enabled())
            self.assertTrue(s.effective_v4_reaction_ir_adapter_enabled())
            self.assertTrue(s.effective_v4_ode_template_v2_enabled())
            self.assertTrue(s.effective_v4_pathway_planner_enabled())
            self.assertTrue(s.effective_v4_pathway_specialist_enabled())
            self.assertTrue(s.effective_v4_crosstalk_coordinator_enabled())
            # P5（3 个）
            self.assertTrue(s.effective_v4_sbml_grounder_enabled())
            self.assertTrue(s.effective_v4_validation_pyramid_enabled())
            self.assertTrue(s.effective_v4_calibration_agent_enabled())
            # P6（2 个）
            self.assertTrue(s.effective_v4_hypothesis_enabled())
            self.assertTrue(s.effective_v4_dynamic_routing_enabled())


class TestFineGrainedEnvOverride(unittest.TestCase):
    """验证 6：细粒度 env override 优先于粗粒度（debug override）。

    场景：粗粒度 ON + 细粒度 env=OFF → effective=OFF
    生产环境用粗粒度 ON 启用整层，但可通过 env 显式关闭某个细粒度 hook 用于调试。
    _resolve_v4_flag 规则 1 在调用时检查 os.environ，因此 patch.dict 可生效。
    """

    def setUp(self):
        self._saved = _clean_v4_env()
        from app.config import Settings

        self.settings = Settings()

    def tearDown(self):
        _restore_v4_env(self._saved)

    def test_scientific_on_but_ontology_env_off(self):
        """SCIENTIFIC=ON + V4_ONTOLOGY_AGENT_ENABLED=false → ontology effective=False。"""
        s = self.settings
        with patch.object(s, "V4_SCIENTIFIC_LAYER_ENABLED", True), \
             patch.dict(os.environ, {"V4_ONTOLOGY_AGENT_ENABLED": "false"}):
            # ontology 被 env override 关闭
            self.assertFalse(s.effective_v4_ontology_enabled())
            # 其他 P1-P4 hook 仍跟随粗粒度 ON（未在 env 设置）
            self.assertTrue(s.effective_v4_pathway_graph_enabled())
            self.assertTrue(s.effective_v4_reaction_ir_enabled())
            self.assertTrue(s.effective_v4_pathway_planner_enabled())
            self.assertTrue(s.effective_v4_crosstalk_coordinator_enabled())

    def test_validation_on_but_validation_pyramid_env_off(self):
        """VALIDATION=ON + V4_VALIDATION_PYRAMID_ENABLED=false → validation_pyramid=False。"""
        s = self.settings
        with patch.object(s, "V4_VALIDATION_ENABLED", True), \
             patch.dict(os.environ, {"V4_VALIDATION_PYRAMID_ENABLED": "false"}):
            self.assertFalse(s.effective_v4_validation_pyramid_enabled())
            # 其他 P5 hook 仍跟随粗粒度 ON
            self.assertTrue(s.effective_v4_sbml_grounder_enabled())
            self.assertTrue(s.effective_v4_calibration_agent_enabled())

    def test_hypothesis_on_but_dynamic_routing_env_off(self):
        """HYPOTHESIS=ON + V4_DYNAMIC_ROUTING_ENABLED=false → dynamic_routing=False。"""
        s = self.settings
        with patch.object(s, "V4_HYPOTHESIS_ENABLED", True), \
             patch.dict(os.environ, {"V4_DYNAMIC_ROUTING_ENABLED": "false"}):
            self.assertFalse(s.effective_v4_dynamic_routing_enabled())
            # hypothesis agent 仍跟随粗粒度 ON
            self.assertTrue(s.effective_v4_hypothesis_enabled())

    def test_fine_env_true_overrides_coarse_off(self):
        """粗粒度 OFF + 细粒度 env=true → effective=True（env override 双向生效）。"""
        s = self.settings
        # 粗粒度保持 OFF，仅设置细粒度 env
        with patch.dict(os.environ, {"V4_ONTOLOGY_AGENT_ENABLED": "true"}):
            # 粗粒度 OFF，但 env 显式设置细粒度 ON → effective ON
            self.assertTrue(s.effective_v4_ontology_enabled())
            # 其他未在 env 设置的细粒度仍为 False
            self.assertFalse(s.effective_v4_pathway_graph_enabled())


class TestBackwardCompatFineAttrPatch(unittest.TestCase):
    """验证 7：粗粒度 OFF + 细粒度属性 ON（@patch）→ effective=ON。

    向后兼容 test_p4_flag_off_isolation.py 等旧测试，这些测试通过
    @patch("app.config.settings.V4_X_ENABLED", False) 直接 patch 细粒度属性。
    当粗粒度 OFF 时，effective_* 跟随细粒度属性值（规则 3）。
    """

    def setUp(self):
        self._saved = _clean_v4_env()
        from app.config import Settings

        self.settings = Settings()

    def tearDown(self):
        _restore_v4_env(self._saved)

    def test_coarse_off_fine_attr_false_returns_false(self):
        """粗粒度 OFF + 细粒度属性=False（@patch False）→ effective=False。

        模拟 test_p4_flag_off_isolation.py 的 @patch(..., False) 场景。
        """
        s = self.settings
        # 默认细粒度属性均为 False
        self.assertFalse(s.V4_SCIENTIFIC_LAYER_ENABLED)
        # patch 细粒度属性为 False（与默认值相同，模拟旧测试）
        with patch.object(s, "V4_PATHWAY_PLANNER_ENABLED", False):
            self.assertFalse(s.effective_v4_pathway_planner_enabled())
        with patch.object(s, "V4_PATHWAY_SPECIALIST_ENABLED", False):
            self.assertFalse(s.effective_v4_pathway_specialist_enabled())
        with patch.object(s, "V4_CROSSTALK_COORDINATOR_ENABLED", False):
            self.assertFalse(s.effective_v4_crosstalk_coordinator_enabled())

    def test_coarse_off_fine_attr_true_returns_true(self):
        """粗粒度 OFF + 细粒度属性=True（@patch True）→ effective=True。

        验证规则 3：粗粒度 OFF 时跟随细粒度属性，允许单独启用某个 hook。
        """
        s = self.settings
        self.assertFalse(s.V4_SCIENTIFIC_LAYER_ENABLED)
        # patch 细粒度属性为 True → effective True（向后兼容单独启用）
        with patch.object(s, "V4_PATHWAY_PLANNER_ENABLED", True):
            self.assertTrue(s.effective_v4_pathway_planner_enabled())
        with patch.object(s, "V4_ONTOLOGY_AGENT_ENABLED", True):
            self.assertTrue(s.effective_v4_ontology_enabled())
        with patch.object(s, "V4_VALIDATION_PYRAMID_ENABLED", True):
            self.assertTrue(s.effective_v4_validation_pyramid_enabled())
        with patch.object(s, "V4_HYPOTHESIS_AGENT_ENABLED", True):
            self.assertTrue(s.effective_v4_hypothesis_enabled())

    def test_p4_isolation_pattern_still_works(self):
        """验证 test_p4_flag_off_isolation.py 的 @patch 模式仍然有效。

        该测试通过 @patch("app.config.settings.V4_X_ENABLED", False) 关闭 hook，
        本测试验证此模式在收敛后仍能正确让 hook 返回 {}。
        """
        from app.pathways.pathway_planner import pathway_planner_hook_node
        from app.pathways.specialist_hook import specialist_hook_node
        from app.crosstalk.coordinator import crosstalk_coordinator_hook_node

        state = {
            "user_input": "EGF binds EGFR",
            "network_json": {"nodes": [], "edges": []},
            "entities": [],
            "mechanism": {},
        }

        # 模拟 test_p4_flag_off_isolation.py 的 @patch 装饰器
        with patch(
            "app.config.settings.V4_PATHWAY_PLANNER_ENABLED", False
        ), patch(
            "app.config.settings.V4_PATHWAY_SPECIALIST_ENABLED", False
        ), patch(
            "app.config.settings.V4_CROSSTALK_COORDINATOR_ENABLED", False
        ):
            # 粗粒度 OFF（默认）+ 细粒度属性 False → effective False → hook 返回 {}
            self.assertEqual(pathway_planner_hook_node(state), {})
            self.assertEqual(specialist_hook_node(state), {})
            self.assertEqual(crosstalk_coordinator_hook_node(state), {})


class TestResolutionPriorityOrder(unittest.TestCase):
    """补充验证：env override > coarse ON > fine attr（优先级从高到低）。"""

    def setUp(self):
        self._saved = _clean_v4_env()
        from app.config import Settings

        self.settings = Settings()

    def tearDown(self):
        _restore_v4_env(self._saved)

    def test_env_override_beats_coarse_on(self):
        """规则 1 > 规则 2：env 显式 false 优先于 coarse=true。"""
        s = self.settings
        with patch.object(s, "V4_SCIENTIFIC_LAYER_ENABLED", True), \
             patch.dict(os.environ, {"V4_PATHWAY_PLANNER_ENABLED": "false"}):
            # coarse=True 但 env=false → effective=false
            self.assertTrue(s.V4_SCIENTIFIC_LAYER_ENABLED)
            self.assertFalse(s.effective_v4_pathway_planner_enabled())

    def test_coarse_on_beats_fine_attr_off(self):
        """规则 2 > 规则 3：coarse=true 优先于 fine_attr=false。"""
        s = self.settings
        with patch.object(s, "V4_SCIENTIFIC_LAYER_ENABLED", True):
            # coarse=True, fine_attr=False（默认）→ effective=True
            self.assertTrue(s.V4_SCIENTIFIC_LAYER_ENABLED)
            self.assertFalse(s.V4_PATHWAY_PLANNER_ENABLED)
            self.assertTrue(s.effective_v4_pathway_planner_enabled())

    def test_fine_attr_used_when_coarse_off(self):
        """规则 3：coarse=false 时 effective 跟随 fine_attr。"""
        s = self.settings
        # coarse=False, fine_attr=False → effective=False
        self.assertFalse(s.V4_SCIENTIFIC_LAYER_ENABLED)
        self.assertFalse(s.V4_PATHWAY_PLANNER_ENABLED)
        self.assertFalse(s.effective_v4_pathway_planner_enabled())

        # patch fine_attr=True → effective=True
        with patch.object(s, "V4_PATHWAY_PLANNER_ENABLED", True):
            self.assertTrue(s.effective_v4_pathway_planner_enabled())


if __name__ == "__main__":
    unittest.main()
