# BioDynamics Agent v4 - Dynamic Router 单元测试（Phase 6 / Task 6.5.6）
#
# 覆盖 SubTask 6.5.1-6.5.6：
# - 6.5.1: DynamicRouter + dynamic_router_hook_node 创建
# - 6.5.2: AGENT_REGISTRY_V4 13 Agent 注册（Ontology/Planner/Specialist Group/
#          Coordinator/ReactionBuilder/MechanismBuilder/ODEBuilder/SBMLGrounder/
#          Calibration/SimulationPlanner/Validation/Hypothesis/ParameterAgent）
# - 6.5.3: PathwayClassDispatcher 基于 v4_pathway_class 分支
#          （单通路→1 Specialist；多通路→N+Coordinator）
# - 6.5.4: FailSafeDispatcher 失败短路 + 超时 30s 回退 + 最大深度 10 + visited set 防环
# - 6.5.6: 本单元测试
#
# 测试策略：
# - mock settings.V4_DYNAMIC_ROUTING_ENABLED / 各 Feature Flag
# - 不调用真实 LLM / 真实 RAG / 真实 SBML 仿真
# - 使用快速超时配置（timeout_seconds=1）测试超时回退
# - 使用 mock Specialist 测试 PathwayClassDispatcher（避免依赖真实 10 Specialist）

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# 确保能导入 app 包（backend/ 在 sys.path）
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.agent_orchestration import (
    AGENT_REGISTRY_V4,
    AgentSpecV4,
    DispatchResult,
    DynamicRouter,
    FailSafeConfig,
    FailSafeDispatcher,
    PathwayClassDispatcher,
    SPECIALIST_AGENT_ID,
    count_agents,
    dynamic_router_hook_node,
    get_agent_spec,
    is_agent_registered,
    list_agent_ids,
)


# =============================================================================
# TestAgentRegistryV4 — 13 Agent 注册表测试
# =============================================================================
class TestAgentRegistryV4:
    """验证 13 Agent 注册表的完整性。"""

    def test_registry_has_13_agents(self):
        """注册表应包含正好 13 个 Agent。"""
        assert len(AGENT_REGISTRY_V4) == 13, (
            f"期望 13 个 Agent，实际 {len(AGENT_REGISTRY_V4)}"
        )
        assert count_agents() == 13

    def test_list_agent_ids_returns_13(self):
        """list_agent_ids 应返回 13 个 agent_id。"""
        ids = list_agent_ids()
        assert len(ids) == 13
        assert isinstance(ids, list)

    def test_all_expected_agent_ids_present(self):
        """所有 13 个预期的 agent_id 都应存在。"""
        expected_ids = {
            "ontology",
            "pathway_planner",
            "pathway_specialist_group",
            "crosstalk_coordinator",
            "reaction_builder",
            "mechanism_builder",
            "ode_builder",
            "sbml_grounder",
            "calibration",
            "simulation_planner",
            "validation",
            "hypothesis",
            "parameter_agent",
        }
        actual_ids = set(AGENT_REGISTRY_V4.keys())
        assert actual_ids == expected_ids, (
            f"缺失: {expected_ids - actual_ids}, 多余: {actual_ids - expected_ids}"
        )

    def test_get_agent_spec_returns_spec_for_known_id(self):
        """get_agent_spec 对已知 agent_id 应返回 AgentSpecV4。"""
        spec = get_agent_spec("ontology")
        assert spec is not None
        assert isinstance(spec, AgentSpecV4)
        assert spec.agent_id == "ontology"
        assert spec.name == "Ontology Agent"
        assert spec.cn_label == "本体标准化"
        assert spec.phase == "P1"
        assert spec.mapped_module == "app.ontology.ontology_agent"
        assert spec.timeout_seconds == 30

    def test_get_agent_spec_returns_none_for_unknown_id(self):
        """get_agent_spec 对未知 agent_id 应返回 None。"""
        assert get_agent_spec("nonexistent_agent") is None
        assert get_agent_spec("") is None

    def test_is_agent_registered(self):
        """is_agent_registered 应正确判断 agent_id 是否注册。"""
        assert is_agent_registered("ontology") is True
        assert is_agent_registered("nonexistent") is False

    def test_specialist_agent_id_constant(self):
        """SPECIALIST_AGENT_ID 应为 'pathway_specialist_group'。"""
        assert SPECIALIST_AGENT_ID == "pathway_specialist_group"
        assert SPECIALIST_AGENT_ID in AGENT_REGISTRY_V4

    def test_all_specs_have_required_fields(self):
        """每个 AgentSpecV4 应包含所有必填字段。"""
        for agent_id, spec in AGENT_REGISTRY_V4.items():
            assert spec.agent_id == agent_id, f"{agent_id} 的 agent_id 不匹配"
            assert spec.name, f"{agent_id} 缺少 name"
            assert spec.cn_label, f"{agent_id} 缺少 cn_label"
            assert spec.description, f"{agent_id} 缺少 description"
            assert spec.phase in ("P1", "P2", "P4", "P5", "P6"), (
                f"{agent_id} phase 非法: {spec.phase}"
            )
            assert spec.mapped_module, f"{agent_id} 缺少 mapped_module"
            assert spec.timeout_seconds > 0, f"{agent_id} timeout_seconds 应 > 0"
            assert spec.icon, f"{agent_id} 缺少 icon"

    def test_specialist_group_phase_is_p4(self):
        """pathway_specialist_group 的 phase 应为 P4。"""
        spec = get_agent_spec(SPECIALIST_AGENT_ID)
        assert spec is not None
        assert spec.phase == "P4"

    def test_task_6_6_agents_are_registered_as_stubs(self):
        """Task 6.6 未实现的 4 个 Agent 也应注册（mapped_module 指向尚不存在的模块）。"""
        stub_agents = {
            "mechanism_builder": "app.agents_v4.mechanism_builder",
            "ode_builder": "app.agents_v4.ode_builder",
            "simulation_planner": "app.agents_v4.simulation_planner",
            "parameter_agent": "app.agents_v4.parameter_agent",
        }
        for agent_id, expected_module in stub_agents.items():
            spec = get_agent_spec(agent_id)
            assert spec is not None, f"{agent_id} 未注册"
            assert spec.mapped_module == expected_module, (
                f"{agent_id} mapped_module 应为 {expected_module}"
            )
            assert spec.phase == "P6"


# =============================================================================
# TestFailSafeDispatcher — fail-safe 调度器测试
# =============================================================================
class TestFailSafeDispatcher:
    """验证 FailSafeDispatcher 的超时/防环/深度保护。"""

    def test_successful_dispatch_returns_success(self):
        """正常调度的 agent_func 应返回 success 结果。"""

        def good_func(state: dict) -> dict:
            return {"v4_ontology_entities": {"entities": []}}

        dispatcher = FailSafeDispatcher()
        result = dispatcher.dispatch(
            agent_id="test_agent",
            agent_func=good_func,
            state={},
        )
        assert result.success is True
        assert result.status == "success"
        assert result.agent_id == "test_agent"
        assert result.output == {"v4_ontology_entities": {"entities": []}}
        assert result.error is None
        assert result.fallback_used is False
        assert result.latency_ms >= 0.0
        assert result.depth == 0

    def test_exception_dispatch_returns_failed(self):
        """agent_func 抛异常时应返回 failed 结果（不抛出）。"""

        def bad_func(state: dict) -> dict:
            raise ValueError("test error")

        dispatcher = FailSafeDispatcher()
        result = dispatcher.dispatch(
            agent_id="bad_agent",
            agent_func=bad_func,
            state={},
        )
        assert result.success is False
        assert result.status == "failed"
        assert result.output == {}
        assert result.error is not None
        assert "ValueError" in result.error
        assert "test error" in result.error
        assert result.fallback_used is False  # 异常不强制回退

    def test_timeout_dispatch_returns_timeout(self):
        """超时的 agent_func 应返回 timeout 结果 + fallback_used=True。"""

        def slow_func(state: dict) -> dict:
            time.sleep(5.0)  # 故意慢
            return {"slow": True}

        # 使用 1s 超时配置
        config = FailSafeConfig(timeout_seconds=1, max_depth=10)
        dispatcher = FailSafeDispatcher(config=config)
        start = time.time()
        result = dispatcher.dispatch(
            agent_id="slow_agent",
            agent_func=slow_func,
            state={},
        )
        elapsed = time.time() - start
        assert result.success is False
        assert result.status == "timeout"
        assert result.output == {}
        assert result.error is not None
        assert "timeout" in result.error.lower()
        assert result.fallback_used is True
        # 应在 ~1s 内返回（允许少量线程调度开销）
        assert elapsed < 3.0, f"超时返回耗时过长: {elapsed:.2f}s"

    def test_depth_exceeded_returns_depth_exceeded(self):
        """depth >= max_depth 时应返回 depth_exceeded + fallback_used=True。"""

        def ok_func(state: dict) -> dict:
            return {"ok": True}

        config = FailSafeConfig(max_depth=3, timeout_seconds=10)
        dispatcher = FailSafeDispatcher(config=config)
        result = dispatcher.dispatch(
            agent_id="depthy_agent",
            agent_func=ok_func,
            state={},
            depth=3,  # 等于 max_depth
        )
        assert result.success is False
        assert result.status == "depth_exceeded"
        assert result.fallback_used is True
        assert result.depth == 3
        assert "max_depth" in result.error

    def test_loop_detected_for_visited_agent(self):
        """visited set 中的 agent_id 二次调度应返回 loop_detected。"""

        def ok_func(state: dict) -> dict:
            return {"ok": True}

        dispatcher = FailSafeDispatcher()
        # 第一次调度成功
        result1 = dispatcher.dispatch(
            agent_id="looping_agent",
            agent_func=ok_func,
            state={},
        )
        assert result1.success is True
        # 第二次调度应被防环检查拦截
        result2 = dispatcher.dispatch(
            agent_id="looping_agent",
            agent_func=ok_func,
            state={},
        )
        assert result2.success is False
        assert result2.status == "loop_detected"
        assert result2.fallback_used is True
        assert "visited" in result2.error.lower() or "loop" in result2.error.lower()

    def test_visited_set_populated_after_dispatch(self):
        """调度后 visited set 应包含已调度的 agent_id。"""
        dispatcher = FailSafeDispatcher()

        def ok_func(state: dict) -> dict:
            return {}

        assert "agent_a" not in dispatcher.get_visited()
        dispatcher.dispatch(agent_id="agent_a", agent_func=ok_func, state={})
        assert "agent_a" in dispatcher.get_visited()

        dispatcher.dispatch(agent_id="agent_b", agent_func=ok_func, state={})
        visited = dispatcher.get_visited()
        assert "agent_a" in visited
        assert "agent_b" in visited
        assert len(visited) == 2

    def test_clear_visited_empties_visited_set(self):
        """clear_visited 应清空 visited set。"""
        dispatcher = FailSafeDispatcher()

        def ok_func(state: dict) -> dict:
            return {}

        dispatcher.dispatch(agent_id="a1", agent_func=ok_func, state={})
        dispatcher.dispatch(agent_id="a2", agent_func=ok_func, state={})
        assert len(dispatcher.get_visited()) == 2

        dispatcher.clear_visited()
        assert len(dispatcher.get_visited()) == 0

        # 清空后可重新调度
        result = dispatcher.dispatch(
            agent_id="a1", agent_func=ok_func, state={}
        )
        assert result.success is True

    def test_get_visited_returns_copy(self):
        """get_visited 应返回副本，外部修改不影响内部状态。"""
        dispatcher = FailSafeDispatcher()

        def ok_func(state: dict) -> dict:
            return {}

        dispatcher.dispatch(agent_id="x1", agent_func=ok_func, state={})
        visited = dispatcher.get_visited()
        visited.add("external_mutation")
        # 内部 visited set 不应被修改
        assert "external_mutation" not in dispatcher.get_visited()

    def test_is_visited_check(self):
        """is_visited 应正确判断 agent_id 是否已调度。"""
        dispatcher = FailSafeDispatcher()

        def ok_func(state: dict) -> dict:
            return {}

        assert dispatcher.is_visited("z1") is False
        dispatcher.dispatch(agent_id="z1", agent_func=ok_func, state={})
        assert dispatcher.is_visited("z1") is True

    def test_disable_visited_check_allows_loop(self):
        """enable_visited_check=False 时允许同一 agent_id 重复调度。"""
        config = FailSafeConfig(enable_visited_check=False)
        dispatcher = FailSafeDispatcher(config=config)

        def ok_func(state: dict) -> dict:
            return {"ok": True}

        result1 = dispatcher.dispatch(
            agent_id="repeat_agent", agent_func=ok_func, state={}
        )
        result2 = dispatcher.dispatch(
            agent_id="repeat_agent", agent_func=ok_func, state={}
        )
        assert result1.success is True
        assert result2.success is True  # 未拦截重复调度

    def test_dispatch_result_to_dict(self):
        """DispatchResult.to_dict 应返回完整字段。"""
        result = DispatchResult(
            success=True,
            agent_id="test",
            agent_name="Test Agent",
            status="success",
            output={"key": "value"},
            error=None,
            latency_ms=42.5,
            fallback_used=False,
            depth=2,
        )
        d = result.to_dict()
        assert d["agent_id"] == "test"
        assert d["agent_name"] == "Test Agent"
        assert d["status"] == "success"
        assert d["output"] == {"key": "value"}
        assert d["error"] is None
        assert d["latency_ms"] == 42.5
        assert d["fallback_used"] is False
        assert d["depth"] == 2

    def test_non_dict_return_marks_failed(self):
        """agent_func 返回非 dict 类型时应标记为 failed。"""

        def bad_return_func(state: dict) -> str:  # 返回 str 而非 dict
            return "not a dict"

        dispatcher = FailSafeDispatcher()
        result = dispatcher.dispatch(
            agent_id="bad_return",
            agent_func=bad_return_func,
            state={},
        )
        assert result.success is False
        assert result.status == "failed"
        assert "非 dict" in result.error or "dict" in result.error.lower()


# =============================================================================
# TestPathwayClassDispatcher — 通路类别分发器测试
# =============================================================================
class TestPathwayClassDispatcher:
    """验证 PathwayClassDispatcher 的解析与分发。"""

    def test_parse_single_pathway(self):
        """单通路字符串应解析为单元素列表。"""
        d = PathwayClassDispatcher()
        assert d.parse_pathway_class("EGFR_RTK") == ["EGFR_RTK"]
        assert d.parse_pathway_class("p53") == ["p53"]

    def test_parse_multi_pathway_with_multi_prefix(self):
        """MULTI:A+B 格式应解析为多元素列表。"""
        d = PathwayClassDispatcher()
        result = d.parse_pathway_class("MULTI:EGFR_RTK+PI3K_AKT_mTOR")
        assert result == ["EGFR_RTK", "PI3K_AKT_mTOR"]

    def test_parse_multi_pathway_with_semicolon(self):
        """分号分隔的多通路应解析为列表。"""
        d = PathwayClassDispatcher()
        result = d.parse_pathway_class("EGFR_RTK;PI3K_AKT_mTOR")
        assert result == ["EGFR_RTK", "PI3K_AKT_mTOR"]

    def test_parse_multi_pathway_with_comma(self):
        """逗号分隔的多通路应解析为列表。"""
        d = PathwayClassDispatcher()
        result = d.parse_pathway_class("EGFR_RTK,PI3K_AKT_mTOR")
        assert result == ["EGFR_RTK", "PI3K_AKT_mTOR"]

    def test_parse_unknown_returns_empty(self):
        """UNKNOWN / 空字符串应返回空列表。"""
        d = PathwayClassDispatcher()
        assert d.parse_pathway_class("UNKNOWN") == []
        assert d.parse_pathway_class("") == []
        assert d.parse_pathway_class(None) == []  # type: ignore[arg-type]

    def test_parse_deduplicates(self):
        """重复的通路应去重。"""
        d = PathwayClassDispatcher()
        result = d.parse_pathway_class("MULTI:EGFR_RTK+EGFR_RTK")
        assert result == ["EGFR_RTK"]

    def test_is_multi_pathway_true_for_multi_prefix(self):
        """MULTI: 前缀应识别为多通路。"""
        d = PathwayClassDispatcher()
        assert d.is_multi_pathway("MULTI:EGFR_RTK+PI3K_AKT_mTOR") is True

    def test_is_multi_pathway_true_for_separator(self):
        """含分隔符应识别为多通路。"""
        d = PathwayClassDispatcher()
        assert d.is_multi_pathway("A+B") is True
        assert d.is_multi_pathway("A;B") is True
        assert d.is_multi_pathway("A,B") is True

    def test_is_multi_pathway_false_for_single(self):
        """单通路应识别为非多通路。"""
        d = PathwayClassDispatcher()
        assert d.is_multi_pathway("EGFR_RTK") is False
        assert d.is_multi_pathway("") is False
        assert d.is_multi_pathway("UNKNOWN") is False

    def test_dispatch_single_specialist_with_mock(self):
        """单通路场景应调度 1 个 Specialist（mock）。"""
        d = PathwayClassDispatcher()

        # 构造 mock registry + mock specialist
        mock_specialist = MagicMock()
        mock_specialist.apply_core.return_value = {
            "species": [{"name": "EGF"}],
            "reactions": [{"id": "r1", "source": "EGF", "target": "EGFR"}],
        }
        mock_registry = MagicMock()
        mock_registry.get_specialist.return_value = mock_specialist

        d._pathway_registry = mock_registry
        d._specialists_imported = True  # 跳过懒加载

        state = {"v4_pathway_class": "EGFR_RTK"}
        results = d.dispatch_specialists(state)

        assert len(results) == 1
        assert results[0]["pathway_class"] == "EGFR_RTK"
        assert results[0]["applied"] is True
        assert results[0]["specialist_name"] == "MagicMock"
        assert len(results[0]["core_reactions"]) == 1
        # 单通路不应调用 Coordinator
        assert not any(r.get("coordinator") for r in results)

    def test_dispatch_multi_specialist_with_coordinator(self):
        """多通路场景应调度 N 个 Specialist + Coordinator。"""
        d = PathwayClassDispatcher()

        # 构造 2 个 mock specialist
        def make_mock(name: str):
            m = MagicMock()
            m.apply_core.return_value = {
                "species": [{"name": name}],
                "reactions": [{"id": f"r_{name}", "source": name}],
            }
            return m

        mock_registry = MagicMock()
        mock_registry.get_specialist.side_effect = lambda pc: make_mock(pc)

        d._pathway_registry = mock_registry
        d._specialists_imported = True

        # mock CrossTalkCoordinator.coordinate
        with patch(
            "app.crosstalk.coordinator.CrossTalkCoordinator.coordinate"
        ) as mock_coord:
            mock_coord.return_value = {
                "v4_crosstalk_edges": [],
                "v4_shared_species": ["RasGTP"],
                "v4_shared_species_sync": {},
                "v4_time_scale_alignment": {},
            }
            state = {"v4_pathway_class": "MULTI:EGFR_RTK+PI3K_AKT_mTOR"}
            results = d.dispatch_specialists(state)

        assert len(results) == 3  # 2 Specialist + 1 Coordinator
        assert results[0]["pathway_class"] == "EGFR_RTK"
        assert results[1]["pathway_class"] == "PI3K_AKT_mTOR"
        # Coordinator 条目
        coord_entry = results[2]
        assert coord_entry.get("coordinator") is True
        assert coord_entry["specialist_name"] == "CrossTalkCoordinator"
        assert coord_entry["applied"] is True

    def test_dispatch_unknown_pathway_returns_empty(self):
        """UNKNOWN 通路应返回空列表。"""
        d = PathwayClassDispatcher()
        state = {"v4_pathway_class": "UNKNOWN"}
        assert d.dispatch_specialists(state) == []

    def test_dispatch_missing_pathway_class_returns_empty(self):
        """缺失 v4_pathway_class 应返回空列表。"""
        d = PathwayClassDispatcher()
        assert d.dispatch_specialists({}) == []

    def test_dispatch_unregistered_specialist_records_failure(self):
        """未注册的 Specialist 应记录 applied=False 但不抛异常。"""
        d = PathwayClassDispatcher()
        mock_registry = MagicMock()
        mock_registry.get_specialist.return_value = None  # 未注册
        d._pathway_registry = mock_registry
        d._specialists_imported = True

        state = {"v4_pathway_class": "UNKNOWN_PATHWAY"}
        results = d.dispatch_specialists(state)
        assert len(results) == 1
        assert results[0]["applied"] is False
        assert "not registered" in results[0]["error"]

    def test_dispatch_specialist_apply_core_exception_isolated(self):
        """单个 Specialist apply_core 异常应被隔离，不影响其他 Specialist。"""
        d = PathwayClassDispatcher()

        # 第一个 specialist 抛异常，第二个正常
        bad_specialist = MagicMock()
        bad_specialist.apply_core.side_effect = RuntimeError("apply_core failed")
        good_specialist = MagicMock()
        good_specialist.apply_core.return_value = {
            "species": [],
            "reactions": [{"id": "ok"}],
        }

        mock_registry = MagicMock()
        mock_registry.get_specialist.side_effect = lambda pc: (
            bad_specialist if pc == "BAD" else good_specialist
        )
        d._pathway_registry = mock_registry
        d._specialists_imported = True

        state = {"v4_pathway_class": "MULTI:BAD+GOOD"}
        # 多通路会调用 Coordinator，patch 掉
        with patch(
            "app.crosstalk.coordinator.CrossTalkCoordinator.coordinate"
        ) as mock_coord:
            mock_coord.return_value = {
                "v4_crosstalk_edges": [],
                "v4_shared_species": [],
                "v4_shared_species_sync": {},
                "v4_time_scale_alignment": {},
            }
            results = d.dispatch_specialists(state)

        # BAD 应失败，GOOD 应成功
        bad_result = next(r for r in results if r["pathway_class"] == "BAD")
        good_result = next(r for r in results if r["pathway_class"] == "GOOD")
        assert bad_result["applied"] is False
        assert "apply_core failed" in bad_result["error"]
        assert good_result["applied"] is True
        assert len(good_result["core_reactions"]) == 1

    def test_dispatch_degrades_on_internal_error(self):
        """整体异常应降级返回空列表（不抛出）。"""
        d = PathwayClassDispatcher()
        # 让 _get_pathway_registry 抛异常
        with patch.object(
            d, "_get_pathway_registry", side_effect=RuntimeError("internal")
        ):
            state = {"v4_pathway_class": "EGFR_RTK"}
            results = d.dispatch_specialists(state)
        assert results == []


# =============================================================================
# TestDynamicRouter — DynamicRouter 主类测试
# =============================================================================
class TestDynamicRouter:
    """验证 DynamicRouter 的路由与调度。"""

    @patch("app.config.settings.V4_DYNAMIC_ROUTING_ENABLED", False)
    def test_route_returns_empty_when_flag_off(self):
        """flag=false 时 route() 应返回 {}。"""
        router = DynamicRouter()
        assert router.route({"v4_pathway_class": "EGFR_RTK"}) == {}

    @patch("app.config.settings.V4_DYNAMIC_ROUTING_ENABLED", True)
    @patch("app.config.settings.V4_SBML_GROUNDER_ENABLED", False)
    @patch("app.config.settings.V4_CALIBRATION_AGENT_ENABLED", False)
    @patch("app.config.settings.V4_HYPOTHESIS_AGENT_ENABLED", False)
    def test_route_returns_v4_agent_dispatches_when_flag_on(self):
        """flag=true 时 route() 应返回 v4_agent_dispatches 列表。"""
        router = DynamicRouter()
        result = router.route({"v4_pathway_class": "EGFR_RTK"})
        assert "v4_agent_dispatches" in result
        assert isinstance(result["v4_agent_dispatches"], list)
        # 应有调度记录（核心 Agent + Task 6.6 stub = 9 个）
        assert len(result["v4_agent_dispatches"]) >= 9

    @patch("app.config.settings.V4_DYNAMIC_ROUTING_ENABLED", False)
    @patch("app.config.settings.V4_SBML_GROUNDER_ENABLED", False)
    @patch("app.config.settings.V4_CALIBRATION_AGENT_ENABLED", False)
    @patch("app.config.settings.V4_HYPOTHESIS_AGENT_ENABLED", False)
    def test_build_dispatch_plan_includes_core_agents(self):
        """build_dispatch_plan 应包含 5 个核心 Agent + 4 个 Task 6.6 stub = 9。"""
        router = DynamicRouter()
        plan = router.build_dispatch_plan({"v4_pathway_class": "EGFR_RTK"})
        # 5 核心 + 4 Task 6.6 stub（mechanism/ode/simulation/parameter）
        assert "ontology" in plan
        assert "pathway_planner" in plan
        assert "pathway_specialist_group" in plan
        assert "reaction_builder" in plan
        assert "validation" in plan
        assert "mechanism_builder" in plan
        assert "ode_builder" in plan
        assert "simulation_planner" in plan
        assert "parameter_agent" in plan
        # 单通路场景不含 coordinator
        assert "crosstalk_coordinator" not in plan

    @patch("app.config.settings.V4_DYNAMIC_ROUTING_ENABLED", False)
    @patch("app.config.settings.V4_SBML_GROUNDER_ENABLED", False)
    @patch("app.config.settings.V4_CALIBRATION_AGENT_ENABLED", False)
    @patch("app.config.settings.V4_HYPOTHESIS_AGENT_ENABLED", False)
    def test_build_dispatch_plan_multi_pathway_adds_coordinator(self):
        """多通路场景应追加 crosstalk_coordinator。"""
        router = DynamicRouter()
        plan = router.build_dispatch_plan(
            {"v4_pathway_class": "MULTI:EGFR_RTK+PI3K_AKT_mTOR"}
        )
        assert "crosstalk_coordinator" in plan

    @patch("app.config.settings.V4_DYNAMIC_ROUTING_ENABLED", False)
    @patch("app.config.settings.V4_SBML_GROUNDER_ENABLED", True)
    @patch("app.config.settings.V4_CALIBRATION_AGENT_ENABLED", True)
    @patch("app.config.settings.V4_HYPOTHESIS_AGENT_ENABLED", True)
    def test_build_dispatch_plan_respects_feature_flags(self):
        """build_dispatch_plan 应按 Feature Flag 追加条件 Agent。"""
        router = DynamicRouter()
        plan = router.build_dispatch_plan({"v4_pathway_class": "EGFR_RTK"})
        assert "sbml_grounder" in plan
        assert "calibration" in plan
        assert "hypothesis" in plan

    @patch("app.config.settings.V4_DYNAMIC_ROUTING_ENABLED", False)
    def test_execute_agent_unknown_returns_empty(self):
        """execute_agent 对未注册 agent_id 应返回 {}。"""
        router = DynamicRouter()
        assert router.execute_agent("nonexistent", {}) == {}

    @patch("app.config.settings.V4_DYNAMIC_ROUTING_ENABLED", False)
    def test_execute_agent_task_6_6_stub_returns_empty(self):
        """Task 6.6 未实现的 Agent 应通过 ImportError 处理返回 {} stub。"""
        router = DynamicRouter()
        result = router.execute_agent("mechanism_builder", {})
        assert result == {}
        result = router.execute_agent("ode_builder", {})
        assert result == {}
        result = router.execute_agent("simulation_planner", {})
        assert result == {}
        result = router.execute_agent("parameter_agent", {})
        assert result == {}

    @patch("app.config.settings.V4_DYNAMIC_ROUTING_ENABLED", True)
    @patch("app.config.settings.V4_SBML_GROUNDER_ENABLED", False)
    @patch("app.config.settings.V4_CALIBRATION_AGENT_ENABLED", False)
    @patch("app.config.settings.V4_HYPOTHESIS_AGENT_ENABLED", False)
    def test_route_records_all_dispatches(self):
        """route() 应为每个调度的 Agent 记录一条 dispatch。"""
        router = DynamicRouter()
        result = router.route({"v4_pathway_class": "EGFR_RTK"})
        dispatches = result["v4_agent_dispatches"]

        # 验证每条 dispatch 含必需字段
        for d in dispatches:
            assert "agent_id" in d
            assert "agent_name" in d
            assert "status" in d
            assert "output" in d
            assert "latency_ms" in d
            assert "fallback_used" in d
            assert "depth" in d

        # 应包含核心 Agent 的调度记录
        agent_ids = {d["agent_id"] for d in dispatches}
        assert "ontology" in agent_ids
        assert "pathway_planner" in agent_ids
        assert "validation" in agent_ids

    @patch("app.config.settings.V4_DYNAMIC_ROUTING_ENABLED", True)
    @patch("app.config.settings.V4_SBML_GROUNDER_ENABLED", False)
    @patch("app.config.settings.V4_CALIBRATION_AGENT_ENABLED", False)
    @patch("app.config.settings.V4_HYPOTHESIS_AGENT_ENABLED", False)
    def test_route_does_not_modify_v3_fields(self):
        """route() 不应修改 v3 字段，仅返回 v4_agent_dispatches。"""
        router = DynamicRouter()
        state = {
            "user_input": "EGF binds EGFR",
            "network_json": {"nodes": []},
            "v4_pathway_class": "EGFR_RTK",
        }
        original_user_input = state["user_input"]
        original_network_json = state["network_json"]

        update = router.route(state)
        # 仅返回 v4_agent_dispatches，不含 v3 字段
        assert set(update.keys()) <= {"v4_agent_dispatches"}
        # state 中的 v3 字段未被修改
        assert state["user_input"] == original_user_input
        assert state["network_json"] == original_network_json

    @patch("app.config.settings.V4_DYNAMIC_ROUTING_ENABLED", True)
    @patch("app.config.settings.V4_SBML_GROUNDER_ENABLED", False)
    @patch("app.config.settings.V4_CALIBRATION_AGENT_ENABLED", False)
    @patch("app.config.settings.V4_HYPOTHESIS_AGENT_ENABLED", False)
    def test_route_with_custom_fail_safe_config(self):
        """route() 应使用自定义 fail_safe_config（如快速超时）。"""
        config = FailSafeConfig(timeout_seconds=1, max_depth=5)
        router = DynamicRouter(fail_safe_config=config)

        # 让所有 Agent 都慢（mock execute_agent 慢返回）
        with patch.object(
            router, "execute_agent", side_effect=lambda aid, s: time.sleep(2) or {}
        ):
            result = router.route({"v4_pathway_class": "EGFR_RTK"})
            dispatches = result["v4_agent_dispatches"]
            # 所有调度应 timeout（execute_agent 慢于 1s）
            assert all(d["status"] == "timeout" for d in dispatches)
            assert all(d["fallback_used"] is True for d in dispatches)


# =============================================================================
# TestDynamicRouterHookNode — LangGraph hook 节点测试
# =============================================================================
class TestDynamicRouterHookNode:
    """验证 dynamic_router_hook_node 的 Feature Flag 隔离。"""

    @patch("app.config.settings.V4_DYNAMIC_ROUTING_ENABLED", False)
    def test_hook_returns_empty_when_flag_off(self):
        """flag=false 时 hook 应返回 {}（不修改 state）。"""
        state = {"user_input": "EGF binds EGFR", "v4_pathway_class": "EGFR_RTK"}
        result = dynamic_router_hook_node(state)
        assert result == {}

    @patch("app.config.settings.V4_DYNAMIC_ROUTING_ENABLED", False)
    def test_hook_flag_off_does_not_pollute_state(self):
        """flag=false 时 hook 不应向 state 写入任何 v4_ 字段。"""
        state = {
            "user_input": "EGF",
            "network_json": {"nodes": []},
        }
        update = dynamic_router_hook_node(state)
        state.update(update)
        v4_keys = [k for k in state.keys() if k.startswith("v4_")]
        assert v4_keys == [], f"flag=false 时不应出现 v4_ 字段: {v4_keys}"

    @patch("app.config.settings.V4_DYNAMIC_ROUTING_ENABLED", True)
    @patch("app.config.settings.V4_SBML_GROUNDER_ENABLED", False)
    @patch("app.config.settings.V4_CALIBRATION_AGENT_ENABLED", False)
    @patch("app.config.settings.V4_HYPOTHESIS_AGENT_ENABLED", False)
    def test_hook_returns_v4_agent_dispatches_when_flag_on(self):
        """flag=true 时 hook 应返回 v4_agent_dispatches 列表。"""
        state = {"v4_pathway_class": "EGFR_RTK"}
        result = dynamic_router_hook_node(state)
        assert "v4_agent_dispatches" in result
        assert isinstance(result["v4_agent_dispatches"], list)
        assert len(result["v4_agent_dispatches"]) >= 1

    @patch("app.config.settings.V4_DYNAMIC_ROUTING_ENABLED", True)
    @patch("app.config.settings.V4_SBML_GROUNDER_ENABLED", False)
    @patch("app.config.settings.V4_CALIBRATION_AGENT_ENABLED", False)
    @patch("app.config.settings.V4_HYPOTHESIS_AGENT_ENABLED", False)
    def test_hook_does_not_modify_v3_fields(self):
        """flag=true 时 hook 也不应修改 v3 字段。"""
        state = {
            "user_input": "EGF",
            "network_json": {"nodes": [{"id": "EGF"}]},
            "parameters": {"k1": 0.1},
            "v4_pathway_class": "EGFR_RTK",
        }
        original = {k: v for k, v in state.items()}
        update = dynamic_router_hook_node(state)
        # 仅返回 v4_agent_dispatches
        assert set(update.keys()) <= {"v4_agent_dispatches"}
        # v3 字段未被修改
        assert state["user_input"] == original["user_input"]
        assert state["network_json"] == original["network_json"]
        assert state["parameters"] == original["parameters"]

    @patch("app.config.settings.V4_DYNAMIC_ROUTING_ENABLED", True)
    @patch("app.config.settings.V4_SBML_GROUNDER_ENABLED", False)
    @patch("app.config.settings.V4_CALIBRATION_AGENT_ENABLED", False)
    @patch("app.config.settings.V4_HYPOTHESIS_AGENT_ENABLED", False)
    def test_hook_degrades_on_internal_error(self):
        """hook 内部异常应降级返回 {}（不抛出，不阻塞主流水线）。"""
        # 让 DynamicRouter.route 抛异常
        with patch.object(
            DynamicRouter, "route", side_effect=RuntimeError("internal error")
        ):
            state = {"v4_pathway_class": "EGFR_RTK"}
            result = dynamic_router_hook_node(state)
        # 异常时应返回 {}（不阻塞）
        assert result == {}

    @patch("app.config.settings.V4_DYNAMIC_ROUTING_ENABLED", True)
    @patch("app.config.settings.V4_SBML_GROUNDER_ENABLED", False)
    @patch("app.config.settings.V4_CALIBRATION_AGENT_ENABLED", False)
    @patch("app.config.settings.V4_HYPOTHESIS_AGENT_ENABLED", False)
    def test_hook_with_empty_state_does_not_raise(self):
        """空 state 也不应抛异常（降级返回空列表或 {}）。"""
        result = dynamic_router_hook_node({})
        # 应返回 v4_agent_dispatches（可能为空列表）或 {}
        assert "v4_agent_dispatches" in result or result == {}


# =============================================================================
# TestPackageExports — 包导出测试
# =============================================================================
class TestPackageExports:
    """验证 agent_orchestration 包的所有导出。"""

    def test_all_exports_available(self):
        """所有 __all__ 中的名称应可导入。"""
        from app.agent_orchestration import __all__ as orchestration_all

        # 验证 __all__ 包含所有期望的导出名
        expected_exports = {
            "AGENT_REGISTRY_V4",
            "AgentSpecV4",
            "SPECIALIST_AGENT_ID",
            "get_agent_spec",
            "list_agent_ids",
            "count_agents",
            "is_agent_registered",
            "DynamicRouter",
            "dynamic_router_hook_node",
            "FailSafeConfig",
            "FailSafeDispatcher",
            "DispatchResult",
            "PathwayClassDispatcher",
        }
        actual_exports = set(orchestration_all)
        assert expected_exports.issubset(actual_exports), (
            f"缺失导出: {expected_exports - actual_exports}"
        )

    def test_dynamic_router_instantiation(self):
        """DynamicRouter 应可正常实例化。"""
        router = DynamicRouter()
        assert router is not None
        assert router._fail_safe_config is not None
        assert router._pathway_dispatcher is not None

    def test_pathway_dispatcher_instantiation(self):
        """PathwayClassDispatcher 应可正常实例化。"""
        d = PathwayClassDispatcher()
        assert d is not None

    def test_fail_safe_dispatcher_instantiation(self):
        """FailSafeDispatcher 应可正常实例化。"""
        dispatcher = FailSafeDispatcher()
        assert dispatcher is not None
        assert dispatcher.config.max_depth == 10
        assert dispatcher.config.timeout_seconds == 30
        assert dispatcher.config.enable_visited_check is True
