# BioDynamics Agent v4 - P6 Hook 端到端冒烟测试 (Phase 6 / Task 6.7)
#
# 覆盖 SubTask 6.7.1-6.7.5：
# - 6.7.1: graph_v3.py 集成 _hypothesis_agent_hook + _dynamic_router_hook
# - 6.7.2: SSE 事件 v4_hypothesis_generated 发射
# - 6.7.3: 7 冒烟用例
# - 6.7.4: flag=false 隔离验证
# - 6.7.5: Feature Flag 4 关键组合测试（全 false / P4 only / P4+P5 / 全开）
#
# 7 冒烟用例：
#   1. test_smoke_flag_all_off_v3_regression: 全 P6 flag=false → hook 返回 {} → v3 行为不变
#   2. test_smoke_hypothesis_generation_produces_list: flag=true + mock state → 生成 v4_hypothesis_list
#   3. test_smoke_hypothesis_quality_check: 生成的假设含必需字段
#   4. test_smoke_egfr_pi3k_dynamic_routing: 多通路 → 调度 specialists + coordinator
#   5. test_smoke_apoptosis_dynamic_routing: 单通路 → 调度 1 specialist
#   6. test_smoke_simulation_failure_fail_safe: 仿真失败 → fail_safe fallback_used=True
#   7. test_smoke_v3_regression_no_v4_fields: 全 flag=false → state 无 v4_ 字段
#
# 4 Feature Flag 组合测试：
#   8. test_flag_combo_all_false: 全 8 flag=false → 所有 hook 返回 {}
#   9. test_flag_combo_p4_only: P4 flag=true, P5/P6=false → P4 执行, P5/P6 返回 {}
#   10. test_flag_combo_p4_p5: P4+P5 flag=true, P6=false → P4+P5 执行, P6 返回 {}
#   11. test_flag_combo_all_on: 全 8 flag=true → 所有 hook 执行
#
# 附加测试：
#   12. test_sse_event_v4_hypothesis_generated: SSE 事件发射验证
#   13. test_dynamic_router_hook_node_flag_off: flag=false → hook 返回 {}
#   14. test_dynamic_router_hook_node_flag_on: flag=true → 返回 v4_agent_dispatches
#   15. test_hypothesis_hook_in_graph: _hypothesis_agent_hook 节点存在于构建的 workflow
#   16. test_dynamic_router_hook_in_graph: _dynamic_router_hook 节点存在于构建的 workflow
#
# 测试策略：
# - 不实际调用 LLM / 真实 RAG / 真实 SBML 仿真
# - mock settings.V4_HYPOTHESIS_AGENT_ENABLED / V4_DYNAMIC_ROUTING_ENABLED
# - mock state dict 含 metrics / v4_validation_report / v4_pathway_class 等
# - 对 dynamic router，mock PathwayClassDispatcher / FailSafeDispatcher
# - 对 hypothesis，mock HypothesisAgent.generate() 返回固定假设列表
# - 不实际运行完整 LangGraph workflow（太慢），直接测试 hook 函数
#
# 运行：cd backend && python -m pytest tests/test_p6_hook_e2e.py -v

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

# 确保能导入 app 包（backend/ 在 sys.path）
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


# =============================================================================
# 测试夹具：构造 mock state / 假设列表
# =============================================================================
def _make_oscillation_metrics() -> dict[str, Any]:
    """构造含振荡特征的 metrics（NF-κB 振荡）。"""
    return {
        "species": {
            "NFkB": {
                "peak": 1.5,
                "peak_time": 30.0,
                "fold_change": 5.0,
                "oscillation": {
                    "oscillatory": True,
                    "period_minutes": 90.0,
                    "n_peaks": 3,
                    "oscillation_type": "damped",
                },
            },
        },
        "overall": {"confidence": 0.8},
    }


def _make_passed_validation_state() -> dict[str, Any]:
    """构造 Validation 通过的 state（含 v4_validation_report.overall_pass=True）。"""
    return {
        "v4_validation_report": {
            "overall_pass": True,
            "level1": {"pass": True},
            "level2": {"pass": True, "skipped": True},
            "level3": {"pass": True, "skipped": True},
            "level4": {"pass": True, "skipped": True},
            "level5": {"pass": True, "skipped": True},
        },
        "v4_grounding_ledger": {
            "entries": [{"pmid": "PMID:14975635", "species": "NFkB"}],
            "integrity": True,
        },
        "v4_pathway_class": "NF_KB",
    }


def _make_mock_hypotheses() -> list[dict[str, Any]]:
    """构造 mock 假设列表（含所有必需字段）。"""
    return [
        {
            "id": "H001",
            "statement": "NFkB 振荡周期由 IKK 反馈环决定",
            "prediction": "抑制 IKK 将消除 NFkB 振荡",
            "experiment_design": {},
            "validation_method": "siRNA knockdown IKK",
            "expected_result": "NFkB 振荡消失",
            "falsifiable": True,
            "supporting_pmids": ["PMID:14975635"],
            "contradicting_pmids": [],
            "strategy": "oscillation",
            "target_species": "NFkB",
            "feedback_node": "IKK",
        },
        {
            "id": "H002",
            "statement": "IKK 浓度阈值为 0.5 μM",
            "prediction": "IKK 浓度低于 0.5 μM 时振荡消失",
            "experiment_design": {},
            "validation_method": "dose-response IKK inhibitor",
            "expected_result": "阈值处振荡切换",
            "falsifiable": True,
            "supporting_pmids": [],
            "contradicting_pmids": [],
            "strategy": "bistability",
            "target_species": "IKK",
        },
    ]


# =============================================================================
# 主测试类：7 冒烟用例 + 4 Flag 组合测试 + 5 附加测试
# =============================================================================
class TestP6HookE2E(unittest.TestCase):
    """P6 Hook 端到端冒烟测试：验证 Hypothesis + Dynamic Router 集成。"""

    # =========================================================================
    # 用例 1：全 P6 flag=false → v3 行为零侵入回归
    # =========================================================================
    @patch("app.config.settings.V4_HYPOTHESIS_AGENT_ENABLED", False)
    @patch("app.config.settings.V4_DYNAMIC_ROUTING_ENABLED", False)
    def test_smoke_flag_all_off_v3_regression(self):
        """用例 1：P6 flags 全 false → hooks 返回 {} → v3 行为不变。

        验证：
        - hypothesis_agent_hook_node 返回 {}
        - dynamic_router_hook_node 返回 {}
        - state 不被 v4_ 字段污染
        """
        from app.hypothesis.hypothesis_agent import hypothesis_agent_hook_node
        from app.agent_orchestration.dynamic_router import dynamic_router_hook_node

        state = {"user_input": "EGF binds EGFR", "v4_pathway_class": "EGFR_RTK"}

        hyp_out = hypothesis_agent_hook_node(state)
        router_out = dynamic_router_hook_node(state)

        self.assertEqual(hyp_out, {})
        self.assertEqual(router_out, {})

        # state 不被污染
        self.assertNotIn("v4_hypothesis_list", state)
        self.assertNotIn("v4_agent_dispatches", state)

    # =========================================================================
    # 用例 2：hypothesis flag=true + mock state → 生成 v4_hypothesis_list
    # =========================================================================
    @patch("app.config.settings.V4_HYPOTHESIS_AGENT_ENABLED", True)
    @patch("app.config.settings.V4_VALIDATION_PYRAMID_ENABLED", True)
    def test_smoke_hypothesis_generation_produces_list(self):
        """用例 2：flag=true + mock state → hypothesis hook 返回 v4_hypothesis_list。

        策略：mock HypothesisAgent.generate 返回固定假设列表，验证 hook 正确写入
        v4_hypothesis_list + v4_hypothesis_generated=True。
        """
        from app.hypothesis.hypothesis_agent import hypothesis_agent_hook_node

        state = _make_passed_validation_state()
        state["metrics"] = _make_oscillation_metrics()

        # mock HypothesisAgent.generate 返回固定假设列表
        mock_hypotheses = _make_mock_hypotheses()
        with patch(
            "app.hypothesis.hypothesis_agent.HypothesisAgent.generate",
            return_value=mock_hypotheses,
        ):
            result = hypothesis_agent_hook_node(state)

        # hook 应返回 v4_hypothesis_list + v4_hypothesis_generated=True
        self.assertIn("v4_hypothesis_list", result)
        self.assertEqual(result["v4_hypothesis_generated"], True)
        self.assertEqual(len(result["v4_hypothesis_list"]), 2)
        self.assertEqual(
            result["v4_hypothesis_list"][0]["id"], "H001"
        )

    # =========================================================================
    # 用例 3：生成的假设含必需字段（id/statement/prediction/falsifiable）
    # =========================================================================
    @patch("app.config.settings.V4_HYPOTHESIS_AGENT_ENABLED", True)
    @patch("app.config.settings.V4_VALIDATION_PYRAMID_ENABLED", True)
    def test_smoke_hypothesis_quality_check(self):
        """用例 3：生成的假设含必需字段（id/statement/prediction/falsifiable）。

        验证假设 schema 完整性：
        - id: 唯一标识（H 开头）
        - statement: 假设陈述
        - prediction: 可证伪预测
        - falsifiable: bool
        """
        from app.hypothesis.hypothesis_agent import hypothesis_agent_hook_node

        state = _make_passed_validation_state()
        state["metrics"] = _make_oscillation_metrics()

        mock_hypotheses = _make_mock_hypotheses()
        with patch(
            "app.hypothesis.hypothesis_agent.HypothesisAgent.generate",
            return_value=mock_hypotheses,
        ):
            result = hypothesis_agent_hook_node(state)

        hypotheses = result["v4_hypothesis_list"]
        self.assertGreaterEqual(len(hypotheses), 1)

        # 验证每个假设含必需字段
        required_fields = ["id", "statement", "prediction", "falsifiable"]
        for hyp in hypotheses:
            for field in required_fields:
                self.assertIn(
                    field, hyp, f"假设 {hyp.get('id', '?')} 缺少字段: {field}"
                )
            # falsifiable 必须是 bool
            self.assertIsInstance(hyp["falsifiable"], bool)
            # id 应以 H 开头
            self.assertTrue(hyp["id"].startswith("H"))

    # =========================================================================
    # 用例 4：多通路 → dynamic router 调度 specialists + coordinator
    # =========================================================================
    @patch("app.config.settings.V4_DYNAMIC_ROUTING_ENABLED", True)
    @patch("app.config.settings.V4_SBML_GROUNDER_ENABLED", False)
    @patch("app.config.settings.V4_CALIBRATION_AGENT_ENABLED", False)
    @patch("app.config.settings.V4_HYPOTHESIS_AGENT_ENABLED", False)
    def test_smoke_egfr_pi3k_dynamic_routing(self):
        """用例 4：多通路 (EGFR_RTK;PI3K_AKT_mTOR) → dispatch plan 含 coordinator。

        策略：仅验证 build_dispatch_plan 含 crosstalk_coordinator
        （route() 会触发真实 Specialist 调度，太重，这里测 plan）
        """
        from app.agent_orchestration.dynamic_router import DynamicRouter

        router = DynamicRouter()
        # 多通路场景
        state = {"v4_pathway_class": "MULTI:EGFR_RTK+PI3K_AKT_mTOR"}
        plan = router.build_dispatch_plan(state)

        # 核心 specialists 都在 plan 中
        self.assertIn("pathway_planner", plan)
        self.assertIn("pathway_specialist_group", plan)
        # 多通路场景应包含 crosstalk_coordinator
        self.assertIn("crosstalk_coordinator", plan)

    # =========================================================================
    # 用例 5：单通路 → dynamic router 调度 1 specialist（不含 coordinator）
    # =========================================================================
    @patch("app.config.settings.V4_DYNAMIC_ROUTING_ENABLED", True)
    @patch("app.config.settings.V4_SBML_GROUNDER_ENABLED", False)
    @patch("app.config.settings.V4_CALIBRATION_AGENT_ENABLED", False)
    @patch("app.config.settings.V4_HYPOTHESIS_AGENT_ENABLED", False)
    def test_smoke_apoptosis_dynamic_routing(self):
        """用例 5：单通路 (APOPTOSIS) → dispatch plan 不含 coordinator。

        策略：验证 build_dispatch_plan 不含 crosstalk_coordinator
        """
        from app.agent_orchestration.dynamic_router import DynamicRouter

        router = DynamicRouter()
        state = {"v4_pathway_class": "APOPTOSIS"}
        plan = router.build_dispatch_plan(state)

        # 核心 specialists 在 plan 中
        self.assertIn("pathway_planner", plan)
        self.assertIn("pathway_specialist_group", plan)
        # 单通路场景不应含 crosstalk_coordinator
        self.assertNotIn("crosstalk_coordinator", plan)

    # =========================================================================
    # 用例 6：仿真失败 → fail_safe fallback_used=True → 无 crash
    # =========================================================================
    @patch("app.config.settings.V4_DYNAMIC_ROUTING_ENABLED", True)
    @patch("app.config.settings.V4_SBML_GROUNDER_ENABLED", False)
    @patch("app.config.settings.V4_CALIBRATION_AGENT_ENABLED", False)
    @patch("app.config.settings.V4_HYPOTHESIS_AGENT_ENABLED", False)
    def test_smoke_simulation_failure_fail_safe(self):
        """用例 6：仿真失败 → fail_safe 返回 fallback_used=True → 无 crash。

        策略：用 FailSafeConfig(timeout_seconds=1) + 慢 agent_func，验证
        dispatch 返回 timeout 状态 + fallback_used=True，且不抛异常。
        """
        from app.agent_orchestration.dynamic_router import (
            DynamicRouter,
            FailSafeConfig,
        )
        import time

        # 用 1s 超时配置构造 router
        config = FailSafeConfig(timeout_seconds=1, max_depth=10)
        router = DynamicRouter(fail_safe_config=config)

        # mock execute_agent 慢返回（>1s 超时）
        def slow_agent(agent_id: str, state: dict) -> dict:
            time.sleep(2)
            return {}

        with patch.object(router, "execute_agent", side_effect=slow_agent):
            # 不应抛异常
            result = router.route({"v4_pathway_class": "EGFR_RTK"})

        # 应返回 v4_agent_dispatches（含至少 1 条 timeout 记录）
        self.assertIn("v4_agent_dispatches", result)
        dispatches = result["v4_agent_dispatches"]
        self.assertGreaterEqual(len(dispatches), 1)

        # 所有 dispatch 应为 timeout + fallback_used=True
        for d in dispatches:
            self.assertEqual(d["status"], "timeout")
            self.assertEqual(d["fallback_used"], True)

    # =========================================================================
    # 用例 7：全 flag=false → state 无 v4_ 字段添加
    # =========================================================================
    @patch("app.config.settings.V4_HYPOTHESIS_AGENT_ENABLED", False)
    @patch("app.config.settings.V4_DYNAMIC_ROUTING_ENABLED", False)
    def test_smoke_v3_regression_no_v4_fields(self):
        """用例 7：全 flag=false → state 不被 v4_ 字段污染。

        验证：
        - 两个 hook 都返回 {}
        - state 中不出现 v4_hypothesis_list / v4_agent_dispatches 等 P6 字段
        """
        from app.hypothesis.hypothesis_agent import hypothesis_agent_hook_node
        from app.agent_orchestration.dynamic_router import dynamic_router_hook_node

        state = {
            "user_input": "EGF binds EGFR",
            "network_json": {"nodes": []},
            "v4_pathway_class": "EGFR_RTK",  # 已存在的 v4 字段（来自 P4）
        }
        original_keys = set(state.keys())

        hyp_out = hypothesis_agent_hook_node(state)
        router_out = dynamic_router_hook_node(state)
        state.update(hyp_out)
        state.update(router_out)

        # 两个 hook 都返回空 dict
        self.assertEqual(hyp_out, {})
        self.assertEqual(router_out, {})

        # state 中不应新增 v4_ 前缀字段
        new_keys = set(state.keys()) - original_keys
        new_v4_keys = [k for k in new_keys if k.startswith("v4_")]
        self.assertEqual(
            new_v4_keys, [], f"flag=false 时不应新增 v4_ 字段: {new_v4_keys}"
        )

    # =========================================================================
    # 用例 8：Feature Flag 组合 - 全 false
    # =========================================================================
    @patch("app.config.settings.V4_ONTOLOGY_AGENT_ENABLED", False)
    @patch("app.config.settings.V4_PATHWAY_PLANNER_ENABLED", False)
    @patch("app.config.settings.V4_PATHWAY_SPECIALIST_ENABLED", False)
    @patch("app.config.settings.V4_CROSSTALK_COORDINATOR_ENABLED", False)
    @patch("app.config.settings.V4_SBML_GROUNDER_ENABLED", False)
    @patch("app.config.settings.V4_VALIDATION_PYRAMID_ENABLED", False)
    @patch("app.config.settings.V4_HYPOTHESIS_AGENT_ENABLED", False)
    @patch("app.config.settings.V4_DYNAMIC_ROUTING_ENABLED", False)
    def test_flag_combo_all_false(self):
        """用例 8：全 8 v4 flag=false → 所有 hook 返回 {}。

        验证：所有 P1/P4/P5/P6 hook 在 flag=false 时都返回空 dict。
        """
        from app.hypothesis.hypothesis_agent import hypothesis_agent_hook_node
        from app.agent_orchestration.dynamic_router import dynamic_router_hook_node
        from app.ontology.ontology_agent import ontology_hook_node
        from app.pathways.pathway_planner import pathway_planner_hook_node
        from app.pathways.specialist_hook import specialist_hook_node
        from app.crosstalk.coordinator import crosstalk_coordinator_hook_node
        from app.sbml_grounder.grounder_agent import sbml_grounder_hook_node
        from app.validation_v2.validation_agent import validation_pyramid_hook_node

        state = {"user_input": "EGF binds EGFR"}

        # 所有 hook 都应返回 {}
        self.assertEqual(ontology_hook_node(state), {})
        self.assertEqual(pathway_planner_hook_node(state), {})
        self.assertEqual(specialist_hook_node(state), {})
        self.assertEqual(crosstalk_coordinator_hook_node(state), {})
        self.assertEqual(sbml_grounder_hook_node(state), {})
        self.assertEqual(validation_pyramid_hook_node(state), {})
        self.assertEqual(hypothesis_agent_hook_node(state), {})
        self.assertEqual(dynamic_router_hook_node(state), {})

    # =========================================================================
    # 用例 9：Feature Flag 组合 - P4 only
    # =========================================================================
    @patch("app.config.settings.V4_ONTOLOGY_AGENT_ENABLED", False)
    @patch("app.config.settings.V4_PATHWAY_PLANNER_ENABLED", True)
    @patch("app.config.settings.V4_PATHWAY_SPECIALIST_ENABLED", True)
    @patch("app.config.settings.V4_CROSSTALK_COORDINATOR_ENABLED", True)
    @patch("app.config.settings.V4_SBML_GROUNDER_ENABLED", False)
    @patch("app.config.settings.V4_VALIDATION_PYRAMID_ENABLED", False)
    @patch("app.config.settings.V4_HYPOTHESIS_AGENT_ENABLED", False)
    @patch("app.config.settings.V4_DYNAMIC_ROUTING_ENABLED", False)
    def test_flag_combo_p4_only(self):
        """用例 9：P4 flags=true, P5/P6=false → P4 执行, P5/P6 返回 {}。

        验证：
        - P4 hooks（pathway_planner/specialist/coordinator）执行并返回非空
        - P5 hooks（sbml_grounder/validation_pyramid）返回 {}
        - P6 hooks（hypothesis/dynamic_router）返回 {}
        """
        from app.pathways.pathway_planner import pathway_planner_hook_node
        from app.pathways.specialist_hook import specialist_hook_node
        from app.crosstalk.coordinator import crosstalk_coordinator_hook_node
        from app.sbml_grounder.grounder_agent import sbml_grounder_hook_node
        from app.validation_v2.validation_agent import validation_pyramid_hook_node
        from app.hypothesis.hypothesis_agent import hypothesis_agent_hook_node
        from app.agent_orchestration.dynamic_router import dynamic_router_hook_node

        state = {"user_input": "EGF binds EGFR receptor"}

        # P4 hooks 应执行并返回非空
        planner_out = pathway_planner_hook_node(state)
        state.update(planner_out)
        self.assertIn("v4_pathway_class", planner_out)
        self.assertEqual(planner_out["v4_pathway_class"], "EGFR_RTK")

        specialist_out = specialist_hook_node(state)
        state.update(specialist_out)
        self.assertIn("v4_specialist_outputs", specialist_out)

        coordinator_out = crosstalk_coordinator_hook_node(state)
        # 单通路场景 coordinator 仍执行（返回 v4_crosstalk_edges=[]）

        # P5/P6 hooks 应返回 {}
        self.assertEqual(sbml_grounder_hook_node(state), {})
        self.assertEqual(validation_pyramid_hook_node(state), {})
        self.assertEqual(hypothesis_agent_hook_node(state), {})
        self.assertEqual(dynamic_router_hook_node(state), {})

    # =========================================================================
    # 用例 10：Feature Flag 组合 - P4 + P5
    # =========================================================================
    @patch("app.config.settings.V4_ONTOLOGY_AGENT_ENABLED", False)
    @patch("app.config.settings.V4_PATHWAY_PLANNER_ENABLED", True)
    @patch("app.config.settings.V4_PATHWAY_SPECIALIST_ENABLED", True)
    @patch("app.config.settings.V4_CROSSTALK_COORDINATOR_ENABLED", True)
    @patch("app.config.settings.V4_SBML_GROUNDER_ENABLED", True)
    @patch("app.config.settings.V4_VALIDATION_PYRAMID_ENABLED", True)
    @patch("app.config.settings.V4_HYPOTHESIS_AGENT_ENABLED", False)
    @patch("app.config.settings.V4_DYNAMIC_ROUTING_ENABLED", False)
    def test_flag_combo_p4_p5(self):
        """用例 10：P4+P5 flags=true, P6=false → P4+P5 执行, P6 返回 {}。

        验证：
        - P4 hooks 执行（v4_pathway_class / v4_specialist_outputs 等被填充）
        - P5 hooks 执行（v4_grounding_ledger / v4_validation_report 被填充）
        - P6 hooks 返回 {}
        """
        from app.pathways.pathway_planner import pathway_planner_hook_node
        from app.pathways.specialist_hook import specialist_hook_node
        from app.crosstalk.coordinator import crosstalk_coordinator_hook_node
        from app.sbml_grounder.grounder_agent import sbml_grounder_hook_node
        from app.validation_v2.validation_agent import validation_pyramid_hook_node
        from app.hypothesis.hypothesis_agent import hypothesis_agent_hook_node
        from app.agent_orchestration.dynamic_router import dynamic_router_hook_node

        state = {
            "user_input": "EGF binds EGFR receptor",
            "sbml_model_id": "BIOMD0000000205",
        }

        # P4 hooks 执行
        planner_out = pathway_planner_hook_node(state)
        state.update(planner_out)
        self.assertIn("v4_pathway_class", planner_out)

        specialist_out = specialist_hook_node(state)
        state.update(specialist_out)
        self.assertIn("v4_specialist_outputs", specialist_out)

        coordinator_out = crosstalk_coordinator_hook_node(state)
        state.update(coordinator_out)

        # P5 hooks 执行（需要 v4_pathway_graph / v4_reaction_ir 等 P4 输出）
        grounder_out = sbml_grounder_hook_node(state)
        state.update(grounder_out)
        # sbml_grounder 可能返回 v4_grounding_ledger 或为空（依赖 sbml_model_text）
        # 这里仅验证 hook 执行了（返回 dict，可能含 v4_grounding_ledger）

        pyramid_out = validation_pyramid_hook_node(state)
        state.update(pyramid_out)
        # validation_pyramid 应返回 v4_validation_report
        self.assertIn("v4_validation_report", pyramid_out)

        # P6 hooks 返回 {}
        self.assertEqual(hypothesis_agent_hook_node(state), {})
        self.assertEqual(dynamic_router_hook_node(state), {})

    # =========================================================================
    # 用例 11：Feature Flag 组合 - 全开
    # =========================================================================
    @patch("app.config.settings.V4_ONTOLOGY_AGENT_ENABLED", False)
    @patch("app.config.settings.V4_PATHWAY_PLANNER_ENABLED", True)
    @patch("app.config.settings.V4_PATHWAY_SPECIALIST_ENABLED", True)
    @patch("app.config.settings.V4_CROSSTALK_COORDINATOR_ENABLED", True)
    @patch("app.config.settings.V4_SBML_GROUNDER_ENABLED", True)
    @patch("app.config.settings.V4_VALIDATION_PYRAMID_ENABLED", True)
    @patch("app.config.settings.V4_HYPOTHESIS_AGENT_ENABLED", True)
    @patch("app.config.settings.V4_DYNAMIC_ROUTING_ENABLED", True)
    def test_flag_combo_all_on(self):
        """用例 11：全 8 flags=true → 所有 P4/P5/P6 hooks 执行。

        验证：
        - P4 hooks 执行（v4_pathway_class 被填充）
        - P5 hooks 执行（v4_validation_report 被填充）
        - P6 hooks 执行（v4_hypothesis_list / v4_agent_dispatches 被填充）

        注：ontology flag 保持 false（避免触发外部 API 调用）
        """
        from app.pathways.pathway_planner import pathway_planner_hook_node
        from app.pathways.specialist_hook import specialist_hook_node
        from app.crosstalk.coordinator import crosstalk_coordinator_hook_node
        from app.sbml_grounder.grounder_agent import sbml_grounder_hook_node
        from app.validation_v2.validation_agent import validation_pyramid_hook_node
        from app.hypothesis.hypothesis_agent import hypothesis_agent_hook_node
        from app.agent_orchestration.dynamic_router import dynamic_router_hook_node

        state = {
            "user_input": "EGF binds EGFR receptor",
            "sbml_model_id": "BIOMD0000000205",
            "metrics": _make_oscillation_metrics(),
        }

        # 1. P4 hooks 执行
        planner_out = pathway_planner_hook_node(state)
        state.update(planner_out)
        self.assertIn("v4_pathway_class", planner_out)

        specialist_out = specialist_hook_node(state)
        state.update(specialist_out)
        self.assertIn("v4_specialist_outputs", specialist_out)

        coordinator_out = crosstalk_coordinator_hook_node(state)
        state.update(coordinator_out)

        # 2. P5 hooks 执行
        grounder_out = sbml_grounder_hook_node(state)
        state.update(grounder_out)
        pyramid_out = validation_pyramid_hook_node(state)
        state.update(pyramid_out)
        self.assertIn("v4_validation_report", pyramid_out)

        # 3. P6 hooks 执行
        # Hypothesis hook：mock generate 返回固定列表（避免真实 RAG/LLM）
        mock_hypotheses = _make_mock_hypotheses()
        with patch(
            "app.hypothesis.hypothesis_agent.HypothesisAgent.generate",
            return_value=mock_hypotheses,
        ):
            hyp_out = hypothesis_agent_hook_node(state)
        self.assertIn("v4_hypothesis_list", hyp_out)
        self.assertEqual(hyp_out["v4_hypothesis_generated"], True)

        # Dynamic Router hook：执行（不 mock，验证返回 v4_agent_dispatches）
        router_out = dynamic_router_hook_node(state)
        self.assertIn("v4_agent_dispatches", router_out)
        self.assertIsInstance(router_out["v4_agent_dispatches"], list)
        self.assertGreaterEqual(len(router_out["v4_agent_dispatches"]), 1)

    # =========================================================================
    # 用例 12：SSE 事件 v4_hypothesis_generated 发射验证
    # =========================================================================
    def test_sse_event_v4_hypothesis_generated(self):
        """用例 12：验证 SSE 事件 v4_hypothesis_generated 会被发射。

        策略：mock output 含 v4_hypothesis_generated=True，调用 _emit_worker_outputs
        前的 SSE 发射逻辑（这里直接验证 _sse_event payload 构造）。
        """
        from app.main import _sse_event

        # 模拟 hook 输出
        output = {
            "v4_hypothesis_generated": True,
            "v4_hypothesis_list": _make_mock_hypotheses(),
        }

        # 模拟 main.py 中的 SSE 发射逻辑
        if isinstance(output, dict) and output.get("v4_hypothesis_generated"):
            sse_payload = {
                "event": "v4_hypothesis_generated",
                "data": {
                    "hypothesis_count": len(output.get("v4_hypothesis_list", [])),
                    "hypothesis_list": output.get("v4_hypothesis_list", []),
                },
            }
            sse_str = _sse_event(sse_payload)

        # 验证 SSE 字符串格式
        self.assertTrue(sse_str.startswith("data: "))
        self.assertTrue(sse_str.endswith("\n\n"))
        # 验证事件名 + hypothesis_count
        import json
        payload = json.loads(sse_str[len("data: "):].strip())
        self.assertEqual(payload["event"], "v4_hypothesis_generated")
        self.assertEqual(payload["data"]["hypothesis_count"], 2)
        self.assertEqual(len(payload["data"]["hypothesis_list"]), 2)

    # =========================================================================
    # 用例 13：dynamic_router_hook_node flag=false → 返回 {}
    # =========================================================================
    @patch("app.config.settings.V4_DYNAMIC_ROUTING_ENABLED", False)
    def test_dynamic_router_hook_node_flag_off(self):
        """用例 13：flag=false 时 dynamic_router_hook_node 返回 {}。"""
        from app.agent_orchestration.dynamic_router import dynamic_router_hook_node

        state = {"v4_pathway_class": "EGFR_RTK"}
        result = dynamic_router_hook_node(state)
        self.assertEqual(result, {})

    # =========================================================================
    # 用例 14：dynamic_router_hook_node flag=true → 返回 v4_agent_dispatches
    # =========================================================================
    @patch("app.config.settings.V4_DYNAMIC_ROUTING_ENABLED", True)
    @patch("app.config.settings.V4_SBML_GROUNDER_ENABLED", False)
    @patch("app.config.settings.V4_CALIBRATION_AGENT_ENABLED", False)
    @patch("app.config.settings.V4_HYPOTHESIS_AGENT_ENABLED", False)
    def test_dynamic_router_hook_node_flag_on(self):
        """用例 14：flag=true 时 dynamic_router_hook_node 返回 v4_agent_dispatches。

        验证：
        - 返回 dict 含 v4_agent_dispatches 键
        - v4_agent_dispatches 是 list
        - 至少 1 条调度记录
        - 不修改 v3 字段
        """
        from app.agent_orchestration.dynamic_router import dynamic_router_hook_node

        state = {
            "user_input": "EGF binds EGFR",
            "network_json": {"nodes": []},
            "v4_pathway_class": "EGFR_RTK",
        }
        original_network_json = state["network_json"]

        result = dynamic_router_hook_node(state)

        self.assertIn("v4_agent_dispatches", result)
        self.assertIsInstance(result["v4_agent_dispatches"], list)
        self.assertGreaterEqual(len(result["v4_agent_dispatches"]), 1)

        # 验证每条 dispatch 含必需字段
        for d in result["v4_agent_dispatches"]:
            self.assertIn("agent_id", d)
            self.assertIn("status", d)
            self.assertIn("fallback_used", d)

        # v3 字段不被修改
        self.assertEqual(state["network_json"], original_network_json)
        # 返回的 dict 仅含 v4_agent_dispatches
        self.assertTrue(set(result.keys()) <= {"v4_agent_dispatches"})

    # =========================================================================
    # 用例 15：_hypothesis_agent_hook 节点存在于构建的 workflow
    # =========================================================================
    def test_hypothesis_hook_in_graph(self):
        """用例 15：_hypothesis_agent_hook 节点存在于 build_workflow_v3() 构建的图中。

        验证：
        - build_workflow_v3() 不抛异常
        - 构建的图中含 _hypothesis_agent_hook 节点
        """
        from app.graph_v3 import build_workflow_v3

        workflow = build_workflow_v3()

        # 通过 workflow.nodes 访问节点（StateGraph.nodes 是 dict-like）
        nodes = workflow.nodes
        self.assertIn("_hypothesis_agent_hook", nodes)

    # =========================================================================
    # 用例 16：_dynamic_router_hook 节点存在于构建的 workflow
    # =========================================================================
    def test_dynamic_router_hook_in_graph(self):
        """用例 16：_dynamic_router_hook 节点存在于 build_workflow_v3() 构建的图中。

        验证：
        - build_workflow_v3() 不抛异常
        - 构建的图中含 _dynamic_router_hook 节点
        """
        from app.graph_v3 import build_workflow_v3

        workflow = build_workflow_v3()

        nodes = workflow.nodes
        self.assertIn("_dynamic_router_hook", nodes)


if __name__ == "__main__":
    unittest.main()
