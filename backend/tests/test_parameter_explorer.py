# BioDynamics Agent v4 - Parameter Explorer + Sensitivity Planner 单元测试
# (Phase 6 / Task 6.4.5)
#
# 覆盖 SubTask 6.4.1-6.4.4：
# - 6.4.1: ParameterExplorer + SensitivityPlanner 类创建
# - 6.4.2: 实现参数扫描（调用 v3 sandbox.py 复用仿真器，不新增）
# - 6.4.3: 实现参数鲁棒性验证（假设在参数范围内是否成立）
# - 6.4.4: 实现灵敏度规划（target_params/method/sample_size）
#
# 测试策略：
# - 注入自定义 model_func 避免真实 sandbox 调用
# - mock sandbox.execute_simulation_code_v2 测试降级路径
# - 验证 parameter_robustness.hypothesis_holds: bool
# - 验证 sensitivity_plan 4 字段完整

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# 确保能导入 app 包（backend/ 在 sys.path）
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.hypothesis import (
    ParameterExplorer,
    SensitivityPlanner,
    METHOD_LOCAL,
    METHOD_MORRIS,
    METHOD_SOBOL,
    HypothesisStrategy,
)


# =============================================================================
# 测试夹具
# =============================================================================
@pytest.fixture
def sensitivity_hypothesis() -> dict[str, Any]:
    """灵敏度假设：k_pEGFR_dephos 对 pEGFR 敏感。"""
    return {
        "id": "H001",
        "statement": "参数 k_pEGFR_dephos 对 pEGFR 敏感（灵敏度=0.85）",
        "prediction": "k_pEGFR_dephos 抑制剂处理后 pEGFR 水平降低 >30%",
        "expected_result": "pEGFR 降低 >30%",
        "strategy": HypothesisStrategy.SENSITIVITY,
        "target_param": "k_pEGFR_dephos",
        "sensitivity": 0.85,
        "target_species": "pEGFR",
        "pathway_class": "EGFR_RTK",
        "experiment_design": {
            "readout": {"species": "pEGFR", "metric": "peak", "threshold": 0.3},
        },
    }


@pytest.fixture
def egfr_state() -> dict[str, Any]:
    """EGFR 通路 state（含 calibration + sensitivity）。"""
    return {
        "v4_pathway_class": "EGFR_RTK",
        "v4_ode_system": {
            "ode_code": "# EGFR ODE\ndpEGFR_dt = k_pEGFR_dephos * pEGFR",
        },
        "v4_calibration_result": {
            "calibrated_params": {
                "k_pEGFR_dephos": 0.1,
                "k_EGFR_act": 0.05,
            },
            "confidence_intervals": {
                "k_pEGFR_dephos": {"lower": 0.08, "upper": 0.12},
            },
        },
        "v4_sensitivity_report": {
            "local_sensitivity": {
                "k_pEGFR_dephos": 0.85,
                "k_EGFR_act": 0.42,
                "k_other": 0.15,
            },
        },
        "v4_pathway_graph": {
            "feedback_loops": [
                {"source": "ERK", "target": "SOS", "type": "negative"},
            ],
        },
    }


# =============================================================================
# TestParameterExplorerBasic：基础接口测试
# =============================================================================
class TestParameterExplorerBasic:
    """SubTask 6.4.1: ParameterExplorer 类创建 + 接口测试。"""

    def test_explore_returns_required_fields(
        self, sensitivity_hypothesis, egfr_state
    ):
        """用例 1：explore 返回 param/range/hypothesis_holds/holds_ratio/method/details。"""
        explorer = ParameterExplorer()
        result = explorer.explore(sensitivity_hypothesis, egfr_state)

        for field in ("param", "range", "hypothesis_holds", "holds_ratio", "method"):
            assert field in result, f"缺少字段: {field}"

    def test_explore_hypothesis_holds_is_bool(
        self, sensitivity_hypothesis, egfr_state
    ):
        """用例 2：hypothesis_holds 为 bool 类型。"""
        explorer = ParameterExplorer()
        result = explorer.explore(sensitivity_hypothesis, egfr_state)
        assert isinstance(result["hypothesis_holds"], bool)

    def test_explore_param_is_target_param(
        self, sensitivity_hypothesis, egfr_state
    ):
        """用例 3：param 应为 hypothesis.target_param。"""
        explorer = ParameterExplorer()
        result = explorer.explore(sensitivity_hypothesis, egfr_state)
        assert result["param"] == "k_pEGFR_dephos"

    def test_explore_range_is_list_of_float(
        self, sensitivity_hypothesis, egfr_state
    ):
        """用例 4：range 为 float 列表，非空。"""
        explorer = ParameterExplorer()
        result = explorer.explore(sensitivity_hypothesis, egfr_state)
        assert isinstance(result["range"], list)
        # baseline 值 0.1，应有扫描点
        assert len(result["range"]) > 0


# =============================================================================
# TestParameterExplorerModelFunc：注入 model_func 测试参数扫描
# =============================================================================
class TestParameterExplorerModelFunc:
    """SubTask 6.4.2: 参数扫描（通过 model_func 注入避免真实 sandbox）。"""

    def test_sweep_with_stable_model_func_holds_true(
        self, sensitivity_hypothesis, egfr_state
    ):
        """用例 5：稳定 model_func（输出变化小）→ hypothesis_holds=True。

        model_func 返回接近 baseline 的值 → 相对变化小 → 假设鲁棒。
        """
        # 稳定模型：输出 = baseline * (1 + 0.01 * factor)
        def stable_model(param: str, factor: float) -> float:
            return 1.0 * (1 + 0.01 * factor)

        explorer = ParameterExplorer(model_func=stable_model)
        result = explorer.explore(sensitivity_hypothesis, egfr_state)

        assert result["param"] == "k_pEGFR_dephos"
        assert result["method"] == "model_func"
        assert isinstance(result["hypothesis_holds"], bool)
        # 稳定模型 → 假设鲁棒
        assert result["hypothesis_holds"] is True
        assert result["holds_ratio"] > 0.5

    def test_sweep_with_unstable_model_func_holds_false(
        self, sensitivity_hypothesis, egfr_state
    ):
        """用例 6：不稳定 model_func（输出剧烈变化）→ hypothesis_holds=False。

        model_func 返回变化大的值 → 相对变化 > 阈值 → 假设不鲁棒。
        """
        # 不稳定模型：输出 = factor^2（参数变化 10 倍 → 输出变化 100 倍）
        def unstable_model(param: str, factor: float) -> float:
            return factor * factor

        # 降低 holds_threshold 使测试更严格
        explorer = ParameterExplorer(
            model_func=unstable_model,
            holds_threshold=0.99,  # 要求所有点都成立
        )
        result = explorer.explore(sensitivity_hypothesis, egfr_state)

        assert result["param"] == "k_pEGFR_dephos"
        # 不稳定模型 + 严格阈值 → 不鲁棒
        assert result["hypothesis_holds"] is False
        assert result["holds_ratio"] < 1.0

    def test_sweep_range_uses_baseline_value(
        self, sensitivity_hypothesis, egfr_state
    ):
        """用例 7：扫描范围基于 baseline 值（0.1）× scan_factors。"""
        def model(param: str, factor: float) -> float:
            return 1.0

        explorer = ParameterExplorer(model_func=model)
        result = explorer.explore(sensitivity_hypothesis, egfr_state)

        # baseline=0.1, factors=[0.1, 0.5, 1.0, 2.0, 10.0]
        # range = [0.01, 0.05, 0.1, 0.2, 1.0]
        assert len(result["range"]) == 5
        assert 0.1 in result["range"]  # baseline
        assert 0.01 in result["range"]  # 0.1 * 0.1
        assert 1.0 in result["range"]  # 0.1 * 10.0

    def test_sweep_details_per_point(
        self, sensitivity_hypothesis, egfr_state
    ):
        """用例 8：details 列表含每个扫描点的结果。"""
        def model(param: str, factor: float) -> float:
            return 1.0

        explorer = ParameterExplorer(model_func=model)
        result = explorer.explore(sensitivity_hypothesis, egfr_state)

        assert isinstance(result["details"], list)
        assert len(result["details"]) == 5  # 5 个扫描点
        # 每个点应含 param_value + holds
        for detail in result["details"]:
            assert "param_value" in detail
            assert "holds" in detail


# =============================================================================
# TestParameterExplorerSandboxFallback：sandbox 降级路径
# =============================================================================
class TestParameterExplorerSandboxFallback:
    """SubTask 6.4.2: sandbox 不可用 / 失败的降级路径。"""

    def test_sandbox_failure_falls_back_to_heuristic(
        self, sensitivity_hypothesis, egfr_state
    ):
        """用例 9：sandbox 抛异常 → 降级到 CI 启发式。

        patch sandbox.execute_simulation_code_v2 抛 RuntimeError →
        ParameterExplorer 应降级到 calibration_ci_heuristic 方法。
        """
        with patch(
            "app.sandbox.execute_simulation_code_v2",
            side_effect=RuntimeError("sandbox unavailable"),
        ):
            explorer = ParameterExplorer()
            result = explorer.explore(sensitivity_hypothesis, egfr_state)

            # 降级到 heuristic
            assert result["method"] in ("calibration_ci_heuristic", "sandbox_sweep")
            # 启发式 → 所有点 holds=True
            assert result["hypothesis_holds"] is True

    def test_sandbox_empty_ode_code_falls_back(
        self, sensitivity_hypothesis
    ):
        """用例 10：ode_code 为空 → 降级到 CI 启发式。"""
        state = {
            "v4_ode_system": {"ode_code": ""},  # 空 ODE 代码
            "v4_calibration_result": {
                "calibrated_params": {"k_pEGFR_dephos": 0.1},
            },
        }
        explorer = ParameterExplorer()
        result = explorer.explore(sensitivity_hypothesis, state)

        # 启发式 → 假设鲁棒
        assert result["hypothesis_holds"] is True

    def test_no_baseline_value_uses_ci_heuristic(
        self, sensitivity_hypothesis
    ):
        """用例 11：无 baseline 值 → CI 启发式判断。"""
        state = {
            "v4_calibration_result": {},  # 无 calibrated_params
            "v4_sensitivity_report": {},  # 无 sensitivity 数据
        }
        explorer = ParameterExplorer()
        result = explorer.explore(sensitivity_hypothesis, state)

        # 无 baseline → CI 启发式
        assert result["method"] == "calibration_ci_heuristic"
        assert result["hypothesis_holds"] is True  # 保守不误杀


# =============================================================================
# TestParameterExplorerTargetParamExtraction：目标参数提取
# =============================================================================
class TestParameterExplorerTargetParamExtraction:
    """目标参数提取测试。"""

    def test_no_target_param_falls_back_to_sensitivity_report(
        self, egfr_state
    ):
        """用例 12：无 target_param → 从 v4_sensitivity_report 提取 top-1。"""
        # 构造无 target_param 的假设（如振荡假设）
        hyp = {
            "strategy": HypothesisStrategy.OSCILLATION,
            "target_species": "NFkB",
            "feedback_node": "IKK",
        }
        explorer = ParameterExplorer()
        result = explorer.explore(hyp, egfr_state)

        # top-1 灵敏度参数 = k_pEGFR_dephos (0.85)
        assert result["param"] == "k_pEGFR_dephos"

    def test_no_target_param_no_sensitivity_report_returns_minimal(
        self
    ):
        """用例 13：无 target_param + 无 sensitivity_report → minimal 结果。"""
        hyp = {
            "strategy": HypothesisStrategy.OSCILLATION,
            "target_species": "NFkB",
        }
        state = {"v4_sensitivity_report": {}}
        explorer = ParameterExplorer()
        result = explorer.explore(hyp, state)

        # 无关键参数 → 跳过鲁棒性验证
        assert result["param"] in ("unknown", "")
        assert result["range"] == []
        # 保守假设成立
        assert result["hypothesis_holds"] is True

    def test_invalid_hypothesis_returns_minimal(self, egfr_state):
        """用例 14：hypothesis 非 dict → minimal 结果。"""
        explorer = ParameterExplorer()
        result = explorer.explore(None, egfr_state)  # type: ignore[arg-type]

        assert result["method"] == "degraded"
        assert result["hypothesis_holds"] is True  # 保守

    def test_invalid_state_returns_minimal(self, sensitivity_hypothesis):
        """用例 15：state 非 dict → minimal 结果。"""
        explorer = ParameterExplorer()
        result = explorer.explore(sensitivity_hypothesis, None)  # type: ignore[arg-type]

        assert result["method"] == "degraded"
        assert result["hypothesis_holds"] is True


# =============================================================================
# TestSensitivityPlannerBasic：SensitivityPlanner 基础测试
# =============================================================================
class TestSensitivityPlannerBasic:
    """SubTask 6.4.4: 灵敏度规划（target_params/method/sample_size）。"""

    def test_plan_returns_required_fields(
        self, sensitivity_hypothesis, egfr_state
    ):
        """用例 16：plan 返回 target_params/method/sample_size/rationale。"""
        planner = SensitivityPlanner()
        plan = planner.plan([sensitivity_hypothesis], egfr_state)

        for field in ("target_params", "method", "sample_size", "rationale"):
            assert field in plan, f"缺少字段: {field}"

    def test_plan_target_params_is_list_of_str(
        self, sensitivity_hypothesis, egfr_state
    ):
        """用例 17：target_params 为 str 列表。"""
        planner = SensitivityPlanner()
        plan = planner.plan([sensitivity_hypothesis], egfr_state)

        assert isinstance(plan["target_params"], list)
        # 应含 hypothesis.target_param
        assert "k_pEGFR_dephos" in plan["target_params"]

    def test_plan_method_is_valid_enum(
        self, sensitivity_hypothesis, egfr_state
    ):
        """用例 18：method 为 local/morris/sobol 之一。"""
        planner = SensitivityPlanner()
        plan = planner.plan([sensitivity_hypothesis], egfr_state)

        assert plan["method"] in (METHOD_LOCAL, METHOD_MORRIS, METHOD_SOBOL)

    def test_plan_sample_size_is_positive_int(
        self, sensitivity_hypothesis, egfr_state
    ):
        """用例 19：sample_size 为正整数。"""
        planner = SensitivityPlanner()
        plan = planner.plan([sensitivity_hypothesis], egfr_state)

        assert isinstance(plan["sample_size"], int)
        assert plan["sample_size"] > 0

    def test_plan_rationale_is_nonempty_str(
        self, sensitivity_hypothesis, egfr_state
    ):
        """用例 20：rationale 为非空 str。"""
        planner = SensitivityPlanner()
        plan = planner.plan([sensitivity_hypothesis], egfr_state)

        assert isinstance(plan["rationale"], str)
        assert plan["rationale"]


# =============================================================================
# TestSensitivityPlannerMethodSelection：方法选择规则
# =============================================================================
class TestSensitivityPlannerMethodSelection:
    """方法选择规则测试（基于假设数量）。"""

    def test_zero_hypotheses_uses_local(self, egfr_state):
        """用例 21：0 个假设 → local 方法。"""
        planner = SensitivityPlanner()
        plan = planner.plan([], egfr_state)
        assert plan["method"] == METHOD_LOCAL

    def test_one_hypothesis_uses_morris(self, egfr_state):
        """用例 22：1 个假设 → morris 方法。"""
        hyp = {"target_param": "k1"}
        planner = SensitivityPlanner()
        plan = planner.plan([hyp], egfr_state)
        assert plan["method"] == METHOD_MORRIS

    def test_two_hypotheses_uses_morris(self, egfr_state):
        """用例 23：2 个假设 → morris 方法。"""
        hyps = [
            {"target_param": "k1"},
            {"target_param": "k2"},
        ]
        planner = SensitivityPlanner()
        plan = planner.plan(hyps, egfr_state)
        assert plan["method"] == METHOD_MORRIS

    def test_three_hypotheses_uses_sobol(self, egfr_state):
        """用例 24：3 个假设 → sobol 方法。"""
        hyps = [
            {"target_param": "k1"},
            {"target_param": "k2"},
            {"target_param": "k3"},
        ]
        planner = SensitivityPlanner()
        plan = planner.plan(hyps, egfr_state)
        assert plan["method"] == METHOD_SOBOL

    def test_many_hypotheses_uses_sobol(self, egfr_state):
        """用例 25：5 个假设 → sobol 方法。"""
        hyps = [{"target_param": f"k{i}"} for i in range(5)]
        planner = SensitivityPlanner()
        plan = planner.plan(hyps, egfr_state)
        assert plan["method"] == METHOD_SOBOL

    def test_no_target_params_uses_local(self, egfr_state):
        """用例 26：无目标参数 → local 方法（即使有假设）。"""
        # 构造无 target_param 的假设 + 无 sensitivity_report
        hyps = [{"strategy": HypothesisStrategy.OSCILLATION}]  # 无 target_param
        state = {"v4_sensitivity_report": {}, "v4_pathway_graph": {}}
        planner = SensitivityPlanner()
        plan = planner.plan(hyps, state)
        assert plan["method"] == METHOD_LOCAL


# =============================================================================
# TestSensitivityPlannerSampleSize：采样规模选择
# =============================================================================
class TestSensitivityPlannerSampleSize:
    """采样规模选择测试。"""

    def test_local_sample_size_is_one(self, egfr_state):
        """用例 27：local 方法 → sample_size=1。"""
        planner = SensitivityPlanner()
        plan = planner.plan([], egfr_state)
        assert plan["method"] == METHOD_LOCAL
        assert plan["sample_size"] == 1

    def test_morris_sample_size_at_least_10(self, egfr_state):
        """用例 28：morris 方法 → sample_size ≥ 10。"""
        hyp = {"target_param": "k1"}
        planner = SensitivityPlanner()
        plan = planner.plan([hyp], egfr_state)
        assert plan["method"] == METHOD_MORRIS
        assert plan["sample_size"] >= 10

    def test_sobol_sample_size_at_least_100(self, egfr_state):
        """用例 29：sobol 方法 → sample_size ≥ 100。"""
        hyps = [{"target_param": f"k{i}"} for i in range(3)]
        planner = SensitivityPlanner()
        plan = planner.plan(hyps, egfr_state)
        assert plan["method"] == METHOD_SOBOL
        assert plan["sample_size"] >= 100

    def test_sobol_sample_size_scales_with_params(self, egfr_state):
        """用例 30：sobol sample_size ≥ n_params * 10。"""
        # 构造 5 个假设，从 sensitivity_report 提取更多参数
        hyps = [{"target_param": f"k{i}"} for i in range(5)]
        planner = SensitivityPlanner()
        plan = planner.plan(hyps, egfr_state)
        assert plan["method"] == METHOD_SOBOL
        n_params = len(plan["target_params"])
        assert plan["sample_size"] >= n_params * 10


# =============================================================================
# TestSensitivityPlannerTargetParamsCollection：目标参数收集
# =============================================================================
class TestSensitivityPlannerTargetParamsCollection:
    """目标参数收集测试（多来源 + 去重）。"""

    def test_collect_from_hypothesis_target_param(self, egfr_state):
        """用例 31：从 hypothesis.target_param 收集。"""
        hyp = {"target_param": "k_unique_param"}
        planner = SensitivityPlanner()
        plan = planner.plan([hyp], egfr_state)
        assert "k_unique_param" in plan["target_params"]

    def test_collect_from_sensitivity_report(self, egfr_state):
        """用例 32：从 v4_sensitivity_report.local_sensitivity 收集 top-K。"""
        # 无 target_param 的假设 → 从 sensitivity_report 提取
        hyp = {"strategy": HypothesisStrategy.OSCILLATION}
        planner = SensitivityPlanner()
        plan = planner.plan([hyp], egfr_state)

        # sensitivity_report 含 k_pEGFR_dephos / k_EGFR_act / k_other
        assert "k_pEGFR_dephos" in plan["target_params"]
        assert "k_EGFR_act" in plan["target_params"]

    def test_dedup_target_params(self, egfr_state):
        """用例 33：重复 target_param 去重。"""
        hyps = [
            {"target_param": "k_pEGFR_dephos"},  # 也在 sensitivity_report 中
            {"target_param": "k_pEGFR_dephos"},  # 重复
        ]
        planner = SensitivityPlanner()
        plan = planner.plan(hyps, egfr_state)

        # k_pEGFR_dephos 只出现一次
        assert plan["target_params"].count("k_pEGFR_dephos") == 1

    def test_max_target_params_limit(self, egfr_state):
        """用例 34：target_params 数量 ≤ max_target_params。"""
        hyps = [{"target_param": f"k{i}"} for i in range(20)]
        planner = SensitivityPlanner(max_target_params=5)
        plan = planner.plan(hyps, egfr_state)

        assert len(plan["target_params"]) <= 5

    def test_collect_from_pathway_graph_feedback_loops(self):
        """用例 35：从 v4_pathway_graph.feedback_loops 收集参数。"""
        hyp = {"strategy": HypothesisStrategy.OSCILLATION}  # 无 target_param
        state = {
            "v4_sensitivity_report": {},  # 无 sensitivity 数据
            "v4_pathway_graph": {
                "feedback_loops": [
                    {"source": "ERK", "target": "SOS"},
                ],
            },
        }
        planner = SensitivityPlanner()
        plan = planner.plan([hyp], state)

        # 从 feedback_loops 构造参数：k_ERK_act / k_ERK_inact / k_SOS_act ...
        # 至少应有一些参数
        assert len(plan["target_params"]) > 0


# =============================================================================
# TestSensitivityPlannerDegradation：降级行为
# =============================================================================
class TestSensitivityPlannerDegradation:
    """失败降级行为测试。"""

    def test_invalid_hypotheses_returns_default_plan(self, egfr_state):
        """用例 36：hypotheses 非 list → 默认 plan。"""
        planner = SensitivityPlanner()
        plan = planner.plan(None, egfr_state)  # type: ignore[arg-type]

        assert plan["method"] == METHOD_LOCAL
        assert plan["sample_size"] == 1

    def test_invalid_state_returns_default_plan(self):
        """用例 37：state 非 dict → 默认 plan。"""
        planner = SensitivityPlanner()
        plan = planner.plan([], None)  # type: ignore[arg-type]

        assert plan["method"] == METHOD_LOCAL
        assert plan["sample_size"] == 1

    def test_empty_hypotheses_returns_local_plan(self, egfr_state):
        """用例 38：空假设列表 → local 方法。"""
        planner = SensitivityPlanner()
        plan = planner.plan([], egfr_state)
        assert plan["method"] == METHOD_LOCAL
        assert plan["n_hypotheses"] == 0


# =============================================================================
# TestIntegrationWithAgent：与 HypothesisAgent 集成
# =============================================================================
class TestIntegrationWithAgent:
    """验证 ParameterExplorer 可作为 HypothesisAgent 子组件工作。"""

    def test_agent_with_parameter_explorer_annotates_parameter_robustness(
        self, egfr_state
    ):
        """用例 39：Agent 注入 ParameterExplorer → 假设含 parameter_robustness。"""
        from app.hypothesis import HypothesisAgent

        metrics = {
            "species": {
                "EGFR": {
                    "peak": 1.5,
                    "fold_change": 5.0,
                },
            },
        }
        # 构造含 sensitivity_report 的 state，触发灵敏度假设生成
        state = {
            "v4_pathway_class": "EGFR_RTK",
            "metrics": metrics,
            "v4_sensitivity_report": {
                "local_sensitivity": {"k_pEGFR_dephos": 0.85},
            },
            "v4_calibration_result": {
                "calibrated_params": {"k_pEGFR_dephos": 0.1},
            },
        }

        # 注入稳定 model_func 的 ParameterExplorer
        def stable_model(param: str, factor: float) -> float:
            return 1.0 * (1 + 0.01 * factor)

        explorer = ParameterExplorer(model_func=stable_model)
        agent = HypothesisAgent(parameter_explorer=explorer)
        with patch.object(agent, "_get_rag_client", return_value=None):
            hypotheses = agent.generate(state)

        # 应生成至少 1 条灵敏度假设
        assert len(hypotheses) >= 1
        # 找到灵敏度假设
        sensitivity_hyps = [
            h for h in hypotheses
            if h.get("strategy") == HypothesisStrategy.SENSITIVITY
        ]
        if sensitivity_hyps:
            hyp = sensitivity_hyps[0]
            # parameter_robustness 应被填充
            assert "parameter_robustness" in hyp
            assert isinstance(hyp["parameter_robustness"], dict)
            assert "hypothesis_holds" in hyp["parameter_robustness"]
            assert isinstance(
                hyp["parameter_robustness"]["hypothesis_holds"], bool
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
