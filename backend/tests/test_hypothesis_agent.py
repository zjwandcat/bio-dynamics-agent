# BioDynamics Agent v4 - Hypothesis Agent 单元测试（Phase 6 / Task 6.1.6）
#
# 覆盖 SubTask 6.1.1-6.1.5：
# - 6.1.1: HypothesisAgent + HypothesisGenerator 创建
# - 6.1.2: 假设生成策略（振荡→反馈环；双稳态→阈值；灵敏度→参数）
# - 6.1.3: Hypothesis schema（id/statement/prediction/experiment_design/
#          validation_method/expected_result/falsifiable/supporting_pmids/
#          contradicting_pmids）
# - 6.1.4: 文献检索验证（mock rag_client）
# - 6.1.5: state 字段 v4_hypothesis_list + Feature Flag V4_HYPOTHESIS_AGENT_ENABLED
#
# 测试策略：
# - mock rag_client.search_params 返回固定文献列表
# - mock settings.V4_HYPOTHESIS_AGENT_ENABLED / V4_VALIDATION_PYRAMID_ENABLED
# - 不调用真实 LLM / 真实 RAG 检索

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
    HypothesisAgent,
    HypothesisGenerator,
    HypothesisStrategy,
    hypothesis_agent_hook_node,
)


# =============================================================================
# 测试夹具
# =============================================================================
@pytest.fixture
def oscillation_metrics() -> dict[str, Any]:
    """构造含振荡特征的 metrics（模拟 NF-κB 振荡）。"""
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
            "IKK": {
                "peak": 2.0,
                "peak_time": 10.0,
                "fold_change": 3.0,
            },
        },
        "overall": {"confidence": 0.8},
    }


@pytest.fixture
def bistability_metrics() -> dict[str, Any]:
    """构造含双稳态特征的 metrics（模拟 Wnt/β-catenin 切换）。"""
    return {
        "species": {
            "beta_catenin": {
                "peak": 5.0,
                "peak_time": 120.0,
                "fold_change": 8.0,
                "bistability": {
                    "bistable": True,
                    "threshold": 2.5,
                },
            },
        },
        "overall": {"confidence": 0.7},
    }


@pytest.fixture
def sensitivity_report() -> dict[str, Any]:
    """构造 P5 Sensitivity 报告（含 local_sensitivity）。"""
    return {
        "local_sensitivity": {
            "k1": 0.85,   # 高灵敏度
            "k2": -0.62,  # 高灵敏度（负）
            "k3": 0.05,   # 低灵敏度
            "Kd": 0.42,   # 中等灵敏度
        },
        "method": "local_only",
        "salib_available": False,
        "warnings": ["SALib 未安装，仅运行 local sensitivity"],
    }


@pytest.fixture
def mixed_metrics(
    oscillation_metrics: dict[str, Any],
    sensitivity_report: dict[str, Any],
) -> dict[str, Any]:
    """构造同时含振荡 + 双稳态 + 灵敏度的混合 metrics。"""
    metrics = dict(oscillation_metrics)
    metrics["species"]["beta_catenin"] = {
        "peak": 5.0,
        "peak_time": 120.0,
        "fold_change": 8.0,
        "bistability": {"bistable": True, "threshold": 2.5},
    }
    return metrics


@pytest.fixture
def passed_validation_state() -> dict[str, Any]:
    """构造 Validation 通过的 state。"""
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
            "entries": [
                {"pmid": "PMID:14975635", "species": "NFkB"},
                {"pmid": "PMID:12906785", "species": "beta_catenin"},
            ],
            "integrity": True,
        },
        "v4_pathway_class": "NF_KB",
    }


@pytest.fixture
def failed_validation_state() -> dict[str, Any]:
    """构造 Validation 失败的 state。"""
    return {
        "v4_validation_report": {
            "overall_pass": False,
            "level1": {"pass": False, "reason": "mass_conservation_violation"},
        },
        "v4_pathway_class": "EGFR_RTK",
    }


@pytest.fixture
def mock_rag_client() -> MagicMock:
    """构造 mock RagClient，search_params 返回固定文献列表。"""
    client = MagicMock()
    client.search_params.return_value = [
        {
            "source": "PMID:14975635",
            "text": "NF-κB oscillation period 1-2h driven by IKK feedback",
            "summary": "Nelson 2004 NF-κB oscillation",
        },
        {
            "source": "PMID:12124381",
            "text": "EGF-induced MAPK cascade not oscillatory in this study",
            "summary": "Schoeberl 2002 pEGFR dynamics",
        },
        {
            "source": "PMID:12906785",
            "text": "Wnt/β-catenin bistability threshold determined by Axin",
            "summary": "Lee 2003 Wnt destruction complex",
        },
    ]
    return client


# =============================================================================
# SubTask 6.1.1 + 6.1.2: HypothesisGenerator 假设生成策略
# =============================================================================
class TestHypothesisGenerator:
    """HypothesisGenerator 单元测试（SubTask 6.1.2 假设生成策略）。"""

    def test_oscillation_strategy_generates_feedback_hypothesis(
        self,
        oscillation_metrics: dict[str, Any],
    ) -> None:
        """振荡特征 → 反馈环假设（spec.md 第 362 行）。"""
        generator = HypothesisGenerator()
        candidates = generator.generate(
            metrics=oscillation_metrics,
            pathway_class="NF_KB",
        )

        # 至少生成 1 条振荡假设
        osc_hyps = [
            h for h in candidates
            if h.get("strategy") == HypothesisStrategy.OSCILLATION
        ]
        assert len(osc_hyps) >= 1, "振荡特征应生成至少 1 条反馈环假设"

        # 验证假设 schema（spec.md 第 355 行）
        hyp = osc_hyps[0]
        assert "id" in hyp and hyp["id"].startswith("H")
        assert "statement" in hyp and "反馈环" in hyp["statement"]
        assert "prediction" in hyp and "下降" in hyp["prediction"]
        assert "expected_result" in hyp
        assert "validation_method" in hyp
        assert hyp["falsifiable"] is True
        assert hyp["supporting_pmids"] == []
        assert hyp["contradicting_pmids"] == []
        # 反馈环节点 Y 应为 IKK（NF_KB 通路特异映射）
        assert hyp["feedback_node"] == "IKK"
        assert hyp["target_species"] == "NFkB"

    def test_bistability_strategy_generates_threshold_hypothesis(
        self,
        bistability_metrics: dict[str, Any],
    ) -> None:
        """双稳态特征 → 阈值假设（spec.md 第 363 行）。"""
        generator = HypothesisGenerator()
        candidates = generator.generate(
            metrics=bistability_metrics,
            pathway_class="WNT",
        )

        bis_hyps = [
            h for h in candidates
            if h.get("strategy") == HypothesisStrategy.BISTABILITY
        ]
        assert len(bis_hyps) >= 1, "双稳态特征应生成至少 1 条阈值假设"

        hyp = bis_hyps[0]
        assert "阈值" in hyp["statement"]
        assert "敲除" in hyp["prediction"]
        # 阈值节点 Z 应为 Axin（WNT 通路特异映射）
        assert hyp["threshold_node"] == "Axin"
        assert hyp["target_species"] == "beta_catenin"

    def test_sensitivity_strategy_generates_parameter_hypothesis(
        self,
        oscillation_metrics: dict[str, Any],
        sensitivity_report: dict[str, Any],
    ) -> None:
        """灵敏度特征 → 参数假设（spec.md 第 364 行）。"""
        generator = HypothesisGenerator()
        candidates = generator.generate(
            metrics=oscillation_metrics,
            v4_sensitivity_report=sensitivity_report,
            pathway_class="NF_KB",
        )

        sens_hyps = [
            h for h in candidates
            if h.get("strategy") == HypothesisStrategy.SENSITIVITY
        ]
        assert len(sens_hyps) >= 1, "灵敏度特征应生成至少 1 条参数假设"

        # top-3 高灵敏度参数：k1=0.85, k2=-0.62, Kd=0.42（k3=0.05 应被过滤）
        param_names = [h["target_param"] for h in sens_hyps]
        assert "k1" in param_names
        assert "k2" in param_names
        assert "k3" not in param_names, "低灵敏度参数 k3=0.05 不应生成假设"

        # 验证 k1 假设（正灵敏度 → "降低"）
        k1_hyp = next(h for h in sens_hyps if h["target_param"] == "k1")
        assert "k1" in k1_hyp["statement"]
        assert k1_hyp["sensitivity"] == pytest.approx(0.85)
        assert "降低" in k1_hyp["prediction"]

        # 验证 k2 假设（负灵敏度 → "升高"）
        k2_hyp = next(h for h in sens_hyps if h["target_param"] == "k2")
        assert "升高" in k2_hyp["prediction"]

    def test_no_features_returns_empty_list(self) -> None:
        """无振荡/双稳态/灵敏度特征 → 返回空列表。"""
        generator = HypothesisGenerator()
        metrics = {
            "species": {
                "EGFR": {"peak": 1.0, "peak_time": 5.0, "fold_change": 2.0},
            },
        }
        candidates = generator.generate(metrics=metrics, pathway_class="EGFR_RTK")
        assert candidates == []

    def test_invalid_metrics_returns_empty_list(self) -> None:
        """无效 metrics（非 dict）→ 返回空列表（失败降级）。"""
        generator = HypothesisGenerator()
        candidates = generator.generate(metrics="invalid", pathway_class="EGFR_RTK")  # type: ignore[arg-type]
        assert candidates == []

    def test_hypothesis_id_increments_sequentially(
        self,
        mixed_metrics: dict[str, Any],
        sensitivity_report: dict[str, Any],
    ) -> None:
        """假设 ID 应顺序递增（H001, H002, ...）。"""
        generator = HypothesisGenerator()
        candidates = generator.generate(
            metrics=mixed_metrics,
            v4_sensitivity_report=sensitivity_report,
            pathway_class="NF_KB",
        )

        ids = [h["id"] for h in candidates]
        # ID 应唯一且递增
        assert len(ids) == len(set(ids)), "假设 ID 应唯一"
        for i, hyp_id in enumerate(ids, start=1):
            assert hyp_id == f"H{i:03d}", f"第 {i} 个假设 ID 应为 H{i:03d}，实际为 {hyp_id}"

    def test_pathway_class_specific_feedback_node_mapping(
        self,
        oscillation_metrics: dict[str, Any],
    ) -> None:
        """通路特异反馈环节点映射（NF_KB→IKK，p53→MDM2，WNT→Axin）。"""
        generator = HypothesisGenerator()

        # NF_KB → IKK
        candidates = generator.generate(metrics=oscillation_metrics, pathway_class="NF_KB")
        assert candidates[0]["feedback_node"] == "IKK"

        # p53 → MDM2
        candidates = generator.generate(metrics=oscillation_metrics, pathway_class="p53")
        assert candidates[0]["feedback_node"] == "MDM2"

        # WNT → Axin
        candidates = generator.generate(metrics=oscillation_metrics, pathway_class="WNT")
        assert candidates[0]["feedback_node"] == "Axin"

        # 未知通路 → "上游激酶"（默认）
        candidates = generator.generate(metrics=oscillation_metrics, pathway_class="UNKNOWN")
        assert candidates[0]["feedback_node"] == "上游激酶"


# =============================================================================
# SubTask 6.1.3 + 6.1.4 + 6.1.5: HypothesisAgent 主类
# =============================================================================
class TestHypothesisAgent:
    """HypothesisAgent 单元测试（SubTask 6.1.3 schema + 6.1.4 文献检索 + 6.1.5 state）。"""

    def test_generate_returns_hypothesis_list_with_full_schema(
        self,
        oscillation_metrics: dict[str, Any],
        passed_validation_state: dict[str, Any],
    ) -> None:
        """HypothesisAgent.generate 返回完整 schema 的假设列表（spec.md 第 355 行）。"""
        state = {**passed_validation_state, "metrics": oscillation_metrics}
        agent = HypothesisAgent(rag_client=None)  # 不做文献检索
        # patch _get_rag_client 避免真实 RagClient 创建（加速测试）
        with patch.object(agent, "_get_rag_client", return_value=None):
            hypotheses = agent.generate(state)

        assert len(hypotheses) >= 1
        for hyp in hypotheses:
            # spec.md 第 355 行完整 schema 校验
            assert "id" in hyp
            assert "statement" in hyp
            assert "prediction" in hyp
            assert "experiment_design" in hyp
            assert "validation_method" in hyp
            assert "expected_result" in hyp
            assert "falsifiable" in hyp
            assert isinstance(hyp["falsifiable"], bool)
            assert "supporting_pmids" in hyp
            assert isinstance(hyp["supporting_pmids"], list)
            assert "contradicting_pmids" in hyp
            assert isinstance(hyp["contradicting_pmids"], list)
            # Level 5 兼容字段
            assert hyp["hypothesis_id"] == hyp["id"]
            assert hyp["falsifying_pmids"] == hyp["contradicting_pmids"]

    def test_validation_failure_short_circuits_to_empty_list(
        self,
        failed_validation_state: dict[str, Any],
        oscillation_metrics: dict[str, Any],
    ) -> None:
        """Validation 失败短路（spec.md 第 394 行）。"""
        # 启用 P5 Validation，且 overall_pass=False
        state = {**failed_validation_state, "metrics": oscillation_metrics}
        agent = HypothesisAgent()
        with patch("app.hypothesis.hypothesis_agent.settings") as mock_settings:
            mock_settings.V4_VALIDATION_PYRAMID_ENABLED = True
            hypotheses = agent.generate(state)

        assert hypotheses == [], "Validation 失败时应短路返回空列表"

    def test_validation_disabled_allows_hypothesis_generation(
        self,
        oscillation_metrics: dict[str, Any],
        passed_validation_state: dict[str, Any],
    ) -> None:
        """P5 Validation 未启用时，Hypothesis 仍可执行（不阻塞）。"""
        state = {**passed_validation_state, "metrics": oscillation_metrics}
        # 删除 validation_report 模拟 P5 未启用
        state.pop("v4_validation_report", None)

        agent = HypothesisAgent(rag_client=None)
        with patch("app.hypothesis.hypothesis_agent.settings") as mock_settings:
            mock_settings.V4_VALIDATION_PYRAMID_ENABLED = False
            with patch.object(agent, "_get_rag_client", return_value=None):
                hypotheses = agent.generate(state)

        assert len(hypotheses) >= 1, "P5 未启用时应允许生成假设"

    def test_literature_search_fills_supporting_and_contradicting_pmids(
        self,
        oscillation_metrics: dict[str, Any],
        passed_validation_state: dict[str, Any],
        mock_rag_client: MagicMock,
    ) -> None:
        """文献检索填充 supporting_pmids / contradicting_pmids（SubTask 6.1.4）。"""
        state = {**passed_validation_state, "metrics": oscillation_metrics}
        agent = HypothesisAgent(rag_client=mock_rag_client)
        hypotheses = agent.generate(state)

        assert len(hypotheses) >= 1
        # 至少有 1 条假设填充了 supporting_pmids
        has_supporting = any(len(h["supporting_pmids"]) > 0 for h in hypotheses)
        assert has_supporting, "文献检索应填充 supporting_pmids"

        # mock_rag_client 第 2 条记录含 " not " → contradicting
        has_contradicting = any(len(h["contradicting_pmids"]) > 0 for h in hypotheses)
        assert has_contradicting, "文献检索应识别 contradicting_pmids（含 'not' 关键词）"

        # 验证 rag_client.search_params 被调用
        assert mock_rag_client.search_params.called

    def test_literature_search_marks_grounded_pmids(
        self,
        oscillation_metrics: dict[str, Any],
        passed_validation_state: dict[str, Any],
        mock_rag_client: MagicMock,
    ) -> None:
        """与 SBML Grounder 交互：grounding_ledger 中的 PMID 标记 grounded（spec.md 第 397 行）。"""
        state = {**passed_validation_state, "metrics": oscillation_metrics}
        agent = HypothesisAgent(rag_client=mock_rag_client)
        hypotheses = agent.generate(state)

        # grounding_ledger 含 PMID:14975635 → supporting_pmids 应标记 |grounded
        all_supporting = []
        for h in hypotheses:
            all_supporting.extend(h["supporting_pmids"])

        grounded_pmids = [p for p in all_supporting if p.endswith("|grounded")]
        assert len(grounded_pmids) >= 1, "grounding_ledger 中的 PMID 应标记 |grounded"
        assert any("PMID:14975635" in p for p in grounded_pmids)

    def test_rag_client_failure_degrades_to_empty_pmids(
        self,
        oscillation_metrics: dict[str, Any],
        passed_validation_state: dict[str, Any],
    ) -> None:
        """rag_client 不可用时降级到空 pmids（不阻塞）。"""
        state = {**passed_validation_state, "metrics": oscillation_metrics}
        # rag_client 创建失败 → _get_rag_client 返回 None
        agent = HypothesisAgent(rag_client=None)
        with patch(
            "app.hypothesis.hypothesis_agent.HypothesisAgent._get_rag_client",
            return_value=None,
        ):
            hypotheses = agent.generate(state)

        assert len(hypotheses) >= 1
        # 所有假设 pmids 应为空
        for h in hypotheses:
            assert h["supporting_pmids"] == []
            assert h["contradicting_pmids"] == []

    def test_exception_returns_empty_list(
        self,
        passed_validation_state: dict[str, Any],
    ) -> None:
        """异常时返回空列表（铁律 #4 失败降级）。"""
        state = {**passed_validation_state, "metrics": "invalid"}  # type: ignore[dict-item]
        agent = HypothesisAgent()
        # metrics 非 dict 时 generate 内部已处理；这里测试 generator 异常
        with patch.object(
            agent._generator,
            "generate",
            side_effect=RuntimeError("generator crashed"),
        ):
            hypotheses = agent.generate(state)
        assert hypotheses == []

    def test_subcomponent_hooks_invoked_when_provided(
        self,
        oscillation_metrics: dict[str, Any],
        passed_validation_state: dict[str, Any],
    ) -> None:
        """子组件钩子（experiment_designer / falsifiability_checker /
        parameter_explorer）被调用时填充对应字段。"""
        state = {**passed_validation_state, "metrics": oscillation_metrics}

        # mock 子组件
        designer = MagicMock()
        designer.design.return_value = {
            "perturbation": "siRNA_IKK",
            "readout": "NFkB_WB",
            "time_points": [0, 30, 60, 90, 120],
            "controls": ["scramble_siRNA"],
            "cell_line": "HeLa",
            "expected_result": "振荡振幅下降 >50%",
        }
        checker = MagicMock()
        checker.check.return_value = {
            "falsifiable": True,
            "falsification_criteria": "振荡振幅下降 <30% 即证伪",
        }
        explorer = MagicMock()
        explorer.explore.return_value = {
            "param": "k_feedback",
            "range": [0.5, 2.0],
            "hypothesis_holds": True,
        }

        agent = HypothesisAgent(
            experiment_designer=designer,
            falsifiability_checker=checker,
            parameter_explorer=explorer,
            rag_client=None,
        )
        with patch.object(agent, "_get_rag_client", return_value=None):
            hypotheses = agent.generate(state)

        assert len(hypotheses) >= 1
        for h in hypotheses:
            assert h["experiment_design"] != {}
            assert "perturbation" in h["experiment_design"]
            assert h["falsification_criteria"] == "振荡振幅下降 <30% 即证伪"
            assert h["parameter_robustness"]["hypothesis_holds"] is True

        # 验证子组件被调用
        assert designer.design.called
        assert checker.check.called
        assert explorer.explore.called


# =============================================================================
# SubTask 6.1.5: Feature Flag 隔离 + LangGraph hook
# =============================================================================
class TestHypothesisHookFlagIsolation:
    """Feature Flag V4_HYPOTHESIS_AGENT_ENABLED 隔离测试。"""

    def test_hook_returns_empty_when_flag_off(
        self,
        passed_validation_state: dict[str, Any],
        oscillation_metrics: dict[str, Any],
    ) -> None:
        """V4_HYPOTHESIS_AGENT_ENABLED=false 时 hook 返回 {}（不修改 state）。"""
        state = {**passed_validation_state, "metrics": oscillation_metrics}
        with patch("app.hypothesis.hypothesis_agent.settings") as mock_settings:
            mock_settings.V4_HYPOTHESIS_AGENT_ENABLED = False
            update = hypothesis_agent_hook_node(state)

        assert update == {}, "flag=false 时 hook 应返回空 dict"

    def test_hook_returns_hypothesis_list_when_flag_on(
        self,
        passed_validation_state: dict[str, Any],
        oscillation_metrics: dict[str, Any],
    ) -> None:
        """V4_HYPOTHESIS_AGENT_ENABLED=true 时 hook 返回 v4_hypothesis_list。"""
        state = {**passed_validation_state, "metrics": oscillation_metrics}
        with patch("app.hypothesis.hypothesis_agent.settings") as mock_settings:
            mock_settings.V4_HYPOTHESIS_AGENT_ENABLED = True
            mock_settings.V4_VALIDATION_PYRAMID_ENABLED = True
            with patch(
                "app.hypothesis.hypothesis_agent.HypothesisAgent._get_rag_client",
                return_value=None,
            ):
                update = hypothesis_agent_hook_node(state)

        assert "v4_hypothesis_list" in update
        assert isinstance(update["v4_hypothesis_list"], list)
        assert update["v4_hypothesis_generated"] is True

    def test_hook_no_v3_pollution_when_flag_off(
        self,
        passed_validation_state: dict[str, Any],
    ) -> None:
        """flag=false 时 hook 不污染 v3 字段。"""
        state = {**passed_validation_state, "metrics": {}}
        state["network_json"] = {"v3_key": "v3_value"}
        state["parameters"] = {"v3_param": 1.0}

        with patch("app.hypothesis.hypothesis_agent.settings") as mock_settings:
            mock_settings.V4_HYPOTHESIS_AGENT_ENABLED = False
            update = hypothesis_agent_hook_node(state)

        # update 应为空，不修改任何 v3 字段
        assert update == {}
        assert state["network_json"] == {"v3_key": "v3_value"}
        assert state["parameters"] == {"v3_param": 1.0}

    def test_hook_exception_degrades_to_empty_dict(
        self,
        passed_validation_state: dict[str, Any],
    ) -> None:
        """hook 异常时降级返回空 dict（不阻塞流水线）。"""
        state = {**passed_validation_state, "metrics": {}}
        with patch("app.hypothesis.hypothesis_agent.settings") as mock_settings:
            mock_settings.V4_HYPOTHESIS_AGENT_ENABLED = True
            mock_settings.V4_VALIDATION_PYRAMID_ENABLED = True
            with patch(
                "app.hypothesis.hypothesis_agent.HypothesisAgent.generate",
                side_effect=RuntimeError("agent crashed"),
            ):
                update = hypothesis_agent_hook_node(state)

        assert update == {}, "异常时 hook 应降级返回空 dict"
