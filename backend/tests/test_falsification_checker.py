# BioDynamics Agent v4 - Falsification Checker 单元测试（Phase 6 / Task 6.3.4）
#
# 覆盖 SubTask 6.3.1-6.3.3：
# - 6.3.1: FalsificationChecker 类创建
# - 6.3.2: 可证伪性规则（必须有可证伪预测 + 对照组 + 定量阈值）
# - 6.3.3: 过滤不可证伪假设
#
# 测试策略：
# - 构造可证伪 / 不可证伪假设用例（覆盖三规则）
# - 验证 check() 输出 falsifiable / falsification_criteria / failure_reasons
# - 验证 filter() 批量过滤行为
# - 不调用真实 LLM / 真实 RAG

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

# 确保能导入 app 包（backend/ 在 sys.path）
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.hypothesis import FalsificationChecker, HypothesisStrategy


# =============================================================================
# 测试夹具：构造各类假设
# =============================================================================
def _make_hypothesis(
    prediction: str = "敲低 IKK 后，NFkB 的振荡振幅下降 >50%",
    expected_result: str = "NFkB 振荡振幅下降 >50%",
    controls: list[str] | None = None,
    threshold: float | None = 0.5,
    statement: str = "NFkB 振荡由 IKK 反馈环决定",
    strategy: str = HypothesisStrategy.OSCILLATION,
) -> dict[str, Any]:
    """构造可配置的假设 dict。

    默认构造一个完全可证伪的假设（三规则全部通过）。
    通过参数覆盖来构造不可证伪的变体。
    """
    if controls is None:
        controls = ["vehicle", "untreated"]
    return {
        "id": "H001",
        "statement": statement,
        "prediction": prediction,
        "expected_result": expected_result,
        "experiment_design": {
            "controls": controls,
            "readout": {
                "species": "NFkB",
                "metric": "oscillation_amplitude",
                "threshold": threshold if threshold is not None else 0.5,
            },
        },
        "strategy": strategy,
    }


@pytest.fixture
def falsifiable_hypothesis() -> dict[str, Any]:
    """完全可证伪的假设（三规则全部通过）。"""
    return _make_hypothesis()


# =============================================================================
# TestFalsificationCheckerRules：三规则分别测试
# =============================================================================
class TestFalsificationCheckerRules:
    """SubTask 6.3.2: 可证伪性三规则测试。"""

    def test_rule1_pass_with_percentage_threshold(self):
        """用例 1：规则 1 通过 - prediction 含 '下降 >50%'。"""
        hyp = _make_hypothesis(prediction="敲低 IKK 后，NFkB 振荡振幅下降 >50%")
        checker = FalsificationChecker()
        result = checker.check(hyp)
        assert result["falsifiable"] is True

    def test_rule1_pass_with_fold_change(self):
        """用例 2：规则 1 通过 - prediction 含 '降低 >2 倍'。"""
        hyp = _make_hypothesis(prediction="处理后 pEGFR 水平降低 >2 倍")
        checker = FalsificationChecker()
        result = checker.check(hyp)
        assert result["falsifiable"] is True

    def test_rule1_pass_with_state_change(self):
        """用例 3：规则 1 通过 - prediction 含 '失去双稳态切换能力'。"""
        hyp = _make_hypothesis(
            prediction="Axin 基因敲除后，beta_catenin 失去双稳态切换能力",
            threshold=0.7,
        )
        checker = FalsificationChecker()
        result = checker.check(hyp)
        assert result["falsifiable"] is True

    def test_rule1_fail_with_vague_prediction(self):
        """用例 4：规则 1 失败 - prediction 含模糊表述 '会变化'。"""
        hyp = _make_hypothesis(prediction="敲低 IKK 后 NFkB 会变化")
        checker = FalsificationChecker()
        result = checker.check(hyp)
        assert result["falsifiable"] is False
        assert any("规则 1" in r for r in result["failure_reasons"])
        assert any("模糊" in r for r in result["failure_reasons"])

    def test_rule1_fail_with_no_direction(self):
        """用例 5：规则 1 失败 - prediction 无方向词 + 无阈值。"""
        hyp = _make_hypothesis(prediction="处理后 NFkB 振荡")
        checker = FalsificationChecker()
        result = checker.check(hyp)
        assert result["falsifiable"] is False
        assert any("规则 1" in r for r in result["failure_reasons"])

    def test_rule1_fail_with_empty_prediction(self):
        """用例 6：规则 1 失败 - prediction 为空。"""
        hyp = _make_hypothesis(prediction="")
        checker = FalsificationChecker()
        result = checker.check(hyp)
        assert result["falsifiable"] is False
        assert any("规则 1" in r for r in result["failure_reasons"])

    def test_rule2_pass_with_vehicle_control(self):
        """用例 7：规则 2 通过 - controls 含 vehicle。"""
        hyp = _make_hypothesis(controls=["vehicle", "untreated"])
        checker = FalsificationChecker()
        result = checker.check(hyp)
        assert result["falsifiable"] is True

    def test_rule2_pass_with_dmso_control(self):
        """用例 8：规则 2 通过 - controls 含 DMSO。"""
        hyp = _make_hypothesis(controls=["DMSO", "untreated"])
        checker = FalsificationChecker()
        result = checker.check(hyp)
        assert result["falsifiable"] is True

    def test_rule2_pass_with_scramble_sirna(self):
        """用例 9：规则 2 通过 - controls 含 scramble siRNA。"""
        hyp = _make_hypothesis(controls=["scramble siRNA"])
        checker = FalsificationChecker()
        result = checker.check(hyp)
        assert result["falsifiable"] is True

    def test_rule2_fail_with_no_standard_control(self):
        """用例 10：规则 2 失败 - controls 不含标准对照。"""
        hyp = _make_hypothesis(controls=["test_group_1", "test_group_2"])
        checker = FalsificationChecker()
        result = checker.check(hyp)
        assert result["falsifiable"] is False
        assert any("规则 2" in r for r in result["failure_reasons"])

    def test_rule2_fail_with_empty_controls(self):
        """用例 11：规则 2 失败 - controls 为空列表。"""
        hyp = _make_hypothesis(controls=[])
        checker = FalsificationChecker()
        result = checker.check(hyp)
        assert result["falsifiable"] is False
        assert any("规则 2" in r for r in result["failure_reasons"])

    def test_rule2_fail_with_missing_experiment_design(self):
        """用例 12：规则 2 失败 - experiment_design 缺失。"""
        hyp = _make_hypothesis()
        hyp["experiment_design"] = None
        checker = FalsificationChecker()
        result = checker.check(hyp)
        assert result["falsifiable"] is False
        assert any("规则 2" in r for r in result["failure_reasons"])

    def test_rule3_pass_with_readout_threshold(self):
        """用例 13：规则 3 通过 - readout.threshold = 0.5。"""
        hyp = _make_hypothesis(threshold=0.5)
        checker = FalsificationChecker()
        result = checker.check(hyp)
        assert result["falsifiable"] is True

    def test_rule3_pass_with_percentage_in_prediction(self):
        """用例 14：规则 3 通过 - prediction 含 '>50%'。"""
        hyp = _make_hypothesis(
            prediction="敲低 IKK 后，NFkB 振荡振幅下降 >50%",
            threshold=None,  # 移除 readout.threshold，强制走 prediction 路径
        )
        # 但 _make_hypothesis 默认 threshold=0.5，需要移除 threshold 字段
        hyp["experiment_design"]["readout"]["threshold"] = 0
        checker = FalsificationChecker()
        result = checker.check(hyp)
        assert result["falsifiable"] is True

    def test_rule3_pass_with_fold_in_expected_result(self):
        """用例 15：规则 3 通过 - expected_result 含 '>2 倍'。"""
        hyp = _make_hypothesis(
            prediction="敲低 IKK 后，NFkB 振荡振幅下降 >50%",
            expected_result="NFkB 振荡振幅降低 >2 倍",
        )
        hyp["experiment_design"]["readout"]["threshold"] = 0
        checker = FalsificationChecker()
        result = checker.check(hyp)
        assert result["falsifiable"] is True

    def test_rule3_fail_with_no_threshold(self):
        """用例 16：规则 3 失败 - 无任何定量阈值。"""
        hyp = _make_hypothesis(
            prediction="敲低 IKK 后，NFkB 振荡振幅下降",  # 无 %/倍/fold
            expected_result="NFkB 振荡振幅下降",  # 无 %/倍/fold
        )
        hyp["experiment_design"]["readout"]["threshold"] = 0
        checker = FalsificationChecker()
        result = checker.check(hyp)
        # 规则 1 也会失败（无阈值），规则 3 失败
        assert result["falsifiable"] is False
        assert any("规则 3" in r for r in result["failure_reasons"])


# =============================================================================
# TestFalsificationCheckerCheck：check() 接口完整性
# =============================================================================
class TestFalsificationCheckerCheck:
    """check() 接口完整性测试。"""

    def test_check_returns_three_required_fields(self, falsifiable_hypothesis):
        """用例 17：check 返回 falsifiable / falsification_criteria / failure_reasons。"""
        checker = FalsificationChecker()
        result = checker.check(falsifiable_hypothesis)

        assert "falsifiable" in result
        assert "falsification_criteria" in result
        assert "failure_reasons" in result
        assert isinstance(result["falsifiable"], bool)
        assert isinstance(result["falsification_criteria"], str)
        assert isinstance(result["failure_reasons"], list)

    def test_check_falsifiable_true_has_empty_failure_reasons(
        self, falsifiable_hypothesis
    ):
        """用例 18：falsifiable=True 时 failure_reasons 为空列表。"""
        checker = FalsificationChecker()
        result = checker.check(falsifiable_hypothesis)
        assert result["falsifiable"] is True
        assert result["failure_reasons"] == []

    def test_check_falsifiable_false_has_nonempty_failure_reasons(self):
        """用例 19：falsifiable=False 时 failure_reasons 非空。"""
        hyp = _make_hypothesis(
            prediction="NFkB 会变化",  # 模糊表述触发规则 1 失败
            controls=[],  # 触发规则 2 失败
        )
        hyp["experiment_design"]["readout"]["threshold"] = 0  # 触发规则 3 失败
        checker = FalsificationChecker()
        result = checker.check(hyp)
        assert result["falsifiable"] is False
        assert len(result["failure_reasons"]) > 0

    def test_check_falsifiable_true_criteria_mentions_prediction(
        self, falsifiable_hypothesis
    ):
        """用例 20：falsifiable=True 时 criteria 含 prediction 文本。"""
        checker = FalsificationChecker()
        result = checker.check(falsifiable_hypothesis)
        assert result["falsifiable"] is True
        # criteria 应含 prediction 内容
        assert "下降" in result["falsification_criteria"]
        assert "证伪" in result["falsification_criteria"]

    def test_check_invalid_hypothesis_returns_falsifiable_false(self):
        """用例 21：hypothesis 非 dict → falsifiable=False。"""
        checker = FalsificationChecker()
        result = checker.check(None)  # type: ignore[arg-type]
        assert result["falsifiable"] is False

    def test_check_exception_degrades_to_falsifiable_true(self):
        """用例 22：异常时降级返回 falsifiable=True（保守不阻塞）。

        构造一个会触发异常的假设（experiment_design.readout 为异常类型）。
        """
        hyp = _make_hypothesis()
        # 制造异常：readout 为不可迭代对象
        hyp["experiment_design"]["readout"] = "invalid_readout"  # type: ignore

        # patch _check_quantitative_threshold 抛异常（模拟内部异常）
        checker = FalsificationChecker()
        with patch.object(
            checker, "_check_quantitative_threshold",
            side_effect=RuntimeError("test exception"),
        ):
            result = checker.check(hyp)
            # 异常降级返回 falsifiable=True
            assert result["falsifiable"] is True
            assert "降级" in result["falsification_criteria"]


# =============================================================================
# TestFalsificationCheckerFilter：filter() 批量过滤
# =============================================================================
class TestFalsificationCheckerFilter:
    """SubTask 6.3.3: 过滤不可证伪假设。"""

    def test_filter_removes_non_falsifiable_hypotheses(self):
        """用例 23：filter 过滤掉不可证伪的假设。"""
        # 1 可证伪 + 1 不可证伪（模糊表述）
        falsifiable_hyp = _make_hypothesis(
            prediction="敲低 IKK 后，NFkB 振荡振幅下降 >50%",
        )
        non_falsifiable_hyp = _make_hypothesis(
            prediction="NFkB 会变化",  # 模糊 → 不可证伪
        )
        hypotheses = [falsifiable_hyp, non_falsifiable_hyp]

        checker = FalsificationChecker()
        filtered = checker.filter(hypotheses)
        # 应只保留 1 个可证伪假设
        assert len(filtered) == 1
        assert filtered[0]["prediction"] == "敲低 IKK 后，NFkB 振荡振幅下降 >50%"

    def test_filter_keeps_all_falsifiable_hypotheses(self):
        """用例 24：filter 保留所有可证伪假设。"""
        h1 = _make_hypothesis(prediction="敲低 X 后 Y 下降 >50%")
        h1["id"] = "H001"
        h2 = _make_hypothesis(prediction="敲低 X 后 Y 下降 >2 倍")
        h2["id"] = "H002"
        h3 = _make_hypothesis(prediction="敲低 X 后 Y 失去双稳态切换能力")
        h3["id"] = "H003"

        checker = FalsificationChecker()
        filtered = checker.filter([h1, h2, h3])
        assert len(filtered) == 3

    def test_filter_returns_empty_when_all_non_falsifiable(self):
        """用例 25：所有假设不可证伪 → filter 返回空列表。"""
        h1 = _make_hypothesis(prediction="X 会变化")
        h2 = _make_hypothesis(prediction="Y 可能改变")

        checker = FalsificationChecker()
        filtered = checker.filter([h1, h2])
        assert filtered == []

    def test_filter_returns_empty_for_empty_input(self):
        """用例 26：filter 空输入 → 返回空列表。"""
        checker = FalsificationChecker()
        assert checker.filter([]) == []

    def test_filter_returns_empty_for_invalid_input(self):
        """用例 27：filter 非 list 输入 → 返回空列表。"""
        checker = FalsificationChecker()
        assert checker.filter(None) == []  # type: ignore[arg-type]
        assert checker.filter("not a list") == []  # type: ignore[arg-type]

    def test_filter_skips_non_dict_items(self):
        """用例 28：filter 跳过非 dict 项。"""
        falsifiable_hyp = _make_hypothesis()
        hypotheses = [falsifiable_hyp, "not a dict", None, 42]

        checker = FalsificationChecker()
        filtered = checker.filter(hypotheses)
        assert len(filtered) == 1
        assert filtered[0]["id"] == "H001"

    def test_filter_annotates_falsifiable_field(self):
        """用例 29：filter 在保留的假设上标注 falsifiable + falsification_criteria。"""
        hyp = _make_hypothesis()
        # 初始 falsifiable 字段可能为 True（Generator 默认），但 filter 会覆盖
        hyp["falsifiable"] = True

        checker = FalsificationChecker()
        filtered = checker.filter([hyp])
        assert len(filtered) == 1
        assert filtered[0]["falsifiable"] is True
        assert "falsification_criteria" in filtered[0]
        assert isinstance(filtered[0]["falsification_criteria"], str)
        assert filtered[0]["falsification_criteria"]  # 非空


# =============================================================================
# TestFalsificationCheckerIntegration：与 HypothesisAgent 集成
# =============================================================================
class TestFalsificationCheckerIntegration:
    """验证 FalsificationChecker 可作为 HypothesisAgent 子组件工作。"""

    def test_agent_with_checker_annotates_falsifiable_field(self):
        """用例 30：HypothesisAgent 注入 FalsificationChecker → 假设含
        falsifiable + falsification_criteria 字段。"""
        from app.hypothesis import HypothesisAgent

        metrics = {
            "species": {
                "EGFR": {
                    "peak": 1.5,
                    "fold_change": 5.0,
                    "oscillation": {
                        "oscillatory": True,
                        "period_minutes": 60.0,
                        "n_peaks": 3,
                    },
                },
            },
        }
        state = {
            "v4_pathway_class": "EGFR_RTK",
            "metrics": metrics,
        }

        checker = FalsificationChecker()
        agent = HypothesisAgent(falsifiability_checker=checker)
        with patch.object(agent, "_get_rag_client", return_value=None):
            hypotheses = agent.generate(state)

        # 应生成至少 1 条假设
        assert len(hypotheses) >= 1
        hyp = hypotheses[0]

        # FalsificationChecker 应已标注 falsifiable + falsification_criteria
        assert "falsifiable" in hyp
        assert isinstance(hyp["falsifiable"], bool)
        # 由 Generator 生成的假设默认可证伪（含明确 prediction + 阈值），
        # 但 experiment_design 为空 dict（无 ExperimentDesigner 注入）→
        # 规则 2 会失败 → falsifiable=False
        # 这是一个合理的设计：未填充 experiment_design 的假设不可证伪
        if hyp["falsifiable"]:
            assert "falsification_criteria" in hyp
            assert hyp["falsification_criteria"]
        else:
            # 不可证伪的假设仍保留在列表中（Agent 的 _check_falsifiability 不删除，
            # 只标注；实际过滤由 filter() 完成）
            assert hyp.get("falsification_criteria", "")

    def test_agent_with_checker_and_designer_fully_falsifiable(self):
        """用例 31：Agent 同时注入 Checker + Designer → 假设完全可证伪。

        完整流水线：
        1. Generator 生成假设（含 prediction + 阈值）
        2. Designer 填充 experiment_design（含 controls + threshold）
        3. Checker 复核可证伪性 → falsifiable=True
        """
        from app.hypothesis import HypothesisAgent, ExperimentDesigner

        import app.pathways.specialists.egfr_specialist  # noqa: F401

        metrics = {
            "species": {
                "EGFR": {
                    "peak": 1.5,
                    "fold_change": 5.0,
                    "oscillation": {
                        "oscillatory": True,
                        "period_minutes": 60.0,
                        "n_peaks": 3,
                    },
                },
            },
        }
        state = {
            "v4_pathway_class": "EGFR_RTK",
            "metrics": metrics,
        }

        designer = ExperimentDesigner()
        checker = FalsificationChecker()
        agent = HypothesisAgent(
            experiment_designer=designer,
            falsifiability_checker=checker,
        )
        with patch.object(agent, "_get_rag_client", return_value=None):
            hypotheses = agent.generate(state)

        assert len(hypotheses) >= 1
        hyp = hypotheses[0]
        # 完整流水线 → falsifiable=True
        assert hyp["falsifiable"] is True
        assert hyp["falsification_criteria"]
        assert "证伪" in hyp["falsification_criteria"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
