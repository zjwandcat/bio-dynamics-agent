# BioDynamics Agent v4 - Level 5 Hypothesis Validation 单元测试
# (Phase 5 / Task 5.6.4)
#
# 测试 Level5HypothesisValidator 主类 + 假设支持检查 + P6 skipped + LangGraph hook。
#
# 测试用例（29 个，>= 20 要求）：
#   - TestLevel5HypothesisValidator: validate() 主入口 + 异常降级
#   - TestLevel5HookNode: Feature Flag + hook 行为
#   - TestHypothesisSupport: 单个假设文献/实验支持/证伪/low_confidence
#   - TestHypothesisListValidation: 多假设聚合 validated/falsified 计数
#   - TestP6NotEnabled: P6 未启用 skipped pass=True
#
# 运行：cd backend && python -m pytest tests/test_level5_hypothesis.py -v

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# 确保能导入 app 包（backend/ 在 sys.path）
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.validation_v2.level5_hypothesis import (
    Level5HypothesisValidator,
    level5_hook_node,
)


# =============================================================================
# Mock 数据
# =============================================================================

# --- 单个假设：有文献支持 ---
MOCK_HYPOTHESIS_WITH_LITERATURE = {
    "hypothesis_id": "H1",
    "statement": "EGF stimulation leads to EGFR autophosphorylation",
    "supporting_pmids": ["PMID:12124381"],  # Schoeberl 2002
}

# --- 单个假设：有实验数据支持 ---
MOCK_HYPOTHESIS_WITH_EXPERIMENT = {
    "hypothesis_id": "H2",
    "statement": "PI3K inhibition reduces AKT phosphorylation",
    "supporting_pmids": [],  # 无文献支持
}

# --- 单个假设：无任何支持（low_confidence）---
MOCK_HYPOTHESIS_NO_SUPPORT = {
    "hypothesis_id": "H3",
    "statement": "Unknown hypothesis without any evidence",
    "supporting_pmids": [],  # 无文献支持
}

# --- 单个假设：被实验数据证伪 ---
MOCK_HYPOTHESIS_FALSIFIED = {
    "hypothesis_id": "H4",
    "statement": "MEK inhibition increases ERK phosphorylation",
    "supporting_pmids": ["PMID:14757805"],  # 有文献支持但被实验证伪
}

# --- 单个假设：含文献证伪（falsifying_pmids）---
MOCK_HYPOTHESIS_WITH_LITERATURE_FALSIFICATION = {
    "hypothesis_id": "H5",
    "statement": "Hypothesis falsified by literature",
    "supporting_pmids": ["PMID:11111111"],
    "falsifying_pmids": ["PMID:22222222"],
}

# --- 实验数据：支持 H2，证伪 H4 ---
MOCK_EXPERIMENTAL_DATA = {
    "validated_hypothesis_ids": ["H2"],
    "falsified_hypothesis_ids": ["H4"],
    "observations": [
        {"experiment": "PI3K inhibitor assay", "result": "AKT pT308 decreased"},
    ],
}

# --- 完整假设列表（含 5 个假设：H1 文献 / H2 实验 / H3 无支持 / H4 证伪 / H5 文献证伪）---
MOCK_FULL_HYPOTHESIS_LIST = [
    MOCK_HYPOTHESIS_WITH_LITERATURE,
    MOCK_HYPOTHESIS_WITH_EXPERIMENT,
    MOCK_HYPOTHESIS_NO_SUPPORT,
    MOCK_HYPOTHESIS_FALSIFIED,
    MOCK_HYPOTHESIS_WITH_LITERATURE_FALSIFICATION,
]


# =============================================================================
# TestLevel5HypothesisValidator: validate() 主入口
# =============================================================================
class TestLevel5HypothesisValidator(unittest.TestCase):
    """测试 Level5HypothesisValidator.validate() 主入口。"""

    def setUp(self):
        self.validator = Level5HypothesisValidator()

    @patch("app.validation_v2.level5_hypothesis.settings")
    def test_validate_p6_enabled_with_hypotheses(self, mock_settings):
        """P6 启用 + 有假设 → 执行验证，pass=True。"""
        mock_settings.V4_HYPOTHESIS_AGENT_ENABLED = True
        state = {
            "v4_hypothesis_list": MOCK_FULL_HYPOTHESIS_LIST,
            "experimental_data": MOCK_EXPERIMENTAL_DATA,
        }
        report = self.validator.validate(state)
        self.assertTrue(report["pass"])
        self.assertGreater(report["hypotheses_validated"], 0)
        self.assertGreaterEqual(report["hypotheses_falsified"], 1)
        self.assertIn("evidence_support", report)
        self.assertIn("low_confidence", report)

    @patch("app.validation_v2.level5_hypothesis.settings")
    def test_validate_invalid_state_type(self, mock_settings):
        """state 不是 dict → pass=False（异常降级）。"""
        mock_settings.V4_HYPOTHESIS_AGENT_ENABLED = True
        report = self.validator.validate("not_a_dict")  # type: ignore[arg-type]
        self.assertFalse(report["pass"])
        self.assertIn("reason", report)

    @patch("app.validation_v2.level5_hypothesis.settings")
    def test_validate_exception_degradation(self, mock_settings):
        """validate 内部异常 → pass=False（不抛异常）。"""
        mock_settings.V4_HYPOTHESIS_AGENT_ENABLED = True
        with patch.object(
            self.validator, "_is_p6_enabled", side_effect=RuntimeError("boom")
        ):
            report = self.validator.validate({"v4_hypothesis_list": []})
        self.assertFalse(report["pass"])

    @patch("app.validation_v2.level5_hypothesis.settings")
    def test_validate_report_has_required_fields(self, mock_settings):
        """验证报告含 spec.md 第 315 行规定的 4 个字段。"""
        mock_settings.V4_HYPOTHESIS_AGENT_ENABLED = True
        state = {
            "v4_hypothesis_list": MOCK_FULL_HYPOTHESIS_LIST,
            "experimental_data": MOCK_EXPERIMENTAL_DATA,
        }
        report = self.validator.validate(state)
        self.assertIn("pass", report)
        self.assertIn("hypotheses_validated", report)
        self.assertIn("hypotheses_falsified", report)
        self.assertIn("evidence_support", report)
        # low_confidence 是新增字段（spec 失败策略要求）
        self.assertIn("low_confidence", report)

    @patch("app.validation_v2.level5_hypothesis.settings")
    def test_validate_pass_is_true_when_low_confidence(self, mock_settings):
        """low_confidence=True 时 pass 仍为 True（不阻塞）。"""
        mock_settings.V4_HYPOTHESIS_AGENT_ENABLED = True
        state = {
            "v4_hypothesis_list": [MOCK_HYPOTHESIS_NO_SUPPORT],
        }
        report = self.validator.validate(state)
        self.assertTrue(report["pass"])
        self.assertTrue(report["low_confidence"])

    @patch("app.validation_v2.level5_hypothesis.settings")
    def test_validate_pass_is_true_when_falsified(self, mock_settings):
        """假设被证伪时 pass 仍为 True（不阻塞）。"""
        mock_settings.V4_HYPOTHESIS_AGENT_ENABLED = True
        state = {
            "v4_hypothesis_list": [MOCK_HYPOTHESIS_FALSIFIED],
            "experimental_data": MOCK_EXPERIMENTAL_DATA,
        }
        report = self.validator.validate(state)
        self.assertTrue(report["pass"])
        self.assertEqual(report["hypotheses_falsified"], 1)


# =============================================================================
# TestLevel5HookNode: Feature Flag + hook 行为
# =============================================================================
class TestLevel5HookNode(unittest.TestCase):
    """测试 level5_hook_node LangGraph 节点。"""

    def test_hook_flag_off_returns_empty(self):
        """V4_VALIDATION_PYRAMID_ENABLED=false → hook 返回 {}。"""
        with patch("app.validation_v2.level5_hypothesis.settings") as mock_settings:
            mock_settings.V4_VALIDATION_PYRAMID_ENABLED = False
            mock_settings.V4_HYPOTHESIS_AGENT_ENABLED = True
            result = level5_hook_node({"v4_hypothesis_list": MOCK_FULL_HYPOTHESIS_LIST})
        self.assertEqual(result, {})

    @patch("app.validation_v2.level5_hypothesis.settings")
    def test_hook_flag_on_writes_level5_report(self, mock_settings):
        """V4_VALIDATION_PYRAMID_ENABLED=true → hook 写入 v4_validation_report.level5。"""
        mock_settings.V4_VALIDATION_PYRAMID_ENABLED = True
        mock_settings.V4_HYPOTHESIS_AGENT_ENABLED = True
        state = {
            "v4_hypothesis_list": MOCK_FULL_HYPOTHESIS_LIST,
            "experimental_data": MOCK_EXPERIMENTAL_DATA,
        }
        result = level5_hook_node(state)
        self.assertIn("v4_validation_report", result)
        self.assertIn("level5", result["v4_validation_report"])
        level5 = result["v4_validation_report"]["level5"]
        self.assertIn("pass", level5)

    @patch("app.validation_v2.level5_hypothesis.settings")
    def test_hook_preserves_existing_report(self, mock_settings):
        """hook 不覆盖已存在的 level1 / level2 / level3 / level4 报告。"""
        mock_settings.V4_VALIDATION_PYRAMID_ENABLED = True
        mock_settings.V4_HYPOTHESIS_AGENT_ENABLED = True
        state = {
            "v4_hypothesis_list": MOCK_FULL_HYPOTHESIS_LIST,
            "v4_validation_report": {
                "level1": {"pass": True},
                "level2": {"pass": False, "track": "skipped"},
                "level3": {"pass": True, "skipped": True},
                "level4": {"pass": True, "benchmarks": []},
            },
        }
        result = level5_hook_node(state)
        report = result["v4_validation_report"]
        self.assertIn("level1", report)
        self.assertIn("level2", report)
        self.assertIn("level3", report)
        self.assertIn("level4", report)
        self.assertIn("level5", report)
        # 旧报告保持原值
        self.assertTrue(report["level1"]["pass"])
        self.assertEqual(report["level2"]["track"], "skipped")

    @patch("app.validation_v2.level5_hypothesis.settings")
    def test_hook_exception_returns_empty(self, mock_settings):
        """hook 内部异常 → 返回 {}（不阻塞主流水线）。"""
        mock_settings.V4_VALIDATION_PYRAMID_ENABLED = True
        mock_settings.V4_HYPOTHESIS_AGENT_ENABLED = True
        with patch(
            "app.validation_v2.level5_hypothesis.Level5HypothesisValidator.validate",
            side_effect=RuntimeError("boom"),
        ):
            result = level5_hook_node({"v4_hypothesis_list": []})
        self.assertEqual(result, {})

    @patch("app.validation_v2.level5_hypothesis.settings")
    def test_hook_p6_not_enabled_writes_skipped(self, mock_settings):
        """P6 未启用时 hook 写入 skipped pass=True。"""
        mock_settings.V4_VALIDATION_PYRAMID_ENABLED = True
        mock_settings.V4_HYPOTHESIS_AGENT_ENABLED = False
        state = {"v4_hypothesis_list": MOCK_FULL_HYPOTHESIS_LIST}
        result = level5_hook_node(state)
        level5 = result["v4_validation_report"]["level5"]
        self.assertTrue(level5["pass"])
        self.assertTrue(level5.get("skipped"))
        self.assertEqual(level5.get("reason"), "P6_hypothesis_agent_not_enabled")


# =============================================================================
# TestHypothesisSupport: 单个假设支持检查
# =============================================================================
class TestHypothesisSupport(unittest.TestCase):
    """测试 _check_hypothesis_support。"""

    def setUp(self):
        self.validator = Level5HypothesisValidator()

    def test_hypothesis_with_literature_support(self):
        """假设有文献支持（supporting_pmids 非空）→ validated=True。"""
        result = self.validator._check_hypothesis_support(
            MOCK_HYPOTHESIS_WITH_LITERATURE, {}
        )
        self.assertTrue(result["validated"])
        self.assertFalse(result["falsified"])
        self.assertFalse(result["low_confidence"])
        self.assertGreater(len(result["evidence_support"]), 0)
        # 文献证据含 type=literature, support=supporting
        ev = result["evidence_support"][0]
        self.assertEqual(ev["type"], "literature")
        self.assertEqual(ev["support"], "supporting")
        self.assertEqual(ev["source"], "PMID:12124381")

    def test_hypothesis_with_experimental_support(self):
        """假设有实验数据支持 → validated=True。"""
        result = self.validator._check_hypothesis_support(
            MOCK_HYPOTHESIS_WITH_EXPERIMENT, MOCK_EXPERIMENTAL_DATA
        )
        self.assertTrue(result["validated"])
        self.assertFalse(result["falsified"])
        self.assertFalse(result["low_confidence"])
        # 实验证据含 type=experimental, support=supporting
        exp_ev = [e for e in result["evidence_support"] if e["type"] == "experimental"]
        self.assertGreater(len(exp_ev), 0)
        self.assertEqual(exp_ev[0]["support"], "supporting")

    def test_hypothesis_no_support_low_confidence(self):
        """假设无任何文献/实验支持 → low_confidence=True。"""
        result = self.validator._check_hypothesis_support(
            MOCK_HYPOTHESIS_NO_SUPPORT, {}
        )
        self.assertFalse(result["validated"])
        self.assertFalse(result["falsified"])
        self.assertTrue(result["low_confidence"])
        self.assertEqual(len(result["evidence_support"]), 0)

    def test_hypothesis_falsified_by_experiment(self):
        """假设被实验数据证伪 → falsified=True（实验证伪优先级最高）。"""
        result = self.validator._check_hypothesis_support(
            MOCK_HYPOTHESIS_FALSIFIED, MOCK_EXPERIMENTAL_DATA
        )
        self.assertTrue(result["falsified"])
        # 文献支持仍存在，但被实验证伪覆盖
        # 注意：is_validated 仍可为 True（有文献支持），但 falsified=True 优先
        self.assertTrue(result["validated"])  # 文献支持仍存在
        # 实验证伪证据
        falsifying_ev = [
            e for e in result["evidence_support"] if e["support"] == "falsifying"
        ]
        self.assertGreater(len(falsifying_ev), 0)

    def test_hypothesis_with_literature_falsification(self):
        """假设含 falsifying_pmids → 文献证伪证据记录但不直接判定 falsified。"""
        result = self.validator._check_hypothesis_support(
            MOCK_HYPOTHESIS_WITH_LITERATURE_FALSIFICATION, {}
        )
        # 无实验数据证伪 → falsified=False
        self.assertFalse(result["falsified"])
        # 仍有文献支持 → validated=True
        self.assertTrue(result["validated"])
        self.assertFalse(result["low_confidence"])
        # 文献证伪证据存在
        lit_falsifying = [
            e
            for e in result["evidence_support"]
            if e["type"] == "literature" and e["support"] == "falsifying"
        ]
        self.assertGreater(len(lit_falsifying), 0)
        self.assertEqual(lit_falsifying[0]["source"], "PMID:22222222")

    def test_hypothesis_evidence_support_includes_required_fields(self):
        """evidence_support 每条含 type / source / support / hypothesis_id。"""
        result = self.validator._check_hypothesis_support(
            MOCK_HYPOTHESIS_WITH_LITERATURE, MOCK_EXPERIMENTAL_DATA
        )
        # H1 不在 validated_hypothesis_ids（H2 才在），但 H1 有文献支持
        ev = result["evidence_support"][0]
        self.assertIn("type", ev)
        self.assertIn("source", ev)
        self.assertIn("support", ev)
        self.assertIn("hypothesis_id", ev)

    def test_hypothesis_result_has_required_fields(self):
        """单假设结果含 hypothesis_id / validated / falsified / evidence_support / low_confidence。"""
        result = self.validator._check_hypothesis_support(
            MOCK_HYPOTHESIS_WITH_LITERATURE, {}
        )
        self.assertIn("hypothesis_id", result)
        self.assertIn("validated", result)
        self.assertIn("falsified", result)
        self.assertIn("evidence_support", result)
        self.assertIn("low_confidence", result)


# =============================================================================
# TestHypothesisListValidation: 多假设聚合验证
# =============================================================================
class TestHypothesisListValidation(unittest.TestCase):
    """测试 _validate_hypothesis_list（聚合 validated/falsified 计数）。"""

    def setUp(self):
        self.validator = Level5HypothesisValidator()

    def test_validate_full_hypothesis_list_aggregation(self):
        """完整假设列表聚合：H1 文献 + H2 实验 → validated=2，H4 证伪 → falsified=1。"""
        report = self.validator._validate_hypothesis_list(
            MOCK_FULL_HYPOTHESIS_LIST, MOCK_EXPERIMENTAL_DATA
        )
        # H1 (文献支持) + H2 (实验支持) → validated=2
        # H4 (实验证伪) → falsified=1
        # H3 (无支持) → low_confidence
        # H5 (含文献证伪但无实验证伪) → validated=True (有文献支持)
        self.assertEqual(report["hypotheses_validated"], 3)  # H1, H2, H5
        self.assertEqual(report["hypotheses_falsified"], 1)  # H4
        self.assertTrue(report["low_confidence"])  # H3 触发
        self.assertTrue(report["pass"])  # 始终 True
        # evidence_support 应包含所有假设的证据
        self.assertGreater(len(report["evidence_support"]), 0)

    def test_validate_empty_hypothesis_list(self):
        """空假设列表 → validated=0, falsified=0, low_confidence=False。"""
        report = self.validator._validate_hypothesis_list([], {})
        self.assertTrue(report["pass"])
        self.assertEqual(report["hypotheses_validated"], 0)
        self.assertEqual(report["hypotheses_falsified"], 0)
        self.assertFalse(report["low_confidence"])
        self.assertEqual(len(report["evidence_support"]), 0)

    def test_validate_all_validated(self):
        """所有假设都有支持 → validated=N, falsified=0, low_confidence=False。"""
        hypotheses = [
            {
                "hypothesis_id": "H1",
                "statement": "test1",
                "supporting_pmids": ["PMID:111"],
            },
            {
                "hypothesis_id": "H2",
                "statement": "test2",
                "supporting_pmids": ["PMID:222"],
            },
        ]
        report = self.validator._validate_hypothesis_list(hypotheses, {})
        self.assertEqual(report["hypotheses_validated"], 2)
        self.assertEqual(report["hypotheses_falsified"], 0)
        self.assertFalse(report["low_confidence"])

    def test_validate_all_low_confidence(self):
        """所有假设都无支持 → low_confidence=True，validated=0。"""
        hypotheses = [
            MOCK_HYPOTHESIS_NO_SUPPORT,
            {
                "hypothesis_id": "H6",
                "statement": "another unsupported",
                "supporting_pmids": [],
            },
        ]
        report = self.validator._validate_hypothesis_list(hypotheses, {})
        self.assertEqual(report["hypotheses_validated"], 0)
        self.assertEqual(report["hypotheses_falsified"], 0)
        self.assertTrue(report["low_confidence"])

    def test_validate_mixed_validated_falsified_low_confidence(self):
        """混合场景：validated + falsified + low_confidence 同时存在。"""
        hypotheses = [
            MOCK_HYPOTHESIS_WITH_LITERATURE,  # validated
            MOCK_HYPOTHESIS_NO_SUPPORT,       # low_confidence
            MOCK_HYPOTHESIS_FALSIFIED,        # falsified (有文献支持但实验证伪)
        ]
        report = self.validator._validate_hypothesis_list(
            hypotheses, MOCK_EXPERIMENTAL_DATA
        )
        self.assertEqual(report["hypotheses_validated"], 1)  # 仅 H1
        self.assertEqual(report["hypotheses_falsified"], 1)  # H4
        self.assertTrue(report["low_confidence"])  # H3 触发
        self.assertTrue(report["pass"])  # 始终 True（不阻塞）

    def test_validate_evidence_support_aggregation(self):
        """evidence_support 正确聚合所有假设的证据。"""
        report = self.validator._validate_hypothesis_list(
            MOCK_FULL_HYPOTHESIS_LIST, MOCK_EXPERIMENTAL_DATA
        )
        # H1 (1 文献) + H2 (1 实验) + H3 (0) + H4 (1 文献 + 1 实验) + H5 (1 文献 + 1 文献)
        # = 1 + 1 + 0 + 2 + 2 = 6
        self.assertEqual(len(report["evidence_support"]), 6)

    def test_validate_pass_always_true(self):
        """无论验证结果如何，pass 始终 True（spec 不阻塞）。"""
        # 即使全部证伪
        hypotheses = [
            {
                "hypothesis_id": "H1",
                "statement": "test",
                "supporting_pmids": ["PMID:111"],
            }
        ]
        experimental_data = {
            "falsified_hypothesis_ids": ["H1"],
        }
        report = self.validator._validate_hypothesis_list(hypotheses, experimental_data)
        self.assertTrue(report["pass"])

    def test_validate_object_hypothesis_supported(self):
        """假设为对象（非 dict）时仍能正常验证。"""
        class MockHypothesis:
            def __init__(self, hid, statement, pmids):
                self.hypothesis_id = hid
                self.statement = statement
                self.supporting_pmids = pmids

        hypotheses = [
            MockHypothesis("H1", "test", ["PMID:111"]),
        ]
        report = self.validator._validate_hypothesis_list(hypotheses, {})
        self.assertEqual(report["hypotheses_validated"], 1)


# =============================================================================
# TestP6NotEnabled: P6 未启用 skipped
# =============================================================================
class TestP6NotEnabled(unittest.TestCase):
    """测试 P6 未启用 skipped 逻辑。"""

    def setUp(self):
        self.validator = Level5HypothesisValidator()

    @patch("app.validation_v2.level5_hypothesis.settings")
    def test_p6_not_enabled_returns_skipped_pass_true(self, mock_settings):
        """P6 未启用 → skipped pass=True（spec.md 第 317 行）。"""
        mock_settings.V4_HYPOTHESIS_AGENT_ENABLED = False
        state = {
            "v4_hypothesis_list": MOCK_FULL_HYPOTHESIS_LIST,
            "experimental_data": MOCK_EXPERIMENTAL_DATA,
        }
        report = self.validator.validate(state)
        self.assertTrue(report["pass"])
        self.assertTrue(report.get("skipped"))
        self.assertEqual(report.get("reason"), "P6_hypothesis_agent_not_enabled")
        self.assertEqual(report["hypotheses_validated"], 0)
        self.assertEqual(report["hypotheses_falsified"], 0)
        self.assertEqual(len(report["evidence_support"]), 0)

    @patch("app.validation_v2.level5_hypothesis.settings")
    def test_p6_enabled_but_empty_hypothesis_list_skipped(self, mock_settings):
        """P6 启用但 v4_hypothesis_list 为空 → skipped pass=True。"""
        mock_settings.V4_HYPOTHESIS_AGENT_ENABLED = True
        state = {
            "v4_hypothesis_list": [],
            "experimental_data": MOCK_EXPERIMENTAL_DATA,
        }
        report = self.validator.validate(state)
        self.assertTrue(report["pass"])
        self.assertTrue(report.get("skipped"))
        self.assertEqual(report.get("reason"), "empty_hypothesis_list")

    @patch("app.validation_v2.level5_hypothesis.settings")
    def test_p6_enabled_but_missing_hypothesis_list_skipped(self, mock_settings):
        """P6 启用但 v4_hypothesis_list 字段缺失 → skipped pass=True。"""
        mock_settings.V4_HYPOTHESIS_AGENT_ENABLED = True
        state = {
            "experimental_data": MOCK_EXPERIMENTAL_DATA,
            # 缺 v4_hypothesis_list 字段
        }
        report = self.validator.validate(state)
        self.assertTrue(report["pass"])
        self.assertTrue(report.get("skipped"))
        self.assertEqual(report.get("reason"), "empty_hypothesis_list")

    @patch("app.validation_v2.level5_hypothesis.settings")
    def test_p6_enabled_but_hypothesis_list_not_list_skipped(self, mock_settings):
        """P6 启用但 v4_hypothesis_list 非 list → skipped pass=True。"""
        mock_settings.V4_HYPOTHESIS_AGENT_ENABLED = True
        state = {
            "v4_hypothesis_list": "not_a_list",  # type: ignore[dict-item]
        }
        report = self.validator.validate(state)
        self.assertTrue(report["pass"])
        self.assertTrue(report.get("skipped"))

    def test_skipped_report_has_required_fields(self):
        """skipped 报告含 spec 规定的字段。"""
        report = self.validator._run_skipped("P6_hypothesis_agent_not_enabled")
        self.assertTrue(report["pass"])
        self.assertEqual(report["hypotheses_validated"], 0)
        self.assertEqual(report["hypotheses_falsified"], 0)
        self.assertEqual(report["evidence_support"], [])
        self.assertTrue(report["skipped"])
        self.assertEqual(report["reason"], "P6_hypothesis_agent_not_enabled")

    def test_skipped_pass_is_true(self):
        """P6 未启用 skipped pass=True（与 Level 2 skipped pass=False 不同）。"""
        report = self.validator._run_skipped("P6_hypothesis_agent_not_enabled")
        self.assertTrue(report["pass"])

    def test_skipped_low_confidence_is_false(self):
        """skipped 时 low_confidence=False（无假设被验证）。"""
        report = self.validator._run_skipped("P6_hypothesis_agent_not_enabled")
        self.assertFalse(report["low_confidence"])

    def test_p6_enabled_check_default_false(self):
        """V4_HYPOTHESIS_AGENT_ENABLED 默认 False（P6 未实现）。"""
        # 不 patch，使用真实 settings（默认 false）
        result = self.validator._is_p6_enabled({})
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
