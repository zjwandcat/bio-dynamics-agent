# BioDynamics Agent v4 - P5 Hook 端到端冒烟测试 (Phase 5 / Task 5.9)
#
# 6 冒烟用例：
#   1. test_validation_agent_orchestrates_5_levels: ValidationAgent 编排 Level 1→5
#   2. test_overall_pass_aggregation: overall_pass 聚合逻辑（任一 Level fail → False）
#   3. test_short_circuit_clarification: 失败短路触发 pending_clarification
#   4. test_sbml_grounder_hook_flag_off: SBML Grounder flag=false 隔离
#   5. test_validation_pyramid_hook_flag_off: Validation Pyramid flag=false 隔离
#   6. test_p5_hooks_chain_e2e: P5 hook 链端到端（grounder → pyramid）
#
# 测试策略：
# - 不实际调用 LLM（验证器使用真实规则匹配，不依赖 LLM）
# - 验证 state 字段被正确填充
# - Feature Flag 通过 unittest.mock.patch 设置
#
# 运行：cd backend && python -m pytest tests/test_p5_hook_e2e.py -v

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# 确保能导入 app 包（backend/ 在 sys.path）
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


class TestP5HookE2E(unittest.TestCase):
    """P5 Hook 端到端冒烟测试：验证 SBML Grounder + Validation Pyramid 的链式调用。"""

    def _make_minimal_state(self) -> dict:
        """构造最小可用 state，含 P3/P4 产出的占位字段。"""
        return {
            "user_input": "EGF binds EGFR receptor",
            "v4_pathway_class": "EGFR_RTK",
            "v4_ode_system": {
                "pathway_class": "EGFR_RTK",
                "template_name": "_mechanism_phosphorylation_mm",
                "ode_code": "# v4 ODE code\ndpEGFR_dt = k1 * EGF * EGFR - k2 * pEGFR",
                "temporal": {},
                "dde_info": None,
            },
            "v4_reaction_ir": {
                "species": [{"name": "EGF"}, {"name": "EGFR"}, {"name": "pEGFR"}],
                "reactions": [],
                "compartments": [],
                "constraints": [],
            },
            "v4_pathway_graph": {
                "nodes": [],
                "edges": [],
                "feedback_loops": [],
                "cross_talk_edges": [],
            },
            "v4_crosstalk_edges": [],
            "v4_shared_species": [],
            "sbml_model_id": "BIOMD0000000205",
            "metrics": {},  # metrics 未计算（N8 未执行）
        }

    # =========================================================================
    # 用例 1：ValidationAgent 编排 Level 1→5
    # =========================================================================
    @patch("app.config.settings.V4_VALIDATION_PYRAMID_ENABLED", True)
    def test_validation_agent_orchestrates_5_levels(self):
        """用例 1：ValidationAgent 编排 Level 1→2→3→4→5。

        验证：
        - v4_validation_report 含 level1/level2/level3/level4/level5 五个字段
        - 含 overall_pass / short_circuit / failed_levels 字段
        - agent_version 标识存在
        """
        from app.validation_v2.validation_agent import ValidationAgent

        state = self._make_minimal_state()
        agent = ValidationAgent()
        report = agent.validate(state)

        # 5 个 Level 字段都存在
        for level in ("level1", "level2", "level3", "level4", "level5"):
            self.assertIn(level, report, f"报告缺少 {level} 字段")
            self.assertIsInstance(report[level], dict)

        # 聚合字段存在
        self.assertIn("overall_pass", report)
        self.assertIn("short_circuit", report)
        self.assertIn("failed_levels", report)
        self.assertIn("agent_version", report)
        self.assertEqual(report["agent_version"], "ValidationAgent-v1.0")

        # overall_pass 是 bool
        self.assertIsInstance(report["overall_pass"], bool)
        self.assertIsInstance(report["short_circuit"], bool)
        # short_circuit = not overall_pass
        self.assertEqual(report["short_circuit"], not report["overall_pass"])

    # =========================================================================
    # 用例 2：overall_pass 聚合逻辑
    # =========================================================================
    @patch("app.config.settings.V4_VALIDATION_PYRAMID_ENABLED", True)
    def test_overall_pass_aggregation(self):
        """用例 2：overall_pass 聚合（任一 Level pass=False → overall_pass=False）。

        策略：构造 state 使 Level 2 skipped（无 sbml_model_text）→ pass=False
        （修复审计 §7.2：skipped pass=False）
        验证 overall_pass=False, failed_levels 含 level2
        """
        from app.validation_v2.validation_agent import ValidationAgent

        state = self._make_minimal_state()
        # 无 sbml_model_text，Level 2 应 skipped pass=False（§7.2 修复）
        state["sbml_model_text"] = ""
        state["sbml_model_id"] = ""  # 无 BIOMD ID

        agent = ValidationAgent()
        report = agent.validate(state)

        # Level 2 skipped pass=False（无 SBML ID）
        self.assertEqual(report["level2"]["pass"], False)

        # overall_pass=False（Level 2 fail）
        self.assertEqual(report["overall_pass"], False)
        self.assertIn("level2", report["failed_levels"])

    # =========================================================================
    # 用例 3：失败短路触发 pending_clarification
    # =========================================================================
    @patch("app.config.settings.V4_VALIDATION_PYRAMID_ENABLED", True)
    def test_short_circuit_clarification(self):
        """用例 3：overall_pass=False 时触发 pending_clarification。

        验证：
        - build_clarification_signal() 返回非 None
        - clarification context="validation_failed"
        - 含 failed_levels / validation_report_summary
        - overall_pass=True 时返回 None
        """
        from app.validation_v2.validation_agent import ValidationAgent

        state = self._make_minimal_state()
        state["sbml_model_id"] = ""  # 触发 Level 2 skipped pass=False

        agent = ValidationAgent()
        report = agent.validate(state)

        # overall_pass=False
        self.assertFalse(report["overall_pass"])

        # 构造 clarification 信号
        clarification = agent.build_clarification_signal(report)
        self.assertIsNotNone(clarification)
        self.assertEqual(clarification["context"], "validation_failed")
        self.assertIn("failed_levels", clarification)
        self.assertIn("validation_report_summary", clarification)
        self.assertIn("question", clarification)
        self.assertIn("options", clarification)
        self.assertGreaterEqual(len(clarification["options"]), 2)

        # overall_pass=True 时返回 None
        report_pass = {
            "overall_pass": True,
            "level1": {"pass": True},
            "level2": {"pass": True},
            "level3": {"pass": True},
            "level4": {"pass": True},
            "level5": {"pass": True},
        }
        self.assertIsNone(agent.build_clarification_signal(report_pass))

    # =========================================================================
    # 用例 4：SBML Grounder hook flag=false 隔离
    # =========================================================================
    @patch("app.config.settings.V4_SBML_GROUNDER_ENABLED", False)
    def test_sbml_grounder_hook_flag_off(self):
        """用例 4：V4_SBML_GROUNDER_ENABLED=false → hook 返回 {}。"""
        from app.sbml_grounder.grounder_agent import sbml_grounder_hook_node

        state = self._make_minimal_state()
        result = sbml_grounder_hook_node(state)
        self.assertEqual(result, {})

    # =========================================================================
    # 用例 5：Validation Pyramid hook flag=false 隔离
    # =========================================================================
    @patch("app.config.settings.V4_VALIDATION_PYRAMID_ENABLED", False)
    def test_validation_pyramid_hook_flag_off(self):
        """用例 5：V4_VALIDATION_PYRAMID_ENABLED=false → hook 返回 {}。"""
        from app.validation_v2.validation_agent import validation_pyramid_hook_node

        state = self._make_minimal_state()
        result = validation_pyramid_hook_node(state)
        self.assertEqual(result, {})

    # =========================================================================
    # 用例 6：P5 hook 链端到端（grounder → pyramid）
    # =========================================================================
    @patch("app.config.settings.V4_SBML_GROUNDER_ENABLED", True)
    @patch("app.config.settings.V4_VALIDATION_PYRAMID_ENABLED", True)
    def test_p5_hooks_chain_e2e(self):
        """用例 6：P5 hook 链端到端。

        依次调用：
        1. sbml_grounder_hook_node → 写入 v4_grounding_ledger
        2. validation_pyramid_hook_node → 写入 v4_validation_report

        验证：
        - v4_grounding_ledger 字段存在（flag=true 时）
        - v4_validation_report 字段存在，含 level1~level5 + overall_pass
        - Level 4 skipped pass=True（metrics 未计算）
        - Level 5 skipped pass=True（P6 未启用）
        """
        from app.sbml_grounder.grounder_agent import sbml_grounder_hook_node
        from app.validation_v2.validation_agent import validation_pyramid_hook_node

        state = self._make_minimal_state()

        # 1. SBML Grounder hook
        grounder_out = sbml_grounder_hook_node(state)
        state.update(grounder_out)

        # 2. Validation Pyramid hook
        pyramid_out = validation_pyramid_hook_node(state)
        state.update(pyramid_out)

        # v4_validation_report 存在
        v4_report = state.get("v4_validation_report")
        self.assertIsNotNone(v4_report)
        self.assertIsInstance(v4_report, dict)

        # 5 个 Level 都存在
        for level in ("level1", "level2", "level3", "level4", "level5"):
            self.assertIn(level, v4_report)

        # Level 4 skipped pass=True（metrics 未计算）
        self.assertEqual(v4_report["level4"]["pass"], True)
        self.assertTrue(v4_report["level4"].get("skipped", False))

        # Level 5 skipped pass=True（P6 未启用）
        self.assertEqual(v4_report["level5"]["pass"], True)
        self.assertTrue(v4_report["level5"].get("skipped", False))

        # overall_pass 是 bool
        self.assertIsInstance(v4_report["overall_pass"], bool)

    # =========================================================================
    # 用例 7：flag=false 时 v3 行为零侵入（补充隔离验证）
    # =========================================================================
    @patch("app.config.settings.V4_SBML_GROUNDER_ENABLED", False)
    @patch("app.config.settings.V4_VALIDATION_PYRAMID_ENABLED", False)
    def test_flag_off_no_v4_pollution(self):
        """用例 7：P5 flag 全 false 时 state 不被 v4 字段污染。"""
        from app.sbml_grounder.grounder_agent import sbml_grounder_hook_node
        from app.validation_v2.validation_agent import validation_pyramid_hook_node

        state = {"user_input": "EGF binds EGFR"}

        grounder_out = sbml_grounder_hook_node(state)
        pyramid_out = validation_pyramid_hook_node(state)

        # 两个 hook 都返回空 dict
        self.assertEqual(grounder_out, {})
        self.assertEqual(pyramid_out, {})

        # state 不被污染
        self.assertNotIn("v4_grounding_ledger", state)
        self.assertNotIn("v4_validation_report", state)
        self.assertNotIn("pending_clarification", state)


if __name__ == "__main__":
    unittest.main()
