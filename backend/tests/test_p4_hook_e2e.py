# BioDynamics Agent v4 - P4 Hook 端到端冒烟测试 (Phase 4 / Task 4.14)
#
# 6 冒烟用例：
#   1. test_egf_single_pathway_smoke: EGF 单通路 → EGFR_RTK
#   2. test_egf_pi3k_multi_pathway_smoke: EGF+PI3K 多通路 → MULTI:EGFR_RTK+PI3K_AKT_mTOR
#   3. test_apoptosis_pathway_smoke: Apoptosis 单通路 → APOPTOSIS
#   4. test_nf_kappa_b_pathway_smoke: NF-κB 单通路 → NF_KB
#   5. test_wnt_pathway_smoke: Wnt 单通路 → WNT
#   6. test_flag_off_no_op_smoke: flag=false → v4 字段不存在
#
# 测试策略：
# - 不实际调用 LLM（Pathway Planner 使用真实规则匹配，不依赖 LLM）
# - Specialist 直接调用 apply_* 方法（不依赖 LLM）
# - 验证 state 字段被正确填充
# - Feature Flag 通过 unittest.mock.patch 设置
#
# 运行：cd backend && python -m pytest tests/test_p4_hook_e2e.py -v

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# 确保能导入 app 包（backend/ 在 sys.path）
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


class TestP4HookE2E(unittest.TestCase):
    """P4 Hook 端到端冒烟测试：验证 3 个 hook 的链式调用与 state 填充。"""

    def _run_p4_hooks(self, user_input: str) -> dict:
        """辅助：依次调用 3 个 P4 hook，返回累积的 state 更新。

        Args:
            user_input: 用户输入文本

        Returns:
            合并了 3 个 hook 输出的 state dict
        """
        from app.pathways.pathway_planner import pathway_planner_hook_node
        from app.pathways.specialist_hook import specialist_hook_node
        from app.crosstalk.coordinator import crosstalk_coordinator_hook_node

        state = {"user_input": user_input}

        # 1. Pathway Planner hook
        planner_out = pathway_planner_hook_node(state)
        state.update(planner_out)

        # 2. Specialist hook（消费 v4_pathway_class + v4_pathway_graph）
        specialist_out = specialist_hook_node(state)
        state.update(specialist_out)

        # 3. Cross-talk Coordinator hook（消费 v4_pathway_class + v4_specialist_outputs）
        coordinator_out = crosstalk_coordinator_hook_node(state)
        state.update(coordinator_out)

        return state

    @patch("app.config.settings.V4_PATHWAY_PLANNER_ENABLED", True)
    @patch("app.config.settings.V4_PATHWAY_SPECIALIST_ENABLED", True)
    @patch("app.config.settings.V4_CROSSTALK_COORDINATOR_ENABLED", True)
    def test_egf_single_pathway_smoke(self):
        """用例 1：EGF 单通路 → v4_pathway_class="EGFR_RTK"。

        验证：
        - v4_pathway_class == "EGFR_RTK"
        - v4_specialist_outputs 非空
        - v4_crosstalk_edges 为空（单通路场景）
        """
        state = self._run_p4_hooks("EGF binds EGFR receptor")

        self.assertEqual(state.get("v4_pathway_class"), "EGFR_RTK")
        specialist_outputs = state.get("v4_specialist_outputs")
        self.assertIsNotNone(specialist_outputs)
        self.assertGreaterEqual(len(specialist_outputs), 1)
        self.assertEqual(specialist_outputs[0]["pathway_class"], "EGFR_RTK")
        # 单通路场景：crosstalk_edges 为空
        self.assertEqual(state.get("v4_crosstalk_edges"), [])

    @patch("app.config.settings.V4_PATHWAY_PLANNER_ENABLED", True)
    @patch("app.config.settings.V4_PATHWAY_SPECIALIST_ENABLED", True)
    @patch("app.config.settings.V4_CROSSTALK_COORDINATOR_ENABLED", True)
    def test_egf_pi3k_multi_pathway_smoke(self):
        """用例 2：EGF + PI3K 多通路 → MULTI:EGFR_RTK+PI3K_AKT_mTOR。

        验证：
        - v4_pathway_class == "MULTI:EGFR_RTK+PI3K_AKT_mTOR"
        - v4_specialist_outputs 含 2 条
        - v4_shared_species 非空（含 AKT 或 Ras）
        - v4_crosstalk_edges 非空（多通路场景）
        """
        state = self._run_p4_hooks(
            "EGF activates EGFR and PI3K activates AKT-mTOR signaling"
        )

        self.assertEqual(
            state.get("v4_pathway_class"),
            "MULTI:EGFR_RTK+PI3K_AKT_mTOR",
        )
        specialist_outputs = state.get("v4_specialist_outputs")
        self.assertIsNotNone(specialist_outputs)
        self.assertEqual(len(specialist_outputs), 2)

        # shared_species 非空
        shared_species = state.get("v4_shared_species")
        self.assertIsNotNone(shared_species)
        self.assertGreaterEqual(len(shared_species), 1)
        # 应包含 AKT（PI3K specialist 的 shared_species）或 Ras（EGFR specialist）
        self.assertTrue(
            "AKT" in shared_species or "Ras" in shared_species,
            f"shared_species 应包含 AKT 或 Ras，实际: {shared_species}",
        )

        # 多通路场景：crosstalk_edges 非空
        crosstalk_edges = state.get("v4_crosstalk_edges")
        self.assertIsNotNone(crosstalk_edges)
        self.assertGreaterEqual(len(crosstalk_edges), 1)

    @patch("app.config.settings.V4_PATHWAY_PLANNER_ENABLED", True)
    @patch("app.config.settings.V4_PATHWAY_SPECIALIST_ENABLED", True)
    @patch("app.config.settings.V4_CROSSTALK_COORDINATOR_ENABLED", True)
    def test_apoptosis_pathway_smoke(self):
        """用例 3：Apoptosis 单通路 → v4_pathway_class="APOPTOSIS"。

        注意：原 task 建议 "TNF + Caspase" 输入，但 TNF 同时是 APOPTOSIS 与
        NF_KB 的关键词，且 "apoptosis" 同时是 p53 与 APOPTOSIS 的关键词，
        会导致多通路匹配。此处改用 "caspase-3 cleaves PARP"（含 caspase
        关键词，唯一命中 APOPTOSIS，与 test_pathway_planner.py 一致）。
        """
        state = self._run_p4_hooks("caspase-3 cleaves PARP")

        self.assertEqual(state.get("v4_pathway_class"), "APOPTOSIS")
        specialist_outputs = state.get("v4_specialist_outputs")
        self.assertIsNotNone(specialist_outputs)
        self.assertGreaterEqual(len(specialist_outputs), 1)
        # 单通路场景：crosstalk_edges 为空
        self.assertEqual(state.get("v4_crosstalk_edges"), [])

    @patch("app.config.settings.V4_PATHWAY_PLANNER_ENABLED", True)
    @patch("app.config.settings.V4_PATHWAY_SPECIALIST_ENABLED", True)
    @patch("app.config.settings.V4_CROSSTALK_COORDINATOR_ENABLED", True)
    def test_nf_kappa_b_pathway_smoke(self):
        """用例 4：NF-κB 单通路 → v4_pathway_class="NF_KB"。

        注意：原 task 建议 "TNF + NF-κB" 输入，但 TNF 同时是 APOPTOSIS 与
        NF_KB 的关键词，会导致多通路匹配。此处改用 "NF-κB signaling pathway"
        （含 NF-κB 关键词，不含 TNF）以唯一命中 NF_KB。与 test_pathway_planner.py
        中 test_nf_kb 的处理方式一致。
        """
        state = self._run_p4_hooks("NF-κB signaling pathway")

        self.assertEqual(state.get("v4_pathway_class"), "NF_KB")
        specialist_outputs = state.get("v4_specialist_outputs")
        self.assertIsNotNone(specialist_outputs)
        self.assertGreaterEqual(len(specialist_outputs), 1)
        # 单通路场景：crosstalk_edges 为空
        self.assertEqual(state.get("v4_crosstalk_edges"), [])

    @patch("app.config.settings.V4_PATHWAY_PLANNER_ENABLED", True)
    @patch("app.config.settings.V4_PATHWAY_SPECIALIST_ENABLED", True)
    @patch("app.config.settings.V4_CROSSTALK_COORDINATOR_ENABLED", True)
    def test_wnt_pathway_smoke(self):
        """用例 5：Wnt 单通路 → v4_pathway_class="WNT"。

        验证：
        - v4_pathway_class == "WNT"
        - v4_specialist_outputs 非空
        - v4_crosstalk_edges 为空（单通路）
        """
        state = self._run_p4_hooks("Wnt signaling β-catenin")

        self.assertEqual(state.get("v4_pathway_class"), "WNT")
        specialist_outputs = state.get("v4_specialist_outputs")
        self.assertIsNotNone(specialist_outputs)
        self.assertGreaterEqual(len(specialist_outputs), 1)
        # 单通路场景：crosstalk_edges 为空
        self.assertEqual(state.get("v4_crosstalk_edges"), [])

    @patch("app.config.settings.V4_PATHWAY_PLANNER_ENABLED", False)
    @patch("app.config.settings.V4_PATHWAY_SPECIALIST_ENABLED", False)
    @patch("app.config.settings.V4_CROSSTALK_COORDINATOR_ENABLED", False)
    def test_flag_off_no_op_smoke(self):
        """用例 6：flag=false → v4 字段不存在，state 不被污染。

        验证：
        - 所有 3 个 hook 返回 {}
        - state 中不出现 v4_pathway_class / v4_specialist_outputs /
          v4_crosstalk_edges / v4_shared_species 等 v4 前缀字段
        - v3 字段（user_input）不受影响
        """
        from app.pathways.pathway_planner import pathway_planner_hook_node
        from app.pathways.specialist_hook import specialist_hook_node
        from app.crosstalk.coordinator import crosstalk_coordinator_hook_node

        state = {"user_input": "EGF binds EGFR"}

        planner_out = pathway_planner_hook_node(state)
        specialist_out = specialist_hook_node(state)
        coordinator_out = crosstalk_coordinator_hook_node(state)

        # 所有 hook 返回空 dict
        self.assertEqual(planner_out, {})
        self.assertEqual(specialist_out, {})
        self.assertEqual(coordinator_out, {})

        # state 不被污染（不出现 v4_ 前缀字段）
        v4_keys = [k for k in state.keys() if k.startswith("v4_")]
        self.assertEqual(v4_keys, [], f"flag=false 时不应出现 v4_ 字段: {v4_keys}")

        # v3 字段不受影响
        self.assertEqual(state.get("user_input"), "EGF binds EGFR")


if __name__ == "__main__":
    unittest.main()
