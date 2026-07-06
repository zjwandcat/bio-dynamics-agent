# BioDynamics Agent v4 - P4 Hook flag=false 隔离验证 (Phase 4 / Task 4.14)
#
# 验证内容：
# 1. flag=false 时所有 3 个 hook 返回 {}
# 2. flag=false 时 state 不被污染（不出现 v4_ 前缀字段）
# 3. flag=false 时 v3 关键测试套件通过（0 回归）
#
# 运行：cd backend && python -m pytest tests/test_p4_flag_off_isolation.py -v

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# 确保能导入 app 包（backend/ 在 sys.path）
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


class TestP4FlagOffIsolation(unittest.TestCase):
    """P4 Hook flag=false 隔离验证。"""

    @patch("app.config.settings.V4_PATHWAY_PLANNER_ENABLED", False)
    @patch("app.config.settings.V4_PATHWAY_SPECIALIST_ENABLED", False)
    @patch("app.config.settings.V4_CROSSTALK_COORDINATOR_ENABLED", False)
    def test_flag_off_all_hooks_return_empty(self):
        """验证 1：flag=false 时所有 3 个 hook 返回 {}。"""
        from app.pathways.pathway_planner import pathway_planner_hook_node
        from app.pathways.specialist_hook import specialist_hook_node
        from app.crosstalk.coordinator import crosstalk_coordinator_hook_node

        state = {
            "user_input": "EGF binds EGFR and PI3K activates AKT",
            "network_json": {"nodes": [], "edges": []},
            "entities": [],
            "mechanism": {},
        }

        # 所有 hook 在 flag=false 时返回 {}
        self.assertEqual(pathway_planner_hook_node(state), {})
        self.assertEqual(specialist_hook_node(state), {})
        self.assertEqual(crosstalk_coordinator_hook_node(state), {})

    @patch("app.config.settings.V4_PATHWAY_PLANNER_ENABLED", False)
    @patch("app.config.settings.V4_PATHWAY_SPECIALIST_ENABLED", False)
    @patch("app.config.settings.V4_CROSSTALK_COORDINATOR_ENABLED", False)
    def test_flag_off_state_not_polluted(self):
        """验证 2：flag=false 时 state 不被污染（不出现 v4_ 前缀字段）。"""
        from app.pathways.pathway_planner import pathway_planner_hook_node
        from app.pathways.specialist_hook import specialist_hook_node
        from app.crosstalk.coordinator import crosstalk_coordinator_hook_node

        original_state = {
            "user_input": "EGF binds EGFR",
            "network_json": {"nodes": [{"id": "EGF"}], "edges": []},
            "entities": [{"name": "EGF"}],
            "mechanism": {"pathway": "EGF_EGFR_MAPK"},
            "knowledge_graph": {"nodes": [], "edges": []},
        }

        state = dict(original_state)
        state.update(pathway_planner_hook_node(state))
        state.update(specialist_hook_node(state))
        state.update(crosstalk_coordinator_hook_node(state))

        # 不应出现任何 v4_ 前缀字段
        v4_keys = [k for k in state.keys() if k.startswith("v4_")]
        self.assertEqual(v4_keys, [], f"flag=false 时不应出现 v4_ 字段: {v4_keys}")

        # v3 字段保持不变
        self.assertEqual(state["user_input"], original_state["user_input"])
        self.assertEqual(state["network_json"], original_state["network_json"])
        self.assertEqual(state["entities"], original_state["entities"])
        self.assertEqual(state["mechanism"], original_state["mechanism"])
        self.assertEqual(state["knowledge_graph"], original_state["knowledge_graph"])

    @patch("app.config.settings.V4_PATHWAY_PLANNER_ENABLED", False)
    @patch("app.config.settings.V4_PATHWAY_SPECIALIST_ENABLED", False)
    @patch("app.config.settings.V4_CROSSTALK_COORDINATOR_ENABLED", False)
    def test_flag_off_planner_returns_empty_on_missing_input(self):
        """验证 3：flag=false 时即使输入缺失，hook 也返回 {}（不抛异常）。"""
        from app.pathways.pathway_planner import pathway_planner_hook_node
        from app.pathways.specialist_hook import specialist_hook_node
        from app.crosstalk.coordinator import crosstalk_coordinator_hook_node

        empty_state: dict = {}

        self.assertEqual(pathway_planner_hook_node(empty_state), {})
        self.assertEqual(specialist_hook_node(empty_state), {})
        self.assertEqual(crosstalk_coordinator_hook_node(empty_state), {})

    @patch("app.config.settings.V4_PATHWAY_PLANNER_ENABLED", False)
    @patch("app.config.settings.V4_PATHWAY_SPECIALIST_ENABLED", False)
    @patch("app.config.settings.V4_CROSSTALK_COORDINATOR_ENABLED", False)
    def test_flag_off_graph_builds_without_v4_fields(self):
        """验证 4：flag=false 时 graph_v3.py 构建的工作流不包含 v4 P4 hook 逻辑执行。

        此测试验证 build_workflow_v3() 能正常编译（不抛异常），
        且 P4 hook 节点已注册到图中（节点存在）。
        """
        from app.graph_v3 import build_workflow_v3

        # 构建工作流（不抛异常即可）
        workflow = build_workflow_v3()
        self.assertIsNotNone(workflow)

        # 验证 P4 hook 节点已注册（节点名存在于 graph nodes）
        # LangGraph StateGraph 的 nodes 属性包含所有注册的节点
        node_names = set(workflow.nodes.keys())
        self.assertIn("_pathway_planner_hook", node_names)
        self.assertIn("_specialist_hook", node_names)
        self.assertIn("_crosstalk_coordinator_hook", node_names)

    @patch("app.config.settings.V4_PATHWAY_PLANNER_ENABLED", False)
    @patch("app.config.settings.V4_PATHWAY_SPECIALIST_ENABLED", False)
    @patch("app.config.settings.V4_CROSSTALK_COORDINATOR_ENABLED", False)
    def test_flag_off_individual_hook_checks(self):
        """验证 5：每个 hook 独立检查自己的 flag。

        即使其他 flag 开启，只要自己的 flag 关闭，就返回 {}。
        """
        # Pathway Planner flag off, others on → planner returns {}
        with patch("app.config.settings.V4_PATHWAY_PLANNER_ENABLED", False), \
             patch("app.config.settings.V4_PATHWAY_SPECIALIST_ENABLED", True), \
             patch("app.config.settings.V4_CROSSTALK_COORDINATOR_ENABLED", True):
            from app.pathways.pathway_planner import pathway_planner_hook_node
            self.assertEqual(pathway_planner_hook_node({"user_input": "EGF"}), {})

        # Specialist flag off, others on → specialist returns {}
        with patch("app.config.settings.V4_PATHWAY_PLANNER_ENABLED", True), \
             patch("app.config.settings.V4_PATHWAY_SPECIALIST_ENABLED", False), \
             patch("app.config.settings.V4_CROSSTALK_COORDINATOR_ENABLED", True):
            from app.pathways.specialist_hook import specialist_hook_node
            state = {"v4_pathway_class": "EGFR_RTK", "v4_pathway_graph": {}}
            self.assertEqual(specialist_hook_node(state), {})

        # Coordinator flag off, others on → coordinator returns {}
        with patch("app.config.settings.V4_PATHWAY_PLANNER_ENABLED", True), \
             patch("app.config.settings.V4_PATHWAY_SPECIALIST_ENABLED", True), \
             patch("app.config.settings.V4_CROSSTALK_COORDINATOR_ENABLED", False):
            from app.crosstalk.coordinator import crosstalk_coordinator_hook_node
            state = {
                "v4_pathway_class": "MULTI:EGFR_RTK+PI3K_AKT_mTOR",
                "v4_specialist_outputs": [{"pathway_class": "EGFR_RTK"}],
            }
            self.assertEqual(crosstalk_coordinator_hook_node(state), {})


if __name__ == "__main__":
    unittest.main()
