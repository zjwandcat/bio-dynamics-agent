# BioDynamics Agent v4 - Task B.2 State 字段合并单元测试
#
# 验证 17 个 v4_ 平铺字段合并为 v4_state dict 的行为：
#   1. set_v4_state 双写（v4_state + 平铺字段）
#   2. get_v4_state 读取（v4_state 优先，回退平铺字段）
#   3. get_v4 按平铺字段名读取（向后兼容）
#   4. normalize_v4_state 从平铺字段重建 v4_state
#   5. merge_v4_state reducer 按 group deep-merge
#   6. 全部 17 字段双路径访问
#   7. v3 行为不受影响（flag off 时 v4_state 缺失/空）
#
# 运行：cd backend && python -m pytest tests/test_v4_state_consolidation.py -v

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# 确保能导入 app 包（backend/ 在 sys.path）
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.state import (
    V4_FIELD_MAP,
    get_v4,
    get_v4_state,
    merge_v4_state,
    normalize_v4_state,
    set_v4_state,
)


class TestSetV4State(unittest.TestCase):
    """set_v4_state 双写测试。"""

    def test_dual_write_creates_both_flat_and_nested(self):
        """set_v4_state 应同时写入平铺字段和 v4_state 嵌套结构。"""
        target = {}
        set_v4_state(target, "ontology", "entities", {"id": "EGFR"})

        # 平铺字段存在（向后兼容）
        self.assertIn("v4_ontology_entities", target)
        self.assertEqual(target["v4_ontology_entities"], {"id": "EGFR"})

        # v4_state 嵌套结构存在
        self.assertIn("v4_state", target)
        self.assertEqual(
            target["v4_state"]["ontology"]["entities"], {"id": "EGFR"}
        )

    def test_dual_write_multiple_groups(self):
        """同一 target 多次调用 set_v4_state 应累积不同 group。"""
        target = {}
        set_v4_state(target, "ontology", "entities", {"a": 1})
        set_v4_state(target, "validation", "report", {"pass": True})
        set_v4_state(target, "router", "dispatches", [{"id": "x"}])

        self.assertEqual(target["v4_ontology_entities"], {"a": 1})
        self.assertEqual(target["v4_validation_report"], {"pass": True})
        self.assertEqual(target["v4_agent_dispatches"], [{"id": "x"}])
        self.assertEqual(target["v4_state"]["ontology"]["entities"], {"a": 1})
        self.assertEqual(target["v4_state"]["validation"]["report"], {"pass": True})
        self.assertEqual(target["v4_state"]["router"]["dispatches"], [{"id": "x"}])

    def test_dual_write_same_group_different_keys(self):
        """同一 group 写入不同 key 应共存。"""
        target = {}
        set_v4_state(target, "specialist", "outputs", [{"path": "EGFR"}])
        set_v4_state(target, "specialist", "crosstalk_edges", [{"src": "A"}])
        set_v4_state(target, "specialist", "shared_species", ["RasGTP"])

        self.assertEqual(len(target["v4_state"]["specialist"]), 3)
        self.assertEqual(target["v4_state"]["specialist"]["outputs"], [{"path": "EGFR"}])
        self.assertEqual(target["v4_state"]["specialist"]["crosstalk_edges"], [{"src": "A"}])
        self.assertEqual(target["v4_state"]["specialist"]["shared_species"], ["RasGTP"])

    def test_overwrite_same_key(self):
        """同一 (group, key) 二次写入应覆盖旧值。"""
        target = {}
        set_v4_state(target, "ontology", "entities", {"v": 1})
        set_v4_state(target, "ontology", "entities", {"v": 2})

        self.assertEqual(target["v4_ontology_entities"], {"v": 2})
        self.assertEqual(target["v4_state"]["ontology"]["entities"], {"v": 2})


class TestGetV4State(unittest.TestCase):
    """get_v4_state 读取测试。"""

    def test_read_from_v4_state(self):
        """v4_state 有值时优先返回。"""
        state = {
            "v4_state": {"ontology": {"entities": {"from": "nested"}}},
            "v4_ontology_entities": {"from": "flat"},
        }
        result = get_v4_state(state, "ontology", "entities")
        self.assertEqual(result, {"from": "nested"})

    def test_fallback_to_flat_field(self):
        """v4_state 缺失时回退到平铺字段。"""
        state = {"v4_ontology_entities": {"from": "flat"}}
        result = get_v4_state(state, "ontology", "entities")
        self.assertEqual(result, {"from": "flat"})

    def test_fallback_when_v4_state_group_missing(self):
        """v4_state 存在但 group 缺失时回退到平铺字段。"""
        state = {
            "v4_state": {"other_group": {}},
            "v4_ontology_entities": {"from": "flat"},
        }
        result = get_v4_state(state, "ontology", "entities")
        self.assertEqual(result, {"from": "flat"})

    def test_fallback_when_v4_state_key_missing(self):
        """v4_state group 存在但 key 缺失时回退到平铺字段。"""
        state = {
            "v4_state": {"ontology": {"other_key": 1}},
            "v4_ontology_entities": {"from": "flat"},
        }
        result = get_v4_state(state, "ontology", "entities")
        self.assertEqual(result, {"from": "flat"})

    def test_default_when_both_missing(self):
        """两处都缺失时返回 default。"""
        state = {}
        result = get_v4_state(state, "ontology", "entities", default="fallback")
        self.assertEqual(result, "fallback")

    def test_default_none(self):
        """两处都缺失且无 default 时返回 None。"""
        state = {}
        result = get_v4_state(state, "ontology", "entities")
        self.assertIsNone(result)


class TestGetV4(unittest.TestCase):
    """get_v4 按平铺字段名读取测试（向后兼容入口）。"""

    def test_read_from_v4_state_by_flat_name(self):
        """通过平铺字段名优先从 v4_state 读取。"""
        state = {
            "v4_state": {"pathway_class": {"class": "EGFR_RTK"}},
            "v4_pathway_class": "OLD_VALUE",
        }
        result = get_v4(state, "v4_pathway_class")
        self.assertEqual(result, "EGFR_RTK")

    def test_fallback_to_flat_by_name(self):
        """v4_state 缺失时通过平铺字段名回退。"""
        state = {"v4_pathway_class": "EGFR_RTK"}
        result = get_v4(state, "v4_pathway_class")
        self.assertEqual(result, "EGFR_RTK")

    def test_unknown_field_name(self):
        """未知字段名直接查 state（不查 v4_state）。"""
        state = {"custom_field": 42}
        result = get_v4(state, "custom_field", default=0)
        self.assertEqual(result, 42)

    def test_default_when_missing(self):
        """字段完全缺失时返回 default。"""
        state = {}
        result = get_v4(state, "v4_ontology_entities", default=[])
        self.assertEqual(result, [])


class TestNormalizeV4State(unittest.TestCase):
    """normalize_v4_state 重建测试。"""

    def test_rebuild_from_flat_fields(self):
        """从平铺字段重建 v4_state。"""
        state = {
            "v4_ontology_entities": {"e": 1},
            "v4_pathway_class": "EGFR_RTK",
            "v4_hypothesis_generated": True,
        }
        normalize_v4_state(state)
        self.assertEqual(state["v4_state"]["ontology"]["entities"], {"e": 1})
        self.assertEqual(state["v4_state"]["pathway_class"]["class"], "EGFR_RTK")
        self.assertEqual(state["v4_state"]["hypothesis"]["generated"], True)

    def test_idempotent(self):
        """多次调用 normalize 幂等，不破坏已有 v4_state 值。"""
        state = {
            "v4_ontology_entities": {"from_flat": True},
            "v4_state": {"ontology": {"entities": {"from_nested": True}}},
        }
        normalize_v4_state(state)
        # v4_state 优先：不应被平铺字段覆盖
        self.assertEqual(
            state["v4_state"]["ontology"]["entities"], {"from_nested": True}
        )
        # 再次调用仍幂等
        normalize_v4_state(state)
        self.assertEqual(
            state["v4_state"]["ontology"]["entities"], {"from_nested": True}
        )

    def test_no_flat_fields_no_change(self):
        """无平铺字段时 v4_state 保持空 dict。"""
        state = {"user_input": "test"}
        normalize_v4_state(state)
        self.assertEqual(state["v4_state"], {})

    def test_rebuild_all_17_fields(self):
        """重建包含全部 17 个字段。"""
        state = {flat: f"value_{flat}" for flat in V4_FIELD_MAP}
        normalize_v4_state(state)
        for flat_field, (group, key) in V4_FIELD_MAP.items():
            self.assertEqual(
                state["v4_state"][group][key],
                f"value_{flat_field}",
                f"Field {flat_field} not normalized correctly",
            )

    def test_returns_same_state_ref(self):
        """返回值是同一 state 引用（便于链式调用）。"""
        state = {"v4_ontology_entities": {}}
        result = normalize_v4_state(state)
        self.assertIs(result, state)


class TestMergeV4StateReducer(unittest.TestCase):
    """merge_v4_state reducer 测试（LangGraph 状态合并）。"""

    def test_merge_different_groups(self):
        """不同 group 合并后共存。"""
        existing = {"ontology": {"entities": 1}}
        new = {"validation": {"report": 2}}
        result = merge_v4_state(existing, new)
        self.assertEqual(result["ontology"]["entities"], 1)
        self.assertEqual(result["validation"]["report"], 2)

    def test_merge_same_group_different_keys(self):
        """同 group 不同 key 合并后共存。"""
        existing = {"specialist": {"outputs": [1]}}
        new = {"specialist": {"crosstalk_edges": [2]}}
        result = merge_v4_state(existing, new)
        self.assertEqual(result["specialist"]["outputs"], [1])
        self.assertEqual(result["specialist"]["crosstalk_edges"], [2])

    def test_merge_same_group_same_key_new_overwrites(self):
        """同 group 同 key 时 new 覆盖 existing。"""
        existing = {"ontology": {"entities": {"old": True}}}
        new = {"ontology": {"entities": {"new": True}}}
        result = merge_v4_state(existing, new)
        self.assertEqual(result["ontology"]["entities"], {"new": True})

    def test_merge_none_existing(self):
        """existing 为 None 时等价于 new 的浅拷贝。"""
        result = merge_v4_state(None, {"ontology": {"entities": 1}})
        self.assertEqual(result["ontology"]["entities"], 1)

    def test_merge_none_new(self):
        """new 为 None 时返回 existing 的浅拷贝。"""
        existing = {"ontology": {"entities": 1}}
        result = merge_v4_state(existing, None)
        self.assertEqual(result["ontology"]["entities"], 1)

    def test_merge_does_not_mutate_existing(self):
        """reducer 不应修改 existing 原始 dict。"""
        existing = {"ontology": {"entities": 1}}
        new = {"validation": {"report": 2}}
        merge_v4_state(existing, new)
        # existing 不应被修改
        self.assertNotIn("validation", existing)

    def test_merge_simulates_langgraph_multi_hook(self):
        """模拟 LangGraph 多 hook 串联合并 v4_state 的场景。"""
        # Hook A 返回 ontology group
        state_v4 = merge_v4_state(None, {"ontology": {"entities": {"a": 1}}})
        # Hook B 返回 specialist group
        state_v4 = merge_v4_state(state_v4, {"specialist": {"outputs": [{"p": "EGFR"}]}})
        # Hook C 返回 specialist 同 group 不同 key
        state_v4 = merge_v4_state(state_v4, {"specialist": {"crosstalk_edges": [{"src": "X"}]}})
        # Hook D 返回 validation group
        state_v4 = merge_v4_state(state_v4, {"validation": {"report": {"pass": True}}})

        self.assertEqual(state_v4["ontology"]["entities"], {"a": 1})
        self.assertEqual(state_v4["specialist"]["outputs"], [{"p": "EGFR"}])
        self.assertEqual(state_v4["specialist"]["crosstalk_edges"], [{"src": "X"}])
        self.assertEqual(state_v4["validation"]["report"], {"pass": True})


class TestAll17FieldsDualAccess(unittest.TestCase):
    """验证全部 17 个字段均可通过新旧两种路径访问。"""

    def test_field_map_has_17_entries(self):
        """V4_FIELD_MAP 应包含恰好 17 个字段。"""
        self.assertEqual(len(V4_FIELD_MAP), 17)

    def test_all_17_fields_dual_write_and_read(self):
        """每个字段通过 set_v4_state 写入后，平铺路径和 v4_state 路径均可读。"""
        for flat_field, (group, key) in V4_FIELD_MAP.items():
            with self.subTest(flat_field=flat_field, group=group, key=key):
                target = {}
                test_value = self._make_test_value(flat_field)
                set_v4_state(target, group, key, test_value)

                # 平铺路径（老代码）
                self.assertIn(
                    flat_field, target,
                    f"Flat field {flat_field} missing after set_v4_state",
                )
                self.assertEqual(target[flat_field], test_value)

                # v4_state 路径（新代码）
                self.assertIn(
                    group, target["v4_state"],
                    f"Group {group} missing in v4_state",
                )
                self.assertIn(
                    key, target["v4_state"][group],
                    f"Key {key} missing in v4_state[{group}]",
                )
                self.assertEqual(target["v4_state"][group][key], test_value)

    def test_all_17_fields_get_v4_state_fallback(self):
        """仅有平铺字段时，get_v4_state 能回退读取全部 17 个字段。"""
        for flat_field, (group, key) in V4_FIELD_MAP.items():
            with self.subTest(flat_field=flat_field):
                test_value = self._make_test_value(flat_field)
                state = {flat_field: test_value}
                result = get_v4_state(state, group, key)
                self.assertEqual(
                    result, test_value,
                    f"get_v4_state fallback failed for {flat_field}",
                )

    def test_all_17_fields_get_v4(self):
        """get_v4 能读取全部 17 个字段（v4_state 优先）。"""
        for flat_field, (group, key) in V4_FIELD_MAP.items():
            with self.subTest(flat_field=flat_field):
                test_value = self._make_test_value(flat_field)
                # 仅 v4_state 有值
                state = {"v4_state": {group: {key: test_value}}}
                result = get_v4(state, flat_field)
                self.assertEqual(
                    result, test_value,
                    f"get_v4 from v4_state failed for {flat_field}",
                )

    def _make_test_value(self, flat_field: str):
        """为不同字段生成类型合适的测试值。"""
        if flat_field in (
            "v4_pathway_class",
        ):
            return "EGFR_RTK"
        if flat_field == "v4_hypothesis_generated":
            return True
        if flat_field in (
            "v4_specialist_outputs",
            "v4_crosstalk_edges",
            "v4_hypothesis_list",
            "v4_agent_dispatches",
            "v4_shared_species",
        ):
            return [{"id": 1}, {"id": 2}] if "species" not in flat_field else ["RasGTP", "AKT"]
        return {"test_key": "test_value"}


class TestBackwardCompatFlatFieldRead(unittest.TestCase):
    """验证老代码直接读平铺字段仍然可用。"""

    def test_old_style_dict_get(self):
        """state.get('v4_xxx') 在 dual-write 后仍可用。"""
        target = {}
        set_v4_state(target, "ontology", "entities", {"e": 1})
        # 老代码风格：state.get("v4_ontology_entities")
        self.assertEqual(target.get("v4_ontology_entities"), {"e": 1})

    def test_old_style_subscript(self):
        """state['v4_xxx'] 在 dual-write 后仍可用。"""
        target = {}
        set_v4_state(target, "validation", "report", {"pass": True})
        self.assertEqual(target["v4_validation_report"], {"pass": True})

    def test_old_style_in_check(self):
        """'v4_xxx' in state 在 dual-write 后仍为 True。"""
        target = {}
        set_v4_state(target, "router", "dispatches", [])
        self.assertIn("v4_agent_dispatches", target)


class TestV3BehaviorUnchanged(unittest.TestCase):
    """验证 v3 行为不受 v4_state 影响（flag off 场景）。"""

    def test_v4_state_absent_when_no_v4_fields(self):
        """无 v4_ 字段的 state 不应被 normalize 创建 v4_state（空 dict 除外）。"""
        state = {"user_input": "simulate EGFR", "network_json": {"nodes": []}}
        normalize_v4_state(state)
        # v4_state 存在但为空 dict（无 v4_ 平铺字段）
        self.assertEqual(state.get("v4_state"), {})

    def test_v3_fields_not_affected_by_normalize(self):
        """normalize 不应触碰 v3 字段。"""
        state = {
            "user_input": "test",
            "network_json": {"nodes": [1, 2]},
            "entities": [{"id": "E1"}],
            "v4_ontology_entities": {"e": 1},
        }
        normalize_v4_state(state)
        self.assertEqual(state["user_input"], "test")
        self.assertEqual(state["network_json"], {"nodes": [1, 2]})
        self.assertEqual(state["entities"], [{"id": "E1"}])

    def test_empty_state_normalize_safe(self):
        """空 state 调用 normalize 不抛异常。"""
        state = {}
        normalize_v4_state(state)
        self.assertEqual(state.get("v4_state"), {})

    def test_merge_v4_state_none_none(self):
        """merge_v4_state(None, None) 返回空 dict（LangGraph 初始状态）。"""
        result = merge_v4_state(None, None)
        self.assertEqual(result, {})


class TestFieldMapIntegrity(unittest.TestCase):
    """V4_FIELD_MAP 完整性测试。"""

    def test_all_groups_present(self):
        """V4_FIELD_MAP 应包含 9 个 group。"""
        groups = {group for group, _ in V4_FIELD_MAP.values()}
        expected = {
            "ontology",
            "reaction_ir",
            "pathway_graph",
            "pathway_class",
            "specialist",
            "grounding",
            "validation",
            "hypothesis",
            "router",
        }
        self.assertEqual(groups, expected)

    def test_reverse_mapping_consistency(self):
        """(group, key) → flat_field 反向映射应与正向映射一致。"""
        from app.state import _V4_REVERSE_MAP

        self.assertEqual(len(_V4_REVERSE_MAP), 17)
        for flat_field, (group, key) in V4_FIELD_MAP.items():
            self.assertEqual(_V4_REVERSE_MAP[(group, key)], flat_field)

    def test_known_field_mappings(self):
        """关键字段映射符合 Task B.2 设计。"""
        self.assertEqual(V4_FIELD_MAP["v4_ontology_entities"], ("ontology", "entities"))
        self.assertEqual(V4_FIELD_MAP["v4_reaction_ir"], ("reaction_ir", "ir"))
        self.assertEqual(V4_FIELD_MAP["v4_pathway_graph"], ("pathway_graph", "graph"))
        self.assertEqual(V4_FIELD_MAP["v4_ode_system"], ("pathway_graph", "ode_system"))
        self.assertEqual(V4_FIELD_MAP["v4_pathway_class"], ("pathway_class", "class"))
        self.assertEqual(V4_FIELD_MAP["v4_specialist_outputs"], ("specialist", "outputs"))
        self.assertEqual(V4_FIELD_MAP["v4_crosstalk_edges"], ("specialist", "crosstalk_edges"))
        self.assertEqual(V4_FIELD_MAP["v4_shared_species"], ("specialist", "shared_species"))
        self.assertEqual(V4_FIELD_MAP["v4_shared_species_sync"], ("specialist", "shared_species_sync"))
        self.assertEqual(V4_FIELD_MAP["v4_time_scale_alignment"], ("specialist", "time_scale_alignment"))
        self.assertEqual(V4_FIELD_MAP["v4_grounding_ledger"], ("grounding", "ledger"))
        self.assertEqual(V4_FIELD_MAP["v4_validation_report"], ("validation", "report"))
        self.assertEqual(V4_FIELD_MAP["v4_calibration_result"], ("validation", "calibration_result"))
        self.assertEqual(V4_FIELD_MAP["v4_sensitivity_report"], ("validation", "sensitivity_report"))
        self.assertEqual(V4_FIELD_MAP["v4_hypothesis_list"], ("hypothesis", "list"))
        self.assertEqual(V4_FIELD_MAP["v4_hypothesis_generated"], ("hypothesis", "generated"))
        self.assertEqual(V4_FIELD_MAP["v4_agent_dispatches"], ("router", "dispatches"))


if __name__ == "__main__":
    unittest.main()
