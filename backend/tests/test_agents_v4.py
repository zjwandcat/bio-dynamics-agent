# BioDynamics Agent v4 - agents_v4 单元测试 (Phase 6 / Task 6.6.5)
#
# 测试 4 个 v4 Agent 模块：
#   - MechanismBuilderAgent（强制 MM/Hill/DDE）
#   - ODEBuilderAgent（从 Reaction IR 渲染 ODE）
#   - SimulationPlannerAgent（选仿真类型/求解器）
#   - ParameterAgent（pathway_tag 隔离 + threshold + provenance）
#
# 测试用例（35+ 个）：
#   - TestMechanismBuilderAgent: build 主入口 + MM 不可降级 + DDE 强制 + flag (10)
#   - TestODEBuilderAgent: build 主入口 + DDE 验证 + 模板选择 + flag (9)
#   - TestSimulationPlannerAgent: plan 主入口 + 仿真类型 + 求解器 + flag (9)
#   - TestParameterAgent: manage 主入口 + 隔离 + threshold + provenance + flag (10)
#   - TestAgentsV4ModuleInit: __init__ 导出 + DynamicRouter 兼容别名 (4)
#
# 运行：cd backend && python -m pytest tests/test_agents_v4.py -v --tb=short

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# 确保能导入 app 包（backend/ 在 sys.path）
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.agents_v4 import (
    MechanismBuilderAgent,
    ODEBuilderAgent,
    ParameterAgent,
    SimulationPlannerAgent,
)
from app.agents_v4.mechanism_builder import MechanismBuilder
from app.agents_v4.ode_builder import ODEBuilder
from app.agents_v4.simulation_planner import SimulationPlanner


# =============================================================================
# Mock 数据
# =============================================================================

# --- 单通路 pathway_graph（p53 振荡通路，含反馈延迟 + 磷酸化）---
MOCK_PATHWAY_GRAPH_P53 = {
    "pathway_class": "p53_signaling",
    "nodes": [
        {"id": "PN_p53", "canonical_name": "p53", "species_type": "protein"},
        {"id": "PN_Mdm2", "canonical_name": "Mdm2", "species_type": "protein"},
    ],
    "edges": [
        {
            "id": "PE_p53_phos",
            "source": "PN_p53",
            "target": "PN_p53",
            "mechanism": "phosphorylation",
            "pathway_tag": "p53_signaling",
            "kinetics_type": "mass_action",  # 故意标错，测试 MM 不可降级
            "is_feedback": False,
        },
        {
            "id": "PE_Mdm2_transcription",
            "source": "PN_p53",
            "target": "PN_Mdm2",
            "mechanism": "transcription",
            "pathway_tag": "p53_signaling",
            "kinetics_type": "mass_action",
            "is_feedback": True,
        },
        {
            "id": "PE_Mdm2_binding",
            "source": "PN_Mdm2",
            "target": "PN_p53",
            "mechanism": "binding",
            "pathway_tag": "p53_signaling",
            "kinetics_type": "mass_action",
            "is_feedback": True,
        },
    ],
    "feedback_loops": [
        {
            "id": "FL_p53_Mdm2_neg",
            "loop_type": "negative",
            "delay_minutes": 30.0,
            "edge_ids": ["PE_Mdm2_transcription", "PE_Mdm2_binding"],
        },
    ],
    "temporal": {
        "requires_dde": True,
        "dde_delay_minutes": 30.0,
        "t_end_minutes": 120.0,
    },
    "version": "v4.0",
}

# --- 单通路 pathway_graph（EGFR，非振荡）---
MOCK_PATHWAY_GRAPH_EGFR = {
    "pathway_class": "EGFR_RTK",
    "nodes": [
        {"id": "PN_EGFR", "canonical_name": "EGFR"},
        {"id": "PN_Ras", "canonical_name": "Ras"},
    ],
    "edges": [
        {
            "id": "PE_EGFR_bind",
            "source": "PN_EGFR",
            "target": "PN_EGFR",
            "mechanism": "binding",
            "pathway_tag": "EGFR_RTK",
            "kinetics_type": "mass_action",
        },
        {
            "id": "PE_Ras_phos",
            "source": "PN_EGFR",
            "target": "PN_Ras",
            "mechanism": "phosphorylation",
            "pathway_tag": "EGFR_RTK",
            "kinetics_type": "Michaelis_Menten",
        },
    ],
    "feedback_loops": [],
    "temporal": {
        "requires_dde": False,
        "dde_delay_minutes": 0.0,
        "t_end_minutes": 60.0,
    },
}

# --- reaction_ir mock ---
MOCK_REACTION_IR = {
    "species": [
        {"id": "SP_001", "name": "p53", "initial_concentration": 0.1,
         "parameters": {"k_deg": 0.01}},
        {"id": "SP_002", "name": "Mdm2", "initial_concentration": 0.0,
         "parameters": {"k_syn": 0.5}},
    ],
    "reactions": [
        {
            "id": "RXN_001",
            "source": "p53",
            "target": "Mdm2",
            "reaction_type": "transcription",
            "reaction_eq": "p53 -> Mdm2_mRNA",
            "parameters": {"k_tx": 0.3, "Km_p53": 0.2},
            "pathway_tag": "p53_signaling",
        },
        {
            "id": "RXN_002",
            "source": "Mdm2",
            "target": "p53",
            "reaction_type": "phosphorylation",
            "reaction_eq": "Mdm2 + p53 -> p53_Mdm2",
            "parameters": {"k_cat": 1.5, "Km": 0.4},
            "pathway_tag": "p53_signaling",
        },
    ],
    "version": "v4.0",
}

# --- v3 parameters mock（扁平格式）---
MOCK_PARAMETERS_FLAT = {
    "k1": 0.1,
    "k2": 0.01,
    "k_cat": 1.5,
    "Km": 0.4,
}

# --- v3 parameters mock（按 edge 分组格式）---
MOCK_PARAMETERS_GROUPED = {
    "RXN_001": {"k_tx": 0.3, "Km_p53": 0.2},
    "RXN_002": {"k_cat": 1.5, "Km": 0.4},
}

# --- v4_calibration_result mock ---
MOCK_CALIBRATION_RESULT = {
    "calibrated_params": {
        "k_cat": 1.8,
        "Km": 0.35,
    },
    "confidence_intervals": {},
    "uncalifiable": [],
    "method": "lmfit",
}

# --- v4_ode_system mock ---
MOCK_ODE_SYSTEM = {
    "ode_code": "def dy_dt(t, y): return [-k1*y[0] + k2*y[1]]",
    "equations": ["d[p53]/dt = -k1*p53 + k2*Mdm2"],
    "parameters": {"k1": 0.1, "k2": 0.01},
    "pathway_class": "p53_signaling",
    "dde_delay_minutes": 30.0,
    "template_name": "oscillatory_feedback.j2",
}


# =============================================================================
# TestMechanismBuilderAgent
# =============================================================================
class TestMechanismBuilderAgent(unittest.TestCase):
    """测试 MechanismBuilderAgent.build() 主入口。"""

    def setUp(self):
        self.agent = MechanismBuilderAgent()

    @patch("app.agents_v4.mechanism_builder.settings")
    def test_build_returns_v4_mechanism_assignments(self, mock_settings):
        """正常 state → 返回 v4_mechanism_assignments 列表。"""
        mock_settings.V4_DYNAMIC_ROUTING_ENABLED = True
        state = {
            "v4_pathway_graph": MOCK_PATHWAY_GRAPH_EGFR,
            "v4_pathway_class": "EGFR_RTK",
        }
        result = self.agent.build(state)
        self.assertIn("v4_mechanism_assignments", result)
        self.assertIsInstance(result["v4_mechanism_assignments"], list)
        self.assertGreater(len(result["v4_mechanism_assignments"]), 0)

    @patch("app.agents_v4.mechanism_builder.settings")
    def test_enzymatic_mechanism_uses_mm(self, mock_settings):
        """酶催化机制（phosphorylation）→ Michaelis_Menten（不是 mass_action）。"""
        mock_settings.V4_DYNAMIC_ROUTING_ENABLED = True
        state = {
            "v4_pathway_graph": MOCK_PATHWAY_GRAPH_EGFR,
            "v4_pathway_class": "EGFR_RTK",
        }
        result = self.agent.build(state)
        assignments = result["v4_mechanism_assignments"]
        phos = [a for a in assignments if a["mechanism"] == "phosphorylation"]
        self.assertGreater(len(phos), 0)
        for a in phos:
            self.assertEqual(a["kinetics_type"], "Michaelis_Menten")

    @patch("app.agents_v4.mechanism_builder.settings")
    def test_transcription_uses_hill(self, mock_settings):
        """转录机制（transcription）→ Hill 动力学。"""
        mock_settings.V4_DYNAMIC_ROUTING_ENABLED = True
        state = {
            "v4_pathway_graph": MOCK_PATHWAY_GRAPH_P53,
            "v4_pathway_class": "p53_signaling",
        }
        result = self.agent.build(state)
        assignments = result["v4_mechanism_assignments"]
        tx = [a for a in assignments if a["mechanism"] == "transcription"]
        self.assertGreater(len(tx), 0)
        for a in tx:
            self.assertEqual(a["kinetics_type"], "Hill")

    @patch("app.agents_v4.mechanism_builder.settings")
    def test_oscillatory_pathway_triggers_dde(self, mock_settings):
        """振荡通路（p53）+ 反馈延迟 → dde_required=True。"""
        mock_settings.V4_DYNAMIC_ROUTING_ENABLED = True
        state = {
            "v4_pathway_graph": MOCK_PATHWAY_GRAPH_P53,
            "v4_pathway_class": "p53_signaling",
        }
        result = self.agent.build(state)
        assignments = result["v4_mechanism_assignments"]
        dde_edges = [a for a in assignments if a["dde_required"]]
        self.assertGreater(len(dde_edges), 0, "振荡通路应至少有一条 dde_required=True 的边")

    @patch("app.agents_v4.mechanism_builder.settings")
    def test_pathway_tag_recorded_per_assignment(self, mock_settings):
        """每条 assignment 记录 pathway_tag（隔离用）。"""
        mock_settings.V4_DYNAMIC_ROUTING_ENABLED = True
        state = {
            "v4_pathway_graph": MOCK_PATHWAY_GRAPH_EGFR,
            "v4_pathway_class": "EGFR_RTK",
        }
        result = self.agent.build(state)
        for a in result["v4_mechanism_assignments"]:
            self.assertIn("pathway_tag", a)
            self.assertTrue(a["pathway_tag"], "pathway_tag 不应为空")

    @patch("app.agents_v4.mechanism_builder.settings")
    def test_mm_not_downgradable_validation(self, mock_settings):
        """MM 不可降级：酶催化机制被标 mass_action → 强制恢复 MM + 记录 warning。"""
        mock_settings.V4_DYNAMIC_ROUTING_ENABLED = True
        # MOCK_PATHWAY_GRAPH_P53 的 phosphorylation 边 kinetics_type="mass_action"（故意标错）
        state = {
            "v4_pathway_graph": MOCK_PATHWAY_GRAPH_P53,
            "v4_pathway_class": "p53_signaling",
        }
        result = self.agent.build(state)
        assignments = result["v4_mechanism_assignments"]
        phos = [a for a in assignments if a["mechanism"] == "phosphorylation"]
        self.assertGreater(len(phos), 0)
        # 强制恢复为 MM
        for a in phos:
            self.assertEqual(a["kinetics_type"], "Michaelis_Menten")
        # warnings 中应包含 MM 不可降级的记录
        mm_warnings = [w for w in result.get("warnings", []) if "MM 不可降级" in w]
        self.assertGreater(len(mm_warnings), 0, "应记录 MM 不可降级 warning")

    @patch("app.agents_v4.mechanism_builder.settings")
    def test_failure_returns_empty_with_warnings(self, mock_settings):
        """pathway_graph 为空 → 返回空 assignments + warnings。"""
        mock_settings.V4_DYNAMIC_ROUTING_ENABLED = True
        result = self.agent.build({"v4_pathway_graph": {}, "v4_pathway_class": ""})
        self.assertEqual(result["v4_mechanism_assignments"], [])
        self.assertGreater(len(result.get("warnings", [])), 0)

    @patch("app.agents_v4.mechanism_builder.settings")
    def test_flag_false_returns_empty(self, mock_settings):
        """V4_DYNAMIC_ROUTING_ENABLED=false → 返回 {}。"""
        mock_settings.V4_DYNAMIC_ROUTING_ENABLED = False
        state = {
            "v4_pathway_graph": MOCK_PATHWAY_GRAPH_P53,
            "v4_pathway_class": "p53_signaling",
        }
        result = self.agent.build(state)
        self.assertEqual(result, {})

    @patch("app.agents_v4.mechanism_builder.settings")
    def test_binding_uses_mass_action(self, mock_settings):
        """结合机制（binding）→ mass_action。"""
        mock_settings.V4_DYNAMIC_ROUTING_ENABLED = True
        state = {
            "v4_pathway_graph": MOCK_PATHWAY_GRAPH_EGFR,
            "v4_pathway_class": "EGFR_RTK",
        }
        result = self.agent.build(state)
        assignments = result["v4_mechanism_assignments"]
        binding = [a for a in assignments if a["mechanism"] == "binding"]
        self.assertGreater(len(binding), 0)
        for a in binding:
            self.assertEqual(a["kinetics_type"], "mass_action")

    @patch("app.agents_v4.mechanism_builder.settings")
    def test_generate_alias_delegates_to_build(self, mock_settings):
        """generate(state) 是 build(state) 的别名（DynamicRouter 兼容）。"""
        mock_settings.V4_DYNAMIC_ROUTING_ENABLED = True
        state = {
            "v4_pathway_graph": MOCK_PATHWAY_GRAPH_EGFR,
            "v4_pathway_class": "EGFR_RTK",
        }
        build_result = self.agent.build(state)
        generate_result = self.agent.generate(state)
        self.assertEqual(
            len(build_result["v4_mechanism_assignments"]),
            len(generate_result["v4_mechanism_assignments"]),
        )


# =============================================================================
# TestODEBuilderAgent
# =============================================================================
class TestODEBuilderAgent(unittest.TestCase):
    """测试 ODEBuilderAgent.build() 主入口。"""

    def setUp(self):
        self.agent = ODEBuilderAgent()

    @patch("app.agents_v4.ode_builder.settings")
    def test_build_returns_v4_ode_system(self, mock_settings):
        """正常 state → 返回 v4_ode_system。"""
        mock_settings.V4_DYNAMIC_ROUTING_ENABLED = True
        state = {
            "v4_reaction_ir": MOCK_REACTION_IR,
            "v4_pathway_class": "p53_signaling",
            "v4_pathway_graph": MOCK_PATHWAY_GRAPH_P53,
        }
        # mock 渲染器避免依赖实际模板
        with patch.object(self.agent, "_render_ode_code", return_value="# ODE code"):
            result = self.agent.build(state)
        self.assertIn("v4_ode_system", result)
        self.assertIsInstance(result["v4_ode_system"], dict)
        self.assertGreater(len(result["v4_ode_system"]), 0)

    @patch("app.agents_v4.ode_builder.settings")
    def test_ode_code_is_nonempty_string(self, mock_settings):
        """ode_code 为非空字符串。"""
        mock_settings.V4_DYNAMIC_ROUTING_ENABLED = True
        state = {
            "v4_reaction_ir": MOCK_REACTION_IR,
            "v4_pathway_class": "p53_signaling",
            "v4_pathway_graph": MOCK_PATHWAY_GRAPH_P53,
        }
        with patch.object(self.agent, "_render_ode_code", return_value="def dy_dt(): pass"):
            result = self.agent.build(state)
        self.assertIsInstance(result["v4_ode_system"]["ode_code"], str)
        self.assertGreater(len(result["v4_ode_system"]["ode_code"]), 0)

    @patch("app.agents_v4.ode_builder.settings")
    def test_dde_enforced_for_oscillatory_pathway(self, mock_settings):
        """振荡通路 → dde_delay_minutes > 0（DDE 强制）。"""
        mock_settings.V4_DYNAMIC_ROUTING_ENABLED = True
        state = {
            "v4_reaction_ir": MOCK_REACTION_IR,
            "v4_pathway_class": "p53_signaling",
            "v4_pathway_graph": MOCK_PATHWAY_GRAPH_P53,
        }
        with patch.object(self.agent, "_render_ode_code", return_value="# dde_delay = 30"):
            result = self.agent.build(state)
        dde_delay = result["v4_ode_system"]["dde_delay_minutes"]
        self.assertGreater(dde_delay, 0, "振荡通路 dde_delay_minutes 必须 > 0")

    @patch("app.agents_v4.ode_builder.settings")
    def test_dde_default_when_delay_missing(self, mock_settings):
        """振荡通路但 temporal 无 dde_delay → 强制设置默认延迟 + warning。"""
        mock_settings.V4_DYNAMIC_ROUTING_ENABLED = True
        graph_no_delay = {
            "pathway_class": "p53_signaling",
            "edges": [],
            "temporal": {"requires_dde": False, "dde_delay_minutes": 0.0},
        }
        state = {
            "v4_reaction_ir": MOCK_REACTION_IR,
            "v4_pathway_class": "p53_signaling",
            "v4_pathway_graph": graph_no_delay,
        }
        with patch.object(self.agent, "_render_ode_code", return_value=""):
            result = self.agent.build(state)
        self.assertGreater(result["v4_ode_system"]["dde_delay_minutes"], 0)
        dde_warnings = [w for w in result.get("warnings", []) if "DDE 强制" in w]
        self.assertGreater(len(dde_warnings), 0)

    @patch("app.agents_v4.ode_builder.settings")
    def test_template_selected_based_on_pathway_class(self, mock_settings):
        """模板名根据 pathway_class 选择（振荡 → oscillatory_feedback.j2）。"""
        mock_settings.V4_DYNAMIC_ROUTING_ENABLED = True
        state = {
            "v4_reaction_ir": MOCK_REACTION_IR,
            "v4_pathway_class": "p53_signaling",
            "v4_pathway_graph": MOCK_PATHWAY_GRAPH_P53,
        }
        with patch.object(self.agent, "_render_ode_code", return_value="# code"):
            result = self.agent.build(state)
        template = result["v4_ode_system"]["template_name"]
        self.assertEqual(template, "oscillatory_feedback.j2")

    @patch("app.agents_v4.ode_builder.settings")
    def test_pathway_tag_isolation_check(self, mock_settings):
        """pathway_tag 隔离：参数标注的 tag 与当前通路不一致 → 记录 warning。"""
        mock_settings.V4_DYNAMIC_ROUTING_ENABLED = True
        # 构造一个含外来 pathway_tag 的 reaction_ir（species 标注为其他通路）
        reaction_ir_foreign = {
            "species": [
                {"name": "p53", "pathway_tag": "EGFR_RTK", "parameters": {"k1": 0.1}},
            ],
            "reactions": [],
        }
        state = {
            "v4_reaction_ir": reaction_ir_foreign,
            "v4_pathway_class": "p53_signaling",
            "v4_pathway_graph": MOCK_PATHWAY_GRAPH_P53,
        }
        with patch.object(self.agent, "_render_ode_code", return_value="# code"):
            result = self.agent.build(state)
        isolation_warnings = [
            w for w in result.get("warnings", []) if "pathway_tag 隔离" in w
        ]
        self.assertGreater(len(isolation_warnings), 0)

    @patch("app.agents_v4.ode_builder.settings")
    def test_failure_returns_empty_with_warnings(self, mock_settings):
        """reaction_ir 为空 → 返回空 ode_system + warnings。"""
        mock_settings.V4_DYNAMIC_ROUTING_ENABLED = True
        result = self.agent.build({
            "v4_reaction_ir": {},
            "v4_pathway_class": "p53_signaling",
        })
        self.assertEqual(result["v4_ode_system"], {})
        self.assertGreater(len(result.get("warnings", [])), 0)

    @patch("app.agents_v4.ode_builder.settings")
    def test_flag_false_returns_empty(self, mock_settings):
        """V4_DYNAMIC_ROUTING_ENABLED=false → 返回 {}。"""
        mock_settings.V4_DYNAMIC_ROUTING_ENABLED = False
        state = {
            "v4_reaction_ir": MOCK_REACTION_IR,
            "v4_pathway_class": "p53_signaling",
        }
        result = self.agent.build(state)
        self.assertEqual(result, {})

    @patch("app.agents_v4.ode_builder.settings")
    def test_equations_extracted_from_reaction_ir(self, mock_settings):
        """从 reaction_ir 提取 equations 列表。"""
        mock_settings.V4_DYNAMIC_ROUTING_ENABLED = True
        state = {
            "v4_reaction_ir": MOCK_REACTION_IR,
            "v4_pathway_class": "p53_signaling",
            "v4_pathway_graph": MOCK_PATHWAY_GRAPH_P53,
        }
        with patch.object(self.agent, "_render_ode_code", return_value="# code"):
            result = self.agent.build(state)
        equations = result["v4_ode_system"]["equations"]
        self.assertIsInstance(equations, list)
        self.assertGreater(len(equations), 0)


# =============================================================================
# TestSimulationPlannerAgent
# =============================================================================
class TestSimulationPlannerAgent(unittest.TestCase):
    """测试 SimulationPlannerAgent.plan() 主入口。"""

    def setUp(self):
        self.agent = SimulationPlannerAgent()

    @patch("app.agents_v4.simulation_planner.settings")
    def test_plan_returns_v4_simulation_plan(self, mock_settings):
        """正常 state → 返回 v4_simulation_plan。"""
        mock_settings.V4_DYNAMIC_ROUTING_ENABLED = True
        state = {
            "v4_ode_system": MOCK_ODE_SYSTEM,
            "v4_pathway_class": "EGFR_RTK",
        }
        result = self.agent.plan(state)
        self.assertIn("v4_simulation_plan", result)
        self.assertIsInstance(result["v4_simulation_plan"], dict)

    @patch("app.agents_v4.simulation_planner.settings")
    def test_simulation_type_is_valid_enum(self, mock_settings):
        """simulation_type 为合法值（ode/dde/stochastic）。"""
        mock_settings.V4_DYNAMIC_ROUTING_ENABLED = True
        state = {
            "v4_ode_system": MOCK_ODE_SYSTEM,
            "v4_pathway_class": "EGFR_RTK",
        }
        result = self.agent.plan(state)
        sim_type = result["v4_simulation_plan"]["simulation_type"]
        self.assertIn(sim_type, {"ode", "dde", "stochastic"})

    @patch("app.agents_v4.simulation_planner.settings")
    def test_solver_matches_simulation_type(self, mock_settings):
        """solver 与 simulation_type 匹配。"""
        mock_settings.V4_DYNAMIC_ROUTING_ENABLED = True
        state = {
            "v4_ode_system": MOCK_ODE_SYSTEM,
            "v4_pathway_class": "EGFR_RTK",
        }
        result = self.agent.plan(state)
        sim_type = result["v4_simulation_plan"]["simulation_type"]
        solver = result["v4_simulation_plan"]["solver"]
        expected = {"ode": "scipy.solve_ivp", "dde": "dde_solver", "stochastic": "gillespie"}
        self.assertEqual(solver, expected[sim_type])

    @patch("app.agents_v4.simulation_planner.settings")
    def test_dde_pathway_uses_dde_solver(self, mock_settings):
        """振荡通路（p53）→ simulation_type="dde", solver="dde_solver"。"""
        mock_settings.V4_DYNAMIC_ROUTING_ENABLED = True
        state = {
            "v4_ode_system": MOCK_ODE_SYSTEM,
            "v4_pathway_class": "p53_signaling",
        }
        result = self.agent.plan(state)
        plan = result["v4_simulation_plan"]
        self.assertEqual(plan["simulation_type"], "dde")
        self.assertEqual(plan["solver"], "dde_solver")

    @patch("app.agents_v4.simulation_planner.settings")
    def test_multi_pathway_uses_multi_time_scales(self, mock_settings):
        """多通路场景 → time_scales 含多个尺度。"""
        mock_settings.V4_DYNAMIC_ROUTING_ENABLED = True
        state = {
            "v4_ode_system": MOCK_ODE_SYSTEM,
            "v4_pathway_class": "MULTI:EGFR_RTK+PI3K_AKT_mTOR",
        }
        result = self.agent.plan(state)
        plan = result["v4_simulation_plan"]
        self.assertTrue(plan["multi_pathway"])
        time_scales = plan["time_scales"]
        self.assertIsInstance(time_scales, list)
        self.assertGreater(len(time_scales), 1, "多通路应有多个时间尺度")

    @patch("app.agents_v4.simulation_planner.settings")
    def test_single_pathway_single_time_scale(self, mock_settings):
        """单通路 → time_scales 仅含一个尺度。"""
        mock_settings.V4_DYNAMIC_ROUTING_ENABLED = True
        state = {
            "v4_ode_system": MOCK_ODE_SYSTEM,
            "v4_pathway_class": "EGFR_RTK",
        }
        result = self.agent.plan(state)
        plan = result["v4_simulation_plan"]
        self.assertFalse(plan["multi_pathway"])
        self.assertEqual(len(plan["time_scales"]), 1)

    @patch("app.agents_v4.simulation_planner.settings")
    def test_default_plan_on_failure(self, mock_settings):
        """异常 → 返回默认 plan（simulation_type="ode"）。"""
        mock_settings.V4_DYNAMIC_ROUTING_ENABLED = True
        # 传入非 dict state 触发异常
        result = self.agent.plan("not_a_dict")  # type: ignore[arg-type]
        plan = result["v4_simulation_plan"]
        self.assertEqual(plan["simulation_type"], "ode")
        self.assertEqual(plan["solver"], "scipy.solve_ivp")

    @patch("app.agents_v4.simulation_planner.settings")
    def test_flag_false_returns_empty(self, mock_settings):
        """V4_DYNAMIC_ROUTING_ENABLED=false → 返回 {}。"""
        mock_settings.V4_DYNAMIC_ROUTING_ENABLED = False
        state = {
            "v4_ode_system": MOCK_ODE_SYSTEM,
            "v4_pathway_class": "EGFR_RTK",
        }
        result = self.agent.plan(state)
        self.assertEqual(result, {})

    @patch("app.agents_v4.simulation_planner.settings")
    def test_dde_pathway_t_end_extended(self, mock_settings):
        """DDE 通路 → t_end 至少为 dde_delay 的 4 倍。"""
        mock_settings.V4_DYNAMIC_ROUTING_ENABLED = True
        ode_system = {
            "dde_delay_minutes": 30.0,
            "pathway_class": "p53_signaling",
        }
        state = {
            "v4_ode_system": ode_system,
            "v4_pathway_class": "p53_signaling",
        }
        result = self.agent.plan(state)
        plan = result["v4_simulation_plan"]
        self.assertGreaterEqual(plan["t_end"], 30.0 * 4.0)


# =============================================================================
# TestParameterAgent
# =============================================================================
class TestParameterAgent(unittest.TestCase):
    """测试 ParameterAgent.manage() 主入口。"""

    def setUp(self):
        self.agent = ParameterAgent()

    @patch("app.agents_v4.parameter_agent.settings")
    def test_manage_returns_v4_parameter_registry(self, mock_settings):
        """正常 state → 返回 v4_parameter_registry。"""
        mock_settings.V4_DYNAMIC_ROUTING_ENABLED = True
        state = {
            "parameters": MOCK_PARAMETERS_FLAT,
            "v4_pathway_class": "p53_signaling",
        }
        result = self.agent.manage(state)
        self.assertIn("v4_parameter_registry", result)
        registry = result["v4_parameter_registry"]
        self.assertIn("params", registry)
        self.assertIn("isolation_valid", registry)
        self.assertIn("warnings", registry)

    @patch("app.agents_v4.parameter_agent.settings")
    def test_params_have_pathway_tag(self, mock_settings):
        """每个参数标注 pathway_tag。"""
        mock_settings.V4_DYNAMIC_ROUTING_ENABLED = True
        state = {
            "parameters": MOCK_PARAMETERS_FLAT,
            "v4_pathway_class": "p53_signaling",
        }
        result = self.agent.manage(state)
        for p in result["v4_parameter_registry"]["params"]:
            self.assertIn("pathway_tag", p)
            self.assertTrue(p["pathway_tag"], "pathway_tag 不应为空")

    @patch("app.agents_v4.parameter_agent.settings")
    def test_params_have_source_provenance(self, mock_settings):
        """每个参数标注 source（provenance）。"""
        mock_settings.V4_DYNAMIC_ROUTING_ENABLED = True
        state = {
            "parameters": MOCK_PARAMETERS_FLAT,
            "v4_pathway_class": "p53_signaling",
        }
        result = self.agent.manage(state)
        for p in result["v4_parameter_registry"]["params"]:
            self.assertIn("source", p)
            self.assertIsInstance(p["source"], str)

    @patch("app.agents_v4.parameter_agent.settings")
    def test_threshold_validation_k_cat_positive(self, mock_settings):
        """k_cat > 0 → threshold_valid=True。"""
        mock_settings.V4_DYNAMIC_ROUTING_ENABLED = True
        state = {
            "parameters": {"k_cat": 1.5, "Km": 0.4},
            "v4_pathway_class": "p53_signaling",
        }
        result = self.agent.manage(state)
        params = result["v4_parameter_registry"]["params"]
        for p in params:
            if p["name"].lower().startswith("k_cat"):
                self.assertTrue(p["threshold_valid"])

    @patch("app.agents_v4.parameter_agent.settings")
    def test_threshold_validation_km_positive(self, mock_settings):
        """Km > 0 → threshold_valid=True。"""
        mock_settings.V4_DYNAMIC_ROUTING_ENABLED = True
        state = {
            "parameters": {"Km": 0.4},
            "v4_pathway_class": "p53_signaling",
        }
        result = self.agent.manage(state)
        params = result["v4_parameter_registry"]["params"]
        for p in params:
            self.assertTrue(p["threshold_valid"], f"Km={p['value']} 应通过阈值验证")

    @patch("app.agents_v4.parameter_agent.settings")
    def test_threshold_validation_negative_value_fails(self, mock_settings):
        """k_cat < 0 → threshold_valid=False + warning。"""
        mock_settings.V4_DYNAMIC_ROUTING_ENABLED = True
        state = {
            "parameters": {"k_cat": -1.0},
            "v4_pathway_class": "p53_signaling",
        }
        result = self.agent.manage(state)
        params = result["v4_parameter_registry"]["params"]
        for p in params:
            if p["name"] == "k_cat":
                self.assertFalse(p["threshold_valid"])
        threshold_warnings = [
            w for w in result["v4_parameter_registry"]["warnings"]
            if "阈值验证失败" in w
        ]
        self.assertGreater(len(threshold_warnings), 0)

    @patch("app.agents_v4.parameter_agent.settings")
    def test_pathway_tag_isolation_no_leakage(self, mock_settings):
        """pathway_tag 隔离：参数标注的 tag 与当前通路一致 → isolation_valid=True。"""
        mock_settings.V4_DYNAMIC_ROUTING_ENABLED = True
        state = {
            "parameters": MOCK_PARAMETERS_FLAT,
            "v4_pathway_class": "p53_signaling",
        }
        result = self.agent.manage(state)
        registry = result["v4_parameter_registry"]
        # 扁平参数默认标注为 pathway_class，无跨通路泄漏
        self.assertTrue(registry["isolation_valid"])

    @patch("app.agents_v4.parameter_agent.settings")
    def test_pathway_tag_isolation_cross_pathway_warning(self, mock_settings):
        """跨通路参数 → isolation_valid=False + warning。"""
        mock_settings.V4_DYNAMIC_ROUTING_ENABLED = True
        # 构造含外来 pathway_tag 的分组参数
        parameters_grouped = {
            "RXN_001": {"k_tx": {"value": 0.3, "pathway_tag": "EGFR_RTK", "source": "RAG"}},
        }
        pathway_graph = {
            "edges": [{"id": "RXN_001", "pathway_tag": "EGFR_RTK"}],
        }
        state = {
            "parameters": parameters_grouped,
            "v4_pathway_class": "p53_signaling",
            "v4_pathway_graph": pathway_graph,
        }
        result = self.agent.manage(state)
        registry = result["v4_parameter_registry"]
        self.assertFalse(registry["isolation_valid"])
        isolation_warnings = [
            w for w in registry["warnings"] if "pathway_tag 隔离" in w
        ]
        self.assertGreater(len(isolation_warnings), 0)

    @patch("app.agents_v4.parameter_agent.settings")
    def test_isolation_valid_flag(self, mock_settings):
        """isolation_valid 标志正确反映隔离状态。"""
        mock_settings.V4_DYNAMIC_ROUTING_ENABLED = True
        # 正常情况：所有参数 tag 一致
        state_ok = {
            "parameters": {"k1": 0.1, "k2": 0.2},
            "v4_pathway_class": "EGFR_RTK",
        }
        result_ok = self.agent.manage(state_ok)
        self.assertTrue(result_ok["v4_parameter_registry"]["isolation_valid"])

    @patch("app.agents_v4.parameter_agent.settings")
    def test_failure_returns_degraded(self, mock_settings):
        """异常 → 返回降级 registry（isolation_valid=True，params=[]）。"""
        mock_settings.V4_DYNAMIC_ROUTING_ENABLED = True
        # 传入非 dict state 触发异常
        result = self.agent.manage("not_a_dict")  # type: ignore[arg-type]
        registry = result["v4_parameter_registry"]
        self.assertEqual(registry["params"], [])
        self.assertTrue(registry["isolation_valid"])
        self.assertGreater(len(registry["warnings"]), 0)

    @patch("app.agents_v4.parameter_agent.settings")
    def test_flag_false_returns_empty(self, mock_settings):
        """V4_DYNAMIC_ROUTING_ENABLED=false → 返回 {}。"""
        mock_settings.V4_DYNAMIC_ROUTING_ENABLED = False
        state = {
            "parameters": MOCK_PARAMETERS_FLAT,
            "v4_pathway_class": "p53_signaling",
        }
        result = self.agent.manage(state)
        self.assertEqual(result, {})

    @patch("app.agents_v4.parameter_agent.settings")
    def test_calibration_params_merged(self, mock_settings):
        """校准参数合并（优先级最高，覆盖同名参数）。"""
        mock_settings.V4_DYNAMIC_ROUTING_ENABLED = True
        state = {
            "parameters": {"k_cat": 1.5, "Km": 0.4, "k1": 0.1},
            "v4_pathway_class": "p53_signaling",
            "v4_calibration_result": MOCK_CALIBRATION_RESULT,
        }
        result = self.agent.manage(state)
        params = result["v4_parameter_registry"]["params"]
        # 校准参数应覆盖原始值
        k_cat = [p for p in params if p["name"] == "k_cat"]
        self.assertGreater(len(k_cat), 0)
        self.assertEqual(k_cat[0]["value"], 1.8)
        self.assertEqual(k_cat[0]["source"], "calibration")


# =============================================================================
# TestAgentsV4ModuleInit
# =============================================================================
class TestAgentsV4ModuleInit(unittest.TestCase):
    """测试 agents_v4 包导出与 DynamicRouter 兼容别名。"""

    def test_init_exports_four_agents(self):
        """__init__.py 导出 4 个 Agent 类。"""
        from app.agents_v4 import (
            MechanismBuilderAgent,
            ODEBuilderAgent,
            ParameterAgent,
            SimulationPlannerAgent,
        )
        self.assertTrue(callable(MechanismBuilderAgent))
        self.assertTrue(callable(ODEBuilderAgent))
        self.assertTrue(callable(SimulationPlannerAgent))
        self.assertTrue(callable(ParameterAgent))

    def test_agent_version_attribute(self):
        """每个 Agent 类有 AGENT_VERSION = "v4.0"。"""
        self.assertEqual(MechanismBuilderAgent.AGENT_VERSION, "v4.0")
        self.assertEqual(ODEBuilderAgent.AGENT_VERSION, "v4.0")
        self.assertEqual(SimulationPlannerAgent.AGENT_VERSION, "v4.0")
        self.assertEqual(ParameterAgent.AGENT_VERSION, "v4.0")

    def test_dynamic_router_compat_aliases(self):
        """DynamicRouter 兼容别名存在（MechanismBuilder/ODEBuilder/SimulationPlanner）。"""
        self.assertIs(MechanismBuilder, MechanismBuilderAgent)
        self.assertIs(ODEBuilder, ODEBuilderAgent)
        self.assertIs(SimulationPlanner, SimulationPlannerAgent)

    def test_generate_method_exists_on_all_agents(self):
        """每个 Agent 类有 generate 方法（DynamicRouter 调度入口）。"""
        for cls in (MechanismBuilderAgent, ODEBuilderAgent,
                    SimulationPlannerAgent, ParameterAgent):
            self.assertTrue(hasattr(cls, "generate"))
            self.assertTrue(callable(getattr(cls, "generate")))


if __name__ == "__main__":
    unittest.main()
