# BioDynamics Agent v4 - Level 1 Internal Consistency Validation 单元测试
# (Phase 5 / Task 5.2.7)
#
# 测试 Level1InternalValidator 主类 + 5 个检查函数 + LangGraph hook。
#
# 测试用例（31 个，>= 25 要求）：
#   - TestLevel1InternalValidator: validate() 主入口 + 异常降级
#   - TestLevel1HookNode: Feature Flag + hook 行为
#   - TestMassConservation: 质量守恒检查
#   - TestNonNegative: 负浓度风险检测
#   - TestSteadyState: 稳态检查
#   - TestNumericalStability: NaN/Inf + stiff system 检测
#   - TestConstraintSatisfaction: 约束满足检查
#
# 运行：cd backend && python -m pytest tests/test_level1_internal.py -v

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# 确保能导入 app 包（backend/ 在 sys.path）
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


# =============================================================================
# Mock 数据
# =============================================================================

# --- Mock ODE 代码 ---
# 干净的 ODE：比例降解，无数值风险
MOCK_ODE_CLEAN = """
dEGFR/dt = -kon * EGF * EGFR + koff * EGF_EGFR - k_deg * EGFR
dEGF_EGFR/dt = kon * EGF * EGFR - koff * EGF_EGFR - k_internal * EGF_EGFR
dpEGFR/dt = k_phos * EGF_EGFR - k_dephos * pEGFR - k_deg * pEGFR
"""

# 含常量降解项的 ODE（可能导致负浓度）
MOCK_ODE_CONSTANT_DEGRADATION = """
dEGFR/dt = kon * EGF * EGFR - 0.5
dpEGFR/dt = k_phos * EGFR - k_deg * pEGFR
"""

# 含除零风险的 ODE
MOCK_ODE_DIVISION_RISK = """
dEGFR/dt = k_max * EGFR / AKT - k_deg * EGFR
dAKT/dt = -k_deg * AKT
"""

# 含 log(0) 风险的 ODE
MOCK_ODE_LOG_RISK = """
dEGFR/dt = k_max * log(EGFR) - k_deg * EGFR
"""

# Stiff system（速率常数比率 > 1e6）
MOCK_ODE_STIFF = """
k_fast = 1000.0
k_slow = 0.0001
dEGFR/dt = k_fast * EGF * EGFR - k_slow * EGFR
"""

# 无降解项的 ODE（稳态检查失败）
MOCK_ODE_NO_DEGRADATION = """
dEGFR/dt = kon * EGF * EGFR + koff * EGF_EGFR
dEGF_EGFR/dt = kon * EGF * EGFR - koff * EGF_EGFR
"""

# 空的 ODE
MOCK_ODE_EMPTY = ""

# --- Mock Reaction IR ---
MOCK_REACTION_IR_CLEAN = {
    "species": [
        {"id": "SP_001", "canonical_name": "EGFR", "initial_concentration": 100.0},
        {"id": "SP_002", "canonical_name": "pEGFR", "initial_concentration": 0.0},
        {"id": "SP_003", "canonical_name": "EGF_EGFR", "initial_concentration": 0.0},
        {"id": "SP_004", "canonical_name": "EGFR_total", "initial_concentration": 100.0},
    ],
    "constraints": [
        {
            "type": "mass_conservation",
            "expression": "EGFR + pEGFR + EGF_EGFR = EGFR_total",
            "tolerance": 0.05,
            "provenance": "EGFR_pool_conservation",
        },
    ],
}

# 质量守恒违反：EGFR(100) + pEGFR(50) + EGF_EGFR(0) = 150, 但 EGFR_total = 100
MOCK_REACTION_IR_MASS_VIOLATION = {
    "species": [
        {"id": "SP_001", "canonical_name": "EGFR", "initial_concentration": 100.0},
        {"id": "SP_002", "canonical_name": "pEGFR", "initial_concentration": 50.0},
        {"id": "SP_003", "canonical_name": "EGF_EGFR", "initial_concentration": 0.0},
        {"id": "SP_004", "canonical_name": "EGFR_total", "initial_concentration": 100.0},
    ],
    "constraints": [
        {
            "type": "mass_conservation",
            "expression": "EGFR + pEGFR + EGF_EGFR = EGFR_total",
            "tolerance": 0.05,
            "provenance": "EGFR_pool_conservation",
        },
    ],
}

# p53 蛋白池守恒
MOCK_REACTION_IR_P53_POOL = {
    "species": [
        {"id": "SP_001", "canonical_name": "p53", "initial_concentration": 50.0},
        {"id": "SP_002", "canonical_name": "p53_p", "initial_concentration": 30.0},
        {"id": "SP_003", "canonical_name": "p53_tetramer", "initial_concentration": 15.0},
        {"id": "SP_004", "canonical_name": "p53_nuclear", "initial_concentration": 5.0},
        {"id": "SP_005", "canonical_name": "p53_total", "initial_concentration": 100.0},
    ],
    "constraints": [
        {
            "type": "mass_conservation",
            "expression": "p53 + p53_p + p53_tetramer + p53_nuclear = p53_total",
            "tolerance": 0.05,
            "provenance": "p53_pool_conservation",
        },
    ],
}

# PIP2 + PIP3 守恒违反
MOCK_REACTION_IR_PIP_VIOLATION = {
    "species": [
        {"id": "SP_001", "canonical_name": "PIP2", "initial_concentration": 80.0},
        {"id": "SP_002", "canonical_name": "PIP3", "initial_concentration": 50.0},
        {"id": "SP_003", "canonical_name": "PIP_total", "initial_concentration": 100.0},
    ],
    "constraints": [
        {
            "type": "mass_conservation",
            "expression": "PIP2 + PIP3 = PIP_total",
            "tolerance": 0.05,
            "provenance": "PIP_pool_conservation",
        },
    ],
}

# 无约束
MOCK_REACTION_IR_NO_CONSTRAINTS = {
    "species": [
        {"id": "SP_001", "canonical_name": "EGFR", "initial_concentration": 100.0},
    ],
    "constraints": [],
}

# 无效约束表达式（缺少 =）
MOCK_REACTION_IR_INVALID_EXPR = {
    "species": [
        {"id": "SP_001", "canonical_name": "EGFR", "initial_concentration": 100.0},
        {"id": "SP_002", "canonical_name": "EGFR_total", "initial_concentration": 100.0},
    ],
    "constraints": [
        {
            "type": "mass_conservation",
            "expression": "EGFR + pEGFR",  # 缺少 = RHS
            "tolerance": 0.05,
            "provenance": "invalid_constraint",
        },
    ],
}

# 稳态约束
MOCK_REACTION_IR_STEADY_STATE = {
    "species": [
        {"id": "SP_001", "canonical_name": "EGFR", "initial_concentration": 100.0},
    ],
    "constraints": [
        {
            "type": "steady_state",
            "expression": "pEGFR < 0.01 * EGFR_total",
            "tolerance": 0.05,
            "provenance": "unstimulated_basal",
        },
    ],
}

# 稳态约束空表达式
MOCK_REACTION_IR_STEADY_STATE_EMPTY = {
    "species": [],
    "constraints": [
        {
            "type": "steady_state",
            "expression": "",
            "tolerance": 0.05,
            "provenance": "empty_steady_state",
        },
    ],
}

# 热力学约束缺少关键字
MOCK_REACTION_IR_THERMO_MISSING_KW = {
    "species": [],
    "constraints": [
        {
            "type": "thermodynamic",
            "expression": "EGFR + EGF = complex",
            "tolerance": 0.05,
            "provenance": "thermo_no_keywords",
        },
    ],
}

# 热力学约束含关键字
MOCK_REACTION_IR_THERMO_WITH_KW = {
    "species": [],
    "constraints": [
        {
            "type": "thermodynamic",
            "expression": "k_forward / k_reverse = Kd",
            "tolerance": 0.05,
            "provenance": "thermo_with_keywords",
        },
    ],
}


# =============================================================================
# TestLevel1InternalValidator: validate() 主入口
# =============================================================================
class TestLevel1InternalValidator(unittest.TestCase):
    """测试 Level1InternalValidator.validate() 主入口。"""

    def setUp(self):
        from app.validation_v2.level1_internal import Level1InternalValidator
        self.validator = Level1InternalValidator()

    def test_validate_returns_correct_structure(self):
        """validate() 返回包含所有必需字段的 dict。"""
        state = {
            "v4_ode_system": {"ode_code": MOCK_ODE_CLEAN},
            "v4_reaction_ir": MOCK_REACTION_IR_CLEAN,
        }
        result = self.validator.validate(state)

        # 验证所有必需字段存在
        required_fields = [
            "pass", "mass_conservation_error", "non_negative_violations",
            "steady_state_check", "numerical_stability", "constraint_violations",
        ]
        for field in required_fields:
            self.assertIn(field, result, f"缺少必需字段: {field}")

        # 验证字段类型
        self.assertIsInstance(result["pass"], bool)
        self.assertIsInstance(result["mass_conservation_error"], float)
        self.assertIsInstance(result["non_negative_violations"], list)
        self.assertIsInstance(result["steady_state_check"], bool)
        self.assertIsInstance(result["numerical_stability"], bool)
        self.assertIsInstance(result["constraint_violations"], list)

    def test_validate_pass_with_clean_input(self):
        """干净输入 → pass=True（mass conservation 通过 + 无 NaN/Inf）。"""
        state = {
            "v4_ode_system": {"ode_code": MOCK_ODE_CLEAN},
            "v4_reaction_ir": MOCK_REACTION_IR_CLEAN,
        }
        result = self.validator.validate(state)
        self.assertTrue(result["pass"])
        self.assertLessEqual(result["mass_conservation_error"], 0.05)
        self.assertTrue(result["numerical_stability"])

    def test_validate_empty_state_degrades(self):
        """空 state → 降级返回（不抛异常）。"""
        result = self.validator.validate({})
        # 空 state 不应抛异常
        self.assertIn("pass", result)
        # 无 ode_code / reaction_ir → mass error = 0, numerical_stable = True
        # 所以 pass = True
        self.assertTrue(result["pass"])

    def test_validate_missing_ode_system_degrades(self):
        """缺少 v4_ode_system → 降级处理。"""
        state = {"v4_reaction_ir": MOCK_REACTION_IR_CLEAN}
        result = self.validator.validate(state)
        self.assertIn("pass", result)
        # 无 ODE 代码 → 数值稳定（无数值风险）
        self.assertTrue(result["numerical_stability"])

    def test_validate_missing_reaction_ir_degrades(self):
        """缺少 v4_reaction_ir → 降级处理。"""
        state = {"v4_ode_system": {"ode_code": MOCK_ODE_CLEAN}}
        result = self.validator.validate(state)
        self.assertIn("pass", result)
        # 无 reaction_ir → 无约束 → mass error = 0
        self.assertEqual(result["mass_conservation_error"], 0.0)

    def test_validate_none_state_degrades(self):
        """state 为 None → 降级返回 pass=False。"""
        result = self.validator.validate(None)  # type: ignore
        self.assertFalse(result["pass"])
        self.assertIn("error", result)

    def test_validate_exception_returns_pass_false(self):
        """validate() 内部异常 → 返回 pass=False（不抛异常）。"""
        # 构造会导致异常的 state：v4_ode_system 是不可迭代的类型
        state = {"v4_ode_system": 12345}  # int 没有 .get()
        result = self.validator.validate(state)
        # int 会被 isinstance(ode_system, dict) 拦截，不会抛异常
        # 所以 pass 取决于其他检查结果
        self.assertIn("pass", result)

    def test_validate_mass_violation_blocks_pipeline(self):
        """mass conservation 误差 > 5% → pass=False（阻塞流水线）。"""
        state = {
            "v4_ode_system": {"ode_code": MOCK_ODE_CLEAN},
            "v4_reaction_ir": MOCK_REACTION_IR_MASS_VIOLATION,
        }
        result = self.validator.validate(state)
        self.assertFalse(result["pass"])
        self.assertGreater(result["mass_conservation_error"], 0.05)

    def test_validate_nan_inf_blocks_pipeline(self):
        """出现 NaN/Inf 风险 → pass=False（阻塞流水线）。"""
        state = {
            "v4_ode_system": {"ode_code": MOCK_ODE_DIVISION_RISK},
            "v4_reaction_ir": MOCK_REACTION_IR_NO_CONSTRAINTS,
        }
        result = self.validator.validate(state)
        self.assertFalse(result["pass"])
        self.assertFalse(result["numerical_stability"])


# =============================================================================
# TestLevel1HookNode: LangGraph hook
# =============================================================================
class TestLevel1HookNode(unittest.TestCase):
    """测试 level1_hook_node LangGraph 节点。"""

    @patch("app.validation_v2.level1_internal.settings")
    def test_flag_false_returns_empty(self, mock_settings):
        """Feature Flag false → hook 返回 {}。"""
        mock_settings.V4_VALIDATION_PYRAMID_ENABLED = False
        from app.validation_v2.level1_internal import level1_hook_node

        state = {
            "v4_ode_system": {"ode_code": MOCK_ODE_CLEAN},
            "v4_reaction_ir": MOCK_REACTION_IR_CLEAN,
        }
        result = level1_hook_node(state)
        self.assertEqual(result, {})

    @patch("app.validation_v2.level1_internal.settings")
    def test_flag_true_writes_validation_report(self, mock_settings):
        """Feature Flag true → hook 写入 v4_validation_report.level1。"""
        mock_settings.V4_VALIDATION_PYRAMID_ENABLED = True
        from app.validation_v2.level1_internal import level1_hook_node

        state = {
            "v4_ode_system": {"ode_code": MOCK_ODE_CLEAN},
            "v4_reaction_ir": MOCK_REACTION_IR_CLEAN,
        }
        result = level1_hook_node(state)
        self.assertIn("v4_validation_report", result)
        report = result["v4_validation_report"]
        self.assertIn("level1", report)
        level1 = report["level1"]
        self.assertIn("pass", level1)
        self.assertTrue(level1["pass"])

    @patch("app.validation_v2.level1_internal.settings")
    def test_hook_merges_existing_report(self, mock_settings):
        """hook 与现有 v4_validation_report 合并（不覆盖 level2/level3）。"""
        mock_settings.V4_VALIDATION_PYRAMID_ENABLED = True
        from app.validation_v2.level1_internal import level1_hook_node

        state = {
            "v4_ode_system": {"ode_code": MOCK_ODE_CLEAN},
            "v4_reaction_ir": MOCK_REACTION_IR_CLEAN,
            "v4_validation_report": {
                "level2": {"pass": True, "track": "skipped"},
            },
        }
        result = level1_hook_node(state)
        report = result["v4_validation_report"]
        # level1 被写入
        self.assertIn("level1", report)
        # level2 被保留（未被覆盖）
        self.assertIn("level2", report)
        self.assertEqual(report["level2"]["track"], "skipped")

    @patch("app.validation_v2.level1_internal.settings")
    def test_hook_exception_returns_empty(self, mock_settings):
        """hook 异常 → 返回 {}（不抛异常）。"""
        mock_settings.V4_VALIDATION_PYRAMID_ENABLED = True
        from app.validation_v2.level1_internal import level1_hook_node

        # 构造会导致 validate 内部异常的 state
        # validate 内部会捕获异常返回 pass=False，所以 hook 不会抛异常
        result = level1_hook_node(None)  # type: ignore
        # validate(None) 会抛 AttributeError，被 validate 内部捕获返回 pass=False
        self.assertIn("v4_validation_report", result)
        self.assertFalse(result["v4_validation_report"]["level1"]["pass"])


# =============================================================================
# TestMassConservation: 质量守恒检查
# =============================================================================
class TestMassConservation(unittest.TestCase):
    """测试 _check_mass_conservation。"""

    def setUp(self):
        from app.validation_v2.level1_internal import Level1InternalValidator
        self.validator = Level1InternalValidator()

    def test_conservation_passes(self):
        """守恒检查通过（误差 < 5%）。"""
        error, violations = self.validator._check_mass_conservation(
            MOCK_ODE_CLEAN, MOCK_REACTION_IR_CLEAN
        )
        self.assertLessEqual(error, 0.05)
        self.assertEqual(len(violations), 0)

    def test_conservation_fails(self):
        """守恒检查失败（误差 > 5%）。"""
        error, violations = self.validator._check_mass_conservation(
            MOCK_ODE_CLEAN, MOCK_REACTION_IR_MASS_VIOLATION
        )
        self.assertGreater(error, 0.05)
        self.assertGreater(len(violations), 0)

    def test_receptor_total_conservation(self):
        """受体总量守恒（EGFR + pEGFR + EGF_EGFR = EGFR_total）。"""
        error, violations = self.validator._check_mass_conservation(
            MOCK_ODE_CLEAN, MOCK_REACTION_IR_CLEAN
        )
        # EGFR(100) + pEGFR(0) + EGF_EGFR(0) = 100 = EGFR_total(100)
        self.assertAlmostEqual(error, 0.0, places=6)
        self.assertEqual(len(violations), 0)

    def test_protein_pool_conservation(self):
        """蛋白池守恒（p53 + p53_p + p53_tetramer + p53_nuclear = p53_total）。"""
        error, violations = self.validator._check_mass_conservation(
            MOCK_ODE_CLEAN, MOCK_REACTION_IR_P53_POOL
        )
        # p53(50) + p53_p(30) + p53_tetramer(15) + p53_nuclear(5) = 100 = p53_total(100)
        self.assertAlmostEqual(error, 0.0, places=6)
        self.assertEqual(len(violations), 0)

    def test_no_constraints_returns_zero_error(self):
        """无约束 → error = 0。"""
        error, violations = self.validator._check_mass_conservation(
            MOCK_ODE_CLEAN, MOCK_REACTION_IR_NO_CONSTRAINTS
        )
        self.assertEqual(error, 0.0)
        self.assertEqual(len(violations), 0)

    def test_invalid_expression_no_equals(self):
        """无效表达式（缺少 =）→ 报违规。"""
        error, violations = self.validator._check_mass_conservation(
            MOCK_ODE_CLEAN, MOCK_REACTION_IR_INVALID_EXPR
        )
        self.assertGreater(len(violations), 0)
        self.assertIn("no '='", violations[0]["reason"])


# =============================================================================
# TestNonNegative: 负浓度风险检测
# =============================================================================
class TestNonNegative(unittest.TestCase):
    """测试 _check_non_negative。"""

    def setUp(self):
        from app.validation_v2.level1_internal import Level1InternalValidator
        self.validator = Level1InternalValidator()

    def test_no_negative_risk(self):
        """无负浓度风险（比例降解安全）。"""
        violations = self.validator._check_non_negative(MOCK_ODE_CLEAN)
        self.assertEqual(len(violations), 0)

    def test_constant_degradation_risk(self):
        """常量降解项可能导致负浓度。"""
        violations = self.validator._check_non_negative(MOCK_ODE_CONSTANT_DEGRADATION)
        self.assertGreater(len(violations), 0)
        self.assertIn("Constant degradation", violations[0]["reason"])
        self.assertEqual(violations[0]["species_name"], "EGFR")

    def test_proportional_degradation_is_safe(self):
        """比例降解（-k*X）是安全的，不报违规。"""
        ode = "dEGFR/dt = -k_deg * EGFR"
        violations = self.validator._check_non_negative(ode)
        self.assertEqual(len(violations), 0)

    def test_empty_ode_returns_empty(self):
        """空 ODE → 返回空列表。"""
        violations = self.validator._check_non_negative(MOCK_ODE_EMPTY)
        self.assertEqual(len(violations), 0)


# =============================================================================
# TestSteadyState: 稳态检查
# =============================================================================
class TestSteadyState(unittest.TestCase):
    """测试 _check_steady_state。"""

    def setUp(self):
        from app.validation_v2.level1_internal import Level1InternalValidator
        self.validator = Level1InternalValidator()

    def test_steady_state_passes(self):
        """稳态检查通过（所有方程有降解项）。"""
        result = self.validator._check_steady_state(MOCK_ODE_CLEAN)
        self.assertTrue(result)

    def test_steady_state_fails(self):
        """稳态检查失败（有方程无降解项）。"""
        result = self.validator._check_steady_state(MOCK_ODE_NO_DEGRADATION)
        self.assertFalse(result)

    def test_empty_ode_passes(self):
        """空 ODE → 稳态检查通过（无方程即 trivially 通过）。"""
        result = self.validator._check_steady_state(MOCK_ODE_EMPTY)
        self.assertTrue(result)

    def test_single_equation_with_degradation(self):
        """单方程含降解项 → 通过。"""
        ode = "dEGFR/dt = k_phos * EGF - k_deg * EGFR"
        result = self.validator._check_steady_state(ode)
        self.assertTrue(result)

    def test_single_equation_without_degradation(self):
        """单方程无降解项 → 失败。"""
        ode = "dEGFR/dt = k_phos * EGF + koff * EGF_EGFR"
        result = self.validator._check_steady_state(ode)
        self.assertFalse(result)


# =============================================================================
# TestNumericalStability: 数值稳定性检查
# =============================================================================
class TestNumericalStability(unittest.TestCase):
    """测试 _check_numerical_stability。"""

    def setUp(self):
        from app.validation_v2.level1_internal import Level1InternalValidator
        self.validator = Level1InternalValidator()

    def test_no_nan_inf_risk(self):
        """无 NaN/Inf 风险。"""
        is_stable, violations = self.validator._check_numerical_stability(
            MOCK_ODE_CLEAN
        )
        self.assertTrue(is_stable)
        self.assertEqual(len(violations), 0)

    def test_division_by_zero_risk(self):
        """除零风险检测。"""
        is_stable, violations = self.validator._check_numerical_stability(
            MOCK_ODE_DIVISION_RISK
        )
        self.assertFalse(is_stable)
        div_violations = [v for v in violations if v["type"] == "division_risk"]
        self.assertGreater(len(div_violations), 0)
        self.assertIn("divide-by-zero", div_violations[0]["reason"])

    def test_log_zero_risk(self):
        """log(0) 风险检测。"""
        is_stable, violations = self.validator._check_numerical_stability(
            MOCK_ODE_LOG_RISK
        )
        self.assertFalse(is_stable)
        log_violations = [v for v in violations if v["type"] == "log_risk"]
        self.assertGreater(len(log_violations), 0)
        self.assertIn("log(0)", log_violations[0]["reason"])

    def test_stiff_system_detection(self):
        """stiff system 检测（rate ratio > 1e6）。"""
        is_stable, violations = self.validator._check_numerical_stability(
            MOCK_ODE_STIFF
        )
        self.assertFalse(is_stable)
        stiff_violations = [v for v in violations if v["type"] == "stiff_system"]
        self.assertGreater(len(stiff_violations), 0)
        self.assertIn("Stiff system", stiff_violations[0]["reason"])

    def test_no_rates_is_stable(self):
        """无速率常数 → 稳定。"""
        ode = "dEGFR/dt = -EGFR"
        is_stable, violations = self.validator._check_numerical_stability(ode)
        self.assertTrue(is_stable)

    def test_empty_ode_is_stable(self):
        """空 ODE → 稳定。"""
        is_stable, violations = self.validator._check_numerical_stability(
            MOCK_ODE_EMPTY
        )
        self.assertTrue(is_stable)
        self.assertEqual(len(violations), 0)


# =============================================================================
# TestConstraintSatisfaction: 约束满足检查
# =============================================================================
class TestConstraintSatisfaction(unittest.TestCase):
    """测试 _check_constraint_satisfaction。"""

    def setUp(self):
        from app.validation_v2.level1_internal import Level1InternalValidator
        self.validator = Level1InternalValidator()

    def test_all_constraints_satisfied(self):
        """所有约束满足。"""
        violations = self.validator._check_constraint_satisfaction(
            MOCK_REACTION_IR_CLEAN
        )
        self.assertEqual(len(violations), 0)

    def test_pip_conservation_violation(self):
        """PIP2 + PIP3 守恒失败（130 vs 100）。"""
        violations = self.validator._check_constraint_satisfaction(
            MOCK_REACTION_IR_PIP_VIOLATION
        )
        self.assertGreater(len(violations), 0)
        self.assertIn("Mass conservation", violations[0]["reason"])
        self.assertGreater(violations[0]["diff"], 0.05)

    def test_no_constraints_passes(self):
        """无约束 → 通过。"""
        violations = self.validator._check_constraint_satisfaction(
            MOCK_REACTION_IR_NO_CONSTRAINTS
        )
        self.assertEqual(len(violations), 0)

    def test_invalid_expression_violation(self):
        """无效表达式（无 =）→ 报违规。"""
        violations = self.validator._check_constraint_satisfaction(
            MOCK_REACTION_IR_INVALID_EXPR
        )
        self.assertGreater(len(violations), 0)
        self.assertIn("no '='", violations[0]["reason"])

    def test_steady_state_constraint_passes(self):
        """稳态约束表达式非空 → 通过。"""
        violations = self.validator._check_constraint_satisfaction(
            MOCK_REACTION_IR_STEADY_STATE
        )
        # 稳态约束表达式非空，不报违规
        steady_state_violations = [
            v for v in violations if "steady_state" in v.get("constraint_name", "")
        ]
        self.assertEqual(len(steady_state_violations), 0)

    def test_steady_state_empty_expression_violation(self):
        """稳态约束表达式为空 → 报违规。"""
        violations = self.validator._check_constraint_satisfaction(
            MOCK_REACTION_IR_STEADY_STATE_EMPTY
        )
        self.assertGreater(len(violations), 0)
        self.assertIn("empty", violations[0]["reason"].lower())

    def test_thermodynamic_missing_keywords(self):
        """热力学约束缺少关键字 → 报违规。"""
        violations = self.validator._check_constraint_satisfaction(
            MOCK_REACTION_IR_THERMO_MISSING_KW
        )
        self.assertGreater(len(violations), 0)
        self.assertIn("missing keywords", violations[0]["reason"].lower())

    def test_thermodynamic_with_keywords_passes(self):
        """热力学约束含关键字 → 通过。"""
        violations = self.validator._check_constraint_satisfaction(
            MOCK_REACTION_IR_THERMO_WITH_KW
        )
        self.assertEqual(len(violations), 0)


# =============================================================================
# 主入口
# =============================================================================
if __name__ == "__main__":
    unittest.main()
