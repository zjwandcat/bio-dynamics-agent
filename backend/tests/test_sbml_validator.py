# P0-4 升级测试：SBML Validator 与 SBML 三角色
# 对应修复提示词1.md §二.6 SBML Validation Layer
#
# 测试覆盖：
# 1. SBMLRole 三角色检测（Primary / Calibration / Validation Oracle / None）
# 2. SBMLValidator.validate 主流程（双轨：libroadrunner / 参数对齐法）
# 3. validation_report 字段完整性（error_diff / peak_time_diff / amplification_diff / sbml_sim_available / method / role / pass）
# 4. 跳过场景（无 SBML / 无 CSV）
# 5. SBML XML 参数提取（_extract_params_from_sbml_xml）
# 6. 物种模糊匹配（_fuzzy_match_species）
# 7. graph_v3 worker_validator 接入（不会破坏现有流水线）
# 8. supervisor.py 注册 SBML Validator Agent

import csv
import os
import sys
import tempfile
import unittest
from pathlib import Path

# 添加 backend 根目录到 sys.path，便于直接运行
_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))


class TestSBMLRole(unittest.TestCase):
    """测试 SBML 三角色检测。"""

    def test_primary_ground_truth_when_biomd_id_in_input(self):
        """用户输入含 BIOMD* ID 且仿真未跑 → Primary Ground Truth"""
        from app.biomodels_client import detect_sbml_role, SBML_ROLE_PRIMARY_GROUND_TRUTH
        user_input = "请基于 BIOMD0000000205 模型的参数仿真 EGF 刺激下 EGFR 磷酸化"
        role = detect_sbml_role(user_input, has_simulation_run=False)
        self.assertEqual(role, SBML_ROLE_PRIMARY_GROUND_TRUTH)

    def test_validation_oracle_after_simulation_run(self):
        """用户输入含 BIOMD* ID 且仿真已跑 → Validation Oracle"""
        from app.biomodels_client import detect_sbml_role, SBML_ROLE_VALIDATION_ORACLE
        user_input = "请基于 BIOMD0000000205 模型的参数仿真 EGF 刺激下 EGFR 磷酸化"
        role = detect_sbml_role(user_input, has_simulation_run=True)
        self.assertEqual(role, SBML_ROLE_VALIDATION_ORACLE)

    def test_calibration_reference_when_pathway_keyword_only(self):
        """用户输入不含 BIOMD* ID 但含通路关键词 → Calibration Reference"""
        from app.biomodels_client import detect_sbml_role, SBML_ROLE_CALIBRATION_REFERENCE
        user_input = "仿真 EGF 刺激下 EGFR 信号级联"
        role = detect_sbml_role(user_input, has_simulation_run=False)
        self.assertEqual(role, SBML_ROLE_CALIBRATION_REFERENCE)

    def test_none_when_no_sbml_signal(self):
        """用户输入无 SBML 信号 → None"""
        from app.biomodels_client import detect_sbml_role, SBML_ROLE_NONE
        user_input = "今天天气怎么样"
        role = detect_sbml_role(user_input, has_simulation_run=False)
        self.assertEqual(role, SBML_ROLE_NONE)

    def test_empty_input_returns_none(self):
        from app.biomodels_client import detect_sbml_role, SBML_ROLE_NONE
        self.assertEqual(detect_sbml_role("", False), SBML_ROLE_NONE)
        self.assertEqual(detect_sbml_role(None, False), SBML_ROLE_NONE)


class TestSBMLParamExtraction(unittest.TestCase):
    """测试从 SBML XML 提取参数（Track B 兜底）。"""

    def test_extract_k_on_k_off_from_xml(self):
        sbml_xml = """
        <sbml>
          <reaction id="r1">
            <kineticLaw>
              <listOfParameters>
                <parameter id="k_on" value="0.003"/>
                <parameter id="k_off" value="0.001"/>
              </listOfParameters>
            </kineticLaw>
          </reaction>
        </sbml>
        """
        from app.sbml_validator import _extract_params_from_sbml_xml
        params = _extract_params_from_sbml_xml(sbml_xml)
        self.assertIn("_binding_", params)
        self.assertAlmostEqual(params["_binding_"]["k_on"], 0.003)
        self.assertAlmostEqual(params["_binding_"]["k_off"], 0.001)

    def test_extract_empty_xml_returns_empty_dict(self):
        from app.sbml_validator import _extract_params_from_sbml_xml
        self.assertEqual(_extract_params_from_sbml_xml(""), {})
        self.assertEqual(_extract_params_from_sbml_xml("<not_sbml></not_sbml>"), {})


class TestFuzzySpeciesMatch(unittest.TestCase):
    """测试 SBML 物种名模糊匹配。"""

    def test_exact_match(self):
        from app.sbml_validator import _fuzzy_match_species
        sbml_species = ["EGF", "EGFR", "pEGFR", "pMAPK"]
        self.assertEqual(_fuzzy_match_species(sbml_species, "pEGFR"), "pEGFR")

    def test_substring_match(self):
        from app.sbml_validator import _fuzzy_match_species
        sbml_species = ["EGF_EGFR_complex"]
        self.assertEqual(_fuzzy_match_species(sbml_species, "EGFR"), "EGF_EGFR_complex")

    def test_no_match_returns_empty(self):
        from app.sbml_validator import _fuzzy_match_species
        sbml_species = ["Ras", "Raf"]
        self.assertEqual(_fuzzy_match_species(sbml_species, "pEGFR"), "")

    def test_empty_input_returns_empty(self):
        from app.sbml_validator import _fuzzy_match_species
        self.assertEqual(_fuzzy_match_species([], "pEGFR"), "")
        self.assertEqual(_fuzzy_match_species(["pEGFR"], ""), "")


class TestSimulationCSVReader(unittest.TestCase):
    """测试仿真 CSV 读取。"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.csv_path = os.path.join(self.tmpdir, "test_simulation.csv")
        with open(self.csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "EGF", "EGFR", "pEGFR", "pMAPK"])
            for i, t in enumerate([0, 5, 10, 15, 20]):
                writer.writerow([t, 0.008, 0.3 - 0.001 * i, 0.001 * i, 0.01 * i])

    def test_read_simulation_csv(self):
        from app.sbml_validator import _read_simulation_csv
        species, times, concs = _read_simulation_csv(self.csv_path)
        self.assertEqual(species, ["EGF", "EGFR", "pEGFR", "pMAPK"])
        self.assertEqual(len(times), 5)
        self.assertEqual(times[0], 0.0)
        self.assertEqual(times[-1], 20.0)
        self.assertEqual(concs["EGF"][0], 0.008)
        # pEGFR = 0.001 * i，i=4 时 pEGFR=0.004
        self.assertEqual(concs["pEGFR"][-1], 0.004)

    def test_read_nonexistent_csv_returns_empty(self):
        from app.sbml_validator import _read_simulation_csv
        species, times, concs = _read_simulation_csv("/nonexistent/path.csv")
        self.assertEqual(species, [])
        self.assertEqual(times, [])
        self.assertEqual(concs, {})


class TestMetricsExtraction(unittest.TestCase):
    """测试指标提取。"""

    def test_extract_species_metrics(self):
        from app.sbml_validator import _extract_species_metrics
        times = [0, 5, 10, 15, 20]
        concs = {"pEGFR": [0, 0.001, 0.005, 0.003, 0.001]}
        m = _extract_species_metrics(times, concs, "pEGFR")
        self.assertAlmostEqual(m["peak"], 0.005)
        self.assertEqual(m["peak_time"], 10.0)
        # AUC > 0
        self.assertGreater(m["auc"], 0)

    def test_compute_amplification(self):
        from app.sbml_validator import _compute_amplification
        times = [0, 5, 10, 15, 20]
        concs = {
            "pEGFR": [0, 0.001, 0.005, 0.003, 0.001],
            "pMAPK": [0, 0.01, 0.05, 0.03, 0.01],
        }
        amp = _compute_amplification(times, concs, "pEGFR", "pMAPK")
        self.assertAlmostEqual(amp, 10.0)  # 0.05 / 0.005 = 10

    def test_amplification_zero_upstream(self):
        from app.sbml_validator import _compute_amplification
        times = [0, 5, 10]
        concs = {"pEGFR": [0, 0, 0], "pMAPK": [0.1, 0.2, 0.1]}
        amp = _compute_amplification(times, concs, "pEGFR", "pMAPK")
        self.assertEqual(amp, 0.0)


class TestSBMLValidatorSkipped(unittest.TestCase):
    """测试 SBMLValidator 在跳过场景的行为。"""

    def test_validate_returns_skipped_when_no_sbml(self):
        from app.sbml_validator import get_sbml_validator
        validator = get_sbml_validator()
        # 无 SBML 信号的用户输入
        report = validator.validate(
            user_input="今天天气怎么样",
            simulation_csv_path="/nonexistent.csv",
        )
        self.assertEqual(report["method"], "skipped")
        self.assertTrue(report["pass"])
        self.assertEqual(report["error_diff"], 0.0)

    def test_validate_returns_skipped_when_csv_unavailable(self):
        from app.sbml_validator import get_sbml_validator
        validator = get_sbml_validator()
        # 含 BIOMD* ID 但 CSV 不存在
        report = validator.validate(
            user_input="请基于 BIOMD0000000205 仿真",
            simulation_csv_path="/nonexistent.csv",
        )
        self.assertEqual(report["method"], "skipped")
        self.assertTrue(report["pass"])

    def test_validate_param_aligned_with_local_sbml(self):
        """使用本地缓存的 BIOMD0000000205.xml 测试 Track B 参数对齐法。"""
        from app.sbml_validator import get_sbml_validator
        sbml_path = _BACKEND_DIR / "data" / "raw" / "BIOMD0000000205.xml"
        if not sbml_path.exists():
            self.skipTest("本地 SBML 缓存 BIOMD0000000205.xml 不存在，跳过")

        sbml_text = sbml_path.read_text(encoding="utf-8", errors="ignore")

        # 构造一个最小仿真 CSV
        tmpdir = tempfile.mkdtemp()
        csv_path = os.path.join(tmpdir, "tpl_simulation.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "EGF", "EGFR", "pEGFR", "pMAPK"])
            for i, t in enumerate([0, 5, 10, 15, 20, 30, 60, 120]):
                writer.writerow([t, 0.008, 0.3 - 0.01 * i, 0.001 * i, 0.01 * i])

        validator = get_sbml_validator()
        report = validator.validate(
            user_input="基于 BIOMD0000000205 仿真",
            simulation_csv_path=csv_path,
            sbml_model_id="BIOMD0000000205",
            sbml_text=sbml_text,
        )
        # 由于 libroadrunner 通常不可用，预期走 Track B 参数对齐法
        # 但若环境装了 libroadrunner，则走 Track A
        self.assertIn(report["method"], ("param_aligned", "libroadrunner", "skipped"))
        # 必须有 pass 字段
        self.assertIn("pass", report)
        # error_diff 字段必须存在
        self.assertIn("error_diff", report)
        # amplification_diff 字段必须存在
        self.assertIn("amplification_diff", report)


class TestGraphV3WorkerValidatorIntegration(unittest.TestCase):
    """测试 graph_v3 中 worker_validator 的接入。"""

    def test_worker_validator_in_worker_names(self):
        from app.graph_v3 import WORKER_NAMES
        self.assertIn("worker_validator", WORKER_NAMES)

    def test_worker_validator_in_full_plan(self):
        from app.graph_v3 import _FULL_PLAN
        self.assertIn("worker_validator", _FULL_PLAN)
        # 必须在 worker_sandbox 之后
        sandbox_idx = _FULL_PLAN.index("worker_sandbox")
        validator_idx = _FULL_PLAN.index("worker_validator")
        self.assertGreater(validator_idx, sandbox_idx)

    def test_worker_validator_function_exists(self):
        from app.graph_v3 import worker_validator
        self.assertTrue(callable(worker_validator))

    def test_worker_validator_returns_skipped_when_no_sbml(self):
        """无 SBML 信号时 worker_validator 应跳过验证，不阻塞流水线。"""
        from app.graph_v3 import worker_validator
        from app.state import BioDynamicsState

        state: BioDynamicsState = {
            "user_input": "今天天气怎么样",  # 无 SBML 信号
            "simulation_csv_path": "/nonexistent.csv",
            "sbml_model_id": "",
            "sbml_model_text": "",
            "ode_model": {"template": ""},
        }
        update = worker_validator(state)
        self.assertIn("validation_report", update)
        report = update["validation_report"]
        self.assertEqual(report["method"], "skipped")
        self.assertTrue(report["pass"])
        self.assertIn("sbml_role", update)

    def test_worker_validator_skips_when_csv_missing(self):
        """有 SBML 角色但 CSV 缺失时应跳过，不抛异常。"""
        from app.graph_v3 import worker_validator
        from app.state import BioDynamicsState

        state: BioDynamicsState = {
            "user_input": "请基于 BIOMD0000000205 仿真 EGF-EGFR 磷酸化",
            "simulation_csv_path": "",  # CSV 缺失
            "sbml_model_id": "",
            "sbml_model_text": "",
            "ode_model": {"template": "Signaling_Cascade_Phos"},
        }
        update = worker_validator(state)
        report = update["validation_report"]
        self.assertEqual(report["method"], "skipped")
        self.assertTrue(report["pass"])


class TestSupervisorRegistration(unittest.TestCase):
    """测试 supervisor.py 中 SBML Validator Agent 的注册。"""

    def test_sbml_validator_agent_in_registry_v2(self):
        from app.supervisor import AGENT_REGISTRY_V2
        names = [a.name for a in AGENT_REGISTRY_V2]
        self.assertIn("SBML Validator Agent", names)

    def test_get_agent_by_node_v2_finds_validator(self):
        from app.supervisor import get_agent_by_node_v2
        agent = get_agent_by_node_v2("worker_validator")
        self.assertIsNotNone(agent)
        self.assertEqual(agent.name, "SBML Validator Agent")


class TestStateFieldAdded(unittest.TestCase):
    """测试 state.py 新增字段。"""

    def test_state_has_sbml_role_field(self):
        from app.state import BioDynamicsState
        # TypedDict 在运行时是 dict，无法直接检查字段，但能确保导入不报错
        self.assertTrue(hasattr(BioDynamicsState, "__annotations__"))
        annotations = BioDynamicsState.__annotations__
        self.assertIn("sbml_role", annotations)
        self.assertIn("validation_report", annotations)


if __name__ == "__main__":
    unittest.main(verbosity=2)
