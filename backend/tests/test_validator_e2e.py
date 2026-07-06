# P0-4 端到端集成测试：worker_validator 在真实流水线中触发
# 模拟 EGF-EGFR 场景跑完 worker_sandbox 后，worker_validator 应正确触发并输出 validation_report

import csv
import os
import sys
import tempfile
import unittest
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))


class TestWorkerValidatorE2E(unittest.TestCase):
    """端到端：worker_validator 在有 BIOMD* ID 与 CSV 的真实场景下触发。"""

    def setUp(self):
        # 构造一个真实的仿真 CSV（pEGFR 在 5-10 min 达峰）
        self.tmpdir = tempfile.mkdtemp()
        self.csv_path = os.path.join(self.tmpdir, "egf_egfr_simulation.csv")
        with open(self.csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "EGF", "EGFR", "pEGFR", "pMAPK"])
            # 模拟 pEGFR 5 min 达峰 0.005，pMAPK 在 30 min 达峰 0.05
            for i, t in enumerate([0, 1, 2, 5, 10, 20, 30, 60, 90, 120]):
                pEGFR = 0.005 * (1 - (1 - t / 5) ** 2) if t <= 5 else 0.005 * (0.8 ** (t / 5 - 1))
                pMAPK = min(0.05, 0.05 * (t / 30)) if t <= 30 else 0.05 * (0.9 ** (t / 30 - 1))
                writer.writerow([t, 0.008, 0.3, pEGFR, pMAPK])

    def test_worker_validator_triggers_validation_with_biomd_id(self):
        """用户输入含 BIOMD* ID + 仿真 CSV 已生成 → 触发验证（不跳过）。"""
        from app.graph_v3 import worker_validator

        # 检查本地 SBML 缓存
        sbml_path = _BACKEND_DIR / "data" / "raw" / "BIOMD0000000205.xml"
        if not sbml_path.exists():
            self.skipTest("本地 SBML 缓存 BIOMD0000000205.xml 不存在，跳过端到端测试")

        sbml_text = sbml_path.read_text(encoding="utf-8", errors="ignore")

        state = {
            "user_input": "请基于 BIOMD0000000205 模型仿真 EGF-EGFR 磷酸化",
            "simulation_csv_path": self.csv_path,
            "sbml_model_id": "BIOMD0000000205",
            "sbml_model_text": sbml_text,
            "ode_model": {"template": "Signaling_Cascade_Phos"},
        }

        update = worker_validator(state)

        # 验证 validation_report 字段完整
        self.assertIn("validation_report", update)
        report = update["validation_report"]
        # 应该走 param_aligned 或 libroadrunner（取决于环境）
        self.assertIn(report["method"], ("param_aligned", "libroadrunner"))
        self.assertIn("error_diff", report)
        self.assertIn("peak_time_diff", report)
        self.assertIn("amplification_diff", report)
        self.assertIn("sbml_sim_available", report)
        self.assertIn("pass", report)
        self.assertIn("role", report)
        # 角色应该是 validation_oracle（仿真已跑完）
        self.assertEqual(report["role"], "validation_oracle")
        # agent_dispatches 应包含 completed 状态
        self.assertIn("agent_dispatches", update)
        dispatches = update["agent_dispatches"]
        self.assertTrue(any(d.get("status") == "completed" or d.get("status") == "completed_with_warning"
                          for d in dispatches))

    def test_worker_validator_skips_gracefully_when_no_biomd(self):
        """用户输入不含 BIOMD* → 跳过验证，但流水线不阻塞。"""
        from app.graph_v3 import worker_validator

        state = {
            "user_input": "仿真 EGF-EGFR 信号级联",
            "simulation_csv_path": self.csv_path,
            "sbml_model_id": "",
            "sbml_model_text": "",
            "ode_model": {"template": "Signaling_Cascade_Phos"},
        }

        update = worker_validator(state)
        report = update["validation_report"]
        # 无 BIOMD* ID 但含通路关键词 → calibration_reference，但无 SBML 文本可用 → 跳过
        # 或直接走参数对齐法
        self.assertIn(report["method"], ("skipped", "param_aligned", "libroadrunner"))
        self.assertTrue(report["pass"])  # 不阻塞流水线

    def test_worker_validator_logs_dispatch_events(self):
        """worker_validator 应生成正确的 dispatch 事件。"""
        from app.graph_v3 import worker_validator

        state = {
            "user_input": "请基于 BIOMD0000000205 仿真",
            "simulation_csv_path": "",  # CSV 缺失，应跳过
            "sbml_model_id": "",
            "sbml_model_text": "",
            "ode_model": {"template": ""},
        }

        update = worker_validator(state)
        dispatches = update.get("agent_dispatches", [])
        # 应至少有 in_progress 和 completed 两个事件
        self.assertGreaterEqual(len(dispatches), 2)
        statuses = [d.get("status") for d in dispatches]
        self.assertIn("in_progress", statuses)


if __name__ == "__main__":
    unittest.main(verbosity=2)
