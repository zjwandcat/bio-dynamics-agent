# BioDynamics Agent v4 - Sensitivity Analysis (Phase 5 / Task 5.8)
#
# 职责：对 ODE 系统的参数执行灵敏度分析（local + sobol + morris 三路径），
# 输出 v4_sensitivity_report（local_sensitivity + sobol + morris + method）。
#
# 设计原则（铁律）：
# 1. Feature Flag V4_CALIBRATION_AGENT_ENABLED=false 时 hook 返回 {}，不执行任何逻辑
#    （Sensitivity Analysis 与 Calibration 共享 flag，spec.md 第 461 行明确）
# 2. 不修改 v3 任何字段（network_json / parameters / ode_model / sbml_validator 等）
# 3. 仅消费 P1/P2/P3/P5 产出（v4_ode_system / v4_calibration_result / state.parameters）
# 4. 失败降级：任何异常都返回空结果或降级路径，不阻塞主流水线
# 5. 输出写入 state["v4_sensitivity_report"]（新增 v4 字段，与 v3 字段共存）
# 6. 依赖隔离：SALib 不可用时仅运行 local sensitivity + warning
#
# 模块结构：
#   - local_sensitivity.py  : LocalSensitivityAnalyzer + forward difference（始终可用）
#   - sobol_analyzer.py     : SobolAnalyzer + SALib try-import 降级 skipped
#   - morris_analyzer.py    : MorrisAnalyzer + SALib try-import 降级 skipped
#   - sensitivity_analyzer.py : SensitivityAnalyzer 编排器 + LangGraph hook

from __future__ import annotations

from app.sensitivity.local_sensitivity import (
    LocalSensitivityAnalyzer,
    LocalSensitivityResult,
)
from app.sensitivity.morris_analyzer import MorrisAnalyzer, MorrisResult
from app.sensitivity.sensitivity_analyzer import (
    SensitivityAnalyzer,
    sensitivity_hook_node,
)
from app.sensitivity.sobol_analyzer import SobolAnalyzer, SobolResult

__all__ = [
    "SensitivityAnalyzer",
    "sensitivity_hook_node",
    "LocalSensitivityAnalyzer",
    "LocalSensitivityResult",
    "SobolAnalyzer",
    "SobolResult",
    "MorrisAnalyzer",
    "MorrisResult",
]
