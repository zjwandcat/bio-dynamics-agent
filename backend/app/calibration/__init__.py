# BioDynamics Agent v4 - Calibration Agent (Phase 5 / Task 5.7)
#
# 职责：用 BioModels reference 或用户实验数据拟合 ODE 系统的参数，
# 输出 calibrated_params + confidence_intervals。
#
# 设计原则（铁律）：
# 1. Feature Flag V4_CALIBRATION_AGENT_ENABLED=false 时 hook 返回 {}，不执行任何逻辑
# 2. 不修改 v3 任何字段（network_json / parameters / ode_model / sbml_validator 等）
# 3. 仅消费 P1/P2/P3 产出（v4_ode_system / v4_reaction_ir / state.parameters）
# 4. 失败降级：任何异常都返回空更新，不阻塞主流水线
# 5. 输出写入 state["v4_calibration_result"]（新增 v4 字段，与 v3 字段共存）
# 6. 依赖隔离：lmfit 不可用时降级到 scipy.optimize.least_squares
#
# 模块结构：
#   - least_squares_fitter.py : lmfit / scipy.least_squares 双路径 + FitResult dataclass
#   - confidence_interval.py  : lmfit stderr / bootstrap 双路径 + ConfidenceInterval dataclass
#   - calibration_agent.py    : CalibrationAgent 主类 + LangGraph hook

from __future__ import annotations

from app.calibration.calibration_agent import CalibrationAgent, calibration_hook_node
from app.calibration.confidence_interval import (
    ConfidenceInterval,
    ConfidenceIntervalEstimator,
)
from app.calibration.least_squares_fitter import FitResult, LeastSquaresFitter

__all__ = [
    "CalibrationAgent",
    "calibration_hook_node",
    "FitResult",
    "LeastSquaresFitter",
    "ConfidenceInterval",
    "ConfidenceIntervalEstimator",
]
