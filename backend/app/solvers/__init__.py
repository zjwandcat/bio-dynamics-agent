# BioDynamics Agent v4 - Solvers Module（Phase 3 新增）
# 对应 v4 Scientific Architecture Part 4 + Migration Plan §3.2。
#
# 职责：提供 ODE / DDE 求解器封装与动力学行为检测器。
#
# 模块：
#   - dde_solver.py: DDE 求解器（jitcdde try-import 降级）
#   - oscillation_detector.py: 振荡检测（p53/NF-κB）
#   - bistability_detector.py: 双稳态检测（Apoptosis/Cell Cycle）
#
# 设计原则（铁律）：
# 1. 不修改 v3 sandbox.py（不可碰清单）
# 2. 检测器纯函数，可被 v4 模板或外部调用
# 3. jitcdde 不可用时降级为 scipy ODE，不抛异常

__all__ = [
    "dde_solver",
    "oscillation_detector",
    "bistability_detector",
]
