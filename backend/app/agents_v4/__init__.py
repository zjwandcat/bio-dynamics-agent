# BioDynamics Agent v4 - agents_v4 包入口（Phase 6 / Task 6.6）
#
# 导出 4 个 v4 Agent 类，供 DynamicRouter 懒加载与外部调用。
#
# 设计原则：
# 1. 仅导出类，不执行任何副作用逻辑（纯模块导出）
# 2. 每个 Agent 的 AGENT_VERSION = "v4.0"，方法入口检查 V4_DYNAMIC_ROUTING_ENABLED
# 3. 兼容 DynamicRouter._get_class_name 的短名别名（MechanismBuilder / ODEBuilder /
#    SimulationPlanner / ParameterAgent）在各自模块内定义
#
# 参考：
# - spec.md Part 6 Dynamic Router（第 393-398 行）
# - tasks.md SubTask 6.6.1-6.6.4

from __future__ import annotations

from app.agents_v4.mechanism_builder import MechanismBuilderAgent
from app.agents_v4.ode_builder import ODEBuilderAgent
from app.agents_v4.parameter_agent import ParameterAgent
from app.agents_v4.simulation_planner import SimulationPlannerAgent

__all__ = [
    "MechanismBuilderAgent",
    "ODEBuilderAgent",
    "SimulationPlannerAgent",
    "ParameterAgent",
]
