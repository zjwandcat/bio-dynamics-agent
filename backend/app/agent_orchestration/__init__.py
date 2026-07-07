# BioDynamics Agent v4 - Agent Orchestration 包（Phase 6 / Task 6.5）
#
# 动态路由编排层：基于 v4_pathway_class 动态编排 13 Agent。
#
# 模块组成：
# - agent_registry_v4.py: 13 Agent 注册表（Ontology/Planner/Specialist Group/
#   Coordinator/ReactionBuilder/MechanismBuilder/ODEBuilder/SBMLGrounder/
#   Calibration/SimulationPlanner/Validation/Hypothesis/ParameterAgent）
# - dynamic_router.py: DynamicRouter 主类 + LangGraph hook 节点
# - pathway_class_dispatcher.py: 基于 v4_pathway_class 的 Specialist 分发器
# - fail_safe.py: 失败短路 + 超时 30s 回退 v3 + 最大调度深度 10 + visited set 防环
#
# 设计原则（铁律）：
# 1. Feature Flag V4_DYNAMIC_ROUTING_ENABLED=false 时 hook 返回 {}，不执行任何逻辑
# 2. flag=true 隐含 V4_PATHWAY_PLANNER_ENABLED=true（Router 依赖 pathway_class）
# 3. 不修改 v3 任何字段；仅新增 v4_agent_dispatches
# 4. P6 Dynamic Router 不修改 P4 PathwayGraph（只读）
# 5. fail_safe：超时 30s 回退 v3 / 最大调度深度 10 / visited set 防环
# 6. 失败降级：任何异常都返回 degraded 结果，不阻塞主流水线
#
# 参考：
# - spec.md Part 6 Dynamic Router（第 393-398 行）
# - tasks.md SubTask 6.5.1-6.5.6

from app.agent_orchestration.agent_registry_v4 import (
    AGENT_REGISTRY_V4,
    AgentSpecV4,
    SPECIALIST_AGENT_ID,
    count_agents,
    get_agent_spec,
    is_agent_registered,
    list_agent_ids,
)
from app.agent_orchestration.dynamic_router import (
    DynamicRouter,
    dynamic_router_hook_node,
)
from app.agent_orchestration.fail_safe import (
    DispatchResult,
    FailSafeConfig,
    FailSafeDispatcher,
)
from app.agent_orchestration.pathway_class_dispatcher import PathwayClassDispatcher

__all__ = [
    # agent_registry_v4
    "AGENT_REGISTRY_V4",
    "AgentSpecV4",
    "SPECIALIST_AGENT_ID",
    "get_agent_spec",
    "list_agent_ids",
    "count_agents",
    "is_agent_registered",
    # dynamic_router
    "DynamicRouter",
    "dynamic_router_hook_node",
    # fail_safe
    "FailSafeConfig",
    "FailSafeDispatcher",
    "DispatchResult",
    # pathway_class_dispatcher
    "PathwayClassDispatcher",
]
