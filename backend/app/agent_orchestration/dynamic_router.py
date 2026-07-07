# BioDynamics Agent v4 - Dynamic Router（Phase 6 / Task 6.5.1 + 6.5.5）
#
# 基于 v4_pathway_class 动态编排 13 Agent，输出 v4_agent_dispatches 调度记录。
#
# 设计原则（铁律）：
# 1. Feature Flag V4_DYNAMIC_ROUTING_ENABLED=false 时返回 {}，不执行任何逻辑
# 2. flag=true 隐含 V4_PATHWAY_PLANNER_ENABLED=true（Router 依赖 pathway_class）
# 3. 不修改 v3 任何字段；仅新增 v4_agent_dispatches
# 4. P6 Dynamic Router 不修改 P4 PathwayGraph（只读）
# 5. fail_safe：超时 30s 回退 v3 / 最大调度深度 10 / visited set 防环
# 6. P6/Task 6.6 未实现的 Agent（mechanism_builder/ode_builder/simulation_planner/
#    parameter_agent）通过 execute_agent 捕获 ImportError 返回 {}（stub）
# 7. 失败降级：任何异常都返回 degraded 结果，不阻塞主流水线
#
# 参考：
# - spec.md Part 6 Dynamic Router（第 393-398 行）
# - spec.md Part 7 Feature Flag 矩阵（第 463 行 + 第 472 行依赖规则）
# - tasks.md SubTask 6.5.1 / 6.5.5

from __future__ import annotations

import logging

from app.agent_orchestration.agent_registry_v4 import get_agent_spec
from app.agent_orchestration.fail_safe import (
    FailSafeConfig,
    FailSafeDispatcher,
)
from app.agent_orchestration.pathway_class_dispatcher import PathwayClassDispatcher
from app.state import set_v4_state

logger = logging.getLogger(__name__)


# =============================================================================
# DynamicRouter 主类
# =============================================================================
class DynamicRouter:
    """v4 动态路由器：基于 v4_pathway_class 编排 13 Agent。

    职责：
    1. 检查 V4_DYNAMIC_ROUTING_ENABLED flag（false → 返回 {}）
    2. 根据 v4_pathway_class 构建调度计划（哪些 Agent 需要调用）
    3. 对每个 Agent 调用 FailSafeDispatcher.dispatch()（含超时/防环/深度保护）
    4. 收集调度结果到 v4_agent_dispatches 列表
    5. 返回 ``{"v4_agent_dispatches": [...]}`` 供 LangGraph 合并到 state

    不职责（铁律）：
    - 不修改 v3 任何字段
    - 不修改 P4 PathwayGraph（只读）
    - 不直接执行 Agent 业务逻辑（通过 execute_agent 懒加载 + 调用）

    用法::

        router = DynamicRouter()
        update = router.route(state)
        # update = {"v4_agent_dispatches": [...]}
    """

    def __init__(self, fail_safe_config: FailSafeConfig | None = None) -> None:
        """初始化路由器。

        Args:
            fail_safe_config: Fail-safe 配置；None 时使用默认配置（max_depth=10,
                timeout=30s）。可通过自定义配置调整超时阈值（如测试场景用 1s）
        """
        self._fail_safe_config: FailSafeConfig = fail_safe_config or FailSafeConfig()
        # 每个 route() 调用使用独立的 FailSafeDispatcher（visited set 隔离）
        self._pathway_dispatcher: PathwayClassDispatcher = PathwayClassDispatcher()

    # -------------------------------------------------------------------------
    # 主入口：route
    # -------------------------------------------------------------------------
    def route(self, state: dict) -> dict:
        """主入口：动态路由 13 Agent，返回 v4_agent_dispatches 更新。

        行为：
        1. 检查 V4_DYNAMIC_ROUTING_ENABLED flag（false → 返回 {}）
        2. 构建 dispatch plan（基于 v4_pathway_class + 各 Feature Flag）
        3. 对每个 agent_id 调用 FailSafeDispatcher.dispatch()
        4. 收集 DispatchResult.to_dict() 到 v4_agent_dispatches 列表
        5. 返回 ``{"v4_agent_dispatches": [...]}``

        Args:
            state: LangGraph 全局状态，读取：
                - ``v4_pathway_class``: 通路类别字符串
                - 其他 Agent 执行所需字段

        Returns:
            flag=false 时返回 {}
            flag=true 时返回 ``{"v4_agent_dispatches": [...]}``
            异常时返回 ``{"v4_agent_dispatches": []}``（不阻塞主流水线）
        """
        # 延迟导入 settings 避免循环依赖（虽然 config 无循环风险，保持一致）
        from app.config import settings

        # 1. Feature Flag 检查（铁律：flag=false 不执行）
        if not settings.effective_v4_dynamic_routing_enabled():
            logger.debug("V4_DYNAMIC_ROUTING_ENABLED effective=false，跳过 Dynamic Router")
            return {}

        try:
            # 2. 构建 dispatch plan
            plan = self.build_dispatch_plan(state)
            logger.info(
                "DynamicRouter: dispatch plan 含 %d 个 Agent: %s",
                len(plan),
                plan,
            )

            # 3. 每个 route() 调用使用独立的 FailSafeDispatcher（visited set 隔离）
            dispatcher = FailSafeDispatcher(self._fail_safe_config)

            # 4. 对每个 agent 调度
            dispatches: list[dict] = []
            for agent_id in plan:
                spec = get_agent_spec(agent_id)
                agent_name = spec.name if spec else agent_id

                # 构造 agent_func：捕获 agent_id 用于 execute_agent
                def _agent_func(s: dict, _aid: str = agent_id) -> dict:
                    return self.execute_agent(_aid, s)

                result = dispatcher.dispatch(
                    agent_id=agent_id,
                    agent_func=_agent_func,
                    state=state,
                    depth=0,
                    agent_name=agent_name,
                )
                dispatches.append(result.to_dict())

                # fail-safe 短路：若 agent timeout 或 depth_exceeded 且 fallback_used=True，
                # 仍继续调度后续 Agent（fail_safe 隔离原则：单 Agent 失败不阻塞其他 Agent）
                # 但若需要回退 v3 流水线，调用方可根据 dispatches 中 fallback_used 标记决定

            logger.info(
                "DynamicRouter: 调度完成，%d 条记录，%d 成功 / %d 失败",
                len(dispatches),
                sum(1 for d in dispatches if d.get("status") == "success"),
                sum(1 for d in dispatches if d.get("status") != "success"),
            )

            return {"v4_agent_dispatches": dispatches}
        except Exception as exc:
            # 任何失败都不阻塞流水线，记录 warning 并返回空更新
            logger.warning(
                "DynamicRouter.route 失败，降级返回空 v4_agent_dispatches: %s",
                exc,
            )
            return {"v4_agent_dispatches": []}

    # -------------------------------------------------------------------------
    # 调度计划构建
    # -------------------------------------------------------------------------
    def build_dispatch_plan(self, state: dict) -> list[str]:
        """根据 state 与 Feature Flags 构建调度计划。

        调度规则（spec.md Part 6 + Feature Flag 矩阵）：
        1. 始终包含核心 Agent：
           - ontology（P1 本体标准化）
           - pathway_planner（P4 通路识别）
           - pathway_specialist_group（P4 10 Specialist 组入口）
           - reaction_builder（P2 反应构建）
           - validation（P5 五层验证）
        2. 多通路场景：追加 crosstalk_coordinator
        3. 条件 Agent（按 Feature Flag）：
           - V4_SBML_GROUNDER_ENABLED → sbml_grounder
           - V4_CALIBRATION_AGENT_ENABLED → calibration
           - V4_HYPOTHESIS_AGENT_ENABLED → hypothesis
        4. P6/Task 6.6 未实现的 Agent（始终包含，execute_agent 会处理 ImportError）：
           - mechanism_builder / ode_builder / simulation_planner / parameter_agent

        Args:
            state: LangGraph 全局状态，读取 v4_pathway_class 判断多通路

        Returns:
            agent_id 列表（按调度顺序）
        """
        from app.config import settings

        plan: list[str] = []

        # 1. 核心 Agent（始终包含）
        plan.extend(
            [
                "ontology",
                "pathway_planner",
                "pathway_specialist_group",
                "reaction_builder",
                "validation",
            ]
        )

        # 2. 多通路场景：追加 crosstalk_coordinator
        pathway_class = state.get("v4_pathway_class", "") or ""
        if self._pathway_dispatcher.is_multi_pathway(pathway_class):
            plan.append("crosstalk_coordinator")

        # 3. 条件 Agent（按 Feature Flag）
        if settings.effective_v4_sbml_grounder_enabled():
            plan.append("sbml_grounder")
        if settings.effective_v4_calibration_agent_enabled():
            plan.append("calibration")
        if settings.effective_v4_hypothesis_enabled():
            plan.append("hypothesis")

        # 4. P6/Task 6.6 未实现的 Agent（始终包含，execute_agent 处理 ImportError）
        plan.extend(
            [
                "mechanism_builder",
                "ode_builder",
                "simulation_planner",
                "parameter_agent",
            ]
        )

        # 去重保持顺序（防止核心 Agent 与条件 Agent 重复）
        seen: set[str] = set()
        unique_plan: list[str] = []
        for aid in plan:
            if aid not in seen:
                unique_plan.append(aid)
                seen.add(aid)
        return unique_plan

    # -------------------------------------------------------------------------
    # Agent 执行（懒加载）
    # -------------------------------------------------------------------------
    def execute_agent(self, agent_id: str, state: dict) -> dict:
        """懒加载 + 执行单个 Agent，返回其输出 dict。

        策略：
        1. 通过 AGENT_REGISTRY_V4 查找 agent_id 的 mapped_module
        2. 懒加载导入对应模块（避免循环依赖与启动开销）
        3. 调用模块中约定的 hook 函数（如 ontology_hook_node / pathway_planner_hook_node）
           或返回 {}（未实现 stub）
        4. P6/Task 6.6 未实现的 Agent（mechanism_builder / ode_builder /
           simulation_planner / parameter_agent）会触发 ImportError，返回 {} stub
        5. 任何异常都返回 {}（fail-safe，不阻塞调度）

        Args:
            agent_id: 路由键
            state: LangGraph 全局状态

        Returns:
            Agent 输出 dict；未实现或失败时返回 {}
        """
        spec = get_agent_spec(agent_id)
        if spec is None:
            logger.warning(
                "DynamicRouter.execute_agent: agent_id=%s 未注册，返回 {}",
                agent_id,
            )
            return {}

        try:
            import importlib

            module = importlib.import_module(spec.mapped_module)

            # 各 Agent 模块的 hook 函数命名约定：<agent_short>_hook_node
            # 或类入口：直接调用类.generate(state)
            hook_func_name = self._get_hook_function_name(agent_id)
            hook_func = getattr(module, hook_func_name, None)

            if hook_func is not None and callable(hook_func):
                # 调用 hook 函数（如 ontology_hook_node(state)）
                result = hook_func(state)
                if isinstance(result, dict):
                    return result
                logger.warning(
                    "DynamicRouter.execute_agent: agent_id=%s hook %s "
                    "返回非 dict 类型 %s，返回 {}",
                    agent_id,
                    hook_func_name,
                    type(result).__name__,
                )
                return {}

            # 未找到 hook 函数：尝试调用类的 generate/run 方法
            class_name = self._get_class_name(agent_id)
            cls = getattr(module, class_name, None)
            if cls is not None:
                instance = cls() if callable(cls) else cls
                # 优先调用 generate(state)，其次 run(state)，最后 execute(state)
                for method_name in ("generate", "run", "execute"):
                    method = getattr(instance, method_name, None)
                    if callable(method):
                        result = method(state)
                        if isinstance(result, dict):
                            return result
                        if isinstance(result, list):
                            # list 类型结果包装为 dict（如 HypothesisAgent.generate 返回 list）
                            return {f"v4_{agent_id}_output": result}
                        logger.warning(
                            "DynamicRouter.execute_agent: agent_id=%s %s.%s "
                            "返回非 dict/list 类型 %s，返回 {}",
                            agent_id,
                            class_name,
                            method_name,
                            type(result).__name__,
                        )
                        return {}

            # 既无 hook 函数也无约定的类方法：返回 {} stub
            logger.debug(
                "DynamicRouter.execute_agent: agent_id=%s 模块 %s 无 hook 函数 %s "
                "也无类 %s，返回 {} stub",
                agent_id,
                spec.mapped_module,
                hook_func_name,
                class_name,
            )
            return {}
        except ImportError as exc:
            # P6/Task 6.6 未实现的 Agent：返回 {} stub
            logger.debug(
                "DynamicRouter.execute_agent: agent_id=%s 模块 %s 不可导入"
                "（Task 6.6 未实现），返回 {} stub: %s",
                agent_id,
                spec.mapped_module,
                exc,
            )
            return {}
        except Exception as exc:
            # 任何异常都返回 {}（fail-safe）
            logger.warning(
                "DynamicRouter.execute_agent: agent_id=%s 执行失败，返回 {}: %s",
                agent_id,
                exc,
            )
            return {}

    # -------------------------------------------------------------------------
    # 内部辅助：hook 函数名与类名映射
    # -------------------------------------------------------------------------
    @staticmethod
    def _get_hook_function_name(agent_id: str) -> str:
        """根据 agent_id 推断 hook 函数名。

        约定：
        - ontology → ontology_hook_node
        - pathway_planner → pathway_planner_hook_node
        - pathway_specialist_group → specialist_hook_node
        - crosstalk_coordinator → crosstalk_coordinator_hook_node
        - hypothesis → hypothesis_agent_hook_node
        - sbml_grounder → sbml_grounder_hook_node（若存在）
        - validation → validation_agent_hook_node（若存在）
        - calibration → calibration_agent_hook_node（若存在）
        - reaction_builder → build_from_pathway_graph（备用，非 hook）
        - 其余（mechanism_builder/ode_builder/simulation_planner/parameter_agent）
          → 无 hook，通过类入口处理
        """
        mapping: dict[str, str] = {
            "ontology": "ontology_hook_node",
            "pathway_planner": "pathway_planner_hook_node",
            "pathway_specialist_group": "specialist_hook_node",
            "crosstalk_coordinator": "crosstalk_coordinator_hook_node",
            "hypothesis": "hypothesis_agent_hook_node",
            "sbml_grounder": "sbml_grounder_hook_node",
            "validation": "validation_agent_hook_node",
            "calibration": "calibration_agent_hook_node",
        }
        return mapping.get(agent_id, f"{agent_id}_hook_node")

    @staticmethod
    def _get_class_name(agent_id: str) -> str:
        """根据 agent_id 推断主类名。

        约定：
        - ontology → OntologyAgent
        - pathway_planner → PathwayPlanner（无类，仅函数；走 hook 路径）
        - crosstalk_coordinator → CrossTalkCoordinator
        - reaction_builder → ReactionBuilder
        - sbml_grounder → SBMLGrounderAgent / GrounderAgent
        - calibration → CalibrationAgent
        - validation → ValidationAgent
        - hypothesis → HypothesisAgent
        - mechanism_builder → MechanismBuilder
        - ode_builder → ODEBuilder
        - simulation_planner → SimulationPlanner
        - parameter_agent → ParameterAgent
        """
        mapping: dict[str, str] = {
            "ontology": "OntologyAgent",
            "crosstalk_coordinator": "CrossTalkCoordinator",
            "reaction_builder": "ReactionBuilder",
            "sbml_grounder": "GrounderAgent",
            "calibration": "CalibrationAgent",
            "validation": "ValidationAgent",
            "hypothesis": "HypothesisAgent",
            "mechanism_builder": "MechanismBuilder",
            "ode_builder": "ODEBuilder",
            "simulation_planner": "SimulationPlanner",
            "parameter_agent": "ParameterAgent",
        }
        return mapping.get(agent_id, "".join(p.capitalize() for p in agent_id.split("_")))


# =============================================================================
# LangGraph hook 节点（Feature Flag 隔离）
# =============================================================================
def dynamic_router_hook_node(state: dict) -> dict:
    """LangGraph 节点：Dynamic Router hook。

    行为：
    - V4_DYNAMIC_ROUTING_ENABLED=false：直接返回空 dict（不修改 state）
    - V4_DYNAMIC_ROUTING_ENABLED=true：调用 DynamicRouter.route()
      写入 state["v4_agent_dispatches"]

    严格遵守不可碰清单：
    - 不修改 v3 任何字段（network_json / parameters / ode_model / sbml_validator 等）
    - 不修改 P4 PathwayGraph（只读）
    - 不生成 ODE / 不做 SBML 验证
    - 失败时降级返回空 dict，不抛异常，不阻塞主流水线

    依赖检查（spec.md 第 472 行）：
    - V4_DYNAMIC_ROUTING_ENABLED=true 隐含 V4_PATHWAY_PLANNER_ENABLED=true
      （Router 依赖 v4_pathway_class 输出）
    - 若 v4_pathway_class 缺失，route() 仍执行，但 dispatch plan 不含
      crosstalk_coordinator（单通路场景）

    Args:
        state: LangGraph 全局状态

    Returns:
        flag=false 时返回 {}
        flag=true 时返回 {"v4_agent_dispatches": [...]}
        异常时返回 {}（不阻塞主流水线）
    """
    from app.config import settings

    # Feature Flag 检查：默认 false，跳过所有逻辑
    if not settings.effective_v4_dynamic_routing_enabled():
        logger.debug("V4_DYNAMIC_ROUTING_ENABLED effective=false，跳过 Dynamic Router hook")
        return {}

    try:
        router = DynamicRouter()
        update = router.route(state)
        logger.info(
            "Dynamic Router hook 完成：v4_agent_dispatches 含 %d 条记录",
            len(update.get("v4_agent_dispatches", [])),
        )
        # Task B.2: 双写 v4_agent_dispatches → v4_state["router"]["dispatches"]
        if "v4_agent_dispatches" in update:
            set_v4_state(update, "router", "dispatches", update["v4_agent_dispatches"])
        return update
    except Exception as exc:
        # 任何失败都不阻塞流水线，记录 warning 并返回空更新
        logger.warning("Dynamic Router hook 失败，降级跳过: %s", exc)
        return {}


__all__ = [
    "DynamicRouter",
    "dynamic_router_hook_node",
]
