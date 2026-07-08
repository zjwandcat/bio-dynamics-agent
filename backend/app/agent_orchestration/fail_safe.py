# BioDynamics Agent v4 - Fail-safe Dispatcher（Phase 6 / Task 6.5.4）
#
# 失败短路 + 超时回退 v3 + 最大调度深度 10 + visited set 防环。
#
# 设计原则（铁律）：
# 1. 任何异常都返回 degraded DispatchResult，不抛出，不阻塞主流水线
# 2. 超时 30s（可配置）→ fallback_used=True，调用方回退 v3 流水线
# 3. 最大调度深度 10（可配置）→ depth_exceeded，fallback_used=True
# 4. visited set 防环：同一 agent_id 二次调度 → loop_detected，fallback_used=True
# 5. 超时实现使用 threading.Thread + join(timeout)（Windows 不支持 signal.alarm）
# 6. 不修改 v3 任何字段；仅返回 DispatchResult 供调用方记录
#
# 参考：
# - spec.md Part 6 Dynamic Router fail_safe（第 393-398 行 + 风险表第 526 行）
# - tasks.md SubTask 6.5.4

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


# =============================================================================
# 配置数据类
# =============================================================================
@dataclass
class FailSafeConfig:
    """Fail-safe 调度器配置。

    Attributes:
        max_depth: 最大调度深度（默认 10），超过返回 depth_exceeded
        timeout_seconds: 单次调度超时阈值（默认 30s），超过返回 timeout
        enable_visited_check: 是否启用 visited set 防环检查（默认 True）
    """

    max_depth: int = 10
    timeout_seconds: int = 30
    enable_visited_check: bool = True


# =============================================================================
# 调度结果数据类
# =============================================================================
@dataclass
class DispatchResult:
    """单次 Agent 调度结果。

    Attributes:
        success: 调度是否成功（True 仅当 status="success"）
        agent_id: 调度的 agent_id
        agent_name: Agent 展示名（来自 AgentSpecV4.name，调度时填入）
        status: 调度状态，取值：
            - ``"success"``：正常完成
            - ``"failed"``：agent_func 抛出异常
            - ``"timeout"``：超过 timeout_seconds
            - ``"depth_exceeded"``：depth >= max_depth
            - ``"loop_detected"``：agent_id 在 visited set 中
        output: Agent 输出 dict（成功时为 agent_func 返回值，失败时为 {}）
        error: 错误信息（status=failed/timeout 时填入，其余为 None）
        latency_ms: 调度耗时（毫秒）
        fallback_used: 是否触发了 v3 回退（True 表示需要回退 v3 流水线）
        depth: 当前调度深度
    """

    success: bool
    agent_id: str
    agent_name: str
    status: str
    output: dict = field(default_factory=dict)
    error: str | None = None
    latency_ms: float = 0.0
    fallback_used: bool = False
    depth: int = 0

    def to_dict(self) -> dict[str, Any]:
        """转换为 dict（供 state.v4_agent_dispatches 记录）。"""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "status": self.status,
            "output": self.output,
            "error": self.error,
            "latency_ms": round(self.latency_ms, 1),
            "fallback_used": self.fallback_used,
            "depth": self.depth,
        }


# =============================================================================
# 主图核心 worker 保护集合（IB-022 修复：主图核心 worker fail-safe 保护）
# =============================================================================
# 主图 4 个核心 worker 必须被 FailSafeDispatcher 包裹，
# is_worker_protected 用于校验某 worker 是否属于受保护集合。
PROTECTED_WORKERS: set[str] = {
    "worker_ode",
    "worker_sandbox",
    "worker_validator",
    "worker_report",
}


def is_worker_protected(worker_name: str) -> bool:
    """检查指定 worker 是否属于主图受 fail-safe 保护的核心 worker。

    Args:
        worker_name: worker 名称（如 ``"worker_ode"``）

    Returns:
        True 表示该 worker 在 PROTECTED_WORKERS 集合中，应被 FailSafeDispatcher 包裹
    """
    return worker_name in PROTECTED_WORKERS


# =============================================================================
# FailSafeDispatcher 主类
# =============================================================================
class FailSafeDispatcher:
    """Fail-safe 调度器：包装 Agent 调用，提供超时/防环/深度保护。

    用法::

        dispatcher = FailSafeDispatcher()
        result = dispatcher.dispatch(
            agent_id="ontology",
            agent_func=lambda state: {"v4_ontology_entities": {...}},
            state=state,
        )
        if result.success:
            merge result.output into state
        else:
            # 回退 v3 流水线（result.fallback_used=True）

    线程安全：单实例非线程安全（visited set 共享）；每个 DynamicRouter.route()
    调用应使用独立实例，或在 route() 入口调用 clear_visited()。
    """

    def __init__(self, config: FailSafeConfig | None = None) -> None:
        """初始化调度器。

        Args:
            config: 配置实例；None 时使用默认配置（max_depth=10, timeout=30s）
        """
        self.config: FailSafeConfig = config or FailSafeConfig()
        # visited set：本调度会话中已调度的 agent_id 集合，防环
        self._visited: set[str] = set()

    # -------------------------------------------------------------------------
    # 主入口：dispatch
    # -------------------------------------------------------------------------
    def dispatch(
        self,
        agent_id: str,
        agent_func: Callable[[dict], dict],
        state: dict,
        depth: int = 0,
        agent_name: str | None = None,
    ) -> DispatchResult:
        """调度单个 Agent，提供 fail-safe 保护。

        Args:
            agent_id: 路由键（如 ``"ontology"``）
            agent_func: Agent 执行函数，签名为 ``func(state: dict) -> dict``
            state: LangGraph 全局状态
            depth: 当前调度深度（默认 0，递归调度时递增）
            agent_name: Agent 展示名（用于结果记录）；None 时使用 agent_id

        Returns:
            DispatchResult：含 status / output / latency_ms / fallback_used 等
        """
        display_name = agent_name or agent_id
        start_time = time.time()

        # 1. 深度检查：depth >= max_depth → depth_exceeded
        if depth >= self.config.max_depth:
            latency_ms = (time.time() - start_time) * 1000
            logger.warning(
                "FailSafeDispatcher: agent_id=%s depth=%d 超过最大深度 %d，"
                "返回 depth_exceeded",
                agent_id,
                depth,
                self.config.max_depth,
            )
            return DispatchResult(
                success=False,
                agent_id=agent_id,
                agent_name=display_name,
                status="depth_exceeded",
                output={},
                error=f"depth {depth} >= max_depth {self.config.max_depth}",
                latency_ms=latency_ms,
                fallback_used=True,
                depth=depth,
            )

        # 2. 防环检查：agent_id 已在 visited set → loop_detected
        if self.config.enable_visited_check and agent_id in self._visited:
            latency_ms = (time.time() - start_time) * 1000
            logger.warning(
                "FailSafeDispatcher: agent_id=%s 已在 visited set 中，"
                "返回 loop_detected",
                agent_id,
            )
            return DispatchResult(
                success=False,
                agent_id=agent_id,
                agent_name=display_name,
                status="loop_detected",
                output={},
                error=f"agent_id {agent_id} already visited (loop detected)",
                latency_ms=latency_ms,
                fallback_used=True,
                depth=depth,
            )

        # 3. 加入 visited set
        self._visited.add(agent_id)

        # 4. 执行 agent_func（带超时保护）
        output: dict = {}
        error: str | None = None
        status: str = "success"
        fallback_used: bool = False

        # 使用 threading 实现 timeout（Windows 不支持 signal.alarm）
        result_holder: dict[str, Any] = {"output": {}, "error": None}

        def _run_agent() -> None:
            try:
                ret = agent_func(state)
                if isinstance(ret, dict):
                    result_holder["output"] = ret
                else:
                    result_holder["error"] = (
                        f"agent_func 返回非 dict 类型: {type(ret).__name__}"
                    )
            except Exception as exc:  # noqa: BLE001
                result_holder["error"] = f"{type(exc).__name__}: {exc}"

        thread = threading.Thread(
            target=_run_agent,
            name=f"FailSafe-{agent_id}",
            daemon=True,
        )
        thread.start()
        thread.join(timeout=self.config.timeout_seconds)

        if thread.is_alive():
            # 超时：线程仍在运行（daemon 线程会被主进程回收，不强制 kill）
            latency_ms = (time.time() - start_time) * 1000
            logger.warning(
                "FailSafeDispatcher: agent_id=%s 超时 (%.1fs)，返回 timeout",
                agent_id,
                self.config.timeout_seconds,
            )
            return DispatchResult(
                success=False,
                agent_id=agent_id,
                agent_name=display_name,
                status="timeout",
                output={},
                error=f"timeout after {self.config.timeout_seconds}s",
                latency_ms=latency_ms,
                fallback_used=True,
                depth=depth,
            )

        # 5. 检查 agent_func 执行结果
        if result_holder["error"] is not None:
            # agent_func 抛出异常或返回非 dict
            error = result_holder["error"]
            status = "failed"
            fallback_used = False  # 异常不强制回退 v3，由调用方决定
            logger.warning(
                "FailSafeDispatcher: agent_id=%s 执行失败: %s",
                agent_id,
                error,
            )
        else:
            output = result_holder["output"]
            status = "success"
            fallback_used = False

        latency_ms = (time.time() - start_time) * 1000
        return DispatchResult(
            success=(status == "success"),
            agent_id=agent_id,
            agent_name=display_name,
            status=status,
            output=output,
            error=error,
            latency_ms=latency_ms,
            fallback_used=fallback_used,
            depth=depth,
        )

    # -------------------------------------------------------------------------
    # visited set 管理
    # -------------------------------------------------------------------------
    def clear_visited(self) -> None:
        """清空 visited set（用于新一轮调度会话开始）。"""
        self._visited.clear()

    def get_visited(self) -> set[str]:
        """返回 visited set 的副本（避免外部修改内部状态）。

        Returns:
            已调度的 agent_id 字符串集合副本
        """
        return set(self._visited)

    def is_visited(self, agent_id: str) -> bool:
        """检查 agent_id 是否已在 visited set 中。

        Args:
            agent_id: 路由键

        Returns:
            True 表示已调度过（再次调度会触发 loop_detected）
        """
        return agent_id in self._visited


__all__ = [
    "FailSafeConfig",
    "DispatchResult",
    "FailSafeDispatcher",
    "PROTECTED_WORKERS",
    "is_worker_protected",
]
