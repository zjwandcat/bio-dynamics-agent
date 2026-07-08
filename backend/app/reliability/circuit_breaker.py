# BioDynamics Agent v4 - Circuit Breaker（Phase 5 / Reliability Engineering）
#
# 断路器：连续失败超过阈值时断开（OPEN），一段时间后半开（HALF_OPEN）试探，
# 试探成功则恢复（CLOSED），失败则重新断开。防止对故障服务持续打流。
#
# 设计原则：
# 1. 三态：CLOSED（正常）/ OPEN（熔断，快速失败）/ HALF_OPEN（试探）
# 2. 线程安全：使用 threading.RLock 保护状态变更
# 3. OPEN 状态下调用直接抛出 CircuitBreakerOpenError，不执行 func
# 4. HALF_OPEN 状态下允许试探调用：成功达 success_threshold → CLOSED，
#    失败 → 重新 OPEN
# 5. recovery_timeout 到期后 OPEN 转 HALF_OPEN（懒触发，下次调用时检查）
# 6. 函数执行不在锁内，避免长时间持锁阻塞其他调用方
#
# 参考：Martin Fowler CircuitBreaker 模式

from __future__ import annotations

import functools
import logging
import threading
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)


class CircuitBreakerOpenError(RuntimeError):
    """断路器处于 OPEN 状态时抛出，表示调用被快速拒绝。

    Attributes:
        name: 断路器名称
        reset_at: 预计进入 HALF_OPEN 的 Unix 时间戳
    """

    def __init__(self, name: str, reset_at: float) -> None:
        self.name = name
        self.reset_at = reset_at
        remaining = max(0.0, reset_at - time.time())
        super().__init__(
            f"断路器 [{name}] 处于 OPEN 状态，{remaining:.1f}s 后进入 HALF_OPEN 试探"
        )


class CircuitBreaker:
    """断路器：连续失败超过阈值时断开，一段时间后半开试探。

    用法（直接调用）::

        cb = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
        try:
            result = cb.call(risky_api, arg1)
        except CircuitBreakerOpenError:
            # 快速失败，走降级路径
            ...

    用法（装饰器）::

        @CircuitBreaker(failure_threshold=5)
        def fetch(url): ...

    Args:
        failure_threshold: 连续失败次数阈值，达到后熔断，默认 5
        recovery_timeout: 熔断后进入 HALF_OPEN 的等待秒数，默认 60
        success_threshold: HALF_OPEN 状态下连续成功次数后恢复 CLOSED，默认 1
        name: 断路器名称（用于日志与错误信息），默认自动生成
        expected_exception: 视为失败的异常类型元组，默认 (Exception,)
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60,
        success_threshold: int = 1,
        name: str | None = None,
        expected_exception: tuple[type[BaseException], ...] = (Exception,),
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold 必须 >= 1")
        if success_threshold < 1:
            raise ValueError("success_threshold 必须 >= 1")
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.name = name or f"cb-{id(self)}"
        self.expected_exception = expected_exception

        self._state = self.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._opened_at = 0.0  # 进入 OPEN 的时间戳
        self._lock = threading.RLock()

    @property
    def state(self) -> str:
        """当前断路器状态（读取时会懒触发 OPEN→HALF_OPEN 转换）。"""
        with self._lock:
            self._maybe_transition_to_half_open()
            return self._state

    @property
    def failure_count(self) -> int:
        """当前连续失败计数。"""
        with self._lock:
            return self._failure_count

    # ------------------------------------------------------------------
    # 状态转换内部方法（调用方需持锁）
    # ------------------------------------------------------------------
    def _maybe_transition_to_half_open(self) -> None:
        """若 OPEN 状态已超过 recovery_timeout，转为 HALF_OPEN。

        必须在持锁状态下调用。
        """
        if self._state == self.OPEN:
            if time.time() - self._opened_at >= self.recovery_timeout:
                self._state = self.HALF_OPEN
                self._success_count = 0
                logger.info(
                    "CircuitBreaker [%s]: OPEN → HALF_OPEN（试探）",
                    self.name,
                )

    # ------------------------------------------------------------------
    # 主入口：call
    # ------------------------------------------------------------------
    def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """通过断路器调用函数。

        Args:
            func: 待执行的可调用对象
            *args / **kwargs: 传给 func 的参数

        Returns:
            func 的返回值

        Raises:
            CircuitBreakerOpenError: 断路器 OPEN 时
            原异常: func 抛出的属于 expected_exception 的异常
        """
        with self._lock:
            self._maybe_transition_to_half_open()
            if self._state == self.OPEN:
                raise CircuitBreakerOpenError(
                    self.name,
                    self._opened_at + self.recovery_timeout,
                )

        # 执行函数（不在锁内，避免长时间持锁阻塞其他调用方）
        try:
            result = func(*args, **kwargs)
        except self.expected_exception as exc:
            self._on_failure()
            raise
        else:
            self._on_success()
            return result

    def _on_success(self) -> None:
        with self._lock:
            if self._state == self.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.success_threshold:
                    self._state = self.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    logger.info(
                        "CircuitBreaker [%s]: HALF_OPEN → CLOSED（恢复）",
                        self.name,
                    )
            elif self._state == self.CLOSED:
                # 正常状态下成功，重置失败计数
                self._failure_count = 0

    def _on_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            if self._state == self.HALF_OPEN:
                # 试探失败，重新断开
                self._state = self.OPEN
                self._opened_at = time.time()
                self._success_count = 0
                logger.warning(
                    "CircuitBreaker [%s]: HALF_OPEN → OPEN（试探失败）",
                    self.name,
                )
            elif self._state == self.CLOSED:
                if self._failure_count >= self.failure_threshold:
                    self._state = self.OPEN
                    self._opened_at = time.time()
                    logger.warning(
                        "CircuitBreaker [%s]: CLOSED → OPEN（连续失败 %d 次）",
                        self.name,
                        self._failure_count,
                    )

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """装饰器用法。"""

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return self.call(func, *args, **kwargs)

        return wrapper

    def reset(self) -> None:
        """手动重置断路器到 CLOSED 状态（用于运维干预或测试）。"""
        with self._lock:
            self._state = self.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._opened_at = 0.0
            logger.info("CircuitBreaker [%s]: 手动重置为 CLOSED", self.name)


__all__ = ["CircuitBreaker", "CircuitBreakerOpenError"]
