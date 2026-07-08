# BioDynamics Agent v4 - Retry 机制（Phase 5 / Reliability Engineering）
#
# 重试策略：指数退避 + 最大尝试次数 + 可选抖动（jitter）。
# 用于包裹瞬时失败的外部调用（网络请求、限流 API 等）。
#
# 设计原则：
# 1. 指数退避：delay = min(base_delay * 2^(attempt-1), max_delay)
# 2. 抖动：默认启用 full jitter（随机 [0, delay)），避免惊群效应
# 3. 可配置重试异常类型，默认仅重试 Exception
# 4. 最后一次失败后抛出原异常，不吞异常
# 5. 每次重试记录 warning 日志，含尝试次数与下次延迟
#
# 参考：AWS Architecture Blog "Exponential Backoff And Jitter"

from __future__ import annotations

import functools
import logging
import random
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)


class RetryPolicy:
    """重试策略：指数退避 + 最大尝试次数。

    用法（直接执行）::

        policy = RetryPolicy(max_attempts=3, base_delay=1.0)
        result = policy.execute(risky_call, arg1, kw1=v1)

    用法（装饰器）::

        retry = RetryPolicy(max_attempts=5)
        @retry
        def fetch(url): ...

    Args:
        max_attempts: 最大尝试次数（含首次），默认 3
        base_delay: 首次重试基础延迟（秒），默认 1.0
        max_delay: 单次延迟上限（秒），默认 60.0
        jitter: 是否启用抖动（full jitter），默认 True
        retry_on: 重试的异常类型元组，默认 (Exception,)
    """

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter: bool = True,
        retry_on: tuple[type[BaseException], ...] = (Exception,),
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts 必须 >= 1")
        if base_delay < 0:
            raise ValueError("base_delay 必须 >= 0")
        if max_delay < 0:
            raise ValueError("max_delay 必须 >= 0")
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter
        self.retry_on = retry_on

    def _compute_delay(self, attempt: int) -> float:
        """计算第 attempt 次重试前的延迟（attempt 从 1 开始，表示首次失败后等待）。

        Returns:
            实际睡眠秒数（已应用 jitter 与上限）
        """
        # 指数退避：base * 2^(attempt-1)
        delay = self.base_delay * (2 ** (attempt - 1))
        delay = min(delay, self.max_delay)
        if self.jitter and delay > 0:
            # full jitter: 在 [0, delay) 区间随机，避免多客户端同步重试
            delay = random.uniform(0, delay)
        return delay

    def execute(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """执行函数，失败时按策略重试。

        Args:
            func: 待执行的可调用对象
            *args / **kwargs: 传给 func 的参数

        Returns:
            func 的返回值

        Raises:
            最后一次尝试失败时抛出原异常
        """
        last_exc: BaseException | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                return func(*args, **kwargs)
            except self.retry_on as exc:
                last_exc = exc
                if attempt >= self.max_attempts:
                    logger.warning(
                        "RetryPolicy: 第 %d/%d 次尝试失败（已达上限，放弃）: %s: %s",
                        attempt,
                        self.max_attempts,
                        type(exc).__name__,
                        exc,
                    )
                    raise
                delay = self._compute_delay(attempt)
                logger.warning(
                    "RetryPolicy: 第 %d/%d 次尝试失败，%.2fs 后重试: %s: %s",
                    attempt,
                    self.max_attempts,
                    delay,
                    type(exc).__name__,
                    exc,
                )
                time.sleep(delay)
        # 理论不可达（循环内已 raise 或 return），保险起见抛出最后一次异常
        assert last_exc is not None
        raise last_exc

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """装饰器用法：将本策略应用到被装饰函数。"""

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return self.execute(func, *args, **kwargs)

        return wrapper


__all__ = ["RetryPolicy"]
