# BioDynamics Agent v4 - Timeout 机制（Phase 5 / Reliability Engineering）
#
# 超时控制：在指定时间内完成函数调用，超时抛出 TimeoutError。
#
# 平台兼容（try-import 降级）：
# - 优先尝试 signal.SIGALRM（仅 Unix 主线程可用，精确中断 CPU 密集/纯 Python 代码）
# - 不可用时降级到 threading.Thread + join(timeout)（Windows / 非主线程）
#   注意：threading 方案无法强制终止线程，超时后函数仍可能在后台 daemon 线程运行，
#   但主流程不再等待其结果（与 agent_orchestration/fail_safe.py 一致的策略）。
#
# 设计原则：
# 1. 跨平台：自动选择 signal 或 threading 实现，调用方无感知
# 2. signal 路径：超时立即抛出，可中断；退出时恢复原 signal handler 与 itimer
# 3. threading 路径：daemon 线程，超时后主流程抛出 TimeoutError，结果被丢弃
# 4. 保留原函数返回值与异常（threading 路径下函数内部异常回传主线程重新抛出）

from __future__ import annotations

import functools
import logging
import signal
import threading
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)

# 检测 signal.SIGALRM 是否可用（仅 Unix；Windows 无此属性）
try:
    _SIGALRM_AVAILABLE = hasattr(signal, "SIGALRM") and hasattr(signal, "setitimer")
except Exception:  # noqa: BLE001 - 极端环境下信号检测本身可能异常
    _SIGALRM_AVAILABLE = False


class Timeout:
    """超时控制：在指定时间内完成函数调用。

    用法（直接执行）::

        to = Timeout(seconds=30)
        result = to.execute(long_running_func, arg1)

    用法（装饰器）::

        @Timeout(seconds=10)
        def fetch(url): ...

    Args:
        seconds: 超时秒数，默认 30
    """

    def __init__(self, seconds: float = 30) -> None:
        if seconds <= 0:
            raise ValueError("seconds 必须 > 0")
        self.seconds = seconds

    def execute(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """执行函数，超时抛出 TimeoutError。

        Args:
            func: 待执行的可调用对象
            *args / **kwargs: 传给 func 的参数

        Returns:
            func 的返回值

        Raises:
            TimeoutError: 超时（内置 TimeoutError，便于上层统一捕获）
            原异常: func 内部抛出的异常
        """
        func_name = getattr(func, "__name__", "unknown")
        # Unix 主线程优先使用 signal（可中断，超时立即生效）
        if (
            _SIGALRM_AVAILABLE
            and threading.current_thread() is threading.main_thread()
        ):
            return self._execute_with_signal(func, args, kwargs, func_name)
        return self._execute_with_thread(func, args, kwargs, func_name)

    # ------------------------------------------------------------------
    # signal 实现（Unix 主线程）
    # ------------------------------------------------------------------
    def _execute_with_signal(
        self,
        func: Callable[..., Any],
        args: tuple,
        kwargs: dict,
        func_name: str,
    ) -> Any:
        """使用 signal.SIGALRM + setitimer 实现超时（Unix 主线程）。

        超时时 SIGALRM 触发 _handler 抛出 TimeoutError，中断 func 执行。
        退出时恢复原有的 signal handler 与 itimer 配置。
        """

        def _handler(signum: int, frame: Any) -> None:  # noqa: ARG001
            raise TimeoutError(
                f"函数 [{func_name}] 执行超时（{self.seconds}s）"
            )

        old_handler = signal.signal(signal.SIGALRM, _handler)
        # setitimer 返回 (previous_value_seconds, previous_interval_seconds)
        old_timer = signal.setitimer(signal.ITIMER_REAL, float(self.seconds))
        try:
            return func(*args, **kwargs)
        finally:
            # 恢复原有 itimer（value, interval），无则置 0 取消
            signal.setitimer(
                signal.ITIMER_REAL,
                old_timer[0] or 0.0,
                old_timer[1] or 0.0,
            )
            signal.signal(signal.SIGALRM, old_handler)

    # ------------------------------------------------------------------
    # threading 实现（Windows / 非主线程）
    # ------------------------------------------------------------------
    def _execute_with_thread(
        self,
        func: Callable[..., Any],
        args: tuple,
        kwargs: dict,
        func_name: str,
    ) -> Any:
        """使用 threading.Thread + join 实现超时（Windows / 非主线程）。

        无法强制终止线程：超时后 daemon 线程仍在后台运行直到进程退出，
        但主流程立即抛出 TimeoutError，不再等待结果。
        函数内部抛出的异常会被回传到主线程重新抛出（超时则不回传）。
        """
        result_holder: dict[str, Any] = {"value": None, "exc": None}

        def _run() -> None:
            try:
                result_holder["value"] = func(*args, **kwargs)
            except BaseException as exc:  # noqa: BLE001 - 捕获所有异常回传主线程
                result_holder["exc"] = exc

        thread = threading.Thread(
            target=_run,
            name=f"Timeout-{func_name}",
            daemon=True,
        )
        thread.start()
        thread.join(timeout=self.seconds)

        if thread.is_alive():
            # 超时：线程仍在运行（daemon 线程会被主进程回收，不强制 kill）
            logger.warning(
                "Timeout: 函数 [%s] 超时（%ss），daemon 线程仍在后台运行",
                func_name,
                self.seconds,
            )
            raise TimeoutError(
                f"函数 [{func_name}] 执行超时（{self.seconds}s）"
            )

        # 线程已结束，回传内部异常或返回值
        if result_holder["exc"] is not None:
            raise result_holder["exc"]
        return result_holder["value"]

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """装饰器用法。"""

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return self.execute(func, *args, **kwargs)

        return wrapper


__all__ = ["Timeout"]
