# BioDynamics Agent v4 - Error Boundary（Phase 5 / Reliability Engineering）
#
# 错误边界：捕获函数抛出的异常，记录结构化日志，返回降级结果。
# 用于包裹不可信/易失败操作（外部 API 调用、SBML 解析、仿真等），
# 防止单点异常向上传播导致整个流水线崩溃。
#
# 设计原则：
# 1. 作为装饰器使用：@ErrorBoundary(fallback_value=...)
# 2. 捕获指定 Exception（含子类），不捕获 BaseException（KeyboardInterrupt 等）
# 3. 记录结构化日志：含函数名 / 异常类型 / 异常消息 / reraise 标记
#    完整 traceback 通过 StructuredLogger.exception 记录
# 4. reraise=True 时记录后重新抛出（用于只记录不吞异常的场景）
# 5. fallback_value 支持工厂函数：传入 callable 时在异常时调用获取降级值
#    （lazy 生成，避免降级值本身在构造时计算）

from __future__ import annotations

import functools
import logging
from typing import Any, Callable

from app.reliability.structured_logging import StructuredLogger

# 结构化日志记录器：错误边界的关键事件（异常捕获）走结构化输出
_slogger = StructuredLogger(__name__)
# 标准日志记录器：用于记录完整 traceback（exc_info=True）
_std_logger = logging.getLogger(__name__)


class ErrorBoundary:
    """错误边界：捕获异常，记录结构化日志，返回降级结果。

    用法（装饰器）::

        @ErrorBoundary(fallback_value={"ok": False})
        def call_external_api(...):
            ...

    用法（直接调用）::

        boundary = ErrorBoundary(fallback_value=None)
        result = boundary.execute(risky_func, arg1, kw1=v1)

    用法（lazy fallback 工厂）::

        @ErrorBoundary(fallback_value=lambda: default_config())
        def load_config(): ...

    Args:
        fallback_value: 降级返回值；若为 callable，则在异常时调用获取降级值
        reraise: True 时记录日志后重新抛出原异常；False（默认）返回降级值
        operation: 可选的操作名称（用于日志），默认从被装饰函数名推断
        exceptions: 捕获的异常类型元组，默认 (Exception,)
    """

    def __init__(
        self,
        fallback_value: Any = None,
        reraise: bool = False,
        operation: str | None = None,
        exceptions: tuple[type[BaseException], ...] = (Exception,),
    ) -> None:
        self.fallback_value = fallback_value
        self.reraise = reraise
        self.operation = operation
        self.exceptions = exceptions

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """装饰器：包裹函数，捕获异常。"""

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return self.execute(func, *args, **kwargs)

        return wrapper

    def execute(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """直接执行函数并应用错误边界保护。

        Args:
            func: 待执行的可调用对象
            *args / **kwargs: 传给 func 的参数

        Returns:
            func 的正常返回值，或异常时的 fallback_value（reraise=True 时不会返回）
        """
        op_name = self.operation or getattr(func, "__name__", "unknown")
        try:
            return func(*args, **kwargs)
        except self.exceptions as exc:
            # 记录结构化日志（含 trace_id / span_id 自动传播）
            _slogger.exception(
                "ErrorBoundary 捕获异常",
                operation=op_name,
                function=op_name,
                exception_type=type(exc).__name__,
                exception_message=str(exc),
                reraise=self.reraise,
            )
            # 标准日志记录完整 traceback（exc_info），便于离线排查
            _std_logger.debug(
                "ErrorBoundary 异常详情 [%s]: %s",
                op_name,
                exc,
                exc_info=True,
            )
            if self.reraise:
                raise
            # fallback_value 为 callable 时 lazy 调用
            if callable(self.fallback_value):
                try:
                    return self.fallback_value()
                except Exception as fb_exc:  # noqa: BLE001
                    _std_logger.error(
                        "ErrorBoundary fallback 工厂抛出异常 [%s]: %s",
                        op_name,
                        fb_exc,
                    )
                    return None
            return self.fallback_value


__all__ = ["ErrorBoundary"]
