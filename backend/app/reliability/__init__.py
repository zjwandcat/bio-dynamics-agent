# BioDynamics Agent v4 - Reliability Engineering 包（Phase 5）
#
# 汇总可靠性工程模块：健康检查 / 错误边界 / 重试 / 断路器 / 超时 / 结构化日志。
# 各模块相互独立，可单独导入使用，也可组合构成弹性调用链：
#
#   from app.reliability import (
#       HealthChecker, ErrorBoundary, RetryPolicy,
#       CircuitBreaker, Timeout, StructuredLogger, trace_context,
#   )
#
# 典型组合：结构化日志 + 错误边界 + 重试 + 断路器 + 超时
#
#   cb = CircuitBreaker(failure_threshold=5)
#   retry = RetryPolicy(max_attempts=3)
#   timeout = Timeout(seconds=10)
#   boundary = ErrorBoundary(fallback_value=None)
#
#   @boundary
#   @retry
#   @cb
#   @timeout
#   def call_external(url): ...

from app.reliability.circuit_breaker import CircuitBreaker, CircuitBreakerOpenError
from app.reliability.error_boundary import ErrorBoundary
from app.reliability.health_check import HealthChecker
from app.reliability.retry import RetryPolicy
from app.reliability.structured_logging import (
    StructuredLogger,
    get_span_id,
    get_trace_id,
    set_span_id,
    set_trace_id,
    trace_context,
)
from app.reliability.timeout import Timeout

__all__ = [
    "HealthChecker",
    "ErrorBoundary",
    "RetryPolicy",
    "CircuitBreaker",
    "CircuitBreakerOpenError",
    "Timeout",
    "StructuredLogger",
    "trace_context",
    "get_trace_id",
    "get_span_id",
    "set_trace_id",
    "set_span_id",
]
