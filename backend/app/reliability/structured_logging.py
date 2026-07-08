# BioDynamics Agent v4 - 结构化日志（Phase 5 / Reliability Engineering）
#
# 提供 JSON 结构化日志，含 trace_id / span_id / timestamp / level / message / context。
# 基于 contextvars 实现跨线程/异步的 trace 传播，便于分布式追踪与日志聚合。
#
# 设计原则：
# 1. 与现有 logging_config.setup_logging 兼容：底层复用标准 logging 模块，
#    但通过专用 logger（bd.structured，propagate=False）独立输出 JSON 行，
#    避免与 root logger 的 JSONFormatter 双重包装
# 2. trace_id / span_id 通过 contextvars 自动传播，无需手动透传
# 3. 每条日志输出单行 JSON，字段固定顺序便于 ELK / Loki 解析
# 4. 零依赖：仅使用标准库（json / logging / uuid / contextvars / datetime）
#
# 使用方式：
#   from app.reliability.structured_logging import StructuredLogger, trace_context
#   logger = StructuredLogger(__name__)
#   with trace_context("req-123"):       # 设置 trace_id
#       logger.info("processing", user_id="u1")  # 自动携带 trace_id + 新 span_id

from __future__ import annotations

import json
import logging
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

# trace_id / span_id 通过 contextvars 传播（async-safe；线程间需 copy_context，
# threading.Thread 默认会复制当前 context，故亦可正常传播）
_trace_id_var: ContextVar[str] = ContextVar("bd_trace_id", default="")
_span_id_var: ContextVar[str] = ContextVar("bd_span_id", default="")


def _new_id() -> str:
    """生成 16 位十六进制 ID（uuid4 前 16 位，足够区分且短小）。"""
    return uuid.uuid4().hex[:16]


def get_trace_id() -> str:
    """获取当前上下文的 trace_id（未设置时返回空串）。"""
    return _trace_id_var.get()


def get_span_id() -> str:
    """获取当前上下文的 span_id（未设置时返回空串）。"""
    return _span_id_var.get()


def set_trace_id(trace_id: str) -> None:
    """显式设置当前上下文的 trace_id。"""
    _trace_id_var.set(trace_id)


def set_span_id(span_id: str) -> None:
    """显式设置当前上下文的 span_id。"""
    _span_id_var.set(span_id)


class trace_context:  # noqa: N801 - 上下文管理器用小写蛇形更自然
    """trace 上下文管理器：进入时设置 trace_id（可选 span_id），退出时恢复。

    用法::

        with trace_context("req-abc"):
            # 此范围内所有 StructuredLogger 自动携带 trace_id="req-abc"
            logger.info("start")
            # 每条日志会自动生成新的 span_id（除非显式指定）
    """

    def __init__(self, trace_id: str | None = None, span_id: str | None = None) -> None:
        self._trace_id = trace_id or _new_id()
        self._span_id = span_id
        self._prev_trace_token: Any = None
        self._prev_span_token: Any = None

    def __enter__(self) -> "trace_context":
        self._prev_trace_token = _trace_id_var.set(self._trace_id)
        self._prev_span_token = _span_id_var.set(self._span_id or _new_id())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._prev_trace_token is not None:
            _trace_id_var.reset(self._prev_trace_token)
        if self._prev_span_token is not None:
            _span_id_var.reset(self._prev_span_token)

    @property
    def trace_id(self) -> str:
        return self._trace_id


# =============================================================================
# 专用 logger + 自定义 formatter（避免与 root JSONFormatter 双重包装）
# =============================================================================
class _StructuredFormatter(logging.Formatter):
    """将 LogRecord 上的 _bd_payload 字段序列化为单行 JSON。"""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] | None = getattr(record, "_bd_payload", None)
        if payload is None:
            # 兜底：未携带 payload 时构造最小结构
            payload = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
        return json.dumps(payload, ensure_ascii=False, default=str)


# 模块级专用 handler / logger：propagate=False 防止向 root 冒泡导致重复输出
_structured_handler = logging.StreamHandler(sys.stdout)
_structured_handler.setFormatter(_StructuredFormatter())

_dedicated_logger = logging.getLogger("bd.structured")
_dedicated_logger.setLevel(logging.INFO)
_dedicated_logger.propagate = False
if not _dedicated_logger.handlers:
    _dedicated_logger.addHandler(_structured_handler)


def _safe_serialize(obj: Any) -> Any:
    """安全序列化：将不可 JSON 序列化的对象转为字符串，避免日志记录本身抛异常。"""
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        if isinstance(obj, dict):
            return {str(k): _safe_serialize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_safe_serialize(v) for v in obj]
        return repr(obj)


class StructuredLogger:
    """结构化日志记录器：输出含 trace_id / span_id 的 JSON 日志。

    底层委托给专用 logger ``bd.structured``（propagate=False），通过自定义
    formatter 直接输出结构化 JSON 行，不经过 root logger 的 JSONFormatter，
    从而避免字段双重包装。

    Attributes:
        name: logger 名称（通常为 __name__）
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._logger = _dedicated_logger

    def log(self, level: str, message: str, **context: Any) -> None:
        """记录结构化日志。

        Args:
            level: 日志级别（DEBUG/INFO/WARNING/ERROR/CRITICAL），不区分大小写
            message: 日志消息
            **context: 任意键值对上下文，序列化到 JSON 的 context 字段
        """
        level_upper = level.upper()
        level_num = getattr(logging, level_upper, logging.INFO)
        if not self._logger.isEnabledFor(level_num):
            return
        trace_id = _trace_id_var.get()
        span_id = _span_id_var.get() or _new_id()
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level_upper,
            "logger": self.name,
            "message": message,
            "trace_id": trace_id,
            "span_id": span_id,
        }
        if context:
            payload["context"] = _safe_serialize(context)
        # 通过 extra 注入 payload，formatter 读取 _bd_payload 输出
        self._logger.log(level_num, message, extra={"_bd_payload": payload})

    # ------------------------------------------------------------------
    # 便捷方法
    # ------------------------------------------------------------------
    def debug(self, message: str, **context: Any) -> None:
        self.log("DEBUG", message, **context)

    def info(self, message: str, **context: Any) -> None:
        self.log("INFO", message, **context)

    def warning(self, message: str, **context: Any) -> None:
        self.log("WARNING", message, **context)

    def error(self, message: str, **context: Any) -> None:
        self.log("ERROR", message, **context)

    def critical(self, message: str, **context: Any) -> None:
        self.log("CRITICAL", message, **context)

    def exception(self, message: str, **context: Any) -> None:
        """记录异常日志（含当前异常的 traceback 到 context）。

        必须在 except 块内调用，否则 traceback 为 "NoneType: None"。
        """
        import traceback

        tb = traceback.format_exc()
        ctx = dict(context)
        ctx["traceback"] = tb
        self.log("ERROR", message, **ctx)


__all__ = [
    "StructuredLogger",
    "trace_context",
    "get_trace_id",
    "get_span_id",
    "set_trace_id",
    "set_span_id",
]
