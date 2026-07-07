# BioDynamics Agent - 统一日志配置模块
# Task G.2：提供 JSON 结构化日志格式化器与全局日志初始化入口。
#
# 设计目标：
# - 统一全后端日志格式为 JSON（便于 ELK / Loki 等日志聚合系统解析）
# - 通过 LOG_LEVEL / LOG_JSON 环境变量控制日志级别与输出格式
# - 兼容现有 logging.getLogger(__name__) 调用，无需修改业务代码
# - 异常信息以 stack trace 形式序列化到 exception 字段，便于离线排查
#
# 使用方式（在 main.py 启动时调用一次）：
#   from app.logging_config import setup_logging
#   setup_logging(level=settings.LOG_LEVEL, json_output=settings.LOG_JSON)

import json
import logging
import sys
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    """JSON 结构化日志格式化器。

    将每条 LogRecord 序列化为单行 JSON，字段包括：
    timestamp / level / logger / message / module / line，并在异常存在时附加
    exception 字段（含完整 stack trace 文本）。
    """

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, ensure_ascii=False)


def setup_logging(level: str = "INFO", json_output: bool = True) -> None:
    """配置 root logger，统一日志格式。

    幂等：重复调用不会叠加 handler（先移除既有 StreamHandler 再添加），
    避免在测试或热重载场景下日志重复输出。

    Args:
        level: 日志级别字符串（DEBUG/INFO/WARNING/ERROR），不区分大小写，
            非法值回退到 INFO。
        json_output: True 使用 JSON 格式化器（生产推荐）；
            False 使用纯文本格式（本地调试更易读）。
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # 幂等：移除已注册的 StreamHandler，避免重复输出
    for handler in list(root_logger.handlers):
        if isinstance(handler, logging.StreamHandler):
            root_logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    if json_output:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
            )
        )
    root_logger.addHandler(handler)
