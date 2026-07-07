"""Task G.2 验证：统一 JSON 日志 + 全局异常处理。

测试内容：
1. JSONFormatter 产出合法 JSON 且包含必需字段
2. setup_logging 配置 root logger（级别 + handler + 幂等）
3. LOG_LEVEL 环境变量控制日志级别（Settings 读取 + setup_logging 应用）
4. 全局异常处理器返回 500 JSON
"""
from __future__ import annotations

import io
import json
import logging
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

# 确保能导入 app 包（backend/ 在 sys.path）
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.logging_config import JSONFormatter, setup_logging


class TestJSONFormatter(unittest.TestCase):
    """JSONFormatter 产出合法 JSON。"""

    def setUp(self) -> None:
        self.logger = logging.getLogger("test.json_formatter")
        self.logger.setLevel(logging.DEBUG)

    def _capture_record(self, level: int, msg: str, exc_info=None) -> dict:
        """构造 LogRecord 并用 JSONFormatter 格式化，返回解析后的 dict。"""
        record = logging.LogRecord(
            name=self.logger.name,
            level=level,
            pathname=__file__,
            lineno=42,
            msg=msg,
            args=(),
            exc_info=exc_info,
        )
        formatter = JSONFormatter()
        output = formatter.format(record)
        return json.loads(output)

    def test_produces_valid_json(self) -> None:
        parsed = self._capture_record(logging.INFO, "hello world")
        self.assertIsInstance(parsed, dict)

    def test_contains_required_fields(self) -> None:
        parsed = self._capture_record(logging.WARNING, "warn msg")
        for field in ("timestamp", "level", "logger", "message", "module", "line"):
            self.assertIn(field, parsed, f"缺少字段 {field}")
        self.assertEqual(parsed["level"], "WARNING")
        self.assertEqual(parsed["message"], "warn msg")
        self.assertEqual(parsed["logger"], "test.json_formatter")
        self.assertEqual(parsed["line"], 42)

    def test_exception_field_present_when_exc_info(self) -> None:
        try:
            raise ValueError("boom")
        except ValueError:
            import sys
            exc_info = sys.exc_info()
        parsed = self._capture_record(logging.ERROR, "with exc", exc_info=exc_info)
        self.assertIn("exception", parsed)
        self.assertIn("ValueError", parsed["exception"])
        self.assertIn("boom", parsed["exception"])

    def test_no_exception_field_when_no_exc_info(self) -> None:
        parsed = self._capture_record(logging.INFO, "no exc")
        self.assertNotIn("exception", parsed)


class TestSetupLogging(unittest.TestCase):
    """setup_logging 配置 root logger。"""

    def setUp(self) -> None:
        # 保存原 handler 列表，测试后恢复
        self._saved_handlers = logging.getLogger().handlers[:]
        self._saved_level = logging.getLogger().level

    def tearDown(self) -> None:
        root = logging.getLogger()
        root.handlers = self._saved_handlers
        root.setLevel(self._saved_level)

    def test_configures_root_logger_level(self) -> None:
        setup_logging(level="DEBUG", json_output=True)
        self.assertEqual(logging.getLogger().level, logging.DEBUG)

    def test_invalid_level_falls_back_to_info(self) -> None:
        setup_logging(level="NOT_A_LEVEL", json_output=True)
        self.assertEqual(logging.getLogger().level, logging.INFO)

    def test_attaches_stream_handler(self) -> None:
        setup_logging(level="INFO", json_output=True)
        handlers = [h for h in logging.getLogger().handlers if isinstance(h, logging.StreamHandler)]
        self.assertGreaterEqual(len(handlers), 1)

    def test_json_formatter_attached_when_json_output(self) -> None:
        setup_logging(level="INFO", json_output=True)
        handlers = [h for h in logging.getLogger().handlers if isinstance(h, logging.StreamHandler)]
        self.assertTrue(any(isinstance(h.formatter, JSONFormatter) for h in handlers))

    def test_idempotent_no_duplicate_handlers(self) -> None:
        setup_logging(level="INFO", json_output=True)
        setup_logging(level="INFO", json_output=True)
        setup_logging(level="INFO", json_output=True)
        stream_handlers = [h for h in logging.getLogger().handlers if isinstance(h, logging.StreamHandler)]
        # 幂等：多次调用只保留一个 StreamHandler（非测试预存的）
        self.assertEqual(len(stream_handlers), 1)

    def test_emits_valid_json_log_line(self) -> None:
        """端到端：logger.info 输出经 handler 后为合法 JSON。"""
        stream = io.StringIO()
        # 直接构造 handler 写入 StringIO，避免污染 stdout
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JSONFormatter())
        root = logging.getLogger()
        root.handlers = [handler]
        root.setLevel(logging.INFO)
        logging.getLogger("test.emit").info("emit test")
        handler.flush()
        line = stream.getvalue().strip()
        self.assertTrue(line, "应有日志输出")
        parsed = json.loads(line)  # 不抛异常即合法 JSON
        self.assertEqual(parsed["message"], "emit test")


class TestLogLevelEnvVar(unittest.TestCase):
    """LOG_LEVEL 环境变量控制日志级别。"""

    def test_settings_reads_log_level_default(self) -> None:
        from app.config import Settings
        os.environ.pop("LOG_LEVEL", None)
        s = Settings()
        self.assertEqual(s.LOG_LEVEL, "INFO")

    def test_settings_reads_log_level_from_env(self) -> None:
        # Settings 类属性在类定义时通过 os.getenv 读取，需 reload 模块才能反映新 env
        import importlib
        import app.config as config_module
        os.environ["LOG_LEVEL"] = "DEBUG"
        try:
            importlib.reload(config_module)
            self.assertEqual(config_module.Settings.LOG_LEVEL, "DEBUG")
        finally:
            os.environ.pop("LOG_LEVEL", None)
            importlib.reload(config_module)

    def test_log_json_default_true(self) -> None:
        from app.config import Settings
        os.environ.pop("LOG_JSON", None)
        self.assertTrue(Settings.LOG_JSON)

    def test_log_json_false_from_env(self) -> None:
        # 同上：需 reload 让类定义重新求值 os.getenv
        import importlib
        import app.config as config_module
        os.environ["LOG_JSON"] = "false"
        try:
            importlib.reload(config_module)
            self.assertFalse(config_module.Settings.LOG_JSON)
        finally:
            os.environ.pop("LOG_JSON", None)
            importlib.reload(config_module)

    def test_setup_logging_applies_settings_level(self) -> None:
        """Settings.LOG_LEVEL 透传给 setup_logging 后实际生效。"""
        saved_handlers = logging.getLogger().handlers[:]
        saved_level = logging.getLogger().level
        try:
            setup_logging(level="ERROR", json_output=False)
            self.assertEqual(logging.getLogger().level, logging.ERROR)
            self.assertFalse(
                any(isinstance(h.formatter, JSONFormatter) for h in logging.getLogger().handlers)
            )
        finally:
            logging.getLogger().handlers = saved_handlers
            logging.getLogger().setLevel(saved_level)


class TestGlobalExceptionHandler(unittest.TestCase):
    """全局异常处理器返回 500 JSON。"""

    @classmethod
    def setUpClass(cls) -> None:
        # 导入 app.main 会触发 setup_logging 与 LLM 客户端初始化（占位 key，无网络请求）
        from app.main import app, global_exception_handler
        cls.app = app
        # 用 staticmethod 包装避免通过实例访问时被绑定为方法（多传一个 self）
        cls.handler = staticmethod(global_exception_handler)

    def test_handler_returns_500_json(self) -> None:
        """直接调用 handler，验证返回 JSONResponse 500。"""
        import asyncio
        from app.main import global_exception_handler
        request = MagicMock()
        request.method = "GET"
        request.url.path = "/api/test"
        exc = RuntimeError("simulated boom")
        response = asyncio.run(global_exception_handler(request, exc))
        self.assertEqual(response.status_code, 500)
        body = json.loads(response.body.decode("utf-8"))
        self.assertEqual(body["error"], "Internal server error")
        self.assertIn("simulated boom", body["detail"])

    def test_handler_registered_on_app(self) -> None:
        """验证 Exception 处理器已注册到 app。"""
        # FastAPI 把 exception_handler 存入 app.exception_handlers
        self.assertIn(Exception, self.app.exception_handlers)

    def test_end_to_end_via_testclient(self) -> None:
        """通过 TestClient 触发未捕获异常，验证 500 JSON 响应。"""
        from fastapi.testclient import TestClient

        # 临时挂一个必定抛异常的路由
        @self.app.get("/__test_g2_raise__")
        def _raise():  # type: ignore[no-untyped-def]
            raise RuntimeError("e2e boom")

        try:
            client = TestClient(self.app, raise_server_exceptions=False)
            resp = client.get("/__test_g2_raise__")
            self.assertEqual(resp.status_code, 500)
            body = resp.json()
            self.assertEqual(body["error"], "Internal server error")
            self.assertIn("e2e boom", body["detail"])
        finally:
            # 清理临时路由：FastAPI 路由存于 app.router.routes（可变列表）
            self.app.router.routes = [
                r for r in self.app.router.routes
                if getattr(r, "path", "") != "/__test_g2_raise__"
            ]


if __name__ == "__main__":
    unittest.main()
