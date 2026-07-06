# BioDynamics Agent - 生产级监控指标
# 对应深度审核报告 §3.3 生产级监控（Metrics）
#
# 设计原则：
# 1. 零依赖优先：默认使用轻量结构化日志记录器，无需安装 prometheus_client
# 2. 可选升级：METRICS_BACKEND=prometheus 时自动切换到 prometheus_client
# 3. 关键指标三件套：
#    - RAG 命中率 (biodynamics_rag_hit_rate)
#    - 各 Worker 执行耗时 (biodynamics_worker_duration_seconds)
#    - 沙箱执行成功率 (biodynamics_sandbox_success_total)
# 4. 禁止散落 print 语句，所有关键事件通过结构化日志记录

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# 抽象指标后端
# -----------------------------------------------------------------------------
class MetricsBackend:
    """指标后端抽象基类。"""

    def record_rag_hit(self, hit_rate: float, found: int, total: int) -> None:
        raise NotImplementedError

    def record_worker_duration(self, worker_name: str, duration_seconds: float, success: bool) -> None:
        raise NotImplementedError

    def record_sandbox_execution(self, success: bool, error_class: str = "none", duration_seconds: float = 0.0) -> None:
        raise NotImplementedError

    def record_validation(self, method: str, passed: bool, structural_confidence: float | None = None) -> None:
        raise NotImplementedError

    def record_degradation(self, mode: str) -> None:
        raise NotImplementedError

    def snapshot(self) -> dict[str, Any]:
        """返回当前所有指标的快照（供 /metrics 端点或日志输出）。"""
        raise NotImplementedError


# -----------------------------------------------------------------------------
# 日志后端（默认，零依赖）
# -----------------------------------------------------------------------------
class LogMetricsBackend(MetricsBackend):
    """轻量结构化日志记录器后端。

    将指标写入 data/metrics/metrics.jsonl，每行一条 JSON。
    同时维护内存中的累计统计供 /metrics 端点查询。
    """

    def __init__(self, log_dir: str) -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self._log_dir / "metrics.jsonl"
        self._lock = threading.Lock()
        # 内存累计统计
        self._rag_hits: list[float] = []
        self._worker_duration: dict[str, list[float]] = defaultdict(list)
        self._worker_success: dict[str, int] = defaultdict(int)
        self._worker_failure: dict[str, int] = defaultdict(int)
        self._sandbox_success: int = 0
        self._sandbox_failure: int = 0
        self._sandbox_errors: dict[str, int] = defaultdict(int)
        self._validation_pass: int = 0
        self._validation_fail: int = 0
        self._degradation_counts: dict[str, int] = defaultdict(int)

    def _write(self, record: dict[str, Any]) -> None:
        record["ts"] = time.time()
        with self._lock:
            # 写入 JSONL 文件
            try:
                with open(self._log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except OSError as exc:
                logger.warning("写入指标日志失败: %s", exc)

    def record_rag_hit(self, hit_rate: float, found: int, total: int) -> None:
        with self._lock:
            self._rag_hits.append(hit_rate)
        self._write({
            "metric": "biodynamics_rag_hit_rate",
            "value": hit_rate,
            "found": found,
            "total": total,
        })

    def record_worker_duration(self, worker_name: str, duration_seconds: float, success: bool) -> None:
        with self._lock:
            self._worker_duration[worker_name].append(duration_seconds)
            if success:
                self._worker_success[worker_name] += 1
            else:
                self._worker_failure[worker_name] += 1
        self._write({
            "metric": "biodynamics_worker_duration_seconds",
            "worker": worker_name,
            "duration": round(duration_seconds, 4),
            "success": success,
        })

    def record_sandbox_execution(self, success: bool, error_class: str = "none", duration_seconds: float = 0.0) -> None:
        with self._lock:
            if success:
                self._sandbox_success += 1
            else:
                self._sandbox_failure += 1
                if error_class and error_class != "none":
                    self._sandbox_errors[error_class] += 1
        self._write({
            "metric": "biodynamics_sandbox_success_total",
            "success": success,
            "error_class": error_class,
            "duration": round(duration_seconds, 4),
        })

    def record_validation(self, method: str, passed: bool, structural_confidence: float | None = None) -> None:
        with self._lock:
            if passed:
                self._validation_pass += 1
            else:
                self._validation_fail += 1
        self._write({
            "metric": "biodynamics_validation_result",
            "method": method,
            "passed": passed,
            "structural_confidence": structural_confidence,
        })

    def record_degradation(self, mode: str) -> None:
        with self._lock:
            self._degradation_counts[mode] += 1
        self._write({
            "metric": "biodynamics_degradation_mode",
            "mode": mode,
        })

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            avg_rag_hit = sum(self._rag_hits) / len(self._rag_hits) if self._rag_hits else 0.0
            sandbox_total = self._sandbox_success + self._sandbox_failure
            sandbox_rate = self._sandbox_success / sandbox_total if sandbox_total > 0 else 0.0
            return {
                "rag_hit_rate_avg": round(avg_rag_hit, 4),
                "rag_hit_rate_samples": len(self._rag_hits),
                "worker_duration_avg": {
                    name: round(sum(durations) / len(durations), 4)
                    for name, durations in self._worker_duration.items()
                },
                "worker_success": dict(self._worker_success),
                "worker_failure": dict(self._worker_failure),
                "sandbox_success": self._sandbox_success,
                "sandbox_failure": self._sandbox_failure,
                "sandbox_success_rate": round(sandbox_rate, 4),
                "sandbox_error_breakdown": dict(self._sandbox_errors),
                "validation_pass": self._validation_pass,
                "validation_fail": self._validation_fail,
                "degradation_counts": dict(self._degradation_counts),
            }


# -----------------------------------------------------------------------------
# Prometheus 后端（可选）
# -----------------------------------------------------------------------------
class PrometheusMetricsBackend(MetricsBackend):
    """prometheus_client 后端。需 pip install prometheus_client。"""

    def __init__(self) -> None:
        try:
            import prometheus_client  # type: ignore[import-untyped]
            from prometheus_client import Counter, Gauge, Histogram  # type: ignore[import-untyped]
            self._prom = prometheus_client
            self._rag_hit_gauge = Gauge(
                "biodynamics_rag_hit_rate",
                "RAG parameter hit rate (0.0-1.0)",
            )
            self._worker_duration = Histogram(
                "biodynamics_worker_duration_seconds",
                "Worker execution duration in seconds",
                labelnames=("worker",),
            )
            self._worker_success = Counter(
                "biodynamics_worker_success_total",
                "Worker execution success count",
                labelnames=("worker",),
            )
            self._worker_failure = Counter(
                "biodynamics_worker_failure_total",
                "Worker execution failure count",
                labelnames=("worker",),
            )
            self._sandbox_success = Counter(
                "biodynamics_sandbox_success_total",
                "Sandbox execution success count",
            )
            self._sandbox_failure = Counter(
                "biodynamics_sandbox_failure_total",
                "Sandbox execution failure count",
                labelnames=("error_class",),
            )
            self._validation_pass = Counter(
                "biodynamics_validation_pass_total",
                "SBML validation pass count",
                labelnames=("method",),
            )
            self._validation_fail = Counter(
                "biodynamics_validation_fail_total",
                "SBML validation fail count",
                labelnames=("method",),
            )
            self._degradation = Counter(
                "biodynamics_degradation_total",
                "Degradation mode trigger count",
                labelnames=("mode",),
            )
            self._available = True
        except ImportError:
            logger.warning("prometheus_client 不可用，降级为日志后端")
            self._available = False
            self._fallback = LogMetricsBackend(settings.METRICS_LOG_DIR)

    def record_rag_hit(self, hit_rate: float, found: int, total: int) -> None:
        if not self._available:
            return self._fallback.record_rag_hit(hit_rate, found, total)
        self._rag_hit_gauge.set(hit_rate)

    def record_worker_duration(self, worker_name: str, duration_seconds: float, success: bool) -> None:
        if not self._available:
            return self._fallback.record_worker_duration(worker_name, duration_seconds, success)
        self._worker_duration.labels(worker=worker_name).observe(duration_seconds)
        if success:
            self._worker_success.labels(worker=worker_name).inc()
        else:
            self._worker_failure.labels(worker=worker_name).inc()

    def record_sandbox_execution(self, success: bool, error_class: str = "none", duration_seconds: float = 0.0) -> None:
        if not self._available:
            return self._fallback.record_sandbox_execution(success, error_class, duration_seconds)
        if success:
            self._sandbox_success.inc()
        else:
            self._sandbox_failure.labels(error_class=error_class).inc()

    def record_validation(self, method: str, passed: bool, structural_confidence: float | None = None) -> None:
        if not self._available:
            return self._fallback.record_validation(method, passed, structural_confidence)
        if passed:
            self._validation_pass.labels(method=method).inc()
        else:
            self._validation_fail.labels(method=method).inc()

    def record_degradation(self, mode: str) -> None:
        if not self._available:
            return self._fallback.record_degradation(mode)
        self._degradation.labels(mode=mode).inc()

    def snapshot(self) -> dict[str, Any]:
        if not self._available:
            return self._fallback.snapshot()
        # Prometheus 通常通过 /metrics 端点暴露，这里返回简化快照
        return {"backend": "prometheus", "endpoint": "/metrics"}

    def generate_latest(self) -> bytes:
        """生成 Prometheus 格式的指标数据（供 /metrics 端点）。"""
        if not self._available:
            return b""
        return self._prom.generate_latest()


# -----------------------------------------------------------------------------
# 全局实例与工厂
# -----------------------------------------------------------------------------
_global_backend: MetricsBackend | None = None
_backend_lock = threading.Lock()


def get_metrics() -> MetricsBackend:
    """获取全局指标后端实例。"""
    global _global_backend
    if _global_backend is None:
        with _backend_lock:
            if _global_backend is None:
                backend = settings.METRICS_BACKEND.lower()
                if backend == "prometheus":
                    _global_backend = PrometheusMetricsBackend()
                elif backend == "off":
                    _global_backend = _NullBackend()
                else:
                    _global_backend = LogMetricsBackend(settings.METRICS_LOG_DIR)
                logger.info("监控指标后端已初始化: %s", backend)
    return _global_backend


class _NullBackend(MetricsBackend):
    """空后端（METRICS_BACKEND=off）。"""

    def record_rag_hit(self, hit_rate: float, found: int, total: int) -> None:
        pass

    def record_worker_duration(self, worker_name: str, duration_seconds: float, success: bool) -> None:
        pass

    def record_sandbox_execution(self, success: bool, error_class: str = "none", duration_seconds: float = 0.0) -> None:
        pass

    def record_validation(self, method: str, passed: bool, structural_confidence: float | None = None) -> None:
        pass

    def record_degradation(self, mode: str) -> None:
        pass

    def snapshot(self) -> dict[str, Any]:
        return {"backend": "off"}


# -----------------------------------------------------------------------------
# 便捷上下文管理器：Worker 执行耗时埋点
# -----------------------------------------------------------------------------
class _WorkerTimer:
    """Worker 执行耗时上下文管理器。"""

    def __init__(self, worker_name: str) -> None:
        self.worker_name = worker_name
        self._start = 0.0
        self._success = True

    def __enter__(self) -> "_WorkerTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        duration = time.perf_counter() - self._start
        success = exc_type is None and self._success
        get_metrics().record_worker_duration(self.worker_name, duration, success)

    def mark_failure(self) -> None:
        """显式标记失败（即使无异常）。"""
        self._success = False


def time_worker(worker_name: str) -> _WorkerTimer:
    """便捷函数：with time_worker("worker_rag") as t: ..."""
    return _WorkerTimer(worker_name)
