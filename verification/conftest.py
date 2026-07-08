"""BioDynamics v4 — Verification Suite 顶层 pytest 配置

本文件为整个 verification/ 目录提供：
  1. 自定义 pytest marker 注册（benchmark / slow / requires_network / requires_roadrunner）
  2. 通用 fixture：reports_dir / cache_dir / rng / pathway_registry
  3. 测试结果收集钩子（pytest_runtest_makereport），把每个用例的
     pass/fail/skip 状态与耗时写入 reports/summary.json，供 dashboard 消费

约束：
  - 仅读取 v4 产物，绝不修改 backend/app/ 下任何文件
  - 所有外部依赖（roadrunner / requests）采用惰性导入，缺失时优雅 skip
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import pytest

# verification/ 根目录
VERIFICATION_ROOT = Path(__file__).parent
# reports/ 输出目录
REPORTS_ROOT = VERIFICATION_ROOT / "reports"

# v4 十条核心通路（与 backend/app/ontology/pathway_registry.py 保持一致）
CORE_PATHWAYS = [
    "EGFR_RTK",
    "MAPK_ERK",
    "PI3K_AKT_mTOR",
    "p53",
    "APOPTOSIS",
    "CELL_CYCLE",
    "JAK_STAT",
    "NF_KB",
    "WNT",
    "TGF_BETA",
]


# --------------------------------------------------------------------------- #
# Marker 注册：避免 pytest 未知 marker 警告
# --------------------------------------------------------------------------- #
def pytest_configure(config: pytest.Config) -> None:
    """注册自定义 marker，并确保 reports 目录存在。"""
    config.addinivalue_line(
        "markers",
        "benchmark: 科学基准 / 回归测试，长时间运行，默认不在快速 CI 中执行",
    )
    config.addinivalue_line(
        "markers",
        "slow: 单用例耗时 > 30s 的慢测试",
    )
    config.addinivalue_line(
        "markers",
        "requires_network: 需要访问外部网络（EBI BioModels / HGNC / UniProt 等）",
    )
    config.addinivalue_line(
        "markers",
        "requires_roadrunner: 需要 libRoadRunner（pip install python-libsbml roadrunner）",
    )
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# 通用 fixture
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def reports_dir() -> Path:
    """返回 reports/ 根目录，确保已创建。"""
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    return REPORTS_ROOT


@pytest.fixture(scope="session")
def cache_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """会话级缓存目录，用于存放下载的 SBML / 临时产物。

    使用 tmp_path_factory 保证不同测试会话互不污染。
    """
    cache = tmp_path_factory.mktemp("v4_cache")
    return cache


@pytest.fixture(scope="session")
def rng() -> random.Random:
    """确定性随机数生成器，保证参数扰动可复现。"""
    return random.Random(20240101)


@pytest.fixture(scope="session")
def pathway_registry() -> list[str]:
    """返回 v4 核心通路清单。"""
    return list(CORE_PATHWAYS)


@pytest.fixture(scope="session")
def metrics_thresholds() -> dict[str, float]:
    """回归测试通用阈值（可被具体用例覆盖）。"""
    return {
        "rmse_max": 15.0,            # RMSE 上界
        "pearson_min": 0.85,         # Pearson 相关系数下界
        "peak_time_tol_min": 2.0,    # 峰值时间容差（分钟）
        "peak_amp_rel_tol": 0.30,    # 峰值高度相对容差 30%
        "auc_rel_tol": 0.25,         # AUC 相对容差 25%
        "steady_state_rel_tol": 0.20,# 稳态相对容差 20%
    }


# --------------------------------------------------------------------------- #
# 结果收集：把每个用例结果追加到 reports/_collected.jsonl
# --------------------------------------------------------------------------- #
@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_makereport(item: pytest.Item, call):
    """捕获每个用例的 outcome / 耗时，写入 reports/_collected.jsonl。"""
    outcome = yield
    report: pytest.TestReport = outcome.get_result()

    # 仅在 call 阶段（实际执行）记录，避免 setup/teardown 重复
    if report.when != "call" and not (report.when == "setup" and report.skipped):
        return

    status = "skip" if report.skipped else ("fail" if report.failed else "pass")
    if report.when == "setup" and report.skipped:
        status = "skip"

    record: dict[str, Any] = {
        "nodeid": item.nodeid,
        "status": status,
        "duration_s": round(report.duration, 4),
        "when": report.when,
    }
    # 标记
    try:
        record["markers"] = [m.name for m in item.iter_markers()]
    except Exception:
        record["markers"] = []

    log_path = REPORTS_ROOT / "_collected.jsonl"
    try:
        REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        # 报告写入失败不应影响测试本身
        pass


# --------------------------------------------------------------------------- #
# 会话结束时聚合 summary.json
# --------------------------------------------------------------------------- #
def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """会话结束：把 _collected.jsonl 聚合为 reports/summary.json。"""
    log_path = REPORTS_ROOT / "_collected.jsonl"
    if not log_path.exists():
        return

    records: list[dict[str, Any]] = []
    try:
        with open(log_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except (OSError, json.JSONDecodeError):
        return

    total = len(records)
    passed = sum(1 for r in records if r.get("status") == "pass")
    failed = sum(1 for r in records if r.get("status") == "fail")
    skipped = sum(1 for r in records if r.get("status") == "skip")
    summary = {
        "total": total,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "records": records,
    }
    out_path = REPORTS_ROOT / "summary.json"
    try:
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)
    except OSError:
        pass
