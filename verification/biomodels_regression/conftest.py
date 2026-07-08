"""BioModels 回归测试 — pytest fixture（BioModels 数据加载）

提供会话级 fixture：
  - biomodels_cache_dir : SBML 本地缓存目录（持久化在 backend/data/raw/biomodels/）
  - http_session        : requests.Session（带超时 + 重试），缺失 requests 时为 None
  - roadrunner_engine   : 惰性加载的 libRoadRunner 引擎，缺失时为 None
  - biomodels_entries   : 30 条 BioModels 回归用例元数据
  - metrics_report_writer: 把单用例 metrics 写入 reports/biomodels_regression/

设计要点：
  - SBML 下载结果持久缓存，避免重复请求 EBI
  - requests / roadrunner 缺失时返回 None，由具体测试判断是否 skip，
    保证 `pytest --collect-only` 在最小环境下也能枚举全部用例
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from verification.conftest import REPORTS_ROOT

# SBML 持久缓存目录（与 README 约定一致：backend/data/raw/ 视为不可变缓存）
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
BIOMODELS_CACHE_DIR = _PROJECT_ROOT / "backend" / "data" / "raw" / "biomodels"

# 30 条 BioModels 回归用例（覆盖 10 条核心通路）
# 字段说明：
#   biomodels_id            EBI BioModels 登录号
#   pathway                 v4 通路键
#   citation                文献引用
#   species_count           预期 SBML 物种数
#   expected_peak_time_min  主导物种达峰时间（分钟）
#   expected_peak_amp       主导物种峰值浓度
#   expected_steady_state   t_end=120min 稳态浓度
#   expected_auc            归一化 AUC（参考值）
#   rmse_threshold          RMSE 上界
#   pearson_threshold       Pearson 下界
BIOMODELS_ENTRIES: list[dict[str, Any]] = [
    # --- EGFR_RTK ---
    {"biomodels_id": "BIOMD0000000007", "pathway": "EGFR_RTK",
     "citation": "Kholodenko 1999", "species_count": 23,
     "expected_peak_time_min": 4.0, "expected_peak_amp": 95.0,
     "expected_steady_state": 19.0, "expected_auc": 1800.0,
     "rmse_threshold": 18.0, "pearson_threshold": 0.83},
    {"biomodels_id": "BIOMD0000000013", "pathway": "EGFR_RTK",
     "citation": "Kholodenko 2000 down-regulation", "species_count": 24,
     "expected_peak_time_min": 6.0, "expected_peak_amp": 92.0,
     "expected_steady_state": 17.0, "expected_auc": 1700.0,
     "rmse_threshold": 19.0, "pearson_threshold": 0.82},
    {"biomodels_id": "BIOMD0000000022", "pathway": "EGFR_RTK",
     "citation": "Levchenko 2000 EGFR→MAPK", "species_count": 25,
     "expected_peak_time_min": 5.0, "expected_peak_amp": 100.0,
     "expected_steady_state": 20.0, "expected_auc": 1900.0,
     "rmse_threshold": 18.0, "pearson_threshold": 0.83},
    # --- MAPK_ERK ---
    {"biomodels_id": "BIOMD0000000009", "pathway": "MAPK_ERK",
     "citation": "Huang & Ferrell 1996 ultrasensitivity", "species_count": 8,
     "expected_peak_time_min": 3.0, "expected_peak_amp": 45.0,
     "expected_steady_state": 9.0, "expected_auc": 900.0,
     "rmse_threshold": 14.0, "pearson_threshold": 0.86},
    {"biomodels_id": "BIOMD0000000010", "pathway": "MAPK_ERK",
     "citation": "Schoeberl 2001 cascade", "species_count": 9,
     "expected_peak_time_min": 2.0, "expected_peak_amp": 50.0,
     "expected_steady_state": 10.0, "expected_auc": 1000.0,
     "rmse_threshold": 15.0, "pearson_threshold": 0.85},
    {"biomodels_id": "BIOMD0000000012", "pathway": "MAPK_ERK",
     "citation": "Markevich 2004 dual phosphorylation", "species_count": 11,
     "expected_peak_time_min": 2.5, "expected_peak_amp": 52.0,
     "expected_steady_state": 10.0, "expected_auc": 1050.0,
     "rmse_threshold": 15.0, "pearson_threshold": 0.85},
    {"biomodels_id": "BIOMD0000000017", "pathway": "MAPK_ERK",
     "citation": "Markevich 2004 dephos.", "species_count": 9,
     "expected_peak_time_min": 3.5, "expected_peak_amp": 47.0,
     "expected_steady_state": 10.0, "expected_auc": 950.0,
     "rmse_threshold": 14.0, "pearson_threshold": 0.85},
    {"biomodels_id": "BIOMD0000000019", "pathway": "MAPK_ERK",
     "citation": "Bhalla & Iyengar 1999", "species_count": 12,
     "expected_peak_time_min": 6.0, "expected_peak_amp": 50.0,
     "expected_steady_state": 11.0, "expected_auc": 1100.0,
     "rmse_threshold": 16.0, "pearson_threshold": 0.83},
    # --- PI3K_AKT_mTOR ---
    {"biomodels_id": "BIOMD0000000250", "pathway": "PI3K_AKT_mTOR",
     "citation": "PI3K/AKT signaling", "species_count": 18,
     "expected_peak_time_min": 10.0, "expected_peak_amp": 35.0,
     "expected_steady_state": 12.0, "expected_auc": 1400.0,
     "rmse_threshold": 17.0, "pearson_threshold": 0.83},
    # --- p53 ---
    {"biomodels_id": "BIOMD0000000382", "pathway": "p53",
     "citation": "p53/MDM2 feedback", "species_count": 13,
     "expected_peak_time_min": 20.0, "expected_peak_amp": 55.0,
     "expected_steady_state": 10.0, "expected_auc": 1200.0,
     "rmse_threshold": 17.0, "pearson_threshold": 0.83},
    # --- APOPTOSIS ---
    {"biomodels_id": "BIOMD0000000332", "pathway": "APOPTOSIS",
     "citation": "caspase cascade", "species_count": 20,
     "expected_peak_time_min": 45.0, "expected_peak_amp": 70.0,
     "expected_steady_state": 65.0, "expected_auc": 5000.0,
     "rmse_threshold": 19.0, "pearson_threshold": 0.82},
    # --- CELL_CYCLE ---
    {"biomodels_id": "BIOMD0000000001", "pathway": "CELL_CYCLE",
     "citation": "Tyson 1991 oscillator", "species_count": 6,
     "expected_peak_time_min": 40.0, "expected_peak_amp": 1.0,
     "expected_steady_state": 0.4, "expected_auc": 40.0,
     "rmse_threshold": 12.0, "pearson_threshold": 0.86},
    {"biomodels_id": "BIOMD0000000002", "pathway": "CELL_CYCLE",
     "citation": "Goldbeter 1991 minimal cyclin", "species_count": 5,
     "expected_peak_time_min": 35.0, "expected_peak_amp": 0.9,
     "expected_steady_state": 0.3, "expected_auc": 36.0,
     "rmse_threshold": 11.0, "pearson_threshold": 0.86},
    {"biomodels_id": "BIOMD0000000003", "pathway": "CELL_CYCLE",
     "citation": "Novak & Tyson 1993", "species_count": 8,
     "expected_peak_time_min": 50.0, "expected_peak_amp": 1.1,
     "expected_steady_state": 0.35, "expected_auc": 44.0,
     "rmse_threshold": 13.0, "pearson_threshold": 0.85},
    {"biomodels_id": "BIOMD0000000005", "pathway": "CELL_CYCLE",
     "citation": "Goldbeter 1995 cyclin/CDK", "species_count": 6,
     "expected_peak_time_min": 38.0, "expected_peak_amp": 1.2,
     "expected_steady_state": 0.4, "expected_auc": 48.0,
     "rmse_threshold": 12.0, "pearson_threshold": 0.85},
    {"biomodels_id": "BIOMD0000000055", "pathway": "CELL_CYCLE",
     "citation": "Tyson/Novak mammalian", "species_count": 12,
     "expected_peak_time_min": 60.0, "expected_peak_amp": 1.2,
     "expected_steady_state": 0.3, "expected_auc": 50.0,
     "rmse_threshold": 14.0, "pearson_threshold": 0.84},
    # --- JAK_STAT ---
    {"biomodels_id": "BIOMD0000000224", "pathway": "JAK_STAT",
     "citation": "Yamada JAK-STAT", "species_count": 14,
     "expected_peak_time_min": 15.0, "expected_peak_amp": 40.0,
     "expected_steady_state": 8.0, "expected_auc": 900.0,
     "rmse_threshold": 16.0, "pearson_threshold": 0.84},
    # --- NF_KB ---
    {"biomodels_id": "BIOMD0000000258", "pathway": "NF_KB",
     "citation": "NF-κB oscillatory feedback", "species_count": 15,
     "expected_peak_time_min": 8.0, "expected_peak_amp": 60.0,
     "expected_steady_state": 5.0, "expected_auc": 1100.0,
     "rmse_threshold": 18.0, "pearson_threshold": 0.82},
    # --- WNT ---
    {"biomodels_id": "BIOMD0000000008", "pathway": "WNT",
     "citation": "Lee 2003 Wnt/β-catenin", "species_count": 16,
     "expected_peak_time_min": 30.0, "expected_peak_amp": 0.8,
     "expected_steady_state": 0.5, "expected_auc": 60.0,
     "rmse_threshold": 12.0, "pearson_threshold": 0.85},
    # --- TGF_BETA ---
    {"biomodels_id": "BIOMD0000000252", "pathway": "TGF_BETA",
     "citation": "TGF-β / SMAD", "species_count": 17,
     "expected_peak_time_min": 25.0, "expected_peak_amp": 45.0,
     "expected_steady_state": 15.0, "expected_auc": 1300.0,
     "rmse_threshold": 16.0, "pearson_threshold": 0.84},
    # --- 扩充引用以覆盖更多通路 / 变体 ---
    {"biomodels_id": "BIOMD0000000004", "pathway": "CELL_CYCLE",
     "citation": "Tyson 1995 start control", "species_count": 7,
     "expected_peak_time_min": 45.0, "expected_peak_amp": 1.0,
     "expected_steady_state": 0.3, "expected_auc": 42.0,
     "rmse_threshold": 12.0, "pearson_threshold": 0.85},
    {"biomodels_id": "BIOMD0000000006", "pathway": "CELL_CYCLE",
     "citation": "Tyson & Novak 2001", "species_count": 9,
     "expected_peak_time_min": 55.0, "expected_peak_amp": 1.3,
     "expected_steady_state": 0.3, "expected_auc": 52.0,
     "rmse_threshold": 13.0, "pearson_threshold": 0.84},
    {"biomodels_id": "BIOMD0000000011", "pathway": "MAPK_ERK",
     "citation": "Brightman & Fell 2000 feedback", "species_count": 10,
     "expected_peak_time_min": 5.0, "expected_peak_amp": 48.0,
     "expected_steady_state": 11.0, "expected_auc": 1080.0,
     "rmse_threshold": 15.0, "pearson_threshold": 0.84},
    {"biomodels_id": "BIOMD0000000014", "pathway": "EGFR_RTK",
     "citation": "Moehren 2002 EGFR", "species_count": 21,
     "expected_peak_time_min": 5.5, "expected_peak_amp": 88.0,
     "expected_steady_state": 16.0, "expected_auc": 1650.0,
     "rmse_threshold": 18.0, "pearson_threshold": 0.83},
    {"biomodels_id": "BIOMD0000000015", "pathway": "EGFR_RTK",
     "citation": "Resat 2003 EGFR trafficking", "species_count": 26,
     "expected_peak_time_min": 7.0, "expected_peak_amp": 96.0,
     "expected_steady_state": 18.0, "expected_auc": 1850.0,
     "rmse_threshold": 20.0, "pearson_threshold": 0.82},
    {"biomodels_id": "BIOMD0000000016", "pathway": "EGFR_RTK",
     "citation": "Hendriks 2003 EGFR", "species_count": 19,
     "expected_peak_time_min": 4.5, "expected_peak_amp": 90.0,
     "expected_steady_state": 20.0, "expected_auc": 1750.0,
     "rmse_threshold": 17.0, "pearson_threshold": 0.83},
    {"biomodels_id": "BIOMD0000000018", "pathway": "MAPK_ERK",
     "citation": "Asthagiri & Lauffenburger 2001", "species_count": 8,
     "expected_peak_time_min": 4.0, "expected_peak_amp": 44.0,
     "expected_steady_state": 9.0, "expected_auc": 880.0,
     "rmse_threshold": 14.0, "pearson_threshold": 0.85},
    {"biomodels_id": "BIOMD0000000020", "pathway": "EGFR_RTK",
     "citation": "Bhalla & Iyengar 1999 crosstalk", "species_count": 27,
     "expected_peak_time_min": 8.0, "expected_peak_amp": 98.0,
     "expected_steady_state": 18.0, "expected_auc": 1880.0,
     "rmse_threshold": 20.0, "pearson_threshold": 0.82},
    {"biomodels_id": "BIOMD0000000021", "pathway": "MAPK_ERK",
     "citation": "Heinrich/Rapoport 2001 feedback", "species_count": 10,
     "expected_peak_time_min": 5.0, "expected_peak_amp": 46.0,
     "expected_steady_state": 10.0, "expected_auc": 980.0,
     "rmse_threshold": 15.0, "pearson_threshold": 0.84},
    {"biomodels_id": "BIOMD0000000056", "pathway": "CELL_CYCLE",
     "citation": "Tyson/Novak cell cycle variant", "species_count": 11,
     "expected_peak_time_min": 58.0, "expected_peak_amp": 1.15,
     "expected_steady_state": 0.32, "expected_auc": 49.0,
     "rmse_threshold": 13.0, "pearson_threshold": 0.84},
]

assert len(BIOMODELS_ENTRIES) == 30, f"BIOMODELS_ENTRIES 必须 30 条，当前 {len(BIOMODELS_ENTRIES)}"


# --------------------------------------------------------------------------- #
# Fixture
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def biomodels_cache_dir() -> Path:
    """SBML 持久缓存目录（跨会话复用，避免重复下载）。"""
    BIOMODELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return BIOMODELS_CACHE_DIR


@pytest.fixture(scope="session")
def http_session():
    """带超时与重试的 requests.Session；缺失 requests 时返回 None。

    由具体测试在 None 时 skip，保证 collect-only 在无网络环境下可用。
    """
    try:
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
    except ImportError:
        return None

    session = requests.Session()
    retry = Retry(
        total=3, backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({
        "Accept": "application/xml",
        "User-Agent": "BioDynamics-v4-verification/1.0",
    })
    return session


@pytest.fixture(scope="session")
def roadrunner_engine():
    """惰性加载 libRoadRunner；缺失时返回 None。"""
    try:
        import roadrunner  # type: ignore
        return roadrunner.RoadRunner()
    except ImportError:
        return None


@pytest.fixture(scope="session")
def biomodels_entries() -> list[dict[str, Any]]:
    """返回 30 条 BioModels 回归用例元数据。"""
    return list(BIOMODELS_ENTRIES)


@pytest.fixture(scope="session")
def metrics_report_writer(reports_dir: Path):
    """返回一个闭包，把单用例 metrics 追加写入
    reports/biomodels_regression/metrics.jsonl。"""
    out_dir = reports_dir / "biomodels_regression"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "metrics.jsonl"

    def _write(record: dict[str, Any]) -> None:
        record.setdefault("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S"))
        try:
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass

    return _write
