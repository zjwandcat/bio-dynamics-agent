"""BioDynamics Agent - 统一 CSV I/O 边界（CSV Encoding Boundary）

单一可信源（single source of truth）for reading/writing simulation.csv across
the pipeline. 所有 CSV 读取点（derived_benchmark_metrics / feature_extractor /
conservation_checker / validation_v2.level1_internal / main / orchestrator /
benchmarks.agent_case_evaluator）必须经过本模块，不得各自实现编码探测。

设计原则（铁律）：
1. 读取端兼容历史文件：UTF-8-SIG → UTF-8 → GB18030 → CP1252。
   优先匹配首个能成功解码的编码，避免 errors=replace 引入乱码。
2. 写入端默认 UTF-8-SIG（带 BOM，Excel 兼容，避免中文列名乱码）。
3. 编码探测逻辑只此一处（防过拟合：不为单 case 硬编码、不各自实现）。
4. 检测到的编码随 artifact manifest 持久化（csv_manifest.encoding/columns/row_count）。
5. 纯 IO 边界，零科学逻辑、零 LLM 调用。

背景：DeepSeek 基准测试中 22 个非运行案例的根因为后处理 CSV 读取用严格 UTF-8，
导致仿真已完成却被标为 fail（参见 DEEPSEEK_MACRO_ANALYSIS.md §B）。
"""

from __future__ import annotations

import csv
import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 编码探测候选顺序：
# BOM 单独检测，避免把普通 UTF-8 错记成 utf-8-sig。
_ENCODING_CANDIDATES: tuple[str, ...] = ("utf-8", "gb18030", "cp1252")

# 写入默认编码（带 BOM，Excel 双击不乱码）
DEFAULT_WRITE_ENCODING: str = "utf-8-sig"


@dataclass
class CsvReadResult:
    """统一 CSV 读取结果（数值时间序列）。

    Attributes:
        times: 时间列数值列表（CSV 第一列）。
        species: {species_name: list[float]} 各物种浓度时序。
        columns: 物种列名列表（不含时间列）。
        encoding: 实际解码使用的编码名（用于 manifest 元数据）。
        row_count: 数据行数（不含表头）。
    """

    times: list[float] = field(default_factory=list)
    species: dict[str, list[float]] = field(default_factory=dict)
    columns: list[str] = field(default_factory=list)
    encoding: str = ""
    row_count: int = 0
    error: str = ""

    @property
    def empty(self) -> bool:
        """CSV 不可读或无有效数值行时返回 True。"""
        return not self.times or not self.species


def decode_csv_text(path: str | Path) -> tuple[str, str]:
    """读取文件并按多编码策略解码，返回 (decoded_text, encoding_used)。

    编码探测顺序：utf-8-sig → gb18030 → cp1252 → utf-8(errors=replace)。
    文件不存在时返回 ("", "")。

    低层 API：调用方需自行解析 decoded_text（如用 csv.DictReader 保留原始字符串）。
    高层 API 见 :func:`read_csv_robust`（直接返回数值时间序列）。

    Args:
        path: CSV 文件路径。

    Returns:
        (decoded_text, encoding_used)。encoding_used 为实际命中的编码名，
        回退场景记为 ``"utf-8(replace)"``。
    """
    p = Path(path)
    if not p.exists():
        return "", ""
    raw = p.read_bytes()
    if not raw:
        return "", ""
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig"), "utf-8-sig"
    for enc in _ENCODING_CANDIDATES:
        try:
            decoded = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        # 全部候选失败 → 保底 utf-8 + replace（不抛异常，可能产生替换字符）
        decoded = raw.decode("utf-8", errors="replace")
        enc = "utf-8(replace)"
    return decoded, enc


def detect_encoding(path: str | Path) -> str:
    """检测 CSV 文件编码（首个能成功解码的候选）。文件不存在返回 ""。"""
    return decode_csv_text(path)[1]


def read_csv_robust(path: str | Path) -> CsvReadResult:
    """统一 CSV 读取：多编码兼容 + 数值时间序列解析。

    兼容 UTF-8/UTF-8-SIG/GB18030/CP1252。文件不存在/空/无有效数值行时
    返回空 :class:`CsvReadResult`（**不抛异常**），调用方据 ``.empty`` 判定。

    解析规则：
    - 第一行为表头，第一列为时间列，其余为物种浓度列。
    - 数值解析失败的行整体跳过（保留有效行）。
    - 列名做 ``.strip()`` 规范化。

    Args:
        path: simulation.csv 文件路径。

    Returns:
        :class:`CsvReadResult`，含 times/species/columns/encoding/row_count。
    """
    decoded, encoding_used = decode_csv_text(path)
    if not decoded:
        return CsvReadResult(encoding=encoding_used)
    if "\x00" in decoded:
        return CsvReadResult(
            encoding=encoding_used,
            error="binary_or_unsupported_encoding: NUL byte detected",
        )
    control_count = sum(
        ord(ch) < 32 and ch not in "\r\n\t" for ch in decoded
    )
    if control_count / max(len(decoded), 1) > 0.01:
        return CsvReadResult(
            encoding=encoding_used,
            error="binary_or_corrupt: excessive control characters",
        )

    f = io.StringIO(decoded, newline="")
    reader = csv.reader(f)
    header: list[str] = []
    rows: list[list[float]] = []
    try:
        for idx, row in enumerate(reader):
            if not row:
                continue
            if idx == 0:
                header = [c.strip() for c in row]
                continue
            try:
                if len(row) != len(header):
                    continue
                rows.append([float(x) for x in row])
            except (TypeError, ValueError):
                continue
    except csv.Error as exc:
        # 二进制乱码 / 畸形 CSV（如未剔除的控制字符）→ 记日志，返回已解析部分。
        # 若连表头都未拿到则返回空（可解释 hard failure）。
        logger.warning("read_csv_robust: csv.Error parsing %s: %s", path, exc)
        if not header:
            return CsvReadResult(encoding=encoding_used, error=f"csv_parse_error: {exc}")

    if not header or not rows:
        return CsvReadResult(
            encoding=encoding_used,
            error="missing_header_or_numeric_rows",
        )
    if len(header) < 2 or header[0].strip().lower() not in {"time", "t"}:
        return CsvReadResult(
            encoding=encoding_used,
            error="invalid_schema: first column must be time or t",
        )

    times = [r[0] for r in rows]
    columns = header[1:]
    species: dict[str, list[float]] = {}
    for col_idx, name in enumerate(columns, start=1):
        species[name] = [
            r[col_idx] if col_idx < len(r) else float("nan") for r in rows
        ]
    return CsvReadResult(
        times=times,
        species=species,
        columns=columns,
        encoding=encoding_used,
        row_count=len(rows),
    )


def write_csv_standard(
    rows: list[list[float]],
    columns: list[str],
    path: str | Path,
    time_unit: str = "min",
) -> dict[str, Any]:
    """统一 CSV 写入：默认 UTF-8-SIG（带 BOM，Excel 兼容）。

    写入格式：表头 ``t,<species1>,<species2>,...``，后续行为数值。
    生成器应优先调用本函数写出 simulation.csv，读取端再以
    :func:`read_csv_robust` 兼容历史文件。

    Args:
        rows: 数据行列表，每行首位为时间，其余为各物种浓度（与 columns 对齐）。
        columns: 物种列名列表（不含时间列）。
        path: 输出 CSV 路径。
        time_unit: 时间单位（如 ``"min"`` / ``"s"``），记入 manifest。

    Returns:
        manifest 元数据 dict，含 encoding/columns/row_count/time_unit。
        调用方应将其写入 artifact manifest（orchestrator_result.csv_manifest）。
    """
    p = Path(path)
    header = ["t"] + list(columns)
    with p.open("w", encoding=DEFAULT_WRITE_ENCODING, newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)
    return {
        "encoding": DEFAULT_WRITE_ENCODING,
        "columns": list(columns),
        "row_count": len(rows),
        "time_unit": time_unit,
    }


def build_csv_manifest(path: str | Path, time_unit: str = "min") -> dict[str, Any]:
    """读取已有 CSV 并构建 manifest 元数据（encoding/columns/row_count/time_unit）。

    供 orchestrator 在 _extract_real_artifacts 中调用：仿真 CSV 由 sandbox 写出
    （科学逻辑，不改动），orchestrator 读取时记录其元数据到 artifact manifest，
    便于下游追溯编码/列结构。

    Args:
        path: simulation.csv 路径。
        time_unit: 时间单位（如已知，默认 ``"min"``）。

    Returns:
        manifest dict。文件不可读时 row_count=0、columns=[]、encoding 记检测值。
    """
    result = read_csv_robust(path)
    return {
        "encoding": result.encoding,
        "columns": list(result.columns),
        "row_count": result.row_count,
        "time_unit": time_unit,
        "error": result.error,
    }


__all__ = [
    "CsvReadResult",
    "decode_csv_text",
    "detect_encoding",
    "read_csv_robust",
    "write_csv_standard",
    "build_csv_manifest",
    "DEFAULT_WRITE_ENCODING",
]
