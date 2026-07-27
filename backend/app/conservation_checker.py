"""BioDynamics Agent - 质量守恒检查器（Mass Conservation Checker）

TASK 5 修复：检测 ODE 仿真结果中蛋白池的质量守恒违规。

规则：
- total(EGFR states) ≈ constant
- total(Grb2 states) ≈ constant
- total(MEK states) ≈ constant
- total(ERK states) ≈ constant

允许误差：< 5%
否则 raise warning：CONSERVATION_VIOLATION
"""

from __future__ import annotations

import csv
import io
import logging
from dataclasses import dataclass, field
from pathlib import Path

from app.csv_io import decode_csv_text
from app.species_ontology import build_conservation_groups

logger = logging.getLogger(__name__)

# 守恒误差阈值（5%）
CONSERVATION_TOLERANCE = 0.05


@dataclass
class ConservationViolation:
    """单个守恒违规。"""
    pool_name: str
    species_in_pool: list[str]
    initial_total: float
    final_total: float
    relative_drift: float  # |final - initial| / initial
    severity: str  # "warning" / "critical"


@dataclass
class ConservationReport:
    """守恒检查报告。"""
    passed: bool
    violations: list[ConservationViolation] = field(default_factory=list)
    summary: str = ""


def check_conservation_from_csv(
    csv_path: str,
    species_names: list[str] | None = None,
) -> ConservationReport:
    """从 simulation.csv 检查质量守恒。

    Args:
        csv_path: simulation.csv 路径
        species_names: 物种名列表（可选，若 None 则从 CSV 表头推断）

    Returns:
        ConservationReport
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        return ConservationReport(passed=False, summary=f"CSV 不存在：{csv_path}")

    # 读取 CSV（多编码兼容，委托 app.csv_io 统一解码边界）
    decoded, _encoding = decode_csv_text(csv_file)
    if not decoded:
        return ConservationReport(passed=False, summary="CSV 为空或不可读")
    reader = csv.reader(io.StringIO(decoded, newline=""))
    header = next(reader, None)
    rows = list(reader)

    if not header or not rows:
        return ConservationReport(passed=False, summary="CSV 为空")

    # 解析物种名（跳过第一列 "t"）
    csv_species = header[1:]
    # 如果提供了 species_names，使用它；否则用 CSV 表头
    all_species = species_names if species_names else csv_species

    # 构建守恒分组
    conservation_groups = build_conservation_groups(all_species)
    if not conservation_groups:
        return ConservationReport(
            passed=True,
            summary="无可守恒分组（单物种池），跳过检查",
        )

    # 读取每列的初始值和终值
    species_initial: dict[str, float] = {}
    species_final: dict[str, float] = {}
    for i, sp in enumerate(csv_species):
        col_idx = i + 1  # 跳过 t 列
        if col_idx >= len(header):
            continue
        try:
            values = [float(row[col_idx]) for row in rows if row and len(row) > col_idx]
            if values:
                species_initial[sp] = values[0]
                species_final[sp] = values[-1]
        except (ValueError, IndexError):
            continue

    # 检查每个守恒分组
    violations: list[ConservationViolation] = []
    for pool_name, pool_species in conservation_groups.items():
        # 仅检查 CSV 中存在的物种
        existing_species = [s for s in pool_species if s in species_initial]
        if len(existing_species) < 2:
            continue

        initial_total = sum(species_initial.get(s, 0.0) for s in existing_species)
        final_total = sum(species_final.get(s, 0.0) for s in existing_species)

        if initial_total <= 0:
            continue

        relative_drift = abs(final_total - initial_total) / initial_total
        if relative_drift > CONSERVATION_TOLERANCE:
            severity = "critical" if relative_drift > 0.20 else "warning"
            violations.append(ConservationViolation(
                pool_name=pool_name,
                species_in_pool=existing_species,
                initial_total=initial_total,
                final_total=final_total,
                relative_drift=relative_drift,
                severity=severity,
            ))
            logger.warning(
                "CONSERVATION_VIOLATION: pool=%s, initial=%.4f, final=%.4f, "
                "drift=%.2f%% (threshold=%.0f%%)",
                pool_name, initial_total, final_total,
                relative_drift * 100, CONSERVATION_TOLERANCE * 100,
            )

    passed = len(violations) == 0
    summary = (
        f"守恒检查{'通过' if passed else '失败'}："
        f"{len(conservation_groups)} 个池，{len(violations)} 个违规"
    )
    return ConservationReport(passed=passed, violations=violations, summary=summary)


def format_conservation_warnings(report: ConservationReport) -> list[str]:
    """将守恒违规格式化为警告字符串列表（供 N8 metadata.warnings 使用）。"""
    warnings = []
    for v in report.violations:
        warnings.append(
            f"CONSERVATION_VIOLATION: {v.pool_name} pool "
            f"(species: {', '.join(v.species_in_pool)}) "
            f"initial={v.initial_total:.4f} → final={v.final_total:.4f} "
            f"drift={v.relative_drift*100:.1f}% [{v.severity}]"
        )
    return warnings
