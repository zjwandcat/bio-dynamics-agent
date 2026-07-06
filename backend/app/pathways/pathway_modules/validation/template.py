# BioDynamics Agent v4 - Pathway Specialist Validation 模块数据结构模板 (Task 4.2.3)
# 定义 ValidationModuleData dataclass，作为 PathwaySpecialistBase.apply_validation()
# 的返回数据结构骨架。具体 Specialist 子类（Task 4.3-4.12）可实例化并填充。

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ValidationModuleData:
    """验证模块数据结构：通路特异 Validation 规则与文献 benchmark。

    由 ``PathwaySpecialistBase.apply_validation()`` 返回，包含通路特异的
    benchmark 规则、容忍度与文献 PMID 引用。
    """

    # Validation 规则列表
    # 每条含 metric_name / expected / tolerance / pmid / comparison / unit
    # 例：{"metric_name": "pEGFR_peak_time_min", "expected": 7.5,
    #      "tolerance": 2.5, "pmid": "Lev Bar-Or 2000"}
    rules: list[dict] = field(default_factory=list)

    # 文献 benchmark 清单
    # 每条含 benchmark_name / value / unit / condition / reference
    benchmarks: list[dict] = field(default_factory=list)

    # 通路特异容忍度（key: metric_name, value: tolerance）
    # 用于覆盖 P5 Level 2 默认阈值（如 NF-κB 容忍更大 peak_time_diff）
    tolerances: dict[str, float] = field(default_factory=dict)

    # 文献 PMID 引用列表
    # 每条含 pmid / citation / pathway_class / metric_name
    pmid_references: list[dict] = field(default_factory=list)


__all__ = ["ValidationModuleData"]
