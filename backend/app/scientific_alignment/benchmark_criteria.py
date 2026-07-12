# BioDynamics Scientific Alignment — Benchmark Criteria (Task 2)
#
# Benchmark 配套科学对齐标准：为 10 通路定义 semantic_criteria 与
# forbidden_patterns，供 benchmark_runner.py 读取。
# 基于 Canonical Reference Library（由 SA_CANONICAL flag 守护）。
#
# 设计要点：
# 1. 只创建本文件，不修改 benchmark_runner.py / __init__.py / config.py。
# 2. Feature Flag 守护：SA_CANONICAL 关闭时返回 skipped=True
#    （v3 默认行为，仅数值校验，不做语义对齐）。
# 3. 缺失 SCIENTIFIC_ALIGNMENT_FIELDS 不 fail，按纯数值降级，但输出 warning。
# 4. semantic_criteria 支持 4 种 check_type：
#    - regex:        re.search(pattern, report_content, IGNORECASE)
#    - keyword:      pattern.lower() in report_content.lower()
#    - semantic:     简化为 keyword 匹配
#    - numeric_range: 从 simulation_metrics[pattern] 取值，检查是否落在 expected 范围
# 5. forbidden_patterns 支持 regex / keyword 两种匹配模式。

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)


# =============================================================================
# SubTask 2.1：SCIENTIFIC_ALIGNMENT_FIELDS 可选字段集
# =============================================================================
# 供 benchmark_runner 读取的科学对齐可选字段集。
# 缺失这些字段不 fail，但会触发 deprecation warning（纯数值降级）。
SCIENTIFIC_ALIGNMENT_FIELDS: list[str] = [
    "semantic_criteria",
    "forbidden_patterns",
    "required_reviews",
    "required_biomodels",
    "required_experiments",
    "expected_dynamics",
    "required_mechanisms",
]


# =============================================================================
# SubTask 2.2：数据结构定义
# =============================================================================
@dataclass
class SemanticCriterion:
    """单条语义级验收标准。"""

    name: str                # 如 "erk_transient_peak"
    description: str         # 人类可读描述
    check_type: str          # "regex" / "keyword" / "semantic" / "numeric_range"
    pattern: str = ""        # regex 或 keyword 模式；numeric_range 时为 metrics 键名
    expected: str = ""       # 期望值（numeric_range 时如 "10-20"）
    weight: float = 1.0      # 权重


@dataclass
class ForbiddenPattern:
    """禁止模式：报告中不应出现的内容。"""

    pattern: str                 # 正则或关键词
    pattern_type: str = "regex"  # "regex" / "keyword"
    reason: str = ""             # 为何禁止


@dataclass
class BenchmarkCriteria:
    """通路对应的 Benchmark 标准集合。"""

    pathway: str
    semantic_criteria: list[SemanticCriterion] = field(default_factory=list)
    forbidden_patterns: list[ForbiddenPattern] = field(default_factory=list)
    required_fields: list[str] = field(default_factory=list)


@dataclass
class CriteriaCheckReport:
    """Benchmark Criteria 检测报告。"""

    enabled: bool
    skipped: bool = False
    pathway: str = ""
    semantic_passed: list[str] = field(default_factory=list)
    semantic_failed: list[str] = field(default_factory=list)
    forbidden_hit: list[str] = field(default_factory=list)   # 命中的禁止模式
    warnings: list[str] = field(default_factory=list)        # 缺失字段 warning
    passed: bool = False     # semantic_failed 空 且 forbidden_hit 空


# =============================================================================
# 10 通路预定义 BenchmarkCriteria
# =============================================================================
# 通路名与 tasks.md Task 2 规格一致。未知通路返回空（warnings 追加 "unknown_pathway"）。


def _sc(
    name: str,
    description: str,
    check_type: str,
    pattern: str = "",
    expected: str = "",
    weight: float = 1.0,
) -> SemanticCriterion:
    """SemanticCriterion 构造快捷函数（缩短定义篇幅）。"""
    return SemanticCriterion(
        name=name,
        description=description,
        check_type=check_type,
        pattern=pattern,
        expected=expected,
        weight=weight,
    )


def _fp(
    pattern: str,
    pattern_type: str = "keyword",
    reason: str = "",
) -> ForbiddenPattern:
    """ForbiddenPattern 构造快捷函数（默认 keyword 匹配）。"""
    return ForbiddenPattern(pattern=pattern, pattern_type=pattern_type, reason=reason)


_PATHWAY_CRITERIA: dict[str, BenchmarkCriteria] = {
    # 1. EGFR / RTK 通路
    "EGFR": BenchmarkCriteria(
        pathway="EGFR",
        semantic_criteria=[
            _sc(
                "erk_transient_peak",
                "ERK 呈瞬时峰值（10-20 min）",
                "numeric_range",
                pattern="erk_peak_time",
                expected="10-20",
            ),
            _sc(
                "dusp_feedback_mentioned",
                "DUSP 负反馈机制被提及",
                "keyword",
                pattern="DUSP",
            ),
            _sc(
                "egfr_early_peak",
                "EGFR 早期峰值（0-5 min）",
                "numeric_range",
                pattern="egfr_peak_time",
                expected="0-5",
            ),
            _sc(
                "nuclear_erk_mentioned",
                "Nuclear ERK 转位被提及",
                "keyword",
                pattern="Nuclear ERK",
            ),
        ],
        forbidden_patterns=[
            _fp(
                "EGFR qPCR as primary validation",
                reason="qPCR 不能作为 EGFR 活性的主要验证手段",
            ),
            _fp(
                "No PubMed evidence",
                reason="必须引用 PubMed 文献证据",
            ),
        ],
    ),
    # 2. MAPK 级联通路
    "MAPK": BenchmarkCriteria(
        pathway="MAPK",
        semantic_criteria=[
            _sc(
                "cascade_phosphorylation",
                "Raf-MEK-ERK 级联磷酸化被提及",
                "keyword",
                pattern="Raf",
            ),
            _sc(
                "negative_feedback",
                "负反馈机制被提及",
                "keyword",
                pattern="negative feedback",
            ),
            _sc(
                "transient_dynamics",
                "瞬态动力学特征被提及",
                "keyword",
                pattern="transient",
            ),
        ],
        forbidden_patterns=[
            _fp(
                "No negative feedback",
                reason="MAPK 通路必须包含负反馈",
            ),
            _fp(
                "Permanent activation",
                reason="ERK 不应永久激活（应为瞬态）",
            ),
        ],
    ),
    # 3. PI3K-AKT-mTOR 通路
    "PI3K-AKT-mTOR": BenchmarkCriteria(
        pathway="PI3K-AKT-mTOR",
        semantic_criteria=[
            _sc(
                "akt_phosphorylation",
                "AKT 磷酸化被提及",
                "keyword",
                pattern="AKT phosphorylation",
            ),
            _sc(
                "mtor_downstream",
                "mTOR 下游效应被提及",
                "keyword",
                pattern="mTOR",
            ),
            _sc(
                "pip3_intermediate",
                "PIP3 中间产物被提及",
                "keyword",
                pattern="PIP3",
            ),
        ],
        forbidden_patterns=[
            _fp(
                "AKT as sole readout",
                reason="AKT 不能作为唯一读数",
            ),
            _fp(
                "No PTEN mention",
                reason="必须提及 PTEN 负调控",
            ),
        ],
    ),
    # 4. p53 通路
    "p53": BenchmarkCriteria(
        pathway="p53",
        semantic_criteria=[
            _sc(
                "mdm2_feedback",
                "MDM2 负反馈被提及",
                "keyword",
                pattern="MDM2",
            ),
            _sc(
                "oscillation_dynamics",
                "p53 振荡动力学被提及",
                "keyword",
                pattern="oscillation",
            ),
            _sc(
                "dna_damage_response",
                "DNA 损伤响应被提及",
                "keyword",
                pattern="DNA damage",
            ),
        ],
        forbidden_patterns=[
            _fp(
                "Linear p53 increase",
                reason="p53 应呈振荡而非线性增加",
            ),
            _fp(
                "No MDM2",
                reason="必须包含 MDM2 负反馈",
            ),
        ],
    ),
    # 5. Apoptosis 凋亡通路
    "Apoptosis": BenchmarkCriteria(
        pathway="Apoptosis",
        semantic_criteria=[
            _sc(
                "caspase_cascade",
                "Caspase 级联被提及",
                "keyword",
                pattern="caspase",
            ),
            _sc(
                "cytochrome_c_release",
                "细胞色素 c 释放被提及",
                "keyword",
                pattern="cytochrome c",
            ),
            _sc(
                "bid_truncation",
                "BID 截断（tBID）被提及",
                "keyword",
                pattern="BID",
            ),
        ],
        forbidden_patterns=[
            _fp(
                "Caspase-9 before CytoC",
                reason="Caspase-9 不应在 CytoC 释放之前激活",
            ),
            _fp(
                "No Bcl-2 family",
                reason="必须提及 Bcl-2 家族调控",
            ),
        ],
    ),
    # 6. Cell Cycle 细胞周期通路
    "Cell Cycle": BenchmarkCriteria(
        pathway="Cell Cycle",
        semantic_criteria=[
            _sc(
                "cyclin_cdk_binding",
                "Cyclin-CDK 结合被提及",
                "keyword",
                pattern="Cyclin",
            ),
            _sc(
                "rb_phosphorylation",
                "Rb 磷酸化被提及",
                "keyword",
                pattern="Rb phosphorylation",
            ),
            _sc(
                "apc_degradation",
                "APC 降解机制被提及",
                "keyword",
                pattern="APC",
            ),
        ],
        forbidden_patterns=[
            _fp(
                "CDK without cyclin",
                reason="CDK 必须与 Cyclin 结合并发挥作用",
            ),
            _fp(
                "No checkpoint",
                reason="必须提及细胞周期检查点",
            ),
        ],
    ),
    # 7. JAK-STAT 通路
    "JAK-STAT": BenchmarkCriteria(
        pathway="JAK-STAT",
        semantic_criteria=[
            _sc(
                "jak_phosphorylation",
                "JAK 磷酸化被提及",
                "keyword",
                pattern="JAK phosphorylation",
            ),
            _sc(
                "stat_dimerization",
                "STAT 二聚化被提及",
                "keyword",
                pattern="dimerization",
            ),
            _sc(
                "nuclear_translocation",
                "核转位被提及",
                "keyword",
                pattern="nuclear translocation",
            ),
        ],
        forbidden_patterns=[
            _fp(
                "STAT before JAK",
                reason="STAT 不应在 JAK 磷酸化之前激活",
            ),
            _fp(
                "No dimerization",
                reason="必须包含 STAT 二聚化步骤",
            ),
        ],
    ),
    # 8. NF-kB 通路
    "NF-kB": BenchmarkCriteria(
        pathway="NF-kB",
        semantic_criteria=[
            _sc(
                "ikb_degradation",
                "IkB 降解被提及",
                "keyword",
                pattern="IkB",
            ),
            _sc(
                "p65_nuclear",
                "p65 核转位被提及",
                "keyword",
                pattern="p65",
            ),
            _sc(
                "ikb_resynthesis",
                "IkB 再合成（负反馈）被提及",
                "keyword",
                pattern="resynthesis",
            ),
        ],
        forbidden_patterns=[
            _fp(
                "NF-kB without IkB",
                reason="NF-kB 通路必须包含 IkB 调控",
            ),
            _fp(
                "No negative feedback",
                reason="必须包含负反馈（IkB 再合成）",
            ),
        ],
    ),
    # 9. Wnt 通路
    "Wnt": BenchmarkCriteria(
        pathway="Wnt",
        semantic_criteria=[
            _sc(
                "beta_catenin_accumulation",
                "β-catenin 累积被提及",
                "keyword",
                pattern="beta-catenin",
            ),
            _sc(
                "apc_axin_complex",
                "APC-Axin 复合物被提及",
                "keyword",
                pattern="Axin",
            ),
            _sc(
                "destruction_complex",
                "破坏复合物（destruction complex）被提及",
                "keyword",
                pattern="destruction complex",
            ),
        ],
        forbidden_patterns=[
            _fp(
                "Beta-catenin before Wnt",
                reason="β-catenin 不应在 Wnt 信号之前累积",
            ),
            _fp(
                "No degradation",
                reason="必须包含 β-catenin 降解机制",
            ),
        ],
    ),
    # 10. TGF-beta 通路
    "TGF-beta": BenchmarkCriteria(
        pathway="TGF-beta",
        semantic_criteria=[
            _sc(
                "smad_phosphorylation",
                "SMAD 磷酸化被提及",
                "keyword",
                pattern="SMAD phosphorylation",
            ),
            _sc(
                "smad4_complex",
                "SMAD4 复合物被提及",
                "keyword",
                pattern="SMAD4",
            ),
            _sc(
                "nuclear_translocation",
                "核转位被提及",
                "keyword",
                pattern="nuclear translocation",
            ),
        ],
        forbidden_patterns=[
            _fp(
                "SMAD4 before SMAD2/3",
                reason="SMAD4 不应在 SMAD2/3 磷酸化之前入核",
            ),
            _fp(
                "No receptor",
                reason="必须提及 TGF-β 受体",
            ),
        ],
    ),
}


def _normalize_pathway(pathway: str) -> str:
    """规范化通路名：精确匹配优先，其次大小写不敏感匹配。

    Returns:
        匹配到的 _PATHWAY_CRITERIA 键；未匹配返回空字符串。
    """
    if not pathway or not isinstance(pathway, str):
        return ""
    key = pathway.strip()
    if key in _PATHWAY_CRITERIA:
        return key
    # 大小写不敏感匹配
    lowered = key.lower()
    for candidate in _PATHWAY_CRITERIA:
        if candidate.lower() == lowered:
            return candidate
    return ""


# =============================================================================
# SubTask 2.3：主函数
# =============================================================================
def get_benchmark_criteria(pathway: str) -> BenchmarkCriteria:
    """获取通路对应的 BenchmarkCriteria。

    Args:
        pathway: 通路标识（如 "EGFR" / "MAPK" / "PI3K-AKT-mTOR"）。

    Returns:
        匹配到的 BenchmarkCriteria；未知通路返回空 criteria
        （pathway=""），调用方可据此追加 "unknown_pathway" warning。
    """
    key = _normalize_pathway(pathway)
    if not key:
        logger.debug("未知通路 %r，返回空 BenchmarkCriteria", pathway)
        return BenchmarkCriteria(pathway="")
    return _PATHWAY_CRITERIA[key]


def _parse_numeric_range(expected: str) -> tuple[float, float] | None:
    """解析 "low-high" 格式的数值范围。

    Returns:
        (low, high) 元组；解析失败返回 None。
    """
    if not expected or not isinstance(expected, str):
        return None
    parts = expected.split("-")
    if len(parts) != 2:
        return None
    try:
        low = float(parts[0].strip())
        high = float(parts[1].strip())
    except ValueError:
        return None
    if low > high:
        low, high = high, low
    return (low, high)


def _check_semantic_criterion(
    criterion: SemanticCriterion,
    report_content: str,
    simulation_metrics: dict[str, Any],
) -> bool:
    """检测单条 semantic_criterion 是否通过。

    - regex:         re.search(pattern, report_content, re.IGNORECASE)
    - keyword:       pattern.lower() in report_content.lower()
    - semantic:      简化为 keyword 匹配
    - numeric_range: 从 simulation_metrics[pattern] 取值，检查是否落在 expected 范围

    Args:
        criterion: 语义标准。
        report_content: 报告文本（文本类检查的输入）。
        simulation_metrics: 仿真指标字典（numeric_range 检查的输入）。

    Returns:
        True 表示通过。
    """
    check_type = criterion.check_type

    # numeric_range：从 simulation_metrics 取值，不依赖 report_content
    if check_type == "numeric_range":
        range_tuple = _parse_numeric_range(criterion.expected)
        if range_tuple is None:
            return False
        low, high = range_tuple
        value = simulation_metrics.get(criterion.pattern)
        if value is None:
            return False
        try:
            num = float(value)
        except (TypeError, ValueError):
            return False
        return low <= num <= high

    # 文本类检查：report_content 为空时直接 fail
    if not report_content:
        return False

    if check_type == "regex":
        try:
            return (
                re.search(criterion.pattern, report_content, re.IGNORECASE)
                is not None
            )
        except re.error as exc:
            logger.warning(
                "semantic_criterion %r 正则编译失败: %s", criterion.name, exc
            )
            return False

    if check_type in ("keyword", "semantic"):
        return criterion.pattern.lower() in report_content.lower()

    # 未知 check_type → fail
    logger.warning(
        "未知 check_type %r (criterion=%s)", check_type, criterion.name
    )
    return False


def _check_forbidden_pattern(
    forbidden: ForbiddenPattern,
    report_content: str,
) -> bool:
    """检测 forbidden_pattern 是否命中。命中返回 True。

    - regex:   re.search(pattern, report_content, re.IGNORECASE)
    - keyword: pattern.lower() in report_content.lower()
    """
    if not report_content:
        return False
    if forbidden.pattern_type == "regex":
        try:
            return (
                re.search(forbidden.pattern, report_content, re.IGNORECASE)
                is not None
            )
        except re.error as exc:
            logger.warning(
                "forbidden_pattern 正则编译失败: %s (%s)",
                forbidden.pattern,
                exc,
            )
            return False
    # keyword 匹配（默认）
    return forbidden.pattern.lower() in report_content.lower()


def validate_yaml_fields(yaml_data: dict[str, Any]) -> list[str]:
    """检查 YAML 是否含 SCIENTIFIC_ALIGNMENT_FIELDS，返回缺失字段列表。

    缺失字段不 fail，仅作为 warnings 输出（纯数值降级）。

    Args:
        yaml_data: benchmark YAML 解析后的字典。

    Returns:
        缺失字段名列表；yaml_data 非 dict 时返回全部字段为缺失。
    """
    if not isinstance(yaml_data, dict):
        return list(SCIENTIFIC_ALIGNMENT_FIELDS)
    missing: list[str] = []
    for fname in SCIENTIFIC_ALIGNMENT_FIELDS:
        if fname not in yaml_data:
            missing.append(fname)
    return missing


def check_benchmark_criteria(
    pathway: str,
    report_content: str = "",
    simulation_metrics: dict[str, Any] | None = None,
    benchmark_yaml: dict[str, Any] | None = None,
) -> CriteriaCheckReport:
    """检测报告是否满足 semantic_criteria 且无 forbidden_patterns。

    流程：
    1. Flag OFF → skipped=True，直接返回（v3 默认仅数值校验）。
    2. 调用 get_benchmark_criteria(pathway) 获取标准；未知通路追加 warning。
    3. 对每条 semantic_criterion 执行 check_type 对应检测。
    4. 对每条 forbidden_pattern 在 report_content 中搜索，命中入 forbidden_hit。
    5. 缺失 SCIENTIFIC_ALIGNMENT_FIELDS → warnings（不 fail）。
    6. passed = (semantic_failed 为空) and (forbidden_hit 为空)。

    Args:
        pathway: 通路标识。
        report_content: 待检测的报告文本。
        simulation_metrics: 仿真指标字典（numeric_range 检测使用）。
        benchmark_yaml: 原 benchmark YAML 内容（用于字段完整性检查）。

    Returns:
        CriteriaCheckReport 检测报告。
    """
    # -------------------------------------------------------------------------
    # Feature Flag 守护：SA_CANONICAL 默认 OFF
    # 铁律：V4_SCIENTIFIC_ALIGNMENT_ENABLED=false 时，子 Flag 永远不生效
    # Flag OFF 时返回 skipped=True（v3 默认，仅数值校验）
    # -------------------------------------------------------------------------
    enabled = settings.is_sa_feature_enabled("CANONICAL")
    report = CriteriaCheckReport(enabled=enabled, pathway=pathway)
    if not enabled:
        report.skipped = True
        return report

    metrics = simulation_metrics or {}

    # 未知通路 → 空 criteria + warning
    key = _normalize_pathway(pathway)
    if not key:
        report.warnings.append(f"unknown_pathway: {pathway!r}")
        report.passed = False
        return report

    criteria = _PATHWAY_CRITERIA[key]

    # 检测 semantic_criteria
    for criterion in criteria.semantic_criteria:
        if _check_semantic_criterion(criterion, report_content, metrics):
            report.semantic_passed.append(criterion.name)
        else:
            report.semantic_failed.append(criterion.name)

    # 检测 forbidden_patterns
    for forbidden in criteria.forbidden_patterns:
        if _check_forbidden_pattern(forbidden, report_content):
            report.forbidden_hit.append(forbidden.pattern)

    # YAML 字段完整性 → warnings（不 fail，纯数值降级）
    if benchmark_yaml is not None:
        missing = validate_yaml_fields(benchmark_yaml)
        for fname in missing:
            report.warnings.append(f"missing_field: {fname}")

    report.passed = (
        len(report.semantic_failed) == 0 and len(report.forbidden_hit) == 0
    )
    return report


__all__ = [
    "SCIENTIFIC_ALIGNMENT_FIELDS",
    "SemanticCriterion",
    "ForbiddenPattern",
    "BenchmarkCriteria",
    "CriteriaCheckReport",
    "get_benchmark_criteria",
    "check_benchmark_criteria",
    "validate_yaml_fields",
]
