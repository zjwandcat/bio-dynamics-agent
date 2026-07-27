# Feature Flag: V4_SCIENTIFIC_REVIEWER_ENABLED
#
# BioDynamics Agent - Scientific Alignment Loop: 12 轴 Scientific Validation Matrix
# （Spec: add-scientific-reviewer-and-validation-matrix, Task 5）
#
# 模块用途：
#   将现有 7 轴 Validation Pyramid 升级为 12 轴 Scientific Validation Matrix，
#   每轴三态 PASS / PARTIAL / FAIL，禁用 High/Medium/Low 主观分级。任一轴 FAIL
#   即整体 Scientific Report 降 Confidence；任一轴 PARTIAL SHALL 在报告中显式标注。
#
# 对应 Spec Requirement：
#   - 12 轴 Scientific Validation Matrix
#
# 12 轴定义（对应 Spec Scenario "12 轴定义"）：
#    1. Mechanism Validation              — 关键机制节点链完整性
#    2. Ontology Validation               — 物种命名规范化（MCP / canonical species 命中率）
#    3. Literature Validation             — Canonical PMID 检索命中率（Top1-Top5）
#    4. BioModels Validation              — Matched ID / Deviation / Missing reactions / Parameter mismatch
#    5. Parameter Validation              — 参数来源占比（BioModels / LLM / Hardcode / Default / Missing）
#    6. Simulation Validation             — 仿真是否成功 + 数值稳定
#    7. Dynamics Validation               — 8 项 Curve Metrics 对照 expected_dynamics
#    8. Experiment Validation             — 机制驱动 + Forbidden 实验拦截
#    9. Evidence Attribution Validation   — 逐句证据标注（ungrounded / undergrounded）
#   10. Scientific Writing Validation     — Honesty 检测（过度声明 / 引用缺失 / 未标注）
#   11. Reproducibility Validation        — Random seed / Parameter snapshot / ODE template
#   12. Benchmark Validation              — 该通路 Golden Benchmark 是否 PASS
#
# 三态判定规则（禁用 High/Medium/Low）：
#   - PASS    — 完全符合 Canonical
#   - PARTIAL — 部分符合（如 6/10 节点匹配，或 4/8 Curve Metric PASS）
#   - FAIL    — 关键缺失或数值严重偏离（如 ERK Peak 45 min vs Expected 10-20 min）
#
# 复用现有模块：
#   - seven_axis_validator.py — 复用其 7 轴基础逻辑设计思想（但禁用 High/Medium/Low）
#   - curve_metrics.py        — Dynamics 轴调用 compute_curve_metrics() / compare_with_expected()
#   - evidence_graph.py       — Evidence Attribution 轴调用 build_from_report() / detect_ungrounded() / detect_undergrounded()
#   - scientific_honesty.py   — Scientific Writing 轴调用 review_report() / classify_claim() / detect_overstatement()
#
# Feature Flag 守护：
#   V4_SCIENTIFIC_REVIEWER_ENABLED 默认 false。关闭时本模块 SHALL 抛出 RuntimeError，
#   提示调用方回退到 seven_axis_validator.py。
#
# 核心导出：
#   from app.scientific_alignment.validation_matrix import (
#       ValidationStatus, AxisResult, ValidationMatrixResult,
#       run_validation_matrix,
#   )

from __future__ import annotations

import logging
import math
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import yaml

from app.config import settings
from app.scientific_alignment import (
    curve_metrics,
    evidence_graph,
    scientific_honesty,
)

logger = logging.getLogger(__name__)


# =============================================================================
# 常量定义（对应 Spec 12 轴定义）
# =============================================================================

# 12 轴名称（顺序固定，使用人类可读的 "X Validation" 格式，便于 get_axis 查询）
AXIS_MECHANISM_VALIDATION: str = "Mechanism Validation"
AXIS_ONTOLOGY_VALIDATION: str = "Ontology Validation"
AXIS_LITERATURE_VALIDATION: str = "Literature Validation"
AXIS_BIOMODELS_VALIDATION: str = "BioModels Validation"
AXIS_PARAMETER_VALIDATION: str = "Parameter Validation"
AXIS_SIMULATION_VALIDATION: str = "Simulation Validation"
AXIS_DYNAMICS_VALIDATION: str = "Dynamics Validation"
AXIS_EXPERIMENT_VALIDATION: str = "Experiment Validation"
AXIS_EVIDENCE_ATTRIBUTION_VALIDATION: str = "Evidence Attribution Validation"
AXIS_SCIENTIFIC_WRITING_VALIDATION: str = "Scientific Writing Validation"
AXIS_REPRODUCIBILITY_VALIDATION: str = "Reproducibility Validation"
AXIS_BENCHMARK_VALIDATION: str = "Benchmark Validation"

# 12 轴顺序元组（用于遍历与校验）
VALIDATION_AXES: tuple[str, ...] = (
    AXIS_MECHANISM_VALIDATION,
    AXIS_ONTOLOGY_VALIDATION,
    AXIS_LITERATURE_VALIDATION,
    AXIS_BIOMODELS_VALIDATION,
    AXIS_PARAMETER_VALIDATION,
    AXIS_SIMULATION_VALIDATION,
    AXIS_DYNAMICS_VALIDATION,
    AXIS_EXPERIMENT_VALIDATION,
    AXIS_EVIDENCE_ATTRIBUTION_VALIDATION,
    AXIS_SCIENTIFIC_WRITING_VALIDATION,
    AXIS_REPRODUCIBILITY_VALIDATION,
    AXIS_BENCHMARK_VALIDATION,
)

# 各轴在综合 Confidence 计算中的权重（PASS=1.0, PARTIAL=0.5, FAIL=0.0）
# 注意：实际计算在 _calculate_confidence 内联实现；此常量仅用于文档化权重定义。
_CONFIDENCE_WEIGHT: dict[str, float] = {
    "PASS": 1.0,
    "PARTIAL": 0.5,
    "FAIL": 0.0,
}

# 非字母数字字符正则：用于节点/物种名归一化（去除 - _ / 空格 . 等所有分隔符）
# 例如 "p-ERK" → "PERK"、"beta_catenin" → "BETACATENIN"
_NON_ALNUM_RE: re.Pattern[str] = re.compile(r"[^A-Za-z0-9]+")

# Literature 轴：Top-N 候选数（Spec 要求 Top1-Top5）
_LITERATURE_TOP_N: int = 5

# Literature 轴通过阈值（Top5 中 Canonical 数 >= 4 → PASS，>= 2 → PARTIAL，否则 FAIL）
_LITERATURE_PASS_HITS: int = 4
_LITERATURE_PARTIAL_HITS: int = 2

# BioModels 轴偏差阈值（deviation <= 0.05 → PASS；<= 0.2 → PARTIAL；> 0.2 → FAIL）
_BIOMODELS_PASS_DEVIATION: float = 0.05
_BIOMODELS_PARTIAL_DEVIATION: float = 0.2

# Parameter 轴：参数来源分类
_PARAM_GROUNDED_SOURCES: frozenset[str] = frozenset({"biomodels", "literature", "canonical"})
_PARAM_UNGROUNDED_SOURCES: frozenset[str] = frozenset({"hardcode", "default", "missing"})

# Simulation 轴：最少时间点数
_SIMULATION_MIN_POINTS: int = 2

# Reproducibility 轴：所需字段
_REPRODUCIBILITY_REQUIRED_FIELDS: tuple[str, ...] = (
    "random_seed",
    "parameter_snapshot",
    "ode_template",
)


# =============================================================================
# 数据结构（对应 Spec Scenario "12 轴定义"）
# =============================================================================


class ValidationStatus(str, Enum):
    """Validation 每轴三态（禁用 High/Medium/Low）。

    对应 Spec Scenario "禁用 High/Medium/Low"——SHALL 仅使用 PASS / PARTIAL / FAIL
    三态，使用 High/Medium/Low 则判 Validation 系统自身 FAIL。

    继承 ``str`` 与 ``Enum`` 便于 ``status == "PASS"`` 字符串比较与 JSON 序列化。
    """

    PASS = "PASS"
    PARTIAL = "PARTIAL"
    FAIL = "FAIL"


# 向后兼容：保留旧 AxisStatus 类（指向 ValidationStatus 的字符串常量）
class AxisStatus:
    """Deprecated: 请使用 ``ValidationStatus`` 枚举。

    保留是为兼容骨架阶段（Task 1）已暴露的接口名。
    """

    PASS: str = ValidationStatus.PASS.value
    PARTIAL: str = ValidationStatus.PARTIAL.value
    FAIL: str = ValidationStatus.FAIL.value


@dataclass
class AxisResult:
    """单轴验证结果。

    对应 Spec Scenario "12 轴定义"——每轴 SHALL 输出 status + evidence + reason。

    Attributes:
        axis: 轴名（如 ``"Mechanism Validation"``），与 ``VALIDATION_AXES`` 一致。
        status: ``ValidationStatus.PASS / PARTIAL / FAIL`` 三态之一。
        evidence: 证据描述（具体引用文件/字段/数值，如
            ``"ERK Peak 45.8 min vs Expected 10-20 min"``）。
        reason: 判定原因（PASS 时可为空，PARTIAL / FAIL 时必须说明）。
        details: 可选明细 dict（如缺哪些节点、哪些指标 FAIL、数值表）。
    """

    axis: str
    status: ValidationStatus
    evidence: str = ""
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationMatrixResult:
    """12 轴 Validation Matrix 完整报告。

    对应 Spec Requirement "12 轴 Scientific Validation Matrix"。

    Attributes:
        pathway: 通路标识（如 ``"egfr"``）。
        axes: 12 项 AxisResult 列表（顺序与 ``VALIDATION_AXES`` 一致）。
        overall_status: 综合状态：
            - 任一轴 FAIL → FAIL
            - 无 FAIL 但有 PARTIAL → PARTIAL
            - 全 PASS → PASS
        overall_confidence: 综合 Confidence（0.0-1.0），
            等于各轴 (PASS=1.0, PARTIAL=0.5, FAIL=0.0) 的均值。
        summary: 人类可读总结（含 Overall Status / Confidence / 失败轴列表）。

    提供便捷方法：
        - ``get_axis(name)`` — 按轴名查找 AxisResult
        - ``to_markdown()`` — 渲染为 Markdown 表格
    """

    pathway: str
    axes: list[AxisResult]
    overall_status: ValidationStatus
    overall_confidence: float
    summary: str = ""

    def get_axis(self, name: str) -> Optional[AxisResult]:
        """按轴名查找 AxisResult。

        Args:
            name: 轴名（如 ``"Dynamics Validation"``），大小写敏感，
                需与 ``VALIDATION_AXES`` 完全一致。

        Returns:
            对应的 AxisResult；未找到返回 None。
        """
        for ax in self.axes:
            if ax.axis == name:
                return ax
        return None

    def to_markdown(self) -> str:
        """渲染为 Markdown 表格。

        输出格式：
            ````markdown
            # Validation Matrix Report

            Pathway: egfr

            | # | Axis | Status | Evidence | Reason |
            |---|------|--------|----------|--------|
            | 1 | Mechanism Validation | PASS | ... | ... |
            ...
            | 12 | Benchmark Validation | PASS | ... | ... |

            **Overall Status**: PASS
            **Overall Confidence**: 1.000
            ````

        Returns:
            Markdown 文本（含表头、12 行数据、Overall Status / Confidence 三段）。
        """
        lines: list[str] = [
            "# Validation Matrix Report",
            "",
            f"Pathway: {self.pathway}",
            "",
            "| # | Axis | Status | Evidence | Reason |",
            "|---|------|--------|----------|--------|",
        ]
        for idx, ax in enumerate(self.axes, start=1):
            evidence = _escape_md_cell(ax.evidence)
            reason = _escape_md_cell(ax.reason)
            lines.append(
                f"| {idx} | {ax.axis} | {ax.status.value} | {evidence} | {reason} |"
            )
        lines.append("")
        lines.append(f"**Overall Status**: {self.overall_status.value}")
        lines.append(f"**Overall Confidence**: {self.overall_confidence:.3f}")
        return "\n".join(lines)


# 向后兼容：保留旧 ValidationMatrixReport 类（指向 ValidationMatrixResult 的别名）
ValidationMatrixReport = ValidationMatrixResult


# =============================================================================
# 内部辅助函数
# =============================================================================


def _escape_md_cell(text: str) -> str:
    """转义 Markdown 表格单元格中的特殊字符（换行 / 管道符）。"""
    if not text:
        return ""
    return text.replace("|", "\\|").replace("\n", " ").replace("\r", "")


def _normalize_name(name: str) -> str:
    """节点/物种名归一化：去除非字母数字字符并大写。

    例：``"p-ERK"`` → ``"PERK"``、``"beta_catenin"`` → ``"BETACATENIN"``、
    ``"EGF_EGFR_binding"`` → ``"EGFEGFRBINDING"``。

    Args:
        name: 原始节点/物种名。

    Returns:
        归一化后的字符串（全大写 + 仅字母数字）。
    """
    if not name:
        return ""
    return _NON_ALNUM_RE.sub("", str(name)).upper()


def _load_canonical_yaml(canonical_yaml_path: str) -> dict[str, Any]:
    """安全加载 Canonical Reference YAML。

    使用 ``yaml.safe_load``，禁止 ``unsafe_load``。

    Args:
        canonical_yaml_path: Canonical YAML 文件绝对路径。

    Returns:
        解析后的 dict。

    Raises:
        FileNotFoundError: 文件不存在。
        ValueError: YAML 顶层非 dict。
        yaml.YAMLError: YAML 解析失败。
    """
    path = Path(canonical_yaml_path)
    if not path.is_file():
        raise FileNotFoundError(f"Canonical YAML 不存在: {canonical_yaml_path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(
            f"Canonical YAML 顶层必须为 dict，实际: {type(data).__name__}"
        )
    return data


def _extract_nodes_from_network(network_json: dict[str, Any]) -> set[str]:
    """从 network_json 提取所有节点/物种名（归一化前）。

    兼容多种结构：
        - ``nodes`` / ``species`` / ``node_list`` / ``species_list``: list[str] 或
          list[dict]（取 ``id`` / ``name`` / ``label`` 字段）
        - ``reactions[*].reactants`` / ``products`` / ``modifiers``: 同上

    Args:
        network_json: 通路网络 JSON dict。

    Returns:
        节点名集合（原始字符串，未归一化）。
    """
    nodes: set[str] = set()
    if not isinstance(network_json, dict):
        return nodes

    def _collect_items(items: Any) -> None:
        if not isinstance(items, list):
            return
        for item in items:
            if isinstance(item, str):
                if item.strip():
                    nodes.add(item.strip())
            elif isinstance(item, dict):
                for id_key in ("id", "name", "label", "species"):
                    val = item.get(id_key)
                    if isinstance(val, str) and val.strip():
                        nodes.add(val.strip())
                        break

    # 顶层节点/物种列表
    for key in ("nodes", "species", "node_list", "species_list"):
        _collect_items(network_json.get(key))

    # 反应物 / 产物 / 调节物
    reactions = network_json.get("reactions", [])
    if isinstance(reactions, list):
        for rxn in reactions:
            if not isinstance(rxn, dict):
                continue
            for key in ("reactants", "products", "modifiers"):
                _collect_items(rxn.get(key))

    return nodes


def _extract_species_from_network(network_json: dict[str, Any]) -> set[str]:
    """从 network_json 提取物种名（用于 Ontology 轴）。

    与 _extract_nodes_from_network 类似，但更专注于物种（species）。
    """
    species: set[str] = set()
    if not isinstance(network_json, dict):
        return species

    def _collect(items: Any) -> None:
        if not isinstance(items, list):
            return
        for item in items:
            if isinstance(item, str):
                if item.strip():
                    species.add(item.strip())
            elif isinstance(item, dict):
                for id_key in ("id", "name", "label", "species"):
                    val = item.get(id_key)
                    if isinstance(val, str) and val.strip():
                        species.add(val.strip())
                        break

    for key in ("species", "nodes", "species_list", "node_list"):
        _collect(network_json.get(key))

    reactions = network_json.get("reactions", [])
    if isinstance(reactions, list):
        for rxn in reactions:
            if not isinstance(rxn, dict):
                continue
            for key in ("reactants", "products", "modifiers"):
                _collect(rxn.get(key))

    return species


def _extract_pmids_from_papers(papers: list[dict[str, Any]]) -> list[str]:
    """从检索论文列表提取 PMID（保留顺序，去重）。

    支持字段名：``pmid`` / ``PMID`` / ``id`` / ``pubmed_id``，
    值可为 ``"12345678"`` 或 ``"PMID:12345678"``。

    Args:
        papers: 检索论文列表（dict）。

    Returns:
        PMID 字符串列表（去除 "PMID:" 前缀，仅保留数字）。
    """
    pmids: list[str] = []
    seen: set[str] = set()
    for paper in papers or []:
        if not isinstance(paper, dict):
            continue
        raw: Any = None
        for key in ("pmid", "PMID", "id", "pubmed_id", "pmid_id"):
            if key in paper:
                raw = paper.get(key)
                break
        if raw is None:
            continue
        # 提取数字部分
        if isinstance(raw, str):
            match = re.search(r"(\d{4,})", raw)
            if match:
                pmid_digit = match.group(1)
            else:
                continue
        elif isinstance(raw, int):
            pmid_digit = str(raw)
        else:
            continue
        if pmid_digit not in seen:
            seen.add(pmid_digit)
            pmids.append(pmid_digit)
    return pmids


def _extract_canonical_pmids(canonical_data: dict[str, Any]) -> list[str]:
    """从 Canonical YAML 提取 canonical_reviews 中的 PMID（数字部分）。

    Args:
        canonical_data: Canonical YAML dict。

    Returns:
        Canonical PMID 数字字符串列表（如 ``["7657691", "11743495", ...]``）。
    """
    reviews = canonical_data.get("canonical_reviews") or []
    if not isinstance(reviews, list):
        return []
    pmids: list[str] = []
    seen: set[str] = set()
    for entry in reviews:
        if not isinstance(entry, str):
            continue
        match = re.search(r"(\d{4,})", entry)
        if match:
            digit = match.group(1)
            if digit not in seen:
                seen.add(digit)
                pmids.append(digit)
    return pmids


def _extract_biomd_ids(items: list[Any]) -> list[str]:
    """从字符串列表提取 BIOMD ID（如 ``"BIOMD0000000010"``）。"""
    result: list[str] = []
    seen: set[str] = set()
    for item in items or []:
        if not isinstance(item, str):
            continue
        match = re.search(r"(BIOMD\d+)", item, re.IGNORECASE)
        if match:
            biomd = match.group(1).upper()
            if biomd not in seen:
                seen.add(biomd)
                result.append(biomd)
    return result


def _build_evidence_pool(
    retrieved_papers: list[dict[str, Any]],
    biomodels_matches: list[dict[str, Any]],
    network_json: dict[str, Any],
    simulation_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构建 evidence_graph 所需的 evidence_pool。

    Args:
        retrieved_papers: 检索论文列表。
        biomodels_matches: BioModels 匹配列表。
        network_json: 通路网络 dict（提取机制节点词）。
        simulation_metrics: 仿真指标（用于 Simulation 证据标签），可为 None。

    Returns:
        evidence_pool dict，含 ``pmids`` / ``biomodels`` / ``simulation_metrics`` /
        ``mechanism_nodes`` 四键。
    """
    pmids = _extract_pmids_from_papers(retrieved_papers)
    biomd_ids: list[str] = []
    for match in biomodels_matches or []:
        if not isinstance(match, dict):
            continue
        for key in ("id", "biomd_id", "model_id", "biomodels_id"):
            val = match.get(key)
            if isinstance(val, str):
                found = re.search(r"(BIOMD\d+)", val, re.IGNORECASE)
                if found:
                    biomd = found.group(1).upper()
                    if biomd not in biomd_ids:
                        biomd_ids.append(biomd)
                break
    mechanism_nodes = sorted(_extract_nodes_from_network(network_json))
    return {
        "pmids": pmids,
        "biomodels": biomd_ids,
        "simulation_metrics": dict(simulation_metrics or {}),
        "mechanism_nodes": mechanism_nodes,
    }


def _is_finite_number(value: Any) -> bool:
    """判断值是否为有限数字（int / float，且非 NaN / Inf）。"""
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    try:
        return not math.isnan(float(value)) and not math.isinf(float(value))
    except (TypeError, ValueError):
        return False


# =============================================================================
# 12 轴评估函数（私有）
# =============================================================================


def _evaluate_mechanism_axis(
    canonical_data: dict[str, Any],
    network_json: dict[str, Any],
) -> AxisResult:
    """评估 Mechanism Validation 轴：机制节点链覆盖度。

    判定规则：
        - 从 Canonical YAML 读取 ``canonical_mechanism.required_nodes``
        - 从 network_json 提取所有节点/物种
        - 归一化后计算 matched / missing
        - 关键节点（出现在 ``known_negative_feedback`` 或 required_nodes 首尾）
          缺失 → FAIL
        - 全部匹配 → PASS
        - 部分非关键节点缺失 → PARTIAL

    Args:
        canonical_data: Canonical YAML dict。
        network_json: 通路网络 dict。

    Returns:
        AxisResult。
    """
    axis_name = AXIS_MECHANISM_VALIDATION
    mech = canonical_data.get("canonical_mechanism") or {}
    required_nodes: list[str] = list(mech.get("required_nodes") or [])
    if not required_nodes:
        return AxisResult(
            axis=axis_name,
            status=ValidationStatus.FAIL,
            evidence="canonical YAML 缺少 canonical_mechanism.required_nodes",
            reason="Canonical 未定义必填节点链，无法校验机制覆盖度",
        )

    # 已知负反馈节点（视为关键节点）
    neg_feedback: list[str] = list(canonical_data.get("known_negative_feedback") or [])
    critical_set = { _normalize_name(n) for n in neg_feedback }
    # 节点链首尾也算关键节点
    if len(required_nodes) >= 1:
        critical_set.add(_normalize_name(required_nodes[0]))
    if len(required_nodes) >= 2:
        critical_set.add(_normalize_name(required_nodes[-1]))

    network_nodes = _extract_nodes_from_network(network_json)
    network_nodes_norm = { _normalize_name(n) for n in network_nodes }

    matched: list[str] = []
    missing: list[str] = []
    missing_critical: list[str] = []
    for node in required_nodes:
        norm = _normalize_name(node)
        if norm in network_nodes_norm:
            matched.append(node)
        else:
            missing.append(node)
            if norm in critical_set:
                missing_critical.append(node)

    coverage = len(matched) / len(required_nodes) if required_nodes else 0.0

    if missing_critical:
        status = ValidationStatus.FAIL
        evidence = (
            f"覆盖率 {coverage:.2f}（{len(matched)}/{len(required_nodes)}），"
            f"缺失关键节点: {missing_critical}"
        )
        reason = (
            f"缺失关键节点 {missing_critical}（负反馈/节点链首尾），"
            "机制不完整"
        )
    elif coverage >= 1.0:
        status = ValidationStatus.PASS
        evidence = (
            f"全部 {len(required_nodes)} 个必填节点匹配（coverage=1.00）"
        )
        reason = ""
    elif coverage >= 0.5:
        status = ValidationStatus.PARTIAL
        evidence = (
            f"覆盖率 {coverage:.2f}（{len(matched)}/{len(required_nodes)}），"
            f"缺失非关键节点: {missing}"
        )
        reason = f"部分非关键节点缺失（{len(missing)} 个）"
    else:
        status = ValidationStatus.FAIL
        evidence = (
            f"覆盖率 {coverage:.2f}（{len(matched)}/{len(required_nodes)}），"
            f"缺失: {missing}"
        )
        reason = f"覆盖率过低（< 0.50），机制严重不完整"

    return AxisResult(
        axis=axis_name,
        status=status,
        evidence=evidence,
        reason=reason,
        details={
            "required_count": len(required_nodes),
            "matched_count": len(matched),
            "missing": missing,
            "missing_critical": missing_critical,
            "coverage": coverage,
        },
    )


def _evaluate_ontology_axis(
    canonical_data: dict[str, Any],
    network_json: dict[str, Any],
) -> AxisResult:
    """评估 Ontology Validation 轴：物种命名规范化与 Canonical 命中率。

    判定规则：
        - 从 Canonical 提取 ``canonical_mechanism.required_nodes`` 作为标准命名集
        - 从 network_json 提取所有 species / nodes
        - 归一化后计算 canonical 命中率
        - 全部命中 → PASS；多数命中 → PARTIAL；少数命中 → FAIL

    无 MCP 时使用 canonical species 命中率作为代理指标。

    Args:
        canonical_data: Canonical YAML dict。
        network_json: 通路网络 dict。

    Returns:
        AxisResult。
    """
    axis_name = AXIS_ONTOLOGY_VALIDATION
    mech = canonical_data.get("canonical_mechanism") or {}
    canonical_nodes: list[str] = list(mech.get("required_nodes") or [])
    if not canonical_nodes:
        return AxisResult(
            axis=axis_name,
            status=ValidationStatus.PARTIAL,
            evidence="Canonical 未定义 required_nodes，Ontology 轴降级 PARTIAL",
            reason="无法校验物种命名规范化（Canonical 缺字段）",
        )

    canonical_norm = { _normalize_name(n): n for n in canonical_nodes }
    network_species = _extract_species_from_network(network_json)
    network_species_norm = { _normalize_name(s) for s in network_species }

    matched_canonical = sum(
        1 for norm in canonical_norm if norm in network_species_norm
    )
    hit_rate = matched_canonical / len(canonical_nodes)

    # 关键节点（负反馈 + 节点链首尾）缺失 → FAIL
    neg_feedback = list(canonical_data.get("known_negative_feedback") or [])
    critical_norm = { _normalize_name(n) for n in neg_feedback }
    if len(canonical_nodes) >= 1:
        critical_norm.add(_normalize_name(canonical_nodes[0]))
    if len(canonical_nodes) >= 2:
        critical_norm.add(_normalize_name(canonical_nodes[-1]))

    missing_critical = [
        canonical_nodes[i]
        for i, norm in enumerate(canonical_norm)
        if norm in critical_norm and norm not in network_species_norm
    ]

    if missing_critical:
        status = ValidationStatus.FAIL
        evidence = (
            f"Canonical 物种命中率 {hit_rate:.2f}，"
            f"缺失关键物种: {missing_critical}"
        )
        reason = (
            f"关键物种 {missing_critical} 未在 network_json 中找到，"
            "命名规范化失败"
        )
    elif hit_rate >= 0.9:
        status = ValidationStatus.PASS
        evidence = (
            f"Canonical 物种命中率 {hit_rate:.2f} "
            f"（{matched_canonical}/{len(canonical_nodes)}）"
        )
        reason = ""
    elif hit_rate >= 0.5:
        status = ValidationStatus.PARTIAL
        evidence = (
            f"Canonical 物种命中率 {hit_rate:.2f} "
            f"（{matched_canonical}/{len(canonical_nodes)}）"
        )
        reason = "部分物种命名未对齐 Canonical"
    else:
        status = ValidationStatus.FAIL
        evidence = (
            f"Canonical 物种命中率 {hit_rate:.2f} "
            f"（{matched_canonical}/{len(canonical_nodes)}）"
        )
        reason = "物种命名严重偏离 Canonical"

    return AxisResult(
        axis=axis_name,
        status=status,
        evidence=evidence,
        reason=reason,
        details={
            "canonical_count": len(canonical_nodes),
            "matched_count": matched_canonical,
            "hit_rate": hit_rate,
            "missing_critical": missing_critical,
        },
    )


def _evaluate_literature_axis(
    canonical_data: dict[str, Any],
    retrieved_papers: list[dict[str, Any]],
) -> AxisResult:
    """评估 Literature Validation 轴：Canonical PMID 检索命中率（Top1-Top5）。

    判定规则：
        - 从 Canonical 提取 ``canonical_reviews`` 中的 PMID 列表
        - 从 retrieved_papers 提取 PMID（按出现顺序）
        - 统计 Top1-Top5 中 Canonical PMID 命中数
        - >= 4 命中 → PASS；>= 2 命中 → PARTIAL；否则 FAIL

    Args:
        canonical_data: Canonical YAML dict。
        retrieved_papers: 检索论文列表。

    Returns:
        AxisResult。
    """
    axis_name = AXIS_LITERATURE_VALIDATION
    canonical_pmids = _extract_canonical_pmids(canonical_data)
    canonical_set = set(canonical_pmids)

    if not canonical_pmids:
        return AxisResult(
            axis=axis_name,
            status=ValidationStatus.PARTIAL,
            evidence="Canonical 未定义 canonical_reviews，无法校验命中率",
            reason="Canonical 缺少 canonical_reviews 字段",
        )

    if not retrieved_papers:
        return AxisResult(
            axis=axis_name,
            status=ValidationStatus.FAIL,
            evidence="retrieved_papers 为空，0 个 Canonical 命中",
            reason="未检索到任何文献，命中率 = 0",
        )

    retrieved_pmids = _extract_pmids_from_papers(retrieved_papers)
    top_n = min(_LITERATURE_TOP_N, len(retrieved_pmids))
    top_n_pmids = retrieved_pmids[:top_n]
    canonical_in_top_n = [pmid for pmid in top_n_pmids if pmid in canonical_set]
    canonical_in_all = [pmid for pmid in retrieved_pmids if pmid in canonical_set]

    hits_in_top_n = len(canonical_in_top_n)
    total_hit_rate = (
        len(canonical_in_all) / len(canonical_pmids)
        if canonical_pmids
        else 0.0
    )

    if hits_in_top_n >= _LITERATURE_PASS_HITS:
        status = ValidationStatus.PASS
        evidence = (
            f"Top{top_n} 中 {hits_in_top_n} 个 Canonical PMID 命中 "
            f"（{canonical_in_top_n}），总命中率 {total_hit_rate:.2f}"
        )
        reason = ""
    elif hits_in_top_n >= _LITERATURE_PARTIAL_HITS:
        status = ValidationStatus.PARTIAL
        evidence = (
            f"Top{top_n} 中 {hits_in_top_n} 个 Canonical PMID 命中 "
            f"（{canonical_in_top_n}），总命中率 {total_hit_rate:.2f}"
        )
        reason = (
            f"Top{top_n} 命中数 {hits_in_top_n} 低于 PASS 阈值 "
            f"{_LITERATURE_PASS_HITS}，经典 Review 未充分优先"
        )
    else:
        status = ValidationStatus.FAIL
        evidence = (
            f"Top{top_n} 中仅 {hits_in_top_n} 个 Canonical PMID 命中 "
            f"（{canonical_in_top_n}），总命中率 {total_hit_rate:.2f}"
        )
        reason = (
            f"Top{top_n} 命中数 {hits_in_top_n} 低于 PARTIAL 阈值 "
            f"{_LITERATURE_PARTIAL_HITS}，经典 Review 排序严重失败"
        )

    return AxisResult(
        axis=axis_name,
        status=status,
        evidence=evidence,
        reason=reason,
        details={
            "canonical_pmids": canonical_pmids,
            "retrieved_count": len(retrieved_pmids),
            "top_n": top_n,
            "hits_in_top_n": hits_in_top_n,
            "hits_in_top_n_pmids": canonical_in_top_n,
            "total_hit_rate": total_hit_rate,
        },
    )


def _evaluate_biomodels_axis(
    canonical_data: dict[str, Any],
    biomodels_matches: list[dict[str, Any]],
    biomodels_validation_result: dict[str, Any] | None,
) -> AxisResult:
    """评估 BioModels Validation 轴：Matched ID / Deviation / Missing reactions。

    判定规则：
        - 若 biomodels_validation_result 提供 status 字段，直接映射
        - 否则：
            - 从 Canonical 提取 ``canonical_models``
            - 从 biomodels_matches 提取 BIOMD ID
            - 计算是否匹配 Canonical
            - 若匹配，根据 deviation 字段判定 PASS / PARTIAL / FAIL

    Args:
        canonical_data: Canonical YAML dict。
        biomodels_matches: BioModels 匹配列表。
        biomodels_validation_result: BioModels Oracle 验证结果 dict（可选）。

    Returns:
        AxisResult。
    """
    axis_name = AXIS_BIOMODELS_VALIDATION
    canonical_models = _extract_biomd_ids(
        list(canonical_data.get("canonical_models") or [])
    )

    # 优先使用 biomodels_validation_result（如已运行 BioModels Oracle）
    if biomodels_validation_result is not None:
        bvr_status = str(
            biomodels_validation_result.get("status")
            or biomodels_validation_result.get("overall_status")
            or ""
        ).upper()
        deviation = biomodels_validation_result.get("deviation")
        if deviation is None:
            deviation = biomodels_validation_result.get("overall_distance")
        missing_reactions = biomodels_validation_result.get("missing_reactions")
        parameter_mismatch = biomodels_validation_result.get(
            "parameter_mismatch"
        )
        matched_id = biomodels_validation_result.get("matched_id") or (
            biomodels_validation_result.get("biomodels_id")
            if isinstance(biomodels_validation_result, dict)
            else None
        )

        # 状态映射
        if bvr_status in ("PASS", "PASSED", "PASSED_"):
            status = ValidationStatus.PASS
        elif bvr_status in ("PARTIAL", "DEGRADED"):
            status = ValidationStatus.PARTIAL
        elif bvr_status in ("FAIL", "FAILED", "ERROR"):
            status = ValidationStatus.FAIL
        else:
            # 无显式 status → 按 deviation 判定
            if deviation is None:
                status = ValidationStatus.PARTIAL
            elif _is_finite_number(deviation):
                dev = float(deviation)
                if dev <= _BIOMODELS_PASS_DEVIATION:
                    status = ValidationStatus.PASS
                elif dev <= _BIOMODELS_PARTIAL_DEVIATION:
                    status = ValidationStatus.PARTIAL
                else:
                    status = ValidationStatus.FAIL
            else:
                status = ValidationStatus.PARTIAL

        evidence_parts: list[str] = []
        if matched_id:
            evidence_parts.append(f"Matched ID={matched_id}")
        if deviation is not None:
            evidence_parts.append(f"Deviation={deviation}")
        if missing_reactions is not None:
            evidence_parts.append(f"Missing reactions={missing_reactions}")
        if parameter_mismatch is not None:
            evidence_parts.append(f"Parameter mismatch={parameter_mismatch}")

        reason = ""
        if status == ValidationStatus.FAIL:
            reason = (
                f"BioModels 验证状态 = {bvr_status or 'FAIL'}，"
                "偏差过大或关键反应缺失"
            )
        elif status == ValidationStatus.PARTIAL:
            reason = (
                f"BioModels 验证状态 = {bvr_status or 'PARTIAL'}，"
                "部分偏差超阈值"
            )

        return AxisResult(
            axis=axis_name,
            status=status,
            evidence="；".join(evidence_parts) if evidence_parts else "无明细",
            reason=reason,
            details={
                "matched_id": matched_id,
                "deviation": deviation,
                "missing_reactions": missing_reactions,
                "parameter_mismatch": parameter_mismatch,
                "canonical_models": canonical_models,
            },
        )

    # 未提供 biomodels_validation_result → 从 biomodels_matches 推断
    if not biomodels_matches:
        if not canonical_models:
            return AxisResult(
                axis=axis_name,
                status=ValidationStatus.PARTIAL,
                evidence="Canonical 未定义 canonical_models 且无 biomodels_matches",
                reason="无 BioModels 数据可校验",
            )
        return AxisResult(
            axis=axis_name,
            status=ValidationStatus.FAIL,
            evidence="无 BioModels 匹配结果",
            reason="BioModels Oracle 未运行或未匹配任何模型",
        )

    # 提取所有匹配的 BIOMD ID
    matched_ids: list[str] = []
    for m in biomodels_matches:
        if not isinstance(m, dict):
            continue
        for key in ("id", "biomd_id", "model_id", "biomodels_id"):
            val = m.get(key)
            if isinstance(val, str):
                found = re.search(r"(BIOMD\d+)", val, re.IGNORECASE)
                if found:
                    biomd = found.group(1).upper()
                    if biomd not in matched_ids:
                        matched_ids.append(biomd)
                break

    canonical_set = set(canonical_models)
    matched_canonical = [bid for bid in matched_ids if bid in canonical_set]

    # 取最佳 deviation
    min_deviation: float | None = None
    for m in biomodels_matches:
        if not isinstance(m, dict):
            continue
        dev = m.get("deviation") or m.get("overall_distance")
        if _is_finite_number(dev):
            d = float(dev)
            if min_deviation is None or d < min_deviation:
                min_deviation = d

    if not canonical_set:
        # 无 Canonical 模型 → 仅根据 deviation 判定
        if min_deviation is not None and min_deviation <= _BIOMODELS_PASS_DEVIATION:
            status = ValidationStatus.PASS
        elif min_deviation is not None and min_deviation <= _BIOMODELS_PARTIAL_DEVIATION:
            status = ValidationStatus.PARTIAL
        else:
            status = ValidationStatus.PARTIAL
    else:
        if not matched_canonical:
            status = ValidationStatus.FAIL
        elif min_deviation is not None and min_deviation <= _BIOMODELS_PASS_DEVIATION:
            status = ValidationStatus.PASS
        elif min_deviation is not None and min_deviation <= _BIOMODELS_PARTIAL_DEVIATION:
            status = ValidationStatus.PARTIAL
        else:
            status = ValidationStatus.PARTIAL if matched_canonical else ValidationStatus.FAIL

    evidence_parts = [
        f"Matched IDs={matched_ids}",
        f"Canonical 命中={matched_canonical}",
    ]
    if min_deviation is not None:
        evidence_parts.append(f"Min Deviation={min_deviation:.4f}")

    reason = ""
    if status == ValidationStatus.FAIL:
        reason = "未匹配任何 Canonical BioModels 模型"
    elif status == ValidationStatus.PARTIAL:
        reason = "匹配 Canonical 模型但偏差较大"

    return AxisResult(
        axis=axis_name,
        status=status,
        evidence="；".join(evidence_parts),
        reason=reason,
        details={
            "matched_ids": matched_ids,
            "canonical_models": canonical_models,
            "matched_canonical": matched_canonical,
            "min_deviation": min_deviation,
        },
    )


def _evaluate_parameter_axis(
    parameter_priors: dict[str, Any],
) -> AxisResult:
    """评估 Parameter Validation 轴：参数来源占比。

    判定规则：
        - 统计每个参数的 source 字段（BioModels / LLM / Hardcode / Default / Missing）
        - 计算 grounded 比例（BioModels + Literature + Canonical）
        - 全部 grounded → PASS；多数 grounded → PARTIAL；多数 ungrounded → FAIL

    Args:
        parameter_priors: 参数先验 dict（参数名 → metadata dict）。

    Returns:
        AxisResult。
    """
    axis_name = AXIS_PARAMETER_VALIDATION
    if not parameter_priors:
        return AxisResult(
            axis=axis_name,
            status=ValidationStatus.FAIL,
            evidence="parameter_priors 为空，0 个参数",
            reason="无参数先验数据，无法验证来源",
        )

    total = 0
    source_counts: dict[str, int] = {}
    grounded = 0
    ungrounded = 0
    ungrounded_params: list[str] = []

    for name, meta in parameter_priors.items():
        total += 1
        if isinstance(meta, dict):
            source = str(meta.get("source") or meta.get("provenance") or "missing")
        else:
            source = "missing"
        source_lower = source.lower()

        if source_lower in _PARAM_GROUNDED_SOURCES:
            grounded += 1
            source_counts["grounded"] = source_counts.get("grounded", 0) + 1
        elif source_lower in _PARAM_UNGROUNDED_SOURCES:
            ungrounded += 1
            source_counts["ungrounded"] = source_counts.get("ungrounded", 0) + 1
            ungrounded_params.append(str(name))
        elif source_lower in ("llm",):
            # LLM 来源：若有 reference 字段则视为 grounded，否则 ungrounded
            has_ref = (
                isinstance(meta, dict)
                and bool(meta.get("reference") or meta.get("pmid"))
            )
            if has_ref:
                grounded += 1
                source_counts["llm_with_ref"] = (
                    source_counts.get("llm_with_ref", 0) + 1
                )
            else:
                ungrounded += 1
                source_counts["llm_no_ref"] = (
                    source_counts.get("llm_no_ref", 0) + 1
                )
                ungrounded_params.append(str(name))
        else:
            # 未知来源：视为 ungrounded
            ungrounded += 1
            source_counts["unknown"] = source_counts.get("unknown", 0) + 1
            ungrounded_params.append(str(name))

    grounded_ratio = grounded / total if total else 0.0

    if grounded_ratio >= 0.9:
        status = ValidationStatus.PASS
        evidence = (
            f"参数来源 grounded 比例 {grounded_ratio:.2f} "
            f"（{grounded}/{total}），分布: {source_counts}"
        )
        reason = ""
    elif grounded_ratio >= 0.5:
        status = ValidationStatus.PARTIAL
        evidence = (
            f"参数来源 grounded 比例 {grounded_ratio:.2f} "
            f"（{grounded}/{total}），分布: {source_counts}，"
            f"ungrounded: {ungrounded_params[:5]}"
        )
        reason = f"{ungrounded} 个参数无 grounded 来源"
    else:
        status = ValidationStatus.FAIL
        evidence = (
            f"参数来源 grounded 比例 {grounded_ratio:.2f} "
            f"（{grounded}/{total}），分布: {source_counts}，"
            f"ungrounded: {ungrounded_params[:5]}"
        )
        reason = f"{ungrounded}/{total} 个参数无 grounded 来源"

    return AxisResult(
        axis=axis_name,
        status=status,
        evidence=evidence,
        reason=reason,
        details={
            "total_params": total,
            "grounded": grounded,
            "ungrounded": ungrounded,
            "grounded_ratio": grounded_ratio,
            "source_counts": source_counts,
            "ungrounded_params": ungrounded_params,
        },
    )


def _evaluate_simulation_axis(
    simulation_csv_path: str,
) -> AxisResult:
    """评估 Simulation Validation 轴：仿真是否成功 + 数值稳定。

    判定规则：
        - 文件不存在 → FAIL
        - CSV 为空 / 时间列未找到 → FAIL
        - 任何数值为 NaN / Inf → FAIL
        - 时间点数 < 2 → FAIL
        - 全部正常 → PASS

    Args:
        simulation_csv_path: simulation.csv 路径。

    Returns:
        AxisResult。
    """
    axis_name = AXIS_SIMULATION_VALIDATION

    if not simulation_csv_path or not os.path.isfile(simulation_csv_path):
        return AxisResult(
            axis=axis_name,
            status=ValidationStatus.FAIL,
            evidence=f"simulation CSV 不存在: {simulation_csv_path}",
            reason="仿真未产出 CSV 文件，仿真失败",
        )

    import csv as csv_module

    try:
        with open(simulation_csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv_module.DictReader(f)
            rows = list(reader)
    except Exception as exc:
        return AxisResult(
            axis=axis_name,
            status=ValidationStatus.FAIL,
            evidence=f"CSV 读取失败: {exc}",
            reason="仿真 CSV 文件损坏",
        )

    if not rows:
        return AxisResult(
            axis=axis_name,
            status=ValidationStatus.FAIL,
            evidence="CSV 文件为空",
            reason="仿真未产出数据行",
        )

    headers = list(rows[0].keys())
    # 检查时间列
    time_col: str | None = None
    for candidate in ("time", "t", "Time", "T"):
        if candidate in headers:
            time_col = candidate
            break
    if time_col is None:
        return AxisResult(
            axis=axis_name,
            status=ValidationStatus.FAIL,
            evidence=f"未找到时间列（CSV headers={headers}）",
            reason="仿真 CSV 缺少时间列",
        )

    # 数值稳定性检查
    nan_count = 0
    inf_count = 0
    total_cells = 0
    for row in rows:
        for col in headers:
            val = row.get(col)
            if val is None or val == "":
                continue
            total_cells += 1
            try:
                num = float(val)
                if math.isnan(num):
                    nan_count += 1
                elif math.isinf(num):
                    inf_count += 1
            except (TypeError, ValueError):
                # 非数值（如时间标签）跳过
                continue

    n_points = len(rows)
    if nan_count > 0 or inf_count > 0:
        return AxisResult(
            axis=axis_name,
            status=ValidationStatus.FAIL,
            evidence=(
                f"CSV 含 {nan_count} 个 NaN + {inf_count} 个 Inf "
                f"（{n_points} 行）"
            ),
            reason="仿真数值不稳定（含 NaN / Inf）",
        )

    if n_points < _SIMULATION_MIN_POINTS:
        return AxisResult(
            axis=axis_name,
            status=ValidationStatus.FAIL,
            evidence=f"CSV 仅 {n_points} 行（< {_SIMULATION_MIN_POINTS}）",
            reason="仿真时间点数不足",
        )

    return AxisResult(
        axis=axis_name,
        status=ValidationStatus.PASS,
        evidence=(
            f"CSV 正常：{n_points} 行 × {len(headers)} 列，"
            f"时间列={time_col}，无 NaN / Inf"
        ),
        reason="",
        details={
            "row_count": n_points,
            "column_count": len(headers),
            "time_column": time_col,
            "nan_count": nan_count,
            "inf_count": inf_count,
        },
    )


def _evaluate_dynamics_axis(
    canonical_data: dict[str, Any],
    simulation_csv_path: str,
) -> AxisResult:
    """评估 Dynamics Validation 轴：8 项 Curve Metrics 对照 expected_dynamics。

    判定规则：
        - 从 Canonical 读取 ``expected_dynamics``
        - 对每个 molecule 调用 ``compute_all_curves`` 计算 CurveMetrics
        - 用 ``compare_with_expected`` 对照 peak_time
        - 自定义检查 must_decline / behavior
        - 关键指标（peak_time / must_decline）FAIL → 轴 FAIL
        - 非关键指标（behavior）FAIL → 轴 PARTIAL
        - 全部 PASS → 轴 PASS

    Args:
        canonical_data: Canonical YAML dict。
        simulation_csv_path: simulation.csv 路径。

    Returns:
        AxisResult。
    """
    axis_name = AXIS_DYNAMICS_VALIDATION

    expected_dynamics = canonical_data.get("expected_dynamics") or {}
    if not isinstance(expected_dynamics, dict) or not expected_dynamics:
        return AxisResult(
            axis=axis_name,
            status=ValidationStatus.FAIL,
            evidence="Canonical 缺少 expected_dynamics 字段",
            reason="无法对照期望动力学（Canonical 缺字段）",
        )

    if not simulation_csv_path or not os.path.isfile(simulation_csv_path):
        return AxisResult(
            axis=axis_name,
            status=ValidationStatus.FAIL,
            evidence=f"simulation CSV 不存在: {simulation_csv_path}",
            reason="仿真未产出 CSV 文件，无法计算 Curve Metrics",
        )

    expected_molecules = list(expected_dynamics.keys())

    # 调用 Task 2 的 compute_all_curves 计算多条曲线指标
    try:
        all_metrics = curve_metrics.compute_all_curves(
            simulation_csv_path, expected_molecules
        )
    except Exception as exc:
        return AxisResult(
            axis=axis_name,
            status=ValidationStatus.FAIL,
            evidence=f"compute_all_curves 失败: {exc}",
            reason="无法从仿真 CSV 计算曲线指标",
        )

    total_checks = 0
    passed_checks = 0
    failed_critical = 0
    failed_non_critical = 0
    failure_details: list[str] = []
    metric_table: list[dict[str, Any]] = []

    for mol, expected_spec in expected_dynamics.items():
        expected_spec = expected_spec or {}
        if mol not in all_metrics:
            total_checks += 1
            failed_critical += 1
            failure_details.append(f"{mol}: 不在仿真 CSV 中")
            metric_table.append({
                "molecule": mol,
                "metric": "(missing)",
                "expected": str(expected_spec),
                "actual": "None",
                "status": "FAIL",
                "reason": "分子不在 CSV 中",
            })
            continue

        actual_metrics = all_metrics[mol]

        # 1. peak_time 检查（关键）
        if "peak_time" in expected_spec:
            total_checks += 1
            expected_range = expected_spec["peak_time"]
            try:
                em = curve_metrics.ExpectedMetric(
                    metric_name="peak_time",
                    expected_range=expected_range,
                )
                comp = curve_metrics.compare_with_expected(actual_metrics, em)
            except Exception as exc:
                comp = None
                failure_details.append(
                    f"{mol}.peak_time 对照异常: {exc}"
                )

            if comp is None:
                failed_critical += 1
            elif comp.status == curve_metrics.MetricStatus.PASS:
                passed_checks += 1
                metric_table.append({
                    "molecule": mol,
                    "metric": "peak_time",
                    "expected": str(expected_range),
                    "actual": str(comp.actual),
                    "status": "PASS",
                    "reason": comp.reason,
                })
            else:
                failed_critical += 1
                failure_details.append(
                    f"{mol}.peak_time: actual={comp.actual}, "
                    f"expected={comp.expected} ({comp.reason})"
                )
                metric_table.append({
                    "molecule": mol,
                    "metric": "peak_time",
                    "expected": str(expected_range),
                    "actual": str(comp.actual),
                    "status": "FAIL",
                    "reason": comp.reason,
                })

        # 2. must_decline 检查（关键）
        if expected_spec.get("must_decline"):
            total_checks += 1
            half_decay = actual_metrics.half_decay
            adapt_ratio = actual_metrics.adaptation_ratio
            # must_decline 通过条件：half_decay 不为 None 且 adaptation_ratio < 0.5
            if half_decay is not None and adapt_ratio < 0.5:
                passed_checks += 1
                metric_table.append({
                    "molecule": mol,
                    "metric": "must_decline",
                    "expected": "True (half_decay not None + adapt_ratio < 0.5)",
                    "actual": (
                        f"half_decay={half_decay:.3f}, "
                        f"adapt_ratio={adapt_ratio:.3f}"
                    ),
                    "status": "PASS",
                    "reason": "",
                })
            else:
                failed_critical += 1
                failure_details.append(
                    f"{mol}.must_decline: half_decay={half_decay}, "
                    f"adaptation_ratio={adapt_ratio:.3f}（期望下降）"
                )
                metric_table.append({
                    "molecule": mol,
                    "metric": "must_decline",
                    "expected": "True",
                    "actual": (
                        f"half_decay={half_decay}, "
                        f"adapt_ratio={adapt_ratio:.3f}"
                    ),
                    "status": "FAIL",
                    "reason": "未观察到峰值后下降",
                })

        # 3. behavior 趋势检查（非关键）
        if "behavior" in expected_spec:
            total_checks += 1
            behavior = str(expected_spec["behavior"]).lower()
            trend_ok = False
            trend_reason = ""
            if "increasing" in behavior or "持续增加" in behavior:
                trend_ok = actual_metrics.trend_slope > 0
                trend_reason = (
                    f"trend_slope={actual_metrics.trend_slope:.4f} > 0"
                    if trend_ok
                    else f"trend_slope={actual_metrics.trend_slope:.4f} ≤ 0"
                )
            elif "decline" in behavior or "decreasing" in behavior:
                trend_ok = (
                    actual_metrics.trend_slope < 0
                    or actual_metrics.half_decay is not None
                )
                trend_reason = (
                    f"trend_slope={actual_metrics.trend_slope:.4f}, "
                    f"half_decay={actual_metrics.half_decay}"
                )
            elif "transient" in behavior:
                trend_ok = actual_metrics.half_decay is not None
                trend_reason = (
                    f"half_decay={actual_metrics.half_decay}（transient 应有下降）"
                )
            elif "rapid" in behavior:
                # 快速上升 + 下降
                trend_ok = (
                    actual_metrics.peak_time is not None
                    and actual_metrics.peak_time < 30.0
                    and actual_metrics.half_decay is not None
                )
                trend_reason = (
                    f"peak_time={actual_metrics.peak_time:.2f}, "
                    f"half_decay={actual_metrics.half_decay}"
                )

            if trend_ok:
                passed_checks += 1
                metric_table.append({
                    "molecule": mol,
                    "metric": "behavior",
                    "expected": str(expected_spec["behavior"]),
                    "actual": trend_reason,
                    "status": "PASS",
                    "reason": "",
                })
            else:
                failed_non_critical += 1
                failure_details.append(
                    f"{mol}.behavior: expected={expected_spec['behavior']}, "
                    f"actual {trend_reason}"
                )
                metric_table.append({
                    "molecule": mol,
                    "metric": "behavior",
                    "expected": str(expected_spec["behavior"]),
                    "actual": trend_reason,
                    "status": "FAIL",
                    "reason": "趋势不匹配",
                })

    # 判定状态
    if failed_critical > 0:
        status = ValidationStatus.FAIL
        evidence = (
            f"{failed_critical} 项关键指标 FAIL（共 {total_checks} 项检查，"
            f"{passed_checks} PASS / {failed_critical} 关键 FAIL / "
            f"{failed_non_critical} 非关键 FAIL）"
        )
        if failure_details:
            evidence += "；失败明细: " + "; ".join(failure_details[:3])
        reason = (
            f"关键动力学指标偏离 Expected（{failed_critical} 项关键 FAIL），"
            "如 ERK Peak 45 min vs Expected 10-20 min"
        )
    elif failed_non_critical > 0:
        status = ValidationStatus.PARTIAL
        evidence = (
            f"{failed_non_critical} 项非关键指标 FAIL（共 {total_checks} 项检查，"
            f"{passed_checks} PASS / {failed_non_critical} 非关键 FAIL）"
        )
        if failure_details:
            evidence += "；失败明细: " + "; ".join(failure_details[:3])
        reason = "部分非关键指标偏离 Expected"
    else:
        status = ValidationStatus.PASS
        evidence = (
            f"全部 {total_checks} 项指标 PASS（覆盖 {len(expected_molecules)} 个分子）"
        )
        reason = ""

    return AxisResult(
        axis=axis_name,
        status=status,
        evidence=evidence,
        reason=reason,
        details={
            "expected_molecules": expected_molecules,
            "computed_molecules": list(all_metrics.keys()),
            "total_checks": total_checks,
            "passed": passed_checks,
            "failed_critical": failed_critical,
            "failed_non_critical": failed_non_critical,
            "failures": failure_details,
            "metric_table": metric_table,
        },
    )


def _evaluate_experiment_axis(
    canonical_data: dict[str, Any],
    experiment_plan: list[dict[str, Any]],
) -> AxisResult:
    """评估 Experiment Validation 轴：机制驱动 + Forbidden 实验拦截。

    判定规则：
        - 实验必须 mechanism-driven（含 ``mechanism_node`` / ``mechanism_target`` 字段）
        - Forbidden 实验（Canonical ``forbidden_experiments``）出现 → FAIL
        - 全部机制驱动且无 forbidden → PASS
        - 部分缺 mechanism 字段 → PARTIAL

    Args:
        canonical_data: Canonical YAML dict。
        experiment_plan: 实验列表。

    Returns:
        AxisResult。
    """
    axis_name = AXIS_EXPERIMENT_VALIDATION
    forbidden = list(canonical_data.get("forbidden_experiments") or [])
    forbidden_lower = [str(f).lower() for f in forbidden]

    if not experiment_plan:
        return AxisResult(
            axis=axis_name,
            status=ValidationStatus.FAIL,
            evidence="experiment_plan 为空",
            reason="无实验计划，无法验证 adaptive dynamics",
        )

    total = len(experiment_plan)
    mechanism_linked = 0
    forbidden_found: list[str] = []

    for exp in experiment_plan:
        if not isinstance(exp, dict):
            continue
        # 检查是否 mechanism-driven
        has_mechanism = bool(
            exp.get("mechanism_node")
            or exp.get("mechanism_target")
            or exp.get("mechanism")
        )
        if has_mechanism:
            mechanism_linked += 1

        # 检查 forbidden
        exp_name = str(
            exp.get("name")
            or exp.get("experiment")
            or exp.get("id")
            or ""
        ).lower()
        for fb in forbidden_lower:
            if fb and fb in exp_name:
                forbidden_found.append(exp_name)
                break

    if forbidden_found:
        return AxisResult(
            axis=axis_name,
            status=ValidationStatus.FAIL,
            evidence=(
                f"发现 {len(forbidden_found)} 个 forbidden 实验: "
                f"{forbidden_found}（Canonical 禁止: {forbidden}）"
            ),
            reason=(
                "包含 forbidden 实验（不能验证 adaptive dynamics），"
                "如 EGFR qPCR / EGF ELISA"
            ),
            details={
                "total_experiments": total,
                "mechanism_linked": mechanism_linked,
                "forbidden_found": forbidden_found,
                "forbidden_defined": forbidden,
            },
        )

    if mechanism_linked == total:
        status = ValidationStatus.PASS
        evidence = (
            f"全部 {total} 个实验均机制驱动，无 forbidden 实验"
        )
        reason = ""
    elif mechanism_linked >= total / 2.0:
        status = ValidationStatus.PARTIAL
        evidence = (
            f"{mechanism_linked}/{total} 个实验机制驱动，无 forbidden 实验"
        )
        reason = f"{total - mechanism_linked} 个实验缺 mechanism 字段"
    else:
        status = ValidationStatus.FAIL
        evidence = (
            f"仅 {mechanism_linked}/{total} 个实验机制驱动（< 50%）"
        )
        reason = "多数实验未关联机制节点"

    return AxisResult(
        axis=axis_name,
        status=status,
        evidence=evidence,
        reason=reason,
        details={
            "total_experiments": total,
            "mechanism_linked": mechanism_linked,
            "forbidden_found": forbidden_found,
            "forbidden_defined": forbidden,
        },
    )


def _evaluate_evidence_attribution_axis(
    report_md: str,
    evidence_pool: dict[str, Any],
) -> AxisResult:
    """评估 Evidence Attribution Validation 轴：逐句证据标注。

    判定规则：
        - 调用 ``evidence_graph.build_from_report(report_md, evidence_pool)``
        - 调用 ``detect_ungrounded(graph)`` → ungrounded 句子数
        - 调用 ``detect_undergrounded(graph)`` → undergrounded 句子数
        - ungrounded > 0 → FAIL（无证据）
        - undergrounded > 0 且 ungrounded == 0 → PARTIAL（仅 [D]Mechanism 无外部证据支撑）
        - 全部 grounded → PASS

    Args:
        report_md: Report Markdown 文本。
        evidence_pool: 证据池 dict。

    Returns:
        AxisResult。
    """
    axis_name = AXIS_EVIDENCE_ATTRIBUTION_VALIDATION

    if not report_md:
        return AxisResult(
            axis=axis_name,
            status=ValidationStatus.FAIL,
            evidence="report_md 为空，无句子可校验",
            reason="报告为空，无证据标注",
        )

    # 调用 Task 3 的 evidence_graph
    try:
        graph = evidence_graph.build_from_report(report_md, evidence_pool)
        ungrounded_nodes = evidence_graph.detect_ungrounded(graph)
        undergrounded_nodes = evidence_graph.detect_undergrounded(graph)
    except Exception as exc:
        return AxisResult(
            axis=axis_name,
            status=ValidationStatus.FAIL,
            evidence=f"evidence_graph 构建异常: {exc}",
            reason="Evidence Graph 构建失败",
        )

    ungrounded_count = len(ungrounded_nodes)
    undergrounded_count = len(undergrounded_nodes)
    total_sentences = len(graph.nodes)

    if ungrounded_count > 0:
        status = ValidationStatus.FAIL
        ungrounded_texts = [
            n.text[:60] for n in ungrounded_nodes[:3]
        ]
        evidence = (
            f"ungrounded 句子 {ungrounded_count}/{total_sentences}，"
            f"示例: {ungrounded_texts}"
        )
        reason = (
            f"{ungrounded_count} 句无任何证据标注（Hallucination Risk=High）"
        )
    elif undergrounded_count > 0:
        status = ValidationStatus.PARTIAL
        undergrounded_texts = [
            n.text[:60] for n in undergrounded_nodes[:3]
        ]
        evidence = (
            f"undergrounded 句子 {undergrounded_count}/{total_sentences}"
            f"（仅 [D]Mechanism 无 [A][B][C] 支撑），示例: {undergrounded_texts}"
        )
        reason = (
            f"{undergrounded_count} 句仅含 [D]Mechanism 而无 [A][B][C] 支撑，"
            "断言不充分"
        )
    else:
        status = ValidationStatus.PASS
        evidence = (
            f"全部 {total_sentences} 句均有 grounded 证据标注"
        )
        reason = ""

    return AxisResult(
        axis=axis_name,
        status=status,
        evidence=evidence,
        reason=reason,
        details={
            "total_sentences": total_sentences,
            "ungrounded_count": ungrounded_count,
            "undergrounded_count": undergrounded_count,
            "ungrounded_sentences": [n.text for n in ungrounded_nodes],
            "undergrounded_sentences": [n.text for n in undergrounded_nodes],
        },
    )


def _evaluate_scientific_writing_axis(
    report_md: str,
) -> AxisResult:
    """评估 Scientific Writing Validation 轴：Honesty 检测。

    判定规则：
        - 调用 ``scientific_honesty.review_report(report_md)``
        - 任何 overstatement 或 citation_missing 违规 → FAIL
        - 仅 unlabeled_claim 违规 → PARTIAL
        - 无违规 → PASS

    Args:
        report_md: Report Markdown 文本。

    Returns:
        AxisResult。
    """
    axis_name = AXIS_SCIENTIFIC_WRITING_VALIDATION

    if not report_md:
        return AxisResult(
            axis=axis_name,
            status=ValidationStatus.FAIL,
            evidence="report_md 为空，无法检测 Honesty",
            reason="报告为空，无声明可校验",
        )

    # 调用 Task 4 的 scientific_honesty
    try:
        review = scientific_honesty.review_report(report_md)
    except Exception as exc:
        return AxisResult(
            axis=axis_name,
            status=ValidationStatus.FAIL,
            evidence=f"scientific_honesty.review_report 异常: {exc}",
            reason="Honesty 检测执行失败",
        )

    violations = list(review.violations)
    overstatement_count = sum(
        1 for d in review.details
        if d.get("violation_type") == "overstatement"
    )
    citation_missing_count = sum(
        1 for d in review.details
        if d.get("violation_type") == "citation_missing"
    )
    unlabeled_count = sum(
        1 for d in review.details
        if d.get("violation_type") == "unlabeled_claim"
    )

    if "overstatement" in violations or "citation_missing" in violations:
        status = ValidationStatus.FAIL
        evidence = (
            f"Honesty 违规: overstatement={overstatement_count}, "
            f"citation_missing={citation_missing_count}, "
            f"unlabeled_claim={unlabeled_count}（status={review.status}）"
        )
        if review.details:
            evidence += (
                "；示例: "
                + str(review.details[0].get("sentence", ""))[:80]
            )
        reason = (
            "存在过度声明或引用缺失违规，违反 Scientific Honesty 准则"
        )
    elif "unlabeled_claim" in violations:
        status = ValidationStatus.PARTIAL
        evidence = (
            f"Honesty 违规: unlabeled_claim={unlabeled_count} "
            f"（status={review.status}）"
        )
        if review.details:
            evidence += (
                "；示例: "
                + str(review.details[0].get("sentence", ""))[:80]
            )
        reason = f"{unlabeled_count} 句未标注四类标签"
    else:
        status = ValidationStatus.PASS
        evidence = (
            f"无 Honesty 违规（status={review.status}，"
            f"共检查 {len(review.details)} 句）"
        )
        reason = ""

    return AxisResult(
        axis=axis_name,
        status=status,
        evidence=evidence,
        reason=reason,
        details={
            "status": review.status,
            "violations": violations,
            "overstatement_count": overstatement_count,
            "citation_missing_count": citation_missing_count,
            "unlabeled_claim_count": unlabeled_count,
            "details": review.details,
        },
    )


def _evaluate_reproducibility_axis(
    reproducibility_metadata: dict[str, Any] | None,
) -> AxisResult:
    """评估 Reproducibility Validation 轴：随机种子 / 参数快照 / ODE 模板。

    判定规则：
        - None → FAIL（无可复现性元数据）
        - 全部三个字段存在且非空 → PASS
        - 部分缺失 → PARTIAL

    Args:
        reproducibility_metadata: 可复现性元数据 dict，需含
            ``random_seed`` / ``parameter_snapshot`` / ``ode_template``。

    Returns:
        AxisResult。
    """
    axis_name = AXIS_REPRODUCIBILITY_VALIDATION

    if reproducibility_metadata is None:
        return AxisResult(
            axis=axis_name,
            status=ValidationStatus.FAIL,
            evidence="reproducibility_metadata 为 None",
            reason="无可复现性元数据，无法保证结果可复现",
        )

    if not isinstance(reproducibility_metadata, dict):
        return AxisResult(
            axis=axis_name,
            status=ValidationStatus.FAIL,
            evidence=f"reproducibility_metadata 类型错误: {type(reproducibility_metadata).__name__}",
            reason="可复现性元数据格式错误",
        )

    present_fields: list[str] = []
    missing_fields: list[str] = []
    for field_name in _REPRODUCIBILITY_REQUIRED_FIELDS:
        val = reproducibility_metadata.get(field_name)
        # random_seed=0 也算有效（不要用 falsy 判断）
        if val is not None and val != "":
            present_fields.append(field_name)
        else:
            missing_fields.append(field_name)

    if not missing_fields:
        status = ValidationStatus.PASS
        evidence = (
            f"全部 {len(_REPRODUCIBILITY_REQUIRED_FIELDS)} 个复现性字段存在: "
            f"{present_fields}"
        )
        reason = ""
    elif present_fields:
        status = ValidationStatus.PARTIAL
        evidence = (
            f"复现性字段 {len(present_fields)}/{len(_REPRODUCIBILITY_REQUIRED_FIELDS)} "
            f"存在（缺失: {missing_fields}）"
        )
        reason = f"缺失 {len(missing_fields)} 个复现性字段: {missing_fields}"
    else:
        status = ValidationStatus.FAIL
        evidence = (
            f"全部 {len(_REPRODUCIBILITY_REQUIRED_FIELDS)} 个复现性字段均缺失"
        )
        reason = "无可复现性信息，结果不可复现"

    return AxisResult(
        axis=axis_name,
        status=status,
        evidence=evidence,
        reason=reason,
        details={
            "present_fields": present_fields,
            "missing_fields": missing_fields,
            "random_seed": reproducibility_metadata.get("random_seed"),
            "has_parameter_snapshot": bool(
                reproducibility_metadata.get("parameter_snapshot")
            ),
            "has_ode_template": bool(
                reproducibility_metadata.get("ode_template")
            ),
        },
    )


def _evaluate_benchmark_axis(
    benchmark_result: dict[str, Any] | None,
) -> AxisResult:
    """评估 Benchmark Validation 轴：通路 Golden Benchmark 是否 PASS。

    判定规则：
        - None → PARTIAL（无 Benchmark 数据可校验）
        - 提供 ``status`` / ``overall_status`` / ``passed`` 字段 → 直接映射
        - 提供 ``acceptance_criteria`` → 按通过率判定
        - 否则 PARTIAL

    Args:
        benchmark_result: Benchmark 结果 dict。

    Returns:
        AxisResult。
    """
    axis_name = AXIS_BENCHMARK_VALIDATION

    if benchmark_result is None:
        return AxisResult(
            axis=axis_name,
            status=ValidationStatus.PARTIAL,
            evidence="benchmark_result 为 None",
            reason="未运行 Benchmark，无法校验收口准则",
        )

    if not isinstance(benchmark_result, dict):
        return AxisResult(
            axis=axis_name,
            status=ValidationStatus.PARTIAL,
            evidence=f"benchmark_result 类型错误: {type(benchmark_result).__name__}",
            reason="Benchmark 结果格式错误",
        )

    # 优先看 status / overall_status / passed
    bvr_status = str(
        benchmark_result.get("status")
        or benchmark_result.get("overall_status")
        or ""
    ).upper()
    passed_flag = benchmark_result.get("passed")
    if passed_flag is None:
        passed_flag = benchmark_result.get("is_passed")

    # acceptance_criteria 列表
    criteria = (
        benchmark_result.get("acceptance_criteria")
        or benchmark_result.get("criteria")
        or []
    )
    if isinstance(criteria, list):
        total_criteria = len(criteria)
        passed_criteria = sum(
            1 for c in criteria
            if isinstance(c, dict) and (
                c.get("passed")
                or c.get("status") == "PASS"
                or c.get("status") == "passed"
            )
        )
    else:
        total_criteria = 0
        passed_criteria = 0

    # 状态映射
    if bvr_status in ("PASS", "PASSED"):
        status = ValidationStatus.PASS
    elif bvr_status in ("FAIL", "FAILED", "ERROR"):
        status = ValidationStatus.FAIL
    elif bvr_status == "PARTIAL":
        status = ValidationStatus.PARTIAL
    elif passed_flag is True:
        status = ValidationStatus.PASS
    elif passed_flag is False:
        status = ValidationStatus.FAIL
    elif total_criteria > 0:
        pass_ratio = passed_criteria / total_criteria
        if pass_ratio >= 0.9:
            status = ValidationStatus.PASS
        elif pass_ratio >= 0.5:
            status = ValidationStatus.PARTIAL
        else:
            status = ValidationStatus.FAIL
    else:
        status = ValidationStatus.PARTIAL

    evidence_parts: list[str] = []
    if bvr_status:
        evidence_parts.append(f"status={bvr_status}")
    if passed_flag is not None:
        evidence_parts.append(f"passed={passed_flag}")
    if total_criteria > 0:
        evidence_parts.append(
            f"acceptance_criteria={passed_criteria}/{total_criteria}"
        )

    reason = ""
    if status == ValidationStatus.FAIL:
        reason = "Benchmark acceptance criteria 未通过"
    elif status == ValidationStatus.PARTIAL:
        reason = "Benchmark 部分通过或未运行"

    return AxisResult(
        axis=axis_name,
        status=status,
        evidence="；".join(evidence_parts) if evidence_parts else "无明细",
        reason=reason,
        details={
            "status": bvr_status,
            "passed": passed_flag,
            "total_criteria": total_criteria,
            "passed_criteria": passed_criteria,
        },
    )


# =============================================================================
# 聚合函数
# =============================================================================


def _aggregate_status(axes: list[AxisResult]) -> ValidationStatus:
    """聚合 12 轴状态为整体状态。

    规则：
        - 任一 FAIL → FAIL
        - 无 FAIL 但有 PARTIAL → PARTIAL
        - 全 PASS → PASS

    Args:
        axes: 12 项 AxisResult 列表。

    Returns:
        综合 ValidationStatus。
    """
    has_fail = any(ax.status == ValidationStatus.FAIL for ax in axes)
    has_partial = any(ax.status == ValidationStatus.PARTIAL for ax in axes)
    if has_fail:
        return ValidationStatus.FAIL
    if has_partial:
        return ValidationStatus.PARTIAL
    return ValidationStatus.PASS


def _calculate_confidence(axes: list[AxisResult]) -> float:
    """计算综合 Confidence（各轴 PASS/PARTIAL/FAIL 均值）。

    公式：``mean(PASS=1.0, PARTIAL=0.5, FAIL=0.0)``

    Args:
        axes: 12 项 AxisResult 列表。

    Returns:
        综合 Confidence（0.0-1.0）。
    """
    if not axes:
        return 0.0
    weights = {
        ValidationStatus.PASS: 1.0,
        ValidationStatus.PARTIAL: 0.5,
        ValidationStatus.FAIL: 0.0,
    }
    total = sum(weights.get(ax.status, 0.0) for ax in axes)
    return total / len(axes)


def _build_summary(
    pathway: str,
    axes: list[AxisResult],
    overall_status: ValidationStatus,
    overall_confidence: float,
) -> str:
    """生成人类可读总结。

    Args:
        pathway: 通路标识。
        axes: 12 项 AxisResult 列表。
        overall_status: 综合状态。
        overall_confidence: 综合 Confidence。

    Returns:
        总结字符串。
    """
    failed_axes = [ax.axis for ax in axes if ax.status == ValidationStatus.FAIL]
    partial_axes = [
        ax.axis for ax in axes if ax.status == ValidationStatus.PARTIAL
    ]
    pass_count = sum(
        1 for ax in axes if ax.status == ValidationStatus.PASS
    )

    parts: list[str] = [
        f"Pathway {pathway} Validation Matrix:",
        f"  Overall Status: {overall_status.value}",
        f"  Overall Confidence: {overall_confidence:.3f}",
        f"  Pass: {pass_count}/{len(axes)}",
    ]
    if failed_axes:
        parts.append(f"  Failed Axes: {failed_axes}")
    if partial_axes:
        parts.append(f"  Partial Axes: {partial_axes}")
    return "\n".join(parts)


# =============================================================================
# 主入口函数
# =============================================================================


def run_validation_matrix(
    pathway: str,
    question: str,
    simulation_csv_path: str,
    report_md: str,
    network_json: dict,
    parameter_priors: dict,
    retrieved_papers: list[dict],
    biomodels_matches: list[dict],
    experiment_plan: list[dict],
    canonical_yaml_path: str,
    biomodels_validation_result: Optional[dict] = None,
    benchmark_result: Optional[dict] = None,
    reproducibility_metadata: Optional[dict] = None,
    # [BENCHMARK CLOSURE / Gap 3] 新增：扁平化仿真指标 dict
    # 供 _build_evidence_pool 构建 evidence_pool.simulation_metrics，
    # 使 Evidence Attribution 轴可检测 [C]Simulation 句子。
    # 旧实现：_build_evidence_pool 硬编码 simulation_metrics=None，
    # 导致 evidence_graph._detect_simulation_metric 永远返回 None →
    # Evidence 轴 grounded_sentence_count=0。
    simulation_metrics_flat: Optional[dict] = None,
) -> ValidationMatrixResult:
    """运行 12 轴 Validation Matrix。

    对应 Spec Requirement "12 轴 Scientific Validation Matrix"。
    所有轴必须实际执行（禁止 placeholder / NotImplementedError）。

    Feature Flag 守护：
        ``V4_SCIENTIFIC_REVIEWER_ENABLED = false`` 时抛出 RuntimeError，
        提示调用方回退到 ``seven_axis_validator.py``。

    Args:
        pathway: 通路标识（如 ``"egfr"``），用于报告标识。
        question: 用户问题文本（保留接口，目前未直接参与 12 轴判定）。
        simulation_csv_path: simulation.csv 文件路径，用于 Simulation 与 Dynamics 轴。
        report_md: Report Markdown 文本，用于 Evidence Attribution 与 Scientific Writing 轴。
        network_json: 通路网络 dict，用于 Mechanism 与 Ontology 轴。
        parameter_priors: 参数先验 dict（参数名 → metadata），用于 Parameter 轴。
        retrieved_papers: 检索论文列表，用于 Literature 轴与 evidence_pool 构建。
        biomodels_matches: BioModels 匹配列表，用于 BioModels 轴与 evidence_pool 构建。
        experiment_plan: 实验推荐列表，用于 Experiment 轴。
        canonical_yaml_path: Canonical YAML 文件路径，提供 12 轴所需的 Expected 数据。
        biomodels_validation_result: BioModels Oracle 验证结果 dict（可选）。
        benchmark_result: Benchmark 结果 dict（可选）。
        reproducibility_metadata: 可复现性元数据 dict（可选）。

    Returns:
        ValidationMatrixResult，含 12 项 AxisResult 与综合状态。

    Raises:
        RuntimeError: ``V4_SCIENTIFIC_REVIEWER_ENABLED = false`` 时抛出。
    """
    # -------------------------------------------------------------------------
    # Feature Flag 守护
    # -------------------------------------------------------------------------
    if not settings.V4_SCIENTIFIC_REVIEWER_ENABLED:
        raise RuntimeError(
            "V4_SCIENTIFIC_REVIEWER_ENABLED is false; "
            "legacy seven_axis_validator should be used instead"
        )

    # -------------------------------------------------------------------------
    # 加载 Canonical YAML
    # -------------------------------------------------------------------------
    try:
        canonical_data = _load_canonical_yaml(canonical_yaml_path)
    except Exception as exc:
        # Canonical 加载失败 → 所有依赖 Canonical 的轴 FAIL
        logger.error("加载 Canonical YAML 失败: %s", exc)
        canonical_data = {}

    # 构建 evidence_pool（供 Evidence Attribution 轴使用）
    # [BENCHMARK CLOSURE / Gap 3] 修复：simulation_metrics 从 None 改为实际值，
    # 使 evidence_graph._detect_simulation_metric 可识别 [C]Simulation 句子。
    # 优先使用调用方提供的扁平化指标（orchestrator 已 _flatten_metrics），
    # 否则降级为空 dict（保留旧行为，避免破坏现有调用方）。
    evidence_pool = _build_evidence_pool(
        retrieved_papers=retrieved_papers,
        biomodels_matches=biomodels_matches,
        network_json=network_json,
        simulation_metrics=simulation_metrics_flat if simulation_metrics_flat is not None else {},
    )

    # -------------------------------------------------------------------------
    # 逐轴评估（每轴独立 try-except，单轴异常不阻塞其他轴）
    # -------------------------------------------------------------------------
    axes: list[AxisResult] = []

    # 1. Mechanism Validation
    try:
        axes.append(_evaluate_mechanism_axis(canonical_data, network_json))
    except Exception as exc:
        logger.error("Mechanism 轴意外异常: %s", exc)
        axes.append(AxisResult(
            axis=AXIS_MECHANISM_VALIDATION,
            status=ValidationStatus.FAIL,
            evidence=f"轴评估异常: {exc}",
            reason="Mechanism 轴执行异常",
        ))

    # 2. Ontology Validation
    try:
        axes.append(_evaluate_ontology_axis(canonical_data, network_json))
    except Exception as exc:
        logger.error("Ontology 轴意外异常: %s", exc)
        axes.append(AxisResult(
            axis=AXIS_ONTOLOGY_VALIDATION,
            status=ValidationStatus.FAIL,
            evidence=f"轴评估异常: {exc}",
            reason="Ontology 轴执行异常",
        ))

    # 3. Literature Validation
    try:
        axes.append(_evaluate_literature_axis(canonical_data, retrieved_papers))
    except Exception as exc:
        logger.error("Literature 轴意外异常: %s", exc)
        axes.append(AxisResult(
            axis=AXIS_LITERATURE_VALIDATION,
            status=ValidationStatus.FAIL,
            evidence=f"轴评估异常: {exc}",
            reason="Literature 轴执行异常",
        ))

    # 4. BioModels Validation
    try:
        axes.append(
            _evaluate_biomodels_axis(
                canonical_data, biomodels_matches, biomodels_validation_result
            )
        )
    except Exception as exc:
        logger.error("BioModels 轴意外异常: %s", exc)
        axes.append(AxisResult(
            axis=AXIS_BIOMODELS_VALIDATION,
            status=ValidationStatus.FAIL,
            evidence=f"轴评估异常: {exc}",
            reason="BioModels 轴执行异常",
        ))

    # 5. Parameter Validation
    try:
        axes.append(_evaluate_parameter_axis(parameter_priors))
    except Exception as exc:
        logger.error("Parameter 轴意外异常: %s", exc)
        axes.append(AxisResult(
            axis=AXIS_PARAMETER_VALIDATION,
            status=ValidationStatus.FAIL,
            evidence=f"轴评估异常: {exc}",
            reason="Parameter 轴执行异常",
        ))

    # 6. Simulation Validation
    try:
        axes.append(_evaluate_simulation_axis(simulation_csv_path))
    except Exception as exc:
        logger.error("Simulation 轴意外异常: %s", exc)
        axes.append(AxisResult(
            axis=AXIS_SIMULATION_VALIDATION,
            status=ValidationStatus.FAIL,
            evidence=f"轴评估异常: {exc}",
            reason="Simulation 轴执行异常",
        ))

    # 7. Dynamics Validation
    try:
        axes.append(
            _evaluate_dynamics_axis(canonical_data, simulation_csv_path)
        )
    except Exception as exc:
        logger.error("Dynamics 轴意外异常: %s", exc)
        axes.append(AxisResult(
            axis=AXIS_DYNAMICS_VALIDATION,
            status=ValidationStatus.FAIL,
            evidence=f"轴评估异常: {exc}",
            reason="Dynamics 轴执行异常",
        ))

    # 8. Experiment Validation
    try:
        axes.append(
            _evaluate_experiment_axis(canonical_data, experiment_plan)
        )
    except Exception as exc:
        logger.error("Experiment 轴意外异常: %s", exc)
        axes.append(AxisResult(
            axis=AXIS_EXPERIMENT_VALIDATION,
            status=ValidationStatus.FAIL,
            evidence=f"轴评估异常: {exc}",
            reason="Experiment 轴执行异常",
        ))

    # 9. Evidence Attribution Validation
    try:
        axes.append(
            _evaluate_evidence_attribution_axis(report_md, evidence_pool)
        )
    except Exception as exc:
        logger.error("Evidence Attribution 轴意外异常: %s", exc)
        axes.append(AxisResult(
            axis=AXIS_EVIDENCE_ATTRIBUTION_VALIDATION,
            status=ValidationStatus.FAIL,
            evidence=f"轴评估异常: {exc}",
            reason="Evidence Attribution 轴执行异常",
        ))

    # 10. Scientific Writing Validation
    try:
        axes.append(_evaluate_scientific_writing_axis(report_md))
    except Exception as exc:
        logger.error("Scientific Writing 轴意外异常: %s", exc)
        axes.append(AxisResult(
            axis=AXIS_SCIENTIFIC_WRITING_VALIDATION,
            status=ValidationStatus.FAIL,
            evidence=f"轴评估异常: {exc}",
            reason="Scientific Writing 轴执行异常",
        ))

    # 11. Reproducibility Validation
    try:
        axes.append(_evaluate_reproducibility_axis(reproducibility_metadata))
    except Exception as exc:
        logger.error("Reproducibility 轴意外异常: %s", exc)
        axes.append(AxisResult(
            axis=AXIS_REPRODUCIBILITY_VALIDATION,
            status=ValidationStatus.FAIL,
            evidence=f"轴评估异常: {exc}",
            reason="Reproducibility 轴执行异常",
        ))

    # 12. Benchmark Validation
    try:
        axes.append(_evaluate_benchmark_axis(benchmark_result))
    except Exception as exc:
        logger.error("Benchmark 轴意外异常: %s", exc)
        axes.append(AxisResult(
            axis=AXIS_BENCHMARK_VALIDATION,
            status=ValidationStatus.FAIL,
            evidence=f"轴评估异常: {exc}",
            reason="Benchmark 轴执行异常",
        ))

    # -------------------------------------------------------------------------
    # 聚合
    # -------------------------------------------------------------------------
    overall_status = _aggregate_status(axes)
    overall_confidence = _calculate_confidence(axes)
    summary = _build_summary(pathway, axes, overall_status, overall_confidence)

    logger.info(
        "12 轴 Validation Matrix 完成: pathway=%s, overall_status=%s, "
        "overall_confidence=%.3f, axes=%d",
        pathway,
        overall_status.value,
        overall_confidence,
        len(axes),
    )

    return ValidationMatrixResult(
        pathway=pathway,
        axes=axes,
        overall_status=overall_status,
        overall_confidence=overall_confidence,
        summary=summary,
    )


# =============================================================================
# Scientific Score Card（Task 7：Scientific Score 计算与对比）
# =============================================================================
#
# 模块用途：
#   为每次 Benchmark 运行计算 Scientific Score（0-100），作为 F5 Loop Merge/Reject
#   的唯一量化依据（禁止肉眼判断）。Score 由五分项加权得出：
#       1. Validation Matrix Score (40%) — 12 轴通过率（PASS=1, PARTIAL=0.5, FAIL=0）
#       2. Curve Metrics Score     (25%) — 8 项指标通过率
#       3. Literature Score         (15%) — Canonical PMIDs hit rate
#       4. Evidence Score           (15%) — grounded 句子比例
#       5. Experiment Score          (5%) — 机制驱动实验通过率
#
# 设计原则：
#   - 唯一成功指标 = Benchmark Pass Rate（禁止肉眼判断）
#   - 输入容错：缺字段或结构不符时该分项 raw_score=0.0（不 crash）
#   - 不引入新依赖（仅标准库 + 已有 import）
#   - 合并到现有模块，不保留独立 score_card.py
# =============================================================================

# 五分项名称与权重（顺序固定，对应 Spec Scenario "Score 计算"）
SCORE_COMPONENTS: dict[str, float] = {
    "validation_matrix": 0.40,   # 1. Validation Matrix Score（40%）
    "curve_metrics": 0.25,       # 2. Curve Metrics Score（25%）
    "literature": 0.15,           # 3. Literature Score（15%）
    "evidence": 0.15,             # 4. Evidence Score（15%）
    "experiment": 0.05,            # 5. Experiment Score（5%）
}

# Score 决策阈值（对应 Spec Scenario "Score 对比"）
# Delta ≥ 0 → MERGE；Delta < 0 → REJECT
SCORE_MERGE_THRESHOLD: float = 0.0


class ScoreDecision:
    """Score Card 决策常量（对应 Spec Scenario "Score 对比"）。

    Attributes:
        MERGE: 前后 Delta ≥ 0，接受修改。
        REJECT: 前后 Delta < 0，拒绝修改。
    """

    MERGE: str = "MERGE"
    REJECT: str = "REJECT"


@dataclass
class ScoreComponent:
    """单分项得分。

    Attributes:
        name: 分项名称（如 ``"validation_matrix"``），与 SCORE_COMPONENTS key 一致。
        weight: 权重（0.0-1.0，对应 SCORE_COMPONENTS）。
        raw_score: 原始分（0.0-1.0，通过率）。
        weighted_score: 加权分（0-100，等于 ``raw_score * weight * 100``）。
    """

    name: str
    weight: float
    raw_score: float = 0.0
    weighted_score: float = 0.0


@dataclass
class ScoreCard:
    """Scientific Score Card 完整报告。

    对应 Spec Requirement "Scientific Score Card"。

    Attributes:
        components: 五分项 ScoreComponent 列表（顺序与 SCORE_COMPONENTS 一致）。
        total_score: 总分（0-100，五分项 weighted_score 之和）。
        benchmark_name: 对应的 Benchmark 名称（如 ``"egfr"``）。
        timestamp: 计算时间戳（``%Y-%m-%d %H:%M:%S`` 格式）。
    """

    components: list[ScoreComponent] = field(default_factory=list)
    total_score: float = 0.0
    benchmark_name: str = ""
    timestamp: str = ""


@dataclass
class ScoreComparison:
    """前后 Score 对比结果（对应 Spec Scenario "Score 对比"）。

    Attributes:
        before: 修改前 ScoreCard。
        after: 修改后 ScoreCard。
        delta_total: 总分变化（``after.total_score - before.total_score``）。
        delta_components: 各分项变化 ``{component_name: delta}``，
            delta = ``after.raw*weight*100 - before.raw*weight*100``。
        decision: ``ScoreDecision.MERGE`` 或 ``ScoreDecision.REJECT``。
    """

    before: ScoreCard
    after: ScoreCard
    delta_total: float = 0.0
    delta_components: dict[str, float] = field(default_factory=dict)
    decision: str = "PENDING"


# =============================================================================
# 辅助函数：字段提取与单分项 raw_score 计算（私有）
# =============================================================================


def _scorecard_get(obj: Any, key: str, default: Any = None) -> Any:
    """从 dict 或 dataclass 实例提取字段值。

    支持两种输入：
        - dict：``obj.get(key, default)``
        - dataclass /普通对象：``getattr(obj, key, default)``

    Args:
        obj: dict 或 dataclass 实例。
        key: 字段名。
        default: 字段缺失时的默认值。

    Returns:
        字段值；不存在时返回 ``default``。
    """
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _normalize_status(status: Any) -> str:
    """将 status 字段标准化为大写字符串。

    支持 ``ValidationStatus`` 枚举（继承 ``str``）与普通字符串。

    Args:
        status: status 字段值。

    Returns:
        大写字符串（``"PASS"`` / ``"PARTIAL"`` / ``"FAIL"``）；无法识别时返回 ``""``。
    """
    if status is None:
        return ""
    return str(status).upper()


def _compute_validation_matrix_raw(result: Any) -> float:
    """计算 12 轴 Validation Matrix 通过率（PASS=1.0, PARTIAL=0.5, FAIL=0）。

    Args:
        result: ``run_validation_matrix()`` 输出（dict 或 ValidationMatrixResult）。

    Returns:
        通过率（0.0-1.0）；axes 为空或缺失时返回 0.0。
    """
    axes = _scorecard_get(result, "axes", [])
    if not isinstance(axes, (list, tuple)):
        return 0.0
    total = len(axes)
    if total == 0:
        return 0.0

    score = 0.0
    for axis in axes:
        status = _normalize_status(_scorecard_get(axis, "status", ""))
        if status == "PASS":
            score += 1.0
        elif status == "PARTIAL":
            score += 0.5
        # FAIL 计 0
    return score / total


def _compute_curve_metrics_raw(result: Any) -> float:
    """计算 8 项 Curve Metrics 通过率。

    支持两种 schema：
        - ``{"metrics": [{"passed": True/False, ...}, ...]}``
        - ``{"metrics": [{"status": "PASS"/"FAIL", ...}, ...]}``

    Args:
        result: Curve Metrics 结果。

    Returns:
        通过率（0.0-1.0）；metrics 为空或缺失时返回 0.0。
    """
    metrics = _scorecard_get(result, "metrics", [])
    if not isinstance(metrics, (list, tuple)):
        return 0.0
    total = len(metrics)
    if total == 0:
        return 0.0

    passed = 0
    for metric in metrics:
        # 优先 passed 字段
        passed_val = _scorecard_get(metric, "passed", None)
        if passed_val is not None:
            if bool(passed_val):
                passed += 1
            continue
        # 退化到 status 字段
        status = _normalize_status(_scorecard_get(metric, "status", ""))
        if status == "PASS":
            passed += 1
    return passed / total


def _compute_literature_raw(result: Any) -> float:
    """计算 Canonical PMIDs hit rate。

    支持多种 schema：
        - ``{"hit_rate": 0.8}``（直接使用）
        - ``{"canonical_pmids": [...], "retrieved_pmids": [...]}``（计算交集）
        - 单 PMID 字符串兼容：``canonical_pmids`` / ``retrieved_pmids`` 为 str

    Args:
        result: Literature 结果。

    Returns:
        hit_rate（0.0-1.0）；canonical 为空或缺失时返回 0.0。
    """
    # 1. 直接 hit_rate 字段
    hit_rate = _scorecard_get(result, "hit_rate", None)
    if hit_rate is not None:
        try:
            return max(0.0, min(1.0, float(hit_rate)))
        except (TypeError, ValueError):
            pass

    # 2. canonical_pmids ∩ retrieved_pmids / canonical_pmids
    canonical = (
        _scorecard_get(result, "canonical_pmids", None)
        or _scorecard_get(result, "canonical_pmid", None)
        or []
    )
    retrieved = (
        _scorecard_get(result, "retrieved_pmids", None)
        or _scorecard_get(result, "retrieved_pmid", None)
        or []
    )

    # 兼容单个 PMID 字符串
    if isinstance(canonical, str):
        canonical = [canonical]
    if isinstance(retrieved, str):
        retrieved = [retrieved]

    if not isinstance(canonical, (list, tuple)) or len(canonical) == 0:
        return 0.0
    if not isinstance(retrieved, (list, tuple)):
        retrieved = []

    canonical_set = {str(p) for p in canonical}
    retrieved_set = {str(p) for p in retrieved}
    if not canonical_set:
        return 0.0
    return len(canonical_set & retrieved_set) / len(canonical_set)


def _compute_evidence_raw(result: Any) -> float:
    """计算 grounded 句子比例。

    支持多种 schema：
        - ``{"grounding_rate": 0.7}``（直接使用）
        - ``{"grounded_sentence_count": 10, "total_sentence_count": 15}``
        - 兼容字段：``grounded_count`` / ``total_count``

    Args:
        result: Evidence Graph 结果。

    Returns:
        grounding_rate（0.0-1.0）；total 为 0 或缺失时返回 0.0。
    """
    # 1. 直接 grounding_rate 字段
    rate = _scorecard_get(result, "grounding_rate", None)
    if rate is not None:
        try:
            return max(0.0, min(1.0, float(rate)))
        except (TypeError, ValueError):
            pass

    # 2. grounded / total
    grounded = (
        _scorecard_get(result, "grounded_sentence_count", None)
        if _scorecard_get(result, "grounded_sentence_count", None) is not None
        else _scorecard_get(result, "grounded_count", 0)
    )
    total = (
        _scorecard_get(result, "total_sentence_count", None)
        if _scorecard_get(result, "total_sentence_count", None) is not None
        else _scorecard_get(result, "total_count", 0)
    )
    try:
        grounded = int(grounded)
        total = int(total)
    except (TypeError, ValueError):
        return 0.0
    if total <= 0:
        return 0.0
    return max(0.0, min(1.0, grounded / total))


def _compute_experiment_raw(result: Any) -> float:
    """计算机制驱动实验通过率。

    支持两种 schema：
        - ``{"experiments": [{"passed": True/False, ...}, ...]}``
        - ``{"experiments": [{"mechanism_driven": True/False, ...}, ...]}``

    Args:
        result: Experiment 结果。

    Returns:
        通过率（0.0-1.0）；experiments 为空或缺失时返回 0.0。
    """
    experiments = _scorecard_get(result, "experiments", [])
    if not isinstance(experiments, (list, tuple)):
        return 0.0
    total = len(experiments)
    if total == 0:
        return 0.0

    passed = 0
    for exp in experiments:
        # 优先 passed 字段
        passed_val = _scorecard_get(exp, "passed", None)
        if passed_val is not None:
            if bool(passed_val):
                passed += 1
            continue
        # 退化到 mechanism_driven 字段
        mech_val = _scorecard_get(exp, "mechanism_driven", None)
        if mech_val is not None and bool(mech_val):
            passed += 1
    return passed / total


# =============================================================================
# 核心函数：compute_scientific_score / compare_scores / render_*_md
# =============================================================================


def compute_scientific_score(
    validation_matrix_result: dict[str, object],
    curve_metrics_result: dict[str, object],
    literature_result: dict[str, object],
    evidence_result: dict[str, object],
    experiment_result: dict[str, object],
    *,
    benchmark_name: str = "",
) -> ScoreCard:
    """计算 Scientific Score（0-100）。

    五分项加权（对应 Spec Scenario "Score 计算"）：
        - Validation Matrix Score (40%) — 12 轴通过率（PASS=1, PARTIAL=0.5, FAIL=0）
        - Curve Metrics Score     (25%) — 8 项指标通过率
        - Literature Score         (15%) — Canonical PMIDs hit rate
        - Evidence Score           (15%) — grounded 句子比例
        - Experiment Score          (5%) — 机制驱动实验通过率

    输入容错：任一分项输入为空 dict 或缺关键字段时，该分项 ``raw_score=0.0``
    （不抛异常，不 crash）。

    Args:
        validation_matrix_result: ``run_validation_matrix()`` 输出（dict 或
            ``ValidationMatrixResult`` 实例）。
        curve_metrics_result: Curve Metrics 结果（含 ``metrics`` 列表）。
        literature_result: Literature 检索结果（含 ``canonical_pmids`` /
            ``retrieved_pmids`` / ``hit_rate``）。
        evidence_result: Evidence Graph 结果（含 ``grounded_sentence_count`` /
            ``total_sentence_count`` / ``grounding_rate``）。
        experiment_result: Experiment 验证结果（含 ``experiments`` 列表）。
        benchmark_name: 对应的 Benchmark 名称（如 ``"egfr"``）。

    Returns:
        ScoreCard，含五分项 ``ScoreComponent`` 明细与总分（0-100，精确到 0.01）。
    """
    # 各分项原始分计算（输入容错：None / 空 dict / 缺字段 → 0.0）
    raw_scores: dict[str, float] = {
        "validation_matrix": _compute_validation_matrix_raw(
            validation_matrix_result or {}
        ),
        "curve_metrics": _compute_curve_metrics_raw(curve_metrics_result or {}),
        "literature": _compute_literature_raw(literature_result or {}),
        "evidence": _compute_evidence_raw(evidence_result or {}),
        "experiment": _compute_experiment_raw(experiment_result or {}),
    }

    # 构建 ScoreComponent 列表（顺序与 SCORE_COMPONENTS 一致）
    components: list[ScoreComponent] = []
    for name, weight in SCORE_COMPONENTS.items():
        raw = max(0.0, min(1.0, raw_scores.get(name, 0.0)))
        weighted = round(raw * weight * 100, 2)
        components.append(
            ScoreComponent(
                name=name,
                weight=weight,
                raw_score=raw,
                weighted_score=weighted,
            )
        )

    # 总分 = sum(raw * weight * 100)，精确到 0.01
    total_score = round(
        sum(c.raw_score * c.weight * 100 for c in components), 2
    )

    # 时间戳：局部 import 避免修改文件顶部 import
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return ScoreCard(
        components=components,
        total_score=total_score,
        benchmark_name=benchmark_name,
        timestamp=timestamp,
    )


def compare_scores(before: ScoreCard, after: ScoreCard) -> ScoreComparison:
    """对比前后 Score，输出 Delta 与 MERGE/REJECT 决策。

    对应 Spec Scenario "Score 对比"——Delta ≥ 0 → MERGE；Delta < 0 → REJECT。

    Args:
        before: 修改前 ScoreCard。
        after: 修改后 ScoreCard。

    Returns:
        ScoreComparison，含总分 Delta、各分项 Delta、MERGE/REJECT 决策。
    """
    delta_total = round(after.total_score - before.total_score, 2)

    # 各分项 delta = after.raw*weight*100 - before.raw*weight*100
    before_map: dict[str, ScoreComponent] = {c.name: c for c in before.components}
    after_map: dict[str, ScoreComponent] = {c.name: c for c in after.components}

    delta_components: dict[str, float] = {}
    for name in SCORE_COMPONENTS.keys():
        b = before_map.get(name)
        a = after_map.get(name)
        b_score = b.raw_score * b.weight * 100 if b else 0.0
        a_score = a.raw_score * a.weight * 100 if a else 0.0
        delta_components[name] = round(a_score - b_score, 2)

    decision = (
        ScoreDecision.MERGE
        if delta_total >= SCORE_MERGE_THRESHOLD
        else ScoreDecision.REJECT
    )

    return ScoreComparison(
        before=before,
        after=after,
        delta_total=delta_total,
        delta_components=delta_components,
        decision=decision,
    )


# 分项展示名映射（用于 Markdown 渲染）
_SCORE_DISPLAY_NAMES: dict[str, str] = {
    "validation_matrix": "Validation Matrix",
    "curve_metrics": "Curve Metrics",
    "literature": "Literature",
    "evidence": "Evidence",
    "experiment": "Experiment",
}


def render_score_card_md(card: ScoreCard) -> str:
    """将 ScoreCard 渲染为 Markdown 文本。

    输出格式对应 Spec Scenario "Score 计算"——总分 SHALL 在 score_card.md 输出，
    含各分项明细。

    Args:
        card: ScoreCard 实例。

    Returns:
        Markdown 格式字符串。
    """
    lines: list[str] = [
        "# Scientific Score Card",
        "",
        f"**Benchmark**: {card.benchmark_name or '(unspecified)'}",
        f"**Timestamp**: {card.timestamp}",
        f"**Total Score**: {card.total_score:.2f} / 100",
        "",
        "## Component Breakdown",
        "",
        "| Component | Weight | Raw Score | Weighted Score |",
        "|-----------|--------|-----------|----------------|",
    ]

    for c in card.components:
        display = _SCORE_DISPLAY_NAMES.get(c.name, c.name)
        weight_pct = f"{int(round(c.weight * 100))}%"
        lines.append(
            f"| {display} | {weight_pct} | {c.raw_score:.2f} | "
            f"{c.weighted_score:.2f} |"
        )

    lines.append(f"| **Total** | 100% | - | **{card.total_score:.2f}** |")

    return "\n".join(lines)


def render_score_comparison_md(comparison: ScoreComparison) -> str:
    """将 ScoreComparison 渲染为前后对比 Markdown。

    输出格式对应 Spec Scenario "Score 对比"——含各分项 Before / After / Delta
    与最终 MERGE/REJECT 决策。

    Args:
        comparison: ScoreComparison 实例。

    Returns:
        Markdown 格式字符串。
    """
    before_map: dict[str, ScoreComponent] = {
        c.name: c for c in comparison.before.components
    }
    after_map: dict[str, ScoreComponent] = {
        c.name: c for c in comparison.after.components
    }

    lines: list[str] = [
        "# Score Comparison",
        "",
        "| Component | Before | After | Delta |",
        "|-----------|--------|-------|-------|",
    ]

    for name in SCORE_COMPONENTS.keys():
        display = _SCORE_DISPLAY_NAMES.get(name, name)
        b = before_map.get(name)
        a = after_map.get(name)
        b_score = b.weighted_score if b else 0.0
        a_score = a.weighted_score if a else 0.0
        delta = comparison.delta_components.get(name, 0.0)
        sign = "+" if delta >= 0 else ""
        lines.append(
            f"| {display} | {b_score:.2f} | {a_score:.2f} | "
            f"{sign}{delta:.2f} |"
        )

    # Total 行
    sign_total = "+" if comparison.delta_total >= 0 else ""
    lines.append(
        f"| **Total** | {comparison.before.total_score:.2f} | "
        f"{comparison.after.total_score:.2f} | "
        f"**{sign_total}{comparison.delta_total:.2f}** |"
    )
    lines.append("")
    lines.append(f"## Decision: {comparison.decision}")

    return "\n".join(lines)


__all__ = [
    "ValidationStatus",
    "AxisStatus",
    "AxisResult",
    "ValidationMatrixResult",
    "ValidationMatrixReport",
    "VALIDATION_AXES",
    "run_validation_matrix",
    # Scientific Score Card（Task 7）
    "SCORE_COMPONENTS",
    "SCORE_MERGE_THRESHOLD",
    "ScoreDecision",
    "ScoreComponent",
    "ScoreCard",
    "ScoreComparison",
    "compute_scientific_score",
    "compare_scores",
    "render_score_card_md",
    "render_score_comparison_md",
]
