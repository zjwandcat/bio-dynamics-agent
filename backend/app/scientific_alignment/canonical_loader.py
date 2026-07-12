# BioDynamics Agent v4 - Scientific Alignment Loop: Canonical Reference Loader (Task 22.2)
#
# Canonical Reference Library 加载器：为每通路提供唯一权威参考。
# 设计目标：所有 Benchmark 的"标准答案"必须基于 Canonical，而非 Retriever 检索结果。
#   Canonical 定义每通路的权威综述、权威 BioModels 模型、机制节点链、
#   动力学行为预期、已知负反馈与自洽规则。
#
# 安全设计（参考 evidence_ranker.py / gold_standard_schema.py）：
#   1. pathway 白名单正则：仅允许 [a-zA-Z0-9_]，拒绝 .. / / \ 等路径遍历字符
#   2. 二次防护：Path.resolve() 后用 relative_to 校验归属 canonical 目录
#   3. 强制 yaml.safe_load，绝不使用 unsafe_load
#
# 依赖：PyYAML（已存在）、Python 标准库；不引入新依赖。
#
# 核心导出：
#   from app.scientific_alignment.canonical_loader import (
#       CanonicalReference, ConsistencyRule, MechanismEdge,
#       load_canonical, get_consistency_rules, validate_canonical,
#       CanonicalMissingError, CanonicalNotFoundError,
#   )

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


# =============================================================================
# 常量
# =============================================================================
# Canonical Reference YAML 根目录：backend/knowledge/canonical/
# 本文件位于 backend/app/scientific_alignment/canonical_loader.py
CANONICAL_DIR: Path = (
    Path(__file__).resolve().parent.parent.parent / "knowledge" / "canonical"
)

# pathway 白名单正则：仅允许 [a-zA-Z0-9_]，防止路径遍历与非法字符注入
_PATHWAY_PATTERN: re.Pattern[str] = re.compile(r"^[a-zA-Z0-9_]+$")

# 必填字段（缺失任一即视为非法 Canonical Reference）
REQUIRED_FIELDS: tuple[str, ...] = (
    "pathway",
    "name",
    "canonical_reviews",
    "canonical_models",
    "canonical_mechanism",
    "expected_behavior",
    "known_negative_feedback",
    "consistency_rules",
)


# =============================================================================
# 异常类
# =============================================================================
class CanonicalNotFoundError(FileNotFoundError):
    """指定 pathway 的 Canonical Reference 文件不存在时抛出。"""


class CanonicalMissingError(ValueError):
    """Canonical Reference 缺失必填字段时抛出。"""


class CanonicalPathwayNameError(ValueError):
    """pathway 参数含非法字符或路径越界时抛出（路径遍历防护）。"""


# =============================================================================
# 数据类
# =============================================================================
@dataclass
class MechanismEdge:
    """机制边：描述两个节点间的因果关系。

    Attributes:
        from_node: 起始节点（大写蛋白名，如 ``"EGF"``）。
        to_node: 目标节点（如 ``"EGFR"``）。
        relation: 关系类型（activation/inhibition/phosphorylation/binding/
            transcription/degradation/cleavage/nuclear_translocation 等）。
    """

    from_node: str = ""
    to_node: str = ""
    relation: str = ""


@dataclass
class CanonicalMechanism:
    """机制节点链与边。

    Attributes:
        required_nodes: 机制节点链（大写蛋白名列表，如 ``["EGF", "EGFR", ...]``）。
        edges: 机制边列表（MechanismEdge）。
    """

    required_nodes: List[str] = field(default_factory=list)
    edges: List[MechanismEdge] = field(default_factory=list)


@dataclass
class ConsistencyRule:
    """自洽规则：供 Consistency Checker 加载并校验仿真结果。

    Attributes:
        rule: 规则的人类可读描述（如 ``"EGFR Peak 不能晚于 ERK Peak"``）。
        assertion: 可判断语句（如 ``"egfr_peak_time < erk_peak_time"``）。
        violation_label: 违规标签（如 ``"egfr_peak_after_erk_peak"``）。
    """

    rule: str = ""
    assertion: str = ""
    violation_label: str = ""


@dataclass
class CanonicalReference:
    """Canonical Reference 数据模型。

    每通路唯一权威参考，包含权威综述、权威 BioModels 模型、机制节点链、
    动力学行为预期、已知负反馈与自洽规则。

    Attributes:
        pathway: 通路标识（如 ``"EGFR_RTK"``）。
        name: 人类可读名称。
        canonical_reviews: 权威综述 PMID 列表（如 ``["PMID:7657691"]``）。
        canonical_models: 权威 BioModels ID 列表（如 ``["BIOMD0000000010"]``）。
        canonical_mechanism: 机制对象，含 ``required_nodes``（节点列表）与
            ``edges``（MechanismEdge 列表）。
        expected_behavior: 动力学行为预期 dict。
        known_negative_feedback: 已知负反馈列表。
        consistency_rules: 自洽规则列表（ConsistencyRule）。
        raw: 原始 YAML dict（便于上层扩展校验，不参与必填校验）。
    """

    pathway: str = ""
    name: str = ""
    canonical_reviews: List[str] = field(default_factory=list)
    canonical_models: List[str] = field(default_factory=list)
    canonical_mechanism: CanonicalMechanism = field(default_factory=CanonicalMechanism)
    expected_behavior: Dict[str, Any] = field(default_factory=dict)
    known_negative_feedback: List[str] = field(default_factory=list)
    consistency_rules: List[ConsistencyRule] = field(default_factory=list)
    raw: Optional[Dict[str, Any]] = None

    @property
    def required_nodes(self) -> List[str]:
        """便捷访问：机制节点链（canonical_mechanism.required_nodes）。"""
        return list(self.canonical_mechanism.required_nodes)

    @property
    def edges(self) -> List[MechanismEdge]:
        """便捷访问：机制边（canonical_mechanism.edges）。"""
        return list(self.canonical_mechanism.edges)


# =============================================================================
# 路径遍历防护
# =============================================================================
def _validate_pathway_name(pathway: str) -> None:
    """校验 pathway 名仅含 [a-zA-Z0-9_]，禁止 .. / / \\ 等字符。

    Args:
        pathway: 通路标识（如 ``"egfr"``）。

    Raises:
        CanonicalPathwayNameError: pathway 为空或含非法字符。
    """
    if not isinstance(pathway, str) or not pathway:
        raise CanonicalPathwayNameError("pathway 不能为空字符串")
    if not _PATHWAY_PATTERN.match(pathway):
        raise CanonicalPathwayNameError(
            f"非法 pathway 标识: {pathway!r}（仅允许 [a-zA-Z0-9_]）"
        )


def _resolve_canonical_path(pathway: str) -> Path:
    """解析 pathway -> Path，并校验解析结果在 canonical 目录内。

    双重保险：先做白名单字符校验（拒绝 .. / 分隔符），再 resolve 后
    用 relative_to 校验归属，确保即使白名单被绕过也不会越界。

    Args:
        pathway: 通路标识（如 ``"egfr"``）。

    Returns:
        解析后的 Canonical YAML 文件绝对路径。

    Raises:
        CanonicalPathwayNameError: pathway 含非法字符或路径越界。
    """
    _validate_pathway_name(pathway)
    base_dir = CANONICAL_DIR.resolve()
    candidate = (base_dir / f"{pathway}.yaml").resolve()
    try:
        candidate.relative_to(base_dir)
    except ValueError as exc:
        raise CanonicalPathwayNameError(
            f"路径遍历攻击检测: pathway={pathway!r} 解析后越出 canonical 目录"
        ) from exc
    return candidate


# =============================================================================
# 解析辅助
# =============================================================================
def _parse_mechanism_edges(raw: Any) -> List[MechanismEdge]:
    """将 YAML 中的 edges 列表解析为 MechanismEdge 列表。

    Args:
        raw: YAML 解析后的原始数据（期望为 list[dict]）。

    Returns:
        MechanismEdge 列表；非 dict 元素被跳过。
    """
    if not isinstance(raw, list):
        return []
    result: List[MechanismEdge] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        result.append(
            MechanismEdge(
                from_node=str(item.get("from", "")),
                to_node=str(item.get("to", "")),
                relation=str(item.get("relation", "")),
            )
        )
    return result


def _parse_canonical_mechanism(raw: Any) -> CanonicalMechanism:
    """将 YAML 中的 canonical_mechanism dict 解析为 CanonicalMechanism 对象。

    Args:
        raw: YAML 解析后的原始数据（期望为 dict 含 required_nodes 与 edges）。

    Returns:
        CanonicalMechanism 实例。
    """
    if not isinstance(raw, dict):
        return CanonicalMechanism()
    nodes_raw = raw.get("required_nodes")
    nodes: List[str] = (
        [str(n) for n in nodes_raw if isinstance(nodes_raw, list)]
        if isinstance(nodes_raw, list)
        else []
    )
    edges = _parse_mechanism_edges(raw.get("edges"))
    return CanonicalMechanism(required_nodes=nodes, edges=edges)


def _parse_consistency_rules(
    raw: Any,
) -> List[ConsistencyRule]:
    """将 YAML 中的 consistency_rules 列表解析为 ConsistencyRule 列表。

    Args:
        raw: YAML 解析后的原始数据（期望为 list[dict]）。

    Returns:
        ConsistencyRule 列表；非 dict 元素被跳过。
    """
    if not isinstance(raw, list):
        return []
    result: List[ConsistencyRule] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        result.append(
            ConsistencyRule(
                rule=str(item.get("rule", "")),
                assertion=str(item.get("assertion", "")),
                violation_label=str(item.get("violation_label", "")),
            )
        )
    return result


# =============================================================================
# 校验函数
# =============================================================================
def validate_canonical(cr: CanonicalReference) -> List[str]:
    """返回缺失必填字段列表（空列表表示 valid）。

    校验范围：
    - 8 个必填字段必须存在（None 视为缺失）
    - pathway / name 必须为非空字符串
    - canonical_mechanism 必须含非空 required_nodes
    - canonical_reviews / canonical_models 允许为空列表（标 TODO 的通路）

    Args:
        cr: CanonicalReference 实例。

    Returns:
        缺失字段名列表；空列表表示完全有效。
    """
    missing: List[str] = []
    for fname in REQUIRED_FIELDS:
        value = getattr(cr, fname, None)
        if value is None:
            missing.append(fname)
    # pathway / name 必须为非空字符串
    if "pathway" not in missing and not cr.pathway.strip():
        missing.append("pathway")
    if "name" not in missing and not cr.name.strip():
        missing.append("name")
    # canonical_mechanism 必须含非空 required_nodes
    if "canonical_mechanism" not in missing:
        if not cr.canonical_mechanism.required_nodes:
            missing.append("canonical_mechanism.required_nodes")
    return missing


# =============================================================================
# 加载函数
# =============================================================================
def load_canonical(pathway: str) -> CanonicalReference:
    """从 ``knowledge/canonical/<pathway>.yaml`` 加载 Canonical Reference。

    做路径遍历防护：
      1. pathway 必须匹配 ``[a-zA-Z0-9_]+`` 白名单（拒绝 ``..`` / ``/`` / ``\\``）
      2. 二次校验：解析后的绝对路径必须在 CANONICAL_DIR 内

    Args:
        pathway: 通路标识（如 ``"egfr"``、``"pi3k_akt_mtor"``）。

    Returns:
        CanonicalReference 实例。

    Raises:
        CanonicalPathwayNameError: pathway 含非法字符或路径越界。
        CanonicalNotFoundError: 文件不存在。
        CanonicalMissingError: YAML 顶层非 dict 或缺失必填字段。
        yaml.YAMLError: YAML 解析失败。
    """
    # 1. 路径解析与防护
    file_path = _resolve_canonical_path(pathway)

    if not file_path.is_file():
        raise CanonicalNotFoundError(
            f"Canonical Reference 文件不存在: {file_path}"
        )

    # 2. 安全加载 YAML（强制 safe_load）
    with file_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict):
        raise CanonicalMissingError(
            f"Canonical YAML 顶层必须为映射/dict，实际: {type(data).__name__}"
        )

    # 3. 构造 CanonicalReference
    cr = CanonicalReference(
        pathway=str(data.get("pathway", "") or ""),
        name=str(data.get("name", "") or ""),
        canonical_reviews=list(data.get("canonical_reviews") or []),
        canonical_models=list(data.get("canonical_models") or []),
        canonical_mechanism=_parse_canonical_mechanism(
            data.get("canonical_mechanism")
        ),
        expected_behavior=dict(data.get("expected_behavior") or {}),
        known_negative_feedback=list(data.get("known_negative_feedback") or []),
        consistency_rules=_parse_consistency_rules(
            data.get("consistency_rules")
        ),
        raw=data,
    )

    # 4. 必填字段校验
    missing = validate_canonical(cr)
    if missing:
        raise CanonicalMissingError(
            f"Canonical Reference {pathway!r} 缺失必填字段: {missing}"
        )

    return cr


def get_consistency_rules(pathway: str) -> List[ConsistencyRule]:
    """便捷方法：加载指定通路的自洽规则列表。

    Args:
        pathway: 通路标识（如 ``"egfr"``）。

    Returns:
        ConsistencyRule 列表。

    Raises:
        与 :func:`load_canonical` 相同。
    """
    cr = load_canonical(pathway)
    return cr.consistency_rules


__all__ = [
    "CanonicalReference",
    "CanonicalMechanism",
    "ConsistencyRule",
    "MechanismEdge",
    "CanonicalMissingError",
    "CanonicalNotFoundError",
    "CanonicalPathwayNameError",
    "REQUIRED_FIELDS",
    "CANONICAL_DIR",
    "load_canonical",
    "get_consistency_rules",
    "validate_canonical",
]
