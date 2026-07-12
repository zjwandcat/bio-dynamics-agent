# BioDynamics Scientific Alignment — Gold Standard Schema (Task 1)
# 机器可读 Gold Standard 数据模型与加载器。
#
# 设计要点：
# 1. Gold Standard 是唯一权威"标准答案"，禁止 Agent/Codex 自行定义；
#    本模块仅负责"加载 + 校验"，不赋予 Agent 改写权限。
# 2. 安全：强制 yaml.safe_load（禁用 unsafe_load）；pathway 参数白名单字符校验
#    （仅 [A-Za-z0-9_]）；Path 解析后必须落在 scientific_alignment 目录内，
#    防止路径遍历（.. / / \ 均被拒绝）。
# 3. forbidden_patterns 同时支持纯字符串与正则模式（编译为 re.Pattern）：
#    优先按正则编译，编译失败则降级为纯字符串匹配。

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Pattern, Union

import yaml

logger = logging.getLogger(__name__)

# Gold Standard YAML 所在目录（backend/benchmarks/scientific_alignment/）
# 本文件位于 backend/app/scientific_alignment/gold_standard_schema.py
_GOLD_STANDARD_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "benchmarks"
    / "scientific_alignment"
)

# pathway 白名单：仅允许字母、数字、下划线（防止路径遍历 / 注入）
_PATHWAY_NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")

# 必填字段（缺失任一即视为非法 Gold Standard）
# 7 个必填：pathway / expected_dynamics / required_mechanisms /
# required_biomodels / required_reviews / required_experiments / forbidden_patterns
REQUIRED_FIELDS = (
    "pathway",
    "expected_dynamics",
    "required_mechanisms",
    "required_biomodels",
    "required_reviews",
    "required_experiments",
    "forbidden_patterns",
)


class GoldStandardMissingError(ValueError):
    """Gold Standard 缺失必填字段时抛出。"""


class GoldStandardNotFoundError(FileNotFoundError):
    """指定 pathway 的 Gold Standard 文件不存在时抛出。"""


class GoldStandardPathwayNameError(ValueError):
    """pathway 参数含非法字符或路径越界时抛出（路径遍历防护）。"""


@dataclass
class GoldStandard:
    """机器可读 Gold Standard 数据模型。

    7 个必填字段 + 可选 name / key_time_scales / key_feedbacks。
    所有字段均带默认值，便于先构造再通过 :func:`validate_gold_standard`
    做完整性校验（缺失字段由校验函数返回 / 抛出）。
    forbidden_patterns 元素可为纯字符串或已编译的正则模式。
    """

    pathway: str = ""
    expected_dynamics: Dict[str, Any] = field(default_factory=dict)
    required_mechanisms: List[str] = field(default_factory=list)
    required_biomodels: List[str] = field(default_factory=list)
    required_reviews: List[str] = field(default_factory=list)
    required_experiments: List[str] = field(default_factory=list)
    forbidden_patterns: List[Union[str, Pattern[str]]] = field(default_factory=list)
    # 可选字段
    name: Optional[str] = None
    key_time_scales: Optional[Dict[str, Any]] = None
    key_feedbacks: Optional[List[str]] = None
    # 原始字典（便于上层做扩展校验/序列化，不参与必填校验）
    raw: Optional[Dict[str, Any]] = None


def _validate_pathway_name(pathway: str) -> None:
    """校验 pathway 名仅含 [A-Za-z0-9_]，禁止 .. / / \\ 等字符。"""
    if not isinstance(pathway, str) or not pathway:
        raise GoldStandardPathwayNameError("pathway 不能为空字符串")
    if not _PATHWAY_NAME_RE.match(pathway):
        raise GoldStandardPathwayNameError(
            f"pathway 含非法字符: {pathway!r}（仅允许 [A-Za-z0-9_]）"
        )


def _resolve_gold_standard_path(pathway: str) -> Path:
    """解析 pathway -> Path，并校验解析结果在 scientific_alignment 目录内。

    双重保险：先做白名单字符校验（拒绝 .. / 分隔符），再 resolve 后
    用 relative_to 校验归属，确保即使白名单被绕过也不会越界。
    """
    _validate_pathway_name(pathway)
    base_dir = _GOLD_STANDARD_DIR.resolve()
    candidate = (base_dir / f"{pathway}.yaml").resolve()
    try:
        candidate.relative_to(base_dir)
    except ValueError as exc:
        raise GoldStandardPathwayNameError(
            f"pathway 解析越界，不在 scientific_alignment 目录内: {candidate}"
        ) from exc
    return candidate


def _compile_forbidden_patterns(
    patterns: List[Any],
) -> List[Union[str, Pattern[str]]]:
    """将 forbidden_patterns 中的字符串编译为正则；编译失败则保留为纯字符串。

    约定：所有 pattern 优先按正则语义编译；若编译失败（含非法正则元字符）
    则降级为纯字符串匹配。这样既支持显式正则模式，也容错普通文本。
    """
    compiled: List[Union[str, Pattern[str]]] = []
    for pat in patterns:
        if isinstance(pat, Pattern):
            compiled.append(pat)
            continue
        if not isinstance(pat, str):
            logger.warning("forbidden_patterns 含非字符串元素: %r，已跳过", pat)
            continue
        try:
            compiled.append(re.compile(pat))
        except re.error as exc:
            logger.debug(
                "forbidden_pattern 编译为正则失败，降级为字符串匹配: %r (%s)",
                pat,
                exc,
            )
            compiled.append(pat)
    return compiled


def validate_gold_standard(gs: GoldStandard) -> List[str]:
    """返回缺失必填字段列表（空列表表示 valid）。

    校验范围：
    - 7 个必填字段必须存在且非空（None / 空容器 / 空字符串视为缺失）
    """
    missing: List[str] = []
    for fname in REQUIRED_FIELDS:
        value = getattr(gs, fname, None)
        if value is None:
            missing.append(fname)
            continue
        # 容器类字段（list/dict/str）为空也视为缺失
        if isinstance(value, (list, dict, str)) and len(value) == 0:
            missing.append(fname)
    return missing


def load_gold_standard(pathway: str) -> GoldStandard:
    """从 benchmarks/scientific_alignment/<pathway>.yaml 加载 Gold Standard。

    Args:
        pathway: 通路标识（如 "egfr"），仅允许 [A-Za-z0-9_]。

    Returns:
        GoldStandard 实例。

    Raises:
        GoldStandardPathwayNameError: pathway 含非法字符或路径越界。
        GoldStandardNotFoundError: 文件不存在。
        GoldStandardMissingError: YAML 顶层非 dict 或缺失必填字段。
        yaml.YAMLError: YAML 解析失败。
    """
    file_path = _resolve_gold_standard_path(pathway)
    if not file_path.is_file():
        raise GoldStandardNotFoundError(f"Gold Standard 文件不存在: {file_path}")

    # 安全：强制 safe_load，绝不使用 unsafe_load
    with file_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict):
        raise GoldStandardMissingError(
            f"Gold Standard YAML 顶层必须为映射/dict，实际: {type(data).__name__}"
        )

    gs = GoldStandard(
        pathway=data.get("pathway", "") or "",
        expected_dynamics=data.get("expected_dynamics") or {},
        required_mechanisms=data.get("required_mechanisms") or [],
        required_biomodels=data.get("required_biomodels") or [],
        required_reviews=data.get("required_reviews") or [],
        required_experiments=data.get("required_experiments") or [],
        forbidden_patterns=_compile_forbidden_patterns(
            data.get("forbidden_patterns") or []
        ),
        name=data.get("name"),
        key_time_scales=data.get("key_time_scales"),
        key_feedbacks=data.get("key_feedbacks"),
        raw=data,
    )

    missing = validate_gold_standard(gs)
    if missing:
        raise GoldStandardMissingError(
            f"Gold Standard {pathway!r} 缺失必填字段: {missing}"
        )

    return gs


__all__ = [
    "GoldStandard",
    "GoldStandardMissingError",
    "GoldStandardNotFoundError",
    "GoldStandardPathwayNameError",
    "REQUIRED_FIELDS",
    "load_gold_standard",
    "validate_gold_standard",
]
