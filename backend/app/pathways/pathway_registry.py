# BioDynamics Agent v4 - Pathway Specialist Plugin Registry (Phase 4 / Task 4.2)
# Specialist 插件注册表：管理 10 个 Pathway Specialist 的注册与查找。
#
# 注意：本文件与 P1 ``app/ontology/pathway_registry.py`` 是不同的注册表：
# - P1 ``ontology/pathway_registry.py``：通路关键词识别注册表（PATHWAY_REGISTRY），
#   用于 Pathway Planner 的规则匹配
# - P4 ``pathways/pathway_registry.py``（本文件）：Specialist 插件注册表
#   （SPECIALIST_REGISTRY），用于按 pathway_class 查找并实例化对应 Specialist
#
# 设计原则：
# 1. 本 Task 仅创建 registry 框架，不注册任何具体 Specialist
#    （10 Specialist 在 Task 4.3-4.12 实现，届时通过 ``@register_specialist``
#    装饰器自动注册到 SPECIALIST_REGISTRY）
# 2. registry 初始为空，但提供完整的注册 / 查询接口
# 3. ``get_specialist`` 每次返回新实例（非单例），避免跨请求状态污染
# 4. 未注册的 pathway_class 查询返回 None，不抛异常（调用方负责降级）

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from app.pathways.pathway_specialist_base import PathwaySpecialistBase

if TYPE_CHECKING:
    from typing import Callable, TypeAlias

logger = logging.getLogger(__name__)

# Type alias: Specialist 类类型（PathwaySpecialistBase 的子类）
if TYPE_CHECKING:
    SpecialistClass: TypeAlias = type[PathwaySpecialistBase]


# =============================================================================
# 全局注册表（初始为空，由 @register_specialist 装饰器自动填充）
# =============================================================================
# key: pathway_class 字符串（如 "EGFR_RTK"）
# value: Specialist 类（type[PathwaySpecialistBase]，非实例）
# 注意：10 Specialist 在 Task 4.3-4.12 实现，届时通过装饰器注册
SPECIALIST_REGISTRY: dict[str, "SpecialistClass"] = {}


def register_specialist(
    cls: "SpecialistClass",
) -> "SpecialistClass":
    """装饰器：将 Specialist 类自动注册到 SPECIALIST_REGISTRY。

    用法::

        @register_specialist
        class EGFRRtkSpecialist(PathwaySpecialistBase):
            pathway_class = "EGFR_RTK"
            ...

    注册逻辑：
    - 读取 ``cls.pathway_class`` 作为 registry key
    - 若 ``pathway_class`` 为空字符串，记录 warning 并跳过注册
      （子类未覆写类属性，属实现错误，但不阻塞导入）
    - 若 key 已存在（重复注册），记录 warning 并覆盖旧值
      （允许覆盖以支持热重载与子类覆写）

    Args:
        cls: 待注册的 Specialist 类（必须是 ``PathwaySpecialistBase`` 子类）。

    Returns:
        原始类（不修改），保持装饰器透明性。
    """
    pathway_class = getattr(cls, "pathway_class", "")
    if not pathway_class:
        logger.warning(
            "register_specialist: 类 %s 未设置 pathway_class，跳过注册",
            cls.__name__,
        )
        return cls

    if pathway_class in SPECIALIST_REGISTRY:
        existing = SPECIALIST_REGISTRY[pathway_class]
        logger.warning(
            "register_specialist: pathway_class='%s' 已注册 (%s)，"
            "将被 %s 覆盖",
            pathway_class,
            existing.__name__,
            cls.__name__,
        )

    SPECIALIST_REGISTRY[pathway_class] = cls
    logger.debug(
        "register_specialist: 已注册 pathway_class='%s' -> %s",
        pathway_class,
        cls.__name__,
    )
    return cls


def get_specialist(pathway_class: str) -> PathwaySpecialistBase | None:
    """按 pathway_class 获取 Specialist 实例。

    每次调用返回新实例（非单例），避免跨请求状态污染。调用方应在使用前
    通过 ``is_specialist_available`` 或返回值 None 检查处理降级。

    [P0-FIX specialist_outputs 为空 / 根因]
    系统中存在两套并行的 pathway_class 命名约定：
      - PATHWAY_REGISTRY (ontology)：全大写键，如 "APOPTOSIS"/"WNT"/"p53"
      - PATHWAY_INITIALIZERS (runner PATHWAY_MAP)：混合大小写键，
        如 "Apoptosis"/"Wnt"/"p53_signaling"
    Specialist 装饰器用 PATHWAY_REGISTRY 形式注册（如 "APOPTOSIS"），
    但当 pathway_planner_hook 失败时（返回 {}），specialist_hook 读到的是
    orchestrator initial_state 的 PATHWAY_INITIALIZERS 形式（如 "Apoptosis"），
    导致 get_specialist("Apoptosis") 返回 None，v4_specialist_outputs 恒为空。

    修复：在查找时做归一化匹配（不修改输入）：
      1. 直接精确匹配（保留原行为）
      2. 大小写不敏感匹配
      3. 通过 _INITIALIZER_TO_REGISTRY 反向映射匹配
         （"Apoptosis" → "APOPTOSIS"，"p53_signaling" → "p53" 等）

    Args:
        pathway_class: 通路类别键（接受 PATHWAY_REGISTRY 或 PATHWAY_INITIALIZERS 形式）。

    Returns:
        Specialist 实例；未注册时返回 ``None``。实例化失败时也返回
        ``None`` 并记录 warning，不抛异常。
    """
    # 1. 直接精确匹配（原行为，最快路径）
    cls = SPECIALIST_REGISTRY.get(pathway_class)
    if cls is not None:
        return _instantiate_specialist(cls, pathway_class)

    # 2-3. 归一化匹配（仅在精确匹配失败时触发，无额外开销）
    normalized = _normalize_pathway_class(pathway_class)
    if normalized != pathway_class:
        cls = SPECIALIST_REGISTRY.get(normalized)
        if cls is not None:
            return _instantiate_specialist(cls, normalized)

    logger.debug(
        "get_specialist: pathway_class='%s' (normalized='%s') 未注册，返回 None",
        pathway_class,
        normalized,
    )
    return None


def _instantiate_specialist(cls: "SpecialistClass", pathway_class: str) -> PathwaySpecialistBase | None:
    """实例化 Specialist 类，捕获所有异常。"""
    try:
        return cls()
    except Exception as exc:
        logger.warning(
            "get_specialist: 实例化 %s (pathway_class='%s') 失败: %s",
            cls.__name__,
            pathway_class,
            exc,
        )
        return None


# [P0-FIX specialist_outputs 为空] 归一化映射表
# PATHWAY_INITIALIZERS 形式 → PATHWAY_REGISTRY 形式
# 与 app.pathways.pathway_planner._REGISTRY_TO_INITIALIZER 互为逆映射。
# 当 PATHWAY_REGISTRY 或 PATHWAY_INITIALIZERS 增删通路时，需同步更新此表。
_INITIALIZER_TO_REGISTRY: dict[str, str] = {
    "EGFR_RTK": "EGFR_RTK",
    "MAPK_ERK": "MAPK_ERK",
    "PI3K_AKT_mTOR": "PI3K_AKT_mTOR",
    "p53_signaling": "p53",
    "Apoptosis": "APOPTOSIS",
    "Cell_Cycle": "CELL_CYCLE",
    "JAK_STAT": "JAK_STAT",
    "NF_kB": "NF_KB",
    "Wnt": "WNT",
    "TGF_beta": "TGF_BETA",
}


def _normalize_pathway_class(pathway_class: str) -> str:
    """将 pathway_class 归一化为 PATHWAY_REGISTRY 形式。

    匹配顺序：
      1. _INITIALIZER_TO_REGISTRY 精确映射（处理 "p53_signaling"→"p53" 等）
      2. 大小写不敏感匹配 SPECIALIST_REGISTRY 已注册键（处理 "Apoptosis"→"APOPTOSIS"）

    Args:
        pathway_class: 任意形式的 pathway_class 字符串

    Returns:
        归一化后的 PATHWAY_REGISTRY 形式；无匹配则原样返回（让上层判定未注册）
    """
    if not pathway_class:
        return pathway_class
    # 1. 反向映射精确匹配
    mapped = _INITIALIZER_TO_REGISTRY.get(pathway_class)
    if mapped:
        return mapped
    # 2. 大小写不敏感匹配（仅在已注册键中查找）
    target_lower = pathway_class.lower()
    for registered_key in SPECIALIST_REGISTRY.keys():
        if registered_key.lower() == target_lower:
            return registered_key
    return pathway_class


def list_specialists() -> list[str]:
    """列出所有已注册的 pathway_class。

    Returns:
        已注册 pathway_class 字符串列表（按注册顺序，Python 3.7+ dict 保序）。
        空列表表示尚未注册任何 Specialist。
    """
    return list(SPECIALIST_REGISTRY.keys())


def is_specialist_available(pathway_class: str) -> bool:
    """检查指定 pathway_class 的 Specialist 是否已注册。

    Args:
        pathway_class: 通路类别键（接受 PATHWAY_REGISTRY 或 PATHWAY_INITIALIZERS 形式）。

    Returns:
        ``True`` 表示已注册且可通过 ``get_specialist`` 获取实例；
        ``False`` 表示未注册。
    """
    if pathway_class in SPECIALIST_REGISTRY:
        return True
    # [P0-FIX] 归一化匹配（与 get_specialist 保持一致）
    normalized = _normalize_pathway_class(pathway_class)
    return normalized in SPECIALIST_REGISTRY


def clear_registry() -> None:
    """清空注册表（仅供测试使用）。

    生产环境不应调用此函数。测试用例在 setup/teardown 中调用以隔离
    不同测试之间的注册状态污染。
    """
    SPECIALIST_REGISTRY.clear()


__all__ = [
    "SPECIALIST_REGISTRY",
    "register_specialist",
    "get_specialist",
    "list_specialists",
    "is_specialist_available",
    "clear_registry",
]
