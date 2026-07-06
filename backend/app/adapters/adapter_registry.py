# BioDynamics Agent v4 - Adapter 注册表 + fail-safe 降级逻辑
# 对应 v4 Migration Plan §约束 2 的 Adapter 失败处理策略。
#
# 职责：
# 1. 注册 v3↔v4 双向 Adapter
# 2. 记录转换失败次数，超过阈值自动禁用 v4 路径（fail-safe）
# 3. 提供 safe_v3_to_v4 / safe_v4_to_v3 入口，失败时返回 None
# 4. 统计 Adapter 调用指标（成功/失败次数）
#
# fail-safe 策略：
# - 转换失败时记录 warning + 返回 None（调用方降级到 v3 路径）
# - 失败次数 > 阈值（默认 5 次）时自动禁用 v4 路径
# - 不阻塞主流水线

from __future__ import annotations

import logging
import threading
from typing import Any

from app.adapters.v3_v4_adapter import v3_to_v4
from app.adapters.v4_v3_adapter import v4_to_v3
from app.reaction_ir_v2.schema import ReactionIRv2

logger = logging.getLogger(__name__)


# =============================================================================
# Adapter 注册表
# =============================================================================
class AdapterRegistry:
    """v3↔v4 双向 Adapter 注册表 + fail-safe 降级。

    线程安全：使用 threading.Lock 保护计数器。
    单例模式：通过 get_adapter_registry() 获取全局实例。

    用法：
        registry = get_adapter_registry()
        ir = registry.safe_v3_to_v4(network_json)
        if ir is None:
            # 降级到 v3 路径
            ...
        else:
            # 使用 v4 Reaction IR
            ...
    """

    # fail-safe 阈值：连续失败次数超过此值时自动禁用
    FAIL_SAFE_THRESHOLD: int = 5

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._v3_to_v4_success: int = 0
        self._v3_to_v4_failure: int = 0
        self._v4_to_v3_success: int = 0
        self._v4_to_v3_failure: int = 0
        self._v3_to_v4_disabled: bool = False
        self._v4_to_v3_disabled: bool = False

    # -------------------------------------------------------------------------
    # v3 → v4
    # -------------------------------------------------------------------------
    def safe_v3_to_v4(
        self,
        network_json: dict[str, Any],
        ontology_entities: dict[str, Any] | None = None,
        pathway_tag: str = "",
        sbml_model_id: str | None = None,
    ) -> ReactionIRv2 | None:
        """安全的 v3 → v4 转换，失败时返回 None。

        fail-safe 逻辑：
        1. 若已禁用（失败次数超阈值），直接返回 None
        2. 调用 v3_to_v4 转换
        3. 成功：success++，重置 failure 计数
        4. 失败：failure++，若超阈值则禁用
        """
        with self._lock:
            if self._v3_to_v4_disabled:
                logger.warning(
                    "safe_v3_to_v4: Adapter 已被 fail-safe 禁用（失败 %d 次），返回 None",
                    self._v3_to_v4_failure,
                )
                return None

        result = v3_to_v4(
            network_json,
            ontology_entities=ontology_entities,
            pathway_tag=pathway_tag,
            sbml_model_id=sbml_model_id,
        )

        with self._lock:
            if result is not None:
                self._v3_to_v4_success += 1
                # 重置失败计数（恢复正常）
                if self._v3_to_v4_failure > 0:
                    self._v3_to_v4_failure = 0
            else:
                self._v3_to_v4_failure += 1
                if self._v3_to_v4_failure >= self.FAIL_SAFE_THRESHOLD:
                    self._v3_to_v4_disabled = True
                    logger.error(
                        "safe_v3_to_v4: v3→v4 Adapter 已被 fail-safe 禁用"
                        "（连续失败 %d 次 ≥ 阈值 %d）",
                        self._v3_to_v4_failure, self.FAIL_SAFE_THRESHOLD,
                    )
        return result

    # -------------------------------------------------------------------------
    # v4 → v3
    # -------------------------------------------------------------------------
    def safe_v4_to_v3(
        self,
        reaction_ir: ReactionIRv2,
    ) -> dict[str, Any] | None:
        """安全的 v4 → v3 转换，失败时返回 None。

        fail-safe 逻辑同 safe_v3_to_v4。
        """
        with self._lock:
            if self._v4_to_v3_disabled:
                logger.warning(
                    "safe_v4_to_v3: Adapter 已被 fail-safe 禁用（失败 %d 次），返回 None",
                    self._v4_to_v3_failure,
                )
                return None

        result = v4_to_v3(reaction_ir)

        with self._lock:
            if result is not None:
                self._v4_to_v3_success += 1
                if self._v4_to_v3_failure > 0:
                    self._v4_to_v3_failure = 0
            else:
                self._v4_to_v3_failure += 1
                if self._v4_to_v3_failure >= self.FAIL_SAFE_THRESHOLD:
                    self._v4_to_v3_disabled = True
                    logger.error(
                        "safe_v4_to_v3: v4→v3 Adapter 已被 fail-safe 禁用"
                        "（连续失败 %d 次 ≥ 阈值 %d）",
                        self._v4_to_v3_failure, self.FAIL_SAFE_THRESHOLD,
                    )
        return result

    # -------------------------------------------------------------------------
    # v3 → v4 → v3 往返转换（用于一致性检查）
    # -------------------------------------------------------------------------
    def roundtrip_v3_to_v4_to_v3(
        self,
        network_json: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, ReactionIRv2 | None]:
        """v3 → v4 → v3 往返转换，用于一致性校验。

        Returns:
            (往返后的 network_json, 中间的 ReactionIRv2)
            任一步失败时对应位置为 None
        """
        ir = self.safe_v3_to_v4(network_json)
        if ir is None:
            return None, None
        roundtrip_json = self.safe_v4_to_v3(ir)
        return roundtrip_json, ir

    # -------------------------------------------------------------------------
    # 指标查询
    # -------------------------------------------------------------------------
    def get_metrics(self) -> dict[str, Any]:
        """返回 Adapter 调用指标。"""
        with self._lock:
            return {
                "v3_to_v4": {
                    "success": self._v3_to_v4_success,
                    "failure": self._v3_to_v4_failure,
                    "disabled": self._v3_to_v4_disabled,
                    "threshold": self.FAIL_SAFE_THRESHOLD,
                },
                "v4_to_v3": {
                    "success": self._v4_to_v3_success,
                    "failure": self._v4_to_v3_failure,
                    "disabled": self._v4_to_v3_disabled,
                    "threshold": self.FAIL_SAFE_THRESHOLD,
                },
            }

    def reset_metrics(self) -> None:
        """重置计数器与禁用状态（用于测试）。"""
        with self._lock:
            self._v3_to_v4_success = 0
            self._v3_to_v4_failure = 0
            self._v4_to_v3_success = 0
            self._v4_to_v3_failure = 0
            self._v3_to_v4_disabled = False
            self._v4_to_v3_disabled = False

    def reset_disabled(self) -> None:
        """仅重置禁用状态（保留计数器，用于运行时手动恢复）。"""
        with self._lock:
            self._v3_to_v4_disabled = False
            self._v4_to_v3_disabled = False
            self._v3_to_v4_failure = 0
            self._v4_to_v3_failure = 0


# =============================================================================
# 全局单例
# =============================================================================
_registry: AdapterRegistry | None = None
_registry_lock = threading.Lock()


def get_adapter_registry() -> AdapterRegistry:
    """获取全局 AdapterRegistry 单例。"""
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = AdapterRegistry()
    return _registry


__all__ = [
    "AdapterRegistry",
    "get_adapter_registry",
]
