# BioDynamics Agent v4 - Adapters（v3↔v4 双向兼容层）
# Phase 2 新增模块：v3 network_json ↔ v4 ReactionIRv2 双向转换。
# 对应 v4 Migration Plan §约束 2 + §2.4。
#
# 设计原则（铁律）：
# 1. Feature Flag V4_REACTION_IR_ADAPTER_ENABLED=false 时 Adapter 不被调用
# 2. 转换失败时返回 None + warning，触发 fail-safe 降级到 v3 路径
# 3. 不阻塞主流水线
# 4. 失败次数超阈值（5 次）自动禁用 v4 路径

from app.adapters.v3_v4_adapter import v3_to_v4
from app.adapters.v4_v3_adapter import v4_to_v3
from app.adapters.adapter_registry import (
    AdapterRegistry,
    get_adapter_registry,
)

__all__ = [
    "v3_to_v4",
    "v4_to_v3",
    "AdapterRegistry",
    "get_adapter_registry",
]
