# BioDynamics Agent v4 - Cross-talk Coordinator Agent 包 (Phase 4 / Task 4.13)
# 协调多通路 shared species 与 cross-talk edges；防止 cross-pathway parameter
# contamination；处理 cross-talk 时间尺度对齐。
#
# 设计原则（铁律）：
# 1. Feature Flag V4_CROSSTALK_COORDINATOR_ENABLED=false 时完全不执行
# 2. 不修改 v3 任何字段；仅新增 v4 字段（v4_crosstalk_edges / v4_shared_species）
# 3. 不修改 Specialist 内部 Reaction；不生成 ODE；不做 SBML 验证（职责边界严格）
# 4. 失败降级：任何异常都返回空更新，不阻塞主流水线
#
# 模块组成：
# - coordinator.py: CrossTalkCoordinator 主类 + LangGraph hook
# - shared_species_sync.py: shared species 同步策略
# - crosstalk_edges.py: cross-talk edge 注入逻辑
# - time_scale_aligner.py: 时间尺度对齐
#
# 参考：
# - spec.md Part 3 Cross-talk Coordinator Agent（第 262-272 行）

from app.crosstalk.coordinator import (
    CrossTalkCoordinator,
    crosstalk_coordinator_hook_node,
)
from app.crosstalk.crosstalk_edges import CrossTalkEdgeInjector
from app.crosstalk.shared_species_sync import SharedSpeciesSync
from app.crosstalk.time_scale_aligner import TimeScaleAligner

__all__ = [
    "CrossTalkCoordinator",
    "crosstalk_coordinator_hook_node",
    "CrossTalkEdgeInjector",
    "SharedSpeciesSync",
    "TimeScaleAligner",
]
