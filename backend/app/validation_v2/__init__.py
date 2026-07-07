# BioDynamics Agent v4 - Validation Pyramid v2 (Phase 5)
#
# P5 Validation Pyramid 主模块，包含 5 层验证：
#   Level 1: Internal Consistency Validation（本模块）
#   Level 2: SBML/BioModels Validation（后续 Task）
#   Level 3: Cross-Pathway Validation（后续 Task）
#   Level 4: Benchmark Validation（后续 Task）
#   Level 5: Hypothesis Validation（后续 Task）
#
# 设计原则：
# - Feature Flag V4_VALIDATION_PYRAMID_ENABLED=false 时所有 Level 跳过
# - 每个 Level 独立 hook，互不依赖
# - 失败降级：任何异常都不阻塞主流水线（flag=false 时）

from app.validation_v2.level1_internal import (
    Level1InternalValidator,
    level1_hook_node,
)
from app.validation_v2.level2_sbml import (
    Level2SBMLValidator,
    level2_hook_node,
)
from app.validation_v2.level3_crosstalk import (
    Level3CrossPathwayValidator,
    level3_hook_node,
)
from app.validation_v2.thresholds import PathwayThresholds

__all__ = [
    "Level1InternalValidator",
    "level1_hook_node",
    "Level2SBMLValidator",
    "level2_hook_node",
    "Level3CrossPathwayValidator",
    "level3_hook_node",
    "PathwayThresholds",
]
