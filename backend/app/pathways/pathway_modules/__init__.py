# BioDynamics Agent v4 - Pathway Specialist Modules 包 (Phase 4 / Task 4.2)
# 5 模块目录骨架：core / feedback / crosstalk / perturbation / validation
# 每个子目录包含一个 dataclass 数据结构模板 (template.py)，
# 作为 PathwaySpecialistBase 各 apply_* 方法的返回数据结构骨架。
#
# 设计原则：
# 1. 本 Task 仅定义 dataclass 骨架，字段默认空，由具体 Specialist 子类
#    （Task 4.3-4.12）在 apply_* 方法中实例化并填充
# 2. dataclass 字段全部使用 default_factory 避免可变默认值共享问题
# 3. 不修改 v3 任何字段；不生成 ODE；不调用 RAG

from app.pathways.pathway_modules.core.template import CoreModuleData
from app.pathways.pathway_modules.crosstalk.template import CrosstalkModuleData
from app.pathways.pathway_modules.feedback.template import FeedbackModuleData
from app.pathways.pathway_modules.perturbation.template import PerturbationModuleData
from app.pathways.pathway_modules.validation.template import ValidationModuleData

__all__ = [
    "CoreModuleData",
    "FeedbackModuleData",
    "CrosstalkModuleData",
    "PerturbationModuleData",
    "ValidationModuleData",
]
