# BioDynamics Agent v4 - Pathway Specialist Feedback 模块数据结构模板 (Task 4.2.3)
# 定义 FeedbackModuleData dataclass，作为 PathwaySpecialistBase.apply_feedback() 的
# 返回数据结构骨架。具体 Specialist 子类（Task 4.3-4.12）可实例化并填充。

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FeedbackModuleData:
    """反馈模块数据结构：通路内反馈环（正/负反馈，含转录延迟）。

    由 ``PathwaySpecialistBase.apply_feedback()`` 返回，包含通路内的
    FeedbackLoop 列表与延迟参数。
    """

    # 反馈环列表（FeedbackLoop 字典）
    # 每条含 loop_id / source / target / sign / delay / mechanism
    feedback_loops: list[dict] = field(default_factory=list)

    # 转录翻译延迟（分钟），0 表示无延迟反馈
    # 用于 DDE 求解器（如 p53-Mdm2 delay=60min，NF-κB-IκBα delay=30min）
    delay_minutes: float = 0.0

    # 反馈类型标签："negative"（负反馈，多数振荡通路）/ "positive"（正反馈，bistable）
    # / "mixed"（同时含正负反馈，如 Apoptosis Caspase 级联）
    loop_type: str = "negative"


__all__ = ["FeedbackModuleData"]
