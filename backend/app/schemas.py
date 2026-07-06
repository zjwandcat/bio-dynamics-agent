# BioDynamics Agent - FastAPI 请求模型
# 定义 /api/chat 接口所需的 Pydantic 结构。

from pydantic import BaseModel, Field


class ClarificationResponse(BaseModel):
    """用户人工干预回答。"""

    selected_option: str = Field(..., description="用户选择的选项 ID：A / B / C")
    free_text: str | None = Field(default=None, description="C 选项时填写的自定义方案")


class ChatRequest(BaseModel):
    """POST /api/chat 接口请求体。"""

    user_input: str = Field(..., min_length=1, description="用户输入的生物学假说或机制描述")
    thread_id: str = Field(..., min_length=1, description="LangGraph 线程标识符")
    mode: str = Field(default="auto_standard", description="运行模式：auto_fast / auto_standard / manual")
    manual_modules: list[str] = Field(
        default_factory=list,
        description="Manual 模式下用户勾选的模块键列表",
    )
    clarification_response: ClarificationResponse | None = Field(
        default=None,
        description="人工干预回答（用于 /api/chat/respond）",
    )


class ClarificationResponseRequest(BaseModel):
    """POST /api/chat/respond 接口请求体。"""

    thread_id: str = Field(..., min_length=1, description="LangGraph 线程标识符")
    clarification_response: ClarificationResponse = Field(..., description="用户干预回答")


class StopRequest(BaseModel):
    """POST /api/chat/stop 接口请求体。"""

    thread_id: str = Field(..., min_length=1, description="需要停止的 LangGraph 线程标识符")


class ClearMemoryRequest(BaseModel):
    """POST /api/chat/clear-memory 接口请求体。"""

    thread_id: str = Field(..., min_length=1, description="需要清空的 LangGraph 线程标识符")
