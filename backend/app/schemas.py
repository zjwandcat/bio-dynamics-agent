# BioDynamics Agent - FastAPI 请求模型
# 定义 /api/chat 接口所需的 Pydantic 结构。

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """POST /api/chat 接口请求体。"""

    user_input: str = Field(..., min_length=1, description="用户输入的生物学假说或机制描述")
    thread_id: str = Field(..., min_length=1, description="LangGraph 线程标识符")


class ClearMemoryRequest(BaseModel):
    """POST /api/chat/clear-memory 接口请求体。"""

    thread_id: str = Field(..., min_length=1, description="需要清空的 LangGraph 线程标识符")
