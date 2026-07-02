# BioDynamics Agent - FastAPI 入口
# 提供 CORS 配置、/api/chat 流式接口、知识库更新与记忆清除接口，驱动 LangGraph 工作流。

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict

from fastapi import BackgroundTasks, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.config import settings
from app.graph import compiled_workflow
from app.schemas import ChatRequest, ClearMemoryRequest
from scripts.update_vector_db import update_vector_db


logger = logging.getLogger(__name__)

app = FastAPI(
    title="BioDynamics Agent",
    description="将生物医学定性假说转化为 ODE 定量模型并执行仿真预测。",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"


NODE_STATUS_MAP = {
    "node1_parse_network": "正在解析生物网络...",
    "node1_5_rag_search": "正在检索文献参数...",
    "node2_generate_code": "正在生成 ODE 仿真代码...",
    "node3_execute_sandbox": "正在执行仿真代码...",
    "node4_audit_and_correct": "正在审计执行结果...",
    "node5_generate_report": "正在生成预测报告...",
}

NODE_NAMES = set(NODE_STATUS_MAP.keys())


def _sse_event(payload: Dict[str, Any]) -> str:
    """将字典封装为 SSE 数据行。"""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.get("/")
async def root() -> Dict[str, str]:
    """健康检查接口。"""
    return {"status": "ok", "service": "BioDynamics Agent"}


@app.post("/api/admin/update-vector-db")
async def update_vector_db_endpoint(background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """接收前端手动触发请求，在后台更新 ChromaDB 知识库。"""

    async def _background_task() -> None:
        try:
            await asyncio.to_thread(update_vector_db, RAW_DATA_DIR)
            logger.info("知识库后台更新完成")
        except Exception as exc:
            logger.error("知识库后台更新失败：%s", exc)

    background_tasks.add_task(_background_task)
    return {
        "status": "started",
        "message": "知识库更新已启动，后台处理中...",
    }


@app.post("/api/chat/clear-memory")
async def clear_memory(request: ClearMemoryRequest) -> Dict[str, Any]:
    """清空指定 thread_id 的内存级短期记忆。"""
    try:
        compiled_workflow.checkpointer.delete_thread(request.thread_id)
    except Exception as exc:
        logger.warning("删除 thread 记忆失败（可能不存在）：%s", exc)
    return {
        "status": "ok",
        "thread_id": request.thread_id,
        "message": "短期记忆已清空",
    }


@app.post("/api/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    """接收用户假说，流式返回 LangGraph 各节点执行状态与最终结果。"""

    async def event_stream():
        initial_state: Dict[str, Any] = {
            "user_input": request.user_input,
            "retry_count": 0,
            "messages": [],
        }
        latest_token_usage: Dict[str, int] | None = None

        try:
            async for event in compiled_workflow.astream_events(
                initial_state,
                {"configurable": {"thread_id": request.thread_id}},
                version="v2",
            ):
                event_name = event.get("event", "")
                event_chain_name = event.get("name", "")
                metadata = event.get("metadata", {}) or {}
                node_name = metadata.get("langgraph_node")

                # 条件边（如 _route_after_parse）会继承父节点名，需按 event 真实名称过滤
                is_actual_node = event_chain_name in NODE_NAMES

                if event_name == "on_chain_start" and is_actual_node:
                    yield _sse_event(
                        {
                            "event": "node_start",
                            "data": NODE_STATUS_MAP[node_name],
                        }
                    )
                    continue

                if event_name != "on_chain_end" or not is_actual_node:
                    continue

                output = event.get("data", {}).get("output", {})
                if not output:
                    output = event.get("data", {})

                # 累计 Token 使用量
                if isinstance(output, dict) and output.get("token_usage"):
                    latest_token_usage = output["token_usage"]

                if node_name == "node1_parse_network":
                    if output.get("need_human_review"):
                        yield _sse_event(
                            {
                                "event": "execution_log",
                                "data": f"需要人工复核：{output.get('review_question', '')}",
                            }
                        )

                elif node_name == "node1_5_rag_search":
                    yield _sse_event(
                        {
                            "event": "rag_ready",
                            "data": {
                                "summary": output.get("rag_summary", ""),
                                "fallback": output.get("rag_fallback", False),
                            },
                        }
                    )
                    sbml_network = output.get("sbml_parsed_network")
                    if sbml_network and sbml_network.get("is_reusable"):
                        yield _sse_event(
                            {
                                "event": "execution_log",
                                "data": "已解析到可复用 SBML 模型。",
                            }
                        )

                elif node_name == "node2_generate_code":
                    yield _sse_event(
                        {
                            "event": "code_generated",
                            "data": output.get("python_code", ""),
                        }
                    )

                elif node_name == "node3_execute_sandbox":
                    yield _sse_event(
                        {
                            "event": "execution_log",
                            "data": output.get("stdout_stderr", ""),
                        }
                    )
                    image_base64 = output.get("image_base64")
                    if image_base64:
                        yield _sse_event(
                            {
                                "event": "image_ready",
                                "data": image_base64,
                            }
                        )

                elif node_name == "node4_audit_and_correct":
                    auditor_status = output.get("auditor_status")
                    if auditor_status == "retry":
                        retry_count = output.get("retry_count", 0)
                        yield _sse_event(
                            {
                                "event": "execution_log",
                                "data": f"仿真出错，正在自动纠错重试 ({retry_count}/3)...",
                            }
                        )
                    elif auditor_status == "failed":
                        yield _sse_event(
                            {
                                "event": "error",
                                "data": output.get("failure_report", "仿真执行失败。"),
                            }
                        )

                elif node_name == "node5_generate_report":
                    yield _sse_event(
                        {
                            "event": "report_ready",
                            "data": output.get("final_report", ""),
                        }
                    )
        except Exception as exc:
            yield _sse_event(
                {
                    "event": "error",
                    "data": f"工作流执行异常：{exc}",
                }
            )
        finally:
            if latest_token_usage:
                yield _sse_event(
                    {
                        "event": "token_usage",
                        "data": latest_token_usage,
                    }
                )
            yield _sse_event({"event": "end", "data": ""})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
