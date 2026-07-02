import asyncio
import json
import traceback
from app.graph import compiled_workflow


async def main():
    user_input = (
        "我研究的是肿瘤微环境。已知肿瘤细胞分泌的TGF-β会抑制CD8+ T细胞活性。"
        "我要测试一种TGF-β抑制剂，请帮我建个模型看看给药后CD8+ T细胞的动态恢复情况。"
    )
    thread_id = "dbg-e2e-full-001"
    try:
        async for event in compiled_workflow.astream_events(
            {"user_input": user_input, "retry_count": 0, "messages": []},
            {"configurable": {"thread_id": thread_id}},
            version="v2",
        ):
            print("--- EVENT ---")
            print(json.dumps({
                "event": event.get("event"),
                "name": event.get("name"),
                "node": event.get("metadata", {}).get("langgraph_node"),
                "data_keys": list(event.get("data", {}).keys()) if isinstance(event.get("data"), dict) else None,
                "output_type": type(event.get("data", {}).get("output")).__name__,
            }, ensure_ascii=False, indent=2))
    except Exception:
        traceback.print_exc()


asyncio.run(main())
