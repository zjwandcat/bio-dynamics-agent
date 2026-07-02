import asyncio
from app.graph import compiled_workflow

async def main():
    async for event in compiled_workflow.astream_events(
        {"user_input": "A activates B", "retry_count": 0, "messages": []},
        {"configurable": {"thread_id": "dbg"}},
        version="v2",
    ):
        print(event.get("event"), event.get("metadata", {}).get("langgraph_node"))

asyncio.run(main())
