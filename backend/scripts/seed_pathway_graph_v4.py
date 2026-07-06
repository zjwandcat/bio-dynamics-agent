#!/usr/bin/env python
# BioDynamics Agent v4 - Pathway Graph Seeder Script (Phase 4 / Task 4.14)
# 独立运行脚本：灌库 biodynamics_pathway_graph_v4 collection。
#
# 用法：
#   cd backend
#   python scripts/seed_pathway_graph_v4.py
#
# 行为：
#   - 遍历 10 通路，调用 PathwayGraphBuilder 构建 PathwayGraph
#   - 序列化为 JSON 写入 ChromaDB biodynamics_pathway_graph_v4 collection
#   - ChromaDB 不可用时降级为 no-op + warning（不抛异常）
#
# 注意：此脚本可独立运行，但不在测试中强制执行（避免依赖 ChromaDB 运行环境）。
# 测试中 mock seed_pathway_graph_v4() 函数，不实际调用 ChromaDB。

from __future__ import annotations

import sys
from pathlib import Path

# 确保 backend/ 在 sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def main() -> int:
    """灌库主入口。"""
    from app.pathways.pathway_graph_seeder import seed_pathway_graph_v4

    print("=" * 60)
    print("BioDynamics v4 - Pathway Graph Seeder")
    print("Collection: biodynamics_pathway_graph_v4")
    print("=" * 60)

    result = seed_pathway_graph_v4()

    print()
    print("Result:")
    for key, value in result.items():
        print(f"  {key}: {value}")

    if result.get("seeded"):
        print()
        print(f"Success: {result['pathways_count']} pathways seeded.")
        return 0
    else:
        print()
        reason = result.get("reason", "unknown")
        print(f"Skipped or failed: {reason}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
