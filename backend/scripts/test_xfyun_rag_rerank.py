#!/usr/bin/env python3
"""测试 xfyun embedding + rerank 在 RAG 链路中的实际效果。"""

import asyncio
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.config import embedding_model, rerank_manager, settings
from app.rag_client import RagClient
from app.rag_collections import get_rag_collections


def _fmt_score(r: dict) -> str:
    for key in ("_rerank_score", "score", "final_score"):
        v = r.get(key)
        if isinstance(v, (int, float)):
            return f"{v:.4f}"
    return "N/A"


async def test_rag_client() -> None:
    print("=" * 60)
    print("测试 1: RagClient.search_params_hybrid (EGF-EGFR 参数检索)")
    print("=" * 60)
    rc = RagClient()
    print(f"RagClient available: {rc.available}")
    print(f"当前 embedding provider: {settings.EMBEDDING_PROVIDER}, model: {settings.EMBEDDING_MODEL}")

    query = "EGF stimulated EGFR phosphorylation kinetic rate human"
    results, summary = rc.search_params_hybrid(query, top_k=5)
    print(f"返回结果数: {len(results)}")
    print(f"检索摘要: total_candidates={summary.get('total_candidates')}, source_distribution={summary.get('source_distribution')}")
    for i, r in enumerate(results[:5]):
        print(
            f"[{i+1}] score={_fmt_score(r)} | "
            f"param={r.get('param_name', '')} | "
            f"value={r.get('value', '')} {r.get('unit', '')} | "
            f"type={r.get('type', '')} | "
            f"source={r.get('source', '')[:50]}"
        )
    print()


async def test_rag_collections() -> None:
    print("=" * 60)
    print("测试 2: RagCollections.search_parameter_hybrid (四路 RAG 参数路)")
    print("=" * 60)
    rag_cols = get_rag_collections()
    print(f"RagCollections available: {rag_cols.available}")
    print(f"Stats: {rag_cols.stats()}")

    query = "EGF EGFR binding affinity Kd human"
    results, summary = rag_cols.search_parameter_hybrid(query, top_k=5)
    print(f"返回结果数: {len(results)}")
    print(f"检索摘要: {summary}")
    for i, r in enumerate(results[:5]):
        print(
            f"[{i+1}] score={_fmt_score(r)} | "
            f"param={r.get('param_name', r.get('name', ''))} | "
            f"value={r.get('value', '')} {r.get('unit', '')} | "
            f"source={r.get('source', r.get('source_model', ''))[:50]}"
        )
    print()


async def test_rerank() -> None:
    print("=" * 60)
    print("测试 3: RerankManager 对当前 agent 信息的排序能力")
    print("=" * 60)
    if rerank_manager is None:
        print("rerank_manager 未初始化，跳过")
        return

    # 模拟 agent 上下文中的一组候选文档
    query = "BioDynamics Agent 默认使用哪个 embedding 模型和 rerank 模型"
    docs = [
        "The default embedding model is sentence-transformers/all-MiniLM-L6-v2.",
        "BioDynamics Agent currently defaults to xop3qwen8bembedding for embedding and xop3qwen8breranker for rerank.",
        "OpenRouter rerank model cohere/rerank-4-pro is also configured.",
        "The primary LLM is glm-5.1 from Zhipu BigModel.",
        "ChromaDB persist directory is ./data/vector_db.",
    ]

    # 触发 health_check 刷新可用列表
    reports = rerank_manager.health_check()
    print("Rerank 候选健康状态:")
    for rep in reports:
        status = "OK" if rep["available"] else "FAIL"
        print(f"  [{status}] {rep['display_name']}")

    results, used = rerank_manager.rerank(query, docs, top_n=3)
    print(f"\n实际使用的 rerank 候选: {used.display_name if used else 'None'}")
    print(f"返回结果数: {len(results)}")
    for i, item in enumerate(results):
        idx = item.get("index", -1)
        score = item.get("relevance_score", item.get("score", "N/A"))
        text = docs[idx] if 0 <= idx < len(docs) else ""
        print(f"[{i+1}] idx={idx} score={score:.4f} | {text[:80]}")
    print()


async def main() -> int:
    await test_rag_client()
    await test_rag_collections()
    await test_rerank()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
