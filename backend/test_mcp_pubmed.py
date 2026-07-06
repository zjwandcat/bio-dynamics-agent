"""测试 MCP PubMed 检索 + 限速验证。"""
import asyncio
import time
from app.mcp_client import get_mcp_client, _ncbi_last_request_ts, _NCBI_MIN_INTERVAL
from app.config import settings


async def test_search_and_rate_limit():
    mcp = get_mcp_client()
    print("=== MCP PubMed 测试 ===")
    print(f"MCP_PUBMED_URL: '{settings.MCP_PUBMED_URL}' (空则走 E-utilities 兜底)")
    print(f"NCBI_EMAIL: '{settings.NCBI_EMAIL}'")
    print(f"NCBI_API_KEY: {'已配置' if settings.NCBI_API_KEY else '未配置（限 3 req/s）'}")
    print(f"限速间隔: {_NCBI_MIN_INTERVAL}s (max 2 req/s)")
    print()

    # 测试 1：单次查询验证功能正确性
    query = "tamoxifen ER alpha inhibition IC50"
    print(f"--- 查询: {query} ---")
    t0 = time.time()
    articles, records, _ = await mcp.search_pubmed(query, max_results=3)
    elapsed = time.time() - t0
    print(f"总耗时: {elapsed:.2f}s (含 esearch + efetch + 0.5s 限速间隔)")
    print(f"返回文章数: {len(articles)}")
    if records:
        r = records[0]
        print(f"  状态: {r['status']}")
    for i, art in enumerate(articles):
        pmid = art.get("pmid", "")
        title = art.get("title", "")[:70]
        print(f"  [{i+1}] PMID:{pmid} - {title}")
    print()

    # 测试 2：连续两次调用验证限速
    print("--- 限速验证：连续两次 search_pubmed ---")
    t0 = time.time()
    await mcp.search_pubmed("aspirin COX inhibition", max_results=1)
    t1 = time.time()
    await mcp.search_pubmed("ibuprofen COX2 inhibitor", max_results=1)
    t2 = time.time()
    # 每次 search_pubmed = 2 次 NCBI 请求 (esearch + efetch)，间隔 0.5s
    # 理论最小耗时: 0.5 + resp1 + 0.5 + resp2 + 0.5 + resp3 + 0.5 + resp4
    print(f"第1次: {t1-t0:.2f}s (2 次 NCBI 请求 + 0.5s 限速)")
    print(f"第2次: {t2-t1:.2f}s (2 次 NCBI 请求 + 0.5s 限速)")
    print(f"总间隔: {t2-t0:.2f}s")
    # 验证：4 次 NCBI 请求至少需要 3 * 0.5s = 1.5s 限速等待
    min_expected = 3 * _NCBI_MIN_INTERVAL
    if t2 - t0 >= min_expected:
        print(f"✅ 限速符合预期（>= {min_expected:.1f}s）")
    else:
        print(f"⚠️ 限速可能不足（期望 >= {min_expected:.1f}s，实际 {t2-t0:.2f}s）")


if __name__ == "__main__":
    asyncio.run(test_search_and_rate_limit())
