"""任务 D：RAG 检索质量诊断脚本

测试 EGFR 通路的关键查询，分析：
1. 命中率（多少查询返回了相关参数）
2. 来源分布（BioModels vs PubMed vs 其他）
3. 参数质量（是否有具体数值，来源是否权威）
4. 查询重写效果
5. 检索方法分布（semantic / bm25 / hybrid）

用法：python scripts/diagnose_rag.py
"""

import sys
import os
import json
from pathlib import Path
from collections import Counter

# 确保从 backend 目录运行
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))
os.chdir(backend_dir)

from app.rag_client import RagClient
from app.rag_collections import get_rag_collections


# EGFR 通路关键查询（覆盖上游/中游/下游/反馈）
TEST_QUERIES = [
    "EGF EGFR binding phosphorylation rate constant",
    "Ras GTPase activation rate k_on",
    "Raf MEK ERK cascade phosphorylation kcat",
    "DUSP negative feedback transcription rate",
    "EGFR internalization degradation rate",
    "SOS Grb2 adapter binding Kd",
    "RasGAP GTP hydrolysis rate",
    "MEK double phosphorylation rate constant",
]


def print_separator(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def diagnose_single_query(rag: RagClient, query: str, idx: int):
    """诊断单个查询的检索质量。"""
    print_separator(f"查询 {idx}: {query}")

    try:
        results, insights = rag.search_params_hybrid(
            query, species_context="Human", top_k=5
        )
    except Exception as e:
        print(f"  ❌ 检索失败: {e}")
        return {"query": query, "hit": False, "error": str(e), "results": []}

    # 检索洞察
    print(f"\n  📝 查询重写:")
    print(f"     原始: {query}")
    print(f"     重写: {insights.get('rewritten_query', 'N/A')}")
    if insights.get("rewrites"):
        print(f"     扩展词: {', '.join(insights['rewrites'][:5])}")

    print(f"\n  📊 检索统计:")
    print(f"     总候选: {insights.get('total_candidates', 0)}")
    print(f"     来源分布: {dict(insights.get('source_distribution', {}))}")

    # 结果分析
    if not results:
        print(f"\n  ❌ 无检索结果")
        return {"query": query, "hit": False, "results": []}

    print(f"\n  ✅ 命中 {len(results)} 条结果:")

    has_numeric_value = False
    has_biomodels_source = False
    source_types = []

    for i, rec in enumerate(results[:5]):
        print(f"\n  --- 结果 {i+1} ---")
        print(f"     参数名: {rec.get('parameter_name', rec.get('name', 'N/A'))}")
        print(f"     数值: {rec.get('value', 'N/A')} {rec.get('unit', '')}")

        value = rec.get('value')
        if value is not None and str(value).replace('.', '').replace('-', '').replace('e', '').replace('E', '').isdigit():
            has_numeric_value = True

        source = rec.get('source', 'N/A')
        print(f"     来源: {source}")
        if 'BioModels' in str(source) or 'BIOMD' in str(source):
            has_biomodels_source = True
        source_types.append(_classify_source(source))

        print(f"     置信度: {rec.get('confidence', 'N/A')}")
        print(f"     检索方法: {rec.get('_retrieval_method', 'N/A')}")
        print(f"     类型: {rec.get('type', 'N/A')}")

        # 相关性分数
        score = rec.get('_rerank_score', rec.get('_relevance_score', 'N/A'))
        print(f"     相关性分数: {score}")

        # 文本片段（截断）
        text = rec.get('text', rec.get('search_text', ''))
        if text:
            print(f"     文本: {text[:120]}...")

    return {
        "query": query,
        "hit": True,
        "result_count": len(results),
        "has_numeric_value": has_numeric_value,
        "has_biomodels_source": has_biomodels_source,
        "source_types": source_types,
        "rewritten": insights.get('rewritten_query', '') != query,
    }


def _classify_source(source: str) -> str:
    """分类来源类型。"""
    s = str(source).lower()
    if 'biomodels' in s or 'biomd' in s:
        return 'BioModels'
    if 'pmid' in s or 'pubmed' in s:
        return 'PubMed'
    if 'chebi' in s or 'uniprot' in s or 'kegg' in s:
        return 'BioDB'
    if 'estimate' in s or 'default' in s:
        return 'Estimate'
    return 'Other'


def diagnose_collection_stats():
    """诊断各 collection 的记录数。"""
    print_separator("ChromaDB Collection 统计")

    try:
        rag_collections = get_rag_collections()
        for name in ['biodynamics_mechanism', 'biodynamics_parameter',
                     'biodynamics_experiment', 'biodynamics_evidence']:
            try:
                coll = rag_collections.get_collection(name)
                count = coll.count()
                print(f"  {name}: {count} 条记录")
            except Exception as e:
                print(f"  {name}: 无法访问 ({e})")
    except Exception as e:
        print(f"  ⚠️  v2 collections 不可用: {e}")

    # legacy collection
    try:
        rag = RagClient()
        if rag.available:
            count = rag._collection.count()
            print(f"  biodynamics_params (legacy): {count} 条记录")
    except Exception as e:
        print(f"  biodynamics_params (legacy): 无法访问 ({e})")


def main():
    print_separator("任务 D：RAG 检索质量诊断")
    print(f"  测试通路: EGFR")
    print(f"  测试查询数: {len(TEST_QUERIES)}")

    # 1. Collection 统计
    diagnose_collection_stats()

    # 2. 初始化 RagClient
    print_separator("初始化 RagClient")
    try:
        rag = RagClient()
        if not rag.available:
            print("  ❌ RagClient 不可用（ChromaDB 未连接）")
            return
        print(f"  ✅ RagClient 就绪")
        print(f"     Collection: {rag.collection_name}")
    except Exception as e:
        print(f"  ❌ 初始化失败: {e}")
        return

    # 3. 逐个查询诊断
    results = []
    for i, query in enumerate(TEST_QUERIES, 1):
        r = diagnose_single_query(rag, query, i)
        results.append(r)

    # 4. 汇总统计
    print_separator("汇总统计")

    hit_count = sum(1 for r in results if r.get('hit'))
    numeric_count = sum(1 for r in results if r.get('has_numeric_value'))
    biomodels_count = sum(1 for r in results if r.get('has_biomodels_source'))
    rewritten_count = sum(1 for r in results if r.get('rewritten'))

    print(f"  命中率: {hit_count}/{len(results)} ({hit_count/len(results)*100:.0f}%)")
    print(f"  有数值结果: {numeric_count}/{len(results)} ({numeric_count/len(results)*100:.0f}%)")
    print(f"  有 BioModels 来源: {biomodels_count}/{len(results)} ({biomodels_count/len(results)*100:.0f}%)")
    print(f"  查询重写生效: {rewritten_count}/{len(results)} ({rewritten_count/len(results)*100:.0f}%)")

    # 来源分布
    all_sources = []
    for r in results:
        all_sources.extend(r.get('source_types', []))
    source_dist = Counter(all_sources)
    print(f"\n  来源分布:")
    for src, cnt in source_dist.most_common():
        print(f"     {src}: {cnt}")

    # 诊断结论
    print_separator("诊断结论")

    if hit_count < len(results) * 0.5:
        print("  ⚠️  命中率低于 50% — 参数库覆盖不足，需要扩充")
    else:
        print("  ✅ 命中率正常")

    if numeric_count < hit_count * 0.5:
        print("  ⚠️  数值参数占比低 — 检索到的多为文本描述，缺乏具体数值")
    else:
        print("  ✅ 数值参数占比正常")

    if biomodels_count < hit_count * 0.3:
        print("  ⚠️  BioModels 来源占比低 — 缺乏权威计算模型参数")
    else:
        print("  ✅ BioModels 来源占比正常")

    if rewritten_count < len(results) * 0.5:
        print("  ⚠️  查询重写生效少 — 可能 LLM 重写失败或未触发")
    else:
        print("  ✅ 查询重写生效正常")


if __name__ == "__main__":
    main()
