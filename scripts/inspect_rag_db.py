"""检查 ChromaDB 中 BIOMD0000000205 模型的参数类型分布。"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import chromadb

client = chromadb.PersistentClient(path=str(Path(__file__).resolve().parent.parent / "backend" / "data" / "vector_db"))
collections = client.list_collections()
print(f"Collections: {[c.name for c in collections]}")

for col_name in [c.name for c in collections]:
    col = client.get_collection(col_name)
    count = col.count()
    print(f"\n=== {col_name} ({count} docs) ===")
    if count == 0:
        continue

    # 抽样 100 条记录看 param_name 分布
    sample = col.get(limit=min(count, 500), include=["metadatas"])
    metas = sample.get("metadatas", [])

    # 统计 param_name 类型
    param_names = Counter()
    sources = Counter()
    biomd_205_count = 0
    for m in metas:
        pn = m.get("param_name", "")
        param_names[pn] += 1
        src = m.get("source", "")
        sources[src[:60]] += 1
        if "BIOMD0000000205" in str(m):
            biomd_205_count += 1

    print(f"  BIOMD0000000205 相关: {biomd_205_count}")
    print(f"  param_name top 15:")
    for name, cnt in param_names.most_common(15):
        print(f"    {name}: {cnt}")
    print(f"  source top 10:")
    for src, cnt in sources.most_common(10):
        print(f"    {src}: {cnt}")

# 专门查询 BIOMD0000000205 的参数
print("\n\n=== BIOMD0000000205 专用查询 ===")
for col_name in [c.name for c in collections]:
    col = client.get_collection(col_name)
    if col.count() == 0:
        continue
    # 查询包含 BIOMD0000000205 的文档
    try:
        results = col.get(where={"source": {"$contains": "BIOMD0000000205"}}, limit=1000, include=["metadatas"])
        metas = results.get("metadatas", [])
        if metas:
            print(f"\n{col_name}: {len(metas)} BIOMD0000000205 records")
            param_names = Counter()
            for m in metas:
                pn = m.get("param_name", "")
                param_names[pn] += 1
            print(f"  param_name 分布:")
            for name, cnt in param_names.most_common(20):
                print(f"    {name}: {cnt}")
            # 打印前 5 条非 initial_concentration 的参数
            print(f"  非 initial_concentration 的前 10 条:")
            non_init = [m for m in metas if not m.get("param_name", "").startswith("initial_concentration")]
            for m in non_init[:10]:
                print(f"    {m.get('param_name')}: {m.get('value')} {m.get('unit')} (source: {m.get('source', '')[:80]})")
    except Exception as e:
        print(f"  {col_name} 查询失败: {e}")
