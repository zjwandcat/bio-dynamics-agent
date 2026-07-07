"""深入检查 BIOMD0000000205 在 ChromaDB 中的参数详情。"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import chromadb

client = chromadb.PersistentClient(path=str(Path(__file__).resolve().parent.parent / "backend" / "data" / "vector_db"))

# 检查 biodynamics_parameter collection（v2 主集合）
col = client.get_collection("biodynamics_parameter")
all_data = col.get(limit=col.count(), include=["metadatas", "documents"])
metas = all_data.get("metadatas", [])
docs = all_data.get("documents", [])

print(f"Total docs: {len(metas)}")

# 找出 BIOMD0000000205 相关的记录
biomd_205_records = []
for i, m in enumerate(metas):
    doc = docs[i] if i < len(docs) else ""
    combined = str(m) + " " + str(doc)
    if "BIOMD0000000205" in combined or "BIOMD0000000205" in str(m):
        biomd_205_records.append((m, doc))

print(f"BIOMD0000000205 records: {len(biomd_205_records)}")

# 统计 param_name
param_names = Counter()
for m, _ in biomd_205_records:
    pn = m.get("param_name", "")
    param_names[pn] += 1

print(f"\nparam_name 分布 (top 30):")
for name, cnt in param_names.most_common(30):
    print(f"  {name}: {cnt}")

# 打印 k1/k2/Kd 等动力学参数
print(f"\n=== 动力学参数（k1/k2/Kd/Ki/V1/V2 等）===")
kinetic_params = []
for m, doc in biomd_205_records:
    pn = m.get("param_name", "")
    if pn in ("k1", "k2", "Kd", "Ki", "K1", "K2", "V1", "V2", "n", "kcat", "Km", "Vmax") or pn.startswith("k"):
        kinetic_params.append((m, doc))

print(f"动力学参数总数: {len(kinetic_params)}")
for m, doc in kinetic_params[:30]:
    print(f"  {m.get('param_name')}: {m.get('value')} {m.get('unit', '')} | context: {m.get('context', '')[:100]} | doc: {doc[:100] if doc else ''}")

# 打印 initial_concentration 参数
print(f"\n=== initial_concentration 参数 ===")
init_params = [(m, doc) for m, doc in biomd_205_records if m.get("param_name", "").startswith("initial_concentration")]
print(f"初始浓度参数总数: {len(init_params)}")
for m, doc in init_params[:15]:
    print(f"  {m.get('param_name')}: {m.get('value')} {m.get('unit', '')} | context: {m.get('context', '')[:100]}")

# 模拟 RAG 查询，看返回什么
print(f"\n=== 模拟 RAG 查询: 'EGF activation EGFR kinetic parameter Kd' ===")
from app.config import embedding_model
query = "EGF activation EGFR 受体 kinetic parameter Kd Km Vmax half-life degradation species Epidermal Growth Factor"
query_vec = embedding_model.embed_query(query)
results = col.query(query_embeddings=[query_vec], n_results=10, include=["metadatas", "documents", "distances"])
metas_q = results.get("metadatas", [[]])[0]
docs_q = results.get("documents", [[]])[0]
dists_q = results.get("distances", [[]])[0]
for i, (m, d, dist) in enumerate(zip(metas_q, docs_q, dists_q)):
    print(f"  [{i}] dist={dist:.4f} param={m.get('param_name', '')} value={m.get('value', '')} {m.get('unit', '')} | context: {m.get('context', '')[:80]} | doc: {d[:80] if d else ''}")
