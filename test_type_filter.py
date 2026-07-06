"""Quick test: verify RAG type-filtered search returns kinetic_rate, not initial_concentration."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))

from app.rag_client import RagClient

rag = RagClient()
print(f"RAG available: {rag.available}")
print(f"Collection: {rag.collection_name}")

query = "EGF activation EGFR kinetic parameter Kd"

# Test 1: No type filter (old behavior)
print("\n=== Test 1: No type filter (old behavior) ===")
results_no_filter = rag.hybrid_search(query, top_k=5)
for i, r in enumerate(results_no_filter[:5]):
    pname = r.get("param_name", "")
    ptype = r.get("type", "N/A")
    pval = r.get("value", "")
    print(f"  [{i+1}] {pname} (type={ptype}) = {pval}")

# Test 2: Exclude initial_concentration (new default)
print("\n=== Test 2: exclude:initial_concentration (new default) ===")
results_filtered = rag.hybrid_search(query, top_k=5, type_filter="exclude:initial_concentration")
for i, r in enumerate(results_filtered[:5]):
    pname = r.get("param_name", "")
    ptype = r.get("type", "N/A")
    pval = r.get("value", "")
    req = r.get("reaction_equation", "N/A")
    print(f"  [{i+1}] {pname} (type={ptype}) = {pval} | reaction: {req}")

# Test 3: Only kinetic_rate
print("\n=== Test 3: type_filter=kinetic_rate only ===")
results_kinetic = rag.hybrid_search(query, top_k=5, type_filter="kinetic_rate")
for i, r in enumerate(results_kinetic[:5]):
    pname = r.get("param_name", "")
    pval = r.get("value", "")
    req = r.get("reaction_equation", "N/A")
    print(f"  [{i+1}] {pname} = {pval} | reaction: {req}")

# Test 4: search_params_hybrid (default excludes initial_concentration)
print("\n=== Test 4: search_params_hybrid (default excludes initial_concentration) ===")
reranked, insights = rag.search_params_hybrid(query, species_context="Human", top_k=5)
for i, r in enumerate(reranked[:5]):
    pname = r.get("param_name", "")
    ptype = r.get("type", "N/A")
    pval = r.get("value", "")
    req = r.get("reaction_equation", "N/A")
    print(f"  [{i+1}] {pname} (type={ptype}) = {pval} | reaction: {req}")
print(f"  Rewritten query: {insights.get('rewritten_query', '')[:100]}")
