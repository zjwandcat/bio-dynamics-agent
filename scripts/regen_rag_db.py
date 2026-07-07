"""Regenerate BIOMD0000000205 processed JSON with type system and verify."""
import sys
import json
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from scripts.fetch_rag_data import parse_sbml_to_records

records = parse_sbml_to_records(Path(__file__).resolve().parent.parent / "backend" / "data" / "raw" / "BIOMD0000000205.xml")
print(f"Total records: {len(records)}")

type_counts = Counter(r.get("type", "unknown") for r in records)
print(f"Type distribution: {dict(type_counts)}")

print("\n--- Kinetic rate samples (with reaction equation) ---")
kinetic_samples = [r for r in records if r.get("type") == "kinetic_rate"][:5]
for r in kinetic_samples:
    pname = r.get("param_name", "")
    pval = r.get("value", "")
    punit = r.get("unit", "")
    req = r.get("reaction_equation", "N/A")
    ctx = r.get("context", "")[:160]
    print(f"  {pname} = {pval} {punit} | reaction: {req}")
    print(f"    context: {ctx}...")

print("\n--- Initial concentration samples ---")
ic_samples = [r for r in records if r.get("type") == "initial_concentration"][:3]
for r in ic_samples:
    pname = r.get("param_name", "")
    pval = r.get("value", "")
    sp = r.get("species", "")
    print(f"  {pname} = {pval} (species={sp})")

out_path = Path("backend/data/processed/BIOMD0000000205.json")
out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\nSaved updated records to {out_path}")

# Verify: check that kinetic_rate records now contain EGF/EGFR keywords
kinetic_with_egf = sum(
    1 for r in records
    if r.get("type") == "kinetic_rate"
    and ("EGF" in r.get("context", "") or "EGFR" in r.get("context", ""))
)
kinetic_total = sum(1 for r in records if r.get("type") == "kinetic_rate")
print(f"\nVerification: {kinetic_with_egf}/{kinetic_total} kinetic_rate records now contain EGF/EGFR keywords")
