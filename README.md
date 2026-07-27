# BioDynamics Agent — AI-Native Biomedical Signaling Pathway Simulation Agent

> **Status (verified 2026-07-19):** Code-complete for the v3 backbone, v4 hook
> chains, and the Scientific Alignment (SA) module. **Benchmark reproducibility
> is NOT achieved.** Of the 10 pathway gold-standard benchmark suite, **1/10
> passes honestly** (APOPTOSIS only). The remaining 9/10 fail with quantitative
> discrepancies (explosions, flat-lines, peak-time violations, mass-conservation
> failures). See the **Benchmark status** and **Known issues** sections for
> root-cause analysis and the priority queue from
> `SCIENTIFIC_CALIBRATION_REPORT.md` (overall score 1.2/10).

This README documents what is actually present in the codebase and what is
actually working as of the verification pass. Every claim below is grounded in
direct code inspection and executed tests (28 backend test files, 1046 passed /
57 failed / 2 skipped), not in marketing copy or other markdown files.

---

## Table of contents

1. [What this system is](#what-this-system-is)
2. [Verified architecture](#verified-architecture)
3. [What actually works](#what-actually-works)
4. [What does NOT work (benchmark evidence)](#what-does-not-work-benchmark-evidence)
5. [Known code-level bugs (with file:line evidence)](#known-code-level-bugs-with-fileline-evidence)
6. [The v3 Supervisor-Worker pipeline](#the-v3-supervisor-worker-pipeline)
7. [v4 hook chains (gated, default OFF)](#v4-hook-chains-gated-default-off)
8. [RAG knowledge base](#rag-knowledge-base)
9. [MCP biomedical terminology integration](#mcp-biomedical-terminology-integration)
10. [Pathway specialists](#pathway-specialists)
11. [Reaction IR & ODE templates](#reaction-ir--ode-templates)
12. [5-level Validation Pyramid](#5-level-validation-pyramid)
13. [Scientific Alignment (SA) — the rigor layer](#scientific-alignment-sa--the-rigor-layer)
14. [Feature flags (rollback-safe by design, not by test)](#feature-flags-rollback-safe-by-design-not-by-test)
15. [Frontend](#frontend)
16. [API reference](#api-reference)
17. [Multi-provider LLM / embedding / rerank](#multi-provider-llm--embedding--rerank)
18. [Testing](#testing)
19. [Benchmark status (10-pathway suite)](#benchmark-status-10-pathway-suite)
20. [Deep root-cause analysis per failing pathway](#deep-root-cause-analysis-per-failing-pathway)
21. [Project structure](#project-structure)
22. [Design principles (aspirational vs verified)](#design-principles-aspirational-vs-verified)
23. [License](#license)

---

## What this system is

BioDynamics Agent is an AI-native agent system that converts a qualitative
biomedical hypothesis expressed in natural language into a quantitative ODE
simulation of a signaling pathway, then wraps the result in a multi-layer
scientific-alignment audit. The intended pipeline is:

1. **Terminology grounding** — biomedical entities normalized via MCP
   ontologies (UMLS, medical-terminologies, OpenBioMed, PubMed search).
2. **Mechanism parsing** — natural-language input parsed into a reaction
   network and knowledge graph.
3. **Knowledge retrieval (RAG)** — kinetic parameters retrieved from a local
   vector store (mechanism / parameter / experiment / evidence collections)
   with hybrid search + rerank + query rewriting, falling back to PubMed
   E-utilities and KEGG/Reactome/UniProt/ChEMBL online lookup.
4. **ODE construction** — Jinja2 ODE template selected by a rule engine and
   rendered with retrieved parameters.
5. **Deterministic simulation** — ODE system integrated in a sandboxed
   executor with non-negativity enforcement and numerical-stability retry.
6. **Validation** — a 5-level pyramid checks mass conservation, SBML/BioModels
   consistency, cross-pathway shared species, benchmark pass criteria, and
   hypothesis falsifiability. **(In practice, the pyramid is a soft gate —
   see Known issues #1.)**
7. **Scientific Alignment post-processing** — a rule-based audit layer.

**Honest summary:** The pipeline scaffolding runs end-to-end, but the
*quantitative correctness* of stages 5–7 is not yet at benchmark-passing
quality. The system is best characterized as an architecture-complete
prototype with known scientific-correctness defects.

---

## Verified architecture

```
┌────────────────────── Frontend (Next.js 16.2.10 + React 19.2.4) ──────────────────────┐
│  /            Minimal Auto-Chat: Header · NL Input · 7-step AI Workflow · Results Tabs  │
│  /advanced    Archived 4-pane Scientific IDE (Project · Workspace · Validation · AI)     │
│  /benchmarks  10-pathway Gold Standard Benchmark runner                                 │
│  /report/[id] Per-simulation report viewer                                              │
│  State: Zustand 5.0.14 store; SSE ingestion in lib/store.ts ingestSSEEvent()            │
└──────────────────────────────────────────────────────────────────────────────────────────┘
                  │ SSE (text/event-stream, POST /api/chat)
                  ▼
┌────────────────────── Backend (FastAPI 0.139.0 + LangGraph 1.2.7) ─────────────────────┐
│  /api/chat                 v3 Supervisor-Worker workflow (compiled_workflow_v3)          │
│  /api/v4/*                 6 REST endpoints (pathways/graph/sim/benchmark/reports/sweep) │
│  /api/admin/*              vector DB update, RAG status                                  │
│  /api/models/status        LLM/embedding/rerank provider health                          │
│                                                                                          │
│  v3 backbone:  pre_router → supervisor → 8 workers + clarification_node                  │
│  v4 hooks:     ontology/dynamic_router · pathway/specialist/crosstalk ·                  │
│                sbml_grounder/calibration/sensitivity · validation_pyramid · hypothesis    │
│  SA post-loop: consistency · critic · multi-dim · validation_rules · scientific_review · │
│                biomodels_calibration (runs after worker_report, flag-gated)               │
│                                                                                          │
│  Storage: ChromaDB 1.5.9 (4 collections) · LangGraph MemorySaver                         │
│  RAG:     query rewrite + semantic + BM25 hybrid + 3-provider rerank + online fallback   │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

**Verified present:**
- `backend/app/graph_v3.py` — v3 LangGraph workflow + v4 hook chains.
- `backend/app/main.py` — FastAPI app, `/api/chat` SSE, SA post-loop trigger.
- `backend/app/supervisor.py` — `AGENT_REGISTRY` (6 v1 agents) and
  `AGENT_REGISTRY_V2` (10 v2 agents). Both verified present.
- `backend/app/v4_endpoints.py` — 6 REST endpoints under `/api/v4`, with the
  10-pathway `_REGISTRY_TO_FRONTEND` mapping.
- `backend/app/config.py` — 3 coarse + 13 fine v4 flags + 18 SA sub-flags.
- `backend/app/scientific_alignment/` — 28-file SA module.
- `frontend/package.json` — Next.js 16.2.10, React 19.2.4, Zustand 5.0.14,
  recharts 3.9.1, @xyflow/react 12.11.2. Verified by direct read.
- `Dockerfile` — multi-stage build (Python 3.11-slim backend + Node 20-slim
  frontend). **Note:** Dockerfile pins Python 3.11-slim while the project
  badge below claims 3.14; the local venv uses Python 3.14.6 in practice.

[![License: MIT](./LICENSE)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%20(Dockerfile)%20|%203.14%20(venv)-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.139.0-009688)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-16.2.10-black)](https://nextjs.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.2.7-green)](https://langchain-ai.github.io/langgraph/)
[![Benchmark](https://img.shields.io/badge/benchmark-1%2F10%20real%20pass-red)](#benchmark-status-10-pathway-suite)

---

## What actually works

The following capabilities were verified by direct code inspection and, where
feasible, by running tests. "Works" means the code path executes and produces
output; it does **not** imply the output is scientifically correct (see next
section).

| Capability | Verified by | Notes |
|---|---|---|
| `/api/chat` SSE workflow streams all node events | code in `graph_v3.py`, `main.py` | End-to-end pipeline runs; produces a markdown report |
| 6 v4 REST endpoints respond | `v4_endpoints.py` | `/api/v4/pathways`, `/graph`, `/simulation/run`, `/benchmark/{class}`, `/reports/{id}`, `/simulation/sweep` |
| 10 pathway specialists registered | `pathways/specialists/` directory listing | `egfr`, `mapk`, `pi3k_akt_mtor`, `jak_stat`, `tgf_beta`, `wnt`, `p53`, `nf_kappa_b`, `apoptosis`, `cell_cycle` |
| 10 canonical reference YAMLs load | `knowledge/canonical/` listing | Path-traversal guard in `canonical_loader.py` verified |
| 10 experiment-chain rule YAMLs exist | `knowledge/experiments/` listing | **But the planner bypasses them by default — see Known issues #4** |
| 10 literature gold standards load | `knowledge/gold_standard/` listing | `literature_*.yaml` per pathway |
| Reaction IR v2 schema validates | `reaction_ir_v2/` | 17+ mechanism types, Pydantic-validated |
| 9 v3 + 9 v4 ODE Jinja2 templates present | `ode_templates/`, `ode_templates_v2/` | Templates render; numerical correctness is a separate question |
| ChromaDB 4 collections persist | `data/vector_db/` directory | `biodynamics_mechanism`, `_parameter`, `_experiment`, `_evidence` |
| RAG hybrid search + rerank executes | `rag_client.py` | 3-provider rerank cascade: Xfyun → OpenRouter → SiliconFlow |
| MCP client registers 4 tools | `mcp_client.py` | OpenBioMed / UMLS / medical-terminologies / pubmed-search; falls back gracefully when `MCP_ENABLED=false` |
| Multi-provider LLM with 0.5 s fallback | `config.py` `FallbackLLM` | Provider chain read from `.env` |
| Frontend Zustand store ingests 30+ SSE event types | `frontend/lib/store.ts` | `ingestSSEEvent` dispatch verified |
| Validation Pyramid executes L1–L5 | `validation_v2/level1_internal.py` … `level5_hypothesis.py` | **But it is a soft gate — see Known issues #1** |
| SA module produces 9 cards | `scientific_alignment/` | Triggered by `main.py::_run_scientific_alignment_postprocess` when master flag ON |

---

## What does NOT work (benchmark evidence)

This is the most important section. The user's explicit goal is *"Benchmark
multiple pathways must be correctly reproducible"* — that goal is **NOT met**.

**Real benchmark pass rate: 1/10** (APOPTOSIS only, and even APOPTOSIS has a
degraded C9 check).

The `BENCHMARK_REPORT.md` at
`backend/data/sa_logs/all_10_pathways/BENCHMARK_REPORT.md` claims "100% pass
rate" — this is **false**. The file's own header reads "总通路数: 1" (total
pathways: 1) while the title says "10 Pathway Benchmark Report". Only
`MAPK_ERK` is listed, and that listing is itself a fake pass (see below).

Per-pathway `12_check_report.json` results, read directly from disk:

| Pathway | `overall_passed` | Real status | Why |
|---|---|---|---|
| APOPTOSIS | `true` | **REAL PASS (with degraded C9)** | Only pathway that actually passes; C9 still degraded |
| MAPK_ERK | `true` | **FAKE PASS** | `csv_max_value=1.0`, `dynamic_species_count=0` — flat-line simulation |
| EGFR_RTK | `false` | FAIL | pEGFR `peak_time=120.0` (expected `[5.0, 10.0]`); 5/12 checks failed |
| PI3K_AKT_mTOR | `false` | FAIL | C9 degraded pass, C11 coverage_ratio=0.0 |
| p53_signaling | `false` | FAIL | C9 degraded pass, C11 coverage_ratio=0.0 |
| NF_kB | `false` | FAIL | C9 degraded pass, C11 coverage_ratio=0.0 |
| JAK_STAT | `false` | FAIL | C9 degraded pass, C11 coverage_ratio=0.0 |
| TGF_beta | `false` | FAIL | `max_fold_change=830` (explosion) |
| Wnt | `false` | FAIL | `max_fold_change=544147.27` (explosion) + `sbml_model_id=BIOMD0000000055` (Arabidopsis circadian misused for WNT) |
| Cell_Cycle | `false` | FAIL | `max_fold_change=16220712` (16M× explosion) |

---

## Known code-level bugs (with file:line evidence)

The following 8 defects were verified by reading the actual source code (not
just outputs). They are the root causes of the benchmark failures above.

### Bug 1 — Validation Pyramid is a soft gate, runs AFTER `worker_report`

**Location:** [backend/app/graph_v3.py:1888-1894](backend/app/graph_v3.py)

The `_validation_pyramid_hook` edge is added **after** `worker_report`, so a
pyramid failure cannot route the workflow back to the simulation engineer for
a retry. The hook also wraps its body in `try/except` and returns `{}` on any
exception, so failures are silently swallowed.

**Code comment found in `validation_v2/validation_agent.py:354-374`:**
> "整个验证金字塔为软门不阻断"

**Impact:** Even when the pyramid detects a quantitative failure, the pipeline
proceeds to emit a "passing" report. This is why 9/10 pathways produce
`overall_passed=false` `12_check_report.json` files yet still emit final
reports.

### Bug 2 — `V4_VALIDATION_PYRAMID_ENABLED` defaults to `false`

**Location:** [backend/app/config.py:333-335](backend/app/config.py)

```python
V4_VALIDATION_PYRAMID_ENABLED: bool = os.getenv(
    "V4_VALIDATION_PYRAMID_ENABLED", "false"
).lower() == "true"
```

**Impact:** Out of the box, the validation pyramid is disabled entirely. The
`12_check_report.json` files in `data/sa_logs/all_10_pathways/` are produced
by the SA module's own checks, not by the L4/L5 pyramid levels. To get the
pyramid to actually run, the user must set `V4_VALIDATION_PYRAMID_ENABLED=true`
in `.env`.

### Bug 3 — Evidence Fuser uses positional matching, not semantic matching

**Location:** [backend/app/scientific_alignment/evidence_fuser.py:148-157, 298-326](backend/app/scientific_alignment/evidence_fuser.py)

```python
def _get_positional_item(evidence_list, index):
    ...
    return evidence_list[index]
```

Code comment found: *"位置匹配：assertion[i] ← evidence_list[i]"*.

**Impact:** Discussion sentences are tagged with evidence by list position,
not by content. If `assertions[0]` discusses pEGFR peak time but
`evidence_list[0]` is a PubMed abstract about apoptosis, the tag `[A]PubMed`
is attached to a wrong statement. This is why the SA report flags
"100% Fake Grounding" in the Evidence Flow audit.

### Bug 4 — `SA_SEVEN_AXIS=OFF` bypasses forbidden-experiment check

**Location:** [backend/app/scientific_alignment/experiment_planner.py:760-765](backend/app/scientific_alignment/experiment_planner.py)

```python
if not settings.is_sa_feature_enabled("SEVEN_AXIS"):
    return ExperimentPlan(enabled=False, skipped=True, ...)
```

The `FORBIDDEN_EXPERIMENTS` list (line 317) and `_check_forbidden` function
(line 627) exist, but the early return at line 760 means **when
`SA_SEVEN_AXIS=OFF` (the default, see `config.py:404`), the forbidden check
is never reached.**

**Impact:** EGFR qPCR-as-primary-phosphorylation-validation experiments leak
into the plan because the rule engine is gated off by default. The
`knowledge/experiments/*.yaml` rule files exist (10 of them, verified) but
are not loaded when the flag is off.

### Bug 5 — BioModels Track B hardcodes `max_relative_error = 1.0`

**Location:** [backend/app/biomodels_client.py:1036](backend/app/biomodels_client.py)

```python
max_relative_error = 1.0  # Track B 无法计算真实相对误差
```

Track A (lines 748-937) uses real RoadRunner simulation to compute
peak/value/amplification diffs. Track B (lines 944-1056) is a degraded
fallback that returns a hardcoded `1.0` relative error, making the SBML
consistency check meaningless when Track A fails (e.g. when `roadrunner` is
not installed or the SBML model is malformed).

**Impact:** When the system falls back to Track B, every BioModels comparison
reports the same `max_relative_error=1.0`, hiding real disagreements between
the agent's ODE simulation and the canonical BioModels reference.

### Bug 6 — Coverage check uses `or` instead of `and`

**Location:** [backend/_run_all_10_pathways.py:981](backend/_run_all_10_pathways.py)

```python
passed = coverage >= min_coverage or pmid_count >= 1
```

**Impact:** A single PubMed hit (`pmid_count >= 1`) bypasses the coverage
threshold entirely. This is why 9/10 pathways report `coverage_ratio=0.0`
yet still mark C11 as `passed=true`. The correct operator should be `and`
(both coverage AND at least one external citation required).

### Bug 7 — "Degraded pass" semantics: `passed=true, degraded=true`

**Location:** [backend/_run_all_10_pathways.py:843-849](backend/_run_all_10_pathways.py)

```python
if sbml_role == "none" or not sbml_model_id:
    return {"passed": True, "degraded": True, ...}
```

**Impact:** When SBML grounding is absent, the check auto-passes with a
`degraded=true` flag. Downstream aggregation typically only inspects
`passed`, so degraded passes are counted as real passes — this is how the
`BENCHMARK_REPORT.md` arrives at its false "100%" claim.

### Bug 8 — `BIOMD0000000205` misassigned to 9 non-EGFR pathways

**Location:** `backend/data/raw/BIOMD0000000205.xml:3-5`

```xml
<model id="Ung2008_EGFR_Endocytosis" name="Ung2008_EGFR_Endocytosis">
```

`BIOMD0000000205` is the **Ung 2008 EGFR Endocytosis** model (curated by
Dharuri & Zong). The SA report found it was used 362 times across 9 non-EGFR
pathways (MAPK, PI3K, p53, NF-κB, JAK-STAT, WNT, TGF-β, APOPTOSIS,
CELL_CYCLE). Additionally, `BIOMD0000000055` (an Arabidopsis circadian clock
model) was used for WNT — see WNT row in the benchmark table.

**Impact:** Every non-EGFR pathway's L2 SBML consistency check is comparing
its ODE simulation against an unrelated EGFR endocytosis model. Track A
numerical diffs are then meaningless; Track B (Bug 5) returns the hardcoded
`1.0`, masking the misassignment.

---

## The v3 Supervisor-Worker pipeline

Implemented in [backend/app/graph_v3.py](backend/app/graph_v3.py) (`build_workflow_v3`).
The supervisor reads an `execution_plan` produced by `pre_router` and dispatches
workers one at a time. Three run modes are supported: `manual` (user-selected
modules), `auto_fast` (skip PK/PD and evidence), and `auto_standard`
(rule-prioritized with LLM fallback for `needs_pkpd` / `needs_evidence`).

| Node | Role | Implementation |
|---|---|---|
| `pre_router` | Analyze run mode, build `execution_plan` | graph_v3.py |
| `supervisor` | Dispatch next worker, trigger human-in-loop | graph_v3.py |
| `clarification_node` | Await user clarification (10-min timeout) | graph_v3.py |
| `worker_mcp` | MCP biomedical term normalization | nodes.py `node0_mcp_term_lookup` |
| `worker_mechanism` | Mechanism parsing + knowledge graph | nodes_v2.py `n1`–`n4` |
| `worker_rag` | Mechanism RAG + parameter RAG (per-edge) | nodes_v2.py `n3`, `n5` |
| `worker_pkpd` | PK/PD inference, dose response, combination synergy | nodes_v2.py |
| `worker_ode` | ODE code generation (template + rule engine) | nodes_v2.py `n6` |
| `worker_sandbox` | Sandboxed simulation, AST precheck, stability retry | nodes_v2.py `n7` |
| `worker_validator` | SBML validation | nodes_v2.py |
| `worker_report` | Scientific features + experiment/evidence RAG + report | nodes_v2.py `n8`–`n11` |

The default full plan is: `worker_mcp → worker_mechanism → worker_rag →
worker_pkpd → worker_ode → worker_sandbox → worker_validator → worker_report`.

**Verified working:** the SSE stream reaches all 8 workers in order and emits
a final markdown report. **Not verified:** quantitative correctness of the
ODE system emitted by `worker_ode` (this is the root cause of benchmark
failures — see Deep root-cause analysis below).

---

## v4 hook chains (gated, default OFF)

All v4 hooks are feature-flagged. **With every flag OFF the graph behaves as
pure v3** — this is verified by code inspection and by the
`test_feature_flag_convergence.py` and `test_p4_flag_off_isolation.py` test
suites (which currently surface ~14 failures, all attributable to Bug 2 —
individual v4 sub-flags defaulting to `true` instead of `false`; see Testing).

| Hook chain | Fires after | Gate flag |
|---|---|---|
| `ontology_hook` | START | `V4_ONTOLOGY_AGENT_ENABLED` |
| `dynamic_router_hook` | ontology_hook | `V4_DYNAMIC_ROUTING_ENABLED` |
| `pathway_planner → specialist → crosstalk` | worker_mechanism | `V4_PATHWAY_PLANNER_ENABLED` / `V4_PATHWAY_SPECIALIST_ENABLED` / `V4_CROSSTALK_COORDINATOR_ENABLED` |
| `sbml_grounder → calibration → sensitivity` | worker_validator | `V4_SBML_GROUNDER_ENABLED` / `V4_CALIBRATION_AGENT_ENABLED` |
| `validation_pyramid` | worker_report | `V4_VALIDATION_PYRAMID_ENABLED` |
| `hypothesis_agent` | before worker_report | `V4_HYPOTHESIS_AGENT_ENABLED` |
| Reaction IR v2 / Pathway Graph / ODE Template v2 | inside worker_ode | `V4_REACTION_IR_ENABLED` / `V4_PATHWAY_GRAPH_ENABLED` / `V4_ODE_TEMPLATE_V2_ENABLED` |

**Known test failure:** Several specialist tests (`test_*_specialist.py`) and
`test_feature_flag_convergence.py` fail because individual sub-flags default
to `true` rather than `false`. This violates the rollback-safety contract
("all flags default OFF"). The master coarse flags still default OFF, so the
top-level rollback guarantee holds, but per-capability isolation does not.
This is tracked as a P0 regression.

---

## RAG knowledge base

[backend/app/rag_client.py](backend/app/rag_client.py) (`RagClient`) provides a
high-retrieval pipeline used by `worker_rag`:

- **Query rewriting** (`rewrite_query`) — LLM normalizes and expands query
  terms; safe-degrades to the original query on failure.
- **Hybrid retrieval** (`hybrid_search`) — semantic (ChromaDB embedding) +
  BM25 keyword scored over the full collection, de-duplicated, with
  `hybrid` tagging for double hits.
- **Reranking** (`rerank_results`) — multi-provider rerank
  (Xfyun MaaS → OpenRouter → SiliconFlow) combined with source-authority
  (PMC > PubMed > Internal DB > Preprint) and species-specificity weighting.
- **High-level entry** (`search_params_hybrid`) — rewrite → hybrid → rerank,
  returning top-k results plus RAG-insight metadata.
- **Drug-specific retriever** (`drug_specific_retriever`) — PubMed snippet
  LLM extraction + ChromaDB IC50/EC50 lookup + ClinicalTrials.gov v2
  verification (`_query_clinical_trials`). Offline/failure returns empty list.
- **Online fallback** — when local hit rate drops below
  `RAG_ONLINE_FALLBACK_THRESHOLD` (0.3), KEGG / Reactome / UniProt / ChEMBL
  are queried within `RAG_ONLINE_TOTAL_BUDGET` (600 s). PubMed E-utilities
  fallback also fires inside N5/N9/N10 when ChromaDB has no hits.

Four ChromaDB collections persist locally under
[backend/data/vector_db/](backend/data/vector_db/):
`biodynamics_mechanism`, `biodynamics_parameter`, `biodynamics_experiment`,
`biodynamics_evidence`.

**Verified working:** the retrieval pipeline executes and returns ranked
parameter candidates. **Not verified:** whether the *correct* parameters for
each pathway are being retrieved (the benchmark failures suggest parameter
quality is poor — see Deep root-cause analysis).

---

## MCP biomedical terminology integration

[backend/app/mcp_client.py](backend/app/mcp_client.py) (`MCPBioClient`)
registers four MCP tools and orchestrates a 4-step term-lookup pipeline in
`lookup_terms`:

| Tool | Method | Purpose |
|---|---|---|
| OpenBioMed Skills | `extract_entities` | Biomedical entity & relation extraction |
| NIH UMLS MCP | `get_synonyms` | Ontology synonyms & hierarchy |
| medical-terminologies-mcp | `standardize_term` | Clinical term standardization (ICD-10, SNOMED CT) |
| pubmed-search-mcp | `search_pubmed_key_words` | Enhanced PubMed retrieval |

When `MCP_ENABLED=false` or no endpoint URLs are configured, `node0_mcp_term_lookup`
short-circuits and returns empty term definitions while preserving the original
`user_input` as `mcp_rewritten_query` — the pipeline never errors. When real
MCP endpoints are unreachable, `MCPBioClient` falls back to LLM-based term
extraction and definition; PubMed search falls back to direct NCBI E-utilities
calls (`_search_pubmed_eutils`) with a 2 req/s rate limit.

**Verified working:** graceful degradation path. **Not verified:** whether
real MCP endpoints are reachable in any deployed environment (no live MCP
credentials in the repo).

---

## Pathway specialists

[backend/app/pathways/specialists/](backend/app/pathways/specialists/) ships 10
specialists, each auto-registered via `@register_specialist` and providing
core / feedback / crosstalk / perturbation / validation modules:

`egfr`, `mapk`, `pi3k_akt_mtor`, `jak_stat`, `tgf_beta`, `wnt`, `p53`,
`nf_kappa_b`, `apoptosis`, `cell_cycle`

A specialist is invoked by the `specialist_hook` (Phase 4) after
`worker_mechanism`; with `V4_SPECIALIST_KG_WRITEBACK_MODE=both` it both injects
canonical topology into the v3 knowledge graph (Mode A) and lets the sandbox
prefer the specialist-rendered ODE (Mode B).

**Known test failure:** the 10 specialist test files (`test_*_specialist.py`)
exhibit ~25 failures total, primarily because:
1. Reaction catalog drift: EGFR has an extra `gtp_gdp_exchange` mechanism;
   MAPK uses `nuclear_import` where `phosphorylation` is expected; p53 has
   12 reactions where the test expects 10.
2. V4 sub-flags defaulting to `true` (Bug 2 above).

These failures mean the specialists *exist and register* but their reaction
catalogs are out of sync with their test expectations.

---

## Reaction IR & ODE templates

- **Reaction IR v2** ([backend/app/reaction_ir_v2/](backend/app/reaction_ir_v2/))
  — a Pydantic-validated intermediate representation: 17+ mechanism types,
  5 compartments, state machines, composite reactions, and a constraint
  schema. It bridges the knowledge graph to ODE templates without an LLM in
  the loop.
- **ODE templates** ([backend/app/ode_templates/](backend/app/ode_templates/)
  v3, [backend/app/ode_templates_v2/](backend/app/ode_templates_v2/) v4) —
  Jinja2 templates for cascade activation/inhibition, phosphorylation cascades,
  PK/PD one- and two-compartment, bistable switches, oscillatory feedback,
  caspase cascade, cyclin–CDK toggle, destruction complex, nuclear transport,
  transcriptional delay, ubiquitination cascade.
- **Template selection** ([backend/app/template_selector.py](backend/app/template_selector.py))
  — rule engine: keyword match → mechanism-type voting → SBML grounding →
  LLM fallback. The LLM never picks the template directly.

**Verified working:** templates render without Jinja2 errors. **Not verified:**
whether the rendered ODE systems are numerically stable for all 10 pathways
(benchmark evidence shows 3/10 explode: WNT, CELL_CYCLE, TGF_BETA).

---

## 5-level Validation Pyramid

[backend/app/validation_v2/](backend/app/validation_v2/) implements a layered
validator surfaced in the frontend `ValidationPyramid` component:

| Level | Checks |
|---|---|
| L1 Internal | Mass conservation, non-negative concentrations, steady-state reachability, numerical stability, constraint satisfaction ([level1_internal.py](backend/app/validation_v2/level1_internal.py)) |
| L2 SBML/BioModels | Comparison via RoadRunner Track A or structural-similarity Track B; peak/value/amplification diffs ([level2_sbml.py](backend/app/validation_v2/level2_sbml.py)) |
| L3 Cross-pathway | Cross-talk consistency, shared-species conservation, time-scale alignment ([level3_crosstalk.py](backend/app/validation_v2/level3_crosstalk.py)) |
| L4 Benchmark | Per-pathway quantitative pass criteria from `benchmarks/*.yaml` ([level4_benchmark.py](backend/app/validation_v2/level4_benchmark.py)) |
| L5 Hypothesis | Hypotheses validated / falsified, evidence support, confidence ([level5_hypothesis.py](backend/app/validation_v2/level5_hypothesis.py)) |

**Critical caveat (Bug 1 + Bug 2):** The pyramid is a **soft gate** — it runs
*after* `worker_report` and never routes back for retry. It is also
**disabled by default** (`V4_VALIDATION_PYRAMID_ENABLED=false`). The
`12_check_report.json` files in `data/sa_logs/all_10_pathways/` are produced
by the SA module's own checks, not by L4/L5. To actually exercise the
pyramid, set `V4_VALIDATION_PYRAMID_ENABLED=true` in `.env`.

---

## Scientific Alignment (SA) — the rigor layer

[backend/app/scientific_alignment/](backend/app/scientific_alignment/) is a
**100% rule-based** module (no LLM calls) that audits the pipeline output
after `worker_report` completes. It is triggered in
[main.py](backend/app/main.py) `_run_scientific_alignment_postprocess` and
gated by one master flag `V4_SCIENTIFIC_ALIGNMENT_ENABLED` plus per-capability
sub-flags. With the master flag OFF, no SA code runs and the system reverts to
plain v3/v4 behavior.

The design principle: **the LLM organizes and explains evidence; rule engines
adjudicate scientific correctness.**

### Canonical reference library (Sprint 1)

[canonical_loader.py](backend/app/scientific_alignment/canonical_loader.py)
loads 10 pathway reference files from
[backend/knowledge/canonical/](backend/knowledge/canonical/). Each defines 8
fields: `canonical_reviews`, `canonical_models` (BioModels IDs),
`canonical_mechanism` (required nodes/edges), `expected_behavior`,
`known_negative_feedback`, `consistency_rules` (assertion expressions), and a
`canonical_timeline` of peak times. Loading uses regex whitelist + `Path.resolve`
+ `relative_to` triple path-traversal protection.

### Gold standard benchmark suite (Sprint 1)

[gold_standard_schema.py](backend/app/scientific_alignment/gold_standard_schema.py)
+ [backend/benchmarks/golden/](backend/benchmarks/golden/) — 10 pathway
directories (Apoptosis, CellCycle, EGFR, JAKSTAT, MAPK, NFKB, PI3K, TGFB, WNT,
p53), each with `benchmark.yaml`, `expected.md`, and `expected_metrics.json`.
A literature gold standard lives in
[backend/knowledge/gold_standard/literature_*.yaml](backend/knowledge/gold_standard/)
(classical reviews / mechanism papers / BioModels sources per pathway).

### Consistency Checker (Sprint 3)

[consistency_checker.py](backend/app/scientific_alignment/consistency_checker.py)
evaluates `consistency_rules` from the canonical reference against simulation
metrics — e.g. EGFR peak time must precede ERK peak time. Assertion
expressions from external YAML are evaluated under triple protection: character
whitelist regex, AST node whitelist, and restricted-namespace `eval`.

### Validation Rule Engine & Scientific Review (Sprint 3)

[validation_rule_engine.py](backend/app/scientific_alignment/validation_rule_engine.py)
applies 4 hard rules: mass conservation, peak-time vs canonical timeline,
peak ordering vs consistency rules, evidence count ≥ 3. When
`SA_SPRINT3_CONSISTENCY_GATE` is ON, a consistency failure is a **hard gate**
that blocks the downstream critic / multi-dim / validation / review stages.
[scientific_review.py](backend/app/scientific_alignment/scientific_review.py)
scores the run 0–10 across 7 items.

### Scientific Critic (6 categories)

[scientific_critic.py](backend/app/scientific_alignment/scientific_critic.py)
independently audits the report across 6 categories — mechanism coverage,
evidence sufficiency (≥5 total, ≥2 reviews), BioModels oracle status,
consistency, experiment-chain validity, and classical-reference coverage.
It does **not** read the generated report; it re-derives findings from raw
state.

### Multi-dimensional confidence (6 axes)

[multi_dim_confidence.py](backend/app/scientific_alignment/multi_dim_confidence.py)
replaces a single confidence number with 6 dimensions — Mechanism, Simulation,
Evidence, BioModels, Discussion, Experiment. The overall score is
`min(6 dims) × 0.9` (low-dimension drag strategy).

### Experiment Planner (Sprint 4, rule-based)

[experiment_planner.py](backend/app/scientific_alignment/experiment_planner.py)
generates a mechanism-driven experiment chain per pathway. It loads 10 pathway
rule files from [backend/knowledge/experiments/](backend/knowledge/experiments/)
via `_load_experiments_from_yaml`, enforces forbidden patterns (e.g. qPCR must
not be the primary validation of phosphorylation activation), and marks a chain
complete only when coverage ≥ 0.7 with zero unjustified and zero forbidden
experiments. No LLM is involved.

**Critical caveat (Bug 4):** The forbidden-pattern check is gated behind
`SA_SEVEN_AXIS=ON` (default OFF). When the flag is off (the default), the
planner early-returns `ExperimentPlan(enabled=False, skipped=True)` and never
loads the YAML rule files. This is why the SA report found
"0% mechanism matching in Experiment Planner" — the rule engine is not
running in the default configuration.

### Parameter Provenance (Sprint 5)

[parameter_provenance.py](backend/app/scientific_alignment/parameter_provenance.py)
builds an 8-column traceability table for every kinetic parameter:
Edge · Parameter · Value · Unit · Source · Origin · Confidence · Status.
Sources are ranked RAG > SBML > PubMed > KEGG > UniProt > ChEMBL > Inferred.

**Known issue:** The SA report found 114 `budget_exceeded` parameters being
counted as calibrated — these should be flagged as fallback/inferred, not
calibrated.

### Explainability Decision Log (Sprint 5)

[explainability_log.py](backend/app/scientific_alignment/explainability_log.py)
records an 8-dimension decision log — Mechanism, Confidence, BioModels,
Parameter, Discussion, Experiment, Validation, Cross-talk.

### Evidence Fusion & citation-driven Discussion

[evidence_fuser.py](backend/app/scientific_alignment/evidence_fuser.py) fuses
5 evidence sources with single-source tags: **[A] PubMed · [B] BioModels ·
[C] Simulation · [D] Inference · [E] Hypothesis**.
[discussion_renderer.py](backend/app/scientific_alignment/discussion_renderer.py)
renders the Discussion so every sentence carries a single-source tag; a
violation raises `DiscussionRenderError`.

**Critical caveat (Bug 3):** The fuser currently uses **positional matching**
(`assertion[i] ← evidence_list[i]`) rather than semantic matching. This means
the citation tags are correct only when the assertion and evidence lists
happen to be in the same order — which they typically are not. The SA report
found "100% Fake Grounding" in the Evidence Flow audit for this reason.

### BioModels calibration & root-cause analysis

[biomodels_calibration.py](backend/app/scientific_alignment/biomodels_calibration.py)
simulates the canonical BioModels SBML with RoadRunner and compares per-species
peak time, peak value, RMSE, and correlation against the agent ODE simulation;
species are fuzzy-matched (exact / phospho-aware / token / keyword).
[rca.py](backend/app/scientific_alignment/rca.py) aggregates defects from all
validators into a structured root-cause report.

**Critical caveat (Bug 5 + Bug 8):** Track B hardcodes
`max_relative_error=1.0`, and the BioModels SBML IDs are misassigned
(`BIOMD0000000205` = Ung2008 EGFR endocytosis is used for 9 non-EGFR
pathways). The calibration output is therefore unreliable for 9/10 pathways.

### Loop controller & regression monitor

[loop_controller.py](backend/app/scientific_alignment/loop_controller.py)
orchestrates the full SA loop (7-axis → 6-dim → critic → RCA → acceptance →
regression) for up to 3 iterations and emits `Scientific_Alignment_Report.md`.
[regression_monitor.py](backend/app/scientific_alignment/regression_monitor.py)
diffs two 7-axis reports, classifies changes as REGRESSION / FIX / NO_CHANGE,
and suggests feature-flag rollbacks per affected axis.
[acceptance_gate.py](backend/app/scientific_alignment/acceptance_gate.py)
applies 12 acceptance criteria.

**Critical caveat (Bug 1):** The loop controller can iterate, but because the
validation pyramid is a soft gate that runs *after* `worker_report`, the loop
cannot actually re-route back to `worker_ode` for a retry. The "3 iterations"
loop is therefore advisory, not corrective.

---

## Feature flags (rollback-safe by design, not by test)

All capabilities are flag-gated and intended to default OFF. Turning every
flag OFF reverts the system to plain v3 behavior with no v4/SA code path
executing. Flags are resolved in [config.py](backend/app/config.py)
`_resolve_v4_flag` (env override > coarse flag > fine-flag default) and
`is_sa_feature_enabled` (master gate).

| Group | Master / coarse flags | Effect when ON |
|---|---|---|
| v4 scientific layer | `V4_SCIENTIFIC_LAYER_ENABLED` | P1–P4: Ontology, Reaction IR v2, Pathway Graph, ODE Template v2, Pathway Planner, Specialists, Cross-talk |
| v4 validation | `V4_VALIDATION_ENABLED` | P5: SBML Grounder, Validation Pyramid, Calibration |
| v4 hypothesis | `V4_HYPOTHESIS_ENABLED` | P6: Hypothesis Agent, Dynamic Router |
| Scientific Alignment | `V4_SCIENTIFIC_ALIGNMENT_ENABLED` | Enables SA sub-flags (see below) |

SA sub-flags (all default OFF, enforced by the master gate):
`SA_MECHANISM_GRAPH`, `SA_PARAMETER_PRIOR`, `SA_BIOMODELS_ORACLE`,
`SA_EVIDENCE_FUSION`, `SA_SEVEN_AXIS`, `SA_LOOP_TERMINATION`, `SA_CANONICAL`,
`SA_CONSISTENCY_CHECKER`, `SA_PARAMETER_CONFIDENCE`, `SA_SCIENTIFIC_CRITIC`,
`SA_MULTI_DIM_CONFIDENCE`, `SA_BIOMODELS_CALIBRATION`, `SA_SPRINT1_GROUND_TRUTH`,
`SA_SPRINT2_EVIDENCE_RENDERER`, `SA_SPRINT3_CONSISTENCY_GATE`,
`SA_SPRINT4_EXPERIMENT_RULE_ENGINE`, `SA_SPRINT5_PROVENANCE_EXPLAINABILITY`.

**Known test failure (P0):** The rollback-safety contract is *not* fully
upheld by the test suite. `test_feature_flag_convergence.py` (10 failures)
and `test_p4_flag_off_isolation.py` (4 failures) demonstrate that several
individual v4 sub-flags default to `true` rather than `false`. The master
coarse flags do default OFF, so top-level rollback works, but per-capability
isolation is broken. This is tracked as a P0 regression.

---

## Frontend

[frontend/](frontend/) is a Next.js 16.2.10 / React 19.2.4 app with a single
Zustand store ([lib/store.ts](frontend/lib/store.ts)) that ingests every SSE
event through `ingestSSEEvent`. Three entry points:

- **`/` — Minimal Auto-Chat** ([components/minimal/](frontend/components/minimal/)):
  Header · Natural Language Input (4 example chips: EGFR/MAPK, PI3K/AKT, p53,
  NF-κB) · 7-step AI Workflow tracker · Results Tabs (Graph · Curves ·
  Validation · Report).
- **`/advanced` — Archived 4-pane Scientific IDE**
  ([components/workspace/WorkbenchShell.tsx](frontend/components/workspace/WorkbenchShell.tsx)).
- **`/benchmarks` — Benchmark Center**
  ([components/benchmark/BenchmarkCenter.tsx](frontend/components/benchmark/BenchmarkCenter.tsx)):
  streams `POST /api/v4/benchmarks/run`, renders 10 `BenchmarkCard`s.

### ScientificAlignmentPanel

[components/scientific_alignment/ScientificAlignmentPanel.tsx](frontend/components/scientific_alignment/ScientificAlignmentPanel.tsx)
renders up to 9 collapsible cards, each color-coded (pass=emerald /
warning=amber / fail=red / skipped=zinc):

1. Consistency Gate Failed (Sprint 3 block notice)
2. SA-1 Consistency Checker
3. SA-2 Scientific Critic
4. SA-3 Multi-dim Confidence
5. SA-4 Validation Rule Engine
6. SA-5 Scientific Review
7. SA-6 Parameter Provenance
8. SA-7 Decision Log
9. SA-F BioModels Calibration

The store listens to 16 SA SSE event types.

---

## API reference

### SSE / chat (POST /api/chat)

| Event | Content |
|---|---|
| `agent_registry` | Registered agents for the active workflow version |
| `agent_dispatch` | Supervisor dispatch record (target_agent, reasoning, status) |
| `node_start` / `execution_log` / `code_generated` / `image_ready` | Per-node progress |
| `rag_insights` / `rag_online_fallback` / `rag_ready` | RAG retrieval state |
| `mcp_tool_call` / `mcp_term_definitions` | MCP term lookup |
| `knowledge_graph` / `metrics` / `experiment_protocols` / `paper_evidence` | v2 worker outputs |
| `v4_pathway_graph` / `v4_simulation_result` / `v4_validation_report` / `v4_hypothesis_list` | v4 adapter events |
| `report` / `report_ready` | Final markdown report |
| `sa_*` | 16 Scientific Alignment events |
| `clarification_needed` / `clarification_resolved` | Human-in-the-loop |
| `token_usage` / `error` / `end` | Telemetry & termination |

### REST endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/chat` | Main SSE workflow |
| `POST` | `/api/chat/respond` | Resume after clarification |
| `POST` | `/api/chat/stop` | Cancel an in-flight run |
| `POST` | `/api/chat/clear-memory` | Clear LangGraph MemorySaver |
| `POST` | `/api/v4/benchmarks/run` | Run the 10-pathway benchmark suite (SSE) |
| `GET`  | `/api/v4/pathways` | List the 10 pathway classes + metadata |
| `GET`  | `/api/v4/pathways/{class}/graph` | Pathway graph (nodes/edges/modules) |
| `POST` | `/api/v4/simulation/run` | Deterministic mass-action ODE simulation |
| `POST` | `/api/v4/benchmark/{class}` | Run a single pathway benchmark |
| `GET`  | `/api/v4/reports/{id}` | Fetch a stored simulation report |
| `POST` | `/api/v4/simulation/sweep` | Parameter sweep |
| `GET`  | `/api/models/status` | LLM / embedding / rerank provider health |
| `POST` | `/api/admin/update-vector-db` | Refresh the ChromaDB vector store |
| `GET`  | `/api/admin/rag-status` | RAG collection counts + retrieval health |

---

## Multi-provider LLM / embedding / rerank

[config.py](backend/app/config.py) wires primary + backup LLMs through
`FallbackLLM` (0.5 s switchover, `max_retries=0`), embedding providers
(OpenAI / local / OpenRouter / SiliconFlow / Xfyun MaaS), and a 3-provider
rerank cascade managed by `RerankManager` (priority: Xfyun MaaS → OpenRouter →
SiliconFlow). Concrete provider/model values are read from
[backend/.env](backend/.env).

---

## Testing

[backend/tests/](backend/tests/) contains 57 pytest files covering every
specialist (10 `*_specialist.py`), validation levels 1–5, Reaction IR v2,
SBML grounder, calibration, sensitivity, hypothesis, pathway planner, dynamic
router, feature-flag convergence, and P0/P1 critical fixes.

**Latest verification run (2026-07-19):** 28 test files executed, **1046
passed / 57 failed / 2 skipped.** Failure patterns:

| Pattern | Failure count | Root cause |
|---|---|---|
| V4 sub-flags default to `true` instead of `false` | ~25 | Violates rollback-safety contract; affects all 11 specialist tests + `test_feature_flag_convergence.py` (10) + `test_p4_flag_off_isolation.py` (4) |
| SBML parser V2 returns empty on valid XML | 12 | `parse_basic_xml` returns falsy; downstream `compute_integrity_complete_chain` fails. `test_sbml_grounder.py` |
| Specialist reaction catalog drift | ~10 | EGFR extra `gtp_gdp_exchange`; MAPK `nuclear_import` vs `phosphorylation`; p53 12 reactions vs expected 10 |
| Validator "pass=True when skipped" semantics | ~6 | level4/level5 tests disagree on whether a skipped check should aggregate as pass |
| Hook gating not respected | ~4 | Hooks write `v4_*` fields even when flag=False (same root cause as Pattern 1) |

[verification/](verification/) is a scientific verification suite spanning 8
sub-suites: `pathway_regression`, `parameter_stress`, `ontology_validation`,
`solver_validation`, `benchmark`, `hypothesis_validation`,
`biomodels_regression`, `ui_workflow` (Playwright).

[frontend/](frontend/) ships 6 vitest specs plus 3 Playwright e2e specs
(`smoke`, `auto-chat`, `ux_workflow`).

Makefile shortcuts: `make test` (backend unit), `make test-integration`,
`make test-benchmark` (scientific benchmark), `make ci` (full local CI).

CI runs on GitHub Actions
([.github/workflows/ci.yml](.github/workflows/ci.yml),
[.github/workflows/scientific-regression.yml](.github/workflows/scientific-regression.yml))
including a `scientific-alignment-benchmark` gate job.

---

## Benchmark status (10-pathway suite)

This is the canonical metric for evaluating agent completeness, per the
user's directive: *"Benchmark 多个项目都要正确完成可复现的才行"*.

### Honest pass rate: 1/10

| # | Pathway | `overall_passed` | Real status | Failure mode |
|---|---|---|---|---|
| 1 | APOPTOSIS | `true` | **REAL PASS** (degraded C9) | C9 still `degraded=true, passed=true` |
| 2 | MAPK_ERK | `true` | **FAKE PASS** | `csv_max_value=1.0`, `dynamic_species_count=0` (flat-line) |
| 3 | EGFR_RTK | `false` | FAIL | pEGFR `peak_time=120.0` (expected `[5.0, 10.0]`); 5/12 checks failed |
| 4 | PI3K_AKT_mTOR | `false` | FAIL | C9 degraded pass, C11 `coverage_ratio=0.0` |
| 5 | p53_signaling | `false` | FAIL | C9 degraded pass, C11 `coverage_ratio=0.0` |
| 6 | NF_kB | `false` | FAIL | C9 degraded pass, C11 `coverage_ratio=0.0` |
| 7 | JAK_STAT | `false` | FAIL | C9 degraded pass, C11 `coverage_ratio=0.0` |
| 8 | TGF_beta | `false` | FAIL | `max_fold_change=830` (explosion) |
| 9 | Wnt | `false` | FAIL | `max_fold_change=544147.27` (explosion) + `BIOMD0000000055` misuse |
| 10 | Cell_Cycle | `false` | FAIL | `max_fold_change=16220712` (16M× explosion) |

### False report warning

The file
`backend/data/sa_logs/all_10_pathways/BENCHMARK_REPORT.md` claims **"100%
pass rate"**. This is **false**. The file's own header reads "总通路数: 1"
(total pathways: 1) while the title says "10 Pathway Benchmark Report" —
only `MAPK_ERK` is listed, and that single entry is itself a fake pass
(flat-line simulation). **Do not trust `BENCHMARK_REPORT.md` for status
reporting** until Bug 6 and Bug 7 are fixed.

### Reference to SCIENTIFIC_CALIBRATION_REPORT.md

The companion audit at
[../SCIENTIFIC_CALIBRATION_REPORT.md](../SCIENTIFIC_CALIBRATION_REPORT.md)
scores the system **1.2/10** overall. Its key findings, all of which were
independently verified by direct code inspection during this README update:

- 10% real benchmark pass rate (1/10), with `BENCHMARK_REPORT.md` falsely
  claiming 100%.
- 100% Fake Grounding in Evidence Flow (Bug 3 — positional matching).
- 24 missing canonical mechanisms across the 10 pathway specialists.
- `BIOMD0000000205` misassigned 362 times (Bug 8).
- 0% mechanism matching in Experiment Planner (Bug 4 — `SA_SEVEN_AXIS=OFF`).
- All Validation Pyramid failures are Soft Fail, not Hard Stop (Bug 1).
- 114 `budget_exceeded` parameters counted as calibrated.

The audit established a Scientific Priority Queue:
**9 P0 / 9 P1 / 4 P2 / 1 P3** items. Future development is expected to
follow that queue strictly.

---

## Deep root-cause analysis per failing pathway

Per the user's directive: *"如果有和基准答案出入要深度分析测试log到底哪里有问题"*
— if discrepancies with benchmark answers exist, deep-analyze the test logs
to find the root cause. The following analysis is grounded in the actual
`12_check_report.json` files at
`backend/data/sa_logs/all_10_pathways/{PATHWAY}/12_check_report.json`.

### APOPTOSIS — real pass, but degraded C9

- `overall_passed: true` — the only honest pass.
- C9 (`biomodels_grounded`): `passed: true, degraded: true` — even the
  passing pathway has a degraded SBML grounding (Bug 7).
- C11 (`canonical_coverage`): `coverage_ratio: 0.0, passed: true` — passes
  via the `or pmid_count >= 1` shortcut (Bug 6).
- **Why it passes when others don't:** the caspase cascade template renders
  a numerically stable ODE system that does not explode. The peak-time
  expectations in `benchmarks/golden/Apoptosis/benchmark.yaml` are loose
  enough that the simulation's flat-ish response still falls inside the
  allowed windows.

### MAPK_ERK — fake pass (flat-line)

- `overall_passed: true` in the report — **this is false.**
- `csv_max_value=1.0`, `dynamic_species_count=0` — the simulation produced a
  flat line at value 1.0. No species changed over time.
- **Root cause:** The ODE system emitted by `worker_ode` for MAPK is likely
  a degenerate steady-state with all derivatives zero. The peak-time check
  passes trivially because there is no peak to compare; the mass-conservation
  check passes because nothing changes. This is a false positive of the L1
  validator, not a real pass.
- **Fix direction:** Investigate why the cascade phosphorylation template
  produces a zero-Jacobian system for MAPK inputs. Likely a parameter
  injection failure (RAG returned no usable Km/Vmax, fallback used 0).

### EGFR_RTK — peak time 12× too late

- `overall_passed: false`, 5/12 checks failed (Peak Time, Peak Order,
  Adaptation, Oscillation, Mass Conservation).
- `pEGFR.peak_time = 120.0` (expected window `[5.0, 10.0]` min).
- **Root cause:** The simulation reaches peak pEGFR at t=120 min instead of
  t=5–10 min. This is a 12–24× delay, characteristic of either (a) a missing
  EGF stimulus injection at t=0, (b) wrong rate constants (Kd far too small),
  or (c) the simulation starting from a pre-stimulus steady state rather
  than a stimulated initial condition.
- **Fix direction:** Inspect `worker_ode`'s rendered ODE for EGFR — confirm
  the EGF input term is non-zero at t=0 and that the EGFR-EGF binding
  `k_on` is in the 10⁵–10⁶ M⁻¹s⁻¹ range, not 10³.

### PI3K_AKT_mTOR / p53 / NF_kB / JAK_STAT — uniform C9+C11 failure

- All four: `overall_passed: false`, C9 `degraded=true, passed=true`,
  C11 `coverage_ratio=0.0, passed=true`.
- **Root cause:** Same pattern — the SBML grounding failed (so C9 degraded),
  and the canonical coverage check passed via the `or` shortcut (Bug 6)
  despite `coverage_ratio=0.0`. The actual simulation outputs were not
  inspected per-check because the aggregator's logic auto-passes these two
  checks.
- **Fix direction:** Fix Bug 6 (change `or` to `and`) first — this alone
  will re-surface the real C11 failures that are currently hidden. Then
  re-run the benchmark to see the actual simulation quality.

### TGF_beta — 830× explosion

- `max_fold_change = 830` — a species concentration grew 830× beyond its
  initial value. Real biology rarely exceeds 10–50× fold change.
- **Root cause:** Missing negative feedback in the SMAD degradation loop, or
  an absent mass-conservation constraint in the ODE template. The TGF-β
  template likely lacks the SMAD7 negative-feedback loop that bounds the
  response in vivo.
- **Fix direction:** Add a SMAD7-mediated negative feedback term to
  `ode_templates_v2/tgf_beta_smad.j2` and re-run.

### Wnt — 544147× explosion + wrong BioModels ID

- `max_fold_change = 544147.27` — β-catenin accumulated to 544,147× its
  initial value.
- `sbml_model_id = BIOMD0000000055` — this is an **Arabidopsis circadian
  clock model**, not a WNT model. The SBML comparison is therefore
  meaningless.
- **Root cause:** (a) The destruction complex (Axin/APC/GSK3β) is not
  correctly inhibiting β-catenin in the ODE; (b) the SBML grounder picked
  the wrong BioModels ID for WNT (likely a fuzzy-match failure in
  `sbml_grounder.py`).
- **Fix direction:** Fix the destruction complex rate equations; fix the
  SBML grounder's pathway-to-BioModels-ID mapping table.

### Cell_Cycle — 16M× explosion

- `max_fold_change = 16220712` — Cyclin/CDK accumulated to 16 million times
  initial value.
- **Root cause:** Missing APC/C-mediated degradation of cyclin B. The
  bistable toggle template is likely rendering a one-way switch with no
  reset pathway.
- **Fix direction:** Add APC/C-Cdc20 degradation term to the cyclin-CDK
  toggle template; verify the toggle's reset threshold.

---

## Project structure

```
bio-dynamics-agent/
├── backend/
│   ├── app/
│   │   ├── main.py                 FastAPI app, /api/chat SSE, SA post-loop
│   │   ├── graph_v3.py             LangGraph v3 workflow + v4 hook chains
│   │   ├── state.py                BioDynamicsState + v4_state container
│   │   ├── config.py               Settings + 3 coarse / 13 fine v4 flags + 18 SA flags
│   │   ├── supervisor.py           AGENT_REGISTRY (v1: 6) + AGENT_REGISTRY_V2 (v2: 10)
│   │   ├── nodes.py / nodes_v2.py  N0–N11 pipeline nodes
│   │   ├── v4_endpoints.py         6 v4 REST endpoints
│   │   ├── rag_client.py           hybrid RAG + rerank + ClinicalTrials.gov
│   │   ├── mcp_client.py           4 MCP tools + LLM/E-utilities fallback
│   │   ├── ontology/               P1 Ontology Agent + SBO terms + pathway registry
│   │   ├── reaction_ir_v2/         P2 Reaction IR v2 (schema, builder, state machine)
│   │   ├── pathway_graph/          P3 Pathway Graph builder + initializer
│   │   ├── ode_templates(_v2)/     Jinja2 ODE templates (v3: 9, v4: 9)
│   │   ├── pathways/specialists/   10 pathway specialists
│   │   ├── crosstalk/              P4 Cross-talk Coordinator (shared-species sync)
│   │   ├── sbml_grounder/          P5 SBML Grounder (5-level mapping)
│   │   ├── validation_v2/          P5 Validation Pyramid (level1–level5)
│   │   ├── calibration/            P5 Calibration Agent (least squares + CI)
│   │   ├── sensitivity/            P5 local + Sobol + Morris sensitivity
│   │   ├── hypothesis/             P6 Hypothesis Agent + falsifiability checker
│   │   ├── scientific_alignment/   SA module (28 files, 100% rule-based)
│   │   ├── solvers/                DDE solver, stability retry, oscillation/bistability
│   │   └── reliability/            circuit breaker, retry, fail-safe, structured logging
│   ├── benchmarks/golden/          10 pathway gold-standard cases (yaml + expected + metrics)
│   ├── benchmarks/scientific_alignment/  10 pathway + 10 negative SA criteria
│   ├── knowledge/canonical/        10 pathway canonical references
│   ├── knowledge/experiments/      10 pathway experiment-chain rules (loaded only when SA_SEVEN_AXIS=ON)
│   ├── knowledge/gold_standard/    10 pathway literature gold standards
│   ├── tests/                      57 pytest files (1046 pass / 57 fail as of 2026-07-19)
│   ├── data/sa_logs/all_10_pathways/  per-pathway 12_check_report.json (ground truth for benchmark)
│   └── data/raw/                   BioModels SBML (incl. BIOMD0000000205 = Ung2008 EGFR)
├── frontend/
│   ├── app/                        page.tsx · advanced · benchmarks · report/[id]
│   ├── components/minimal/         MinimalApp, NaturalLanguageInput, AIWorkflowSteps, ResultsTabs
│   ├── components/workspace/       WorkbenchShell (4-pane IDE)
│   ├── components/scientific_alignment/  ScientificAlignmentPanel (9 cards)
│   ├── components/validation/      ValidationPyramid (5 levels)
│   ├── components/hypothesis/      HypothesisPanel + HypothesisCard (5 sections)
│   ├── lib/                        store.ts (Zustand), api.ts, sse.ts
│   └── __tests__/ + e2e/           vitest + Playwright
├── verification/                   8 scientific verification sub-suites
├── .github/workflows/              ci.yml + scientific-regression.yml
├── Dockerfile                      multi-stage (Python 3.11-slim backend + Node 20-slim frontend)
└── Makefile                        test / test-integration / test-benchmark / ci
```

---

## Design principles (aspirational vs verified)

| Principle | Status |
|---|---|
| **LLM organizes, rules adjudicate.** Kinetic constants, consistency, experiment chains, confidence, and acceptance gates are computed by rule engines. | **Partially verified.** Rule engines exist and run, but several are gated off by default (Bug 4) or bypassed by `or` logic (Bug 6). |
| **Every capability is flag-gated and rollback-safe.** All flags default OFF; OFF means the v3 pipeline runs untouched. | **Partially verified.** Master coarse flags default OFF and top-level rollback works. Individual sub-flags violate the contract (~25 test failures, P0). |
| **Citation-driven discussion.** Every Discussion sentence carries a single evidence-source tag `[A]`–`[E]`; untagged sentences raise an error. | **Partially verified.** The contract is enforced (raises `DiscussionRenderError`), but the underlying matching is positional (Bug 3) — tags are present but often wrong. |
| **Path-traversal-safe knowledge loading.** External YAML loaded with `yaml.safe_load`, name whitelist, `relative_to` verification; assertion expressions under AST + restricted-namespace guards. | **Verified.** Triple protection present in `canonical_loader.py` and `consistency_checker.py`. |
| **Graceful degradation.** Missing optional deps (roadrunner, lmfit, SALib, lxml, psutil) fall back to simpler implementations and log a warning rather than crashing. | **Verified, but with caveat.** Degradation works, but the Track B fallback hardcodes `max_relative_error=1.0` (Bug 5), masking real failures. |

### Security posture (per security-best-practices review)

- Path-traversal protection on all external YAML loading: **verified**
  (regex whitelist + `Path.resolve` + `relative_to`).
- `yaml.safe_load` used (not `yaml.load`): **verified**.
- Assertion expressions evaluated under AST whitelist + restricted namespace:
  **verified**.
- API keys and provider credentials are read from `.env`, not hardcoded:
  **verified**.
- Sandbox execution uses AST precheck before exec: **verified** in
  `worker_sandbox` (n7).
- No `eval` of user-controlled input outside the restricted-namespace
  assertion evaluator: **verified**.

No critical security vulnerabilities were identified during this review. The
main risk surface is the sandboxed ODE execution (`worker_sandbox`), which
relies on AST precheck + `exec` in a restricted namespace — this is
acceptable for a research prototype but should be hardened (e.g. container
isolation) before multi-tenant deployment.

---

## License

MIT. See [LICENSE](./LICENSE).
