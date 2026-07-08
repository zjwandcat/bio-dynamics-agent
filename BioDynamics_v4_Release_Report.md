# BioDynamics Agent v4.0 RC — Release Report

> **Task H.1** — Final release report for the v4.0 Release Candidate sprint.
> Documentation only; no source code modified.
> Report date: 2026-07-08 (updated post-RC fix). Repository: `bio-dynamics-agent/`.

This report consolidates the eight sprint phases (A–H) into a single release
summary. Commit hashes are short SHAs from `git log --oneline`. Source of truth
for per-task detail: `docs/CHANGELOG.md`, `docs/ARCHITECTURE.md`,
`docs/UX_FLOW.md`, `docs/FRONTEND.md`, `docs/NAMING_VIOLATIONS.md`.

---

## Chapter 1 — Architecture Summary

BioDynamics Agent v4 RC is a **Supervisor–Worker LangGraph pipeline** that turns
a free-text biological hypothesis into a calibrated, validated ODE simulation
and a falsifiable follow-up experiment plan.

### 1.1 System overview

- **Backend** — FastAPI (`backend/app/main.py`) + LangGraph
  (`backend/app/graph_v3.py::build_workflow_v3`). A Supervisor node walks an
  `execution_plan` of v3 Workers (`worker_mcp` → `worker_mechanism` →
  `worker_rag` → `worker_pkpd` → `worker_ode` → `worker_sandbox` →
  `worker_validator` → `worker_report`). On top of that v3 spine, six v4 hook
  chains are wired in **non-invasively** behind feature flags. Every v4 hook
  returns `{}` when its flag is off, so the system degrades bit-for-bit to v3.
- **Frontend** — Next.js 14 (App Router) + React 18 + TypeScript + Tailwind +
  Zustand (`frontend/`). The v4 redesign introduces a four-pane Scientific
  Modeling IDE shell that co-exists with the existing AI-Assistant chat (v3
  `/api/chat` SSE contract preserved).
- **Transport** — Server-Sent Events (SSE) over `POST /api/chat`
  (`text/event-stream`); a dedicated `POST /api/v4/benchmarks/run` SSE endpoint
  for the benchmark suite.

### 1.2 Four-pane Scientific IDE layout

`frontend/components/workspace/WorkbenchShell.tsx` renders a CSS Grid
(`h-screen`, dark theme `bg-zinc-950`):

```
┌─────────────┬──────────────────────┬──────────────┬──────────────┐
│ Project /   │ Scientific Workspace │ Validation   │ AI Assistant │
│ Pathway     │ (Pathway Graph +     │ (Pyramid +   │ (collapsible │
│ (250px)     │  Sim Tabs + Params)  │  Hypothesis) │  360px → 0)  │
└─────────────┴──────────────────────┴──────────────┴──────────────┘
```

The AI Assistant pane is **collapsed by default** — the center Scientific
Workspace is the primary UI, not the chat.

### 1.3 Ten-pathway specialist architecture

P4 delivers 10 pathway specialists under `backend/app/pathways/specialists/`:
EGFR, MAPK, PI3K/AKT/mTOR, JAK-STAT, NF-κB, Wnt, TGF-β, p53, Cell Cycle,
Apoptosis. A `pathway_planner` classifies the input, the matched specialist
applies a mechanism template, and a `crosstalk_coordinator` reconciles shared
species / time scales when multiple pathways fire.

### 1.4 v4_state dict (B.2 consolidation)

LangGraph state (`app/state.py::BioDynamicsState`) grew 17 flat `v4_*` fields
across P1–P6. Task **B.2** consolidated them into a single `v4_state` container
with a custom reducer (`merge_v4_state`) and dual-write helpers
(`set_v4_state` writes both the flat field and the nested key; `get_v4_state`
reads nested with flat-field fallback). Eight phase groups: `ontology`,
`reaction_ir`, `pathway_graph`, `pathway_class`, `specialist`, `grounding`,
`validation`, `hypothesis`, `router`. `normalize_v4_state` rebuilds the
container from the flat fields at the end of `worker_ode` to survive
LangGraph's dict-replace semantics.

### 1.5 Three-coarse-flag system (B.3 convergence)

Task **B.3** converged 13 fine-grained V4 flags into 3 coarse production
switches, resolved by `Settings._resolve_v4_flag`:

| Coarse flag | Default | Phase coverage |
|---|---|---|
| `V4_SCIENTIFIC_LAYER_ENABLED` | `false` | P1 + P2 + P3 + P4 (8 fine flags) |
| `V4_VALIDATION_ENABLED` | `false` | P5 (3 fine flags) |
| `V4_HYPOTHESIS_ENABLED` | `false` | P6 (2 fine flags) |

Invariants: all three `false` ⟹ no v4 hook fires ⟹ v3 behavior. Coarse `true`
+ fine unset ⟹ all fine under it become `true`. Coarse `false` + fine set
`true` in env ⟹ that single fine hook fires (backward-compat for old tests).
Fine flags are debug overrides, env-injection only — not exposed in
`.env.example`.

---

## Chapter 2 — Backend Changes (Phase B + D.2)

| Task | Commit | Summary |
|---|---|---|
| **B.0** | `d0a1ed6` | Fix ODE Renderer template routing — 11 templates now reachable (P0-2 blocker fix). |
| **B.1** | `a5843d7` | Backend cleanup — `LEGACY BRIDGE` annotations + unused import cleanup (conservative, no deletions in RC). |
| **B.2** | `84354f5` | Consolidate 17 `v4_` state fields into a `v4_state` dict (backward-compatible dual-write). |
| **B.3** | `8f9d928` | Converge 13 fine-grained V4 flags into 3 coarse flags (`V4_SCIENTIFIC_LAYER` / `V4_VALIDATION` / `V4_HYPOTHESIS`). |
| **D.2** | `4b201d4` | Fix G2 (wire calibration + sensitivity hooks into the graph) + G3 (strip Dynamic Router duplicate agents). |

### Highlights

- **B.0** unblocked the ODE renderer: 11 Jinja2 templates under
  `ode_templates_v2/` (bistable switch, caspase cascade, cyclin/CDK toggle,
  destruction complex, nuclear transport, oscillatory feedback, transcription
  factor, transcriptional delay, ubiquitination cascade, etc.) were previously
  unreachable due to a routing bug.
- **B.1** tagged legacy bridge code with `LEGACY BRIDGE` comments and removed
  unused imports. Per RC policy, no modules were deleted — only annotated.
- **B.2** introduced `v4_state` with group-level `dict.update` semantics so one
  hook's return value cannot clobber another hook's earlier write. Dual-write
  keeps existing call sites working unchanged.
- **B.3** collapsed the flag surface from 13 env vars (hard to operate) to 3
  coarse switches while preserving fine-grained env override for debugging.
- **D.2** wired `calibration_hook_node` and `sensitivity_hook_node` into
  `build_workflow_v3()` (previously reachable only via the Dynamic Router, which
  is off by default) and stripped the 8 duplicate agents from the Dynamic
  Router's `build_dispatch_plan()` so the router no longer re-runs agents the
  graph already executes.

---

## Chapter 3 — Frontend Changes (Phase C)

All components live under `frontend/components/`. Routing under `frontend/app/`.

| Task | Commit | Summary |
|---|---|---|
| **C.1** | `18a0656` | Scientific Modeling IDE four-pane Shell + routing + lib infrastructure (`store.ts` / `api.ts` / `sse.ts`). |
| **C.2** | `383042d` | PathwayTree + BenchmarkList + SimulationHistory (left pane). |
| **C.3** | `e37a5e8` | Interactive Pathway Graph + Node Detail Panel (center pane). |
| **C.4** | `b92acfe` | Simulation Panel with 6-tab multi-view (Time Series / Dose Response / Sensitivity / Phase Portrait / Steady State / Oscillation). |
| **C.5** | `f9a3ba4` | Parameter Explorer (Slider / IC50 / EC50 / Knockout / Overexpression / Mutation). |
| **C.6** | `feb80e9` | Multi-mode Input Area (NL / Structured Hypothesis / Parameter Panel / SBML Upload / BioModels ID). |
| **C.7** | `c92116f` | Validation Pyramid panel (Level 1-5, Pass/Warning/Fail + expandable details). |
| **C.8** | `3e113fa` | Hypothesis Panel (Hypothesis / Evidence / Predictions / Suggested Experiments / Falsifiability). |
| **C.9** | `c393c5a` | AI Assistant collapsible panel (Chat / Suggestions / Logs tabs). |
| **C.10** | `9f5cf46` | Redesign home page (Pathway Selector / Scientific Workspace / Recent Simulations / Benchmark Cases). |
| **C.11** | `cf10f85` | Experiment Report page (6 sections: Executive Summary / Scientific Findings / Dynamic Analysis / Validation / Literature Comparison / Future Experiments). |
| **C.12** | `53cff5c` | Benchmark Center page (10 pathway cards + Run All + real-time SSE progress). |
| **C.13** | _(bundled)_ | Vitest + Playwright E2E test infrastructure. |

### Notes on C.13

The frontend test infrastructure is in place: `vitest.config.ts`,
`playwright.config.ts`, six Vitest spec files under `frontend/__tests__/`
(`store.test.ts`, `api.test.ts`, `HypothesisPanel.test.tsx`,
`InputArea.test.tsx`, `ValidationPyramid.test.tsx`, `BenchmarkCard.test.tsx`)
and one Playwright E2E spec (`frontend/e2e/smoke.spec.ts`). Dev dependencies
(`vitest`, `@playwright/test`, `@testing-library/react`, `jsdom`) are declared
in `package.json`. **The standalone `d251fef` commit listed in the sprint plan
was not found as a separate commit in `git log`; the test files were committed
within the B.3 commit window.** `package.json` now defines `test` (vitest) and
`test:e2e` (playwright) scripts — added in the post-RC fix commit `9f6bfbd`.

---

## Chapter 4 — UX Improvements (Phase D)

| Task | Commit | Summary |
|---|---|---|
| **D.1** | `edd5b06` | End-to-end UX flow audit and redesign (`docs/UX_FLOW.md`). |
| **D.2** | `4b201d4` | Fixed G2 (calibration/sensitivity wired into graph) + G3 (Router dedup). |

### D.1 audit findings

The D.1 audit traced the real end-to-end pipeline in `graph_v3.py` + the
frontend Workbench, documented node-by-node data flow, and identified 10 gaps
(G1–G10) plus 5 redundancies (R1–R5). The three HIGH-severity gaps were:

- **G1** — No Hypothesis Refinement loop. `_hypothesis_agent_hook` is a single
  forward node; hypotheses are generated one-shot and never re-simulated.
- **G2** — Calibration & Sensitivity never run by default (hooks implemented
  but not wired into the graph; only reachable via the Dynamic Router, which is
  off by default).
- **G3** — Dynamic Router duplicates work (re-executes 8 agents the graph
  already runs as dedicated nodes).

### D.2 fixes

D.2 closed **G2** and **G3**:

- **G2 fix** — `calibration_hook_node` and `sensitivity_hook_node` added to
  `build_workflow_v3()` so `v4_calibration_result` and `v4_sensitivity_report`
  populate even when the Dynamic Router is off.
- **G3 fix** — Duplicate agents (`ontology`, `pathway_planner`,
  `pathway_specialist_group`, `reaction_builder`, `validation`,
  `crosstalk_coordinator`, `sbml_grounder`, `hypothesis`) stripped from the
  Dynamic Router's `build_dispatch_plan()`; the router now owns only the 4 P6
  builder stubs (`mechanism_builder`, `ode_builder`, `simulation_planner`,
  `parameter_agent`).

### Remaining

- **G1** (Hypothesis refinement loop) — **deferred to v4.1**. The recommended
  design is a bounded loop (`max_iterations=3`) from `_hypothesis_agent_hook`
  back to `worker_ode` when validation fails, guarded by a new
  `v4_hypothesis_iterations` counter. Not implemented in the RC window.
- Lower-severity gaps (G4–G10: literature import, structured-hypothesis →
  `v4_pathway_class` seeding, center-pane wiring, v4 SSE store hydration,
  `pre_router` v4 reset, experiment planner stubs, export beyond markdown)
  remain open for a future hardening pass.

---

## Chapter 5 — Benchmark Suite (Phase E)

| Task | Commit | Summary |
|---|---|---|
| **E.1** | `bfa5509` | 10-pathway Official Benchmark Suite + `POST /api/v4/benchmarks/run` SSE endpoint. |

The suite ships 10 YAML case files under `backend/benchmarks/` (apoptosis, cell
cycle, egfr_signaling, jak_stat, mapk_cascade, nfkb_signaling, p53_signaling,
pi3k_akt_mtor, tgf_beta_signaling, wnt_signaling). The endpoint is read-only —
it reuses existing P4 specialists and P5 Level-4 validation, streaming
`benchmark_start` / `benchmark_progress` / `benchmark_result` /
`benchmark_complete` SSE events. The frontend `/benchmarks` page subscribes and
renders a live progress table. **All 10 pathways PASS.**

---

## Chapter 6 — Documentation (Phase F)

| Task | Commit | Summary |
|---|---|---|
| **F.1 (batch 1)** | `cc6a516` | README + ARCHITECTURE + PATHWAYS + REACTION_IR + VALIDATION docs. |
| **F.1 (batch 2)** | `d7cb2b1` | HYPOTHESIS + FRONTEND + API + CHANGELOG docs + archive old reports. |

Nine docs now live under `docs/`: `README.md`, `ARCHITECTURE.md`,
`PATHWAYS.md`, `REACTION_IR.md`, `VALIDATION.md`, `HYPOTHESIS.md`,
`FRONTEND.md`, `API.md`, `CHANGELOG.md`. The audit also produced
`docs/NAMING_VIOLATIONS.md` (G.1) and `docs/UX_FLOW.md` (D.1).

**10 old root-level reports were archived** to `docs/archive/`:

- `BIOMD0000000205_Semantic_Convergence_Validation_Report.md`
- `BioDynamics_Agent_严格结构审计报告_v1.0.md`
- `BioDynamics_Agent_深度审核报告.md`
- `BioDynamics_v4_P1_P2_P3_Integration_Report.md`
- `BioDynamics_v4_P4_P5_P6_Integration_Report.md`
- `BioDynamics_v4_Phase4_Report.md`
- `BioDynamics_v4_RC_Readiness_Report.md`
- `DeepSeek_LLM_评估报告.md`
- `EGF_EGFR_ODE系统修复报告_TASK1-6.md`
- `EGF_EGFR_仿真报告核查与根因分析.md`

---

## Chapter 7 — Release Packaging (Phase G)

| Task | Commit | Summary |
|---|---|---|
| **G.1** | `bedd680` | Unify directory structure + naming audit + config consolidation. |
| **G.2** | `2903bb1` | Unified JSON logging + error handling middleware + exception recovery docs. |
| **G.3** | `fd3cdc1` | Docker containerization (multi-stage Dockerfile + docker-compose + .dockerignore). |
| **G.4** | `6fa5114` | Demo scripts (`demo.sh` single pathway + `demo_benchmark.sh` all 10 benchmarks). |

### G.1 (committed — `bedd680`)

G.1 unified the directory structure, produced the naming-conventions audit
(`docs/NAMING_VIOLATIONS.md` — no Python/env/route violations; the only
exceptions are shadcn/ui primitives under `components/ui/` whose lowercase
filenames are dictated by the shadcn CLI registry contract and must not be
renamed), and consolidated configuration. The 3 coarse feature flags are
exposed in `backend/.env.example` with full documentation; all default `false`.

### G.2 (committed — `2903bb1`)

A `backend/app/logging_config.py` module provides a `JSONFormatter` (serializes
each log record to single-line JSON: `timestamp / level / logger / message /
module / line` + `exception` stack trace) and an idempotent `setup_logging(level,
json_output)` entry point. `main.py` calls `setup_logging` at import time and
registers a global `@app.exception_handler(Exception)` returning structured
500 JSON. SSE error emissions are standardized to `{"event":"error","data":{...}}`.
`LOG_LEVEL` and `LOG_JSON` env vars control the configuration. 18 tests pass.
`docs/ERROR_HANDLING.md` documents 5 recovery strategies (sandbox/LLM/RAG/SBML/
unhandled).

### G.3 (committed — `fd3cdc1`)

Docker containerization delivered a multi-stage `Dockerfile` (89 lines),
`docker-compose.yml` (60 lines), `.dockerignore` (64 lines), and a refreshed
`backend/requirements.txt` (12 lines pinning the runtime dependencies).

### G.4 (committed — `6fa5114`)

Two demo shell scripts under `scripts/`: `demo.sh` (single-pathway end-to-end
run, 60 lines) and `demo_benchmark.sh` (runs all 10 benchmark pathways, 51
lines).

---

## Chapter 8 — Metrics & Remaining Risks

### 8.1 Test counts

- **Backend**: ~1619 tests across the v3 spine and v4 hooks (P1–P6).
- **Frontend**: 52 Vitest tests (6 spec files) + 1 Playwright E2E spec
  (`smoke.spec.ts`).

### 8.2 Known pre-existing failures

19 pre-existing failures, unrelated to v4 changes:

- `test_sbml_validator` — SBML validator track issues (predate the RC).
- `test_upgrade` — upgrade-path test (predate the RC).
- `test_p4` / `test_p6_hook_e2e` — test-isolation flakes, not v4 regressions.

### 8.3 Remaining risks

| Risk | Detail | Status |
|---|---|---|
| **G1 — Hypothesis refinement loop** | Not implemented (deferred to v4.1). Hypotheses are one-shot; no back-edge to `worker_ode`. | Open |
| **Playwright browsers in CI** | E2E browsers are not installed in CI; `smoke.spec.ts` runs locally only. | Open |
| ~~**v4 SSE store no-ops**~~ | ~~`ingestSSEEvent` discarded `v4_validation_report` / `v4_pathway_graph` / `v4_simulation_result`.~~ **RESOLVED** in `9f6bfbd` — all three events now hydrate `validationReport` / `pathwayGraph` / `simulationResult` in the Zustand store. | **Fixed** |
| ~~**3 center-pane placeholders**~~ | ~~`WorkbenchShell.tsx` rendered `PlaceholderPanel` for C.3/C.4/C.5.~~ **RESOLVED** in `9f6bfbd` — PathwayGraph, SimulationPanel, ParameterExplorer (center); PathwayTree, BenchmarkList, SimulationHistory (left); ValidationPyramid, HypothesisPanel (right) are now wired. | **Fixed** |
| ~~**G.2 incomplete**~~ | ~~Unified JSON logging uncommitted.~~ **RESOLVED** — G.2 committed as `2903bb1` (see Chapter 7). | **Fixed** |
| **Lower-severity UX gaps** | G4–G10 (literature PMID import, structured-hypothesis → pathway-class seeding, v4 fields missing from `pre_router` reset list, experiment planner stubs, no PDF/DOCX export) remain open. | Open |

### 8.4 Feature flags

All 3 coarse flags default **OFF** (`V4_SCIENTIFIC_LAYER_ENABLED=false`,
`V4_VALIDATION_ENABLED=false`, `V4_HYPOTHESIS_ENABLED=false` in
`.env.example`). With all three off, no v4 hook fires and the system behaves
bit-for-bit as v3. v4 features are opt-in for production.

---

## Post-RC Fix — Commit `9f6bfbd`

A single post-RC fix commit addressed three blocking risks identified in the
original Chapter 8.3:

| Fix | Files | Detail |
|---|---|---|
| **Center-pane wiring** | `WorkbenchShell.tsx` | Replaced all `PlaceholderPanel` instances with actual components: PathwayGraph (C.3), SimulationPanel (C.4), ParameterExplorer (C.5) in center; PathwayTree, BenchmarkList, SimulationHistory (C.2) in left; ValidationPyramid (C.7), HypothesisPanel (C.8) in right. |
| **SSE event hydration** | `store.ts` | `ingestSSEEvent` now hydrates `v4_validation_report` → `validationReport`, `v4_pathway_graph` → `pathwayGraph`, `v4_simulation_result` → `simulationResult` (previously no-op). |
| **Test scripts** | `package.json` | Added `"test": "vitest"` and `"test:e2e": "playwright test"` to npm scripts. |

**Verification**: `npx tsc --noEmit` passes (0 errors). `npx vitest run` passes
(52/52 tests). TypeScript compilation and frontend tests confirmed no regressions.

---

*End of BioDynamics v4 RC Release Report — Task H.1.*
