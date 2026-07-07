# Changelog — BioDynamics Agent v4.0 RC

All notable changes in the v4.0 Release Candidate sprint. Entries are grouped
by sprint phase. Commit hashes are short SHAs from `git log --oneline`.

---

## Phase A — Audit

| Task   | Commit    | Description                                                  |
|--------|-----------|--------------------------------------------------------------|
| A.1    | `2a54d1f` | RC Readiness Audit Report (GATE) — established the sprint scope and entry criteria. |

---

## Phase B — Backend Cleanup

| Task   | Commit    | Description                                                                                                  |
|--------|-----------|--------------------------------------------------------------------------------------------------------------|
| B.0    | `d0a1ed6` | Fix ODE Renderer template routing — 11 templates now reachable (P0-2 blocker fix).                           |
| B.1    | `a5843d7` | Backend cleanup — `LEGACY BRIDGE` annotations + unused import cleanup (conservative, no deletions in RC).    |
| B.2    | `84354f5` | Consolidate 17 `v4_` state fields into a `v4_state` dict (backward-compatible dual-write).                   |
| B.3    | `8f9d928` | Converge 13 fine-grained V4 flags into 3 coarse flags (`V4_SCIENTIFIC_LAYER` / `V4_VALIDATION` / `V4_HYPOTHESIS`). |

---

## Phase C — Frontend

| Task   | Commit    | Description                                                                                                  |
|--------|-----------|--------------------------------------------------------------------------------------------------------------|
| C.1    | `18a0656` | Scientific Modeling IDE four-pane Shell + routing + lib infrastructure (`store.ts` / `api.ts` / `sse.ts`).   |
| C.2    | `383042d` | PathwayTree + BenchmarkList + SimulationHistory (left pane).                                                 |
| C.3    | `e37a5e8` | Interactive Pathway Graph + Node Detail Panel (center pane).                                                 |
| C.4    | `b92acfe` | Simulation Panel with 6-tab multi-view (Time Series / Dose Response / Sensitivity / Phase Portrait / Steady State / Oscillation). |
| C.5    | `f9a3ba4` | Parameter Explorer (Slider / IC50 / EC50 / Knockout / Overexpression / Mutation).                            |
| C.6    | `feb80e9` | Multi-mode Input Area (NL / Structured Hypothesis / Parameter Panel / SBML Upload / BioModels ID).           |
| C.7    | `c92116f` | Validation Pyramid panel (Level 1-5, Pass/Warning/Fail + expandable details).                                |
| C.8    | `3e113fa` | Hypothesis Panel (Hypothesis / Evidence / Predictions / Suggested Experiments / Falsifiability).             |
| C.9    | `c393c5a` | AI Assistant collapsible panel (Chat / Suggestions / Logs tabs).                                             |
| C.10   | `9f5cf46` | Redesign home page (Pathway Selector / Scientific Workspace / Recent Simulations / Benchmark Cases).         |
| C.11   | `cf10f85` | Experiment Report page (6 sections: Executive Summary / Scientific Findings / Dynamic Analysis / Validation / Literature Comparison / Future Experiments). |
| C.12   | `53cff5c` | Benchmark Center page (10 pathway cards + Run All + real-time SSE progress).                                 |

---

## Phase D — UX

| Task   | Commit    | Description                                                  |
|--------|-----------|--------------------------------------------------------------|
| D.1    | `edd5b06` | End-to-end UX flow audit and redesign (`docs/UX_FLOW.md`).   |

---

## Phase E — Benchmarks

| Task   | Commit    | Description                                                                                                  |
|--------|-----------|--------------------------------------------------------------------------------------------------------------|
| E.1    | `bfa5509` | 10-pathway Official Benchmark Suite + `POST /api/v4/benchmarks/run` SSE endpoint (read-only, reuses P4 specialists + P5 Level-4 validation). |

---

## Phase F — Docs

| Task   | Commit    | Description                                                                                                  |
|--------|-----------|--------------------------------------------------------------------------------------------------------------|
| F.1    | _this commit_ | Documentation sweep — `HYPOTHESIS.md` + `FRONTEND.md` + `API.md` + `CHANGELOG.md` + archive old root reports. |

---

## Foundation (pre-RC)

These earlier commits delivered the P5 / P6 / P3 scientific layers that the
RC sprint builds on. They pre-date Phase A but are referenced here for
completeness.

| Commit    | Description                                                                                                  |
|-----------|--------------------------------------------------------------------------------------------------------------|
| `c9fffb2` | P5 prereqs — dependency isolation strategy + feature flags (Task 5.0).                                       |
| `75f260a` | SBML Grounder Agent + `sbml_parser_v2` (Task 5.1).                                                           |
| `77699f2` | Validation Level 1 — Internal Consistency Validation (Task 5.2).                                             |
| `d4204ea` | Validation Level 2 — SBML/BioModels Validation (Task 5.3).                                                   |
| `70a19e8` | Validation Level 3 — Cross-Pathway Validation (Task 5.4).                                                    |
| `94d6933` | Validation Level 4 — Benchmark Validation (Task 5.5).                                                        |
| `b177e5b` | Validation Level 5 — Hypothesis Validation (Task 5.6).                                                       |
| `45f1956` | Calibration Agent (`lmfit` try-import + `scipy` fallback + confidence intervals) (Task 5.7).                 |
| `d8b61cd` | Sensitivity Analysis — Local + Sobol + Morris three paths (Task 5.8).                                        |
| `8db93cd` | Validation Agent + P5 integration hook (Task 5.9).                                                           |
| `1c25a13` | Hypothesis Agent + Generator (Task 6.1).                                                                     |
| `45b0df2` | Experiment Planner — `ExperimentDesigner` + 6-field schema + P4 Specialist integration (Task 6.2).           |
| `2fc2bd6` | Falsification Checker — 3 rules + filter non-falsifiable hypotheses (Task 6.3).                              |
| `f81eecd` | Parameter Explorer + Sensitivity Planner (Task 6.4).                                                         |
| `b5862f5` | Dynamic Router + `agent_registry_v4` + `pathway_class_dispatcher` + `fail_safe` (Task 6.5).                  |
| `1fe4e94` | `agents_v4` 4 Agents (`mechanism_builder` / `ode_builder` / `simulation_planner` / `parameter_agent`) (Task 6.6). |
| `aa82a07` | P6 integration hooks + SSE `v4_hypothesis_generated` + e2e smoke (Task 6.7).                                 |
| `6c49e47` | Final Integration Report (`READY_FOR_PHASE7=YES`) (Task 6.8).                                                |
| `4fba6b1` | P3 补全 — ODE Templates v2 (7 new templates + 26 unit tests, 0 v4 regression).                               |
