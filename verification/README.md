# BioDynamics v4 — Verification Suite

This directory holds the full verification & regression suite for BioDynamics v4.
It is **independent of production code** — every test here only *reads* v4 outputs
(state, ODE systems, reaction IR, pathway graphs) and asserts against reference
truth. No file under `backend/app/` should be modified to make these tests pass.

The suite is organized as 10 themed sub-directories. Each one owns its own
config (`*.yaml`) + pytest module(s) and writes its artifacts to `reports/`.

---

## Directory map

| # | Directory | Purpose | Input | Output | Execution |
|---|-----------|---------|-------|--------|-----------|
| 1 | `biomodels_regression/` | Download 30+ EBI BioModels reference SBML models, re-simulate each with roadrunner (gold standard) and compare against BioDynamics v4 simulation output. Metrics: RMSE, Pearson, peak time/amplitude, steady state, AUC. | `biomodels_config.yaml` (30 entries), EBI BioModels REST API, cached SBML under `backend/data/raw/` | Per-model metrics JSON + pass/fail matrix in `reports/biomodels_regression/` | `pytest verification/biomodels_regression/test_biomodels_regression.py` |
| 2 | `pathway_regression/` | 50 mechanism-level regression cases (5 per pathway × 10 pathways). Each case asserts species count, reaction mechanism, ODE term presence, and validation-level pass/fail for a single biochemical step. | `pathway_config.yaml` (50 entries), v4 pathway specialist output | Per-case assertion report + skip list of known P0 failures in `reports/pathway_regression/` | `pytest verification/pathway_regression/test_pathway_regression.py` |
| 3 | `hypothesis_validation/` | Validates the P6 Hypothesis Agent: generated hypotheses must be falsifiable, grounded in v4 state, and produce a testable experiment design. | v4 hypothesis list (`v4_hypothesis_list`), falsifiability checker output | Hypothesis quality score + falsifiability matrix in `reports/hypothesis_validation/` | `pytest verification/hypothesis_validation/` |
| 4 | `parameter_stress/` | Parameter robustness / stress tests: ±50% perturbation of kinetic parameters, Morris & Sobol global sensitivity, detection of parameter regimes that flip qualitative behavior (oscillation/mono-stable/bi-stable). | v4 ODE system + parameter set, sensitivity analyzer | Sensitivity indices, bifurcation table in `reports/parameter_stress/` | `pytest verification/parameter_stress/` |
| 5 | `solver_validation/` | Numerical solver validation: DDE solver correctness vs analytic solutions, oscillation/bistability detector accuracy, stiff-system stability, conservation of mass over long horizons. | v4 solvers (`dde_solver`, `oscillation_detector`, `bistability_detector`) | Solver error table + detector ROC in `reports/solver_validation/` | `pytest verification/solver_validation/` |
| 6 | `ontology_validation/` | Validates the P1 Ontology Layer: SBO term assignment, 5-level species grounding, canonical species merging, KEGG/Reactome pathway ID resolution, HGNC/UniProt/CHEBI client caching. | v4 grounding ledger, ontology clients | Grounding coverage report + mislabel list in `reports/ontology_validation/` | `pytest verification/ontology_validation/` |
| 7 | `ui_workflow/` | End-to-end UI / SSE workflow tests: hypothesis → plan → simulate → report streaming, clarification-needed short-circuit, multi-pathway crosstalk rendering. Uses Playwright against the v4 frontend. | Running v4 backend + frontend (or recorded SSE fixtures) | Playwright trace + screenshot diffs in `reports/ui_workflow/` | `pytest verification/ui_workflow/` (requires `PLAYWRIGHT_BROWSERS_PATH`) |
| 8 | `benchmark/` | Official v4 benchmark suite (EGFR / MAPK / NF-κB / p53 / Wnt etc.) — reproduces published literature dynamics and asserts pathway-specific thresholds (peak time, amplification, half-life, mass-conservation error). | `backend/benchmarks/*.yaml` literature benchmarks | Benchmark pass/fail + performance (runtime/memory) table in `reports/benchmark/` | `pytest verification/benchmark/` |
| 9 | `reports/` | Aggregated machine-readable artifacts produced by all suites above (JSON / CSV / JUnit XML). Consumed by the dashboard and CI. | Writes from every other directory | `reports/<suite>/*.json`, `reports/junit.xml`, `reports/summary.json` | Auto-populated by pytest runs; no manual execution |
| 10 | `dashboard/` | Static HTML dashboard generator that visualizes the latest `reports/` run: per-suite pass rate, per-pathway heat-map, known-failure ledger, trend over commits. | `reports/summary.json` | `dashboard/index.html` (+ assets) | `python verification/dashboard/build_dashboard.py` (or `pytest` plugin) |

---

## Pathway coverage

All suites target the same 10 core pathways defined in
`backend/app/ontology/pathway_registry.py`:

`EGFR_RTK` · `MAPK_ERK` · `PI3K_AKT_mTOR` · `p53` · `APOPTOSIS` ·
`CELL_CYCLE` · `JAK_STAT` · `NF_KB` · `WNT` · `TGF_BETA`

---

## Validation levels (L1–L5)

Pathway regression cases are tagged with the deepest validation pyramid level
they exercise (defined in `backend/app/validation_v2/`):

- **L1** — Internal Consistency (mass conservation / non-negative / steady state)
- **L2** — SBML / BioModels structural & dynamic comparison
- **L3** — Cross-Pathway crosstalk & shared-species conservation
- **L4** — Benchmark against published literature thresholds
- **L5** — Hypothesis validation (falsifiability & experiment design)

---

## Running the suite

From the `bio-dynamics-agent/` project root:

```bash
# Run a single suite
pytest verification/biomodels_regression/ -v

# Run everything (slow; downloads BioModels on first run)
pytest verification/ -v --junitxml=verification/reports/junit.xml

# Build the dashboard from the latest report
python verification/dashboard/build_dashboard.py
```

> **Note:** Several cases are marked `@pytest.mark.skip` to document known P0
> bugs in the v4 ODE Renderer (FM-001/002/003 — reads non-existent fields and
> emits zero-flux ODEs). These are intentional skips, not failures; they will be
> un-skipped once the ODE Renderer fix lands.

---

## Conventions

- **Read-only**: suites must not mutate `backend/`. Any fixture data the suite
  needs lives under the suite's own directory or under `backend/data/raw/`
  (treated as an immutable cache).
- **Config-driven**: every parametrized suite is driven by a `*_config.yaml` so
  cases can be added without touching Python.
- **Deterministic**: tests set explicit seeds and fixed `t_end` so re-runs are
  reproducible.
