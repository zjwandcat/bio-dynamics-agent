# BioDynamics Agent v4 RC — Validation Pyramid, Calibration & Sensitivity

> **Task F.1** — Documentation only. Reference for the 5-level Validation Pyramid,
> Calibration Agent, and Sensitivity Analyzer introduced in Phase 5.
> Source of truth: `backend/app/validation_v2/`, `backend/app/calibration/`,
> `backend/app/sensitivity/`.

---

## 1. Overview

The v4 Validation Pyramid is a **five-tier, short-circuiting validation chain**
that runs after `worker_ode` produces a Reaction IR + ODE system and before
`worker_report` emits the final markdown. Every tier is gated behind the coarse
flag `V4_VALIDATION_PYRAMID_ENABLED` (default `false`); when the flag is off the
entire pyramid returns `{}` and the v3 pipeline is untouched.

Two adjacent agents extend the pyramid with quantitative rigor:

- **Calibration Agent** (`V4_CALIBRATION_AGENT_ENABLED`) — fits ODE parameters
  to BioModels reference or user-supplied experimental data using
  `lmfit` (primary) or `scipy.optimize.least_squares` (fallback), and reports
  95% confidence intervals.
- **Sensitivity Analyzer** (`V4_CALIBRATION_AGENT_ENABLED`, shared flag) — runs
  local forward-difference sensitivity (always), plus Sobol and Morris global
  sensitivity when `SALib` is installed.

> **Wiring note:** `calibration_hook_node` and `sensitivity_hook_node` are fully
> implemented but **not wired into `build_workflow_v3()`** by default. They are
> reachable via the Dynamic Router's `execute_agent("calibration"|"sensitivity")`
> path, itself gated behind `V4_DYNAMIC_ROUTING_ENABLED`. See `docs/ARCHITECTURE.md`
> §4 for the flag hierarchy.

---

## 2. Validation Pyramid — 5 Levels

```
                ┌────────────────────────────────────┐
                │   ValidationAgent.validate(state)  │
                └────────────────┬───────────────────┘
                                 │  sequential, short-circuit on fail
   L1 Internal ─────────────────►│
   L2 SBML/BioModels ───────────►│
   L3 Cross-Pathway ────────────►│
   L4 Benchmark ────────────────►│
   L5 Hypothesis ───────────────►│
                                 ▼
              overall_pass = L1 ∧ L2 ∧ L3 ∧ L4 ∧ L5
              overall_pass=False → pending_clarification
                                  → clarification_needed SSE
```

### Aggregation rules

| Rule | Behavior | Source |
|------|----------|--------|
| `overall_pass` | Logical AND of all 5 level `pass` flags | `validation_agent.py::validate` |
| Short-circuit | Any level `pass=False` → `overall_pass=False`, `failed_levels` populated | spec.md §280-317 |
| `skipped` semantics | L2 skipped → `pass=False` (blocking); L3 single-pathway skipped → `pass=True`; L5 P6-off skipped → `pass=True`; L4 no-metrics skipped → `pass=True` | audit §7.2, spec.md §289/§299/§317 |
| Failure isolation | Each level wrapped in try/except; one level's exception does not abort others | `_run_level` |
| Existing-report reuse | If a level hook already wrote its report, `ValidationAgent` reuses it instead of recomputing | `_run_level` |

### 2.1 Level 1 — Internal Consistency

**File:** `backend/app/validation_v2/level1_internal.py`
**Class:** `Level1InternalValidator` · **Hook:** `level1_hook_node`
**Flag:** `V4_VALIDATION_PYRAMID_ENABLED`

Checks the ODE system for structural sanity **before** any simulation runs.
Consumes `v4_ode_system` (ode_code) and `v4_reaction_ir` (constraints + species).

| # | Check | Method | Failure impact |
|---|-------|--------|----------------|
| 1 | Mass conservation | Parse `reaction_ir.constraints[type=mass_conservation]` expressions (e.g. `EGFR + pEGFR + EGF_EGFR = EGFR_total`); sum LHS initial concentrations, compare to RHS, compute relative error | Error > 5% → `pass=False` |
| 2 | Non-negative concentration | Regex-scan `dX/dt = ...` for constant degradation terms (e.g. `- 0.5` not followed by `* X`); proportional degradation `-k*X` is safe | Recorded as violation (does not alone block) |
| 3 | Steady state | Check each `dX/dt` equation has a self-degradation term (`-k*X` / `-X*k` / `-X`); simplified necessary-condition check | Recorded (does not alone block) |
| 4 | Numerical stability | Detect division-by-variable, `log()` of expression, and stiff systems (`max_rate/min_rate > 1e6`) | Any violation → `pass=False` |
| 5 | Constraint satisfaction | Validate all P2 `Constraint` schema entries (mass_conservation / steady_state / non_negative / enzymatic / thermodynamic) | Recorded (does not alone block) |

**Pass criteria:** `mass_conservation_error ≤ 0.05` **AND** `numerical_stability = True`

**Constants:**
- `MASS_CONSERVATION_THRESHOLD = 0.05` (5%, spec.md §281)
- `STIFF_SYSTEM_RATIO = 1e6`

**Output fields:**
```python
{
    "pass": bool,
    "mass_conservation_error": float,          # 0.0 = no error or no constraints
    "non_negative_violations": list[dict],     # {species_name, equation, reason}
    "steady_state_check": bool,
    "numerical_stability": bool,
    "constraint_violations": list[dict],       # {constraint_name, expected, actual, diff, reason}
}
```

### 2.2 Level 2 — SBML / BioModels Reference

**File:** `backend/app/validation_v2/level2_sbml.py`
**Class:** `Level2SBMLValidator` · **Hook:** `level2_hook_node`
**Flag:** `V4_VALIDATION_PYRAMID_ENABLED`

Compares the v4 ODE simulation against a curated BioModels SBML reference
(`sbml_model_id`). Three tracks depending on `roadrunner` availability and
inputs.

| Track | When | Method | Pass criteria |
|-------|------|--------|---------------|
| **A** | `roadrunner` installed AND `sbml_model_text` present | Run SBML sim (0–60 min, 200 pts) + v4 ODE sim; extract peak/peak_time/amplification per species; align species by ontology ID; compute mean relative/absolute diffs | `peak_time_diff ≤ thresholds.peak_time_diff` AND `amplification_diff ≤ thresholds.amplification_diff` |
| **B** | `roadrunner` missing OR Track A threw | Structural similarity score (4 dimensions: species-count ratio, reaction-count ratio, kinetics-type overlap, ontology-aligned species ratio); each 0.25, summed/averaged | `similarity_score ≥ 0.6` |
| **skipped** | Missing `sbml_model_id` or `v4_ode_system` | — | `pass=False` (blocking — audit §7.2 fix) |

**Key audit fixes baked in:**
- **§7.2** — Track B diff metrics (`peak_diff`, `peak_time_diff`, `amplification_diff`) are forced to `None` (never `0`), preventing false "perfect match" signals.
- **§7.2** — `skipped` state returns `pass=False`, not the Oracle's default `pass=True`, so upstream errors cannot silently pass.
- **§10.3** — Species alignment uses **ontology IDs** (HGNC → UniProt priority), not string matching. Local ontology KB (`_LOCAL_ONTOLOGY`) is a fallback for v4 species lacking `ontology_ref`.

**Pathway-specific thresholds** (`backend/app/validation_v2/thresholds.py::PathwayThresholds`):

| Pathway | `peak_time_diff` (min) | `amplification_diff` | Rationale |
|---------|------------------------|----------------------|-----------|
| EGFR / EGFR_RTK | 2.0 | 0.20 | Fast receptor activation, low jitter |
| MAPK | 2.0 | 0.30 | Cascade amplification |
| NF_KB / NF_KAPPAB | 30.0 | 0.50 | Oscillation phase hard to align (spec.md §291) |
| P53 | 30.0 | 1.00 | Pulse amplitude high biological variance (spec.md §291) |
| WNT | 60.0 | 0.50 | Slow β-catenin steady-state kinetics |
| PI3K_AKT_mTOR | 5.0 | 0.30 | — |
| APOPTOSIS | 30.0 | 0.50 | — |
| CELL_CYCLE | 60.0 | 0.50 | — |
| JAK_STAT | 5.0 | 0.30 | — |
| TGF_BETA | 30.0 | 0.50 | — |
| MULTI | 30.0 | 0.50 | Multi-pathway uses most lenient |
| default | 5.0 | 0.30 | Unrecognized pathway |

Lookup priority: exact key → prefix match (`MULTI:EGFR_RTK+...` → `MULTI`) → substring match → default.

**Output fields:**
```python
# Track A
{"pass": bool, "track": "A", "peak_diff": float|None, "peak_time_diff": float|None,
 "amplification_diff": float|None, "sbml_sim_available": True,
 "method": "roadrunner_simulation", "sbml_model_id": str, "pathway_class": str,
 "thresholds_applied": dict, "aligned_species_count": int}

# Track B
{"pass": bool, "track": "B", "peak_diff": None, "peak_time_diff": None,
 "amplification_diff": None, "sbml_sim_available": False,
 "method": "structural_similarity", "similarity_score": float}

# Skipped
{"pass": False, "track": "skipped", "peak_diff": None, "peak_time_diff": None,
 "amplification_diff": None, "sbml_sim_available": False, "method": "skipped: <reason>"}
```

### 2.3 Level 3 — Cross-Pathway

**File:** `backend/app/validation_v2/level3_crosstalk.py`
**Class:** `Level3CrossPathwayValidator` · **Hook:** `level3_hook_node`
**Flag:** `V4_VALIDATION_PYRAMID_ENABLED`

Validates multi-pathway consistency. Consumes P4 Coordinator outputs
(`v4_crosstalk_edges`, `v4_shared_species`, `v4_time_scale_alignment`,
`v4_specialist_outputs`).

| # | Check | Method | Failure impact |
|---|-------|--------|----------------|
| 1 | Cross-talk consistency | For each cross-talk edge, verify `target_pathway`'s species set contains `target_node` (i.e. the edge is accepted as input on the receiving side) | Inconsistency → `pass=False` |
| 2 | Shared species conservation | For each shared species, sum produced (products) vs consumed (substrates) across all specialist reactions; compute `\|produced - consumed\| / max(produced, consumed)` | Max error > 10% → `pass=False` |
| 3 | Time-scale alignment | Check `v4_time_scale_alignment.unified_max_step` is a positive finite number | Invalid/missing → `pass=False` |

**Single-pathway shortcut:** If `v4_pathway_class` lacks the `MULTI:` prefix,
Level 3 returns `skipped` with `pass=True` (no cross-talk to validate). This
differs from Level 2's skipped (which is `pass=False`).

**Pass criteria (multi-pathway):** `crosstalk_consistency ∧ (shared_species_conservation ≤ 0.10) ∧ time_scale_alignment`

**Constant:** `SHARED_SPECIES_CONSERVATION_THRESHOLD = 0.10` (10%, spec.md §298)

**Output fields:**
```python
{
    "pass": bool,
    "crosstalk_consistency": bool,
    "shared_species_conservation": float,     # max error across shared species (0.0 = conserved)
    "time_scale_alignment": bool,
    # skipped (single-pathway) adds:
    "skipped": True, "reason": "single_pathway"
}
```

### 2.4 Level 4 — Benchmark

**File:** `backend/app/validation_v2/level4_benchmark.py`
**Class:** `Level4BenchmarkValidator` · **Hook:** `level4_hook_node`
**Flag:** `V4_VALIDATION_PYRAMID_ENABLED`

Compares simulation `metrics` (from N8 scientific_features) against
pathway-specific literature benchmarks curated in `BENCHMARK_REGISTRY`.

**Benchmark registry** (5 pathways, spec.md §304-309):

| Pathway | Benchmark | Metric | Expected range | Tolerance | Source PMID |
|---------|-----------|--------|----------------|-----------|-------------|
| EGFR_RTK | pEGFR 5-10 min peak | `peak_time_minutes` | (5.0, 10.0) | 2.0 min | PMID:12124381 (Schoeberl 2002) |
| MAPK_ERK | Zero-order ultrasensitivity Hill > 2 | `hill_coefficient` | (2.0, None) | 0.5 | PMID:14757805 (Markevich 2004) |
| NF_KB | Oscillation period 1-2 h | `oscillation_period_hours` | (1.0, 2.0) | 0.5 h | PMID:14975635 (Nelson 2004) |
| p53 | Pulse period 5-7 h | `pulse_period_hours` | (5.0, 7.0) | 1.0 h | PMID:10644694 (Lev Bar-Or 2000) |
| WNT | β-catenin steady-state < 10 nM (Wnt off) | `steady_state_nM` | (None, 10.0) | 2.0 nM | PMID:12906785 (Lee 2003) |

**Evaluation logic per benchmark:**
1. Extract `actual` from `metrics[benchmark.metric]`.
2. If `actual` is `None`/missing → `pass=False` (missing data is failure).
3. If `actual` ∈ `expected_range` → `pass=True`, `diff=0`.
4. If `actual` ∉ range but `diff ≤ tolerance` → `pass=True` (within tolerance).
5. Else → `pass=False`.

**Pass criteria:** `all(benchmark.pass)` — any single benchmark failure blocks.

**Multi-pathway support:** `MULTI:EGFR_RTK+PI3K_AKT_mTOR` is parsed via
`parse_pathway_class`; each recognized sub-pathway yields its benchmark
(deduplicated by metric key).

**Metrics-not-computed shortcut:** If `state.metrics` is empty (Pyramid hook
runs before `worker_report`), Level 4 returns `skipped` with `pass=True` to
avoid blocking the pipeline before N8 runs.

**Output fields:**
```python
{
    "pass": bool,
    "benchmarks": [
        {"name": str, "source_pmid": str, "expected": {"range": [min, max], "tolerance": float},
         "actual": float|None, "diff": float|None, "pass": bool, "pathway_class": str}
    ],
    "method": "benchmark_validation" | "no_benchmark_matched" | "failed: <reason>",
    "pathway_class": str
}
```

### 2.5 Level 5 — Hypothesis

**File:** `backend/app/validation_v2/level5_hypothesis.py`
**Class:** `Level5HypothesisValidator` · **Hook:** `level5_hook_node`
**Flag:** `V4_VALIDATION_PYRAMID_ENABLED` (gates hook); `V4_HYPOTHESIS_AGENT_ENABLED` (gates P6)

Validates P6 Hypothesis Agent output (`v4_hypothesis_list`) against literature
PMIDs and optional user-supplied experimental data. This is an **interface
definition** consumed by P6 — when P6 is off, Level 5 auto-skips.

**Per-hypothesis evaluation:**
| Evidence source | Field | Effect |
|-----------------|-------|--------|
| Literature support | `hypothesis.supporting_pmids` | `validated=True` if non-empty |
| Literature falsification | `hypothesis.falsifying_pmids` | Recorded as evidence (does not alone falsify) |
| Experimental support | `experimental_data.validated_hypothesis_ids` contains `hypothesis_id` | `validated=True` |
| Experimental falsification | `experimental_data.falsified_hypothesis_ids` contains `hypothesis_id` | `falsified=True` (highest priority) |

**Decision rules:**
- `validated = has_literature_support OR has_experimental_support`
- `falsified = experimental_falsification` (overrides validated)
- `low_confidence = NOT validated AND NOT falsified` (no support of any kind)

**Pass criteria:** Always `pass=True` (spec.md §316 — low_confidence and
falsified hypotheses do **not** block the pipeline; they are recorded for
human review).

**Skipped conditions (both return `pass=True`):**
- P6 not enabled (`V4_HYPOTHESIS_AGENT_ENABLED=false`)
- P6 enabled but `v4_hypothesis_list` is empty

**Output fields:**
```python
{
    "pass": True,                              # always True (non-blocking)
    "hypotheses_validated": int,
    "hypotheses_falsified": int,
    "evidence_support": list[dict],            # {type, source, support, hypothesis_id}
    "low_confidence": bool,                    # True if any hypothesis lacks support
    # skipped adds:
    "skipped": True, "reason": "P6_hypothesis_agent_not_enabled" | "empty_hypothesis_list"
}
```

---

## 3. Calibration Agent

**Files:** `backend/app/calibration/`
**Class:** `CalibrationAgent` (`calibration_agent.py`) · **Hook:** `calibration_hook_node`
**Flag:** `V4_CALIBRATION_AGENT_ENABLED` (shared with Sensitivity)
**`AGENT_VERSION = "v4.0"`**

Fits ODE parameters to reference data (BioModels simulation or user-supplied
experimental data) and reports 95% confidence intervals.

### 3.1 Method

`CalibrationAgent.calibrate(state)` orchestrates:

1. **Extract target params** (priority order):
   - `v4_grounding_ledger.ode_equations[].parameter_ids` (P4 output)
   - `state.parameters` keys (v3 params)
   - `v4_ode_system.parameters` keys (P3 output)
2. **Extract reference data** from `experimental_data.user_data` (or
   `experimental_data` directly), looking for `observations` / `values` lists.
3. **Fit** via `LeastSquaresFitter.fit(target_params, reference_data)`.
4. **Estimate CIs** via `ConfidenceIntervalEstimator.estimate(fit_result, n_samples=100)`.
5. **Mark uncalibrated** params (fit failed OR CI estimation failed).

### 3.2 LeastSquaresFitter — dual backend

**File:** `backend/app/calibration/least_squares_fitter.py`

| Backend | When | Library | Method | Bounds | Max evals |
|---------|------|---------|--------|--------|-----------|
| `lmfit` | `LMFIT_AVAILABLE=True` | `lmfit.minimize` | `leastsq` (Levenberg-Marquardt) | per-param `[1e-6, 1e3]`, init `1.0` | library default |
| `least_squares` | `LMFIT_AVAILABLE=False` (fallback) | `scipy.optimize.least_squares` | `trf` (trust-region reflective) | same bounds | `max_nfev=200` |

**`FitResult` dataclass:** `success`, `params` (dict), `cost` (residual sum of
squares), `nfev`, `message`, `method` (`"lmfit"` / `"least_squares"`),
`residuals` (list, for bootstrap CI), `raw` (original optimizer object).

**Default model_func:** placeholder returning parameter-product list —
**production must inject a real ODE forward-simulator** (roadrunner /
`scipy.solve_ivp`).

### 3.3 ConfidenceIntervalEstimator — dual path

**File:** `backend/app/calibration/confidence_interval.py`
**`confidence_level = 0.95`** · **`Z_95 = 1.96`** · **`DEFAULT_CI_N_SAMPLES = 100`**

| Path | When | Formula |
|------|------|---------|
| `lmfit` | `fit_result.method == "lmfit"` AND `LMFIT_AVAILABLE=True` AND `raw` present | `CI = mean ± 1.96 * stderr` (stderr from `lmfit.MinimizerResult.params[name].stderr`) |
| `bootstrap` | Otherwise (scipy path OR lmfit stderr missing) | Resample residuals `n_samples=100` times with Box-Muller normal noise; empirical percentiles `[2.5%, 97.5%]`; `std_error ≈ (upper - lower) / (2 * 1.96)` |

Params with `stderr=None` or `stderr ≤ 0` are marked `uncalibrated=True`.

### 3.4 Output

```python
{"v4_calibration_result": {
    "calibrated_params": dict,            # {param_name: fitted_value}
    "confidence_intervals": dict,         # {param_name: {lower, upper, std_error, method, uncalibrated}}
    "uncalifiable": list[str],            # params that failed fit or CI
    "method": "lmfit" | "least_squares" | "none",
    "agent_version": "v4.0",
    "warnings": list[str],
    "fallback": bool                      # True if entire agent degraded
}}
```

**Convergence criteria:** `fit_result.success=True` (lmfit/scipy native flag).
On failure, all target params are added to `uncalifiable` and the result is
non-blocking (`fallback=True`, empty `calibrated_params`).

**Task B.2 dual-write:** the hook also writes
`v4_state.validation.calibration_result` via `set_v4_state`.

---

## 4. Sensitivity Analyzer

**Files:** `backend/app/sensitivity/`
**Class:** `SensitivityAnalyzer` (`sensitivity_analyzer.py`) · **Hook:** `sensitivity_hook_node`
**Flag:** `V4_CALIBRATION_AGENT_ENABLED` (shared with Calibration, spec.md §461)
**`AGENT_VERSION = "v4.0"`**

Orchestrates three sensitivity methods: **local** (always runs, no deps),
**Sobol** and **Morris** (both require `SALib`).

### 4.1 Local Sensitivity (always available)

**File:** `backend/app/sensitivity/local_sensitivity.py`
**Class:** `LocalSensitivityAnalyzer` · **`delta = 0.01` (1%) · `relative = True`**

**Method:** Forward difference. For each parameter `p`:
- `perturbed = p * (1 + delta)` (relative mode, default)
- `sensitivity = (f(perturbed) - f(baseline)) / f(baseline)` if baseline ≠ 0
- `sensitivity = f(perturbed) - f(baseline)` if baseline = 0 (absolute, avoids div-by-zero)

Failed perturbations (exception / NaN / Inf) → `sensitivity = 0.0` + warning.

**`LocalSensitivityResult`:** `param_name`, `sensitivity`, `baseline`,
`perturbed`, `method` (`"forward_difference_relative"` /
`"forward_difference_absolute"`).

### 4.2 Sobol (SALib-dependent)

**File:** `backend/app/sensitivity/sobol_analyzer.py`

Variance-based global sensitivity. Outputs:
- **S1** — first-order indices (main effect of each param)
- **ST** — total-order indices (main + interactions)
- **S2** — second-order indices (pairwise interactions, optional)

When `SALib_AVAILABLE=False` → `method="skipped"`, warnings propagated.

### 4.3 Morris (SALib-dependent)

**File:** `backend/app/sensitivity/morris_analyzer.py`

Elementary-effects screening method. Outputs:
- **mu** — mean of elementary effects (directional)
- **mu_star** — mean of absolute elementary effects (main effect, non-directional)
- **sigma** — std of elementary effects (interaction/non-linearity)

When `SALib_AVAILABLE=False` → `method="skipped"`, warnings propagated.

### 4.4 Aggregation

**`method` field:**
- `"full"` — both Sobol and Morris ran successfully
- `"local_only"` — at least one of Sobol/Morris skipped (SALib missing or failed)
- `"skipped"` — entire analyzer degraded (no params, exception)

### 4.5 Output

```python
{"v4_sensitivity_report": {
    "local_sensitivity": dict,             # {param_name: sensitivity_value}
    "sobol": {                             # None if skipped silently
        "S1": dict, "ST": dict, "S2": dict|None,
        "method": "sobol" | "skipped",
        "n_samples": int, "warnings": list
    } | None,
    "morris": {                            # None if skipped silently
        "mu": dict, "sigma": dict, "mu_star": dict,
        "method": "morris" | "skipped",
        "n_trajectories": int, "warnings": list
    } | None,
    "method": "full" | "local_only" | "skipped",
    "salib_available": bool,
    "warnings": list[str],
    "agent_version": "v4.0",
    "fallback": bool                       # True if entire analyzer degraded
}}
```

**Task B.2 dual-write:** the hook also writes
`v4_state.validation.sensitivity_report` via `set_v4_state`.

---

## 5. SSE Event Flow to Frontend

Validation results reach the Next.js frontend through two channels: the
**`v4_state` container** (Task B.2 dual-write, polled by the workbench store)
and **SSE events** streamed from `/api/chat`.

### 5.1 State container path (primary)

```
validation_pyramid_hook_node
   ├── state["v4_validation_report"]            ← flat v4 field
   └── v4_state["validation"]["report"]         ← nested container (Task B.2)

calibration_hook_node
   ├── state["v4_calibration_result"]
   └── v4_state["validation"]["calibration_result"]

sensitivity_hook_node
   ├── state["v4_sensitivity_report"]
   └── v4_state["validation"]["sensitivity_report"]
```

The frontend Zustand store (`useWorkbenchStore`) hydrates from the final
`v4_state` snapshot returned in the SSE `end` event and renders the
**Validation pane** (4-pane Workbench, pane 3):
- Validation Pyramid (L1–L5 pass/fail badges + drill-down)
- Hypothesis panel (L5 evidence + low_confidence flag)
- Evidence & Warnings (constraint violations, benchmark diffs, CI warnings)

### 5.2 SSE event path (clarification short-circuit)

When `overall_pass=False`, `ValidationAgent.build_clarification_signal(report)`
constructs a `pending_clarification` dict that the Supervisor picks up,
emitting the standard clarification SSE sequence:

```
backend                                          frontend
─────────                                        ────────
validation_pyramid_hook_node
  → state["pending_clarification"] = {
      context: "validation_failed",
      question: "v4 Validation Pyramid 验证失败：...",
      options: [continue, stop],
      failed_levels: [...],
      validation_report_summary: {level1..level5: bool}
    }
supervisor
  → emits clarification_needed SSE ──────────────► ingestSSEEvent
                                                   → ClarificationDialog
                                                     (user picks continue/stop)
user response ──── clarification_resolved SSE ───► supervisor resumes
```

**Clarification context:** `"validation_failed"`
**Options:**
- `continue` — keep the `low_confidence` marker and proceed to `worker_report`
- `stop` — terminate the current simulation (no report generated)

**10-min timeout:** If the user does not respond within
`_CLARIFICATION_TIMEOUT_SECONDS = 600`, the clarification auto-cancels and the
run proceeds with `low_confidence`.

### 5.3 Benchmark SSE flow (separate endpoint)

Benchmark runs invoked from the frontend `/benchmarks` page use a dedicated
endpoint `POST /api/v4/benchmarks/run` (see `backend/app/main.py`), which
streams its own SSE event sequence (per-benchmark `benchmark_start` /
`benchmark_result` / `benchmark_suite_complete`) and writes results to
`v4_state.validation.report.level4` for the Validation pane to consume.

---

## 6. Feature Flag Summary

| Flag | Gates | Default | Effect when off |
|------|-------|---------|-----------------|
| `V4_VALIDATION_PYRAMID_ENABLED` | L1–L5 hooks, `ValidationAgent` | `false` | All 5 level hooks return `{}`; v3 `worker_validator` runs unchanged |
| `V4_CALIBRATION_AGENT_ENABLED` | Calibration hook + Sensitivity hook (shared, spec.md §461) | `false` | Both hooks return `{}`; no parameter fitting or sensitivity analysis |
| `V4_HYPOTHESIS_AGENT_ENABLED` | P6 Hypothesis Agent (Level 5 input) | `false` | Level 5 auto-skips with `pass=True` (no hypotheses to validate) |
| `V4_DYNAMIC_ROUTING_ENABLED` | Dynamic Router (reachability path for Calibration/Sensitivity hooks) | `false` | Calibration/Sensitivity hooks unreachable (not wired into main graph) |

> **Coarse → fine resolution:** See `docs/ARCHITECTURE.md` §4 for the full
> 3-coarse → 13-fine flag hierarchy and `_resolve_v4_flag` priority rules.

---

## 7. Dependency Isolation

| Dependency | Used by | Graceful degradation |
|------------|---------|----------------------|
| `roadrunner` | L2 Track A | Falls back to Track B structural similarity; `ROADRUNNER_AVAILABLE` flag |
| `lmfit` | Calibration fitter, CI estimator | Falls back to `scipy.optimize.least_squares` + bootstrap CI; `LMFIT_AVAILABLE` flag |
| `SALib` | Sobol + Morris analyzers | Skipped (`method="skipped"`); local sensitivity still runs; `SALIB_AVAILABLE` flag |
| `numpy` | scipy backend, residual math | Imported lazily inside methods; absence falls back to pure-Python paths |

All imports are `try/except` at module load (`backend/app/config.py`) and
guarded by `*_AVAILABLE` booleans, so a missing optional dependency never
crashes the pyramid — it only degrades a single level or method.

---

## 8. Cross-References

- `docs/ARCHITECTURE.md` §3 — `v4_state.validation` container schema
- `docs/ARCHITECTURE.md` §4 — Feature flag hierarchy and resolution rules
- `docs/ARCHITECTURE.md` §5 — LangGraph node sequence (Pyramid hook after `worker_ode`)
- `docs/ARCHITECTURE.md` §6 — SSE event flow and worker→event mapping
- `docs/PATHWAYS.md` §1 — 10-pathway catalog with benchmark BIOMD IDs and pass criteria
- `docs/PATHWAYS.md` §4 — Cross-talk matrix and shared species (Level 3 inputs)
- `docs/REACTION_IR.md` §1 — `Constraint` schema (Level 1 input)
- `docs/UX_FLOW.md` §3 — Node 15 (`_validation_pyramid_hook`) full I/O table
- `spec.md` Part 4 (§276-346) — Validation Pyramid, Calibration, Sensitivity spec
