# Hypothesis Layer (Phase 6 / v4)

The Hypothesis Layer is the top tier of the v4 Validation Pyramid (Level 5).
It converts simulation results + scientific features + sensitivity analysis
into **falsifiable scientific hypotheses**, each packaged with a concrete
wet-lab experiment design and a Popper-style falsifiability verdict.

This document covers the backend agent pipeline
(`backend/app/hypothesis/`) and the frontend panel
(`frontend/components/hypothesis/`).

---

## 1. Overview

`HypothesisAgent` is the main coordinator (spec.md Part 5, lines 352-358).
It runs as a LangGraph hook node (`hypothesis_agent_hook_node`) gated by the
`V4_HYPOTHESIS_AGENT_ENABLED` feature flag. When the flag is `false` the hook
returns `{}` and touches nothing — the v3 pipeline is unaffected.

Pipeline (`HypothesisAgent.generate`):

1. **Validation gate** — only runs when `v4_validation_report.overall_pass=True`
   (Validation failure short-circuits to an empty list, never enters Hypothesis).
2. **Generate candidates** — `HypothesisGenerator` emits candidate hypotheses
   from metrics / feature_metadata / sensitivity_report / pathway_class.
   Three strategies: `oscillation`, `bistability`, `sensitivity`.
3. **Experiment design** (Task 6.2) — `ExperimentDesigner` fills
   `experiment_design` for each hypothesis via the P4 Specialist
   Perturbation Module.
4. **Falsifiability check** (Task 6.3) — `FalsificationChecker` re-derives the
   `falsifiable` flag using Karl Popper's 3 rules; non-falsifiable hypotheses
   are filtered out.
5. **Parameter robustness** (Task 6.4) — `ParameterExplorer` validates that
   the prediction holds under parameter perturbation.
6. **Literature search** — reuses `rag_client.RagClient.search_params` to fill
   `supporting_pmids` / `contradicting_pmids`, cross-checked against the
   P5 SBML Grounding Ledger (grounded PMIDs get a `|grounded` suffix).
7. **Return** `v4_hypothesis_list`.

Iron rules (铁律):

- Never mutate any v3 field — only consumes `v4_*` fields + `metrics` /
  `feature_metadata`.
- Failure degrades to an empty list + warning; the report pipeline is never
  blocked.
- Read-only on the P5 Validation Report.

---

## 2. Hypothesis Data Structure

Each item in `v4_hypothesis_list` is a dict with the following fields
(mirrored in `frontend/components/hypothesis/types.ts` as `Hypothesis`).
All fields are optional because every sub-component degrades gracefully.

| Field                     | Type                         | Source                         | Description                                                          |
|---------------------------|------------------------------|--------------------------------|----------------------------------------------------------------------|
| `id`                      | `str`                        | Generator                      | Stable hypothesis identifier.                                        |
| `hypothesis_id`           | `str`                        | Generator                      | Alias of `id` (frontend accepts either).                            |
| `statement`               | `str`                        | Generator                      | Natural-language hypothesis statement.                              |
| `prediction`              | `str`                        | Generator                      | Testable prediction, e.g. "knockdown of X reduces Y oscillation amplitude >50%". |
| `experiment_design`       | `dict` (6-field)             | ExperimentDesigner (Task 6.2)  | Wet-lab validation design — see §3.                                  |
| `validation_method`       | `str`                        | Generator                      | How the hypothesis is validated (e.g. "oscillation amplitude comparison"). |
| `expected_result`         | `str`                        | Generator / ExperimentDesigner | Expected experimental outcome.                                       |
| `falsifiable`             | `bool`                       | FalsificationChecker (Task 6.3)| Popper 3-rule verdict.                                               |
| `falsification_criteria`  | `str`                        | FalsificationChecker           | Human-readable criteria under which the hypothesis is falsified.     |
| `supporting_pmids`        | `list[str]`                  | RAG literature search          | PMIDs whose text supports the prediction; grounded ones get `\|grounded`. |
| `contradicting_pmids`     | `list[str]`                  | RAG literature search          | PMIDs whose text contradicts the prediction.                        |
| `falsifying_pmids`        | `list[str]`                  | RAG literature search          | Alias of `contradicting_pmids` (Level 5 compat).                    |
| `strategy`                | `"oscillation" \| "bistability" \| "sensitivity"` | Generator | Generation strategy that produced this hypothesis.                  |
| `target_species`          | `str`                        | Generator                      | Primary species the hypothesis targets.                             |
| `feedback_node`           | `str`                        | Generator                      | Feedback-loop node (oscillation/bistability strategies).            |
| `threshold_node`          | `str`                        | Generator                      | Bistability threshold node.                                         |
| `target_param`            | `str`                        | Generator                      | Sensitivity strategy parameter name.                                |
| `sensitivity`             | `float`                      | Generator                      | Normalized sensitivity coefficient for the target_param.            |
| `pathway_class`           | `str`                        | state                          | Pathway class the hypothesis belongs to.                            |
| `parameter_robustness`    | `dict`                       | ParameterExplorer (Task 6.4)   | Robustness analysis result.                                         |
| `confidence`              | `float` (0-1)                | Generator                      | Optional confidence score; otherwise derived client-side.           |

---

## 3. Experiment Designer (Task 6.2)

`ExperimentDesigner.design(hypothesis, state) -> dict` produces a 6-field
`experiment_design`. The frontend `ExperimentCard` renders these as
`cell_line` / `treatment` / `measurement` / `timepoint` / `control` /
`expected_outcome`.

### 3.1 `cell_line`

Default cell line selected from the pathway class via `_PATHWAY_CELL_LINE_MAP`:

| Pathway class     | Cell line |
|-------------------|-----------|
| `EGFR_RTK`        | A431      |
| `MAPK`            | A375      |
| `PI3K_AKT_MTOR`   | MCF7      |
| `P53`             | MCF7      |
| `NF_KB`           | HEK293    |
| `WNT`             | HEK293T   |
| `TGF_BETA`        | HaCaT     |
| `JAK_STAT`        | HELA      |
| `APOPTOSIS`       | HELA      |
| `CELL_CYCLE`      | HELA      |

Fallback: `HEK293`. Multi-pathway classes (`MULTI:EGFR_RTK+PI3K_AKT_MTOR`)
match on the first hit.

### 3.2 `perturbation` (treatment)

`{type, agent, target, dose, duration, mechanism, description}`.

Selection priority:

1. Match `target_species` / `feedback_node` / `threshold_node` / species
   extracted from `target_param` against the P4 Specialist
   `apply_perturbation()` candidate list.
2. Strategy preference — `bistability` → knockout; `oscillation`/`sensitivity`
   → drug.
3. First candidate from the Specialist.
4. If the Specialist is unavailable, a degraded default is built
   (`siRNA-<target>` for bistability, `anti-<target> inhibitor` otherwise).

Normalized types: `drug` (IC50 dose), `knockout` (complete KO, 48-72h
CRISPR/Cas9), `inhibition` (10 µM).

### 3.3 `readout` (measurement)

`{species, metric, threshold}`.

- `species` — `target_species` → `readout_species` → `feedback_node` →
  `threshold_node`.
- `metric` — `oscillation_amplitude` (oscillation) /
  `on_off_ratio` (bistability) / `peak` (sensitivity).
- `threshold` — extracted from `prediction` text via the regex
  `>?\s*(\d+(?:\.\d+)?)\s*%` (e.g. ">50%" → `0.5`); defaults per strategy.

### 3.4 `time_points` (timepoint)

Sampling scheme in minutes, chosen by strategy:

- Default: `[0, 5, 15, 30, 60, 120]`
- Oscillation (dense, captures multiple cycles):
  `[0, 15, 30, 45, 60, 90, 120, 180, 240]`
- Bistability (long, captures ON/OFF switching):
  `[0, 30, 60, 120, 240, 480, 720, 1440]`

### 3.5 `controls` (control)

Always includes `vehicle` and `untreated`. `drug` perturbations add `DMSO`;
`knockout` perturbations add `scramble siRNA`. De-duplicated, order-preserving.

### 3.6 `expected_result` (expected_outcome)

Priority: `hypothesis.expected_result` → `hypothesis.prediction` → default
template "实验组与对照组相比，readout 出现显著差异（p<0.05）".

On any failure the designer returns a `_minimal_design` so the hypothesis is
never left without a design.

---

## 4. Falsifiability Indicator (Task 6.3)

`FalsificationChecker.check(hypothesis) -> dict` applies Karl Popper's three
rules. A hypothesis is `falsifiable=True` only if **all three** pass.

| Rule | Check                                                                                                                  |
|------|------------------------------------------------------------------------------------------------------------------------|
| 1    | **Testable prediction** — `prediction` must contain a direction word (下降/升高/消除/失去/出现…) **and** a quantitative threshold (`%`/倍/fold), or an explicit state change (消除振荡 / 失去双稳态). Vague terms (会变化 / 可能改变 / 有所反应 …) fail the rule. |
| 2    | **Control group** — `experiment_design.controls` must be a non-empty list containing at least one standard control (`vehicle` / `untreated` / `dmso` / `scramble`). |
| 3    | **Quantitative threshold** — any of: `readout.threshold > 0`, `prediction` matches `%`/倍/fold/`>N`, or `expected_result` matches the same. |

Output:

```json
{
  "falsifiable": true,
  "falsification_criteria": "假设可证伪：prediction='...'; 若实验组与对照组相比未达到该阈值或方向相反，则假设被证伪。",
  "failure_reasons": []
}
```

`FalsificationChecker.filter(hypotheses)` drops non-falsifiable hypotheses
from the list (spec.md line 379). On exception the checker degrades to
`falsifiable=True` (conservative — never accidentally kills a hypothesis).

The frontend (`types.ts → deriveFalsifiability`) re-derives the three
per-rule booleans client-side because the backend only persists
`falsifiable` + `falsification_criteria`. The `HypothesisCard` renders the
three rules as a checklist ("Testable prediction / Control group /
Quantitative threshold").

---

## 5. SSE Events

Two v4 SSE events flow through the existing `/api/chat` stream (the v3
contract). They are emitted by `backend/app/main.py::_v3_event_stream` when
the Hypothesis Agent hook writes `v4_hypothesis_generated=True` to the
LangGraph state, and ingested by `frontend/lib/store.ts::ingestSSEEvent`.

### 5.1 `v4_hypothesis_list`

Direct list payload. Emitted by the v3→v4 adapter forwarding path.

- **event**: `"v4_hypothesis_list"`
- **data**: `Hypothesis[]` (the full list, replaces the store's
  `hypothesisList`).

### 5.2 `v4_hypothesis_generated`

Emitted from `main.py` when the Hypothesis hook completes:

```python
yield _sse_event({
    "event": "v4_hypothesis_generated",
    "data": {
        "hypothesis_count": <int>,
        "hypothesis_list": <Hypothesis[]>,
    },
})
```

The store accepts several shapes for robustness:
- `data` is an array → replaces `hypothesisList`.
- `data.v4_hypothesis_list` is an array → replaces `hypothesisList`.
- `data.hypotheses` is an array → replaces `hypothesisList`.
- `data.hypothesis` is a single object → appended to `hypothesisList`.

### Store hydration (`ingestSSEEvent`)

```ts
case "v4_hypothesis_list": {
  if (Array.isArray(eventData)) set({ hypothesisList: eventData });
  break;
}
case "v4_hypothesis_generated": { /* shape-tolerant unwrap above */ break; }
```

---

## 6. Frontend — HypothesisPanel

`frontend/components/hypothesis/HypothesisPanel.tsx` is the right-pane panel
(Task C.8). It subscribes to `useWorkbenchStore`'s `hypothesisList` and
renders one `HypothesisCard` per hypothesis (first card expanded by default).

Overall status badge:
- **generated** (emerald) — `hypothesisList.length > 0`
- **pending** (blue, pulsing dot while streaming) — list empty, stream running

Empty state shows a spinner ("Generating hypotheses…") while streaming, or a
dormant lightbulb ("Run a simulation to generate hypotheses") otherwise.

### 6.1 HypothesisCard — 5 accordion sections

Each card collapses to a summary header (id + strategy badge + falsifiability
badge + statement + confidence bar). When expanded it reveals 5 sections
(`HypothesisCard.tsx`):

1. **Hypothesis** (`Lightbulb` icon) — `statement`, plus inline metadata:
   `target_species`, `feedback_node`, `threshold_node`, `target_param`,
   `|sensitivity|`, and `validation_method`.
2. **Evidence** (`FileText`) — two PMID lists:
   - Supporting (emerald) — PMIDs from `supporting_pmids`; grounded PMIDs
     (SBML ledger cross-check) show a `·g` suffix and a tooltip.
   - Contradicting (red) — PMIDs from `contradicting_pmids` /
     `falsifying_pmids`.
   Each PMID links to `pubmed.ncbi.nlm.nih.gov/<pmid>/`.
3. **Predictions** (`TrendingUp`) — `prediction` text + `expected_result`.
4. **Suggested Experiments** (`FlaskConical`) — renders `ExperimentCard`
   with the 6-field design (`cell_line` / `treatment` / `measurement` /
   `timepoint` / `control` / `expected_outcome`) plus an "Export" button
   that copies the design JSON to the clipboard for ELN hand-off.
5. **Falsifiability** (`ShieldCheck`) — the Popper 3-rule checklist
   (`FalsifiabilityChecklist`): Testable prediction / Control group /
   Quantitative threshold, each with a ✓/✕, plus the `falsification_criteria`
   text and any `failure_reasons`.

### 6.2 Strategy badges

| Strategy      | Color   |
|---------------|---------|
| `oscillation` | purple  |
| `bistability` | amber   |
| `sensitivity` | cyan    |

### 6.3 Confidence bar

`deriveConfidence(h)` (in `types.ts`) returns a 0-1 score: uses backend
`confidence` if present, otherwise estimates from falsifiability (+0.2),
supporting-PMID count (+0.1 each, cap 2), experiment design presence (+0.1),
and validation method (+0.05). Rendered as a colored bar
(emerald ≥70%, yellow ≥40%, red otherwise).
