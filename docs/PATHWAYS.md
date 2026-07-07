# BioDynamics Agent v4 — 10-Pathway Catalog

> Task F.1 (batch 1) — documentation only. Source of truth:
> `backend/app/pathways/specialists/*_specialist.py` (10 specialists) +
> `backend/benchmarks/*.yaml` (10 benchmark specs).

This document catalogs the 10 cancer signaling pathways supported by v4 RC.
For each pathway: biological class, key species, key reactions, biological
role, the Pathway Specialist used, ODE system size, and the benchmark
BioModels / PMID reference. The final section is a cross-talk matrix showing
which pathways interact via shared species.

---

## 1. Pathway Catalog

### 1.1 EGFR RTK Signaling

| Field | Value |
|---|---|
| **pathway_class** | `EGFR_RTK` |
| **Class** | Receptor tyrosine kinase (RTK) — growth factor signaling |
| **Specialist** | `app/pathways/specialists/egfr_specialist.py` — `EGFRRtkSpecialist` |
| **ODE template** | `oscillatory_feedback.j2` (phosphorylation-cascade carrier) |
| **Benchmark YAML** | `backend/benchmarks/egfr_signaling.yaml` |
| **Benchmark source** | Levchenko 2000 (PMID:11923475) / Schoeberl 2001 (PMID:11483517) |
| **BIOMD reference** | `BIOMD0000000022` (Levchenko 2000 EGFR model) |
| **Key species** | `EGF`, `EGFR`, `pEGFR`, `Shc`, `pShc`, `Grb2`, `SOS`, `Ras`, `RasGTP`, `Raf`, `pRaf` |
| **Key reactions** | EGF+EGFR binding (mass_action) → EGFR dimerization → EGFR autophosphorylation (Michaelis-Menten) → Shc phosphorylation (hetero-phosphorylation, pEGFR catalytic) → Grb2/SOS recruitment → Ras GDP/GTP exchange (GEF catalytic) → Raf phosphorylation |
| **Biological role** | Upstream growth-factor receptor. Initiates the Ras→Raf→MEK→ERK cascade and cross-talks with PI3K/AKT/mTOR through RasGTP and with MAPK through shared RasGTP/Raf/pRaf. |
| **ODE system size** | ~11 species, ~12 reactions (no DDE) |
| **Pass criteria** | pEGFR peak 5–10 min; MAPK amplification 10–100×; EGFR internalization t½ 10–15 min; mass conservation <5% error |

### 1.2 MAPK / ERK Cascade

| Field | Value |
|---|---|
| **pathway_class** | `MAPK_ERK` |
| **Class** | Three-tier kinase cascade (downstream of RTK) |
| **Specialist** | `mapk_specialist.py` — `MAPKSpecialist` |
| **ODE template** | `oscillatory_feedback.j2` |
| **Benchmark YAML** | `backend/benchmarks/mapk_cascade.yaml` |
| **Benchmark source** | Schoeberl 2001 (PMID:11483517) / Goldbeter & Koshland 1981 (PMID:1941687) |
| **Key species** | `RasGTP`, `Raf`, `pRaf`, `MEK`, `pMEK`, `ppMEK`, `ERK`, `pERK`, `ppERK` |
| **Key reactions** | Raf phosphorylation (Michaelis-Menten) → MEK dual phosphorylation (zero-order ultrasensitivity) → ERK dual phosphorylation (zero-order ultrasensitivity) |
| **Biological role** | Signal amplification 10–100× from Raf to ERK. Classic zero-order ultrasensitivity regime (saturated enzyme). Downstream of EGFR via shared RasGTP/pRaf. |
| **ODE system size** | ~9 species, ~6 dual-phosphorylation reactions |
| **Pass criteria** | MAPK amplification 10–100×; Hill coefficient 2–10; ERK peak 2–8 min; mass conservation <5% error |

### 1.3 PI3K / AKT / mTOR

| Field | Value |
|---|---|
| **pathway_class** | `PI3K_AKT_mTOR` |
| **Class** | Lipid kinase / survival cascade |
| **Specialist** | `pi3k_akt_mtor_specialist.py` — `PI3KAKTmTORSpecialist` |
| **ODE template** | `oscillatory_feedback.j2` |
| **Benchmark YAML** | `backend/benchmarks/pi3k_akt_mtor.yaml` |
| **Benchmark source** | Mazzoletti 2009 (PMID:19211571) |
| **Key species** | `PI3K`, `PIP2`, `PIP3`, `PTEN`, `PDK1`, `AKT`, `pAKT`, `TSC2`, `pTSC2`, `Rheb`, `mTORC1`, `S6K1`, `pS6K1` |
| **Key reactions** | PI3K: PIP2→PIP3 (mass_action) ↔ PTEN: PIP3→PIP2 (mass_action); PDK1: AKT→pAKT (Michaelis-Menten); pAKT: TSC2→pTSC2 (inactivation); Rheb accumulation → mTORC1 activation → S6K1 phosphorylation |
| **Biological role** | Cell survival, growth, metabolism. Cross-talks with EGFR through PI3K recruitment by pEGFR; PIP2/PIP3 mass conservation ~1.0 is a key L1 invariant. |
| **ODE system size** | ~13 species, ~7 reactions |
| **Pass criteria** | pAKT peak 30–60 min; PIP2+PIP3 mass conservation 0.95–1.05; pS6K1 delayed 30–60 min after pAKT |

### 1.4 Wnt / β-catenin

| Field | Value |
|---|---|
| **pathway_class** | `WNT` |
| **Class** | Destruction complex + transcriptional switch |
| **Specialist** | `wnt_specialist.py` — `WntSpecialist` |
| **ODE template** | `destruction_complex.j2` |
| **Benchmark YAML** | `backend/benchmarks/wnt_signaling.yaml` |
| **Benchmark source** | Lee 2003 (PMID:12906785) / Polakis 2002 (PMID:12064617) |
| **Key species** | `Wnt`, `Frizzled`, `LRP6`, `pLRP6`, `Dvl`, `Axin`, `APC`, `GSK3b`, `CK1`, `bcatenin`, `bcatenin_nuclear`, `TCF`, `Axin2_mRNA` |
| **Key reactions** | Wnt+Frizzled+LRP6 binding → LRP6 phosphorylation → Dvl recruitment → Axin recruitment to membrane → destruction complex (Axin+APC+GSK3β+CK1) dissociation → β-catenin accumulation → nuclear translocation → TCF/LEF transcription → Axin2 mRNA (negative feedback, 30-min transcriptional delay) |
| **Biological role** | Development, stem cell self-renewal, Wnt/β-catenin often dysregulated in CRC. The destruction complex is the canonical **CompositeReaction** example (3-step coupling). |
| **ODE system size** | ~13 species, with composite destruction-complex reaction |
| **Pass criteria** | β-catenin steady-state <10 nM without Wnt; Axin2 mRNA peak 60–120 min; destruction complex intact assembly without Wnt; mass conservation <5% |

### 1.5 p53 Tumor Suppressor

| Field | Value |
|---|---|
| **pathway_class** | `p53` / `P53_SIGNALING` |
| **Class** | DDE pulse oscillator (DNA damage response) |
| **Specialist** | `p53_specialist.py` — `P53Specialist` |
| **ODE template** | `transcriptional_delay.j2` |
| **Benchmark YAML** | `backend/benchmarks/p53_signaling.yaml` |
| **Benchmark source** | Lev Bar-Or 2000 (PMID:10644692) |
| **Key species** | `ATM`, `p53`, `p53_p`, `p53_tet`, `p53_nuclear`, `Mdm2`, `Mdm2_mRNA`, `p21`, `Bax`, `PUMA` |
| **Key reactions** | DNA damage → ATM activation → p53 phosphorylation → p53 tetramerization → nuclear import → Mdm2 transcription (Hill) + translation (first-order, **60–120 min DDE delay**) → Mdm2-mediated p53 degradation (proteasomal) → **negative-feedback pulse oscillation** |
| **Biological role** | Tumor suppression. p53-Mdm2 DDE negative feedback produces 5–7 hour pulse oscillation. Cross-talks with Apoptosis via Bax/PUMA and with Cell Cycle via p21. |
| **ODE system size** | ~10 species, requires DDE solver (`dde_solver.py`) |
| **Pass criteria** | p53 pulse period 5–7 hours; Mdm2 transcription delay 60–120 min; p53 phosphorylation response 5–30 min; mass conservation <5% |

### 1.6 NF-κB Signaling

| Field | Value |
|---|---|
| **pathway_class** | `NF_KB` |
| **Class** | DDE nuclear oscillation (inflammation) |
| **Specialist** | `nf_kappa_b_specialist.py` — `NfKappaBSpecialist` |
| **ODE template** | `transcriptional_delay.j2` |
| **Benchmark YAML** | `backend/benchmarks/nfkb_signaling.yaml` |
| **Benchmark source** | Nelson 2004 (PMID:14976212) |
| **Key species** | `TNF`, `TNFR`, `IKK`, `IkBa`, `pIkBa`, `NFkB`, `NFkB_nuclear`, `IkBa_mRNA`, `A20` |
| **Key reactions** | TNF+TNFR binding → IKK activation → IκBα phosphorylation → IκBα proteasomal degradation → NF-κB release → nuclear import → IκBα transcription (Hill, 30–60 min delay) + translation → IκBα binding to NF-κB → nuclear export → **negative-feedback DDE oscillation** |
| **Biological role** | Innate immunity, inflammation, cancer cell survival. NF-κB nuclear oscillation period 1–2 hours under sustained TNF. |
| **ODE system size** | ~9 species, requires DDE solver |
| **Pass criteria** | NF-κB nuclear oscillation period 1–2 hours; IκBα mRNA delay 30–60 min; oscillation duration 6–20 hours; mass conservation <5% |

### 1.7 JAK-STAT

| Field | Value |
|---|---|
| **pathway_class** | `JAK_STAT` |
| **Class** | Transcription factor + nuclear translocation (single pulse) |
| **Specialist** | `jak_stat_specialist.py` — `JakStatSpecialist` |
| **ODE template** | `transcription_factor.j2` |
| **Benchmark YAML** | `backend/benchmarks/jak_stat.yaml` |
| **Benchmark source** | Timm 2003 / Schwartz 2003 (PMID:15286703) |
| **Key species** | `IL6`, `JAK`, `STAT5`, `pSTAT5`, `pSTAT5_nuclear`, `SOCS_mRNA`, `SOCS`, `CIS` |
| **Key reactions** | IL-6/EPO + receptor binding → JAK autophosphorylation → STAT5 tyrosine phosphorylation → pSTAT5 dimerization → nuclear import → SOCS transcription (Hill) + translation (30–60 min delay) → SOCS binds JAK (inhibition) → **single-pulse negative feedback** |
| **Biological role** | Cytokine signaling, immune cell proliferation. Unlike p53/NF-κB, JAK-STAT produces a single pulse (not sustained oscillation) due to SOCS strong negative feedback. |
| **ODE system size** | ~8 species, with nuclear import/export |
| **Pass criteria** | pSTAT5 peak 5–15 min; SOCS mRNA delayed peak 30–60 min; STAT5 nuclear/cytoplasmic ratio single pulse (non-oscillatory) |

### 1.8 TGF-β Signaling

| Field | Value |
|---|---|
| **pathway_class** | `TGF_BETA` |
| **Class** | Transcription factor + nuclear translocation (Smad) |
| **Specialist** | `tgf_beta_specialist.py` — `TgfBetaSpecialist` |
| **ODE template** | `transcription_factor.j2` |
| **Benchmark YAML** | `backend/benchmarks/tgf_beta_signaling.yaml` |
| **Benchmark source** | Clarke 2009 / Massagué 1998 (PMID:9674480) / Schmierer 2007 (PMID:17721552) |
| **Key species** | `TGFB`, `TbRI`, `TbRII`, `Smad2`, `pSmad2`, `Smad4`, `pSmad2_Smad4`, `pSmad2_Smad4_nuclear`, `SMAD7_mRNA`, `SMAD7` |
| **Key reactions** | TGF-β + TβRI/II binding → TβRI phosphorylation → Smad2 (Ser465/467) phosphorylation (Michaelis-Menten) → pSmad2:Smad4 heteromer formation → nuclear accumulation → SMAD7 transcription + translation (30–60 min delay) → SMAD7 inhibits TβRI (negative feedback) |
| **Biological role** | Tumor suppression → tumor promotion (context-dependent). Drug target: Galunisertib (TβRI inhibitor, IC50=51 nM) is the canonical use case in the README. |
| **ODE system size** | ~10 species, with nuclear import |
| **Pass criteria** | pSmad2 peak 5–15 min; pSmad2-Smad4 nuclear accumulation 15–30 min; SMAD7 mRNA delayed 30–60 min |

### 1.9 Apoptosis (Intrinsic + Extrinsic)

| Field | Value |
|---|---|
| **pathway_class** | `APOPTOSIS` |
| **Class** | Bistable caspase cascade (point-of-no-return) |
| **Specialist** | `apoptosis_specialist.py` — `ApoptosisSpecialist` |
| **ODE template** | `caspase_cascade.j2` (with MOMP bistability detection) |
| **Benchmark YAML** | `backend/benchmarks/apoptosis.yaml` |
| **Benchmark source** | Rehm 2006 / Green & Kroemer 2004 (PMID:15241432) |
| **Key species** | `BH3`, `Bax`, `Bak`, `MOMP`, `Cyt_c`, `Apaf1`, `Casp9`, `Casp3`, `Casp8`, `tBid`, `DISC`, `Casp6`, `XIAP`, `PARP` |
| **Key reactions** | Intrinsic: BH3-only activation → Bax/Bak oligomerization → MOMP (bistable) → Cyt c release → Apaf-1 + Casp9 apoptosome → Casp3 activation (bistable). Extrinsic: death receptor → DISC → Casp8 → Casp3 → Casp6 → PARP cleavage |
| **Biological role** | Programmed cell death. MOMP is the **point-of-no-return** with bistable all-or-none behavior. Cross-talks with p53 (p53 induces Bax/PUMA). |
| **ODE system size** | ~14 species, with bistable MOMP switch |
| **Pass criteria** | Cyt c release precedes Casp3 by 5–15 min; MOMP bistable all-or-none; Casp3 activation threshold 0.1–0.5; mass conservation <5% |

### 1.10 Cell Cycle Regulation

| Field | Value |
|---|---|
| **pathway_class** | `CELL_CYCLE` |
| **Class** | Cyclin-CDK toggle + delayed negative feedback oscillator |
| **Specialist** | `cell_cycle_specialist.py` — `CellCycleSpecialist` |
| **ODE template** | `cyclin_cdk_toggle.j2` |
| **Benchmark YAML** | `backend/benchmarks/cell_cycle.yaml` |
| **Benchmark source** | Tyson 1991 / Pomerening 2005 (PMID:11389814) / Yao 2008 (PMID:12064617) |
| **Key species** | `CyclinB`, `Cdk1`, `CyclinB_Cdk1`, `APC_C`, `Cdc20`, `Securin`, `Separase`, `CyclinD1`, `Cdk4`, `Rb`, `pRb`, `E2F` |
| **Key reactions** | CyclinB-CDK1 binding → active CyclinB_Cdk1 → Cdc20 activation → APC/C activation → Securin degradation → Separase release + CyclinB degradation (8–12h delayed negative feedback oscillation). G1/S: CyclinD1+Cdk4 → Rb phosphorylation (bistable) → E2F release |
| **Biological role** | Cell proliferation control. Cross-talks with p53 (p21 inhibits Cyclin-CDK) and with Wnt (β-catenin induces Cyclin D1). |
| **ODE system size** | ~12 species, with Cyclin-CDK toggle + Rb-E2F bistability |
| **Pass criteria** | CyclinB-APC/C oscillation period 8–12 hours; Rb-E2F bistable G1/S switch; Cyclin D1 peak 60–240 min in G1; mass conservation <5% |

---

## 2. Specialist Module Template

Every specialist extends `PathwaySpecialistBase`
(`app/pathways/pathway_specialist_base.py`) and ships 5 modules
(`app/pathways/pathway_modules/`):

| Module | File | Purpose |
|---|---|---|
| Core | `core/template.py` | Defines `_CORE_SPECIES` (with `species_type`, `compartment`, `shared` flags) and `_CORE_REACTIONS` (mechanism, kinetics, reactants, products, modifiers). |
| Feedback | `feedback/template.py` | Pathway-specific feedback loops (e.g. SOCS→JAK, Mdm2→p53, Axin2→destruction complex). |
| Crosstalk | `crosstalk/template.py` | Returns cross-talk Reaction fragments **from this pathway's side**. Cross-talk Coordinator merges them. |
| Perturbation | `perturbation/template.py` | Drug/ligand perturbations (e.g. Galunisertib inhibits TβRI, PD0325901 inhibits MEK). |
| Validation | `validation/template.py` | Per-pathway `_VALIDATION_RULES` consumed by L4 Benchmark Validator. |

All specialists self-register via `@register_specialist` decorator
(`app/pathways/pathway_registry.py`), keyed by `pathway_class` (e.g.
`"EGFR_RTK"`). `get_specialist(pathway_class)` returns a fresh instance per
call (no cross-request state pollution).

---

## 3. Benchmark YAML Schema

Each `backend/benchmarks/<pathway>.yaml` follows this schema:

```yaml
pathway_class: EGFR_RTK              # matches specialist registry key
name: "EGFR RTK Signaling"
input:
  hypothesis: "<free-text hypothesis>"
  pathway_class: EGFR_RTK
  species: [...]                     # initial species list
  duration: 120                      # minutes
ground_truth:
  source: "Levchenko 2000 / Schoeberl 2001"
  pmid: "PMID:11923475"
  key_findings:                      # literature-known biological facts
    - "pEGFR peaks at 5-10 min after EGF stimulation"
    - "MAPK signal amplification 10-100x from Ras to ERK"
expected_dynamics:                   # qualitative dynamics for L4
  - species: "pEGFR"
    behavior: "peak"
    timing: "5-10 min"
    threshold: ">0"
validation:                          # which Validation Pyramid levels apply
  level1_internal: true
  level2_sbml: true
  level3_crosstalk: true
  level4_benchmark: true
pass_criteria:                       # quantitative L4 pass criteria
  - criterion: "pEGFR peak occurs between 5-10 min"
    metric_name: "pEGFR_peak_time"
    expected_min: 5.0
    expected_max: 10.0
    tolerance: 2.0
    unit: "minutes"
performance:                         # L4 performance budget
  max_runtime_seconds: 60
  max_memory_mb: 500
```

10 benchmark files: `apoptosis.yaml`, `cell_cycle.yaml`,
`egfr_signaling.yaml`, `jak_stat.yaml`, `mapk_cascade.yaml`,
`nfkb_signaling.yaml`, `p53_signaling.yaml`, `pi3k_akt_mtor.yaml`,
`tgf_beta_signaling.yaml`, `wnt_signaling.yaml`.

The runner (`app/benchmark_runner.py::BenchmarkRunner`) iterates all 10
sequentially. The HTTP endpoint `POST /api/v4/benchmarks/run` streams
progress as SSE.

---

## 4. Cross-talk Matrix

Cross-talk is computed by `app/crosstalk/coordinator.py` from each
specialist's `crosstalk` module. Shared species are auto-detected by species
name overlap between specialist outputs and consumed by Level 3 Cross-Pathway
validation.

### 4.1 Pairwise interactions

| Pathway (row) ↓ \ (col) → | EGFR | MAPK | PI3K | Wnt | p53 | NF-κB | JAK-STAT | TGF-β | Apoptosis | Cell Cycle |
|---|---|---|---|---|---|---|---|---|---|---|
| **EGFR** | — | RasGTP, Raf, pRaf | PI3K (recruited by pEGFR) | — | — | — | — | — | — | CyclinD1 induction |
| **MAPK** | RasGTP, pRaf (upstream) | — | — | — | — | — | — | — | — | ERK→CyclinD1 |
| **PI3K** | pEGFR (recruits PI3K) | — | — | — | AKT→p53 inhibition | — | — | — | AKT→Casp9 inhibition | AKT→CyclinD1 |
| **Wnt** | — | — | — | — | — | — | — | — | — | β-catenin→CyclinD1 |
| **p53** | — | — | AKT (inhibits p53) | — | — | — | — | — | Bax, PUMA | p21→CDK inhibition |
| **NF-κB** | — | — | — | — | — | — | — | — | — | — |
| **JAK-STAT** | — | — | — | — | — | — | — | — | — | — |
| **TGF-β** | — | — | — | — | — | — | — | — | — | p15/p21→CDK inhibition |
| **Apoptosis** | — | — | — | — | Bax/PUMA (p53-induced) | — | — | — | — | — |
| **Cell Cycle** | — | ERK→CyclinD1 | — | β-catenin→CyclinD1 | p21→CDK | — | — | p15/p21→CDK | — | — |

### 4.2 Canonical shared species

The Cross-talk Coordinator auto-detects shared species by matching
`canonical_name` across specialist outputs. The most common shared species
written to `state["v4_shared_species"]`:

- **`RasGTP`** — EGFR ↔ MAPK (RasGTP is the upstream activator of Raf)
- **`pRaf`** — EGFR ↔ MAPK (Raf phosphorylation shared node)
- **`PI3K`** — EGFR ↔ PI3K/AKT/mTOR (pEGFR recruits PI3K to membrane)
- **`AKT` / `pAKT`** — PI3K ↔ p53 (AKT inhibits p53) ↔ Apoptosis (AKT
  inhibits Casp9) ↔ Cell Cycle (AKT induces CyclinD1)
- **`p53` / `p21`** — p53 ↔ Cell Cycle (p21 inhibits Cyclin-CDK) ↔
  Apoptosis (p53 induces Bax/PUMA)
- **`Bax` / `PUMA`** — p53 ↔ Apoptosis
- **`CyclinD1`** — EGFR/MAPK/PI3K/Wnt ↔ Cell Cycle (induced by ERK, AKT,
  β-catenin)
- **`β-catenin`** — Wnt ↔ Cell Cycle (induces CyclinD1)

### 4.3 Cross-talk Coordinator behavior

`app/crosstalk/coordinator.py::crosstalk_coordinator_hook_node`:

1. Reads all `v4_specialist_outputs` (one entry per detected pathway class).
2. For each pair `(pathway_A, pathway_B)`, scans species name overlap to
   detect shared species.
3. Builds `v4_crosstalk_edges` (list of `CrossTalkEdge`):
   `{id, source_pathway, target_pathway, source_node, target_node,
     mechanism, shared_species}`.
4. Builds `v4_shared_species_sync` — canonical variable mapping so that the
   same ODE variable (e.g. `RasGTP`) is shared across pathway ODEs in
   multi-pathway simulations.
5. Computes `v4_time_scale_alignment` — unified `max_step` across pathways
   (e.g. EGFR second-scale + p53 hour-scale must be reconciled).

Multi-pathway detection: `v4_pathway_class` is `"MULTI:EGFR_RTK+PI3K_AKT_mTOR"`
when more than one pathway is identified. The Coordinator then activates;
single-pathway cases produce empty cross-talk output and Level 3 validation
is `skipped pass=True`.

---

## 5. Pathway Planner

`app/pathways/pathway_planner.py::pathway_planner_hook_node` runs after
`worker_mechanism` (when `effective_v4_pathway_planner_enabled()` is true).
It uses the keyword-based `app/ontology/pathway_registry.py` (PATHWAY_REGISTRY
with curated keywords for each of the 10 classes) to identify which
pathway(s) the user input is about, then writes:

- `v4_pathway_class` — single class (`"EGFR_RTK"`) or
  `"MULTI:EGFR_RTK+PI3K_AKT_mTOR"` for multi-pathway, or `"UNKNOWN"`.
- Pre-identified cross-talk edges (passed to Coordinator).

The Planner output drives Specialist dispatch (Pathway Specialist Hook calls
`get_specialist(pathway_class)` for each detected class) and the Dynamic
Router (P6).
