# BioDynamics Agent v4 — Reaction IR v2

> Task F.1 (batch 1) — documentation only. Source of truth:
> `backend/app/reaction_ir_v2/` (schema, mechanism_types, builder,
> composite_reaction, state_machine, constraints, validation_rules) and
> `backend/app/ode_renderer_v2.py`.

Reaction IR v2 is the structured intermediate representation that sits
between the LLM-parsed Knowledge Graph (KG) and the Jinja2 ODE templates. It
replaces the v3 flat `network_json` (just `nodes` + `edges` with
`interaction` strings) with a fully-typed Pydantic model that captures
species ontology, 17 mechanism types, kinetics, modifiers, compartments,
state machines, composite reactions, and constraints. **No LLM is in the
loop** between KG and ODE — the IR is rule-built and template-rendered.

This document covers: (1) the schema, (2) the 17 mechanism types, (3)
composite reactions, (4) how IR maps to ODE equations, (5) a worked
EGFR example.

---

## 1. Reaction IR v2 Schema

Defined in `app/reaction_ir_v2/schema.py`. All models use Pydantic v2 for
JSON serialization + runtime validation. The top-level container is
`ReactionIRv2`, which holds 6 component lists + metadata.

### 1.1 Top-level container: `ReactionIRv2`

```python
class ReactionIRv2(BaseModel):
    species: list[SpeciesV2]              # 1. Species
    reactions: list[ReactionV2]           # 2. Reactions
    composite_reactions: list[CompositeReaction]  # 3. Composite reactions
    state_machines: list[StateMachine]    # 4. State machines
    compartments: list[Compartment]       # 5. Compartments
    constraints: list[Constraint]         # 6. Constraints
    version: str = "v4.0"
    source: str = "v4_native"             # v4_native / v3_downgraded / sbml_parsed
    warnings: list[str] = []
```

Helpers: `species_by_id(id)`, `species_by_name(name)`, `reaction_by_id(id)`,
`reactions_for_species(species_id)`, `to_dict()`, `from_dict(data)`.

State storage: `state["v4_reaction_ir"]` = `ReactionIRv2.to_dict()` output.
Also dual-written to `state["v4_state"]["reaction_ir"]["ir"]` per Task B.2.

### 1.2 Component 1 — `SpeciesV2`

Replaces v3's flat `node` dict. Each species carries ontology refs,
compartment, initial concentration, and full provenance.

```python
class SpeciesV2(BaseModel):
    id: str                                  # "SP_001"
    canonical_name: str                      # "EGFR"
    display_name: str = ""                   # "EGF Receptor"
    ontology: OntologyRef                    # HGNC, UniProt, ChEBI, GO, SBO
    species_type: str = "protein"            # ligand/receptor/kinase/phosphatase/
                                             #   adapter/gtpase/transcription_factor/
                                             #   complex/mrna/destruction_complex/...
    state_machine: str | None = None         # ID of associated StateMachine
    compartment: str = "cytoplasm"           # extracellular/membrane/cytoplasm/nucleus/mitochondria
    initial_concentration: float = 0.0       # nM
    concentration_unit: str = "nM"           # nM / molecule_per_cell
    source_sbml: str | None = None
    source_pmid: str | None = None
    source_uniprot: str | None = None
```

`OntologyRef` carries `hgnc_id`, `uniprot_id`, `chebi_id`, `go_terms`,
`sbo_term`, and a `verified` flag (false in v3-downgrade mode when API
verification is unavailable).

### 1.3 Component 2 — `ReactionV2`

Replaces v3's flat `edge` dict. Captures the full biochemical reaction:
reactants, products, modifiers (enzymes), mechanism, kinetics, compartment,
parameters, and provenance.

```python
class ReactionV2(BaseModel):
    id: str                                  # "RXN_001"
    reaction_type: str                       # one of 17 MechanismType values
    kinetics_type: str = "mass_action"       # mass_action/Michaelis_Menten/Hill/Boolean/hybrid
    reactants: list[SpeciesRef]              # substrate species with stoichiometry + role
    products: list[SpeciesRef]               # product species
    modifiers: list[Modifier]                # enzymes / catalysts / allosteric modulators
    compartments: list[str] = []             # involved compartments
    parameter_context: str = ""              # "EGF-EGFR binding kon/koff"
    pathway_tag: str = ""                    # "EGFR_RTK"
    provenance: Provenance                   # source_sbml_reaction / pmid / kegg
    constraints: list[Constraint] = []       # per-reaction constraints
```

`SpeciesRef` = `{species_id, stoichiometry=1, role=substrate|product|enzyme|cofactor}`.
`Modifier` = `{species_id, modifier_type=catalytic|allosteric|inhibitory|activating, site?}`.

### 1.4 Component 3 — `CompositeReaction`

Ordered collection of sub-reactions expressing a multi-step coupled
mechanism. The canonical example is the Wnt destruction complex (3-step
sequential coupling: formation → β-catenin binding → ubiquitination).

```python
class CompositeReaction(BaseModel):
    id: str                                  # "CR_WNT_DESTRUCTION"
    name: str                                # "Wnt destruction complex"
    sub_reactions: list[str]                 # ordered RXN_* IDs
    coupling_type: str = "sequential"        # sequential/branched/cyclic
    intermediate_species: list[str] = []     # SP_* IDs of intermediates
    net_reaction: str = ""                   # "β-catenin → ∅"
```

Composite reactions are **not** flattened into a single reaction — the
intermediate species and sub-reaction semantics are preserved for ODE
rendering and validation.

### 1.5 Component 4 — `StateMachine`

Expresses the discrete states a protein can occupy (e.g. EGFR: monomer →
EGF-bound → dimer → phosphorylated dimer → Grb2-bound → internalized →
degraded). Used to validate reaction consistency and to drive ODE template
selection.

```python
class StateMachine(BaseModel):
    id: str                                  # "EGFR_STATE_MACHINE"
    species: str                             # "EGFR"
    states: list[State]                      # {name, species_id, is_initial}
    transitions: list[Transition]            # {from_state, to_state, reaction_id, trigger}
```

`Transition.trigger` is one of `ligand_binding`, `phosphorylation`,
`internalization`, `degradation`, etc.

### 1.6 Component 5 — `Compartment`

Cellular compartment with volume ratio and transport reactions. Compartment
names are restricted to: `extracellular`, `membrane`, `cytoplasm`,
`nucleus`, `mitochondria`.

```python
class Compartment(BaseModel):
    name: str                                # 5 allowed values
    size: float = 1.0                        # volume ratio (cytoplasm=0.5, nucleus=0.1, ...)
    transport_reactions: list[str] = []      # RXN_* IDs of transport reactions
```

### 1.7 Component 6 — `Constraint`

Declarative constraint schema for validation. Constraint types:
`mass_conservation`, `steady_state`, `non_negative`, `enzymatic`,
`thermodynamic`. Each constraint has scope (`species`/`reaction`/`pathway`/
`global`), an expression string, a tolerance, and provenance.

```python
class Constraint(BaseModel):
    type: str                                # 5 allowed values
    scope: str = "species"                   # species/reaction/pathway/global
    expression: str                          # "EGFR + pEGFR + EGF_EGFR = EGFR_total"
    tolerance: float = 0.05                  # 5%
    provenance: str = ""                     # "Schoeberl 2002"
```

Constraints are consumed by Level 1 Internal Validator — see
[`docs/VALIDATION.md`](./VALIDATION.md).

---

## 2. The 17 Reaction Mechanism Types

Defined in `app/reaction_ir_v2/mechanism_types.py::MechanismType` (str Enum)
and `app/ontology/sbo_terms.py`. Each mechanism has a corresponding SBO term
and a default kinetics type. The 17 types are grouped into 6 biological
categories:

### 2.1 Modification (3)

| Mechanism | Enum | SBO term | Default kinetics | Notes |
|---|---|---|---|---|
| Phosphorylation | `PHOSPHORYLATION` | SBO:0000216 | Michaelis_Menten (forced) | Kinase-substrate. **Auditor §3.1 fix**: enzymatic, never degrades to mass-action. |
| Dephosphorylation | `DEPHOSPHORYLATION` | SBO:0000330 | Michaelis_Menten (forced) | Phosphatase-substrate. Also enzymatic. |
| Ubiquitination | `UBIQUITINATION` | SBO:0000211 | mass_action (E3-dependent) | E3 ligase tags substrate for proteasomal degradation. |

### 2.2 Binding / Assembly (5)

| Mechanism | Enum | SBO term | Default kinetics |
|---|---|---|---|
| Binding | `BINDING` | SBO:0000177 | mass_action (reversible, kon/koff) |
| Dissociation | `DISSOCIATION` | SBO:0000179 | first_order |
| Dimerization | `DIMERIZATION` | SBO:0000526 | mass_action (dimer-specific) |
| Complex formation | `COMPLEX_FORMATION` | SBO:0000526 | mass_action (multi-component) |
| Sequestration | `SEQUESTRATION` | SBO:0000177 | mass_action (masking binding) |

### 2.3 Cleavage / Exchange (2)

| Mechanism | Enum | SBO term | Default kinetics |
|---|---|---|---|
| Cleavage | `CLEAVAGE` | SBO:0000178 | Michaelis_Menten (forced) — Caspase, Notch NICD release |
| GTP/GDP exchange | `GTP_GDP_EXCHANGE` | SBO:0000333 | mass_action (GEF/GAP catalytic) |

### 2.4 Gene Expression (2)

| Mechanism | Enum | SBO term | Default kinetics |
|---|---|---|---|
| Transcription | `TRANSCRIPTION` | SBO:0000183 | Hill (TF concentration → mRNA) |
| Translation | `TRANSLATION` | SBO:0000184 | first_order (mRNA-dependent) |

### 2.5 Transport (3)

| Mechanism | Enum | SBO term | Default kinetics |
|---|---|---|---|
| Nuclear import | `NUCLEAR_IMPORT` | SBO:0000186 | first_order (cargo-specific) |
| Nuclear export | `NUCLEAR_EXPORT` | SBO:0000185 | first_order (cargo-specific) |
| Cytoplasm translocation | `CYTOPLASM_TRANSLOCATION` | SBO:0000186 | first_order |

### 2.6 Degradation (2)

| Mechanism | Enum | SBO term | Default kinetics |
|---|---|---|---|
| Degradation | `DEGRADATION` | SBO:0000179 | first_order (spontaneous protein) |
| Proteasomal degradation | `PROTEASOMAL_DEGRADATION` | SBO:0000179 | mass_action (ubiquitin-dependent) |

### 2.7 Regulation (2)

| Mechanism | Enum | SBO term | Default kinetics |
|---|---|---|---|
| Inhibition | `INHIBITION` | SBO:0000169 | hybrid (Emax or mass_action depending on context) |
| Activation | `ACTIVATION` | SBO:0000170 | hybrid |

### 2.8 Helper functions

- `is_enzymatic_mechanism(name)` — true for `phosphorylation`,
  `dephosphorylation`, `cleavage` (forces Michaelis-Menten, prevents
  mass-action degradation).
- `is_transport_mechanism(name)` — true for the 3 transport mechanisms
  (drives cross-compartment validation rule).
- `is_degradation_mechanism(name)` — true for both degradation types.
- `get_mechanism_category(name)` — returns the 6 category names.
- `v3_interaction_to_mechanism(interaction)` — maps v3 `network_json.edges[].interaction`
  (`activation`, `inhibition`, `phosphorylation`, ...) to v4 MechanismType.
  Used by the v3→v4 Adapter. Note: v3 `activation` maps to v4 `ACTIVATION`
  (not forced to `PHOSPHORYLATION`) — this is the **Auditor §4.2 fix**.

### 2.9 Kinetics normalization

`_normalize_kinetics_name(raw)` in `mechanism_types.py` normalizes P1's
fine-grained kinetics names (e.g. `michaelis_menten_irreversible`,
`mass_action_reversible`, `first_order_mrna`) to the 5 values accepted by
`ReactionV2.kinetics_type`:

- `mass_action` (covers `mass_action*`, `first_order*`)
- `Michaelis_Menten`
- `Hill`
- `Boolean`
- `hybrid` (from `mixed`)

---

## 3. Composite Reaction Support

`app/reaction_ir_v2/composite_reaction.py::CompositeReactionBuilder` is a
fluent builder for assembling `CompositeReaction` objects.

### 3.1 Builder API

```python
builder = CompositeReactionBuilder("CR_WNT_DESTRUCTION", "Wnt destruction complex")
builder.set_coupling("sequential")
builder.add_sub_reaction("RXN_DC_FORMATION")        # Axin+APC+GSK3β+CK1 → DC
builder.add_sub_reaction("RXN_DC_BC3ATENIN_BINDING") # DC + β-catenin → DC_βcatenin
builder.add_sub_reaction("RXN_DC_BC3ATENIN_UBIQ")   # DC_βcatenin → DC + ub_βcatenin
builder.add_intermediate("SP_DC_BC3ATENIN")
builder.set_net_reaction("β-catenin → ∅")
cr = builder.build()
```

### 3.2 Coupling types

- **`sequential`** — sub-reactions execute in order; intermediates flow
  between them. Default for destruction complex, apoptosome assembly.
- **`branched`** — sub-reactions diverge from a common precursor (e.g.
  Casp3 → Casp6 + PARP cleavage).
- **`cyclic`** — sub-reactions form a loop (e.g. Cdc20-APC/C self-amplifying
  loop).

### 3.3 Design rules

1. Composite reactions are **not flattened** — intermediate species and
   sub-reaction semantics are preserved.
2. `sub_reactions` is **ordered** (sequential coupling assumes order).
3. `net_reaction` describes the overall stoichiometric effect.
4. Intermediate species are explicitly listed in `intermediate_species` so
   that mass-conservation checks can exclude them from steady-state invariants.

### 3.4 ODE rendering

The ODE renderer treats each sub-reaction as a separate ODE term, then
collapses intermediates that appear only as both product and reactant within
the composite (mass-conserving collapse). This preserves the dynamic
behavior of the multi-step mechanism without inflating the ODE system with
explicit intermediate variables.

---

## 4. How Reaction IR Maps to ODE Equations

`app/ode_renderer_v2.py::ODERendererV2` consumes `ReactionIRv2` +
`PathwayGraph` and produces executable Python ODE code via Jinja2 templates
in `app/ode_templates_v2/`.

### 4.1 Data flow

```
user input
  → LLM → network_json (v3)
  → Adapter v3_to_v4 → ReactionIRv2 (P2)
  → PathwayGraphBuilder → PathwayGraph (P3, adds feedback_loops, cross_talk_edges, temporal)
  → ODERendererV2.render(pathway_class, reaction_ir, pathway_graph)
  → Jinja2 template (e.g. oscillatory_feedback.j2)
  → Python ODE code (string)
  → sandbox.py execute_simulation_code_v2(code)
  → simulation.csv + image_base64
```

### 4.2 Template selection

`_PATHWAY_TEMPLATE_MAP` in `ode_renderer_v2.py` routes `pathway_class` →
template file. All 10 specialist pathway classes plus initializer pathway
keys are covered (case-insensitive):

| pathway_class | Template | Why |
|---|---|---|
| `EGFR_RTK`, `MAPK_ERK`, `PI3K_AKT_MTOR` | `oscillatory_feedback.j2` | Phosphorylation cascades. Template contains `_ode_rhs` phosphorylation branch (Michaelis-Menten). |
| `P53`, `P53_SIGNALING`, `NF_KB` | `transcriptional_delay.j2` | DDE pulse oscillators (60–120 min transcriptional delay). |
| `JAK_STAT`, `TGF_BETA` | `transcription_factor.j2` | TF + nuclear translocation, single pulse. |
| `WNT` | `destruction_complex.j2` | Composite destruction-complex reaction. |
| `APOPTOSIS` | `caspase_cascade.j2` | Bistable MOMP switch + caspase cascade. |
| `CELL_CYCLE` | `cyclin_cdk_toggle.j2` | Cyclin-CDK toggle + Rb-E2F bistability. |

Unknown `pathway_class` falls back to `oscillatory_feedback.j2` (safest —
supports all mechanism types + DDE downgrade).

### 4.3 Per-mechanism ODE term mapping

Each `ReactionV2.reaction_type` maps to a Jinja2 fragment that emits a
specific ODE term:

| Mechanism | ODE term generated |
|---|---|
| `binding` (mass_action, reversible) | `d[AB]/dt = kon*A*B - koff*AB`; `dA/dt = -kon*A*B + koff*AB` |
| `dissociation` | `d[A]/dt = k_diss*AB`; `d[AB]/dt = -k_diss*AB` |
| `phosphorylation` (MM, enzymatic) | `d[pX]/dt = Vmax*X / (Km + X)` (catalytic modifier forces MM) |
| `dephosphorylation` (MM) | `d[X]/dt = Vmax*pX / (Km + pX)` |
| `ubiquitination` (E3-dependent) | `d[ubX]/dt = k_cat*E3*X` |
| `proteasomal_degradation` | `d[X]/dt = -k_deg*ubX` |
| `transcription` (Hill) | `d[mRNA]/dt = Vmax * TF^n / (K^n + TF^n) - k_deg*mRNA` |
| `translation` (first-order) | `d[Protein]/dt = k_tl*mRNA - k_deg*Protein` |
| `nuclear_import` / `nuclear_export` | `d[X_nuc]/dt = k_imp*X_cyt - k_exp*X_nuc` |
| `gtp_gdp_exchange` (GEF/GAP) | `d[RasGTP]/dt = k_gef*GEF*RasGDP - k_gap*GAP*RasGTP` |
| `inhibition` (drug/small molecule) | `d[X_active]/dt = -k_inh*Drug*X_active` (Emax if PK/PD profile present) |
| `activation` | Pathway-specific (typically phosphorylation or expression up-regulation) |

### 4.4 DDE handling

When `PathwayGraph.temporal.requires_dde = true` (p53, NF-κB), the renderer
includes the `_dde_helpers.j2` fragment and emits
`y_delayed = ddeint(rhs, t, y_history, tau_delay_minutes)` instead of
`scipy.integrate.solve_ivp`. The `dde_delay_minutes` value comes from
`PathwayGraph.temporal.dde_delay_minutes`.

### 4.5 Constraint injection

Reaction IR `constraints` are not directly rendered into ODE code; they're
consumed by Level 1 Internal Validator after simulation to verify that the
ODE solution respects them (mass conservation, non-negative, etc.).

---

## 5. Example: EGFR → Reaction IR → ODE

This is the canonical worked example. Source: `egfr_specialist.py` core
module + Levchenko 2000 (BIOMD0000000022).

### 5.1 User input

> "EGF stimulation leads to EGFR phosphorylation and MAPK cascade activation"

### 5.2 Network JSON (v3, after worker_mechanism)

```json
{
  "nodes": [
    {"id": "EGF", "name": "EGF", "type": "ligand"},
    {"id": "EGFR", "name": "EGFR", "type": "receptor"},
    {"id": "pEGFR", "name": "pEGFR", "type": "protein"},
    {"id": "RasGTP", "name": "RasGTP", "type": "protein"}
  ],
  "edges": [
    {"source": "EGF", "target": "EGFR", "interaction": "binding"},
    {"source": "EGFR", "target": "pEGFR", "interaction": "phosphorylation"},
    {"source": "pEGFR", "target": "RasGTP", "interaction": "activation"}
  ]
}
```

### 5.3 Adapter v3_to_v4 → ReactionIRv2

`app/adapters/v3_v4_adapter.py` walks `network_json.edges`, applies
`v3_interaction_to_mechanism()` per edge, and builds `ReactionV2` objects:

```python
{
  "species": [
    {"id": "SP_001", "canonical_name": "EGF",
     "species_type": "ligand", "compartment": "extracellular",
     "initial_concentration": 100.0, "ontology": {"verified": false}},
    {"id": "SP_002", "canonical_name": "EGFR",
     "species_type": "receptor", "compartment": "membrane",
     "initial_concentration": 100.0},
    {"id": "SP_003", "canonical_name": "pEGFR",
     "species_type": "protein", "compartment": "membrane",
     "initial_concentration": 0.0},
    {"id": "SP_004", "canonical_name": "RasGTP",
     "species_type": "gtpase", "compartment": "membrane",
     "initial_concentration": 0.0, "shared": true}
  ],
  "reactions": [
    {
      "id": "RXN_001",
      "reaction_type": "binding",
      "kinetics_type": "mass_action",
      "reactants": [{"species_id": "SP_001", "role": "substrate"},
                    {"species_id": "SP_002", "role": "substrate"}],
      "products":   [{"species_id": "SP_001", "stoichiometry": 1, "role": "product"},
                     {"species_id": "SP_002", "stoichiometry": 1, "role": "product"}],
      "modifiers": [],
      "compartments": ["extracellular", "membrane"],
      "parameter_context": "EGF-EGFR binding kon/koff",
      "pathway_tag": "EGFR_RTK"
    },
    {
      "id": "RXN_002",
      "reaction_type": "phosphorylation",
      "kinetics_type": "Michaelis_Menten",
      "reactants": [{"species_id": "SP_002", "role": "substrate"}],
      "products":   [{"species_id": "SP_003", "role": "product"}],
      "modifiers":  [{"species_id": "SP_002", "modifier_type": "catalytic"}],
      "compartments": ["membrane"],
      "parameter_context": "EGFR autophosphorylation Vmax/Km",
      "pathway_tag": "EGFR_RTK"
    },
    {
      "id": "RXN_003",
      "reaction_type": "gtp_gdp_exchange",
      "kinetics_type": "mass_action",
      "reactants": [{"species_id": "SP_003", "role": "enzyme"}],
      "products":   [{"species_id": "SP_004", "role": "product"}],
      "modifiers":  [{"species_id": "SP_003", "modifier_type": "catalytic"}],
      "compartments": ["membrane"],
      "pathway_tag": "EGFR_RTK"
    }
  ],
  "compartments": [
    {"name": "extracellular", "size": 1.0},
    {"name": "membrane",      "size": 0.1},
    {"name": "cytoplasm",     "size": 0.5}
  ],
  "constraints": [
    {"type": "mass_conservation", "scope": "pathway",
     "expression": "EGFR + pEGFR + EGF_EGFR = EGFR_total",
     "tolerance": 0.05, "provenance": "Levchenko 2000"}
  ],
  "version": "v4.0",
  "source": "v3_downgraded",
  "warnings": ["ontology not verified (v3 downgraded mode)"]
}
```

### 5.4 PathwayGraph (P3)

`PathwayGraphBuilder.build(pathway_class="EGFR_RTK", reaction_ir=...)` adds:

- `feedback_loops`: `[]` (EGFR core has no feedback — feedback enters via MAPK)
- `cross_talk_edges`: `[{"source_pathway":"EGFR_RTK","target_pathway":"MAPK_ERK","shared_species":["RasGTP","pRaf"]}]`
- `temporal`: `{"requires_dde": false, "t_end_minutes": 120.0}`

### 5.5 ODE rendering

`ODERendererV2.render(pathway_class="EGFR_RTK", reaction_ir, pathway_graph)`
selects `oscillatory_feedback.j2` and renders:

```python
import numpy as np
from scipy.integrate import solve_ivp

# Species: EGF, EGFR, pEGFR, RasGTP
species_names = ["EGF", "EGFR", "pEGFR", "RasGTP"]
y0 = [100.0, 100.0, 0.0, 0.0]
t_end = 120.0
n_eval = 200

# Parameters (from RAG / SBML grounding)
params = {
  "EGF":     {"kon": 1.0e6, "koff": 0.6},
  "EGFR":    {"Vmax": 1.0e5, "Km": 50.0},
  "RasGTP":  {"k_gef": 0.1, "k_gap": 0.01}
}

def _ode_rhs(t, y):
    EGF, EGFR, pEGFR, RasGTP = y
    # RXN_001: EGF + EGFR binding (mass_action, reversible)
    # (simplified — full template includes EGF_EGFR complex)
    dEGF_dt = -params["EGF"]["kon"] * EGF * EGFR + params["EGF"]["koff"] * (100.0 - EGFR)
    dEGFR_dt = -params["EGF"]["kon"] * EGF * EGFR + params["EGF"]["koff"] * (100.0 - EGFR)
    # RXN_002: EGFR autophosphorylation (Michaelis-Menten, EGFR catalyzes itself)
    d_pEGFR_dt = (params["EGFR"]["Vmax"] * EGFR) / (params["EGFR"]["Km"] + EGFR)
    dEGFR_dt -= d_pEGFR_dt
    # RXN_003: pEGFR → RasGTP (GEF-catalyzed, mass_action)
    dRasGTP_dt = params["RasGTP"]["k_gef"] * pEGFR * (1.0 - RasGTP/100.0)
    return [dEGF_dt, dEGFR_dt, d_pEGFR_dt, dRasGTP_dt]

t_eval = np.linspace(0, t_end, n_eval)
sol = solve_ivp(_ode_rhs, [0, t_end], y0, t_eval=t_eval, method="LSODA")
# ... CSV + image export follows
```

### 5.6 Sandbox execution

`sandbox.execute_simulation_code_v2(code)` runs the rendered Python in an
isolated process with timeout + AST pre-flight check, captures `stdout`,
`simulation.csv`, and a base64 PNG plot. Outputs are written to state:
`execution_result`, `simulation_csv_path`, `image_base64`, and the parsed
markers `ic50`/`ic90`/`hed`/`dose_response_data` if present in stdout.

### 5.7 Validation chain

After sandbox, the worker_validator + v4 SBML Grounder + Validation Pyramid
hooks fire (when their flags are on):

- L1 Internal: mass conservation check on `EGFR + pEGFR + EGF_EGFR = EGFR_total`
  (5% tolerance from `constraints[0]`).
- L2 SBML: `BIOMD0000000022` loaded via BioModels API, simulated with
  roadrunner (Track A) or structural similarity (Track B), pEGFR peak time
  compared to template output.
- L3 Cross-talk: shared `RasGTP` with MAPK Specialist checked for sync
  consistency.
- L4 Benchmark: `pass_criteria` from `egfr_signaling.yaml` — pEGFR peak
  5–10 min, MAPK amplification 10–100×, EGFR internalization t½ 10–15 min.
- L5 Hypothesis: skipped (P6 disabled by default); `pass=True` per spec.

See [`docs/VALIDATION.md`](./VALIDATION.md) for full Pyramid details.
