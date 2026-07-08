# BioDynamics Agent — Scientific Modeling IDE for Signal Pathway Simulation

> Turn a free-text biological hypothesis into a calibrated, validated ODE
> simulation of a cancer signaling pathway — in one workbench.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.139-009688)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-16-black)](https://nextjs.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.x-green)](https://langchain-ai.github.io/langgraph/)
[![CI](https://img.shields.io/badge/CI-GitHub_Actions-2088FF)](./.github/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/verification-223%20passed-brightgreen)](./verification/)

---

## Overview

BioDynamics Agent is an end-to-end AI agent platform that bridges the
**"qualitative hypothesis ↔ quantitative model"** gap in computational systems
biology and translational oncology. A researcher types a mechanism or drug
hypothesis in plain English (e.g. *"Galunisertib (TGF-β inhibitor, IC50=51 nM)
blocks TGF-β signaling — model its synergy with anti-PD-1 on CD8⁺ T-cell
recovery"*). The system then closes the loop:

**Mechanism parsing → Knowledge graph → RAG parameter retrieval → PK/PD
inference → ODE generation → Sandbox simulation → Scientific feature
extraction → SBML validation → Literature evidence → Markdown prediction
report.**

The v4.1 release layers a **Scientific Modeling IDE** on top of the v3
Supervisor–Worker pipeline. It ships **10 cancer-pathway Specialists** (EGFR,
MAPK, PI3K/AKT/mTOR, Wnt, p53, NF-κB, JAK-STAT, TGF-β, Apoptosis, Cell Cycle),
a structured **Reaction IR v2** intermediate representation with 17+ mechanism
types, a **5-level Validation Pyramid** (Internal → SBML → Cross-talk →
Benchmark → Hypothesis), and a **Hypothesis Layer** that emits falsifiable
predictions with experiment designs. Progress streams to a 4-pane Next.js
workbench via Server-Sent Events.

All v4 functionality is gated behind 3 coarse feature flags. With all flags
off, the system degrades bit-for-bit to v3 behavior.

---

## Quick Start

### Prerequisites

- **Python 3.11+** (3.14 tested)
- **Node.js 20+** (Next.js 16)
- An OpenAI-compatible LLM endpoint (OpenAI, BigModel, DeepSeek, OpenRouter, etc.)

> **No API key?** The Scientific Workspace, pathway selector, benchmark
> center, and validation pyramid UI are all browsable without an LLM key. Only
> running a new simulation requires `OPENAI_API_KEY`.

### Option A — One-click (Windows)

```bat
cd bio-dynamics-agent
scripts\start-dev.bat
```

The script auto-detects `backend\.venv` or `backend\venv`, checks model
connectivity, then launches the backend (uvicorn :8000) and frontend
(Next.js :3000) in separate windows and opens the browser.

### Option B — Manual setup

#### Backend

```bash
cd bio-dynamics-agent/backend
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

pip install -r requirements.txt

copy .env.example .env          # Windows
# cp .env.example .env          # macOS/Linux
# Edit .env: set OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend

```bash
cd bio-dynamics-agent/frontend
npm install
npm run dev          # http://localhost:3000
```

### Run

1. Open the frontend at `http://localhost:3000`.
2. Click **Launch Workspace** to enter the 4-pane Scientific Modeling IDE.
3. Type a hypothesis in the AI Assistant pane; SSE events stream as the
   Supervisor–Worker pipeline executes.
4. (Optional) Visit `/benchmarks` to run the 10-pathway Official Benchmark
   Suite via `/api/v4/benchmarks/run`.

### Option C — Docker

```bash
cp backend/.env.example backend/.env   # fill in OPENAI_API_KEY
docker compose up --build
# backend  → http://localhost:8000
# frontend → http://localhost:3000
```

---

## Architecture Overview

```
┌──────────────────────── Frontend (Next.js 16 + React 19) ────────────────────────┐
│  /                landing page (pathway selector, recent sims, benchmarks)        │
│  /workspace       4-pane Scientific IDE:                                          │
│                   [Project] [Scientific Workspace] [Validation] [AI Assistant]    │
│  /benchmarks      10-pathway Official Benchmark Suite runner                      │
│  /report/[id]     per-simulation report viewer                                    │
│  State: Zustand store (useWorkbenchStore) · SSE event stream from /api/chat       │
└──────────────────────────────────────────────────────────────────────────────────┘
                  │ SSE (text/event-stream)
                  ▼
┌──────────────────────── Backend (FastAPI + LangGraph) ───────────────────────────┐
│  /api/chat                 v3+v4 LangGraph workflow (compiled_workflow_v3)       │
│  /api/v4/benchmarks/run    10-pathway Official Benchmark Suite (SSE)             │
│  /api/admin/*              vector DB update, RAG status                          │
│  /api/models/status        LLM/embedding/rerank provider health                  │
│  /api/chat/clear-memory    clear LangGraph MemorySaver                            │
│                                                                                   │
│  7-node LangGraph workflow → 5 specialized agents (supervisor.py AGENT_REGISTRY): │
│   node0  MCP term lookup      → Terminology Agent (async, app/mcp_client.py)      │
│   node1  parse network        → Mechanism Parsing Agent                           │
│   node1.5 RAG search          → Parameter Retrieval Agent (hybrid search)        │
│   node1.6 PK/PD inference     → PK/PD Agent                                       │
│   node2  generate ODE code    → ODE Engineer Agent                                │
│   node3  execute sandbox      → Simulation Engineer Agent (owns sandbox)         │
│   node4  audit & correct      → Validation Agent                                  │
│   node5  render report        → Report Agent                                      │
│                                                                                   │
│  v4 hook chains (gated by 3 coarse flags, see Configuration):                    │
│   • P1 Ontology hook        → pre_router                                          │
│   • P6 Dynamic Router hook  → pre_router                                           │
│   • P4 Pathway Planner → Specialist → Cross-talk   after worker_mechanism         │
│   • P5 SBML Grounder → Validation Pyramid          after worker_ode               │
│   • P6 Hypothesis Agent                            before worker_report           │
│                                                                                   │
│  Storage: ChromaDB (mechanism/parameter/experiment/evidence collections)          │
│  RAG: query rewrite + hybrid retrieval + multi-provider rerank (xfyun/OR/SF)      │
└──────────────────────────────────────────────────────────────────────────────────┘
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full module matrix,
state management, and SSE event flow.

---

## Key Features

- **7-node LangGraph workflow** — node0 (MCP term lookup) → node1 (mechanism
  parsing) → node1.5 (RAG parameter search) → node1.6 (PK/PD inference) →
  node2 (ODE generation) → node3 (sandbox execution) → node4 (audit & correct)
  → node5 (report). Node 2 prioritizes real parameters from RAG results; only
  falls back to estimation when parameters are missing.
- **10 cancer-pathway Specialists** — EGFR, MAPK, PI3K/AKT/mTOR, Wnt, p53,
  NF-κB, JAK-STAT, TGF-β, Apoptosis, Cell Cycle. Each ships core/feedback/
  cross-talk/perturbation/validation modules and is auto-registered via
  `@register_specialist`.
- **Reaction IR v2** — a Pydantic-validated intermediate representation with
  17+ mechanism types, 5 compartments, state machines, composite reactions,
  and constraint schema. Bridges KG → ODE templates without LLM in the loop.
- **5-level Validation Pyramid** — L1 Internal consistency (mass conservation,
  non-negative, numerical stability), L2 SBML/BioModels (roadrunner Track A
  or structural similarity Track B), L3 Cross-pathway shared-species, L4
  Benchmark (per-pathway pass criteria from `benchmarks/*.yaml`), L5
  Hypothesis falsifiability.
- **Hypothesis Layer** — generates falsifiable predictions with experiment
  designs, supporting/contradicting PMIDs, and validation methods.
- **Benchmark Suite** — 10 YAML-defined benchmark cases
  (`backend/benchmarks/*.yaml`) with literature PMIDs, expected dynamics, and
  quantitative pass criteria. Run via `POST /api/v4/benchmarks/run` (SSE).
- **4-pane Scientific IDE** — Project, Scientific Workspace, Validation, and
  collapsible AI Assistant. Streams SSE events into a Zustand store.
- **3-coarse-flag feature gates** — `V4_SCIENTIFIC_LAYER_ENABLED` /
  `V4_VALIDATION_ENABLED` / `V4_HYPOTHESIS_ENABLED`; all off = v3 behavior.
- **Multi-provider LLM/Embedding/Rerank** — primary + backup LLM with
  automatic failover; embedding (OpenAI / local / OpenRouter / SiliconFlow /
  Xfyun MaaS); rerank (Xfyun MaaS > OpenRouter > SiliconFlow).
- **MCP integration** — Node 0 looks up biomedical terminology via MCP tools
  (OpenBioMed Skills, medical-terminologies, PubMed search, UMLS). When
  `MCP_ENABLED=false` or endpoint URLs are empty, gracefully falls back to LLM
  internal knowledge — never errors.
- **Human-in-the-loop clarification** — Supervisor triggers clarification
  when parameters are missing, KG has cycles, or PK/PD modeling choice
  diverges; 10-minute timeout auto-cancel.

---

## Configuration

Production exposes **3 coarse feature flags** (see [`backend/.env.example`](backend/.env.example)).
13 fine-grained flags remain as internal debug overrides (env injection) and
take precedence over the coarse flags when set explicitly.

| Coarse flag | Covers (fine flags) | Effect when `true` |
|---|---|---|
| `V4_SCIENTIFIC_LAYER_ENABLED` | P1 Ontology, P2 Reaction IR + Adapter, P3 Pathway Graph + ODE Template v2, P4 Pathway Planner + Specialist + Cross-talk Coordinator | All P1–P4 hooks fire; Specialists load; Reaction IR v2 built from `network_json` |
| `V4_VALIDATION_ENABLED` | P5 SBML Grounder, Validation Pyramid (5 levels), Calibration Agent, Sensitivity Analyzer | SBML Grounder builds ODE↔Reaction↔SBML↔Parameter↔PMID chain; Validation Pyramid runs L1–L5 |
| `V4_HYPOTHESIS_ENABLED` | P6 Hypothesis Agent, Dynamic Router | Hypothesis Agent emits `v4_hypothesis_list`; Dynamic Router records agent dispatches |

All three default to `false` — equivalent to v3 behavior (no v4 hook fires).

LLM / embedding / rerank / RAG online fallback / MCP tool endpoints / PubMed
NCBI API key settings are documented inline in
[`backend/.env.example`](backend/.env.example).

### Optional scientific dependencies

The following are **try-imported** (see the dependency-isolation matrix in
[`app/config.py`](backend/app/config.py)); if missing, the pipeline degrades
gracefully to a simplified fallback and logs a warning:

| Package | Enables | Install |
|---|---|---|
| `python-roadrunner` | L2 SBML Validation Track A (else structural similarity) | `pip install python-roadrunner` |
| `lmfit` | Calibration Agent (else `scipy.optimize.least_squares`) | `pip install lmfit` |
| `SALib` | Sobol/Morris global sensitivity (else local sensitivity only) | `pip install SALib` |
| `lxml` | Faster SBML XML parsing in `sbml_parser_v2` (else `xml.etree`) | `pip install lxml` |

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET`  | `/` | Health check / root info |
| `POST` | `/api/chat` | Main SSE workflow (text/event-stream). Streams node progress + final report. |
| `POST` | `/api/v4/benchmarks/run` | Run the 10-pathway Official Benchmark Suite (SSE). |
| `GET`  | `/api/models/status` | LLM / embedding / rerank provider health. |
| `POST` | `/api/admin/update-vector-db` | Refresh the ChromaDB vector store. |
| `GET`  | `/api/admin/rag-status` | RAG collection counts + retrieval health. |
| `POST` | `/api/chat/clear-memory` | Clear the LangGraph MemorySaver. |
| `POST` | `/api/chat/respond` | Resume after a clarification. |
| `POST` | `/api/chat/stop` | Cancel an in-flight run. |

Full request/response schemas: [`docs/API.md`](docs/API.md).

---

## Testing

### Backend unit tests

```bash
cd bio-dynamics-agent/backend
python -m pytest tests/ -v --tb=short
```

Includes `tests/test_critical_fixes.py` (27 TDD cases covering 14 critical
issue fixes — 25 passed / 2 skipped due to `jitcdde` availability).

### Verification Suite (`verification/`)

The scientific correctness gate — 7 active sub-suites, **223 passed,
282 skipped, 0 failed** (last run 2026-07-08):

| Sub-suite | passed | skipped | failed | Purpose |
|---|---|---|---|---|
| `pathway_regression` | 51 | 54 | 0 | 10 pathways × 5 dynamics cases, literature-baselined |
| `parameter_stress` | 58 | 2 | 0 | ±10%/±30%/±50% perturbation stability |
| `ontology_validation` | 15 | 40 | 0 | SBO / Reactome / KEGGG mapping (network skips) |
| `solver_validation` | 45 | 6 | 0 | RK45/LSODA/BDF/Radau, DDE Heun, Padé, bistability |
| `benchmark` | 50 | 8 | 0 | 10 pathways × 5 cases, scientific pass criteria |
| `hypothesis_validation` | 0 | 124 | 0 | LLM-batch hypothesis checks (deferred to v4.2+) |
| `biomodels_regression` | 0 | 48 | 0 | roadrunner + v4 sim service (deferred to v4.2+) |

```bash
cd bio-dynamics-agent
python -m pytest verification/ --tb=short
```

### Frontend tests

```bash
cd bio-dynamics-agent/frontend
npm run test          # vitest — 52 spec cases (store / api / panels)
npm run test:e2e      # playwright — smoke.spec.ts (needs `npx playwright install`)
```

### Makefile shortcuts

```bash
make test              # backend unit tests
make test-integration  # integration + e2e
make test-benchmark    # scientific benchmark suite
make test-coverage     # coverage report (htmlcov/)
make ci                # full local CI pipeline
```

---

## Deployment

### Docker (recommended for production)

A multi-stage [`Dockerfile`](Dockerfile) builds backend (Python 3.11-slim) and
frontend (Node 20-slim production runtime) images;
[`docker-compose.yml`](docker-compose.yml) wires both services with
`./backend/data` volume-mounted for ChromaDB / sandbox logs / metrics
persistence. ChromaDB and Qdrant use **local file-based persistence** — no
separate database container is required.

```bash
cp backend/.env.example backend/.env   # fill in OPENAI_API_KEY
docker compose up --build
```

### CI/CD pipeline (GitHub Actions)

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) defines a 9-stage
gated pipeline on push/PR to `main`/`master`/`develop`:

1. `unit-tests` → 2. `integration-tests` → 3. `scientific-benchmarks` →
4. `biomodels-regression` → 5. `parameter-stress` → 6. `playwright-ui` →
7. `performance-benchmark` → 8. `coverage-report` (artifact upload) →
9. `release-gate` (all gates must pass before release approval).

### Demo scripts

| Script | Purpose |
|---|---|
| [`scripts/start-dev.bat`](scripts/start-dev.bat) | One-click Windows dev startup (backend + frontend + browser) |
| [`scripts/demo.sh`](scripts/demo.sh) | Single-pathway end-to-end run |
| [`scripts/demo_benchmark.sh`](scripts/demo_benchmark.sh) | Run all 10 benchmark pathways |
| [`scripts/regen_rag_db.py`](scripts/regen_rag_db.py) | Rebuild the ChromaDB vector store from BioModels XML |
| [`scripts/inspect_rag_db.py`](scripts/inspect_rag_db.py) | Inspect vector DB contents |

---

## Project Structure

```
bio-dynamics-agent/
├── backend/
│   ├── app/
│   │   ├── main.py                  FastAPI app, /api/chat SSE, /api/v4/benchmarks/run
│   │   ├── graph_v3.py              LangGraph workflow + v4 hook chains
│   │   ├── state.py                 BioDynamicsState TypedDict + v4_state container
│   │   ├── config.py                Settings + 3 coarse / 13 fine feature flags
│   │   ├── supervisor.py            AGENT_REGISTRY — 5 agents → 7 nodes
│   │   ├── nodes.py / nodes_v2.py   v1/v2 pipeline nodes
│   │   ├── reaction_ir.py           v3 Reaction IR (KG → reaction graph)
│   │   ├── ode_renderer_v2.py        v4 ODE renderer (Reaction IR → ODE code)
│   │   ├── mcp_client.py            Node 0 MCP term lookup (async)
│   │   ├── rag_client.py            Node 1.5 hybrid RAG + ClinicalTrials.gov v2
│   │   ├── ontology/                P1 Ontology Agent + SBO terms + pathway registry
│   │   ├── reaction_ir_v2/          P2 Reaction IR v2 (schema, builder, composite, state machine)
│   │   ├── adapters/                v3↔v4 Adapter registry (v3_v4 / v4_v3)
│   │   ├── pathway_graph/           P3 Pathway Graph builder + initializer
│   │   ├── ode_templates_v2/        P3 v4 Jinja2 templates (oscillatory_feedback, bistable_switch, ...)
│   │   ├── pathways/                P4 Pathway Planner + Specialist base + 10 specialists
│   │   ├── crosstalk/               P4 Cross-talk Coordinator (shared species sync)
│   │   ├── sbml_grounder/           P5 SBML Grounder Agent (5-level mapping)
│   │   ├── validation_v2/           P5 Validation Pyramid (level1–level5 + thresholds)
│   │   ├── calibration/             P5 Calibration Agent (least squares + CI)
│   │   ├── sensitivity/             P5 Sensitivity (local + Sobol + Morris)
│   │   ├── hypothesis/              P6 Hypothesis Agent + falsifiability checker
│   │   ├── agent_orchestration/     P6 Dynamic Router + pathway class dispatcher
│   │   ├── agents_v4/               P6 agent registry (mechanism/ode/param/sim builder)
│   │   ├── reliability/             circuit breaker, retry, fail-safe, structured logging
│   │   ├── solvers/                 DDE solver, numerical stability retry, oscillation/bistability detection
│   │   └── ...                      sandbox, sbml_parser, biomodels_client, kg_builder, ...
│   ├── benchmarks/                  10 pathway YAML benchmark specs (Task E.1)
│   ├── tests/                       pytest unit tests + test_critical_fixes.py
│   ├── data/                        raw SBML (BIOMD*), vector DB, metrics, sandbox logs
│   ├── .env.example
│   └── requirements.txt
├── frontend/
│   ├── app/
│   │   ├── page.tsx                 landing page
│   │   ├── workspace/page.tsx       4-pane Scientific IDE
│   │   ├── benchmarks/page.tsx      benchmark suite runner
│   │   └── report/[id]/page.tsx     per-sim report viewer
│   ├── components/                  WorkbenchShell, panels, AI Assistant, charts, XyFlow graph
│   ├── lib/                         store.ts (Zustand), api.ts, sse.ts, i18n.tsx
│   ├── __tests__/                   vitest specs
│   ├── e2e/                         playwright smoke.spec.ts
│   ├── package.json                 Next.js 16, React 19, Zustand, Recharts, XyFlow
│   └── ...
├── verification/                    scientific Verification Suite (7 sub-suites, 505 cases)
├── scripts/                         start-dev.bat, demo.sh, demo_benchmark.sh, rag_db tools
├── docs/                            ARCHITECTURE / PATHWAYS / REACTION_IR / VALIDATION / API / ...
├── .github/workflows/ci.yml         9-stage CI/CD pipeline
├── Dockerfile                       multi-stage (backend + frontend)
├── docker-compose.yml               backend:8000 + frontend:3000
├── Makefile                         test / test-integration / test-benchmark / ci shortcuts
├── LICENSE                         MIT
└── README.md                        this file
```

---

## Documentation

In-depth docs live under [`docs/`](docs/):

| Doc | Content |
|---|---|
| [`ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Module matrix, state management, SSE event flow |
| [`PATHWAYS.md`](docs/PATHWAYS.md) | 10 pathway specialists + mechanism templates |
| [`REACTION_IR.md`](docs/REACTION_IR.md) | Reaction IR v2 schema + 17 mechanism types |
| [`VALIDATION.md`](docs/VALIDATION.md) | 5-level Validation Pyramid internals |
| [`HYPOTHESIS.md`](docs/HYPOTHESIS.md) | Falsifiable hypothesis layer + experiment design |
| [`FRONTEND.md`](docs/FRONTEND.md) | 4-pane Scientific IDE component map |
| [`API.md`](docs/API.md) | REST/SSE endpoint schemas |
| [`UX_FLOW.md`](docs/UX_FLOW.md) | End-to-end UX flow + gap audit |
| [`ERROR_HANDLING.md`](docs/ERROR_HANDLING.md) | 5 recovery strategies + unified JSON logging |
| [`CHANGELOG.md`](docs/CHANGELOG.md) | Version history |
| [`NAMING_VIOLATIONS.md`](docs/NAMING_VIOLATIONS.md) | Naming-convention audit |

Release & audit reports at the repo root:
[`BioDynamics_v4_Release_Report.md`](BioDynamics_v4_Release_Report.md),
[`BioDynamics_v4_1_Reliability_Report.md`](BioDynamics_v4_1_Reliability_Report.md),
[`BioDynamics_v4_1_Comprehensive_Audit_Report.md`](BioDynamics_v4_1_Comprehensive_Audit_Report.md),
[`BioDynamics_v4_1_Improvement_Plan.md`](BioDynamics_v4_1_Improvement_Plan.md),
[`BioDynamics_v4_Issue_Backlog.md`](BioDynamics_v4_Issue_Backlog.md).

---

## Roadmap & Known Limitations

v4.1 passes the Verification Suite (223 passed / 0 failed) and validates 10
cancer-pathway dynamics against literature baselines (p53 oscillation,
NF-κB damped oscillation, Caspase bistability, Cyclin-CDK cycling, etc.).
The following are deferred to **v4.2+** (tracked in
[`BioDynamics_v4_Issue_Backlog.md`](BioDynamics_v4_Issue_Backlog.md)):

- **IB-006** — Full DDE solver (`jitcdde` dependency, Python 3.14 compat)
- **IB-009** — Reaction IR `else` branch rewrite for 8 mechanism types
- **IB-010 / IB-011** — Modifier schema (Ki/Kact/n_hill/inhibition_type, multi-site)
- **IB-012** — Numeric mass-conservation verification (currently token-only)
- **IB-016** — L4 benchmark coverage (5/10 → 10/10 pathways)
- **IB-018 / IB-020** — Calibration & Sensitivity agents (currently placeholders)
- **IB-097** — Performance test suite (17 cases, currently skeletons)
- **G1** — Hypothesis refinement loop (bounded re-simulation back-edge)
- `biomodels_regression` + `hypothesis_validation` — full LLM/roadrunner integration
- Playwright E2E browser install in CI

---

## License

MIT. See [`LICENSE`](./LICENSE).
