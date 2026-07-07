# BioDynamics Agent — Scientific Modeling IDE for Signal Pathway Simulation

> Turn a free-text biological hypothesis into a calibrated, validated ODE
> simulation of a cancer signaling pathway — in one workbench.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![Next.js](https://img.shields.io/badge/Next.js-16-black)](https://nextjs.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.x-green)](https://langchain-ai.github.io/langgraph/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.139-009688)](https://fastapi.tiangolo.com/)

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

The v4 RC release layers a **Scientific Modeling IDE** on top of the v3
Supervisor–Worker pipeline. It ships **10 cancer-pathway Specialists** (EGFR,
MAPK, PI3K/AKT/mTOR, Wnt, p53, NF-κB, JAK-STAT, TGF-β, Apoptosis, Cell Cycle),
a structured **Reaction IR v2** intermediate representation with 17 mechanism
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

### Backend setup

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

### Frontend setup

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
│  /api/chat            v3 Supervisor–Worker workflow (compiled_workflow_v3)        │
│  /api/v4/benchmarks/run   10-pathway Official Benchmark Suite (Task E.1)          │
│  /api/admin/*         vector DB update, RAG status, model status                  │
│                                                                                   │
│  v3 spine: pre_router → supervisor → {worker_mcp, worker_mechanism,               │
│             worker_rag, worker_pkpd, worker_ode, worker_sandbox,                  │
│             worker_validator, worker_report} → supervisor                         │
│                                                                                   │
│  v4 hook chains (gated by 3 coarse flags, see Configuration):                    │
│   • P1 Ontology hook        → pre_router                                          │
│   • P6 Dynamic Router hook  → pre_router                                           │
│   • P4 Pathway Planner → Specialist → Cross-talk   after worker_mechanism         │
│   • P5 SBML Grounder → Validation Pyramid          after worker_ode               │
│   • P6 Hypothesis Agent                            before worker_report           │
│                                                                                   │
│  Storage: ChromaDB (4 collections: mechanism/parameter/experiment/evidence)       │
│  RAG: query rewrite + hybrid retrieval + multi-provider rerank (xfyun/OR/SF)      │
└──────────────────────────────────────────────────────────────────────────────────┘
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full module matrix,
state management, and SSE event flow.

---

## Key Features

- **10 cancer-pathway Specialists** — EGFR, MAPK, PI3K/AKT/mTOR, Wnt, p53,
  NF-κB, JAK-STAT, TGF-β, Apoptosis, Cell Cycle. Each ships core/feedback/
  cross-talk/perturbation/validation modules and is auto-registered via
  `@register_specialist`.
- **Reaction IR v2** — a Pydantic-validated intermediate representation with
  17 mechanism types, 5 compartments, state machines, composite reactions,
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
- **3-tier feature flags** — coarse `V4_SCIENTIFIC_LAYER_ENABLED` /
  `V4_VALIDATION_ENABLED` / `V4_HYPOTHESIS_ENABLED` flags; all off = v3
  behavior.
- **Multi-provider LLM/Embedding/Rerank** — primary + backup LLM with
  automatic failover; embedding (OpenAI / local / OpenRouter / SiliconFlow /
  Xfyun MaaS); rerank (Xfyun MaaS > OpenRouter > SiliconFlow).
- **Human-in-the-loop clarification** — Supervisor triggers clarification
  when parameters are missing, KG has cycles, or PK/PD modeling choice
  diverges; 10-minute timeout auto-cancel.

---

## Configuration

Production exposes **3 coarse feature flags** (see `backend/.env.example`).
13 fine-grained flags remain as internal debug overrides (env injection) and
take precedence over the coarse flags when set explicitly.

| Coarse flag | Covers (fine flags) | Effect when `true` |
|---|---|---|
| `V4_SCIENTIFIC_LAYER_ENABLED` | P1 Ontology, P2 Reaction IR + Adapter, P3 Pathway Graph + ODE Template v2, P4 Pathway Planner + Specialist + Cross-talk Coordinator | All P1–P4 hooks fire; Specialists load; Reaction IR v2 built from `network_json` |
| `V4_VALIDATION_ENABLED` | P5 SBML Grounder, Validation Pyramid (5 levels), Calibration Agent, Sensitivity Analyzer | SBML Grounder builds ODE↔Reaction↔SBML↔Parameter↔PMID chain; Validation Pyramid runs L1–L5 |
| `V4_HYPOTHESIS_ENABLED` | P6 Hypothesis Agent, Dynamic Router | Hypothesis Agent emits `v4_hypothesis_list`; Dynamic Router records 13-agent dispatches |

All three default to `false` — equivalent to v3 behavior (no v4 hook fires).

LLM / RAG / sandbox / MCP settings are documented inline in
`backend/.env.example`.

---

## Project Structure

```
bio-dynamics-agent/
├── backend/
│   ├── app/
│   │   ├── main.py                  FastAPI app, /api/chat SSE, /api/v4/benchmarks/run
│   │   ├── graph_v3.py              LangGraph Supervisor–Worker + v4 hook chains
│   │   ├── state.py                 BioDynamicsState TypedDict + v4_state container
│   │   ├── config.py                Settings + 3 coarse / 13 fine feature flags
│   │   ├── nodes.py / nodes_v2.py   v1/v2 pipeline nodes
│   │   ├── supervisor.py            v2 supervisor registry (legacy compat)
│   │   ├── reaction_ir.py           v3 Reaction IR (KG → reaction graph)
│   │   ├── ode_renderer_v2.py       v4 ODE renderer (Reaction IR → ODE code)
│   │   ├── ontology/                P1 Ontology Agent + SBO terms + pathway registry
│   │   ├── reaction_ir_v2/          P2 Reaction IR v2 (schema, builder, composite, state machine)
│   │   ├── adapters/                v3↔v4 Adapter registry (v3_v4 / v4_v3)
│   │   ├── pathway_graph/           P3 Pathway Graph builder + initializer
│   │   ├── ode_templates_v2/        P3 v4 Jinja2 templates (oscillatory_feedback, ...)
│   │   ├── pathways/                P4 Pathway Planner + Specialist base + 10 specialists
│   │   ├── crosstalk/               P4 Cross-talk Coordinator (shared species sync)
│   │   ├── sbml_grounder/           P5 SBML Grounder Agent (5-level mapping)
│   │   ├── validation_v2/           P5 Validation Pyramid (level1–level5 + thresholds)
│   │   ├── calibration/             P5 Calibration Agent (least squares + CI)
│   │   ├── sensitivity/             P5 Sensitivity (local + Sobol + Morris)
│   │   ├── hypothesis/              P6 Hypothesis Agent + falsifiability checker
│   │   ├── agent_orchestration/     P6 Dynamic Router + pathway class dispatcher
│   │   ├── agents_v4/               P6 13-agent registry (mechanism/ode/param/sim builder)
│   │   ├── ode_templates/           v3 Jinja2 templates (Simple/Cascade/Combination/...)
│   │   └── ...                      sandbox, sbml_parser, biomodels_client, rag_client, etc.
│   ├── benchmarks/                  10 pathway YAML benchmark specs (Task E.1)
│   ├── data/                        raw SBML (BIOMD*), vector DB, metrics, sandbox logs
│   ├── .env.example
│   └── requirements.txt
├── frontend/
│   ├── app/
│   │   ├── page.tsx                 landing page
│   │   ├── workspace/page.tsx       4-pane Scientific IDE
│   │   ├── benchmarks/page.tsx      benchmark suite runner
│   │   └── report/[id]/page.tsx     per-sim report viewer
│   ├── components/                  WorkbenchShell, panels, AI Assistant, charts
│   ├── package.json                 Next.js 16, React 19, Zustand, Recharts, XyFlow
│   └── ...
├── docs/                            ARCHITECTURE / PATHWAYS / REACTION_IR / VALIDATION / UX_FLOW
├── ARCHITECTURE.md                  (legacy audit docs)
├── LICENSE                          MIT
└── README.md                        this file
```

---

## License

MIT. See [`LICENSE`](./LICENSE).
