# BioDynamics Agent: Codex Repository Instructions

These instructions apply to the whole repository. Before a non-trivial change,
read `PROJECT_STATE.json`, then the relevant reverse-engineering documents under
`docs/00_PROJECT_OVERVIEW.md` through `docs/11_NEXT_STEPS.md`. Use
`docs/CODEX_MAINTENANCE_GUIDE.md` as the compact maintenance/search guide.

## Scope And Baseline

- The application repository is this directory, `bio-dynamics-agent/`. The
  sibling `C:\Users\27553\Desktop\gzlab\backend` contains only data and is not
  the agent backend.
- Treat the checked-out working tree as the user's source of truth. It may be
  newer than `HEAD`, older than the remote, or both. Never discard, reset, or
  overwrite existing changes to make the tree match Git.
- At the 2026-07-20 inspection, the branch was `main` at `b7a2820`, the local
  tracking ref reported `ahead 10`, and the tree contained extensive tracked and
  untracked work. That is a snapshot, not a claim that the repository is current.
- Before editing, run `git status --short --branch`, `git diff --name-only`, and
  `git log -3 --oneline`. Do not fetch, pull, rebase, or switch branches unless
  the user asks for it or the task explicitly requires it.

## Start From The Real Runtime

- Backend/API entry: `backend/app/main.py` (`app.main:app`).
- Active chat graph: `backend/app/graph_v3.py::compiled_workflow_v3`.
- Shared state contract: `backend/app/state.py::BioDynamicsState`.
- Configuration and flags: `backend/app/config.py::Settings`.
- Computational nodes: `backend/app/nodes_v2.py`; selected legacy capabilities
  are still reused from `backend/app/nodes.py`.
- V4 deterministic REST endpoints: `backend/app/v4_endpoints.py`.
- Frontend product entry: `frontend/app/page.tsx` ->
  `frontend/components/minimal/MinimalApp.tsx`.
- Frontend transport/state: `frontend/lib/api.ts`, `frontend/lib/sse.ts`, and
  `frontend/lib/store.ts`.

Do not start a production-flow investigation in `backend/app/graph.py`; it is a
legacy bridge. Do not infer current behavior from `ARCHITECTURE.md` alone; that
file describes an older v2 topology.

## Reverse Engineering Index

- Architecture and ownership: `docs/01_ARCHITECTURE.md`,
  `docs/02_CODE_STRUCTURE.md`, `docs/DEPENDENCY_GRAPH.md`.
- State and scientific pipelines: `docs/03_DATA_FLOW.md` through
  `docs/07_BENCHMARK_SYSTEM.md`.
- Current risks and governance: `docs/08_KNOWN_PROBLEMS.md` through
  `docs/11_NEXT_STEPS.md`.
- Configuration and prompts: `docs/CONFIG_REFERENCE.md`,
  `docs/PROMPT_INDEX.md`.
- Stable intent: `docs/DESIGN_DECISIONS.md`, `docs/PROJECT_PHILOSOPHY.md`.

`PROJECT_STATE.json` is a static snapshot, not live telemetry. Check its
`generated_at`, `generation_kind`, Git metadata, and benchmark evidence fields
before relying on its counts or statuses.

## Required Change Discipline

- Preserve the `/api/chat` SSE wire contract unless the change deliberately
  updates backend emission, frontend parsing, Zustand ingestion, components,
  tests, and `docs/API.md` together.
- A new workflow field normally requires coordinated updates to
  `BioDynamicsState`, request initialization/reset in `main.py`, the producing
  worker or hook, SSE emission, frontend store hydration, and tests.
- V4 data is represented both by flat `v4_*` fields and the grouped `v4_state`.
  Use the helpers in `state.py`; do not update only one representation.
- Keep feature-flag behavior reversible. Resolve V4 flags through the
  `effective_v4_*` methods and SA flags through `is_sa_feature_enabled()`.
- Treat scientific output validity separately from pipeline completion. A
  successful HTTP response, rendered plot, or generated report does not prove a
  benchmark passed.
- Never expose or print `backend/.env`, API keys, credentials, or raw secret
  values. `backend/.env.example` is only a partial catalog; `config.py` is the
  configuration source of truth.
- Keep runtime artifacts out of source searches and edits: `backend/data/`,
  `backend/logs/`, `backend/_*`, `frontend/node_modules/`, `frontend/.next/`,
  `test_outputs_*`, caches, images, CSV files, and diagnostic logs.

## Frontend Rule

Before editing anything under `frontend/`, also read `frontend/AGENTS.md`. This
checkout uses Next.js 16 with changed APIs; consult the relevant local guide in
`frontend/node_modules/next/dist/docs/` before relying on remembered Next.js
conventions.

## Verification

Choose tests by the surface changed; do not begin with the real 10-pathway
benchmark for a narrow edit.

```powershell
# Backend targeted test (from repository root)
Set-Location backend
python -m pytest tests/<relevant_test>.py -q

# Backend suite
python -m pytest tests -q

# Frontend unit/build checks
Set-Location ..\frontend
npm test -- --run
npm run build

# Scientific verification suites (from repository root)
Set-Location ..\verification
python -m pytest <relevant_suite> -q
```

The local test directories are currently excluded by `.gitignore` even though
the Makefile and CI reference them. Verify that required tests are actually
tracked before treating a clean-clone or CI result as representative.
