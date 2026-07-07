# API Reference (v3 + v4)

Backend entry point: `backend/app/main.py` (FastAPI, `title="BioDynamics
Agent"`, `version="0.5.0"`). CORS allows `settings.FRONTEND_URL`. All
endpoints are served from this single app — v3 routes keep the `/api`
prefix (the legacy contract that must NOT be version-bumped), v4 routes use
the `/api/v4` prefix.

SSE wire format: every event is a single `data:` line whose payload is a
JSON object `{"event": "<name>", "data": <value>}`, terminated by a blank
line. The frontend (`lib/sse.ts`) parses each frame and forwards
`{event, data}` to the store.

---

## 1. REST Endpoints

### 1.1 Health

#### `GET /`
Health check.

- **Request body**: none
- **Response**: `200`
  ```json
  { "status": "ok", "service": "BioDynamics Agent", "version": "v3" }
  ```

---

## 2. v3 Endpoints (legacy contract — `/api` prefix)

### 2.1 `POST /api/admin/update-vector-db`

Trigger a background ChromaDB vector-db rebuild over `data/raw/`.

- **Request body**: none (empty)
- **Response**: `200`
  ```json
  { "status": "started", "message": "知识库更新已启动，后台处理中..." }
  ```
- The actual rebuild runs in a `BackgroundTasks` worker thread; the response
  returns immediately.

### 2.2 `GET /api/admin/rag-status`

Return the RAG knowledge-base status.

- **Request body**: none
- **Response**: `200` `RagStatusData`
  ```json
  {
    "databases": [
      { "name": "PubMed", "type": "online_api", "collection": "biodynamics_params" },
      { "name": "KEGG", "type": "online_api", "collection": "biodynamics_mechanism" },
      { "name": "Reactome", "type": "online_api", "collection": "biodynamics_mechanism" },
      { "name": "UniProt", "type": "online_api", "collection": "biodynamics_mechanism" },
      { "name": "ChEMBL", "type": "online_api", "collection": "biodynamics_parameter" },
      { "name": "BioModels (SBML)", "type": "local_file", "collection": "biodynamics_parameter" },
      { "name": "ClinicalTrials.gov", "type": "online_api", "collection": null }
    ],
    "collections": {
      "mechanism": <int>,
      "parameter": <int>,
      "experiment": <int>,
      "evidence": <int>,
      "legacy_params": <int>
    },
    "online_fallback_enabled": <bool>,
    "online_fallback_threshold": <float>
  }
  ```
- User-imported files in `data/raw/` (`.txt/.md/.json/.xml/.sbml/.csv`) are
  appended to `databases` as a `user_import` entry.

### 2.3 `GET /api/models/status`

Return the active LLM / embedding / rerank provider configuration.

- **Request body**: none
- **Response**: `200` `ModelStatusData`
  ```json
  {
    "llm":      { "provider": "...", "model": "...", "base_url": "..." },
    "backup_llm": { "provider": "...", "model": "...", "base_url": "..." } | null,
    "embedding": { "provider": "...", "model": "..." },
    "rerank": {
      "provider": "...",
      "selection_mode": "...",
      "provider_priority": ["..."],
      "candidates": [{ "provider": "...", "model": "...", "display_name": "..." }]
    }
  }
  ```

### 2.4 `POST /api/chat/clear-memory`

Clear the LangGraph MemorySaver short-term memory for a thread and clean up
clarification events.

- **Request body**: `ClearMemoryRequest`
  ```json
  { "thread_id": "<string>" }
  ```
- **Response**: `200`
  ```json
  { "status": "ok", "thread_id": "<string>", "message": "短期记忆已清空" }
  ```

### 2.5 `POST /api/chat/respond`

Submit a human-in-the-loop clarification answer and wake the
`clarification_node`.

- **Request body**: `ClarificationResponseRequest`
  ```json
  {
    "thread_id": "<string>",
    "clarification_response": { "selected_option": "<string>", "free_text": "<string?>" }
  }
  ```
- **Response**: `200`
  ```json
  { "status": "ok", "thread_id": "<string>" }
  ```

### 2.6 `POST /api/chat/stop`

Stop the in-flight generation for a thread (wakes the clarification node and
ends the run).

- **Request body**: `StopRequest`
  ```json
  { "thread_id": "<string>" }
  ```
- **Response**: `200`
  ```json
  { "status": "ok", "thread_id": "<string>" }
  ```

---

## 3. SSE Endpoints

### 3.1 `POST /api/chat` (v3 — Supervisor-Worker streaming)

The primary chat endpoint. Streams the v3 Supervisor-Worker workflow events
as `text/event-stream`. This is the **v3 contract** that the frontend
`lib/sse.ts::streamChat` subscribes to — it must not be altered or
version-bumped.

- **Request body**: `ChatRequest`
  ```json
  {
    "user_input": "<string>",
    "thread_id": "<string>",
    "mode": "auto_fast" | "auto_standard" | "manual",
    "manual_modules": ["<string>", ...]
  }
  ```
- **Response**: `200` `text/event-stream`

#### Event types emitted (in order, selected; full list in `frontend/lib/store.ts::ingestSSEEvent`)

| Event                      | data shape                                                                                                                                | Description                                                              |
|----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------|
| `config`                   | `{ model_name: string }`                                                                                                                  | Emitted at stream start so the UI shows the real model.                  |
| `agent_registry`           | `AgentRegistryItem[]`                                                                                                                     | Emitted once after `pre_router` produces `execution_plan`; filtered to agents active in this run. |
| `node_start`               | `string`                                                                                                                                  | Localized status text for the node about to execute.                     |
| `workflow_v3_state`        | `{ current_node, status, mode }`                                                                                                          | Current v3 node + run mode.                                              |
| `agent_dispatch`           | `{ target_agent, reasoning?, status, latency_ms?, node_name? }`                                                                           | Per-agent dispatch; filtered to the v3 agent whitelist.                  |
| `clarification_needed`     | `ClarificationRequest`                                                                                                                    | Human-in-the-loop prompt (emitted once).                                 |
| `clarification_resolved`   | `""`                                                                                                                                      | Close the clarification dialog.                                          |
| `mcp_term_definitions`     | `{ definitions, tokens_saved, rewritten_query }`                                                                                          | MCP term-standardization output (`worker_mcp`).                          |
| `mcp_tool_call`            | `MCPToolCall`                                                                                                                             | Individual MCP tool call.                                                |
| `knowledge_graph`          | `{ node_count, edge_count, is_acyclic, topology_signature }`                                                                              | KG summary (`worker_mechanism`).                                         |
| `execution_log`            | `string`                                                                                                                                  | Mechanism plan line / sandbox stdout (`worker_mechanism`/`worker_sandbox`). |
| `rag_insights`             | `RAGInsightsData`                                                                                                                         | RAG insights (`worker_rag`).                                             |
| `rag_online_fallback`      | `{ triggered, hit_rate, message }`                                                                                                        | Online fallback notice (`worker_rag`).                                   |
| `rag_ready`                | `{ summary, fallback, hit_rate }`                                                                                                         | RAG readiness (`worker_rag`).                                            |
| `pkpd_profile`             | `pkpd profile dict`                                                                                                                       | PK/PD profile (`worker_pkpd`).                                           |
| `drug_regimen`             | `drug regimen array`                                                                                                                      | Drug regimen (`worker_pkpd`); rendered inside `combination_synergy`.     |
| `rule_violations`          | `Array<{ rule_name, edge_key?, message, severity }>`                                                                                      | ODE rule-engine violations (`worker_ode`).                               |
| `code_generated`           | `string` (ODE code)                                                                                                                       | Generated ODE code (`worker_ode`).                                       |
| `image_ready`              | `string` (base64 PNG)                                                                                                                     | Simulation figure (`worker_sandbox`).                                    |
| `simulation_csv`           | `string` (path)                                                                                                                           | CSV export path (`worker_sandbox`).                                      |
| `dose_response`            | `{ ...dose_response_data, ic50, ic90, hed }`                                                                                              | Dose-response (`worker_sandbox`).                                        |
| `metrics`                  | `{ species, overall, combo }`                                                                                                             | Scientific features (`worker_report`).                                   |
| `experiment_protocols`     | `Array<Record>`                                                                                                                           | Recommended protocols (`worker_report`).                                 |
| `paper_evidence`           | `Array<Record>`                                                                                                                           | Literature evidence (`worker_report`).                                   |
| `report`                   | `{ markdown, llm_filled_json?, forbidden_terms_violations? }`                                                                             | v2 report object (`worker_report`).                                      |
| `report_ready`             | `string` (markdown)                                                                                                                       | Final markdown report (`worker_report`).                                 |
| **`v4_hypothesis_generated`** | `{ hypothesis_count: int, hypothesis_list: Hypothesis[] }`                                                                             | Emitted when the Hypothesis Agent hook writes `v4_hypothesis_generated=True` (Phase 6). |
| **`v4_hypothesis_list`**   | `Hypothesis[]`                                                                                                                            | Forwarded by the v3→v4 adapter.                                          |
| `token_usage`              | `{ total_tokens, mcp_tokens_saved?, model_name }`                                                                                         | Emitted in `finally` before `end`.                                       |
| `error`                    | `string`                                                                                                                                  | Workflow exception.                                                       |
| `end`                      | `""`                                                                                                                                      | Stream terminator (always emitted last).                                 |

The `_v3_event_stream` generator resets all structured state fields per
request (preventing cross-request pollution), accumulates token usage +
MCP tokens saved, and runs `cleanup_clarification_events(thread_id)` in
`finally`.

---

## 4. v4 Endpoints (`/api/v4` prefix)

### 4.1 `POST /api/v4/benchmarks/run` (Task E.1 — IMPLEMENTED)

Run the 10-pathway Official Benchmark Suite, streaming progress via SSE.
Read-only — reuses existing P4 specialists + P5 Level-4 validation; does
not invoke scientific code-mutation paths. Each pathway runs sequentially
in a worker thread (`asyncio.to_thread`) so the event loop keeps streaming.

- **Request body**: none (empty)
- **Response**: `200` `text/event-stream`

#### Event types (in order)

| Event                | data shape                                                                                              | Description                                                          |
|----------------------|---------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------|
| `benchmark_start`    | `{ pathway_class: "__suite__", name: "BioDynamics v4 Official Benchmark Suite", total: <int> }`        | Suite-level opener.                                                  |
| `benchmark_start`    | `{ pathway_class, name }`                                                                               | Per-pathway opener (one per pathway class).                          |
| `benchmark_progress` | `{ pathway_class, step: "loading_specialist" }`                                                         | Specialist loading.                                                  |
| `benchmark_progress` | `{ pathway_class, step: "validation_complete" }`                                                       | Validation complete.                                                 |
| `benchmark_result`   | full single-pathway result dict (see `BenchmarkRunner.run_benchmark` schema, includes `status` field)  | Per-pathway result. Failures isolated as `status="fail"`.            |
| `benchmark_complete` | `{ total, passed, failed, results: [...], runtime_seconds }`                                            | Final summary.                                                       |
| `error`              | `string`                                                                                                | Stream-level error.                                                  |
| `end`                | `""`                                                                                                    | Stream terminator.                                                   |

The benchmark suite covers all 10 pathway classes (apoptosis, cell_cycle,
egfr_signaling, jak_stat, mapk_cascade, nfkb_signaling, p53_signaling,
pi3k_akt_mtor, tgf_beta_signaling, wnt_signaling) defined in
`backend/benchmarks/*.yaml`.

### 4.2 Planned v4 endpoints (frontend client stubs — NOT yet implemented)

The following endpoints are typed in `frontend/lib/api.ts` and ready to call,
but the backend routes do not exist yet (they land in later sprint tasks).
They are listed here for contract planning.

| Method | Path                                            | Request body                | Response type            | Description                                      |
|--------|-------------------------------------------------|-----------------------------|--------------------------|--------------------------------------------------|
| GET    | `/api/v4/pathways`                              | —                           | `PathwaySummary[]`       | List all available pathway classes.              |
| GET    | `/api/v4/pathways/{pathwayClass}/graph`         | —                           | `PathwayGraphData`       | Fetch pathway graph (nodes/edges/modules).       |
| POST   | `/api/v4/simulation/run`                        | `SimulationParams`          | `SimulationResult`       | Run a single simulation.                         |
| POST   | `/api/v4/simulation/sweep`                      | `ParameterSweepParams`      | `ParameterSweepResult`   | 1-D parameter sweep for sensitivity exploration. |
| POST   | `/api/v4/benchmark/{pathwayClass}`              | `{}`                        | `BenchmarkResult`        | Run a single-pathway benchmark.                  |
| GET    | `/api/v4/reports/{id}`                          | —                           | `ExperimentReport`       | Fetch a persisted experiment report.             |

Additional planned endpoints referenced by the v4 component design
(SBML import, BioModels fetch): `POST /api/v4/sbml/import`,
`GET /api/v4/biomodels/{id}` — contracts to be finalized when the
corresponding input components (`input/SbmlUpload.tsx`,
`input/BioModelsFetcher.tsx`) wire up to the backend.
