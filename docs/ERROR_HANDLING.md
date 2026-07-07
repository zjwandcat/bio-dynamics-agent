# Error Handling & Exception Recovery

> Task G.2 — Unified logging, error handling, and exception recovery strategies
> for the BioDynamics Agent backend (FastAPI + LangGraph).

This document defines the canonical recovery strategy for each failure domain in
the pipeline. Every strategy lists: **trigger condition**, **recovery action**,
and **user-visible message**. All recovery paths log structured JSON entries
(see `app/logging_config.py`) and never crash the SSE stream.

---

## 1. Sandbox Failure → Degrade to Template-Only ODE

| Field | Value |
|-------|-------|
| **Trigger** | `worker_sandbox` raises an exception, returns `error_class != "none"`, or the generated ODE code fails to execute within `SANDBOX_TIMEOUT` (default 60s). |
| **Recovery action** | `worker_sandbox` retries up to `max_retries` (mode-dependent, default 3). On each retry it re-invokes `n6_ode_generator` with `correction_suggestion` from the audit hook. If all retries are exhausted, the run degrades to `DEGRADATION_MODE=template_only` — the sandbox emits the last error to `stdout_stderr`, sets `sandbox_failure_reason`, and the report node renders a partial report using the template-generated ODE without LLM augmentation. |
| **User-visible message** | SSE `execution_log` carries the sandbox stdout/stderr; `sandbox_failure_reason` is surfaced in the final report. The stream continues to `worker_report` rather than aborting. |

Code: `app/graph_v3.py::worker_sandbox` (retry loop ~L1132), `app/sandbox.py`.

---

## 2. LLM Failure → Retry 3× with Exponential Backoff, then Fallback to Template

| Field | Value |
|-------|-------|
| **Trigger** | Primary LLM (`ChatOpenAI`) raises `RateLimitError`, `APIError`, `Timeout`, or returns malformed/empty content. Backup LLM (if configured) also fails. |
| **Recovery action** | (1) The OpenAI client itself retries with `max_retries=2` and built-in exponential backoff. (2) `FallbackLLM` (config.py) switches to the backup LLM after a 0.5s delay. (3) `_StructuredOutputRunnable` strips markdown code fences and falls back to regex JSON extraction. (4) If the LLM still cannot produce structured output, the consuming worker degrades: `worker_ode` falls back to `ode_templates/` (rule-engine-selected template with default parameters), `worker_mechanism` falls back to rule-based entity extraction. |
| **User-visible message** | Workflow continues; `rag_fallback=True` / template name is reported via `mechanism` event. No error event is emitted unless the entire worker aborts (then SSE `error` with `code="workflow_exception"`). |

Code: `app/config.py::FallbackLLM`, `_StructuredOutputRunnable`; `app/graph_v3.py::worker_ode`.

---

## 3. RAG Failure → Fallback to Hardcoded Pathway Data

| Field | Value |
|-------|-------|
| **Trigger** | ChromaDB collection is empty/unreachable, embedding API fails, retrieval hit-rate falls below `RAG_ONLINE_FALLBACK_THRESHOLD` (default 0.3), or all retrieved params are marked `is_fallback=True`. |
| **Recovery action** | (1) If `RAG_ONLINE_FALLBACK=true`, `worker_rag` queries online databases (KEGG / Reactome / UniProt / ChEMBL) within `RAG_ONLINE_TOTAL_BUDGET` (default 600s) to supplement. (2) If online fallback also fails or is disabled, the worker marks every edge parameter with `is_fallback=True` and `rag_fallback=True`. (3) `pre_router` detects "all fallback" and sets `DEGRADATION_MODE` to `template_only` for downstream workers, which use hardcoded default parameter ranges from the template selector. |
| **User-visible message** | SSE `rag_ready` event carries `fallback: true`; `rag_online_fallback` event is emitted when online supplement triggers. `rag_insights` notes the degradation. |

Code: `app/graph_v3.py::worker_rag`, `pre_router` all-fallback detection (~L486); `app/rag_client.py`.

---

## 4. SBML Parse Failure → Return Error Event, User Can Retry

| Field | Value |
|-------|-------|
| **Trigger** | `sbml_parser` / `sbml_parser_v2` cannot parse the uploaded SBML XML (malformed XML, unsupported SBML Level/Version, missing `species`/`reaction` elements), or `worker_validator` cannot load the reference model for Level 2 validation. |
| **Recovery action** | The parser logs a warning and returns an empty/partial structure rather than raising. `worker_validator` catches the exception, sets `validation_report.method="skipped"`, `sbml_sim_available=False`, `pass=True` (non-blocking), and records `details.reason`. The pipeline continues; the user is informed that SBML validation was skipped. If the failure originates from a user-uploaded file in `/api/admin/update-vector-db`, the background task logs the error and the endpoint returns `status="started"` (non-blocking). |
| **User-visible message** | SSE `agent_dispatch` for `worker_validator` reports `"验证异常：{exc}"` or `"SBML 验证：method=skipped"`. The user can re-upload a corrected SBML file and retry. If the failure is fatal (e.g. user-supplied model is unusable), SSE `error` event with `code="workflow_exception"` is emitted and the stream terminates gracefully with an `end` event. |

Code: `app/sbml_parser.py`, `app/sbml_grounder/sbml_parser_v2.py`, `app/graph_v3.py::worker_validator` (~L1318).

---

## 5. Unhandled Exception → Global Exception Handler (500 JSON)

| Field | Value |
|-------|-------|
| **Trigger** | Any exception that propagates out of a route handler without being caught by route-level try/except. |
| **Recovery action** | `app.main.global_exception_handler` (registered via `@app.exception_handler(Exception)`) catches the exception, logs it at `ERROR` level with full stack trace (`exc_info=True`) in JSON format, and returns a `500` JSON response: `{"error": "Internal server error", "detail": str(exc)}`. |
| **User-visible message** | HTTP 500 JSON body `{"error": "Internal server error", "detail": "..."}`. For SSE endpoints, the stream's own try/except emits a structured error event `{"event": "error", "data": {"message": "...", "code": "workflow_exception"}}` followed by an `end` event, so the client always receives a terminal event. |

Code: `app/main.py::global_exception_handler`.

---

## SSE Error Event Format (Task G.2 standard)

All SSE error events MUST use the structured shape below so the frontend can
branch on `code`:

```
data: {"event": "error", "data": {"message": "...", "code": "..."}}\n\n
```

Defined codes:

| Code | Emitted by | Meaning |
|------|-----------|---------|
| `workflow_exception` | `/api/chat` stream | Unhandled exception inside the v3 workflow. |
| `benchmark_stream_error` | `/api/v4/benchmarks/run` | Unhandled exception inside the benchmark runner. |

---

## Logging Configuration

| Env var | Default | Purpose |
|---------|---------|---------|
| `LOG_LEVEL` | `INFO` | Root logger level (`DEBUG`/`INFO`/`WARNING`/`ERROR`). Case-insensitive; invalid values fall back to `INFO`. |
| `LOG_JSON` | `true` | `true` → JSON formatter (production, log-aggregation friendly). `false` → plain text formatter (local debugging). |

Logging is initialized once at import time in `app/main.py` via
`setup_logging(level=settings.LOG_LEVEL, json_output=settings.LOG_JSON)`.
The call is idempotent — repeated invocations replace existing `StreamHandler`s
rather than stacking them, so it is safe under test reloads and uvicorn
`--reload`.
