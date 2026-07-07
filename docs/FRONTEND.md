# Frontend Architecture (v4)

Next.js 14 (App Router) + React 18 + TypeScript + Tailwind CSS + Zustand.
The v4 redesign introduces a four-pane **Scientific Modeling IDE** shell that
co-exists with the existing AI-Assistant chat (v3 `/api/chat` SSE contract).

---

## 1. Four-Pane Scientific IDE Layout

`frontend/components/workspace/WorkbenchShell.tsx` (Task C.1) renders the
workbench as a CSS Grid (`h-screen`, dark theme `bg-zinc-950`). The grid
template columns change when the AI Assistant pane is toggled:

```
aiOpen=false:  "250px minmax(0, 1fr) 320px"
aiOpen=true:   "250px minmax(0, 1fr) 320px 360px"
```

```
┌─────────────┬──────────────────────┬──────────────┬──────────────┐
│ Project /   │ Scientific Workspace │ Validation   │ AI Assistant │
│ Pathway     │ (Pathway Graph +     │ (Pyramid +   │ (collapsible │
│ (250px)     │  Sim Tabs + Params)  │  Hypothesis) │  360px → 0)  │
└─────────────┴──────────────────────┴──────────────┴──────────────┘
```

Above the grid sits `WorkbenchHeader` (model/RAG status, AI toggle) and an
optional `updateDbStatus` banner. The AI Assistant pane is **collapsed by
default** (`uiState.aiAssistantOpen=false`) — the center Scientific Workspace
is the primary UI, not the chat.

### Pane contents

| Pane            | Width     | Components                                                                                         |
|-----------------|-----------|----------------------------------------------------------------------------------------------------|
| **Left** — Project / Pathway | 250px | `RunControls` + `PathwayTree` (C.2) + `BenchmarkList` (C.2/C.12) + `SimulationHistory` (C.2/C.11) |
| **Center** — Scientific Workspace | `1fr` | `PathwayGraph` + `NodeDetailPanel` (C.3) / `SimulationTabs` (C.4) / `ParameterExplorer` (C.5) |
| **Right** — Validation | 320px | `ValidationPyramid` + `ValidationDetail` (C.7) / `HypothesisPanel` (C.8) / Evidence & Warnings |
| **Far Right** — AI Assistant | 360px (collapsible) | `AIAssistantPanel` (C.9): Chat / Suggestions / Logs tabs |

On mount the shell hydrates admin status once (`refreshRagStatus()` +
`refreshModelStatus()`).

---

## 2. Component Directory (grouped by pane)

All paths under `frontend/components/`. Files marked with their originating
task where applicable.

### 2.1 Workspace shell
- `workspace/WorkbenchShell.tsx` — four-pane grid shell (C.1)
- `workspace/WorkbenchHeader.tsx` — top header (model/RAG status, AI toggle)
- `workspace/RunControls.tsx` — run-mode controls in the left pane
- `workspace/PlaceholderPanel.tsx` — placeholder slot used before C.2–C.8 land

### 2.2 Left pane — Project / Pathway
- `pathway/LeftPane.tsx` — left pane container
- `pathway/PathwayTree.tsx` — pathway class hierarchy tree + module selection (C.2)
- `pathway/BenchmarkList.tsx` — BioModels benchmark reference list (C.2)
- `pathway/SimulationHistory.tsx` — experiment & simulation history (C.2)
- `pathway/PathwayWorkspace.tsx` — pathway workspace wrapper
- `pathway/NodeDetailPanel.tsx` — selected graph node detail (C.3)

### 2.3 Center pane — Scientific Workspace
- `pathway/PathwayGraph.tsx` — interactive pathway mechanism graph (C.3)
- `simulation/SimulationPanel.tsx` — simulation panel container (C.4)
- `simulation/SimulationTabs.tsx` — 6-tab multi-view container (C.4)
- `simulation/TimeSeriesChart.tsx` — time-series view
- `simulation/DoseResponseChart.tsx` — dose-response view
- `simulation/SensitivityChart.tsx` — sensitivity view
- `simulation/PhasePortraitChart.tsx` — phase-portrait view
- `simulation/SteadyStateChart.tsx` — steady-state view
- `simulation/OscillationChart.tsx` — oscillation view
- `parameter/ParameterPanel.tsx` — parameter editor (C.5)
- `parameter/ParameterExplorer.tsx` — slider/IC50/EC50/KO/OE/Mutation explorer (C.5)

### 2.4 Right pane — Validation
- `validation/ValidationPanel.tsx` — validation panel container (C.7)
- `validation/ValidationPyramid.tsx` — 5-level pyramid (Level 1-5) (C.7)
- `validation/ValidationDetail.tsx` — expandable per-level details (C.7)
- `hypothesis/HypothesisPanel.tsx` — hypothesis list panel (C.8)
- `hypothesis/HypothesisCard.tsx` — single hypothesis card (5 sections)
- `hypothesis/ExperimentCard.tsx` — 6-field experiment design card
- `hypothesis/types.ts` — `Hypothesis` types + `deriveFalsifiability` helpers

### 2.5 Far-right pane — AI Assistant (collapsible)
- `ai_assistant/AIAssistantPanel.tsx` — Chat / Suggestions / Logs tabs (C.9)
- `ai_assistant/ChatInput.tsx` — chat input box
- `ai_assistant/ChatMessage.tsx` — single chat message renderer
- `ai_assistant/SuggestionsPanel.tsx` — suggestions tab
- `ai_assistant/LogsPanel.tsx` — logs tab
- `ai_assistant/ClarificationDialog.tsx` — human-in-the-loop dialog
- `ai_assistant/WorkflowVisualization.tsx` — pipeline step visualization
- `ai_assistant/AgentWorkflowTracker.tsx` — agent dispatch tracker
- `ai_assistant/MCPToolPanel.tsx` — MCP tool-call panel
- `ai_assistant/TermDefinitionCard.tsx` — MCP term definition card
- `ai_assistant/RAGInsightPanel.tsx` — RAG insights panel
- `ai_assistant/DoseResponseCurve.tsx` — dose-response curve card
- `ai_assistant/TokenPerformanceBadge.tsx` — token-usage badge

### 2.6 Home / Report / Benchmarks pages
- `home/PathwayCard.tsx` — pathway selector card (C.10)
- `home/BenchmarkCards.tsx` — benchmark case cards (C.10)
- `home/RecentSimulations.tsx` — recent simulations list (C.10)
- `input/InputArea.tsx` — multi-mode input container (C.6)
- `input/SbmlUpload.tsx` — SBML file upload (C.6)
- `input/BioModelsFetcher.tsx` — BioModels ID fetcher (C.6)
- `report/ReportViewer.tsx` — report page container (C.11)
- `report/ReportSection.tsx` — report section block (C.11)
- `report/DynamicAnalysis.tsx` — dynamic analysis section (C.11)
- `report/LiteratureComparison.tsx` — literature comparison section (C.11)
- `benchmark/BenchmarkCenter.tsx` — benchmark center page (C.12)
- `benchmark/BenchmarkCard.tsx` — single pathway benchmark card (C.12)

### 2.7 Legacy chat (mirrored under `chat/`)
A duplicate set of the AI-Assistant components exists under `chat/`
(`ChatMessage.tsx`, `ChatInput.tsx`, `ControlBar.tsx`, `ClarificationDialog.tsx`,
`WorkflowVisualization.tsx`, `AgentWorkflowTracker.tsx`, `MCPToolPanel.tsx`,
`RAGInsightPanel.tsx`, `DoseResponseCurve.tsx`, `TermDefinitionCard.tsx`,
`TokenPerformanceBadge.tsx`) — kept for the legacy single-page chat before the
workbench migration.

### 2.8 UI primitives (`ui/`)
shadcn/ui-style primitives: `button.tsx`, `card.tsx`, `badge.tsx`, `tabs.tsx`,
`input.tsx`, `textarea.tsx`, `scroll-area.tsx`, `separator.tsx`, `avatar.tsx`.

---

## 3. Routing (App Router)

`frontend/app/`:

| Route            | File                          | Description                                                                 |
|------------------|-------------------------------|-----------------------------------------------------------------------------|
| `/`              | `app/page.tsx`                | Home — Pathway Selector / Scientific Workspace preview / Recent Simulations / Benchmark Cases (C.10) |
| `/workspace`     | `app/workspace/page.tsx`      | Four-pane Scientific IDE shell (`WorkbenchShell`) (C.1)                     |
| `/report/[id]`   | `app/report/[id]/page.tsx`    | Experiment Report page — 6 sections (Executive Summary / Scientific Findings / Dynamic Analysis / Validation / Literature Comparison / Future Experiments) (C.11) |
| `/benchmarks`    | `app/benchmarks/page.tsx`     | Benchmark Center — 10 pathway cards + Run All + real-time SSE progress (C.12) |

`app/layout.tsx` is the root layout: `<html lang="en" className="dark">`,
Geist Sans + Geist Mono fonts, `metadata.title = "BioDynamics Agent"`. The
body is a `min-h-full flex flex-col` container.

---

## 4. State Management — Zustand Store

`frontend/lib/store.ts` exports a single global store
`useWorkbenchStore` (created with `create<WorkbenchStore>`). It owns three
slices:

### 4.1 v4 Scientific Workspace domain state

```ts
currentPathway: PathwayClass | null;
simulationResult: SimulationResult | null;
validationReport: unknown | null;
hypothesisList: unknown[];
pathwayGraph: PathwayGraphData | null;
agentDispatches: DispatchData[];
```

Setters: `setCurrentPathway`, `setPathwayGraph`, `setSimulationResult`.

### 4.2 AI Assistant chat state (migrated verbatim from legacy `app/page.tsx`)

```ts
messages: Message[];          // superset of all v1/v2/v3/v4 SSE payloads
input: string;
isStreaming: boolean;
threadId: string;
tokenUsage: number;
modelName: string;
agents: AgentState[];
clarification: ClarificationRequest | null;
currentNode: string | undefined;
lastRagHitRate: number | undefined;
lastLatencyMs: number | undefined;
mcpTokensSavedTotal: number | undefined;
pipelineSteps: V2PipelineStep[];
pipelineCurrent: string;
pipelineStepIndex: number;
pipelineTotal: number;
pipelineName: "v1" | "v2";
pipelineStatus: "starting" | "running" | "completed" | "failed";
```

### 4.3 Admin / control + UI state

```ts
ragStatus: RagStatusData | null;
modelStatus: ModelStatusData | null;
controlBarState: ControlBarState;   // { mode: RunMode, manualModules: string[] }
isUpdatingDb: boolean;
updateDbStatus: string;
uiState: { aiAssistantOpen: boolean };  // collapsed by default
```

### 4.4 Actions

- **UI / domain** — `setInput`, `setControlBarState`, `toggleAIPanel`,
  `setAIPanelOpen`, `setCurrentPathway`, `setPathwayGraph`, `setSimulationResult`.
- **Admin** — `refreshRagStatus`, `refreshModelStatus`, `handleUpdateVectorDb`.
- **Chat lifecycle** — `sendMessage(text)`, `stopGeneration()`, `clearMemory()`,
  `submitClarification(answer)`.
- **SSE ingestion** — `ingestSSEEvent(event)` (internal, exported for tests).

`RunMode = "auto_fast" | "auto_standard" | "manual"`.

Module-level (non-reactive) accumulators live outside the store:
`abortController`, `mcpToolCallsAcc`, `streamCodeGenCount`.

---

## 5. SSE Event Handling

### 5.1 `lib/sse.ts` — stream transport

`streamChat(payload, handlers, signal)` POSTs to `${API_BASE}/api/chat`
(the v3 SSE contract — must NOT be version-bumped). It reads the
`text/event-stream` response frame-by-frame, parses each `data:` line as
JSON `{"event": "...", "data": ...}`, and forwards `{event, data}` to
`handlers.onEvent`. `[DONE]` sentinel and malformed JSON frames are skipped
silently. `AbortError` (user hits Stop) is treated as benign.

`ChatStreamPayload = { user_input, thread_id, mode, manual_modules }`.

### 5.2 `store.ingestSSEEvent(event)` — the big switch

`sendMessage()` calls `streamChat` with `onEvent: (e) => get().ingestSSEEvent(e)`.
The switch (ported from legacy `app/page.tsx`) handles every v1/v2/v3/v4
event type. Selected cases:

| Event                      | Store action                                                                       |
|----------------------------|------------------------------------------------------------------------------------|
| `config`                   | set `modelName` from `data.model_name`                                             |
| `agent_registry`           | replace `agents` with the registry payload                                         |
| `workflow_v3_state`        | set `currentNode`                                                                  |
| `agent_dispatch`           | upsert agent in `agents` + append to `agentDispatches` + attach latency metric     |
| `clarification_needed`     | set `clarification`                                                                |
| `clarification_resolved`   | clear `clarification`                                                              |
| `rag_insights`             | append `rag_insights` message                                                      |
| `mcp_tool_call`            | accumulate into `mcpToolCallsAcc` + status update                                  |
| `mcp_term_definitions`     | append `mcp_tools` + `mcp_terms` messages, set `mcpTokensSavedTotal`               |
| `node_start`               | update trailing status message                                                     |
| `code_generated`           | append `code` message (increment `streamCodeGenCount` for retry detection)         |
| `execution_log`            | append `log` message (annotated if retry)                                          |
| `image_ready`              | append `image` message (base64)                                                    |
| `rag_ready`                | set `lastRagHitRate` + attach to last agent message                                |
| `rag_online_fallback`      | append status message when triggered                                               |
| `pkpd_profile`             | append `pkpd_profile` message                                                      |
| `dose_response`            | append `dose_response` message                                                     |
| `combination_synergy`      | append `combination_synergy` message                                               |
| `report_ready`             | append `report` message with RAG/latency/MCP metrics                               |
| `token_usage`              | set `tokenUsage`, `modelName`, `mcpTokensSavedTotal`                               |
| `error`                    | append error text message                                                          |
| `end`                      | remove trailing status                                                             |
| `workflow_v2_state`        | update `pipelineName/Status/StepIndex/Total/Current` + pipeline step               |
| `knowledge_graph`          | append KG summary text message                                                     |
| `rule_violations`          | append violations text message                                                     |
| `metrics`                  | append scientific-features text message with `metrics`                             |
| `experiment_protocols`     | append protocols text message                                                      |
| `paper_evidence`           | append evidence text message                                                       |
| `report`                   | append `report` message from `data.markdown`                                       |
| `simulation_csv`           | status update with CSV path                                                        |
| **`v4_hypothesis_list`**   | `set({ hypothesisList: eventData })` (replaces list)                               |
| **`v4_hypothesis_generated`** | shape-tolerant unwrap (array / `v4_hypothesis_list` / `hypotheses` / single `hypothesis`) → set/append `hypothesisList` |
| `v4_validation_report` / `v4_pathway_graph` / `v4_simulation_result` | TODO(C.7/C.4): hydrate respective slices |

Internal message helpers: `appendMessage`, `updateLastStatus`,
`removeTrailingStatus`, `setLastAgentTokenUsage`, `attachMetricsToLastAgent`,
`handleAgentDispatch`.

---

## 6. API Layer — `lib/api.ts`

`API_BASE` defaults to `http://localhost:8000` (override via
`NEXT_PUBLIC_API_BASE`). Two prefixes:

- `V4_PREFIX = "/api/v4"` — new BioDynamics endpoints.
- `V3_PREFIX = "/api"` — legacy chat + admin contract (do not version-bump).

Shared helpers: `getJSON<T>`, `postJSON<T>`, `postRaw` (all honor an optional
`AbortSignal`).

### 6.1 v4 client stubs (typed; backend routes land in later sprint tasks)

| Function             | Method | Path                                            | Request body                | Response type            |
|----------------------|--------|-------------------------------------------------|-----------------------------|--------------------------|
| `fetchPathways`      | GET    | `/api/v4/pathways`                              | —                           | `PathwaySummary[]`       |
| `fetchPathwayGraph`  | GET    | `/api/v4/pathways/{pathwayClass}/graph`         | —                           | `PathwayGraphData`       |
| `runSimulation`      | POST   | `/api/v4/simulation/run`                        | `SimulationParams`          | `SimulationResult`       |
| `parameterSweep`     | POST   | `/api/v4/simulation/sweep`                      | `ParameterSweepParams`      | `ParameterSweepResult`   |
| `runBenchmark`       | POST   | `/api/v4/benchmark/{pathwayClass}`              | `{}`                        | `BenchmarkResult`        |
| `fetchReport`        | GET    | `/api/v4/reports/{id}`                          | —                           | `ExperimentReport`       |

> Note: `runBenchmark(pathwayClass)` (single-pathway POST) is the typed client
> stub. The actually implemented backend SSE endpoint is
> `POST /api/v4/benchmarks/run` (full 10-pathway suite) — see `docs/API.md`.

Key v4 domain types: `PathwayClass` (10 values), `PathwaySummary`,
`PathwayGraphData`, `SimulationParams`, `SimulationResult`, `BenchmarkResult`,
`ParameterSweepParams`, `ParameterSweepResult`, `ExperimentReport`.

### 6.2 v3 legacy endpoints (kept verbatim — AI Assistant + admin contracts)

| Function              | Method | Path                            | Request body                         | Response type     |
|-----------------------|--------|---------------------------------|--------------------------------------|-------------------|
| `fetchRagStatus`      | GET    | `/api/admin/rag-status`         | —                                    | `RagStatusData`   |
| `fetchModelStatus`    | GET    | `/api/models/status`            | —                                    | `ModelStatusData` |
| `updateVectorDb`      | POST   | `/api/admin/update-vector-db`   | `{}`                                 | `Response`        |
| `clearChatMemory`     | POST   | `/api/chat/clear-memory`        | `{ thread_id }`                      | `Response`        |
| `stopChat`            | POST   | `/api/chat/stop`                | `{ thread_id }`                      | `Response`        |
| `respondClarification`| POST   | `/api/chat/respond`             | `{ thread_id, clarification_response }` | `Response`     |

The chat SSE stream itself is opened by `lib/sse.ts::streamChat` against
`POST /api/chat` — see §5.1. See `docs/API.md` for the full backend contract.
