/**
 * Unified API client for BioDynamics v4.
 *
 * Versioning contract (Task C.1):
 * - NEW endpoints (pathway graph / benchmarks / reports / simulation) use the
 *   `/api/v4/...` prefix. These backend routes do not exist yet — they will be
 *   implemented in later sprint tasks (C.2–C.13). The client stubs below are
 *   typed and ready to call.
 * - The AI Assistant chat SSE stream keeps using the existing `/api/chat`
 *   endpoint (the v3 contract). It must NOT be broken — see `lib/sse.ts`.
 * - Legacy admin/control endpoints (`/api/admin/*`, `/api/models/*`,
 *   `/api/chat/*` non-SSE) are kept here verbatim so migrated components keep
 *   working against the current backend.
 */

export const API_BASE =
  typeof process !== "undefined" && process.env.NEXT_PUBLIC_API_BASE
    ? process.env.NEXT_PUBLIC_API_BASE
    : "http://localhost:8000";

/** V4 API prefix for all new BioDynamics endpoints. */
export const V4_PREFIX = "/api/v4";

/** Legacy chat / admin prefix (v3 contract — do not version-bump). */
export const V3_PREFIX = "/api";

// ---------------------------------------------------------------------------
// Shared HTTP helpers
// ---------------------------------------------------------------------------

async function parseJSON<T>(response: Response): Promise<T> {
  if (!response.ok) {
    throw new Error(`HTTP ${response.status} ${response.statusText}`);
  }
  return (await response.json()) as T;
}

/** GET helper. */
export async function getJSON<T>(path: string, signal?: AbortSignal): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "GET",
    headers: { Accept: "application/json" },
    signal,
  });
  return parseJSON<T>(res);
}

/** POST helper with a JSON body. */
export async function postJSON<T>(
  path: string,
  body: unknown,
  signal?: AbortSignal
): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  return parseJSON<T>(res);
}

/** POST helper that returns the raw Response (for non-JSON endpoints). */
export async function postRaw(
  path: string,
  body: unknown,
  signal?: AbortSignal
): Promise<Response> {
  return fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
}

// ---------------------------------------------------------------------------
// V4 domain types (placeholders — refined as C.2–C.13 land)
// ---------------------------------------------------------------------------

/** Canonical pathway class identifiers (mirrors backend pathway registry). */
export type PathwayClass =
  | "egfr"
  | "mapk"
  | "pi3k_akt_mtor"
  | "jak_stat"
  | "nf_kappa_b"
  | "wnt"
  | "tgf_beta"
  | "p53"
  | "apoptosis"
  | "cell_cycle";

export interface PathwaySummary {
  pathway_class: PathwayClass;
  display_name: string;
  category: "core" | "feedback" | "crosstalk" | "perturbation" | "validation";
  species_count?: number;
  description?: string;
}

/** Pathway graph node/edge payload for the Scientific Workspace graph view. */
export interface PathwayGraphData {
  pathway_class: PathwayClass;
  nodes: Array<{
    id: string;
    label: string;
    species: string;
    node_type: "species" | "reaction" | "module" | "perturbation";
    compartment?: string;
  }>;
  edges: Array<{
    source: string;
    target: string;
    relation: "activation" | "inhibition" | "phosphorylation" | "binding" | "catalysis";
    sbo_term?: string;
  }>;
  modules?: Array<{ id: string; label: string; member_ids: string[] }>;
}

export interface SimulationParams {
  pathway_class: PathwayClass;
  duration: number;
  steps: number;
  parameters: Record<string, number>;
  initial_conditions?: Record<string, number>;
  perturbations?: Array<{
    target: string;
    kind: "knockout" | "overexpression" | "inhibit" | "dose";
    value?: number;
    start_time?: number;
  }>;
}

export interface SimulationResult {
  run_id: string;
  pathway_class: PathwayClass;
  time_points: number[];
  species: Record<string, number[]>;
  metrics?: Record<string, number>;
  csv_path?: string;
  image_base64?: string;
}

export interface BenchmarkResult {
  pathway_class: PathwayClass;
  biomd_id: string;
  reference_rmse: number;
  simulated_rmse: number;
  peak_error_pct: number;
  passed: boolean;
  comparison_chart_base64?: string;
}

export interface ParameterSweepParams extends SimulationParams {
  sweep_parameter: string;
  sweep_values: number[];
}

export interface ParameterSweepResult {
  run_id: string;
  sweep_parameter: string;
  sweep_values: number[];
  response_series: Record<string, number[]>;
}

export interface ExperimentReport {
  id: string;
  pathway_class: PathwayClass;
  title: string;
  created_at: string;
  markdown: string;
  validation?: unknown;
  metrics?: Record<string, number>;
}

// ---------------------------------------------------------------------------
// V4 API client stubs (backend endpoints added in later tasks)
// ---------------------------------------------------------------------------

/** List all available pathway classes for the left PathwayTree pane. */
export function fetchPathways(signal?: AbortSignal): Promise<PathwaySummary[]> {
  return getJSON<PathwaySummary[]>(`${V4_PREFIX}/pathways`, signal);
}

/** Fetch the pathway graph (nodes/edges/modules) for the center graph view. */
export function fetchPathwayGraph(
  pathwayClass: PathwayClass,
  signal?: AbortSignal
): Promise<PathwayGraphData> {
  return getJSON<PathwayGraphData>(
    `${V4_PREFIX}/pathways/${encodeURIComponent(pathwayClass)}/graph`,
    signal
  );
}

/** Run a single simulation. Returns the full result payload. */
export function runSimulation(
  params: SimulationParams,
  signal?: AbortSignal
): Promise<SimulationResult> {
  return postJSON<SimulationResult>(`${V4_PREFIX}/simulation/run`, params, signal);
}

/** Run a benchmark against a BioModels reference for the given pathway. */
export function runBenchmark(
  pathwayClass: PathwayClass,
  signal?: AbortSignal
): Promise<BenchmarkResult> {
  return postJSON<BenchmarkResult>(
    `${V4_PREFIX}/benchmark/${encodeURIComponent(pathwayClass)}`,
    {},
    signal
  );
}

/** Fetch a persisted experiment report by id (used by /report/[id]). */
export function fetchReport(id: string, signal?: AbortSignal): Promise<ExperimentReport> {
  return getJSON<ExperimentReport>(`${V4_PREFIX}/reports/${encodeURIComponent(id)}`, signal);
}

/** Run a 1-D parameter sweep for sensitivity exploration. */
export function parameterSweep(
  params: ParameterSweepParams,
  signal?: AbortSignal
): Promise<ParameterSweepResult> {
  return postJSON<ParameterSweepResult>(
    `${V4_PREFIX}/simulation/sweep`,
    params,
    signal
  );
}

// ---------------------------------------------------------------------------
// Legacy v3 endpoints (kept verbatim — AI Assistant + admin contracts)
// ---------------------------------------------------------------------------

/** LLM / embedding / rerank provider status (mirrors `/api/models/status`).
 *  支持三链路容灾：primary LLM → backup LLM → backup2 LLM。
 */
export interface ModelStatusData {
  llm: { provider: string; model: string; base_url: string };
  backup_llm: { provider: string; model: string; base_url: string } | null;
  backup2_llm: { provider: string; model: string; base_url: string } | null;
  user_selected_llm: string | null;
  embedding: { provider: string; model: string };
  rerank: {
    provider: string;
    selection_mode: string;
    provider_priority: string[];
    candidates: Array<{ provider: string; model: string; display_name: string }>;
  };
}

/** LLM 模型可选项（来自 /api/llm/models）。 */
export interface LlmModelOption {
  model: string;
  provider: string;
  base_url: string;
  role: "primary" | "backup" | "backup2" | "unknown";
}

/** /api/llm/models 响应：可选模型列表 + 当前主用 + 容灾链路顺序。 */
export interface LlmModelsData {
  models: LlmModelOption[];
  current: string;
  chain: string[];
}

/** /api/llm/select 响应：切换结果。 */
export interface LlmSelectResult {
  ok: boolean;
  message?: string;
  primary?: string;
  backup?: string;
  backup2?: string;
  chain?: string[];
  current?: string;
  error?: string;
  available?: string[];
}

/** RAG knowledge-base status (mirrors `/api/admin/rag-status`). */
export interface RagStatusData {
  databases: Array<{ name: string; type: string; collection?: string; files?: string[] }>;
  collections: Record<string, number>;
  online_fallback_enabled: boolean;
  online_fallback_threshold: number;
}

/** RAG knowledge-base status (databases / collections / online fallback). */
export function fetchRagStatus(signal?: AbortSignal): Promise<RagStatusData> {
  return getJSON<RagStatusData>(`${V3_PREFIX}/admin/rag-status`, signal);
}

/** LLM / embedding / rerank provider status shown in the header control. */
export function fetchModelStatus(signal?: AbortSignal): Promise<ModelStatusData> {
  return getJSON<ModelStatusData>(`${V3_PREFIX}/models/status`, signal);
}

/** 获取所有可用 LLM 模型列表（供前端切换 UI 渲染）。 */
export function fetchLlmModels(signal?: AbortSignal): Promise<LlmModelsData> {
  return getJSON<LlmModelsData>(`${V3_PREFIX}/llm/models`, signal);
}

/** 切换主 LLM 模型（重新组合三链路容灾顺序）。
 *  传空字符串或 "default" 恢复默认链路。
 */
export function selectLlm(model: string): Promise<LlmSelectResult> {
  return postJSON<LlmSelectResult>(`${V3_PREFIX}/llm/select`, { model });
}

/** Trigger a vector-db rebuild. */
export function updateVectorDb() {
  return postRaw(`${V3_PREFIX}/admin/update-vector-db`, {});
}

/** Clear the LangGraph checkpointer memory for a thread. */
export function clearChatMemory(threadId: string) {
  return postRaw(`${V3_PREFIX}/chat/clear-memory`, { thread_id: threadId });
}

/** Ask the backend to stop the in-flight chat stream for a thread. */
export function stopChat(threadId: string) {
  return postRaw(`${V3_PREFIX}/chat/stop`, { thread_id: threadId });
}

/** Submit a human-in-the-loop clarification answer. */
export function respondClarification(
  threadId: string,
  answer: { selected_option: string; free_text?: string }
) {
  return postRaw(`${V3_PREFIX}/chat/respond`, {
    thread_id: threadId,
    clarification_response: answer,
  });
}
