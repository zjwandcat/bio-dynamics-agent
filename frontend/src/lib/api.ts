// API 客户端：封装与后端的所有非 SSE HTTP 调用
// 基于真实端点（注意路径与 spec 中假设的差异）。
//
// 端点说明：
// - POST /api/chat/respond          人工干预响应
// - POST /api/chat/stop             终止流程
// - POST /api/chat/clear-memory     清除会话（注意路径是 /api/chat/clear-memory，不是 /api/admin/clear-memory）
// - GET  /api/models/status         获取当前 LLM 供应商与模型（不是 /api/models）
// - GET  /api/llm/models            获取所有可用 LLM 模型列表
// - POST /api/llm/select            切换主 LLM 模型
// - GET  /api/v4/pathways           获取 10 条通路列表
// - GET  /api/v4/pathways/{class}/graph  获取通路图

// 从 Vite 环境变量读取后端地址，未配置时回退到本地默认值
const BASE_URL: string =
  (import.meta as unknown as { env?: Record<string, string | undefined> }).env
    ?.VITE_API_BASE_URL || 'http://localhost:8000';

/** 通用 JSON 解析与错误处理 */
async function parseJSON<T>(response: Response): Promise<T> {
  if (!response.ok) {
    throw new Error(`HTTP ${response.status} ${response.statusText}`);
  }
  return (await response.json()) as T;
}

/** 统一的 POST + JSON 请求封装 */
async function postJSON<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
    body: JSON.stringify(body),
  });
  return parseJSON<T>(res);
}

/** 统一的 GET 请求封装 */
async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    method: 'GET',
    headers: { Accept: 'application/json' },
  });
  return parseJSON<T>(res);
}

// ---------------------------------------------------------------------------
// 聊天控制接口
// ---------------------------------------------------------------------------

/** POST /api/chat/respond 的人工干预响应请求体 */
interface ClarificationResponsePayload {
  thread_id: string;
  clarification_response: {
    selected_option: string;
    free_text?: string;
  };
}

/**
 * 提交人工干预响应。
 * 端点：POST /api/chat/respond
 * 请求体：{ thread_id, clarification_response: { selected_option, free_text? } }
 * 响应：{ status: 'ok', thread_id }
 *
 * @param threadId 当前会话线程 ID
 * @param selectedOption 用户选择的选项 ID
 * @param freeText 可选的自由文本补充
 */
export async function respondToClarification(
  threadId: string,
  selectedOption: string,
  freeText?: string,
): Promise<{ status: string; thread_id: string }> {
  const payload: ClarificationResponsePayload = {
    thread_id: threadId,
    clarification_response: {
      selected_option: selectedOption,
      ...(freeText !== undefined ? { free_text: freeText } : {}),
    },
  };
  return postJSON<{ status: string; thread_id: string }>('/api/chat/respond', payload);
}

/**
 * 终止当前会话的 SSE 流。
 * 端点：POST /api/chat/stop
 * 请求体：{ thread_id }
 *
 * @param threadId 当前会话线程 ID
 */
export async function stopStream(
  threadId: string,
): Promise<{ status: string; thread_id: string }> {
  return postJSON<{ status: string; thread_id: string }>('/api/chat/stop', {
    thread_id: threadId,
  });
}

/**
 * 清除指定会话的记忆。
 * 端点：POST /api/chat/clear-memory（注意：不是 /api/admin/clear-memory）
 * 请求体：{ thread_id }
 *
 * @param threadId 当前会话线程 ID
 */
export async function clearMemory(
  threadId: string,
): Promise<{ status: string; thread_id: string; message: string }> {
  return postJSON<{ status: string; thread_id: string; message: string }>(
    '/api/chat/clear-memory',
    { thread_id: threadId },
  );
}

// ---------------------------------------------------------------------------
// LLM 模型管理接口
// ---------------------------------------------------------------------------

/** 单个 LLM 供应商配置 */
interface LlmProviderInfo {
  provider: string;
  model: string;
  base_url: string;
}

/**
 * 获取当前 LLM 供应商与模型状态（含三链路容灾）。
 * 端点：GET /api/models/status（注意：不是 /api/models）
 */
export async function getModelsStatus(): Promise<{
  llm: LlmProviderInfo;
  backup_llm: LlmProviderInfo | null;
  backup2_llm: LlmProviderInfo | null;
  user_selected_llm: string | null;
}> {
  return getJSON<{
    llm: LlmProviderInfo;
    backup_llm: LlmProviderInfo | null;
    backup2_llm: LlmProviderInfo | null;
    user_selected_llm: string | null;
  }>('/api/models/status');
}

/**
 * 获取所有可用 LLM 模型列表。
 * 端点：GET /api/llm/models
 */
export async function getLlmModels(): Promise<{
  models: Array<{ model: string; provider: string; base_url: string; role: string }>;
  current: string;
  chain: string[];
}> {
  return getJSON<{
    models: Array<{ model: string; provider: string; base_url: string; role: string }>;
    current: string;
    chain: string[];
  }>('/api/llm/models');
}

/**
 * 切换主 LLM 模型。
 * 端点：POST /api/llm/select
 * 请求体：{ model }
 *
 * @param model 目标模型名称
 */
export async function selectLlm(model: string): Promise<unknown> {
  return postJSON<unknown>('/api/llm/select', { model });
}

// ---------------------------------------------------------------------------
// 知识库管理接口
// ---------------------------------------------------------------------------

/** 知识库更新状态取值 */
export type KbUpdateStatus = 'idle' | 'running' | 'success' | 'failed';

/** GET /api/admin/kb-update-status 响应结构 */
export interface KbUpdateStatusResponse {
  status: KbUpdateStatus;
  started_at: number | null;
  finished_at: number | null;
  message: string;
  stats: {
    files_processed: number;
    chunks_inserted: number;
    collection_name: string;
    persist_dir: string;
  } | null;
}

/**
 * 触发知识库后台更新。
 * 端点：POST /api/admin/update-vector-db
 * 响应：{ status: 'started' | 'already_running', message }
 */
export async function triggerKbUpdate(): Promise<{
  status: string;
  message: string;
}> {
  return postJSON<{ status: string; message: string }>(
    '/api/admin/update-vector-db',
    {},
  );
}

/**
 * 查询知识库更新当前状态（供前端轮询）。
 * 端点：GET /api/admin/kb-update-status
 */
export async function getKbUpdateStatus(): Promise<KbUpdateStatusResponse> {
  return getJSON<KbUpdateStatusResponse>('/api/admin/kb-update-status');
}

// ---------------------------------------------------------------------------
// V4 通路相关接口
// ---------------------------------------------------------------------------

/** 通路列表条目结构 */
interface PathwaySummary {
  pathway_class: string;
  display_name: string;
  category: string;
  species_count: number;
  description: string;
}

/**
 * 获取 10 条通路列表。
 * 端点：GET /api/v4/pathways
 */
export async function getPathways(): Promise<PathwaySummary[]> {
  return getJSON<PathwaySummary[]>('/api/v4/pathways');
}

/** 通路图节点结构 */
interface PathwayGraphNode {
  id: string;
  label: string;
  species: string;
  node_type: string;
  compartment?: string;
}

/** 通路图边结构 */
interface PathwayGraphEdge {
  source: string;
  target: string;
  relation: string;
}

/**
 * 获取指定通路的通路图数据。
 * 端点：GET /api/v4/pathways/{pathway_class}/graph
 *
 * @param pathwayClass 通路类标识（如 'egfr'）
 */
export async function getPathwayGraph(pathwayClass: string): Promise<{
  pathway_class: string;
  nodes: PathwayGraphNode[];
  edges: PathwayGraphEdge[];
  modules: unknown[];
  display_name: string;
  source_sbml: string;
  source_kegg: string;
}> {
  const encoded = encodeURIComponent(pathwayClass);
  return getJSON<{
    pathway_class: string;
    nodes: PathwayGraphNode[];
    edges: PathwayGraphEdge[];
    modules: unknown[];
    display_name: string;
    source_sbml: string;
    source_kegg: string;
  }>(`/api/v4/pathways/${encoded}/graph`);
}
