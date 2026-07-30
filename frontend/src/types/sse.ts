// SSE 事件类型定义与节点阶段映射常量
// 基于后端 v3 Supervisor-Worker 工作流真实 SSE 契约（已调研确认）。
// 注意：多个事件的 data 是裸字符串而非对象，详见各接口注释。

// ---------------------------------------------------------------------------
// 通用 payload
// ---------------------------------------------------------------------------

/** SSE 事件 payload 统一格式 */
export interface SSEPayload {
  event: string;
  /** 可能是字符串或对象，按事件类型区分 */
  data: unknown;
}

// ---------------------------------------------------------------------------
// 各事件 data 结构定义
// ---------------------------------------------------------------------------

/** config 事件：当前 LLM 配置 */
export interface ConfigData {
  model_name: string;
}

/** agent_registry 事件：单个 agent 定义项 */
export interface AgentDef {
  name: string;
  cn_label: string;
  description: string;
  icon: string;
  mapped_node: string;
}

// 说明：node_start / execution_log / code_generated / image_ready /
// report_ready / clarification_resolved / end 这几个事件的 data 是裸字符串，
// 因此不单独定义结构接口，使用 string 即可。

/** workflow_v3_state 事件：当前节点状态来源（而非 node_start） */
export interface WorkflowV3StateData {
  current_node: string;
  status: string;
  mode: string;
}

/** agent_dispatch 事件：主管调度信息 */
export interface AgentDispatchData {
  target_agent: string;
  reasoning: string;
  status: string;
  timestamp: number;
  node_name: string;
  latency_ms: number;
}

/** clarification_needed 事件：追问选项 */
export interface ClarificationOption {
  id: string;
  label: string;
}

/** clarification_needed 事件：主管请求人工干预 */
export interface ClarificationNeededData {
  question: string;
  options: ClarificationOption[];
  context: string;
}

/** mcp_term_definitions 事件：术语标准化结果 */
export interface McpTermDefinitionsData {
  definitions: unknown[];
  tokens_saved: number;
  rewritten_query: string;
}

/** knowledge_graph 事件：知识图谱拓扑信息 */
export interface KnowledgeGraphData {
  node_count: number;
  edge_count: number;
  is_acyclic: boolean;
  topology_signature: string;
}

/** v4_pathway_graph 事件：通路图节点 */
export interface PathwayNode {
  id: string;
  label: string;
  species: string;
  node_type: string;
  compartment?: string;
}

/** v4_pathway_graph 事件：通路图边 */
export interface PathwayEdge {
  source: string;
  target: string;
  relation: string;
}

/** v4_pathway_graph 事件：完整通路图数据 */
export interface V4PathwayGraphData {
  pathway_class: string;
  nodes: PathwayNode[];
  edges: PathwayEdge[];
  modules: unknown[];
}

/** rag_insights 事件：RAG 检索洞察 */
export interface RagInsightsData {
  rewritten_query: string;
  rewrites: unknown[];
  source_distribution: Record<string, number>;
  total_candidates: number;
  top_selections: unknown[];
  hit_rate: number;
  drug_candidates: unknown[];
  online_fallback_enabled: boolean;
}

/** rag_online_fallback 事件：在线兜底触发情况 */
export interface RagOnlineFallbackData {
  triggered: boolean;
  hit_rate: number;
  message: string;
}

/** rag_ready 事件：RAG 完成摘要 */
export interface RagReadyData {
  summary: string;
  fallback: boolean;
  hit_rate: number;
}

/** v4_simulation_result 事件：仿真结果 */
export interface V4SimulationResultData {
  run_id: string;
  pathway_class: string;
  time_points: number[];
  species: Record<string, number[]>;
  metrics: Record<string, unknown>;
  csv_path: string | null;
  image_base64: string | null;
}

/** report 事件：完整报告（含 markdown 子字段） */
export interface ReportData {
  markdown: string;
  sprint2_evidence_bundle?: unknown;
  sprint5_provenance?: unknown;
  sprint5_decision_log?: unknown;
}

/** v4_validation_report 事件：五级验证报告 */
export interface V4ValidationReportData {
  metrics?: unknown;
  report_markdown?: string;
  experiment_protocols?: unknown[];
  paper_evidence?: unknown[];
  confidence?: number;
  passed?: boolean;
  level1?: unknown;
  level2?: unknown;
  level3?: unknown;
  level4?: unknown;
  level5?: unknown;
  overall_pass?: boolean;
  failed_levels?: string[];
  short_circuit?: boolean;
  agent_version?: string;
}

// 说明：experiment_protocols / paper_evidence / v4_hypothesis_list 这几个事件的
// data 是数组，使用 unknown[] 即可。

/** token_usage 事件：token 消耗统计（注意字段是 model_name 而非 model） */
export interface TokenUsageData {
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  mcp_tokens_saved?: number;
  model_name: string;
}

/** error 事件：错误信息 */
export interface ErrorData {
  message: string;
  code: string;
}

/** sa_* 事件统一基类（敏感性分析系列，均带可选 pathway 字段） */
export interface SaDataBase {
  pathway?: string;
}

// 说明：dose_response 事件使用 SaDataBase 衍生结构；
// simulation_csv 事件的 data 是字符串（CSV 内容）。

// ---------------------------------------------------------------------------
// SSE 事件名联合类型
// ---------------------------------------------------------------------------

/** 所有 SSE 事件名（字面量联合类型），`sa_${string}` 覆盖敏感性分析系列事件 */
export type SSEEventName =
  | 'config'
  | 'agent_registry'
  | 'node_start'
  | 'execution_log'
  | 'code_generated'
  | 'image_ready'
  | 'report_ready'
  | 'clarification_resolved'
  | 'end'
  | 'workflow_v3_state'
  | 'agent_dispatch'
  | 'clarification_needed'
  | 'mcp_term_definitions'
  | 'knowledge_graph'
  | 'v4_pathway_graph'
  | 'rag_insights'
  | 'rag_online_fallback'
  | 'rag_ready'
  | 'v4_simulation_result'
  | 'report'
  | 'v4_validation_report'
  | 'experiment_protocols'
  | 'paper_evidence'
  | 'v4_hypothesis_list'
  | 'token_usage'
  | 'error'
  | 'dose_response'
  | 'simulation_csv'
  | `sa_${string}`;

/** 单个解析后的 SSE 事件 */
export interface SSEEvent {
  /** 事件名（见 SSEEventName） */
  event: string;
  /** 事件数据，按事件类型可能是字符串或对象 */
  data: unknown;
}

// ---------------------------------------------------------------------------
// 节点名 → 中文阶段名映射（基于真实 v3 节点名）
// ---------------------------------------------------------------------------

/** 节点名到中文阶段名的映射 */
export const NODE_STAGE_MAP: Record<string, string> = {
  pre_router: '运行模式分析',
  supervisor: '主管调度',
  worker_mcp: '术语标准化',
  worker_mechanism: '机制解析与图谱',
  worker_rag: '知识检索 (RAG)',
  worker_pkpd: 'PK/PD 推断',
  worker_ode: 'ODE 方程生成',
  worker_sandbox: '沙箱仿真执行',
  worker_report: '预测报告生成',
  clarification_node: '人工干预',
};

// ---------------------------------------------------------------------------
// 节点名 → "正在做什么"文案映射
// ---------------------------------------------------------------------------

/** 节点名到"正在做什么"文案的映射 */
export const NODE_ACTION_TEXT: Record<string, string> = {
  pre_router: '正在分析运行模式并生成执行计划...',
  supervisor: '主管正在调度下一个 Worker...',
  worker_mcp: '正在查询生物医学术语定义...',
  worker_mechanism: '正在解析信号通路机制与构建知识图谱...',
  worker_rag: '正在从 RAG 数据库检索参数与文献...',
  worker_pkpd: '正在推断 PK/PD 参数...',
  worker_ode: '正在生成 ODE 方程代码...',
  worker_sandbox: '正在沙箱中执行仿真...',
  worker_report: '正在生成预测报告...',
  clarification_node: '等待用户回答追问...',
};

// ---------------------------------------------------------------------------
// 节点名 → lucide 图标名映射（用字符串，组件层映射为实际图标组件）
// ---------------------------------------------------------------------------

/** 节点名到 lucide 图标名（字符串）的映射 */
export const NODE_ICON_NAME: Record<string, string> = {
  pre_router: 'Route',
  supervisor: 'GitMerge',
  worker_mcp: 'BookOpen',
  worker_mechanism: 'Network',
  worker_rag: 'Search',
  worker_pkpd: 'Syringe',
  worker_ode: 'Code',
  worker_sandbox: 'FlaskConical',
  worker_report: 'FileText',
  clarification_node: 'HelpCircle',
};

// ---------------------------------------------------------------------------
// 节点名 → 真实 Agent 名字映射（基于后端 AGENT_REGISTRY_V2）
// 后端 _dispatch_for_v3_worker 发送的 target_agent 是中文短语，
// 这里用 node_name 映射到 supervisor.py 中定义的真实 Agent 英文名 + 中文标签
// ---------------------------------------------------------------------------

/** 真实 Agent 信息（对应后端 AGENT_REGISTRY_V2） */
export interface AgentInfo {
  /** 英文 Agent 名（如 "Mechanism Analysis Agent"） */
  name: string;
  /** 中文短标签（如 "机制解析"） */
  cnLabel: string;
}

/** 节点名 → 真实 Agent 信息映射（对应 supervisor.py AGENT_REGISTRY_V2）
 *  注意：worker_ode 和 worker_sandbox 同属 Simulation Engineer Agent；
 *  worker_validator 是 P0-4 新增的 SBML Validator Agent；
 *  共 10 个 Agent 中有 7 个会出现在 v3 工作流中（其余在 v2 内部节点触发）。 */
export const NODE_NAME_TO_AGENT: Record<string, AgentInfo> = {
  pre_router: { name: 'Pre-Router', cnLabel: '运行模式分析' },
  supervisor: { name: 'Supervisor', cnLabel: '主管调度' },
  worker_mcp: { name: 'Terminology Agent', cnLabel: '术语标准化' },
  worker_mechanism: { name: 'Mechanism Analysis Agent', cnLabel: '机制解析' },
  worker_rag: { name: 'Knowledge Retrieval Agent', cnLabel: '知识检索' },
  worker_pkpd: { name: 'PK/PD Modeling Agent', cnLabel: 'PK/PD 建模' },
  worker_ode: { name: 'Simulation Engineer Agent', cnLabel: '仿真工程' },
  worker_sandbox: { name: 'Simulation Engineer Agent', cnLabel: '仿真工程' },
  worker_validator: { name: 'SBML Validator Agent', cnLabel: 'SBML 验证' },
  worker_report: { name: 'Scientific Report Agent', cnLabel: '报告生成' },
  clarification_node: { name: 'Human-in-the-Loop', cnLabel: '人工干预' },
};

/** 英文 Agent 名 → 中文标签反查表（供 TopBar 等组件根据 currentAgent 反查中文标签） */
export const AGENT_NAME_TO_CN_LABEL: Record<string, string> = Object.values(
  NODE_NAME_TO_AGENT
).reduce((acc, info) => {
  if (!acc[info.name]) acc[info.name] = info.cnLabel;
  return acc;
}, {} as Record<string, string>);

// ---------------------------------------------------------------------------
// 阶段时间轴节点定义
// ---------------------------------------------------------------------------

/** 阶段时间轴单个节点定义 */
export interface StageNode {
  /** 节点名（如 'worker_mcp'） */
  key: string;
  /** 中文名（如 '术语标准化'） */
  label: string;
  /** "正在做什么"文案 */
  actionText: string;
  /** lucide 图标名 */
  iconName: string;
}

/** 按 v3 工作流顺序排列的阶段时间轴节点 */
export const STAGE_NODES: StageNode[] = [
  { key: 'pre_router', label: '运行模式分析', actionText: '正在分析运行模式并生成执行计划...', iconName: 'Route' },
  { key: 'worker_mcp', label: '术语标准化', actionText: '正在查询生物医学术语定义...', iconName: 'BookOpen' },
  { key: 'worker_mechanism', label: '机制解析与图谱', actionText: '正在解析信号通路机制与构建知识图谱...', iconName: 'Network' },
  { key: 'worker_rag', label: '知识检索 (RAG)', actionText: '正在从 RAG 数据库检索参数与文献...', iconName: 'Search' },
  { key: 'worker_pkpd', label: 'PK/PD 推断', actionText: '正在推断 PK/PD 参数...', iconName: 'Syringe' },
  { key: 'worker_ode', label: 'ODE 方程生成', actionText: '正在生成 ODE 方程代码...', iconName: 'Code' },
  { key: 'worker_sandbox', label: '沙箱仿真执行', actionText: '正在沙箱中执行仿真...', iconName: 'FlaskConical' },
  { key: 'worker_report', label: '预测报告生成', actionText: '正在生成预测报告...', iconName: 'FileText' },
];

// ---------------------------------------------------------------------------
// 消息类型定义
// ---------------------------------------------------------------------------

/** 聊天消息角色 */
export type MessageRole = 'user' | 'agent' | 'system';

/** 单条聊天消息 */
export interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;
  timestamp: number;
  /** 如果有图片 */
  imageBase64?: string;
  /** 如果有代码 */
  code?: string;
}

// ---------------------------------------------------------------------------
// 阶段时间轴状态定义
// ---------------------------------------------------------------------------

/** 阶段时间轴节点状态 */
export type StageStatus = 'pending' | 'running' | 'done' | 'failed';

/** 阶段时间轴节点运行时状态 */
export interface StageState {
  key: string;
  status: StageStatus;
  startTime?: number;
  endTime?: number;
  durationMs?: number;
}

// ---------------------------------------------------------------------------
// 连接状态
// ---------------------------------------------------------------------------

/** 与后端的 SSE 连接状态 */
export type ConnectionStatus = 'disconnected' | 'connecting' | 'connected' | 'error';
