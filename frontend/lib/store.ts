/**
 * Global BioDynamics v4 workbench store (Zustand).
 *
 * Owns:
 * - The migrated AI-Assistant chat state + the full SSE event-ingestion logic
 *   (lifted verbatim from the legacy `app/page.tsx` switch statement so the
 *   existing `/api/chat` v3 contract keeps working bit-for-bit).
 * - The new v4 Scientific-Workspace domain state (current pathway, simulation
 *   result, validation report, hypothesis list, pathway graph, agent dispatches).
 * - UI state (which panes are open/collapsed).
 *
 * Type-only imports from `@/components/ai_assistant/*` are erased at compile
 * time, so there is no runtime cycle with the AI Assistant panel.
 */

import { create } from "zustand";
import type { MCPToolCall } from "@/components/ai_assistant/MCPToolPanel";
import type { RAGInsightsData } from "@/components/ai_assistant/RAGInsightPanel";
import type { TermDefinition } from "@/components/ai_assistant/TermDefinitionCard";
import type { DoseResponseData } from "@/components/ai_assistant/DoseResponseCurve";
import type {
  AgentState,
  AgentStatus,
} from "@/components/ai_assistant/AgentWorkflowTracker";
import type { V2PipelineStep } from "@/components/ai_assistant/WorkflowVisualization";
import type {
  ClarificationRequest,
  ClarificationAnswer,
} from "@/components/ai_assistant/ClarificationDialog";
import {
  clearChatMemory,
  fetchModelStatus,
  fetchRagStatus,
  respondClarification,
  stopChat,
  updateVectorDb,
  type ModelStatusData,
  type RagStatusData,
  type PathwayClass,
  type PathwayGraphData,
  type SimulationResult,
} from "@/lib/api";
import { streamChat, type SSEEvent } from "@/lib/sse";

// ---------------------------------------------------------------------------
// Shared UI / state types
// ---------------------------------------------------------------------------

export type RunMode = "auto_fast" | "auto_standard" | "manual";

export interface ControlBarState {
  mode: RunMode;
  manualModules: string[];
}

/** A single chat message — superset of all v1/v2/v3/v4 SSE payloads. */
export interface Message {
  id: string;
  role: "user" | "agent";
  content: string;
  type:
    | "text"
    | "code"
    | "image"
    | "log"
    | "status"
    | "report"
    | "rag_insights"
    | "mcp_tools"
    | "mcp_terms"
    | "dose_response"
    | "combination_synergy"
    | "pkpd_profile";
  tokenUsage?: number;
  ragInsights?: RAGInsightsData;
  ragHitRate?: number;
  latencyMs?: number;
  mcpToolCalls?: MCPToolCall[];
  mcpTokensSaved?: number;
  mcpTermDefinitions?: TermDefinition[];
  doseResponseData?: DoseResponseData;
  synergyData?: {
    synergy_assessment: string;
    combination_index: Record<string, number>;
    drug_regimen: Array<{
      drug_name: string;
      dose: number;
      ec50: number;
      emax: number;
      gamma: number;
      target: string;
    }>;
  };
  pkpdProfile?: {
    drug_name: string;
    drug_target: string;
    route: string;
    compartment: string;
    pk_params: Record<string, number>;
    pd_params: Record<string, number>;
  };
  knowledgeGraph?: {
    node_count: number;
    edge_count: number;
    is_acyclic: boolean;
    topology_signature: string;
  };
  metrics?: {
    species?: Record<string, unknown>;
    overall?: Record<string, unknown>;
    combo?: Record<string, unknown>;
  };
  experimentProtocols?: Array<Record<string, unknown>>;
  paperEvidence?: Array<Record<string, unknown>>;
  v2Report?: {
    markdown: string;
    llm_filled_json?: Record<string, unknown>;
    forbidden_terms_violations?: string[];
  };
  modelName?: string;
}

/** `agent_dispatch` SSE event payload. */
interface DispatchData {
  target_agent: string;
  reasoning?: string;
  status: string;
  latency_ms?: number;
  node_name?: string;
}

/** `agent_registry` SSE event payload. */
interface AgentRegistryItem {
  name: string;
  cn_label: string;
  description: string;
  icon: string;
  mapped_node: string;
}

// ---------------------------------------------------------------------------
// Module-level (non-reactive) stream accumulators
// ---------------------------------------------------------------------------

let abortController: AbortController | null = null;
/** MCP tool-call accumulator for the current stream (mirrors legacy ref). */
let mcpToolCallsAcc: MCPToolCall[] = [];
/** code_generated counter for the current stream (retry detection). */
let streamCodeGenCount = 0;

function generateId() {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

// ---------------------------------------------------------------------------
// Store shape
// ---------------------------------------------------------------------------

export interface UIState {
  /** AI Assistant pane is collapsible and collapsed by default. */
  aiAssistantOpen: boolean;
}

export interface WorkbenchStore {
  // --- v4 Scientific Workspace domain state ---
  currentPathway: PathwayClass | null;
  simulationResult: SimulationResult | null;
  validationReport: unknown | null;
  hypothesisList: unknown[];
  pathwayGraph: PathwayGraphData | null;
  agentDispatches: DispatchData[];

  // --- AI Assistant chat state (migrated from page.tsx) ---
  messages: Message[];
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

  // --- admin / control state ---
  ragStatus: RagStatusData | null;
  modelStatus: ModelStatusData | null;
  controlBarState: ControlBarState;
  isUpdatingDb: boolean;
  updateDbStatus: string;

  // --- UI state ---
  uiState: UIState;

  // --- actions: UI / domain ---
  setInput: (value: string) => void;
  setControlBarState: (state: ControlBarState) => void;
  toggleAIPanel: () => void;
  setAIPanelOpen: (open: boolean) => void;
  setCurrentPathway: (pathway: PathwayClass | null) => void;
  setPathwayGraph: (graph: PathwayGraphData | null) => void;
  setSimulationResult: (result: SimulationResult | null) => void;

  // --- actions: admin ---
  refreshRagStatus: () => Promise<void>;
  refreshModelStatus: () => Promise<void>;
  handleUpdateVectorDb: () => Promise<void>;

  // --- actions: chat lifecycle ---
  sendMessage: (text: string) => Promise<void>;
  stopGeneration: () => Promise<void>;
  clearMemory: () => Promise<void>;
  submitClarification: (answer: ClarificationAnswer) => Promise<void>;

  // --- actions: SSE ingestion (internal, exported for tests) ---
  ingestSSEEvent: (event: SSEEvent) => void;
}

export const useWorkbenchStore = create<WorkbenchStore>((set, get) => {
  // --- internal message helpers (operate on current messages) ---
  const appendMessage = (message: Omit<Message, "id">) => {
    set((state) => ({
      messages: [...state.messages, { ...message, id: generateId() }],
    }));
  };

  const updateLastStatus = (content: string) => {
    set((state) => {
      const prev = state.messages;
      const last = prev[prev.length - 1];
      if (last && last.role === "agent" && last.type === "status") {
        return { messages: [...prev.slice(0, -1), { ...last, content }] };
      }
      return {
        messages: [
          ...prev,
          { id: generateId(), role: "agent", content, type: "status" },
        ],
      };
    });
  };

  const removeTrailingStatus = () => {
    set((state) => {
      let i = state.messages.length - 1;
      while (
        i >= 0 &&
        state.messages[i].role === "agent" &&
        state.messages[i].type === "status"
      ) {
        i -= 1;
      }
      return { messages: state.messages.slice(0, i + 1) };
    });
  };

  const setLastAgentTokenUsage = (usage: number, model?: string) => {
    set((state) => {
      const prev = state.messages;
      for (let i = prev.length - 1; i >= 0; i--) {
        if (prev[i].role === "agent") {
          const updated = [...prev];
          updated[i] = { ...updated[i], tokenUsage: usage, modelName: model };
          return { messages: updated };
        }
      }
      return { messages: prev };
    });
  };

  const attachMetricsToLastAgent = (metrics: {
    ragHitRate?: number;
    latencyMs?: number;
  }) => {
    set((state) => {
      const prev = state.messages;
      for (let i = prev.length - 1; i >= 0; i--) {
        if (prev[i].role === "agent") {
          const updated = [...prev];
          updated[i] = {
            ...updated[i],
            ragHitRate: metrics.ragHitRate ?? updated[i].ragHitRate,
            latencyMs: metrics.latencyMs ?? updated[i].latencyMs,
          };
          return { messages: updated };
        }
      }
      return { messages: prev };
    });
  };

  const handleAgentDispatch = (dispatch: DispatchData) => {
    const status = dispatch.status as AgentStatus;
    set((state) => {
      const exists = state.agents.some((a) => a.name === dispatch.target_agent);
      if (!exists) {
        return {
          agents: [
            ...state.agents,
            {
              name: dispatch.target_agent,
              cn_label: dispatch.target_agent.replace(" Agent", ""),
              description: dispatch.reasoning || "",
              icon: "file",
              status,
              reasoning: dispatch.reasoning,
              latency_ms: dispatch.latency_ms,
            },
          ],
          agentDispatches: [...state.agentDispatches, dispatch],
        };
      }
      return {
        agents: state.agents.map((a) =>
          a.name === dispatch.target_agent
            ? {
                ...a,
                status,
                reasoning: dispatch.reasoning || a.reasoning,
                latency_ms: dispatch.latency_ms ?? a.latency_ms,
              }
            : a
        ),
        agentDispatches: [...state.agentDispatches, dispatch],
      };
    });

    if (dispatch.latency_ms !== undefined && dispatch.latency_ms > 0) {
      set({ lastLatencyMs: dispatch.latency_ms });
      attachMetricsToLastAgent({ latencyMs: dispatch.latency_ms });
    }
  };

  // --- the big SSE event switch (ported from legacy page.tsx) ---
  const ingestSSEEvent = (event: SSEEvent) => {
    const eventType = event.event;
    const eventData = event.data;
    const state = get();

    switch (eventType) {
      case "config": {
        if (typeof eventData === "object" && eventData !== null) {
          const cfg = eventData as { model_name?: string };
          if (cfg.model_name) set({ modelName: cfg.model_name });
        }
        break;
      }

      case "agent_registry": {
        if (Array.isArray(eventData)) {
          const registry = eventData as AgentRegistryItem[];
          set({
            agents: registry.map((item) => ({
              name: item.name,
              cn_label: item.cn_label,
              description: item.description,
              icon: item.icon,
              status: "idle" as AgentStatus,
            })),
          });
        }
        break;
      }

      case "workflow_v3_state": {
        if (eventData && typeof eventData === "object") {
          const data = eventData as {
            current_node?: string;
            status?: string;
            mode?: string;
          };
          if (data.current_node) set({ currentNode: data.current_node });
        }
        break;
      }

      case "agent_dispatch": {
        if (eventData && typeof eventData === "object") {
          handleAgentDispatch(eventData as DispatchData);
        }
        break;
      }

      case "clarification_needed": {
        if (eventData && typeof eventData === "object") {
          set({ clarification: eventData as ClarificationRequest });
        }
        break;
      }

      case "clarification_resolved": {
        set({ clarification: null });
        break;
      }

      case "rag_insights": {
        if (eventData && typeof eventData === "object") {
          removeTrailingStatus();
          appendMessage({
            role: "agent",
            content: "",
            type: "rag_insights",
            ragInsights: eventData as RAGInsightsData,
          });
        }
        break;
      }

      case "mcp_tool_call": {
        if (eventData && typeof eventData === "object") {
          const call = eventData as MCPToolCall;
          mcpToolCallsAcc = [...mcpToolCallsAcc, call];
          updateLastStatus(`${call.icon} [MCP] ${call.tool_name}：${call.action}`);
        }
        break;
      }

      case "mcp_term_definitions": {
        if (eventData && typeof eventData === "object") {
          const data = eventData as {
            definitions?: TermDefinition[];
            tokens_saved?: number;
            rewritten_query?: string;
          };
          if (data.definitions && data.definitions.length > 0) {
            removeTrailingStatus();
            if (mcpToolCallsAcc.length > 0) {
              appendMessage({
                role: "agent",
                content: "",
                type: "mcp_tools",
                mcpToolCalls: mcpToolCallsAcc,
                mcpTokensSaved: data.tokens_saved ?? 0,
              });
            }
            appendMessage({
              role: "agent",
              content: "",
              type: "mcp_terms",
              mcpTermDefinitions: data.definitions,
            });
            if (data.tokens_saved !== undefined) {
              set({ mcpTokensSavedTotal: data.tokens_saved });
            }
          }
        }
        break;
      }

      case "node_start": {
        const text = typeof eventData === "string" ? eventData : "正在处理...";
        updateLastStatus(text);
        break;
      }

      case "code_generated": {
        streamCodeGenCount += 1;
        removeTrailingStatus();
        const code = typeof eventData === "string" ? eventData : "";
        appendMessage({ role: "agent", content: code, type: "code" });
        break;
      }

      case "execution_log": {
        removeTrailingStatus();
        const rawLog = typeof eventData === "string" ? eventData : "";
        const isRetry = streamCodeGenCount > 1;
        const content = isRetry
          ? `⚠️ 仿真出错，正在自动纠错重试 (${streamCodeGenCount - 1}/3)...\n${rawLog}`
          : rawLog;
        appendMessage({ role: "agent", content, type: "log" });
        break;
      }

      case "image_ready": {
        removeTrailingStatus();
        const imageBase64 = typeof eventData === "string" ? eventData : "";
        appendMessage({ role: "agent", content: imageBase64, type: "image" });
        break;
      }

      case "rag_ready": {
        if (eventData && typeof eventData === "object") {
          const data = eventData as { hit_rate?: number; summary?: string };
          if (data.hit_rate !== undefined) {
            set({ lastRagHitRate: data.hit_rate });
            attachMetricsToLastAgent({ ragHitRate: data.hit_rate });
          }
        }
        break;
      }

      case "rag_online_fallback": {
        if (eventData && typeof eventData === "object") {
          const data = eventData as {
            triggered?: boolean;
            hit_rate?: number;
            message?: string;
          };
          if (data.triggered) {
            appendMessage({
              role: "agent",
              content:
                data.message || "本地 RAG 命中不足，已自动查询在线数据库补充",
              type: "status",
            });
          }
        }
        break;
      }

      case "pkpd_profile": {
        if (eventData && typeof eventData === "object") {
          removeTrailingStatus();
          appendMessage({
            role: "agent",
            content: "",
            type: "pkpd_profile",
            pkpdProfile: eventData as Message["pkpdProfile"],
          });
        }
        break;
      }

      case "drug_regimen": {
        // Drug regimen data is rendered inside combination_synergy; no-op here.
        break;
      }

      case "dose_response": {
        if (eventData && typeof eventData === "object") {
          removeTrailingStatus();
          appendMessage({
            role: "agent",
            content: "",
            type: "dose_response",
            doseResponseData: eventData as DoseResponseData,
          });
        }
        break;
      }

      case "combination_synergy": {
        if (eventData && typeof eventData === "object") {
          removeTrailingStatus();
          appendMessage({
            role: "agent",
            content: "",
            type: "combination_synergy",
            synergyData: eventData as Message["synergyData"],
          });
        }
        break;
      }

      case "report_ready": {
        removeTrailingStatus();
        const report = typeof eventData === "string" ? eventData : "";
        appendMessage({
          role: "agent",
          content: report,
          type: "report",
          ragHitRate: state.lastRagHitRate,
          latencyMs: state.lastLatencyMs,
          mcpTokensSaved: state.mcpTokensSavedTotal,
        });
        break;
      }

      case "token_usage": {
        if (typeof eventData === "object" && eventData !== null) {
          const usage = eventData as {
            total_tokens?: number;
            mcp_tokens_saved?: number;
            model_name?: string;
          };
          const total = usage.total_tokens ?? 0;
          set({ tokenUsage: total });
          if (usage.model_name) set({ modelName: usage.model_name });
          setLastAgentTokenUsage(total, usage.model_name);
          if (usage.mcp_tokens_saved !== undefined) {
            set({ mcpTokensSavedTotal: usage.mcp_tokens_saved });
          }
        }
        break;
      }

      case "error": {
        removeTrailingStatus();
        const errorText =
          typeof eventData === "string" ? eventData : "发生未知错误";
        appendMessage({ role: "agent", content: errorText, type: "text" });
        break;
      }

      case "end": {
        removeTrailingStatus();
        break;
      }

      // ---- v2 workflow events ----
      case "workflow_v2_state": {
        if (eventData && typeof eventData === "object") {
          const data = eventData as {
            current_node?: string;
            step_index?: number;
            total_steps?: number;
            pipeline?: "v1" | "v2";
            status?: "starting" | "running" | "completed" | "failed";
          };
          set((s) => ({
            pipelineName: data.pipeline ?? s.pipelineName,
            pipelineStatus: data.status ?? s.pipelineStatus,
            pipelineStepIndex:
              typeof data.step_index === "number"
                ? data.step_index
                : s.pipelineStepIndex,
            pipelineTotal:
              typeof data.total_steps === "number"
                ? data.total_steps
                : s.pipelineTotal,
          }));
          if (data.current_node !== undefined) {
            const node = data.current_node;
            set({ pipelineCurrent: node });
            if (node !== "") {
              set((s) => {
                const exists = s.pipelineSteps.find(
                  (st) => st.node_name === node
                );
                if (exists) {
                  return {
                    pipelineSteps: s.pipelineSteps.map((st) =>
                      st.node_name === node
                        ? { ...st, status: "running" }
                        : st
                    ),
                  };
                }
                return {
                  pipelineSteps: [
                    ...s.pipelineSteps,
                    {
                      node_name: node,
                      cn_label: node,
                      icon: "cpu",
                      status: "running",
                    },
                  ],
                };
              });
            }
          }
        }
        break;
      }

      case "knowledge_graph": {
        if (eventData && typeof eventData === "object") {
          const kg = eventData as Message["knowledgeGraph"];
          removeTrailingStatus();
          appendMessage({
            role: "agent",
            content: `📊 知识图谱：${kg?.node_count ?? 0} 节点 / ${kg?.edge_count ?? 0} 边（拓扑 ${kg?.topology_signature ?? "?"}, 无环=${kg?.is_acyclic ?? "?"}）`,
            type: "text",
            knowledgeGraph: kg,
          });
        }
        break;
      }

      case "rule_violations": {
        if (Array.isArray(eventData)) {
          const violations = eventData as Array<{
            rule_name: string;
            edge_key?: string | null;
            message: string;
            severity: string;
          }>;
          removeTrailingStatus();
          appendMessage({
            role: "agent",
            content: `⚠️ Rule Engine 违规 ${violations.length} 条：\n${violations
              .map((v) => `- [${v.severity}] ${v.rule_name}: ${v.message}`)
              .join("\n")}`,
            type: "text",
          });
        }
        break;
      }

      case "metrics": {
        if (eventData && typeof eventData === "object") {
          const metrics = eventData as Message["metrics"];
          removeTrailingStatus();
          const species = metrics?.species || {};
          const lines = Object.entries(species)
            .slice(0, 5)
            .map(
              ([sp, m]) =>
                `- ${sp}: peak=${(m as { peak?: number }).peak?.toFixed(2) ?? "?"}, t_peak=${(m as { peak_time?: number }).peak_time?.toFixed(2) ?? "?"}h`
            )
            .join("\n");
          appendMessage({
            role: "agent",
            content: `📈 科学特征提取完成（${Object.keys(species).length} 个物种）：\n${lines}`,
            type: "text",
            metrics,
          });
        }
        break;
      }

      case "experiment_protocols": {
        if (Array.isArray(eventData)) {
          const protocols = eventData as Array<Record<string, unknown>>;
          removeTrailingStatus();
          appendMessage({
            role: "agent",
            content: `🧪 实验方案推荐（${protocols.length} 个）：\n${protocols
              .slice(0, 5)
              .map((p) => `- ${p.name ?? "?"} (${p.detection_method ?? "?"})`)
              .join("\n")}`,
            type: "text",
            experimentProtocols: protocols,
          });
        }
        break;
      }

      case "paper_evidence": {
        if (Array.isArray(eventData)) {
          const evidence = eventData as Array<Record<string, unknown>>;
          removeTrailingStatus();
          appendMessage({
            role: "agent",
            content: `📚 文献证据（${evidence.length} 篇）：\n${evidence
              .slice(0, 5)
              .map((e) => `- ${e.title ?? "?"} (PMID:${e.pmid ?? "?"})`)
              .join("\n")}`,
            type: "text",
            paperEvidence: evidence,
          });
        }
        break;
      }

      case "report": {
        if (eventData && typeof eventData === "object") {
          const data = eventData as Message["v2Report"];
          removeTrailingStatus();
          if (data?.markdown) {
            appendMessage({
              role: "agent",
              content: data.markdown,
              type: "report",
              v2Report: data,
              ragHitRate: state.lastRagHitRate,
              latencyMs: state.lastLatencyMs,
              mcpTokensSaved: state.mcpTokensSavedTotal,
            });
          }
        }
        break;
      }

      case "simulation_csv": {
        if (typeof eventData === "string" && eventData) {
          updateLastStatus(`📁 simulation.csv 已生成：${eventData}`);
        }
        break;
      }

      // ---- v4 events (forwarded by the v3→v4 adapter; handled in C.7–C.8) ----
      case "v4_hypothesis_generated":
      case "v4_validation_report":
      case "v4_pathway_graph":
      case "v4_simulation_result":
        // TODO(C.7/C.8): hydrate validationReport / hypothesisList / pathwayGraph.
        break;

      default:
        break;
    }
  };

  return {
    // --- v4 domain state ---
    currentPathway: null,
    simulationResult: null,
    validationReport: null,
    hypothesisList: [],
    pathwayGraph: null,
    agentDispatches: [],

    // --- chat state ---
    messages: [],
    input: "",
    isStreaming: false,
    threadId: generateId(),
    tokenUsage: 0,
    modelName: "",
    agents: [],
    clarification: null,
    currentNode: undefined,
    lastRagHitRate: undefined,
    lastLatencyMs: undefined,
    mcpTokensSavedTotal: undefined,
    pipelineSteps: [],
    pipelineCurrent: "",
    pipelineStepIndex: 0,
    pipelineTotal: 0,
    pipelineName: "v1",
    pipelineStatus: "starting",

    // --- admin / control ---
    ragStatus: null,
    modelStatus: null,
    controlBarState: { mode: "auto_standard", manualModules: [] },
    isUpdatingDb: false,
    updateDbStatus: "",

    // --- UI state: AI Assistant collapsed by default ---
    uiState: { aiAssistantOpen: false },

    // --- actions: UI / domain ---
    setInput: (value) => set({ input: value }),
    setControlBarState: (next) => set({ controlBarState: next }),
    toggleAIPanel: () =>
      set((s) => ({
        uiState: {
          ...s.uiState,
          aiAssistantOpen: !s.uiState.aiAssistantOpen,
        },
      })),
    setAIPanelOpen: (open) =>
      set((s) => ({ uiState: { ...s.uiState, aiAssistantOpen: open } })),
    setCurrentPathway: (pathway) => set({ currentPathway: pathway }),
    setPathwayGraph: (graph) => set({ pathwayGraph: graph }),
    setSimulationResult: (result) => set({ simulationResult: result }),

    // --- actions: admin ---
    refreshRagStatus: async () => {
      try {
        const data = await fetchRagStatus();
        set({ ragStatus: data });
      } catch (err) {
        console.error("获取 RAG 状态失败", err);
      }
    },
    refreshModelStatus: async () => {
      try {
        const data = await fetchModelStatus();
        set({ modelStatus: data });
      } catch (err) {
        console.error("[Workbench] 获取模型状态失败:", err);
      }
    },
    handleUpdateVectorDb: async () => {
      set({ isUpdatingDb: true, updateDbStatus: "" });
      try {
        const response = await updateVectorDb();
        if (!response.ok) throw new Error(`请求失败：${response.status}`);
        set({ updateDbStatus: "知识库更新已启动" });
        setTimeout(() => get().refreshRagStatus(), 2000);
      } catch (err) {
        const message = err instanceof Error ? err.message : "更新失败";
        set({ updateDbStatus: message });
      } finally {
        set({ isUpdatingDb: false });
        setTimeout(() => set({ updateDbStatus: "" }), 3000);
      }
    },

    // --- actions: chat lifecycle ---
    sendMessage: async (text) => {
      const userInput = text.trim();
      const state = get();
      if (!userInput || state.isStreaming) return;

      // New thread per query to keep LangGraph checkpointer state isolated.
      const newThreadId = generateId();
      set({ threadId: newThreadId });

      appendMessage({ role: "user", content: userInput, type: "text" });
      set({
        input: "",
        isStreaming: true,
        agents: [],
        lastRagHitRate: undefined,
        lastLatencyMs: undefined,
        mcpTokensSavedTotal: undefined,
        clarification: null,
        currentNode: undefined,
        pipelineSteps: [],
        pipelineCurrent: "",
        pipelineStepIndex: 0,
        pipelineTotal: 0,
        pipelineName: "v1",
        pipelineStatus: "starting",
      });
      mcpToolCallsAcc = [];
      streamCodeGenCount = 0;

      abortController = new AbortController();
      await streamChat(
        {
          user_input: userInput,
          thread_id: newThreadId,
          mode: state.controlBarState.mode,
          manual_modules: state.controlBarState.manualModules,
        },
        {
          onEvent: (event) => get().ingestSSEEvent(event),
          onError: (err) => {
            removeTrailingStatus();
            appendMessage({
              role: "agent",
              content: err.message,
              type: "text",
            });
          },
          onDone: () => {
            set({ isStreaming: false });
            abortController = null;
          },
        },
        abortController.signal
      );
    },

    stopGeneration: async () => {
      abortController?.abort();
      const { threadId } = get();
      try {
        await stopChat(threadId);
      } catch (err) {
        console.error("停止生成请求失败", err);
      }
      set({ isStreaming: false, clarification: null });
    },

    clearMemory: async () => {
      const { threadId } = get();
      try {
        await clearChatMemory(threadId);
      } catch (err) {
        console.error("清空记忆失败", err);
      }
      mcpToolCallsAcc = [];
      streamCodeGenCount = 0;
      set({
        messages: [],
        agents: [],
        threadId: generateId(),
        tokenUsage: 0,
        modelName: "",
        lastRagHitRate: undefined,
        lastLatencyMs: undefined,
        mcpTokensSavedTotal: undefined,
        clarification: null,
        currentNode: undefined,
        pipelineSteps: [],
        pipelineCurrent: "",
        pipelineStepIndex: 0,
        pipelineTotal: 0,
        pipelineName: "v1",
        pipelineStatus: "starting",
        agentDispatches: [],
      });
    },

    submitClarification: async (answer) => {
      const { threadId } = get();
      try {
        await respondClarification(threadId, answer);
      } catch (err) {
        console.error("提交人工干预回答失败", err);
      }
    },

    ingestSSEEvent,
  };
});
