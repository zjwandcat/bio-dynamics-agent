"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import { ChatMessage, ChatMessageProps } from "@/components/chat/ChatMessage";
import { ChatInput } from "@/components/chat/ChatInput";
import { Button } from "@/components/ui/button";
import { Trash2, RefreshCw } from "lucide-react";
import {
  AgentWorkflowTracker,
  type AgentState,
  type AgentStatus,
} from "@/components/chat/AgentWorkflowTracker";
import {
  WorkflowVisualization,
  type V2PipelineStep,
} from "@/components/chat/WorkflowVisualization";
import { ControlBar, ControlBarState } from "@/components/chat/ControlBar";
import type { ClarificationRequest, ClarificationAnswer } from "@/components/chat/ClarificationDialog";
import type { RAGInsightsData } from "@/components/chat/RAGInsightPanel";
import type { MCPToolCall } from "@/components/chat/MCPToolPanel";
import type { TermDefinition } from "@/components/chat/TermDefinitionCard";
import type { DoseResponseData } from "@/components/chat/DoseResponseCurve";

interface Message {
  id: string;
  role: ChatMessageProps["role"];
  content: string;
  type: ChatMessageProps["type"];
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
  // v2 字段
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

function generateId() {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

const API_BASE = "http://localhost:8000";

// 后端 agent_dispatch 事件的数据结构
interface DispatchData {
  target_agent: string;
  reasoning?: string;
  status: string;
  latency_ms?: number;
  node_name?: string;
}

// 后端 agent_registry 下发的智能体元信息
interface AgentRegistryItem {
  name: string;
  cn_label: string;
  description: string;
  icon: string;
  mapped_node: string;
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [threadId, setThreadId] = useState(() => generateId());
  const [tokenUsage, setTokenUsage] = useState(0);
  // 当前后端使用的 LLM 模型名（从 SSE config/token_usage 事件获取，展示真实模型而非硬编码）
  const [modelName, setModelName] = useState<string>("");
  const [isUpdatingDb, setIsUpdatingDb] = useState(false);
  const [updateDbStatus, setUpdateDbStatus] = useState("");
  // RAG 知识库状态（已加载数据库列表 + collection 文档数）
  const [ragStatus, setRagStatus] = useState<{
    databases: Array<{ name: string; type: string; collection?: string; files?: string[] }>;
    collections: Record<string, number>;
    online_fallback_enabled: boolean;
    online_fallback_threshold: number;
  } | null>(null);
  // v3 右侧控制栏状态
  const [controlBarState, setControlBarState] = useState<ControlBarState>({
    mode: "auto_standard",
    manualModules: [],
  });
  // v3 人在环路干预状态
  const [clarification, setClarification] = useState<ClarificationRequest | null>(null);
  const [currentNode, setCurrentNode] = useState<string | undefined>(undefined);
  // 智能体工作流追踪器状态
  const [agents, setAgents] = useState<AgentState[]>([]);
  // 最新性能指标（供最后一条 agent 消息的 TokenPerformanceBadge 展示）
  const [lastRagHitRate, setLastRagHitRate] = useState<number | undefined>(undefined);
  const [lastLatencyMs, setLastLatencyMs] = useState<number | undefined>(undefined);
  // MCP 工具调用累加器：流式过程中收集所有工具调用记录，结束后插入面板消息
  // 使用 ref 避免闭包陈旧引用问题（SSE 事件跨 tick 到达时 state 可能未更新）
  const mcpToolCallsAccRef = useRef<MCPToolCall[]>([]);
  // MCP Token 节省总量（供最终报告消息的 TokenPerformanceBadge 展示）
  const [mcpTokensSavedTotal, setMcpTokensSavedTotal] = useState<number | undefined>(undefined);
  // v2 工作流可视化状态
  const [pipelineSteps, setPipelineSteps] = useState<V2PipelineStep[]>([]);
  const [pipelineCurrent, setPipelineCurrent] = useState<string>("");
  const [pipelineStepIndex, setPipelineStepIndex] = useState(0);
  const [pipelineTotal, setPipelineTotal] = useState(0);
  const [pipelineName, setPipelineName] = useState<"v1" | "v2">("v1");
  const [pipelineStatus, setPipelineStatus] = useState<"starting" | "running" | "completed" | "failed">("starting");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  const appendMessage = useCallback((message: Omit<Message, "id">) => {
    setMessages((prev) => [...prev, { ...message, id: generateId() }]);
  }, []);

  const updateLastStatus = useCallback((content: string) => {
    setMessages((prev) => {
      const last = prev[prev.length - 1];
      if (last && last.role === "agent" && last.type === "status") {
        return [...prev.slice(0, -1), { ...last, content }];
      }
      return [
        ...prev,
        { id: generateId(), role: "agent" as const, content, type: "status" as const },
      ];
    });
  }, []);

  const removeTrailingStatus = useCallback(() => {
    setMessages((prev) => {
      let i = prev.length - 1;
      while (i >= 0 && prev[i].role === "agent" && prev[i].type === "status") {
        i -= 1;
      }
      return prev.slice(0, i + 1);
    });
  }, []);

  const setLastAgentTokenUsage = useCallback((usage: number, model?: string) => {
    setMessages((prev) => {
      for (let i = prev.length - 1; i >= 0; i--) {
        if (prev[i].role === "agent") {
          const updated = [...prev];
          updated[i] = { ...updated[i], tokenUsage: usage, modelName: model };
          return updated;
        }
      }
      return prev;
    });
  }, []);

  // 将性能指标附加到最后一条 agent 消息
  const attachMetricsToLastAgent = useCallback(
    (metrics: { ragHitRate?: number; latencyMs?: number }) => {
      setMessages((prev) => {
        for (let i = prev.length - 1; i >= 0; i--) {
          if (prev[i].role === "agent") {
            const updated = [...prev];
            updated[i] = {
              ...updated[i],
              ragHitRate: metrics.ragHitRate ?? updated[i].ragHitRate,
              latencyMs: metrics.latencyMs ?? updated[i].latencyMs,
            };
            return updated;
          }
        }
        return prev;
      });
    },
    []
  );

  // 处理 agent_dispatch 事件：更新工作流追踪器中对应智能体的状态
  const handleAgentDispatch = useCallback((dispatch: DispatchData) => {
    const status = dispatch.status as AgentStatus;
    setAgents((prev) => {
      // 若该智能体不在列表中（如 Report Generator），动态追加
      const exists = prev.some((a) => a.name === dispatch.target_agent);
      if (!exists) {
        return [
          ...prev,
          {
            name: dispatch.target_agent,
            cn_label: dispatch.target_agent.replace(" Agent", ""),
            description: dispatch.reasoning || "",
            icon: "file",
            status,
            reasoning: dispatch.reasoning,
            latency_ms: dispatch.latency_ms,
          },
        ];
      }
      return prev.map((a) =>
        a.name === dispatch.target_agent
          ? {
              ...a,
              status,
              reasoning: dispatch.reasoning || a.reasoning,
              latency_ms: dispatch.latency_ms ?? a.latency_ms,
            }
          : a
      );
    });

    // 记录最新延迟供 TokenPerformanceBadge 展示
    if (dispatch.latency_ms !== undefined && dispatch.latency_ms > 0) {
      setLastLatencyMs(dispatch.latency_ms);
      attachMetricsToLastAgent({ latencyMs: dispatch.latency_ms });
    }
  }, [attachMetricsToLastAgent]);

  const handleClearMemory = useCallback(async () => {
    try {
      await fetch(`${API_BASE}/api/chat/clear-memory`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ thread_id: threadId }),
      });
    } catch (err) {
      console.error("清空记忆失败", err);
    }
    // 彻底重置所有面板与对话状态
    setMessages([]);
    setAgents([]);
    setThreadId(generateId());
    setTokenUsage(0);
    setModelName("");
    setLastRagHitRate(undefined);
    setLastLatencyMs(undefined);
    mcpToolCallsAccRef.current = [];
    setMcpTokensSavedTotal(undefined);
    setClarification(null);
    setCurrentNode(undefined);
    setPipelineSteps([]);
    setPipelineCurrent("");
    setPipelineStepIndex(0);
    setPipelineTotal(0);
    setPipelineName("v1");
    setPipelineStatus("starting");
  }, [threadId]);

  const handleUpdateVectorDb = useCallback(async () => {
    setIsUpdatingDb(true);
    setUpdateDbStatus("");
    try {
      const response = await fetch(`${API_BASE}/api/admin/update-vector-db`, {
        method: "POST",
      });
      if (!response.ok) {
        throw new Error(`请求失败：${response.status}`);
      }
      setUpdateDbStatus("知识库更新已启动");
      // 更新完成后刷新 RAG 状态
      setTimeout(fetchRagStatus, 2000);
    } catch (err) {
      const message = err instanceof Error ? err.message : "更新失败";
      setUpdateDbStatus(message);
    } finally {
      setIsUpdatingDb(false);
      setTimeout(() => setUpdateDbStatus(""), 3000);
    }
  }, []);

  // 获取 RAG 知识库状态
  const fetchRagStatus = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/api/admin/rag-status`);
      if (response.ok) {
        const data = await response.json();
        setRagStatus(data);
      }
    } catch (err) {
      console.error("获取 RAG 状态失败", err);
    }
  }, []);

  // 页面加载时获取 RAG 状态
  useEffect(() => {
    fetchRagStatus();
  }, [fetchRagStatus]);

  // v3：停止生成
  const handleStop = useCallback(async () => {
    abortControllerRef.current?.abort();
    try {
      await fetch(`${API_BASE}/api/chat/stop`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ thread_id: threadId }),
      });
    } catch (err) {
      console.error("停止生成请求失败", err);
    }
    setIsStreaming(false);
    setClarification(null);
  }, [threadId]);

  // v3：提交人工干预回答
  const handleClarificationSubmit = useCallback(
    async (answer: ClarificationAnswer) => {
      try {
        const response = await fetch(`${API_BASE}/api/chat/respond`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            thread_id: threadId,
            clarification_response: answer,
          }),
        });
        if (!response.ok) {
          throw new Error(`提交失败：${response.status}`);
        }
      } catch (err) {
        console.error("提交人工干预回答失败", err);
      }
    },
    [threadId]
  );

  const handleSend = useCallback(async () => {
    const userInput = input.trim();
    if (!userInput || isStreaming) return;

    // 每次新查询生成新的 thread_id，确保 LangGraph checkpointer 创建全新状态，
    // 彻底阻断 operator.add reducer 字段与 raw_cache 的跨请求数据污染。
    const newThreadId = generateId();
    setThreadId(newThreadId);

    appendMessage({ role: "user", content: userInput, type: "text" });
    setInput("");
    setIsStreaming(true);
    setAgents([]); // 重置工作流追踪器
    setLastRagHitRate(undefined);
    setLastLatencyMs(undefined);
    mcpToolCallsAccRef.current = []; // 重置 MCP 工具调用累加器
    setMcpTokensSavedTotal(undefined);
    setClarification(null);
    setCurrentNode(undefined);
    // 重置 v2 流水线状态
    setPipelineSteps([]);
    setPipelineCurrent("");
    setPipelineStepIndex(0);
    setPipelineTotal(0);
    setPipelineName("v1");
    setPipelineStatus("starting");

    let codeGenCount = 0;

    try {
      abortControllerRef.current = new AbortController();
      const response = await fetch(`${API_BASE}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_input: userInput,
          thread_id: newThreadId,
          mode: controlBarState.mode,
          manual_modules: controlBarState.manualModules,
        }),
        signal: abortControllerRef.current.signal,
      });

      if (!response.ok) {
        throw new Error(`请求失败：${response.status} ${response.statusText}`);
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error("响应流不可用");
      }

      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed.startsWith("data:")) continue;

          const payload = trimmed.slice(5).trim();
          if (payload === "[DONE]") continue;
          if (!payload) continue;

          let event: { event?: string; data?: unknown } = {};
          try {
            event = JSON.parse(payload);
          } catch {
            continue;
          }

          const eventType = event.event;
          const eventData = event.data;

          switch (eventType) {
            // 配置事件：后端下发当前使用的模型名等配置信息
            case "config": {
              if (typeof eventData === "object" && eventData !== null) {
                const cfg = eventData as { model_name?: string };
                if (cfg.model_name) {
                  setModelName(cfg.model_name);
                }
              }
              break;
            }
            // 智能体注册表：初始化工作流追踪器
            case "agent_registry": {
              if (Array.isArray(eventData)) {
                const registry = eventData as AgentRegistryItem[];
                setAgents(
                  registry.map((item) => ({
                    name: item.name,
                    cn_label: item.cn_label,
                    description: item.description,
                    icon: item.icon,
                    status: "idle" as AgentStatus,
                  }))
                );
              }
              break;
            }

            // v3 工作流状态：更新当前节点
            case "workflow_v3_state": {
              if (eventData && typeof eventData === "object") {
                const data = eventData as { current_node?: string; status?: string; mode?: string };
                if (data.current_node) {
                  setCurrentNode(data.current_node);
                }
              }
              break;
            }

            // 智能体调度事件：更新追踪器状态
            case "agent_dispatch": {
              if (eventData && typeof eventData === "object") {
                handleAgentDispatch(eventData as DispatchData);
              }
              break;
            }

            // 人工干预：需要用户回答
            case "clarification_needed": {
              if (eventData && typeof eventData === "object") {
                setClarification(eventData as ClarificationRequest);
              }
              break;
            }

            // 人工干预：已被后端消费，关闭面板
            case "clarification_resolved": {
              setClarification(null);
              break;
            }

            // RAG 洞察数据：插入 RAG 洞察面板消息
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

            // MCP 工具调用记录：累加到 ref，流结束后统一渲染面板
            case "mcp_tool_call": {
              if (eventData && typeof eventData === "object") {
                const call = eventData as MCPToolCall;
                mcpToolCallsAccRef.current = [...mcpToolCallsAccRef.current, call];
                const statusText = `${call.icon} [MCP] ${call.tool_name}：${call.action}`;
                updateLastStatus(statusText);
              }
              break;
            }

            // MCP 术语定义：插入术语定义卡片消息
            case "mcp_term_definitions": {
              if (eventData && typeof eventData === "object") {
                const data = eventData as {
                  definitions?: TermDefinition[];
                  tokens_saved?: number;
                  rewritten_query?: string;
                };
                if (data.definitions && data.definitions.length > 0) {
                  removeTrailingStatus();
                  // 先插入 MCP 工具调用面板（若已累积工具调用）
                  const accCalls = mcpToolCallsAccRef.current;
                  if (accCalls.length > 0) {
                    appendMessage({
                      role: "agent",
                      content: "",
                      type: "mcp_tools",
                      mcpToolCalls: accCalls,
                      mcpTokensSaved: data.tokens_saved ?? 0,
                    });
                  }
                  // 插入术语定义卡片
                  appendMessage({
                    role: "agent",
                    content: "",
                    type: "mcp_terms",
                    mcpTermDefinitions: data.definitions,
                  });
                  if (data.tokens_saved !== undefined) {
                    setMcpTokensSavedTotal(data.tokens_saved);
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
              codeGenCount += 1;
              removeTrailingStatus();
              const code = typeof eventData === "string" ? eventData : "";
              appendMessage({ role: "agent", content: code, type: "code" });
              break;
            }

            case "execution_log": {
              removeTrailingStatus();
              const rawLog = typeof eventData === "string" ? eventData : "";
              const isRetry = codeGenCount > 1;
              const content = isRetry
                ? `⚠️ 仿真出错，正在自动纠错重试 (${codeGenCount - 1}/3)...\n${rawLog}`
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
                  setLastRagHitRate(data.hit_rate);
                  attachMetricsToLastAgent({ ragHitRate: data.hit_rate });
                }
              }
              break;
            }

            // 在线数据库补充通知：本地 RAG 命中不足时自动查询 KEGG/Reactome/UniProt/ChEMBL
            case "rag_online_fallback": {
              if (eventData && typeof eventData === "object") {
                const data = eventData as { triggered?: boolean; hit_rate?: number; message?: string };
                if (data.triggered) {
                  appendMessage({
                    role: "agent",
                    content: data.message || "本地 RAG 命中不足，已自动查询在线数据库补充",
                    type: "status",
                  });
                }
              }
              break;
            }

            // PK/PD 推断结果：渲染 PK/PD 模型卡片
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

            // 药物方案：不单独渲染，数据已包含在 combination_synergy 中
            case "drug_regimen": {
              break;
            }

            // 剂量-反应曲线：渲染 Sigmoid Emax 曲线与 IC50/IC90/HED
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

            // 联合用药协同评估：渲染 Chou-Talalay CI
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
                ragHitRate: lastRagHitRate,
                latencyMs: lastLatencyMs,
                mcpTokensSaved: mcpTokensSavedTotal,
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
                setTokenUsage(total);
                // 同步更新模型名（token_usage 事件在流末尾携带，确保最终展示准确）
                if (usage.model_name) {
                  setModelName(usage.model_name);
                }
                setLastAgentTokenUsage(total, usage.model_name);
                // 若后端在 token_usage 事件中携带了 MCP 节省量，更新到最终报告
                if (usage.mcp_tokens_saved !== undefined) {
                  setMcpTokensSavedTotal(usage.mcp_tokens_saved);
                }
              }
              break;
            }

            case "error": {
              removeTrailingStatus();
              const errorText = typeof eventData === "string" ? eventData : "发生未知错误";
              appendMessage({ role: "agent", content: errorText, type: "text" });
              break;
            }

            case "end": {
              removeTrailingStatus();
              break;
            }

            // ---- v2 工作流事件 ----
            case "workflow_v2_state": {
              if (eventData && typeof eventData === "object") {
                const data = eventData as {
                  current_node?: string;
                  step_index?: number;
                  total_steps?: number;
                  pipeline?: "v1" | "v2";
                  status?: "starting" | "running" | "completed" | "failed";
                };
                if (data.pipeline) setPipelineName(data.pipeline);
                if (data.status) setPipelineStatus(data.status);
                if (typeof data.step_index === "number") {
                  setPipelineStepIndex(data.step_index);
                }
                if (typeof data.total_steps === "number") {
                  setPipelineTotal(data.total_steps);
                }
                if (data.current_node !== undefined) {
                  setPipelineCurrent(data.current_node);
                  setPipelineSteps((prev) => {
                    const exists = prev.find((s) => s.node_name === data.current_node);
                    if (data.current_node === "") {
                      return prev;
                    }
                    if (exists) {
                      return prev.map((s) =>
                        s.node_name === data.current_node
                          ? { ...s, status: "running" }
                          : s
                      );
                    }
                    return [
                      ...prev,
                      {
                        node_name: data.current_node!,
                        cn_label: data.current_node!,
                        icon: "cpu",
                        status: "running",
                      },
                    ];
                  });
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
                    ragHitRate: lastRagHitRate,
                    latencyMs: lastLatencyMs,
                    mcpTokensSaved: mcpTokensSavedTotal,
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

            default:
              break;
          }
        }
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : "连接后端失败";
      removeTrailingStatus();
      appendMessage({ role: "agent", content: message, type: "text" });
    } finally {
      setIsStreaming(false);
      abortControllerRef.current = null;
    }
  }, [
    input,
    isStreaming,
    controlBarState,
    appendMessage,
    updateLastStatus,
    removeTrailingStatus,
    setLastAgentTokenUsage,
    handleAgentDispatch,
    attachMetricsToLastAgent,
    lastRagHitRate,
    lastLatencyMs,
    mcpTokensSavedTotal,
  ]);

  useEffect(() => {
    return () => {
      abortControllerRef.current?.abort();
    };
  }, []);

  return (
    <div className="flex h-screen flex-col bg-zinc-900 text-zinc-100">
      <header className="flex h-14 items-center justify-between border-b border-zinc-800 px-6">
        <h1 className="text-lg font-semibold">BioDynamics Agent</h1>
        <div className="flex items-center gap-2">
          {/* 已加载数据库徽章 */}
          {ragStatus && ragStatus.databases.length > 0 && (
            <div className="hidden md:flex items-center gap-1 mr-2">
              {ragStatus.databases.map((db) => (
                <span
                  key={db.name}
                  className={`inline-flex items-center rounded-full px-2 py-0.5 text-[10px] font-medium ${
                    db.type === "online_api"
                      ? "bg-emerald-900/40 text-emerald-300 border border-emerald-700/50"
                      : db.type === "local_file"
                        ? "bg-blue-900/40 text-blue-300 border border-blue-700/50"
                        : "bg-zinc-800/60 text-zinc-400 border border-zinc-700/50"
                  }`}
                  title={db.type === "online_api" ? "在线 API" : db.type === "local_file" ? "本地文件" : "用户导入"}
                >
                  {db.name}
                </span>
              ))}
            </div>
          )}
          <Button
            variant="outline"
            size="sm"
            onClick={handleUpdateVectorDb}
            disabled={isUpdatingDb}
            className="h-8 gap-1.5 border-zinc-700 bg-zinc-800/80 text-zinc-200 hover:bg-zinc-700 hover:text-zinc-100"
          >
            <RefreshCw
              className={`h-3.5 w-3.5 ${isUpdatingDb ? "animate-spin" : ""}`}
            />
            <span className="hidden sm:inline">
              {isUpdatingDb ? "更新中..." : "更新知识库"}
            </span>
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={handleClearMemory}
            disabled={isStreaming}
            className="h-8 gap-1.5 border-zinc-700 bg-zinc-800/80 text-zinc-200 hover:bg-red-900/30 hover:text-red-200"
          >
            <Trash2 className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">清除当前对话</span>
          </Button>
        </div>
      </header>

      {updateDbStatus && (
        <div className="border-b border-zinc-800 bg-zinc-800/50 px-6 py-1.5 text-center text-xs text-zinc-400">
          {updateDbStatus}
        </div>
      )}

      {/* 智能体工作流追踪器：流式过程中或完成后展示 */}
      {agents.length > 0 && (
        <div className="border-b border-zinc-800 bg-zinc-900/95 px-4 py-2 sm:px-6 lg:px-8">
          <div className="mx-auto max-w-3xl">
            <AgentWorkflowTracker agents={agents} />
            {/* v2 流水线可视化：横向 Stepper + 进度条 */}
            {pipelineSteps.length > 0 && (
              <div className="mt-2">
                <WorkflowVisualization
                  steps={pipelineSteps}
                  currentNode={pipelineCurrent}
                  stepIndex={pipelineStepIndex}
                  totalSteps={pipelineTotal || pipelineSteps.length}
                  pipeline={pipelineName}
                  status={pipelineStatus}
                />
              </div>
            )}
          </div>
        </div>
      )}

      <main className="flex flex-1 flex-row overflow-hidden">
        {/* 左侧聊天区：70% */}
        <section className="flex h-full w-[70%] flex-col">
          <div className="flex-1 overflow-y-auto px-4 py-6 sm:px-6 lg:px-8">
            <div className="mx-auto max-w-3xl space-y-5">
              {messages.length === 0 && (
                <div className="flex flex-col items-center justify-center gap-3 pt-32 text-zinc-500">
                  <h2 className="text-2xl font-semibold text-zinc-300">BioDynamics Agent</h2>
                  <p className="text-sm">输入生物学假说，Agent 将自动建模并运行仿真。</p>
                </div>
              )}
              {messages.map((msg) => (
                <ChatMessage
                  key={msg.id}
                  role={msg.role}
                  content={msg.content}
                  type={msg.type}
                  tokenUsage={msg.tokenUsage}
                  ragInsights={msg.ragInsights}
                  ragHitRate={msg.ragHitRate}
                  latencyMs={msg.latencyMs}
                  mcpToolCalls={msg.mcpToolCalls}
                  mcpTokensSaved={msg.mcpTokensSaved}
                  mcpTermDefinitions={msg.mcpTermDefinitions}
                  doseResponseData={msg.doseResponseData}
                  synergyData={msg.synergyData}
                  pkpdProfile={msg.pkpdProfile}
                  modelName={msg.modelName || modelName}
                />
              ))}
              <div ref={messagesEndRef} />
            </div>
          </div>

          <div className="border-t border-zinc-800 bg-zinc-900/95 px-4 pb-4 pt-3 sm:px-6 lg:px-8">
            <div className="mx-auto max-w-3xl">
              <ChatInput
                value={input}
                onChange={setInput}
                onSend={handleSend}
                disabled={isStreaming}
              />
            </div>
          </div>
        </section>

        {/* 右侧控制栏：30%，最小宽度 360px */}
        <ControlBar
          state={controlBarState}
          onChange={setControlBarState}
          currentNode={currentNode}
          tokenUsage={tokenUsage}
          isStreaming={isStreaming}
          clarification={clarification}
          onClarificationSubmit={handleClarificationSubmit}
          onStop={handleStop}
          ragStatus={ragStatus}
        />
      </main>
    </div>
  );
}
