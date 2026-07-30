// useChatStream: SSE 客户端 Hook
// 通过 @microsoft/fetch-event-source 连接后端 POST /api/chat，解析并分发 SSE 事件。
// 严格遵循后端 v3 Supervisor-Worker 工作流的 SSE 真实契约（详见 src/types/sse.ts）。
// 注意：多个事件的 data 是裸字符串而非对象，已在各 case 中按需处理。

import { useCallback, useRef, useState } from 'react';
import { fetchEventSource } from '@microsoft/fetch-event-source';
import { STAGE_NODES, NODE_ACTION_TEXT, NODE_NAME_TO_AGENT } from '../types/sse';
import * as api from '../lib/api';
import type {
  SSEEvent,
  ChatMessage,
  StageState,
  StageStatus,
  ConnectionStatus,
  MessageRole,
  ConfigData,
  AgentDef,
  WorkflowV3StateData,
  AgentDispatchData,
  ClarificationNeededData,
  ClarificationOption,
  KnowledgeGraphData,
  V4PathwayGraphData,
  V4SimulationResultData,
  V4ValidationReportData,
  RagInsightsData,
  RagOnlineFallbackData,
  RagReadyData,
  TokenUsageData,
  ErrorData,
  ReportData,
} from '../types/sse';

// ---------------------------------------------------------------------------
// 环境变量
// ---------------------------------------------------------------------------

// 从 Vite 环境变量读取后端地址与默认 LLM，未配置时回退到默认值
// （项目无 vite-env.d.ts，此处沿用 api.ts 的 import.meta 类型断言写法）
const ENV = (import.meta as unknown as { env?: Record<string, string | undefined> }).env ?? {};
const BASE_URL: string = ENV.VITE_API_BASE_URL || 'http://localhost:8000';
const DEFAULT_LLM: string = ENV.VITE_DEFAULT_LLM || '';

// ---------------------------------------------------------------------------
// 内部类型
// ---------------------------------------------------------------------------

/** 报告类聚合状态（对应各类 report / 结果事件） */
interface ReportsState {
  pathwayGraph?: V4PathwayGraphData;
  code?: string;
  imageBase64?: string;
  simulationResult?: V4SimulationResultData;
  validationReport?: V4ValidationReportData;
  experimentProtocols?: unknown[];
  paperEvidence?: unknown[];
  markdown?: string;
  knowledgeGraph?: KnowledgeGraphData;
  ragInsights?: RagInsightsData;
}

/** 追问（人工干预）状态 */
interface ClarificationState {
  question: string;
  options: ClarificationOption[];
  context: string;
}

/** Hook 对外暴露的完整状态 */
interface ChatStreamState {
  messages: ChatMessage[];
  stages: StageState[];
  currentLLM: string;
  currentAgent: string | null;
  currentStage: string | null;
  currentActionText: string;
  reports: ReportsState;
  clarification: ClarificationState | null;
  connectionStatus: ConnectionStatus;
  isStreaming: boolean;
  errorMessage: string | null;
  threadId: string;
}

/** Hook 返回值：状态 + 控制方法 */
interface UseChatStreamReturn extends ChatStreamState {
  /** agent_registry 事件提供的 agent 列表（用于时间轴展示） */
  agentRegistry: AgentDef[];
  sendMessage: (text: string) => Promise<void>;
  respondToClarification: (selectedOption: string, freeText?: string) => Promise<void>;
  stopStream: () => Promise<void>;
  clearMemory: () => Promise<void>;
  reset: () => void;
}

// ---------------------------------------------------------------------------
// 工具函数
// ---------------------------------------------------------------------------

/** 生成阶段时间轴的初始状态（全部 pending） */
const initialStages = (): StageState[] =>
  STAGE_NODES.map(s => ({ key: s.key, status: 'pending' as StageStatus }));

/** 生成一条聊天消息（id 形如 msg_<时间戳>_<随机>） */
const makeMessage = (
  role: MessageRole,
  content: string,
  extra?: Partial<ChatMessage>,
): ChatMessage => ({
  id: `msg_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
  role,
  content,
  timestamp: Date.now(),
  ...(extra ?? {}),
});

// ---------------------------------------------------------------------------
// Hook 实现
// ---------------------------------------------------------------------------

export function useChatStream(): UseChatStreamReturn {
  // 核心状态
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [stages, setStages] = useState<StageState[]>(initialStages);
  const [currentLLM, setCurrentLLM] = useState<string>(DEFAULT_LLM);
  const [currentAgent, setCurrentAgent] = useState<string | null>(null);
  const [currentStage, setCurrentStage] = useState<string | null>(null);
  const [currentActionText, setCurrentActionText] = useState<string>('');
  const [reports, setReports] = useState<ReportsState>({});
  const [clarification, setClarification] = useState<ClarificationState | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('disconnected');
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [threadId, setThreadId] = useState<string>('');
  const [agentRegistry, setAgentRegistry] = useState<AgentDef[]>([]);

  // Refs：用于在稳定的回调中读取最新值，避免闭包陷阱
  const currentLLMRef = useRef<string>(DEFAULT_LLM);
  const isStreamingRef = useRef<boolean>(false);

  // -------------------------------------------------------------------------
  // 事件分发：根据事件名更新对应状态
  // -------------------------------------------------------------------------

  const handleEvent = useCallback((event: string, data: unknown) => {
    switch (event) {
      case 'config': {
        // 当前 LLM 配置
        const cfg = data as ConfigData;
        if (cfg?.model_name) {
          currentLLMRef.current = cfg.model_name;
          setCurrentLLM(cfg.model_name);
        }
        break;
      }
      case 'agent_registry': {
        // agent 列表（用于时间轴展示）
        const list = Array.isArray(data) ? (data as AgentDef[]) : [];
        setAgentRegistry(list);
        break;
      }
      case 'node_start': {
        // data 是裸字符串（状态文案），非空则作为系统消息追加
        const text = typeof data === 'string' ? data : '';
        if (text) {
          setMessages(prev => [...prev, makeMessage('system', text)]);
        }
        break;
      }
      case 'workflow_v3_state': {
        // 当前节点状态来源：更新 currentStage / currentActionText / stages
        const st = data as WorkflowV3StateData;
        const node = st?.current_node;
        if (!node) break;
        setCurrentStage(node);
        setCurrentActionText(NODE_ACTION_TEXT[node] ?? '');
        // 同步更新 currentAgent：当阶段切换时，根据节点名映射到真实 Agent 名
        // 这样即使 agent_dispatch 事件延迟或丢失，顶部状态栏也能显示正确的 Agent
        const agentInfoForStage = NODE_NAME_TO_AGENT[node];
        if (agentInfoForStage && st.status === 'running') {
          setCurrentAgent(agentInfoForStage.name);
        } else if (st.status === 'completed' && node === 'worker_report') {
          // 报告生成完成时清空 Agent
          setCurrentAgent(null);
        }
        setStages(prev => {
          const next = prev.map(s => ({ ...s }));
          if (st.status === 'running') {
            // 切换到新节点：将之前处于 running 的阶段标记为 done
            for (const s of next) {
              if (s.status === 'running' && s.key !== node) {
                s.status = 'done';
                s.endTime = Date.now();
                if (s.startTime) s.durationMs = s.endTime - s.startTime;
              }
            }
            // 当前节点阶段标记为 running，并记录 startTime
            const target = next.find(s => s.key === node);
            if (target && target.status !== 'running') {
              target.status = 'running';
              target.startTime = Date.now();
            }
          } else if (st.status === 'completed') {
            // 当前节点阶段标记为 done，记录 endTime 与 durationMs
            const target = next.find(s => s.key === node);
            if (target) {
              target.status = 'done';
              target.endTime = Date.now();
              if (target.startTime) target.durationMs = target.endTime - target.startTime;
            }
          }
          return next;
        });
        break;
      }
      case 'agent_dispatch': {
        // 主管调度信息：更新 currentAgent
        // 后端 target_agent 可能是中文短语，优先用 node_name 映射到真实 Agent 名
        const ad = data as AgentDispatchData;
        if (ad?.status === 'completed') {
          setCurrentAgent(null);
        } else {
          // 优先用 node_name 映射到真实 Agent 名（来自 supervisor.py AGENT_REGISTRY_V2）
          const agentInfo = ad?.node_name ? NODE_NAME_TO_AGENT[ad.node_name] : undefined;
          if (agentInfo) {
            setCurrentAgent(agentInfo.name);
          } else if (ad?.target_agent) {
            // fallback：直接用后端 target_agent（可能是中文短语）
            setCurrentAgent(ad.target_agent);
          }
        }
        break;
      }
      case 'clarification_needed': {
        // 设置追问状态，等待用户响应（不阻塞 SSE 流）
        const cn = data as ClarificationNeededData;
        setClarification({
          question: cn?.question ?? '',
          options: cn?.options ?? [],
          context: cn?.context ?? '',
        });
        break;
      }
      case 'clarification_resolved': {
        // 清空追问状态
        setClarification(null);
        break;
      }
      case 'code_generated': {
        // data 是 Python 代码字符串
        const code = typeof data === 'string' ? data : '';
        setReports(prev => ({ ...prev, code }));
        if (code) {
          setMessages(prev => [
            ...prev,
            makeMessage('agent', '```python\n' + code + '\n```', { code }),
          ]);
        }
        break;
      }
      case 'image_ready': {
        // data 是 base64 字符串
        const b64 = typeof data === 'string' ? data : '';
        setReports(prev => ({ ...prev, imageBase64: b64 }));
        if (b64) {
          setMessages(prev => [...prev, makeMessage('agent', '仿真结果图', { imageBase64: b64 })]);
        }
        break;
      }
      case 'report': {
        // data.markdown 设置 reports.markdown
        const r = data as ReportData;
        if (r?.markdown) {
          setReports(prev => ({ ...prev, markdown: r.markdown }));
        }
        break;
      }
      case 'report_ready': {
        // data 是 markdown 字符串
        const md = typeof data === 'string' ? data : '';
        setReports(prev => ({ ...prev, markdown: md }));
        if (md) {
          setMessages(prev => [...prev, makeMessage('agent', md)]);
        }
        break;
      }
      case 'v4_pathway_graph': {
        const g = data as V4PathwayGraphData;
        setReports(prev => ({ ...prev, pathwayGraph: g }));
        break;
      }
      case 'v4_simulation_result': {
        const sr = data as V4SimulationResultData;
        setReports(prev => ({
          ...prev,
          simulationResult: sr,
          ...(sr?.image_base64 ? { imageBase64: sr.image_base64 } : {}),
        }));
        break;
      }
      case 'v4_validation_report': {
        const vr = data as V4ValidationReportData;
        setReports(prev => ({ ...prev, validationReport: vr }));
        break;
      }
      case 'experiment_protocols': {
        // data 是数组
        const arr = Array.isArray(data) ? data : [];
        setReports(prev => ({ ...prev, experimentProtocols: arr }));
        break;
      }
      case 'paper_evidence': {
        // data 是数组
        const arr = Array.isArray(data) ? data : [];
        setReports(prev => ({ ...prev, paperEvidence: arr }));
        break;
      }
      case 'knowledge_graph': {
        const kg = data as KnowledgeGraphData;
        setReports(prev => ({ ...prev, knowledgeGraph: kg }));
        break;
      }
      case 'rag_insights': {
        const ri = data as RagInsightsData;
        setReports(prev => ({ ...prev, ragInsights: ri }));
        break;
      }
      case 'rag_online_fallback': {
        // 触发在线兜底时追加系统消息
        const rof = data as RagOnlineFallbackData;
        if (rof?.triggered && rof.message) {
          setMessages(prev => [...prev, makeMessage('system', rof.message)]);
        }
        break;
      }
      case 'rag_ready': {
        // 可选追加 RAG 完成摘要
        const rr = data as RagReadyData;
        if (rr?.summary) {
          setMessages(prev => [...prev, makeMessage('system', rr.summary)]);
        }
        break;
      }
      case 'execution_log': {
        // data 是字符串；内容简短时作为系统消息
        const text = typeof data === 'string' ? data : '';
        if (text && text.length <= 200) {
          setMessages(prev => [...prev, makeMessage('system', text)]);
        }
        break;
      }
      case 'token_usage': {
        // 字段是 model_name；与当前不同则切换并追加系统消息
        const tu = data as TokenUsageData;
        if (tu?.model_name && tu.model_name !== currentLLMRef.current) {
          currentLLMRef.current = tu.model_name;
          setCurrentLLM(tu.model_name);
          setMessages(prev => [...prev, makeMessage('system', `LLM 已切换至 ${tu.model_name}`)]);
        }
        break;
      }
      case 'error': {
        const e = data as ErrorData;
        const msg = e?.message || '未知错误';
        setErrorMessage(msg);
        setMessages(prev => [...prev, makeMessage('system', `错误: ${msg}`)]);
        break;
      }
      case 'end': {
        // 流结束：复位运行状态，所有 running 阶段标记为 done
        setIsStreaming(false);
        isStreamingRef.current = false;
        setConnectionStatus('disconnected');
        setStages(prev =>
          prev.map(s =>
            s.status === 'running'
              ? {
                  ...s,
                  status: 'done',
                  endTime: Date.now(),
                  ...(s.startTime ? { durationMs: Date.now() - s.startTime } : {}),
                }
              : s,
          ),
        );
        break;
      }
      default:
        // 其余事件（mcp_term_definitions / metrics / sa_* / simulation_csv /
        // dose_response / v4_hypothesis_list 等）目前简化处理，直接忽略
        break;
    }
  }, []);

  // -------------------------------------------------------------------------
  // 发送消息：发起 SSE 流
  // -------------------------------------------------------------------------

  const sendMessage = useCallback(
    async (text: string) => {
      if (!text.trim() || isStreamingRef.current) return;
      // 生成线程 ID：web_<时间戳>_<随机>
      const newThreadId = `web_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
      setThreadId(newThreadId);
      setMessages(prev => [...prev, makeMessage('user', text)]);
      setIsStreaming(true);
      isStreamingRef.current = true;
      setConnectionStatus('connecting');
      // 重置阶段时间轴为 pending
      setStages(initialStages());

      try {
        // fetchEventSource 返回可关闭的 Promise，连接关闭时 resolve
        await fetchEventSource(`${BASE_URL}/api/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
          body: JSON.stringify({
            user_input: text,
            thread_id: newThreadId,
            mode: 'auto_standard',
          }),
          openWhenHidden: true,
          onopen: async (response) => {
            if (response.ok) {
              setConnectionStatus('connected');
              setMessages(prev => [...prev, makeMessage('system', '已连接后端')]);
            } else {
              setConnectionStatus('error');
              setErrorMessage(`连接失败: ${response.status}`);
            }
          },
          onmessage: (ev) => {
            try {
              const payload = JSON.parse(ev.data) as SSEEvent;
              handleEvent(payload.event, payload.data);
            } catch {
              // 忽略无法解析的 SSE 消息，避免中断流
            }
          },
          onclose: () => {
            setIsStreaming(false);
            isStreamingRef.current = false;
            setConnectionStatus('disconnected');
          },
          onerror: (err: unknown) => {
            setConnectionStatus('error');
            setErrorMessage(err instanceof Error ? err.message : '连接错误');
            setIsStreaming(false);
            isStreamingRef.current = false;
            throw err; // 抛出以停止自动重试
          },
        });
      } catch (err) {
        setConnectionStatus('error');
        setErrorMessage(err instanceof Error ? err.message : String(err));
        setIsStreaming(false);
        isStreamingRef.current = false;
      }
    },
    [handleEvent],
  );

  // -------------------------------------------------------------------------
  // 回应追问：不阻塞 SSE 流，UI 提交后后端继续推送后续事件
  // -------------------------------------------------------------------------

  const respondToClarification = useCallback(
    async (selectedOption: string, freeText?: string) => {
      if (!clarification || !threadId) return;
      await api.respondToClarification(threadId, selectedOption, freeText);
      setClarification(null);
    },
    [clarification, threadId],
  );

  // -------------------------------------------------------------------------
  // 终止当前 SSE 流
  // -------------------------------------------------------------------------

  const stopStream = useCallback(async () => {
    if (!threadId) return;
    await api.stopStream(threadId);
    setIsStreaming(false);
    isStreamingRef.current = false;
  }, [threadId]);

  // -------------------------------------------------------------------------
  // 清除会话记忆
  // -------------------------------------------------------------------------

  const clearMemory = useCallback(async () => {
    if (!threadId) return;
    await api.clearMemory(threadId);
    setMessages([]);
    setStages(initialStages());
  }, [threadId]);

  // -------------------------------------------------------------------------
  // 重置全部状态
  // -------------------------------------------------------------------------

  const reset = useCallback(() => {
    setMessages([]);
    setStages(initialStages());
    setCurrentLLM(DEFAULT_LLM);
    currentLLMRef.current = DEFAULT_LLM;
    setCurrentAgent(null);
    setCurrentStage(null);
    setCurrentActionText('');
    setReports({});
    setClarification(null);
    setConnectionStatus('disconnected');
    setIsStreaming(false);
    isStreamingRef.current = false;
    setErrorMessage(null);
    setThreadId('');
    setAgentRegistry([]);
  }, []);

  return {
    messages,
    stages,
    currentLLM,
    currentAgent,
    currentStage,
    currentActionText,
    reports,
    clarification,
    connectionStatus,
    isStreaming,
    errorMessage,
    threadId,
    agentRegistry,
    sendMessage,
    respondToClarification,
    stopStream,
    clearMemory,
    reset,
  };
}
