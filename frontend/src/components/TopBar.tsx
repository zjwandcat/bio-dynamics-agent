// 顶部状态栏组件：展示项目标识、当前 LLM/Agent/阶段状态、SSE 连接状态以及知识库刷新按钮
import { useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import { Activity, Brain, Bot, Database, RefreshCw } from 'lucide-react';
import type { ConnectionStatus } from '../types/sse';
import { AGENT_NAME_TO_CN_LABEL } from '../types/sse';
import {
  triggerKbUpdate,
  getKbUpdateStatus,
  type KbUpdateStatus,
} from '../lib/api';

/** 顶部状态栏 Props */
export interface TopBarProps {
  /** 当前使用的 LLM 模型名 */
  currentLLM: string;
  /** 当前调度的 Agent 中文名（为空表示空闲） */
  currentAgent: string | null;
  /** 当前运行节点 key（如 'worker_ode'） */
  currentStage: string | null;
  /** 中文阶段名（可选，如 "ODE 方程生成"） */
  currentStageLabel?: string;
  /** 与后端的 SSE 连接状态 */
  connectionStatus: ConnectionStatus;
}

/** 连接状态对应的圆点颜色、文案与是否需要脉冲动画 */
const CONNECTION_CONFIG: Record<
  ConnectionStatus,
  { dotClass: string; label: string; pulse: boolean }
> = {
  connected: { dotClass: 'bg-emerald-500', label: '已连接', pulse: false },
  connecting: { dotClass: 'bg-yellow-400', label: '连接中', pulse: true },
  disconnected: { dotClass: 'bg-slate-500', label: '未连接', pulse: false },
  error: { dotClass: 'bg-rose-500', label: '错误', pulse: false },
};

/** 知识库刷新状态对应的圆点颜色、文案与是否脉冲 */
const KB_STATUS_CONFIG: Record<
  KbUpdateStatus,
  { dotClass: string; label: string; pulse: boolean }
> = {
  idle: { dotClass: 'bg-slate-500', label: '知识库就绪', pulse: false },
  running: { dotClass: 'bg-yellow-400', label: '刷新中', pulse: true },
  success: { dotClass: 'bg-emerald-500', label: '知识库最新', pulse: false },
  failed: { dotClass: 'bg-rose-500', label: '刷新失败', pulse: false },
};

/** 轮询间隔：刷新中时 2 秒查询一次后端状态 */
const KB_POLL_INTERVAL_MS = 2000;

/**
 * 顶部状态栏：三栏布局，深色主题
 * - 左侧：项目 logo（Activity 图标） + 名称（渐变文字）
 * - 中间：LLM / Agent / 阶段 三组状态卡片
 * - 右侧：知识库刷新按钮 + 状态灯 / SSE 连接状态指示灯
 */
export default function TopBar({
  currentLLM,
  currentAgent,
  currentStage,
  currentStageLabel,
  connectionStatus,
}: TopBarProps) {
  // 判断 LLM 是否处于 fallback 状态：名称包含 "fallback" 即视为兜底模型
  const isFallback = currentLLM.toLowerCase().includes('fallback');
  // 阶段卡片显示值：优先使用中文阶段名，其次节点 key，最后回退到 "待命"
  const stageValue = currentStageLabel || currentStage || '待命';
  // Agent 中文标签：根据英文 Agent 名反查（如 "Mechanism Analysis Agent" → "机制解析"）
  const agentCnLabel = currentAgent ? AGENT_NAME_TO_CN_LABEL[currentAgent] : undefined;
  const conn = CONNECTION_CONFIG[connectionStatus];

  // 知识库刷新状态：组件挂载时先查询一次后端真实状态，避免总是显示 idle
  const [kbStatus, setKbStatus] = useState<KbUpdateStatus>('idle');
  const [kbMessage, setKbMessage] = useState<string>('');
  // 轮询定时器引用：仅在 running 期间启用
  const pollTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  /** 拉取后端知识库刷新状态；若仍 running 则继续排队下一次轮询 */
  const fetchKbStatus = async () => {
    try {
      const resp = await getKbUpdateStatus();
      setKbStatus(resp.status);
      setKbMessage(resp.message || '');
      // running 期间持续轮询；success/failed/idle 时停止
      if (resp.status === 'running') {
        pollTimerRef.current = setTimeout(fetchKbStatus, KB_POLL_INTERVAL_MS);
      }
    } catch (err) {
      // 网络异常时降级为 idle，避免误导用户
      setKbStatus('idle');
      setKbMessage('状态查询失败');
    }
  };

  // 组件挂载时同步一次后端状态；卸载时清理轮询定时器
  useEffect(() => {
    fetchKbStatus();
    return () => {
      if (pollTimerRef.current) {
        clearTimeout(pollTimerRef.current);
        pollTimerRef.current = null;
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /** 点击"更新知识库"按钮：触发后端刷新并立即进入轮询 */
  const handleRefreshKb = async () => {
    // 刷新中时禁止重复点击
    if (kbStatus === 'running') return;
    setKbStatus('running');
    setKbMessage('知识库更新已启动，后台处理中...');
    try {
      await triggerKbUpdate();
      // 启动轮询，等待后端 running → success/failed
      pollTimerRef.current = setTimeout(fetchKbStatus, KB_POLL_INTERVAL_MS);
    } catch (err) {
      setKbStatus('failed');
      setKbMessage('触发更新失败：' + (err instanceof Error ? err.message : String(err)));
    }
  };

  const kbCfg = KB_STATUS_CONFIG[kbStatus];

  return (
    <div className="h-14 bg-slate-900 border-b border-slate-800 flex items-center justify-between px-4">
      {/* 左侧：项目 logo + 名称 */}
      <div className="flex items-center gap-2">
        <Activity className="w-6 h-6 text-indigo-400" />
        <span className="text-lg font-bold bg-gradient-to-r from-indigo-400 to-purple-400 bg-clip-text text-transparent">
          BioDynamics Agent
        </span>
      </div>

      {/* 中间：三组状态卡片 */}
      <div className="flex gap-4">
        {/* LLM 卡片 */}
        <div className="bg-slate-800 rounded-lg px-3 py-1.5 flex items-center gap-2">
          <Brain className="w-4 h-4 text-indigo-400" />
          <span className="text-xs text-slate-500">LLM</span>
          <span className="text-sm text-slate-200 font-medium">{currentLLM || '-'}</span>
          {isFallback && (
            <span className="text-xs text-orange-400 bg-orange-500/10 px-1.5 py-0.5 rounded">
              fallback
            </span>
          )}
        </div>

        {/* Agent 卡片 */}
        <div className="bg-slate-800 rounded-lg px-3 py-1.5 flex items-center gap-2">
          <Bot className="w-4 h-4 text-indigo-400" />
          <span className="text-xs text-slate-500">Agent</span>
          {currentAgent ? (
            <>
              <span className="text-sm text-slate-200 font-medium">{currentAgent}</span>
              {agentCnLabel && (
                <span className="text-xs text-indigo-300 bg-indigo-500/10 px-1.5 py-0.5 rounded">
                  {agentCnLabel}
                </span>
              )}
            </>
          ) : (
            <span className="text-sm text-slate-400">空闲</span>
          )}
        </div>

        {/* 阶段卡片 */}
        <div className="bg-slate-800 rounded-lg px-3 py-1.5 flex items-center gap-2">
          <Activity className="w-4 h-4 text-indigo-400" />
          <span className="text-xs text-slate-500">阶段</span>
          <span className="text-sm text-slate-200 font-medium">{stageValue}</span>
        </div>
      </div>

      {/* 右侧：知识库刷新按钮 + SSE 连接状态指示灯 */}
      <div className="flex items-center gap-4">
        {/* 知识库刷新按钮：点击触发后端更新，旁边圆点反映当前状态 */}
        <button
          type="button"
          onClick={handleRefreshKb}
          disabled={kbStatus === 'running'}
          title={kbMessage || kbCfg.label}
          className="flex items-center gap-2 bg-slate-800 hover:bg-slate-700 disabled:opacity-60 disabled:cursor-not-allowed rounded-lg px-3 py-1.5 transition-colors"
        >
          <Database className="w-4 h-4 text-indigo-400" />
          <span className="text-xs text-slate-400">知识库</span>
          {/* 状态圆点：刷新中黄色脉冲，成功绿色，失败红色，空闲灰色 */}
          {kbCfg.pulse ? (
            <motion.span
              className={`w-2 h-2 rounded-full ${kbCfg.dotClass}`}
              animate={{ opacity: [1, 0.3, 1] }}
              transition={{ duration: 1, repeat: Infinity }}
            />
          ) : (
            <span className={`w-2 h-2 rounded-full ${kbCfg.dotClass}`} />
          )}
          {/* 刷新图标：running 时旋转动画 */}
          <RefreshCw
            className={`w-3.5 h-3.5 text-slate-400 ${kbStatus === 'running' ? 'animate-spin' : ''}`}
          />
          <span className="text-xs text-slate-300">{kbCfg.label}</span>
        </button>

        {/* SSE 连接状态指示灯 */}
        <div className="flex items-center gap-2">
          {conn.pulse ? (
            <motion.span
              className={`w-2 h-2 rounded-full ${conn.dotClass}`}
              animate={{ opacity: [1, 0.3, 1] }}
              transition={{ duration: 1, repeat: Infinity }}
            />
          ) : (
            <span className={`w-2 h-2 rounded-full ${conn.dotClass}`} />
          )}
          <span className="text-sm text-slate-400">{conn.label}</span>
        </div>
      </div>
    </div>
  );
}
