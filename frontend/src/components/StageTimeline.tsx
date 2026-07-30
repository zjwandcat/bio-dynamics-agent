// 左侧阶段时间轴组件：展示 8 个阶段节点的执行进度与运行状态
import { motion } from 'framer-motion';
import {
  Route,
  BookOpen,
  Network,
  Search,
  Syringe,
  Code,
  FlaskConical,
  FileText,
  CheckCircle,
  XCircle,
  Loader2,
} from 'lucide-react';
import type { ReactNode, ComponentType } from 'react';
import { STAGE_NODES } from '../types/sse';
import type { StageState, StageStatus } from '../types/sse';

/** 阶段时间轴 Props */
export interface StageTimelineProps {
  /** 来自 useChatStream 的阶段状态列表 */
  stages: StageState[];
  /** 当前运行节点 key */
  currentStage: string | null;
  /** 当前正在做什么的文案 */
  currentActionText: string;
}

/** iconName → lucide 图标组件映射 */
const ICON_MAP: Record<string, ComponentType<{ className?: string }>> = {
  Route,
  BookOpen,
  Network,
  Search,
  Syringe,
  Code,
  FlaskConical,
  FileText,
};

/** 阶段状态对应的中文文案 */
const STATUS_TEXT: Record<StageStatus, string> = {
  pending: '等待中',
  running: '执行中',
  done: '已完成',
  failed: '失败',
};

/** 将毫秒耗时格式化为秒（如 1200ms → "1.2s"） */
const formatDuration = (ms?: number): string => {
  if (ms == null) return '';
  return `${(ms / 1000).toFixed(1)}s`;
};

/**
 * 左侧阶段时间轴：展示 8 个阶段节点的执行进度与状态
 * - 顶部：标题 "执行阶段" + 进度条（已完成 / 总数）
 * - 主体：垂直列表，每个节点显示图标圆、名称、状态与耗时
 *         当前 running 节点下方额外显示 currentActionText
 */
export default function StageTimeline({
  stages,
  currentStage,
  currentActionText,
}: StageTimelineProps) {
  // 构建节点 key → 状态的映射，便于快速查找
  const stageMap = new Map(stages.map(s => [s.key, s]));
  // 已完成节点数量
  const doneCount = stages.filter(s => s.status === 'done').length;
  const totalCount = STAGE_NODES.length;
  // 进度条填充百分比
  const progressPercent = totalCount > 0 ? (doneCount / totalCount) * 100 : 0;

  return (
    <div className="w-72 bg-slate-900 border-r border-slate-800 flex flex-col">
      {/* 顶部标题 + 进度条 */}
      <div className="px-4 py-3 border-b border-slate-800">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-slate-200">执行阶段</span>
          <span className="text-xs text-slate-500">
            执行进度 {doneCount}/{totalCount}
          </span>
        </div>
        <div className="bg-slate-800 rounded-full h-1.5 overflow-hidden">
          <div
            className="bg-indigo-500 rounded-full h-1.5 transition-all"
            style={{ width: `${progressPercent}%` }}
          />
        </div>
      </div>

      {/* 主体：节点列表 */}
      <div className="flex-1 overflow-y-auto px-4 py-3">
        {STAGE_NODES.map((node, idx) => {
          const state = stageMap.get(node.key);
          const status: StageStatus = state?.status ?? 'pending';
          const duration = state?.durationMs;
          const isRunning = status === 'running';
          // 通过 currentStage 判断是否为当前活跃节点
          const isCurrentNode = node.key === currentStage;
          const isLast = idx === STAGE_NODES.length - 1;
          const Icon = ICON_MAP[node.iconName] ?? Route;

          // 根据 status 决定图标圆的样式与内容
          let circleClass = 'bg-slate-800 text-slate-500';
          let circleContent: ReactNode = <Icon className="w-5 h-5" />;
          if (isRunning) {
            circleClass = 'bg-indigo-500 text-white';
          } else if (status === 'done') {
            circleClass = 'bg-emerald-500 text-white';
            circleContent = <CheckCircle className="w-5 h-5" />;
          } else if (status === 'failed') {
            circleClass = 'bg-rose-500 text-white';
            circleContent = <XCircle className="w-5 h-5" />;
          }

          return (
            <div key={node.key}>
              <div className="flex items-center gap-3">
                {/* 图标圆（running 时脉冲动画 scale 1→1.1→1, opacity 0.8→1→0.8） */}
                {isRunning ? (
                  <motion.div
                    className={`w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0 ${circleClass}`}
                    animate={{ scale: [1, 1.1, 1], opacity: [0.8, 1, 0.8] }}
                    transition={{ duration: 1.2, repeat: Infinity }}
                  >
                    {circleContent}
                  </motion.div>
                ) : (
                  <div
                    className={`w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0 ${circleClass}`}
                  >
                    {circleContent}
                  </div>
                )}

                {/* 右侧：节点名称 + 状态 + 耗时 */}
                <div className="flex flex-col flex-1 min-w-0">
                  <span className="text-sm text-slate-200">{node.label}</span>
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-slate-500">{STATUS_TEXT[status]}</span>
                    {status === 'done' && duration != null && (
                      <span className="text-xs text-slate-500">
                        ({formatDuration(duration)})
                      </span>
                    )}
                  </div>
                </div>
              </div>

              {/* 当前 running 节点下方显示 currentActionText（带 Loader2 旋转图标） */}
              {isRunning && isCurrentNode && currentActionText && (
                <div className="ml-12 mt-1 mb-1 flex items-center gap-1.5">
                  <Loader2 className="w-3 h-3 text-indigo-300 animate-spin" />
                  <span className="text-xs text-indigo-300 italic">{currentActionText}</span>
                </div>
              )}

              {/* 节点间竖线连接（最后一个节点不显示） */}
              {!isLast && <div className="ml-5 w-px h-4 bg-slate-700" />}
            </div>
          );
        })}
      </div>
    </div>
  );
}
