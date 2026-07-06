"use client";

import React from "react";
import { motion } from "framer-motion";
import {
  Check,
  X,
  Loader2,
  ChevronRight,
  Network,
  Search,
  FlaskConical,
  BarChart3,
  Beaker,
  Library,
  FileText,
  Sliders,
  GitBranch,
  type LucideIcon,
} from "lucide-react";
import { cn } from "@/lib/utils";

// v2 节点 → 图标映射（与 supervisor AGENT_REGISTRY_V2.icon 一致）
const ICON_MAP: Record<string, LucideIcon> = {
  network: Network,
  search: Search,
  "git-branch": GitBranch,
  sliders: Sliders,
  "flask-conical": FlaskConical,
  "bar-chart-3": BarChart3,
  beaker: Beaker,
  library: Library,
  "file-text": FileText,
  cpu: Loader2,
};

export type V2NodeStatus = "idle" | "running" | "completed" | "failed" | "skipped";

export interface V2PipelineStep {
  node_name: string;
  cn_label: string;
  icon: string;
  status: V2NodeStatus;
  latency_ms?: number;
  started_at?: number;
  finished_at?: number;
}

interface WorkflowVisualizationProps {
  steps: V2PipelineStep[];
  currentNode: string;
  stepIndex: number;
  totalSteps: number;
  pipeline: "v1" | "v2";
  status: "starting" | "running" | "completed" | "failed";
  className?: string;
}

/**
 * v2 工作流可视化
 * 横向 Stepper + 进度条，12 节点（v1 也可复用）。
 * - running: 当前节点加载动画
 * - completed: 绿色对勾
 * - failed: 红色叉号
 * - idle: 灰色空心
 * 鼠标悬停查看节点状态详情。
 */
export function WorkflowVisualization({
  steps,
  currentNode,
  stepIndex,
  totalSteps,
  pipeline,
  status,
  className,
}: WorkflowVisualizationProps) {
  if (!steps || steps.length === 0) return null;

  const progressPct = totalSteps > 0 ? (stepIndex / totalSteps) * 100 : 0;

  return (
    <div
      className={cn(
        "rounded-lg border border-white/10 bg-black/40 p-3",
        className
      )}
    >
      {/* 头部：流水线标识 + 进度 */}
      <div className="mb-2 flex items-center justify-between text-xs">
        <div className="flex items-center gap-2">
          <span
            className={cn(
              "rounded-full px-2 py-0.5 font-mono text-[10px] uppercase tracking-wide",
              pipeline === "v2"
                ? "bg-purple-900/40 text-purple-300"
                : "bg-zinc-800 text-zinc-400"
            )}
          >
            {pipeline} · 12-node pipeline
          </span>
          <span className="text-zinc-500">
            {status === "completed"
              ? "✓ 全部完成"
              : status === "failed"
                ? "✗ 失败"
                : `${stepIndex}/${totalSteps}`}
          </span>
        </div>
        <div className="text-zinc-500">
          当前节点：
          <span className="ml-1 font-mono text-zinc-300">{currentNode || "—"}</span>
        </div>
      </div>

      {/* 进度条 */}
      <div className="mb-3 h-1.5 w-full overflow-hidden rounded-full bg-zinc-800/60">
        <motion.div
          className={cn(
            "h-full rounded-full",
            status === "failed"
              ? "bg-red-500"
              : status === "completed"
                ? "bg-green-500"
                : "bg-gradient-to-r from-blue-500 to-purple-500"
          )}
          initial={{ width: 0 }}
          animate={{ width: `${progressPct}%` }}
          transition={{ duration: 0.4 }}
        />
      </div>

      {/* 节点列表（紧凑横向） */}
      <div className="flex flex-wrap items-center gap-1.5">
        {steps.map((step, idx) => {
          const IconComp = ICON_MAP[step.icon] ?? Loader2;
          const isLast = idx === steps.length - 1;
          return (
            <React.Fragment key={step.node_name}>
              <motion.div
                className={cn(
                  "flex items-center gap-1.5 rounded-md border px-2 py-1 text-[11px] transition-colors",
                  step.status === "running" &&
                    "border-blue-500 bg-blue-500/10 text-blue-200",
                  step.status === "completed" &&
                    "border-green-500/60 bg-green-500/10 text-green-200",
                  step.status === "failed" &&
                    "border-red-500/60 bg-red-500/10 text-red-200",
                  step.status === "skipped" &&
                    "border-zinc-700 bg-zinc-800/30 text-zinc-500 line-through",
                  step.status === "idle" &&
                    "border-zinc-700 bg-zinc-800/40 text-zinc-500"
                )}
                title={`${step.cn_label} · ${step.status}${
                  step.latency_ms ? ` · ${Math.round(step.latency_ms)}ms` : ""
                }`}
                initial={{ opacity: 0, y: 4 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: idx * 0.05, duration: 0.2 }}
              >
                {step.status === "running" ? (
                  <Loader2 className="h-3 w-3 animate-spin" />
                ) : step.status === "completed" ? (
                  <Check className="h-3 w-3" />
                ) : step.status === "failed" ? (
                  <X className="h-3 w-3" />
                ) : (
                  <IconComp className="h-3 w-3" />
                )}
                <span className="font-mono">{step.node_name}</span>
              </motion.div>
              {!isLast && (
                <ChevronRight className="h-3 w-3 text-zinc-600" />
              )}
            </React.Fragment>
          );
        })}
      </div>
    </div>
  );
}
