"use client";

import React from "react";
import { motion } from "framer-motion";
import {
  Loader2,
  Check,
  X,
  Network,
  Search,
  FlaskConical,
  ShieldCheck,
  FileText,
  type LucideIcon,
} from "lucide-react";
import { cn } from "@/lib/utils";

export type AgentStatus = "idle" | "running" | "completed" | "failed";

export interface AgentState {
  name: string;
  cn_label: string;
  description: string;
  icon: string;
  status: AgentStatus;
  reasoning?: string;
  latency_ms?: number;
}

// 智能体图标映射：后端下发 icon 字符串 → 前端 lucide 组件
const ICON_MAP: Record<string, LucideIcon> = {
  network: Network,
  search: Search,
  "flask-conical": FlaskConical,
  "shield-check": ShieldCheck,
  cpu: Loader2,
  file: FileText,
};

interface AgentWorkflowTrackerProps {
  agents: AgentState[];
  className?: string;
}

/**
 * 智能体工作流追踪器
 * 对应 1233.md 第四部分 §1：水平 Step Wizard，监听 agent_dispatch 事件动态点亮。
 * running 时加载动画，completed 时绿色对勾，failed 时红色叉号。
 * 鼠标悬停显示 Orchestrator 的 reasoning。
 */
export function AgentWorkflowTracker({
  agents,
  className,
}: AgentWorkflowTrackerProps) {
  if (!agents || agents.length === 0) return null;

  return (
    <div
      className={cn(
        "flex items-center justify-between gap-2 rounded-lg border border-white/10 bg-black/30 p-3",
        className
      )}
    >
      {agents.map((agent, idx) => {
        const IconComp = ICON_MAP[agent.icon] ?? Loader2;
        const isLast = idx === agents.length - 1;

        return (
          <React.Fragment key={agent.name}>
            <motion.div
              className="flex flex-col items-center"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: idx * 0.15, duration: 0.3 }}
            >
              <div
                className={cn(
                  "flex h-9 w-9 items-center justify-center rounded-full border-2 transition-colors duration-300",
                  agent.status === "running" &&
                    "border-blue-500 bg-blue-500/20 animate-pulse",
                  agent.status === "completed" &&
                    "border-green-500 bg-green-500/20",
                  agent.status === "failed" &&
                    "border-red-500 bg-red-500/20",
                  agent.status === "idle" && "border-zinc-700 bg-zinc-800/40"
                )}
                title={agent.reasoning || agent.description}
              >
                {agent.status === "running" && (
                  <Loader2 className="h-4 w-4 animate-spin text-blue-400" />
                )}
                {agent.status === "completed" && (
                  <Check className="h-4 w-4 text-green-400" />
                )}
                {agent.status === "failed" && (
                  <X className="h-4 w-4 text-red-400" />
                )}
                {agent.status === "idle" && (
                  <IconComp className="h-4 w-4 text-zinc-500" />
                )}
              </div>
              <span className="mt-1.5 max-w-[72px] truncate text-center text-[11px] text-zinc-400">
                {agent.cn_label}
              </span>
              {agent.latency_ms !== undefined && agent.latency_ms > 0 && (
                <span className="text-[9px] text-zinc-600">
                  {Math.round(agent.latency_ms)}ms
                </span>
              )}
            </motion.div>

            {!isLast && (
              <motion.div
                className={cn(
                  "h-0.5 flex-1 rounded-full transition-colors duration-500",
                  agent.status === "completed"
                    ? "bg-green-500/40"
                    : "bg-zinc-700/60"
                )}
                initial={{ scaleX: 0 }}
                animate={{ scaleX: 1 }}
                transition={{ delay: idx * 0.15 + 0.1, duration: 0.3 }}
              />
            )}
          </React.Fragment>
        );
      })}
    </div>
  );
}
