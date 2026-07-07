"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronDown, Zap, Clock, CheckCircle2, AlertCircle, XCircle } from "lucide-react";
import { cn } from "@/lib/utils";

// MCP 工具调用记录（与后端 ToolCallRecord.to_dict() 对应）
export interface MCPToolCall {
  tool_key: string;
  tool_name: string;
  icon: string;
  action: string;
  status: "success" | "fallback" | "failed";
  input_summary: string;
  output_summary: string;
  latency_ms: number;
  tokens_saved: number;
  detail?: Record<string, unknown>;
}

interface MCPToolPanelProps {
  toolCalls: MCPToolCall[];
  tokensSaved: number;
  className?: string;
}

// 状态对应的图标与颜色
const STATUS_CONFIG = {
  success: {
    icon: CheckCircle2,
    color: "text-green-400",
    bg: "bg-green-500/10",
    label: "成功",
  },
  fallback: {
    icon: AlertCircle,
    color: "text-amber-400",
    bg: "bg-amber-500/10",
    label: "降级",
  },
  failed: {
    icon: XCircle,
    color: "text-red-400",
    bg: "bg-red-500/10",
    label: "失败",
  },
} as const;

/**
 * MCP 工具调用状态面板
 * 展示每次 MCP 工具调用的状态、耗时与 Token 节省量，
 * 让用户直观感知 MCP 工具链的工作过程与价值。
 */
export function MCPToolPanel({
  toolCalls,
  tokensSaved,
  className,
}: MCPToolPanelProps) {
  const [expanded, setExpanded] = useState(true);
  const [expandedDetail, setExpandedDetail] = useState<number | null>(null);

  if (toolCalls.length === 0) return null;

  return (
    <div
      className={cn(
        "rounded-lg border border-white/10 bg-zinc-900/60",
        className
      )}
    >
      {/* 折叠头部 */}
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        className="flex w-full items-center justify-between px-3 py-2 text-left"
      >
        <div className="flex items-center gap-2">
          <span className="text-sm">🔧</span>
          <span className="text-xs font-medium text-zinc-200">
            MCP 工具调用
          </span>
          <span className="rounded-full bg-zinc-700/50 px-2 py-0.5 text-[10px] text-zinc-400">
            {toolCalls.length} 次调用
          </span>
        </div>
        <div className="flex items-center gap-2">
          {tokensSaved > 0 && (
            <span className="flex items-center gap-1 rounded-full bg-emerald-500/10 px-2 py-0.5 text-[10px] text-emerald-300">
              <Zap className="h-2.5 w-2.5" />
              节省 {tokensSaved} Token
            </span>
          )}
          <ChevronDown
            className={cn(
              "h-4 w-4 text-zinc-500 transition-transform",
              expanded && "rotate-180"
            )}
          />
        </div>
      </button>

      <AnimatePresence initial={false}>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25 }}
            className="overflow-hidden"
          >
            <div className="space-y-1.5 px-3 pb-3">
              {toolCalls.map((call, idx) => {
                const config = STATUS_CONFIG[call.status] ?? STATUS_CONFIG.fallback;
                const StatusIcon = config.icon;
                const hasDetail =
                  call.detail &&
                  Object.keys(call.detail).length > 0;

                return (
                  <div
                    key={idx}
                    className="rounded-md border border-zinc-700/50 bg-zinc-800/40 p-2"
                  >
                    {/* 工具调用头部行 */}
                    <div className="flex items-start gap-2">
                      <span className="mt-0.5 text-sm flex-shrink-0">
                        {call.icon}
                      </span>
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-1.5">
                          <span className="truncate text-xs font-medium text-zinc-200">
                            {call.tool_name}
                          </span>
                          <span
                            className={cn(
                              "flex flex-shrink-0 items-center gap-0.5 rounded-full px-1.5 py-0.5 text-[9px]",
                              config.bg,
                              config.color
                            )}
                          >
                            <StatusIcon className="h-2 w-2" />
                            {config.label}
                          </span>
                        </div>
                        <p className="mt-0.5 text-[11px] text-zinc-400">
                          {call.action}
                        </p>
                      </div>
                    </div>

                    {/* 输出摘要 */}
                    <p className="mt-1 ml-6 text-[10px] text-zinc-500">
                      {call.output_summary}
                    </p>

                    {/* 耗时与 Token 节省 */}
                    <div className="mt-1 ml-6 flex items-center gap-3 text-[10px] text-zinc-600">
                      {call.latency_ms > 0 && (
                        <span className="flex items-center gap-0.5">
                          <Clock className="h-2.5 w-2.5" />
                          {call.latency_ms >= 1000
                            ? `${(call.latency_ms / 1000).toFixed(1)}s`
                            : `${Math.round(call.latency_ms)}ms`}
                        </span>
                      )}
                      {call.tokens_saved > 0 && (
                        <span className="flex items-center gap-0.5 text-emerald-500">
                          <Zap className="h-2.5 w-2.5" />
                          -{call.tokens_saved}
                        </span>
                      )}
                    </div>

                    {/* 可展开的详情区域 */}
                    {hasDetail && (
                      <div className="mt-1 ml-6">
                        <button
                          type="button"
                          onClick={() =>
                            setExpandedDetail(
                              expandedDetail === idx ? null : idx
                            )
                          }
                          className="text-[10px] text-zinc-500 transition-colors hover:text-zinc-300"
                        >
                          {expandedDetail === idx ? "收起详情" : "查看详情"}
                        </button>
                        <AnimatePresence>
                          {expandedDetail === idx && (
                            <motion.pre
                              initial={{ height: 0, opacity: 0 }}
                              animate={{ height: "auto", opacity: 1 }}
                              exit={{ height: 0, opacity: 0 }}
                              transition={{ duration: 0.2 }}
                              className="mt-1 max-h-40 overflow-auto rounded bg-black/40 p-2 text-[9px] text-zinc-400"
                            >
                              <code className="whitespace-pre-wrap break-all">
                                {JSON.stringify(call.detail, null, 2)}
                              </code>
                            </motion.pre>
                          )}
                        </AnimatePresence>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
