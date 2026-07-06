"use client";

import React from "react";
import { Cpu, Target, Timer, Zap } from "lucide-react";
import { cn } from "@/lib/utils";

interface TokenPerformanceBadgeProps {
  model?: string;
  ragHitRate?: number;
  latencyMs?: number;
  tokenUsage?: number;
  mcpTokensSaved?: number;
  className?: string;
}

/**
 * 实时 Token / 性能监控标签
 * 对应 1233.md 第四部分 §3：聊天气泡右下角的小型状态标签。
 * 展示 Model / RAG Hit Rate / Latency / Tokens / MCP Token 节省。
 */
export function TokenPerformanceBadge({
  model = "",
  ragHitRate,
  latencyMs,
  tokenUsage,
  mcpTokensSaved,
  className,
}: TokenPerformanceBadgeProps) {
  const hitRatePct =
    ragHitRate !== undefined ? Math.round(ragHitRate * 100) : null;
  const latencyLabel =
    latencyMs !== undefined
      ? latencyMs >= 1000
        ? `${(latencyMs / 1000).toFixed(1)}s`
        : `${Math.round(latencyMs)}ms`
      : null;

  return (
    <div
      className={cn(
        "flex flex-wrap items-center gap-x-3 gap-y-0.5 text-[10px] text-zinc-500",
        className
      )}
    >
      {model && (
        <span className="flex items-center gap-1">
          <Cpu className="h-2.5 w-2.5" />
          <span className="text-zinc-400">{model}</span>
        </span>
      )}

      {hitRatePct !== null && (
        <span className="flex items-center gap-1">
          <Target className="h-2.5 w-2.5" />
          <span
            className={cn(
              hitRatePct >= 70
                ? "text-green-400"
                : hitRatePct >= 40
                ? "text-yellow-400"
                : "text-red-400"
            )}
          >
            RAG {hitRatePct}%
          </span>
        </span>
      )}

      {latencyLabel && (
        <span className="flex items-center gap-1">
          <Timer className="h-2.5 w-2.5" />
          <span>{latencyLabel}</span>
        </span>
      )}

      {tokenUsage !== undefined && tokenUsage > 0 && (
        <span>Tokens: {tokenUsage.toLocaleString()}</span>
      )}

      {mcpTokensSaved !== undefined && mcpTokensSaved > 0 && (
        <span className="flex items-center gap-0.5 text-emerald-400">
          <Zap className="h-2.5 w-2.5" />
          MCP -{mcpTokensSaved}
        </span>
      )}
    </div>
  );
}
