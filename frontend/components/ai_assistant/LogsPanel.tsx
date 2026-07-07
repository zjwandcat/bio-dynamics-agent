"use client";

import React, { useEffect, useRef } from "react";
import { Trash2, Loader2, Check, X, Activity } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

/** Shape of a single agent dispatch entry (mirrors `DispatchData` in store). */
export interface LogEntry {
  target_agent: string;
  reasoning?: string;
  status: string;
  latency_ms?: number;
  node_name?: string;
}

export interface LogsPanelProps {
  dispatches: LogEntry[];
  onClear?: () => void;
  className?: string;
}

type StatusLabel = "running" | "done" | "error" | "idle";

interface NormalizedStatus {
  label: StatusLabel;
  color: string;
  bg: string;
  icon: React.ReactNode;
}

/**
 * Map the free-form `status` string emitted by the backend onto a small fixed
 * set of UI states. The SSE contract uses a mix of `running` / `completed` /
 * `failed` (agent_dispatch) and the occasional `start` / `done` / `error` — we
 * collapse all of them into the four buckets below.
 */
function normalizeStatus(status: string): NormalizedStatus {
  const s = status.toLowerCase();
  if (
    s === "running" ||
    s === "start" ||
    s === "started" ||
    s === "pending" ||
    s === "in_progress"
  ) {
    return {
      label: "running",
      color: "text-blue-300",
      bg: "bg-blue-500/10",
      icon: <Loader2 className="h-3 w-3 animate-spin" />,
    };
  }
  if (
    s === "completed" ||
    s === "done" ||
    s === "success" ||
    s === "ok" ||
    s === "finished"
  ) {
    return {
      label: "done",
      color: "text-emerald-300",
      bg: "bg-emerald-500/10",
      icon: <Check className="h-3 w-3" />,
    };
  }
  if (s === "failed" || s === "error" || s === "errored" || s === "aborted") {
    return {
      label: "error",
      color: "text-red-300",
      bg: "bg-red-500/10",
      icon: <X className="h-3 w-3" />,
    };
  }
  return {
    label: "idle",
    color: "text-zinc-500",
    bg: "bg-zinc-700/10",
    icon: <span className="h-1.5 w-1.5 rounded-full bg-zinc-500" />,
  };
}

function formatTime(date: Date): string {
  return date.toLocaleTimeString("en-US", { hour12: false });
}

/**
 * Logs tab content for the AI Assistant panel.
 *
 * Renders a real-time stream of agent dispatches (the `agent_dispatch` SSE
 * events accumulated in the global store). Each row shows the wall-clock
 * timestamp it was first seen, the target agent, the action / reasoning, the
 * normalized status (running / done / error), and the latency in ms. The panel
 * auto-scrolls to the latest entry and exposes a "Clear" button to flush the
 * visible log buffer.
 *
 * Timestamps are tracked client-side via a ref Map keyed by dispatch index,
 * because the backend `agent_dispatch` payload does not carry a timestamp.
 */
export function LogsPanel({ dispatches, onClear, className }: LogsPanelProps) {
  const endRef = useRef<HTMLDivElement>(null);
  const timestampsRef = useRef<Map<number, string>>(new Map());

  // Record a first-seen timestamp for each new dispatch index. The Map is
  // keyed by array index, which is stable for the lifetime of a dispatch
  // (the store only ever appends, and a clear resets the array to []).
  dispatches.forEach((_, idx) => {
    if (!timestampsRef.current.has(idx)) {
      timestampsRef.current.set(idx, formatTime(new Date()));
    }
  });

  // Reset the timestamp map whenever the dispatch buffer is cleared so stale
  // entries don't bleed into the next run.
  useEffect(() => {
    if (dispatches.length === 0) {
      timestampsRef.current.clear();
    }
  }, [dispatches.length]);

  // Auto-scroll to the bottom whenever a new dispatch arrives.
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [dispatches.length]);

  return (
    <div className={cn("flex h-full min-h-0 flex-col", className)}>
      {/* Header strip with count + clear button */}
      <div className="flex shrink-0 items-center justify-between border-b border-zinc-800 px-3 py-1.5">
        <div className="flex items-center gap-1.5">
          <Activity className="h-3 w-3 text-zinc-500" />
          <span className="text-[11px] font-medium uppercase tracking-wide text-zinc-500">
            Agent Dispatch Log
          </span>
          {dispatches.length > 0 && (
            <span className="rounded-full bg-zinc-800 px-1.5 text-[10px] text-zinc-400">
              {dispatches.length}
            </span>
          )}
        </div>
        <Button
          variant="ghost"
          size="icon-xs"
          onClick={onClear}
          disabled={dispatches.length === 0}
          title="Clear logs"
          className="text-zinc-400 hover:text-red-300"
        >
          <Trash2 className="h-3 w-3" />
        </Button>
      </div>

      {/* Scrollable log stream */}
      <div className="min-h-0 flex-1 overflow-auto">
        {dispatches.length === 0 ? (
          <div className="flex flex-col items-center justify-center gap-2 pt-20 text-center text-zinc-500">
            <Activity className="h-8 w-8 text-zinc-700" />
            <p className="text-xs">No dispatch logs yet</p>
            <p className="px-6 text-[11px] leading-relaxed text-zinc-600">
              Agent dispatches from the next simulation run will stream here in
              real time.
            </p>
          </div>
        ) : (
          <div className="space-y-1 p-2 font-mono text-[11px]">
            {dispatches.map((d, idx) => {
              const ts = timestampsRef.current.get(idx) ?? "--:--:--";
              const status = normalizeStatus(d.status);
              const action =
                d.reasoning ||
                (d.node_name ? `node: ${d.node_name}` : "dispatch");
              return (
                <div
                  key={idx}
                  className={cn(
                    "rounded border border-zinc-800 px-2 py-1.5",
                    status.bg
                  )}
                >
                  <div className="flex items-center gap-2">
                    <span className="shrink-0 text-zinc-600">[{ts}]</span>
                    <span
                      className={cn("flex shrink-0 items-center gap-1", status.color)}
                    >
                      {status.icon}
                      {status.label}
                    </span>
                    <span className="truncate font-semibold text-zinc-200">
                      {d.target_agent}
                    </span>
                    {d.latency_ms !== undefined && d.latency_ms > 0 && (
                      <span className="ml-auto shrink-0 text-zinc-500">
                        {Math.round(d.latency_ms)}ms
                      </span>
                    )}
                  </div>
                  {d.reasoning && (
                    <div
                      className="mt-0.5 truncate text-zinc-500"
                      title={action}
                    >
                      {action}
                    </div>
                  )}
                </div>
              );
            })}
            <div ref={endRef} />
          </div>
        )}
      </div>
    </div>
  );
}
