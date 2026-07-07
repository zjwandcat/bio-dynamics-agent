"use client";

import React, { useState } from "react";
import {
  Play,
  CheckCircle2,
  XCircle,
  Loader2,
  ChevronDown,
  ChevronRight,
  Clock,
  AlertCircle,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import type {
  BenchmarkCheck,
  BenchmarkResultEvent,
} from "@/lib/benchmarkSse";

// ---------------------------------------------------------------------------
// Static definition (one per pathway) — owned by the page, passed in.
// ---------------------------------------------------------------------------

export interface BenchmarkDef {
  pathway_class: string;
  name: string;
  description: string;
}

// ---------------------------------------------------------------------------
// Runtime state for a single card — owned by the page, passed in.
// ---------------------------------------------------------------------------

export type CardStatus = "idle" | "running" | "pass" | "fail";

export interface BenchmarkCardState {
  status: CardStatus;
  /** Current progress step label while running (e.g. "loading_specialist"). */
  step?: string;
  /** Last completed result payload (set when status is pass/fail). */
  result?: BenchmarkResultEvent;
  /** Epoch ms of the last completed run. */
  lastRunAt?: number;
  /** Error message captured from the stream (fatal per-card error). */
  error?: string;
}

export interface BenchmarkCardProps {
  def: BenchmarkDef;
  state: BenchmarkCardState;
  /** True while a "Run All" suite is in flight — disables the per-card Run button. */
  suiteRunning: boolean;
  /** Trigger a single-card run. */
  onRun: (pathwayClass: string) => void;
}

// ---------------------------------------------------------------------------
// Status badge
// ---------------------------------------------------------------------------

function StatusBadge({ status }: { status: CardStatus }) {
  switch (status) {
    case "running":
      return (
        <Badge
          variant="outline"
          className="h-5 gap-1 border-blue-700/60 bg-blue-500/10 px-2 text-[11px] text-blue-300"
        >
          <Loader2 className="h-3 w-3 animate-spin" />
          Running
        </Badge>
      );
    case "pass":
      return (
        <Badge
          variant="outline"
          className="h-5 gap-1 border-emerald-700/60 bg-emerald-500/10 px-2 text-[11px] text-emerald-300"
        >
          <CheckCircle2 className="h-3 w-3" />
          Pass
        </Badge>
      );
    case "fail":
      return (
        <Badge
          variant="outline"
          className="h-5 gap-1 border-red-700/60 bg-red-500/10 px-2 text-[11px] text-red-300"
        >
          <XCircle className="h-3 w-3" />
          Fail
        </Badge>
      );
    default:
      return (
        <Badge
          variant="outline"
          className="h-5 gap-1 border-zinc-700 bg-zinc-800/40 px-2 text-[11px] text-zinc-400"
        >
          Not Run
        </Badge>
      );
  }
}

// ---------------------------------------------------------------------------
// Pass-criteria checklist row
// ---------------------------------------------------------------------------

function CriterionRow({ check }: { check: BenchmarkCheck }) {
  return (
    <li className="flex items-start gap-2">
      <span className="mt-0.5 shrink-0" aria-hidden>
        {check.passed ? (
          <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400" />
        ) : (
          <XCircle className="h-3.5 w-3.5 text-red-400" />
        )}
      </span>
      <div className="min-w-0 flex-1 space-y-0.5">
        <div className="flex items-baseline justify-between gap-2">
          <span className="text-[11px] font-medium text-zinc-200">
            {check.criterion || check.metric_name || "criterion"}
          </span>
          <span className="shrink-0 font-mono text-[10px] text-zinc-500">
            {check.metric_name}
          </span>
        </div>
        {check.detail && (
          <p
            className={cn(
              "text-[10px] leading-snug",
              check.passed ? "text-emerald-300/80" : "text-red-300/90"
            )}
          >
            {check.detail}
          </p>
        )}
      </div>
    </li>
  );
}

// ---------------------------------------------------------------------------
// Card
// ---------------------------------------------------------------------------

function formatTimestamp(ms: number): string {
  try {
    return new Date(ms).toLocaleTimeString("en-US", {
      hour12: false,
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  } catch {
    return "";
  }
}

export function BenchmarkCard({
  def,
  state,
  suiteRunning,
  onRun,
}: BenchmarkCardProps) {
  const [expanded, setExpanded] = useState(false);
  const { status, step, result, lastRunAt, error } = state;

  const running = status === "running";
  const disabled = running || suiteRunning;
  const checks = result?.checks ?? [];
  const passedChecks = checks.filter((c) => c.passed).length;
  const canExpand = checks.length > 0 || !!error || (result?.errors?.length ?? 0) > 0;
  const showExpanded = expanded;

  return (
    <div
      className={cn(
        "flex flex-col rounded-xl border bg-zinc-950/60 p-3 transition-colors",
        status === "pass" && "border-emerald-800/50",
        status === "fail" && "border-red-800/50",
        status === "running" && "border-blue-800/60",
        status === "idle" && "border-zinc-800"
      )}
    >
      {/* Header row: name + status */}
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0 flex-1">
          <h3 className="truncate text-sm font-semibold text-zinc-100">
            {def.name}
          </h3>
          <p className="mt-0.5 font-mono text-[10px] text-zinc-500">
            {def.pathway_class}
          </p>
        </div>
        <StatusBadge status={status} />
      </div>

      {/* Description */}
      <p className="mt-2 line-clamp-2 text-[11px] leading-relaxed text-zinc-400">
        {def.description}
      </p>

      {/* Meta row: runtime + last-run timestamp */}
      <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-[10px] text-zinc-500">
        {result?.runtime_seconds !== undefined && (
          <span className="inline-flex items-center gap-1">
            <Clock className="h-3 w-3" />
            {result.runtime_seconds.toFixed(3)}s
          </span>
        )}
        {lastRunAt && (
          <span className="inline-flex items-center gap-1">
            <Clock className="h-3 w-3" />
            {formatTimestamp(lastRunAt)}
          </span>
        )}
        {running && step && (
          <span className="inline-flex items-center gap-1 text-blue-300/80">
            <Loader2 className="h-3 w-3 animate-spin" />
            {step === "loading_specialist"
              ? "loading specialist"
              : step === "validation_complete"
                ? "validation complete"
                : step}
          </span>
        )}
        {checks.length > 0 && (
          <span className="text-zinc-400">
            {passedChecks}/{checks.length} criteria
          </span>
        )}
      </div>

      {/* Errors (non-fatal, from runner) */}
      {result?.errors && result.errors.length > 0 && (
        <div className="mt-2 flex items-start gap-1.5 rounded-md border border-amber-800/50 bg-amber-500/5 px-2 py-1">
          <AlertCircle className="mt-0.5 h-3 w-3 shrink-0 text-amber-400" />
          <p className="text-[10px] leading-snug text-amber-300/90">
            {result.errors.join("; ")}
          </p>
        </div>
      )}

      {/* Fatal stream error */}
      {error && (
        <div className="mt-2 flex items-start gap-1.5 rounded-md border border-red-800/50 bg-red-500/5 px-2 py-1">
          <AlertCircle className="mt-0.5 h-3 w-3 shrink-0 text-red-400" />
          <p className="text-[10px] leading-snug text-red-300/90">{error}</p>
        </div>
      )}

      {/* Actions */}
      <div className="mt-3 flex items-center gap-2">
        <Button
          type="button"
          size="xs"
          variant={status === "idle" ? "default" : "outline"}
          disabled={disabled}
          onClick={() => onRun(def.pathway_class)}
        >
          {running ? (
            <Loader2 className="h-3 w-3 animate-spin" />
          ) : (
            <Play className="h-3 w-3" />
          )}
          {running ? "Running…" : "Run"}
        </Button>

        {canExpand && (
          <Button
            type="button"
            size="xs"
            variant="ghost"
            onClick={() => setExpanded((v) => !v)}
            aria-expanded={showExpanded}
          >
            {showExpanded ? (
              <ChevronDown className="h-3 w-3" />
            ) : (
              <ChevronRight className="h-3 w-3" />
            )}
            {showExpanded ? "Hide" : "Details"}
          </Button>
        )}
      </div>

      {/* Expandable criteria checklist */}
      {showExpanded && canExpand && (
        <div className="mt-2 space-y-1.5 rounded-md border border-zinc-800 bg-zinc-950/60 p-2">
          <div className="text-[10px] font-medium uppercase tracking-wide text-zinc-500">
            Pass Criteria
          </div>
          {checks.length > 0 ? (
            <ul className="space-y-1.5">
              {checks.map((check, idx) => (
                <CriterionRow
                  key={`${def.pathway_class}-${idx}-${check.metric_name}`}
                  check={check}
                />
              ))}
            </ul>
          ) : (
            <p className="text-[10px] text-zinc-500">
              No criteria evaluated for this run.
            </p>
          )}
        </div>
      )}
    </div>
  );
}

export default BenchmarkCard;
