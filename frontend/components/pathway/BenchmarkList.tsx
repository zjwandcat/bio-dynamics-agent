"use client";

import React, { useState } from "react";
import {
  Play,
  PlayCircle,
  CheckCircle2,
  XCircle,
  Loader2,
  BarChart3,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";

/**
 * BenchmarkList — left-pane benchmark runner (Task C.2).
 *
 * Lists 10 BioModels reference benchmarks (one per pathway) with a per-row
 * "Run" button and a "Run All" action. Each row shows a pass/fail/running
 * badge once the benchmark has been executed.
 *
 * The real implementation will stream results from
 * `POST /api/v4/benchmarks/run` (SSE). Until that endpoint lands, clicking
 * "Run" dispatches a local action that simulates the run lifecycle
 * (running → passed) so the UI is fully exercisable today.
 */

interface BenchmarkDef {
  id: string;
  pathwayClass: string;
  displayName: string;
  biomdId: string;
}

const BENCHMARKS: BenchmarkDef[] = [
  { id: "bm_egfr", pathwayClass: "egfr", displayName: "EGFR RTK Signaling", biomdId: "BIOMD0000000017" },
  { id: "bm_mapk", pathwayClass: "mapk", displayName: "MAPK Cascade", biomdId: "BIOMD0000000010" },
  { id: "bm_pi3k", pathwayClass: "pi3k_akt_mtor", displayName: "PI3K-AKT-mTOR", biomdId: "BIOMD0000000032" },
  { id: "bm_p53", pathwayClass: "p53", displayName: "p53 Signaling", biomdId: "BIOMD0000000059" },
  { id: "bm_apoptosis", pathwayClass: "apoptosis", displayName: "Apoptosis", biomdId: "BIOMD0000000033" },
  { id: "bm_cell_cycle", pathwayClass: "cell_cycle", displayName: "Cell Cycle", biomdId: "BIOMD0000000055" },
  { id: "bm_jak_stat", pathwayClass: "jak_stat", displayName: "JAK-STAT", biomdId: "BIOMD0000000056" },
  { id: "bm_nfkb", pathwayClass: "nf_kappa_b", displayName: "NF-κB Signaling", biomdId: "BIOMD0000000027" },
  { id: "bm_wnt", pathwayClass: "wnt", displayName: "Wnt Signaling", biomdId: "BIOMD0000000052" },
  { id: "bm_tgfb", pathwayClass: "tgf_beta", displayName: "TGF-β Signaling", biomdId: "BIOMD0000000049" },
];

type RunStatus = "idle" | "running" | "passed" | "failed";

function StatusBadge({ status }: { status: RunStatus }) {
  if (status === "idle") return null;
  if (status === "running") {
    return (
      <Badge
        variant="outline"
        className="h-4 gap-0.5 border-amber-700/60 bg-amber-500/10 px-1 text-[9px] text-amber-300"
      >
        <Loader2 className="h-2.5 w-2.5 animate-spin" />
        running
      </Badge>
    );
  }
  if (status === "passed") {
    return (
      <Badge
        variant="outline"
        className="h-4 gap-0.5 border-emerald-700/60 bg-emerald-500/10 px-1 text-[9px] text-emerald-300"
      >
        <CheckCircle2 className="h-2.5 w-2.5" />
        pass
      </Badge>
    );
  }
  return (
    <Badge
      variant="outline"
      className="h-4 gap-0.5 border-red-700/60 bg-red-500/10 px-1 text-[9px] text-red-300"
    >
      <XCircle className="h-2.5 w-2.5" />
      fail
    </Badge>
  );
}

export function BenchmarkList() {
  const [statuses, setStatuses] = useState<Record<string, RunStatus>>({});
  const [runAllLoading, setRunAllLoading] = useState(false);

  const runOne = (id: string) =>
    new Promise<void>((resolve) => {
      setStatuses((s) => ({ ...s, [id]: "running" }));
      // Placeholder for `POST /api/v4/benchmarks/run` SSE — dispatches a store
      // action today; the real stream wiring lands in a later sprint task.
      window.setTimeout(() => {
        setStatuses((s) => ({ ...s, [id]: "passed" }));
        resolve();
      }, 600);
    });

  const runAll = async () => {
    setRunAllLoading(true);
    for (const b of BENCHMARKS) {
      await runOne(b.id);
    }
    setRunAllLoading(false);
  };

  const completedCount = Object.values(statuses).filter(
    (s) => s === "passed" || s === "failed"
  ).length;

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex shrink-0 items-center justify-between px-0.5 pb-1.5">
        <div className="flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wide text-zinc-500">
          <BarChart3 className="h-3.5 w-3.5" />
          Benchmarks
          <span className="text-zinc-600">({completedCount}/{BENCHMARKS.length})</span>
        </div>
        <Button
          type="button"
          size="xs"
          variant="default"
          disabled={runAllLoading}
          onClick={runAll}
        >
          <PlayCircle className="h-3 w-3" />
          {runAllLoading ? "Running…" : "Run All"}
        </Button>
      </div>

      <ScrollArea className="min-h-0 flex-1">
        <div className="space-y-1 pr-1.5">
          {BENCHMARKS.map((b) => {
            const status = statuses[b.id] ?? "idle";
            const running = status === "running" || runAllLoading;
            return (
              <div
                key={b.id}
                className={cn(
                  "flex items-center gap-2 rounded-md border border-zinc-800 bg-zinc-950/50 px-2.5 py-1.5",
                  status !== "idle" && "bg-zinc-900/60"
                )}
              >
                <div className="min-w-0 flex-1">
                  <div className="truncate text-xs font-medium text-zinc-200">
                    {b.displayName}
                  </div>
                  <div className="truncate text-[10px] text-zinc-500">
                    {b.biomdId}
                  </div>
                </div>
                <StatusBadge status={status} />
                <Button
                  type="button"
                  size="xs"
                  variant="outline"
                  disabled={running}
                  onClick={() => runOne(b.id)}
                >
                  <Play className="h-3 w-3" />
                  {status === "running" ? "…" : "Run"}
                </Button>
              </div>
            );
          })}
        </div>
      </ScrollArea>
    </div>
  );
}
