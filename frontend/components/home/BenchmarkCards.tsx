"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import {
  Play,
  PlayCircle,
  CheckCircle2,
  XCircle,
  Loader2,
  ArrowUpRight,
  BarChart3,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";

/**
 * Benchmark Cases section (Task C.10, section 04).
 *
 * Grid of 10 BioModels reference benchmarks (one per pathway). Each card shows
 * a pass / fail / not-run status badge and a "Run" button; "Run All" sweeps
 * the set. Statuses persist across reloads via `localStorage`
 * (`biodynamics:benchmark_status`).
 *
 * The benchmark list mirrors `BenchmarkList.BENCHMARKS`. The actual run streams
 * from `POST /api/v4/benchmark/<class>` (Task C.12); until that endpoint lands,
 * "Run" simulates the lifecycle locally (running → passed) so the home-page
 * surface is fully exercisable today.
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

const STORAGE_KEY = "biodynamics:benchmark_status";

function loadStatuses(): Record<string, RunStatus> {
  if (typeof window === "undefined") return {};
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
      return {};
    }
    return parsed as Record<string, RunStatus>;
  } catch {
    return {};
  }
}

function saveStatuses(s: Record<string, RunStatus>) {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(s));
  } catch {
    /* ignore quota / private-mode errors */
  }
}

function StatusBadge({ status }: { status: RunStatus }) {
  if (status === "idle") {
    return (
      <span className="inline-flex items-center gap-1 rounded-full border border-zinc-700 px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider text-zinc-500">
        not run
      </span>
    );
  }
  if (status === "running") {
    return (
      <span className="inline-flex items-center gap-1 rounded-full border border-amber-700/60 bg-amber-500/10 px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider text-amber-300">
        <Loader2 className="h-2.5 w-2.5 animate-spin" />
        running
      </span>
    );
  }
  if (status === "passed") {
    return (
      <span className="inline-flex items-center gap-1 rounded-full border border-emerald-700/60 bg-emerald-500/10 px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider text-emerald-300">
        <CheckCircle2 className="h-2.5 w-2.5" />
        pass
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1 rounded-full border border-red-700/60 bg-red-500/10 px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider text-red-300">
      <XCircle className="h-2.5 w-2.5" />
      fail
    </span>
  );
}

export function BenchmarkCases() {
  const [statuses, setStatuses] = useState<Record<string, RunStatus>>({});
  const [runAllLoading, setRunAllLoading] = useState(false);

  useEffect(() => {
    setStatuses(loadStatuses());
  }, []);

  const apply = (next: Record<string, RunStatus>) => {
    setStatuses(next);
    saveStatuses(next);
  };

  const runOne = (id: string) =>
    new Promise<void>((resolve) => {
      apply({ ...loadStatuses(), [id]: "running" });
      // Placeholder for `POST /api/v4/benchmark/<class>` — real wiring in C.12.
      window.setTimeout(() => {
        apply({ ...loadStatuses(), [id]: "passed" });
        resolve();
      }, 700);
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
    <section className="py-14">
      <div className="mb-6 flex items-end justify-between gap-4 border-b border-zinc-800 pb-3">
        <div>
          <div className="font-mono text-[11px] uppercase tracking-[0.2em] text-zinc-500">
            04 / Benchmarks
          </div>
          <h2 className="mt-1 text-xl font-semibold tracking-tight text-zinc-100">
            Benchmark Cases
            <span className="ml-2 font-mono text-sm font-normal text-zinc-500">
              {completedCount}/{BENCHMARKS.length}
            </span>
          </h2>
          <p className="mt-0.5 text-sm text-zinc-500">
            Compare simulated dynamics against curated BioModels references.
          </p>
        </div>
        <Button
          size="sm"
          onClick={runAll}
          disabled={runAllLoading}
          className="gap-1 bg-blue-600 hover:bg-blue-700"
        >
          <PlayCircle className="h-3.5 w-3.5" />
          {runAllLoading ? "Running…" : "Run All Benchmarks"}
        </Button>
      </div>

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5">
        {BENCHMARKS.map((b) => {
          const status = statuses[b.id] ?? "idle";
          const running = status === "running" || runAllLoading;
          return (
            <div
              key={b.id}
              className={cn(
                "flex h-full flex-col rounded-lg border border-zinc-800 bg-zinc-900/40 p-4",
                status !== "idle" && "border-zinc-700 bg-zinc-900/70"
              )}
            >
              <div className="min-w-0">
                <div className="truncate text-sm font-medium text-zinc-100">
                  {b.displayName}
                </div>
                <div className="mt-0.5 font-mono text-[10px] text-zinc-500">
                  {b.biomdId}
                </div>
              </div>
              <div className="mt-3 flex-1">
                <StatusBadge status={status} />
              </div>
              <div className="mt-3 flex items-center gap-2">
                <Button
                  size="xs"
                  variant="outline"
                  disabled={running}
                  onClick={() => runOne(b.id)}
                  className="gap-1"
                >
                  <Play className="h-3 w-3" />
                  {status === "running" ? "…" : "Run"}
                </Button>
                <Link
                  href="/benchmarks"
                  className="inline-flex items-center gap-1 text-[11px] text-zinc-500 hover:text-zinc-200"
                >
                  details
                  <ArrowUpRight className="h-3 w-3" />
                </Link>
              </div>
            </div>
          );
        })}
      </div>

      <div className="mt-4 flex items-center justify-end">
        <Link
          href="/benchmarks"
          className="inline-flex items-center gap-1.5 text-xs text-zinc-400 hover:text-zinc-100"
        >
          <BarChart3 className="h-3.5 w-3.5" />
          Open Benchmark Center
        </Link>
      </div>
    </section>
  );
}
