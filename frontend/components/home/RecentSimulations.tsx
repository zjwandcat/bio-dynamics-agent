"use client";

import React, { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import {
  History,
  Clock,
  CheckCircle2,
  XCircle,
  Loader2,
  ArrowUpRight,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { useWorkbenchStore } from "@/lib/store";
import type { PathwayClass, SimulationResult } from "@/lib/api";

/**
 * Recent Simulations section (Task C.10, section 03).
 *
 * Reads recent simulation runs from `localStorage` (`biodynamics:sim_history`,
 * the same key written by the Simulation Tabs pane in C.4 / SimulationHistory).
 * "Open" hydrates the global workbench store with the persisted result + pathway
 * and navigates to `/workspace?pathway=<class>` so the run reopens in context.
 *
 * Storage shape mirrors `SimulationHistory.HistoryEntry`.
 */

const STORAGE_KEY = "biodynamics:sim_history";

interface HistoryEntry {
  run_id: string;
  pathway_class: PathwayClass;
  pathway_name: string;
  /** Unix epoch ms. */
  timestamp: number;
  status: "running" | "completed" | "failed";
  /** Persisted result payload — loaded into the store on "Open". */
  result?: SimulationResult;
}

function loadHistory(): HistoryEntry[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed as HistoryEntry[];
  } catch {
    return [];
  }
}

function formatTimestamp(ms: number): string {
  try {
    return new Date(ms).toLocaleString(undefined, {
      month: "short",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return "";
  }
}

function StatusIcon({ status }: { status: HistoryEntry["status"] }) {
  if (status === "running") {
    return <Loader2 className="h-3.5 w-3.5 animate-spin text-amber-400" />;
  }
  if (status === "completed") {
    return <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400" />;
  }
  return <XCircle className="h-3.5 w-3.5 text-red-400" />;
}

export function RecentSimulations() {
  const router = useRouter();
  const setSimulationResult = useWorkbenchStore((s) => s.setSimulationResult);
  const setCurrentPathway = useWorkbenchStore((s) => s.setCurrentPathway);
  const [entries, setEntries] = useState<HistoryEntry[]>([]);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    setEntries(loadHistory());
    setLoaded(true);
    const handler = (e: StorageEvent) => {
      if (e.key === STORAGE_KEY) setEntries(loadHistory());
    };
    window.addEventListener("storage", handler);
    return () => window.removeEventListener("storage", handler);
  }, []);

  const open = (entry: HistoryEntry) => {
    if (entry.result) setSimulationResult(entry.result);
    setCurrentPathway(entry.pathway_class);
    router.push(`/workspace?pathway=${entry.pathway_class}`);
  };

  return (
    <section className="py-14">
      <div className="mb-6 flex items-end justify-between gap-4 border-b border-zinc-800 pb-3">
        <div>
          <div className="font-mono text-[11px] uppercase tracking-[0.2em] text-zinc-500">
            03 / History
          </div>
          <h2 className="mt-1 text-xl font-semibold tracking-tight text-zinc-100">
            Recent Simulations
          </h2>
          <p className="mt-0.5 text-sm text-zinc-500">
            Reopen a recent run from local history.
          </p>
        </div>
        <Link
          href="/workspace"
          className="hidden text-xs text-zinc-400 hover:text-zinc-100 sm:inline"
        >
          launch workspace →
        </Link>
      </div>

      {loaded && entries.length === 0 ? (
        <div className="flex flex-col items-center justify-center gap-2 rounded-lg border border-dashed border-zinc-800 bg-zinc-900/30 px-6 py-14 text-center">
          <History className="h-6 w-6 text-zinc-700" />
          <p className="text-sm text-zinc-400">
            No simulations yet. Launch the workspace to get started.
          </p>
        </div>
      ) : (
        <div className="divide-y divide-zinc-800 overflow-hidden rounded-lg border border-zinc-800 bg-zinc-900/30">
          {entries.slice(0, 8).map((entry) => (
            <div
              key={entry.run_id}
              className="flex items-center gap-3 px-4 py-3"
            >
              <StatusIcon status={entry.status} />
              <div className="min-w-0 flex-1">
                <div className="truncate text-sm font-medium text-zinc-100">
                  {entry.pathway_name}
                </div>
                <div className="flex items-center gap-1.5 font-mono text-[11px] text-zinc-500">
                  <Clock className="h-3 w-3" />
                  {formatTimestamp(entry.timestamp)}
                </div>
              </div>
              <Button
                size="xs"
                variant="outline"
                onClick={() => open(entry)}
                className="gap-1"
              >
                Open
                <ArrowUpRight className="h-3 w-3" />
              </Button>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
