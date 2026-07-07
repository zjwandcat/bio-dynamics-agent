"use client";

import React, { useEffect, useState } from "react";
import {
  History,
  Clock,
  CheckCircle2,
  XCircle,
  Loader2,
  ExternalLink,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useWorkbenchStore } from "@/lib/store";
import type { PathwayClass, SimulationResult } from "@/lib/api";

/**
 * SimulationHistory — left-pane recent-simulation list (Task C.2).
 *
 * Reads recent simulation runs from `localStorage` and lets the user reopen
 * one by loading its result back into the global workbench store
 * (`setSimulationResult` + `setCurrentPathway`). Simulations are written by
 * the Simulation Tabs pane (Task C.4); until then the list shows its empty
 * state.
 */

const STORAGE_KEY = "biodynamics:sim_history";

export interface HistoryEntry {
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
    return <Loader2 className="h-3 w-3 animate-spin text-amber-400" />;
  }
  if (status === "completed") {
    return <CheckCircle2 className="h-3 w-3 text-emerald-400" />;
  }
  return <XCircle className="h-3 w-3 text-red-400" />;
}

export function SimulationHistory() {
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
    if (!entry.result) return;
    setSimulationResult(entry.result);
    setCurrentPathway(entry.pathway_class);
  };

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex shrink-0 items-center gap-1.5 px-0.5 pb-1.5 text-[11px] font-medium uppercase tracking-wide text-zinc-500">
        <History className="h-3.5 w-3.5" />
        Simulation History
      </div>

      <ScrollArea className="min-h-0 flex-1">
        <div className="pr-1.5">
          {loaded && entries.length === 0 ? (
            <div className="flex flex-col items-center justify-center gap-1.5 py-10 text-center">
              <History className="h-5 w-5 text-zinc-700" />
              <p className="text-xs text-zinc-500">No simulations yet</p>
              <p className="text-[10px] text-zinc-600">
                Run a simulation to populate history
              </p>
            </div>
          ) : (
            <div className="space-y-1">
              {entries.map((entry) => {
                const disabled = !entry.result;
                return (
                  <div
                    key={entry.run_id}
                    className="flex items-center gap-2 rounded-md border border-zinc-800 bg-zinc-950/50 px-2.5 py-1.5"
                  >
                    <StatusIcon status={entry.status} />
                    <div className="min-w-0 flex-1">
                      <div className="truncate text-xs font-medium text-zinc-200">
                        {entry.pathway_name}
                      </div>
                      <div className="flex items-center gap-1 text-[10px] text-zinc-500">
                        <Clock className="h-2.5 w-2.5" />
                        {formatTimestamp(entry.timestamp)}
                      </div>
                    </div>
                    <Button
                      type="button"
                      size="xs"
                      variant="outline"
                      disabled={disabled}
                      onClick={() => open(entry)}
                      title={disabled ? "Result not persisted" : "Load into workspace"}
                    >
                      <ExternalLink className="h-3 w-3" />
                      Open
                    </Button>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </ScrollArea>
    </div>
  );
}
