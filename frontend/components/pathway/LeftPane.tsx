"use client";

import React, { useState } from "react";
import { Network, BarChart3, History } from "lucide-react";
import { cn } from "@/lib/utils";
import { PathwayTree } from "@/components/pathway/PathwayTree";
import { BenchmarkList } from "@/components/pathway/BenchmarkList";
import { SimulationHistory } from "@/components/pathway/SimulationHistory";

/**
 * LeftPane — left workbench pane container (Task C.2).
 *
 * Combines the PathwayTree, BenchmarkList and SimulationHistory components in
 * a 3-tab layout ("Pathways" | "Benchmarks" | "History"). This is the
 * component that replaces the left-pane placeholders in `WorkbenchShell`.
 *
 * Tab bar is a manual button group (no shadcn Tabs component installed). The
 * active tab fills the remaining height; each child manages its own
 * ScrollArea so long lists scroll independently.
 */

type Tab = "pathways" | "benchmarks" | "history";

const TABS: { id: Tab; label: string; icon: React.ReactNode }[] = [
  { id: "pathways", label: "Pathways", icon: <Network className="h-3.5 w-3.5" /> },
  { id: "benchmarks", label: "Benchmarks", icon: <BarChart3 className="h-3.5 w-3.5" /> },
  { id: "history", label: "History", icon: <History className="h-3.5 w-3.5" /> },
];

export function LeftPane() {
  const [tab, setTab] = useState<Tab>("pathways");

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div
        role="tablist"
        aria-label="Left pane"
        className="flex shrink-0 gap-1 rounded-lg border border-zinc-800 bg-zinc-900/60 p-1"
      >
        {TABS.map((t) => {
          const active = tab === t.id;
          return (
            <button
              key={t.id}
              role="tab"
              type="button"
              aria-selected={active}
              onClick={() => setTab(t.id)}
              className={cn(
                "flex flex-1 items-center justify-center gap-1.5 rounded-md px-2 py-1.5 text-xs font-medium transition-colors",
                active
                  ? "bg-blue-500/15 text-blue-200"
                  : "text-zinc-400 hover:bg-zinc-800/60 hover:text-zinc-200"
              )}
            >
              {t.icon}
              {t.label}
            </button>
          );
        })}
      </div>

      <div className="mt-2 min-h-0 flex-1 overflow-hidden">
        {tab === "pathways" && <PathwayTree />}
        {tab === "benchmarks" && <BenchmarkList />}
        {tab === "history" && <SimulationHistory />}
      </div>
    </div>
  );
}
