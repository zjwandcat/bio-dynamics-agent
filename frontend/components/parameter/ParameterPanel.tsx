"use client";

import React, { useState } from "react";
import { Sliders } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { ParameterExplorer } from "./ParameterExplorer";

/**
 * Parameter Panel — center-pane wrapper around `ParameterExplorer` (Task C.5).
 *
 * Provides the "Parameter Explorer" header with a live modified/baseline
 * status badge, and hosts the explorer itself. This component replaces the
 * center-pane "Parameter Editor" placeholder from `WorkbenchShell` (the wiring
 * is performed by the integration task — `WorkbenchShell` is intentionally not
 * edited here).
 *
 * The modified/baseline status is reported upward from `ParameterExplorer`
 * via the `onStatusChange` callback so the header badge stays in sync with the
 * explorer's working state without lifting the parameter state out of it.
 */
export function ParameterPanel() {
  const [modified, setModified] = useState(false);

  return (
    <section className="flex h-full min-h-0 flex-col rounded-lg border border-zinc-800 bg-zinc-900/60">
      <header className="flex shrink-0 items-center justify-between gap-2 border-b border-zinc-800 px-3 py-2">
        <div className="flex items-center gap-2">
          <Sliders className="h-3.5 w-3.5 text-emerald-400" />
          <h3 className="text-xs font-semibold text-zinc-200">
            Parameter Explorer
          </h3>
        </div>
        <Badge
          variant="outline"
          className={
            modified
              ? "border-amber-600/50 text-[10px] text-amber-300"
              : "border-zinc-700 text-[10px] text-zinc-500"
          }
        >
          {modified ? "modified" : "baseline"}
        </Badge>
      </header>

      <div className="flex min-h-0 flex-1 flex-col p-2">
        <ParameterExplorer onStatusChange={setModified} />
      </div>
    </section>
  );
}
