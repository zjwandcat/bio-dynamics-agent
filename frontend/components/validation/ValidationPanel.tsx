"use client";

import React from "react";
import { ShieldCheck } from "lucide-react";
import { useWorkbenchStore } from "@/lib/store";
import { ValidationPyramid, type ValidationReport } from "./ValidationPyramid";

/**
 * ValidationPanel — right-pane wrapper around the Validation Pyramid.
 *
 * Reads the `validationReport` slice of the workbench store (hydrated by the
 * `v4_validation_report` SSE event) and renders either the pyramid or the
 * empty state. This component is the replacement for the right-pane
 * "Validation Pyramid" placeholder in `WorkbenchShell` (wired in by the
 * integration task).
 *
 * Task C.7.
 */
export function ValidationPanel() {
  const report = useWorkbenchStore(
    (s) => s.validationReport
  ) as ValidationReport | null;

  return (
    <section className="flex min-h-0 flex-col rounded-lg border border-zinc-800 bg-zinc-900/60">
      <header className="flex shrink-0 items-center justify-between gap-2 border-b border-zinc-800 px-3 py-2">
        <div className="flex items-center gap-2">
          <ShieldCheck className="h-3.5 w-3.5 text-emerald-400" />
          <h3 className="text-xs font-semibold text-zinc-200">
            Validation Pyramid
          </h3>
        </div>
        {report?.agent_version && (
          <span className="font-mono text-[10px] text-zinc-500">
            {report.agent_version}
          </span>
        )}
      </header>

      <div className="flex min-h-0 flex-1 flex-col p-2.5">
        {report ? (
          <ValidationPyramid report={report} />
        ) : (
          <EmptyState />
        )}
      </div>
    </section>
  );
}

function EmptyState() {
  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-2 py-10 text-center">
      <ShieldCheck className="h-7 w-7 text-zinc-700" />
      <p className="text-xs text-zinc-400">No validation report yet</p>
      <p className="text-[10px] text-zinc-600">
        Run a simulation to validate
      </p>
    </div>
  );
}
