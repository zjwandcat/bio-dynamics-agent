"use client";

import React from "react";
import { Lightbulb, Loader2 } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useWorkbenchStore } from "@/lib/store";
import type { Hypothesis, HypothesisStatus } from "./types";
import { HypothesisCard } from "./HypothesisCard";

/**
 * Hypothesis Panel — right pane, below validation (Task C.8).
 *
 * Subscribes to the workbench store's `hypothesisList` (hydrated from the
 * `v4_hypothesis_list` / `v4_hypothesis_generated` SSE events by the store's
 * `ingestSSEEvent`) and renders one `HypothesisCard` per hypothesis. Each card
 * expands into 5 accordion sections (Hypothesis / Evidence / Predictions /
 * Suggested Experiments / Falsifiability).
 *
 * Overall status: Generated (list non-empty) / Pending (streaming) / Failed.
 */
export function HypothesisPanel() {
  const hypothesisList = useWorkbenchStore((s) => s.hypothesisList);
  const isStreaming = useWorkbenchStore((s) => s.isStreaming);

  const hypotheses = hypothesisList as Hypothesis[];
  const status: HypothesisStatus =
    hypotheses.length > 0 ? "generated" : "pending";

  return (
    <section className="flex h-full min-h-0 flex-col rounded-lg border border-zinc-800 bg-zinc-900/60">
      {/* Header */}
      <header className="flex shrink-0 items-center justify-between gap-2 border-b border-zinc-800 px-3 py-2">
        <div className="flex items-center gap-2">
          <Lightbulb className="h-3.5 w-3.5 text-amber-400" />
          <h3 className="text-xs font-semibold text-zinc-200">Hypothesis</h3>
          <StatusBadge status={status} streaming={isStreaming} />
        </div>
        {hypotheses.length > 0 && (
          <span className="text-[10px] text-zinc-500">
            {hypotheses.length} {hypotheses.length === 1 ? "item" : "items"}
          </span>
        )}
      </header>

      {/* Body */}
      {hypotheses.length === 0 ? (
        <EmptyState streaming={isStreaming} />
      ) : (
        <ScrollArea className="min-h-0 flex-1">
          <div className="space-y-2 p-2.5">
            {hypotheses.map((h, idx) => (
              <HypothesisCard
                key={h.id ?? h.hypothesis_id ?? idx}
                hypothesis={h}
                defaultExpanded={idx === 0}
              />
            ))}
          </div>
        </ScrollArea>
      )}
    </section>
  );
}

function StatusBadge({
  status,
  streaming,
}: {
  status: HypothesisStatus;
  streaming: boolean;
}) {
  if (status === "generated") {
    return (
      <Badge
        variant="outline"
        className="h-4 border-emerald-700/50 bg-emerald-500/10 text-[9px] text-emerald-300"
      >
        generated
      </Badge>
    );
  }
  // pending — pulse a dot while the stream is running
  return (
    <Badge
      variant="outline"
      className="h-4 border-blue-700/50 bg-blue-500/10 text-[9px] text-blue-300"
    >
      {streaming && (
        <span className="mr-1 h-1.5 w-1.5 animate-pulse rounded-full bg-blue-400" />
      )}
      pending
    </Badge>
  );
}

function EmptyState({ streaming }: { streaming: boolean }) {
  return (
    <div className="flex min-h-0 flex-1 flex-col items-center justify-center gap-2 p-6 text-center">
      {streaming ? (
        <>
          <Loader2 className="h-6 w-6 animate-spin text-amber-400" />
          <p className="text-xs text-zinc-400">Generating hypotheses…</p>
        </>
      ) : (
        <>
          <Lightbulb className="h-6 w-6 text-zinc-700" />
          <p className="text-xs text-zinc-400">
            Run a simulation to generate hypotheses
          </p>
        </>
      )}
    </div>
  );
}
