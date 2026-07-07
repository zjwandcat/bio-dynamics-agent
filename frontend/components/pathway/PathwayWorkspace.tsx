"use client";

import React, { useCallback, useEffect, useMemo, useState } from "react";
import { Loader2, AlertTriangle, GitBranch } from "lucide-react";
import { useWorkbenchStore } from "@/lib/store";
import {
  fetchPathwayGraph,
  type PathwayClass,
  type PathwayGraphData,
} from "@/lib/api";
import {
  PathwayGraph,
  getFallbackEGFRGraph,
  type RichPathwayGraphData,
  type RichPathwayNode,
} from "./PathwayGraph";
import { NodeDetailPanel } from "./NodeDetailPanel";
import { cn } from "@/lib/utils";

/**
 * Center-pane wrapper that ties together the interactive PathwayGraph and the
 * NodeDetailPanel overlay.
 *
 * Responsibilities:
 *  - Watch `currentPathway` from the global store and fetch the graph payload
 *    from `/api/v4/pathways/<class>/graph` (via `fetchPathwayGraph`).
 *  - On API failure, fall back to a hardcoded EGFR pathway graph so the
 *    workbench always shows something useful.
 *  - Keep the store's `pathwayGraph` mirror in sync (used later by the
 *    SSE hydration in C.7/C.8).
 *  - Own the "selected node" state and render the NodeDetailPanel as an
 *    overlay on the right edge of the graph.
 *
 * This component replaces the center-pane "Pathway Graph" placeholder from
 * WorkbenchShell (integration is wired in the C.13 integration task).
 */
export function PathwayWorkspace({
  className,
}: {
  className?: string;
}) {
  const currentPathway = useWorkbenchStore((s) => s.currentPathway);
  const setPathwayGraph = useWorkbenchStore((s) => s.setPathwayGraph);

  const [graph, setGraph] = useState<RichPathwayGraphData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [usingFallback, setUsingFallback] = useState(false);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);

  // ---- Fetch graph whenever the selected pathway class changes ----
  useEffect(() => {
    if (!currentPathway) {
      setGraph(null);
      setPathwayGraph(null);
      setError(null);
      setUsingFallback(false);
      setSelectedNodeId(null);
      setLoading(false);
      return;
    }

    const controller = new AbortController();
    setLoading(true);
    setError(null);
    setUsingFallback(false);
    setSelectedNodeId(null);

    fetchPathwayGraph(currentPathway as PathwayClass, controller.signal)
      .then((data: PathwayGraphData) => {
        // The placeholder contract is a subset of the rich backend payload;
        // cast through unknown so optional rich fields are tolerated.
        const rich = data as unknown as RichPathwayGraphData;
        setGraph(rich);
        setPathwayGraph(data);
        setLoading(false);
      })
      .catch((err: unknown) => {
        if (controller.signal.aborted) return;
        // Hardcoded fallback: EGFR pathway graph (API unavailable).
        const fallback = getFallbackEGFRGraph();
        setGraph(fallback);
        setPathwayGraph(fallback as unknown as PathwayGraphData);
        setUsingFallback(true);
        setError(err instanceof Error ? err.message : "API unavailable");
        setLoading(false);
      });

    return () => controller.abort();
  }, [currentPathway, setPathwayGraph]);

  // ---- Selection handling ----
  const handleNodeClick = useCallback(
    (nodeId: string) => {
      setSelectedNodeId((prev) => (prev === nodeId ? null : nodeId));
    },
    []
  );

  const handleClosePanel = useCallback(() => {
    setSelectedNodeId(null);
  }, []);

  const selectedNode = useMemo<RichPathwayNode | null>(() => {
    if (!graph || !selectedNodeId) return null;
    return graph.nodes.find((n) => n.id === selectedNodeId) ?? null;
  }, [graph, selectedNodeId]);

  return (
    <section
      className={cn(
        "relative flex min-h-0 min-w-0 flex-col overflow-hidden rounded-lg border border-zinc-800 bg-zinc-900/60",
        className
      )}
    >
      {/* Header */}
      <header className="flex shrink-0 items-center justify-between gap-2 border-b border-zinc-800 px-3 py-2">
        <div className="flex items-center gap-2">
          <GitBranch className="h-3.5 w-3.5 text-zinc-400" />
          <h3 className="text-xs font-semibold text-zinc-200">
            Pathway Graph
          </h3>
          {currentPathway && (
            <span className="rounded bg-zinc-800 px-1.5 py-0.5 text-[10px] font-medium text-zinc-400">
              {currentPathway}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {loading && (
            <span className="inline-flex items-center gap-1 text-[10px] text-zinc-500">
              <Loader2 className="h-3 w-3 animate-spin" />
              Loading…
            </span>
          )}
          {usingFallback && !loading && (
            <span
              className="inline-flex items-center gap-1 rounded border border-amber-800/50 bg-amber-900/30 px-1.5 py-0.5 text-[10px] text-amber-300"
              title={error ? `API unavailable: ${error}` : "API unavailable"}
            >
              <AlertTriangle className="h-3 w-3" />
              Fallback: EGFR
            </span>
          )}
          {!currentPathway && !loading && (
            <span className="text-[10px] text-zinc-600">no pathway selected</span>
          )}
        </div>
      </header>

      {/* Graph + detail panel overlay */}
      <div className="relative flex min-h-0 flex-1 overflow-hidden">
        <PathwayGraph
          graph={graph}
          onNodeClick={handleNodeClick}
          className="min-h-0 flex-1"
        />

        {selectedNode && (
          <NodeDetailPanel
            node={selectedNode}
            graph={graph}
            onClose={handleClosePanel}
            className="absolute top-2 right-2 bottom-2 z-20 w-[300px]"
          />
        )}
      </div>
    </section>
  );
}
