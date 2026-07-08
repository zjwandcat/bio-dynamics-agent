"use client";

import React, { useEffect } from "react";
import { AlertTriangle } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useWorkbenchStore } from "@/lib/store";
import { WorkbenchHeader } from "@/components/workspace/WorkbenchHeader";
import { RunControls } from "@/components/workspace/RunControls";
import { PlaceholderPanel } from "@/components/workspace/PlaceholderPanel";
import { AIAssistantPanel } from "@/components/ai_assistant/AIAssistantPanel";
import { PathwayTree } from "@/components/pathway/PathwayTree";
import { BenchmarkList } from "@/components/pathway/BenchmarkList";
import { SimulationHistory } from "@/components/pathway/SimulationHistory";
import { PathwayGraph, type PathwayGraphProps } from "@/components/pathway/PathwayGraph";
import { SimulationPanel } from "@/components/simulation/SimulationPanel";
import { ParameterExplorer } from "@/components/parameter/ParameterExplorer";
import {
  ValidationPyramid,
  type ValidationReport,
} from "@/components/validation/ValidationPyramid";
import { HypothesisPanel } from "@/components/hypothesis/HypothesisPanel";

/**
 * Scientific Modeling IDE four-pane workbench shell.
 *
 * Layout (CSS Grid):
 *   ┌─────────────┬──────────────────────┬──────────────┬──────────────┐
 *   │ Project /   │ Scientific Workspace │ Validation   │ AI Assistant │
 *   │ Pathway     │ (Pathway Graph +     │ (Pyramid +   │ (collapsible │
 *   │ (250px)     │  Sim Tabs + Params)  │  Hypothesis) │  360px → 0)  │
 *   └─────────────┴──────────────────────┴──────────────┴──────────────┘
 *
 * The AI Assistant pane is collapsible (collapsed by default) and is NOT the
 * primary UI — the center Scientific Workspace is. Placeholder slots are
 * filled in C.2–C.8.
 */
export function WorkbenchShell() {
  const aiOpen = useWorkbenchStore((s) => s.uiState.aiAssistantOpen);
  const updateDbStatus = useWorkbenchStore((s) => s.updateDbStatus);
  const refreshRagStatus = useWorkbenchStore((s) => s.refreshRagStatus);
  const refreshModelStatus = useWorkbenchStore((s) => s.refreshModelStatus);
  const pathwayGraph = useWorkbenchStore((s) => s.pathwayGraph);
  const validationReport = useWorkbenchStore((s) => s.validationReport);

  // Hydrate admin status once on mount (RAG + model providers).
  useEffect(() => {
    refreshRagStatus();
    refreshModelStatus();
  }, [refreshRagStatus, refreshModelStatus]);

  const gridTemplateColumns = aiOpen
    ? "250px minmax(0, 1fr) 320px 360px"
    : "250px minmax(0, 1fr) 320px";

  return (
    <div className="flex h-screen flex-col bg-zinc-950 text-zinc-100">
      <WorkbenchHeader />

      {updateDbStatus && (
        <div className="border-b border-zinc-800 bg-zinc-900/80 px-4 py-1.5 text-center text-xs text-zinc-400">
          {updateDbStatus}
        </div>
      )}

      <div
        className="grid min-h-0 flex-1 overflow-hidden"
        style={{ gridTemplateColumns, transition: "grid-template-columns 200ms ease" }}
      >
        {/* ── Left pane: Project / Pathway ── */}
        <div className="flex h-full min-w-0 flex-col border-r border-zinc-800 bg-zinc-950">
          <div className="shrink-0 border-b border-zinc-800 px-3 py-2">
            <span className="text-[11px] font-medium uppercase tracking-wide text-zinc-500">
              Project / Pathway
            </span>
          </div>
          <ScrollArea className="min-h-0 flex-1">
            <div className="space-y-3 p-2.5">
              <RunControls />
              <PathwayTree />
              <BenchmarkList />
              <SimulationHistory />
            </div>
          </ScrollArea>
        </div>

        {/* ── Center pane: Scientific Workspace ── */}
        <div className="flex h-full min-w-0 flex-col bg-zinc-950">
          <div className="shrink-0 border-b border-zinc-800 px-3 py-2">
            <span className="text-[11px] font-medium uppercase tracking-wide text-zinc-500">
              Scientific Workspace
            </span>
          </div>
          <div className="flex min-h-0 flex-1 flex-col gap-2 overflow-hidden p-2.5">
            <PathwayGraph
              graph={pathwayGraph as PathwayGraphProps["graph"]}
              className="min-h-0 flex-1"
            />
            <SimulationPanel />
            <div className="h-80 shrink-0 overflow-hidden">
              <ParameterExplorer />
            </div>
          </div>
        </div>

        {/* ── Right pane: Validation ── */}
        <div className="flex h-full min-w-0 flex-col border-l border-zinc-800 bg-zinc-950">
          <div className="shrink-0 border-b border-zinc-800 px-3 py-2">
            <span className="text-[11px] font-medium uppercase tracking-wide text-zinc-500">
              Validation
            </span>
          </div>
          <ScrollArea className="min-h-0 flex-1">
            <div className="space-y-3 p-2.5">
              <ValidationPyramid
                report={(validationReport as ValidationReport) ?? {}}
              />
              <HypothesisPanel />
              <PlaceholderPanel
                title="Evidence & Warnings"
                description="文献证据与一致性告警"
                taskRef="C.7"
                icon={<AlertTriangle className="h-3.5 w-3.5" />}
              />
            </div>
          </ScrollArea>
        </div>

        {/* ── Far-right pane: AI Assistant (collapsible) ── */}
        {aiOpen && (
          <div className="flex h-full min-w-0 flex-col bg-zinc-950">
            <AIAssistantPanel />
          </div>
        )}
      </div>
    </div>
  );
}
