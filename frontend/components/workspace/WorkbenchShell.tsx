"use client";

import React, { useEffect } from "react";
import { AlertTriangle } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useWorkbenchStore } from "@/lib/store";
import { useTranslation } from "@/lib/i18n";
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
 *   │ (250px)     │  Sim Tabs + Params)  │  Hypothesis) │  320px → 0)  │
 *   └─────────────┴──────────────────────┴──────────────┴──────────────┘
 *
 * The AI Assistant pane is collapsible (默认展开，开源用户进入即可见 AI 输入框)
 * and is NOT the primary UI — the center Scientific Workspace is.
 *
 * Scroll / responsive notes:
 *   • Center pane scrolls vertically as a whole so users on 100% zoom / small
 *     screens can always reach the parameter controls and the AI input.
 *   • ParameterExplorer is capped at 35vh min+max to avoid eating the graph.
 *   • Right panes use ScrollArea for independent vertical scrolling.
 */
export function WorkbenchShell() {
  const { t } = useTranslation();
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
    ? "minmax(200px, 250px) minmax(0, 1fr) 300px minmax(280px, 320px)"
    : "minmax(200px, 250px) minmax(0, 1fr) 300px";

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
              {t("pane.project")}
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
        <div className="flex h-full min-w-0 flex-col overflow-hidden bg-zinc-950">
          <div className="shrink-0 border-b border-zinc-800 px-3 py-2">
            <span className="text-[11px] font-medium uppercase tracking-wide text-zinc-500">
              {t("pane.workspace")}
            </span>
          </div>
          {/* 使用浏览器原生滚动条（overflow-y-scroll），确保在 Edge / Chrome 中滚动条始终可见 */}
          <div
            data-testid="center-scroll"
            className="min-h-0 flex-1 overflow-y-scroll overscroll-contain pr-1"
            style={{ scrollbarGutter: "stable" }}
          >
            <div className="flex flex-col gap-2 p-2.5">
              <PathwayGraph
                graph={pathwayGraph as PathwayGraphProps["graph"]}
                className="shrink-0"
                style={{ height: "clamp(140px, 22vh, 260px)", minHeight: 140 }}
              />
              <div
                className="min-h-[120px] shrink-0"
                style={{ height: "clamp(140px, 20vh, 220px)" }}
              >
                <SimulationPanel />
              </div>
              <div
                className="min-h-[180px] shrink-0"
                style={{ height: "clamp(200px, 36vh, 420px)" }}
              >
                <ParameterExplorer />
              </div>
            </div>
          </div>
        </div>

        {/* ── Right pane: Validation ── */}
        <div className="flex h-full min-w-0 flex-col border-l border-zinc-800 bg-zinc-950">
          <div className="shrink-0 border-b border-zinc-800 px-3 py-2">
            <span className="text-[11px] font-medium uppercase tracking-wide text-zinc-500">
              {t("pane.validation")}
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
