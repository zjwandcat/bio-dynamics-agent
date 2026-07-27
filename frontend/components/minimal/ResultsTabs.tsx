"use client";

import React, { useState } from "react";
import { Network, Activity, ShieldCheck, FileText, ImageIcon } from "lucide-react";
import { cn } from "@/lib/utils";
import { useWorkbenchStore } from "@/lib/store";
import {
  PathwayGraph,
  type PathwayGraphProps,
} from "@/components/pathway/PathwayGraph";
import { TimeSeriesChart } from "@/components/simulation/TimeSeriesChart";
import {
  ValidationPyramid,
  type ValidationReport,
} from "@/components/validation/ValidationPyramid";

/**
 * ResultsTabs —— 极简 Results 面板的 4 个 Tab。
 *
 * 直取 store 顶层字段（不再反查 messages），与极简 Auto-Chat 闭环一致：
 *   Graph       → pathwayGraph + knowledgeGraph 摘要
 *   Curves      → simulationResult.time_points/species + simulationImage（PNG）
 *   Validation  → validationReport（5 级校验金字塔）
 *   Report      → reportMarkdown（科学报告）
 *
 * 仅在任意结果就绪时渲染；空闲态由 AIWorkflowSteps 的 IdleHint 处理。
 */

type TabKey = "graph" | "curves" | "validation" | "report";

const TABS: { key: TabKey; label: string; icon: typeof Network }[] = [
  { key: "graph", label: "Graph", icon: Network },
  { key: "curves", label: "Curves", icon: Activity },
  { key: "validation", label: "Validation", icon: ShieldCheck },
  { key: "report", label: "Report", icon: FileText },
];

export function ResultsTabs({ className }: { className?: string }) {
  const [active, setActive] = useState<TabKey>("graph");

  const pathwayGraph = useWorkbenchStore((s) => s.pathwayGraph);
  const knowledgeGraph = useWorkbenchStore((s) => s.knowledgeGraph);
  const simulationResult = useWorkbenchStore((s) => s.simulationResult);
  const simulationImage = useWorkbenchStore((s) => s.simulationImage);
  const validationReport = useWorkbenchStore((s) => s.validationReport);
  const reportMarkdown = useWorkbenchStore((s) => s.reportMarkdown);
  const isStreaming = useWorkbenchStore((s) => s.isStreaming);

  // 仅当存在任一结果时才渲染整个 Results 区块
  const hasAnyResult =
    pathwayGraph ||
    simulationResult ||
    simulationImage ||
    validationReport ||
    reportMarkdown;
  if (!hasAnyResult && !isStreaming) return null;

  return (
    <section
      className={cn("rounded-xl border border-zinc-800 bg-zinc-950", className)}
      aria-label="simulation results"
    >
      {/* Tab 头 */}
      <div className="flex items-center gap-1 border-b border-zinc-800 px-2">
        {TABS.map((tab) => {
          const Icon = tab.icon;
          const available =
            tab.key === "graph"
              ? !!pathwayGraph
              : tab.key === "curves"
                ? !!simulationResult || !!simulationImage
                : tab.key === "validation"
                  ? !!validationReport
                  : !!reportMarkdown;
          return (
            <button
              key={tab.key}
              type="button"
              onClick={() => setActive(tab.key)}
              className={cn(
                "relative inline-flex items-center gap-1.5 px-3 py-2.5 text-[13px] font-medium transition-colors",
                active === tab.key
                  ? "text-blue-300"
                  : "text-zinc-500 hover:text-zinc-300",
                available && "text-zinc-300"
              )}
            >
              <Icon className="h-3.5 w-3.5" />
              {tab.label}
              {available && active !== tab.key && (
                <span className="absolute bottom-0 left-2 right-2 h-px bg-emerald-500/40" />
              )}
              {active === tab.key && (
                <span className="absolute bottom-0 left-2 right-2 h-0.5 rounded-full bg-blue-500" />
              )}
            </button>
          );
        })}
      </div>

      {/* Tab 内容 */}
      <div className="p-4">
        {active === "graph" && (
          <GraphTab
            graph={pathwayGraph as PathwayGraphProps["graph"]}
            kgSummary={knowledgeGraph ?? null}
          />
        )}
        {active === "curves" && (
          <CurvesTab
            simulationResult={simulationResult}
            simulationImage={simulationImage}
            streaming={isStreaming}
          />
        )}
        {active === "validation" && (
          <ValidationTab report={validationReport as ValidationReport} />
        )}
        {active === "report" && <ReportTab markdown={reportMarkdown} />}
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// Graph tab
// ---------------------------------------------------------------------------

function GraphTab({
  graph,
  kgSummary,
}: {
  graph: PathwayGraphProps["graph"];
  kgSummary:
    | {
        node_count: number;
        edge_count: number;
        is_acyclic: boolean;
        topology_signature: string;
      }
    | null;
}) {
  if (!graph) {
    return <EmptyTab hint="通路图将在「Pathway Recognition」步骤完成后显示。" />;
  }
  return (
    <div className="space-y-3">
      <PathwayGraph graph={graph} className="h-[360px]" />
      {kgSummary && (
        <div className="flex flex-wrap gap-x-5 gap-y-1 border-t border-zinc-800 pt-3 font-mono text-[11px] text-zinc-500">
          <span>
            nodes: <span className="text-zinc-300">{kgSummary.node_count}</span>
          </span>
          <span>
            edges: <span className="text-zinc-300">{kgSummary.edge_count}</span>
          </span>
          <span>
            acyclic:{" "}
            <span
              className={cn(
                kgSummary.is_acyclic ? "text-emerald-400" : "text-amber-400"
              )}
            >
              {kgSummary.is_acyclic ? "yes" : "no"}
            </span>
          </span>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Curves tab
// ---------------------------------------------------------------------------

function CurvesTab({
  simulationResult,
  simulationImage,
  streaming,
}: {
  simulationResult:
    | {
        time_points: number[];
        species: Record<string, number[]>;
      }
    | null;
  simulationImage: string | null;
  streaming: boolean;
}) {
  // 优先交互式曲线（可读数据），其次后端渲染的 PNG
  if (simulationResult && simulationResult.time_points?.length > 0) {
    return (
      <div className="space-y-3">
        <TimeSeriesChart
          timePoints={simulationResult.time_points}
          species={simulationResult.species}
          className="h-[360px]"
        />
        {simulationImage && (
          <details className="border-t border-zinc-800 pt-3 text-xs text-zinc-500">
            <summary className="cursor-pointer select-none">
              后端渲染图（PNG）
            </summary>
            <img
              src={`data:image/png;base64,${simulationImage}`}
              alt="backend simulation plot"
              className="mt-2 w-full rounded-lg border border-zinc-800"
            />
          </details>
        )}
      </div>
    );
  }

  if (simulationImage) {
    return (
      <div className="space-y-2">
        <img
          src={`data:image/png;base64,${simulationImage}`}
          alt="simulation result"
          className="w-full rounded-lg border border-zinc-800"
        />
        {streaming && (
          <p className="text-center text-xs text-zinc-500">仿真进行中…</p>
        )}
      </div>
    );
  }

  return (
    <EmptyTab hint={streaming ? "正在求解 ODE…" : "仿真曲线将在「Simulation」步骤完成后显示。"} />
  );
}

// ---------------------------------------------------------------------------
// Validation tab
// ---------------------------------------------------------------------------

function ValidationTab({ report }: { report: ValidationReport }) {
  if (!report) {
    return (
      <EmptyTab hint="校验报告将在「Validation」步骤完成后显示。" />
    );
  }
  return <ValidationPyramid report={report} />;
}

// ---------------------------------------------------------------------------
// Report tab
// ---------------------------------------------------------------------------

function ReportTab({ markdown }: { markdown: string | null }) {
  if (!markdown) {
    return (
      <EmptyTab hint="科学报告将在「Scientific Report」步骤完成后显示。" />
    );
  }
  return (
    <article className="prose prose-invert prose-sm max-w-none whitespace-pre-wrap break-words rounded-lg bg-zinc-900/40 p-4 font-mono text-[13px] leading-relaxed text-zinc-200">
      {markdown}
    </article>
  );
}

// ---------------------------------------------------------------------------
// 空态占位
// ---------------------------------------------------------------------------

function EmptyTab({ hint }: { hint: string }) {
  return (
    <div className="flex h-[320px] flex-col items-center justify-center gap-2 text-center">
      <ImageIcon className="h-6 w-6 text-zinc-700" />
      <p className="text-sm text-zinc-500">{hint}</p>
    </div>
  );
}
