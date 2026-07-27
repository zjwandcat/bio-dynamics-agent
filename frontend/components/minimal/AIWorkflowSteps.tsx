"use client";

import React from "react";
import { Check, Loader2, Circle } from "lucide-react";
import { cn } from "@/lib/utils";
import { useWorkbenchStore } from "@/lib/store";

/**
 * AIWorkflowSteps —— 极简 Auto-Chat 流程的 7 步进度跟踪器。
 *
 * 设计原则：不引入新的 store 字段，而是从既有 SSE 注入的顶层状态直接
 * 派生 7 个阶段的状态（pending / running / done）：
 *
 *   1. Ontology              ← knowledge_graph SSE（knowledgeGraph 非空）
 *   2. Pathway Recognition   ← v4_pathway_graph SSE（pathwayGraph 非空）
 *   3. Reaction Graph         ← knowledgeGraph.edge_count > 0
 *   4. ODE Construction       ← messages 中出现 code 类型（node2 产物）
 *   5. Simulation            ← v4_simulation_result / image_ready（仿真图）
 *   6. Validation            ← v4_validation_report（validationReport）
 *   7. Scientific Report     ← report / report_ready（reportMarkdown）
 *
 * "running" = 第一个尚未完成的阶段（仅在 isStreaming 时高亮）。
 */
type StageStatus = "pending" | "running" | "done";

interface Stage {
  key: string;
  label: string;
  hint: string;
}

const STAGES: Stage[] = [
  { key: "ontology", label: "Ontology", hint: "术语标准化" },
  { key: "pathway", label: "Pathway Recognition", hint: "通路识别" },
  { key: "reaction", label: "Reaction Graph", hint: "反应图谱" },
  { key: "ode", label: "ODE Construction", hint: "方程构建" },
  { key: "simulation", label: "Simulation", hint: "动力学仿真" },
  { key: "validation", label: "Validation", hint: "科学校验" },
  { key: "report", label: "Scientific Report", hint: "证据报告" },
];

/** 从 store 派生每个阶段的完成态。 */
function useStageStatuses(): StageStatus[] {
  const knowledgeGraph = useWorkbenchStore((s) => s.knowledgeGraph);
  const pathwayGraph = useWorkbenchStore((s) => s.pathwayGraph);
  const simulationResult = useWorkbenchStore((s) => s.simulationResult);
  const simulationImage = useWorkbenchStore((s) => s.simulationImage);
  const validationReport = useWorkbenchStore((s) => s.validationReport);
  const reportMarkdown = useWorkbenchStore((s) => s.reportMarkdown);
  const hasCode = useWorkbenchStore((s) =>
    s.messages.some((m) => m.role === "agent" && m.type === "code")
  );
  const isStreaming = useWorkbenchStore((s) => s.isStreaming);

  const done = [
    knowledgeGraph !== null,
    pathwayGraph !== null,
    !!knowledgeGraph && (knowledgeGraph.edge_count ?? 0) > 0,
    hasCode,
    simulationResult !== null || simulationImage !== null,
    validationReport !== null,
    reportMarkdown !== null,
  ];

  // 第一个未完成阶段在 streaming 时为 running
  const firstUndoneIdx = done.findIndex((d) => !d);
  return STAGES.map((_, i) => {
    if (done[i]) return "done";
    if (isStreaming && i === firstUndoneIdx) return "running";
    return "pending";
  });
}

export function AIWorkflowSteps({ className }: { className?: string }) {
  const statuses = useStageStatuses();
  const isStreaming = useWorkbenchStore((s) => s.isStreaming);

  return (
    <ol
      className={cn(
        "grid grid-cols-2 gap-2 sm:grid-cols-4 lg:grid-cols-7",
        className
      )}
      aria-label="AI workflow progress"
    >
      {STAGES.map((stage, i) => {
        const status = statuses[i];
        return (
          <li
            key={stage.key}
            className={cn(
              "flex flex-col gap-1.5 rounded-lg border px-3 py-2.5 transition-colors",
              status === "done" &&
                "border-emerald-500/30 bg-emerald-500/5",
              status === "running" &&
                "border-blue-500/40 bg-blue-500/5",
              status === "pending" &&
                "border-zinc-800 bg-zinc-900/40"
            )}
          >
            <div className="flex items-center gap-2">
              {status === "done" && (
                <Check className="h-3.5 w-3.5 shrink-0 text-emerald-400" />
              )}
              {status === "running" && (
                <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin text-blue-400" />
              )}
              {status === "pending" && (
                <Circle className="h-3.5 w-3.5 shrink-0 text-zinc-600" />
              )}
              <span
                className={cn(
                  "text-[12px] font-medium leading-tight",
                  status === "done" && "text-emerald-200",
                  status === "running" && "text-blue-200",
                  status === "pending" && "text-zinc-500"
                )}
              >
                {stage.label}
              </span>
            </div>
            <span className="text-[10px] uppercase tracking-wide text-zinc-600">
              {stage.hint}
            </span>
          </li>
        );
      })}
    </ol>
  );
}

/**
 * 顶层空闲态：未开始任何仿真时显示的占位（避免空 panel）。
 * 仅当既未 streaming 又没有任何结果时渲染。
 */
export function WorkflowIdleHint() {
  const isStreaming = useWorkbenchStore((s) => s.isStreaming);
  const hasAnyResult = useWorkbenchStore((s) =>
    Boolean(
      s.pathwayGraph ||
        s.simulationResult ||
        s.simulationImage ||
        s.validationReport ||
        s.reportMarkdown
    )
  );
  if (isStreaming || hasAnyResult) return null;
  return (
    <div className="rounded-lg border border-dashed border-zinc-800 bg-zinc-900/30 px-4 py-8 text-center">
      <p className="text-sm text-zinc-500">
        输入一句生物学假说，AI 将自动完成{" "}
        <span className="text-zinc-300">通路识别 → 仿真 → 校验 → 报告</span>{" "}
        全流程。
      </p>
    </div>
  );
}
