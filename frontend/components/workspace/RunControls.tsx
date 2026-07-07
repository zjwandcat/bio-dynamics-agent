"use client";

import React, { useState } from "react";
import {
  Zap,
  Settings2,
  Hand,
  Activity,
  Cpu,
  Search,
  FlaskConical,
  Beaker,
  Library,
  FileText,
  AlertCircle,
  CheckCircle2,
  type LucideIcon,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { useWorkbenchStore, type RunMode } from "@/lib/store";

/**
 * Run controls — absorbs the mode selector + manual module checkbox logic from
 * the legacy `ControlBar.tsx`. Rendered in the left pane of the WorkbenchShell.
 */
const MODES: { value: RunMode; label: string; icon: React.ReactNode; desc: string }[] = [
  { value: "auto_fast", label: "Auto Fast", icon: <Zap className="h-3.5 w-3.5" />, desc: "极简流程，单智能体直跑" },
  { value: "auto_standard", label: "Auto Standard", icon: <Settings2 className="h-3.5 w-3.5" />, desc: "默认模式，LLM 动态裁剪流程" },
  { value: "manual", label: "Manual", icon: <Hand className="h-3.5 w-3.5" />, desc: "用户勾选所需模块" },
];

interface ModuleDef {
  key: string;
  label: string;
  icon: React.ReactNode;
  required?: boolean;
}

const MODULES: ModuleDef[] = [
  { key: "terminology_mcp", label: "术语标准化 (MCP)", icon: <Activity className="h-3.5 w-3.5" /> },
  { key: "mechanism_graph", label: "机制解析与图谱", icon: <Cpu className="h-3.5 w-3.5" /> },
  { key: "mechanism_parameter_rag", label: "知识检索 (RAG)", icon: <Search className="h-3.5 w-3.5" /> },
  { key: "pkpd_inference", label: "PK/PD 推断", icon: <FlaskConical className="h-3.5 w-3.5" /> },
  { key: "sandbox_execute", label: "沙箱仿真执行", icon: <Beaker className="h-3.5 w-3.5" />, required: true },
  { key: "dose_analysis", label: "剂量递增分析", icon: <Activity className="h-3.5 w-3.5" /> },
  { key: "experiment_evidence_rag", label: "实验与文献检索", icon: <Library className="h-3.5 w-3.5" /> },
  { key: "report_generation", label: "预测报告生成", icon: <FileText className="h-3.5 w-3.5" /> },
];

const MODULE_ORDER = MODULES.map((m) => m.key);

export function RunControls() {
  const controlBarState = useWorkbenchStore((s) => s.controlBarState);
  const setControlBarState = useWorkbenchStore((s) => s.setControlBarState);
  const isStreaming = useWorkbenchStore((s) => s.isStreaming);
  const [tooltip, setTooltip] = useState<string | null>(null);

  const { mode, manualModules } = controlBarState;

  // Auto-add dependency modules when report_generation is selected (legacy behavior).
  React.useEffect(() => {
    if (mode !== "manual") return;
    const hasReport = manualModules.includes("report_generation");
    const hasSandbox = manualModules.includes("sandbox_execute");
    const hasMechanism = manualModules.includes("mechanism_graph");
    if (hasReport && (!hasSandbox || !hasMechanism)) {
      const added = [
        hasSandbox ? null : "sandbox_execute",
        hasMechanism ? null : "mechanism_graph",
      ].filter(Boolean) as string[];
      const next = [...manualModules, ...added].sort(
        (a, b) => MODULE_ORDER.indexOf(a) - MODULE_ORDER.indexOf(b)
      );
      setControlBarState({ ...controlBarState, manualModules: next });
      setTooltip("已自动补充依赖项：沙箱仿真执行、机制解析与图谱");
      const t = setTimeout(() => setTooltip(null), 3000);
      return () => clearTimeout(t);
    }
  }, [manualModules, mode, setControlBarState, controlBarState]);

  const setMode = (m: RunMode) =>
    setControlBarState({ ...controlBarState, mode: m });

  const toggleModule = (key: string) => {
    if (mode !== "manual") return;
    if (MODULES.find((m) => m.key === key)?.required) return;
    const next = manualModules.includes(key)
      ? manualModules.filter((k) => k !== key)
      : [...manualModules, key];
    setControlBarState({ ...controlBarState, manualModules: next });
  };

  return (
    <div className="space-y-2">
      <div className="text-[11px] font-medium uppercase tracking-wide text-zinc-500">
        运行模式
      </div>
      <div className="space-y-1.5">
        {MODES.map((opt) => (
          <button
            key={opt.value}
            type="button"
            disabled={isStreaming}
            onClick={() => setMode(opt.value)}
            className={cn(
              "flex w-full items-center gap-2 rounded-md border px-2.5 py-2 text-left transition-colors",
              mode === opt.value
                ? "border-blue-500/50 bg-blue-500/10 text-blue-100"
                : "border-zinc-800 bg-zinc-950/50 text-zinc-300 hover:bg-zinc-800",
              isStreaming && "cursor-not-allowed opacity-60"
            )}
          >
            <span className={mode === opt.value ? "text-blue-400" : "text-zinc-500"}>
              {opt.icon}
            </span>
            <div className="min-w-0 flex-1">
              <div className="text-xs font-medium">{opt.label}</div>
              <div className="truncate text-[10px] text-zinc-500">{opt.desc}</div>
            </div>
            {mode === opt.value && <CheckCircle2 className="h-3.5 w-3.5 text-blue-400" />}
          </button>
        ))}
      </div>

      {mode === "manual" && (
        <div className="space-y-1.5">
          <div className="text-[11px] font-medium uppercase tracking-wide text-zinc-500">
            模块勾选
          </div>
          {tooltip && (
            <div className="flex items-center gap-1.5 rounded-md bg-amber-500/10 px-2 py-1.5 text-[11px] text-amber-300">
              <AlertCircle className="h-3 w-3 shrink-0" />
              {tooltip}
            </div>
          )}
          {MODULES.map((mod) => {
            const checked = manualModules.includes(mod.key);
            return (
              <label
                key={mod.key}
                className={cn(
                  "flex cursor-pointer items-center gap-2 rounded-md border px-2.5 py-1.5 text-xs transition-colors",
                  checked
                    ? "border-blue-500/40 bg-blue-500/10 text-zinc-100"
                    : "border-zinc-800 bg-zinc-950/50 text-zinc-400 hover:bg-zinc-800",
                  mod.required && "opacity-90"
                )}
              >
                <input
                  type="checkbox"
                  className="h-3.5 w-3.5 accent-blue-500"
                  checked={checked}
                  disabled={mod.required || isStreaming}
                  onChange={() => toggleModule(mod.key)}
                />
                <span className="text-zinc-500">{mod.icon}</span>
                <span className="flex-1">{mod.label}</span>
                {mod.required && (
                  <Badge variant="outline" className="h-4 text-[9px] border-zinc-700 text-zinc-500">
                    必须
                  </Badge>
                )}
              </label>
            );
          })}
        </div>
      )}
    </div>
  );
}

// Re-export for type consumers (keeps the icon re-export path stable).
export type { LucideIcon };
