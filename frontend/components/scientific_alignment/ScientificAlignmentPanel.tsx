"use client";

/**
 * ScientificAlignmentPanel — Task 21 Step 2: 科学对齐闭环前端渲染。
 *
 * 订阅 workbench store 的三个 SA 报告切片：
 *   - saConsistencyReport  （sa_consistency_report SSE 事件）
 *   - saCriticReport       （sa_critic_report SSE 事件）
 *   - saMultiDimConfidence （sa_multi_dim_confidence SSE 事件）
 *
 * 渲染三张可折叠卡片：
 *   1. Consistency Checker —— 机制级因果时序校验（passed/violations）
 *   2. Scientific Critic   —— 6 类独立审稿（overall_status/findings）
 *   3. Multi-dim Confidence —— 6 维置信度（overall_confidence/axes）
 *
 * 后端 Feature Flag OFF 时三个切片均为 null，面板显示空态。
 * 视觉风格对齐 ValidationPyramid（暗色主题 + 状态徽章 + 左边框色编码）。
 */

import React, { useState } from "react";
import {
  ShieldAlert,
  ChevronRight,
  CheckCircle2,
  XCircle,
  Gauge,
  Microscope,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useWorkbenchStore } from "@/lib/store";

// ---------------------------------------------------------------------------
// 类型定义（镜像后端 SSE payload 结构）
// ---------------------------------------------------------------------------

interface ConsistencyViolation {
  rule?: string;
  assertion?: string;
  violation_label?: string;
  observed_values?: Record<string, number | string>;
  message?: string;
}

interface ConsistencyReport {
  passed?: boolean;
  rules_checked?: number;
  rules_evaluated?: number;
  violation_count?: number;
  violations?: ConsistencyViolation[];
  pathway?: string;
}

interface CriticFinding {
  category?: string;
  severity?: string; // "critical" | "major" | "minor" | "info"
  finding?: string;
  evidence?: string;
  suggestion?: string;
}

interface CriticReport {
  overall_status?: string; // "passed" | "failed" | "warning"
  findings_count?: number;
  findings?: CriticFinding[];
  pathway?: string;
}

interface ConfidenceAxis {
  axis_name?: string;
  score?: number;
  status?: string; // "pass" | "warning" | "fail"
  sub_scores?: Record<string, number>;
}

interface MultiDimConfidenceReport {
  overall_confidence?: number;
  axes?: ConfidenceAxis[];
  pathway?: string;
}

// ---------------------------------------------------------------------------
// 状态 → 视觉映射
// ---------------------------------------------------------------------------

type CheckStatus = "pass" | "warning" | "fail" | "skipped";

const STATUS_BORDER: Record<CheckStatus, string> = {
  pass: "border-l-emerald-500",
  warning: "border-l-amber-500",
  fail: "border-l-red-500",
  skipped: "border-l-zinc-600",
};

const STATUS_BADGE: Record<CheckStatus, { label: string; cls: string }> = {
  pass: { label: "PASS", cls: "bg-emerald-500/15 text-emerald-300" },
  warning: { label: "WARN", cls: "bg-amber-500/15 text-amber-300" },
  fail: { label: "FAIL", cls: "bg-red-500/15 text-red-300" },
  skipped: { label: "SKIP", cls: "bg-zinc-500/15 text-zinc-400" },
};

function StatusBadge({ status }: { status: CheckStatus }) {
  const cfg = STATUS_BADGE[status];
  return (
    <span
      className={cn(
        "shrink-0 rounded px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-wide",
        cfg.cls
      )}
    >
      {cfg.label}
    </span>
  );
}

// severity → status 映射（Critic findings 用）
function severityToStatus(severity?: string): CheckStatus {
  if (!severity) return "skipped";
  const s = severity.toLowerCase();
  if (s === "critical" || s === "major" || s === "high") return "fail";
  if (s === "minor" || s === "medium") return "warning";
  if (s === "info" || s === "low") return "pass";
  return "skipped";
}

// overall_status → status 映射
function overallStatusToStatus(overall?: string): CheckStatus {
  if (!overall) return "skipped";
  const s = overall.toLowerCase();
  if (s === "passed" || s === "pass") return "pass";
  if (s === "warning" || s === "warn") return "warning";
  if (s === "failed" || s === "fail") return "fail";
  return "skipped";
}

// score (0~1) → status 映射
function scoreToStatus(score?: number): CheckStatus {
  if (score === undefined || score === null) return "skipped";
  if (score >= 0.7) return "pass";
  if (score >= 0.4) return "warning";
  return "fail";
}

// ---------------------------------------------------------------------------
// 1. Consistency Checker 卡片
// ---------------------------------------------------------------------------

function ConsistencyCard({ report }: { report: ConsistencyReport }) {
  const [expanded, setExpanded] = useState(false);
  const passed = report.passed === true;
  const status: CheckStatus = passed ? "pass" : "fail";
  const violations = report.violations ?? [];
  const rulesChecked = report.rules_checked ?? 0;
  const rulesEvaluated = report.rules_evaluated ?? 0;

  return (
    <div className="overflow-hidden rounded-md border border-zinc-800 bg-zinc-900/70">
      <button
        type="button"
        onClick={() => setExpanded(!expanded)}
        aria-expanded={expanded}
        className={cn(
          "flex w-full items-center gap-2 border-l-2 px-2.5 py-2 text-left transition-colors hover:bg-zinc-800/40",
          STATUS_BORDER[status]
        )}
      >
        <span className="shrink-0 text-zinc-400" aria-hidden>
          <CheckCircle2 className="h-3.5 w-3.5" />
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-1.5">
            <span className="font-mono text-[10px] text-zinc-500">SA-1</span>
            <span className="truncate text-[11px] font-semibold text-zinc-100">
              Consistency Checker
            </span>
          </div>
          <div className="truncate text-[9px] text-zinc-500">
            {rulesEvaluated}/{rulesChecked} rules · {violations.length} violation(s)
          </div>
        </div>
        <StatusBadge status={status} />
        <ChevronRight
          className={cn(
            "h-3.5 w-3.5 shrink-0 text-zinc-500 transition-transform",
            expanded && "rotate-90"
          )}
          aria-hidden
        />
      </button>
      {expanded && (
        <div className="border-t border-zinc-800/80 px-3 py-2">
          {violations.length === 0 ? (
            <p className="text-[10px] text-emerald-400">
              ✓ 所有机制级因果时序规则已通过
            </p>
          ) : (
            <div className="space-y-2">
              {violations.map((v, i) => (
                <div
                  key={i}
                  className="rounded border border-red-900/40 bg-red-950/20 px-2 py-1.5"
                >
                  <div className="flex items-start gap-1.5">
                    <XCircle className="mt-0.5 h-3 w-3 shrink-0 text-red-400" />
                    <div className="min-w-0 flex-1">
                      <div className="text-[10px] font-semibold text-red-200">
                        {v.rule || v.violation_label || "未命名规则"}
                      </div>
                      {v.assertion && (
                        <div className="mt-0.5 font-mono text-[9px] text-zinc-500">
                          {v.assertion}
                        </div>
                      )}
                      {v.message && (
                        <div className="mt-0.5 text-[10px] text-zinc-400">
                          {v.message}
                        </div>
                      )}
                      {v.observed_values &&
                        Object.keys(v.observed_values).length > 0 && (
                          <div className="mt-1 flex flex-wrap gap-1">
                            {Object.entries(v.observed_values).map(
                              ([k, val]) => (
                                <span
                                  key={k}
                                  className="rounded bg-zinc-800/60 px-1 py-0.5 font-mono text-[9px] text-zinc-400"
                                >
                                  {k}={typeof val === "number"
                                    ? val.toFixed(2)
                                    : String(val)}
                                </span>
                              )
                            )}
                          </div>
                        )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// 2. Scientific Critic 卡片
// ---------------------------------------------------------------------------

const CATEGORY_LABELS: Record<string, string> = {
  consistency: "一致性",
  evidence: "证据",
  references: "引用",
  parameters: "参数",
  mechanism: "机制",
  dynamics: "动力学",
  methodology: "方法学",
  interpretation: "解读",
};

function CriticCard({ report }: { report: CriticReport }) {
  const [expanded, setExpanded] = useState(false);
  const status = overallStatusToStatus(report.overall_status);
  const findings = report.findings ?? [];

  return (
    <div className="overflow-hidden rounded-md border border-zinc-800 bg-zinc-900/70">
      <button
        type="button"
        onClick={() => setExpanded(!expanded)}
        aria-expanded={expanded}
        className={cn(
          "flex w-full items-center gap-2 border-l-2 px-2.5 py-2 text-left transition-colors hover:bg-zinc-800/40",
          STATUS_BORDER[status]
        )}
      >
        <span className="shrink-0 text-zinc-400" aria-hidden>
          <Microscope className="h-3.5 w-3.5" />
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-1.5">
            <span className="font-mono text-[10px] text-zinc-500">SA-2</span>
            <span className="truncate text-[11px] font-semibold text-zinc-100">
              Scientific Critic
            </span>
          </div>
          <div className="truncate text-[9px] text-zinc-500">
            {findings.length} finding(s) · {report.overall_status || "—"}
          </div>
        </div>
        <StatusBadge status={status} />
        <ChevronRight
          className={cn(
            "h-3.5 w-3.5 shrink-0 text-zinc-500 transition-transform",
            expanded && "rotate-90"
          )}
          aria-hidden
        />
      </button>
      {expanded && (
        <div className="border-t border-zinc-800/80 px-3 py-2">
          {findings.length === 0 ? (
            <p className="text-[10px] text-emerald-400">
              ✓ 审稿未发现科学性问题
            </p>
          ) : (
            <div className="space-y-2">
              {findings.map((f, i) => {
                const fStatus = severityToStatus(f.severity);
                return (
                  <div
                    key={i}
                    className={cn(
                      "rounded border px-2 py-1.5",
                      fStatus === "fail" &&
                        "border-red-900/40 bg-red-950/20",
                      fStatus === "warning" &&
                        "border-amber-900/40 bg-amber-950/20",
                      fStatus === "pass" &&
                        "border-emerald-900/40 bg-emerald-950/20",
                      fStatus === "skipped" &&
                        "border-zinc-800 bg-zinc-900/40"
                    )}
                  >
                    <div className="flex items-center gap-1.5">
                      <span className="rounded bg-zinc-800/60 px-1 py-0.5 text-[9px] font-medium text-zinc-400">
                        {CATEGORY_LABELS[f.category ?? ""] ?? f.category ?? "—"}
                      </span>
                      {f.severity && (
                        <span
                          className={cn(
                            "rounded px-1 py-0.5 text-[8px] font-semibold uppercase",
                            STATUS_BADGE[fStatus].cls
                          )}
                        >
                          {f.severity}
                        </span>
                      )}
                    </div>
                    {f.finding && (
                      <p className="mt-1 text-[10px] text-zinc-200">
                        {f.finding}
                      </p>
                    )}
                    {f.evidence && (
                      <p className="mt-0.5 text-[9px] italic text-zinc-500">
                        evidence: {f.evidence}
                      </p>
                    )}
                    {f.suggestion && (
                      <p className="mt-0.5 text-[9px] text-amber-300/80">
                        ↳ {f.suggestion}
                      </p>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// 3. Multi-dim Confidence 卡片
// ---------------------------------------------------------------------------

function MultiDimConfidenceCard({
  report,
}: {
  report: MultiDimConfidenceReport;
}) {
  const [expanded, setExpanded] = useState(false);
  const overall = report.overall_confidence ?? 0;
  const status = scoreToStatus(overall);
  const axes = report.axes ?? [];

  return (
    <div className="overflow-hidden rounded-md border border-zinc-800 bg-zinc-900/70">
      <button
        type="button"
        onClick={() => setExpanded(!expanded)}
        aria-expanded={expanded}
        className={cn(
          "flex w-full items-center gap-2 border-l-2 px-2.5 py-2 text-left transition-colors hover:bg-zinc-800/40",
          STATUS_BORDER[status]
        )}
      >
        <span className="shrink-0 text-zinc-400" aria-hidden>
          <Gauge className="h-3.5 w-3.5" />
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-1.5">
            <span className="font-mono text-[10px] text-zinc-500">SA-3</span>
            <span className="truncate text-[11px] font-semibold text-zinc-100">
              Multi-dim Confidence
            </span>
          </div>
          <div className="truncate text-[9px] text-zinc-500">
            {axes.length} axes · overall {(overall * 100).toFixed(1)}%
          </div>
        </div>
        <StatusBadge status={status} />
        <ChevronRight
          className={cn(
            "h-3.5 w-3.5 shrink-0 text-zinc-500 transition-transform",
            expanded && "rotate-90"
          )}
          aria-hidden
        />
      </button>
      {expanded && (
        <div className="border-t border-zinc-800/80 px-3 py-2">
          {axes.length === 0 ? (
            <p className="text-[10px] text-zinc-500">无置信度轴数据</p>
          ) : (
            <div className="space-y-1.5">
              {axes.map((ax, i) => {
                const axStatus = scoreToStatus(ax.score);
                const score = ax.score ?? 0;
                return (
                  <div
                    key={i}
                    className="flex items-center gap-2 rounded bg-zinc-800/30 px-2 py-1"
                  >
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center justify-between gap-1">
                        <span className="truncate text-[10px] text-zinc-300">
                          {ax.axis_name || `axis-${i}`}
                        </span>
                        <span className="font-mono text-[9px] text-zinc-500">
                          {(score * 100).toFixed(0)}%
                        </span>
                      </div>
                      {/* 进度条 */}
                      <div className="mt-1 h-1 overflow-hidden rounded-full bg-zinc-700/50">
                        <div
                          className={cn(
                            "h-full rounded-full",
                            axStatus === "pass" && "bg-emerald-500",
                            axStatus === "warning" && "bg-amber-500",
                            axStatus === "fail" && "bg-red-500",
                            axStatus === "skipped" && "bg-zinc-600"
                          )}
                          style={{ width: `${Math.min(score * 100, 100)}%` }}
                        />
                      </div>
                    </div>
                    <StatusBadge status={axStatus} />
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// 主面板
// ---------------------------------------------------------------------------

export function ScientificAlignmentPanel() {
  const consistency = useWorkbenchStore(
    (s) => s.saConsistencyReport
  ) as ConsistencyReport | null;
  const critic = useWorkbenchStore(
    (s) => s.saCriticReport
  ) as CriticReport | null;
  const multiDim = useWorkbenchStore(
    (s) => s.saMultiDimConfidence
  ) as MultiDimConfidenceReport | null;

  const hasAny = consistency || critic || multiDim;

  if (!hasAny) {
    // SA 报告未生成时不渲染（避免空态占用右栏空间）
    return null;
  }

  return (
    <section className="space-y-2">
      {/* 标题栏 */}
      <div className="flex items-center justify-between rounded-md border border-zinc-800 bg-zinc-900/70 px-2.5 py-1.5">
        <div className="flex items-center gap-1.5">
          <ShieldAlert className="h-3.5 w-3.5 text-amber-400" />
          <span className="text-[11px] font-semibold text-zinc-200">
            Scientific Alignment
          </span>
        </div>
        <span className="font-mono text-[9px] text-zinc-500">
          {consistency?.pathway || critic?.pathway || multiDim?.pathway || ""}
        </span>
      </div>

      {/* 三张卡片 */}
      {consistency && <ConsistencyCard report={consistency} />}
      {critic && <CriticCard report={critic} />}
      {multiDim && <MultiDimConfidenceCard report={multiDim} />}
    </section>
  );
}
