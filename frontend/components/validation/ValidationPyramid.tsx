"use client";

import React, { useMemo, useState } from "react";
import {
  ShieldCheck,
  ChevronRight,
  Network,
  GitCompare,
  BarChart3,
  Lightbulb,
} from "lucide-react";
import { cn } from "@/lib/utils";
import {
  ValidationDetail,
  type CheckStatus,
  type ValidationCheck,
} from "./ValidationDetail";

/**
 * ValidationPyramid — the 5-level validation pyramid.
 *
 * Subscribes to the workbench store's `validationReport` (hydrated by the
 * `v4_validation_report` SSE event) and renders Levels 1–5 as a vertical stack
 * of cards with a subtle pyramid taper (widest at the top). Each card carries
 * a colour-coded left border + status badge and expands on click to reveal the
 * per-check detail (delegated to `ValidationDetail`).
 *
 * Report shape mirrors `backend/app/validation_v2/*`:
 *   { level1, level2, level3, level4, level5, overall_pass, failed_levels, … }
 *
 * Task C.7.
 */

// ---------------------------------------------------------------------------
// Report types (loose mirror of the backend validation_v2 report dicts)
// ---------------------------------------------------------------------------

interface LevelBase {
  pass?: boolean;
  skipped?: boolean;
  reason?: string;
  method?: string;
  error?: string;
}

interface Level1Report extends LevelBase {
  mass_conservation_error?: number;
  non_negative_violations?: Array<{ species_name?: string; reason?: string }>;
  steady_state_check?: boolean;
  numerical_stability?: boolean;
  constraint_violations?: Array<{ constraint_name?: string; reason?: string }>;
}

interface Level2Report extends LevelBase {
  track?: "A" | "B" | "skipped" | string;
  peak_diff?: number | null;
  peak_time_diff?: number | null;
  amplification_diff?: number | null;
  sbml_sim_available?: boolean;
  similarity_score?: number;
  thresholds_applied?: { peak_time_diff?: number; amplification_diff?: number };
  sbml_model_id?: string;
  pathway_class?: string;
}

interface Level3Report extends LevelBase {
  crosstalk_consistency?: boolean;
  shared_species_conservation?: number;
  time_scale_alignment?: boolean;
}

interface BenchmarkItem {
  name?: string;
  source_pmid?: string;
  expected?: { range?: Array<number | null>; tolerance?: number };
  actual?: number | null;
  diff?: number | null;
  pass?: boolean;
  reason?: string;
}

interface Level4Report extends LevelBase {
  benchmarks?: BenchmarkItem[];
  pathway_class?: string;
}

interface Level5Report extends LevelBase {
  hypotheses_validated?: number;
  hypotheses_falsified?: number;
  evidence_support?: Array<{
    type?: string;
    source?: string;
    support?: string;
  }>;
  low_confidence?: boolean;
}

export interface ValidationReport {
  level1?: Level1Report;
  level2?: Level2Report;
  level3?: Level3Report;
  level4?: Level4Report;
  level5?: Level5Report;
  overall_pass?: boolean;
  short_circuit?: boolean;
  failed_levels?: string[];
  agent_version?: string;
}

type LevelKey = "level1" | "level2" | "level3" | "level4" | "level5";

// ---------------------------------------------------------------------------
// Level catalogue
// ---------------------------------------------------------------------------

interface LevelConfig {
  key: LevelKey;
  number: number;
  name: string;
  icon: React.ReactNode;
  /** Horizontal inset (px) — produces the pyramid silhouette (widest at top). */
  taper: number;
}

const LEVELS: LevelConfig[] = [
  {
    key: "level1",
    number: 1,
    name: "Internal Consistency",
    icon: <Network className="h-3.5 w-3.5" />,
    taper: 0,
  },
  {
    key: "level2",
    number: 2,
    name: "SBML / BioModels",
    icon: <GitCompare className="h-3.5 w-3.5" />,
    taper: 6,
  },
  {
    key: "level3",
    number: 3,
    name: "Cross-Pathway",
    icon: <Network className="h-3.5 w-3.5" />,
    taper: 12,
  },
  {
    key: "level4",
    number: 4,
    name: "Benchmark",
    icon: <BarChart3 className="h-3.5 w-3.5" />,
    taper: 18,
  },
  {
    key: "level5",
    number: 5,
    name: "Hypothesis",
    icon: <Lightbulb className="h-3.5 w-3.5" />,
    taper: 24,
  },
];

// ---------------------------------------------------------------------------
// Status derivation
// ---------------------------------------------------------------------------

const MASS_TOL = 0.05; // Level 1 mass conservation threshold (spec.md L281)
const SHARED_TOL = 0.1; // Level 3 shared species conservation (spec.md L298)
const SIM_PASS = 0.6; // Level 2 Track B structural similarity pass bar

function baseStatus(l: LevelBase): CheckStatus {
  if (l.skipped) return "skipped";
  if (l.pass === true) return "pass";
  return "fail";
}

function deriveLevelStatus(key: LevelKey, report: ValidationReport): CheckStatus {
  switch (key) {
    case "level1": {
      const l = report.level1;
      if (!l) return "skipped";
      const base = baseStatus(l);
      if (base !== "pass") return base;
      if ((l.non_negative_violations?.length ?? 0) > 0) return "warning";
      if ((l.constraint_violations?.length ?? 0) > 0) return "warning";
      if (l.steady_state_check === false) return "warning";
      return "pass";
    }
    case "level2": {
      const l = report.level2;
      if (!l) return "skipped";
      const base = baseStatus(l);
      if (base !== "pass") return base;
      // Track B is structural-only (no real simulation) → warn even on pass.
      if (l.track === "B") return "warning";
      return "pass";
    }
    case "level3": {
      const l = report.level3;
      if (!l) return "skipped";
      return baseStatus(l);
    }
    case "level4": {
      const l = report.level4;
      if (!l) return "skipped";
      return baseStatus(l);
    }
    case "level5": {
      const l = report.level5;
      if (!l) return "skipped";
      const base = baseStatus(l);
      if (base !== "pass") return base;
      if (l.low_confidence) return "warning";
      if ((l.hypotheses_falsified ?? 0) > 0) return "warning";
      return "pass";
    }
  }
}

// ---------------------------------------------------------------------------
// Per-level check flattening → ValidationCheck[]
// ---------------------------------------------------------------------------

function fmtPct(n: number, digits = 1): string {
  return `${(n * 100).toFixed(digits)}%`;
}

function buildLevel1Checks(l: Level1Report): ValidationCheck[] {
  const checks: ValidationCheck[] = [];
  const massErr = l.mass_conservation_error ?? 0;
  checks.push({
    name: "Mass Conservation",
    status: massErr <= MASS_TOL ? "pass" : "fail",
    value: fmtPct(massErr, 2),
    threshold: `≤ ${fmtPct(MASS_TOL, 0)}`,
    message:
      massErr > MASS_TOL
        ? `Exceeds tolerance (${fmtPct(massErr)} > ${fmtPct(MASS_TOL, 0)})`
        : undefined,
  });

  const nn = l.non_negative_violations ?? [];
  checks.push({
    name: "Non-Negative Concentrations",
    status: nn.length > 0 ? "warning" : "pass",
    value: nn.length ? `${nn.length} flag(s)` : "clean",
    message: nn.length ? nn[0]?.reason : undefined,
  });

  checks.push({
    name: "Steady State Reachability",
    status: l.steady_state_check ? "pass" : "warning",
    value: l.steady_state_check ? "reachable" : "at risk",
  });

  checks.push({
    name: "Numerical Stability (NaN/Inf)",
    status: l.numerical_stability ? "pass" : "fail",
    value: l.numerical_stability ? "stable" : "unstable",
    message: l.numerical_stability
      ? undefined
      : "NaN/Inf or stiff-system risk detected",
  });

  const cv = l.constraint_violations ?? [];
  checks.push({
    name: "Constraint Satisfaction",
    status: cv.length > 0 ? "warning" : "pass",
    value: cv.length ? `${cv.length} violation(s)` : "satisfied",
    message: cv.length ? cv[0]?.reason : undefined,
  });

  if (l.error) {
    checks.push({ name: "Error", status: "fail", message: l.error });
  }
  return checks;
}

function buildLevel2Checks(l: Level2Report): ValidationCheck[] {
  const checks: ValidationCheck[] = [];
  const track = l.track ?? "skipped";
  checks.push({
    name: "Comparison Track",
    status:
      track === "A" ? "pass" : track === "B" ? "warning" : "skipped",
    value:
      track === "A"
        ? "Track A · roadrunner"
        : track === "B"
          ? "Track B · structural"
          : "skipped",
    message: l.method,
  });

  if (l.peak_diff !== null && l.peak_diff !== undefined) {
    checks.push({
      name: "Peak Difference",
      status: "pass",
      value: fmtPct(l.peak_diff),
    });
  }

  if (l.peak_time_diff !== null && l.peak_time_diff !== undefined) {
    const th = l.thresholds_applied?.peak_time_diff ?? 5;
    checks.push({
      name: "Peak Time Difference",
      status: l.peak_time_diff <= th ? "pass" : "fail",
      value: `${l.peak_time_diff.toFixed(2)} min`,
      threshold: `≤ ${th} min`,
    });
  }

  if (l.amplification_diff !== null && l.amplification_diff !== undefined) {
    const th = l.thresholds_applied?.amplification_diff ?? 0.3;
    checks.push({
      name: "Amplification Difference",
      status: l.amplification_diff <= th ? "pass" : "fail",
      value: fmtPct(l.amplification_diff),
      threshold: `≤ ${fmtPct(th, 0)}`,
    });
  }

  if (l.similarity_score !== undefined) {
    checks.push({
      name: "Structural Similarity",
      status: l.similarity_score >= SIM_PASS ? "pass" : "fail",
      value: l.similarity_score.toFixed(3),
      threshold: `≥ ${SIM_PASS.toFixed(2)}`,
    });
  }

  if (l.sbml_model_id) {
    checks.push({
      name: "SBML Model",
      status: "pass",
      value: l.sbml_model_id,
    });
  }

  if (l.error) {
    checks.push({ name: "Error", status: "fail", message: l.error });
  }
  return checks;
}

function buildLevel3Checks(l: Level3Report): ValidationCheck[] {
  const checks: ValidationCheck[] = [];
  checks.push({
    name: "Cross-talk Consistency",
    status: l.crosstalk_consistency ? "pass" : "fail",
    value: l.crosstalk_consistency ? "consistent" : "inconsistent",
  });

  const cons = l.shared_species_conservation ?? 0;
  checks.push({
    name: "Shared Species Conservation",
    status: cons <= SHARED_TOL ? "pass" : "fail",
    value: fmtPct(cons, 2),
    threshold: `≤ ${fmtPct(SHARED_TOL, 0)}`,
  });

  checks.push({
    name: "Time-Scale Alignment",
    status: l.time_scale_alignment ? "pass" : "fail",
    value: l.time_scale_alignment ? "aligned" : "misaligned",
  });

  if (l.error) {
    checks.push({ name: "Error", status: "fail", message: l.error });
  }
  return checks;
}

function formatRange(range?: Array<number | null>): string {
  if (!range || range.length !== 2) return "—";
  const lo = range[0];
  const hi = range[1];
  return `[${lo === null || lo === undefined ? "−∞" : lo}, ${hi === null || hi === undefined ? "+∞" : hi}]`;
}

function buildLevel4Checks(l: Level4Report): ValidationCheck[] {
  const benchmarks = l.benchmarks ?? [];
  if (benchmarks.length === 0) {
    return [
      {
        name: "Benchmark Match",
        status: "pass",
        value: "no benchmark matched",
        message: l.method ?? "no_benchmark_matched",
      },
    ];
  }
  return benchmarks.map((bm) => {
    const rangeStr = formatRange(bm.expected?.range);
    const tol = bm.expected?.tolerance ?? 0;
    return {
      name: bm.name ?? "benchmark",
      status: bm.pass ? "pass" : "fail",
      value:
        bm.actual === null || bm.actual === undefined
          ? "missing"
          : bm.actual.toFixed(3),
      threshold: `${rangeStr} ± ${tol}`,
      message:
        bm.reason ??
        (bm.pass
          ? undefined
          : `deviation ${bm.diff?.toFixed(3) ?? "?"}${bm.source_pmid ? ` · ${bm.source_pmid}` : ""}`),
    } as ValidationCheck;
  });
}

function buildLevel5Checks(l: Level5Report): ValidationCheck[] {
  const checks: ValidationCheck[] = [];
  checks.push({
    name: "Hypotheses Validated",
    status: "pass",
    value: `${l.hypotheses_validated ?? 0}`,
  });
  checks.push({
    name: "Hypotheses Falsified",
    status: (l.hypotheses_falsified ?? 0) > 0 ? "warning" : "pass",
    value: `${l.hypotheses_falsified ?? 0}`,
  });
  const ev = l.evidence_support ?? [];
  checks.push({
    name: "Evidence Support",
    status: ev.length > 0 ? "pass" : "warning",
    value: `${ev.length} item(s)`,
  });
  checks.push({
    name: "Confidence",
    status: l.low_confidence ? "warning" : "pass",
    value: l.low_confidence ? "low" : "normal",
  });
  return checks;
}

function buildChecks(key: LevelKey, report: ValidationReport): ValidationCheck[] {
  switch (key) {
    case "level1":
      return report.level1 ? buildLevel1Checks(report.level1) : [];
    case "level2":
      return report.level2 ? buildLevel2Checks(report.level2) : [];
    case "level3":
      return report.level3 ? buildLevel3Checks(report.level3) : [];
    case "level4":
      return report.level4 ? buildLevel4Checks(report.level4) : [];
    case "level5":
      return report.level5 ? buildLevel5Checks(report.level5) : [];
  }
}

function levelNotes(key: LevelKey, report: ValidationReport): string | undefined {
  const l = report[key];
  if (!l) return undefined;
  return l.reason ?? l.method;
}

// ---------------------------------------------------------------------------
// Status → visual maps
// ---------------------------------------------------------------------------

const STATUS_BORDER: Record<CheckStatus, string> = {
  pass: "border-l-emerald-500",
  warning: "border-l-amber-500",
  fail: "border-l-red-500",
  skipped: "border-l-zinc-600",
};

const BADGE_STYLE: Record<CheckStatus, { label: string; cls: string }> = {
  pass: { label: "Pass", cls: "bg-emerald-500/15 text-emerald-300" },
  warning: { label: "Warn", cls: "bg-amber-500/15 text-amber-300" },
  fail: { label: "Fail", cls: "bg-red-500/15 text-red-300" },
  skipped: { label: "Skip", cls: "bg-zinc-500/15 text-zinc-400" },
};

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function StatusBadge({ status }: { status: CheckStatus }) {
  const cfg = BADGE_STYLE[status];
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

function ScoreHeader({
  passed,
  total,
  overallPass,
}: {
  passed: number;
  total: number;
  overallPass: boolean;
}) {
  return (
    <div className="flex items-center justify-between rounded-md border border-zinc-800 bg-zinc-900/70 px-3 py-2">
      <div className="flex items-center gap-2">
        <ShieldCheck
          className={cn(
            "h-4 w-4",
            overallPass ? "text-emerald-400" : "text-red-400"
          )}
        />
        <div className="leading-tight">
          <div className="text-[11px] font-semibold text-zinc-100">
            {passed}/{total} levels passed
          </div>
          <div className="text-[9px] uppercase tracking-wide text-zinc-500">
            {overallPass ? "validation clear" : "validation failed"}
          </div>
        </div>
      </div>
      <span
        className={cn(
          "rounded px-2 py-0.5 text-[10px] font-bold uppercase tracking-wide",
          overallPass
            ? "bg-emerald-500/15 text-emerald-300"
            : "bg-red-500/15 text-red-300"
        )}
      >
        {overallPass ? "All Pass" : "Failed"}
      </span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export interface ValidationPyramidProps {
  report: ValidationReport;
  className?: string;
}

export function ValidationPyramid({
  report,
  className,
}: ValidationPyramidProps) {
  const [expanded, setExpanded] = useState<LevelKey | null>(null);

  const rows = useMemo(
    () =>
      LEVELS.map((cfg) => {
        const status = deriveLevelStatus(cfg.key, report);
        const checks = buildChecks(cfg.key, report);
        const notes = levelNotes(cfg.key, report);
        return { cfg, status, checks, notes };
      }),
    [report]
  );

  const passedCount = rows.filter(
    (r) => report[r.cfg.key]?.pass === true
  ).length;
  const overallPass = report.overall_pass === true;

  return (
    <div className={cn("space-y-2", className)}>
      <ScoreHeader
        passed={passedCount}
        total={LEVELS.length}
        overallPass={overallPass}
      />

      <div className="space-y-1.5">
        {rows.map(({ cfg, status, checks, notes }) => {
          const isOpen = expanded === cfg.key;
          return (
            <div
              key={cfg.key}
              className="overflow-hidden rounded-md border border-zinc-800 bg-zinc-900/70"
              style={{ marginLeft: cfg.taper, marginRight: cfg.taper }}
            >
              <button
                type="button"
                onClick={() => setExpanded(isOpen ? null : cfg.key)}
                aria-expanded={isOpen}
                className={cn(
                  "flex w-full items-center gap-2 border-l-2 px-2.5 py-2 text-left transition-colors hover:bg-zinc-800/40 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-zinc-600",
                  STATUS_BORDER[status]
                )}
              >
                <span className="shrink-0 text-zinc-400" aria-hidden>
                  {cfg.icon}
                </span>
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-1.5">
                    <span className="font-mono text-[10px] text-zinc-500">
                      L{cfg.number}
                    </span>
                    <span className="truncate text-[11px] font-semibold text-zinc-100">
                      {cfg.name}
                    </span>
                  </div>
                </div>
                <StatusBadge status={status} />
                <ChevronRight
                  className={cn(
                    "h-3.5 w-3.5 shrink-0 text-zinc-500 transition-transform",
                    isOpen && "rotate-90"
                  )}
                  aria-hidden
                />
              </button>
              {isOpen && checks.length > 0 && (
                <ValidationDetail
                  level={cfg.number}
                  checks={checks}
                  notes={notes}
                />
              )}
              {isOpen && checks.length === 0 && (
                <div className="border-t border-zinc-800/80 px-3 py-2 text-[10px] text-zinc-500">
                  No checks available for this level.
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
