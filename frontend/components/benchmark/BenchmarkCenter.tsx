"use client";

import React, { useCallback, useMemo, useRef, useState } from "react";
import Link from "next/link";
import {
  BarChart3,
  ArrowLeft,
  PlayCircle,
  StopCircle,
  CheckCircle2,
  XCircle,
  Clock,
  Activity,
  Loader2,
  AlertCircle,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { BenchmarkCard, type BenchmarkDef, type BenchmarkCardState } from "./BenchmarkCard";
import {
  runAllBenchmarksStream,
  runBenchmarkStream,
  SUITE_MARKER,
  type BenchmarkSummary,
  type BenchmarkResultEvent,
  type BenchmarkStreamEvent,
} from "@/lib/benchmarkSse";

// ---------------------------------------------------------------------------
// Static benchmark registry — mirrors backend/benchmarks/*.yaml and the
// sorted order used by BenchmarkRunner.list_benchmarks() (Python's sort:
// uppercase identifiers first, lowercase "p53" last).
// ---------------------------------------------------------------------------

const BENCHMARKS: BenchmarkDef[] = [
  {
    pathway_class: "APOPTOSIS",
    name: "Apoptosis (Intrinsic + Extrinsic)",
    description:
      "Intrinsic apoptosis: BH3-only -> Bax/Bak -> MOMP -> Cyt c -> Apaf-1 -> Casp9 -> Casp3 (bistable, point-of-no-return).",
  },
  {
    pathway_class: "CELL_CYCLE",
    name: "Cell Cycle Regulation",
    description:
      "CyclinB-Cdk1-APC/C delayed negative feedback produces 8-12h oscillation; Rb-E2F bistable G1/S switch.",
  },
  {
    pathway_class: "EGFR_RTK",
    name: "EGFR RTK Signaling",
    description:
      "EGF stimulation leads to EGFR phosphorylation and MAPK cascade activation.",
  },
  {
    pathway_class: "JAK_STAT",
    name: "JAK-STAT Signaling",
    description:
      "IL-6/EPO stimulation activates JAK->STAT5 phosphorylation; STAT5->SOCS transcriptional negative feedback produces single pulse.",
  },
  {
    pathway_class: "MAPK_ERK",
    name: "MAPK / ERK Signaling Cascade",
    description:
      "Three-tier MAPK cascade (Raf->MEK->ERK) exhibits zero-order ultrasensitivity and signal amplification.",
  },
  {
    pathway_class: "NF_KB",
    name: "NF-κB Signaling",
    description:
      "TNF stimulation activates IKK->IκBα phosphorylation->degradation->NF-κB release->nuclear translocation; NF-κB-IκBα DDE negative feedback produces nuclear oscillation.",
  },
  {
    pathway_class: "PI3K_AKT_mTOR",
    name: "PI3K / AKT / mTOR Signaling",
    description:
      "Growth factor stimulation activates PI3K->PIP3->AKT->TSC2->Rheb->mTORC1->S6K1 cascade.",
  },
  {
    pathway_class: "TGF_BETA",
    name: "TGF-β Signaling",
    description:
      "TGF-β stimulation activates TβRI/II->Smad2/3 phosphorylation->pSmad2:Smad4 heteromer->nuclear accumulation->transcription; SMAD7 transcriptional negative feedback.",
  },
  {
    pathway_class: "WNT",
    name: "Wnt / β-catenin Signaling",
    description:
      "Wnt+Frizzled+LRP5/6->Dvl->LRP6 phosphorylation->Axin recruitment->destruction complex dissociation->β-catenin accumulation->nuclear translocation->TCF/LEF transcription.",
  },
  {
    pathway_class: "p53",
    name: "p53 Tumor Suppressor Signaling",
    description:
      "DNA damage triggers ATM-mediated p53 phosphorylation, p53-Mdm2 DDE negative feedback produces pulse oscillation.",
  },
];

const TOTAL = BENCHMARKS.length;

const IDLE_STATE: BenchmarkCardState = { status: "idle" };

// ---------------------------------------------------------------------------
// BenchmarkCenter
// ---------------------------------------------------------------------------

export function BenchmarkCenter() {
  const [states, setStates] = useState<Record<string, BenchmarkCardState>>({});
  const [suiteRunning, setSuiteRunning] = useState(false);
  const [summary, setSummary] = useState<BenchmarkSummary | null>(null);
  const [suiteError, setSuiteError] = useState<string | null>(null);

  const suiteAbortRef = useRef<AbortController | null>(null);
  const cardAbortRef = useRef<Record<string, AbortController>>({});

  // --- state helpers ------------------------------------------------------

  const patchCard = useCallback(
    (pathwayClass: string, patch: Partial<BenchmarkCardState>) => {
      setStates((prev) => ({
        ...prev,
        [pathwayClass]: { ...(prev[pathwayClass] ?? IDLE_STATE), ...patch },
      }));
    },
    []
  );

  // --- result applier (shared by single-card + run-all) -------------------

  const applyResult = useCallback(
    (pathwayClass: string, result: BenchmarkResultEvent) => {
      patchCard(pathwayClass, {
        status: result.status === "pass" ? "pass" : "fail",
        result,
        step: undefined,
        lastRunAt: Date.now(),
        error: undefined,
      });
    },
    [patchCard]
  );

  // --- single-card run ----------------------------------------------------

  const runOne = useCallback(
    async (pathwayClass: string) => {
      if (suiteRunning) return;
      // One standalone run at a time: cancel any in-flight per-card run.
      Object.values(cardAbortRef.current).forEach((ac) => ac.abort());
      cardAbortRef.current = {};

      const ac = new AbortController();
      cardAbortRef.current[pathwayClass] = ac;

      patchCard(pathwayClass, {
        status: "running",
        step: undefined,
        result: undefined,
        error: undefined,
        lastRunAt: undefined,
      });
      setSuiteError(null);

      let gotResult = false;
      await runBenchmarkStream(
        pathwayClass,
        (evt: BenchmarkStreamEvent) => {
          switch (evt.event) {
            case "benchmark_start": {
              const data = evt.data as { pathway_class?: string } | null;
              if (data?.pathway_class === pathwayClass) {
                patchCard(pathwayClass, { status: "running", step: undefined });
              }
              break;
            }
            case "benchmark_progress": {
              const data = evt.data as
                | { pathway_class?: string; step?: string }
                | null;
              if (data?.pathway_class === pathwayClass) {
                patchCard(pathwayClass, { status: "running", step: data.step });
              }
              break;
            }
            case "benchmark_result": {
              const data = evt.data as BenchmarkResultEvent | null;
              if (data && data.pathway_class === pathwayClass) {
                gotResult = true;
                applyResult(pathwayClass, data);
              }
              break;
            }
            case "error": {
              const msg =
                typeof evt.data === "string"
                  ? evt.data
                  : "Benchmark stream error.";
              patchCard(pathwayClass, { status: "fail", error: msg });
              break;
            }
            default:
              break;
          }
        },
        ac.signal
      );

      delete cardAbortRef.current[pathwayClass];
      // If the stream ended without a result, surface it as a failure.
      if (!gotResult) {
        setStates((prev) => {
          const cur = prev[pathwayClass];
          if (cur && cur.status === "running") {
            return {
              ...prev,
              [pathwayClass]: {
                ...cur,
                status: "fail",
                error: "No result received from benchmark stream.",
              },
            };
          }
          return prev;
        });
      }
    },
    [patchCard, applyResult, suiteRunning]
  );

  // --- run all -----------------------------------------------------------

  const runAll = useCallback(async () => {
    if (suiteRunning) return;
    // Cancel any in-flight per-card run.
    Object.values(cardAbortRef.current).forEach((ac) => ac.abort());
    cardAbortRef.current = {};

    const ac = new AbortController();
    suiteAbortRef.current = ac;

    setSuiteRunning(true);
    setSuiteError(null);
    setSummary(null);
    // Reset every card to idle for a clean suite run.
    setStates({});

    await runAllBenchmarksStream(
      (evt: BenchmarkStreamEvent) => {
        switch (evt.event) {
          case "benchmark_start": {
            const data = evt.data as
              | { pathway_class?: string; name?: string; total?: number }
              | null;
            if (!data) break;
            if (data.pathway_class && data.pathway_class !== SUITE_MARKER) {
              patchCard(data.pathway_class, {
                status: "running",
                step: undefined,
                result: undefined,
                error: undefined,
              });
            }
            break;
          }
          case "benchmark_progress": {
            const data = evt.data as
              | { pathway_class?: string; step?: string }
              | null;
            if (data?.pathway_class) {
              patchCard(data.pathway_class, {
                status: "running",
                step: data.step,
              });
            }
            break;
          }
          case "benchmark_result": {
            const data = evt.data as BenchmarkResultEvent | null;
            if (data?.pathway_class) {
              applyResult(data.pathway_class, data);
            }
            break;
          }
          case "benchmark_complete": {
            if (evt.data && typeof evt.data === "object") {
              setSummary(evt.data as BenchmarkSummary);
            }
            break;
          }
          case "error": {
            const msg =
              typeof evt.data === "string"
                ? evt.data
                : "Benchmark suite stream error.";
            setSuiteError(msg);
            break;
          }
          default:
            break;
        }
      },
      (completeSummary: BenchmarkSummary) => {
        setSummary(completeSummary);
      },
      ac.signal
    );

    suiteAbortRef.current = null;
    setSuiteRunning(false);
  }, [patchCard, applyResult, suiteRunning]);

  const stopAll = useCallback(() => {
    suiteAbortRef.current?.abort();
    Object.values(cardAbortRef.current).forEach((ac) => ac.abort());
    cardAbortRef.current = {};
    suiteAbortRef.current = null;
    setSuiteRunning(false);
    // Mark any still-running cards as stopped.
    setStates((prev) => {
      const next: Record<string, BenchmarkCardState> = {};
      for (const [k, v] of Object.entries(prev)) {
        next[k] =
          v.status === "running"
            ? { ...v, status: "idle", step: undefined, error: "Stopped by user." }
            : v;
      }
      return next;
    });
  }, []);

  // --- derived summary stats ---------------------------------------------

  const stats = useMemo(() => {
    let passed = 0;
    let failed = 0;
    let completed = 0;
    let runtimeSum = 0;
    let runtimeCount = 0;
    for (const def of BENCHMARKS) {
      const st = states[def.pathway_class];
      if (!st) continue;
      if (st.status === "pass") {
        passed += 1;
        completed += 1;
      } else if (st.status === "fail") {
        failed += 1;
        completed += 1;
      }
      if (st.result?.runtime_seconds !== undefined) {
        runtimeSum += st.result.runtime_seconds;
        runtimeCount += 1;
      }
    }
    const avgRuntime = runtimeCount > 0 ? runtimeSum / runtimeCount : 0;
    const totalRuntime =
      summary?.runtime_seconds ?? runtimeSum;
    return {
      passed,
      failed,
      completed,
      avgRuntime,
      totalRuntime,
      runCount: completed,
    };
  }, [states, summary]);

  const overallStatus: "all-pass" | "has-fail" | "empty" | "partial" =
    stats.completed === 0
      ? "empty"
      : stats.failed === 0
        ? "all-pass"
        : stats.passed === 0
          ? "has-fail"
          : "partial";

  const progressPct =
    TOTAL > 0 ? Math.round((stats.completed / TOTAL) * 100) : 0;

  // --- render -------------------------------------------------------------

  return (
    <main className="flex min-h-screen flex-col bg-zinc-950 text-zinc-100">
      {/* Header */}
      <header className="flex h-12 shrink-0 items-center gap-3 border-b border-zinc-800 px-4">
        <Link
          href="/workspace"
          className="flex items-center gap-1.5 text-xs text-zinc-400 hover:text-zinc-100"
        >
          <ArrowLeft className="h-3.5 w-3.5" />
          Back to Workspace
        </Link>
        <div className="flex items-center gap-2">
          <BarChart3 className="h-4 w-4 text-blue-400" />
          <span className="text-sm font-semibold">Benchmark Center</span>
        </div>
        <div className="ml-auto flex items-center gap-2">
          {suiteRunning ? (
            <Button
              type="button"
              size="sm"
              variant="destructive"
              onClick={stopAll}
            >
              <StopCircle className="h-3.5 w-3.5" />
              Stop
            </Button>
          ) : (
            <Button
              type="button"
              size="sm"
              onClick={runAll}
              disabled={suiteRunning}
            >
              <PlayCircle className="h-3.5 w-3.5" />
              Run All Benchmarks
            </Button>
          )}
        </div>
      </header>

      <div className="flex flex-1 flex-col gap-4 p-4">
        {/* Summary bar */}
        <section className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-4">
          <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
            <SummaryStat
              icon={<CheckCircle2 className="h-4 w-4 text-emerald-400" />}
              label="Passed"
              value={`${stats.passed}/${TOTAL}`}
              valueClass="text-emerald-300"
            />
            <SummaryStat
              icon={<XCircle className="h-4 w-4 text-red-400" />}
              label="Failed"
              value={`${stats.failed}/${TOTAL}`}
              valueClass="text-red-300"
            />
            <SummaryStat
              icon={<Clock className="h-4 w-4 text-zinc-400" />}
              label="Total Runtime"
              value={`${stats.totalRuntime.toFixed(3)}s`}
            />
            <SummaryStat
              icon={<Activity className="h-4 w-4 text-blue-400" />}
              label="Avg Runtime"
              value={
                stats.avgRuntime > 0
                  ? `${stats.avgRuntime.toFixed(3)}s`
                  : "—"
              }
            />
            <div className="ml-auto flex items-center gap-2">
              <OverallStatusBadge status={overallStatus} />
            </div>
          </div>

          {/* Progress bar (run-all) */}
          {suiteRunning && (
            <div className="mt-3">
              <div className="mb-1 flex items-center justify-between text-[10px] text-zinc-500">
                <span className="inline-flex items-center gap-1">
                  <Loader2 className="h-3 w-3 animate-spin" />
                  Running suite…
                </span>
                <span>
                  {stats.completed}/{TOTAL} ({progressPct}%)
                </span>
              </div>
              <div className="h-1.5 w-full overflow-hidden rounded-full bg-zinc-800">
                <div
                  className="h-full rounded-full bg-blue-500 transition-all duration-300"
                  style={{ width: `${progressPct}%` }}
                />
              </div>
            </div>
          )}

          {/* Suite-level error */}
          {suiteError && (
            <div className="mt-3 flex items-start gap-2 rounded-md border border-red-800/50 bg-red-500/5 px-3 py-2">
              <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-red-400" />
              <p className="text-xs leading-snug text-red-300/90">
                {suiteError}
              </p>
            </div>
          )}
        </section>

        {/* Grid of benchmark cards */}
        <section className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {BENCHMARKS.map((def) => (
            <BenchmarkCard
              key={def.pathway_class}
              def={def}
              state={states[def.pathway_class] ?? IDLE_STATE}
              suiteRunning={suiteRunning}
              onRun={runOne}
            />
          ))}
        </section>

        {/* Footer note */}
        <p className="mt-2 text-center text-[10px] text-zinc-600">
          Streaming from{" "}
          <code className="font-mono text-zinc-500">
            POST /api/v4/benchmarks/run
          </code>{" "}
          — 10-pathway Official Benchmark Suite (Task E.1 backend).
        </p>
      </div>
    </main>
  );
}

// ---------------------------------------------------------------------------
// Small presentational helpers
// ---------------------------------------------------------------------------

function SummaryStat({
  icon,
  label,
  value,
  valueClass,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  valueClass?: string;
}) {
  return (
    <div className="flex items-center gap-2">
      <span className="shrink-0" aria-hidden>
        {icon}
      </span>
      <div className="leading-tight">
        <div className="text-[10px] uppercase tracking-wide text-zinc-500">
          {label}
        </div>
        <div className={cn("text-sm font-semibold", valueClass ?? "text-zinc-100")}>
          {value}
        </div>
      </div>
    </div>
  );
}

function OverallStatusBadge({
  status,
}: {
  status: "all-pass" | "has-fail" | "empty" | "partial";
}) {
  if (status === "empty") {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-full border border-zinc-700 bg-zinc-800/40 px-3 py-1 text-xs text-zinc-400">
        <Clock className="h-3.5 w-3.5" />
        Awaiting run
      </span>
    );
  }
  if (status === "all-pass") {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-full border border-emerald-700/60 bg-emerald-500/10 px-3 py-1 text-xs text-emerald-300">
        <CheckCircle2 className="h-3.5 w-3.5" />
        All passing
      </span>
    );
  }
  if (status === "partial") {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-full border border-amber-700/60 bg-amber-500/10 px-3 py-1 text-xs text-amber-300">
        <AlertCircle className="h-3.5 w-3.5" />
        Partial pass
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full border border-red-700/60 bg-red-500/10 px-3 py-1 text-xs text-red-300">
      <XCircle className="h-3.5 w-3.5" />
      Failures detected
    </span>
  );
}

export default BenchmarkCenter;
