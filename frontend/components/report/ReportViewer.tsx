"use client";

import React, { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import {
  ArrowLeft,
  FileText,
  Loader2,
  AlertTriangle,
  FlaskConical,
  Microscope,
  ShieldCheck,
  BookOpen,
  Lightbulb,
  Download,
  Printer,
  Link2,
  Check,
  ClipboardList,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { fetchReport, type ExperimentReport, type PathwayClass } from "@/lib/api";
import type { SimulationResultExtras } from "@/components/simulation/shared";
import {
  type ValidationReport,
  ValidationPyramid,
} from "@/components/validation/ValidationPyramid";
import {
  type Hypothesis,
  deriveFalsifiability,
} from "@/components/hypothesis/types";
import { ExperimentCard } from "@/components/hypothesis/ExperimentCard";

import { ReportSection } from "./ReportSection";
import { DynamicAnalysis } from "./DynamicAnalysis";
import { LiteratureComparison, type BenchmarkComparison } from "./LiteratureComparison";

// ---------------------------------------------------------------------------
// Extended report types (v4 report payload — superset of ExperimentReport)
// ---------------------------------------------------------------------------

/** A single scientific finding with its biological significance + confidence. */
export interface ScientificFinding {
  observation: string;
  biological_significance: string;
  /** 0–1 confidence score. */
  confidence: number;
}

/**
 * The full v4 experiment report. Extends `ExperimentReport` with the rich
 * structured payloads the 6-section report page renders. All extra fields are
 * optional — the backend populates them as the v4 endpoints land; the page
 * degrades gracefully when a section's data is absent.
 */
export interface ReportData extends ExperimentReport {
  // Executive summary
  pathway_display_name?: string;
  hypothesis?: string;
  key_finding?: string;
  simulation_run_at?: string;
  // Dynamic analysis
  simulation_result?: SimulationResultExtras;
  // Validation
  validation_report?: ValidationReport;
  // Literature
  benchmarks?: BenchmarkComparison[];
  // Findings
  findings?: ScientificFinding[];
  // Future experiments
  hypotheses?: Hypothesis[];
}

// ---------------------------------------------------------------------------
// Pathway display labels
// ---------------------------------------------------------------------------

const PATHWAY_LABELS: Record<PathwayClass, string> = {
  egfr: "EGFR Signaling",
  mapk: "MAPK Cascade",
  pi3k_akt_mtor: "PI3K / AKT / mTOR",
  jak_stat: "JAK-STAT",
  nf_kappa_b: "NF-κB",
  wnt: "Wnt / β-catenin",
  tgf_beta: "TGF-β",
  p53: "p53 Network",
  apoptosis: "Apoptosis",
  cell_cycle: "Cell Cycle",
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatDate(iso?: string): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function pathwayLabel(report: ReportData): string {
  return report.pathway_display_name ?? PATHWAY_LABELS[report.pathway_class] ?? report.pathway_class;
}

/** Count how many of the 5 validation levels passed. */
function countPassedLevels(vr?: ValidationReport): { passed: number; total: number } {
  const total = 5;
  if (!vr) return { passed: 0, total };
  const levels = [vr.level1, vr.level2, vr.level3, vr.level4, vr.level5];
  const passed = levels.filter((l) => l?.pass === true).length;
  return { passed, total };
}

function confidencePct(c: number): number {
  if (!Number.isFinite(c)) return 0;
  return Math.round(Math.max(0, Math.min(1, c)) * 100);
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

type LoadStatus = "loading" | "ready" | "error" | "empty";

export function ReportViewer({ id }: { id: string }) {
  const [status, setStatus] = useState<LoadStatus>("loading");
  const [report, setReport] = useState<ReportData | null>(null);
  const [errorMsg, setErrorMsg] = useState("");
  const [copiedLink, setCopiedLink] = useState(false);
  const [copiedMd, setCopiedMd] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    setStatus("loading");
    setErrorMsg("");
    fetchReport(id, controller.signal)
      .then((data) => {
        const r = data as ReportData;
        setReport(r);
        // "empty" = fetched but has no structured content at all
        const hasContent =
          r.markdown ||
          r.simulation_result ||
          r.validation_report ||
          (r.findings && r.findings.length) ||
          (r.benchmarks && r.benchmarks.length) ||
          (r.hypotheses && r.hypotheses.length);
        setStatus(hasContent ? "ready" : "empty");
      })
      .catch((err: unknown) => {
        if (controller.signal.aborted) return;
        setErrorMsg(err instanceof Error ? err.message : "Failed to load report.");
        setStatus("error");
      });
    return () => controller.abort();
  }, [id]);

  const handleExportMarkdown = useCallback(() => {
    if (!report) return;
    const md = generateMarkdown(report);
    const blob = new Blob([md], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `report-${report.id}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [report]);

  const handleCopyMarkdown = useCallback(async () => {
    if (!report) return;
    try {
      await navigator.clipboard.writeText(generateMarkdown(report));
      setCopiedMd(true);
      setTimeout(() => setCopiedMd(false), 1500);
    } catch {
      /* clipboard unavailable */
    }
  }, [report]);

  const handlePrint = useCallback(() => {
    window.print();
  }, []);

  const handleCopyLink = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(window.location.href);
      setCopiedLink(true);
      setTimeout(() => setCopiedLink(false), 1500);
    } catch {
      /* clipboard unavailable */
    }
  }, []);

  // -------------------------------------------------------------------------
  // State: loading
  // -------------------------------------------------------------------------
  if (status === "loading") {
    return (
      <div className="flex flex-1 flex-col items-center justify-center gap-3 p-12 text-center">
        <Loader2 className="h-8 w-8 animate-spin text-emerald-400" />
        <p className="text-sm text-zinc-400">Loading report…</p>
        <span className="rounded-full border border-zinc-700 px-2 py-0.5 font-mono text-[10px] text-zinc-500">
          {id}
        </span>
      </div>
    );
  }

  // -------------------------------------------------------------------------
  // State: error
  // -------------------------------------------------------------------------
  if (status === "error") {
    return (
      <div className="flex flex-1 flex-col items-center justify-center gap-3 p-12 text-center">
        <AlertTriangle className="h-10 w-10 text-red-400" />
        <h2 className="text-lg font-semibold text-zinc-200">Report not found</h2>
        <p className="max-w-md text-sm text-zinc-500">
          {errorMsg || "The requested report could not be loaded."}
        </p>
        <Link href="/workspace">
          <Button variant="outline" size="sm">
            <ArrowLeft className="h-3.5 w-3.5" />
            Back to Workspace
          </Button>
        </Link>
      </div>
    );
  }

  // -------------------------------------------------------------------------
  // State: empty
  // -------------------------------------------------------------------------
  if (status === "empty" || !report) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center gap-3 p-12 text-center">
        <ClipboardList className="h-10 w-10 text-zinc-600" />
        <h2 className="text-lg font-semibold text-zinc-200">
          No report content yet
        </h2>
        <p className="max-w-md text-sm text-zinc-500">
          Run a simulation to generate a report. Reports are persisted once a
          simulation completes and the validation pipeline finishes.
        </p>
        <Link href="/workspace">
          <Button variant="outline" size="sm">
            <ArrowLeft className="h-3.5 w-3.5" />
            Back to Workspace
          </Button>
        </Link>
      </div>
    );
  }

  // -------------------------------------------------------------------------
  // State: ready — render the 6-section report
  // -------------------------------------------------------------------------
  const validation = report.validation_report ?? (report.validation as ValidationReport | undefined);
  const { passed, total } = countPassedLevels(validation);
  const findings = report.findings ?? [];
  const benchmarks = report.benchmarks ?? [];
  const hypotheses = report.hypotheses ?? [];

  return (
    <div className="mx-auto flex w-full max-w-4xl flex-col gap-4 p-4 pb-12">
      {/* Report header + export toolbar */}
      <div className="flex flex-col gap-3 border-b border-zinc-800 pb-3">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <h1 className="truncate text-xl font-bold text-zinc-100">
              {report.title || "Experiment Report"}
            </h1>
            <p className="mt-0.5 text-xs text-zinc-500">
              {pathwayLabel(report)} ·{" "}
              <span className="font-mono">{report.pathway_class}</span>
            </p>
          </div>
          <div className="flex shrink-0 flex-wrap items-center gap-1.5 print:hidden">
            <Button variant="outline" size="sm" onClick={handleExportMarkdown}>
              <Download className="h-3.5 w-3.5" />
              Markdown
            </Button>
            <Button variant="outline" size="sm" onClick={handleCopyMarkdown}>
              {copiedMd ? (
                <Check className="h-3.5 w-3.5 text-emerald-400" />
              ) : (
                <FileText className="h-3.5 w-3.5" />
              )}
              {copiedMd ? "Copied" : "Copy MD"}
            </Button>
            <Button variant="outline" size="sm" onClick={handlePrint}>
              <Printer className="h-3.5 w-3.5" />
              PDF
            </Button>
            <Button variant="outline" size="sm" onClick={handleCopyLink}>
              {copiedLink ? (
                <Check className="h-3.5 w-3.5 text-emerald-400" />
              ) : (
                <Link2 className="h-3.5 w-3.5" />
              )}
              {copiedLink ? "Copied" : "Copy Link"}
            </Button>
          </div>
        </div>
      </div>

      {/* 1. Executive Summary */}
      <ReportSection number={1} title="Executive Summary" icon={<Microscope className="h-3.5 w-3.5" />}>
        <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-2">
          <SummaryField label="Pathway" value={pathwayLabel(report)} />
          <SummaryField
            label="Pathway class"
            value={report.pathway_class}
            mono
          />
          <SummaryField
            label="Generated"
            value={formatDate(report.simulation_run_at ?? report.created_at)}
          />
          <SummaryField
            label="Validation"
            value={`${passed}/${total} levels passed`}
            valueClass={
              passed === total
                ? "text-emerald-300"
                : passed > 0
                  ? "text-amber-300"
                  : "text-red-300"
            }
          />
        </div>
        <div className="mt-3 space-y-2">
          <div>
            <div className="text-[10px] font-medium uppercase tracking-wide text-zinc-500">
              Hypothesis tested
            </div>
            <p className="mt-0.5 text-sm leading-relaxed text-zinc-200">
              {report.hypothesis || "—"}
            </p>
          </div>
          <div>
            <div className="text-[10px] font-medium uppercase tracking-wide text-zinc-500">
              Key finding
            </div>
            <p className="mt-0.5 text-sm leading-relaxed text-zinc-200">
              {report.key_finding || "—"}
            </p>
          </div>
        </div>
      </ReportSection>

      {/* 2. Scientific Findings */}
      <ReportSection number={2} title="Scientific Findings" icon={<FlaskConical className="h-3.5 w-3.5" />}>
        {findings.length === 0 ? (
          <EmptyInline text="No structured findings were extracted for this report." />
        ) : (
          <ul className="space-y-2.5">
            {findings.map((f, idx) => (
              <li
                key={idx}
                className="rounded-md border border-zinc-800 bg-zinc-900/40 p-2.5"
              >
                <div className="flex items-start gap-2">
                  <span className="mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center rounded-full bg-emerald-500/15 font-mono text-[9px] font-semibold text-emerald-300">
                    {idx + 1}
                  </span>
                  <div className="min-w-0 flex-1 space-y-1">
                    <p className="text-[12px] leading-snug text-zinc-200">
                      <span className="font-medium text-zinc-100">Observation: </span>
                      {f.observation}
                    </p>
                    <p className="text-[12px] leading-snug text-zinc-400">
                      <span className="font-medium text-zinc-300">Significance: </span>
                      {f.biological_significance}
                    </p>
                  </div>
                  <ConfidenceBadge confidence={f.confidence} />
                </div>
              </li>
            ))}
          </ul>
        )}
      </ReportSection>

      {/* 3. Dynamic Analysis */}
      <ReportSection number={3} title="Dynamic Analysis" icon={<Microscope className="h-3.5 w-3.5" />}>
        <DynamicAnalysis result={report.simulation_result} />
      </ReportSection>

      {/* 4. Validation */}
      <ReportSection number={4} title="Validation" icon={<ShieldCheck className="h-3.5 w-3.5" />}>
        {validation ? (
          <div className="space-y-3">
            <ValidationMetricsRow report={validation} />
            <ValidationPyramid report={validation} />
          </div>
        ) : (
          <EmptyInline text="No validation report attached — run the validation pipeline to populate this section." />
        )}
      </ReportSection>

      {/* 5. Literature Comparison */}
      <ReportSection number={5} title="Literature Comparison" icon={<BookOpen className="h-3.5 w-3.5" />}>
        <LiteratureComparison benchmarks={benchmarks} />
      </ReportSection>

      {/* 6. Future Experiments */}
      <ReportSection number={6} title="Future Experiments" icon={<Lightbulb className="h-3.5 w-3.5" />}>
        {hypotheses.length === 0 ? (
          <EmptyInline text="No suggested experiments — the Hypothesis Layer has not generated any yet." />
        ) : (
          <div className="grid grid-cols-1 gap-2.5 md:grid-cols-2">
            {hypotheses.map((h, idx) => {
              const design = h.experiment_design;
              if (!design) return null;
              const falsifiability = deriveFalsifiability(h);
              return (
                <ExperimentCard
                  key={h.id ?? h.hypothesis_id ?? idx}
                  experiment={design}
                  falsifiability={falsifiability}
                />
              );
            })}
          </div>
        )}
      </ReportSection>

      {/* Footer */}
      <div className="mt-2 flex items-center justify-between border-t border-zinc-800 pt-3 text-[10px] text-zinc-600">
        <span className="font-mono">Report ID: {report.id}</span>
        <span>BioDynamics v4 · Experiment Report</span>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Inline sub-components
// ---------------------------------------------------------------------------

function SummaryField({
  label,
  value,
  mono,
  valueClass,
}: {
  label: string;
  value: string;
  mono?: boolean;
  valueClass?: string;
}) {
  return (
    <div className="rounded-md border border-zinc-800 bg-zinc-900/40 px-2.5 py-1.5">
      <div className="text-[10px] font-medium uppercase tracking-wide text-zinc-500">
        {label}
      </div>
      <div
        className={cn(
          "mt-0.5 text-[12px] text-zinc-200",
          mono && "font-mono",
          valueClass
        )}
      >
        {value}
      </div>
    </div>
  );
}

function ConfidenceBadge({ confidence }: { confidence: number }) {
  const pct = confidencePct(confidence);
  const cls =
    pct >= 75
      ? "bg-emerald-500/15 text-emerald-300"
      : pct >= 50
        ? "bg-amber-500/15 text-amber-300"
        : "bg-red-500/15 text-red-300";
  return (
    <span
      className={cn(
        "shrink-0 rounded px-1.5 py-0.5 text-[10px] font-semibold",
        cls
      )}
    >
      {pct}% conf
    </span>
  );
}

function EmptyInline({ text }: { text: string }) {
  return (
    <div className="flex items-center gap-2 rounded-md border border-dashed border-zinc-800 bg-zinc-900/30 px-3 py-4 text-[11px] text-zinc-500">
      <AlertTriangle className="h-3 w-3 shrink-0 text-zinc-600" />
      {text}
    </div>
  );
}

/** Compact 3-metric summary row above the full ValidationPyramid. */
function ValidationMetricsRow({ report }: { report: ValidationReport }) {
  const massErr = report.level1?.mass_conservation_error;
  const sbmlScore = report.level2?.similarity_score;
  const benchmarkPass = report.level4?.benchmarks?.filter((b) => b.pass).length ?? 0;
  const benchmarkTotal = report.level4?.benchmarks?.length ?? 0;

  return (
    <div className="grid grid-cols-3 gap-2">
      <MetricTile
        label="Mass Conservation Error"
        value={massErr !== undefined ? `${(massErr * 100).toFixed(2)}%` : "—"}
        good={massErr !== undefined && massErr <= 0.05}
      />
      <MetricTile
        label="SBML Fit Score"
        value={sbmlScore !== undefined ? sbmlScore.toFixed(3) : "—"}
        good={sbmlScore !== undefined && sbmlScore >= 0.6}
      />
      <MetricTile
        label="Benchmark Score"
        value={benchmarkTotal > 0 ? `${benchmarkPass}/${benchmarkTotal}` : "—"}
        good={benchmarkTotal > 0 && benchmarkPass === benchmarkTotal}
      />
    </div>
  );
}

function MetricTile({
  label,
  value,
  good,
}: {
  label: string;
  value: string;
  good?: boolean;
}) {
  return (
    <div className="rounded-md border border-zinc-800 bg-zinc-900/50 px-2.5 py-1.5">
      <div className="text-[10px] font-medium uppercase tracking-wide text-zinc-500">
        {label}
      </div>
      <div
        className={cn(
          "mt-0.5 text-sm font-semibold",
          good === true && "text-emerald-300",
          good === false && "text-red-300",
          good === undefined && "text-zinc-200"
        )}
      >
        {value}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Markdown export
// ---------------------------------------------------------------------------

function generateMarkdown(r: ReportData): string {
  const lines: string[] = [];
  const validation = r.validation_report ?? (r.validation as ValidationReport | undefined);
  const { passed, total } = countPassedLevels(validation);

  lines.push(`# ${r.title || "Experiment Report"}`);
  lines.push("");
  lines.push(`**Pathway:** ${pathwayLabel(r)} (${r.pathway_class})  `);
  lines.push(`**Report ID:** ${r.id}  `);
  lines.push(`**Generated:** ${formatDate(r.simulation_run_at ?? r.created_at)}  `);
  lines.push(`**Validation:** ${passed}/${total} levels passed`);
  lines.push("");
  lines.push("---");
  lines.push("");

  // 1. Executive Summary
  lines.push("## 1. Executive Summary");
  lines.push("");
  lines.push(`**Hypothesis tested:** ${r.hypothesis || "—"}`);
  lines.push("");
  lines.push(`**Key finding:** ${r.key_finding || "—"}`);
  lines.push("");

  // 2. Scientific Findings
  lines.push("## 2. Scientific Findings");
  lines.push("");
  const findings = r.findings ?? [];
  if (findings.length === 0) {
    lines.push("_No structured findings extracted._");
  } else {
    findings.forEach((f, i) => {
      lines.push(`${i + 1}. **Observation:** ${f.observation}`);
      lines.push(`   - **Biological significance:** ${f.biological_significance}`);
      lines.push(`   - **Confidence:** ${confidencePct(f.confidence)}%`);
    });
  }
  lines.push("");

  // 3. Dynamic Analysis
  lines.push("## 3. Dynamic Analysis");
  lines.push("");
  const sim = r.simulation_result;
  if (sim && sim.time_points?.length && sim.species) {
    lines.push(`Time course: ${sim.time_points.length} points, ${Object.keys(sim.species).length} species.`);
    lines.push("");
    // Steady-state table
    lines.push("### Steady-State Values");
    lines.push("");
    lines.push("| Species | Steady-state |");
    lines.push("|---------|-------------|");
    for (const [name, series] of Object.entries(sim.species)) {
      const tailLen = Math.max(1, Math.floor(series.length * 0.1));
      const tail = series.slice(-tailLen);
      const val = tail.reduce((a, b) => a + b, 0) / tail.length;
      lines.push(`| ${name} | ${val.toFixed(4)} |`);
    }
    lines.push("");
  } else {
    lines.push("_No simulation result available._");
    lines.push("");
  }

  // 4. Validation
  lines.push("## 4. Validation");
  lines.push("");
  if (validation) {
    lines.push(`Overall: ${validation.overall_pass ? "PASS" : "FAIL"} (${passed}/${total} levels passed)`);
    lines.push("");
    if (validation.level1?.mass_conservation_error !== undefined) {
      lines.push(`- Mass conservation error: ${(validation.level1.mass_conservation_error * 100).toFixed(2)}%`);
    }
    if (validation.level2?.similarity_score !== undefined) {
      lines.push(`- SBML fit score: ${validation.level2.similarity_score.toFixed(3)}`);
    }
    if (validation.level4?.benchmarks?.length) {
      const bp = validation.level4.benchmarks.filter((b) => b.pass).length;
      lines.push(`- Benchmarks: ${bp}/${validation.level4.benchmarks.length} passed`);
    }
  } else {
    lines.push("_No validation report attached._");
  }
  lines.push("");

  // 5. Literature Comparison
  lines.push("## 5. Literature Comparison");
  lines.push("");
  const benchmarks = r.benchmarks ?? [];
  if (benchmarks.length === 0) {
    lines.push("_No literature benchmarks matched._");
  } else {
    lines.push("| Metric | Simulated | Literature | PMID | Match |");
    lines.push("|--------|-----------|-----------|------|-------|");
    for (const bm of benchmarks) {
      lines.push(
        `| ${bm.metric} | ${bm.simulated_value} | ${bm.literature_value} | ${bm.source_pmid ?? "—"} | ${bm.match} |`
      );
    }
  }
  lines.push("");

  // 6. Future Experiments
  lines.push("## 6. Future Experiments");
  lines.push("");
  const hypotheses = r.hypotheses ?? [];
  if (hypotheses.length === 0) {
    lines.push("_No suggested experiments._");
  } else {
    hypotheses.forEach((h, i) => {
      const d = h.experiment_design;
      if (!d) return;
      const f = deriveFalsifiability(h);
      lines.push(`${i + 1}. **Cell line:** ${d.cell_line ?? "—"}`);
      lines.push(`   - **Treatment:** ${d.perturbation?.agent ?? "—"} (${d.perturbation?.type ?? "—"})`);
      lines.push(`   - **Measurement:** ${d.readout?.species ?? "—"} (${d.readout?.metric ?? "—"})`);
      lines.push(`   - **Timepoint:** ${(d.time_points ?? []).join(", ") || "—"}`);
      lines.push(`   - **Control:** ${(d.controls ?? []).join(", ") || "—"}`);
      lines.push(`   - **Expected outcome:** ${d.expected_result ?? "—"}`);
      lines.push(`   - **Falsifiable:** ${f.overall ? "yes" : "no"}`);
    });
  }
  lines.push("");
  lines.push("---");
  lines.push("");
  lines.push("*BioDynamics v4 — Experiment Report*");

  return lines.join("\n");
}
