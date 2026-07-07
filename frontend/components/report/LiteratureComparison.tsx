"use client";

import { Check, X, AlertCircle, BookOpen } from "lucide-react";
import { cn } from "@/lib/utils";
import { formatNumber } from "@/components/simulation/shared";

/** A single row comparing a simulated metric against a literature benchmark. */
export interface BenchmarkComparison {
  metric: string;
  simulated_value: number;
  literature_value: number;
  source_pmid?: string;
  match: "yes" | "no" | "partial";
  notes?: string;
}

interface LiteratureComparisonProps {
  benchmarks: BenchmarkComparison[];
  className?: string;
}

/**
 * Literature Comparison — Section 5 of the Experiment Report.
 *
 * Tabular comparison of simulation results against published literature
 * benchmarks. Each row carries the metric name, the simulated value, the
 * literature reference value, the source PMID, and a match verdict
 * (yes / partial / no) rendered as a colour-coded badge.
 */
export function LiteratureComparison({
  benchmarks,
  className,
}: LiteratureComparisonProps) {
  if (!benchmarks.length) {
    return (
      <div
        className={cn(
          "flex flex-col items-center justify-center gap-2 rounded-md border border-dashed border-zinc-800 py-10 text-center",
          className
        )}
      >
        <BookOpen className="h-5 w-5 text-zinc-700" />
        <p className="text-xs text-zinc-500">
          No literature benchmarks matched this simulation.
        </p>
        <p className="text-[10px] text-zinc-600">
          Benchmarks populate once the Level-4 validation runs against curated
          BioModels references.
        </p>
      </div>
    );
  }

  const passed = benchmarks.filter((b) => b.match === "yes").length;

  return (
    <div className={cn("space-y-2", className)}>
      <div className="flex items-center gap-2 text-[11px] text-zinc-400">
        <span>
          {passed}/{benchmarks.length} metrics match literature
        </span>
        <span className="h-1 w-1 rounded-full bg-zinc-700" />
        <span className="text-zinc-500">
          sources cited by PMID
        </span>
      </div>
      <div className="overflow-x-auto rounded-md border border-zinc-800">
        <table className="w-full text-left text-[11px]">
          <thead className="bg-zinc-900/80 text-[10px] uppercase tracking-wide text-zinc-500">
            <tr>
              <th className="px-2.5 py-1.5 font-medium">Metric</th>
              <th className="px-2.5 py-1.5 font-medium">Simulated</th>
              <th className="px-2.5 py-1.5 font-medium">Literature</th>
              <th className="px-2.5 py-1.5 font-medium">Source (PMID)</th>
              <th className="px-2.5 py-1.5 font-medium">Match</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-zinc-800/60">
            {benchmarks.map((bm, idx) => (
              <tr key={`${bm.metric}-${idx}`} className="hover:bg-zinc-800/30">
                <td className="px-2.5 py-1.5 text-zinc-200">{bm.metric}</td>
                <td className="px-2.5 py-1.5 font-mono text-zinc-200">
                  {formatNumber(bm.simulated_value)}
                </td>
                <td className="px-2.5 py-1.5 font-mono text-zinc-300">
                  {formatNumber(bm.literature_value)}
                </td>
                <td className="px-2.5 py-1.5 font-mono text-cyan-400">
                  {bm.source_pmid ? `PMID:${bm.source_pmid}` : "—"}
                </td>
                <td className="px-2.5 py-1.5">
                  <MatchBadge match={bm.match} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function MatchBadge({ match }: { match: "yes" | "no" | "partial" }) {
  const cfg = {
    yes: {
      icon: <Check className="h-3 w-3" />,
      label: "match",
      cls: "bg-emerald-500/15 text-emerald-300",
    },
    partial: {
      icon: <AlertCircle className="h-3 w-3" />,
      label: "partial",
      cls: "bg-amber-500/15 text-amber-300",
    },
    no: {
      icon: <X className="h-3 w-3" />,
      label: "no match",
      cls: "bg-red-500/15 text-red-300",
    },
  }[match];

  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] font-medium",
        cfg.cls
      )}
    >
      {cfg.icon}
      {cfg.label}
    </span>
  );
}
