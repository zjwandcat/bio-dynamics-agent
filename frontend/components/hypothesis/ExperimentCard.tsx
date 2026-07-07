"use client";

import React, { useState } from "react";
import { FlaskConical, Copy, Check } from "lucide-react";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import type {
  ExperimentDesign,
  FalsifiabilityCheck,
  Perturbation,
  Readout,
} from "./types";

interface ExperimentCardProps {
  experiment: ExperimentDesign;
  falsifiability?: FalsifiabilityCheck;
  className?: string;
}

/**
 * Suggested experiment card — renders the ExperimentDesigner 6-field schema:
 * `cell_line`, `treatment` (perturbation), `measurement` (readout), `timepoint`
 * (time_points), `control` (controls), `expected_outcome` (expected_result).
 *
 * The "Export" button copies the design to the clipboard as JSON so it can be
 * handed off to a wet-lab protocol or ELN.
 */
export function ExperimentCard({
  experiment,
  falsifiability,
  className,
}: ExperimentCardProps) {
  const [copied, setCopied] = useState(false);

  const perturbation = experiment.perturbation ?? {};
  const readout = experiment.readout ?? {};
  const timePoints = experiment.time_points ?? [];
  const controls = experiment.controls ?? [];
  const cellLine = experiment.cell_line ?? "";
  const expectedResult = experiment.expected_result ?? "";

  const handleExport = async () => {
    const payload = {
      cell_line: cellLine || null,
      treatment: {
        type: perturbation.type ?? null,
        agent: perturbation.agent ?? null,
        target: perturbation.target ?? null,
        dose: perturbation.dose ?? null,
        duration: perturbation.duration ?? null,
        mechanism: perturbation.mechanism ?? null,
      },
      measurement: {
        species: readout.species ?? null,
        metric: readout.metric ?? null,
        threshold: readout.threshold ?? null,
      },
      timepoint: timePoints,
      control: controls,
      expected_outcome: expectedResult || null,
    };
    try {
      await navigator.clipboard.writeText(JSON.stringify(payload, null, 2));
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard unavailable — silent no-op */
    }
  };

  return (
    <div
      className={cn(
        "rounded-md border border-zinc-700/60 bg-zinc-800/40 p-2.5",
        className
      )}
    >
      <div className="mb-2 flex items-center justify-between">
        <div className="flex items-center gap-1.5">
          <FlaskConical className="h-3 w-3 text-emerald-400" />
          <span className="text-[11px] font-semibold text-zinc-200">
            Suggested Experiment
          </span>
        </div>
        <div className="flex items-center gap-1.5">
          {falsifiability && (
            <Badge
              variant="outline"
              className={cn(
                "h-4 text-[9px]",
                falsifiability.overall
                  ? "border-emerald-700/50 text-emerald-300"
                  : "border-red-800/50 text-red-300"
              )}
            >
              {falsifiability.overall ? "falsifiable" : "not falsifiable"}
            </Badge>
          )}
          <Button
            variant="ghost"
            size="icon-xs"
            onClick={handleExport}
            title="Copy experiment as JSON"
          >
            {copied ? (
              <Check className="h-3 w-3 text-emerald-400" />
            ) : (
              <Copy className="h-3 w-3 text-zinc-400" />
            )}
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-1.5">
        <Field label="Cell line" value={cellLine || "—"} />
        <Field label="Treatment" value={formatTreatment(perturbation)} />
        <Field label="Measurement" value={formatMeasurement(readout)} />
        <Field
          label="Timepoint"
          value={timePoints.length > 0 ? `${timePoints.join(", ")} min` : "—"}
        />
        <Field
          label="Control"
          value={controls.length > 0 ? controls.join(", ") : "—"}
        />
      </div>

      <div className="mt-2 border-t border-zinc-700/50 pt-1.5">
        <div className="text-[10px] uppercase tracking-wide text-zinc-500">
          Expected outcome
        </div>
        <p className="mt-0.5 text-[11px] leading-snug text-zinc-300">
          {expectedResult || "—"}
        </p>
      </div>
    </div>
  );
}

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-start gap-2 text-[10px]">
      <span className="w-20 shrink-0 text-zinc-500">{label}</span>
      <span className="min-w-0 flex-1 break-words text-zinc-300">{value}</span>
    </div>
  );
}

function formatTreatment(p: Perturbation): string {
  const parts: string[] = [];
  if (p.agent) parts.push(p.agent);
  if (p.type) parts.push(`(${p.type})`);
  if (p.dose) parts.push(`· ${p.dose}`);
  if (p.duration) parts.push(`· ${p.duration}`);
  return parts.length > 0 ? parts.join(" ") : "—";
}

function formatMeasurement(r: Readout): string {
  const parts: string[] = [];
  if (r.species) parts.push(r.species);
  if (r.metric) parts.push(`(${r.metric})`);
  if (typeof r.threshold === "number" && r.threshold > 0) {
    const pct = r.threshold <= 1 ? r.threshold * 100 : r.threshold;
    parts.push(`· threshold ${pct.toFixed(1)}%`);
  }
  return parts.length > 0 ? parts.join(" ") : "—";
}
