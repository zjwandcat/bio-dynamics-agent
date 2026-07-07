"use client";

import React from "react";
import { Waves, GitCompare, Table2 } from "lucide-react";
import { cn } from "@/lib/utils";

import { TimeSeriesChart } from "@/components/simulation/TimeSeriesChart";
import { DoseResponseChart } from "@/components/simulation/DoseResponseChart";
import { PhasePortraitChart } from "@/components/simulation/PhasePortraitChart";
import {
  type SimulationResultExtras,
  type SteadyStateItem,
  type OscillationInfo,
  detectOscillation,
  formatNumber,
  inferTimeUnit,
} from "@/components/simulation/shared";

interface DynamicAnalysisProps {
  /** Simulation result payload (time-series + optional analysis extras). */
  result?: SimulationResultExtras;
  className?: string;
}

/**
 * Dynamic Analysis — Section 3 of the Experiment Report.
 *
 * Embeds the simulation panel's Recharts views (Time Series, Dose Response,
 * Phase Portrait) directly in the report document, plus a steady-state values
 * table and an oscillation characterization block (period / amplitude /
 * damping) computed via the shared `detectOscillation` heuristic.
 *
 * Each chart is shown only when its data is available — dose-response and
 * phase-portrait are conditional; the time-series is the always-on backbone.
 */
export function DynamicAnalysis({ result, className }: DynamicAnalysisProps) {
  const timePoints = result?.time_points ?? [];
  const species = result?.species ?? {};
  const doseResponse = result?.dose_response;
  const steadyState = result?.steady_state;
  const oscillation = result?.oscillation;

  // Derive oscillation per species when the backend didn't supply it.
  const oscillationRows: OscillationInfo[] = React.useMemo(() => {
    if (!timePoints.length || !Object.keys(species).length) return [];
    if (oscillation?.length) return oscillation;
    return Object.entries(species).map(([name, series]) => {
      const detected = detectOscillation(series, timePoints);
      return {
        species: name,
        is_oscillating: detected.isOscillating,
        period: detected.period,
        amplitude: detected.amplitude,
        damping_ratio: detected.dampingRatio,
        dominant_frequency: detected.dominantFrequency,
      };
    });
  }, [oscillation, species, timePoints]);

  // Steady-state table rows (derive from tail samples when backend omitted).
  const steadyRows: SteadyStateItem[] = React.useMemo(() => {
    if (steadyState?.length) return steadyState;
    if (!Object.keys(species).length) return [];
    return Object.entries(species).map(([name, series]) => {
      const tailLen = Math.max(1, Math.floor(series.length * 0.1));
      const tail = series.slice(-tailLen);
      const value = tail.reduce((a, b) => a + b, 0) / tail.length;
      return { species: name, value };
    });
  }, [steadyState, species]);

  if (!result || !timePoints.length || !Object.keys(species).length) {
    return (
      <div
        className={cn(
          "flex items-center justify-center rounded-md border border-dashed border-zinc-800 py-10 text-center",
          className
        )}
      >
        <p className="text-xs text-zinc-500">
          No dynamic analysis data — simulation result unavailable for this report.
        </p>
      </div>
    );
  }

  const unit = inferTimeUnit(timePoints);
  const oscillatingSpecies = oscillationRows.filter((o) => o.is_oscillating);

  return (
    <div className={cn("space-y-4", className)}>
      {/* Time Series — always shown */}
      <div className="h-64 rounded-lg border border-zinc-800 bg-zinc-950/40 p-2.5">
        <TimeSeriesChart timePoints={timePoints} species={species} />
      </div>

      {/* Dose Response — conditional */}
      {doseResponse && doseResponse.concentrations?.length ? (
        <div className="h-64 rounded-lg border border-zinc-800 bg-zinc-950/40 p-2.5">
          <DoseResponseChart data={doseResponse} />
        </div>
      ) : null}

      {/* Phase Portrait — conditional on oscillation */}
      {oscillatingSpecies.length > 0 && Object.keys(species).length >= 2 ? (
        <div className="h-64 rounded-lg border border-zinc-800 bg-zinc-950/40 p-2.5">
          <PhasePortraitChart timePoints={timePoints} species={species} />
        </div>
      ) : null}

      {/* Steady-state values table */}
      <div>
        <div className="mb-1.5 flex items-center gap-1.5">
          <Table2 className="h-3 w-3 text-emerald-400" />
          <h4 className="text-xs font-semibold text-zinc-200">
            Steady-State Values
          </h4>
          {!steadyState?.length && (
            <span className="text-[10px] text-zinc-600">
              (estimated from final 10% of samples)
            </span>
          )}
        </div>
        <div className="overflow-hidden rounded-md border border-zinc-800">
          <table className="w-full text-left text-[11px]">
            <thead className="bg-zinc-900/80 text-[10px] uppercase tracking-wide text-zinc-500">
              <tr>
                <th className="px-2.5 py-1.5 font-medium">Species</th>
                <th className="px-2.5 py-1.5 font-medium">Steady-state</th>
                <th className="px-2.5 py-1.5 font-medium">95% CI</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-800/60">
              {steadyRows.map((row) => (
                <tr key={row.species} className="hover:bg-zinc-800/30">
                  <td className="px-2.5 py-1.5 font-mono text-zinc-300">
                    {row.species}
                  </td>
                  <td className="px-2.5 py-1.5 text-zinc-200">
                    {formatNumber(row.value)}
                  </td>
                  <td className="px-2.5 py-1.5 text-zinc-500">
                    {row.ci_low !== undefined && row.ci_high !== undefined
                      ? `[${formatNumber(row.ci_low)}, ${formatNumber(row.ci_high)}]`
                      : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Oscillation analysis — conditional */}
      {oscillatingSpecies.length > 0 ? (
        <div>
          <div className="mb-1.5 flex items-center gap-1.5">
            <Waves className="h-3 w-3 text-emerald-400" />
            <h4 className="text-xs font-semibold text-zinc-200">
              Oscillation Analysis
            </h4>
            <span className="rounded bg-emerald-500/15 px-1.5 py-0.5 text-[9px] font-medium text-emerald-300">
              {oscillatingSpecies.length} oscillating species
            </span>
          </div>
          <div className="overflow-hidden rounded-md border border-zinc-800">
            <table className="w-full text-left text-[11px]">
              <thead className="bg-zinc-900/80 text-[10px] uppercase tracking-wide text-zinc-500">
                <tr>
                  <th className="px-2.5 py-1.5 font-medium">Species</th>
                  <th className="px-2.5 py-1.5 font-medium">Period ({unit})</th>
                  <th className="px-2.5 py-1.5 font-medium">Amplitude</th>
                  <th className="px-2.5 py-1.5 font-medium">Damping ζ</th>
                  <th className="px-2.5 py-1.5 font-medium">Freq (/{unit})</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-800/60">
                {oscillatingSpecies.map((o) => (
                  <tr key={o.species} className="hover:bg-zinc-800/30">
                    <td className="px-2.5 py-1.5 font-mono text-zinc-300">
                      {o.species}
                    </td>
                    <td className="px-2.5 py-1.5 text-zinc-200">
                      {formatNumber(o.period)}
                    </td>
                    <td className="px-2.5 py-1.5 text-zinc-200">
                      {formatNumber(o.amplitude)}
                    </td>
                    <td className="px-2.5 py-1.5 text-zinc-200">
                      {formatNumber(o.damping_ratio)}
                    </td>
                    <td className="px-2.5 py-1.5 text-zinc-200">
                      {formatNumber(o.dominant_frequency)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ) : (
        <div className="flex items-center gap-1.5 rounded-md border border-zinc-800 bg-zinc-900/40 px-2.5 py-1.5 text-[11px] text-zinc-500">
          <GitCompare className="h-3 w-3" />
          No sustained oscillation detected across the simulated time course.
        </div>
      )}
    </div>
  );
}
