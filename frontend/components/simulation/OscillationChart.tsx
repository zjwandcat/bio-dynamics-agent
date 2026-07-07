"use client";

import React from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Waves } from "lucide-react";

import {
  type OscillationInfo,
  buildTimeSeriesRows,
  CHART_COLORS,
  CHART_THEME,
  detectOscillation,
  formatNumber,
  inferTimeUnit,
  TOOLTIP_STYLE,
} from "./shared";

interface OscillationChartProps {
  timePoints: number[];
  species: Record<string, number[]>;
  /** Pre-computed oscillation info per species (optional). */
  oscillation?: OscillationInfo[];
  className?: string;
}

/**
 * Time-series view with oscillation detection overlay.
 *
 * For the selected species the chart annotates the detected period, amplitude
 * and damping ratio, draws reference lines at detected peak times, and shows a
 * compact frequency (DFT) sidebar with the dominant frequency. Detection is a
 * lightweight peak-finding heuristic — see `detectOscillation` in shared.ts.
 */
export function OscillationChart({
  timePoints,
  species,
  oscillation,
  className,
}: OscillationChartProps) {
  const speciesNames = Object.keys(species);
  const [selected, setSelected] = React.useState(speciesNames[0] ?? "");

  React.useEffect(() => {
    if (!speciesNames.includes(selected)) setSelected(speciesNames[0] ?? "");
  }, [speciesNames, selected]);

  if (!timePoints.length || !speciesNames.length) {
    return (
      <div className={className ?? "flex h-full items-center justify-center"}>
        <p className="text-xs text-zinc-500">No oscillation data available.</p>
      </div>
    );
  }

  const unit = inferTimeUnit(timePoints);
  const series = species[selected] ?? [];
  const rows = buildTimeSeriesRows(timePoints, { [selected]: series });

  // Prefer backend-provided characterization; fall back to local detection.
  const provided = oscillation?.find((o) => o.species === selected);
  const detected =
    provided ??
    (series.length >= 8
      ? normalizeOscillation(detectOscillation(series, timePoints), selected)
      : undefined);

  const peaks = detected?.period
    ? findPeakIndices(series).slice(0, 6)
    : [];

  return (
    <div className={className ?? "h-full w-full"}>
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <Waves className="h-3.5 w-3.5 text-emerald-400" />
        <h4 className="text-xs font-medium text-zinc-200">
          Oscillation Analysis
        </h4>
        <select
          value={selected}
          onChange={(e) => setSelected(e.target.value)}
          className="h-6 rounded-md border border-zinc-700 bg-zinc-900 px-1.5 text-[11px] text-zinc-200 outline-none focus:border-zinc-500"
        >
          {speciesNames.map((opt) => (
            <option key={opt} value={opt}>
              {opt}
            </option>
          ))}
        </select>
        {detected?.is_oscillating ? (
          <span className="rounded bg-emerald-500/15 px-1.5 py-0.5 text-[10px] font-medium text-emerald-300">
            oscillating
          </span>
        ) : (
          <span className="rounded bg-zinc-800 px-1.5 py-0.5 text-[10px] text-zinc-400">
            no clear oscillation
          </span>
        )}
      </div>

      <div className="flex h-[calc(100%-1.75rem)] min-h-0 gap-2">
        <div className="min-w-0 flex-1">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={rows} margin={{ top: 8, right: 8, bottom: 8, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={CHART_THEME.grid} />
              <XAxis
                dataKey="time"
                type="number"
                tick={{ fill: CHART_THEME.tick, fontSize: 10 }}
                stroke={CHART_THEME.axis}
                label={{
                  value: `Time (${unit})`,
                  position: "insideBottomRight",
                  offset: -4,
                  fill: CHART_THEME.tick,
                  fontSize: 10,
                }}
              />
              <YAxis
                tick={{ fill: CHART_THEME.tick, fontSize: 10 }}
                stroke={CHART_THEME.axis}
              />
              <Tooltip
                contentStyle={TOOLTIP_STYLE}
                labelFormatter={(label) =>
                  `t = ${typeof label === "number" ? label.toFixed(2) : label} ${unit}`
                }
                formatter={(value) => [
                  typeof value === "number" ? value.toFixed(4) : String(value),
                  selected,
                ]}
              />
              <Legend wrapperStyle={{ fontSize: "10px" }} />
              <Line
                type="monotone"
                dataKey={selected}
                stroke={CHART_COLORS[5]}
                strokeWidth={1.75}
                dot={false}
                activeDot={{ r: 3 }}
                isAnimationActive={false}
              />
              {peaks.map((pi, idx) =>
                timePoints[pi] !== undefined ? (
                  <ReferenceLine
                    key={`peak-${idx}`}
                    x={timePoints[pi]}
                    stroke="#f59e0b"
                    strokeDasharray="2 3"
                    strokeOpacity={0.6}
                  />
                ) : null
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>

        <OscillationSidebar detected={detected} unit={unit} />
      </div>
    </div>
  );
}

function OscillationSidebar({
  detected,
  unit,
}: {
  detected: OscillationInfo | undefined;
  unit: string;
}) {
  return (
    <aside className="w-28 shrink-0 space-y-1.5 rounded-lg bg-zinc-900/60 p-2 ring-1 ring-zinc-800">
      <div className="text-[10px] font-medium uppercase tracking-wide text-zinc-500">
        Metrics
      </div>
      <Metric
        label="Period"
        value={
          detected?.period !== undefined
            ? `${formatNumber(detected.period)} ${unit}`
            : "—"
        }
      />
      <Metric
        label="Amplitude"
        value={detected?.amplitude !== undefined ? formatNumber(detected.amplitude) : "—"}
      />
      <Metric
        label="Damping ζ"
        value={detected?.damping_ratio !== undefined ? formatNumber(detected.damping_ratio) : "—"}
      />
      <div className="!mt-2 border-t border-zinc-800 pt-1.5">
        <div className="text-[10px] font-medium uppercase tracking-wide text-zinc-500">
          Frequency
        </div>
        <Metric
          label="Dominant"
          value={
            detected?.dominant_frequency !== undefined
              ? `${formatNumber(detected.dominant_frequency)} /${unit}`
              : "—"
          }
        />
      </div>
    </aside>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col">
      <span className="text-[9px] uppercase tracking-wide text-zinc-600">
        {label}
      </span>
      <span className="text-[11px] font-medium text-zinc-200">{value}</span>
    </div>
  );
}

/** Local-detection result → normalized `OscillationInfo`. */
function normalizeOscillation(
  result: ReturnType<typeof detectOscillation>,
  species: string
): OscillationInfo {
  return {
    species,
    is_oscillating: result.isOscillating,
    period: result.period,
    amplitude: result.amplitude,
    damping_ratio: result.dampingRatio,
    dominant_frequency: result.dominantFrequency,
  };
}

/** Local maxima indices above the mean (for reference-line overlay). */
function findPeakIndices(values: number[]): number[] {
  const n = values.length;
  if (n < 3) return [];
  const mean = values.reduce((a, b) => a + b, 0) / n;
  const peaks: number[] = [];
  for (let i = 1; i < n - 1; i++) {
    if (values[i] > values[i - 1] && values[i] >= values[i + 1] && values[i] > mean) {
      peaks.push(i);
    }
  }
  return peaks;
}
