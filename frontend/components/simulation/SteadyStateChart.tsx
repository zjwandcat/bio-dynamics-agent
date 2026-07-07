"use client";

import React from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ErrorBar,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Equal } from "lucide-react";

import {
  type SteadyStateItem,
  CHART_COLORS,
  CHART_THEME,
  formatNumber,
  TOOLTIP_STYLE,
} from "./shared";

interface SteadyStateChartProps {
  /** Explicit steady-state items; if omitted, derived from the last sample. */
  items?: SteadyStateItem[];
  timePoints: number[];
  species: Record<string, number[]>;
  className?: string;
}

/**
 * Horizontal bar chart of steady-state concentrations per species.
 *
 * Bars are sorted descending by concentration so the dominant species rise to
 * the top. Optional confidence intervals render as error bars when the backend
 * supplies them (e.g. from calibration); otherwise the last sampled value of
 * each species is used as the steady-state estimate.
 */
export function SteadyStateChart({
  items,
  timePoints,
  species,
  className,
}: SteadyStateChartProps) {
  const data: SteadyStateItem[] = React.useMemo(() => {
    if (items?.length) return items;
    // Derive: take the mean of the last 10% of samples as a robust steady state.
    const derived: SteadyStateItem[] = Object.entries(species).map(
      ([name, series]) => {
        const tailLen = Math.max(1, Math.floor(series.length * 0.1));
        const tail = series.slice(-tailLen);
        const value = tail.reduce((a, b) => a + b, 0) / tail.length;
        return { species: name, value };
      }
    );
    return derived;
  }, [items, species]);

  if (!timePoints.length || !data.length) {
    return (
      <div className={className ?? "flex h-full items-center justify-center"}>
        <p className="text-xs text-zinc-500">No steady-state data available.</p>
      </div>
    );
  }

  const sorted = [...data].sort((a, b) => b.value - a.value);
  const chartData = sorted.map((d) => ({
    species: d.species,
    value: d.value,
    errorRange:
      d.ci_low !== undefined && d.ci_high !== undefined
        ? [d.value - d.ci_low, d.ci_high - d.value]
        : undefined,
  }));

  return (
    <div className={className ?? "h-full w-full"}>
      <div className="mb-2 flex items-center gap-2">
        <Equal className="h-3.5 w-3.5 text-emerald-400" />
        <h4 className="text-xs font-medium text-zinc-200">
          Steady-State Concentrations
        </h4>
        {!items?.length && (
          <span className="text-[10px] text-zinc-600">
            (estimated from final samples)
          </span>
        )}
      </div>
      <div className="h-[calc(100%-1.75rem)] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            layout="vertical"
            data={chartData}
            margin={{ top: 8, right: 16, bottom: 8, left: 8 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke={CHART_THEME.grid} />
            <XAxis
              type="number"
              tick={{ fill: CHART_THEME.tick, fontSize: 10 }}
              stroke={CHART_THEME.axis}
              label={{
                value: "Concentration",
                position: "insideBottomRight",
                offset: -4,
                fill: CHART_THEME.tick,
                fontSize: 10,
              }}
            />
            <YAxis
              type="category"
              dataKey="species"
              tick={{ fill: CHART_THEME.tick, fontSize: 10 }}
              stroke={CHART_THEME.axis}
              width={72}
            />
            <Tooltip
              contentStyle={TOOLTIP_STYLE}
              formatter={(value) => [
                typeof value === "number" ? formatNumber(value) : String(value),
                "Concentration",
              ]}
            />
            <Bar
              dataKey="value"
              name="Steady state"
              radius={[0, 3, 3, 0]}
              isAnimationActive={false}
            >
              {chartData.map((_, idx) => (
                <Cell key={idx} fill={CHART_COLORS[idx % CHART_COLORS.length]} />
              ))}
              {chartData.some((d) => d.errorRange) && (
                <ErrorBar
                  dataKey="errorRange"
                  width={4}
                  strokeWidth={1}
                  stroke="#f59e0b"
                />
              )}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
