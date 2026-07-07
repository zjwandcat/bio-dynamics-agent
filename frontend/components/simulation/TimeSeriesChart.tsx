"use client";

import React from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Activity } from "lucide-react";

import {
  buildTimeSeriesRows,
  CHART_COLORS,
  CHART_THEME,
  inferTimeUnit,
  TOOLTIP_STYLE,
} from "./shared";

interface TimeSeriesChartProps {
  timePoints: number[];
  species: Record<string, number[]>;
  className?: string;
}

/**
 * Multi-species concentration-vs-time line chart.
 *
 * The primary view of the Simulation Panel: every species in the result is
 * plotted as its own line over the shared time axis. All six tabs share the
 * same result, so this component is pure — switching tabs does not re-run.
 */
export function TimeSeriesChart({
  timePoints,
  species,
  className,
}: TimeSeriesChartProps) {
  const speciesNames = Object.keys(species);
  const data = React.useMemo(
    () => buildTimeSeriesRows(timePoints, species),
    [timePoints, species]
  );
  const unit = inferTimeUnit(timePoints);

  if (!timePoints.length || !speciesNames.length) {
    return (
      <div className={className ?? "flex h-full items-center justify-center"}>
        <p className="text-xs text-zinc-500">No time-series data available.</p>
      </div>
    );
  }

  return (
    <div className={className ?? "h-full w-full"}>
      <div className="mb-2 flex items-center gap-2">
        <Activity className="h-3.5 w-3.5 text-emerald-400" />
        <h4 className="text-xs font-medium text-zinc-200">
          Concentration vs Time
        </h4>
        <span className="text-[10px] text-zinc-500">
          {speciesNames.length} species · {timePoints.length} points
        </span>
      </div>
      <div className="h-[calc(100%-1.75rem)] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
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
              label={{
                value: "Concentration",
                angle: -90,
                position: "insideLeft",
                fill: CHART_THEME.tick,
                fontSize: 10,
              }}
            />
            <Tooltip
              contentStyle={TOOLTIP_STYLE}
              labelFormatter={(label) =>
                `t = ${typeof label === "number" ? label.toFixed(2) : label} ${unit}`
              }
              formatter={(value, name) => [
                typeof value === "number" ? value.toFixed(4) : String(value),
                name,
              ]}
            />
            <Legend wrapperStyle={{ fontSize: "10px" }} />
            {speciesNames.map((name, idx) => (
              <Line
                key={name}
                type="monotone"
                dataKey={name}
                stroke={CHART_COLORS[idx % CHART_COLORS.length]}
                strokeWidth={1.75}
                dot={false}
                activeDot={{ r: 3 }}
                isAnimationActive={false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
