"use client";

import React from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceDot,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { GitCompare } from "lucide-react";

import { CHART_COLORS, CHART_THEME, TOOLTIP_STYLE } from "./shared";

interface PhasePortraitChartProps {
  timePoints: number[];
  species: Record<string, number[]>;
  className?: string;
}

/**
 * Phase portrait — species A vs species B trajectory over time.
 *
 * Useful for spotting limit cycles (closed orbits) and attractors. The X and Y
 * species are selectable via dropdowns. Direction of travel is shown with
 * arrowheads sampled along the trajectory; a terminal dot marks the final
 * state so the user can see where the system ends up.
 */
export function PhasePortraitChart({
  timePoints,
  species,
  className,
}: PhasePortraitChartProps) {
  const speciesNames = Object.keys(species);

  const [xKey, setXKey] = React.useState(speciesNames[0] ?? "");
  const [yKey, setYKey] = React.useState(speciesNames[1] ?? speciesNames[0] ?? "");

  // Keep selections valid when the result species set changes.
  React.useEffect(() => {
    if (!speciesNames.includes(xKey)) setXKey(speciesNames[0] ?? "");
    if (!speciesNames.includes(yKey)) setYKey(speciesNames[1] ?? speciesNames[0] ?? "");
  }, [speciesNames, xKey, yKey]);

  if (!timePoints.length || speciesNames.length < 2) {
    return (
      <div className={className ?? "flex h-full items-center justify-center"}>
        <div className="text-center">
          <GitCompare className="mx-auto mb-2 h-5 w-5 text-zinc-600" />
          <p className="text-xs text-zinc-500">
            Phase portrait needs at least 2 species.
          </p>
        </div>
      </div>
    );
  }

  const xSeries = species[xKey] ?? [];
  const ySeries = species[yKey] ?? [];
  const data = timePoints.map((_, i) => ({
    x: xSeries[i] ?? 0,
    y: ySeries[i] ?? 0,
    t: timePoints[i],
  }));

  // Sample arrowhead indices along the trajectory.
  const n = data.length;
  const arrowEvery = Math.max(1, Math.floor(n / 6));
  const arrowIndices = new Set<number>();
  for (let i = arrowEvery; i < n - 1; i += arrowEvery) arrowIndices.add(i);

  const renderDot = (props: {
    cx?: number;
    cy?: number;
    index?: number;
    payload?: { x: number; y: number };
  }) => {
    const { cx, cy, index = 0, payload } = props;
    if (cx === undefined || cy === undefined) return <g />;
    if (!arrowIndices.has(index) || !payload) return <g />;
    // Tangent direction from the previous sample.
    const prev = data[Math.max(0, index - 1)];
    const dx = payload.x - prev.x;
    const dy = payload.y - prev.y;
    const len = Math.hypot(dx, dy);
    if (len === 0) return <g />;
    // Rotate an arrowhead polygon by the tangent angle.
    const angle = (Math.atan2(dy, dx) * 180) / Math.PI;
    return (
      <g transform={`translate(${cx},${cy}) rotate(${angle})`}>
        <polygon
          points="0,0 -6,-3 -6,3"
          fill={CHART_COLORS[2]}
          stroke={CHART_COLORS[2]}
        />
      </g>
    );
  };

  const last = data[n - 1];

  return (
    <div className={className ?? "h-full w-full"}>
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <GitCompare className="h-3.5 w-3.5 text-emerald-400" />
        <h4 className="text-xs font-medium text-zinc-200">Phase Portrait</h4>
        <SpeciesSelect
          label="X"
          value={xKey}
          options={speciesNames}
          onChange={setXKey}
        />
        <span className="text-zinc-600">vs</span>
        <SpeciesSelect
          label="Y"
          value={yKey}
          options={speciesNames}
          onChange={setYKey}
        />
      </div>
      <div className="h-[calc(100%-1.75rem)] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={CHART_THEME.grid} />
            <XAxis
              dataKey="x"
              type="number"
              tick={{ fill: CHART_THEME.tick, fontSize: 10 }}
              stroke={CHART_THEME.axis}
              label={{
                value: xKey,
                position: "insideBottomRight",
                offset: -4,
                fill: CHART_COLORS[1],
                fontSize: 10,
              }}
            />
            <YAxis
              type="number"
              dataKey="y"
              tick={{ fill: CHART_THEME.tick, fontSize: 10 }}
              stroke={CHART_THEME.axis}
              label={{
                value: yKey,
                angle: -90,
                position: "insideLeft",
                fill: CHART_COLORS[0],
                fontSize: 10,
              }}
            />
            <Tooltip
              contentStyle={TOOLTIP_STYLE}
              formatter={(value, name) => [
                typeof value === "number" ? value.toFixed(4) : String(value),
                name === "y" ? yKey : xKey,
              ]}
              labelFormatter={(label) =>
                `x = ${typeof label === "number" ? label.toFixed(4) : label}`
              }
            />
            <Line
              type="monotone"
              dataKey="y"
              stroke={CHART_COLORS[3]}
              strokeWidth={1.5}
              dot={renderDot}
              activeDot={{ r: 3 }}
              isAnimationActive={false}
            />
            {last && (
              <ReferenceDot
                x={last.x}
                y={last.y}
                r={4}
                fill={CHART_COLORS[2]}
                stroke="#fff"
                strokeWidth={1}
                label={{
                  value: "end",
                  position: "top",
                  fill: CHART_COLORS[2],
                  fontSize: 9,
                }}
              />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function SpeciesSelect({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: string[];
  onChange: (v: string) => void;
}) {
  return (
    <label className="flex items-center gap-1 text-[11px] text-zinc-400">
      <span className="font-medium text-zinc-500">{label}</span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="h-6 rounded-md border border-zinc-700 bg-zinc-900 px-1.5 text-[11px] text-zinc-200 outline-none focus:border-zinc-500"
      >
        {options.map((opt) => (
          <option key={opt} value={opt}>
            {opt}
          </option>
        ))}
      </select>
    </label>
  );
}
