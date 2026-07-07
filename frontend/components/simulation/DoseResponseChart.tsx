"use client";

import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceDot,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { TrendingUp } from "lucide-react";

import {
  type DoseResponseExtras,
  CHART_THEME,
  formatScientific,
  TOOLTIP_STYLE,
} from "./shared";

interface DoseResponseChartProps {
  data?: DoseResponseExtras;
  className?: string;
}

/**
 * Sigmoidal dose-response curve on a log-scale X axis with IC50/EC50 markers.
 *
 * Adapts the existing chat `DoseResponseCurve` for the Simulation Panel. Dose
 * response is not produced by every run, so when the payload is absent the
 * chart shows an explicit "not available" state rather than an empty plot.
 */
export function DoseResponseChart({
  data,
  className,
}: DoseResponseChartProps) {
  if (!data || !data.concentrations?.length || !data.effects?.length) {
    return (
      <div className={className ?? "flex h-full items-center justify-center"}>
        <div className="text-center">
          <TrendingUp className="mx-auto mb-2 h-5 w-5 text-zinc-600" />
          <p className="text-xs text-zinc-500">
            No dose-response data for this run.
          </p>
          <p className="mt-1 text-[10px] text-zinc-600">
            Run a dose-escalation simulation to populate this view.
          </p>
        </div>
      </div>
    );
  }

  const { concentrations, effects, ic50, ec50, ic90, hed, drug_name } = data;
  const chartData = concentrations.map((conc, idx) => ({
    concentration: conc,
    effect: effects[idx] ?? 0,
  }));
  const maxEffect = effects.length > 0 ? Math.max(...effects) : 100;
  const midpoint = 50;

  return (
    <div className={className ?? "h-full w-full"}>
      <div className="mb-2 flex items-center gap-2">
        <TrendingUp className="h-3.5 w-3.5 text-emerald-400" />
        <h4 className="text-xs font-medium text-zinc-200">
          Dose-Response Curve (Sigmoid Emax)
          {drug_name ? ` — ${drug_name}` : ""}
        </h4>
      </div>

      <div className="mb-2 grid grid-cols-3 gap-1.5">
        <Stat label="IC50" value={ic50} color="text-emerald-400" />
        <Stat label="EC50" value={ec50} color="text-cyan-400" />
        <Stat label="IC90" value={ic90} color="text-blue-400" />
      </div>

      <div className="h-[calc(100%-4.5rem)] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={CHART_THEME.grid} />
            <XAxis
              dataKey="concentration"
              type="number"
              scale="log"
              domain={["auto", "auto"]}
              tick={{ fill: CHART_THEME.tick, fontSize: 10 }}
              stroke={CHART_THEME.axis}
              label={{
                value: "Dose (log)",
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
                value: "Response (%)",
                angle: -90,
                position: "insideLeft",
                fill: CHART_THEME.tick,
                fontSize: 10,
              }}
              domain={[0, Math.max(maxEffect * 1.05, 100)]}
            />
            <Tooltip
              contentStyle={TOOLTIP_STYLE}
              formatter={(value) => [
                typeof value === "number" ? value.toFixed(2) : String(value),
                "Response (%)",
              ]}
              labelFormatter={(label) =>
                `Dose: ${formatScientific(typeof label === "number" ? label : Number(label))}`
              }
            />
            <Legend wrapperStyle={{ fontSize: "10px" }} />
            <Line
              type="monotone"
              dataKey="effect"
              name="Emax response"
              stroke="#10b981"
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4 }}
              isAnimationActive={false}
            />
            {ic50 !== undefined && ic50 && Number.isFinite(ic50) && (
              <ReferenceDot
                x={ic50}
                y={midpoint}
                r={5}
                fill="#10b981"
                stroke="#10b981"
                label={{ value: "IC50", position: "top", fill: "#10b981", fontSize: 10 }}
              />
            )}
            {ec50 !== undefined && ec50 && Number.isFinite(ec50) && (
              <ReferenceDot
                x={ec50}
                y={midpoint}
                r={5}
                fill="#06b6d4"
                stroke="#06b6d4"
                label={{ value: "EC50", position: "bottom", fill: "#06b6d4", fontSize: 10 }}
              />
            )}
            {ic90 !== undefined && ic90 && Number.isFinite(ic90) && (
              <ReferenceDot
                x={ic90}
                y={90}
                r={5}
                fill="#3b82f6"
                stroke="#3b82f6"
                label={{ value: "IC90", position: "top", fill: "#3b82f6", fontSize: 10 }}
              />
            )}
            {hed !== undefined && hed && Number.isFinite(hed) && (
              <ReferenceLine
                x={hed}
                stroke="#f59e0b"
                strokeDasharray="5 5"
                label={{
                  value: "HED",
                  position: "insideTopRight",
                  fill: "#f59e0b",
                  fontSize: 10,
                }}
              />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function Stat({
  label,
  value,
  color,
}: {
  label: string;
  value: number | null | undefined;
  color: string;
}) {
  return (
    <div className="rounded-lg bg-zinc-800/80 p-1.5 text-center">
      <div className="text-[10px] uppercase tracking-wide text-zinc-500">
        {label}
      </div>
      <div className={`text-xs font-semibold ${color}`}>
        {formatScientific(value)}
      </div>
    </div>
  );
}
