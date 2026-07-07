"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { BarChart3 } from "lucide-react";

import {
  type SensitivityExtras,
  CHART_COLORS,
  CHART_THEME,
  TOOLTIP_STYLE,
} from "./shared";

interface SensitivityChartProps {
  data?: SensitivityExtras;
  className?: string;
}

/**
 * Parameter sensitivity index bar chart.
 *
 * Supports both Sobol (S1 first-order + ST total-order, grouped bars) and
 * Morris (mu* single bar) indices. Bars are sorted by the magnitude of the
 * primary index so the most influential parameters surface to the left.
 */
export function SensitivityChart({
  data,
  className,
}: SensitivityChartProps) {
  if (!data || !data.indices?.length) {
    return (
      <div className={className ?? "flex h-full items-center justify-center"}>
        <div className="text-center">
          <BarChart3 className="mx-auto mb-2 h-5 w-5 text-zinc-600" />
          <p className="text-xs text-zinc-500">
            No sensitivity analysis available.
          </p>
          <p className="mt-1 text-[10px] text-zinc-600">
            Sobol / Morris indices appear here when a sensitivity run completes.
          </p>
        </div>
      </div>
    );
  }

  const isSobol = data.method === "sobol";
  // Sort by the primary index magnitude (ST for Sobol, mu* for Morris).
  const sorted = [...data.indices].sort((a, b) => {
    const av = isSobol ? (a.ST ?? a.S1 ?? 0) : a.mu_star ?? 0;
    const bv = isSobol ? (b.ST ?? b.S1 ?? 0) : b.mu_star ?? 0;
    return Math.abs(bv) - Math.abs(av);
  });

  return (
    <div className={className ?? "h-full w-full"}>
      <div className="mb-2 flex items-center gap-2">
        <BarChart3 className="h-3.5 w-3.5 text-emerald-400" />
        <h4 className="text-xs font-medium text-zinc-200">
          Parameter Sensitivity
        </h4>
        <span className="rounded bg-zinc-800 px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-zinc-400">
          {data.method}
        </span>
      </div>
      <div className="h-[calc(100%-1.75rem)] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={sorted}
            margin={{ top: 8, right: 16, bottom: 8, left: 0 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke={CHART_THEME.grid} />
            <XAxis
              dataKey="parameter"
              tick={{ fill: CHART_THEME.tick, fontSize: 9 }}
              stroke={CHART_THEME.axis}
              angle={-30}
              textAnchor="end"
              height={56}
              interval={0}
            />
            <YAxis
              tick={{ fill: CHART_THEME.tick, fontSize: 10 }}
              stroke={CHART_THEME.axis}
              label={{
                value: "Sensitivity index",
                angle: -90,
                position: "insideLeft",
                fill: CHART_THEME.tick,
                fontSize: 10,
              }}
            />
            <Tooltip
              contentStyle={TOOLTIP_STYLE}
              formatter={(value, name) => [
                typeof value === "number" ? value.toFixed(4) : String(value),
                name,
              ]}
            />
            <Legend wrapperStyle={{ fontSize: "10px" }} />
            {isSobol ? (
              <>
                <Bar dataKey="S1" name="S1 (first-order)" fill="#10b981" radius={[3, 3, 0, 0]}>
                  {sorted.map((_, idx) => (
                    <Cell key={idx} fill={CHART_COLORS[0]} />
                  ))}
                </Bar>
                <Bar dataKey="ST" name="ST (total-order)" fill="#3b82f6" radius={[3, 3, 0, 0]}>
                  {sorted.map((_, idx) => (
                    <Cell key={idx} fill={CHART_COLORS[1]} />
                  ))}
                </Bar>
              </>
            ) : (
              <Bar
                dataKey="mu_star"
                name="μ* (Morris)"
                fill="#f59e0b"
                radius={[3, 3, 0, 0]}
              />
            )}
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
