"use client";

import React from "react";
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

export interface DoseResponseData {
  concentrations: number[];
  effects: number[];
  ic50?: number | null;
  ic90?: number | null;
  hed?: number | null;
  drug_name?: string;
}

interface DoseResponseCurveProps {
  data: DoseResponseData;
  className?: string;
}

function formatScientific(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return "N/A";
  }
  if (value === 0) return "0";
  if (value >= 0.01 && value < 10000) {
    return value.toFixed(3).replace(/\.?0+$/, "");
  }
  return value.toExponential(2);
}

export function DoseResponseCurve({ data, className }: DoseResponseCurveProps) {
  const { concentrations, effects, ic50, ic90, hed, drug_name } = data;

  const chartData = concentrations.map((conc, idx) => ({
    concentration: conc,
    effect: effects[idx] ?? 0,
  }));

  const maxEffect = effects.length > 0 ? Math.max(...effects) : 100;

  return (
    <div className={className}>
      <div className="mb-3 flex items-center gap-2">
        <TrendingUp className="h-4 w-4 text-emerald-400" />
        <h4 className="text-sm font-medium text-zinc-100">
          剂量-反应曲线（Sigmoid Emax）
          {drug_name ? ` — ${drug_name}` : ""}
        </h4>
      </div>

      <div className="mb-3 grid grid-cols-3 gap-2">
        <div className="rounded-lg bg-zinc-800/80 p-2 text-center">
          <div className="text-[10px] uppercase tracking-wide text-zinc-500">IC50 (nM)</div>
          <div className="text-sm font-semibold text-emerald-400">{formatScientific(ic50)}</div>
        </div>
        <div className="rounded-lg bg-zinc-800/80 p-2 text-center">
          <div className="text-[10px] uppercase tracking-wide text-zinc-500">IC90 (nM)</div>
          <div className="text-sm font-semibold text-blue-400">{formatScientific(ic90)}</div>
        </div>
        <div className="rounded-lg bg-zinc-800/80 p-2 text-center">
          <div className="text-[10px] uppercase tracking-wide text-zinc-500">HED</div>
          <div className="text-sm font-semibold text-amber-400">{formatScientific(hed)}</div>
        </div>
      </div>

      <div className="h-64 w-full rounded-lg bg-zinc-800/40 p-2">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#3f3f46" />
            <XAxis
              dataKey="concentration"
              type="number"
              scale="log"
              domain={["auto", "auto"]}
              tick={{ fill: "#a1a1aa", fontSize: 10 }}
              stroke="#52525b"
              label={{
                value: "药物浓度 (nM, log)",
                position: "insideBottomRight",
                offset: -4,
                fill: "#a1a1aa",
                fontSize: 10,
              }}
            />
            <YAxis
              tick={{ fill: "#a1a1aa", fontSize: 10 }}
              stroke="#52525b"
              label={{
                value: "效应 (%)",
                angle: -90,
                position: "insideLeft",
                fill: "#a1a1aa",
                fontSize: 10,
              }}
              domain={[0, Math.max(maxEffect * 1.05, 100)]}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "#18181b",
                border: "1px solid #3f3f46",
                borderRadius: "0.5rem",
                fontSize: "0.75rem",
              }}
              formatter={(value) => [
                typeof value === "number" ? value.toFixed(2) : String(value),
                "效应 (%)",
              ]}
              labelFormatter={(label) =>
                `浓度: ${formatScientific(typeof label === "number" ? label : Number(label))} nM`
              }
            />
            <Legend wrapperStyle={{ fontSize: "10px" }} />
            <Line
              type="monotone"
              dataKey="effect"
              name="Emax 效应"
              stroke="#10b981"
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4 }}
            />
            {ic50 && Number.isFinite(ic50) && (
              <ReferenceDot
                x={ic50}
                y={50}
                r={5}
                fill="#10b981"
                stroke="#10b981"
                label={{ value: "IC50", position: "top", fill: "#10b981", fontSize: 10 }}
              />
            )}
            {ic90 && Number.isFinite(ic90) && (
              <ReferenceDot
                x={ic90}
                y={90}
                r={5}
                fill="#3b82f6"
                stroke="#3b82f6"
                label={{ value: "IC90", position: "top", fill: "#3b82f6", fontSize: 10 }}
              />
            )}
            {hed && Number.isFinite(hed) && (
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
