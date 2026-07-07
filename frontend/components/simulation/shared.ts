/**
 * Shared types, chart theme, and helpers for the Simulation Panel charts.
 *
 * All six tabs share the SAME `SimulationResult` from the workbench store.
 * Some analyses (dose-response, sensitivity) are not produced by every run,
 * so the chart components accept `SimulationResultExtras` — a superset of
 * `SimulationResult` with optional analysis payloads. When the backend later
 * enriches the simulation result, the charts pick up the extras automatically.
 */

import type { SimulationResult } from "@/lib/api";

// ---------------------------------------------------------------------------
// Chart theme (dark — matches the existing DoseResponseCurve styling)
// ---------------------------------------------------------------------------

/** Distinct line colors for multi-species charts. */
export const CHART_COLORS = [
  "#10b981", // emerald
  "#3b82f6", // blue
  "#f59e0b", // amber
  "#ef4444", // red
  "#a855f7", // purple
  "#06b6d4", // cyan
  "#ec4899", // pink
  "#84cc16", // lime
  "#f97316", // orange
  "#6366f1", // indigo
];

export const CHART_THEME = {
  grid: "#3f3f46",
  axis: "#52525b",
  tick: "#a1a1aa",
  tooltipBg: "#18181b",
  tooltipBorder: "#3f3f46",
} as const;

/** Shared Recharts tooltip content style for every chart. */
export const TOOLTIP_STYLE = {
  backgroundColor: CHART_THEME.tooltipBg,
  border: `1px solid ${CHART_THEME.tooltipBorder}`,
  borderRadius: "0.5rem",
  fontSize: "0.75rem",
} as const;

// ---------------------------------------------------------------------------
// Extended simulation types (forward-compatible analysis payloads)
// ---------------------------------------------------------------------------

/** Dose-response curve payload (mirrors the existing chat DoseResponseData). */
export interface DoseResponseExtras {
  concentrations: number[];
  effects: number[];
  ic50?: number | null;
  ec50?: number | null;
  ic90?: number | null;
  hed?: number | null;
  drug_name?: string;
}

/** A single parameter sensitivity index row. */
export interface SensitivityItem {
  parameter: string;
  /** Sobol first-order index. */
  S1?: number;
  /** Sobol total-order index. */
  ST?: number;
  /** Morris mean of absolute elementary effects. */
  mu_star?: number;
}

/** Sensitivity analysis payload. */
export interface SensitivityExtras {
  method: "sobol" | "morris" | "local";
  indices: SensitivityItem[];
}

/** Steady-state value for a single species (with optional confidence band). */
export interface SteadyStateItem {
  species: string;
  value: number;
  ci_low?: number;
  ci_high?: number;
}

/** Oscillation characterization for a single species. */
export interface OscillationInfo {
  species: string;
  is_oscillating: boolean;
  period?: number;
  amplitude?: number;
  damping_ratio?: number;
  dominant_frequency?: number;
}

/**
 * `SimulationResult` extended with the optional analysis payloads the charts
 * can render. The store owns a plain `SimulationResult`; charts cast to this
 * superset and gracefully no-op when a payload is absent.
 */
export interface SimulationResultExtras extends SimulationResult {
  dose_response?: DoseResponseExtras;
  sensitivity?: SensitivityExtras;
  steady_state?: SteadyStateItem[];
  oscillation?: OscillationInfo[];
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Format a number in scientific notation for compact axis/label display. */
export function formatScientific(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return "N/A";
  }
  if (value === 0) return "0";
  const abs = Math.abs(value);
  if (abs >= 0.01 && abs < 10000) {
    return value.toFixed(3).replace(/\.?0+$/, "");
  }
  return value.toExponential(2);
}

/** Format a plain numeric value with up to 4 significant digits. */
export function formatNumber(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return "N/A";
  }
  if (value === 0) return "0";
  const abs = Math.abs(value);
  if (abs >= 1000 || abs < 0.001) return value.toExponential(2);
  return value.toFixed(4).replace(/\.?0+$/, "");
}

/** Infer a stable time-unit label from the time-points range (minutes default). */
export function inferTimeUnit(timePoints: number[]): string {
  if (!timePoints.length) return "min";
  const span = timePoints[timePoints.length - 1] - timePoints[0];
  if (span >= 1440) return "min";
  return "min";
}

/**
 * Detect dominant period / amplitude / damping for an oscillating series via
 * simple peak detection. Returns null when the series does not oscillate.
 *
 * This is a lightweight, dependency-free heuristic suitable for the panel's
 * at-a-glance annotations — not a substitute for a full spectral analysis.
 */
export function detectOscillation(
  values: number[],
  timePoints: number[]
): {
  isOscillating: boolean;
  period?: number;
  amplitude?: number;
  dampingRatio?: number;
  dominantFrequency?: number;
} {
  const n = values.length;
  if (n < 8 || timePoints.length < n) return { isOscillating: false };

  // Peak detection: local maxima above the mean.
  const mean = values.reduce((a, b) => a + b, 0) / n;
  const peaks: number[] = [];
  for (let i = 1; i < n - 1; i++) {
    if (values[i] > values[i - 1] && values[i] >= values[i + 1] && values[i] > mean) {
      peaks.push(i);
    }
  }
  if (peaks.length < 2) return { isOscillating: false };

  // Average inter-peak period.
  const periodSamples: number[] = [];
  for (let i = 1; i < peaks.length; i++) {
    periodSamples.push(timePoints[peaks[i]] - timePoints[peaks[i - 1]]);
  }
  const period =
    periodSamples.reduce((a, b) => a + b, 0) / periodSamples.length;

  // Amplitude: mean peak-to-trough swing around the mean.
  const troughs: number[] = [];
  for (let i = 1; i < n - 1; i++) {
    if (values[i] < values[i - 1] && values[i] <= values[i + 1] && values[i] < mean) {
      troughs.push(values[i]);
    }
  }
  const peakVals = peaks.map((i) => values[i]);
  const ampPeaks =
    peakVals.reduce((a, b) => a + b, 0) / (peakVals.length || 1);
  const ampTroughs =
    troughs.length > 0
      ? troughs.reduce((a, b) => a + b, 0) / troughs.length
      : mean;
  const amplitude = (ampPeaks - ampTroughs) / 2;

  // Damping ratio estimate from successive peak amplitudes (log-decrement).
  let dampingRatio: number | undefined;
  if (peakVals.length >= 2) {
    const a1 = peakVals[0] - mean;
    const a2 = peakVals[peakVals.length - 1] - mean;
    if (a1 > 0 && a2 > 0 && a1 !== a2) {
      const delta = Math.log(a1 / a2) / (peakVals.length - 1);
      const zeta = delta / Math.sqrt(4 * Math.PI ** 2 + delta ** 2);
      if (Number.isFinite(zeta) && zeta > 0 && zeta < 2) dampingRatio = zeta;
    }
  }

  // Dominant frequency via a naive DFT over the de-meaned signal.
  const dominantFrequency = naiveDominantFrequency(values, timePoints);

  // Require a non-trivial swing relative to the mean to call it oscillation.
  const isOscillating =
    amplitude > 1e-9 &&
    Math.abs(mean) > 0 &&
    amplitude / (Math.abs(mean) || 1) > 0.01;

  return {
    isOscillating,
    period: isOscillating ? period : undefined,
    amplitude: isOscillating ? amplitude : undefined,
    dampingRatio,
    dominantFrequency,
  };
}

/** Naive DFT returning the dominant (non-DC) frequency in cycles per time-unit. */
function naiveDominantFrequency(
  values: number[],
  timePoints: number[]
): number | undefined {
  const n = values.length;
  if (n < 8) return undefined;
  const mean = values.reduce((a, b) => a + b, 0) / n;
  const span = timePoints[n - 1] - timePoints[0];
  if (span <= 0) return undefined;

  let bestMag = 0;
  let bestK = 0;
  // Scan the first half of the spectrum (Nyquist).
  const maxK = Math.min(n / 2, 64);
  for (let k = 1; k <= maxK; k++) {
    let re = 0;
    let im = 0;
    for (let t = 0; t < n; t++) {
      const angle = (-2 * Math.PI * k * t) / n;
      re += (values[t] - mean) * Math.cos(angle);
      im += (values[t] - mean) * Math.sin(angle);
    }
    const mag = Math.sqrt(re * re + im * im);
    if (mag > bestMag) {
      bestMag = mag;
      bestK = k;
    }
  }
  if (bestK === 0) return undefined;
  return bestK / span;
}

/** Build a `{ time, ...species }` row array for Recharts time-series charts. */
export function buildTimeSeriesRows(
  timePoints: number[],
  species: Record<string, number[]>
): Array<Record<string, number>> {
  return timePoints.map((t, i) => {
    const row: Record<string, number> = { time: t };
    for (const [name, series] of Object.entries(species)) {
      row[name] = series[i] ?? 0;
    }
    return row;
  });
}
