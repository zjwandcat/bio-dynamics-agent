"use client";

import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  Activity,
  AlertCircle,
  Check,
  CheckCircle2,
  Dna,
  Loader2,
  Play,
  RefreshCw,
  Sliders,
  TrendingUp,
  XCircle,
} from "lucide-react";
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

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useWorkbenchStore } from "@/lib/store";
import {
  parameterSweep,
  type ParameterSweepParams,
  type PathwayClass,
  type SimulationParams,
  type SimulationResult,
} from "@/lib/api";
import { cn } from "@/lib/utils";

// ---------------------------------------------------------------------------
// Self-contained chart theme + formatter (mirrors the Simulation Panel palette
// so the dose-response preview matches the rest of the workbench without
// coupling this folder to the simulation feature module).
// ---------------------------------------------------------------------------

const CHART = {
  grid: "#3f3f46",
  axis: "#52525b",
  tick: "#a1a1aa",
  tooltipBg: "#18181b",
  tooltipBorder: "#3f3f46",
} as const;

const TOOLTIP_STYLE = {
  backgroundColor: CHART.tooltipBg,
  border: `1px solid ${CHART.tooltipBorder}`,
  borderRadius: "0.5rem",
  fontSize: "0.75rem",
} as const;

function formatVal(value: number | null | undefined): string {
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

// ---------------------------------------------------------------------------
// Parameter catalog & mutation presets
// ---------------------------------------------------------------------------

interface ParamSpec {
  key: string;
  label: string;
  default: number;
  min: number;
  max: number;
  step: number;
  unit: string;
  description: string;
}

const PARAM_CATALOG: ParamSpec[] = [
  {
    key: "k_cat",
    label: "k_cat",
    default: 1.0,
    min: 0.01,
    max: 100,
    step: 0.01,
    unit: "1/s",
    description: "Catalytic turnover rate",
  },
  {
    key: "Km",
    label: "Km",
    default: 10.0,
    min: 0.1,
    max: 1000,
    step: 0.1,
    unit: "µM",
    description: "Michaelis constant",
  },
  {
    key: "k_phos",
    label: "k_phos",
    default: 0.5,
    min: 0.001,
    max: 10,
    step: 0.001,
    unit: "1/s",
    description: "Phosphorylation rate",
  },
  {
    key: "k_dephos",
    label: "k_dephos",
    default: 0.1,
    min: 0.001,
    max: 10,
    step: 0.001,
    unit: "1/s",
    description: "Dephosphorylation rate",
  },
  {
    key: "Vmax",
    label: "Vmax",
    default: 5.0,
    min: 0.01,
    max: 100,
    step: 0.01,
    unit: "µM/s",
    description: "Max reaction velocity",
  },
  {
    key: "k_deg",
    label: "k_deg",
    default: 0.01,
    min: 0.0001,
    max: 1,
    step: 0.0001,
    unit: "1/s",
    description: "Degradation / clearance rate",
  },
];

const BASELINE_PARAMS: Record<string, number> = Object.fromEntries(
  PARAM_CATALOG.map((p) => [p.key, p.default])
);

/** Sweep factors mirror the backend ParameterExplorer defaults (0.1×…10×). */
const SWEEP_FACTORS = [0.1, 0.5, 1.0, 2.0, 10.0];

const DEFAULT_SPECIES = [
  "EGF",
  "EGFR",
  "Ras_GTP",
  "Raf",
  "MEK_p",
  "ERK_p",
  "AKT_p",
  "mTORC1",
];

interface MutationPreset {
  key: string;
  label: string;
  description: string;
  overrides: Record<string, number>;
}

const MUTATION_PRESETS: MutationPreset[] = [
  {
    key: "none",
    label: "None (wild-type)",
    description: "Baseline kinetic parameters",
    overrides: {},
  },
  {
    key: "ras_constitutive",
    label: "Ras* constitutive activation",
    description: "High phosphorylation, near-zero dephosphorylation",
    overrides: { k_phos: 8.0, k_dephos: 0.001 },
  },
  {
    key: "pten_loss",
    label: "PTEN loss-of-function",
    description: "Reduced degradation → pathway hyperactivation",
    overrides: { k_deg: 0.0001 },
  },
  {
    key: "kinase_dead",
    label: "Kinase-dead (K→M)",
    description: "Catalytic activity abolished",
    overrides: { k_cat: 0.001, Vmax: 0.01 },
  },
  {
    key: "efflux_up",
    label: "Drug resistance (efflux ↑)",
    description: "Elevated clearance / degradation",
    overrides: { k_deg: 0.8 },
  },
];

type Mode = "slider" | "ic50" | "knockout" | "overexpression" | "mutation";

const MODES: {
  value: Mode;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
}[] = [
  { value: "slider", label: "Slider", icon: Sliders },
  { value: "ic50", label: "IC50/EC50", icon: Activity },
  { value: "knockout", label: "Knockout", icon: XCircle },
  { value: "overexpression", label: "Overexpr.", icon: TrendingUp },
  { value: "mutation", label: "Mutation", icon: Dna },
];

// ---------------------------------------------------------------------------
// Sweep snapshot
// ---------------------------------------------------------------------------

interface SweepSnapshot {
  parameters: Record<string, number>;
  perturbations: NonNullable<SimulationParams["perturbations"]>;
  sweepParameter: string;
  sweepValues: number[];
}

/** Build a log-spaced dose array for the IC50/EC50 preview curve. */
function logspace(start: number, end: number, points: number): number[] {
  const out: number[] = [];
  const a = Math.log10(start);
  const b = Math.log10(end);
  for (let i = 0; i < points; i++) {
    out.push(Math.pow(10, a + ((b - a) * i) / (points - 1)));
  }
  return out;
}

// ---------------------------------------------------------------------------
// ParameterExplorer
// ---------------------------------------------------------------------------

interface ParameterExplorerProps {
  /** Notified when the modified/baseline status changes (consumed by the wrapper). */
  onStatusChange?: (modified: boolean) => void;
}

/**
 * Parameter Explorer — center-pane parameter editing surface (Task C.5).
 *
 * Five control modes (tabs):
 *   1. Slider      — adjust k_cat / Km / k_phos / k_dephos / Vmax / k_deg.
 *   2. IC50/EC50   — dose-response preview (Hill equation) + sweep trigger.
 *   3. Knockout    — force species concentration to 0.
 *   4. Overexpr.   — multiply a species' baseline by a fold change.
 *   5. Mutation    — preset kinetic mutations + custom parameter overrides.
 *
 * Parameter changes are debounced (500ms) and trigger a `parameterSweep` call
 * against `/api/v4/simulation/sweep`. The sweep response is adapted into the
 * `SimulationResult` shape (`time_points` ← `sweep_values`, `species` ←
 * `response_series`) so the Simulation Tabs re-render with the parameter
 * response curve — a re-simulation without re-running the full pipeline.
 *
 * "Apply Changes" flushes the debounce immediately; "Reset to Baseline"
 * restores defaults and fires one baseline sweep.
 */
export function ParameterExplorer({ onStatusChange }: ParameterExplorerProps) {
  const simulationResult = useWorkbenchStore((s) => s.simulationResult);
  const setSimulationResult = useWorkbenchStore((s) => s.setSimulationResult);
  const currentPathway = useWorkbenchStore((s) => s.currentPathway);

  // ── Working state ──
  const [paramValues, setParamValues] =
    useState<Record<string, number>>(BASELINE_PARAMS);
  const [lastTouched, setLastTouched] = useState<string>("k_cat");
  const [knockoutSet, setKnockoutSet] = useState<Set<string>>(new Set());
  const [overexprMap, setOverexprMap] = useState<Record<string, number>>({});
  const [mutationKey, setMutationKey] = useState<string>("none");
  const [customMutation, setCustomMutation] = useState<Record<string, number>>(
    {}
  );
  const [ic50, setIc50] = useState(1.0);
  const [ec50, setEc50] = useState(1.0);
  const [hillCoef, setHillCoef] = useState(1.0);

  // ── Apply lifecycle ──
  const [dirty, setDirty] = useState(false);
  const [isApplying, setIsApplying] = useState(false);
  const [applyError, setApplyError] = useState("");
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Species list: prefer the current simulation result's species, else defaults.
  const speciesList = useMemo(() => {
    const keys = simulationResult ? Object.keys(simulationResult.species) : [];
    return keys.length > 0 ? keys : DEFAULT_SPECIES;
  }, [simulationResult]);

  // Effective parameters after the active mutation preset + custom overrides.
  const effectiveParams = useMemo(() => {
    const preset = MUTATION_PRESETS.find((m) => m.key === mutationKey);
    return {
      ...paramValues,
      ...(preset?.overrides ?? {}),
      ...customMutation,
    };
  }, [paramValues, mutationKey, customMutation]);

  const isModified = useMemo(() => {
    for (const p of PARAM_CATALOG) {
      if (Math.abs((effectiveParams[p.key] ?? 0) - p.default) > 1e-12) {
        return true;
      }
    }
    if (knockoutSet.size > 0) return true;
    if (Object.keys(overexprMap).length > 0) return true;
    if (mutationKey !== "none") return true;
    return false;
  }, [effectiveParams, knockoutSet, overexprMap, mutationKey]);

  useEffect(() => {
    onStatusChange?.(isModified);
  }, [isModified, onStatusChange]);

  // ── Build a sweep snapshot from current state ──
  const buildSnapshot = useCallback((): SweepSnapshot => {
    const perturbations: NonNullable<SimulationParams["perturbations"]> = [];
    for (const sp of knockoutSet) {
      perturbations.push({ target: sp, kind: "knockout", value: 0 });
    }
    for (const [sp, fold] of Object.entries(overexprMap)) {
      if (fold && fold !== 1) {
        perturbations.push({
          target: sp,
          kind: "overexpression",
          value: fold,
        });
      }
    }
    const sweepParameter = lastTouched || PARAM_CATALOG[0].key;
    const baseVal =
      effectiveParams[sweepParameter] ??
      BASELINE_PARAMS[sweepParameter] ??
      1;
    const sweepValues = SWEEP_FACTORS.map((f) =>
      Number((baseVal * f).toFixed(10))
    );
    return {
      parameters: effectiveParams,
      perturbations,
      sweepParameter,
      sweepValues,
    };
  }, [effectiveParams, knockoutSet, overexprMap, lastTouched]);

  // ── Run the sweep and push the adapted result into the store ──
  const runSweep = useCallback(
    async (snapshot: SweepSnapshot) => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;
      setIsApplying(true);
      setApplyError("");
      try {
        const params: ParameterSweepParams = {
          pathway_class: (currentPathway ?? "egfr") as PathwayClass,
          duration: 120,
          steps: 200,
          parameters: snapshot.parameters,
          perturbations: snapshot.perturbations,
          sweep_parameter: snapshot.sweepParameter,
          sweep_values: snapshot.sweepValues,
        };
        const sweepResult = await parameterSweep(params, controller.signal);
        // Adapt the sweep response into the SimulationResult shape so the
        // Simulation Tabs (which read `simulationResult`) reflect the parameter
        // response curve: time_points ← sweep_values, species ← response_series.
        const adapted: SimulationResult = {
          run_id: sweepResult.run_id,
          pathway_class: (currentPathway ?? "egfr") as PathwayClass,
          time_points: sweepResult.sweep_values,
          species: sweepResult.response_series,
        };
        setSimulationResult(adapted);
      } catch (err) {
        if (controller.signal.aborted) return;
        const msg =
          err instanceof Error ? err.message : "Parameter sweep failed.";
        setApplyError(msg);
      } finally {
        if (abortRef.current === controller) {
          abortRef.current = null;
        }
        setIsApplying(false);
      }
    },
    [currentPathway, setSimulationResult]
  );

  // ── Debounced auto re-simulation (500ms) on any parameter change ──
  useEffect(() => {
    if (!dirty) return;
    if (debounceRef.current) clearTimeout(debounceRef.current);
    const snapshot = buildSnapshot();
    debounceRef.current = setTimeout(() => {
      void runSweep(snapshot);
    }, 500);
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [dirty, buildSnapshot, runSweep]);

  // Cancel any in-flight sweep / pending debounce on unmount.
  useEffect(() => {
    return () => {
      abortRef.current?.abort();
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, []);

  // ── Actions ──
  const markDirty = () => setDirty(true);

  const handleSliderChange = (key: string, value: number) => {
    setParamValues((prev) => ({ ...prev, [key]: value }));
    setLastTouched(key);
    markDirty();
  };

  const handleApply = () => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    void runSweep(buildSnapshot());
  };

  const handleReset = () => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    setParamValues(BASELINE_PARAMS);
    setKnockoutSet(new Set());
    setOverexprMap({});
    setMutationKey("none");
    setCustomMutation({});
    setIc50(1.0);
    setEc50(1.0);
    setHillCoef(1.0);
    setDirty(false);
    setApplyError("");
    // Fire one baseline sweep so the view returns to baseline immediately.
    const sweepParameter = PARAM_CATALOG[0].key;
    const baselineSnapshot: SweepSnapshot = {
      parameters: BASELINE_PARAMS,
      perturbations: [],
      sweepParameter,
      sweepValues: SWEEP_FACTORS.map((f) =>
        Number((BASELINE_PARAMS[sweepParameter] * f).toFixed(10))
      ),
    };
    void runSweep(baselineSnapshot);
  };

  // ── Dose-response preview data (IC50/EC50 mode, computed locally) ──
  const dosePreview = useMemo(() => {
    const concs = logspace(0.001, 1000, 60);
    const n = hillCoef > 0 ? hillCoef : 1;
    const inhib = concs.map(
      (c) => 100 / (1 + Math.pow(c / (ic50 || 1), n))
    );
    const activ = concs.map(
      (c) => (100 * Math.pow(c, n)) / (Math.pow(ec50 || 1, n) + Math.pow(c, n))
    );
    return concs.map((c, i) => ({
      concentration: c,
      inhibition: inhib[i],
      activation: activ[i],
    }));
  }, [ic50, ec50, hillCoef]);

  const currentPerturbations = buildSnapshot().perturbations;

  // ---------------------------------------------------------------------------

  return (
    <div className="flex h-full min-h-0 flex-col">
      <Tabs
        defaultValue="slider"
        className="flex min-h-0 flex-1 flex-col gap-2"
      >
        <TabsList className="shrink-0">
          {MODES.map((m) => {
            const Icon = m.icon;
            return (
              <TabsTrigger key={m.value} value={m.value}>
                <Icon className="h-3 w-3" />
                <span className="truncate">{m.label}</span>
              </TabsTrigger>
            );
          })}
        </TabsList>

        <div className="min-h-0 flex-1 overflow-hidden">
          <TabsContent
            value="slider"
            className="h-full overflow-auto rounded-lg bg-zinc-900/40 p-2 ring-1 ring-zinc-800"
          >
            <SliderMode
              values={paramValues}
              onChange={handleSliderChange}
            />
          </TabsContent>
          <TabsContent
            value="ic50"
            className="h-full overflow-auto rounded-lg bg-zinc-900/40 p-2 ring-1 ring-zinc-800"
          >
            <IC50Mode
              ic50={ic50}
              ec50={ec50}
              hill={hillCoef}
              data={dosePreview}
              onIc50={(v) => {
                setIc50(v);
                markDirty();
              }}
              onEc50={(v) => {
                setEc50(v);
                markDirty();
              }}
              onHill={(v) => {
                setHillCoef(v);
                markDirty();
              }}
            />
          </TabsContent>
          <TabsContent
            value="knockout"
            className="h-full overflow-auto rounded-lg bg-zinc-900/40 p-2 ring-1 ring-zinc-800"
          >
            <KnockoutMode
              species={speciesList}
              selected={knockoutSet}
              onToggle={(sp) => {
                setKnockoutSet((prev) => {
                  const next = new Set(prev);
                  if (next.has(sp)) next.delete(sp);
                  else next.add(sp);
                  return next;
                });
                markDirty();
              }}
            />
          </TabsContent>
          <TabsContent
            value="overexpression"
            className="h-full overflow-auto rounded-lg bg-zinc-900/40 p-2 ring-1 ring-zinc-800"
          >
            <OverexpressionMode
              species={speciesList}
              folds={overexprMap}
              onChange={(sp, fold) => {
                setOverexprMap((prev) => {
                  const next = { ...prev };
                  if (fold === 1) delete next[sp];
                  else next[sp] = fold;
                  return next;
                });
                markDirty();
              }}
            />
          </TabsContent>
          <TabsContent
            value="mutation"
            className="h-full overflow-auto rounded-lg bg-zinc-900/40 p-2 ring-1 ring-zinc-800"
          >
            <MutationMode
              current={mutationKey}
              custom={customMutation}
              onSelect={(key) => {
                setMutationKey(key);
                markDirty();
              }}
              onCustom={(key, val) => {
                setCustomMutation((prev) => ({ ...prev, [key]: val }));
                markDirty();
              }}
              onClearCustom={(key) => {
                setCustomMutation((prev) => {
                  const next = { ...prev };
                  delete next[key];
                  return next;
                });
                markDirty();
              }}
            />
          </TabsContent>
        </div>
      </Tabs>

      {/* Footer: current vs baseline comparison + actions */}
      <div className="mt-2 shrink-0 space-y-2 border-t border-zinc-800 pt-2">
        <ComparisonStrip
          params={effectiveParams}
          perturbations={currentPerturbations}
          knockoutCount={knockoutSet.size}
          overexprCount={Object.keys(overexprMap).length}
          mutationKey={mutationKey}
        />
        {applyError && (
          <div className="flex items-center gap-1.5 rounded-md bg-red-500/10 px-2 py-1 text-[10px] text-red-300">
            <AlertCircle className="h-3 w-3 shrink-0" />
            <span className="truncate">{applyError}</span>
          </div>
        )}
        <div className="flex items-center gap-1.5">
          <Button size="sm" onClick={handleApply} disabled={isApplying}>
            {isApplying ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <Play className="h-3.5 w-3.5" />
            )}
            Apply Changes
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={handleReset}
            disabled={isApplying}
          >
            <RefreshCw className="h-3.5 w-3.5" />
            Reset to Baseline
          </Button>
          {isApplying && (
            <span className="text-[10px] text-zinc-500">re-simulating…</span>
          )}
        </div>
      </div>
    </div>
  );
}

// ===========================================================================
// Slider mode
// ===========================================================================

function SliderMode({
  values,
  onChange,
}: {
  values: Record<string, number>;
  onChange: (key: string, value: number) => void;
}) {
  return (
    <div className="space-y-2.5">
      {PARAM_CATALOG.map((p) => {
        const val = values[p.key] ?? p.default;
        const changed = Math.abs(val - p.default) > 1e-12;
        return (
          <div
            key={p.key}
            className="rounded-md border border-zinc-800 bg-zinc-950/40 px-2.5 py-2"
          >
            <div className="flex items-center justify-between gap-2">
              <div className="flex min-w-0 items-center gap-1.5">
                <span className="text-xs font-medium text-zinc-200">
                  {p.label}
                </span>
                <span className="text-[10px] text-zinc-500">{p.unit}</span>
                {changed && (
                  <Badge
                    variant="outline"
                    className="h-3.5 border-amber-600/50 px-1 text-[9px] text-amber-300"
                  >
                    modified
                  </Badge>
                )}
              </div>
              <span className="font-mono text-[11px] text-emerald-300">
                {formatVal(val)}
              </span>
            </div>
            <input
              type="range"
              min={p.min}
              max={p.max}
              step={p.step}
              value={val}
              onChange={(e) => onChange(p.key, Number(e.target.value))}
              className="mt-1.5 h-1.5 w-full cursor-pointer accent-emerald-400"
            />
            <div className="mt-1 flex items-center justify-between gap-2 text-[10px] text-zinc-500">
              <span>min {formatVal(p.min)}</span>
              <span className="truncate text-zinc-600">{p.description}</span>
              <span>max {formatVal(p.max)}</span>
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ===========================================================================
// IC50 / EC50 mode
// ===========================================================================

function IC50Mode({
  ic50,
  ec50,
  hill,
  data,
  onIc50,
  onEc50,
  onHill,
}: {
  ic50: number;
  ec50: number;
  hill: number;
  data: Array<{ concentration: number; inhibition: number; activation: number }>;
  onIc50: (v: number) => void;
  onEc50: (v: number) => void;
  onHill: (v: number) => void;
}) {
  return (
    <div className="flex h-full min-h-0 flex-col gap-2">
      <div className="grid shrink-0 grid-cols-3 gap-1.5">
        <NumField label="IC50 (µM)" value={ic50} onChange={onIc50} step={0.01} />
        <NumField label="EC50 (µM)" value={ec50} onChange={onEc50} step={0.01} />
        <NumField
          label="Hill (γ)"
          value={hill}
          onChange={onHill}
          step={0.1}
          min={0.1}
        />
      </div>
      <div className="min-h-[140px] flex-1">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={data}
            margin={{ top: 8, right: 12, bottom: 8, left: 0 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke={CHART.grid} />
            <XAxis
              dataKey="concentration"
              type="number"
              scale="log"
              domain={["auto", "auto"]}
              tick={{ fill: CHART.tick, fontSize: 10 }}
              stroke={CHART.axis}
            />
            <YAxis
              tick={{ fill: CHART.tick, fontSize: 10 }}
              stroke={CHART.axis}
              domain={[0, 100]}
            />
            <Tooltip
              contentStyle={TOOLTIP_STYLE}
              formatter={(value) => [
                typeof value === "number" ? value.toFixed(2) : String(value),
                "%",
              ]}
              labelFormatter={(label) =>
                `Dose: ${formatVal(
                  typeof label === "number" ? label : Number(label)
                )}`
              }
            />
            <Line
              type="monotone"
              dataKey="inhibition"
              name="Inhibition (%)"
              stroke="#ef4444"
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="activation"
              name="Activation (%)"
              stroke="#10b981"
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
            {ic50 > 0 && Number.isFinite(ic50) && (
              <ReferenceDot
                x={ic50}
                y={50}
                r={4}
                fill="#ef4444"
                stroke="#ef4444"
                label={{
                  value: "IC50",
                  position: "top",
                  fill: "#ef4444",
                  fontSize: 10,
                }}
              />
            )}
            {ec50 > 0 && Number.isFinite(ec50) && (
              <ReferenceDot
                x={ec50}
                y={50}
                r={4}
                fill="#10b981"
                stroke="#10b981"
                label={{
                  value: "EC50",
                  position: "bottom",
                  fill: "#10b981",
                  fontSize: 10,
                }}
              />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>
      <p className="shrink-0 text-[10px] text-zinc-500">
        Preview only — Apply Changes runs a parameter sweep against the current
        pathway.
      </p>
    </div>
  );
}

function NumField({
  label,
  value,
  onChange,
  step = 0.01,
  min,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  step?: number;
  min?: number;
}) {
  return (
    <div>
      <label className="text-[10px] uppercase tracking-wide text-zinc-500">
        {label}
      </label>
      <Input
        type="number"
        step={step}
        min={min}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="h-7 text-xs"
      />
    </div>
  );
}

// ===========================================================================
// Knockout mode
// ===========================================================================

function KnockoutMode({
  species,
  selected,
  onToggle,
}: {
  species: string[];
  selected: Set<string>;
  onToggle: (sp: string) => void;
}) {
  return (
    <div className="space-y-1.5">
      <p className="text-[10px] text-zinc-500">
        Force a species concentration to 0 (knockout) at t=0.
      </p>
      {species.map((sp) => {
        const on = selected.has(sp);
        return (
          <label
            key={sp}
            className="flex cursor-pointer items-center gap-2 rounded-md border border-zinc-800 bg-zinc-950/40 px-2.5 py-1.5 hover:bg-zinc-900/60"
          >
            <input
              type="checkbox"
              checked={on}
              onChange={() => onToggle(sp)}
              className="h-3.5 w-3.5 accent-emerald-400"
            />
            <span className="text-xs text-zinc-200">{sp}</span>
            {on && (
              <Badge
                variant="outline"
                className="ml-auto h-3.5 border-red-500/50 px-1 text-[9px] text-red-300"
              >
                KO
              </Badge>
            )}
          </label>
        );
      })}
    </div>
  );
}

// ===========================================================================
// Overexpression mode
// ===========================================================================

function OverexpressionMode({
  species,
  folds,
  onChange,
}: {
  species: string[];
  folds: Record<string, number>;
  onChange: (sp: string, fold: number) => void;
}) {
  return (
    <div className="space-y-1.5">
      <p className="text-[10px] text-zinc-500">
        Overexpress a species (fold change over baseline). 1× = no change.
      </p>
      {species.map((sp) => {
        const fold = folds[sp] ?? 1;
        return (
          <div
            key={sp}
            className="rounded-md border border-zinc-800 bg-zinc-950/40 px-2.5 py-1.5"
          >
            <div className="flex items-center justify-between gap-2">
              <span className="text-xs text-zinc-200">{sp}</span>
              <span className="font-mono text-[11px] text-emerald-300">
                {fold}×
              </span>
            </div>
            <input
              type="range"
              min={0}
              max={100}
              step={1}
              value={fold}
              onChange={(e) => onChange(sp, Number(e.target.value))}
              className="mt-1.5 h-1.5 w-full cursor-pointer accent-emerald-400"
            />
            <div className="mt-1 flex gap-1">
              {[1, 10, 100].map((f) => (
                <button
                  key={f}
                  type="button"
                  onClick={() => onChange(sp, f)}
                  className={cn(
                    "rounded px-1.5 py-0.5 text-[10px]",
                    fold === f
                      ? "bg-emerald-500/20 text-emerald-300"
                      : "bg-zinc-800 text-zinc-400 hover:bg-zinc-700"
                  )}
                >
                  {f}×
                </button>
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ===========================================================================
// Mutation mode
// ===========================================================================

function MutationMode({
  current,
  custom,
  onSelect,
  onCustom,
  onClearCustom,
}: {
  current: string;
  custom: Record<string, number>;
  onSelect: (key: string) => void;
  onCustom: (key: string, val: number) => void;
  onClearCustom: (key: string) => void;
}) {
  const preset = MUTATION_PRESETS.find((m) => m.key === current);
  return (
    <div className="space-y-2">
      <div className="space-y-1">
        <p className="text-[10px] text-zinc-500">
          Preset mutations alter kinetic parameters directly.
        </p>
        {MUTATION_PRESETS.map((m) => {
          const active = m.key === current;
          return (
            <button
              key={m.key}
              type="button"
              onClick={() => onSelect(m.key)}
              className={cn(
                "w-full rounded-md border px-2.5 py-1.5 text-left",
                active
                  ? "border-emerald-500/50 bg-emerald-500/10"
                  : "border-zinc-800 bg-zinc-950/40 hover:bg-zinc-900/60"
              )}
            >
              <div className="flex items-center gap-1.5">
                <span className="text-xs font-medium text-zinc-200">
                  {m.label}
                </span>
                {active && <Check className="h-3 w-3 text-emerald-400" />}
              </div>
              <p className="text-[10px] text-zinc-500">{m.description}</p>
            </button>
          );
        })}
      </div>

      <div className="space-y-1.5 border-t border-zinc-800 pt-2">
        <p className="text-[10px] font-medium uppercase tracking-wide text-zinc-500">
          Custom override
        </p>
        {PARAM_CATALOG.map((p) => {
          const overridden = p.key in custom;
          const val = custom[p.key] ?? BASELINE_PARAMS[p.key];
          return (
            <div key={p.key} className="flex items-center gap-2">
              <span className="w-16 shrink-0 text-xs text-zinc-300">
                {p.label}
              </span>
              <Input
                type="number"
                step={p.step}
                min={p.min}
                max={p.max}
                value={val}
                onChange={(e) => onCustom(p.key, Number(e.target.value))}
                className="h-6 w-24 text-xs"
              />
              <span className="text-[10px] text-zinc-500">{p.unit}</span>
              {overridden && (
                <>
                  <Badge
                    variant="outline"
                    className="h-3.5 border-amber-600/50 px-1 text-[9px] text-amber-300"
                  >
                    override
                  </Badge>
                  <button
                    type="button"
                    onClick={() => onClearCustom(p.key)}
                    className="text-[10px] text-zinc-500 hover:text-zinc-300"
                    title="Clear override"
                  >
                    clear
                  </button>
                </>
              )}
            </div>
          );
        })}
      </div>

      {preset && preset.key !== "none" && (
        <div className="rounded-md bg-zinc-800/50 px-2 py-1.5 text-[10px] text-zinc-400">
          <span className="text-zinc-300">Active preset:</span> {preset.label}
          <div className="mt-0.5">
            {Object.entries(preset.overrides)
              .map(([k, v]) => `${k}=${formatVal(v)}`)
              .join(", ")}
          </div>
        </div>
      )}
    </div>
  );
}

// ===========================================================================
// Comparison strip (current vs baseline)
// ===========================================================================

function ComparisonStrip({
  params,
  perturbations,
  knockoutCount,
  overexprCount,
  mutationKey,
}: {
  params: Record<string, number>;
  perturbations: NonNullable<SimulationParams["perturbations"]>;
  knockoutCount: number;
  overexprCount: number;
  mutationKey: string;
}) {
  const changed = PARAM_CATALOG.filter(
    (p) => Math.abs((params[p.key] ?? 0) - p.default) > 1e-12
  );
  const hasPerturb = perturbations.length > 0 || mutationKey !== "none";

  if (changed.length === 0 && !hasPerturb) {
    return (
      <div className="flex items-center gap-1.5 text-[10px] text-zinc-500">
        <CheckCircle2 className="h-3 w-3 text-emerald-400" />
        All parameters at baseline
      </div>
    );
  }

  return (
    <div className="flex flex-wrap items-center gap-1">
      {changed.map((p) => {
        const cur = params[p.key] ?? 0;
        const base = p.default;
        const ratio = base !== 0 ? cur / base : 0;
        return (
          <Badge
            key={p.key}
            variant="outline"
            className="h-4 border-zinc-700 px-1.5 text-[9px] text-zinc-300"
            title={`${p.label}: baseline ${formatVal(base)} → ${formatVal(
              cur
            )} ${p.unit}`}
          >
            {p.label}: {formatVal(base)} →{" "}
            <span className="text-amber-300">{formatVal(cur)}</span> (×
            {ratio.toFixed(2)})
          </Badge>
        );
      })}
      {knockoutCount > 0 && (
        <Badge
          variant="outline"
          className="h-4 border-red-500/50 px-1.5 text-[9px] text-red-300"
        >
          {knockoutCount} KO
        </Badge>
      )}
      {overexprCount > 0 && (
        <Badge
          variant="outline"
          className="h-4 border-cyan-500/50 px-1.5 text-[9px] text-cyan-300"
        >
          {overexprCount} OE
        </Badge>
      )}
      {mutationKey !== "none" && (
        <Badge
          variant="outline"
          className="h-4 border-purple-500/50 px-1.5 text-[9px] text-purple-300"
        >
          {MUTATION_PRESETS.find((m) => m.key === mutationKey)?.label ??
            mutationKey}
        </Badge>
      )}
    </div>
  );
}
