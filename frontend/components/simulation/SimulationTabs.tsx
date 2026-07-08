"use client";

import {
  Activity,
  AlertCircle,
  FlaskConical,
  GitCompare,
  Loader2,
  Play,
  TrendingUp,
  Waves,
  BarChart3,
  Equal,
} from "lucide-react";

import { useWorkbenchStore } from "@/lib/store";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { useTranslation } from "@/lib/i18n";

import type { SimulationResultExtras } from "./shared";
import { TimeSeriesChart } from "./TimeSeriesChart";
import { DoseResponseChart } from "./DoseResponseChart";
import { SensitivityChart } from "./SensitivityChart";
import { PhasePortraitChart } from "./PhasePortraitChart";
import { SteadyStateChart } from "./SteadyStateChart";
import { OscillationChart } from "./OscillationChart";

/** Simulation execution status, owned by the parent `SimulationPanel`. */
export type SimStatus = "idle" | "running" | "complete" | "error";

interface SimulationTabsProps {
  status: SimStatus;
  /** Error message surfaced when `status === "error"`. */
  errorMessage?: string;
  /** Invoke a new simulation run (wired to the parent's run handler). */
  onRun: () => void;
  className?: string;
}

function useTabs(t: (key: string) => string) {
  return [
    { value: "time_series" as const, label: t("sim.tab.timeSeries"), icon: Activity },
    { value: "dose_response" as const, label: t("sim.tab.doseResponse"), icon: TrendingUp },
    { value: "sensitivity" as const, label: t("sim.tab.sensitivity"), icon: BarChart3 },
    { value: "phase_portrait" as const, label: t("sim.tab.phasePortrait"), icon: GitCompare },
    { value: "steady_state" as const, label: t("sim.tab.steadyState"), icon: Equal },
    { value: "oscillation" as const, label: t("sim.tab.oscillation"), icon: Waves },
  ];
}

/**
 * Six-tab multi-view over a single shared simulation result.
 *
 * All tabs read the SAME `simulationResult` from the workbench store —
 * switching tabs only swaps the view, it never re-runs the simulation. Empty
 * and loading states are rendered here so the parent panel stays a thin shell
 * around the run action + status.
 */
export function SimulationTabs({
  status,
  errorMessage,
  onRun,
  className,
}: SimulationTabsProps) {
  const { t } = useTranslation();
  const TABS = useTabs(t);
  const result = useWorkbenchStore((s) => s.simulationResult);

  // ── Loading: spinner while a simulation is running ──
  if (status === "running") {
    return (
      <div
        className={
          className ??
          "flex h-full flex-col items-center justify-center gap-3 text-zinc-400"
        }
      >
        <Loader2 className="h-6 w-6 animate-spin text-emerald-400" />
        <p className="text-xs">{t("sim.running")}</p>
      </div>
    );
  }

  // ── Error: no result to show — surface the error with a retry ──
  if (status === "error" && !result) {
    return (
      <div
        className={
          className ??
          "flex h-full flex-col items-center justify-center gap-3 text-center"
        }
      >
        <AlertCircle className="h-6 w-6 text-red-400" />
        <p className="max-w-sm text-xs text-zinc-400">
          {errorMessage || t("sim.error")}
        </p>
        <Button size="sm" variant="outline" onClick={onRun}>
          <Play className="h-3.5 w-3.5" />
          {t("sim.retry")}
        </Button>
      </div>
    );
  }

  // ── Empty: no result yet — prompt the user to run ──
  if (!result) {
    return (
      <div
        className={
          className ??
          "flex h-full flex-col items-center justify-center gap-3 text-center"
        }
      >
        <FlaskConical className="h-6 w-6 text-zinc-600" />
        <p className="text-xs text-zinc-400">{t("sim.empty")}</p>
        <Button size="sm" onClick={onRun}>
          <Play className="h-3.5 w-3.5" />
          {t("sim.run")}
        </Button>
      </div>
    );
  }

  // ── Result available: render the 6-tab multi-view ──
  const extras = result as SimulationResultExtras;
  const { time_points, species } = result;

  return (
    <Tabs
      defaultValue="time_series"
      className={className ?? "flex h-full min-h-0 flex-col"}
    >
      <TabsList className="shrink-0">
        {TABS.map((tab) => {
          const Icon = tab.icon;
          return (
            <TabsTrigger key={tab.value} value={tab.value}>
              <Icon className="h-3 w-3" />
              <span className="truncate">{tab.label}</span>
            </TabsTrigger>
          );
        })}
      </TabsList>

      <TabsContent value="time_series" className="min-h-0 flex-1 rounded-lg bg-zinc-900/40 p-2 ring-1 ring-zinc-800">
        <TimeSeriesChart timePoints={time_points} species={species} />
      </TabsContent>
      <TabsContent value="dose_response" className="min-h-0 flex-1 rounded-lg bg-zinc-900/40 p-2 ring-1 ring-zinc-800">
        <DoseResponseChart data={extras.dose_response} />
      </TabsContent>
      <TabsContent value="sensitivity" className="min-h-0 flex-1 rounded-lg bg-zinc-900/40 p-2 ring-1 ring-zinc-800">
        <SensitivityChart data={extras.sensitivity} />
      </TabsContent>
      <TabsContent value="phase_portrait" className="min-h-0 flex-1 rounded-lg bg-zinc-900/40 p-2 ring-1 ring-zinc-800">
        <PhasePortraitChart timePoints={time_points} species={species} />
      </TabsContent>
      <TabsContent value="steady_state" className="min-h-0 flex-1 rounded-lg bg-zinc-900/40 p-2 ring-1 ring-zinc-800">
        <SteadyStateChart
          items={extras.steady_state}
          timePoints={time_points}
          species={species}
        />
      </TabsContent>
      <TabsContent value="oscillation" className="min-h-0 flex-1 rounded-lg bg-zinc-900/40 p-2 ring-1 ring-zinc-800">
        <OscillationChart
          timePoints={time_points}
          species={species}
          oscillation={extras.oscillation}
        />
      </TabsContent>
    </Tabs>
  );
}
