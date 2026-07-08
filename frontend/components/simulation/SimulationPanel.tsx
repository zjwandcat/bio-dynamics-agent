"use client";

import React, { useCallback, useState } from "react";
import { Loader2, Play, FlaskConical } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useWorkbenchStore } from "@/lib/store";
import { runSimulation, type PathwayClass, type SimulationParams } from "@/lib/api";
import { useTranslation } from "@/lib/i18n";

import { SimulationTabs, type SimStatus } from "./SimulationTabs";

/**
 * Simulation Panel — the center-pane wrapper around `SimulationTabs`.
 *
 * Owns the run lifecycle + status (idle / running / complete / error) and
 * renders the "Run Simulation" header button. The tab views themselves live in
 * `SimulationTabs`; this component replaces the center pane "Simulation Tabs"
 * placeholder from `WorkbenchShell` (wired in by the integration task).
 *
 * The run calls the v4 simulation endpoint (`/api/v4/simulation/run`). The
 * backend route lands in a later sprint task; until then a failed run surfaces
 * a clear error state without crashing the panel.
 */
export function SimulationPanel() {
  const { t } = useTranslation();
  const simulationResult = useWorkbenchStore((s) => s.simulationResult);
  const setSimulationResult = useWorkbenchStore((s) => s.setSimulationResult);
  const currentPathway = useWorkbenchStore((s) => s.currentPathway);

  const [status, setStatus] = useState<SimStatus>(
    simulationResult ? "complete" : "idle"
  );
  const [errorMessage, setErrorMessage] = useState("");

  const handleRun = useCallback(async () => {
    setStatus("running");
    setErrorMessage("");
    try {
      const params: SimulationParams = {
        pathway_class: (currentPathway ?? "egfr") as PathwayClass,
        duration: 120,
        steps: 200,
        parameters: {},
      };
      const result = await runSimulation(params);
      setSimulationResult(result);
      setStatus("complete");
    } catch (err) {
      const msg =
        err instanceof Error ? err.message : "Simulation request failed.";
      setErrorMessage(msg);
      setStatus("error");
    }
  }, [currentPathway, setSimulationResult]);

  return (
    <section className="flex h-full min-h-0 flex-col rounded-lg border border-zinc-800 bg-zinc-900/60">
      <header className="flex shrink-0 items-center justify-between gap-2 border-b border-zinc-800 px-3 py-2">
        <div className="flex items-center gap-2">
          <FlaskConical className="h-3.5 w-3.5 text-emerald-400" />
          <h3 className="text-xs font-semibold text-zinc-200">{t("sim.title")}</h3>
          <StatusBadge status={status} t={t} />
        </div>
        <Button
          size="sm"
          onClick={handleRun}
          disabled={status === "running"}
        >
          {status === "running" ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
          ) : (
            <Play className="h-3.5 w-3.5" />
          )}
          {t("sim.run")}
        </Button>
      </header>

      <div className="flex min-h-0 flex-1 flex-col p-2">
        <SimulationTabs
          status={status}
          errorMessage={errorMessage}
          onRun={handleRun}
          className="min-h-0 flex-1"
        />
      </div>
    </section>
  );
}

function StatusBadge({ status, t }: { status: SimStatus; t: (key: string) => string }) {
  const map: Record<
    SimStatus,
    { labelKey: string; variant: "default" | "secondary" | "outline" | "destructive"; className: string }
  > = {
    idle: {
      labelKey: "sim.status.idle",
      variant: "outline",
      className: "border-zinc-700 text-zinc-500",
    },
    running: {
      labelKey: "sim.status.running",
      variant: "secondary",
      className: "bg-blue-500/15 text-blue-300",
    },
    complete: {
      labelKey: "sim.status.complete",
      variant: "secondary",
      className: "bg-emerald-500/15 text-emerald-300",
    },
    error: {
      labelKey: "sim.status.error",
      variant: "destructive",
      className: "",
    },
  };
  const cfg = map[status];
  return (
    <Badge variant={cfg.variant} className={`h-4 text-[9px] ${cfg.className}`}>
      {t(cfg.labelKey)}
    </Badge>
  );
}
