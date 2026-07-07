"use client";

import React, { useState, useRef, useCallback } from "react";
import {
  MessageSquare,
  FlaskConical,
  Sliders,
  FileCode,
  Database,
  Send,
  Loader2,
  AlertCircle,
  Plus,
  Trash2,
  Upload as UploadIcon,
  CheckCircle2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import { useWorkbenchStore } from "@/lib/store";
import {
  runSimulation,
  type PathwayClass,
} from "@/lib/api";
import {
  SbmlUpload,
  type SbmlUploadHandle,
  type SbmlUploadResult,
} from "./SbmlUpload";
import {
  BioModelsFetcher,
  type BioModelsMetadata,
} from "./BioModelsFetcher";

/**
 * InputArea — multi-mode scientific input panel (Task C.6).
 *
 * Replaces the legacy single chat box with a 5-mode tabbed input that routes
 * user intent to the correct backend contract:
 *
 *   1. Natural Language     → existing `/api/chat` SSE stream (v3) via the
 *                             global store's `sendMessage` action. This is the
 *                             AI Assistant path — it must NOT be version-bumped.
 *   2. Structured Hypothesis→ `POST /api/v4/simulation/run` with a typed
 *                             `SimulationParams` payload.
 *   3. Parameter Panel      → `POST /api/v4/simulation/run` with a key/value
 *                             parameter table (importable from JSON).
 *   4. SBML Upload          → `POST /api/v4/sbml/import` (multipart) via the
 *                             SbmlUpload sub-component's imperative `upload()`.
 *   5. BioModels ID         → `GET  /api/v4/biomodels/{id}` for metadata, then
 *                             Submit triggers `POST /api/v4/simulation/run`.
 *
 * The prominent Submit button lives in a sticky footer (bottom-right) and its
 * behaviour is dispatched per active mode. Dark zinc theme matches the
 * WorkbenchShell palette.
 */

type InputMode = "nl" | "structured" | "params" | "sbml" | "biomodels";

interface ModeDef {
  value: InputMode;
  label: string;
  icon: React.ReactNode;
}

const MODES: ModeDef[] = [
  { value: "nl", label: "Natural Language", icon: <MessageSquare className="h-3.5 w-3.5" /> },
  { value: "structured", label: "Structured", icon: <FlaskConical className="h-3.5 w-3.5" /> },
  { value: "params", label: "Parameters", icon: <Sliders className="h-3.5 w-3.5" /> },
  { value: "sbml", label: "SBML Upload", icon: <FileCode className="h-3.5 w-3.5" /> },
  { value: "biomodels", label: "BioModels ID", icon: <Database className="h-3.5 w-3.5" /> },
];

const PATHWAY_OPTIONS: { value: PathwayClass; label: string }[] = [
  { value: "egfr", label: "EGFR" },
  { value: "mapk", label: "MAPK" },
  { value: "pi3k_akt_mtor", label: "PI3K / AKT / mTOR" },
  { value: "jak_stat", label: "JAK / STAT" },
  { value: "nf_kappa_b", label: "NF-κB" },
  { value: "wnt", label: "Wnt" },
  { value: "tgf_beta", label: "TGF-β" },
  { value: "p53", label: "p53" },
  { value: "apoptosis", label: "Apoptosis" },
  { value: "cell_cycle", label: "Cell Cycle" },
];

const PERTURBATION_OPTIONS: { value: string; label: string }[] = [
  { value: "", label: "— none —" },
  { value: "knockout", label: "Knockout" },
  { value: "overexpression", label: "Overexpression" },
  { value: "inhibit", label: "Inhibit" },
  { value: "dose", label: "Dose" },
];

interface ParamRow {
  id: string;
  name: string;
  value: string;
  unit: string;
}

function makeId() {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
}

const selectClass =
  "h-8 w-full rounded-lg border border-zinc-700 bg-zinc-950/50 px-2.5 py-1 text-sm text-zinc-100 outline-none transition-colors focus-visible:border-blue-500 focus-visible:ring-2 focus-visible:ring-blue-500/30";

const labelClass = "text-[11px] font-medium text-zinc-400";

const inputClass =
  "border-zinc-700 bg-zinc-950/50 text-zinc-100 placeholder:text-zinc-600";

export function InputArea() {
  const [mode, setMode] = useState<InputMode>("nl");

  // --- NL mode (routes through the existing v3 /api/chat SSE stream) ---
  const sendMessage = useWorkbenchStore((s) => s.sendMessage);
  const isStreaming = useWorkbenchStore((s) => s.isStreaming);
  const setSimulationResult = useWorkbenchStore((s) => s.setSimulationResult);
  const [nlText, setNlText] = useState("");

  // --- Structured Hypothesis mode ---
  const [stPathway, setStPathway] = useState<PathwayClass>("egfr");
  const [stHypothesis, setStHypothesis] = useState("");
  const [stSpecies, setStSpecies] = useState("");
  const [stDuration, setStDuration] = useState(60);
  const [stPerturbation, setStPerturbation] = useState("");

  // --- Parameter Panel mode ---
  const [paramRows, setParamRows] = useState<ParamRow[]>([
    { id: makeId(), name: "", value: "", unit: "" },
  ]);

  // --- SBML Upload mode ---
  const sbmlRef = useRef<SbmlUploadHandle>(null);
  const [sbmlFile, setSbmlFile] = useState<File | null>(null);
  const [, setSbmlResult] = useState<SbmlUploadResult | null>(null);

  // --- BioModels ID mode ---
  const [biomodel, setBiomodel] = useState<BioModelsMetadata | null>(null);

  // --- Submit feedback ---
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [submitOk, setSubmitOk] = useState<string | null>(null);

  const addParamRow = () => {
    setParamRows((rs) => [...rs, { id: makeId(), name: "", value: "", unit: "" }]);
  };

  const removeParamRow = (id: string) => {
    setParamRows((rs) => (rs.length > 1 ? rs.filter((r) => r.id !== id) : rs));
  };

  const updateParamRow = (
    id: string,
    field: keyof ParamRow,
    value: string
  ) => {
    setParamRows((rs) =>
      rs.map((r) => (r.id === id ? { ...r, [field]: value } : r))
    );
  };

  const importParamsJson = () => {
    const fileInput = document.createElement("input");
    fileInput.type = "file";
    fileInput.accept = ".json,application/json";
    fileInput.onchange = () => {
      const f = fileInput.files?.[0];
      if (!f) return;
      const reader = new FileReader();
      reader.onload = () => {
        try {
          const parsed = JSON.parse(String(reader.result));
          let entries: unknown[] = [];
          if (Array.isArray(parsed)) {
            entries = parsed;
          } else if (parsed && typeof parsed === "object") {
            const obj = parsed as Record<string, unknown>;
            if (Array.isArray(obj.parameters)) {
              entries = obj.parameters as unknown[];
            } else {
              entries = Object.entries(obj).map(([name, value]) => ({
                name,
                value,
                unit: "",
              }));
            }
          } else {
            throw new Error("JSON root must be an object or array");
          }
          const rows: ParamRow[] = entries
            .map((entry) => {
              const e = (entry ?? {}) as Record<string, unknown>;
              return {
                id: makeId(),
                name: String(e.name ?? e.key ?? e.parameter ?? ""),
                value: String(e.value ?? ""),
                unit: String(e.unit ?? e.units ?? ""),
              };
            })
            .filter((r) => r.name || r.value);
          if (rows.length === 0) {
            throw new Error("No parameter entries found in JSON");
          }
          setParamRows(rows);
          setSubmitError(null);
        } catch (err) {
          setSubmitError(
            `Import failed: ${err instanceof Error ? err.message : "invalid JSON"}`
          );
        }
      };
      reader.readAsText(f);
    };
    fileInput.click();
  };

  const canSubmit = (): boolean => {
    if (submitting || isStreaming) return false;
    switch (mode) {
      case "nl":
        return nlText.trim().length > 0;
      case "structured":
        return Boolean(stHypothesis.trim()) && stDuration > 0;
      case "params": {
        const valid = paramRows.filter((r) => r.name.trim() && r.value.trim());
        return valid.length > 0;
      }
      case "sbml":
        return sbmlFile !== null;
      case "biomodels":
        return biomodel !== null;
      default:
        return false;
    }
  };

  const flashOk = (msg: string) => {
    setSubmitOk(msg);
    setTimeout(() => setSubmitOk(null), 3000);
  };

  const handleSubmit = useCallback(async () => {
    setSubmitError(null);
    setSubmitOk(null);

    // NL mode: delegate to the store's v3 SSE chat action.
    if (mode === "nl") {
      const text = nlText.trim();
      if (!text) return;
      setNlText("");
      await sendMessage(text);
      return;
    }

    setSubmitting(true);
    try {
      if (mode === "structured") {
        const species = stSpecies
          .split(",")
          .map((s) => s.trim())
          .filter(Boolean);
        const initialConditions =
          species.length > 0
            ? Object.fromEntries(species.map((s) => [s, 1.0]))
            : undefined;
        const perturbations =
          stPerturbation && species.length > 0
            ? [
                {
                  target: species[0],
                  kind: stPerturbation as
                    | "knockout"
                    | "overexpression"
                    | "inhibit"
                    | "dose",
                },
              ]
            : undefined;
        const result = await runSimulation({
          pathway_class: stPathway,
          duration: stDuration,
          steps: Math.max(10, Math.round(stDuration)),
          parameters: {},
          initial_conditions: initialConditions,
          perturbations,
        });
        setSimulationResult(result);
        flashOk(`Simulation queued · run_id ${result.run_id}`);
      } else if (mode === "params") {
        const valid = paramRows.filter((r) => r.name.trim() && r.value.trim());
        if (valid.length === 0) throw new Error("No valid parameter rows");
        const parameters: Record<string, number> = {};
        for (const r of valid) {
          const n = Number(r.value);
          parameters[r.name.trim()] = Number.isFinite(n) ? n : 0;
        }
        const result = await runSimulation({
          pathway_class: "egfr",
          duration: 60,
          steps: 60,
          parameters,
        });
        setSimulationResult(result);
        flashOk(`Simulation queued · run_id ${result.run_id}`);
      } else if (mode === "sbml") {
        const result = await sbmlRef.current?.upload();
        if (result) {
          setSbmlResult(result);
          flashOk("SBML imported successfully");
        }
      } else if (mode === "biomodels") {
        if (!biomodel) throw new Error("No BioModels model loaded");
        const pathway: PathwayClass = biomodel.pathway_class ?? "egfr";
        const result = await runSimulation({
          pathway_class: pathway,
          duration: 60,
          steps: 60,
          parameters: {},
        });
        setSimulationResult(result);
        flashOk(`Simulation queued · run_id ${result.run_id}`);
      }
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : "Submit failed");
    } finally {
      setSubmitting(false);
    }
  }, [
    mode,
    nlText,
    sendMessage,
    stPathway,
    stHypothesis,
    stSpecies,
    stDuration,
    stPerturbation,
    paramRows,
    biomodel,
    setSimulationResult,
  ]);

  const busy = submitting || isStreaming;

  return (
    <section className="flex h-full min-h-0 flex-col rounded-lg border border-zinc-800 bg-zinc-900/60">
      {/* ── Horizontal tab strip ── */}
      <div className="shrink-0 border-b border-zinc-800 px-2 py-2">
        <div className="flex items-center gap-1 overflow-x-auto">
          {MODES.map((m) => {
            const active = mode === m.value;
            return (
              <button
                key={m.value}
                type="button"
                onClick={() => setMode(m.value)}
                className={cn(
                  "inline-flex shrink-0 items-center gap-1.5 rounded-md px-2.5 py-1.5 text-xs font-medium transition-colors",
                  active
                    ? "bg-blue-600 text-white"
                    : "text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
                )}
              >
                {m.icon}
                {m.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* ── Mode body ── */}
      <div className="min-h-0 flex-1 overflow-auto p-3">
        {mode === "nl" && (
          <div className="space-y-2">
            <div className={labelClass}>Hypothesis (free text)</div>
            <Textarea
              value={nlText}
              onChange={(e) => setNlText(e.target.value)}
              placeholder="e.g. Inhibition of MEK reduces ERK phosphorylation in a dose-dependent manner..."
              rows={5}
              className={cn("resize-none", inputClass)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
                  e.preventDefault();
                  handleSubmit();
                }
              }}
            />
            <div className="text-[10px] text-zinc-500">
              ⌘/Ctrl + Enter to submit · routes through the existing AI Assistant stream
            </div>
          </div>
        )}

        {mode === "structured" && (
          <div className="space-y-3">
            <div className="space-y-1">
              <div className={labelClass}>Pathway</div>
              <select
                value={stPathway}
                onChange={(e) => setStPathway(e.target.value as PathwayClass)}
                className={selectClass}
              >
                {PATHWAY_OPTIONS.map((o) => (
                  <option key={o.value} value={o.value}>
                    {o.label}
                  </option>
                ))}
              </select>
            </div>
            <div className="space-y-1">
              <div className={labelClass}>Hypothesis</div>
              <Textarea
                value={stHypothesis}
                onChange={(e) => setStHypothesis(e.target.value)}
                placeholder="State the hypothesis to test..."
                rows={3}
                className={cn("resize-none", inputClass)}
              />
            </div>
            <div className="grid grid-cols-2 gap-2">
              <div className="space-y-1">
                <div className={labelClass}>Species of interest</div>
                <Input
                  value={stSpecies}
                  onChange={(e) => setStSpecies(e.target.value)}
                  placeholder="ERK, MEK, RAF"
                  className={inputClass}
                />
                <div className="text-[10px] text-zinc-500">Comma-separated</div>
              </div>
              <div className="space-y-1">
                <div className={labelClass}>Duration (min)</div>
                <Input
                  type="number"
                  min={1}
                  value={stDuration}
                  onChange={(e) =>
                    setStDuration(Number(e.target.value) || 0)
                  }
                  className={inputClass}
                />
              </div>
            </div>
            <div className="space-y-1">
              <div className={labelClass}>Perturbation (optional)</div>
              <select
                value={stPerturbation}
                onChange={(e) => setStPerturbation(e.target.value)}
                className={selectClass}
              >
                {PERTURBATION_OPTIONS.map((o) => (
                  <option key={o.value} value={o.value}>
                    {o.label}
                  </option>
                ))}
              </select>
            </div>
          </div>
        )}

        {mode === "params" && (
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <div className={labelClass}>Parameter Table</div>
              <Button
                type="button"
                size="xs"
                variant="outline"
                onClick={importParamsJson}
                className="border-zinc-700 bg-zinc-900/60 text-zinc-300 hover:bg-zinc-800"
              >
                <UploadIcon className="h-3 w-3" /> Import JSON
              </Button>
            </div>
            <div className="space-y-1.5">
              <div className="grid grid-cols-[1fr_1fr_70px_28px] gap-1.5 px-1 text-[10px] uppercase tracking-wide text-zinc-500">
                <div>Name</div>
                <div>Value</div>
                <div>Unit</div>
                <div />
              </div>
              {paramRows.map((r) => (
                <div
                  key={r.id}
                  className="grid grid-cols-[1fr_1fr_70px_28px] items-center gap-1.5"
                >
                  <Input
                    value={r.name}
                    onChange={(e) => updateParamRow(r.id, "name", e.target.value)}
                    placeholder="k1"
                    className={cn("h-7 text-xs", inputClass)}
                  />
                  <Input
                    value={r.value}
                    onChange={(e) => updateParamRow(r.id, "value", e.target.value)}
                    placeholder="0.1"
                    className={cn("h-7 text-xs", inputClass)}
                  />
                  <Input
                    value={r.unit}
                    onChange={(e) => updateParamRow(r.id, "unit", e.target.value)}
                    placeholder="1/min"
                    className={cn("h-7 text-xs", inputClass)}
                  />
                  <button
                    type="button"
                    onClick={() => removeParamRow(r.id)}
                    className="flex h-7 w-7 items-center justify-center rounded text-zinc-500 hover:bg-zinc-800 hover:text-red-400"
                    aria-label="Remove row"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                </div>
              ))}
            </div>
            <Button
              type="button"
              size="xs"
              variant="ghost"
              onClick={addParamRow}
              className="text-zinc-400 hover:bg-zinc-800"
            >
              <Plus className="h-3 w-3" /> Add row
            </Button>
          </div>
        )}

        {mode === "sbml" && (
          <div className="space-y-2">
            <div className={labelClass}>SBML Model Upload</div>
            <SbmlUpload
              ref={sbmlRef}
              onFileSelect={setSbmlFile}
              onUploaded={setSbmlResult}
            />
          </div>
        )}

        {mode === "biomodels" && (
          <div className="space-y-2">
            <div className={labelClass}>BioModels Reference</div>
            <BioModelsFetcher onModelChange={setBiomodel} />
          </div>
        )}

        {/* ── Submit feedback ── */}
        {submitError && (
          <div className="mt-3 flex items-start gap-1.5 rounded-md border border-red-700/40 bg-red-900/10 px-2.5 py-2 text-[11px] text-red-300">
            <AlertCircle className="mt-0.5 h-3 w-3 shrink-0" />
            {submitError}
          </div>
        )}
        {submitOk && (
          <div className="mt-3 flex items-start gap-1.5 rounded-md border border-emerald-700/40 bg-emerald-900/10 px-2.5 py-2 text-[11px] text-emerald-300">
            <CheckCircle2 className="mt-0.5 h-3 w-3 shrink-0" />
            {submitOk}
          </div>
        )}
      </div>

      {/* ── Sticky submit footer ── */}
      <div className="flex shrink-0 items-center justify-between gap-2 border-t border-zinc-800 px-3 py-2">
        <div className="text-[10px] text-zinc-500">
          {mode === "nl" && "Routes via /api/chat (v3 SSE)"}
          {mode === "structured" && "Routes via /api/v4/simulation/run"}
          {mode === "params" && "Routes via /api/v4/simulation/run"}
          {mode === "sbml" && "Routes via /api/v4/sbml/import"}
          {mode === "biomodels" && "Routes via /api/v4/simulation/run"}
        </div>
        <Button
          type="button"
          onClick={handleSubmit}
          disabled={!canSubmit()}
          className="bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50"
        >
          {busy ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
          ) : (
            <Send className="h-3.5 w-3.5" />
          )}
          Submit
        </Button>
      </div>
    </section>
  );
}
