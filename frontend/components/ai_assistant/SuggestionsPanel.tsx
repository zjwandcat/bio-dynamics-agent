"use client";

import React from "react";
import { Lightbulb, FlaskConical, Sliders, Check } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export type SuggestionType =
  | "hypothesis_refinement"
  | "parameter_adjustment"
  | "experiment_suggestion";

export interface Suggestion {
  id: string;
  type: SuggestionType;
  text: string;
  source?: string;
}

/** Shape of a single agent dispatch entry (mirrors `DispatchData` in store). */
export interface DispatchEntry {
  target_agent: string;
  reasoning?: string;
  status: string;
  latency_ms?: number;
  node_name?: string;
}

export interface SuggestionsPanelProps {
  hypothesisList: unknown[];
  agentDispatches: DispatchEntry[];
  onApply?: (suggestion: Suggestion) => void;
  className?: string;
}

const TYPE_META: Record<
  SuggestionType,
  { label: string; icon: React.ReactNode; badgeClass: string }
> = {
  hypothesis_refinement: {
    label: "Hypothesis",
    icon: <Lightbulb className="h-3 w-3" />,
    badgeClass: "border-amber-700/50 text-amber-300",
  },
  parameter_adjustment: {
    label: "Parameter",
    icon: <Sliders className="h-3 w-3" />,
    badgeClass: "border-blue-700/50 text-blue-300",
  },
  experiment_suggestion: {
    label: "Experiment",
    icon: <FlaskConical className="h-3 w-3" />,
    badgeClass: "border-emerald-700/50 text-emerald-300",
  },
};

/**
 * Pull a human-readable statement out of a heterogeneous hypothesis object.
 * The v4 hypothesis generator emits objects with `statement`, but we tolerate
 * `text` / `hypothesis` / bare strings for forward-compatibility.
 */
function extractHypothesisText(h: unknown): string {
  if (typeof h === "string") return h;
  if (h && typeof h === "object") {
    const obj = h as {
      statement?: string;
      text?: string;
      hypothesis?: string;
      description?: string;
    };
    return (
      obj.statement || obj.text || obj.hypothesis || obj.description || ""
    );
  }
  return "";
}

/**
 * Derive a flat list of actionable suggestions from the v4 hypothesis list and
 * the live agent-dispatch stream. Each hypothesis yields a refinement card and
 * an experiment-suggestion card; parameter-related agent dispatches yield
 * parameter-adjustment cards.
 */
function deriveSuggestions(
  hypothesisList: unknown[],
  agentDispatches: DispatchEntry[]
): Suggestion[] {
  const suggestions: Suggestion[] = [];

  hypothesisList.forEach((h, idx) => {
    const text = extractHypothesisText(h) || JSON.stringify(h);
    suggestions.push({
      id: `hyp-refine-${idx}`,
      type: "hypothesis_refinement",
      text,
      source: "hypothesis_generator",
    });
    suggestions.push({
      id: `exp-suggest-${idx}`,
      type: "experiment_suggestion",
      text: `Design experiment to test: ${text}`,
      source: "hypothesis_generator",
    });
  });

  agentDispatches.forEach((d, idx) => {
    const agentLower = d.target_agent.toLowerCase();
    const nodeLower = d.node_name?.toLowerCase() ?? "";
    const isParameterAgent =
      agentLower.includes("parameter") ||
      agentLower.includes("calibrat") ||
      nodeLower.includes("parameter") ||
      nodeLower.includes("calibrat");
    if (isParameterAgent && d.reasoning) {
      suggestions.push({
        id: `param-adj-${idx}`,
        type: "parameter_adjustment",
        text: d.reasoning,
        source: d.target_agent,
      });
    }
  });

  return suggestions;
}

/**
 * Suggestions tab content for the AI Assistant panel.
 *
 * Shows AI-generated suggestions derived from the v4 hypothesis list and the
 * agent-dispatch stream. Each card carries a type badge, the suggestion text,
 * and an "Apply" button that forwards the selection to the parent (which
 * typically pre-fills the chat input and switches to the Chat tab).
 */
export function SuggestionsPanel({
  hypothesisList,
  agentDispatches,
  onApply,
  className,
}: SuggestionsPanelProps) {
  const suggestions = deriveSuggestions(hypothesisList, agentDispatches);

  return (
    <div className={cn("h-full overflow-auto", className)}>
      <div className="space-y-2 p-3">
        {suggestions.length === 0 ? (
          <div className="flex flex-col items-center justify-center gap-2 pt-20 text-center text-zinc-500">
            <Lightbulb className="h-8 w-8 text-zinc-700" />
            <p className="text-xs">No suggestions yet</p>
            <p className="px-6 text-[11px] leading-relaxed text-zinc-600">
              Run a simulation to generate hypothesis refinements, parameter
              adjustments, and experiment suggestions.
            </p>
          </div>
        ) : (
          suggestions.map((s) => {
            const meta = TYPE_META[s.type];
            return (
              <div
                key={s.id}
                className="rounded-lg border border-zinc-700 bg-zinc-800/40 p-2.5"
              >
                <div className="mb-1.5 flex items-center justify-between gap-2">
                  <Badge
                    variant="outline"
                    className={cn("gap-1 text-[10px]", meta.badgeClass)}
                  >
                    {meta.icon}
                    {meta.label}
                  </Badge>
                  {s.source && (
                    <span className="truncate text-[10px] text-zinc-600" title={s.source}>
                      {s.source}
                    </span>
                  )}
                </div>
                <p className="mb-2 text-xs leading-relaxed text-zinc-300">
                  {s.text}
                </p>
                <Button
                  size="xs"
                  variant="outline"
                  onClick={() => onApply?.(s)}
                  className="border-zinc-600 text-zinc-300 hover:bg-zinc-700/50 hover:text-zinc-100"
                >
                  <Check className="h-3 w-3" />
                  Apply
                </Button>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
