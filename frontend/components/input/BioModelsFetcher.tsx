"use client";

import React, { useState, useCallback } from "react";
import {
  Loader2,
  CheckCircle2,
  AlertCircle,
  Search,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { getJSON, V4_PREFIX, type PathwayClass } from "@/lib/api";

/**
 * BioModels Fetcher — sub-component for the InputArea "BioModels ID" tab
 * (Task C.6).
 *
 * Flow:
 * 1. User types a BIOMD identifier (e.g. `BIOMD0000000010`).
 * 2. The "Fetch" button issues `GET /api/v4/biomodels/{id}` (via the shared
 *    `getJSON` helper, which targets `API_BASE`).
 * 3. On success, the model metadata (name, species / reaction counts,
 *    publication, mapped pathway class) is previewed inline.
 * 4. The parent InputArea reads the fetched metadata (via `onModelChange`) and
 *    triggers a simulation when the user hits the prominent Submit button.
 *
 * Error handling: format validation (BIOMD + 10 digits) runs client-side; HTTP
 * and network errors surface as a red inline banner.
 */

export interface BioModelsMetadata {
  biomd_id: string;
  name?: string;
  title?: string;
  species_count?: number;
  reactions_count?: number;
  publication?: string;
  pathway_class?: PathwayClass;
  [key: string]: unknown;
}

export interface BioModelsFetcherProps {
  /** Called with the fetched metadata (or null on reset / failure). */
  onModelChange?: (model: BioModelsMetadata | null) => void;
}

/** BIOMD identifiers are `BIOMD` + exactly 10 digits. */
const BIOMD_RE = /^BIOMD\d{10}$/i;

export function BioModelsFetcher({ onModelChange }: BioModelsFetcherProps) {
  const [id, setId] = useState("");
  const [loading, setLoading] = useState(false);
  const [model, setModel] = useState<BioModelsMetadata | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fetchModel = useCallback(async () => {
    const trimmed = id.trim();
    if (!trimmed) {
      setError("Please enter a BIOMD ID (e.g. BIOMD0000000010)");
      setModel(null);
      onModelChange?.(null);
      return;
    }
    if (!BIOMD_RE.test(trimmed)) {
      setError("Invalid format. Expected BIOMD followed by 10 digits.");
      setModel(null);
      onModelChange?.(null);
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const data = await getJSON<BioModelsMetadata>(
        `${V4_PREFIX}/biomodels/${encodeURIComponent(trimmed)}`
      );
      setModel(data);
      onModelChange?.(data);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Failed to fetch model";
      setError(msg);
      setModel(null);
      onModelChange?.(null);
    } finally {
      setLoading(false);
    }
  }, [id, onModelChange]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      e.preventDefault();
      fetchModel();
    }
  };

  return (
    <div className="space-y-2.5">
      <div className="flex items-center gap-1.5">
        <Input
          value={id}
          onChange={(e) => setId(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="BIOMD0000000010"
          className="font-mono text-xs border-zinc-700 bg-zinc-950/50 text-zinc-100 placeholder:text-zinc-600"
        />
        <Button
          type="button"
          size="sm"
          variant="outline"
          onClick={fetchModel}
          disabled={loading || !id.trim()}
          className="shrink-0 border-zinc-700 bg-zinc-900/60 text-zinc-200 hover:bg-zinc-800"
        >
          {loading ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
          ) : (
            <Search className="h-3.5 w-3.5" />
          )}
          Fetch
        </Button>
      </div>

      <div className="text-[10px] text-zinc-500">
        Format: BIOMD followed by 10 digits (e.g. BIOMD0000000010)
      </div>

      {error && (
        <div className="flex items-start gap-1.5 rounded-md border border-red-700/40 bg-red-900/10 px-2.5 py-2 text-[11px] text-red-300">
          <AlertCircle className="mt-0.5 h-3 w-3 shrink-0" />
          {error}
        </div>
      )}

      {model && !error && (
        <div className="space-y-2 rounded-md border border-emerald-700/40 bg-emerald-900/10 px-2.5 py-2.5">
          <div className="flex items-center gap-1.5 text-[11px] font-medium text-emerald-300">
            <CheckCircle2 className="h-3 w-3" />
            Model fetched
          </div>
          <div className="space-y-1 text-[11px] text-zinc-300">
            <div>
              <span className="text-zinc-500">Name:</span>{" "}
              {model.name || model.title || "—"}
            </div>
            <div className="flex gap-3">
              <span>
                <span className="text-zinc-500">Species:</span>{" "}
                {model.species_count ?? "—"}
              </span>
              <span>
                <span className="text-zinc-500">Reactions:</span>{" "}
                {model.reactions_count ?? "—"}
              </span>
            </div>
            {model.publication && (
              <div className="truncate">
                <span className="text-zinc-500">Publication:</span>{" "}
                {model.publication}
              </div>
            )}
            {model.pathway_class && (
              <div>
                <span className="text-zinc-500">Pathway:</span>{" "}
                <span className="font-mono">{model.pathway_class}</span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
