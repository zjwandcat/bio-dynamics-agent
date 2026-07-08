"use client";

import React, { useEffect, useMemo, useState } from "react";
import {
  Network,
  GitBranch,
  Activity,
  ShieldAlert,
  Skull,
  RefreshCw,
  Zap,
  Flame,
  Waves,
  TrendingDown,
  Sparkles,
  Dna,
  CheckCircle2,
  Loader2,
  type LucideIcon,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useWorkbenchStore } from "@/lib/store";
import { useTranslation } from "@/lib/i18n";
import {
  fetchPathways,
  type PathwayClass,
  type PathwaySummary,
} from "@/lib/api";

/**
 * PathwayTree — left-pane pathway selector (Task C.2).
 *
 * Renders 10 canonical signaling pathways grouped by category, plus an
 * "Auto Detect" mode that leaves `currentPathway = null` so the backend can
 * auto-detect the pathway from user input.
 *
 * Data source: `GET /api/v4/pathways` on mount. If the endpoint is unavailable
 * (backend not implemented yet), falls back to a hardcoded list mirroring the
 * backend `ontology/pathway_registry.py` (10 pathways).
 *
 * Selecting a pathway dispatches `setCurrentPathway(pathwayClass)` on the
 * global workbench store. The active pathway is highlighted with the same
 * blue accent used by `RunControls`.
 */

export interface PathwayItem {
  /** Display id (fallback uses the spec keys, e.g. "egfr_signaling"). */
  id: string;
  /** Canonical pathway class — written into the global store. */
  pathwayClass: PathwayClass;
  displayName: string;
  description: string;
  icon: LucideIcon;
  /** Number of modeled species / specialist modules (badge). */
  speciesCount: number;
  category: PathwaySummary["category"];
}

/** Icon per canonical pathway class (covers all 10 PathwayClass values). */
const ICON_BY_CLASS: Record<PathwayClass, LucideIcon> = {
  egfr: Network,
  mapk: GitBranch,
  pi3k_akt_mtor: Activity,
  p53: ShieldAlert,
  apoptosis: Skull,
  cell_cycle: RefreshCw,
  jak_stat: Zap,
  nf_kappa_b: Flame,
  wnt: Waves,
  tgf_beta: TrendingDown,
};

const CATEGORY_LABEL: Record<PathwaySummary["category"], string> = {
  core: "Core Signaling",
  feedback: "Feedback Control",
  crosstalk: "Crosstalk",
  perturbation: "Perturbation",
  validation: "Validation",
};

const CATEGORY_ORDER: PathwaySummary["category"][] = [
  "core",
  "feedback",
  "crosstalk",
  "perturbation",
  "validation",
];

/**
 * Hardcoded fallback (10 pathways). Keys follow the C.2 spec; each maps 1:1
 * to a canonical `PathwayClass` for store integration. Descriptions distilled
 * from `backend/app/ontology/pathway_registry.py`.
 */
const FALLBACK_PATHWAYS: PathwayItem[] = [
  {
    id: "egfr_signaling",
    pathwayClass: "egfr",
    displayName: "EGFR RTK Signaling",
    description:
      "Ligand binding → receptor dimerization → autophosphorylation → downstream signaling",
    icon: Network,
    speciesCount: 12,
    category: "core",
  },
  {
    id: "mapk_cascade",
    pathwayClass: "mapk",
    displayName: "MAPK Cascade",
    description: "Ras → Raf → MEK → ERK three-tier kinase cascade amplification",
    icon: GitBranch,
    speciesCount: 8,
    category: "core",
  },
  {
    id: "pi3k_akt_mtor",
    pathwayClass: "pi3k_akt_mtor",
    displayName: "PI3K-AKT-mTOR",
    description:
      "Growth factor → PI3K → PIP3 → AKT → mTOR; metabolism & survival control",
    icon: Activity,
    speciesCount: 15,
    category: "core",
  },
  {
    id: "p53_signaling",
    pathwayClass: "p53",
    displayName: "p53 Signaling",
    description:
      "DNA damage → p53 phosphorylation → MDM2 feedback loop → arrest / apoptosis",
    icon: ShieldAlert,
    speciesCount: 10,
    category: "feedback",
  },
  {
    id: "apoptosis",
    pathwayClass: "apoptosis",
    displayName: "Apoptosis",
    description:
      "Extrinsic (death receptor) + intrinsic (mitochondrial) + Caspase cascade",
    icon: Skull,
    speciesCount: 14,
    category: "perturbation",
  },
  {
    id: "cell_cycle",
    pathwayClass: "cell_cycle",
    displayName: "Cell Cycle",
    description: "Cyclin-CDK drives G1/S/G2/M transitions; Rb/E2F regulation",
    icon: RefreshCw,
    speciesCount: 16,
    category: "core",
  },
  {
    id: "jak_stat",
    pathwayClass: "jak_stat",
    displayName: "JAK-STAT",
    description:
      "Cytokine receptor → JAK phosphorylation → STAT dimer nuclear entry → transcription",
    icon: Zap,
    speciesCount: 9,
    category: "core",
  },
  {
    id: "nfkb_signaling",
    pathwayClass: "nf_kappa_b",
    displayName: "NF-κB Signaling",
    description:
      "TNF/TLR → IKK → IκBα degradation → NF-κB nuclear entry; inflammation core",
    icon: Flame,
    speciesCount: 11,
    category: "crosstalk",
  },
  {
    id: "wnt_signaling",
    pathwayClass: "wnt",
    displayName: "Wnt Signaling",
    description:
      "Wnt → Frizzled/LRP → destruction complex dissociation → β-catenin accumulation",
    icon: Waves,
    speciesCount: 13,
    category: "crosstalk",
  },
  {
    id: "tgf_beta_signaling",
    pathwayClass: "tgf_beta",
    displayName: "TGF-β Signaling",
    description:
      "TGF-β → TβR → SMAD2/3 phosphorylation → SMAD4 complex nuclear entry",
    icon: TrendingDown,
    speciesCount: 10,
    category: "crosstalk",
  },
];

function pathwayItemFromSummary(s: PathwaySummary): PathwayItem {
  const Icon = ICON_BY_CLASS[s.pathway_class] ?? Network;
  return {
    id: s.pathway_class,
    pathwayClass: s.pathway_class,
    displayName: s.display_name,
    description: s.description ?? "",
    icon: Icon,
    speciesCount: s.species_count ?? 1,
    category: s.category,
  };
}

export function PathwayTree() {
  const { t } = useTranslation();
  const currentPathway = useWorkbenchStore((s) => s.currentPathway);
  const setCurrentPathway = useWorkbenchStore((s) => s.setCurrentPathway);

  const [pathways, setPathways] = useState<PathwayItem[]>(FALLBACK_PATHWAYS);
  const [loading, setLoading] = useState(true);
  const [usingFallback, setUsingFallback] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();
    (async () => {
      setLoading(true);
      try {
        const data = await fetchPathways(controller.signal);
        if (cancelled) return;
        if (Array.isArray(data) && data.length > 0) {
          setPathways(data.map(pathwayItemFromSummary));
          setUsingFallback(false);
        } else {
          setPathways(FALLBACK_PATHWAYS);
          setUsingFallback(true);
        }
      } catch {
        if (cancelled) return;
        setPathways(FALLBACK_PATHWAYS);
        setUsingFallback(true);
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, []);

  const isAuto = currentPathway === null;

  const grouped = useMemo(() => {
    const map = new Map<PathwaySummary["category"], PathwayItem[]>();
    for (const p of pathways) {
      const arr = map.get(p.category) ?? [];
      arr.push(p);
      map.set(p.category, arr);
    }
    return map;
  }, [pathways]);

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex shrink-0 items-center justify-between px-0.5 pb-1.5">
        <div className="text-[11px] font-medium uppercase tracking-wide text-zinc-500">
          {t("pathway.selection")}
        </div>
        {usingFallback && (
          <Badge
            variant="outline"
            className="h-4 border-zinc-700 px-1 text-[9px] text-zinc-500"
          >
            {t("pathway.offline")}
          </Badge>
        )}
      </div>

      <ScrollArea className="min-h-0 flex-1">
        <div className="space-y-2.5 pr-1.5">
          {/* Auto-detect mode */}
          <button
            type="button"
            onClick={() => setCurrentPathway(null)}
            aria-pressed={isAuto}
            className={cn(
              "flex w-full items-start gap-2 rounded-md border px-2.5 py-2 text-left transition-colors",
              isAuto
                ? "border-blue-500/50 bg-blue-500/10 text-blue-100"
                : "border-zinc-800 bg-zinc-950/50 text-zinc-300 hover:bg-zinc-800/60"
            )}
          >
            <span
              className={cn(
                "mt-0.5",
                isAuto ? "text-blue-400" : "text-zinc-500"
              )}
            >
              <Sparkles className="h-3.5 w-3.5" />
            </span>
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-1.5">
                <span className="text-xs font-medium">{t("pathway.autoDetect")}</span>
                {isAuto && <CheckCircle2 className="h-3 w-3 text-blue-400" />}
              </div>
              <div className="truncate text-[10px] text-zinc-500">
                {t("pathway.autoDetect.hint")}
              </div>
            </div>
          </button>

          {loading ? (
            <div className="flex items-center justify-center gap-1.5 py-6 text-[11px] text-zinc-500">
              <Loader2 className="h-3 w-3 animate-spin" />
              {t("pathway.loading")}
            </div>
          ) : (
            <div className="space-y-2.5">
              {CATEGORY_ORDER.filter((c) => grouped.has(c)).map((cat) => (
                <div key={cat} className="space-y-1">
                  <div className="px-0.5 text-[10px] font-medium uppercase tracking-wide text-zinc-600">
                    {CATEGORY_LABEL[cat]}
                  </div>
                  <div className="space-y-1">
                    {grouped.get(cat)!.map((item) => {
                      const active = currentPathway === item.pathwayClass;
                      const Icon = item.icon;
                      return (
                        <button
                          key={item.id}
                          type="button"
                          onClick={() => setCurrentPathway(item.pathwayClass)}
                          aria-pressed={active}
                          title={item.description}
                          className={cn(
                            "flex w-full items-start gap-2 rounded-md border px-2.5 py-2 text-left transition-colors",
                            active
                              ? "border-blue-500/50 bg-blue-500/10 text-blue-100"
                              : "border-zinc-800 bg-zinc-950/50 text-zinc-300 hover:bg-zinc-800/60"
                          )}
                        >
                          <span
                            className={cn(
                              "mt-0.5",
                              active ? "text-blue-400" : "text-zinc-500"
                            )}
                          >
                            <Icon className="h-3.5 w-3.5" />
                          </span>
                          <div className="min-w-0 flex-1">
                            <div className="flex items-center justify-between gap-1.5">
                              <span className="truncate text-xs font-medium">
                                {item.displayName}
                              </span>
                              <Badge
                                variant="outline"
                                title="species count"
                                className="h-4 shrink-0 gap-0.5 border-zinc-700 px-1 text-[9px] text-zinc-500"
                              >
                                <Dna className="h-2.5 w-2.5" />
                                {item.speciesCount}
                              </Badge>
                            </div>
                            <div className="mt-0.5 line-clamp-2 text-[10px] leading-tight text-zinc-500">
                              {item.description}
                            </div>
                          </div>
                        </button>
                      );
                    })}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </ScrollArea>
    </div>
  );
}
