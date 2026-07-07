"use client";

import React from "react";
import { useRouter } from "next/navigation";
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
  ArrowUpRight,
  type LucideIcon,
} from "lucide-react";
import { useWorkbenchStore } from "@/lib/store";
import type { PathwayClass } from "@/lib/api";

/**
 * Pathway Selector section (Task C.10, section 02).
 *
 * Grid of the Auto-Detect card + 10 canonical signaling pathway cards. The
 * pathway list mirrors the C.2 `PathwayTree` fallback data
 * (backend `ontology/pathway_registry.py`). Clicking a card hydrates the
 * global workbench store's `currentPathway` and navigates to
 * `/workspace?pathway=<class>` so the workbench opens with the selection
 * already applied.
 */

interface PathwayEntry {
  pathwayClass: PathwayClass;
  displayName: string;
  description: string;
  icon: LucideIcon;
  speciesCount: number;
}

const PATHWAYS: PathwayEntry[] = [
  {
    pathwayClass: "egfr",
    displayName: "EGFR RTK Signaling",
    description:
      "Ligand binding → receptor dimerization → autophosphorylation → downstream signaling.",
    icon: Network,
    speciesCount: 12,
  },
  {
    pathwayClass: "mapk",
    displayName: "MAPK Cascade",
    description:
      "Ras → Raf → MEK → ERK three-tier kinase cascade amplification.",
    icon: GitBranch,
    speciesCount: 8,
  },
  {
    pathwayClass: "pi3k_akt_mtor",
    displayName: "PI3K-AKT-mTOR",
    description:
      "Growth factor → PI3K → PIP3 → AKT → mTOR; metabolism & survival control.",
    icon: Activity,
    speciesCount: 15,
  },
  {
    pathwayClass: "p53",
    displayName: "p53 Signaling",
    description:
      "DNA damage → p53 phosphorylation → MDM2 feedback loop → arrest / apoptosis.",
    icon: ShieldAlert,
    speciesCount: 10,
  },
  {
    pathwayClass: "apoptosis",
    displayName: "Apoptosis",
    description:
      "Extrinsic (death receptor) + intrinsic (mitochondrial) + caspase cascade.",
    icon: Skull,
    speciesCount: 14,
  },
  {
    pathwayClass: "cell_cycle",
    displayName: "Cell Cycle",
    description:
      "Cyclin-CDK drives G1/S/G2/M transitions; Rb/E2F regulation.",
    icon: RefreshCw,
    speciesCount: 16,
  },
  {
    pathwayClass: "jak_stat",
    displayName: "JAK-STAT",
    description:
      "Cytokine receptor → JAK phosphorylation → STAT dimer nuclear entry.",
    icon: Zap,
    speciesCount: 9,
  },
  {
    pathwayClass: "nf_kappa_b",
    displayName: "NF-κB Signaling",
    description:
      "TNF/TLR → IKK → IκBα degradation → NF-κB nuclear entry; inflammation core.",
    icon: Flame,
    speciesCount: 11,
  },
  {
    pathwayClass: "wnt",
    displayName: "Wnt Signaling",
    description:
      "Wnt → Frizzled/LRP → destruction complex dissociation → β-catenin accumulation.",
    icon: Waves,
    speciesCount: 13,
  },
  {
    pathwayClass: "tgf_beta",
    displayName: "TGF-β Signaling",
    description:
      "TGF-β → TβR → SMAD2/3 phosphorylation → SMAD4 complex nuclear entry.",
    icon: TrendingDown,
    speciesCount: 10,
  },
];

export function PathwaySelector() {
  const router = useRouter();
  const setCurrentPathway = useWorkbenchStore((s) => s.setCurrentPathway);

  const open = (pathwayClass: PathwayClass | null) => {
    setCurrentPathway(pathwayClass);
    router.push(
      pathwayClass ? `/workspace?pathway=${pathwayClass}` : "/workspace"
    );
  };

  return (
    <section className="py-14">
      <div className="mb-6 flex items-end justify-between gap-4 border-b border-zinc-800 pb-3">
        <div>
          <div className="font-mono text-[11px] uppercase tracking-[0.2em] text-zinc-500">
            02 / Pathways
          </div>
          <h2 className="mt-1 text-xl font-semibold tracking-tight text-zinc-100">
            Pathway Selector
          </h2>
          <p className="mt-0.5 text-sm text-zinc-500">
            Pick a signaling pathway to model, or let the agent auto-detect from
            your query.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
        {/* Auto-detect card */}
        <button
          type="button"
          onClick={() => open(null)}
          className="group flex h-full flex-col rounded-lg border border-blue-500/40 bg-blue-500/5 p-4 text-left transition-colors hover:border-blue-500/70 hover:bg-blue-500/10"
        >
          <div className="flex items-center justify-between">
            <span className="flex h-8 w-8 items-center justify-center rounded-md bg-blue-500/15 text-blue-300">
              <Sparkles className="h-4 w-4" />
            </span>
            <span className="font-mono text-[10px] uppercase tracking-wider text-blue-400/70">
              auto
            </span>
          </div>
          <div className="mt-3 text-sm font-medium text-zinc-100">Auto Detect</div>
          <p className="mt-1 flex-1 text-xs leading-relaxed text-zinc-400">
            Let the backend auto-detect the pathway from your natural-language
            query.
          </p>
          <span className="mt-3 inline-flex items-center gap-1 text-xs font-medium text-blue-300">
            Open in Workspace
            <ArrowUpRight className="h-3 w-3 transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
          </span>
        </button>

        {PATHWAYS.map((p) => {
          const Icon = p.icon;
          return (
            <button
              key={p.pathwayClass}
              type="button"
              onClick={() => open(p.pathwayClass)}
              className="group flex h-full flex-col rounded-lg border border-zinc-800 bg-zinc-900/40 p-4 text-left transition-colors hover:border-zinc-700 hover:bg-zinc-900"
            >
              <div className="flex items-center justify-between">
                <span className="flex h-8 w-8 items-center justify-center rounded-md bg-zinc-800 text-zinc-300">
                  <Icon className="h-4 w-4" />
                </span>
                <span className="inline-flex items-center gap-1 rounded-full border border-zinc-700 px-1.5 py-0.5 font-mono text-[10px] text-zinc-400">
                  <Dna className="h-2.5 w-2.5" />
                  {p.speciesCount}
                </span>
              </div>
              <div className="mt-3 text-sm font-medium text-zinc-100">
                {p.displayName}
              </div>
              <p className="mt-1 flex-1 line-clamp-2 text-xs leading-relaxed text-zinc-400">
                {p.description}
              </p>
              <span className="mt-3 inline-flex items-center gap-1 text-xs font-medium text-zinc-300 group-hover:text-blue-300">
                Open in Workspace
                <ArrowUpRight className="h-3 w-3 transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
              </span>
            </button>
          );
        })}
      </div>
    </section>
  );
}
