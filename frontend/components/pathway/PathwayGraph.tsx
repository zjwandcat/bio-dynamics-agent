"use client";

import "@xyflow/react/dist/style.css";
import React, { useCallback, useEffect, useMemo } from "react";
import {
  ReactFlow,
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  Handle,
  Position,
  MarkerType,
  BaseEdge,
  getSmoothStepPath,
  EdgeLabelRenderer,
  useNodesState,
  useEdgesState,
  type Node,
  type Edge,
  type NodeProps,
  type EdgeProps,
  type NodeTypes,
  type EdgeTypes,
  type NodeMouseHandler,
} from "@xyflow/react";
import { cn } from "@/lib/utils";
import type { PathwayGraphData } from "@/lib/api";

// =============================================================================
// Rich pathway graph types
// -----------------------------------------------------------------------------
// The frontend `PathwayGraphData` (lib/api.ts) is a placeholder contract that
// will be refined as C.2–C.13 land. The backend `PathwayGraph` schema
// (backend/app/pathway_graph/schema.py) is much richer: it carries
// species_type, ontology refs (HGNC/UniProt/ChEBI/GO/SBO), states, cross-talk
// and feedback flags. The component below consumes BOTH shapes — every extra
// field is optional, so the simple placeholder payload still renders while the
// rich backend payload lights up the detail panel.
// =============================================================================

export type SpeciesType =
  | "protein"
  | "gene"
  | "complex"
  | "ligand"
  | "drug"
  | "chemical"
  | "mrna"
  | "rna";

export type Compartment =
  | "extracellular"
  | "membrane"
  | "cytoplasm"
  | "nucleus"
  | "mitochondria";

export type EdgeRelation =
  | "activation"
  | "inhibition"
  | "phosphorylation"
  | "binding"
  | "catalysis";

export interface PathwayStateInfo {
  name: string;
  state_type?: string;
  site?: string | null;
}

export interface RichPathwayNode {
  id: string;
  label: string;
  species: string;
  node_type?: "species" | "reaction" | "module" | "perturbation";
  species_type?: SpeciesType;
  compartment?: Compartment;
  // Ontology refs (rich backend payload)
  hgnc_id?: string | null;
  uniprot_id?: string | null;
  chebi_id?: string | null;
  go_terms?: string[];
  sbo_term?: string | null;
  // Provenance / flags
  source_pmid?: string | null;
  is_shared?: boolean;
  shared_with?: string[];
  states?: PathwayStateInfo[];
  /** Rendered as a faint dashed node — used for cross-talk external targets. */
  isExternal?: boolean;
}

export interface RichPathwayEdge {
  source: string;
  target: string;
  relation: EdgeRelation;
  sbo_term?: string;
  is_crosstalk?: boolean;
  is_feedback?: boolean;
}

export interface RichPathwayGraphData {
  pathway_class: string;
  nodes: RichPathwayNode[];
  edges: RichPathwayEdge[];
  modules?: Array<{ id: string; label: string; member_ids: string[] }>;
}

// =============================================================================
// Color & layout maps
// =============================================================================

interface SpeciesColor {
  bg: string;
  border: string;
  text: string;
  label: string;
}

const SPECIES_COLORS: Record<SpeciesType, SpeciesColor> = {
  protein: { bg: "#1e3a5f", border: "#3b82f6", text: "#bfdbfe", label: "Protein" },
  chemical: { bg: "#052e16", border: "#22c55e", text: "#bbf7d0", label: "Chemical" },
  gene: { bg: "#3b0764", border: "#a855f7", text: "#e9d5ff", label: "Gene" },
  complex: { bg: "#431407", border: "#f97316", text: "#fed7aa", label: "Complex" },
  ligand: { bg: "#042f2e", border: "#14b8a6", text: "#99f6e4", label: "Ligand" },
  drug: { bg: "#500724", border: "#ec4899", text: "#fbcfe8", label: "Drug" },
  mrna: { bg: "#1e1b4b", border: "#6366f1", text: "#c7d2fe", label: "mRNA" },
  rna: { bg: "#1e1b4b", border: "#6366f1", text: "#c7d2fe", label: "RNA" },
};

const EDGE_COLORS: Record<EdgeRelation, string> = {
  activation: "#22c55e",
  inhibition: "#ef4444",
  phosphorylation: "#3b82f6",
  binding: "#a1a1aa",
  catalysis: "#f59e0b",
};

const EDGE_LABELS: Record<EdgeRelation, string> = {
  activation: "activation",
  inhibition: "inhibition",
  phosphorylation: "phospho",
  binding: "binding",
  catalysis: "catalysis",
};

const COMPARTMENT_ORDER: Compartment[] = [
  "extracellular",
  "membrane",
  "cytoplasm",
  "nucleus",
  "mitochondria",
];

const COMPARTMENT_Y: Record<Compartment, number> = {
  extracellular: 0,
  membrane: 150,
  cytoplasm: 320,
  nucleus: 510,
  mitochondria: 680,
};

const COMPARTMENT_BG: Record<Compartment, string> = {
  extracellular: "rgba(148, 163, 184, 0.05)",
  membrane: "rgba(251, 191, 36, 0.05)",
  cytoplasm: "rgba(59, 130, 246, 0.04)",
  nucleus: "rgba(168, 85, 247, 0.05)",
  mitochondria: "rgba(236, 72, 153, 0.05)",
};

const COMPARTMENT_LABEL: Record<Compartment, string> = {
  extracellular: "EXTRACELLULAR",
  membrane: "MEMBRANE",
  cytoplasm: "CYTOPLASM",
  nucleus: "NUCLEUS",
  mitochondria: "MITOCHONDRIA",
};

const NODE_SPACING_X = 175;
const COMPARTMENT_HEIGHT = 120;

// =============================================================================
// Fallback EGFR pathway graph (used when /api/v4 backend is unavailable)
// -----------------------------------------------------------------------------
// Topology mirrors backend/app/pathway_graph/initializer.py EGFR_RTK so the
// workbench shows a meaningful graph even before the backend route lands.
// =============================================================================

export function getFallbackEGFRGraph(): RichPathwayGraphData {
  return {
    pathway_class: "egfr",
    nodes: [
      {
        id: "PN_EGF",
        label: "EGF",
        species: "EGF",
        species_type: "ligand",
        compartment: "extracellular",
        chebi_id: "CHEBI:64854",
        sbo_term: "SBO:0000280",
        go_terms: ["GO:0008083", "GO:0038153"],
        source_pmid: "PMID:16543144",
      },
      {
        id: "PN_EGFR",
        label: "EGFR",
        species: "EGFR",
        species_type: "protein",
        compartment: "membrane",
        hgnc_id: "HGNC:3236",
        uniprot_id: "P00533",
        sbo_term: "SBO:0000252",
        go_terms: ["GO:0005006", "GO:0007173", "GO:0007165"],
        source_pmid: "PMID:16543144",
        is_shared: true,
        shared_with: ["pi3k_akt_mtor"],
      },
      {
        id: "PN_pEGFR",
        label: "pEGFR",
        species: "pEGFR",
        species_type: "protein",
        compartment: "membrane",
        hgnc_id: "HGNC:3236",
        uniprot_id: "P00533",
        sbo_term: "SBO:0000216",
        go_terms: ["GO:0005006", "GO:0007173"],
        states: [
          { name: "phosphorylated", state_type: "phosphorylation", site: "Tyr1068" },
        ],
        source_pmid: "PMID:16543144",
        is_shared: true,
        shared_with: ["pi3k_akt_mtor"],
      },
      {
        id: "PN_Shc",
        label: "Shc",
        species: "Shc",
        species_type: "protein",
        compartment: "cytoplasm",
        hgnc_id: "HGNC:7946",
        uniprot_id: "P29353",
        go_terms: ["GO:0007265", "GO:0007165"],
      },
      {
        id: "PN_pShc",
        label: "pShc",
        species: "pShc",
        species_type: "protein",
        compartment: "cytoplasm",
        hgnc_id: "HGNC:7946",
        uniprot_id: "P29353",
        states: [{ name: "phosphorylated", state_type: "phosphorylation" }],
      },
      {
        id: "PN_Grb2",
        label: "Grb2",
        species: "Grb2",
        species_type: "protein",
        compartment: "cytoplasm",
        hgnc_id: "HGNC:2447",
        uniprot_id: "P62993",
        go_terms: ["GO:0007265"],
      },
      {
        id: "PN_SOS",
        label: "SOS1",
        species: "SOS",
        species_type: "protein",
        compartment: "cytoplasm",
        hgnc_id: "HGNC:11187",
        uniprot_id: "Q07889",
        go_terms: ["GO:0005085", "GO:0007265"],
      },
      {
        id: "PN_Ras",
        label: "Ras",
        species: "Ras",
        species_type: "protein",
        compartment: "membrane",
        hgnc_id: "HGNC:7989",
        uniprot_id: "P01116",
        is_shared: true,
        shared_with: ["mapk"],
      },
      {
        id: "PN_RasGTP",
        label: "RasGTP",
        species: "RasGTP",
        species_type: "protein",
        compartment: "membrane",
        hgnc_id: "HGNC:7989",
        uniprot_id: "P01116",
        states: [{ name: "active", state_type: "conformational" }],
        is_shared: true,
        shared_with: ["mapk"],
      },
      {
        id: "PN_Raf",
        label: "Raf",
        species: "Raf",
        species_type: "protein",
        compartment: "cytoplasm",
        hgnc_id: "HGNC:9830",
        uniprot_id: "P04049",
        is_shared: true,
        shared_with: ["mapk"],
      },
      {
        id: "PN_pRaf",
        label: "pRaf",
        species: "pRaf",
        species_type: "protein",
        compartment: "cytoplasm",
        hgnc_id: "HGNC:9830",
        uniprot_id: "P04049",
        states: [{ name: "phosphorylated", state_type: "phosphorylation", site: "Ser259" }],
      },
      {
        id: "PN_PI3K_ext",
        label: "PI3K →",
        species: "PI3K",
        species_type: "protein",
        compartment: "cytoplasm",
        isExternal: true,
      },
    ],
    edges: [
      { source: "PN_EGF", target: "PN_EGFR", relation: "binding" },
      { source: "PN_EGFR", target: "PN_pEGFR", relation: "phosphorylation" },
      { source: "PN_pEGFR", target: "PN_Shc", relation: "binding" },
      { source: "PN_Shc", target: "PN_pShc", relation: "phosphorylation" },
      { source: "PN_pEGFR", target: "PN_Grb2", relation: "binding" },
      { source: "PN_Grb2", target: "PN_SOS", relation: "binding" },
      { source: "PN_SOS", target: "PN_RasGTP", relation: "catalysis" },
      { source: "PN_RasGTP", target: "PN_pRaf", relation: "phosphorylation" },
      {
        source: "PN_pEGFR",
        target: "PN_PI3K_ext",
        relation: "activation",
        is_crosstalk: true,
      },
    ],
  };
}

// =============================================================================
// Helpers
// =============================================================================

function inferSpeciesType(node: RichPathwayNode): SpeciesType {
  if (node.species_type) return node.species_type;
  const name = node.label || node.species || "";
  if (/complex|cdk\d.*_/i.test(name) || name.includes("_Cdk")) return "complex";
  if (/^mRNA|_mRNA$/i.test(name)) return "mrna";
  if (node.node_type === "perturbation") return "drug";
  return "protein";
}

function resolveSpeciesType(node: RichPathwayNode): SpeciesType {
  return inferSpeciesType(node);
}

/**
 * Compartment-based auto-layout.
 *
 * dagre is not installed in this project, so we use a biologically-meaningful
 * built-in layout: nodes are grouped by compartment (extracellular at the top,
 * then membrane, cytoplasm, nucleus, mitochondria) and spread horizontally
 * within each compartment band. Compartment background rectangles are emitted
 * as low-z-index group nodes.
 */
function layoutGraph(graph: RichPathwayGraphData): {
  nodes: Node[];
  edges: Edge[];
  compartments: Node[];
} {
  const byCompartment = new Map<Compartment, RichPathwayNode[]>();
  for (const n of graph.nodes) {
    const c = (n.compartment as Compartment) ?? "cytoplasm";
    if (!byCompartment.has(c)) byCompartment.set(c, []);
    byCompartment.get(c)!.push(n);
  }

  const speciesNodes: Node[] = [];
  const compartmentNodes: Node[] = [];

  for (const comp of COMPARTMENT_ORDER) {
    const group = byCompartment.get(comp);
    if (!group || group.length === 0) continue;

    const count = group.length;
    const bandWidth = Math.max(count * NODE_SPACING_X, 440);
    const startX = -bandWidth / 2;
    const y = COMPARTMENT_Y[comp];

    group.forEach((n, i) => {
      const x = startX + i * NODE_SPACING_X + NODE_SPACING_X / 2;
      speciesNodes.push({
        id: n.id,
        type: "species",
        position: { x, y },
        data: {
          label: n.label,
          speciesType: resolveSpeciesType(n),
          compartment: comp,
          isShared: n.is_shared === true,
          isExternal: n.isExternal === true,
        },
        draggable: true,
      });
    });

    compartmentNodes.push({
      id: `compartment-${comp}`,
      type: "compartment",
      position: { x: startX - 30, y: y - 38 },
      data: {
        label: COMPARTMENT_LABEL[comp],
        width: bandWidth + 60,
        height: COMPARTMENT_HEIGHT,
        color: COMPARTMENT_BG[comp],
        isCompartment: true,
      },
      draggable: false,
      selectable: false,
      zIndex: -1,
    });
  }

  const edges: Edge[] = graph.edges.map((e, i) => {
    const relation = e.relation ?? "binding";
    const color = EDGE_COLORS[relation] ?? "#a1a1aa";
    return {
      id: `e-${i}-${e.source}-${e.target}`,
      source: e.source,
      target: e.target,
      type: "reaction",
      data: {
        relation,
        isCrosstalk: e.is_crosstalk === true,
        isFeedback: e.is_feedback === true,
        label: EDGE_LABELS[relation] ?? relation,
      },
      markerEnd: { type: MarkerType.ArrowClosed, color, width: 16, height: 16 },
    };
  });

  return { nodes: speciesNodes, edges, compartments: compartmentNodes };
}

// =============================================================================
// Custom node: species
// =============================================================================

interface SpeciesNodeData {
  label: string;
  speciesType: SpeciesType;
  compartment?: Compartment;
  isShared?: boolean;
  isExternal?: boolean;
  // compartment-only fields (shared data shape for a single NodeTypes map)
  width?: number;
  height?: number;
  color?: string;
  isCompartment?: boolean;
}

function SpeciesNode({ data, selected }: NodeProps) {
  const d = data as unknown as SpeciesNodeData;
  const colors = SPECIES_COLORS[d.speciesType ?? "protein"];
  const isRect = d.speciesType === "complex";
  return (
    <>
      <Handle type="target" position={Position.Top} style={{ opacity: 0 }} />
      <Handle type="target" position={Position.Left} style={{ opacity: 0 }} />
      <div
        className={cn(
          "flex items-center justify-center border-2 px-3 py-1.5 text-center text-[11px] font-semibold shadow-sm transition-all",
          isRect ? "rounded-lg" : "rounded-full",
          d.isExternal && "border-dashed opacity-70"
        )}
        style={{
          background: colors.bg,
          borderColor: colors.border,
          color: colors.text,
          minWidth: isRect ? 84 : 64,
          ...(selected
            ? { boxShadow: `0 0 0 3px ${colors.border}, 0 0 12px ${colors.border}66` }
            : {}),
        }}
        title={
          d.isShared
            ? `${d.label} — ${colors.label} (shared species)`
            : `${d.label} — ${colors.label}`
        }
      >
        <span className="truncate">{d.label}</span>
        {d.isShared && (
          <span className="ml-1 text-[9px] opacity-80" aria-hidden>
            ⇄
          </span>
        )}
      </div>
      <Handle type="source" position={Position.Bottom} style={{ opacity: 0 }} />
      <Handle type="source" position={Position.Right} style={{ opacity: 0 }} />
    </>
  );
}

// =============================================================================
// Custom node: compartment background
// =============================================================================

function CompartmentNode({ data }: NodeProps) {
  const d = data as unknown as SpeciesNodeData;
  return (
    <div
      className="flex items-start rounded-xl border border-dashed p-2"
      style={{
        width: d.width,
        height: d.height,
        background: d.color,
        borderColor: "rgba(255,255,255,0.08)",
      }}
    >
      <span className="text-[10px] font-semibold tracking-wider text-zinc-500 uppercase">
        {d.label}
      </span>
    </div>
  );
}

// =============================================================================
// Custom edge: reaction
// =============================================================================

interface ReactionEdgeData {
  relation: EdgeRelation;
  isCrosstalk?: boolean;
  isFeedback?: boolean;
  label: string;
}

function ReactionEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  data,
  markerEnd,
}: EdgeProps) {
  const d = (data ?? {}) as unknown as ReactionEdgeData;
  const color = EDGE_COLORS[d.relation ?? "binding"] ?? "#a1a1aa";
  const [edgePath, labelX, labelY] = getSmoothStepPath({
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
    borderRadius: 8,
  });

  const dasharray = d.isCrosstalk
    ? "6 4"
    : d.isFeedback
      ? "2 4"
      : undefined;

  return (
    <>
      <BaseEdge
        id={id}
        path={edgePath}
        markerEnd={markerEnd}
        style={{
          stroke: color,
          strokeWidth: d.isCrosstalk ? 1.4 : 2,
          strokeDasharray: dasharray,
          opacity: d.isCrosstalk ? 0.75 : 1,
        }}
      />
      <EdgeLabelRenderer>
        <div
          style={{
            position: "absolute",
            transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)`,
            pointerEvents: "none",
            borderColor: `${color}66`,
          }}
          className="rounded border bg-zinc-900/90 px-1 py-0.5 text-[9px] font-medium text-zinc-300"
        >
          {d.label}
        </div>
      </EdgeLabelRenderer>
    </>
  );
}

// =============================================================================
// nodeTypes / edgeTypes (module scope so ReactFlow doesn't warn on re-creation)
// =============================================================================

const nodeTypes: NodeTypes = {
  species: SpeciesNode,
  compartment: CompartmentNode,
};

const edgeTypes: EdgeTypes = {
  reaction: ReactionEdge,
};

function miniMapNodeColor(node: Node): string {
  const d = node.data as unknown as SpeciesNodeData | undefined;
  if (d?.isCompartment) return "#27272a";
  return SPECIES_COLORS[(d?.speciesType ?? "protein") as SpeciesType]?.border ?? "#3b82f6";
}

// =============================================================================
// PathwayGraph component
// =============================================================================

export interface PathwayGraphProps {
  graph: RichPathwayGraphData | null;
  /** Called when a species node is clicked (opens the detail panel). */
  onNodeClick?: (nodeId: string) => void;
  className?: string;
  style?: React.CSSProperties;
}

export function PathwayGraph({
  graph,
  onNodeClick,
  className,
  style,
}: PathwayGraphProps) {
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);

  // Rebuild the flow whenever the graph payload changes.
  useEffect(() => {
    if (!graph) {
      setNodes([]);
      setEdges([]);
      return;
    }
    const { nodes: speciesNodes, edges: reactionEdges, compartments } =
      layoutGraph(graph);
    setNodes([...compartments, ...speciesNodes]);
    setEdges(reactionEdges);
  }, [graph, setNodes, setEdges]);

  const handleNodeClick = useCallback<NodeMouseHandler<Node>>(
    (_event, node) => {
      if (node.type === "species") onNodeClick?.(node.id);
    },
    [onNodeClick]
  );

  const hasGraph = (graph?.nodes.length ?? 0) > 0;

  // Legend entries (memoised so ReactFlow panel doesn't re-render needlessly).
  const legend = useMemo(
    () =>
      (["protein", "chemical", "gene", "complex", "ligand", "drug"] as SpeciesType[]).map(
        (t) => ({ type: t, color: SPECIES_COLORS[t] })
      ),
    []
  );

  const edgeLegend = useMemo(
    () =>
      (["activation", "inhibition", "phosphorylation", "binding", "catalysis"] as EdgeRelation[]).map(
        (r) => ({ relation: r, color: EDGE_COLORS[r] })
      ),
    []
  );

  return (
    <div
      className={cn("relative h-full w-full bg-zinc-950", className)}
      style={style}
    >
      {hasGraph ? (
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onNodeClick={handleNodeClick}
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
          fitView
          fitViewOptions={{ padding: 0.18 }}
          minZoom={0.2}
          maxZoom={2}
          proOptions={{ hideAttribution: true }}
          colorMode="dark"
          nodesDraggable
          nodesConnectable={false}
          elementsSelectable
        >
          <Background
            variant={BackgroundVariant.Dots}
            gap={16}
            size={1}
            color="#27272a"
          />
          <Controls
            className="!border-zinc-800 !bg-zinc-900/90 !text-zinc-300"
            showInteractive={false}
          />
          <MiniMap
            pannable
            zoomable
            className="!bg-zinc-900 !border-zinc-800"
            maskColor="rgba(9, 9, 11, 0.7)"
            nodeColor={miniMapNodeColor}
            nodeStrokeWidth={2}
          />
          {/* Legend */}
          <div className="pointer-events-none absolute left-2 top-2 z-10 flex flex-col gap-1.5 rounded-lg border border-zinc-800 bg-zinc-900/85 p-2 text-[10px] text-zinc-300 backdrop-blur">
            <div className="font-semibold tracking-wide text-zinc-400 uppercase">
              Species
            </div>
            <div className="grid grid-cols-2 gap-x-2 gap-y-1">
              {legend.map((l) => (
                <div key={l.type} className="flex items-center gap-1.5">
                  <span
                    className="inline-block h-2.5 w-2.5 rounded-full"
                    style={{ background: l.color.border }}
                  />
                  <span className="text-zinc-400">{l.color.label}</span>
                </div>
              ))}
            </div>
            <div className="mt-1 font-semibold tracking-wide text-zinc-400 uppercase">
              Reaction
            </div>
            <div className="grid grid-cols-2 gap-x-2 gap-y-1">
              {edgeLegend.map((l) => (
                <div key={l.relation} className="flex items-center gap-1.5">
                  <span
                    className="inline-block h-0.5 w-4 rounded"
                    style={{ background: l.color }}
                  />
                  <span className="text-zinc-400">{l.relation}</span>
                </div>
              ))}
            </div>
            <div className="mt-1 flex items-center gap-1.5">
              <span
                className="inline-block h-0.5 w-4 rounded border-t border-dashed border-zinc-400"
                style={{ borderColor: "#a1a1aa" }}
              />
              <span className="text-zinc-400">cross-talk</span>
            </div>
          </div>
        </ReactFlow>
      ) : (
        <div className="flex h-full w-full items-center justify-center p-6 text-center">
          <div className="max-w-xs space-y-1">
            <p className="text-sm text-zinc-400">No pathway graph loaded</p>
            <p className="text-xs text-zinc-600">
              Select a pathway class from the header dropdown to render its
              interactive mechanism graph.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

/** Re-export for the store-side hydration contract (PathwayGraphData). */
export type { PathwayGraphData };
