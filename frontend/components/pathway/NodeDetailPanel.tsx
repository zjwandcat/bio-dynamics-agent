"use client";

import React, { useMemo } from "react";
import { X, ExternalLink, FileText, FlaskConical, BookOpen } from "lucide-react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";
import type {
  RichPathwayNode,
  RichPathwayEdge,
  RichPathwayGraphData,
  EdgeRelation,
  SpeciesType,
} from "./PathwayGraph";

// =============================================================================
// Display helpers
// =============================================================================

const SPECIES_LABEL: Record<SpeciesType, string> = {
  protein: "Protein",
  chemical: "Chemical",
  gene: "Gene",
  complex: "Complex",
  ligand: "Ligand",
  drug: "Drug",
  mrna: "mRNA",
  rna: "RNA",
};

const RELATION_LABEL: Record<EdgeRelation, string> = {
  activation: "Activation",
  inhibition: "Inhibition",
  phosphorylation: "Phosphorylation",
  binding: "Binding",
  catalysis: "Catalysis",
};

const RELATION_COLOR: Record<EdgeRelation, string> = {
  activation: "text-emerald-400",
  inhibition: "text-red-400",
  phosphorylation: "text-blue-400",
  binding: "text-zinc-400",
  catalysis: "text-amber-400",
};

function Truthy({ value, children }: { value: unknown; children: React.ReactNode }) {
  if (value === null || value === undefined || value === "") return null;
  return <>{children}</>;
}

function Field({
  label,
  value,
  mono,
}: {
  label: string;
  value: React.ReactNode;
  mono?: boolean;
}) {
  const empty =
    value === null ||
    value === undefined ||
    value === "" ||
    (Array.isArray(value) && value.length === 0);
  return (
    <div className="flex flex-col gap-0.5 py-1">
      <span className="text-[10px] font-medium tracking-wide text-zinc-500 uppercase">
        {label}
      </span>
      <span
        className={cn(
          "text-xs text-zinc-200",
          mono && "font-mono text-[11px]",
          empty && "text-zinc-600"
        )}
      >
        {empty ? "—" : value}
      </span>
    </div>
  );
}

// =============================================================================
// Reaction participation
// =============================================================================

interface ReactionRow {
  edge: RichPathwayEdge;
  partnerId: string;
  partnerLabel: string;
  /** Role of the selected species within this reaction. */
  role: "reactant" | "product" | "modifier";
}

/**
 * Classify a species' role in a reaction edge. In the simplified pathway graph
 * every edge is binary (source → target); we treat the source as the
 * reactant/modifier and the target as the product. Inhibition / catalysis are
 * modifier roles; phosphorylation / binding / activation are reactant→product.
 */
function classifyRole(
  edge: RichPathwayEdge,
  selectedId: string
): "reactant" | "product" | "modifier" {
  const isSource = edge.source === selectedId;
  if (edge.relation === "inhibition" || edge.relation === "catalysis") {
    // The source acts as a modifier on the target.
    return isSource ? "modifier" : "product";
  }
  return isSource ? "reactant" : "product";
}

// =============================================================================
// NodeDetailPanel
// =============================================================================

export interface NodeDetailPanelProps {
  node: RichPathwayNode | null;
  graph: RichPathwayGraphData | null;
  onClose: () => void;
  className?: string;
}

export function NodeDetailPanel({
  node,
  graph,
  onClose,
  className,
}: NodeDetailPanelProps) {
  // Reactions the selected species participates in.
  const reactions = useMemo<ReactionRow[]>(() => {
    if (!node || !graph) return [];
    const labelById = new Map(graph.nodes.map((n) => [n.id, n.label]));
    const rows: ReactionRow[] = [];
    for (const edge of graph.edges) {
      if (edge.source !== node.id && edge.target !== node.id) continue;
      const partnerId = edge.source === node.id ? edge.target : edge.source;
      rows.push({
        edge,
        partnerId,
        partnerLabel: labelById.get(partnerId) ?? partnerId,
        role: classifyRole(edge, node.id),
      });
    }
    return rows;
  }, [node, graph]);

  // Evidence rows: prefer the node's own source_pmid; fall back to PMIDs cited
  // on incident reaction edges (rich backend payload carries source_pmid/edge).
  const evidence = useMemo(() => {
    const rows: Array<{ pmid: string; source: string; detail: string }> = [];
    if (node?.source_pmid) {
      rows.push({
        pmid: node.source_pmid,
        source: "Pathway registry",
        detail: `${node.label} initial topology`,
      });
    }
    if (graph && node) {
      for (const edge of graph.edges) {
        if (edge.source !== node.id && edge.target !== node.id) continue;
        const pmid = (edge as RichPathwayEdge & { source_pmid?: string })
          .source_pmid;
        if (pmid && !rows.some((r) => r.pmid === pmid)) {
          rows.push({
            pmid,
            source: "Reaction edge",
            detail: `${RELATION_LABEL[edge.relation]} → ${
              edge.source === node.id ? "downstream" : "upstream"
            } partner`,
          });
        }
      }
    }
    return rows;
  }, [node, graph]);

  if (!node) return null;

  const speciesLabel = SPECIES_LABEL[node.species_type ?? "protein"];

  return (
    <Card
      className={cn(
        "flex flex-col gap-0 overflow-hidden rounded-lg border-zinc-800 bg-zinc-900/95 p-0 ring-zinc-800 backdrop-blur",
        className
      )}
    >
      {/* Header */}
      <CardHeader className="flex-row items-start justify-between gap-2 border-b border-zinc-800 py-2.5 pr-2">
        <div className="flex min-w-0 flex-col gap-1">
          <CardTitle className="flex items-center gap-2 text-sm">
            <span className="truncate text-zinc-100">{node.label}</span>
            {node.is_shared && (
              <Badge
                variant="outline"
                className="border-amber-700/50 text-[9px] text-amber-300"
              >
                shared
              </Badge>
            )}
          </CardTitle>
          <div className="flex items-center gap-1.5 text-[10px] text-zinc-500">
            <span className="rounded bg-zinc-800 px-1.5 py-0.5 text-zinc-400">
              {speciesLabel}
            </span>
            {node.compartment && (
              <span className="rounded bg-zinc-800 px-1.5 py-0.5 text-zinc-400">
                {node.compartment}
              </span>
            )}
            <span className="truncate font-mono text-zinc-600">{node.id}</span>
          </div>
        </div>
        <Button
          variant="ghost"
          size="icon-xs"
          onClick={onClose}
          title="Close detail panel"
          className="text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200"
        >
          <X className="h-3.5 w-3.5" />
        </Button>
      </CardHeader>

      {/* Tabs */}
      <Tabs
        defaultValue="ontology"
        className="flex min-h-0 flex-1 flex-col"
      >
        <div className="shrink-0 border-b border-zinc-800 px-2 py-1.5">
          <TabsList>
            <TabsTrigger value="ontology" className="gap-1">
              <BookOpen className="h-3 w-3" />
              Ontology
            </TabsTrigger>
            <TabsTrigger value="reaction" className="gap-1">
              <FlaskConical className="h-3 w-3" />
              Reaction
            </TabsTrigger>
            <TabsTrigger value="evidence" className="gap-1">
              <FileText className="h-3 w-3" />
              Evidence
            </TabsTrigger>
          </TabsList>
        </div>

        <ScrollArea className="min-h-0 flex-1">
          {/* ---- Ontology tab ---- */}
          <TabsContent value="ontology" className="px-3 py-2">
            <CardContent className="flex flex-col gap-0 px-0">
              <Field label="Species name" value={node.species || node.label} />
              <Field
                label="Type"
                value={
                  <span className="rounded bg-zinc-800 px-1.5 py-0.5 text-[11px] text-zinc-300">
                    {speciesLabel}
                  </span>
                }
              />
              <Field
                label="Compartment"
                value={node.compartment ?? "—"}
              />
              <Field
                label="HGNC ID"
                value={node.hgnc_id}
                mono
              />
              <Field
                label="UniProt ID"
                value={
                  node.uniprot_id ? (
                    <span
                      className="inline-flex items-center gap-1 font-mono text-[11px] text-blue-300"
                    >
                      {node.uniprot_id}
                    </span>
                  ) : (
                    "—"
                  )
                }
              />
              <Field
                label="ChEBI ID"
                value={node.chebi_id}
                mono
              />
              <Field label="SBO term" value={node.sbo_term} mono />
              <Field
                label="GO terms"
                value={
                  node.go_terms && node.go_terms.length > 0 ? (
                    <div className="flex flex-wrap gap-1">
                      {node.go_terms.map((g) => (
                        <span
                          key={g}
                          className="rounded bg-zinc-800 px-1.5 py-0.5 font-mono text-[10px] text-zinc-300"
                        >
                          {g}
                        </span>
                      ))}
                    </div>
                  ) : (
                    "—"
                  )
                }
              />
              <Field
                label="States"
                value={
                  node.states && node.states.length > 0 ? (
                    <div className="flex flex-col gap-1">
                      {node.states.map((s, i) => (
                        <div
                          key={`${s.name}-${i}`}
                          className="flex items-center gap-1.5 text-[11px]"
                        >
                          <span className="rounded bg-purple-900/40 px-1.5 py-0.5 text-purple-300">
                            {s.state_type ?? "state"}
                          </span>
                          <span className="text-zinc-300">{s.name}</span>
                          <Truthy value={s.site}>
                            <span className="font-mono text-[10px] text-zinc-500">
                              @{s.site}
                            </span>
                          </Truthy>
                        </div>
                      ))}
                    </div>
                  ) : (
                    "—"
                  )
                }
              />
              {node.is_shared && node.shared_with && node.shared_with.length > 0 && (
                <Field
                  label="Shared with"
                  value={
                    <div className="flex flex-wrap gap-1">
                      {node.shared_with.map((p) => (
                        <span
                          key={p}
                          className="rounded bg-amber-900/40 px-1.5 py-0.5 text-[10px] text-amber-300"
                        >
                          {p}
                        </span>
                      ))}
                    </div>
                  }
                />
              )}
            </CardContent>
          </TabsContent>

          {/* ---- Reaction tab ---- */}
          <TabsContent value="reaction" className="px-3 py-2">
            <CardContent className="flex flex-col gap-2 px-0">
              <p className="text-[10px] text-zinc-500">
                {reactions.length === 0
                  ? "No reactions recorded for this species."
                  : `${reactions.length} reaction${
                      reactions.length === 1 ? "" : "s"
                    } involving ${node.label}.`}
              </p>
              <div className="flex flex-col gap-1.5">
                {reactions.map((r, i) => {
                  const rel = r.edge.relation;
                  return (
                    <div
                      key={`${r.edge.source}-${r.edge.target}-${i}`}
                      className="rounded-md border border-zinc-800 bg-zinc-900/60 px-2 py-1.5"
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span
                          className={cn(
                            "text-[11px] font-semibold",
                            RELATION_COLOR[rel]
                          )}
                        >
                          {RELATION_LABEL[rel]}
                        </span>
                        <Badge
                          variant="outline"
                          className="border-zinc-700 text-[9px] text-zinc-400"
                        >
                          {r.role}
                        </Badge>
                      </div>
                      <div className="mt-1 flex items-center gap-1.5 text-[11px] text-zinc-300">
                        <span className="truncate">{node.label}</span>
                        <span className="text-zinc-600">
                          {r.role === "modifier" ? " ⊣ " : " → "}
                        </span>
                        <span className="truncate font-medium text-zinc-200">
                          {r.partnerLabel}
                        </span>
                      </div>
                      <div className="mt-0.5 font-mono text-[9px] text-zinc-600">
                        {r.partnerId}
                      </div>
                      {(r.edge.is_crosstalk || r.edge.is_feedback) && (
                        <div className="mt-1 flex gap-1">
                          {r.edge.is_crosstalk && (
                            <span className="rounded bg-zinc-800 px-1 py-0.5 text-[9px] text-zinc-400">
                              cross-talk
                            </span>
                          )}
                          {r.edge.is_feedback && (
                            <span className="rounded bg-zinc-800 px-1 py-0.5 text-[9px] text-zinc-400">
                              feedback
                            </span>
                          )}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </CardContent>
          </TabsContent>

          {/* ---- Evidence tab ---- */}
          <TabsContent value="evidence" className="px-3 py-2">
            <CardContent className="flex flex-col gap-2 px-0">
              <p className="text-[10px] text-zinc-500">
                {evidence.length === 0
                  ? "No literature references attached to this species."
                  : `${evidence.length} reference${
                      evidence.length === 1 ? "" : "s"
                    }.`}
              </p>
              <div className="flex flex-col gap-1.5">
                {evidence.map((e, i) => (
                  <div
                    key={`${e.pmid}-${i}`}
                    className="rounded-md border border-zinc-800 bg-zinc-900/60 px-2 py-1.5"
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="inline-flex items-center gap-1 font-mono text-[11px] text-blue-300">
                        <ExternalLink className="h-3 w-3" />
                        {e.pmid}
                      </span>
                      <span className="text-[9px] text-zinc-500">
                        {e.source}
                      </span>
                    </div>
                    <div className="mt-1 text-[11px] text-zinc-400">
                      {e.detail}
                    </div>
                  </div>
                ))}
                {evidence.length === 0 && (
                  <div className="rounded-md border border-dashed border-zinc-800 px-2 py-3 text-center text-[11px] text-zinc-600">
                    No PMID citations available for this node.
                  </div>
                )}
              </div>
            </CardContent>
          </TabsContent>
        </ScrollArea>
      </Tabs>
    </Card>
  );
}
