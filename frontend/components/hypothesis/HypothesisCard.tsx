"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  ChevronDown,
  Lightbulb,
  FileText,
  TrendingUp,
  FlaskConical,
  ShieldCheck,
  ExternalLink,
  AlertCircle,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import {
  deriveConfidence,
  deriveFalsifiability,
  isGroundedPmid,
  normalizePmid,
  type FalsifiabilityCheck,
  type Hypothesis,
  type HypothesisStrategy,
} from "./types";
import { ExperimentCard } from "./ExperimentCard";

interface HypothesisCardProps {
  hypothesis: Hypothesis;
  className?: string;
  defaultExpanded?: boolean;
}

const STRATEGY_META: Record<
  HypothesisStrategy,
  { label: string; className: string }
> = {
  oscillation: {
    label: "oscillation",
    className: "border-purple-700/50 bg-purple-500/10 text-purple-300",
  },
  bistability: {
    label: "bistability",
    className: "border-amber-700/50 bg-amber-500/10 text-amber-300",
  },
  sensitivity: {
    label: "sensitivity",
    className: "border-cyan-700/50 bg-cyan-500/10 text-cyan-300",
  },
};

/**
 * Individual hypothesis display card (Task C.8 / Step 3).
 *
 * Collapsed: hypothesis text + strategy badge + confidence bar + falsifiability
 * status. Expanded: 5 accordion sections — Hypothesis / Evidence / Predictions
 * / Suggested Experiments / Falsifiability.
 */
export function HypothesisCard({
  hypothesis,
  className,
  defaultExpanded = false,
}: HypothesisCardProps) {
  const [expanded, setExpanded] = useState(defaultExpanded);

  const strategy = hypothesis.strategy;
  const strategyMeta = strategy ? STRATEGY_META[strategy] : null;
  const falsifiability = deriveFalsifiability(hypothesis);
  const confidence = deriveConfidence(hypothesis);
  const supporting = hypothesis.supporting_pmids ?? [];
  const contradicting =
    hypothesis.contradicting_pmids ?? hypothesis.falsifying_pmids ?? [];
  const experiment = hypothesis.experiment_design;

  return (
    <div
      className={cn(
        "rounded-lg border border-zinc-800 bg-zinc-900/60",
        className
      )}
    >
      {/* Summary header */}
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        className="flex w-full items-start justify-between gap-2 px-3 py-2 text-left"
      >
        <div className="flex min-w-0 items-start gap-2">
          <Lightbulb className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber-400" />
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-1.5">
              {hypothesis.id && (
                <span className="font-mono text-[10px] text-zinc-500">
                  {hypothesis.id}
                </span>
              )}
              {strategyMeta && (
                <Badge
                  variant="outline"
                  className={cn("h-4 text-[9px]", strategyMeta.className)}
                >
                  {strategyMeta.label}
                </Badge>
              )}
              <FalsifiabilityBadge falsifiable={falsifiability.overall} />
            </div>
            <p className="mt-1 text-xs leading-snug text-zinc-200">
              {hypothesis.statement ?? "(no statement)"}
            </p>
          </div>
        </div>
        <ChevronDown
          className={cn(
            "mt-1 h-4 w-4 shrink-0 text-zinc-500 transition-transform",
            expanded && "rotate-180"
          )}
        />
      </button>

      {/* Confidence bar */}
      <div className="px-3 pb-2">
        <ConfidenceBar value={confidence} />
      </div>

      {/* Expandable details: 5 accordion sections */}
      <AnimatePresence initial={false}>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="space-y-1 border-t border-zinc-800 px-3 py-2">
              {/* 1. Hypothesis */}
              <Section
                title="Hypothesis"
                icon={<Lightbulb className="h-3 w-3" />}
              >
                <p className="text-[11px] leading-relaxed text-zinc-300">
                  {hypothesis.statement ?? "—"}
                </p>
                <div className="mt-1.5 flex flex-wrap gap-x-3 gap-y-0.5 text-[10px] text-zinc-500">
                  {hypothesis.target_species && (
                    <span>
                      <span className="text-zinc-600">target:</span>{" "}
                      {hypothesis.target_species}
                    </span>
                  )}
                  {hypothesis.feedback_node && (
                    <span>
                      <span className="text-zinc-600">feedback:</span>{" "}
                      {hypothesis.feedback_node}
                    </span>
                  )}
                  {hypothesis.threshold_node && (
                    <span>
                      <span className="text-zinc-600">threshold node:</span>{" "}
                      {hypothesis.threshold_node}
                    </span>
                  )}
                  {hypothesis.target_param && (
                    <span>
                      <span className="text-zinc-600">param:</span>{" "}
                      {hypothesis.target_param}
                    </span>
                  )}
                  {typeof hypothesis.sensitivity === "number" && (
                    <span>
                      <span className="text-zinc-600">|S|:</span>{" "}
                      {Math.abs(hypothesis.sensitivity).toFixed(3)}
                    </span>
                  )}
                </div>
                {hypothesis.validation_method && (
                  <p className="mt-1.5 text-[10px] text-zinc-500">
                    <span className="text-zinc-600">Validation:</span>{" "}
                    {hypothesis.validation_method}
                  </p>
                )}
              </Section>

              {/* 2. Evidence */}
              <Section title="Evidence" icon={<FileText className="h-3 w-3" />}>
                {supporting.length === 0 && contradicting.length === 0 ? (
                  <p className="text-[11px] text-zinc-600">
                    No literature evidence linked yet.
                  </p>
                ) : (
                  <div className="space-y-2">
                    <PmidList
                      label="Supporting"
                      pmids={supporting}
                      tone="support"
                    />
                    <PmidList
                      label="Contradicting"
                      pmids={contradicting}
                      tone="contradict"
                    />
                  </div>
                )}
              </Section>

              {/* 3. Predictions */}
              <Section
                title="Predictions"
                icon={<TrendingUp className="h-3 w-3" />}
              >
                <p className="text-[11px] leading-relaxed text-zinc-300">
                  {hypothesis.prediction || "—"}
                </p>
                {hypothesis.expected_result && (
                  <p className="mt-1.5 text-[10px] text-zinc-500">
                    <span className="text-zinc-600">Expected:</span>{" "}
                    {hypothesis.expected_result}
                  </p>
                )}
              </Section>

              {/* 4. Suggested Experiments */}
              <Section
                title="Suggested Experiments"
                icon={<FlaskConical className="h-3 w-3" />}
              >
                {experiment && Object.keys(experiment).length > 0 ? (
                  <ExperimentCard
                    experiment={experiment}
                    falsifiability={falsifiability}
                  />
                ) : (
                  <p className="text-[11px] text-zinc-600">
                    No experiment design available.
                  </p>
                )}
              </Section>

              {/* 5. Falsifiability */}
              <Section
                title="Falsifiability"
                icon={<ShieldCheck className="h-3 w-3" />}
              >
                <FalsifiabilityChecklist
                  check={falsifiability}
                  criteria={hypothesis.falsification_criteria}
                />
              </Section>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function Section({
  title,
  icon,
  defaultOpen = true,
  children,
}: {
  title: string;
  icon: React.ReactNode;
  defaultOpen?: boolean;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="rounded-md border border-zinc-800/80 bg-zinc-900/40">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between px-2 py-1.5 text-left"
      >
        <span className="flex items-center gap-1.5 text-[11px] font-medium text-zinc-300">
          <span className="text-zinc-500">{icon}</span>
          {title}
        </span>
        <ChevronDown
          className={cn(
            "h-3 w-3 text-zinc-600 transition-transform",
            open && "rotate-180"
          )}
        />
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.15 }}
            className="overflow-hidden"
          >
            <div className="px-2 pb-2 pt-0.5">{children}</div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function ConfidenceBar({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const color =
    pct >= 70 ? "bg-emerald-500" : pct >= 40 ? "bg-yellow-500" : "bg-red-500";
  return (
    <div className="flex items-center gap-2">
      <span className="text-[10px] text-zinc-600">confidence</span>
      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-zinc-700">
        <div
          className={cn("h-full rounded-full transition-all duration-500", color)}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-[10px] text-zinc-500">{pct}%</span>
    </div>
  );
}

function FalsifiabilityBadge({ falsifiable }: { falsifiable: boolean }) {
  return (
    <Badge
      variant="outline"
      className={cn(
        "h-4 text-[9px]",
        falsifiable
          ? "border-emerald-700/50 text-emerald-300"
          : "border-red-800/50 text-red-300"
      )}
    >
      {falsifiable ? "falsifiable" : "not falsifiable"}
    </Badge>
  );
}

function PmidList({
  label,
  pmids,
  tone,
}: {
  label: string;
  pmids: string[];
  tone: "support" | "contradict";
}) {
  if (pmids.length === 0) return null;
  const toneClass =
    tone === "support"
      ? "border-emerald-700/50 text-emerald-300"
      : "border-red-800/50 text-red-300";
  return (
    <div>
      <div className="mb-1 text-[10px] text-zinc-500">
        {label} ({pmids.length})
      </div>
      <div className="flex flex-wrap gap-1">
        {pmids.map((raw, idx) => {
          const pmid = normalizePmid(String(raw));
          const grounded = isGroundedPmid(String(raw));
          return (
            <a
              key={`${pmid}-${idx}`}
              href={`https://pubmed.ncbi.nlm.nih.gov/${pmid}/`}
              target="_blank"
              rel="noopener noreferrer"
              className={cn(
                "inline-flex items-center gap-0.5 rounded border px-1.5 py-0.5 text-[10px] transition-colors hover:bg-white/5",
                toneClass
              )}
              title={grounded ? "grounded via SBML ledger" : undefined}
            >
              <FileText className="h-2.5 w-2.5" />
              PMID:{pmid}
              {grounded && <span className="text-[9px] opacity-70">·g</span>}
              <ExternalLink className="h-2.5 w-2.5" />
            </a>
          );
        })}
      </div>
    </div>
  );
}

function FalsifiabilityChecklist({
  check,
  criteria,
}: {
  check: FalsifiabilityCheck;
  criteria?: string;
}) {
  const rules: { label: string; pass: boolean }[] = [
    { label: "Testable prediction", pass: check.has_testable_prediction },
    { label: "Control group", pass: check.has_control_group },
    { label: "Quantitative threshold", pass: check.has_quantitative_threshold },
  ];
  return (
    <div className="space-y-1.5">
      <div className="flex items-center gap-1.5">
        {check.overall ? (
          <ShieldCheck className="h-3.5 w-3.5 text-emerald-400" />
        ) : (
          <AlertCircle className="h-3.5 w-3.5 text-red-400" />
        )}
        <span
          className={cn(
            "text-[11px] font-medium",
            check.overall ? "text-emerald-300" : "text-red-300"
          )}
        >
          {check.overall ? "Falsifiable (Popper 3-rule pass)" : "Not falsifiable"}
        </span>
      </div>
      <div className="space-y-1">
        {rules.map((r) => (
          <div key={r.label} className="flex items-center gap-1.5 text-[10px]">
            <span
              className={cn(
                "flex h-3 w-3 items-center justify-center rounded-full text-[8px]",
                r.pass
                  ? "bg-emerald-500/20 text-emerald-400"
                  : "bg-red-500/20 text-red-400"
              )}
            >
              {r.pass ? "✓" : "✕"}
            </span>
            <span className="text-zinc-400">{r.label}</span>
          </div>
        ))}
      </div>
      {criteria && (
        <p className="rounded bg-zinc-800/50 px-1.5 py-1 text-[10px] leading-relaxed text-zinc-500">
          {criteria}
        </p>
      )}
      {check.failure_reasons.length > 0 && !check.overall && (
        <div className="space-y-0.5">
          {check.failure_reasons.map((r, idx) => (
            <p key={idx} className="text-[10px] text-red-300/80">
              · {r}
            </p>
          ))}
        </div>
      )}
    </div>
  );
}
