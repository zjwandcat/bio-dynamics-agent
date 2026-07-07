/**
 * Shared TypeScript types + helpers for the Hypothesis Panel (Task C.8).
 *
 * Mirrors the backend Hypothesis schema produced by
 * `backend/app/hypothesis/hypothesis_agent.py` and its sub-components
 * (hypothesis_generator / experiment_designer / falsifiability_checker).
 *
 * All fields are optional because the backend degrades gracefully and may emit
 * partial dicts (e.g. `experiment_design` is `{}` until ExperimentPlanner runs,
 * `supporting_pmids` is `[]` until the RAG literature search runs).
 */

export type HypothesisStrategy = "oscillation" | "bistability" | "sensitivity";

export type HypothesisStatus = "generated" | "pending" | "failed";

export interface Perturbation {
  type?: string;
  agent?: string;
  target?: string;
  dose?: string;
  duration?: string;
  mechanism?: string;
  description?: string;
}

export interface Readout {
  species?: string;
  metric?: string;
  threshold?: number;
}

/** ExperimentDesigner 6-field schema. */
export interface ExperimentDesign {
  perturbation?: Perturbation;
  readout?: Readout;
  time_points?: number[];
  controls?: string[];
  cell_line?: string;
  expected_result?: string;
}

/** Popper 3-rule falsifiability check results. */
export interface FalsifiabilityCheck {
  has_testable_prediction: boolean;
  has_control_group: boolean;
  has_quantitative_threshold: boolean;
  overall: boolean;
  criteria?: string;
  failure_reasons: string[];
}

export interface Hypothesis {
  id?: string;
  hypothesis_id?: string;
  statement?: string;
  prediction?: string;
  experiment_design?: ExperimentDesign;
  validation_method?: string;
  expected_result?: string;
  falsifiable?: boolean;
  falsification_criteria?: string;
  supporting_pmids?: string[];
  contradicting_pmids?: string[];
  falsifying_pmids?: string[];
  strategy?: HypothesisStrategy;
  target_species?: string;
  feedback_node?: string;
  threshold_node?: string;
  target_param?: string;
  sensitivity?: number;
  pathway_class?: string;
  parameter_robustness?: Record<string, unknown>;
  confidence?: number;
  status?: HypothesisStatus;
}

// ---------------------------------------------------------------------------
// Falsifiability derivation (mirrors backend FalsificationChecker 3 rules)
// ---------------------------------------------------------------------------
// The backend only persists `falsifiable` + `falsification_criteria` on the
// hypothesis dict (not the per-rule verdicts), so we re-derive the 3 rule
// booleans client-side to render the Popper checklist.

const VAGUE_KEYWORDS = [
  "会变化", "可能改变", "有所反应", "出现一些",
  "可能影响", "有所变化", "发生改变", "产生反应", "存在差异",
];

const STANDARD_CONTROLS = ["vehicle", "untreated", "dmso", "scramble"];

const QUANT_RE = /\d+(?:\.\d+)?\s*(?:%|倍|fold)|>\s*\d+(?:\.\d+)?/i;
const DIRECTION_RE = /下降|降低|升高|增加|消除|消失|失去|出现|减少|增大/;
const STATE_CHANGE_RE =
  /消除\s*\w*\s*(?:振荡|切换|响应|信号)|失去\s*\w*\s*(?:双稳态|切换能力|振荡)/;

export function deriveFalsifiability(h: Hypothesis): FalsifiabilityCheck {
  const prediction = h.prediction ?? "";

  // Rule 1 — testable prediction (direction + quantitative threshold, no vague terms)
  let has_testable_prediction = false;
  if (prediction.trim()) {
    const lower = prediction.toLowerCase();
    const isVague = VAGUE_KEYWORDS.some((k) => lower.includes(k));
    if (!isVague) {
      const hasQuant = QUANT_RE.test(prediction);
      has_testable_prediction =
        (DIRECTION_RE.test(prediction) && hasQuant) ||
        STATE_CHANGE_RE.test(prediction) ||
        hasQuant;
    }
  }

  // Rule 2 — standard control group
  const controls = h.experiment_design?.controls ?? [];
  let has_control_group = false;
  if (Array.isArray(controls) && controls.length > 0) {
    const joined = controls.map((c) => String(c).toLowerCase()).join(" ");
    has_control_group = STANDARD_CONTROLS.some((k) => joined.includes(k));
  }

  // Rule 3 — quantitative threshold
  let has_quantitative_threshold = false;
  const threshold = h.experiment_design?.readout?.threshold;
  if (typeof threshold === "number" && threshold > 0) {
    has_quantitative_threshold = true;
  }
  if (
    !has_quantitative_threshold &&
    (QUANT_RE.test(h.prediction ?? "") || QUANT_RE.test(h.expected_result ?? ""))
  ) {
    has_quantitative_threshold = true;
  }

  const failure_reasons: string[] = [];
  if (!has_testable_prediction) {
    failure_reasons.push(
      "Rule 1 failed: prediction lacks a testable direction + quantitative threshold"
    );
  }
  if (!has_control_group) {
    failure_reasons.push(
      "Rule 2 failed: no standard control group (vehicle / untreated / DMSO / scramble)"
    );
  }
  if (!has_quantitative_threshold) {
    failure_reasons.push("Rule 3 failed: no quantitative threshold detected");
  }

  const overall =
    h.falsifiable !== undefined
      ? h.falsifiable
      : has_testable_prediction &&
        has_control_group &&
        has_quantitative_threshold;

  return {
    has_testable_prediction,
    has_control_group,
    has_quantitative_threshold,
    overall,
    criteria: h.falsification_criteria,
    failure_reasons,
  };
}

/**
 * Derive a 0–1 confidence score for display. Uses the backend `confidence`
 * field when present; otherwise estimates from falsifiability, evidence count,
 * experiment design, and validation method.
 */
export function deriveConfidence(h: Hypothesis): number {
  if (typeof h.confidence === "number" && h.confidence >= 0 && h.confidence <= 1) {
    return h.confidence;
  }
  let score = 0.4;
  if (h.falsifiable !== false) score += 0.2;
  const supporting = h.supporting_pmids?.length ?? 0;
  score += Math.min(supporting, 2) * 0.1;
  if (h.experiment_design && Object.keys(h.experiment_design).length > 0) score += 0.1;
  if (h.validation_method) score += 0.05;
  return Math.min(score, 0.95);
}

/** Strip the optional `|grounded` suffix the backend appends to grounded PMIDs. */
export function normalizePmid(pmid: string): string {
  return pmid.split("|")[0].trim();
}

/** Whether a PMID was marked as grounded by the SBML Grounder ledger. */
export function isGroundedPmid(pmid: string): boolean {
  return pmid.includes("|grounded");
}
