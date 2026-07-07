"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Tooltip as RechartsTooltip,
} from "recharts";
import {
  ChevronDown,
  Search,
  FileText,
  ExternalLink,
  RefreshCw,
} from "lucide-react";
import { cn } from "@/lib/utils";

export interface RewriteItem {
  original: string;
  standardized: string;
  reason?: string;
}

export interface TopSelection {
  parameter: string;
  value: string;
  source: string;
  pmid?: string;
  confidence_score: number;
  species?: string;
  context?: string;
  edge?: string;
}

export interface DrugCandidate {
  drug_name: string;
  target_name?: string;
  ic50?: number;
  ec50?: number;
  clinical_dose?: string;
  source?: string;
  is_clinical_candidate?: boolean;
  clinical_trial_info?: Array<{
    nct_id: string;
    phase: string;
    condition: string;
    status: string;
  }>;
}

export interface RAGInsightsData {
  rewritten_query?: string;
  rewrites?: RewriteItem[];
  expanded_terms?: string[];
  source_distribution?: Record<string, number>;
  total_candidates?: number;
  top_selections?: TopSelection[];
  hit_rate?: number;
  drug_candidates?: DrugCandidate[];
}

interface RAGInsightPanelProps {
  insights: RAGInsightsData;
  className?: string;
  onReplaceParam?: (selection: TopSelection) => void;
}

// 来源类型对应的饼图颜色
const SOURCE_COLORS: Record<string, string> = {
  PMC: "#10b981",
  PubMed: "#3b82f6",
  "Internal DB": "#a78bfa",
  Preprint: "#f59e0b",
};

const DEFAULT_COLORS = ["#6366f1", "#ec4899", "#14b8a6", "#f97316"];

/**
 * RAG 深度洞察面板
 * 对应 1233.md 第四部分 §2：可折叠手风琴面板，展示查询重写对比、来源分布饼图、Top 参数卡片。
 */
export function RAGInsightPanel({
  insights,
  className,
  onReplaceParam,
}: RAGInsightPanelProps) {
  const [expanded, setExpanded] = useState(true);

  const rewrites = insights.rewrites || [];
  const sourceDist = insights.source_distribution || {};
  const topSelections = (insights.top_selections || []).slice(0, 3);
  const totalCandidates = insights.total_candidates || 0;

  const pieData = Object.entries(sourceDist).map(([name, value]) => ({
    name,
    value,
  }));

  return (
    <div
      className={cn(
        "rounded-lg border border-white/10 bg-zinc-900/60",
        className
      )}
    >
      {/* 折叠头部 */}
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        className="flex w-full items-center justify-between px-3 py-2 text-left"
      >
        <div className="flex items-center gap-2">
          <Search className="h-3.5 w-3.5 text-blue-400" />
          <span className="text-xs font-medium text-zinc-200">
            RAG 深度洞察
          </span>
          <span className="rounded-full bg-blue-500/10 px-2 py-0.5 text-[10px] text-blue-300">
            {totalCandidates} 篇候选 · 命中率{" "}
            {Math.round((insights.hit_rate ?? 0) * 100)}%
          </span>
        </div>
        <ChevronDown
          className={cn(
            "h-4 w-4 text-zinc-500 transition-transform",
            expanded && "rotate-180"
          )}
        />
      </button>

      <AnimatePresence initial={false}>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25 }}
            className="overflow-hidden"
          >
            <div className="space-y-4 px-3 pb-3">
              {/* 1. Query Rewriting 对比 */}
              {rewrites.length > 0 && (
                <section>
                  <h4 className="mb-1.5 text-[11px] font-medium uppercase tracking-wide text-zinc-500">
                    查询重写
                  </h4>
                  <div className="space-y-1">
                    {rewrites.map((rw, idx) => (
                      <div
                        key={idx}
                        className="flex flex-wrap items-center gap-1.5 text-xs"
                      >
                        <span className="rounded bg-red-500/10 px-1.5 py-0.5 text-red-300 line-through">
                          {rw.original}
                        </span>
                        <span className="text-zinc-600">→</span>
                        <span className="rounded bg-green-500/10 px-1.5 py-0.5 text-green-300">
                          {rw.standardized}
                        </span>
                        {rw.reason && (
                          <span className="text-[10px] text-zinc-600">
                            ({rw.reason})
                          </span>
                        )}
                      </div>
                    ))}
                  </div>
                  {insights.rewritten_query && (
                    <p className="mt-1.5 rounded bg-zinc-800/50 px-2 py-1 text-[11px] text-zinc-400">
                      <span className="text-zinc-500">标准化查询：</span>
                      {insights.rewritten_query}
                    </p>
                  )}
                </section>
              )}

              {/* 2. 来源分布饼图 */}
              {pieData.length > 0 && (
                <section>
                  <h4 className="mb-1.5 text-[11px] font-medium uppercase tracking-wide text-zinc-500">
                    来源分布
                  </h4>
                  <div className="flex items-center gap-3">
                    <div className="h-32 w-32 flex-shrink-0">
                      <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                          <Pie
                            data={pieData}
                            dataKey="value"
                            nameKey="name"
                            cx="50%"
                            cy="50%"
                            innerRadius={28}
                            outerRadius={55}
                            paddingAngle={2}
                          >
                            {pieData.map((entry, idx) => (
                              <Cell
                                key={idx}
                                fill={
                                  SOURCE_COLORS[entry.name] ??
                                  DEFAULT_COLORS[idx % DEFAULT_COLORS.length]
                                }
                              />
                            ))}
                          </Pie>
                          <RechartsTooltip
                            contentStyle={{
                              background: "#18181b",
                              border: "1px solid #3f3f46",
                              borderRadius: "6px",
                              fontSize: "11px",
                            }}
                          />
                        </PieChart>
                      </ResponsiveContainer>
                    </div>
                    <div className="space-y-1">
                      {pieData.map((entry, idx) => (
                        <div
                          key={idx}
                          className="flex items-center gap-1.5 text-[11px]"
                        >
                          <span
                            className="h-2 w-2 rounded-full"
                            style={{
                              background:
                                SOURCE_COLORS[entry.name] ??
                                DEFAULT_COLORS[idx % DEFAULT_COLORS.length],
                            }}
                          />
                          <span className="text-zinc-400">{entry.name}</span>
                          <span className="text-zinc-600">{entry.value}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </section>
              )}

              {/* 3. Top 参数卡片 */}
              {topSelections.length > 0 && (
                <section>
                  <h4 className="mb-1.5 text-[11px] font-medium uppercase tracking-wide text-zinc-500">
                    Top {topSelections.length} 参数
                  </h4>
                  <div className="grid gap-2">
                    {topSelections.map((sel, idx) => (
                      <ParamCard
                        key={idx}
                        selection={sel}
                        onReplace={onReplaceParam}
                      />
                    ))}
                  </div>
                </section>
              )}

              {/* 4. 药物候选（知识图谱） */}
              <DrugCandidateSection candidates={insights.drug_candidates} />

              {topSelections.length === 0 &&
                rewrites.length === 0 &&
                (!insights.drug_candidates ||
                  insights.drug_candidates.length === 0) && (
                  <p className="py-2 text-center text-xs text-zinc-600">
                    本次未检索到可用文献参数，已回退到估算值。
                  </p>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

/**
 * 单个参数卡片：参数值 + 置信度条 + PMID 引文徽章 + 替换按钮
 */
function ParamCard({
  selection,
  onReplace,
}: {
  selection: TopSelection;
  onReplace?: (s: TopSelection) => void;
}) {
  const confidencePct = Math.round(
    (selection.confidence_score ?? 0) * 100
  );
  const pmid = selection.pmid?.trim();

  return (
    <div className="rounded-md border border-zinc-700/60 bg-zinc-800/40 p-2.5">
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0 flex-1">
          <div className="flex items-baseline gap-1.5">
            <span className="text-sm font-semibold text-zinc-100">
              {selection.parameter}
            </span>
            <span className="text-sm text-blue-300">{selection.value}</span>
          </div>
          {selection.edge && (
            <span className="mt-0.5 inline-block rounded bg-zinc-700/50 px-1.5 py-0.5 text-[10px] text-zinc-400">
              {selection.edge}
            </span>
          )}
          {selection.species && (
            <span className="ml-1 inline-block text-[10px] text-zinc-500">
              · {selection.species}
            </span>
          )}
        </div>
        {pmid && (
          <a
            href={`https://pubmed.ncbi.nlm.nih.gov/${pmid}/`}
            target="_blank"
            rel="noopener noreferrer"
            className="flex flex-shrink-0 items-center gap-0.5 rounded bg-blue-500/10 px-1.5 py-0.5 text-[10px] text-blue-300 transition-colors hover:bg-blue-500/20"
          >
            <FileText className="h-2.5 w-2.5" />
            PMID:{pmid}
            <ExternalLink className="h-2.5 w-2.5" />
          </a>
        )}
      </div>

      {/* 置信度条 */}
      <div className="mt-2 flex items-center gap-2">
        <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-zinc-700">
          <div
            className={cn(
              "h-full rounded-full transition-all duration-500",
              confidencePct >= 80
                ? "bg-green-500"
                : confidencePct >= 50
                ? "bg-yellow-500"
                : "bg-red-500"
            )}
            style={{ width: `${confidencePct}%` }}
          />
        </div>
        <span className="text-[10px] text-zinc-500">{confidencePct}%</span>
      </div>

      {selection.context && (
        <p className="mt-1.5 line-clamp-2 text-[10px] text-zinc-600">
          {selection.context}
        </p>
      )}

      {onReplace && (
        <button
          type="button"
          onClick={() => onReplace(selection)}
          className="mt-1.5 flex items-center gap-1 text-[10px] text-zinc-500 transition-colors hover:text-blue-300"
        >
          <RefreshCw className="h-2.5 w-2.5" />
          替换参数
        </button>
      )}
    </div>
  );
}

/**
 * 药物候选卡片列表：展示靶点相关药物、IC50/EC50、临床状态
 */
function DrugCandidateSection({
  candidates,
}: {
  candidates?: DrugCandidate[];
}) {
  if (!candidates || candidates.length === 0) {
    return null;
  }

  return (
    <section>
      <h4 className="mb-1.5 text-[11px] font-medium uppercase tracking-wide text-zinc-500">
        药物候选（知识图谱）
      </h4>
      <div className="grid gap-2">
        {candidates.map((cand, idx) => (
          <div
            key={idx}
            className={cn(
              "rounded-md border p-2.5",
              cand.is_clinical_candidate
                ? "border-emerald-500/30 bg-emerald-500/5"
                : "border-zinc-700/60 bg-zinc-800/40"
            )}
          >
            <div className="flex items-start justify-between gap-2">
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-1.5">
                  <span className="text-sm font-semibold text-zinc-100">
                    {cand.drug_name}
                  </span>
                  {cand.is_clinical_candidate && (
                    <span className="rounded-full bg-emerald-500/15 px-1.5 py-0.5 text-[10px] text-emerald-400">
                      临床候选
                    </span>
                  )}
                </div>
                {cand.target_name && (
                  <div className="mt-0.5 text-[10px] text-zinc-500">
                    靶点：{cand.target_name}
                  </div>
                )}
              </div>
            </div>

            <div className="mt-2 flex flex-wrap gap-2 text-[10px]">
              {cand.ic50 !== undefined && (
                <span className="rounded bg-zinc-800 px-1.5 py-0.5 text-zinc-300">
                  IC50 {formatScientific(cand.ic50)} nM
                </span>
              )}
              {cand.ec50 !== undefined && (
                <span className="rounded bg-zinc-800 px-1.5 py-0.5 text-zinc-300">
                  EC50 {formatScientific(cand.ec50)} nM
                </span>
              )}
              {cand.clinical_dose && (
                <span className="rounded bg-zinc-800 px-1.5 py-0.5 text-zinc-300">
                  {cand.clinical_dose}
                </span>
              )}
            </div>

            {cand.source && (
              <p className="mt-1.5 text-[10px] text-zinc-600">
                来源：{cand.source}
              </p>
            )}

            {cand.clinical_trial_info && cand.clinical_trial_info.length > 0 && (
              <div className="mt-2 space-y-1">
                {cand.clinical_trial_info.slice(0, 2).map((trial) => (
                  <a
                    key={trial.nct_id}
                    href={`https://clinicaltrials.gov/study/${trial.nct_id}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-1 text-[10px] text-emerald-400 hover:text-emerald-300"
                  >
                    <ExternalLink className="h-2.5 w-2.5" />
                    {trial.nct_id} · {trial.phase} · {trial.status}
                  </a>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </section>
  );
}

function formatScientific(value: number | undefined): string {
  if (value === undefined || !Number.isFinite(value)) {
    return "N/A";
  }
  if (value === 0) {
    return "0";
  }
  const absValue = Math.abs(value);
  if (absValue >= 1 && absValue < 1000) {
    return value.toFixed(2);
  }
  return value.toExponential(2);
}
