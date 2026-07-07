"use client";

import React from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import {
  DoseResponseCurve,
  type DoseResponseData,
} from "@/components/ai_assistant/DoseResponseCurve";
import {
  RAGInsightPanel,
  type RAGInsightsData,
  type TopSelection,
} from "@/components/ai_assistant/RAGInsightPanel";
import { TokenPerformanceBadge } from "@/components/ai_assistant/TokenPerformanceBadge";
import {
  MCPToolPanel,
  type MCPToolCall,
} from "@/components/ai_assistant/MCPToolPanel";
import {
  TermDefinitionCard,
  type TermDefinition,
} from "@/components/ai_assistant/TermDefinitionCard";

export interface ChatMessageProps {
  role: "user" | "agent";
  content?: string;
  type?:
    | "text"
    | "code"
    | "image"
    | "log"
    | "status"
    | "report"
    | "rag_insights"
    | "mcp_tools"
    | "mcp_terms"
    | "dose_response"
    | "combination_synergy"
    | "pkpd_profile";
  tokenUsage?: number;
  ragInsights?: RAGInsightsData;
  ragHitRate?: number;
  latencyMs?: number;
  modelName?: string;
  mcpToolCalls?: MCPToolCall[];
  mcpTokensSaved?: number;
  mcpTermDefinitions?: TermDefinition[];
  doseResponseData?: DoseResponseData;
  synergyData?: {
    synergy_assessment: string;
    combination_index: Record<string, number>;
    drug_regimen: Array<{
      drug_name: string;
      dose: number;
      ec50: number;
      emax: number;
      gamma: number;
      target: string;
    }>;
  };
  pkpdProfile?: {
    drug_name: string;
    drug_target: string;
    route: string;
    compartment: string;
    pk_params: Record<string, number>;
    pd_params: Record<string, number>;
  };
  onReplaceParam?: (selection: TopSelection) => void;
}

export function ChatMessage({
  role,
  content = "",
  type = "text",
  tokenUsage,
  ragInsights,
  ragHitRate,
  latencyMs,
  modelName = "",
  mcpToolCalls,
  mcpTokensSaved,
  mcpTermDefinitions,
  doseResponseData,
  synergyData,
  pkpdProfile,
}: ChatMessageProps) {
  const isUser = role === "user";

  if (isUser) {
    return (
      <div className="flex justify-end">
        <div className="max-w-[80%] rounded-2xl bg-blue-600 px-4 py-2.5 text-zinc-100">
          <p className="whitespace-pre-wrap text-sm">{content}</p>
        </div>
      </div>
    );
  }

  const renderAgentContent = () => {
    switch (type) {
      case "status":
        return (
          <div className="flex items-center gap-2 text-sm text-zinc-400">
            <motion.span
              className="inline-flex gap-1"
              initial={{ opacity: 0.6 }}
              animate={{ opacity: 1 }}
              transition={{
                repeat: Infinity,
                repeatType: "reverse",
                duration: 0.8,
              }}
            >
              <motion.span
                className="h-1.5 w-1.5 rounded-full bg-zinc-400"
                animate={{ y: [0, -4, 0] }}
                transition={{ repeat: Infinity, duration: 0.6, delay: 0 }}
              />
              <motion.span
                className="h-1.5 w-1.5 rounded-full bg-zinc-400"
                animate={{ y: [0, -4, 0] }}
                transition={{ repeat: Infinity, duration: 0.6, delay: 0.15 }}
              />
              <motion.span
                className="h-1.5 w-1.5 rounded-full bg-zinc-400"
                animate={{ y: [0, -4, 0] }}
                transition={{ repeat: Infinity, duration: 0.6, delay: 0.3 }}
              />
            </motion.span>
            <span className="whitespace-pre-wrap">{content}</span>
          </div>
        );

      case "code":
      case "report":
        return (
          <div className="markdown-content text-sm text-zinc-100">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                code(props) {
                  const { children, className } = props;
                  const match = /language-(\w+)/.exec(className || "");
                  const language = match ? match[1] : "text";
                  return (
                    <SyntaxHighlighter
                      PreTag="div"
                      language={language}
                      style={oneDark}
                      customStyle={{
                        margin: "0.75rem 0",
                        borderRadius: "0.5rem",
                        fontSize: "0.8125rem",
                      }}
                    >
                      {String(children).replace(/\n$/, "")}
                    </SyntaxHighlighter>
                  );
                },
              }}
            >
              {content}
            </ReactMarkdown>
          </div>
        );

      case "image":
        return (
          <img
            src={`data:image/png;base64,${content}`}
            alt="simulation result"
            className="max-w-full rounded-lg border border-zinc-700"
          />
        );

      case "log":
        return (
          <pre className="terminal-log max-h-80 overflow-auto rounded-lg bg-black p-3 text-xs text-zinc-200">
            <code className="whitespace-pre-wrap break-all">{content}</code>
          </pre>
        );

      case "rag_insights":
        return ragInsights ? (
          <RAGInsightPanel insights={ragInsights} />
        ) : null;

      case "mcp_tools":
        return mcpToolCalls && mcpToolCalls.length > 0 ? (
          <MCPToolPanel
            toolCalls={mcpToolCalls}
            tokensSaved={mcpTokensSaved ?? 0}
          />
        ) : null;

      case "mcp_terms":
        return mcpTermDefinitions && mcpTermDefinitions.length > 0 ? (
          <TermDefinitionCard definitions={mcpTermDefinitions} />
        ) : null;

      case "dose_response":
        return doseResponseData ? (
          <DoseResponseCurve data={doseResponseData} />
        ) : null;

      case "combination_synergy":
        return synergyData ? (
          <div className="space-y-2 text-sm text-zinc-100">
            <div className="flex items-center gap-2">
              <span
                className={cn(
                  "rounded-full px-2 py-0.5 text-xs font-medium",
                  synergyData.synergy_assessment === "潜在协同"
                    ? "bg-emerald-500/20 text-emerald-400"
                    : synergyData.synergy_assessment === "拮抗风险"
                    ? "bg-red-500/20 text-red-400"
                    : "bg-zinc-500/20 text-zinc-400"
                )}
              >
                {synergyData.synergy_assessment}
              </span>
              <span className="text-xs text-zinc-500">Chou-Talalay CI</span>
            </div>
            <div className="grid grid-cols-3 gap-2">
              {Object.entries(synergyData.combination_index).map(([key, ci]) => (
                <div
                  key={key}
                  className="rounded-lg bg-zinc-800/80 p-2 text-center"
                >
                  <div className="text-[10px] uppercase tracking-wide text-zinc-500">
                    {key.replace("fa_", "fa=")}
                  </div>
                  <div
                    className={cn(
                      "text-sm font-semibold",
                      Number(ci) < 0.8
                        ? "text-emerald-400"
                        : Number(ci) > 1.2
                        ? "text-red-400"
                        : "text-zinc-200"
                    )}
                  >
                    {Number(ci).toFixed(2)}
                  </div>
                </div>
              ))}
            </div>
            <div className="text-xs text-zinc-400">
              药物组合：
              {synergyData.drug_regimen.map((d) => d.drug_name).join(" + ")}
            </div>
          </div>
        ) : null;

      case "pkpd_profile":
        return pkpdProfile ? (
          <div className="space-y-2 text-sm text-zinc-100">
            <div className="flex items-center gap-2">
              <span className="text-xs font-medium text-zinc-300">
                PK/PD 模型
              </span>
              {pkpdProfile.route && (
                <span className="rounded-full bg-blue-500/20 px-2 py-0.5 text-xs text-blue-400">
                  {pkpdProfile.route}
                </span>
              )}
              {pkpdProfile.compartment && (
                <span className="rounded-full bg-purple-500/20 px-2 py-0.5 text-xs text-purple-400">
                  {pkpdProfile.compartment}
                </span>
              )}
            </div>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="rounded-lg bg-zinc-800/80 p-2">
                <div className="mb-1 text-zinc-500">药物 / 靶点</div>
                <div>
                  {pkpdProfile.drug_name} → {pkpdProfile.drug_target}
                </div>
              </div>
              <div className="rounded-lg bg-zinc-800/80 p-2">
                <div className="mb-1 text-zinc-500">PD 参数</div>
                <div>
                  Emax={pkpdProfile.pd_params?.Emax}, EC50={pkpdProfile.pd_params?.EC50}, γ={pkpdProfile.pd_params?.gamma}
                </div>
              </div>
            </div>
          </div>
        ) : null;

      case "text":
      default:
        return (
          <p className="whitespace-pre-wrap text-sm text-zinc-100">{content}</p>
        );
    }
  };

  return (
    <div className={cn("flex", isUser ? "justify-end" : "justify-start")}>
      <div
        className={cn(
          "max-w-[85%] rounded-2xl px-4 py-3",
          type === "log" || type === "code" || type === "report"
            ? "w-full bg-zinc-800/80"
            : type === "rag_insights" ||
              type === "mcp_tools" ||
              type === "mcp_terms" ||
              type === "dose_response" ||
              type === "combination_synergy" ||
              type === "pkpd_profile"
            ? "w-full bg-zinc-800/40"
            : "bg-zinc-800"
        )}
      >
        {renderAgentContent()}
        {!isUser &&
          (tokenUsage ||
            ragHitRate !== undefined ||
            latencyMs !== undefined ||
            (mcpTokensSaved ?? 0) > 0) && (
            <div className="mt-2 border-t border-zinc-700/50 pt-1.5">
              <TokenPerformanceBadge
                model={modelName}
                ragHitRate={ragHitRate}
                latencyMs={latencyMs}
                tokenUsage={tokenUsage}
                mcpTokensSaved={mcpTokensSaved}
              />
            </div>
          )}
      </div>
    </div>
  );
}
