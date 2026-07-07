"use client";

import React, { useEffect, useRef, useState } from "react";
import { MessageSquare, Lightbulb, ScrollText, X, Trash2, Square } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { ChatMessage } from "@/components/ai_assistant/ChatMessage";
import { ChatInput } from "@/components/ai_assistant/ChatInput";
import {
  AgentWorkflowTracker,
} from "@/components/ai_assistant/AgentWorkflowTracker";
import {
  WorkflowVisualization,
} from "@/components/ai_assistant/WorkflowVisualization";
import {
  ClarificationDialog,
} from "@/components/ai_assistant/ClarificationDialog";
import { useWorkbenchStore } from "@/lib/store";

type Tab = "chat" | "suggestions" | "logs";

const TABS: { id: Tab; label: string; icon: React.ReactNode }[] = [
  { id: "chat", label: "Chat", icon: <MessageSquare className="h-3.5 w-3.5" /> },
  { id: "suggestions", label: "Suggestions", icon: <Lightbulb className="h-3.5 w-3.5" /> },
  { id: "logs", label: "Logs", icon: <ScrollText className="h-3.5 w-3.5" /> },
];

/**
 * AI Assistant panel — a collapsible, tabbed side panel.
 *
 * Per the Scientific Modeling IDE spec the AI Assistant is NOT the primary UI:
 * it defaults to collapsed and is hosted in the far-right (collapsible) pane of
 * the WorkbenchShell. Chat keeps using the existing `/api/chat` v3 SSE contract
 * via the global store.
 */
export function AIAssistantPanel() {
  const [tab, setTab] = useState<Tab>("chat");
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const messages = useWorkbenchStore((s) => s.messages);
  const input = useWorkbenchStore((s) => s.input);
  const isStreaming = useWorkbenchStore((s) => s.isStreaming);
  const modelName = useWorkbenchStore((s) => s.modelName);
  const agents = useWorkbenchStore((s) => s.agents);
  const pipelineSteps = useWorkbenchStore((s) => s.pipelineSteps);
  const pipelineCurrent = useWorkbenchStore((s) => s.pipelineCurrent);
  const pipelineStepIndex = useWorkbenchStore((s) => s.pipelineStepIndex);
  const pipelineTotal = useWorkbenchStore((s) => s.pipelineTotal);
  const pipelineName = useWorkbenchStore((s) => s.pipelineName);
  const pipelineStatus = useWorkbenchStore((s) => s.pipelineStatus);
  const clarification = useWorkbenchStore((s) => s.clarification);
  const tokenUsage = useWorkbenchStore((s) => s.tokenUsage);
  const currentNode = useWorkbenchStore((s) => s.currentNode);
  const agentDispatches = useWorkbenchStore((s) => s.agentDispatches);
  const ragStatus = useWorkbenchStore((s) => s.ragStatus);
  const modelStatus = useWorkbenchStore((s) => s.modelStatus);
  const hypothesisList = useWorkbenchStore((s) => s.hypothesisList);

  const setInput = useWorkbenchStore((s) => s.setInput);
  const sendMessage = useWorkbenchStore((s) => s.sendMessage);
  const stopGeneration = useWorkbenchStore((s) => s.stopGeneration);
  const clearMemory = useWorkbenchStore((s) => s.clearMemory);
  const submitClarification = useWorkbenchStore((s) => s.submitClarification);
  const setAIPanelOpen = useWorkbenchStore((s) => s.setAIPanelOpen);

  // Auto-scroll the chat transcript to the latest message.
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <aside className="flex h-full w-full flex-col border-l border-zinc-800 bg-zinc-900/95">
      {/* Panel header */}
      <div className="flex h-10 shrink-0 items-center justify-between border-b border-zinc-800 px-3">
        <div className="flex items-center gap-2">
          <MessageSquare className="h-4 w-4 text-blue-400" />
          <span className="text-sm font-semibold text-zinc-100">AI Assistant</span>
          {isStreaming && (
            <span className="flex items-center gap-1 text-[10px] text-blue-400">
              <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-blue-400" />
              运行中
            </span>
          )}
        </div>
        <div className="flex items-center gap-1">
          <Button
            variant="ghost"
            size="icon-xs"
            onClick={clearMemory}
            disabled={isStreaming}
            title="清除当前对话"
            className="text-zinc-400 hover:text-red-300"
          >
            <Trash2 className="h-3.5 w-3.5" />
          </Button>
          <Button
            variant="ghost"
            size="icon-xs"
            onClick={() => setAIPanelOpen(false)}
            title="折叠 AI 助手"
            className="text-zinc-400 hover:text-zinc-100"
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* Tab bar */}
      <div className="flex shrink-0 border-b border-zinc-800">
        {TABS.map((t) => (
          <button
            key={t.id}
            type="button"
            onClick={() => setTab(t.id)}
            className={cn(
              "flex flex-1 items-center justify-center gap-1.5 px-2 py-2 text-xs font-medium transition-colors",
              tab === t.id
                ? "border-b-2 border-blue-500 text-blue-300"
                : "text-zinc-500 hover:text-zinc-300"
            )}
          >
            {t.icon}
            {t.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      {tab === "chat" && (
        <div className="flex min-h-0 flex-1 flex-col">
          {/* Workflow tracker strip */}
          {(agents.length > 0 || pipelineSteps.length > 0) && (
            <div className="shrink-0 border-b border-zinc-800 bg-zinc-900/95 px-3 py-2">
              {agents.length > 0 && <AgentWorkflowTracker agents={agents} />}
              {pipelineSteps.length > 0 && (
                <div className={cn(agents.length > 0 && "mt-2")}>
                  <WorkflowVisualization
                    steps={pipelineSteps}
                    currentNode={pipelineCurrent}
                    stepIndex={pipelineStepIndex}
                    totalSteps={pipelineTotal || pipelineSteps.length}
                    pipeline={pipelineName}
                    status={pipelineStatus}
                  />
                </div>
              )}
            </div>
          )}

          {/* Messages */}
          <ScrollArea className="min-h-0 flex-1">
            <div className="space-y-4 px-3 py-4">
              {messages.length === 0 && (
                <div className="flex flex-col items-center justify-center gap-2 pt-20 text-center text-zinc-500">
                  <MessageSquare className="h-8 w-8 text-zinc-700" />
                  <p className="text-xs">
                    输入生物学假说，AI 将自动建模并运行仿真。
                  </p>
                </div>
              )}
              {messages.map((msg) => (
                <ChatMessage
                  key={msg.id}
                  role={msg.role}
                  content={msg.content}
                  type={msg.type}
                  tokenUsage={msg.tokenUsage}
                  ragInsights={msg.ragInsights}
                  ragHitRate={msg.ragHitRate}
                  latencyMs={msg.latencyMs}
                  mcpToolCalls={msg.mcpToolCalls}
                  mcpTokensSaved={msg.mcpTokensSaved}
                  mcpTermDefinitions={msg.mcpTermDefinitions}
                  doseResponseData={msg.doseResponseData}
                  synergyData={msg.synergyData}
                  pkpdProfile={msg.pkpdProfile}
                  modelName={msg.modelName || modelName}
                />
              ))}
              <div ref={messagesEndRef} />
            </div>
          </ScrollArea>

          {/* Clarification dialog (human-in-the-loop) */}
          {clarification && (
            <div className="shrink-0 border-t border-zinc-800 px-3 py-2">
              <ClarificationDialog
                request={clarification}
                onSubmit={submitClarification}
                onStop={stopGeneration}
                disabled={!isStreaming}
              />
            </div>
          )}

          {/* Input bar */}
          <div className="shrink-0 border-t border-zinc-800 bg-zinc-900/95 px-3 pb-3 pt-2">
            {isStreaming && (
              <div className="mb-1.5 flex justify-end">
                <Button
                  variant="outline"
                  size="xs"
                  onClick={stopGeneration}
                  className="border-red-800 text-red-300 hover:bg-red-950/50 hover:text-red-200"
                >
                  <Square className="mr-1 h-3 w-3" />
                  停止生成
                </Button>
              </div>
            )}
            <ChatInput
              value={input}
              onChange={setInput}
              onSend={() => sendMessage(input)}
              disabled={isStreaming}
            />
          </div>
        </div>
      )}

      {tab === "suggestions" && (
        <ScrollArea className="min-h-0 flex-1">
          <div className="space-y-3 p-3">
            <div className="flex items-center gap-2 text-xs font-medium text-zinc-300">
              <Lightbulb className="h-4 w-4 text-amber-400" />
              假说与建议
            </div>
            {hypothesisList.length === 0 ? (
              <div className="rounded-lg border border-dashed border-zinc-700 p-6 text-center text-xs text-zinc-500">
                暂无建议。运行仿真后，假设生成器（C.7）将在此推荐可证伪的生物学假说与实验方案。
              </div>
            ) : (
              <div className="space-y-2">
                {hypothesisList.map((h, idx) => (
                  <div
                    key={idx}
                    className="rounded-lg border border-zinc-700 bg-zinc-800/40 p-2.5 text-xs text-zinc-300"
                  >
                    {String((h as { statement?: string }).statement ?? JSON.stringify(h))}
                  </div>
                ))}
              </div>
            )}
          </div>
        </ScrollArea>
      )}

      {tab === "logs" && (
        <ScrollArea className="min-h-0 flex-1">
          <div className="space-y-3 p-3 text-xs">
            {/* Live execution status */}
            <section className="rounded-lg border border-zinc-800 bg-zinc-900 p-2.5">
              <div className="mb-1.5 text-[11px] font-medium uppercase tracking-wide text-zinc-500">
                执行状态
              </div>
              <div className="space-y-1 text-zinc-300">
                <div className="flex justify-between">
                  <span className="text-zinc-500">当前节点</span>
                  <span className="font-mono">
                    {currentNode ? currentNode.replace("worker_", "").replace("_", " ") : "待机"}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-zinc-500">Token 消耗</span>
                  <span>{tokenUsage}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-zinc-500">模型</span>
                  <span className="truncate text-zinc-400" title={modelName}>
                    {modelName || "—"}
                  </span>
                </div>
              </div>
            </section>

            {/* Model providers */}
            {modelStatus && (
              <section className="rounded-lg border border-zinc-800 bg-zinc-900 p-2.5">
                <div className="mb-1.5 text-[11px] font-medium uppercase tracking-wide text-zinc-500">
                  模型供应商
                </div>
                <div className="space-y-1 text-zinc-300">
                  <div className="flex items-center justify-between">
                    <span className="text-zinc-500">主 LLM</span>
                    <Badge variant="outline" className="text-[10px] border-zinc-700 text-zinc-300">
                      {modelStatus.llm.provider}
                    </Badge>
                  </div>
                  <div className="truncate text-zinc-400" title={modelStatus.llm.model}>
                    {modelStatus.llm.model}
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-zinc-500">Embedding</span>
                    <Badge variant="outline" className="text-[10px] border-zinc-700 text-zinc-300">
                      {modelStatus.embedding.provider}
                    </Badge>
                  </div>
                  <div className="truncate text-zinc-400" title={modelStatus.embedding.model}>
                    {modelStatus.embedding.model}
                  </div>
                </div>
              </section>
            )}

            {/* RAG status */}
            {ragStatus && (
              <section className="rounded-lg border border-zinc-800 bg-zinc-900 p-2.5">
                <div className="mb-1.5 text-[11px] font-medium uppercase tracking-wide text-zinc-500">
                  知识库
                </div>
                <div className="flex flex-wrap gap-1">
                  {ragStatus.databases.map((db) => (
                    <Badge
                      key={db.name}
                      variant="outline"
                      className={cn(
                        "text-[10px]",
                        db.type === "online_api"
                          ? "border-emerald-700/50 text-emerald-300"
                          : db.type === "local_file"
                            ? "border-blue-700/50 text-blue-300"
                            : "border-zinc-600/50 text-zinc-400"
                      )}
                    >
                      {db.name}
                    </Badge>
                  ))}
                </div>
              </section>
            )}

            {/* Agent dispatch log */}
            <section className="rounded-lg border border-zinc-800 bg-zinc-900 p-2.5">
              <div className="mb-1.5 text-[11px] font-medium uppercase tracking-wide text-zinc-500">
                智能体调度日志
              </div>
              {agentDispatches.length === 0 ? (
                <p className="text-zinc-600">暂无调度记录。</p>
              ) : (
                <div className="max-h-60 space-y-1 overflow-auto">
                  {agentDispatches.map((d, idx) => (
                    <div key={idx} className="font-mono text-[10px] text-zinc-400">
                      <span className="text-zinc-500">[{d.status}]</span>{" "}
                      {d.target_agent}
                      {d.latency_ms ? ` · ${Math.round(d.latency_ms)}ms` : ""}
                    </div>
                  ))}
                </div>
              )}
            </section>
          </div>
        </ScrollArea>
      )}
    </aside>
  );
}
