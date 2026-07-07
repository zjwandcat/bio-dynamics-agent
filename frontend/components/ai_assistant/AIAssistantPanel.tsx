"use client";

import React, { useEffect, useRef, useState } from "react";
import {
  MessageSquare,
  Lightbulb,
  ScrollText,
  X,
  Trash2,
  Square,
  ChevronDown,
  ChevronUp,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
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
import {
  SuggestionsPanel,
  type Suggestion,
} from "@/components/ai_assistant/SuggestionsPanel";
import { LogsPanel } from "@/components/ai_assistant/LogsPanel";
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
 *
 * Collapse behaviour: the panel is mounted/unmounted by `WorkbenchShell` based
 * on `uiState.aiAssistantOpen` (the far-right grid column collapses to 0). The
 * close button here flips that flag; the header's toggle button re-opens it.
 * (The "thin vertical rail" rendering for the collapsed state is owned by the
 * shell — adding it here would have no effect because the panel is unmounted
 * when collapsed.)
 */
export function AIAssistantPanel() {
  const [tab, setTab] = useState<Tab>("chat");
  const [workflowOpen, setWorkflowOpen] = useState(true);
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
  const agentDispatches = useWorkbenchStore((s) => s.agentDispatches);
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

  // Apply a suggestion from the Suggestions tab: pre-fill the chat input with a
  // type-appropriate prompt and switch to the Chat tab so the user can review
  // and send it.
  const handleApplySuggestion = (suggestion: Suggestion) => {
    const prefix =
      suggestion.type === "hypothesis_refinement"
        ? "Refine hypothesis: "
        : suggestion.type === "parameter_adjustment"
          ? "Apply parameter adjustment: "
          : "Run experiment: ";
    setInput(`${prefix}${suggestion.text}`);
    setTab("chat");
  };

  // Clear only the visible agent-dispatch log buffer (the store exposes no
  // dedicated action, so we patch state directly via the Zustand setter).
  const handleClearLogs = () => {
    useWorkbenchStore.setState({ agentDispatches: [] });
  };

  const hasWorkflow = agents.length > 0 || pipelineSteps.length > 0;

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
          {/* Collapsible workflow tracker strip */}
          {hasWorkflow && (
            <div className="shrink-0 border-b border-zinc-800 bg-zinc-900/95">
              <button
                type="button"
                onClick={() => setWorkflowOpen((v) => !v)}
                className="flex w-full items-center justify-between px-3 py-1.5 text-[11px] font-medium uppercase tracking-wide text-zinc-500 hover:text-zinc-300"
              >
                <span>Workflow</span>
                {workflowOpen ? (
                  <ChevronUp className="h-3.5 w-3.5" />
                ) : (
                  <ChevronDown className="h-3.5 w-3.5" />
                )}
              </button>
              {workflowOpen && (
                <div className="px-3 pb-2">
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
        <SuggestionsPanel
          hypothesisList={hypothesisList}
          agentDispatches={agentDispatches}
          onApply={handleApplySuggestion}
          className="min-h-0 flex-1"
        />
      )}

      {tab === "logs" && (
        <LogsPanel
          dispatches={agentDispatches}
          onClear={handleClearLogs}
          className="min-h-0 flex-1"
        />
      )}
    </aside>
  );
}
