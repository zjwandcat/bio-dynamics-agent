"use client";

import React from "react";
import { Atom, ExternalLink, AlertCircle } from "lucide-react";
import Link from "next/link";
import { useWorkbenchStore } from "@/lib/store";
import { NaturalLanguageInput } from "./NaturalLanguageInput";
import { AIWorkflowSteps, WorkflowIdleHint } from "./AIWorkflowSteps";
import { ResultsTabs } from "./ResultsTabs";

/**
 * MinimalApp —— 极简 Auto-Chat 仿真产品主界面。
 *
 * 单栏纵向布局：
 *   Header → Natural Language Input → AI Workflow → Results Tabs
 *
 * 所有 SSE 事件由 store.ingestSSEEvent 统一消费，本组件只读顶层派生状态。
 * 错误反馈：当 sendMessage 的 onError 触发时，store 会 append 一条
 * role=agent type=text 的错误消息；本组件取最后一条 agent text 作为
 * 实时状态/错误行展示，不再用脆弱的中文关键词过滤。
 */
export function MinimalApp() {
  const isStreaming = useWorkbenchStore((s) => s.isStreaming);
  const messages = useWorkbenchStore((s) => s.messages);
  const currentNode = useWorkbenchStore((s) => s.currentNode);

  // 取最后一条 agent text 消息作为状态/错误反馈（含 SSE error 与 onError）。
  // store 的 error 事件和 onError 回调都会 append role=agent type=text，
  // 所以这里能稳定捕获连接失败、流读取失败等所有错误。
  const lastAgentText = messages.reduceRight<
    | { content: string; isError: boolean }
    | undefined
  >((acc, m) => {
    if (acc) return acc;
    if (m.role === "agent" && m.type === "text" && m.content) {
      // 识别错误类消息（后端 error 事件 / 网络失败 / 流错误）
      const isError =
        /错误|error|失败|failed|refused|fetch|连接|timeout/i.test(m.content);
      return { content: m.content, isError };
    }
    return undefined;
  }, undefined);

  return (
    <main className="min-h-screen bg-zinc-950 text-zinc-100">
      {/* ── Header ── */}
      <header className="border-b border-zinc-900 bg-zinc-950/80 backdrop-blur">
        <div className="mx-auto flex max-w-4xl items-center justify-between px-4 py-4">
          <div className="flex items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-blue-600/15">
              <Atom className="h-4 w-4 text-blue-400" />
            </div>
            <div>
              <h1 className="text-base font-semibold leading-tight text-zinc-100">
                BioDynamics
              </h1>
              <p className="font-mono text-[10px] uppercase tracking-wider text-zinc-500">
                One Prompt → One Simulation → One Scientific Report
              </p>
            </div>
          </div>
          <Link
            href="/advanced"
            className="inline-flex items-center gap-1.5 rounded-md border border-zinc-800 px-2.5 py-1 text-[11px] text-zinc-500 transition-colors hover:border-zinc-600 hover:text-zinc-300"
            title="归档保留的高级四栏 IDE"
          >
            Advanced
            <ExternalLink className="h-3 w-3" />
          </Link>
        </div>
      </header>

      {/* ── 主体单栏 ── */}
      <div className="mx-auto max-w-4xl space-y-6 px-4 py-8">
        {/* 01 / Natural Language Input */}
        <section aria-label="hypothesis input">
          <SectionLabel index="01" title="Natural Language Input" />
          <NaturalLanguageInput className="mt-2" />
        </section>

        {/* 02 / AI Workflow */}
        <section aria-label="ai workflow">
          <SectionLabel index="02" title="AI Workflow" />
          <div className="mt-2 space-y-3">
            <AIWorkflowSteps />
            {/* 实时状态行：当前节点 */}
            {(isStreaming || currentNode) && (
              <p className="font-mono text-[11px] text-zinc-500">
                {isStreaming ? (
                  <>
                    <span className="mr-1.5 inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-blue-400" />
                    {currentNode ? `running: ${currentNode}` : "running…"}
                  </>
                ) : (
                  <span className="text-emerald-500">done</span>
                )}
              </p>
            )}
            {/* 错误/状态反馈：捕获所有 agent text（含网络错误） */}
            {lastAgentText && !isStreaming && (
              <div
                className={
                  lastAgentText.isError
                    ? "flex items-start gap-2 rounded-md border border-red-900/40 bg-red-950/30 px-3 py-2 text-xs text-red-300"
                    : "rounded-md border border-zinc-800 bg-zinc-900/40 px-3 py-2 text-xs text-zinc-400"
                }
                role={lastAgentText.isError ? "alert" : "status"}
              >
                {lastAgentText.isError && (
                  <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                )}
                <span className="break-words">{lastAgentText.content}</span>
              </div>
            )}
            <WorkflowIdleHint />
          </div>
        </section>

        {/* 03 / Results */}
        <section aria-label="results">
          <SectionLabel index="03" title="Results" />
          <ResultsTabs className="mt-2" />
        </section>
      </div>

      <footer className="border-t border-zinc-900 px-4 py-6 text-center font-mono text-[10px] text-zinc-600">
        BioDynamics Agent · AI-native biomedical signaling pathway simulation
      </footer>
    </main>
  );
}

/** 小节标签：序号 + 标题，呼应"科学仪器"质感。 */
function SectionLabel({ index, title }: { index: string; title: string }) {
  return (
    <div className="flex items-center gap-2">
      <span className="font-mono text-[10px] text-zinc-600">{index}</span>
      <h2 className="text-[11px] font-medium uppercase tracking-wider text-zinc-500">
        {title}
      </h2>
      <div className="h-px flex-1 bg-zinc-900" />
    </div>
  );
}
