"use client";

import React from "react";
import { Atom, ExternalLink } from "lucide-react";
import Link from "next/link";
import { useWorkbenchStore, type Message } from "@/lib/store";
import { NaturalLanguageInput } from "./NaturalLanguageInput";
import { AIWorkflowSteps, WorkflowIdleHint } from "./AIWorkflowSteps";
import { ResultsTabs } from "./ResultsTabs";

/**
 * MinimalApp —— 极简 Auto-Chat 仿真产品主界面。
 *
 * 单栏纵向布局，对齐用户指定的 5 模块结构：
 *
 *   ┌───────────────────────────────────────┐
 *   │  Header（BioDynamics · 品牌标语）       │
 *   ├───────────────────────────────────────┤
 *   │  Natural Language Input（假说 + 模拟）   │
 *   ├───────────────────────────────────────┤
 *   │  AI Workflow（7 步自动进度）            │
 *   ├───────────────────────────────────────┤
 *   │  Results Tabs（Graph|Curves|Validation|Report）│
 *   └───────────────────────────────────────┘
 *
 * 不展示任何空壳 / 高级功能（Sensitivity、SBML、Calibration 等，见
 * TODO.md）。所有 SSE 事件由 store.ingestSSEEvent 统一消费，本组件只读
 * 顶层派生状态，保持极简与解耦。
 */
export function MinimalApp() {
  const isStreaming = useWorkbenchStore((s) => s.isStreaming);
  // 订阅稳定的 messages 数组引用（仅在 messages 真实变化时变更），
  // 在渲染体内派生 lastError，避免每次 store 更新都产生新数组引用
  // 触发无谓重渲染（曾导致 E2E 中按钮 "unstable" 点击超时）。
  const messages = useWorkbenchStore((s) => s.messages);
  const lastError = messages.reduceRight<Message | undefined>(
    (acc, m) => acc ?? (m.type === "text" && m.content.includes("错误") ? m : undefined),
    undefined
  );
  const currentNode = useWorkbenchStore((s) => s.currentNode);

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
            {/* 实时状态行：当前节点 / 错误 */}
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
            {lastError && (
              <p className="rounded-md border border-red-900/40 bg-red-950/30 px-3 py-2 text-xs text-red-300">
                {lastError.content}
              </p>
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
