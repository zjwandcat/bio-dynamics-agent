"use client";

import React, { useState, useRef, useEffect } from "react";
import { ArrowUp, Square } from "lucide-react";
import { cn } from "@/lib/utils";
import { useWorkbenchStore } from "@/lib/store";

/**
 * NaturalLanguageInput —— 极简 Auto-Chat 的唯一入口。
 *
 * 一个文本框 + 一个发送按钮，调用 store.sendMessage(text) → /api/chat SSE。
 * 配 4 条经典通路假说示例 chip，点击即填入，帮助新用户 5 秒上手。
 * 回车发送（Shift+回车换行）；streaming 时按钮变为停止。
 */

const EXAMPLES: { label: string; text: string }[] = [
  {
    label: "EGFR / MAPK",
    text: "EGF stimulation induces transient ERK activation through the EGFR-MAPK cascade.",
  },
  {
    label: "PI3K / AKT",
    text: "PTEN loss causes sustained AKT hyperactivation via the PI3K pathway.",
  },
  {
    label: "p53",
    text: "DNA damage induces oscillatory p53 dynamics with a 5-6 hour period.",
  },
  {
    label: "NF-κB",
    text: "TNF-alpha induces oscillatory nuclear NF-kB localization with a 1-2 hour period.",
  },
];

export function NaturalLanguageInput({ className }: { className?: string }) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const sendMessage = useWorkbenchStore((s) => s.sendMessage);
  const stopGeneration = useWorkbenchStore((s) => s.stopGeneration);
  const isStreaming = useWorkbenchStore((s) => s.isStreaming);

  // 自适应高度：内容多时撑高，发送后回缩
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, [value]);

  const handleSubmit = () => {
    const trimmed = value.trim();
    if (!trimmed || isStreaming) return;
    void sendMessage(trimmed);
    setValue("");
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleExample = (text: string) => {
    setValue(text);
    textareaRef.current?.focus();
  };

  return (
    <div className={cn("space-y-3", className)}>
      <div
        className={cn(
          "flex items-end gap-2 rounded-xl border bg-zinc-900/60 p-2 transition-colors focus-within:border-blue-500/50 focus-within:bg-zinc-900",
          isStreaming
            ? "border-blue-500/40"
            : "border-zinc-700 hover:border-zinc-600"
        )}
      >
        <textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          rows={1}
          placeholder="用一句话描述生物学假说，例如：EGF stimulation induces transient ERK activation…"
          disabled={isStreaming}
          className="flex-1 resize-none bg-transparent px-2 py-2 text-sm leading-relaxed text-zinc-100 placeholder:text-zinc-600 focus:outline-none disabled:opacity-60"
        />
        {isStreaming ? (
          <button
            type="button"
            onClick={() => void stopGeneration()}
            className="inline-flex h-9 shrink-0 items-center gap-1.5 rounded-lg bg-red-600/90 px-3 text-sm font-medium text-white transition-colors hover:bg-red-500"
          >
            <Square className="h-3.5 w-3.5" />
            停止
          </button>
        ) : (
          <button
            type="button"
            onClick={handleSubmit}
            disabled={!value.trim()}
            className="inline-flex h-9 shrink-0 items-center gap-1.5 rounded-lg bg-blue-600 px-4 text-sm font-medium text-white transition-colors hover:bg-blue-500 disabled:cursor-not-allowed disabled:bg-zinc-700 disabled:text-zinc-500"
          >
            模拟
            <ArrowUp className="h-3.5 w-3.5" />
          </button>
        )}
      </div>

      {/* 示例 chip —— 新用户零配置上手 */}
      <div className="flex flex-wrap gap-2">
        {EXAMPLES.map((ex) => (
          <button
            key={ex.label}
            type="button"
            onClick={() => handleExample(ex.text)}
            disabled={isStreaming}
            className="rounded-full border border-zinc-800 bg-zinc-900/60 px-3 py-1 text-[11px] text-zinc-400 transition-colors hover:border-zinc-600 hover:text-zinc-200 disabled:opacity-40"
          >
            {ex.label}
          </button>
        ))}
      </div>
    </div>
  );
}
