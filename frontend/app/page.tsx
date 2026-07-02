"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import { ChatMessage, ChatMessageProps } from "@/components/chat/ChatMessage";
import { ChatInput } from "@/components/chat/ChatInput";
import { Button } from "@/components/ui/button";
import { Trash2, RefreshCw } from "lucide-react";

interface Message {
  id: string;
  role: ChatMessageProps["role"];
  content: string;
  type: ChatMessageProps["type"];
  tokenUsage?: number;
}

function generateId() {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

const API_BASE = "http://localhost:8000";

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [threadId, setThreadId] = useState(() => generateId());
  const [tokenUsage, setTokenUsage] = useState(0);
  const [isUpdatingDb, setIsUpdatingDb] = useState(false);
  const [updateDbStatus, setUpdateDbStatus] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  const appendMessage = useCallback((message: Omit<Message, "id">) => {
    setMessages((prev) => [...prev, { ...message, id: generateId() }]);
  }, []);

  const updateLastStatus = useCallback((content: string) => {
    setMessages((prev) => {
      const last = prev[prev.length - 1];
      if (last && last.role === "agent" && last.type === "status") {
        return [...prev.slice(0, -1), { ...last, content }];
      }
      return [
        ...prev,
        { id: generateId(), role: "agent" as const, content, type: "status" as const },
      ];
    });
  }, []);

  const removeTrailingStatus = useCallback(() => {
    setMessages((prev) => {
      let i = prev.length - 1;
      while (i >= 0 && prev[i].role === "agent" && prev[i].type === "status") {
        i -= 1;
      }
      return prev.slice(0, i + 1);
    });
  }, []);

  const setLastAgentTokenUsage = useCallback((usage: number) => {
    setMessages((prev) => {
      for (let i = prev.length - 1; i >= 0; i--) {
        if (prev[i].role === "agent") {
          const updated = [...prev];
          updated[i] = { ...updated[i], tokenUsage: usage };
          return updated;
        }
      }
      return prev;
    });
  }, []);

  const handleClearMemory = useCallback(async () => {
    try {
      await fetch(`${API_BASE}/api/chat/clear-memory`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ thread_id: threadId }),
      });
    } catch (err) {
      // 即使后端清空失败，前端也重置当前会话
      console.error("清空记忆失败", err);
    }
    setMessages([]);
    setThreadId(generateId());
    setTokenUsage(0);
  }, [threadId]);

  const handleUpdateVectorDb = useCallback(async () => {
    setIsUpdatingDb(true);
    setUpdateDbStatus("");
    try {
      const response = await fetch(`${API_BASE}/api/admin/update-vector-db`, {
        method: "POST",
      });
      if (!response.ok) {
        throw new Error(`请求失败：${response.status}`);
      }
      setUpdateDbStatus("知识库更新已启动");
    } catch (err) {
      const message = err instanceof Error ? err.message : "更新失败";
      setUpdateDbStatus(message);
    } finally {
      setIsUpdatingDb(false);
      setTimeout(() => setUpdateDbStatus(""), 3000);
    }
  }, []);

  const handleSend = useCallback(async () => {
    const userInput = input.trim();
    if (!userInput || isStreaming) return;

    appendMessage({ role: "user", content: userInput, type: "text" });
    setInput("");
    setIsStreaming(true);

    let codeGenCount = 0;

    try {
      abortControllerRef.current = new AbortController();
      const response = await fetch(`${API_BASE}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_input: userInput,
          thread_id: threadId,
        }),
        signal: abortControllerRef.current.signal,
      });

      if (!response.ok) {
        throw new Error(`请求失败：${response.status} ${response.statusText}`);
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error("响应流不可用");
      }

      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed.startsWith("data:")) continue;

          const payload = trimmed.slice(5).trim();
          if (payload === "[DONE]") continue;
          if (!payload) continue;

          let event: { event?: string; data?: unknown } = {};
          try {
            event = JSON.parse(payload);
          } catch {
            continue;
          }

          const eventType = event.event;
          const eventData = event.data;

          switch (eventType) {
            case "node_start": {
              const text = typeof eventData === "string" ? eventData : "正在处理...";
              updateLastStatus(text);
              break;
            }

            case "code_generated": {
              codeGenCount += 1;
              removeTrailingStatus();
              const code = typeof eventData === "string" ? eventData : "";
              appendMessage({ role: "agent", content: code, type: "code" });
              break;
            }

            case "execution_log": {
              removeTrailingStatus();
              const rawLog = typeof eventData === "string" ? eventData : "";
              const isRetry = codeGenCount > 1;
              const content = isRetry
                ? `⚠️ 仿真出错，正在自动纠错重试 (${codeGenCount - 1}/3)...\n${rawLog}`
                : rawLog;
              appendMessage({ role: "agent", content, type: "log" });
              break;
            }

            case "image_ready": {
              removeTrailingStatus();
              const imageBase64 = typeof eventData === "string" ? eventData : "";
              appendMessage({ role: "agent", content: imageBase64, type: "image" });
              break;
            }

            case "report_ready": {
              removeTrailingStatus();
              const report = typeof eventData === "string" ? eventData : "";
              appendMessage({ role: "agent", content: report, type: "report" });
              break;
            }

            case "token_usage": {
              if (typeof eventData === "object" && eventData !== null) {
                const usage = eventData as { total_tokens?: number };
                const total = usage.total_tokens ?? 0;
                setTokenUsage(total);
                setLastAgentTokenUsage(total);
              }
              break;
            }

            case "error": {
              removeTrailingStatus();
              const errorText = typeof eventData === "string" ? eventData : "发生未知错误";
              appendMessage({ role: "agent", content: errorText, type: "text" });
              break;
            }

            case "end": {
              removeTrailingStatus();
              break;
            }

            default:
              break;
          }
        }
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : "连接后端失败";
      removeTrailingStatus();
      appendMessage({ role: "agent", content: message, type: "text" });
    } finally {
      setIsStreaming(false);
      abortControllerRef.current = null;
    }
  }, [
    input,
    isStreaming,
    threadId,
    appendMessage,
    updateLastStatus,
    removeTrailingStatus,
    setLastAgentTokenUsage,
  ]);

  useEffect(() => {
    return () => {
      abortControllerRef.current?.abort();
    };
  }, []);

  return (
    <div className="flex h-screen flex-col bg-zinc-900 text-zinc-100">
      <header className="flex h-14 items-center justify-between border-b border-zinc-800 px-6">
        <h1 className="text-lg font-semibold">BioDynamics Agent</h1>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={handleUpdateVectorDb}
            disabled={isUpdatingDb}
            className="h-8 gap-1.5 border-zinc-700 bg-zinc-800/80 text-zinc-200 hover:bg-zinc-700 hover:text-zinc-100"
          >
            <RefreshCw
              className={`h-3.5 w-3.5 ${isUpdatingDb ? "animate-spin" : ""}`}
            />
            <span className="hidden sm:inline">
              {isUpdatingDb ? "更新中..." : "更新知识库"}
            </span>
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={handleClearMemory}
            disabled={isStreaming}
            className="h-8 gap-1.5 border-zinc-700 bg-zinc-800/80 text-zinc-200 hover:bg-red-900/30 hover:text-red-200"
          >
            <Trash2 className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">清除当前对话</span>
          </Button>
        </div>
      </header>

      {updateDbStatus && (
        <div className="border-b border-zinc-800 bg-zinc-800/50 px-6 py-1.5 text-center text-xs text-zinc-400">
          {updateDbStatus}
        </div>
      )}

      <main className="flex flex-1 flex-col overflow-hidden">
        <div className="flex-1 overflow-y-auto px-4 py-6 sm:px-6 lg:px-8">
          <div className="mx-auto max-w-3xl space-y-5">
            {messages.length === 0 && (
              <div className="flex flex-col items-center justify-center gap-3 pt-32 text-zinc-500">
                <h2 className="text-2xl font-semibold text-zinc-300">BioDynamics Agent</h2>
                <p className="text-sm">输入生物学假说，Agent 将自动建模并运行仿真。</p>
              </div>
            )}
            {messages.map((msg) => (
              <ChatMessage
                key={msg.id}
                role={msg.role}
                content={msg.content}
                type={msg.type}
                tokenUsage={msg.tokenUsage}
              />
            ))}
            <div ref={messagesEndRef} />
          </div>
        </div>

        <div className="border-t border-zinc-800 bg-zinc-900/95 px-4 pb-4 pt-3 sm:px-6 lg:px-8">
          <div className="mx-auto max-w-3xl">
            <ChatInput
              value={input}
              onChange={setInput}
              onSend={handleSend}
              disabled={isStreaming}
            />
          </div>
        </div>
      </main>
    </div>
  );
}
