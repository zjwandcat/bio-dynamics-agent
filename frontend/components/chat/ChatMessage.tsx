"use client";

import React from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

export interface ChatMessageProps {
  role: "user" | "agent";
  content?: string;
  type?: "text" | "code" | "image" | "log" | "status" | "report";
  tokenUsage?: number;
}

export function ChatMessage({
  role,
  content = "",
  type = "text",
  tokenUsage,
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
            : "bg-zinc-800"
        )}
      >
        {renderAgentContent()}
        {!isUser && tokenUsage !== undefined && tokenUsage > 0 && (
          <div className="mt-2 border-t border-zinc-700/50 pt-1.5 text-right text-xs text-zinc-500">
            Tokens: {tokenUsage.toLocaleString()}
          </div>
        )}
      </div>
    </div>
  );
}
