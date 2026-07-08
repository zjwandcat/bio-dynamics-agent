"use client";

import React from "react";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { Send } from "lucide-react";

export interface ChatInputProps {
  value: string;
  onChange: (value: string) => void;
  onSend: () => void;
  disabled?: boolean;
}

export function ChatInput({
  value,
  onChange,
  onSend,
  disabled = false,
}: ChatInputProps) {
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!disabled && value.trim()) {
        onSend();
      }
    }
  };

  return (
    <div className="flex items-end gap-2 rounded-2xl border border-zinc-700 bg-zinc-800/80 p-2 backdrop-blur-sm">
      <Textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="输入生物学假说，例如：A 抑制 B..."
        disabled={disabled}
        rows={1}
        className="field-sizing-fixed max-h-40 min-h-0 resize-none border-0 bg-transparent px-3 py-2.5 text-zinc-100 placeholder:text-zinc-500 focus-visible:ring-0 focus-visible:ring-offset-0"
      />
      <Button
        onClick={onSend}
        disabled={disabled || !value.trim()}
        size="icon"
        aria-label="发送"
        className="shrink-0 rounded-xl bg-blue-600 hover:bg-blue-700 disabled:opacity-50"
      >
        <Send className="h-4 w-4" />
      </Button>
    </div>
  );
}
