"use client";

import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Square, SquareCheck, Loader2, SquareX } from "lucide-react";

export interface ClarificationOption {
  id: string;
  label: string;
}

export interface ClarificationRequest {
  question: string;
  options: ClarificationOption[];
  context?: string;
}

export interface ClarificationAnswer {
  selected_option: string;
  free_text?: string;
}

interface ClarificationDialogProps {
  request: ClarificationRequest;
  onSubmit: (answer: ClarificationAnswer) => void;
  onStop: () => void;
  disabled?: boolean;
}

export function ClarificationDialog({
  request,
  onSubmit,
  onStop,
  disabled = false,
}: ClarificationDialogProps) {
  const [selected, setSelected] = useState<string | null>(null);
  const [freeText, setFreeText] = useState("");

  const customOption = request.options.find((o) => o.id === "C");
  const isCustomSelected = selected === "C";

  const handleSubmit = () => {
    if (!selected) return;
    onSubmit({
      selected_option: selected,
      free_text: isCustomSelected ? freeText : undefined,
    });
  };

  return (
    <Card className="border-amber-500/50 bg-amber-950/20">
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-2 text-sm font-semibold text-amber-300">
          <Loader2 className="h-4 w-4 animate-spin" />
          需要人工干预
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <p className="text-sm text-zinc-200">{request.question}</p>

        <div className="space-y-2">
          {request.options.map((option) => {
            const isSelected = selected === option.id;
            const isCustom = option.id === "C";
            return (
              <button
                key={option.id}
                type="button"
                disabled={disabled}
                onClick={() => setSelected(option.id)}
                className={`flex w-full items-start gap-2 rounded-md border px-3 py-2 text-left text-sm transition-colors ${
                  isSelected
                    ? "border-amber-500/70 bg-amber-500/10 text-amber-100"
                    : "border-zinc-700 bg-zinc-900/50 text-zinc-300 hover:bg-zinc-800"
                } ${disabled ? "opacity-60 cursor-not-allowed" : ""}`}
              >
                <span className="mt-0.5 shrink-0">
                  {isSelected ? (
                    <SquareCheck className="h-4 w-4 text-amber-400" />
                  ) : (
                    <Square className="h-4 w-4 text-zinc-500" />
                  )}
                </span>
                <span className="flex-1">
                  <span className="font-medium">{option.id}.</span> {option.label}
                </span>
              </button>
            );
          })}
        </div>

        {customOption && isCustomSelected && (
          <Textarea
            placeholder="请填写自定义方案..."
            value={freeText}
            onChange={(e) => setFreeText(e.target.value)}
            disabled={disabled}
            className="min-h-[80px] border-zinc-700 bg-zinc-900 text-zinc-100 placeholder:text-zinc-500 focus-visible:ring-amber-500"
          />
        )}

        <div className="flex items-center gap-2 pt-1">
          <Button
            size="sm"
            onClick={handleSubmit}
            disabled={disabled || !selected || (isCustomSelected && !freeText.trim())}
            className="bg-amber-600 text-white hover:bg-amber-500"
          >
            {disabled ? <Loader2 className="mr-1 h-3.5 w-3.5 animate-spin" /> : null}
            提交回答
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={onStop}
            disabled={disabled}
            className="border-red-800 text-red-300 hover:bg-red-950/50 hover:text-red-200"
          >
            <SquareX className="mr-1 h-3.5 w-3.5" />
            停止生成
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
