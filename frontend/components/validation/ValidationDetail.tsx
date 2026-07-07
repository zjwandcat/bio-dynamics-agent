"use client";

import React from "react";
import {
  CheckCircle2,
  XCircle,
  AlertTriangle,
  MinusCircle,
} from "lucide-react";
import { cn } from "@/lib/utils";

/**
 * ValidationDetail — expanded view for a single validation level.
 *
 * Rendered inside a `ValidationPyramid` level card when the user clicks to
 * expand it. Receives a pre-flattened list of `ValidationCheck` items (one per
 * sub-check the level performed) so this component stays a pure presentational
 * leaf with no knowledge of the backend report shape.
 *
 * Task C.7.
 */

export type CheckStatus = "pass" | "warning" | "fail" | "skipped";

export interface ValidationCheck {
  /** Display name of the sub-check (e.g. "Mass Conservation"). */
  name: string;
  status: CheckStatus;
  /** Measured value, already formatted for display (e.g. "3.20%"). */
  value?: string;
  /** Threshold, already formatted (e.g. "≤ 5%"). */
  threshold?: string;
  /** Error / warning detail line. Rendered in the status colour. */
  message?: string;
}

export interface ValidationDetailProps {
  /** Level number (1–5), used for stable list keys. */
  level: number;
  checks: ValidationCheck[];
  /** Optional level-level note (e.g. skipped reason or method). */
  notes?: string;
  className?: string;
}

const STATUS_ICON: Record<CheckStatus, React.ReactNode> = {
  pass: <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400" />,
  warning: <AlertTriangle className="h-3.5 w-3.5 text-amber-400" />,
  fail: <XCircle className="h-3.5 w-3.5 text-red-400" />,
  skipped: <MinusCircle className="h-3.5 w-3.5 text-zinc-500" />,
};

const STATUS_MESSAGE_COLOR: Record<CheckStatus, string> = {
  pass: "text-emerald-300/80",
  warning: "text-amber-300/90",
  fail: "text-red-300/90",
  skipped: "text-zinc-500",
};

export function ValidationDetail({
  level,
  checks,
  notes,
  className,
}: ValidationDetailProps) {
  return (
    <div
      className={cn(
        "space-y-2 border-t border-zinc-800/80 bg-zinc-950/40 px-2.5 py-2",
        className
      )}
    >
      {notes && (
        <p className="text-[10px] leading-relaxed text-zinc-500">{notes}</p>
      )}

      <ul className="space-y-1.5">
        {checks.map((check, idx) => (
          <li
            key={`${level}-${idx}-${check.name}`}
            className="space-y-0.5"
          >
            <div className="flex items-start gap-2">
              <span className="mt-0.5 shrink-0" aria-hidden>
                {STATUS_ICON[check.status]}
              </span>
              <div className="min-w-0 flex-1">
                <div className="flex items-baseline justify-between gap-2">
                  <span className="truncate text-[11px] font-medium text-zinc-200">
                    {check.name}
                  </span>
                  {check.value !== undefined && (
                    <span className="shrink-0 font-mono text-[10px] text-zinc-400">
                      {check.value}
                    </span>
                  )}
                </div>
                {(check.threshold || check.message) && (
                  <div className="mt-0.5 flex flex-wrap items-center gap-x-2 gap-y-0.5 text-[10px] leading-snug">
                    {check.threshold && (
                      <span className="text-zinc-500">
                        threshold {check.threshold}
                      </span>
                    )}
                    {check.message && (
                      <span
                        className={cn(
                          "break-words",
                          STATUS_MESSAGE_COLOR[check.status]
                        )}
                      >
                        {check.message}
                      </span>
                    )}
                  </div>
                )}
              </div>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}
