"use client";

import React from "react";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";

/**
 * A labelled placeholder for a workbench pane slot that has not been
 * implemented yet. Each placeholder cites the sprint task that will fill it
 * (C.2–C.12) so the workbench communicates its own roadmap visually.
 */
export interface PlaceholderPanelProps {
  title: string;
  description?: string;
  /** Sprint task id that will implement this pane, e.g. "C.3". */
  taskRef?: string;
  icon?: React.ReactNode;
  className?: string;
  children?: React.ReactNode;
}

export function PlaceholderPanel({
  title,
  description,
  taskRef,
  icon,
  className,
  children,
}: PlaceholderPanelProps) {
  return (
    <section
      className={cn(
        "flex min-h-0 flex-col rounded-lg border border-zinc-800 bg-zinc-900/60",
        className
      )}
    >
      <header className="flex shrink-0 items-center justify-between border-b border-zinc-800 px-3 py-2">
        <div className="flex items-center gap-2">
          {icon && <span className="text-zinc-400">{icon}</span>}
          <h3 className="text-xs font-semibold text-zinc-200">{title}</h3>
        </div>
        {taskRef && (
          <Badge
            variant="outline"
            className="border-zinc-700 text-[10px] text-zinc-500"
          >
            {taskRef}
          </Badge>
        )}
      </header>
      <div className="flex min-h-0 flex-1 flex-col items-center justify-center gap-1.5 p-4 text-center">
        {children ? (
          children
        ) : (
          <>
            <p className="text-xs text-zinc-400">{description ?? "Loading..."}</p>
            {taskRef && (
              <p className="text-[10px] text-zinc-600">
                将在 Task {taskRef} 中实现
              </p>
            )}
          </>
        )}
      </div>
    </section>
  );
}
