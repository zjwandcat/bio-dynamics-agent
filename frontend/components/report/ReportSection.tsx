"use client";

import React, { useState } from "react";
import { ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";

interface ReportSectionProps {
  /** 1-based section number shown as a leading badge. */
  number: number;
  title: string;
  icon?: React.ReactNode;
  /** When true the section can be collapsed; defaults to true. */
  collapsible?: boolean;
  /** Initial expanded state; defaults to true. */
  defaultOpen?: boolean;
  /** Optional right-aligned actions (e.g. export buttons). */
  actions?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}

/**
 * ReportSection — reusable wrapper for the 6 sections of the Experiment Report.
 *
 * Renders a numbered title bar with an icon + optional actions, and a body that
 * can be collapsed (chevron toggle). All sections default to expanded so the
 * report reads top-to-bottom on first load; collapsing is for navigation once
 * the reader has scanned the document.
 */
export function ReportSection({
  number,
  title,
  icon,
  collapsible = true,
  defaultOpen = true,
  actions,
  children,
  className,
}: ReportSectionProps) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <section
      className={cn(
        "overflow-hidden rounded-lg border border-zinc-800 bg-zinc-900/60",
        className
      )}
    >
      <div className="flex items-center gap-2.5 border-b border-zinc-800/80 bg-zinc-900/40 px-4 py-2.5">
        <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-md bg-emerald-500/15 font-mono text-[10px] font-semibold text-emerald-300">
          {number}
        </span>
        {icon && <span className="shrink-0 text-zinc-400">{icon}</span>}
        <h2 className="min-w-0 flex-1 truncate text-sm font-semibold text-zinc-100">
          {title}
        </h2>
        {actions && <div className="flex shrink-0 items-center gap-1.5">{actions}</div>}
        {collapsible && (
          <button
            type="button"
            onClick={() => setOpen((v) => !v)}
            aria-expanded={open}
            aria-label={open ? "Collapse section" : "Expand section"}
            className="shrink-0 rounded p-1 text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-200"
          >
            <ChevronDown
              className={cn(
                "h-3.5 w-3.5 transition-transform",
                open ? "rotate-0" : "-rotate-90"
              )}
            />
          </button>
        )}
      </div>
      {open && (
        <div className="px-4 py-3.5 text-zinc-300">{children}</div>
      )}
    </section>
  );
}
