import type { Metadata } from "next";
import Link from "next/link";
import { FileText, ArrowLeft } from "lucide-react";

import { ReportViewer } from "@/components/report/ReportViewer";

export const metadata: Metadata = {
  title: "Experiment Report — BioDynamics v4",
  description: "Persisted experiment report viewer",
};

/**
 * /report/[id] — Experiment Report viewer (Task C.11).
 *
 * Server component that awaits the async `params` (Next.js 16) and delegates
 * all interactivity — fetching, loading/error states, export buttons, the 6
 * collapsible report sections, and the embedded Recharts views — to the client
 * `ReportViewer` component.
 *
 * The report payload is fetched client-side via `fetchReport(id)` from
 * `/api/v4/reports/:id`. The page renders a fixed dark-themed shell (header
 * with a back-link + report id badge) and lets the viewer fill the body.
 */
export default async function ReportPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;

  return (
    <main className="flex min-h-screen flex-col bg-zinc-950 text-zinc-100">
      <header className="flex h-12 shrink-0 items-center gap-3 border-b border-zinc-800 px-4 print:hidden">
        <Link
          href="/workspace"
          className="flex items-center gap-1.5 text-xs text-zinc-400 hover:text-zinc-100"
        >
          <ArrowLeft className="h-3.5 w-3.5" />
          Back to Workspace
        </Link>
        <div className="flex items-center gap-2">
          <FileText className="h-4 w-4 text-blue-400" />
          <span className="text-sm font-semibold">Experiment Report</span>
        </div>
        <span className="ml-auto rounded-full border border-zinc-700 px-2 py-0.5 font-mono text-[10px] text-zinc-500">
          {id}
        </span>
      </header>
      <div className="flex flex-1 flex-col overflow-y-auto">
        <ReportViewer id={id} />
      </div>
    </main>
  );
}
