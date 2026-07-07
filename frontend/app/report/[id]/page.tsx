import type { Metadata } from "next";
import Link from "next/link";
import { FileText, ArrowLeft } from "lucide-react";

export const metadata: Metadata = {
  title: "Experiment Report — BioDynamics v4",
  description: "Persisted experiment report viewer",
};

/**
 * /report/[id] — Experiment Report viewer placeholder.
 *
 * Will be implemented in Task C.11 to render a persisted simulation report
 * (markdown + validation + metrics) fetched via `fetchReport(id)` from
 * `/api/v4/reports/:id`.
 *
 * NOTE: Next.js 16 made `params` async (async request APIs). It must be
 * awaited before use.
 */
export default async function ReportPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;

  return (
    <main className="flex min-h-screen flex-col bg-zinc-950 text-zinc-100">
      <header className="flex h-12 shrink-0 items-center gap-3 border-b border-zinc-800 px-4">
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
      <div className="flex flex-1 flex-col items-center justify-center gap-3 p-8 text-center">
        <FileText className="h-10 w-10 text-zinc-700" />
        <h1 className="text-lg font-semibold text-zinc-200">Experiment Report</h1>
        <p className="max-w-md text-sm text-zinc-500">
          持久化实验报告查看器（Markdown / 验证结果 / 指标）将在
          Task C.11 中实现，届时将通过 <code className="text-zinc-400">/api/v4/reports/{id}</code> 拉取。
        </p>
        <span className="rounded-full border border-zinc-700 px-2 py-0.5 text-[10px] text-zinc-500">
          C.11
        </span>
      </div>
    </main>
  );
}
