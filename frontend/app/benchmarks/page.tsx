import type { Metadata } from "next";
import Link from "next/link";
import { BarChart3, ArrowLeft } from "lucide-react";

export const metadata: Metadata = {
  title: "Benchmarks — BioDynamics v4",
  description: "BioModels benchmark center",
};

/**
 * /benchmarks — Benchmark Center placeholder.
 *
 * Will be implemented in Task C.12 as the BioModels reference-comparison hub
 * (RMSE / peak-error / half-life validation against curated BIOMD models).
 */
export default function BenchmarksPage() {
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
          <BarChart3 className="h-4 w-4 text-blue-400" />
          <span className="text-sm font-semibold">Benchmark Center</span>
        </div>
      </header>
      <div className="flex flex-1 flex-col items-center justify-center gap-3 p-8 text-center">
        <BarChart3 className="h-10 w-10 text-zinc-700" />
        <h1 className="text-lg font-semibold text-zinc-200">Benchmark Center</h1>
        <p className="max-w-md text-sm text-zinc-500">
          BioModels 参考对照与基准验证（RMSE / 峰值误差 / 半衰期）将在
          Task C.12 中实现。
        </p>
        <span className="rounded-full border border-zinc-700 px-2 py-0.5 text-[10px] text-zinc-500">
          C.12
        </span>
      </div>
    </main>
  );
}
