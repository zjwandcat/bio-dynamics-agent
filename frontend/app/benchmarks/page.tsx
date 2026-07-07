import type { Metadata } from "next";
import { BenchmarkCenter } from "@/components/benchmark/BenchmarkCenter";

export const metadata: Metadata = {
  title: "Benchmarks — BioDynamics v4",
  description: "BioModels benchmark center — 10-pathway Official Benchmark Suite",
};

/**
 * /benchmarks — Benchmark Center (Task C.12).
 *
 * Server-component shell that preserves the route metadata and renders the
 * client `BenchmarkCenter`, which owns the 10 pathway cards, the "Run All"
 * action, and the real-time SSE progress wired to
 * `POST /api/v4/benchmarks/run` (Task E.1 backend).
 */
export default function BenchmarksPage() {
  return <BenchmarkCenter />;
}
