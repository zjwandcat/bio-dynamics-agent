import Link from "next/link";
import { Atom, BarChart3 } from "lucide-react";
import { PathwaySelector } from "@/components/home/PathwayCard";
import { RecentSimulations } from "@/components/home/RecentSimulations";
import { BenchmarkCases } from "@/components/home/BenchmarkCards";

/**
 * Home page (Task C.10) — BioDynamics Agent landing surface.
 *
 * A scientific-instrument console, not a marketing page. Four sections encode
 * the researcher's workflow as a sequence: orient (hero) → select pathway →
 * reopen recent work → validate against benchmarks. The hero's graph-paper
 * grid backdrop + monospaced section markers evoke simulation plot axes / a
 * lab notebook, reusing the workbench's own zinc-950 + blue palette so the
 * home page reads as the same instrument as /workspace.
 *
 * The hero is a server component (CTAs are plain Links); the three lower
 * sections are client components because they read localStorage and hydrate
 * the global workbench store before navigating.
 */
export default function HomePage() {
  return (
    <main className="min-h-screen bg-zinc-950 text-zinc-100">
      {/* ── 01 / Hero ─────────────────────────────────────────────────── */}
      <section
        className="relative overflow-hidden border-b border-zinc-800"
        style={{
          backgroundImage:
            "linear-gradient(to right, rgba(255,255,255,0.035) 1px, transparent 1px), linear-gradient(to bottom, rgba(255,255,255,0.035) 1px, transparent 1px)",
          backgroundSize: "32px 32px",
        }}
      >
        {/* blue glow */}
        <div
          aria-hidden
          className="pointer-events-none absolute -top-40 left-1/4 h-80 w-[40rem] rounded-full opacity-20 blur-3xl"
          style={{
            background:
              "radial-gradient(circle, #3b82f6 0%, transparent 70%)",
          }}
        />
        <div className="relative mx-auto max-w-6xl px-6 py-20 sm:py-28">
          <div className="font-mono text-[11px] uppercase tracking-[0.2em] text-blue-400/80">
            BioDynamics · v4 RC
          </div>
          <h1 className="mt-4 text-5xl font-semibold tracking-tight text-zinc-50 sm:text-6xl">
            BioDynamics Agent
          </h1>
          <p className="mt-3 text-lg text-zinc-300">
            Scientific Modeling IDE for Signal Pathway Simulation
          </p>
          <p className="mt-4 max-w-2xl text-sm leading-relaxed text-zinc-400">
            Build, simulate, and validate ODE models of cancer signaling
            networks against curated BioModels references. Ten canonical
            pathways, agent-driven hypothesis generation, and a five-tier
            validation pyramid — in one workbench.
          </p>

          <div className="mt-8 flex flex-wrap items-center gap-3">
            <Link
              href="/workspace"
              className="inline-flex h-10 items-center gap-2 rounded-lg bg-blue-600 px-5 text-sm font-medium text-white transition-colors hover:bg-blue-500"
            >
              <Atom className="h-4 w-4" />
              Launch Workspace
            </Link>
            <Link
              href="/benchmarks"
              className="inline-flex h-10 items-center gap-2 rounded-lg border border-zinc-700 bg-zinc-900 px-5 text-sm font-medium text-zinc-200 transition-colors hover:bg-zinc-800"
            >
              <BarChart3 className="h-4 w-4" />
              Run Benchmarks
            </Link>
          </div>

          {/* instrument readout strip */}
          <div className="mt-10 flex flex-wrap items-center gap-x-6 gap-y-2 border-t border-zinc-800/80 pt-4 font-mono text-[11px] uppercase tracking-wider text-zinc-500">
            <span>
              <span className="text-zinc-300">10</span> pathways
            </span>
            <span className="text-zinc-700">·</span>
            <span>
              <span className="text-zinc-300">10</span> benchmarks
            </span>
            <span className="text-zinc-700">·</span>
            <span>
              <span className="text-zinc-300">5-tier</span> validation
            </span>
            <span className="text-zinc-700">·</span>
            <span className="text-blue-400/80">sprint C · RC</span>
          </div>
        </div>
      </section>

      {/* ── 02 / 03 / 04 — lower sections ────────────────────────────── */}
      <div className="mx-auto max-w-6xl px-6">
        <PathwaySelector />
        <RecentSimulations />
        <BenchmarkCases />
      </div>

      <footer className="border-t border-zinc-800 px-6 py-6 text-center font-mono text-[11px] text-zinc-600">
        BioDynamics Agent · v4 RC — Scientific Modeling IDE
      </footer>
    </main>
  );
}
