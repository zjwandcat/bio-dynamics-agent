/**
 * SSE client for the BioDynamics v4 Official Benchmark Suite (Task C.12).
 *
 * Subscribes to `POST /api/v4/benchmarks/run` — the streaming endpoint added
 * in backend Task E.1. The backend runs all 10 pathway benchmarks
 * sequentially and emits Server-Sent Events in this order:
 *
 *   - benchmark_start    { pathway_class, name }   (suite-level uses "__suite__")
 *   - benchmark_progress { pathway_class, step }
 *   - benchmark_result   { pathway_class, name, status, checks, runtime_seconds, errors }
 *   - benchmark_complete { total, passed, failed, results, runtime_seconds }
 *   - error              string
 *   - end                ""
 *
 * The SSE wire format is `data: {"event": "...", "data": {...}}\n\n`, mirroring
 * the existing `/api/chat` contract parsed in `lib/sse.ts`.
 *
 * Two entry points:
 *   - runAllBenchmarksStream: relays every event (used by "Run All").
 *   - runBenchmarkStream:     connects to the same endpoint but only relays
 *     events for a single `pathwayClass`, aborting once that pathway's
 *     `benchmark_result` arrives (used by the per-card "Run" button).
 */

import { API_BASE, V4_PREFIX } from "./api";

const BENCHMARKS_RUN_PATH = `${V4_PREFIX}/benchmarks/run`;

/** Marker emitted by the backend for the suite-level start event. */
export const SUITE_MARKER = "__suite__";

// ---------------------------------------------------------------------------
// Event payload types (mirror backend/app/benchmark_runner.py schema)
// ---------------------------------------------------------------------------

export type BenchmarkStatus = "pass" | "fail";

export interface BenchmarkCheck {
  criterion: string;
  metric_name: string;
  passed: boolean;
  detail: string;
}

export interface BenchmarkResultEvent {
  pathway_class: string;
  name: string;
  status: BenchmarkStatus;
  checks: BenchmarkCheck[];
  runtime_seconds: number;
  errors: string[];
}

export interface BenchmarkSummary {
  total: number;
  passed: number;
  failed: number;
  results: BenchmarkResultEvent[];
  runtime_seconds: number;
}

export interface BenchmarkStartData {
  pathway_class: string;
  name?: string;
  total?: number;
}

export interface BenchmarkProgressData {
  pathway_class: string;
  step: string;
}

export type BenchmarkEventType =
  | "benchmark_start"
  | "benchmark_progress"
  | "benchmark_result"
  | "benchmark_complete"
  | "error"
  | "end";

export interface BenchmarkStreamEvent {
  event: BenchmarkEventType;
  data: unknown;
}

// ---------------------------------------------------------------------------
// Internal SSE frame parser (shared by both entry points)
// ---------------------------------------------------------------------------

/**
 * Open a POST SSE connection to /api/v4/benchmarks/run and feed every parsed
 * event to `onEvent`. Resolves when the stream closes (success, error, or
 * abort). The optional `shouldAbort` predicate lets the per-pathway runner
 * short-circuit once its target result arrives.
 *
 * Returns nothing; callers own the AbortController via the `signal` arg.
 */
async function openBenchmarkStream(
  onEvent: (event: BenchmarkStreamEvent) => void,
  onError: (error: Error) => void,
  signal?: AbortSignal,
  shouldAbort?: (event: BenchmarkStreamEvent) => boolean
): Promise<void> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${BENCHMARKS_RUN_PATH}`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify({}),
      signal,
    });
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") return;
    onError(err instanceof Error ? err : new Error("连接后端失败"));
    return;
  }

  if (!response.ok) {
    onError(new Error(`请求失败：${response.status} ${response.statusText}`));
    return;
  }

  const reader = response.body?.getReader();
  if (!reader) {
    onError(new Error("响应流不可用"));
    return;
  }

  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed.startsWith("data:")) continue;

        const payloadText = trimmed.slice(5).trim();
        if (!payloadText || payloadText === "[DONE]") continue;

        let parsed: { event?: string; data?: unknown } = {};
        try {
          parsed = JSON.parse(payloadText);
        } catch {
          continue;
        }

        const evt: BenchmarkStreamEvent = {
          event: (parsed.event as BenchmarkEventType) ?? "",
          data: parsed.data,
        };
        onEvent(evt);

        if (shouldAbort?.(evt)) {
          // Target event received — cancel the underlying stream.
          try {
            await reader.cancel();
          } catch {
            /* noop */
          }
          return;
        }
      }
    }
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") return;
    onError(err instanceof Error ? err : new Error("流读取失败"));
  } finally {
    try {
      reader.releaseLock();
    } catch {
      /* noop */
    }
  }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Run a single pathway benchmark by streaming from
 * `/api/v4/benchmarks/run` and relaying only the events whose
 * `pathway_class` matches `pathwayClass` (plus terminal `error`/`end`).
 *
 * The backend always runs all 10 sequentially; this function cancels the
 * stream as soon as the matching `benchmark_result` arrives so the remaining
 * pathways are not executed unnecessarily.
 *
 * @param pathwayClass  Target pathway_class identifier (e.g. "EGFR_RTK").
 * @param onEvent       Called for each matching event.
 * @param signal        Optional AbortSignal to cancel the run.
 */
export async function runBenchmarkStream(
  pathwayClass: string,
  onEvent: (event: BenchmarkStreamEvent) => void,
  signal?: AbortSignal
): Promise<void> {
  await openBenchmarkStream(
    (evt) => {
      // Relay only events relevant to this pathway, plus terminal events.
      const pc = (evt.data as { pathway_class?: string } | null)?.pathway_class;
      const isMatch = pc === pathwayClass;
      const isTerminal = evt.event === "error" || evt.event === "end";
      if (isMatch || isTerminal) {
        onEvent(evt);
      }
    },
    (err) => {
      onEvent({ event: "error", data: err.message });
    },
    signal,
    // Abort once the matching result lands.
    (evt) =>
      evt.event === "benchmark_result" &&
      (evt.data as { pathway_class?: string } | null)?.pathway_class ===
        pathwayClass
  );
}

/**
 * Run the full 10-pathway suite, relaying every SSE event to `onEvent`.
 *
 * @param onEvent     Called for every parsed event (start/progress/result/complete/error/end).
 * @param onComplete  Called once with the final summary when `benchmark_complete` arrives.
 * @param signal      Optional AbortSignal to cancel the run.
 */
export async function runAllBenchmarksStream(
  onEvent: (event: BenchmarkStreamEvent) => void,
  onComplete: (summary: BenchmarkSummary) => void,
  signal?: AbortSignal
): Promise<void> {
  await openBenchmarkStream(
    (evt) => {
      onEvent(evt);
      if (evt.event === "benchmark_complete" && evt.data && typeof evt.data === "object") {
        onComplete(evt.data as BenchmarkSummary);
      }
    },
    (err) => {
      onEvent({ event: "error", data: err.message });
    },
    signal
  );
}

/** Type guard: extract a typed benchmark_result payload from a stream event. */
export function isBenchmarkResult(
  event: BenchmarkStreamEvent
): event is BenchmarkStreamEvent & { data: BenchmarkResultEvent } {
  return (
    event.event === "benchmark_result" &&
    event.data !== null &&
    typeof event.data === "object" &&
    "status" in (event.data as Record<string, unknown>)
  );
}
