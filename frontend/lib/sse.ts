/**
 * SSE manager for the BioDynamics AI Assistant chat stream.
 *
 * Contract (Task C.1):
 * - This subscribes to the EXISTING `/api/chat` endpoint — the v3 SSE contract
 *   shared by `backend/app/main.py`. It must NOT be altered or version-bumped.
 * - All event types emitted by the backend (v1/v2/v3/v4 flows) are parsed here
 *   and forwarded verbatim to the caller via `onEvent`. The caller (the global
 *   store) owns the event-type-specific state transitions.
 *
 * The parsing logic is lifted directly from the previous `app/page.tsx` inline
 * reader so behaviour is preserved bit-for-bit.
 */

import { API_BASE, V3_PREFIX } from "./api";

/** A single parsed SSE event from the chat stream. */
export interface SSEEvent {
  /** The `event` field emitted by the backend (e.g. `agent_dispatch`). */
  event: string;
  /** The `data` field emitted by the backend (object, array, string, or null). */
  data: unknown;
}

/** Payload sent to `/api/chat` to start a stream. */
export interface ChatStreamPayload {
  user_input: string;
  thread_id: string;
  mode: string;
  manual_modules: string[];
}

/** Callbacks invoked while streaming. */
export interface StreamChatHandlers {
  /** Called for every parsed SSE event. */
  onEvent: (event: SSEEvent) => void;
  /** Called if the fetch / stream fails (network error, non-2xx, etc.). */
  onError?: (error: Error) => void;
  /** Called once the stream finishes (success or after error). Always runs once. */
  onDone?: () => void;
}

/**
 * Start a chat SSE stream against `/api/chat`.
 *
 * Returns the AbortController so the caller can cancel the in-flight request
 * (used by the Stop button). The promise resolves when the stream closes.
 */
export async function streamChat(
  payload: ChatStreamPayload,
  handlers: StreamChatHandlers,
  signal?: AbortSignal
): Promise<void> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${V3_PREFIX}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal,
    });
  } catch (err) {
    const error =
      err instanceof Error ? err : new Error("连接后端失败");
    handlers.onError?.(error);
    handlers.onDone?.();
    return;
  }

  if (!response.ok) {
    handlers.onError?.(
      new Error(`请求失败：${response.status} ${response.statusText}`)
    );
    handlers.onDone?.();
    return;
  }

  const reader = response.body?.getReader();
  if (!reader) {
    handlers.onError?.(new Error("响应流不可用"));
    handlers.onDone?.();
    return;
  }

  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      // SSE frames are newline-delimited in this backend's streaming format.
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed.startsWith("data:")) continue;

        const payloadText = trimmed.slice(5).trim();
        if (payloadText === "[DONE]") continue;
        if (!payloadText) continue;

        let parsed: { event?: string; data?: unknown } = {};
        try {
          parsed = JSON.parse(payloadText);
        } catch {
          // Malformed JSON frames are silently skipped (matches legacy behavior).
          continue;
        }

        handlers.onEvent({
          event: parsed.event ?? "",
          data: parsed.data,
        });
      }
    }
  } catch (err) {
    // AbortError is expected when the user hits Stop — treat as benign.
    if (err instanceof DOMException && err.name === "AbortError") {
      handlers.onDone?.();
      return;
    }
    const error = err instanceof Error ? err : new Error("流读取失败");
    handlers.onError?.(error);
  } finally {
    try {
      reader.releaseLock();
    } catch {
      /* noop */
    }
    handlers.onDone?.();
  }
}
