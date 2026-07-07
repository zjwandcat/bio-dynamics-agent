/**
 * Unit tests for the unified BioDynamics v4 API client (lib/api.ts).
 *
 * Mocks global `fetch` and asserts the V4 endpoints are hit with the
 * correct method, path, and JSON body. Covers:
 *   - getJSON / postJSON shared helpers
 *   - runSimulation (POST /api/v4/simulation/run)
 *   - fetchReport (GET /api/v4/reports/:id)
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  API_BASE,
  V4_PREFIX,
  getJSON,
  postJSON,
  runSimulation,
  fetchReport,
  fetchPathways,
  type SimulationParams,
} from "@/lib/api";

/** Build a minimal ok Response stub. */
function okResponse(body: unknown): Response {
  return {
    ok: true,
    status: 200,
    statusText: "OK",
    json: async () => body,
  } as Response;
}

describe("api client — shared helpers", () => {
  let fetchMock: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    fetchMock = vi.fn();
    globalThis.fetch = fetchMock as unknown as typeof fetch;
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("getJSON issues a GET with the Accept header", async () => {
    fetchMock.mockResolvedValueOnce(okResponse({ ok: true }));
    const data = await getJSON<{ ok: boolean }>("/api/v4/pathways");
    expect(data).toEqual({ ok: true });
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`${API_BASE}/api/v4/pathways`);
    expect(init.method).toBe("GET");
    expect(init.headers.Accept).toBe("application/json");
  });

  it("postJSON issues a POST with a JSON body", async () => {
    fetchMock.mockResolvedValueOnce(okResponse({ id: "x" }));
    const data = await postJSON<{ id: string }>("/api/v4/foo", { a: 1 });
    expect(data).toEqual({ id: "x" });
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`${API_BASE}/api/v4/foo`);
    expect(init.method).toBe("POST");
    expect(init.headers["Content-Type"]).toBe("application/json");
    expect(init.body).toBe(JSON.stringify({ a: 1 }));
  });

  it("getJSON throws on a non-ok response", async () => {
    fetchMock.mockResolvedValueOnce({
      ok: false,
      status: 500,
      statusText: "Internal Server Error",
    } as Response);
    await expect(getJSON("/api/v4/oops")).rejects.toThrow(/HTTP 500/);
  });
});

describe("api client — runSimulation", () => {
  let fetchMock: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    fetchMock = vi.fn();
    globalThis.fetch = fetchMock as unknown as typeof fetch;
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("POSTs to /api/v4/simulation/run with the simulation params", async () => {
    const params: SimulationParams = {
      pathway_class: "egfr",
      duration: 60,
      steps: 60,
      parameters: { k1: 0.1 },
    };
    const resultBody = {
      run_id: "run-123",
      pathway_class: "egfr",
      time_points: [0, 1],
      species: { ERK: [0, 1] },
    };
    fetchMock.mockResolvedValueOnce(okResponse(resultBody));

    const result = await runSimulation(params);

    expect(result).toEqual(resultBody);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`${API_BASE}${V4_PREFIX}/simulation/run`);
    expect(init.method).toBe("POST");
    expect(init.body).toBe(JSON.stringify(params));
  });
});

describe("api client — fetchReport", () => {
  let fetchMock: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    fetchMock = vi.fn();
    globalThis.fetch = fetchMock as unknown as typeof fetch;
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("GETs /api/v4/reports/:id with the id URL-encoded", async () => {
    const report = {
      id: "rep-1",
      pathway_class: "mapk",
      title: "MAPK run",
      created_at: "2026-01-01",
      markdown: "# Report",
    };
    fetchMock.mockResolvedValueOnce(okResponse(report));

    const result = await fetchReport("rep-1");

    expect(result).toEqual(report);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`${API_BASE}${V4_PREFIX}/reports/rep-1`);
    expect(init.method).toBe("GET");
  });

  it("URL-encodes special characters in the report id", async () => {
    fetchMock.mockResolvedValueOnce(
      okResponse({ id: "a b", pathway_class: "egfr", title: "", created_at: "", markdown: "" })
    );
    await fetchReport("a b");
    const [url] = fetchMock.mock.calls[0];
    expect(url).toContain("/reports/a%20b");
  });
});

describe("api client — fetchPathways", () => {
  let fetchMock: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    fetchMock = vi.fn();
    globalThis.fetch = fetchMock as unknown as typeof fetch;
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("GETs /api/v4/pathways", async () => {
    fetchMock.mockResolvedValueOnce(okResponse([]));
    await fetchPathways();
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`${API_BASE}${V4_PREFIX}/pathways`);
    expect(init.method).toBe("GET");
  });
});
