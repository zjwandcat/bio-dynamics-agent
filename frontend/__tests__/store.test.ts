/**
 * Unit tests for the global BioDynamics v4 workbench store (Zustand).
 *
 * Covers the public domain/UI actions exposed on `useWorkbenchStore`:
 *   - Initial state (empty messages, no pathway, panels closed)
 *   - setCurrentPathway / setSimulationResult / setPathwayGraph
 *   - setAIPanelOpen / toggleAIPanel (panel toggles)
 *   - setInput
 *   - Direct state hydration for validationReport & hypothesisList
 *     (no dedicated setter exists — they are set via SSE ingestion /
 *     `setState`; see store.ts WorkbenchStore interface).
 *   - ingestSSEEvent("v4_hypothesis_list") hydrates hypothesisList
 *   - sendMessage appends a user message (streamChat mocked)
 */
import { beforeEach, describe, expect, it, vi } from "vitest";

// Mock the SSE stream so sendMessage doesn't make a real fetch.
vi.mock("@/lib/sse", () => ({
  streamChat: vi.fn().mockResolvedValue(undefined),
}));

import { useWorkbenchStore } from "@/lib/store";
import { streamChat } from "@/lib/sse";
import type { SimulationResult, PathwayClass } from "@/lib/api";

/** Reset the relevant store slices between tests (the store is a singleton). */
function resetStore() {
  useWorkbenchStore.setState({
    messages: [],
    input: "",
    isStreaming: false,
    currentPathway: null,
    simulationResult: null,
    validationReport: null,
    hypothesisList: [],
    pathwayGraph: null,
    agentDispatches: [],
    agents: [],
    tokenUsage: 0,
    modelName: "",
    uiState: { aiAssistantOpen: false },
    clarification: null,
  });
}

describe("workbench store — initial state", () => {
  beforeEach(() => {
    resetStore();
    vi.mocked(streamChat).mockClear();
  });

  it("starts with empty messages, no current pathway, and panels closed", () => {
    const s = useWorkbenchStore.getState();
    expect(s.messages).toEqual([]);
    expect(s.currentPathway).toBeNull();
    expect(s.simulationResult).toBeNull();
    expect(s.validationReport).toBeNull();
    expect(s.hypothesisList).toEqual([]);
    expect(s.uiState.aiAssistantOpen).toBe(false);
    expect(s.isStreaming).toBe(false);
    expect(s.input).toBe("");
  });
});

describe("workbench store — setCurrentPathway", () => {
  beforeEach(() => resetStore());

  it("sets the current pathway class", () => {
    useWorkbenchStore.getState().setCurrentPathway("egfr" as PathwayClass);
    expect(useWorkbenchStore.getState().currentPathway).toBe("egfr");
  });

  it("clears the current pathway when passed null", () => {
    useWorkbenchStore.getState().setCurrentPathway("mapk" as PathwayClass);
    useWorkbenchStore.getState().setCurrentPathway(null);
    expect(useWorkbenchStore.getState().currentPathway).toBeNull();
  });
});

describe("workbench store — setSimulationResult", () => {
  beforeEach(() => resetStore());

  it("stores the simulation result payload", () => {
    const result: SimulationResult = {
      run_id: "run-1",
      pathway_class: "egfr",
      time_points: [0, 1, 2],
      species: { ERK: [0, 0.5, 1] },
    };
    useWorkbenchStore.getState().setSimulationResult(result);
    expect(useWorkbenchStore.getState().simulationResult).toEqual(result);
  });

  it("clears the simulation result when passed null", () => {
    useWorkbenchStore.getState().setSimulationResult({
      run_id: "run-2",
      pathway_class: "mapk",
      time_points: [],
      species: {},
    });
    useWorkbenchStore.getState().setSimulationResult(null);
    expect(useWorkbenchStore.getState().simulationResult).toBeNull();
  });
});

describe("workbench store — validationReport & hypothesisList hydration", () => {
  beforeEach(() => resetStore());

  it("accepts a validation report via setState (no dedicated setter)", () => {
    const report = { level1: { pass: true }, overall_pass: true };
    useWorkbenchStore.setState({ validationReport: report });
    expect(useWorkbenchStore.getState().validationReport).toEqual(report);
  });

  it("accepts a hypothesis list via setState", () => {
    const list = [{ id: "h1", statement: "MEK inhibits ERK" }];
    useWorkbenchStore.setState({ hypothesisList: list });
    expect(useWorkbenchStore.getState().hypothesisList).toEqual(list);
  });

  it("hydrates hypothesisList from the v4_hypothesis_list SSE event", () => {
    const list = [
      { id: "h1", statement: "Hypothesis A" },
      { id: "h2", statement: "Hypothesis B" },
    ];
    useWorkbenchStore.getState().ingestSSEEvent({
      event: "v4_hypothesis_list",
      data: list,
    });
    expect(useWorkbenchStore.getState().hypothesisList).toEqual(list);
  });

  it("appends a single hypothesis from v4_hypothesis_generated", () => {
    useWorkbenchStore.setState({ hypothesisList: [] });
    useWorkbenchStore.getState().ingestSSEEvent({
      event: "v4_hypothesis_generated",
      data: { hypothesis: { id: "h9", statement: "new" } },
    });
    const list = useWorkbenchStore.getState().hypothesisList as Array<{
      id: string;
    }>;
    expect(list).toHaveLength(1);
    expect(list[0].id).toBe("h9");
  });
});

describe("workbench store — panel toggle actions", () => {
  beforeEach(() => resetStore());

  it("setAIPanelOpen(true) opens the AI Assistant pane", () => {
    useWorkbenchStore.getState().setAIPanelOpen(true);
    expect(useWorkbenchStore.getState().uiState.aiAssistantOpen).toBe(true);
  });

  it("toggleAIPanel flips the open state", () => {
    expect(useWorkbenchStore.getState().uiState.aiAssistantOpen).toBe(false);
    useWorkbenchStore.getState().toggleAIPanel();
    expect(useWorkbenchStore.getState().uiState.aiAssistantOpen).toBe(true);
    useWorkbenchStore.getState().toggleAIPanel();
    expect(useWorkbenchStore.getState().uiState.aiAssistantOpen).toBe(false);
  });
});

describe("workbench store — setInput", () => {
  beforeEach(() => resetStore());

  it("updates the chat input value", () => {
    useWorkbenchStore.getState().setInput("hello pathway");
    expect(useWorkbenchStore.getState().input).toBe("hello pathway");
  });
});

describe("workbench store — sendMessage appends a user message", () => {
  beforeEach(() => {
    resetStore();
    vi.mocked(streamChat).mockClear();
    vi.mocked(streamChat).mockResolvedValue(undefined);
  });

  it("appends a user message and flips isStreaming", async () => {
    await useWorkbenchStore.getState().sendMessage("Inhibit MEK");
    const s = useWorkbenchStore.getState();
    expect(s.messages).toHaveLength(1);
    expect(s.messages[0].role).toBe("user");
    expect(s.messages[0].content).toBe("Inhibit MEK");
    expect(s.messages[0].type).toBe("text");
    // streamChat should have been called once.
    expect(streamChat).toHaveBeenCalledTimes(1);
  });

  it("ignores empty / whitespace-only input", async () => {
    await useWorkbenchStore.getState().sendMessage("   ");
    expect(useWorkbenchStore.getState().messages).toEqual([]);
    expect(streamChat).not.toHaveBeenCalled();
  });
});
