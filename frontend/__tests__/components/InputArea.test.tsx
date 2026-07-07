/**
 * Unit tests for the InputArea component (Task C.6).
 *
 * InputArea reads three fields from the workbench store (sendMessage,
 * isStreaming, setSimulationResult) and conditionally renders the SbmlUpload /
 * BioModelsFetcher sub-components. Both are mocked here so the test focuses on
 * the tab-switching + per-mode form rendering contract.
 */
import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

// --- Mocks -------------------------------------------------------------

const mockStoreState = {
  sendMessage: vi.fn(),
  isStreaming: false,
  setSimulationResult: vi.fn(),
};

vi.mock("@/lib/store", () => ({
  useWorkbenchStore: (selector: (s: typeof mockStoreState) => unknown) =>
    selector(mockStoreState),
}));

vi.mock("@/components/input/SbmlUpload", () => ({
  SbmlUpload: () => <div data-testid="sbml-upload-mock" />,
}));
vi.mock("@/components/input/BioModelsFetcher", () => ({
  BioModelsFetcher: () => <div data-testid="biomodels-fetcher-mock" />,
}));

import { InputArea } from "@/components/input/InputArea";

// -----------------------------------------------------------------------

const TAB_LABELS = [
  "Natural Language",
  "Structured",
  "Parameters",
  "SBML Upload",
  "BioModels ID",
];

describe("InputArea — mode tabs", () => {
  it("renders all 5 mode tabs", () => {
    render(<InputArea />);
    for (const label of TAB_LABELS) {
      expect(screen.getByText(label)).toBeInTheDocument();
    }
  });
});

describe("InputArea — Natural Language mode (default)", () => {
  it("shows a textarea with the NL hypothesis placeholder", () => {
    render(<InputArea />);
    expect(
      screen.getByPlaceholderText(/Inhibition of MEK/i)
    ).toBeInTheDocument();
  });

  it('routes via the v3 /api/chat SSE stream (footer note)', () => {
    render(<InputArea />);
    expect(screen.getByText(/Routes via \/api\/chat/i)).toBeInTheDocument();
  });
});

describe("InputArea — Structured Hypothesis mode", () => {
  it("switches to the structured form when the Structured tab is clicked", async () => {
    render(<InputArea />);
    await userEvent.click(screen.getByText("Structured"));

    // Pathway select + Hypothesis textarea + Species input + Duration input
    expect(screen.getByText("Pathway")).toBeInTheDocument();
    expect(screen.getByText("Hypothesis")).toBeInTheDocument();
    expect(screen.getByText("Species of interest")).toBeInTheDocument();
    expect(screen.getByText("Duration (min)")).toBeInTheDocument();
    expect(screen.getByText("Perturbation (optional)")).toBeInTheDocument();
  });

  it("routes via the v4 simulation endpoint (footer note)", async () => {
    render(<InputArea />);
    await userEvent.click(screen.getByText("Structured"));
    expect(
      screen.getByText(/Routes via \/api\/v4\/simulation\/run/i)
    ).toBeInTheDocument();
  });

  it("shows the structured-hypothesis textarea prompt", async () => {
    render(<InputArea />);
    await userEvent.click(screen.getByText("Structured"));
    expect(
      screen.getByPlaceholderText("State the hypothesis to test...")
    ).toBeInTheDocument();
  });
});

describe("InputArea — Parameters mode", () => {
  it("renders the parameter table header when Parameters is active", async () => {
    render(<InputArea />);
    await userEvent.click(screen.getByText("Parameters"));
    expect(screen.getByText("Parameter Table")).toBeInTheDocument();
    expect(screen.getByText("Import JSON")).toBeInTheDocument();
  });
});

describe("InputArea — SBML Upload mode", () => {
  it("renders the SBML upload sub-component when active", async () => {
    render(<InputArea />);
    await userEvent.click(screen.getByText("SBML Upload"));
    expect(screen.getByText("SBML Model Upload")).toBeInTheDocument();
    expect(screen.getByTestId("sbml-upload-mock")).toBeInTheDocument();
  });
});

describe("InputArea — BioModels ID mode", () => {
  it("renders the BioModels fetcher sub-component when active", async () => {
    render(<InputArea />);
    await userEvent.click(screen.getByText("BioModels ID"));
    expect(screen.getByText("BioModels Reference")).toBeInTheDocument();
    expect(screen.getByTestId("biomodels-fetcher-mock")).toBeInTheDocument();
  });
});
