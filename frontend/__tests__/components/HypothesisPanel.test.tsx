/**
 * Unit tests for the HypothesisPanel component (Task C.8).
 *
 * HypothesisPanel reads `hypothesisList` and `isStreaming` from the workbench
 * store and renders one HypothesisCard per hypothesis. The store is mocked so
 * each test can control the list contents. ExperimentCard is mocked to keep
 * the test focused on the panel/card contract (5 accordion sections).
 */
import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

// --- Mutable mock store state -----------------------------------------

let mockStoreState: {
  hypothesisList: unknown[];
  isStreaming: boolean;
} = { hypothesisList: [], isStreaming: false };

vi.mock("@/lib/store", () => ({
  useWorkbenchStore: (selector: (s: typeof mockStoreState) => unknown) =>
    selector(mockStoreState),
}));

// Mock ExperimentCard to avoid pulling in deeper dependency chains; the 5
// section titles live in HypothesisCard, not ExperimentCard.
vi.mock("@/components/hypothesis/ExperimentCard", () => ({
  ExperimentCard: () => <div data-testid="experiment-card-mock" />,
}));

import { HypothesisPanel } from "@/components/hypothesis/HypothesisPanel";
import type { Hypothesis } from "@/components/hypothesis/types";

// -----------------------------------------------------------------------

beforeEach(() => {
  mockStoreState = { hypothesisList: [], isStreaming: false };
});

describe("HypothesisPanel — empty state", () => {
  it("renders the panel header", () => {
    render(<HypothesisPanel />);
    expect(screen.getByText("Hypothesis")).toBeInTheDocument();
  });

  it('shows the "Run a simulation to generate hypotheses" empty state', () => {
    render(<HypothesisPanel />);
    expect(
      screen.getByText(/Run a simulation to generate hypotheses/i)
    ).toBeInTheDocument();
  });

  it("shows a pending status badge when not streaming and list is empty", () => {
    render(<HypothesisPanel />);
    expect(screen.getByText("pending")).toBeInTheDocument();
  });
});

describe("HypothesisPanel — populated state", () => {
  it("renders one card per hypothesis and the item count", () => {
    const hypotheses: Hypothesis[] = [
      {
        id: "h1",
        statement: "MEK inhibition reduces ERK phosphorylation by 50%.",
        strategy: "sensitivity",
      },
      {
        id: "h2",
        statement: "NF-κB exhibits damped oscillation under TNF stimulus.",
        strategy: "oscillation",
      },
    ];
    mockStoreState = { hypothesisList: hypotheses, isStreaming: false };

    render(<HypothesisPanel />);

    // The first card is expanded by default (defaultExpanded={idx === 0}),
    // so its statement appears in both the summary and the expanded section.
    // Use getAllByText and assert at least one match for each statement.
    expect(
      screen.getAllByText("MEK inhibition reduces ERK phosphorylation by 50%.")
        .length
    ).toBeGreaterThanOrEqual(1);
    expect(
      screen.getAllByText("NF-κB exhibits damped oscillation under TNF stimulus.")
        .length
    ).toBeGreaterThanOrEqual(1);
    // Item count ("2 items")
    expect(screen.getByText(/2 items/i)).toBeInTheDocument();
    // Status flips to "generated"
    expect(screen.getByText("generated")).toBeInTheDocument();
  });

  it("renders all 5 accordion sections when a card is expanded", async () => {
    const hypotheses: Hypothesis[] = [
      {
        id: "h1",
        statement: "p53 pulses under DNA damage.",
        strategy: "oscillation",
        prediction: "p53 amplitude decreases >30% with MDM2 overexpression.",
        experiment_design: {
          perturbation: { type: "overexpression", target: "MDM2" },
          readout: { species: "p53", metric: "amplitude", threshold: 0.3 },
          controls: ["vehicle"],
          time_points: [0, 30, 60],
          cell_line: "MCF7",
          expected_result: "Reduced p53 pulse amplitude",
        },
        supporting_pmids: ["12345678"],
      },
    ];
    mockStoreState = { hypothesisList: hypotheses, isStreaming: false };

    render(<HypothesisPanel />);

    // The first card is expanded by default (defaultExpanded={idx === 0}),
    // so the 5 accordion sections are already visible. "Hypothesis" appears
    // both as the panel header <h3> and as a section title — use getAllByText.
    expect(screen.getAllByText("Hypothesis").length).toBeGreaterThanOrEqual(2);
    expect(screen.getByText("Evidence")).toBeInTheDocument();
    expect(screen.getByText("Predictions")).toBeInTheDocument();
    expect(screen.getByText("Suggested Experiments")).toBeInTheDocument();
    expect(screen.getByText("Falsifiability")).toBeInTheDocument();
  });
});

describe("HypothesisPanel — streaming state", () => {
  it("shows the generating spinner text while streaming with an empty list", () => {
    mockStoreState = { hypothesisList: [], isStreaming: true };
    render(<HypothesisPanel />);
    expect(screen.getByText(/Generating hypotheses/i)).toBeInTheDocument();
  });
});
