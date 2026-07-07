/**
 * Unit tests for the BenchmarkCard component.
 *
 * BenchmarkCard is a pure presentational component driven entirely by its
 * props (`def`, `state`, `suiteRunning`, `onRun`) — it does not read the
 * Zustand store, so no store mocking is required.
 */
import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import {
  BenchmarkCard,
  type BenchmarkDef,
  type BenchmarkCardState,
} from "@/components/benchmark/BenchmarkCard";

const def: BenchmarkDef = {
  pathway_class: "EGFR_RTK",
  name: "EGFR RTK Signaling",
  description: "EGF stimulation leads to EGFR phosphorylation.",
};

const idleState: BenchmarkCardState = { status: "idle" };

describe("BenchmarkCard — idle state", () => {
  it("renders the pathway name and pathway_class", () => {
    render(
      <BenchmarkCard
        def={def}
        state={idleState}
        suiteRunning={false}
        onRun={() => {}}
      />
    );
    expect(screen.getByText("EGFR RTK Signaling")).toBeInTheDocument();
    expect(screen.getByText("EGFR_RTK")).toBeInTheDocument();
  });

  it('shows the "Not Run" status badge initially', () => {
    render(
      <BenchmarkCard
        def={def}
        state={idleState}
        suiteRunning={false}
        onRun={() => {}}
      />
    );
    expect(screen.getByText("Not Run")).toBeInTheDocument();
  });

  it('renders a "Run" button', () => {
    render(
      <BenchmarkCard
        def={def}
        state={idleState}
        suiteRunning={false}
        onRun={() => {}}
      />
    );
    expect(screen.getByRole("button", { name: /Run/i })).toBeInTheDocument();
  });

  it("calls onRun with the pathway_class when Run is clicked", async () => {
    const onRun = vi.fn();
    render(
      <BenchmarkCard
        def={def}
        state={idleState}
        suiteRunning={false}
        onRun={onRun}
      />
    );
    await userEvent.click(screen.getByRole("button", { name: /^Run$/ }));
    expect(onRun).toHaveBeenCalledTimes(1);
    expect(onRun).toHaveBeenCalledWith("EGFR_RTK");
  });
});

describe("BenchmarkCard — running / pass / fail states", () => {
  it('shows the "Running" badge and disables Run while running', () => {
    render(
      <BenchmarkCard
        def={def}
        state={{ status: "running", step: "loading_specialist" }}
        suiteRunning={false}
        onRun={() => {}}
      />
    );
    expect(screen.getByText("Running")).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /Running/i })
    ).toBeDisabled();
  });

  it('shows the "Pass" badge for a passing result', () => {
    render(
      <BenchmarkCard
        def={def}
        state={{
          status: "pass",
          result: {
            pathway_class: "EGFR_RTK",
            name: "EGFR RTK",
            status: "pass",
            checks: [],
            runtime_seconds: 1.23,
            errors: [],
          },
        }}
        suiteRunning={false}
        onRun={() => {}}
      />
    );
    expect(screen.getByText("Pass")).toBeInTheDocument();
  });

  it('shows the "Fail" badge for a failing result', () => {
    render(
      <BenchmarkCard
        def={def}
        state={{
          status: "fail",
          result: {
            pathway_class: "EGFR_RTK",
            name: "EGFR RTK",
            status: "fail",
            checks: [],
            runtime_seconds: 0.5,
            errors: [],
          },
        }}
        suiteRunning={false}
        onRun={() => {}}
      />
    );
    expect(screen.getByText("Fail")).toBeInTheDocument();
  });

  it("disables Run when a suite run is in progress", () => {
    render(
      <BenchmarkCard
        def={def}
        state={idleState}
        suiteRunning={true}
        onRun={() => {}}
      />
    );
    expect(
      screen.getByRole("button", { name: /Run/i })
    ).toBeDisabled();
  });
});
