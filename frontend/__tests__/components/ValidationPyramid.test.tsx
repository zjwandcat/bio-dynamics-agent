/**
 * Unit tests for the ValidationPyramid component.
 *
 * ValidationPyramid is driven by its `report` prop (no store dependency).
 * Covers:
 *   - Renders all 5 level cards
 *   - Status badges reflect the report (skipped on empty report)
 *   - Empty-state report (all levels skipped)
 *   - Expanding a level reveals its checks
 */
import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import {
  ValidationPyramid,
  type ValidationReport,
} from "@/components/validation/ValidationPyramid";

const LEVEL_NAMES = [
  "Internal Consistency",
  "SBML / BioModels",
  "Cross-Pathway",
  "Benchmark",
  "Hypothesis",
];

describe("ValidationPyramid — 5 levels render", () => {
  it("renders all 5 level cards", () => {
    render(<ValidationPyramid report={{}} />);
    for (const name of LEVEL_NAMES) {
      expect(screen.getByText(name)).toBeInTheDocument();
    }
  });

  it("renders the L1–L5 level number markers", () => {
    render(<ValidationPyramid report={{}} />);
    expect(screen.getByText("L1")).toBeInTheDocument();
    expect(screen.getByText("L2")).toBeInTheDocument();
    expect(screen.getByText("L3")).toBeInTheDocument();
    expect(screen.getByText("L4")).toBeInTheDocument();
    expect(screen.getByText("L5")).toBeInTheDocument();
  });
});

describe("ValidationPyramid — empty state (all skipped)", () => {
  it("shows Skip badges for every level when the report is empty", () => {
    render(<ValidationPyramid report={{}} />);
    // 5 level badges + the overall header badge ("Failed")
    const skipBadges = screen.getAllByText("Skip");
    expect(skipBadges).toHaveLength(5);
  });

  it("shows 0/5 levels passed in the score header", () => {
    render(<ValidationPyramid report={{}} />);
    expect(screen.getByText(/0\/5 levels passed/i)).toBeInTheDocument();
  });

  it('marks the overall header as "Failed" when overall_pass is not true', () => {
    render(<ValidationPyramid report={{}} />);
    expect(screen.getByText("Failed")).toBeInTheDocument();
  });
});

describe("ValidationPyramid — status badges with a populated report", () => {
  it("shows Pass badges for passing levels", () => {
    const report: ValidationReport = {
      overall_pass: true,
      level1: {
        pass: true,
        mass_conservation_error: 0.01,
        steady_state_check: true,
        numerical_stability: true,
      },
      level2: { pass: true, track: "A" },
      level3: { pass: true, crosstalk_consistency: true, shared_species_conservation: 0.05, time_scale_alignment: true },
      level4: { pass: true, benchmarks: [] },
      level5: { pass: true, hypotheses_validated: 2, hypotheses_falsified: 0 },
    };
    render(<ValidationPyramid report={report} />);
    // Overall header should say "All Pass"
    expect(screen.getByText("All Pass")).toBeInTheDocument();
    // At least 5 Pass badges (one per level)
    const passBadges = screen.getAllByText("Pass");
    expect(passBadges.length).toBeGreaterThanOrEqual(5);
  });

  it("shows Fail badges for failing levels", () => {
    const report: ValidationReport = {
      overall_pass: false,
      level1: { pass: false, error: "boom" },
    };
    render(<ValidationPyramid report={report} />);
    expect(screen.getByText("Fail")).toBeInTheDocument();
  });
});

describe("ValidationPyramid — expandable detail", () => {
  it("expands a level to reveal its checks on click", async () => {
    const report: ValidationReport = {
      overall_pass: false,
      level1: {
        pass: true,
        mass_conservation_error: 0.01,
        steady_state_check: true,
        numerical_stability: true,
      },
    };
    render(<ValidationPyramid report={report} />);
    // Level 1 header button contains "Internal Consistency"
    const level1Button = screen.getByText("Internal Consistency").closest("button")!;
    await userEvent.click(level1Button);
    // Level 1 builds a "Mass Conservation" check
    expect(await screen.findByText("Mass Conservation")).toBeInTheDocument();
  });
});
