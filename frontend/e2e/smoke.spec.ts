/**
 * E2E smoke tests for the BioDynamics v4 frontend (Task C.13).
 *
 * Covers the critical user journeys against the running Next.js dev server:
 *   1. Home page loads and shows the Pathway Selector
 *   2. Clicking a pathway card navigates to /workspace
 *   3. /workspace renders the 4-pane workbench layout
 *   4. /benchmarks renders all 10 benchmark cards
 *
 * The InputArea mode-switching journey is covered by the Vitest component
 * suite (__tests__/components/InputArea.test.tsx); it is skipped here because
 * InputArea is not yet wired into any route (the workbench center pane still
 * uses PlaceholderPanel until the integration tasks land).
 *
 * Playwright auto-starts the dev server via playwright.config.ts `webServer`.
 */
import { test, expect } from "@playwright/test";

test.describe("BioDynamics v4 — E2E smoke", () => {
  test("home page loads and shows the Pathway Selector", async ({ page }) => {
    await page.goto("/");

    // Hero title
    await expect(page.getByRole("heading", { name: "BioDynamics Agent" })).toBeVisible();

    // Pathway Selector section heading
    await expect(page.getByRole("heading", { name: "Pathway Selector" })).toBeVisible();

    // At least one pathway card is present (EGFR RTK Signaling)
    await expect(page.getByText("EGFR RTK Signaling").first()).toBeVisible();
  });

  test("clicking a pathway card navigates to /workspace", async ({ page }) => {
    await page.goto("/");

    // Click the MAPK Cascade pathway card.
    const mapkCard = page.getByText("MAPK Cascade").first();
    await mapkCard.click();

    await expect(page).toHaveURL(/\/workspace/);
  });

  test("/workspace renders the 4-pane workbench layout", async ({ page }) => {
    await page.goto("/workspace");

    // Three visible pane headers (the AI Assistant pane is collapsed by default).
    await expect(page.getByText("Project / Pathway")).toBeVisible();
    await expect(page.getByText("Scientific Workspace")).toBeVisible();
    await expect(page.getByText("Validation")).toBeVisible();

    // Center pane placeholder slots.
    await expect(page.getByText("Pathway Graph")).toBeVisible();
    await expect(page.getByText("Simulation Tabs")).toBeVisible();
    await expect(page.getByText("Parameter Editor")).toBeVisible();
  });

  // InputArea mode-switching is exercised at the component level
  // (InputArea.test.tsx). InputArea is not yet mounted on any route — the
  // workbench center pane uses PlaceholderPanel until the C.2–C.8 integration
  // lands — so the E2E journey is skipped here.
  test.skip("switch input modes in InputArea", async ({ page }) => {
    await page.goto("/workspace");
    // NOTE: covered by __tests__/components/InputArea.test.tsx
  });

  test("/benchmarks renders all 10 benchmark cards", async ({ page }) => {
    await page.goto("/benchmarks");

    // Benchmark Center header
    await expect(page.getByText("Benchmark Center")).toBeVisible();

    // 10 "Not Run" status badges (one per card in the idle state).
    await expect(page.getByText("Not Run", { exact: true })).toHaveCount(10);

    // The "Run All Benchmarks" action is present.
    await expect(
      page.getByRole("button", { name: /Run All Benchmarks/i })
    ).toBeVisible();
  });
});
