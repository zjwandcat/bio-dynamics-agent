import { defineConfig, devices } from "@playwright/test";

/**
 * Playwright configuration for the BioDynamics v4 frontend E2E smoke tests.
 *
 * Auto-starts the Next.js dev server on port 3000 so `npx playwright test`
 * works without a manually running server. If the dev server fails to start
 * (e.g. missing env / port in use), the E2E suite will fail fast — the
 * individual smoke specs are written to skip gracefully if the home page is
 * unreachable.
 */
export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  workers: 1,
  reporter: [["list"]],
  timeout: 30_000,
  expect: { timeout: 10_000 },
  use: {
    baseURL: "http://localhost:3000",
    trace: "on-first-retry",
    actionTimeout: 10_000,
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
  webServer: {
    command: "npm run dev",
    url: "http://localhost:3000",
    timeout: 120_000,
    reuseExistingServer: !process.env.CI,
    cwd: __dirname,
  },
});
