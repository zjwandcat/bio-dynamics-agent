import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import path from "node:path";

/**
 * Vitest configuration for the BioDynamics v4 frontend.
 *
 * - jsdom environment so @testing-library/react can mount components.
 * - The `@/` path alias mirrors `tsconfig.json` (`"@/*": ["./*"]`).
 * - Setup file registers @testing-library/jest-dom matchers.
 */
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./"),
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./vitest.setup.ts"],
    include: ["__tests__/**/*.test.{ts,tsx}"],
    css: false,
  },
});
