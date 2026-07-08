import { test, expect } from '@playwright/test';

test.describe('Scientific Workflow: Hypothesis → Simulation → Validation → Report', () => {
  
  test('Complete hypothesis-to-report workflow', async ({ page }) => {
    // Step 1: Navigate to workspace
    await page.goto('/workspace');
    await expect(page.locator('[data-testid="workbench-shell"]')).toBeVisible();
    
    // Step 2: Enter hypothesis
    const inputArea = page.locator('[data-testid="input-area"]');
    await inputArea.fill('EGF stimulation activates EGFR-MAPK signaling cascade');
    
    // Step 3: Submit
    await page.click('[data-testid="run-button"]');
    
    // Step 4: Wait for pathway graph
    await expect(page.locator('[data-testid="pathway-graph"]')).toBeVisible({ timeout: 30000 });
    
    // Step 5: Check simulation panel renders
    await expect(page.locator('[data-testid="simulation-panel"]')).toBeVisible({ timeout: 30000 });
    
    // Step 6: Check validation pyramid
    const pyramid = page.locator('[data-testid="validation-pyramid"]');
    await expect(pyramid).toBeVisible({ timeout: 30000 });
    
    // Step 7: Verify at least one level shows result
    const levels = page.locator('[data-testid^="validation-level-"]');
    await expect(levels.first()).toBeVisible();
    
    // Step 8: Check hypothesis panel
    await expect(page.locator('[data-testid="hypothesis-panel"]')).toBeVisible();
    
    // Step 9: Navigate to report
    await page.click('[data-testid="view-report"]');
    await expect(page).toHaveURL(/\/report\//);
    
    // Step 10: Verify report content
    await expect(page.locator('h1')).toContainText(/experiment.*report/i);
  });

  test('Parameter exploration workflow', async ({ page }) => {
    await page.goto('/workspace');
    
    // Wait for parameter explorer
    const paramExplorer = page.locator('[data-testid="parameter-explorer"]');
    await expect(paramExplorer).toBeVisible({ timeout: 30000 });
    
    // Modify a parameter
    const paramInput = paramExplorer.locator('input[type="number"]').first();
    await paramInput.fill('0.5');
    
    // Run simulation
    await page.click('[data-testid="run-button"]');
    
    // Verify simulation updates
    await expect(page.locator('[data-testid="simulation-panel"]')).toBeVisible({ timeout: 30000 });
  });

  test('Benchmark center workflow', async ({ page }) => {
    await page.goto('/benchmarks');
    await expect(page.locator('h1')).toContainText(/benchmark/i);
    
    // Run benchmark
    const runButton = page.locator('[data-testid="run-benchmark"]');
    if (await runButton.isVisible()) {
      await runButton.click();
      
      // Wait for results
      await expect(page.locator('[data-testid="benchmark-results"]')).toBeVisible({ timeout: 60000 });
    }
  });

  test('Screenshot: full workspace state', async ({ page }) => {
    await page.goto('/workspace');
    await page.waitForTimeout(2000);
    await page.screenshot({ path: 'reports/workspace-full.png', fullPage: true });
  });
});

test.describe('Error handling', () => {
  test('Handles simulation failure gracefully', async ({ page }) => {
    await page.goto('/workspace');
    
    // Enter invalid input
    await page.locator('[data-testid="input-area"]').fill('invalid pathway xyz123');
    await page.click('[data-testid="run-button"]');
    
    // Should show error, not crash
    const errorNotification = page.locator('[data-testid="error-notification"]');
    await expect(errorNotification).toBeVisible({ timeout: 10000 });
  });
});
