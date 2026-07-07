import { test, expect } from '@playwright/test';

test.describe('Site health', () => {
  test('homepage loads successfully', async ({ page }) => {
    const response = await page.goto('/');
    expect(response?.status()).toBeLessThan(400);
  });

  test('page has expected title', async ({ page }) => {
    await page.goto('/');
    await expect(page).toHaveTitle(/Bhagavad Gita Handbook/i);
  });

  test('no uncaught JS errors on load', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(err.message));

    await page.goto('/');
    await page.waitForLoadState('networkidle');

    expect(errors, `Uncaught JS errors: ${errors.join('; ')}`).toEqual([]);
  });

  test('no failed network requests on load', async ({ page }) => {
    const failed: string[] = [];
    page.on('requestfailed', (req) => {
      failed.push(`${req.method()} ${req.url()} - ${req.failure()?.errorText}`);
    });

    await page.goto('/');
    await page.waitForLoadState('networkidle');

    expect(failed, `Failed requests: ${failed.join('; ')}`).toEqual([]);
  });

  test('page renders visible content (not a blank shell)', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    const bodyText = await page.locator('body').innerText();
    expect(bodyText.trim().length).toBeGreaterThan(50);
  });
});
