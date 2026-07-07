import { defineConfig, devices } from '@playwright/test';

/**
 * Playwright config for daily health/functionality checks against the
 * live Bhagavad Gita Handbook site.
 * https://the-bhagavad-gita-handbook.netlify.app/
 */
export default defineConfig({
  testDir: './tests',
  timeout: 30_000,
  expect: { timeout: 10_000 },
  fullyParallel: true,
  retries: 2, // live-site flakiness (network, cold Netlify function starts, etc.)
  reporter: [
    ['list'],
    ['html', { open: 'never' }],
  ],
  use: {
    baseURL: 'https://the-bhagavad-gita-handbook.netlify.app/',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
});
