# Daily Site Checks - Bhagavad Gita Handbook

Automated Playwright tests that run daily against the live site:
https://the-bhagavad-gita-handbook.netlify.app/

## What's included

- `playwright.config.ts` - test config, points at the live Netlify URL
- `tests/site-health.spec.ts` - basic uptime/health checks (page loads, no JS errors, no failed requests, content renders)
- `tests/functionality.spec.ts` - functional checks (Sanskrit verse rendering, audio player, chat input, chapter navigation, language selector)
- `.github/workflows/daily-site-check.yml` - runs the suite daily at 6 AM UTC, plus on-demand via "Run workflow"

## Setup (if merging into an existing repo)

If this repo already has a `package.json`, don't overwrite it - instead:

1. Add `@playwright/test` to `devDependencies`:
   ```
   npm install -D @playwright/test
   ```
2. Merge the `scripts` entries from this `package.json` into the existing one.
3. Drop the `playwright.config.ts` and `tests/` folder in as-is.
4. Add the workflow file under `.github/workflows/`.

## Before merging the PR

The `functionality.spec.ts` file has several `TODO` selectors that are best-effort
guesses, since the site is a client-rendered SPA and I couldn't inspect the live DOM
directly. Run this locally first to record real selectors:

```bash
npm install
npx playwright install --with-deps
npx playwright codegen https://the-bhagavad-gita-handbook.netlify.app/
```

Click through: a chapter link, the audio play button, the chat input, and (if present)
the language selector. Codegen will print real Playwright locators you can paste in place
of the TODOs.

Then run the suite locally to confirm it passes cleanly:

```bash
npm test
```

## Notifications

- **Email (default, already active):** GitHub automatically emails whoever triggered
  the workflow (or repo watchers, depending on notification settings) when a scheduled
  run fails. No extra setup needed.
- **Discord (add later):** instructions are commented directly in
  `.github/workflows/daily-site-check.yml` - just add a `DISCORD_WEBHOOK_URL` repo
  secret and uncomment the notify step.

## Adjusting the schedule

The cron `0 6 * * *` runs daily at 6:00 AM UTC. Cron in GitHub Actions is always UTC -
adjust the hour if you want it to land at a specific local time.
