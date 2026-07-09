import { test, expect, type Page, type APIRequestContext } from '@playwright/test';

/**
 * Validates that external media links on each chapter's /listen page actually
 * resolve (not broken/404), covering:
 *  - MP3 audio sources (<audio><source src="...">)
 *  - YouTube links/embeds
 *  - SoundCloud links/embeds
 *
 * Uses HTTP HEAD requests (falling back to GET if HEAD isn't supported) rather
 * than loading full media files, to keep this fast and bandwidth-light.
 */

const TOTAL_CHAPTERS = 18;

// Some hosts don't support HEAD or block automated requests outright (returning
// 403 for anything without a real browser session). We only treat genuine
// "not found" style failures as test failures.
async function checkLinkIsReachable(
  request: APIRequestContext,
  url: string
): Promise<{ ok: boolean; status: number | null; error?: string }> {
  try {
    let response = await request.head(url, { failOnStatusCode: false, timeout: 15_000 });
    if (response.status() === 405) {
      // HEAD not allowed - fall back to GET
      response = await request.get(url, { failOnStatusCode: false, timeout: 15_000 });
    }
    return { ok: response.status() < 400, status: response.status() };
  } catch (err) {
    return { ok: false, status: null, error: (err as Error).message };
  }
}

async function collectMediaLinks(page: Page) {
  const mp3Links = await page.locator('audio source[src], audio[src]').evaluateAll((elements) =>
    elements.map((el) => el.getAttribute('src')).filter((src): src is string => !!src)
  );

  const youtubeLinks = await page
    .locator(
      'a[href*="youtube.com"], a[href*="youtu.be"], iframe[src*="youtube.com"], iframe[src*="youtu.be"]'
    )
    .evaluateAll((elements) =>
      elements
        .map((el) => el.getAttribute('href') || el.getAttribute('src'))
        .filter((url): url is string => !!url)
    );

  const soundcloudLinks = await page
    .locator('a[href*="soundcloud.com"], iframe[src*="soundcloud.com"]')
    .evaluateAll((elements) =>
      elements
        .map((el) => el.getAttribute('href') || el.getAttribute('src'))
        .filter((url): url is string => !!url)
    );

  // Resolve relative MP3 paths (e.g. "/Chapter1-Arjuna-Visada-Yoga.mp3") against the page origin
  const pageOrigin = new URL(page.url()).origin;
  const resolvedMp3Links = mp3Links.map((src) =>
    src.startsWith('http') ? src : new URL(src, pageOrigin).toString()
  );

  return { mp3Links: resolvedMp3Links, youtubeLinks, soundcloudLinks };
}

test.describe('Media link validation - /listen pages', () => {
  for (let chapter = 1; chapter <= TOTAL_CHAPTERS; chapter++) {
    test(`chapter ${chapter}: all MP3/YouTube/SoundCloud links are reachable`, async ({
      page,
      request,
    }) => {
      const response = await page.goto(`/chapter/${chapter}/listen`);
      expect(response?.status(), `Chapter ${chapter} listen page failed to load`).toBeLessThan(400);
      await page.waitForLoadState('networkidle');

      const { mp3Links, youtubeLinks, soundcloudLinks } = await collectMediaLinks(page);

      // At minimum, the primary MP3 source should exist (confirmed via earlier
      // inspection - every chapter has a default local MP3 audio source).
      expect(mp3Links.length, `Chapter ${chapter}: no MP3 source found on listen page`).toBeGreaterThan(0);

      const allLinks = [
        ...mp3Links.map((url) => ({ url, type: 'MP3' })),
        ...youtubeLinks.map((url) => ({ url, type: 'YouTube' })),
        ...soundcloudLinks.map((url) => ({ url, type: 'SoundCloud' })),
      ];

      for (const { url, type } of allLinks) {
        const result = await checkLinkIsReachable(request, url);
        expect(
          result.ok,
          `Chapter ${chapter} [${type}] link unreachable: ${url} (status: ${result.status ?? 'error'}${
            result.error ? `, ${result.error}` : ''
          })`
        ).toBeTruthy();
      }
    });
  }
});
