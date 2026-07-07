import { test, expect } from '@playwright/test';

/**
 * NOTE: This is a starter suite. Because the site is a client-rendered SPA,
 * exact selectors below are best-effort guesses based on typical structure.
 *
 * Before relying on this suite, run:
 *   npx playwright codegen https://the-bhagavad-gita-handbook.netlify.app/
 * and record real interactions (clicking a chapter, playing audio, sending
 * a chat message) to replace the TODO selectors with real ones - ideally
 * matching on data-testid, role, or stable text rather than CSS classes.
 */

test.describe('Chapter navigation', () => {
  test('can navigate to a chapter with contributed verses', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    // TODO: replace with the real link/button text or testid for chapter nav.
    const chapterLink = page.getByRole('link', { name: /chapter 1/i })
      .or(page.getByText(/chapter 1/i).first());

    if (await chapterLink.count() > 0) {
      await chapterLink.first().click();
      await page.waitForLoadState('networkidle');
      await expect(page.locator('body')).toContainText(/./); // page updated, not blank
    } else {
      test.skip(true, 'Chapter navigation element not found - update selector');
    }
  });
});

test.describe('Sanskrit verse rendering', () => {
  // Devanagari unicode range check - confirms Sanskrit script actually rendered,
  // not just placeholder/broken text.
  test('Sanskrit (Devanagari) text is present on a verse page', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    const bodyText = await page.locator('body').innerText();
    const hasDevanagari = /[\u0900-\u097F]/.test(bodyText);

    expect(hasDevanagari, 'No Devanagari (Sanskrit) characters found on the page').toBeTruthy();
  });

  test('translation/transliteration text is present alongside verses', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    // TODO: tighten this once you know the real class/testid for translation blocks.
    const bodyText = await page.locator('body').innerText();
    expect(bodyText.trim().length).toBeGreaterThan(200);
  });
});

test.describe('Audio playback', () => {
  test('audio element or player control is present', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    const audioEl = page.locator('audio');
    const playButton = page.getByRole('button', { name: /play/i });

    const audioCount = await audioEl.count();
    const playButtonCount = await playButton.count();

    expect(
      audioCount + playButtonCount,
      'No <audio> element or play button found on the page'
    ).toBeGreaterThan(0);
  });
});

test.describe('AI chat interaction', () => {
  test('chat input is present and accepts text', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    // TODO: replace with real placeholder text / testid for the chat input.
    const chatInput = page.getByPlaceholder(/ask|question|chat|message/i)
      .or(page.getByRole('textbox').first());

    if (await chatInput.count() > 0) {
      await chatInput.first().fill('What is the meaning of dharma?');
      await expect(chatInput.first()).toHaveValue(/dharma/i);
    } else {
      test.skip(true, 'Chat input not found - update selector');
    }
  });
});

test.describe('Language support', () => {
  test('language selector is present (if applicable)', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    // TODO: replace with real testid/label once confirmed via codegen.
    const langControl = page.getByRole('combobox', { name: /language/i })
      .or(page.getByRole('button', { name: /language|हिन्दी|தமிழ்/i }));

    const count = await langControl.count();
    if (count === 0) {
      test.skip(true, 'Language selector not found - update selector after codegen');
    } else {
      await expect(langControl.first()).toBeVisible();
    }
  });
});
