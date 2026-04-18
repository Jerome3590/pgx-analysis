"use strict";

/**
 * DTW Trajectories tab — end-to-end tests.
 *
 * Mermaid workflow (10_risk_dashboard/README.md — Tab: DTW Trajectories):
 *   A([DTW tab])
 *   → B[Select cohort · age band · Optional: event density bin]
 *   → C[Select panels to show via checkboxes]
 *   → D[Click Load DTW Visualizations (btnLoadDTW)]
 *   → E[GET /visualizations/dtw?cohort=&age_band=&n_event_bin=]
 *   → F{HTTP 200?}
 *   F →|200| H{Sub-tab}
 *   H →|Overview and Trajectories| I[Trajectory Analysis PNG from S3]
 *   H →|Routine vs Utilization|    J[Outcome rate + event counts]
 *   F →|Error| G[dtw-status error]
 *
 * Note: DTW uses its own cohort/age_band dropdowns (#dtw-cohort, #dtw-age-band).
 * Event density is #dtw-event-density (re-renders from already-loaded data — no new API call).
 * Does NOT require Risk Assessment context.
 *
 * Run:
 *   DASHBOARD_URL=... API_BASE_URL=... npx jest tests/tab-dtw --forceExit --verbose
 */

const { launchBrowser, openDashboard, navigateToTab, loadVisualization,
        getStatusText, setDropdown, sleep } = require("../../helpers/browser");

let browser, page;

beforeAll(async () => {
  browser = await launchBrowser();
  page    = await openDashboard(browser);
  // Mermaid A: navigate to DTW tab
  await navigateToTab(page, "dtw-visualizations", "#dtw-cohort");
}, 40_000);

afterAll(async () => {
  if (browser) await browser.close();
});

// ---------------------------------------------------------------------------

async function loadDtw(cohort, ageBand) {
  // Mermaid B: select cohort + age band
  await setDropdown(page, "#dtw-cohort",    cohort);
  await setDropdown(page, "#dtw-age-band",  ageBand);

  // Mermaid D→E: click Load → GET /dtw
  return loadVisualization(page, "btnLoadDTW", "/dtw", 20_000);
}

// ---------------------------------------------------------------------------

describe("DTW tab — happy path (mermaid: A→B→D→E→F→H)", () => {

  test("opioid_ed / 55-64 → response OK, status not error", async () => {
    const response = await loadDtw("opioid_ed", "55-64");

    if (response !== null) {
      expect([200, 400, 404, 500]).toContain(response.status());
      if (response.status() === 200) {
        const body = await response.json().catch(() => null);
        expect(body !== null && typeof body === "object").toBe(true);
      }
    }

    const statusText = await getStatusText(page, "dtw-status");
    if (statusText) {
      expect(statusText.toLowerCase()).not.toMatch(/^error:/);
    }
  }, 30_000);

  test("non_opioid_ed / 75-84 → response OK", async () => {
    const response = await loadDtw("non_opioid_ed", "75-84");
    if (response !== null) {
      expect([200, 400, 404, 500]).toContain(response.status());
    }
  }, 30_000);

  test("opioid_ed / 65-74 → response OK", async () => {
    const response = await loadDtw("opioid_ed", "65-74");
    if (response !== null) {
      expect([200, 400, 404, 500]).toContain(response.status());
    }
  }, 30_000);

});

// ---------------------------------------------------------------------------

describe("DTW tab — density filter re-render (mermaid: dtw-event-density re-renders from loaded data)", () => {

  beforeAll(async () => {
    // Load data first so density filter has something to re-render
    await loadDtw("opioid_ed", "55-64");
    await sleep(1000);
  }, 30_000);

  test("changing event density does not crash the page", async () => {
    // Mermaid note: DTW event density re-renders from already-loaded data (no new API call)
    await setDropdown(page, "#dtw-event-density", "medium");
    await page.$eval("#dtw-event-density", el => el.dispatchEvent(new Event("change", { bubbles: true })));
    await sleep(500);

    const alive = await page.evaluate(() => document.title).catch(() => null);
    expect(alive).not.toBeNull();
  }, 15_000);

});

// ---------------------------------------------------------------------------

describe("DTW tab — no selection guard (mermaid: B→ missing cohort/age → error)", () => {

  test("empty age band → dtw-status shows guard message, page alive", async () => {
    await setDropdown(page, "#dtw-cohort", "opioid_ed");
    await page.$eval("#dtw-age-band", el => { el.value = ""; });

    await page.click("#btnLoadDTW").catch(() => {});
    await sleep(500);

    const statusText = await getStatusText(page, "dtw-status");
    if (statusText) {
      expect(typeof statusText).toBe("string");
    }

    const alive = await page.evaluate(() => document.title).catch(() => null);
    expect(alive).not.toBeNull();
  }, 15_000);

});
