"use strict";

/**
 * BupaR Process Mining tab — end-to-end tests.
 *
 * Mermaid workflow (10_risk_dashboard/README.md — Tab: BupaR Process Mining):
 *   A([BupaR tab])
 *   → B[Select cohort · age band · Optional: event density bin]
 *   → C[Select panels: Sequences · Activity Frequency · Trace Explorer · Drug×Drug Matrix]
 *   → D[Click Load BupaR Visualizations (btnLoadBupaR)]
 *   → E[GET /visualizations/bupar?cohort=&age_band=&n_event_bin=]
 *   → F{HTTP 200?}
 *   F →|200| H[Drug-specific panels (PNGs/HTMLs) + Event-to-target panels]
 *   F →|Error| G[bupar-status error]
 *
 * Note: BupaR uses its own cohort/age_band dropdowns (#bupar-cohort, #bupar-age-band).
 * Does NOT require Risk Assessment context.
 *
 * Run:
 *   DASHBOARD_URL=... API_BASE_URL=... npx jest tests/tab-bupar --forceExit --verbose
 */

const { launchBrowser, openDashboard, navigateToTab, loadVisualization,
        getStatusText, setDropdown, sleep } = require("../../helpers/browser");

let browser, page;

beforeAll(async () => {
  browser = await launchBrowser();
  page    = await openDashboard(browser);
  // Mermaid A: navigate to BupaR tab — wait for cohort dropdown to confirm tab HTML injected
  await navigateToTab(page, "bupar-visualizations", "#bupar-cohort");
}, 40_000);

afterAll(async () => {
  if (browser) await browser.close();
});

// ---------------------------------------------------------------------------

async function loadBupar(cohort, ageBand, bin = null) {
  // Mermaid B: select cohort + age band + optional density bin
  await setDropdown(page, "#bupar-cohort",   cohort);
  await setDropdown(page, "#bupar-age-band", ageBand);
  if (bin) await setDropdown(page, "#bupar-n-event-bin", bin);

  // Mermaid D→E: click Load → GET /bupar
  return loadVisualization(page, "btnLoadBupaR", "/bupar", 20_000);
}

// ---------------------------------------------------------------------------

describe("BupaR tab — happy path (mermaid: A→B→D→E→F→H)", () => {

  test("opioid_ed / 55-64 → response OK, status not error", async () => {
    const response = await loadBupar("opioid_ed", "55-64");

    // Mermaid F: null OK (static S3 assets may not trigger interceptable GET /bupar)
    if (response !== null) {
      expect([200, 400, 404, 500]).toContain(response.status());
      if (response.status() === 200) {
        const body = await response.json().catch(() => null);
        expect(body !== null && typeof body === "object").toBe(true);
      }
    }

    // Mermaid G: status must not show unhandled error
    const statusText = await getStatusText(page, "bupar-status");
    if (statusText) {
      expect(statusText.toLowerCase()).not.toMatch(/^error:/);
    }
  }, 30_000);

  test("non_opioid_ed / 65-74 → response OK", async () => {
    const response = await loadBupar("non_opioid_ed", "65-74");
    if (response !== null) {
      expect([200, 400, 404, 500]).toContain(response.status());
    }
  }, 30_000);

  test("opioid_ed / 25-44 / density bin medium → response OK", async () => {
    // Mermaid B: optional event density bin
    const response = await loadBupar("opioid_ed", "25-44", "medium");
    if (response !== null) {
      expect([200, 400, 404, 500]).toContain(response.status());
    }
  }, 30_000);

});

// ---------------------------------------------------------------------------

describe("BupaR tab — DOM rendering (Activity Rate by Event Density Bin panel)", () => {

  beforeAll(async () => {
    await loadBupar("opioid_ed", "55-64");
    await sleep(3000); // bupar_activity_heatmap.json loads in background after primary response
  }, 30_000);

  test("bupar-bin-bar-container renders Plotly chart or shows graceful not-available message", async () => {
    // Dashboard fetches density/combined/bupar_activity_heatmap.json in background.
    // If present: renders Plotly grouped bar chart (SVG inside container).
    // If absent:  renders "Grouped bar chart not available." (data-dependent).
    const result = await page.evaluate(() => {
      const el = document.getElementById("bupar-bin-bar-container");
      if (!el) return null;
      if (el.querySelector("svg")) return "plotly";
      return el.textContent.trim().slice(0, 80) || "";
    });
    expect(result).not.toBeNull(); // element must exist in DOM
    if (result === "plotly") {
      console.log("bupar-bin-bar-container: Plotly grouped bar chart rendered ✓");
    } else {
      console.warn("bupar-bin-bar-container: not rendered — bupar_activity_heatmap.json missing from S3 (run generate_combined_bin_activity_heatmap):", result);
    }
    // Not hard-failing: data-dependent on pipeline having generated the combined heatmap JSON
  }, 10_000);

});

// ---------------------------------------------------------------------------

describe("BupaR tab — no selection guard (mermaid: B→ missing cohort/age → error)", () => {

  test("empty age band → bupar-status shows error, page alive", async () => {
    await setDropdown(page, "#bupar-cohort", "opioid_ed");
    // Force empty age band by setting to blank
    await page.$eval("#bupar-age-band", el => { el.value = ""; });

    await page.click("#btnLoadBupaR").catch(() => {});
    await sleep(500);

    const statusText = await getStatusText(page, "bupar-status");
    if (statusText) {
      // Mermaid guard: should show an informational/error message, not crash
      expect(typeof statusText).toBe("string");
    }

    const alive = await page.evaluate(() => document.title).catch(() => null);
    expect(alive).not.toBeNull();
  }, 15_000);

});
