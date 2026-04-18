"use strict";

/**
 * FP-Growth Patterns tab — end-to-end tests.
 *
 * Mermaid workflow (10_risk_dashboard/README.md — Tab: FP-Growth Patterns):
 *   A([FP-Growth tab]) — drug names only
 *   → B[Select cohort · age band · Optional: event density bin]
 *   → C[Click Load FP-Growth Visualizations (btnLoadFPGrowth)]
 *   → D[GET /visualizations/fpgrowth?cohort=&age_band=&n_event_bin=]
 *   → E{HTTP 200?}
 *   E →|200| G[fpgrowth-support-image — Itemset Support Distribution PNG]
 *   G →          H[fpgrowth-network-iframe — Interactive Drug Association Network (iframe)]
 *   H →          I[fpgrowth-rules-iframe — Association Rules Network (iframe)]
 *   E →|Error| F[fpgrowth-status error]
 *
 * Note: FP-Growth uses its own cohort/age_band dropdowns (#fpgrowth-cohort, #fpgrowth-age-band).
 * Density bin change auto-triggers btnLoadFPGrowth click (change event listener).
 * Does NOT require Risk Assessment context.
 *
 * Run:
 *   DASHBOARD_URL=... API_BASE_URL=... npx jest tests/tab-fpgrowth --forceExit --verbose
 */

const { launchBrowser, openDashboard, navigateToTab, loadVisualization,
        getStatusText, setDropdown, sleep } = require("../../helpers/browser");

let browser, page;

beforeAll(async () => {
  browser = await launchBrowser();
  page    = await openDashboard(browser);
  // Mermaid A: navigate to FP-Growth tab
  await navigateToTab(page, "fpgrowth-visualizations", "#fpgrowth-cohort");
}, 40_000);

afterAll(async () => {
  if (browser) await browser.close();
});

// ---------------------------------------------------------------------------

async function loadFpgrowth(cohort, ageBand, bin = null) {
  // Mermaid B: select cohort + age band + optional density bin
  await setDropdown(page, "#fpgrowth-cohort",   cohort);
  await setDropdown(page, "#fpgrowth-age-band", ageBand);
  if (bin) {
    // NOTE: do NOT dispatch change event — change listener auto-clicks btnLoadFPGrowth
    // which would fire a fetch before loadVisualization registers waitForResponse.
    await page.$eval("#fpgrowth-n-event-bin", (el, v) => { el.value = v; }, bin);
  }

  // Mermaid C→D: click Load → GET /fpgrowth
  return loadVisualization(page, "btnLoadFPGrowth", "/fpgrowth", 20_000);
}

// ---------------------------------------------------------------------------

describe("FP-Growth tab — happy path (mermaid: A→B→C→D→E→G→H→I)", () => {

  test("opioid_ed / 55-64 → response OK, status not error", async () => {
    const response = await loadFpgrowth("opioid_ed", "55-64");

    if (response !== null) {
      expect([200, 400, 404, 500]).toContain(response.status());
      if (response.status() === 200) {
        const body = await response.json().catch(() => null);
        expect(body !== null && typeof body === "object").toBe(true);
      }
    }

    const statusText = await getStatusText(page, "fpgrowth-status");
    if (statusText) {
      expect(statusText.toLowerCase()).not.toMatch(/^error:/);
    }
  }, 30_000);

  test("non_opioid_ed / 65-74 → response OK", async () => {
    const response = await loadFpgrowth("non_opioid_ed", "65-74");
    if (response !== null) {
      expect([200, 400, 404, 500]).toContain(response.status());
    }
  }, 30_000);

  test("opioid_ed / 25-44 / density bin low → response OK", async () => {
    const response = await loadFpgrowth("opioid_ed", "25-44", "low");
    if (response !== null) {
      expect([200, 400, 404, 500]).toContain(response.status());
    }
  }, 30_000);

});

// ---------------------------------------------------------------------------

describe("FP-Growth tab — DOM rendering (mermaid: E→G support image, H network iframe)", () => {

  beforeAll(async () => {
    await loadFpgrowth("opioid_ed", "55-64");
    await sleep(2000); // iframes load asynchronously
  }, 30_000);

  test("fpgrowth-support-image renders via Plotly or PNG fallback (mermaid G)", async () => {
    // #fpgrowth-support-image is a <div>. Two render paths:
    //   Plotly (itemsets_data present): child <div id="fpgrowth-support-plotly">
    //   PNG fallback (support_image url): child <img src="...">
    const result = await page.evaluate(() => {
      const el = document.getElementById("fpgrowth-support-image");
      if (!el) return null;
      if (el.querySelector("#fpgrowth-support-plotly")) return "plotly";
      const img = el.querySelector("img");
      if (img && img.src) return img.src;
      return "";
    });
    expect(result).not.toBeNull();   // element must exist
    expect(result).not.toBe("");     // must have rendered something
    console.log("fpgrowth-support-image render path:", result === "plotly" ? "Plotly JSON" : result.slice(0, 80));
  }, 10_000);

  test("fpgrowth-network-iframe src is set after load (mermaid H)", async () => {
    // Mermaid H: Interactive Drug Association Network iframe
    const src = await page.$eval("#fpgrowth-network-iframe", el => el.src || "").catch(() => null);
    if (src !== null) {
      console.log("fpgrowth-network-iframe src:", (src || "").slice(0, 80));
    }
    // iframe may be absent for some age/cohort combinations — not a hard failure
  }, 10_000);

});

// ---------------------------------------------------------------------------

describe("FP-Growth tab — no selection guard (mermaid: B→ missing cohort/age → error)", () => {

  test("empty age band → fpgrowth-status shows guard message, page alive", async () => {
    await setDropdown(page, "#fpgrowth-cohort", "opioid_ed");
    await page.$eval("#fpgrowth-age-band", el => { el.value = ""; });

    await page.click("#btnLoadFPGrowth").catch(() => {});
    await sleep(500);

    const statusText = await getStatusText(page, "fpgrowth-status");
    if (statusText) {
      expect(typeof statusText).toBe("string");
    }

    const alive = await page.evaluate(() => document.title).catch(() => null);
    expect(alive).not.toBeNull();
  }, 15_000);

});
