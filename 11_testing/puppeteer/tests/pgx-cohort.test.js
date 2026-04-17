"use strict";

/**
 * PGx Cohort Network tab — DOM rendering tests.
 *
 * viz.test.js validates the API returns valid JSON; this suite validates what
 * the frontend DOES with that response:
 *   1. cohort-pgx-iframe.src is set to a real URL (not about:blank)
 *   2. cohort-pgx-status shows success (not error)
 *   3. cohort-pgx-citations-section becomes visible after load
 *   4. cohort-pgx-citations-content has per-gene citation entries
 *   5. cohort-pgx-radar-section becomes visible (radar optional — warn only)
 *
 * Run:
 *   npx jest tests/pgx-cohort --forceExit --verbose
 */

const { launchBrowser, openDashboard, sleep } = require("../helpers/browser");

let browser, page;

beforeAll(async () => {
  browser = await launchBrowser();
  page    = await openDashboard(browser);
}, 40_000);

afterAll(async () => {
  if (browser) await browser.close();
});

/** Navigate to the PGx Cohort tab and wait for its container to inject. */
async function openPgxCohortTab() {
  await page.evaluate(() => {
    const btn = document.querySelector('.tab-button[data-tab="cohort-pgx-visualizations"]');
    if (btn) btn.click();
  });
  await page.waitForSelector("#cohort-pgx-cohort", { timeout: 10_000 });
}

/** Set cohort + age band dropdowns and click Load. */
async function loadCohortPgxNetwork(cohort = "opioid_ed", ageBand = "55-64") {
  await page.$eval("#cohort-pgx-cohort",   (el, v) => { el.value = v; }, cohort);
  await page.$eval("#cohort-pgx-age-band", (el, v) => { el.value = v; }, ageBand);

  const [response] = await Promise.all([
    page.waitForResponse(
      r => (r.url().includes("/cohort_pgx") || r.url().includes("network_topology") || r.url().includes("pubmed_citations")),
      { timeout: 15_000 }
    ).catch(() => null),
    page.$eval("#btnLoadCohortPgx", el => el.click()),
  ]);

  // Extra settle for async citation + radar loads that fire after the primary response
  await sleep(3000);
  return response;
}

/** Read a DOM element's visibility + text/attribute. */
async function domState(selector, attr = null) {
  return page.evaluate((sel, a) => {
    const el = document.querySelector(sel);
    if (!el) return null;
    const style = window.getComputedStyle(el);
    return {
      visible: style.display !== "none" && style.visibility !== "hidden" && el.offsetParent !== null,
      text:    el.textContent?.trim().slice(0, 300),
      attr:    a ? el.getAttribute(a) : undefined,
      count:   el.children?.length ?? 0,
    };
  }, selector, attr);
}

// ---------------------------------------------------------------------------

describe("PGx Cohort Network tab — DOM rendering", () => {

  beforeAll(async () => {
    await openPgxCohortTab();
  }, 20_000);

  test("iframe src is set to a real network topology URL (not about:blank)", async () => {
    await loadCohortPgxNetwork("opioid_ed", "55-64");

    const iframe = await domState("#cohort-pgx-iframe", "src");
    expect(iframe).not.toBeNull();

    const src = iframe.attr || "";
    expect(src).not.toBe("about:blank");
    expect(src).not.toBe("");
    expect(src.toLowerCase()).toMatch(/network_topology|cohort_pgx/);

    console.log("iframe src:", src.slice(0, 100));
  }, 30_000);

  test("status message shows success after load", async () => {
    const status = await domState("#cohort-pgx-status");
    expect(status).not.toBeNull();
    expect(status.text.toLowerCase()).not.toMatch(/error|not available/);
    expect(status.text.toLowerCase()).toMatch(/loaded|success|network/);
    console.log("cohort-pgx-status:", status.text.slice(0, 100));
  }, 10_000);

  test("citations section becomes visible with gene entries", async () => {
    const section = await domState("#cohort-pgx-citations-section");
    expect(section).not.toBeNull();
    expect(section.visible).toBe(true);

    // Gene citation entries — rendered as collapsible divs or list items
    const entries = await page.evaluate(() => {
      const candidates = [
        "#cohort-pgx-citations-content > div",
        "#cohort-pgx-citations-content .citation-gene",
        "#cohort-pgx-citations-content details",
        "#cohort-pgx-citations-content li",
      ];
      for (const sel of candidates) {
        const els = document.querySelectorAll(sel);
        if (els.length > 0) return {
          selector: sel,
          count: els.length,
          genes: [...els].slice(0, 3).map(e => e.textContent.trim().slice(0, 40)),
        };
      }
      return null;
    });

    expect(entries).not.toBeNull();
    expect(entries.count).toBeGreaterThan(0);
    console.log(`Citations: ${entries.count} entries via "${entries.selector}" — e.g. ${entries.genes}`);
  }, 10_000);

  test("radar section visible or gracefully absent (data-dependent)", async () => {
    const radar = await domState("#cohort-pgx-radar-section");
    if (radar && radar.visible) {
      // Radar chart container should have Plotly SVG content
      const hasPlot = await page.evaluate(() => {
        const el = document.getElementById("cohort-pgx-radar-section");
        return el ? el.querySelector("svg") !== null : false;
      });
      console.log("Radar section visible, has SVG:", hasPlot);
      // Not failing if radar data missing on S3 — just log
    } else {
      console.warn("cohort-pgx-radar-section not visible — pgx_radar_data.json may not be on S3 for this band");
    }
  }, 10_000);

  test("non_opioid_ed / 65-74 cohort network loads without error", async () => {
    await loadCohortPgxNetwork("non_opioid_ed", "65-74");

    const status = await domState("#cohort-pgx-status");
    expect(status).not.toBeNull();
    expect(status.text.toLowerCase()).not.toMatch(/^error/);

    const iframe = await domState("#cohort-pgx-iframe", "src");
    const src = iframe?.attr || "";
    expect(src).not.toBe("about:blank");
    console.log("non_opioid_ed/65-74 iframe src:", src.slice(0, 100));
  }, 30_000);

});
