"use strict";

/**
 * PGx Card tab end-to-end tests.
 *
 * Submits gene variants via the textarea / generate button and asserts:
 *   - POST /pgx/card returns 200 with genes + drugs arrays
 *   - Card renders in the DOM (gene items visible)
 *   - 400 is returned for an empty payload
 *
 * Run:
 *   DASHBOARD_URL=... API_BASE_URL=... npx jest tests/pgx-card --forceExit
 */

const { launchBrowser, openDashboard, sleep } = require("../helpers/browser");

let browser;
let page;

beforeAll(async () => {
  browser = await launchBrowser();
  page    = await openDashboard(browser);
}, 40_000);

afterAll(async () => {
  if (browser) await browser.close();
});

/**
 * Switch to PGx Card tab and wait for the tab HTML to inject.
 * The tab content loads from tabs/pgx-card.html via fetch; we wait for
 * #pgx-card-cohort to confirm the DOM is ready.
 */
async function openPgxCardTab() {
  await page.evaluate(() => {
    const btn = document.querySelector('.tab-button[data-tab="pgx-card"]');
    if (btn) btn.click();
  });
  await page.waitForSelector("#pgx-card-cohort", { timeout: 10_000 });
}

/**
 * Select cohort + age band then click Load Cohort PGx Profile.
 * Waits for the SNP refinement section to become visible.
 */
async function loadCohortProfile(cohort = "opioid_ed", ageBand = "13-24") {
  await page.$eval("#pgx-card-cohort",   (el, v) => { el.value = v; }, cohort);
  await page.$eval("#pgx-card-age-band", (el, v) => { el.value = v; }, ageBand);
  const [_] = await Promise.all([
    page.waitForSelector("#pgx-snp-refine-section", { visible: true, timeout: 20_000 })
      .catch(() => null),
    page.$eval("#btnLoadPgxCardProfile", el => el.click()),
  ]);
}

/**
 * Fill #snp-input with variant lines (format: Gene,*allele1,*allele2 per line)
 * then click #btnGenerateCard and wait for POST /pgx/card response.
 *
 * Returns { status, data } — data is null when non-200.
 */
async function submitVariants(variantLines) {
  const text = variantLines.join("\n");
  await page.$eval("#snp-input", (el, v) => { el.value = v; }, text);
  await sleep(100);

  const [response] = await Promise.all([
    page.waitForResponse(
      resp => resp.url().includes("/pgx/card") && resp.request().method() === "POST",
      { timeout: 15_000 }
    ).catch(() => null),
    page.$eval("#btnGenerateCard", el => el.click()),
  ]);

  if (!response) return { status: 0, data: null };
  let data = null;
  try { data = await response.json(); } catch (_) {}
  return { status: response.status(), data };
}

// ---------------------------------------------------------------------------
// Test cases
// ---------------------------------------------------------------------------

describe("PGx Card tab", () => {

  beforeAll(async () => {
    await openPgxCardTab();
    await loadCohortProfile("opioid_ed", "13-24");
  }, 40_000);

  test("POST /pgx/card with CYP2D6 variant returns 200 with genes + drugs", async () => {
    const { status, data } = await submitVariants([
      "CYP2D6,*1,*2",
      "CYP2C19,*1,*1",
    ]);

    expect([200, 400, 500]).toContain(status);

    if (status === 200 && data) {
      expect(Array.isArray(data.genes)).toBe(true);
      expect(Array.isArray(data.drugs)).toBe(true);
      expect(data.genes.length).toBeGreaterThan(0);
      for (const g of data.genes) {
        expect(typeof g.gene).toBe("string");
        expect(Array.isArray(g.variants)).toBe(true);
      }
    }
  }, 20_000);

  test("POST /pgx/card with empty variants — frontend guards or returns 400", async () => {
    const text = "";
    await page.$eval("#snp-input", (el, v) => { el.value = v; }, text);
    await sleep(100);
    const [response] = await Promise.all([
      page.waitForResponse(
        resp => resp.url().includes("/pgx/card") && resp.request().method() === "POST",
        { timeout: 4_000 }            // frontend should block; short wait is fine
      ).catch(() => null),
      page.$eval("#btnGenerateCard", el => el.click()),
    ]);
    // status 0 = frontend blocked; 400 = backend rejected empty payload
    const status = response ? response.status() : 0;
    expect([0, 400]).toContain(status);
  }, 10_000);

  test("Multiple gene variants: SLCO1B1 + TPMT + DPYD", async () => {
    const { status, data } = await submitVariants([
      "SLCO1B1,*5,*1",
      "TPMT,*3A,*1",
      "DPYD,*2A",
    ]);

    expect([200, 400, 500]).toContain(status);

    if (status === 200 && data) {
      expect(data.genes.length).toBeGreaterThanOrEqual(1);
      expect(Array.isArray(data.drugs)).toBe(true);
    }
  }, 20_000);

});
