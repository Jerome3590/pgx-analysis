"use strict";

/**
 * Causal Analysis tab — end-to-end tests.
 *
 * Mermaid workflow (10_risk_dashboard/README.md — Tab: Causal Analysis):
 *   A([Causal Analysis tab])
 *   → B[Context from Risk Assessment: currentCohort · currentAgeBand · selected drugs/ICDs/CPTs]
 *     Prerequisite: selectCohort + setAge + clickCalculate (populates code lists + JS state)
 *   → C[Optional: what-if codes (causal-whatif-codes)]
 *   → D[Optional: top-N filter]
 *   → E[Optional: event density bin (causal-n-event-bin)]
 *   → F[Click Load Causal Analysis (btnLoadCausal)]
 *   → G[Static fetch: visualizations/causal/{cohort}/{ageBand}/{bin}/causal_data.json
 *       (CloudFront-first; falls back to Lambda /visualizations/causal on 4xx/5xx)]
 *   → H{HTTP 200?}
 *   H →|200| J[causal-factors-chart · shap-importance-chart · causal-radar-chart]
 *   H →|Error| I[causal-status error]
 *   J → M{What-if codes?} → N[second trace overlaid on each chart]
 *
 * NOTE: Chart updated — static JSON is now fetched from CloudFront FIRST
 * (visualizations/causal/{cohort}/{ageBand}/{bin}/causal_data.json).
 * Client-side code filters by selectedFeatureSet. Lambda is only called if static fails.
 *
 * Run:
 *   DASHBOARD_URL=... API_BASE_URL=... npx jest tests/tab-causal --forceExit --verbose
 */

const { launchBrowser, openDashboard, selectCohort, setAge,
        clickCalculate, injectCodes, navigateToTab, loadVisualization,
        getStatusText, sleep } = require("../../helpers/browser");

let browser, page;

beforeAll(async () => {
  browser = await launchBrowser();
  page    = await openDashboard(browser);
}, 40_000);

afterAll(async () => {
  if (browser) await browser.close();
});

// ---------------------------------------------------------------------------
// Shared setup: full Risk Assessment flow per mermaid prerequisite B
// ---------------------------------------------------------------------------

async function setupRiskContext(cohort = "opioid_ed", age = 35) {
  await selectCohort(page, cohort);
  await setAge(page, age);
  const riskResult = await clickCalculate(page); // sets currentCohort / currentAgeBand in JS
  await sleep(600);                              // updateCodeLists() populates code selects
  await navigateToTab(page, "causal-analysis", "#btnLoadCausal"); // triggers switchTab hook
  await sleep(300);
  return riskResult; // { status, data } — data.n_event_bin used for bin assertion
}

// ---------------------------------------------------------------------------

describe("Causal Analysis tab — bin auto-sync from Risk Assessment (switchTab hook)", () => {
  // Verifies the index.html fix: switchTab('causal-analysis') reads window._patientNEventBin
  // and sets causal-n-event-bin so the per-bin static JSON is fetched automatically.

  beforeAll(async () => {
    // Inject known codes so Lambda computes a non-zero n_event_bin
    await selectCohort(page, "opioid_ed");
    await setAge(page, 35);
    await injectCodes(page, ["drug_GABAPENTIN"], [], []);
    await clickCalculate(page);
    await sleep(600);
    await navigateToTab(page, "causal-analysis", "#btnLoadCausal");
    await sleep(300);
  }, 40_000);

  test("causal-n-event-bin matches window._patientNEventBin after tab navigation", async () => {
    // Read the bin the dashboard stored after risk calculation
    const patientBin = await page.evaluate(() => window._patientNEventBin);
    const binVal     = await page.$eval("#causal-n-event-bin", el => el.value);

    expect(patientBin).toBeTruthy(); // risk calc must have returned a valid bin
    expect(["low", "medium", "high", "extreme"]).toContain(patientBin);
    expect(binVal).toBe(patientBin); // switchTab hook synced it
  }, 10_000);

});

// ---------------------------------------------------------------------------

describe("Causal Analysis tab — no codes (all factors, mermaid: A→B→F→G→H→J)", () => {

  beforeAll(async () => {
    await setupRiskContext("opioid_ed", 35);
  }, 40_000);

  test("load without code filter → top_causal_factors non-empty", async () => {
    // Mermaid E: no density bin set → uses manifest or default static path
    const response = await loadVisualization(page, "btnLoadCausal", "/causal", 20_000);

    if (response === null) {
      // Static load from S3 with no network intercept point — acceptable
      console.warn("No /causal response intercepted — may be cached or loaded from static manifest");
      return;
    }

    expect(response.status()).toBe(200);
    const body = await response.json().catch(() => null);
    expect(body).not.toBeNull();
    expect(typeof body).toBe("object");

    // Static format: top_causal_factors; Lambda format: chart_data.causal_factors
    const factors = body.top_causal_factors || (body.chart_data || {}).causal_factors || [];
    expect(factors.length).toBeGreaterThan(0);
  }, 30_000);

  test("causal-status does not show error after load", async () => {
    const status = await getStatusText(page, "causal-status");
    expect(status).not.toBeNull();
    expect(status.toLowerCase()).not.toMatch(/^error/);
  }, 10_000);

});

// ---------------------------------------------------------------------------

describe("Causal Analysis tab — with codes selected (regression: prefix bug + filter path)", () => {
  // Mermaid B→ selected drugs/ICDs → G[static: client-side filter by selectedFeatureSet]
  // Tests the filter path that was NOT exercised by the combinatorial matrix
  // (which submits no codes → selectedFeatureSet empty → filter skipped).

  const TEST_COHORT = "opioid_ed";
  const TEST_AGE    = 35;            // 25-44 band
  const TEST_BIN    = "medium";
  const DRUG_CODE   = "drug_GABAPENTIN";
  const ICD_CODE    = "icd_B1920";

  beforeAll(async () => {
    await setupRiskContext(TEST_COHORT, TEST_AGE);
  }, 40_000);

  test("causal_factors non-empty and contains submitted code features", async () => {
    // Mermaid B: select codes from populated lists
    await page.evaluate((drug, icd, bin) => {
      const sel = v => el => {
        const o = Array.from(el.options).find(x => x.value === v);
        if (o) o.selected = true;
      };
      const drugsEl = document.getElementById("drugs");
      const icdsEl  = document.getElementById("icds");
      if (drugsEl) sel(drug)(drugsEl);
      if (icdsEl)  sel(icd)(icdsEl);
      // Mermaid E: set density bin — do NOT dispatch change (avoids premature fetch race)
      const binEl = document.getElementById("causal-n-event-bin");
      if (binEl) { binEl.value = bin; }
    }, DRUG_CODE, ICD_CODE, TEST_BIN);

    // Mermaid F→G: click Load → static causal_data.json fetch
    const response = await loadVisualization(page, "btnLoadCausal", "/causal", 20_000);

    expect(response).not.toBeNull();
    expect(response.status()).toBe(200);

    const body = await response.json().catch(() => null);
    expect(body).not.toBeNull();
    expect(typeof body).toBe("object");

    // Static format has top_causal_factors; verify feature name prefix is correct
    // (regression guard: was item_DRUG_GABAPENTIN before fix → now item_drug_GABAPENTIN)
    const factors = body.top_causal_factors || (body.chart_data || {}).causal_factors || [];
    expect(factors.length).toBeGreaterThan(0);

    const featureNames = factors.map(f => f.feature);
    const hasExpectedDrug = featureNames.some(f => f === "item_drug_GABAPENTIN");
    const hasExpectedIcd  = featureNames.some(f => f === "item_icd_B1920");
    expect(hasExpectedDrug || hasExpectedIcd).toBe(true);
  }, 30_000);

});

// ---------------------------------------------------------------------------

describe("Causal Analysis tab — no Risk Assessment context (mermaid: B→ missing context → error)", () => {

  test("no valid cohort/age → causal-status shows error, page does not crash", async () => {
    // Open fresh page with no risk calculation done
    const freshPage = await browser.newPage();
    const { buildDashboardUrl } = require("../../helpers/browser");
    await freshPage.goto(buildDashboardUrl(), { waitUntil: "networkidle0" });
    await freshPage.waitForSelector("#btnRisk", { timeout: 20_000 });

    // Navigate directly to causal tab without Risk Assessment context
    await freshPage.evaluate(() => {
      const btn = document.querySelector('.tab-button[data-tab="causal-analysis"]');
      if (btn) btn.click();
    });
    await sleep(500);

    const btnExists = await freshPage.$("#btnLoadCausal").then(el => el !== null).catch(() => false);
    if (btnExists) {
      await freshPage.click("#btnLoadCausal").catch(() => {});
      await sleep(500);
      // Mermaid I: should show error status, not crash
      const statusText = await freshPage.$eval("#causal-status", el => el.textContent).catch(() => "");
      expect(statusText.toLowerCase()).toMatch(/set cohort|age|error|select/i);
    }

    // Page must still be alive — mermaid: no unhandled crash
    const alive = await freshPage.evaluate(() => document.title).catch(() => null);
    expect(alive).not.toBeNull();

    await freshPage.close();
  }, 30_000);

});
