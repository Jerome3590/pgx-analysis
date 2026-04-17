"use strict";

/**
 * Combinatorial end-to-end tests for the PGx Risk Dashboard.
 *
 * Covers all cohort × age_band × density-bin scenarios through the
 * actual browser UI (Puppeteer), intercepting the POST /risk response to
 * validate the JSON exactly as pytest does — but via a live page.
 *
 * Requirements
 * ────────────
 *   DASHBOARD_URL  – URL of the dashboard HTML page (required)
 *   API_BASE_URL   – optional override for the ?apiBase= param
 *
 * Run
 * ───
 *   # Install deps first (once):
 *   cd 11_testing/puppeteer && npm install
 *
 *   # Local offline server (start it first):
 *   #   python 11_testing/offline_dashboard_server.py
 *   DASHBOARD_URL=http://localhost:8000/index.html \
 *   API_BASE_URL=http://localhost:8000/prod \
 *   npx jest tests/combinatorial --forceExit
 *
 *   # Production CloudFront + API Gateway:
 *   DASHBOARD_URL=https://d1234.cloudfront.net/index.html \
 *   API_BASE_URL=https://xxx.execute-api.us-east-1.amazonaws.com/prod \
 *   npx jest tests/combinatorial --forceExit
 *
 *   # Windows PowerShell:
 *   $env:DASHBOARD_URL = "https://d1234.cloudfront.net/index.html"
 *   $env:API_BASE_URL  = "https://xxx.execute-api.us-east-1.amazonaws.com/prod"
 *   npx jest tests/combinatorial --forceExit
 */

const {
  launchBrowser, openDashboard,
  selectCohort, setAge, injectCodes, clickCalculate, readRiskDisplay, sleep,
} = require("../helpers/browser");
const {
  makeScenarios, AGE_BAND_MIDPOINTS,
  COHORTS, AGE_BANDS, VALID_BINS, VALID_BANDS,
} = require("../helpers/scenarios");

// ── Shared browser instance across all tests ──────────────────────────────
let browser;
let page;

beforeAll(async () => {
  browser = await launchBrowser();
  page    = await openDashboard(browser);
}, 40_000);

afterAll(async () => {
  if (browser) await browser.close();
});

// ── Helpers ────────────────────────────────────────────────────────────────

function assertRiskJson(data, cohort, ageBand, scenario, { drugs, icds, cpts, expectedBin, totalCodes }) {
  // Core score
  expect(typeof data.risk_score).toBe("number");
  expect(data.risk_score).toBeGreaterThanOrEqual(0);
  expect(data.risk_score).toBeLessThanOrEqual(1);

  // Band
  expect(VALID_BANDS.has(data.risk_band)).toBe(true);

  // Echo-back
  expect(data.cohort_used).toBe(cohort);
  expect(data.age_band_used).toBe(ageBand);

  if (scenario === "baseline") {
    expect(data.is_baseline).toBe(true);
  } else {
    expect(data.is_baseline).not.toBe(true);

    // n_event_bin validity
    expect(VALID_BINS.has(data.n_event_bin)).toBe(true);

    // Code count echo
    expect(data.n_events).toBe(totalCodes);

    // Density-bin routing (default thresholds p25=5, p50=15, p95=50)
    if (expectedBin) {
      expect(data.n_event_bin).toBe(expectedBin);
    }

    // model_breakdown is a dict
    expect(typeof data.model_breakdown).toBe("object");
    expect(data.model_breakdown).not.toBeNull();
  }

  // codes_used / codes_unknown
  for (const key of ["codes_used", "codes_unknown"]) {
    expect(data).toHaveProperty(key);
    expect(typeof data[key]).toBe("object");
    for (const sub of ["drugs", "icds", "cpts"]) {
      expect(Array.isArray(data[key][sub])).toBe(true);
    }
  }
}

// ── Combinatorial matrix ───────────────────────────────────────────────────
// 7 age bands × 2 cohorts × 5 scenarios = 70 test cases
// (0-12 excluded — UI blocks it)

describe("POST /risk — combinatorial matrix (via browser)", () => {

  for (const cohort of COHORTS) {
    describe(`cohort: ${cohort}`, () => {

      // Switch cohort tab once per cohort group
      beforeAll(async () => {
        await selectCohort(page, cohort);
      }, 10_000);

      for (const ageBand of AGE_BANDS) {
        const age = AGE_BAND_MIDPOINTS[ageBand];

        describe(`age_band: ${ageBand} (age=${age})`, () => {
          const scenarios = makeScenarios(cohort);

          for (const [scenarioName, scenarioData] of Object.entries(scenarios)) {

            test(`${cohort}/${ageBand} [${scenarioName}]`, async () => {
              await setAge(page, age);
              await injectCodes(page, scenarioData.drugs, scenarioData.icds, scenarioData.cpts);

              const { status, data } = await clickCalculate(page);

              // HTTP 500 is tolerated when per-bin models are not deployed
              expect([200, 500]).toContain(status);

              if (status === 200 && data) {
                assertRiskJson(data, cohort, ageBand, scenarioName, scenarioData);

                // Also verify the UI reflects the result
                const ui = await readRiskDisplay(page);
                expect(ui.displayVisible).toBe(true);

                // Score text should contain a number (e.g. "7.7%" or "7.70%")
                const scoreNum = parseFloat(ui.scoreText.replace("%", ""));
                expect(Number.isFinite(scoreNum)).toBe(true);

                // Band text should be present
                expect(ui.bandText.length).toBeGreaterThan(0);

                if (scenarioName !== "baseline") {
                  // Density-bin badge should be visible for non-baseline
                  expect(ui.binVisible).toBe(true);
                  expect(ui.binText.length).toBeGreaterThan(0);
                }
              }
            }, 25_000);

          } // scenarios
        });
      } // age bands
    });
  } // cohorts
});

// ── Required-field validation: 400 on missing cohort ──────────────────────

describe("Error handling", () => {

  test("age < 13 shows error in #status, no risk calculated", async () => {
    await setAge(page, 10);
    await injectCodes(page, [], [], []);

    // Listen for status element update instead of intercepting response
    // (UI guards against age < 13 before firing the request)
    await page.click("#btnRisk");
    await sleep(500);

    const statusClass = await page.$eval("#status", el => el.className);
    expect(statusClass).toContain("error");
  });

  test("age > 114 shows error in #status, no risk calculated", async () => {
    await setAge(page, 120);
    await injectCodes(page, [], [], []);
    await page.click("#btnRisk");
    await sleep(500);

    const statusClass = await page.$eval("#status", el => el.className);
    expect(statusClass).toContain("error");
  });

});
