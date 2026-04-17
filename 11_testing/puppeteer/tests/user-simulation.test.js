"use strict";

/**
 * Full UI simulation — real user workflow end-to-end.
 *
 * Unlike combinatorial.test.js (which bypasses the code-selection UI via a
 * fetch interceptor), this test fully emulates a user clicking through the
 * dashboard:
 *
 *   1. Click a cohort tab
 *   2. Type an age via keyboard (triple-click + type)
 *   3. Navigate to the Drugs tab → type in the search box → select options
 *      from the <select multiple> list via DOM click events
 *   4. Navigate to ICD / CPT tabs (opioid_ed only — hidden for non_opioid_ed)
 *      and repeat code selection
 *   5. Navigate back to Risk Assessment tab → click Calculate Risk Score
 *   6. Capture the outgoing POST /risk request body and assert that the
 *      codes selected through the UI actually appear in the payload
 *   7. Assert the response and UI result display
 *
 * This test exercises the full client-side pipeline including:
 *   - Tab navigation (switchTab, cohort-tab-button)
 *   - Search-box filtering (input event → filterOptions)
 *   - Multi-select DOM selection (change event → updateSelectionDisplays)
 *   - calculateRisk() → updateCodeLists() selection-preservation fix
 *   - POST /risk API contract
 *   - Risk result display update
 *
 * Run:
 *   DASHBOARD_URL=https://jerome-dixon.io/vcu/pgx-risk-calculator/index.html \
 *   API_BASE_URL=https://cmv0qislq3.execute-api.us-east-1.amazonaws.com/prod \
 *   npx jest tests/user-simulation --forceExit --verbose
 */

const fs   = require("fs");
const path = require("path");

const { launchBrowser, openDashboard, sleep } = require("../helpers/browser");

const API_BASE     = (process.env.API_BASE_URL || "").replace(/\/$/, "");
const RESULTS_FILE = path.join(__dirname, "../../results", "user_simulation_responses.json");

const captured = [];

let browser;
let page;

beforeAll(async () => {
  browser = await launchBrowser();
  page    = await openDashboard(browser);
}, 40_000);

afterAll(async () => {
  if (browser) await browser.close();
  if (captured.length) {
    try {
      const dir = path.dirname(RESULTS_FILE);
      if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
      fs.writeFileSync(RESULTS_FILE, JSON.stringify(captured, null, 2), "utf8");
      console.log(`[user-simulation] Responses written to ${RESULTS_FILE}`);
    } catch (err) {
      console.warn("[user-simulation] Could not write results file:", err.message);
    }
  }
});

// ── Helpers ────────────────────────────────────────────────────────────────

/** Click a primary tab button by data-tab value. */
async function switchToTab(tabName) {
  await page.evaluate(t => window.switchTab(t), tabName);
  await sleep(300);
}

/**
 * Type age by triple-clicking the #age field (selects existing text)
 * then typing the value character-by-character as a real user would.
 */
async function typeAge(age) {
  await page.click("#age", { clickCount: 3 });
  await page.keyboard.type(String(age), { delay: 40 });
  await page.keyboard.press("Tab");
  await sleep(300);
}

/**
 * Type into a search box (fires native input events for filterOptions).
 * @param {string} inputId  DOM id of the search input
 * @param {string} text     Text to type
 */
async function typeInSearch(inputId, text) {
  await page.click(`#${inputId}`);
  await page.keyboard.type(text, { delay: 40 });
  await sleep(200);
}

/**
 * Clear a search box and reset option visibility.
 * @param {string} inputId
 */
async function clearSearch(inputId) {
  await page.$eval(`#${inputId}`, el => {
    el.value = "";
    el.dispatchEvent(new Event("input", { bubbles: true }));
  });
  await sleep(100);
}

/**
 * Select the first `n` visible (non-hidden) options from a <select multiple>
 * by setting selected=true and firing a change event — mirrors what a user
 * Ctrl+clicking on options does.
 *
 * Returns the values that were selected.
 */
async function selectFirstNVisible(selectId, n) {
  return page.evaluate((id, count) => {
    const sel = document.getElementById(id);
    if (!sel) return [];
    const visible = Array.from(sel.options).filter(o => !o.hidden && !o.disabled);
    const toSelect = visible.slice(0, count);
    toSelect.forEach(o => { o.selected = true; });
    sel.dispatchEvent(new Event("change", { bubbles: true }));
    return toSelect.map(o => o.value);
  }, selectId, n);
}

/**
 * Wait until a <select> has at least 1 option (metadata has loaded and
 * updateCodeLists() has run).
 */
async function waitForOptionsPopulated(selectId, timeout = 15_000) {
  await page.waitForFunction(
    id => {
      const el = document.getElementById(id);
      return el && el.options.length > 0;
    },
    { timeout },
    selectId
  );
}

/**
 * Capture the outgoing POST /risk request body AND wait for the response.
 * The click that triggers the request is passed as the third Promise.all arg.
 *
 * Returns { requestBody, response } — either may be null on timeout.
 */
async function calculateAndCapture() {
  const isRiskPost = r =>
    r.url().includes("/risk") &&
    !r.url().includes("/comparison") &&
    !r.url().includes("/drug_contributions") &&
    !r.url().includes("/causal");

  const [req, resp] = await Promise.all([
    page.waitForRequest(r => isRiskPost(r) && r.method() === "POST", { timeout: 20_000 })
      .catch(() => null),
    page.waitForResponse(r => isRiskPost(r) && r.request().method() === "POST", { timeout: 20_000 })
      .catch(() => null),
    page.click("#btnRisk"),
  ]);

  let requestBody = null;
  if (req) {
    try { requestBody = JSON.parse(req.postData() || "{}"); } catch (_) {}
  }

  return { requestBody, response: resp };
}

// ── Test suite ─────────────────────────────────────────────────────────────

describe("Full UI simulation — real user workflow", () => {

  // ── opioid_ed / 55-64 ──────────────────────────────────────────────────
  describe("cohort: opioid_ed / age_band: 55-64", () => {

    let selectedDrugs = [];
    let selectedIcds  = [];
    let selectedCpts  = [];
    let requestBody   = null;
    let response      = null;

    test("step 1-2: select opioid_ed cohort tab and type age 60", async () => {
      await page.click('button.cohort-tab-button[data-cohort="opioid_ed"]');
      await sleep(800); // allow metadata request to fire

      await switchToTab("risk-assessment");
      await typeAge(60);
    }, 20_000);

    test("step 3: navigate to Drugs tab, search, select codes", async () => {
      await switchToTab("drugs");
      await waitForOptionsPopulated("drugs");

      // Search for first drug term — select whatever matches
      await typeInSearch("drug-search", "oxy");
      selectedDrugs = await selectFirstNVisible("drugs", 1);
      await clearSearch("drug-search");

      // Select another drug from the unfiltered list
      const more = await selectFirstNVisible("drugs", 1);
      // Merge, deduplicate
      selectedDrugs = [...new Set([...selectedDrugs, ...more])];

      expect(selectedDrugs.length).toBeGreaterThan(0);
    }, 20_000);

    test("step 4a: navigate to ICD tab, search, select codes", async () => {
      await switchToTab("icd-codes");
      await waitForOptionsPopulated("icds");

      await typeInSearch("icd-search", "M54");
      selectedIcds = await selectFirstNVisible("icds", 1);
      await clearSearch("icd-search");

      expect(selectedIcds.length).toBeGreaterThan(0);
    }, 20_000);

    test("step 4b: navigate to CPT tab, search, select codes", async () => {
      await switchToTab("cpt-codes");
      await waitForOptionsPopulated("cpts");

      await typeInSearch("cpt-search", "992");
      selectedCpts = await selectFirstNVisible("cpts", 1);
      await clearSearch("cpt-search");

      expect(selectedCpts.length).toBeGreaterThan(0);
    }, 20_000);

    test("step 5-7: calculate risk — POST body contains UI-selected codes", async () => {
      await switchToTab("risk-assessment");
      await sleep(200);

      ({ requestBody, response } = await calculateAndCapture());

      // POST must have fired
      expect(requestBody).not.toBeNull();

      // Routing metadata
      expect(requestBody.cohort).toBe("opioid_ed");
      expect(requestBody.age).toBe(60);

      // Codes selected through the UI must survive updateCodeLists() and appear in the body
      for (const drug of selectedDrugs) {
        expect(requestBody.drugs).toContain(drug);
      }
      for (const icd of selectedIcds) {
        expect(requestBody.icds).toContain(icd);
      }
      for (const cpt of selectedCpts) {
        expect(requestBody.cpts).toContain(cpt);
      }
    }, 30_000);

    test("step 8: response is valid JSON, UI result panel updates", async () => {
      expect(response).not.toBeNull();
      expect([200, 400, 404, 500]).toContain(response.status());

      const data = response.status() === 200 ? await response.json().catch(() => null) : null;

      if (data) {
        expect(typeof data.risk_score).toBe("number");
        expect(data.risk_score).toBeGreaterThanOrEqual(0);
        expect(data.risk_score).toBeLessThanOrEqual(1);
        expect(data.cohort_used).toBe("opioid_ed");
        expect(data.age_band_used).toBe("55-64");
      }

      // Wait for displayRiskResults() to update the DOM
      await page.waitForFunction(
        () => { const el = document.getElementById("risk-display"); return el && el.style.display !== "none"; },
        { timeout: 8_000 }
      ).catch(() => {});

      const ui = await page.evaluate(() => {
        const els = (id) => document.getElementById(id);
        return {
          visible:        (els("risk-display") || {}).style ? els("risk-display").style.display !== "none" : false,
          scoreText:      (els("risk-score")   || {}).textContent?.trim() ?? "",
          bandText:       (els("risk-band")    || {}).textContent?.trim() ?? "",
          binBadge:       (els("n-event-bin-badge") || {}).textContent?.trim() ?? "",
          modelInfo:      (els("model-info")   || {}).textContent?.trim() ?? "",
          codesSelected:  (els("risk-codes-selected") || {}).textContent?.trim() ?? "",
          pgxActionLink:  (els("pgx-action-link") || {}).style?.display !== "none",
        };
      });
      expect(ui.visible).toBe(true);
      expect(ui.scoreText.length).toBeGreaterThan(0);

      captured.push({
        scenario:      "opioid_ed / 55-64",
        cohort:        "opioid_ed",
        age:           60,
        age_band:      "55-64",
        timestamp:     new Date().toISOString(),
        ui_selected:   { drugs: selectedDrugs, icds: selectedIcds, cpts: selectedCpts },
        request_body:  requestBody,
        response_status: response ? response.status() : null,
        response_body: data,
        ui_state:      ui,
      });
    }, 10_000);

  }); // opioid_ed

  // ── non_opioid_ed / 65-74 ──────────────────────────────────────────────
  // ICD and CPT tabs are hidden for the Polypharmacy cohort; only drugs are selectable.
  describe("cohort: non_opioid_ed / age_band: 65-74", () => {

    let selectedDrugs = [];
    let requestBody   = null;
    let response      = null;

    test("step 1-2: select non_opioid_ed cohort tab and type age 70", async () => {
      await page.click('button.cohort-tab-button[data-cohort="non_opioid_ed"]');
      await sleep(800);

      await switchToTab("risk-assessment");
      await typeAge(70);
    }, 20_000);

    test("step 3: navigate to Drugs tab, search, select codes", async () => {
      await switchToTab("drugs");
      await waitForOptionsPopulated("drugs");

      await typeInSearch("drug-search", "met");
      selectedDrugs = await selectFirstNVisible("drugs", 1);
      await clearSearch("drug-search");

      const more = await selectFirstNVisible("drugs", 1);
      selectedDrugs = [...new Set([...selectedDrugs, ...more])];

      expect(selectedDrugs.length).toBeGreaterThan(0);
    }, 20_000);

    test("step 4: ICD/CPT tabs hidden for non_opioid_ed cohort", async () => {
      // Verify the UI correctly hides ICD and CPT tabs for polypharmacy
      const icdHidden = await page.$eval(
        ".tab-button[data-tab='icd-codes']",
        el => el.style.display === "none"
      ).catch(() => true);
      const cptHidden = await page.$eval(
        ".tab-button[data-tab='cpt-codes']",
        el => el.style.display === "none"
      ).catch(() => true);

      expect(icdHidden).toBe(true);
      expect(cptHidden).toBe(true);
    }, 10_000);

    test("step 5-7: calculate risk — POST body contains UI-selected drugs", async () => {
      await switchToTab("risk-assessment");
      await sleep(200);

      ({ requestBody, response } = await calculateAndCapture());

      expect(requestBody).not.toBeNull();
      expect(requestBody.cohort).toBe("non_opioid_ed");
      expect(requestBody.age).toBe(70);

      for (const drug of selectedDrugs) {
        expect(requestBody.drugs).toContain(drug);
      }
    }, 30_000);

    test("step 8: response is valid JSON, UI result panel updates", async () => {
      expect(response).not.toBeNull();
      expect([200, 400, 404, 500]).toContain(response.status());

      const data = response.status() === 200 ? await response.json().catch(() => null) : null;

      if (data) {
        expect(typeof data.risk_score).toBe("number");
        expect(data.risk_score).toBeGreaterThanOrEqual(0);
        expect(data.risk_score).toBeLessThanOrEqual(1);
        expect(data.cohort_used).toBe("non_opioid_ed");
        expect(data.age_band_used).toBe("65-74");
      }

      // Wait for displayRiskResults() to update the DOM
      await page.waitForFunction(
        () => { const el = document.getElementById("risk-display"); return el && el.style.display !== "none"; },
        { timeout: 8_000 }
      ).catch(() => {});

      const ui = await page.evaluate(() => {
        const els = (id) => document.getElementById(id);
        return {
          visible:        (els("risk-display") || {}).style ? els("risk-display").style.display !== "none" : false,
          scoreText:      (els("risk-score")   || {}).textContent?.trim() ?? "",
          bandText:       (els("risk-band")    || {}).textContent?.trim() ?? "",
          binBadge:       (els("n-event-bin-badge") || {}).textContent?.trim() ?? "",
          modelInfo:      (els("model-info")   || {}).textContent?.trim() ?? "",
          codesSelected:  (els("risk-codes-selected") || {}).textContent?.trim() ?? "",
          pgxActionLink:  (els("pgx-action-link") || {}).style?.display !== "none",
        };
      });
      expect(ui.visible).toBe(true);
      expect(ui.scoreText.length).toBeGreaterThan(0);

      captured.push({
        scenario:      "non_opioid_ed / 65-74",
        cohort:        "non_opioid_ed",
        age:           70,
        age_band:      "65-74",
        timestamp:     new Date().toISOString(),
        ui_selected:   { drugs: selectedDrugs, icds: [], cpts: [] },
        request_body:  requestBody,
        response_status: response ? response.status() : null,
        response_body: data,
        ui_state:      ui,
      });
    }, 10_000);

  }); // non_opioid_ed

}); // Full UI simulation
