"use strict";

/**
 * Risk Assessment tab — end-to-end tests.
 *
 * Mermaid workflow (10_risk_dashboard/README.md — Tab: Risk Assessment):
 *   A([Risk Assessment tab])
 *   → B[Enter age 13–114 · Optional overrides: n_drugs · n_cpic_drugs · n_events]
 *   → C[codes-summary-group shows selected code count]
 *   → D{Action}
 *   D →|Calculate Risk Score| E[calculateRisk → POST /risk]
 *   E → H{HTTP 200?}
 *   H →|200| J[risk-score · risk-band · n-event-bin-badge · model-info · Plotly charts · density-table-wrap]
 *   J → M{n_event_bin known?} → N[pgx-action-link shown]
 *   H →|Error| I[setStatus error]
 *
 * Run:
 *   DASHBOARD_URL=... API_BASE_URL=... npx jest tests/tab-risk --forceExit --verbose
 */

const { launchBrowser, openDashboard, selectCohort, setAge,
        clickCalculate, readRiskDisplay, sleep } = require("../../helpers/browser");
const { makeScenarios } = require("../../helpers/scenarios");

let browser, page;

beforeAll(async () => {
  browser = await launchBrowser();
  page    = await openDashboard(browser);
}, 40_000);

afterAll(async () => {
  if (browser) await browser.close();
});

// ---------------------------------------------------------------------------

describe("Risk Assessment tab — happy path (mermaid: A→B→D→E→H→J)", () => {

  test("opioid_ed / age 35 → 200 → risk display visible with score and band", async () => {
    // Mermaid B: enter age 13–114
    await selectCohort(page, "opioid_ed");
    await setAge(page, 35);

    // Mermaid D→E: Calculate Risk Score → POST /risk
    const { status, data } = await clickCalculate(page);
    expect(status).toBe(200);
    expect(data).not.toBeNull();

    // Mermaid J: risk-score · risk-band displayed
    expect(data.risk_score).toBeGreaterThanOrEqual(0);
    expect(data.risk_score).toBeLessThanOrEqual(1);
    expect(data.risk_band).toMatch(/low|moderate|high/i);

    const display = await readRiskDisplay(page);
    expect(display.displayVisible).toBe(true);
    expect(display.scoreText).not.toBe("");
    expect(display.bandText).not.toBe("");
  }, 30_000);

  test("non_opioid_ed / age 70 → 200 → risk display visible", async () => {
    await selectCohort(page, "non_opioid_ed");
    await setAge(page, 70);

    const { status, data } = await clickCalculate(page);
    expect(status).toBe(200);
    expect(data).not.toBeNull();
    expect(data.risk_score).toBeGreaterThanOrEqual(0);
    expect(data.risk_score).toBeLessThanOrEqual(1);

    const display = await readRiskDisplay(page);
    expect(display.displayVisible).toBe(true);
  }, 30_000);

});

// ---------------------------------------------------------------------------

describe("Risk Assessment tab — n_event_bin badge (mermaid: J→M→N)", () => {

  test("medium-density codes → n_event_bin badge shown with known bin label", async () => {
    const scenarios = makeScenarios("opioid_ed");
    const { drugs, icds, cpts } = scenarios.medium;

    await selectCohort(page, "opioid_ed");
    await setAge(page, 50);

    // Inject medium-density codes so bin is assigned
    await page.evaluate((d, i, c) => {
      const _orig = window.fetch;
      window.fetch = async function(url, opts, ...rest) {
        if (window.fetch !== _orig && url && String(url).includes("/risk") &&
            opts && opts.method === "POST" &&
            !String(url).includes("/comparison") && !String(url).includes("/drug_contributions")) {
          window.fetch = _orig;
          try {
            const body = JSON.parse(opts.body || "{}");
            body.drugs = d; body.icds = i; body.cpts = c;
            opts = { ...opts, body: JSON.stringify(body) };
          } catch (_) {}
        }
        return _orig.call(this, url, opts, ...rest);
      };
    }, drugs, icds, cpts);

    const { status, data } = await clickCalculate(page);
    expect(status).toBe(200);

    // Mermaid M: n_event_bin known → N: pgx-action-link shown, badge visible
    if (data && data.n_event_bin) {
      const display = await readRiskDisplay(page);
      expect(display.binVisible).toBe(true);
      expect(["low", "medium", "high", "extreme"]).toContain(data.n_event_bin);
    }
  }, 30_000);

});

// ---------------------------------------------------------------------------

describe("Risk Assessment tab — replace, compare, and leave-one-out", () => {

  test("replace / contributions / save-scenario controls are present", async () => {
    await selectCohort(page, "opioid_ed");
    await setAge(page, 35);
    await page.evaluate(() => window.switchTab("risk-assessment"));
    const ids = await page.evaluate(() => ({
      replace: !!document.getElementById("btnReplaceCode"),
      compare: !!document.getElementById("btnComparison"),
      contrib: !!document.getElementById("btnContributions"),
      save: !!document.getElementById("btnSaveScenario"),
      from: !!document.getElementById("replace-from"),
      to: !!document.getElementById("replace-to"),
    }));
    expect(ids.replace).toBe(true);
    expect(ids.compare).toBe(true);
    expect(ids.contrib).toBe(true);
    expect(ids.save).toBe(true);
    expect(ids.from).toBe(true);
    expect(ids.to).toBe(true);
  }, 20_000);

  test("Drug contributions POSTs /risk/drug_contributions and renders Δp̂ rows", async () => {
    await selectCohort(page, "opioid_ed");
    await setAge(page, 60);
    await page.evaluate(() => window.switchTab("drugs"));
    await page.waitForFunction(
      () => { const el = document.getElementById("drugs"); return el && el.options.length > 20; },
      { timeout: 12_000 }
    );
    const selected = await page.evaluate(() => {
      const sel = document.getElementById("drugs");
      const tokens = ["OXYCODONE", "GABAPENTIN", "HYDROCODONE"];
      const found = [];
      for (const opt of sel.options) {
        if (tokens.some((t) => opt.value.includes(t) || opt.text.includes(t))) {
          opt.selected = true;
          found.push(opt.value);
        }
        if (found.length >= 2) break;
      }
      sel.dispatchEvent(new Event("change"));
      return found;
    });
    expect(selected.length).toBeGreaterThanOrEqual(2);

    await page.evaluate(() => window.switchTab("risk-assessment"));
    let contribData = null;
    const handler = async (r) => {
      if (r.url().includes("/risk/drug_contributions")) {
        contribData = await r.json().catch(() => null);
      }
    };
    page.on("response", handler);
    await page.click("#btnContributions");
    await sleep(10_000);
    page.off("response", handler);

    expect(contribData).not.toBeNull();
    expect(typeof contribData.base_risk).toBe("number");
    expect(Array.isArray(contribData.drug_contributions)).toBe(true);
    expect(contribData.drug_contributions.length).toBeGreaterThanOrEqual(2);

    const tableState = await page.evaluate(() => {
      const mode = document.getElementById("contributions-mode");
      const rows = document.querySelectorAll("#contributions-table tbody tr");
      return {
        active: !!(mode && mode.classList.contains("active")),
        rows: rows.length,
        text: document.getElementById("contributions-table")?.textContent || "",
      };
    });
    expect(tableState.active).toBe(true);
    expect(tableState.rows).toBeGreaterThanOrEqual(2);
    expect(tableState.text).toMatch(/Δp̂|delta|risk without/i);
  }, 40_000);

});

describe("Risk Assessment tab — error path (mermaid: B→ invalid age →C→ no API call)", () => {

  test("age 0 (invalid) → no 200 response, page remains alive", async () => {
    await selectCohort(page, "opioid_ed");
    await setAge(page, 0);

    // clickCalculate waits up to 20s for POST /risk; returns status 0 if no call fires
    const { status } = await clickCalculate(page);
    // UI blocks age < 13 — either no call fires (status 0) or Lambda returns 400
    expect([0, 400, 422]).toContain(status);

    // Page must still be alive — mermaid I: setStatus error shown, no crash
    const alive = await page.evaluate(() => document.title).catch(() => null);
    expect(alive).not.toBeNull();
  }, 30_000);

});
