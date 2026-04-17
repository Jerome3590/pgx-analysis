"use strict";

/**
 * Full UI simulation — all cohorts × all age bands, with importance-driven
 * risk-increase verification.
 *
 * For each cohort/age-band:
 *   1. Fetch metadata from the live API → extract top-importance drugs/ICDs/CPTs.
 *   2. Baseline run  — no codes selected, low n_events (→ low utilization).
 *   3. High-risk run — top-importance codes selected, high n_events.
 *   4. Assert: high_risk_score > baseline_score  (model responds to features).
 *
 * Results → 11_testing/results/full_coverage_results.csv
 *
 * Run:
 *   npx jest tests/full-coverage --forceExit --verbose
 */

const fs   = require("fs");
const path = require("path");

const { launchBrowser, openDashboard, API_BASE_URL, sleep } = require("../helpers/browser");

// ── Constants ──────────────────────────────────────────────────────────────

const API_BASE  = (API_BASE_URL || "https://cmv0qislq3.execute-api.us-east-1.amazonaws.com/prod").replace(/\/$/, "");

const COHORTS   = ["opioid_ed", "non_opioid_ed"];
const AGE_BANDS = ["0-12", "13-24", "25-44", "45-54", "55-64", "65-74", "75-84", "85-114"];

const BAND_AGE = {
  "0-12":   6,  "13-24":  18, "25-44":  35, "45-54":  50,
  "55-64":  60, "65-74":  70, "75-84":  80, "85-114": 90,
};

const TOP_N_DRUGS = 3;   // top-importance drugs to select
const TOP_N_ICDS  = 2;   // top-importance ICDs (opioid_ed only)
const TOP_N_CPTS  = 1;   // top-importance CPTs (opioid_ed only)

const N_EVENTS_BASELINE  = 5;    // low utilization → low bin
const N_EVENTS_HIGH_RISK = 200;  // higher utilization → higher bin, more features

const RESULTS_DIR = path.join(__dirname, "../../results");
const CSV_FILE    = path.join(RESULTS_DIR, "full_coverage_results.csv");

const CSV_HEADER = [
  "timestamp", "cohort", "age_band", "age_input",
  "top_drugs", "top_icds", "top_icds_importance", "top_drugs_importance",
  "baseline_status", "baseline_score", "baseline_band", "baseline_bin",
  "highrisk_status", "highrisk_score", "highrisk_band", "highrisk_bin",
  "highrisk_n_events_for_bin", "highrisk_model_used", "highrisk_bin_model",
  "highrisk_models_failed",
  "risk_increased",          // highrisk_score > baseline_score
  "score_delta",             // highrisk_score - baseline_score
  "ui_visible", "ui_score_text", "ui_band_text", "ui_bin_badge",
  "duration_ms", "error",
].join(",");

// ── CSV helpers ────────────────────────────────────────────────────────────

function csvEscape(v) {
  if (v === null || v === undefined) return "";
  const s = Array.isArray(v) ? v.join("|") : String(v);
  return s.includes(",") || s.includes('"') || s.includes("\n")
    ? `"${s.replace(/"/g, '""')}"` : s;
}

function appendCsvRow(row) {
  const line = CSV_HEADER.split(",").map(h => csvEscape(row[h])).join(",");
  fs.appendFileSync(CSV_FILE, line + "\n", "utf8");
}

// ── Metadata fetch ─────────────────────────────────────────────────────────

/** Fetch metadata for a cohort and extract top-importance codes per type. */
async function fetchTopFeatures(cohort, band) {
  const resp = await fetch(`${API_BASE}/metadata?cohort=${cohort}`);
  if (!resp.ok) throw new Error(`Metadata fetch failed: ${resp.status}`);
  const meta   = await resp.json();
  const codes  = (meta.codes || {})[band] || {};

  const topBy = (arr, n) =>
    [...(arr || [])]
      .sort((a, b) => (b.importance || 0) - (a.importance || 0))
      .slice(0, n);

  const drugs = topBy(codes.drugs, TOP_N_DRUGS);
  const icds  = topBy(codes.icds,  TOP_N_ICDS);
  const cpts  = topBy(codes.cpts,  TOP_N_CPTS);

  return { drugs, icds, cpts };
}

// ── Page helpers ────────────────────────────────────────────────────────────

async function switchToTab(page, tabName) {
  await page.evaluate(t => window.switchTab(t), tabName);
  await sleep(300);
}

async function selectCohort(page, cohort) {
  await page.click(`button.cohort-tab-button[data-cohort="${cohort}"]`);
  await sleep(800);
}

async function typeAge(page, age) {
  await page.click("#age", { clickCount: 3 });
  await page.keyboard.type(String(age), { delay: 40 });
  await page.keyboard.press("Tab");
  await sleep(200);
}

async function setNEvents(page, n) {
  const el = await page.$("#inputNevents");
  if (!el) return;
  await page.$eval("#inputNevents", (node, v) => {
    node.value = v;
    node.dispatchEvent(new Event("input",  { bubbles: true }));
    node.dispatchEvent(new Event("change", { bubbles: true }));
  }, String(n));
  await sleep(150);
}

async function clearAllSelections(page) {
  await page.evaluate(() => {
    ["drugs", "icds", "cpts"].forEach(id => {
      const sel = document.getElementById(id);
      if (sel) { Array.from(sel.options).forEach(o => { o.selected = false; }); }
    });
  });
  await sleep(100);
}

async function waitForOptions(page, selectId, timeout = 15_000) {
  await page.waitForFunction(
    id => { const el = document.getElementById(id); return el && el.options.length > 0; },
    { timeout }, selectId
  );
}

/**
 * Select specific codes by value in a <select multiple>.
 * Returns the values that were actually found and selected.
 */
async function selectByValues(page, selectId, values) {
  return page.evaluate((id, vals) => {
    const sel = document.getElementById(id);
    if (!sel || !vals.length) return [];
    const selected = [];
    Array.from(sel.options).forEach(o => {
      if (vals.includes(o.value)) { o.selected = true; selected.push(o.value); }
    });
    if (selected.length) sel.dispatchEvent(new Event("change", { bubbles: true }));
    return selected;
  }, selectId, values);
}

async function calculateAndCapture(page) {
  const isRiskPost = r =>
    r.url().includes("/risk") &&
    !r.url().includes("/comparison") &&
    !r.url().includes("/drug_contributions") &&
    !r.url().includes("/causal");

  const [req, resp] = await Promise.all([
    page.waitForRequest(r => isRiskPost(r) && r.method() === "POST", { timeout: 25_000 }).catch(() => null),
    page.waitForResponse(r => isRiskPost(r) && r.request().method() === "POST", { timeout: 25_000 }).catch(() => null),
    page.click("#btnRisk"),
  ]);

  let requestBody = null;
  if (req) { try { requestBody = JSON.parse(req.postData() || "{}"); } catch (_) {} }
  return { requestBody, response: resp };
}

async function waitForRiskDisplay(page) {
  await page.waitForFunction(
    () => { const el = document.getElementById("risk-display"); return el && el.style.display !== "none"; },
    { timeout: 8_000 }
  ).catch(() => {});
}

async function readUiState(page) {
  return page.evaluate(() => {
    const g = id => document.getElementById(id);
    return {
      visible:   !!(g("risk-display")?.style.display !== "none"),
      scoreText: g("risk-score")?.textContent?.trim()        ?? "",
      bandText:  g("risk-band")?.textContent?.trim()         ?? "",
      binBadge:  g("n-event-bin-badge")?.textContent?.trim() ?? "",
    };
  });
}

async function parseRiskResponse(response) {
  if (!response || response.status() !== 200) return null;
  return response.json().catch(() => null);
}

// ── Scenarios ───────────────────────────────────────────────────────────────

const SCENARIOS = [];
for (const cohort of COHORTS)
  for (const band of AGE_BANDS)
    SCENARIOS.push({ cohort, band, age: BAND_AGE[band] });

// ── Suite ───────────────────────────────────────────────────────────────────

let browser, page;

beforeAll(async () => {
  if (!fs.existsSync(RESULTS_DIR)) fs.mkdirSync(RESULTS_DIR, { recursive: true });
  fs.writeFileSync(CSV_FILE, CSV_HEADER + "\n", "utf8");
  browser = await launchBrowser();
  page    = await openDashboard(browser);
}, 60_000);

afterAll(async () => {
  if (browser) await browser.close();
  console.log(`\n[full-coverage] Results → ${CSV_FILE}`);
});

describe("Full coverage — all cohorts × age bands (importance-driven risk increase)", () => {

  test.each(SCENARIOS)(
    "$cohort / $band",
    async ({ cohort, band, age }) => {
      const t0  = Date.now();
      const row = {
        timestamp: new Date().toISOString(),
        cohort, age_band: band, age_input: age,
        top_drugs: null, top_icds: null,
        top_drugs_importance: null, top_icds_importance: null,
        baseline_status: null, baseline_score: null,
        baseline_band:   null, baseline_bin:   null,
        highrisk_status: null, highrisk_score:  null,
        highrisk_band:   null, highrisk_bin:    null,
        highrisk_n_events_for_bin: null,
        highrisk_model_used: null, highrisk_bin_model: null,
        highrisk_models_failed: null,
        risk_increased: null, score_delta: null,
        ui_visible: null, ui_score_text: null,
        ui_band_text: null, ui_bin_badge: null,
        duration_ms: null, error: null,
      };

      try {
        // ── 1. Load top-importance features from metadata API ──────────────
        const features = await fetchTopFeatures(cohort, band);
        const topDrugValues = features.drugs.map(d => d.code);
        const topIcdValues  = features.icds.map(d => d.code);
        const topCptValues  = features.cpts.map(d => d.code);

        row.top_drugs            = topDrugValues.join("|");
        row.top_icds             = topIcdValues.join("|");
        row.top_drugs_importance = features.drugs.map(d => d.importance?.toFixed(4)).join("|");
        row.top_icds_importance  = features.icds.map(d => d.importance?.toFixed(4)).join("|");

        // ── 2. BASELINE run — no codes, low n_events ──────────────────────
        await selectCohort(page, cohort);
        await switchToTab(page, "risk-assessment");
        await typeAge(page, age);
        await setNEvents(page, N_EVENTS_BASELINE);
        await clearAllSelections(page);

        await switchToTab(page, "risk-assessment");
        await sleep(200);
        const baseResult = await calculateAndCapture(page);
        const baseData   = await parseRiskResponse(baseResult.response);

        row.baseline_status = baseResult.response?.status() ?? null;
        if (baseData) {
          row.baseline_score = baseData.risk_score;
          row.baseline_band  = baseData.risk_band;
          row.baseline_bin   = baseData.n_event_bin;
        }

        // ── 3. HIGH-RISK run — top-importance codes, high n_events ─────────
        await selectCohort(page, cohort);
        await switchToTab(page, "risk-assessment");
        await typeAge(page, age);
        await setNEvents(page, N_EVENTS_HIGH_RISK);

        // Select top drugs
        await switchToTab(page, "drugs");
        await waitForOptions(page, "drugs");
        await selectByValues(page, "drugs", topDrugValues);

        // Select top ICDs / CPTs (opioid_ed only)
        if (cohort === "opioid_ed") {
          if (topIcdValues.length) {
            await switchToTab(page, "icd-codes");
            await waitForOptions(page, "icds");
            await selectByValues(page, "icds", topIcdValues);
          }
          if (topCptValues.length) {
            await switchToTab(page, "cpt-codes");
            await waitForOptions(page, "cpts");
            await selectByValues(page, "cpts", topCptValues);
          }
        }

        await switchToTab(page, "risk-assessment");
        await sleep(200);
        const highResult = await calculateAndCapture(page);
        const highData   = await parseRiskResponse(highResult.response);

        row.highrisk_status = highResult.response?.status() ?? null;
        if (highData) {
          row.highrisk_score           = highData.risk_score;
          row.highrisk_band            = highData.risk_band;
          row.highrisk_bin             = highData.n_event_bin;
          row.highrisk_n_events_for_bin = highData.n_events_for_bin;
          row.highrisk_model_used      = highData.model_used;
          row.highrisk_bin_model       = highData.bin_model_used;
          row.highrisk_models_failed   = (highData.ensemble_info?.models_failed || []).join("|");
        }

        // ── 4. Risk-increase assertion ─────────────────────────────────────
        if (row.baseline_score !== null && row.highrisk_score !== null) {
          row.score_delta    = +(row.highrisk_score - row.baseline_score).toFixed(6);
          row.risk_increased = row.highrisk_score > row.baseline_score;
        }

        // ── 5. UI state after high-risk run ───────────────────────────────
        await waitForRiskDisplay(page);
        const ui = await readUiState(page);
        row.ui_visible    = ui.visible;
        row.ui_score_text = ui.scoreText;
        row.ui_band_text  = ui.bandText;
        row.ui_bin_badge  = ui.binBadge;

        // ── Assertions ────────────────────────────────────────────────────
        expect(row.baseline_status).toBe(200);
        expect(row.highrisk_status).toBe(200);
        expect(typeof row.baseline_score).toBe("number");
        expect(typeof row.highrisk_score).toBe("number");
        expect(row.highrisk_score).toBeGreaterThanOrEqual(0);
        expect(row.highrisk_score).toBeLessThanOrEqual(1);
        expect(row.ui_visible).toBe(true);

        // Core: selecting known high-importance features must increase risk
        expect(row.risk_increased).toBe(true);

      } catch (err) {
        row.error       = err.message?.slice(0, 300);
        row.duration_ms = Date.now() - t0;
        appendCsvRow(row);
        throw err;
      }

      row.duration_ms = Date.now() - t0;
      appendCsvRow(row);
    },
    120_000
  );
});
