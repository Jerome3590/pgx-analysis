"use strict";

/**
 * Browser / page helpers shared across test files.
 *
 * DASHBOARD_URL   – full URL to the dashboard HTML (required)
 *                   e.g. https://d1234.cloudfront.net/index.html
 *                        or http://localhost:8000/index.html
 *
 * API_BASE_URL    – optional override for the ?apiBase= query param.
 *                   Defaults to the dashboard's own production API_BASE if not set.
 *                   e.g. http://localhost:8000/prod
 *                        or https://xxx.execute-api.us-east-1.amazonaws.com/prod
 */

const puppeteer = require("puppeteer");

const DASHBOARD_URL = process.env.DASHBOARD_URL || "http://localhost:8000/index.html";
const API_BASE_URL  = process.env.API_BASE_URL  || null;

function buildDashboardUrl() {
  const u = new URL(DASHBOARD_URL);
  if (API_BASE_URL) u.searchParams.set("apiBase", API_BASE_URL);
  return u.toString();
}

async function launchBrowser() {
  return puppeteer.launch({
    headless: "new",
    args: ["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"],
  });
}

/**
 * Open the dashboard and wait until the Risk Assessment tab content has loaded
 * (waits for #btnRisk to exist in the DOM, which requires the tab HTML fetch to complete).
 */
async function openDashboard(browser) {
  const page = await browser.newPage();
  page.setDefaultTimeout(30_000);
  await page.goto(buildDashboardUrl(), { waitUntil: "networkidle0" });
  await page.waitForSelector("#btnRisk", { timeout: 20_000 });
  return page;
}

/**
 * Switch the top-level cohort tab.
 * @param {import('puppeteer').Page} page
 * @param {"opioid_ed"|"non_opioid_ed"} cohort
 */
async function selectCohort(page, cohort) {
  await page.click(`button.cohort-tab-button[data-cohort="${cohort}"]`);
  // Brief settle: cohort click fires metadata load + indicator update
  await page.waitForTimeout(300);
}

/**
 * Set the age input.  The frontend derives age_band from this value.
 * @param {import('puppeteer').Page} page
 * @param {number} age
 */
async function setAge(page, age) {
  await page.$eval("#age", (el, v) => { el.value = v; el.dispatchEvent(new Event("input", { bubbles: true })); }, age);
}

/**
 * Inject code values directly into the multi-select DOM elements,
 * bypassing the search-and-click UI (which requires metadata to be loaded first).
 *
 * Creates options and marks them selected so getMultiSelectValues() returns them.
 */
async function injectCodes(page, drugs, icds, cpts) {
  await page.evaluate((d, i, c) => {
    function fill(id, values) {
      const el = document.getElementById(id);
      if (!el) return;
      while (el.options.length) el.remove(0);
      values.forEach(v => {
        const opt = new Option(v, v, true, true);
        el.add(opt);
      });
      el.dispatchEvent(new Event("change", { bubbles: true }));
    }
    fill("drugs", d);
    fill("icds",  i);
    fill("cpts",  c);
  }, drugs, icds, cpts);
}

/**
 * Click Calculate Risk Score, wait for the POST /risk response,
 * and return { status, data } — data is null on non-200.
 *
 * Resolves after the response arrives OR after a 20 s timeout (returns null).
 */
async function clickCalculate(page) {
  const [response] = await Promise.all([
    page.waitForResponse(
      resp =>
        resp.url().includes("/risk") &&
        resp.request().method() === "POST" &&
        !resp.url().includes("/comparison") &&
        !resp.url().includes("/drug_contributions") &&
        !resp.url().includes("/causal"),
      { timeout: 20_000 }
    ).catch(() => null),
    page.click("#btnRisk"),
  ]);

  if (!response) return { status: 0, data: null };

  const status = response.status();
  let data = null;
  if (status === 200) {
    try { data = await response.json(); } catch (_) {}
  }
  return { status, data };
}

/**
 * Read visible UI text from risk result elements.
 * Returns { scoreText, bandClass, binText, displayVisible }.
 */
async function readRiskDisplay(page) {
  return page.evaluate(() => {
    const display = document.getElementById("risk-display");
    const visible  = display ? display.style.display !== "none" : false;
    const scoreEl  = document.getElementById("risk-score");
    const bandEl   = document.getElementById("risk-band");
    const binEl    = document.getElementById("n-event-bin-badge");
    return {
      displayVisible: visible,
      scoreText:  scoreEl  ? scoreEl.textContent.trim()  : "",
      bandText:   bandEl   ? bandEl.textContent.trim()   : "",
      bandClass:  bandEl   ? bandEl.className            : "",
      binText:    binEl    ? binEl.textContent.trim()    : "",
      binVisible: binEl    ? binEl.style.display !== "none" : false,
    };
  });
}

module.exports = {
  DASHBOARD_URL,
  API_BASE_URL,
  buildDashboardUrl,
  launchBrowser,
  openDashboard,
  selectCohort,
  setAge,
  injectCodes,
  clickCalculate,
  readRiskDisplay,
};
