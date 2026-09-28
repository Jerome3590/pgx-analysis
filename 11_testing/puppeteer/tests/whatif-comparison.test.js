/**
 * Multi-scenario comparison + replace/swap.
 * Contract:
 *   - Save two user-defined code sets, then Compare Scenarios
 *   - POST /risk/comparison sends scenarios.length >= 2
 *   - DOM shows baseline (optional) plus both user scenario cards
 *   - Replace / swap changes the selected drug before a later Calculate Risk Score
 */

const { launchBrowser, openDashboard, selectCohort, sleep } = require("../helpers/browser");

const COHORT = "opioid_ed";
const AGE = 60;
const TEST_DRUG_A = "drug_OXYCODONE_HYDROCHLORIDE";
const TEST_DRUG_B = "drug_GABAPENTIN";

let browser, page;

beforeAll(async () => {
  browser = await launchBrowser();
  page = await openDashboard(browser);
}, 30_000);

afterAll(async () => {
  if (browser) await browser.close();
});

async function switchToTab(page, tabName) {
  await page.evaluate((t) => window.switchTab(t), tabName);
  await sleep(300);
}

async function waitForOptions(page, selectId, timeout = 12_000) {
  await page.waitForFunction(
    (id) => { const el = document.getElementById(id); return el && el.options.length > 0; },
    { timeout },
    selectId
  );
}

async function selectByValues(page, selectId, values) {
  return page.evaluate((id, vals) => {
    const sel = document.getElementById(id);
    if (!sel || !vals.length) return [];
    for (const opt of sel.options) opt.selected = false;
    const found = [];
    for (const opt of sel.options) {
      if (vals.includes(opt.value) || vals.some((v) => opt.text.includes(v))) {
        opt.selected = true;
        found.push(opt.value);
      }
    }
    sel.dispatchEvent(new Event("change"));
    return found;
  }, selectId, values);
}

async function pickSecondDrug(page, excludeValue) {
  return page.evaluate((exclude) => {
    const sel = document.getElementById("drugs");
    if (!sel) return [];
    const tokens = ["GABAPENTIN", "HYDROCODONE", "TRAMADOL", "MORPHINE", "OXYCODONE"];
    for (const opt of sel.options) {
      if (opt.value === exclude) continue;
      if (tokens.some((t) => opt.value.includes(t) || opt.text.includes(t))) {
        for (const o of sel.options) o.selected = false;
        opt.selected = true;
        sel.dispatchEvent(new Event("change"));
        return [opt.value];
      }
    }
    for (const opt of sel.options) {
      if (opt.value !== exclude) {
        for (const o of sel.options) o.selected = false;
        opt.selected = true;
        sel.dispatchEvent(new Event("change"));
        return [opt.value];
      }
    }
    return [];
  }, excludeValue);
}

test("Compare Scenarios scores two saved user-defined code sets side-by-side", async () => {
  await selectCohort(page, COHORT);
  await switchToTab(page, "risk-assessment");
  await page.evaluate((a) => {
    const el = document.getElementById("age");
    if (el) { el.value = String(a); el.dispatchEvent(new Event("input")); }
  }, AGE);
  await sleep(500);

  await switchToTab(page, "drugs");
  await waitForOptions(page, "drugs");
  const selectedA = await selectByValues(page, "drugs", [TEST_DRUG_A, "OXYCODONE"]);
  expect(selectedA.length).toBeGreaterThan(0);

  await switchToTab(page, "risk-assessment");
  await page.evaluate(() => {
    const name = document.getElementById("scenario-name-input");
    if (name) name.value = "Scenario A";
    const baseline = document.getElementById("include-baseline");
    if (baseline) baseline.checked = true;
  });
  await page.click("#btnSaveScenario");
  await sleep(300);

  await switchToTab(page, "drugs");
  const selectedB = await pickSecondDrug(page, selectedA[0]);
  expect(selectedB.length).toBeGreaterThan(0);

  await switchToTab(page, "risk-assessment");
  await page.evaluate(() => {
    const name = document.getElementById("scenario-name-input");
    if (name) name.value = "Scenario B";
  });
  await page.click("#btnSaveScenario");
  await sleep(300);

  const savedCount = await page.evaluate(() => (window._savedRiskScenarios || []).length);
  expect(savedCount).toBeGreaterThanOrEqual(2);

  let compData = null;
  let reqBody = null;
  const reqHandler = (req) => {
    if (req.url().includes("risk/comparison")) {
      reqBody = req.postData();
    }
  };
  const respHandler = async (r) => {
    if (r.url().includes("risk/comparison")) {
      compData = await r.json().catch(() => null);
    }
  };
  page.on("request", reqHandler);
  page.on("response", respHandler);

  await page.click("#btnComparison");
  await sleep(8000);
  page.off("request", reqHandler);
  page.off("response", respHandler);

  expect(compData).not.toBeNull();
  expect(typeof compData.base_risk).toBe("number");
  expect(compData.scenarios.length).toBeGreaterThanOrEqual(2);

  const parsed = reqBody ? JSON.parse(reqBody) : {};
  expect(Array.isArray(parsed.scenarios)).toBe(true);
  expect(parsed.scenarios.length).toBeGreaterThanOrEqual(2);

  const cards = await page.evaluate(() =>
    [...document.querySelectorAll(".scenario-card")].map((c) => ({
      name: c.querySelector(".scenario-name")?.textContent?.trim(),
      risk: c.querySelector(".scenario-risk")?.textContent?.trim(),
      delta: c.querySelector(".scenario-delta")?.textContent?.trim(),
    }))
  );

  expect(cards.length).toBeGreaterThanOrEqual(3);
  expect(cards[0].name).toMatch(/Baseline|Reference|Base/i);
  const names = cards.map((c) => c.name).join(" ");
  expect(names).toMatch(/Scenario A/i);
  expect(names).toMatch(/Scenario B/i);

  console.log("POST /risk/comparison body:", reqBody);
  console.log(`Base risk: ${compData.base_risk.toFixed(4)}`);
  console.log(`Scenarios:`, JSON.stringify(compData.scenarios));
  console.log(`DOM cards:`, JSON.stringify(cards));
}, 45_000);

test("Replace / swap exchanges a selected drug in the live selection", async () => {
  await selectCohort(page, COHORT);
  await switchToTab(page, "risk-assessment");
  await page.evaluate((a) => {
    const el = document.getElementById("age");
    if (el) { el.value = String(a); el.dispatchEvent(new Event("input")); }
  }, AGE);
  await sleep(400);

  await switchToTab(page, "drugs");
  await waitForOptions(page, "drugs");
  const selected = await selectByValues(page, "drugs", [TEST_DRUG_A, "OXYCODONE"]);
  expect(selected.length).toBeGreaterThan(0);

  await switchToTab(page, "risk-assessment");
  await page.waitForSelector("#replace-from", { timeout: 8_000 });
  await sleep(200);

  const swapped = await page.evaluate(() => {
    const typeEl = document.getElementById("replace-type");
    const fromEl = document.getElementById("replace-from");
    const toEl = document.getElementById("replace-to");
    if (!typeEl || !fromEl || !toEl) return { ok: false, error: "replace controls missing" };
    typeEl.value = "drug";
    typeEl.dispatchEvent(new Event("change"));
    if (!fromEl.options.length || !toEl.options.length) {
      return { ok: false, error: "empty replace lists", from: fromEl.options.length, to: toEl.options.length };
    }
    const fromCode = fromEl.value;
    const toCode = toEl.value;
    const ok = window.replacePatientCode("drug", fromCode, toCode);
    const drugs = [...document.getElementById("drugs").options].filter((o) => o.selected).map((o) => o.value);
    return { ok, fromCode, toCode, drugs };
  });

  expect(swapped.ok).toBe(true);
  expect(swapped.toCode).toBeTruthy();
  expect(swapped.drugs).toContain(swapped.toCode);
  expect(swapped.drugs).not.toContain(swapped.fromCode);

  const status = await page.evaluate(() => document.getElementById("status")?.textContent || "");
  expect(status).toMatch(/Replaced|Calculate Risk Score/i);
}, 30_000);
