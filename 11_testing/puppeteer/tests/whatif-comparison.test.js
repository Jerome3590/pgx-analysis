/**
 * What-if comparison panel smoke test.
 * Verifies that clicking "Compare Scenarios" with codes selected:
 *   1. POSTs to /risk/comparison with base={no codes} and scenarios=[{current selection}]
 *   2. Gets a valid 200 response with base_risk and one scenario
 *   3. Renders two scenario-cards in the DOM (Base + Current Selection)
 *   4. Current Selection card shows higher risk than Base (for opioid_ed drugs)
 */

const { launchBrowser, openDashboard, selectCohort, sleep } = require("../helpers/browser");

const COHORT   = "opioid_ed";
const AGE      = 60;
const AGE_BAND = "55-64";
const TEST_DRUG = "drug_OXYCODONE_HYDROCHLORIDE";

let browser, page;

beforeAll(async () => {
  browser = await launchBrowser();
  page    = await openDashboard(browser);
}, 30_000);

afterAll(async () => {
  if (browser) await browser.close();
});

async function switchToTab(page, tabName) {
  await page.evaluate(t => window.switchTab(t), tabName);
  await sleep(300);
}

async function waitForOptions(page, selectId, timeout = 12_000) {
  await page.waitForFunction(
    id => { const el = document.getElementById(id); return el && el.options.length > 0; },
    { timeout }, selectId
  );
}

async function selectByValues(page, selectId, values) {
  return page.evaluate((id, vals) => {
    const sel = document.getElementById(id);
    if (!sel || !vals.length) return [];
    const found = [];
    for (const opt of sel.options) {
      if (vals.includes(opt.value) || vals.some(v => opt.text.includes(v))) {
        opt.selected = true;
        found.push(opt.value);
      }
    }
    sel.dispatchEvent(new Event("change"));
    return found;
  }, selectId, values);
}

test("Compare Scenarios renders Base + Current Selection cards with valid delta", async () => {
  // ── 1. Select cohort + age ────────────────────────────────────────────────
  await selectCohort(page, COHORT);
  await switchToTab(page, "risk-assessment");
  await page.evaluate(a => {
    const el = document.getElementById("age");
    if (el) { el.value = String(a); el.dispatchEvent(new Event("input")); }
  }, AGE);
  await sleep(500);

  // ── 2. Switch to Drugs tab, wait for options, select drug ─────────────────
  await switchToTab(page, "drugs");
  await waitForOptions(page, "drugs");
  const selected = await selectByValues(page, "drugs", [TEST_DRUG, "OXYCODONE"]);
  expect(selected.length).toBeGreaterThan(0);

  // ── 3. Switch back to Risk Assessment tab ────────────────────────────────
  await switchToTab(page, "risk-assessment");
  await sleep(400);

  // Debug: read drugs select state just before clicking compare
  const drugsBeforeClick = await page.evaluate(() => {
    const sel = document.getElementById("drugs");
    return sel ? [...sel.options].filter(o => o.selected).map(o => o.value) : [];
  });
  console.log("Drugs selected before btnComparison click:", drugsBeforeClick);

  // ── 5. Intercept /risk/comparison request + response ─────────────────────
  let compData = null;
  let reqBody  = null;
  page.on("request", req => {
    if (req.url().includes("risk/comparison")) {
      reqBody = req.postData();
      console.log("POST /risk/comparison body:", reqBody);
    }
  });
  const respHandler = async r => {
    if (r.url().includes("risk/comparison")) {
      compData = await r.json().catch(() => null);
    }
  };
  page.on("response", respHandler);

  // ── 6. Click Compare Scenarios ────────────────────────────────────────────
  await page.click("#btnComparison");
  await sleep(5000);
  page.off("response", respHandler);

  // ── 7. Assert API response ────────────────────────────────────────────────
  expect(compData).not.toBeNull();
  expect(typeof compData.base_risk).toBe("number");
  expect(compData.scenarios).toHaveLength(1);
  expect(compData.scenarios[0].risk_score).toBeGreaterThan(compData.base_risk);
  expect(compData.scenarios[0].delta).toBeGreaterThan(0);

  // ── 8. Assert DOM cards ───────────────────────────────────────────────────
  const cards = await page.evaluate(() =>
    [...document.querySelectorAll(".scenario-card")].map(c => ({
      name:  c.querySelector(".scenario-name")?.textContent?.trim(),
      risk:  c.querySelector(".scenario-risk")?.textContent?.trim(),
      delta: c.querySelector(".scenario-delta")?.textContent?.trim(),
    }))
  );

  expect(cards).toHaveLength(2);
  expect(cards[0].name).toBe("Base");
  expect(cards[1].name).toBe("Current Selection");
  // Delta string should contain "+" (risk increased)
  expect(cards[1].delta).toMatch(/\+/);

  console.log(`Base risk: ${compData.base_risk.toFixed(4)}`);
  console.log(`Current Selection risk: ${compData.scenarios[0].risk_score.toFixed(4)}  delta=${compData.scenarios[0].delta.toFixed(4)}`);
  console.log(`DOM cards:`, JSON.stringify(cards));
}, 30_000);
