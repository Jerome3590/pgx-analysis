"use strict";

/**
 * One-pass live screenshot tour for DASHBOARD_USE_CASES.md (UC01–UC09).
 * Writes PNGs into Google Drive sync + repo mirror. Does not invent UI.
 */
const fs = require("fs");
const path = require("path");
const { launchBrowser, openDashboard, selectCohort, setAge, sleep } = require("./helpers/browser");

const DRIVE_ROOT = process.env.PGX_UC_DRIVE_ROOT
  || "G:\\My Drive\\PGx_Dashboard_Use_Cases";
const REPO_ROOT = path.resolve(__dirname, "..", "..", "10_risk_dashboard", "docs", "use_case_training");

const DIRS = {
  UC01: "UC01_cohort_risk",
  UC02: "UC02_scenario_analysis",
  UC03: "UC03_density_bin_exploration",
  UC04: "UC04_feature_importance",
  UC05: "UC05_pattern_process",
  UC06: "UC06_claims_pgx_card",
  UC07: "UC07_personalized_pgx_card",
  UC08: "UC08_cohort_vs_card",
  UC09: "UC09_documentation",
};

function ensureShotDirs(ucKey) {
  const names = [path.join(DRIVE_ROOT, DIRS[ucKey], "screenshots"), path.join(REPO_ROOT, DIRS[ucKey], "screenshots")];
  for (const dir of names) fs.mkdirSync(dir, { recursive: true });
  return names;
}

async function shot(page, ucKey, filename) {
  const dirs = ensureShotDirs(ucKey);
  await sleep(400);
  const buf = await page.screenshot({ fullPage: true, type: "png" });
  for (const dir of dirs) {
    const dest = path.join(dir, filename);
    fs.writeFileSync(dest, buf);
    console.log("saved", dest);
  }
}

async function switchTab(page, tab) {
  await page.evaluate((name) => {
    if (typeof window.switchTab === "function") window.switchTab(name);
  }, tab);
  await sleep(900);
}

function sel(id) {
  return id.startsWith("#") ? id : `#${id}`;
}

async function selectFirstMatching(page, selectId, searchId, query) {
  try {
  if (searchId) {
    const sid = sel(searchId);
    await page.waitForSelector(sid, { timeout: 10000 });
    await page.click(sid, { clickCount: 3 });
    await page.keyboard.press("Backspace");
    await page.type(sid, query, { delay: 20 });
    await sleep(400);
  }
    const picked = await page.evaluate((sid, q) => {
      const box = document.getElementById(sid);
      if (!box) return null;
      const needle = String(q).toLowerCase();
      let opt = [...box.options].find((o) => (o.textContent || "").toLowerCase().includes(needle));
      if (!opt) opt = [...box.options].find((o) => o.value);
      if (!opt) return null;
      [...box.options].forEach((o) => { o.selected = false; });
      opt.selected = true;
      box.dispatchEvent(new Event("change", { bubbles: true }));
      return opt.textContent.trim();
    }, selectId, query);
    return picked;
  } catch (err) {
    console.warn("selectFirstMatching failed", selectId, err.message);
    return null;
  }
}

async function clickIf(page, selector) {
  const el = await page.$(selector);
  if (!el) return false;
  await el.click();
  return true;
}

async function waitForAny(page, selectors, timeout = 20000) {
  const start = Date.now();
  while (Date.now() - start < timeout) {
    for (const sel of selectors) {
      const handle = await page.$(sel);
      if (handle) {
        const vis = await page.evaluate((s) => {
          const el = document.querySelector(s);
          if (!el) return false;
          const st = window.getComputedStyle(el);
          return st.display !== "none" && st.visibility !== "hidden";
        }, sel);
        if (vis) return sel;
      }
    }
    await sleep(250);
  }
  return null;
}

async function main() {
  const browser = await launchBrowser();
  const page = await openDashboard(browser);
  await page.setViewport({ width: 1440, height: 900 });

  // ----- UC01 cohort risk -----
  await selectCohort(page, "opioid_ed");
  await switchTab(page, "risk-assessment");
  await setAge(page, 45);
  await shot(page, "UC01", "01-cohort-and-age.png");

  await switchTab(page, "drugs");
  await page.click("#drug-search", { clickCount: 3 });
  await page.keyboard.press("Backspace");
  await page.type("#drug-search", "oxy", { delay: 20 });
  await sleep(400);
  await page.evaluate(() => {
    const sel = document.getElementById("drugs");
    if (!sel) return;
    const needles = ["oxy", "codeine", "hydrocodone"];
    [...sel.options].forEach((o) => {
      const t = (o.textContent || "").toLowerCase();
      o.selected = needles.some((n) => t.includes(n));
    });
    sel.dispatchEvent(new Event("change", { bubbles: true }));
  });
  await shot(page, "UC01", "02-drugs-select.png");

  await switchTab(page, "icd-codes");
  await selectFirstMatching(page, "icds", "icd-search", "F11");
  await switchTab(page, "cpt-codes");
  await selectFirstMatching(page, "cpts", "cpt-search", "992");
  await shot(page, "UC01", "03-icd-cpt-codes.png");

  await switchTab(page, "risk-assessment");
  await Promise.all([
    page.waitForResponse((r) => r.url().includes("/risk") && r.request().method() === "POST", { timeout: 25000 }).catch(() => null),
    clickIf(page, "#btnRisk"),
  ]);
  await waitForAny(page, ["#risk-display", "#risk-score", ".risk-score"], 20000);
  await shot(page, "UC01", "04-calculate-risk.png");

  await page.evaluate(() => {
    const name = document.getElementById("scenario-name-input") || document.getElementById("scenario-name");
    if (name) name.value = "Baseline set A";
  });
  await clickIf(page, "#btnSaveScenario");
  await sleep(300);
  await clickIf(page, "#btnReplaceCode");
  await shot(page, "UC01", "05-replace-swap-code.png");

  await switchTab(page, "drugs");
  await selectFirstMatching(page, "drugs", "drug-search", "hydrocodone");
  await switchTab(page, "risk-assessment");
  await page.evaluate(() => {
    const name = document.getElementById("scenario-name-input") || document.getElementById("scenario-name");
    if (name) name.value = "Swap set B";
  });
  await clickIf(page, "#btnSaveScenario");
  await Promise.all([
    page.waitForResponse((r) => r.url().includes("/risk") && r.request().method() === "POST", { timeout: 25000 }).catch(() => null),
    clickIf(page, "#btnComparison"),
  ]);
  await sleep(1500);
  await shot(page, "UC01", "06-save-and-compare.png");

  await Promise.all([
    page.waitForResponse((r) => r.url().includes("/drug_contributions") && r.request().method() === "POST", { timeout: 25000 }).catch(() => null),
    clickIf(page, "#btnContributions"),
  ]);
  await sleep(1200);
  await shot(page, "UC01", "07-drug-contributions.png");

  // ----- UC02 scenario -----
  await shot(page, "UC02", "01-risk-context.png");
  await switchTab(page, "scenario-analysis");
  await shot(page, "UC02", "02-scenario-tab.png");
  await Promise.all([
    page.waitForResponse((r) => /scenario|causal|ffa|shap/i.test(r.url()), { timeout: 25000 }).catch(() => null),
    clickIf(page, "#btnLoadScenario"),
  ]);
  await sleep(2500);
  await shot(page, "UC02", "03-load-scenario-analysis.png");
  await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight * 0.45));
  await shot(page, "UC02", "04-ffa-shap-charts.png");
  await clickIf(page, "#btnClearScenarioFilters");
  await sleep(1200);
  await shot(page, "UC02", "05-clear-filters.png");

  // ----- UC03 density bin -----
  await switchTab(page, "risk-assessment");
  await shot(page, "UC03", "01-event-density-badge.png");

  await switchTab(page, "bupar-visualizations");
  await Promise.all([
    page.waitForResponse((r) => /bupar/i.test(r.url()), { timeout: 25000 }).catch(() => null),
    clickIf(page, "#btnLoadBupaR"),
  ]);
  await sleep(2500);
  await shot(page, "UC03", "02-bupar-loaded.png");
  await shot(page, "UC05", "01-bupar.png");

  await switchTab(page, "dtw-visualizations");
  await Promise.all([
    page.waitForResponse((r) => /dtw/i.test(r.url()), { timeout: 25000 }).catch(() => null),
    clickIf(page, "#btnLoadDTW"),
  ]);
  await sleep(2500);
  await shot(page, "UC03", "03-dtw-loaded.png");
  await shot(page, "UC05", "02-dtw.png");

  await switchTab(page, "fpgrowth-visualizations");
  await Promise.all([
    page.waitForResponse((r) => /fpgrowth/i.test(r.url()), { timeout: 25000 }).catch(() => null),
    clickIf(page, "#btnLoadFPGrowth"),
  ]);
  await sleep(2500);
  await shot(page, "UC03", "04-fpgrowth-loaded.png");
  await shot(page, "UC05", "03-fpgrowth.png");

  await switchTab(page, "cohort-pgx-visualizations");
  await Promise.all([
    page.waitForResponse((r) => /cohort_pgx|cohort-pgx/i.test(r.url()), { timeout: 25000 }).catch(() => null),
    clickIf(page, "#btnLoadCohortPgx"),
  ]);
  await sleep(2500);
  await shot(page, "UC03", "05-pgx-cohort-loaded.png");
  await shot(page, "UC08", "01-pgx-cohort-network.png");

  await switchTab(page, "pgx-card");
  await Promise.all([
    page.waitForResponse((r) => /cohort|pgx/i.test(r.url()), { timeout: 20000 }).catch(() => null),
    clickIf(page, "#btnLoadPgxCardProfile"),
  ]);
  await sleep(2000);
  await shot(page, "UC03", "06-pgx-card-cohort-profile.png");
  await shot(page, "UC06", "01-view-pgx-card.png");
  await shot(page, "UC06", "02-load-cohort-profile.png");
  await page.evaluate(() => window.scrollTo(0, 400));
  await shot(page, "UC06", "03-radar-and-genes.png");

  // ----- UC04 feature importance -----
  await switchTab(page, "feature-importance-visualizations");
  await shot(page, "UC04", "01-feature-importance-tab.png");
  await page.evaluate(() => {
    const view = document.getElementById("fi-view") || document.getElementById("fi-cohort") || document.querySelector("#feature-importance-visualizations select");
    if (view) {
      view.value = view.value;
      view.dispatchEvent(new Event("change", { bubbles: true }));
    }
  });
  await shot(page, "UC04", "02-view-and-top-n.png");
  await Promise.all([
    page.waitForResponse((r) => /feature_importance|feature-importance/i.test(r.url()), { timeout: 25000 }).catch(() => null),
    clickIf(page, "#btnLoadFeatureImportance"),
  ]);
  await sleep(2500);
  await shot(page, "UC04", "03-heatmap-loaded.png");

  // ----- UC05 drug networks -----
  await switchTab(page, "cytoscape-visualizations");
  await Promise.all([
    page.waitForResponse((r) => /cytoscape|network|fpgrowth/i.test(r.url()), { timeout: 25000 }).catch(() => null),
    clickIf(page, "#btnLoadCytoscape"),
  ]);
  await sleep(3000);
  await shot(page, "UC05", "04-drug-networks.png");

  // ----- UC07 personalized card -----
  await switchTab(page, "pgx-card");
  await page.evaluate(() => {
    const pid = document.getElementById("patient-id");
    if (pid) pid.value = "training-demo";
    const snp = document.getElementById("snp-input");
    if (snp) snp.value = "CYP2D6,*1,*4\nCYP2C19,*1,*2";
  });
  await shot(page, "UC07", "01-gene-data-entry.png");
  await page.click("#pgx-drug-search", { clickCount: 3 }).catch(() => {});
  await page.keyboard.press("Backspace").catch(() => {});
  await page.type("#pgx-drug-search", "clopidogrel", { delay: 20 }).catch(() => {});
  await sleep(400);
  await page.$eval(".pgx-suggest-item", (el) => el.click()).catch(() => {});
  await page.evaluate(() => {
    const scope = document.getElementById("pgx-drug-scope");
    if (scope) scope.value = "SELECTED";
  });
  await shot(page, "UC07", "02-apcd-meds-and-scope.png");
  await Promise.all([
    page.waitForResponse((r) => r.url().includes("/pgx/card") && r.request().method() === "POST", { timeout: 25000 }).catch(() => null),
    clickIf(page, "#btnGenerateCard"),
  ]);
  await sleep(2000);
  await shot(page, "UC07", "03-generate-pgx-results.png");
  await page.evaluate(() => {
    const el = document.getElementById("pgx-card-display");
    if (el) el.scrollIntoView();
  });
  await shot(page, "UC07", "04-action-queue-matrix.png");
  await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
  await shot(page, "UC07", "05-exports.png");
  // Distinct clips (do not rely on this fullPage tour):
  //   recapture_uc01_to_uc06.js, recapture_uc07.js, recapture_uc08.js, recapture_uc09.js

  await browser.close();
  console.log("TOUR_DONE");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
