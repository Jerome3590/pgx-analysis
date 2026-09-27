"use strict";

/**
 * Recapture UC01–UC06 as distinct live clips (not full-page duplicates).
 * Writes Drive pack + repo mirror. Does not invent UI.
 */
const fs = require("fs");
const path = require("path");
const puppeteer = require("puppeteer");

const DASHBOARD = process.env.DASHBOARD_URL || "https://pgx.jerome-dixon.io/?v=20260927-unphased";
const DRIVE = process.env.PGX_UC_DRIVE_ROOT || "G:\\My Drive\\PGx_Dashboard_Use_Cases";
const REPO = path.resolve(__dirname, "..", "..", "10_risk_dashboard", "docs", "use_case_training");

const DIRS = {
  UC01: "UC01_cohort_risk",
  UC02: "UC02_scenario_analysis",
  UC03: "UC03_density_bin_exploration",
  UC04: "UC04_feature_importance",
  UC05: "UC05_pattern_process",
  UC06: "UC06_claims_pgx_card",
};

function dests(uc) {
  return [
    path.join(DRIVE, DIRS[uc], "screenshots"),
    path.join(REPO, DIRS[uc], "screenshots"),
  ];
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function save(page, uc, name, selector) {
  const dirs = dests(uc);
  dirs.forEach((dir) => fs.mkdirSync(dir, { recursive: true }));
  const el = selector ? await page.$(selector) : null;
  if (selector && !el) throw new Error("missing selector " + selector + " for " + uc + " " + name);
  const buf = el
    ? await el.screenshot({ type: "png" })
    : await page.screenshot({ type: "png", fullPage: false });
  for (const dir of dirs) {
    const file = path.join(dir, name);
    fs.writeFileSync(file, buf);
    console.log("saved", uc, name, buf.length);
  }
}

async function switchTab(page, tab) {
  await page.evaluate((name) => {
    if (typeof window.switchTab === "function") window.switchTab(name);
  }, tab);
  await sleep(700);
}

async function selectMatching(page, selectId, searchId, pred) {
  if (searchId) {
    const box = await page.$(searchId);
    if (box) {
      await box.click({ clickCount: 3 });
      await page.keyboard.press("Backspace");
      await box.type(pred.query || "", { delay: 20 });
      await sleep(400);
    }
  }
  const picked = await page.evaluate((sid, needle, exactish) => {
    const sel = document.getElementById(sid);
    if (!sel) return null;
    const n = String(needle).toLowerCase();
    const opt = [...sel.options].find((o) => {
      const t = (o.textContent || "").toLowerCase();
      if (exactish) return t.includes(n) && !/acetaminophen|hydroxyzine/.test(t);
      return t.includes(n);
    });
    if (!opt) return null;
    [...sel.options].forEach((o) => { o.selected = false; });
    opt.selected = true;
    sel.dispatchEvent(new Event("change", { bubbles: true }));
    return opt.textContent.trim();
  }, selectId, pred.query, !!pred.exactish);
  console.log("picked", selectId, picked);
  return picked;
}

async function addDrugOption(page, query) {
  const box = await page.$("#drug-search");
  await box.click({ clickCount: 3 });
  await page.keyboard.press("Backspace");
  await box.type(query, { delay: 20 });
  await sleep(400);
  const picked = await page.evaluate((q) => {
    const sel = document.getElementById("drugs");
    if (!sel) return null;
    const n = q.toLowerCase();
    const opt = [...sel.options].find((o) => {
      const t = (o.textContent || "").toLowerCase();
      return t.includes(n) && !/acetaminophen|hydroxyzine/.test(t);
    });
    if (!opt) return null;
    opt.selected = true;
    sel.dispatchEvent(new Event("change", { bubbles: true }));
    return opt.textContent.trim();
  }, query);
  console.log("drug", picked);
  return picked;
}

(async () => {
  const browser = await puppeteer.launch({
    headless: "new",
    defaultViewport: { width: 1440, height: 1100 },
    args: ["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"],
  });
  const page = await browser.newPage();
  page.setDefaultTimeout(45000);
  await page.goto(DASHBOARD, { waitUntil: "networkidle0" });
  await page.waitForSelector("#btnRisk");
  await page.evaluate(() => {
    const btn = document.querySelector("button.cohort-tab-button[data-cohort='opioid_ed']");
    if (btn) btn.click();
    const age = document.getElementById("age");
    if (age) {
      age.value = "45";
      age.dispatchEvent(new Event("input", { bubbles: true }));
    }
  });
  await sleep(800);

  await switchTab(page, "risk-assessment");
  await page.evaluate(() => {
    if (document.getElementById("uc01-setup-clip")) return;
    const tab = document.getElementById("risk-assessment-tab");
    const clip = document.createElement("div");
    clip.id = "uc01-setup-clip";
    const sub = tab.querySelector(".subtitle");
    const controls = tab.querySelector(".controls");
    if (sub) clip.appendChild(sub);
    if (controls) clip.appendChild(controls);
    tab.insertBefore(clip, tab.firstChild);
  });
  await save(page, "UC01", "01-cohort-and-age.png", "#uc01-setup-clip");

  await switchTab(page, "drugs");
  await page.waitForSelector("#drug-search");
  await addDrugOption(page, "oxycodone");
  await addDrugOption(page, "hydrocodone");
  await addDrugOption(page, "gabapentin");
  await sleep(300);
  await save(page, "UC01", "02-drugs-select.png", "#drugs-tab .tab-content-inner");

  await switchTab(page, "icd-codes");
  await selectMatching(page, "icds", "#icd-search", { query: "F11" });
  await sleep(200);
  await switchTab(page, "cpt-codes");
  await selectMatching(page, "cpts", "#cpt-search", { query: "99214" });
  await sleep(200);
  await page.evaluate(() => {
    if (document.getElementById("uc01-codes-clip")) return;
    const clip = document.createElement("div");
    clip.id = "uc01-codes-clip";
    const icd = document.querySelector("#icd-codes-tab .tab-content-inner");
    const cpt = document.querySelector("#cpt-codes-tab .tab-content-inner");
    const host = document.getElementById("cpt-codes-tab");
    if (!host) return;
    host.insertBefore(clip, host.firstChild);
    if (icd) clip.appendChild(icd.cloneNode(true));
    if (cpt) clip.appendChild(cpt.cloneNode(true));
  });
  await save(page, "UC01", "03-icd-cpt-codes.png", "#uc01-codes-clip");

  await switchTab(page, "risk-assessment");
  await Promise.all([
    page.waitForResponse((r) => r.url().includes("/risk") && r.request().method() === "POST" && !/comparison|drug_contributions/.test(r.url()), { timeout: 30000 }).catch(() => null),
    page.click("#btnRisk"),
  ]);
  await page.waitForFunction(() => {
    const el = document.getElementById("risk-display");
    return el && getComputedStyle(el).display !== "none";
  }, { timeout: 20000 });
  await sleep(800);
  await page.$eval("#risk-display", (el) => el.scrollIntoView({ block: "start" }));
  await save(page, "UC01", "04-calculate-risk.png", "#risk-display");
  await save(page, "UC02", "01-risk-context.png", "#risk-display");

  await page.evaluate(() => {
    if (document.getElementById("uc03-badge-clip")) return;
    const clip = document.createElement("div");
    clip.id = "uc03-badge-clip";
    const table = document.getElementById("density-table-wrap");
    const badge = document.getElementById("n-event-bin-badge");
    const band = document.getElementById("risk-band");
    const display = document.getElementById("risk-display");
    if (!display) return;
    display.parentNode.insertBefore(clip, display);
    if (table) clip.appendChild(table);
    if (band) clip.appendChild(band.cloneNode(true));
    if (badge) clip.appendChild(badge.cloneNode(true));
  });
  await save(page, "UC03", "01-event-density-badge.png", "#uc03-badge-clip");

  await page.evaluate(() => {
    const name = document.getElementById("scenario-name-input");
    if (name) name.value = "Baseline set A";
  });
  await page.click("#btnSaveScenario");
  await sleep(300);

  await page.evaluate(() => {
    const type = document.getElementById("replace-type");
    if (type) {
      type.value = "drug";
      type.dispatchEvent(new Event("change", { bubbles: true }));
    }
  });
  await sleep(200);
  await page.evaluate(() => {
    const from = document.getElementById("replace-from");
    if (!from || !from.options.length) return;
    const oxy = [...from.options].find((o) => /oxycodone/i.test(o.textContent || ""));
    if (oxy) from.value = oxy.value;
    from.dispatchEvent(new Event("change", { bubbles: true }));
  });
  const toSearch = await page.$("#replace-to-search");
  if (toSearch) {
    await toSearch.click({ clickCount: 3 });
    await page.keyboard.press("Backspace");
    await toSearch.type("buprenorphine", { delay: 20 });
    await sleep(400);
  }
  await page.evaluate(() => {
    const to = document.getElementById("replace-to");
    if (!to) return;
    const opt = [...to.options].find((o) => /buprenorphine/i.test(o.textContent || o.value || ""));
    if (opt) to.value = opt.value;
  });
  await page.click("#btnReplaceCode");
  await sleep(400);
  await Promise.all([
    page.waitForResponse((r) => r.url().includes("/risk") && r.request().method() === "POST" && !/comparison|drug_contributions/.test(r.url()), { timeout: 30000 }).catch(() => null),
    page.click("#btnRisk"),
  ]);
  await sleep(800);
  await page.evaluate(() => {
    if (document.getElementById("uc01-replace-clip")) return;
    const clip = document.createElement("div");
    clip.id = "uc01-replace-clip";
    const panel = document.getElementById("replace-panel");
    const saved = document.getElementById("saved-scenarios-panel");
    const status = document.getElementById("status");
    const host = document.getElementById("scenario-modeling");
    if (!host) return;
    host.parentNode.insertBefore(clip, host);
    if (panel) clip.appendChild(panel);
    if (saved) clip.appendChild(saved);
    if (status) clip.appendChild(status.cloneNode(true));
  });
  await save(page, "UC01", "05-replace-swap-code.png", "#uc01-replace-clip");

  await page.evaluate(() => {
    const name = document.getElementById("scenario-name-input");
    if (name) name.value = "Swap set B";
  });
  await page.click("#btnSaveScenario");
  await sleep(300);
  await Promise.all([
    page.waitForResponse((r) => r.url().includes("/risk") && r.request().method() === "POST", { timeout: 30000 }).catch(() => null),
    page.click("#btnComparison"),
  ]);
  await page.waitForFunction(() => {
    const el = document.getElementById("comparison-scenarios");
    return el && el.children.length > 0;
  }, { timeout: 15000 }).catch(() => null);
  await sleep(600);
  await page.$eval("#comparison-mode", (el) => el.scrollIntoView({ block: "start" }));
  await save(page, "UC01", "06-save-and-compare.png", "#comparison-mode");

  await Promise.all([
    page.waitForResponse((r) => r.url().includes("/drug_contributions"), { timeout: 30000 }).catch(() => null),
    page.click("#btnContributions"),
  ]);
  await page.waitForFunction(() => {
    const el = document.getElementById("contributions-table");
    return el && el.innerHTML.trim().length > 20;
  }, { timeout: 15000 }).catch(() => null);
  await sleep(600);
  await page.$eval("#contributions-mode", (el) => el.scrollIntoView({ block: "start" }));
  await save(page, "UC01", "07-drug-contributions.png", "#contributions-mode");

  await page.$eval("#pgx-action-link", (el) => el.scrollIntoView({ block: "center" })).catch(() => {});
  await save(page, "UC06", "01-view-pgx-card.png", "#pgx-action-link");

  await switchTab(page, "scenario-analysis");
  await page.waitForSelector("#btnLoadScenario");
  await save(page, "UC02", "02-scenario-tab.png", "#scenario-analysis-tab .controls");
  await Promise.all([
    page.waitForResponse((r) => /scenario|causal|ffa|shap/i.test(r.url()), { timeout: 30000 }).catch(() => null),
    page.click("#btnLoadScenario"),
  ]);
  await sleep(2000);
  await page.evaluate(() => {
    if (document.getElementById("uc02-loaded-clip")) return;
    const clip = document.createElement("div");
    clip.id = "uc02-loaded-clip";
    const status = document.getElementById("scenario-status");
    const panels = document.querySelector("#scenario-analysis-tab .panels");
    const host = document.getElementById("scenario-analysis-tab");
    host.insertBefore(clip, panels || host.lastChild);
    if (status) clip.appendChild(status);
    if (panels) clip.appendChild(panels);
  });
  await save(page, "UC02", "03-load-scenario-analysis.png", "#uc02-loaded-clip");
  await save(page, "UC02", "04-ffa-shap-charts.png", "#scenario-analysis-tab .panels");

  await Promise.all([
    page.waitForResponse((r) => /scenario|causal|ffa|shap/i.test(r.url()), { timeout: 30000 }).catch(() => null),
    page.click("#btnClearScenarioFilters"),
  ]);
  await sleep(2000);
  await save(page, "UC02", "05-clear-filters.png", "#uc02-loaded-clip");

  async function loadViz(tab, btn, waitRe) {
    await switchTab(page, tab);
    await page.waitForSelector(btn);
    await Promise.all([
      page.waitForResponse((r) => waitRe.test(r.url()), { timeout: 30000 }).catch(() => null),
      page.click(btn),
    ]);
    await sleep(2500);
  }

  await loadViz("bupar-visualizations", "#btnLoadBupaR", /bupar/i);
  await page.evaluate(() => {
    if (document.getElementById("uc-bupar-clip")) return;
    const tab = document.getElementById("bupar-visualizations-tab");
    const clip = document.createElement("div");
    clip.id = "uc-bupar-clip";
    const controls = tab.querySelector(".controls");
    const status = document.getElementById("bupar-status");
    const seq = document.getElementById("bupar-sequence-image");
    tab.insertBefore(clip, tab.firstChild);
    if (controls) clip.appendChild(controls);
    if (status) clip.appendChild(status);
    if (seq) clip.appendChild(seq);
  });
  await save(page, "UC03", "02-bupar-loaded.png", "#uc-bupar-clip");
  await save(page, "UC05", "01-bupar.png", "#uc-bupar-clip");

  await loadViz("dtw-visualizations", "#btnLoadDTW", /dtw/i);
  await page.evaluate(() => {
    if (document.getElementById("uc-dtw-clip")) return;
    const tab = document.getElementById("dtw-visualizations-tab");
    const clip = document.createElement("div");
    clip.id = "uc-dtw-clip";
    const controls = tab.querySelector(".controls");
    const status = document.getElementById("dtw-status");
    const img = document.getElementById("dtw-overview-image");
    tab.insertBefore(clip, tab.firstChild);
    if (controls) clip.appendChild(controls);
    if (status) clip.appendChild(status);
    if (img) clip.appendChild(img);
  });
  await save(page, "UC03", "03-dtw-loaded.png", "#uc-dtw-clip");
  await save(page, "UC05", "02-dtw.png", "#uc-dtw-clip");

  await loadViz("fpgrowth-visualizations", "#btnLoadFPGrowth", /fpgrowth/i);
  await page.evaluate(() => {
    if (document.getElementById("uc-fp-clip")) return;
    const tab = document.getElementById("fpgrowth-visualizations-tab");
    const clip = document.createElement("div");
    clip.id = "uc-fp-clip";
    const controls = tab.querySelector(".controls");
    const status = document.getElementById("fpgrowth-status");
    const img = document.getElementById("fpgrowth-support-image");
    tab.insertBefore(clip, tab.firstChild);
    if (controls) clip.appendChild(controls);
    if (status) clip.appendChild(status);
    if (img) clip.appendChild(img);
  });
  await save(page, "UC03", "04-fpgrowth-loaded.png", "#uc-fp-clip");
  await save(page, "UC05", "03-fpgrowth.png", "#uc-fp-clip");

  await loadViz("cohort-pgx-visualizations", "#btnLoadCohortPgx", /cohort_pgx|network_topology/i);
  await page.waitForFunction(() => {
    const iframe = document.getElementById("cohort-pgx-iframe");
    return iframe && iframe.src && iframe.src !== "about:blank";
  }, { timeout: 20000 }).catch(() => null);
  await sleep(1500);
  await page.evaluate(() => {
    if (document.getElementById("uc-cohort-clip")) return;
    const tab = document.getElementById("cohort-pgx-visualizations-tab");
    const clip = document.createElement("div");
    clip.id = "uc-cohort-clip";
    const controls = tab.querySelector(".controls");
    const status = document.getElementById("cohort-pgx-status");
    const panel = [...tab.querySelectorAll(".panel")].find((p) =>
      /Gene–Drug–Phenotype Network Topology/i.test((p.querySelector("h2") || {}).textContent || "")
    );
    tab.insertBefore(clip, tab.firstChild);
    if (controls) clip.appendChild(controls);
    if (status) clip.appendChild(status);
    if (panel) clip.appendChild(panel);
  });
  await save(page, "UC03", "05-pgx-cohort-loaded.png", "#uc-cohort-clip");

  await switchTab(page, "pgx-card");
  await page.waitForSelector("#btnLoadPgxCardProfile");
  await page.evaluate(() => {
    const cohort = document.getElementById("pgx-card-cohort");
    const age = document.getElementById("pgx-card-age-band");
    if (cohort) {
      cohort.value = "opioid_ed";
      cohort.dispatchEvent(new Event("change", { bubbles: true }));
    }
    if (age) {
      age.value = "45-54";
      age.dispatchEvent(new Event("change", { bubbles: true }));
    }
  });
  await page.evaluate(() => {
    if (document.getElementById("uc06-form-clip")) return;
    const clip = document.createElement("div");
    clip.id = "uc06-form-clip";
    const tab = document.getElementById("pgx-card-tab");
    const sub = tab.querySelector(".subtitle");
    const inputs = tab.querySelector(".pgx-input-section");
    const status = document.getElementById("pgx-card-status");
    tab.insertBefore(clip, tab.firstChild);
    if (sub) clip.appendChild(sub);
    if (inputs) clip.appendChild(inputs);
    if (status) clip.appendChild(status);
  });
  await save(page, "UC06", "02-load-cohort-profile.png", "#uc06-form-clip");

  await Promise.all([
    page.waitForResponse((r) => /pubmed_citations|pgx_radar|cohort_pgx/i.test(r.url()), { timeout: 30000 }).catch(() => null),
    page.click("#btnLoadPgxCardProfile"),
  ]);
  await page.waitForFunction(() => {
    const sec = document.getElementById("pgx-cohort-profile-section");
    return sec && getComputedStyle(sec).display !== "none";
  }, { timeout: 25000 });
  await sleep(1200);
  await save(page, "UC03", "06-pgx-card-cohort-profile.png", "#pgx-cohort-profile-section");
  await save(page, "UC06", "03-radar-and-genes.png", "#pgx-cohort-profile-section");

  await switchTab(page, "feature-importance-visualizations");
  await page.waitForSelector("#btnLoadFeatureImportance");
  await save(page, "UC04", "01-feature-importance-tab.png", "#feature-importance-visualizations-tab .controls");
  await page.evaluate(() => {
    const view = document.getElementById("fi-cohort");
    const top = document.getElementById("fi-top-n");
    if (view) {
      view.value = "combined";
      view.dispatchEvent(new Event("change", { bubbles: true }));
    }
    if (top) {
      top.value = "20";
      top.dispatchEvent(new Event("change", { bubbles: true }));
    }
  });
  await sleep(200);
  await save(page, "UC04", "02-view-and-top-n.png", "#feature-importance-visualizations-tab .controls");
  await Promise.all([
    page.waitForResponse((r) => /feature_importance|feature-importance/i.test(r.url()), { timeout: 30000 }).catch(() => null),
    page.click("#btnLoadFeatureImportance"),
  ]);
  await sleep(2500);
  await save(page, "UC04", "03-heatmap-loaded.png", "#fi-single-panel");

  await loadViz("cytoscape-visualizations", "#btnLoadCytoscape", /cytoscape|network|fpgrowth/i);
  await page.waitForFunction(() => {
    const iframe = document.getElementById("cytoscape-iframe");
    return iframe && iframe.src && iframe.src !== "about:blank";
  }, { timeout: 20000 }).catch(() => null);
  await sleep(2000);
  await page.evaluate(() => {
    if (document.getElementById("uc-cyto-clip")) return;
    const tab = document.getElementById("cytoscape-visualizations-tab");
    const clip = document.createElement("div");
    clip.id = "uc-cyto-clip";
    const controls = tab.querySelector(".controls");
    const status = document.getElementById("cytoscape-status");
    const panel = tab.querySelector(".panel");
    tab.insertBefore(clip, tab.firstChild);
    if (controls) clip.appendChild(controls);
    if (status) clip.appendChild(status);
    if (panel) clip.appendChild(panel);
  });
  await save(page, "UC05", "04-drug-networks.png", "#uc-cyto-clip");

  await browser.close();
  console.log("UC01_TO_UC06_RECAPTURE_DONE");
})().catch((err) => {
  console.error(err);
  process.exit(1);
});
