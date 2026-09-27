"use strict";

const fs = require("fs");
const path = require("path");
const puppeteer = require("puppeteer");

const DASHBOARD = process.env.DASHBOARD_URL || "https://pgx.jerome-dixon.io/?v=20260927-unphased";
const DRIVE = process.env.PGX_UC_DRIVE_ROOT || "G:\\My Drive\\PGx_Dashboard_Use_Cases";
const REPO = path.resolve(__dirname, "..", "..", "10_risk_dashboard", "docs", "use_case_training");
const UC = "UC04_feature_importance";

function dests() {
  return [path.join(DRIVE, UC, "screenshots"), path.join(REPO, UC, "screenshots")];
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function save(page, name, selector) {
  dests().forEach((dir) => fs.mkdirSync(dir, { recursive: true }));
  const el = selector ? await page.$(selector) : null;
  if (selector && !el) throw new Error("missing " + selector);
  const buf = el
    ? await el.screenshot({ type: "png" })
    : await page.screenshot({ type: "png", fullPage: false });
  for (const dir of dests()) {
    fs.writeFileSync(path.join(dir, name), buf);
    console.log("saved", name, buf.length);
  }
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
    if (typeof window.switchTab === "function") window.switchTab("feature-importance-visualizations");
  });
  await page.waitForSelector("#btnLoadFeatureImportance");
  await sleep(400);

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
  await save(page, "02-view-and-top-n.png", "#feature-importance-visualizations-tab .controls");

  await page.evaluate(() => {
    const view = document.getElementById("fi-cohort");
    const top = document.getElementById("fi-top-n");
    if (view) {
      view.value = "opioid_ed";
      view.dispatchEvent(new Event("change", { bubbles: true }));
    }
    if (top) {
      top.value = "20";
      top.dispatchEvent(new Event("change", { bubbles: true }));
    }
  });
  await Promise.all([
    page.waitForResponse((r) => /feature_importance|feature-importance/i.test(r.url()), { timeout: 30000 }).catch(() => null),
    page.click("#btnLoadFeatureImportance"),
  ]);
  await sleep(2500);
  await save(page, "03-heatmap-loaded.png", "#fi-single-panel");

  await browser.close();
  console.log("UC04_TOP20_DONE");
})().catch((err) => {
  console.error(err);
  process.exit(1);
});
