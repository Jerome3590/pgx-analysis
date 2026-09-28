"use strict";

/**
 * Recapture UC09 screenshots as distinct live clips (not full-page duplicates).
 * Writes Drive pack + repo mirror. Does not invent UI.
 */
const fs = require("fs");
const path = require("path");
const puppeteer = require("puppeteer");

const DASHBOARD = process.env.DASHBOARD_URL || "https://pgx.jerome-dixon.io/?v=20260928-genealogy-note";
const DRIVE = process.env.PGX_UC_DRIVE_ROOT || "G:\\My Drive\\PGx_Dashboard_Use_Cases";
const REPO = path.resolve(__dirname, "..", "..", "10_risk_dashboard", "docs", "use_case_training");
const UC = "UC09_documentation";

function dests() {
  return [
    path.join(DRIVE, UC, "screenshots"),
    path.join(REPO, UC, "screenshots"),
  ];
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function save(page, name, selector) {
  const dirs = dests();
  dirs.forEach((dir) => fs.mkdirSync(dir, { recursive: true }));
  const el = selector ? await page.$(selector) : null;
  const buf = el
    ? await el.screenshot({ type: "png" })
    : await page.screenshot({ type: "png", fullPage: false });
  for (const dir of dirs) {
    const file = path.join(dir, name);
    fs.writeFileSync(file, buf);
    console.log("saved", file, buf.length);
  }
}

async function wrapSections(page, clipId, headingRes) {
  await page.evaluate((id, patterns) => {
    if (document.getElementById(id)) return;
    const root = document.querySelector("#documentation-tab .doc-content") || document.getElementById("documentation-tab");
    if (!root) return;
    const headings = [...root.querySelectorAll("h2")];
    const regs = patterns.map((p) => new RegExp(p, "i"));
    const start = headings.find((el) => regs.some((re) => re.test(el.textContent || "")));
    if (!start) return;
    const wrap = document.createElement("div");
    wrap.id = id;
    const nodes = [];
    let n = start;
    while (n) {
      if (n !== start && n.tagName === "H2") {
        const keepGoing = regs.some((re) => re.test(n.textContent || ""));
        if (!keepGoing) break;
      }
      nodes.push(n);
      n = n.nextElementSibling;
    }
    start.parentNode.insertBefore(wrap, start);
    nodes.forEach((el) => wrap.appendChild(el));
  }, clipId, headingRes);
}

(async () => {
  const browser = await puppeteer.launch({
    headless: "new",
    defaultViewport: { width: 1440, height: 1100 },
    executablePath: process.env.PUPPETEER_EXECUTABLE_PATH || undefined,
  });
  const page = await browser.newPage();
  page.setDefaultTimeout(45000);
  await page.goto(DASHBOARD, { waitUntil: "networkidle0" });
  await page.waitForSelector("#btnRisk");
  await page.evaluate(() => {
    if (typeof window.switchTab === "function") window.switchTab("documentation");
  });
  await page.waitForSelector("#documentation-tab h2", { timeout: 15000 });
  await page.waitForFunction(() => {
    const metrics = document.getElementById("doc-metrics-container");
    const manifest = document.getElementById("doc-manifest-container");
    const metricsReady = metrics && !/Loading metrics/i.test(metrics.textContent || "");
    const manifestReady = manifest && !/Loading manifest/i.test(manifest.textContent || "");
    return metricsReady && manifestReady;
  }, { timeout: 25000 }).catch(() => null);
  await sleep(800);

  await wrapSections(page, "uc09-howto-clip", ["^Overview$", "^Tabs$"]);
  await page.$eval("#uc09-howto-clip", (el) => el.scrollIntoView({ block: "start" }));
  await sleep(200);
  await save(page, "01-documentation-how-to.png", "#uc09-howto-clip");

  await wrapSections(page, "uc09-metrics-clip", ["Model performance"]);
  await page.evaluate(() => {
    const box = document.querySelector("#uc09-metrics-clip #doc-metrics-container");
    if (box) [...box.children].slice(8).forEach((c) => c.remove());
  });
  await page.$eval("#uc09-metrics-clip", (el) => el.scrollIntoView({ block: "start" }));
  await sleep(200);
  await save(page, "02-model-performance.png", "#uc09-metrics-clip");

  await wrapSections(page, "uc09-manifest-clip", ["visual artifact|manifest"]);
  await page.$eval("#uc09-manifest-clip", (el) => el.scrollIntoView({ block: "start" }));
  await sleep(200);
  await save(page, "03-visual-artifacts.png", "#uc09-manifest-clip");

  await wrapSections(page, "uc09-density-clip", ["Feature importance sources", "Event density bins"]);
  await page.$eval("#uc09-density-clip", (el) => el.scrollIntoView({ block: "start" }));
  await sleep(200);
  await save(page, "04-density-bins.png", "#uc09-density-clip");

  await wrapSections(page, "uc09-unphased-clip", ["Unphased DNA files"]);
  await page.$eval("#uc09-unphased-clip", (el) => el.scrollIntoView({ block: "start" }));
  await sleep(200);
  await save(page, "05-unphased-dna.png", "#uc09-unphased-clip");

  await browser.close();
  console.log("UC09_RECAPTURE_DONE");
})().catch((err) => {
  console.error(err);
  process.exit(1);
});
