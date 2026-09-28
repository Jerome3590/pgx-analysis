"use strict";

/**
 * Recapture UC07 screenshots as distinct live clips (not full-page duplicates).
 * Writes Drive pack + repo mirror. Does not invent UI.
 */
const fs = require("fs");
const path = require("path");
const puppeteer = require("puppeteer");

const DASHBOARD = process.env.DASHBOARD_URL || "https://pgx.jerome-dixon.io/?v=20260928-genealogy-note";
const DRIVE = process.env.PGX_UC_DRIVE_ROOT || "G:\\My Drive\\PGx_Dashboard_Use_Cases";
const REPO = path.resolve(__dirname, "..", "..", "10_risk_dashboard", "docs", "use_case_training");
const UC = "UC07_personalized_pgx_card";

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

async function addDrug(page, query) {
  const search = await page.$("#pgx-drug-search");
  if (!search) throw new Error("pgx-drug-search missing");
  await search.click({ clickCount: 3 });
  await page.keyboard.press("Backspace");
  await search.type(query, { delay: 25 });
  await sleep(500);
  const item = await page.$(".pgx-suggest-item");
  if (!item) throw new Error("no suggest item for " + query);
  await item.click();
  await sleep(250);
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
    if (typeof window.switchTab === "function") window.switchTab("pgx-card");
  });
  await page.waitForSelector("#snp-input", { timeout: 15000 });
  await sleep(800);

  await page.evaluate(() => {
    const pid = document.getElementById("patient-id");
    if (pid) pid.value = "training-demo";
    const snp = document.getElementById("snp-input");
    if (snp) {
      snp.value = "CYP2D6,*1,*4\nCYP2C19,*1,*2\nrs4149056,TC";
    }
  });
  await page.evaluate(() => {
    if (document.getElementById("uc07-gene-clip")) return;
    const section = document.getElementById("pgx-snp-refine-section");
    const box = section && section.firstElementChild;
    if (!box) return;
    const geneWrap = document.createElement("div");
    geneWrap.id = "uc07-gene-clip";
    const medWrap = document.createElement("div");
    medWrap.id = "uc07-med-clip";
    const kids = [...box.children];
    const split = kids.findIndex((el) => el.tagName === "H2" && /Virginia APCD/i.test(el.textContent || ""));
    kids.slice(0, split >= 0 ? split : kids.length).forEach((el) => geneWrap.appendChild(el));
    kids.slice(split >= 0 ? split : kids.length).forEach((el) => medWrap.appendChild(el));
    box.appendChild(geneWrap);
    box.appendChild(medWrap);
  });
  await page.$eval("#uc07-gene-clip", (el) => el.scrollIntoView({ block: "start" }));
  await sleep(300);
  await save(page, "01-gene-data-entry.png", "#uc07-gene-clip");

  await addDrug(page, "clopidogrel");
  await addDrug(page, "gabapentin");
  await addDrug(page, "alprazolam");
  await page.evaluate(() => {
    const scope = document.getElementById("pgx-drug-scope");
    if (scope) {
      scope.value = "SELECTED";
      scope.dispatchEvent(new Event("change", { bubbles: true }));
    }
    const clinician = document.querySelector("input[name='pgx-view-mode'][value='clinician']");
    if (clinician) clinician.click();
  });
  await page.$eval("h2", () => {});
  await page.$eval("#uc07-med-clip", (el) => el.scrollIntoView({ block: "start" }));
  await sleep(300);
  await save(page, "02-apcd-meds-and-scope.png", "#uc07-med-clip");

  page.on("console", (msg) => console.log("PAGE", msg.type(), msg.text().slice(0, 300)));
  page.on("request", (req) => {
    if (req.method() === "POST") console.log("POST", req.url());
  });
  await Promise.all([
    page.waitForResponse((r) => r.url().includes("/pgx/card") && r.request().method() === "POST", { timeout: 90000 }),
    page.click("#btnGenerateCard"),
  ]).catch(async (err) => {
    const status = await page.$eval("#pgx-status", (el) => el.textContent || "").catch(() => "");
    console.error("GENERATE_STATUS", status);
    throw err;
  });
  await page.waitForFunction(() => {
    const box = document.getElementById("pgx-card-display");
    return box && getComputedStyle(box).display !== "none";
  }, { timeout: 20000 });
  await sleep(800);

  await page.evaluate(() => {
    if (document.getElementById("uc07-results-clip")) return;
    const card = document.querySelector("#pgx-card-display .pgx-card");
    if (!card) return;
    const clip = document.createElement("div");
    clip.id = "uc07-results-clip";
    const keep = [
      card.querySelector(".pgx-card-header"),
      document.getElementById("pgx-summary-cards"),
      document.getElementById("pgx-report-notice"),
      document.getElementById("pgx-testing-card"),
    ].filter(Boolean);
    if (!keep.length) return;
    keep[0].parentNode.insertBefore(clip, keep[0]);
    keep.forEach((el) => clip.appendChild(el));
  });
  await page.$eval("#uc07-results-clip", (el) => el.scrollIntoView({ block: "start" }));
  await sleep(200);
  await save(page, "03-generate-pgx-results.png", "#uc07-results-clip");

  await page.evaluate(() => {
    if (document.getElementById("uc07-queue-clip")) return;
    const card = document.querySelector("#pgx-card-display .pgx-card");
    if (!card) return;
    const queueClip = document.createElement("div");
    queueClip.id = "uc07-queue-clip";
    const restClip = document.createElement("div");
    restClip.id = "uc07-rest-clip";
    const sections = [...card.querySelectorAll(":scope > .pgx-card-section")];
    sections.forEach((sec) => {
      const title = (sec.querySelector("h3") || {}).textContent || "";
      if (/action queue|actionability matrix/i.test(title)) queueClip.appendChild(sec);
      else restClip.appendChild(sec);
    });
    card.appendChild(queueClip);
    card.appendChild(restClip);
  });
  await page.$eval("#uc07-queue-clip", (el) => el.scrollIntoView({ block: "start" }));
  await sleep(200);
  await save(page, "04-action-queue-matrix.png", "#uc07-queue-clip");

  await page.$eval("#uc07-rest-clip", (el) => el.scrollIntoView({ block: "start" }));
  await sleep(200);
  await save(page, "06-triplets-genes-pharmacy.png", "#uc07-rest-clip");

  await page.evaluate(() => {
    const row = document.querySelector(".pgx-export-row");
    if (row) row.scrollIntoView({ block: "center" });
  });
  await sleep(200);
  await save(page, "05-exports.png", ".pgx-export-row");

  await browser.close();
  console.log("UC07_RECAPTURE_DONE");
})().catch((err) => {
  console.error(err);
  process.exit(1);
});
