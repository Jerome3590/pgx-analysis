"use strict";

/**
 * Recapture UC08 screenshots as distinct live clips (not full-page near-duplicates).
 * Writes Drive pack + repo mirror. Does not invent UI.
 */
const fs = require("fs");
const path = require("path");
const puppeteer = require("puppeteer");

const DASHBOARD = process.env.DASHBOARD_URL || "https://pgx.jerome-dixon.io/?v=20260928-genealogy-note";
const DRIVE = process.env.PGX_UC_DRIVE_ROOT || "G:\\My Drive\\PGx_Dashboard_Use_Cases";
const REPO = path.resolve(__dirname, "..", "..", "10_risk_dashboard", "docs", "use_case_training");
const UC = "UC08_cohort_vs_card";

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
    const age = document.getElementById("age");
    if (age) {
      age.value = "45";
      age.dispatchEvent(new Event("input", { bubbles: true }));
    }
  });
  await sleep(600);

  await page.evaluate(() => {
    if (typeof window.switchTab === "function") window.switchTab("cohort-pgx-visualizations");
  });
  await page.waitForSelector("#btnLoadCohortPgx", { timeout: 15000 });
  await page.evaluate(() => {
    const cohort = document.getElementById("cohort-pgx-cohort");
    const age = document.getElementById("cohort-pgx-age-band");
    if (cohort) cohort.value = "opioid_ed";
    if (age) {
      const opt = [...age.options].find((o) => o.value === "45-54")
        || [...age.options].find((o) => /45/.test(o.value || ""));
      if (opt) age.value = opt.value;
      age.dispatchEvent(new Event("change", { bubbles: true }));
    }
  });
  await Promise.all([
    page.waitForResponse((r) => /cohort_pgx|network_topology/i.test(r.url()), { timeout: 30000 }).catch(() => null),
    page.click("#btnLoadCohortPgx"),
  ]);
  await page.waitForFunction(() => {
    const iframe = document.getElementById("cohort-pgx-iframe");
    return iframe && iframe.src && iframe.src !== "about:blank";
  }, { timeout: 25000 });
  await sleep(2500);

  await page.evaluate(() => {
    if (document.getElementById("uc08-network-clip")) return;
    const tab = document.getElementById("cohort-pgx-visualizations-tab");
    if (!tab) return;
    const clip = document.createElement("div");
    clip.id = "uc08-network-clip";
    const keep = [];
    const subtitle = tab.querySelector(".subtitle");
    const controls = tab.querySelector(".controls");
    const status = document.getElementById("cohort-pgx-status");
    const networkPanel = [...tab.querySelectorAll(".panel")].find((p) =>
      /Gene–Drug–Phenotype Network Topology/i.test((p.querySelector("h2") || {}).textContent || "")
    );
    [subtitle, controls, status, networkPanel].forEach((el) => {
      if (el) keep.push(el);
    });
    if (!keep.length) return;
    keep[0].parentNode.insertBefore(clip, keep[0]);
    keep.forEach((el) => clip.appendChild(el));
  });
  await page.$eval("#uc08-network-clip", (el) => el.scrollIntoView({ block: "start" }));
  await sleep(300);
  await save(page, "01-pgx-cohort-network.png", "#uc08-network-clip");

  await page.evaluate(() => {
    if (typeof window.switchTab === "function") window.switchTab("pgx-card");
  });
  await page.waitForSelector("#btnLoadPgxCardProfile", { timeout: 15000 });
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
  await Promise.all([
    page.waitForResponse((r) => /pubmed_citations|pgx_radar|cohort_pgx/i.test(r.url()), { timeout: 30000 }).catch(() => null),
    page.click("#btnLoadPgxCardProfile"),
  ]);
  await page.waitForFunction(() => {
    const sec = document.getElementById("pgx-cohort-profile-section");
    return sec && getComputedStyle(sec).display !== "none";
  }, { timeout: 25000 });
  await sleep(1500);

  await page.evaluate(() => {
    if (document.getElementById("uc08-profile-clip")) return;
    const tab = document.getElementById("pgx-card-tab");
    if (!tab) return;
    const clip = document.createElement("div");
    clip.id = "uc08-profile-clip";
    const keep = [];
    const subtitle = tab.querySelector(".subtitle");
    const inputs = tab.querySelector(".pgx-input-section");
    const status = document.getElementById("pgx-card-status");
    const profile = document.getElementById("pgx-cohort-profile-section");
    [subtitle, inputs, status, profile].forEach((el) => {
      if (el) keep.push(el);
    });
    if (!keep.length) return;
    keep[0].parentNode.insertBefore(clip, keep[0]);
    keep.forEach((el) => clip.appendChild(el));
  });
  await page.$eval("#uc08-profile-clip", (el) => el.scrollIntoView({ block: "start" }));
  await sleep(300);
  await save(page, "03-roles-compared.png", "#uc08-profile-clip");

  await page.evaluate(() => {
    const pid = document.getElementById("patient-id");
    if (pid) pid.value = "training-demo";
    const snp = document.getElementById("snp-input");
    if (snp) snp.value = "CYP2D6,*1,*4\nCYP2C19,*1,*2\nrs4149056,TC";
    const patient = document.querySelector("input[name='pgx-view-mode'][value='patient']");
    if (patient) patient.click();
  });
  await addDrug(page, "clopidogrel");
  page.on("console", (msg) => console.log("PAGE", msg.type(), msg.text().slice(0, 300)));
  page.on("request", (req) => {
    if (req.method() === "POST") console.log("POST", req.url());
  });
  await page.$eval("#btnGenerateCard", (el) => el.scrollIntoView({ block: "center" }));
  await sleep(300);
  await Promise.all([
    page.waitForResponse((r) => r.url().includes("/pgx/card") && r.request().method() === "POST", { timeout: 90000 }),
    page.click("#btnGenerateCard"),
  ]).catch(async (err) => {
    const status = await page.$eval("#pgx-status", (el) => (el.textContent || "").trim()).catch(() => "");
    console.error("GENERATE_STATUS", status);
    throw err;
  });
  await page.waitForFunction(() => {
    const box = document.getElementById("pgx-card-display");
    return box && getComputedStyle(box).display !== "none";
  }, { timeout: 20000 });
  await sleep(800);

  await page.evaluate(() => {
    if (document.getElementById("uc08-patient-clip")) return;
    const clip = document.createElement("div");
    clip.id = "uc08-patient-clip";
    const viewRow = [...document.querySelectorAll("#pgx-snp-refine-section .control-group")].find((el) =>
      /Patient/.test(el.textContent || "") && el.querySelector("input[name='pgx-view-mode']")
    );
    const card = document.getElementById("pgx-card-display");
    const header = card && card.querySelector(".pgx-card-header");
    const notice = document.getElementById("pgx-report-notice");
    const testing = document.getElementById("pgx-testing-card");
    const queue = document.getElementById("pgx-action-queue");
    const parent = (viewRow && viewRow.parentNode) || (card && card.parentNode);
    if (!parent) return;
    parent.insertBefore(clip, viewRow || card);
    if (viewRow) clip.appendChild(viewRow);
    if (header) clip.appendChild(header);
    if (notice) clip.appendChild(notice);
    if (testing) clip.appendChild(testing);
    if (queue) clip.appendChild(queue);
  });
  await page.$eval("#uc08-patient-clip", (el) => el.scrollIntoView({ block: "start" }));
  await sleep(300);
  await save(page, "02-pgx-card-patient.png", "#uc08-patient-clip");

  await browser.close();
  console.log("UC08_RECAPTURE_DONE");
})().catch((err) => {
  console.error(err);
  process.exit(1);
});
