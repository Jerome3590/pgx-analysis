"use strict";

/**
 * Diagnose empty drug-name filters on the live PGx dashboard.
 * Usage (from 11_testing/puppeteer):
 *   node debug_drug_filter.js
 */
const puppeteer = require("puppeteer");

const DASHBOARD_URL =
  process.env.DASHBOARD_URL || "https://pgx.jerome-dixon.io/";
const API_BASE_URL = process.env.API_BASE_URL || "";

function url() {
  const u = new URL(DASHBOARD_URL);
  if (API_BASE_URL) u.searchParams.set("apiBase", API_BASE_URL);
  return u.toString();
}

async function main() {
  const browser = await puppeteer.launch({
    headless: "new",
    args: ["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"],
  });
  const page = await browser.newPage();
  page.setDefaultTimeout(45_000);
  const failed = [];
  page.on("console", (msg) => {
    const t = msg.text();
    if (/error|fail|metadata|vocab|drug/i.test(t)) {
      failed.push(`[console.${msg.type()}] ${t.slice(0, 300)}`);
    }
  });
  page.on("pageerror", (err) => failed.push(`[pageerror] ${err.message}`));
  page.on("response", (resp) => {
    const u = resp.url();
    if (/metadata|pgx_reference|combination|available/i.test(u)) {
      failed.push(`[http ${resp.status()}] ${u}`);
    }
  });

  console.log("OPEN", url());
  await page.goto(url(), { waitUntil: "networkidle0" });
  await page.waitForSelector("#btnRisk", { timeout: 25_000 });

  const risk = await page.evaluate(() => {
    const drugs = document.getElementById("drugs");
    const age = document.getElementById("age");
    const opts = drugs ? [...drugs.options].map((o) => o.textContent).filter(Boolean) : [];
    return {
      age: age && age.value,
      drugCount: opts.length,
      sample: opts.slice(0, 8),
      metadataKeys: window.currentMetadata
        ? Object.keys(window.currentMetadata)
        : "currentMetadata not on window",
    };
  });
  console.log("RISK DRUG SELECT", JSON.stringify(risk, null, 2));

  await page.evaluate(() => {
    const btn = document.querySelector('.tab-button[data-tab="pgx-card"]');
    if (btn) btn.click();
  });
  await page.waitForSelector("#pgx-drug-search", { timeout: 15_000 });

  const before = await page.evaluate(() => {
    const wf = window.PgxWorkflow;
    return {
      inited: !!(wf && wf.state && wf.state._inited),
      vocab: wf && wf.state ? wf.state.vocab.length : -1,
      sample: wf && wf.state ? wf.state.vocab.slice(0, 12) : [],
      comboKeys: wf && wf.state ? Object.keys(wf.state.combinations || {}).length : -1,
      searchBound: !!(document.getElementById("pgx-drug-search") || {}).dataset.bound,
    };
  });
  console.log("PGX VOCAB BEFORE TYPE", JSON.stringify(before, null, 2));

  await page.click("#pgx-drug-search");
  await page.type("#pgx-drug-search", "co", { delay: 40 });
  await new Promise((r) => setTimeout(r, 400));

  const after = await page.evaluate(() => {
    const list = document.getElementById("pgx-drug-suggest");
    const items = list ? [...list.querySelectorAll(".pgx-suggest-item")].map((e) => e.textContent) : [];
    const empty = list ? list.querySelector(".pgx-suggest-empty") : null;
    return {
      hidden: list ? list.hidden : null,
      itemCount: items.length,
      items: items.slice(0, 15),
      emptyText: empty ? empty.textContent : null,
      inputValue: (document.getElementById("pgx-drug-search") || {}).value,
    };
  });
  console.log("PGX SUGGEST AFTER 'co'", JSON.stringify(after, null, 2));

  await page.$eval("#pgx-card-cohort", (el) => { el.value = "opioid_ed"; });
  await page.$eval("#pgx-card-age-band", (el) => { el.value = "25-44"; });
  await page.click("#btnLoadPgxCardProfile");
  await new Promise((r) => setTimeout(r, 2500));

  const afterLoad = await page.evaluate(() => {
    const wf = window.PgxWorkflow;
    const genes = document.getElementById("pgx-cohort-genes-content");
    return {
      vocab: wf && wf.state ? wf.state.vocab.length : -1,
      sample: wf && wf.state ? wf.state.vocab.slice(0, 12) : [],
      status: (document.getElementById("pgx-card-status") || {}).textContent,
      geneHtmlLen: genes ? genes.innerHTML.length : 0,
    };
  });
  console.log("AFTER LOAD PROFILE", JSON.stringify(afterLoad, null, 2));

  await page.click("#pgx-drug-search", { clickCount: 3 });
  await page.type("#pgx-drug-search", "oxy", { delay: 40 });
  await new Promise((r) => setTimeout(r, 400));
  const oxy = await page.evaluate(() => {
    const list = document.getElementById("pgx-drug-suggest");
    const items = list ? [...list.querySelectorAll(".pgx-suggest-item")].map((e) => e.textContent) : [];
    return { hidden: list ? list.hidden : null, items: items.slice(0, 15), empty: list && list.textContent };
  });
  console.log("PGX SUGGEST AFTER 'oxy'", JSON.stringify(oxy, null, 2));

  console.log("NETWORK/CONSOLE");
  failed.forEach((l) => console.log(l));

  await browser.close();
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
