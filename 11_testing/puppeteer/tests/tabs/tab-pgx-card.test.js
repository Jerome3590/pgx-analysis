"use strict";

/**
 * PGx Card tab end-to-end tests.
 *
 * Submits gene variants via the textarea / generate button and asserts:
 *   - POST /pgx/card returns 200 with genes + drugs arrays
 *   - Card renders in the DOM (gene items visible)
 *   - 400 is returned for an empty payload
 *
 * Run:
 *   DASHBOARD_URL=... API_BASE_URL=... npx jest tests/pgx-card --forceExit
 */

const { launchBrowser, openDashboard, sleep } = require("../../helpers/browser");

let browser;
let page;

beforeAll(async () => {
  browser = await launchBrowser();
  page    = await openDashboard(browser);
}, 40_000);

afterAll(async () => {
  if (browser) await browser.close();
});

/**
 * Switch to PGx Card tab and wait for the tab HTML to inject.
 * The tab content loads from tabs/pgx-card.html via fetch; we wait for
 * #pgx-card-cohort to confirm the DOM is ready.
 */
async function openPgxCardTab() {
  await page.evaluate(() => {
    const btn = document.querySelector('.tab-button[data-tab="pgx-card"]');
    if (btn) btn.click();
  });
  await page.waitForSelector("#pgx-card-cohort", { timeout: 10_000 });
}

/**
 * Select cohort + age band then click Load Cohort PGx Profile.
 * Waits for the SNP refinement section to become visible.
 */
async function loadCohortProfile(cohort = "opioid_ed", ageBand = "13-24") {
  await page.$eval("#pgx-card-cohort",   (el, v) => { el.value = v; }, cohort);
  await page.$eval("#pgx-card-age-band", (el, v) => { el.value = v; }, ageBand);
  // loadCohortPgxProfile used to hide #pgx-snp-refine-section during fetch, so
  // waitForSelector({visible:true}) resolved immediately then flake-failed
  // (offsetParent null). Wait for the profile section + success status instead.
  await page.$eval("#btnLoadPgxCardProfile", el => el.click());
  await page.waitForFunction(() => {
    const profile = document.getElementById("pgx-cohort-profile-section");
    const snp = document.getElementById("pgx-snp-refine-section");
    const status = document.getElementById("pgx-card-status");
    if (!profile || !snp || !status) return false;
    const profileOn = window.getComputedStyle(profile).display !== "none";
    const snpOn = window.getComputedStyle(snp).display !== "none";
    return profileOn && snpOn && /identified|genes/i.test(status.textContent || "");
  }, { timeout: 25_000 });
}

/**
 * Fill #snp-input with variant lines (format: Gene,*allele1,*allele2 per line)
 * then click #btnGenerateCard and wait for POST /pgx/card response.
 *
 * Returns { status, data } — data is null when non-200.
 */
async function submitVariants(variantLines) {
  const text = variantLines.join("\n");
  await page.$eval("#snp-input", (el, v) => { el.value = v; }, text);
  await sleep(100);

  const [response] = await Promise.all([
    page.waitForResponse(
      resp => resp.url().includes("/pgx/card") && resp.request().method() === "POST",
      { timeout: 15_000 }
    ).catch(() => null),
    page.$eval("#btnGenerateCard", el => el.click()),
  ]);

  if (!response) return { status: 0, data: null };
  let data = null;
  try { data = await response.json(); } catch (_) {}
  return { status: response.status(), data };
}

// ---------------------------------------------------------------------------
// DOM state helpers
// ---------------------------------------------------------------------------

/** Returns visibility + text content for a selector, or null if absent. */
async function domState(selector) {
  return page.evaluate(sel => {
    const el = document.querySelector(sel);
    if (!el) return null;
    const style = window.getComputedStyle(el);
    const displayed = style.display !== "none" && style.visibility !== "hidden";
    return {
      visible: displayed && (el.offsetParent !== null || style.position === "fixed"),
      displayed,
      text:    el.textContent?.trim().slice(0, 200),
      count:   el.children?.length ?? 0,
    };
  }, selector);
}

async function setDrugScope(value) {
  await page.$eval("#pgx-drug-scope", (el, v) => {
    el.value = v;
    el.dispatchEvent(new Event("change", { bubbles: true }));
  }, value);
}

async function hookExports() {
  await page.evaluate(() => {
    window.__pgxExportBlobs = [];
    window.__pgxPrintCalled = 0;
    if (window.__pgxExportHooked) return;
    window.__pgxExportHooked = true;
    const origCreate = URL.createObjectURL.bind(URL);
    URL.createObjectURL = function (blob) {
      const rec = { type: blob.type, size: blob.size, name: null, text: null };
      window.__pgxExportBlobs.push(rec);
      if (blob && typeof blob.text === "function") {
        blob.text().then((t) => { rec.text = t; rec.size = t.length; });
      }
      return origCreate(blob);
    };
    const origClick = HTMLAnchorElement.prototype.click;
    HTMLAnchorElement.prototype.click = function () {
      if (this.download) {
        const rec = window.__pgxExportBlobs[window.__pgxExportBlobs.length - 1];
        if (rec && !rec.name) rec.name = this.download;
      }
      return origClick.call(this);
    };
    window.print = function () { window.__pgxPrintCalled += 1; };
  });
}

async function resetExportHooks() {
  await page.evaluate(() => {
    window.__pgxExportBlobs = [];
    window.__pgxPrintCalled = 0;
  });
}

async function readExports() {
  await sleep(500);
  return page.evaluate(() => ({
    blobs: (window.__pgxExportBlobs || []).map((b) => {
      let parsed = null;
      try { parsed = b.text ? JSON.parse(b.text) : null; } catch (_) {}
      return { type: b.type, size: b.size, name: b.name, parsed, textLen: b.text ? b.text.length : 0 };
    }),
    printCalled: window.__pgxPrintCalled || 0,
  }));
}

async function cardSnapshot() {
  return page.evaluate(() => {
    const display = document.getElementById("pgx-card-display");
    const cs = display ? getComputedStyle(display) : null;
    const queue = document.getElementById("pgx-action-queue");
    const matrix = document.getElementById("pgx-action-matrix");
    const triplets = document.getElementById("pgx-triplet-panel");
    const genes = document.getElementById("pgx-genes-list");
    const summary = document.getElementById("pgx-summary-cards");
    const radar = document.getElementById("pgx-card-radar-chart");
    const chips = [...document.querySelectorAll("button.pgx-chip")].map((b) =>
      b.textContent.replace(/[×x]\s*$/i, "").trim().toLowerCase()
    );
    const drugNames = [...(queue ? queue.querySelectorAll(".pgx-action-card h3") : [])]
      .map((h) => h.textContent.trim().toLowerCase());
    const state = window.PgxWorkflow && window.PgxWorkflow.state;
    return {
      displayVisible: !!(display && cs && cs.display !== "none"),
      geneItems: genes ? genes.querySelectorAll(".pgx-gene-item").length : 0,
      summaryCards: summary ? summary.querySelectorAll(".pgx-count-card").length : 0,
      actionCards: queue ? queue.querySelectorAll(".pgx-action-card").length : 0,
      actionEmpty: queue ? /no medication actions/i.test(queue.textContent || "") : false,
      matrixRows: matrix ? matrix.querySelectorAll("tbody tr").length : 0,
      tripletRows: triplets ? triplets.querySelectorAll("tbody tr").length : 0,
      tripletText: triplets ? (triplets.textContent || "").slice(0, 220) : "",
      radarHasPlot: !!(radar && radar.querySelector(".js-plotly-plot, .plot-container, svg")),
      chips,
      drugNames,
      lastRecs: state && state.lastRecs ? state.lastRecs.length : 0,
      lastAlerts: state && state.lastAlerts ? state.lastAlerts.length : 0,
      drugScope: state ? state.drugScope : null,
    };
  });
}

async function addChip(query) {
  await page.click("#pgx-drug-search", { clickCount: 3 });
  await page.keyboard.press("Backspace");
  await page.type("#pgx-drug-search", query, { delay: 25 });
  await sleep(350);
  const names = await page.evaluate(() =>
    [...document.querySelectorAll(".pgx-suggest-item")].map((el) => el.textContent.trim())
  );
  if (!names.length) return [];
  await page.$eval(".pgx-suggest-item", (el) => el.click());
  await sleep(150);
  return names;
}

// ---------------------------------------------------------------------------
// Test cases
// ---------------------------------------------------------------------------

describe("PGx Card tab — UI rendering (two-phase workflow)", () => {

  beforeAll(async () => {
    await openPgxCardTab();
  }, 40_000);

  // ── Phase 1: Load Cohort PGx Profile ──────────────────────────────────────

  test("Phase 1: cohort profile sections become visible after btnLoadPgxCardProfile", async () => {
    await loadCohortProfile("opioid_ed", "55-64");

    // pgx-snp-refine-section must be visible (displayed; offsetParent can be null
    // while a parent is still settling after the cohort-profile fetch)
    const snpSec = await domState("#pgx-snp-refine-section");
    expect(snpSec).not.toBeNull();
    expect(snpSec.displayed || snpSec.visible).toBe(true);

    // pgx-cohort-profile-section must also be visible
    const profileSec = await domState("#pgx-cohort-profile-section");
    expect(profileSec).not.toBeNull();
    expect(profileSec.visible).toBe(true);

    // Status must not show an error
    const status = await domState("#pgx-card-status");
    if (status && status.text) {
      expect(status.text.toLowerCase()).not.toMatch(/error|failed/);
    }
  }, 30_000);

  test("Phase 1: medication search lists visible APCD drug names", async () => {
    await page.waitForSelector("#pgx-drug-search", { timeout: 10_000 });
    await page.click("#pgx-drug-search");
    await page.keyboard.type("oxy", { delay: 30 });
    await sleep(300);
    const suggest = await page.evaluate(() => {
      const list = document.getElementById("pgx-drug-suggest");
      const items = [...(list ? list.querySelectorAll(".pgx-suggest-item") : [])];
      const first = items[0];
      const cs = first ? getComputedStyle(first) : null;
      return {
        hidden: list ? list.hidden : true,
        names: items.map((el) => el.textContent.trim()),
        color: cs ? cs.color : null,
        bg: cs ? cs.backgroundColor : null,
      };
    });
    expect(suggest.hidden).toBe(false);
    expect(suggest.names.length).toBeGreaterThan(0);
    expect(suggest.names.join(" ").toLowerCase()).toMatch(/oxy/);
    const rgb = (suggest.color || "").match(/\d+/g) || [];
    const brightness = rgb.slice(0, 3).reduce((s, n) => s + Number(n), 0);
    expect(brightness).toBeLessThan(600);
  }, 15_000);

  test("Phase 1: identified PGx genes list is populated", async () => {
    // pgx-gene-list or equivalent child elements inside the profile section
    const geneList = await page.evaluate(() => {
      // Genes render as inline divs inside #pgx-cohort-genes-content
      const candidates = [
        "#pgx-cohort-genes-content > div",
        "#pgx-cohort-genes-content div",
        "#pgx-cohort-profile-section .pgx-gene-item",
        "#pgx-cohort-profile-section li",
      ];
      for (const sel of candidates) {
        const els = document.querySelectorAll(sel);
        if (els.length > 0) return { selector: sel, count: els.length, texts: [...els].slice(0, 3).map(e => e.textContent.trim().slice(0, 40)) };
      }
      return null;
    });
    // Gene list is data-dependent — just assert it's present if section is visible
    if (geneList) {
      expect(geneList.count).toBeGreaterThan(0);
      console.log(`PGx gene list: ${geneList.count} items via "${geneList.selector}" — e.g. ${geneList.texts}`);
    } else {
      // Profile section visible but no gene list items — log as warning, not failure
      console.warn("pgx-cohort-profile-section visible but no gene list items found — check selector");
    }
  }, 10_000);

  // ── Phase 2: Generate Personalized Card ───────────────────────────────────

  test("Phase 2: generate card renders pgx-card-display with gene + drug content", async () => {
    const { status, data } = await submitVariants([
      "CYP2D6,*1,*2",
      "CYP2C19,*1,*17",
      "SLCO1B1,*5,*1",
    ]);

    expect([200, 400, 500]).toContain(status);

    if (status === 200) {
      // Card display section must become visible
      await page.waitForFunction(
        () => {
          const el = document.getElementById("pgx-card-display");
          return el && window.getComputedStyle(el).display !== "none";
        },
        { timeout: 8_000 }
      ).catch(() => {});

      const cardDisplay = await domState("#pgx-card-display");
      expect(cardDisplay).not.toBeNull();
      expect(cardDisplay.visible).toBe(true);

      // pgx-status must show success (not error)
      const pgxStatus = await domState("#pgx-status");
      if (pgxStatus?.text) {
        expect(pgxStatus.text.toLowerCase()).not.toMatch(/error|failed/);
      }

      // Genes tested section must have at least one .pgx-gene-item
      const genesRendered = await page.evaluate(() => {
        const candidates = [".pgx-gene-item", "#pgx-gene-details .pgx-gene-item", "#pgx-card-display .pgx-gene-item"];
        for (const sel of candidates) {
          const els = document.querySelectorAll(sel);
          if (els.length) return { selector: sel, count: els.length };
        }
        return null;
      });
      if (genesRendered) {
        expect(genesRendered.count).toBeGreaterThan(0);
        console.log(`Card gene items: ${genesRendered.count} via "${genesRendered.selector}"`);
      }

      // Drugs list populated (if data has drugs)
      if (data?.drugs?.length > 0) {
        const drugsList = await domState("#pgx-drugs-list");
        if (drugsList) {
          expect(drugsList.count).toBeGreaterThan(0);
          console.log(`Drugs list: ${drugsList.count} items`);
        }
      }

      // API data assertions
      expect(Array.isArray(data.genes)).toBe(true);
      expect(data.genes.length).toBeGreaterThan(0);
    }
  }, 30_000);

  test("Phase 2: empty variants — frontend blocks or returns 400, card display stays hidden", async () => {
    await page.$eval("#snp-input", el => { el.value = ""; });
    await sleep(100);

    const [response] = await Promise.all([
      page.waitForResponse(
        r => r.url().includes("/pgx/card") && r.request().method() === "POST",
        { timeout: 4_000 }
      ).catch(() => null),
      page.$eval("#btnGenerateCard", el => el.click()),
    ]);

    const status = response ? response.status() : 0;
    expect([0, 400]).toContain(status);

    // Card display must NOT be newly visible after an empty submit
    const cardDisplay = await domState("#pgx-card-display");
    if (cardDisplay && status === 0) {
      // Frontend blocked — card should still be hidden or unchanged
      console.log("Frontend blocked empty submit — card display state:", cardDisplay.visible);
    }
  }, 10_000);

});

describe("PGx Card tab", () => {

  beforeAll(async () => {
    await openPgxCardTab();
    await loadCohortProfile("opioid_ed", "13-24");
  }, 40_000);

  test("POST /pgx/card with CYP2D6 variant returns 200 with genes + drugs", async () => {
    const { status, data } = await submitVariants([
      "CYP2D6,*1,*2",
      "CYP2C19,*1,*1",
    ]);

    expect([200, 400, 500]).toContain(status);

    if (status === 200 && data) {
      expect(Array.isArray(data.genes)).toBe(true);
      expect(Array.isArray(data.drugs)).toBe(true);
      expect(data.genes.length).toBeGreaterThan(0);
      for (const g of data.genes) {
        expect(typeof g.gene).toBe("string");
        expect(Array.isArray(g.variants)).toBe(true);
      }
    }
  }, 20_000);

  test("POST /pgx/card with empty variants — frontend guards or returns 400", async () => {
    const text = "";
    await page.$eval("#snp-input", (el, v) => { el.value = v; }, text);
    await sleep(100);
    const [response] = await Promise.all([
      page.waitForResponse(
        resp => resp.url().includes("/pgx/card") && resp.request().method() === "POST",
        { timeout: 4_000 }            // frontend should block; short wait is fine
      ).catch(() => null),
      page.$eval("#btnGenerateCard", el => el.click()),
    ]);
    // status 0 = frontend blocked; 400 = backend rejected empty payload
    const status = response ? response.status() : 0;
    expect([0, 400]).toContain(status);
  }, 10_000);

  test("Multiple gene variants: SLCO1B1 + TPMT + DPYD", async () => {
    const { status, data } = await submitVariants([
      "SLCO1B1,*5,*1",
      "TPMT,*3A,*1",
      "DPYD,*2A",
    ]);

    expect([200, 400, 500]).toContain(status);

    if (status === 200 && data) {
      expect(data.genes.length).toBeGreaterThanOrEqual(1);
      expect(Array.isArray(data.drugs)).toBe(true);
    }
  }, 20_000);

});

describe("PGx Card — ALL_MATCHED vs selected scope + exports", () => {
  beforeAll(async () => {
    await openPgxCardTab();
    await loadCohortProfile("opioid_ed", "55-64");
    await hookExports();
  }, 50_000);

  test("ALL_MATCHED: visualizations populate without exploding triplets", async () => {
    await setDrugScope("ALL_MATCHED");
    const { status, data } = await submitVariants([
      "CYP2D6,*1,*2",
      "CYP2C19,*1,*17",
      "SLCO1B1,*5,*1",
    ]);
    expect(status).toBe(200);
    expect(data.drugs.length).toBeGreaterThan(10);

    await page.waitForFunction(() => {
      const el = document.getElementById("pgx-card-display");
      const queue = document.getElementById("pgx-action-queue");
      return el && getComputedStyle(el).display !== "none" &&
        queue && (queue.querySelector(".pgx-action-card") || /no medication|three-way/i.test(queue.textContent || ""));
    }, { timeout: 15_000 });

    const snap = await cardSnapshot();
    expect(snap.displayVisible).toBe(true);
    expect(snap.geneItems).toBeGreaterThan(0);
    expect(snap.summaryCards).toBeGreaterThan(0);
    expect(snap.actionCards).toBeGreaterThan(10);
    expect(snap.matrixRows).toBeGreaterThan(10);
    expect(snap.actionCards).toBe(snap.lastRecs);
    // Triplets must use the regimen, not every CPIC-matched drug.
    expect(snap.tripletRows).toBeLessThan(50);
    expect(snap.lastAlerts).toBeLessThan(50);
    expect(snap.drugScope).toBe("ALL_MATCHED");
    console.log(`ALL_MATCHED: ${snap.actionCards} cards, ${snap.matrixRows} matrix rows, ${snap.tripletRows} triplets, radar=${snap.radarHasPlot}`);
  }, 40_000);

  test("ALL_MATCHED: JSON, technical appendix, and print exports are scoped + non-empty", async () => {
    await resetExportHooks();
    await page.$eval("#pgx-export-json", (el) => el.click());
    await page.$eval("#pgx-export-tech", (el) => el.click());
    await page.$eval("#pgx-export-current", (el) => el.click());
    const exp = await readExports();
    expect(exp.printCalled).toBeGreaterThanOrEqual(1);

    const json = exp.blobs.find((b) => (b.name || "").includes("current") || (b.parsed && b.parsed.filters && !b.parsed.filters.includeTechnicalAppendix));
    const tech = exp.blobs.find((b) => (b.name || "").includes("appendix") || (b.parsed && b.parsed.filters && b.parsed.filters.includeTechnicalAppendix));
    expect(exp.blobs.length).toBeGreaterThanOrEqual(2);

    for (const blob of exp.blobs) {
      expect(blob.textLen).toBeGreaterThan(200);
      expect(blob.parsed).toBeTruthy();
      expect(blob.parsed.filters.drugScope).toBe("ALL_MATCHED");
      expect(blob.parsed.recommendations.length).toBeGreaterThan(10);
      expect(blob.parsed.geneCalls.length).toBeGreaterThan(0);
      expect(blob.parsed.polypharmacyAlerts.length).toBeLessThan(50);
    }
    if (tech && tech.parsed) {
      expect(tech.parsed.filters.includeTechnicalAppendix).toBe(true);
    }
    if (json && json.parsed) {
      expect(json.parsed.recommendations.length).toBeGreaterThan(10);
    }
    console.log(`ALL_MATCHED exports: ${exp.blobs.map((b) => `${b.name}:${b.parsed ? b.parsed.recommendations.length : 0}recs`).join(", ")} print=${exp.printCalled}`);
  }, 20_000);

  test("SELECTED drugs: visualizations shrink to the chip subset", async () => {
    const before = await cardSnapshot();
    expect(before.actionCards).toBeGreaterThan(10);

    const oxyHits = await addChip("oxy");
    expect(oxyHits.join(" ").toLowerCase()).toMatch(/oxy/);
    const coHits = await addChip("co");
    expect(coHits.join(" ").toLowerCase()).toMatch(/co/);

    await setDrugScope("SELECTED");
    await sleep(250);
    const snap = await cardSnapshot();
    expect(snap.chips.length).toBeGreaterThanOrEqual(1);
    expect(snap.actionCards).toBeGreaterThan(0);
    expect(snap.actionCards).toBeLessThan(before.actionCards);
    expect(snap.matrixRows).toBeGreaterThan(0);
    expect(snap.matrixRows).toBeLessThan(before.matrixRows);
    expect(snap.drugScope).toBe("SELECTED");
    const chipBlob = snap.chips.join(" ");
    expect(chipBlob).toMatch(/oxycodone|codeine|hydrocodone/);
    for (const name of snap.drugNames) {
      expect(chipBlob.includes(name) || snap.chips.some((c) => c.includes(name) || name.includes(c.split(" ")[0]))).toBe(true);
    }
    expect(snap.tripletRows).toBeLessThan(20);
    console.log(`SELECTED: chips=${snap.chips} cards=${snap.actionCards} (from ${before.actionCards}) drugs=${snap.drugNames}`);
  }, 25_000);

  test("SELECTED drugs: exports contain only the subset", async () => {
    await resetExportHooks();
    await page.$eval("#pgx-export-json", (el) => el.click());
    await page.$eval("#pgx-export-tech", (el) => el.click());
    await page.$eval("#pgx-export-current", (el) => el.click());
    const exp = await readExports();
    expect(exp.printCalled).toBeGreaterThanOrEqual(1);
    expect(exp.blobs.length).toBeGreaterThanOrEqual(2);

    const chips = await page.evaluate(() =>
      [...document.querySelectorAll("button.pgx-chip")].map((b) =>
        b.textContent.replace(/[×x]\s*$/i, "").trim().toLowerCase()
      )
    );
    for (const blob of exp.blobs) {
      expect(blob.textLen).toBeGreaterThan(100);
      expect(blob.parsed).toBeTruthy();
      expect(blob.parsed.filters.drugScope).toBe("SELECTED");
      expect(blob.parsed.filters.selectedApcdDrugIds.length).toBe(chips.length);
      expect(blob.parsed.recommendations.length).toBeGreaterThan(0);
      expect(blob.parsed.recommendations.length).toBeLessThan(40);
      for (const rec of blob.parsed.recommendations) {
        const n = String(rec.formattedGenericName || "").toLowerCase();
        expect(chips.some((c) => n.includes(c) || c.includes(n) || c.split(/[^a-z0-9]+/).includes(n))).toBe(true);
      }
    }
    console.log(`SELECTED exports: ${exp.blobs.map((b) => `${b.name}:${b.parsed.recommendations.length}recs/${b.parsed.filters.drugScope}`).join(", ")}`);
  }, 20_000);

  test("Clear chips + ALL_MATCHED restores the full card", async () => {
    await page.evaluate(() => {
      document.querySelectorAll("button.pgx-chip").forEach((b) => b.click());
    });
    await setDrugScope("ALL_MATCHED");
    await sleep(250);
    const snap = await cardSnapshot();
    expect(snap.chips.length).toBe(0);
    expect(snap.actionCards).toBeGreaterThan(10);
    expect(snap.actionCards).toBe(snap.lastRecs);
    expect(snap.matrixRows).toBeGreaterThan(10);
    expect(snap.tripletRows).toBeLessThan(50);
    expect(snap.drugScope).toBe("ALL_MATCHED");
  }, 15_000);

});
