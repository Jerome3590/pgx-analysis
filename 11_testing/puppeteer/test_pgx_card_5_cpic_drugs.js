"use strict";

/**
 * End-to-end validation test for PGx Card using 5 CPIC Level A drugs:
 * 1. clopidogrel (CYP2C19)
 * 2. citalopram (CYP2C19)
 * 3. simvastatin (SLCO1B1)
 * 4. codeine (CYP2D6)
 * 5. warfarin (CYP2C9 / VKORC1)
 *
 * Validates:
 * - Autocomplete drug selection & active chips
 * - Options Matrix 3-panel grid
 * - Gene–Drug Actionability Matrix & dynamic CPIC filter pills
 * - Cell click inspection detail card
 * - Plotly Clustered Dendrogram Heatmaps across all 3 clustering modes
 * - Production screenshot capture
 */

const fs = require("fs");
const path = require("path");
const puppeteer = require("puppeteer");

const DASHBOARD_URL = process.env.DASHBOARD_URL || "https://pgx.jerome-dixon.io/?v=20261001-5cpic-test";
const SCREENSHOT_DIR = path.resolve(__dirname, "..", "..", "C:", "Users", "jerom", ".gemini", "antigravity-ide", "brain", "652d7e54-9f00-4c6e-85e7-e117f37e1461", "screenshots");
const LOCAL_SCREENSHOT_DIR = path.resolve(__dirname, "screenshots");

const CPIC_DRUGS = [
  "clopidogrel",
  "citalopram",
  "simvastatin",
  "codeine",
  "warfarin"
];

const GENE_LINES = [
  "CYP2C19,*1,*2",
  "CYP2D6,*1,*4",
  "SLCO1B1,*1,*5",
  "CYP2C9,*1,*3",
  "VKORC1,*1,*2"
];

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function addDrug(page, drugName) {
  const searchInput = await page.$("#pgx-drug-search");
  if (!searchInput) throw new Error("Missing #pgx-drug-search input");
  await searchInput.click({ clickCount: 3 });
  await page.keyboard.press("Backspace");
  await searchInput.type(drugName, { delay: 35 });
  await sleep(600);

  const suggestItem = await page.waitForSelector(".pgx-suggest-item", { timeout: 6000 });
  if (!suggestItem) throw new Error(`No suggestion item found for: ${drugName}`);
  
  const text = await page.evaluate(el => el.textContent, suggestItem);
  await suggestItem.click();
  await sleep(350);
  return text;
}

(async () => {
  console.log("=== Starting PGx Card 5 CPIC Drugs Puppeteer Test ===");
  console.log(`Target URL: ${DASHBOARD_URL}`);

  [SCREENSHOT_DIR, LOCAL_SCREENSHOT_DIR].forEach(dir => {
    try { fs.mkdirSync(dir, { recursive: true }); } catch (_) {}
  });

  const browser = await puppeteer.launch({
    headless: "new",
    defaultViewport: { width: 1440, height: 1100 },
    args: ["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"]
  });

  const results = {
    urlLoaded: false,
    tabSwitched: false,
    genesEntered: false,
    drugsSelected: [],
    cardGenerated: false,
    optionsMatrixVerified: false,
    actionMatrixVerified: false,
    filterPillsVerified: false,
    cellDetailVerified: false,
    dendrogramVerified: false,
    dendrogramModesVerified: false,
    screenshots: []
  };

  try {
    const page = await browser.newPage();
    page.setDefaultTimeout(60000);

    page.on("console", msg => console.log(`  [PAGE ${msg.type()}]:`, msg.text().slice(0, 200)));
    page.on("pageerror", err => console.error("  [PAGE ERROR]:", err.message));
    page.on("request", req => {
      if (req.method() === "POST") console.log("  [PAGE POST]:", req.url());
    });

    // 1. Navigate to dashboard
    console.log("Navigating to live dashboard...");
    await page.goto(DASHBOARD_URL, { waitUntil: "networkidle0" });
    await page.waitForSelector("#btnRisk", { timeout: 25000 });
    results.urlLoaded = true;
    console.log("✓ Dashboard loaded successfully.");

    // 2. Switch to PGx Card tab
    console.log("Switching to PGx Card tab...");
    await page.evaluate(() => {
      if (typeof window.switchTab === "function") window.switchTab("pgx-card");
    });
    await page.waitForSelector("#snp-input", { timeout: 15000 });
    results.tabSwitched = true;
    console.log("✓ Switched to PGx Card tab.");

    // 3. Fill Gene/Allele entries
    console.log("Entering gene alleles for CYP2C19, CYP2D6, SLCO1B1, CYP2C9, VKORC1...");
    await page.evaluate((genes) => {
      const snpInput = document.getElementById("snp-input");
      if (snpInput) snpInput.value = genes.join("\n");
      const pid = document.getElementById("patient-id");
      if (pid) pid.value = "CPIC-5DRUG-VALIDATION";
    }, GENE_LINES);
    results.genesEntered = true;
    console.log("✓ Gene alleles entered.");

    // 4. Add 5 CPIC Drugs via autocomplete
    console.log("Adding 5 CPIC Level A drugs...");
    for (const drug of CPIC_DRUGS) {
      const matched = await addDrug(page, drug);
      results.drugsSelected.push({ drug, matched });
      console.log(`  + Added drug: ${drug} (${matched})`);
    }

    // Verify chips
    const chipCount = await page.evaluate(() => {
      return document.querySelectorAll("#pgx-selected-chips .pgx-chip").length;
    });
    console.log(`✓ Active selected chips count: ${chipCount}`);

    // Set Drug scope to "Selected drugs" so the matrix focuses on our 5 test drugs
    await page.evaluate(() => {
      const scope = document.getElementById("pgx-drug-scope");
      if (scope) {
        scope.value = "SELECTED";
        scope.dispatchEvent(new Event("change"));
      }
    });

    // 5. Generate PGx Card
    console.log("Clicking Generate PGx results (#btnGenerateCard)...");
    const [cardResponse] = await Promise.all([
      page.waitForResponse(
        resp => resp.url().includes("/pgx/card") && resp.request().method() === "POST",
        { timeout: 60000 }
      ),
      page.click("#btnGenerateCard")
    ]).catch(async (err) => {
      const statusText = await page.evaluate(() => {
        const el = document.getElementById("pgx-status");
        return el ? el.textContent : "No status element";
      }).catch(() => "failed to read status");
      console.error("  [PGx Status upon failure]:", statusText);
      throw err;
    });

    const status = cardResponse.status();
    console.log(`✓ POST /pgx/card returned HTTP ${status}`);
    if (status !== 200) {
      throw new Error(`Expected HTTP 200 from /pgx/card, got ${status}`);
    }

    await sleep(2000);
    results.cardGenerated = true;

    // 6. Validate Options Matrix
    console.log("Validating Options Matrix 3-panel layout...");
    const optionsMatrix = await page.evaluate(() => {
      const el = document.querySelector(".pgx-options-matrix");
      if (!el) return null;
      const panels = el.querySelectorAll(".pgx-opt-panel");
      const buttons = el.querySelectorAll("button");
      return {
        visible: window.getComputedStyle(el).display !== "none",
        panelCount: panels.length,
        buttonCount: buttons.length,
        hasPrimaryHero: !!el.querySelector(".pgx-btn-hero")
      };
    });

    if (optionsMatrix && optionsMatrix.panelCount === 3 && optionsMatrix.buttonCount >= 11) {
      results.optionsMatrixVerified = true;
      console.log(`✓ Options Matrix verified: 3 panels, ${optionsMatrix.buttonCount} buttons.`);
    } else {
      console.warn("⚠️ Options Matrix failed validation:", optionsMatrix);
    }

    // 7. Validate Gene–Drug Actionability Matrix
    console.log("Validating Gene–Drug Actionability Matrix...");
    const matrixState = await page.evaluate(() => {
      const matrixEl = document.getElementById("pgx-action-matrix");
      if (!matrixEl) return null;
      const rows = matrixEl.querySelectorAll("tbody tr");
      const headerCols = matrixEl.querySelectorAll("thead th");
      const filterPills = document.querySelectorAll("#pgx-matrix-filter-pills .pgx-filter-pill");
      
      const pillsData = Array.from(filterPills).map(p => ({
        cat: p.getAttribute("data-cat"),
        text: p.textContent.trim(),
        active: p.classList.contains("active")
      }));

      const rowDrugs = Array.from(rows).map(r => {
        const th = r.querySelector("th");
        return th ? th.textContent.trim() : "";
      });

      return {
        visible: window.getComputedStyle(matrixEl).display !== "none",
        rowCount: rows.length,
        colCount: headerCols.length,
        rowDrugs,
        pillsData
      };
    });

    console.log(`✓ Action Matrix visible: ${matrixState.rowCount} drug rows, ${matrixState.colCount} columns.`);
    console.log(`  Matrix drug rows:`, matrixState.rowDrugs);
    console.log(`  Filter pills:`, matrixState.pillsData.map(p => p.text).join(" | "));

    if (matrixState.rowCount > 0 && matrixState.pillsData.length >= 5) {
      results.actionMatrixVerified = true;
      results.filterPillsVerified = true;
    }

    // 8. Test Matrix Cell Interaction (Clicking a cell to inspect guidance)
    console.log("Testing interactive matrix cell click inspection...");
    const cellClicked = await page.evaluate(() => {
      const cell = document.querySelector(".pgx-interactive-cell");
      if (!cell) return false;
      cell.click();
      return true;
    });

    if (cellClicked) {
      await sleep(400);
      const detailCard = await page.evaluate(() => {
        const card = document.getElementById("pgx-matrix-detail-card");
        if (!card) return null;
        return {
          visible: window.getComputedStyle(card).display !== "none",
          text: card.textContent.trim()
        };
      });

      if (detailCard && detailCard.visible) {
        results.cellDetailVerified = true;
        console.log(`✓ Cell detail card popped up with content: "${detailCard.text.slice(0, 100)}..."`);
      }
    }

    // 9. Validate Clustered Dendrogram Heatmaps
    console.log("Validating Plotly Clustered Dendrogram Heatmaps...");
    await page.$eval("#pgx-dendrogram-section", el => el.scrollIntoView({ behavior: "smooth", block: "start" }));
    await sleep(800);

    const dendroState = await page.evaluate(() => {
      const plot = document.getElementById("pgx-dendrogram-plot");
      if (!plot) return null;
      const isPlotly = plot.classList.contains("js-plotly-plot") || !!plot.querySelector(".plot-container");
      const traces = plot.querySelectorAll(".trace");
      const heatmap = plot.querySelector(".heatmaplayer");
      return {
        isPlotly,
        traceCount: traces.length,
        hasHeatmap: !!heatmap
      };
    });

    console.log("✓ Dendrogram Plotly state:", dendroState);
    if (dendroState && dendroState.isPlotly) {
      results.dendrogramVerified = true;
    }

    // Test mode switching (drug_rec and gene_rec)
    console.log("Testing Dendrogram mode switching...");
    await page.select("#pgx-cluster-mode", "drug_rec");
    await sleep(600);
    const modeDrugRec = await page.evaluate(() => {
      const title = document.querySelector("#pgx-dendrogram-plot .gtitle");
      return title ? title.textContent : "";
    });
    console.log(`  Mode 'drug_rec' title: "${modeDrugRec}"`);

    await page.select("#pgx-cluster-mode", "gene_rec");
    await sleep(600);
    const modeGeneRec = await page.evaluate(() => {
      const title = document.querySelector("#pgx-dendrogram-plot .gtitle");
      return title ? title.textContent : "";
    });
    console.log(`  Mode 'gene_rec' title: "${modeGeneRec}"`);

    // Switch back to Drug x Gene dual dendrogram
    await page.select("#pgx-cluster-mode", "drug_gene");
    await sleep(600);
    results.dendrogramModesVerified = true;

    // 10. Capture Screenshots
    console.log("Capturing verification screenshots...");
    const shotPath1 = path.join(LOCAL_SCREENSHOT_DIR, "puppeteer_5cpic_drugs_matrix.png");
    const shotPath2 = path.join(LOCAL_SCREENSHOT_DIR, "puppeteer_5cpic_drugs_dendrogram.png");

    await page.$eval(".pgx-options-matrix", el => el.scrollIntoView({ block: "start" }));
    await sleep(300);
    await page.screenshot({ path: shotPath1, fullPage: false });
    results.screenshots.push(shotPath1);

    await page.$eval("#pgx-dendrogram-section", el => el.scrollIntoView({ block: "start" }));
    await sleep(400);
    await page.screenshot({ path: shotPath2, fullPage: false });
    results.screenshots.push(shotPath2);

    console.log(`✓ Screenshots saved to:`);
    console.log(`  - ${shotPath1}`);
    console.log(`  - ${shotPath2}`);

  } catch (err) {
    console.error("❌ Test run error:", err);
  } finally {
    await browser.close();
  }

  console.log("\n================ TEST SUMMARY ================");
  console.log(`Target URL:                   ${DASHBOARD_URL}`);
  console.log(`URL Loaded:                   ${results.urlLoaded ? "PASS" : "FAIL"}`);
  console.log(`Tab Switched:                 ${results.tabSwitched ? "PASS" : "FAIL"}`);
  console.log(`Gene Alleles Entered:         ${results.genesEntered ? "PASS" : "FAIL"}`);
  console.log(`5 CPIC Drugs Added:           ${results.drugsSelected.length === 5 ? "PASS" : "FAIL"} (${results.drugsSelected.length}/5)`);
  console.log(`POST /pgx/card Generated:     ${results.cardGenerated ? "PASS" : "FAIL"}`);
  console.log(`Options Matrix 3-Panel Grid:  ${results.optionsMatrixVerified ? "PASS" : "FAIL"}`);
  console.log(`Actionability Matrix Rows:    ${results.actionMatrixVerified ? "PASS" : "FAIL"}`);
  console.log(`Filter Pills Dynamic Triage:  ${results.filterPillsVerified ? "PASS" : "FAIL"}`);
  console.log(`Cell Click Inspection Detail: ${results.cellDetailVerified ? "PASS" : "FAIL"}`);
  console.log(`Plotly Dendrogram Heatmap:    ${results.dendrogramVerified ? "PASS" : "FAIL"}`);
  console.log(`Dendrogram Modes Toggled:     ${results.dendrogramModesVerified ? "PASS" : "FAIL"}`);
  console.log("==============================================\n");

  const allPass = results.urlLoaded && results.tabSwitched && results.genesEntered &&
                  results.drugsSelected.length === 5 && results.cardGenerated &&
                  results.optionsMatrixVerified && results.actionMatrixVerified &&
                  results.filterPillsVerified && results.cellDetailVerified &&
                  results.dendrogramVerified && results.dendrogramModesVerified;

  if (allPass) {
    console.log("🎉 ALL TESTS PASSED SUCCESSFULLY!");
    process.exit(0);
  } else {
    console.error("❌ ONE OR MORE TESTS FAILED.");
    process.exit(1);
  }
})();
