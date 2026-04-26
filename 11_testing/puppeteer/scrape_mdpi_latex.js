"use strict";

const puppeteer = require("puppeteer");

(async () => {
  const browser = await puppeteer.launch({
    headless: "new",
    args: [
      "--no-sandbox",
      "--disable-setuid-sandbox",
      "--disable-blink-features=AutomationControlled",
      "--disable-dev-shm-usage",
    ],
  });

  const page = await browser.newPage();

  await page.setUserAgent(
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
  );
  await page.setExtraHTTPHeaders({ "Accept-Language": "en-US,en;q=0.9" });

  await page.evaluateOnNewDocument(() => {
    Object.defineProperty(navigator, "webdriver", { get: () => undefined });
  });

  console.log("Navigating to MDPI LaTeX page...");
  await page.goto("https://www.mdpi.com/journal/dhi/instructions", {
    waitUntil: "networkidle2",
    timeout: 30000,
  });

  const content = await page.evaluate(() => document.body.innerText);
  console.log("\n=== MDPI DHI Author Instructions ===\n");
  // Find the article types table section
  const lower = content.toLowerCase();
  const markers = ['article types', 'manuscript length', 'word limit', 'max. words', 'maximum word'];
  for (const m of markers) {
    const idx = lower.indexOf(m);
    if (idx !== -1) {
      console.log(`\nFound "${m}" at position ${idx}:`);
      console.log(content.slice(Math.max(0, idx - 50), idx + 3000));
      break;
    }
  }
  // Also print the first 2000 chars to see page structure
  console.log("\n--- PAGE TOP ---");
  console.log(content.slice(0, 2000));

  await browser.close();
})();
