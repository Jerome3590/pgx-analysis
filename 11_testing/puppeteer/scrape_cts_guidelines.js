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

  console.log("Navigating to CTS Guide to Authors...");
  await page.goto(
    "https://ascpt.onlinelibrary.wiley.com/page/journal/17528062/guidetoauthors",
    { waitUntil: "networkidle2", timeout: 30000 }
  );

  const content = await page.evaluate(() => document.body.innerText);

  const fs = require("fs");
  fs.writeFileSync("cts_guidelines_raw.txt", content, "utf8");
  console.log(`Saved ${content.length} chars to cts_guidelines_raw.txt`);

  await browser.close();
})();
