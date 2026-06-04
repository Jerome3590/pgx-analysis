"use strict";

const puppeteer = require("puppeteer");
const fs = require("fs");
const path = require("path");

const OUT_DIR = path.join(__dirname, "cts_articles");
if (!fs.existsSync(OUT_DIR)) fs.mkdirSync(OUT_DIR);

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

  // Step 1: Find the most recent issue URL from the LOI page
  console.log("Loading list of issues...");
  await page.goto("https://ascpt.onlinelibrary.wiley.com/loi/17528062", {
    waitUntil: "networkidle2",
    timeout: 30000,
  });

  const latestIssueUrl = await page.evaluate(() => {
    // Issue links are typically /toc/17528062/YYYY/NN/N
    const links = Array.from(document.querySelectorAll('a[href*="/toc/"]'));
    for (const a of links) {
      if (a.href.includes("17528062")) return a.href;
    }
    // Fallback: any link with /toc/
    const all = Array.from(document.querySelectorAll('a[href]'));
    for (const a of all) {
      if (a.href.match(/\/toc\/17528062\/\d/)) return a.href;
    }
    return null;
  });

  console.log("Latest issue URL:", latestIssueUrl);

  let articleLinks = [];
  if (latestIssueUrl) {
    await page.goto(latestIssueUrl, { waitUntil: "networkidle2", timeout: 30000 });
    articleLinks = await page.evaluate(() => {
      const seen = new Set();
      const results = [];
      // Article links in TOC are usually inside .issue-item or similar, pointing to /doi/abs/ or /doi/full/
      const candidates = Array.from(document.querySelectorAll('a[href]'));
      for (const a of candidates) {
        const href = a.href;
        const text = a.innerText.trim();
        if (
          href.match(/\/doi\/(abs|full|10\.)\/10\.1111\/cts/) &&
          !seen.has(href) &&
          text.length > 20
        ) {
          seen.add(href);
          results.push({ href, title: text.slice(0, 150) });
          if (results.length >= 3) break;
        }
      }
      return results;
    });
  }

  // Fallback: use known recent article DOIs from CTS
  if (articleLinks.length < 3) {
    console.log("TOC scrape insufficient, using known recent CTS DOIs...");
    const fallback = [
      { href: "https://ascpt.onlinelibrary.wiley.com/doi/full/10.1111/cts.70134", title: "Recent CTS article 1" },
      { href: "https://ascpt.onlinelibrary.wiley.com/doi/full/10.1111/cts.70120", title: "Recent CTS article 2" },
      { href: "https://ascpt.onlinelibrary.wiley.com/doi/full/10.1111/cts.70110", title: "Recent CTS article 3" },
    ];
    for (const f of fallback) {
      if (articleLinks.length < 3) articleLinks.push(f);
    }
  }

  console.log(`Found ${articleLinks.length} article(s):`);
  articleLinks.forEach((a, i) => console.log(`  ${i + 1}. ${a.title || "(no title)"} — ${a.href}`));

  // Step 2: Scrape each article full text
  for (let i = 0; i < articleLinks.length; i++) {
    const { href, title } = articleLinks[i];
    console.log(`\nScraping article ${i + 1}: ${href}`);
    try {
      await page.goto(href, { waitUntil: "networkidle2", timeout: 30000 });
      const content = await page.evaluate(() => document.body.innerText);
      const filename = path.join(OUT_DIR, `article_${i + 1}.txt`);
      fs.writeFileSync(filename, `URL: ${href}\nTITLE: ${title}\n\n${content}`, "utf8");
      console.log(`  Saved ${content.length} chars → ${filename}`);
    } catch (e) {
      console.error(`  Error scraping article ${i + 1}: ${e.message}`);
    }
  }

  await browser.close();
  console.log("\nDone.");
})();
