"use strict";

/**
 * Attach to Chrome CDP and create one NotebookLM notebook per PGx UC.
 *
 * Source path: paste Google Drive file view URLs (README + screenshots).
 * Do NOT use the Drive picker iframe. Do NOT use Fast Research / website scrape.
 *
 * Connect pattern matches cana-forge google_alerts.js:
 *   puppeteer.connect({ browserURL }) then browser.disconnect() — never browser.close().
 *
 * Default CDP: http://127.0.0.1:9226
 */
const fs = require("fs");
const path = require("path");
const puppeteer = require("puppeteer");

const DEFAULT_BROWSER_URL = process.env.CHROME_BROWSER_URL || "http://127.0.0.1:9226";
const REPO_ROOT = path.resolve(__dirname, "..", "..");
const URLS_PATH = process.env.NLM_URLS_JSON
  || path.join(REPO_ROOT, "10_risk_dashboard", "docs", "use_case_training", "notebooklm_drive_urls.json");
const PROMPT_PATH = path.join(REPO_ROOT, "10_risk_dashboard", "docs", "use_case_training", "NOTEBOOKLM_PROMPTS.md");
const DRIVE_PACK = process.env.PGX_UC_DRIVE_ROOT || "G:\\My Drive\\PGx_Dashboard_Use_Cases";
const REPO_PACK = path.join(REPO_ROOT, "10_risk_dashboard", "docs", "use_case_training");
const RESULTS = path.join(REPO_ROOT, "11_testing", "results");
const EMAIL = "jerome.dixon90@gmail.com";
const FORBIDDEN = ["canallc", "jdixon@canallc"];
const NLM_HOSTS = ["notebooklm.google.com", "notebook.google.com"];

function parseArgs(argv) {
  const out = {
    browserUrl: DEFAULT_BROWSER_URL,
    urls: URLS_PATH,
    only: "",
    dryRun: false,
    skipGenerate: false,
    skipDownload: false,
    waitLoginMs: 180000,
  };
  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--dry-run") out.dryRun = true;
    else if (arg === "--skip-generate") out.skipGenerate = true;
    else if (arg === "--skip-download") out.skipDownload = true;
    else if (arg === "--browser-url") {
      i += 1;
      out.browserUrl = argv[i] || out.browserUrl;
    } else if (arg === "--urls") {
      i += 1;
      out.urls = argv[i] || out.urls;
    } else if (arg === "--only") {
      i += 1;
      out.only = String(argv[i] || "").toUpperCase();
    } else if (arg === "--wait-login-ms") {
      i += 1;
      out.waitLoginMs = Number(argv[i] || out.waitLoginMs);
    }
  }
  return out;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function loadJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function promptText() {
  const body = fs.readFileSync(PROMPT_PATH, "utf8").trim();
  const extra = [
    "Hard rule: use only the uploaded screenshots as visual source of truth.",
    "Do not generate dashboard images, mock UIs, or fictional screens.",
    "Describe or cite the real PNGs; slides should embed or refer to uploaded screenshots.",
  ].join(" ");
  return `${body}\n\n${extra}`;
}

function destDirs(uc) {
  return [
    path.join(DRIVE_PACK, uc, "notebooklm"),
    path.join(REPO_PACK, uc, "notebooklm"),
  ];
}

function ensureDirs(uc) {
  for (const dir of destDirs(uc)) fs.mkdirSync(dir, { recursive: true });
}

async function dumpPage(page, name) {
  fs.mkdirSync(RESULTS, { recursive: true });
  const data = await page.evaluate(() => {
    const buttons = [...document.querySelectorAll("button, a, [role='button']")]
      .map((el) => ({
        text: (el.innerText || "").trim().slice(0, 120),
        aria: (el.getAttribute("aria-label") || "").trim().slice(0, 160),
      }))
      .filter((x) => x.text || x.aria)
      .slice(0, 80);
    const inputs = [...document.querySelectorAll("input, textarea, [contenteditable='true']")]
      .map((el) => ({
        tag: el.tagName,
        type: el.type || "",
        placeholder: el.placeholder || "",
        aria: el.getAttribute("aria-label") || "",
      }))
      .slice(0, 30);
    return {
      href: location.href,
      title: document.title,
      buttons,
      inputs,
      body: (document.body.innerText || "").slice(0, 2500),
    };
  });
  fs.writeFileSync(path.join(RESULTS, `nlm_${name}.json`), JSON.stringify(data, null, 2));
  try {
    await page.screenshot({ path: path.join(RESULTS, `nlm_${name}.png`), fullPage: false });
  } catch (err) {
    console.log("SHOT fail", name, err.message);
  }
  return data;
}

function pageLooksForbidden(text) {
  const low = String(text || "").toLowerCase();
  return FORBIDDEN.some((token) => low.includes(token));
}

async function pagesOf(browser) {
  const pages = [];
  for (const page of await browser.pages()) {
    pages.push(page);
  }
  return pages;
}

async function findNotebookPage(browser) {
  const pages = await pagesOf(browser);
  const nlm = pages.find((page) => NLM_HOSTS.some((host) => page.url().includes(host)));
  if (nlm) return nlm;
  const accounts = pages.find((page) => page.url().includes("accounts.google.com"));
  return accounts || pages[0];
}

async function waitSignedIn(page, waitLoginMs) {
  const deadline = Date.now() + waitLoginMs;
  while (Date.now() < deadline) {
    const url = page.url();
    const body = await page.evaluate(() => (document.body && document.body.innerText) || "").catch(() => "");
    if (pageLooksForbidden(`${url}\n${body}`)) {
      throw new Error("Forbidden CANA / canallc account is visible. Sign in as jerome.dixon90@gmail.com only.");
    }
    const onNlm = NLM_HOSTS.some((host) => url.includes(host)) && !url.includes("accounts.google.com");
    const signedChooser = body.toLowerCase().includes(EMAIL) && url.includes("accounts.google.com");
    if (onNlm) {
      console.log("signed_in_notebook", url);
      return true;
    }
    if (signedChooser) {
      const chip = await page.$(`[data-identifier="${EMAIL}"]`);
      if (chip) {
        console.log("clicking jerome.dixon90 chip");
        await chip.click();
        await sleep(2500);
        continue;
      }
    }
    if (url.includes("accounts.google.com") && /challenge\/pwd|password/i.test(`${url}\n${body}`)) {
      console.log("BLOCKER_PASSWORD_REQUIRED — finish sign-in as jerome.dixon90 in the debug Chrome window.");
      return false;
    }
    if (url.includes("accounts.google.com")) {
      console.log("waiting_for_login", url.slice(0, 120));
    }
    await sleep(3000);
  }
  return false;
}

async function clickFirst(page, locators, label) {
  for (const locator of locators) {
    const handle = await page.$(locator).catch(() => null);
    if (handle) {
      await handle.click();
      console.log("clicked", label, locator);
      return true;
    }
  }
  const clicked = await page.evaluate((needles) => {
    const nodes = [...document.querySelectorAll("button, a, [role='button'], [role='menuitem']")];
    const hit = nodes.find((el) => {
      const text = `${el.innerText || ""} ${el.getAttribute("aria-label") || ""}`.toLowerCase();
      return needles.some((needle) => text.includes(needle));
    });
    if (!hit) return false;
    hit.click();
    return true;
  }, locators.filter((x) => !x.startsWith("[") && !x.startsWith("#")));
  if (clicked) {
    console.log("clicked_text", label);
    return true;
  }
  console.log("missing", label);
  return false;
}

async function clickByText(page, needles, label) {
  const clicked = await page.evaluate((want) => {
    const nodes = [...document.querySelectorAll("button, a, [role='button'], [role='tab'], [role='menuitem'], span, div")];
    const hit = nodes.find((el) => {
      const text = `${el.innerText || ""}\n${el.getAttribute("aria-label") || ""}`.toLowerCase();
      return want.every((needle) => text.includes(needle));
    });
    if (!hit) return false;
    (hit.closest("button, a, [role='button']") || hit).click();
    return true;
  }, needles.map((n) => n.toLowerCase()));
  console.log(clicked ? "clicked" : "missing", label);
  return clicked;
}

async function renameNotebook(page, title) {
  const renamed = await page.evaluate((want) => {
    const input = [...document.querySelectorAll("input")]
      .find((el) => /untitled|notebook/i.test(`${el.value || ""} ${el.getAttribute("aria-label") || ""}`));
    if (!input) return false;
    input.focus();
    input.value = want;
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
    return true;
  }, title);
  if (!renamed) {
    const handle = await page.$("input");
    if (handle) {
      await handle.click({ clickCount: 3 });
      await handle.type(title, { delay: 15 });
    }
  }
  await page.keyboard.press("Enter").catch(() => {});
  console.log("rename", title, "ok=", renamed);
}

async function pasteUrl(page, url) {
  const filled = await page.evaluate((value) => {
    const candidates = [...document.querySelectorAll("input, textarea")]
      .filter((el) => {
        const hint = `${el.placeholder || ""} ${el.getAttribute("aria-label") || ""} ${el.type || ""}`.toLowerCase();
        return (
          el.type === "url"
          || hint.includes("url")
          || hint.includes("website")
          || hint.includes("link")
          || hint.includes("paste")
          || hint.includes("http")
        );
      });
    const box = candidates[0] || document.querySelector("input[type='url']");
    if (!box) return false;
    box.focus();
    box.value = value;
    box.dispatchEvent(new Event("input", { bubbles: true }));
    box.dispatchEvent(new Event("change", { bubbles: true }));
    return true;
  }, url);
  if (!filled) {
    const box = await page.$("input[type='url'], textarea, input[type='text']");
    if (!box) throw new Error("No URL paste field after opening Websites/URL add");
    await box.click({ clickCount: 3 });
    await box.type(url, { delay: 8 });
  }
  const submitted = await clickByText(page, ["insert"], "insert")
    || await clickByText(page, ["add"], "add")
    || await clickByText(page, ["submit"], "submit")
    || await page.keyboard.press("Enter").then(() => true);
  console.log("pasted_url", url, "submitted=", submitted);
  await sleep(2500);
}

async function addSourcesByUrl(page, files) {
  const webAlready = await page.evaluate(() => {
    const nodes = [...document.querySelectorAll("button, [role='button']")];
    return nodes.some((el) => /^(web)$/i.test((el.getAttribute("aria-label") || "").trim()));
  });
  const opened = webAlready
    || await clickFirst(page, ["[aria-label='Add source']", "[aria-label='Add sources']"], "add_source_aria")
    || await clickByText(page, ["add sources"], "add_sources")
    || await clickByText(page, ["add source"], "add_source")
    || await clickByText(page, ["add a source"], "add_a_source");
  if (!opened) throw new Error("Could not open Add sources");
  await sleep(1500);

  const websites = await clickByText(page, ["websites"], "websites");
  if (!websites) {
    const web = await clickByText(page, ["web"], "web_chip");
    if (!web) throw new Error("Could not open Websites/URL paste (refusing Drive picker)");
  }
  await sleep(1200);

  const picker = await page.evaluate(() =>
    [...document.querySelectorAll("iframe")].some((f) => /picker|onepick|drive.google/i.test(f.src || ""))
  );
  if (picker) {
    await page.keyboard.press("Escape").catch(() => {});
    throw new Error("Drive picker iframe appeared; refusing that path. Use Websites URL paste.");
  }

  for (const file of files) {
    await pasteUrl(page, file.url);
  }
}

async function pasteStudioPrompt(page, text) {
  const box = await page.$("textarea[aria-label='Query box'], textarea[placeholder*='Ask a question']");
  if (box) {
    await box.click();
    await box.type(text.slice(0, 4000), { delay: 2 });
    console.log("pasted_prompt_chat", text.length);
    return;
  }
  const typed = await page.evaluate((value) => {
    const areas = [...document.querySelectorAll("textarea, [contenteditable='true']")];
    const target = areas.find((el) => /studio|prompt|overview|custom/i.test(
      `${el.placeholder || ""} ${el.getAttribute("aria-label") || ""}`
    )) || areas[areas.length - 1];
    if (!target) return false;
    target.focus();
    if (target.tagName === "TEXTAREA") target.value = value;
    else target.textContent = value;
    target.dispatchEvent(new Event("input", { bubbles: true }));
    return true;
  }, text);
  console.log("pasted_prompt", typed);
}

async function generateStudio(page) {
  for (const label of ["Audio Overview", "Video Overview", "Slide Deck"]) {
    const ok = await clickByText(page, [label.toLowerCase()], label);
    if (ok) await sleep(2000);
    const customize = await page.$("textarea, [contenteditable='true']");
    if (customize) {
      await pasteStudioPrompt(page, promptText());
      await clickByText(page, ["generate"], "generate")
        || await clickByText(page, ["create"], "create");
      await sleep(3000);
    }
  }
}

async function setDownloadPath(page, dest) {
  fs.mkdirSync(dest, { recursive: true });
  const client = await page.target().createCDPSession();
  try {
    await client.send("Browser.setDownloadBehavior", {
      behavior: "allow",
      downloadPath: dest,
      eventsEnabled: true,
    });
  } catch (_err) {
    await client.send("Page.setDownloadBehavior", {
      behavior: "allow",
      downloadPath: dest,
    });
  }
  return dest;
}

function copyNewDownloads(fromDir, destDirsList, startedAt) {
  if (!fs.existsSync(fromDir)) return [];
  const copied = [];
  for (const name of fs.readdirSync(fromDir)) {
    if (name.endsWith(".crdownload")) continue;
    const src = path.join(fromDir, name);
    const st = fs.statSync(src);
    if (!st.isFile() || st.mtimeMs < startedAt - 1000) continue;
    for (const dest of destDirsList) {
      fs.mkdirSync(dest, { recursive: true });
      const target = path.join(dest, name);
      fs.copyFileSync(src, target);
      copied.push(target);
    }
  }
  return copied;
}

async function downloadIfPresent(page, uc) {
  const dests = destDirs(uc);
  ensureDirs(uc);
  const startedAt = Date.now();
  const primary = dests[0];
  await setDownloadPath(page, primary);
  for (const label of ["Download", "Export", "Save"]) {
    await clickByText(page, [label.toLowerCase()], label);
    await sleep(1500);
  }
  const chromeDownloads = path.join(process.env.USERPROFILE || "", "Downloads");
  const copied = [
    ...copyNewDownloads(primary, dests, startedAt),
    ...copyNewDownloads(chromeDownloads, dests, startedAt),
  ];
  console.log("downloads_copied", copied.length);
  return copied;
}

async function createNotebook(page, notebook) {
  const home = NLM_HOSTS.find((host) => page.url().includes(host)) ? page.url() : "https://notebooklm.google.com/";
  if (!/\/notebook\/[0-9a-f-]{8,}/i.test(page.url()) || page.url().includes("notebooklm.google.com/?") || page.url().endsWith("notebooklm.google.com/")) {
    await page.goto("https://notebooklm.google.com/", { waitUntil: "domcontentloaded", timeout: 60000 });
    await sleep(2500);
  } else {
    await page.goto("https://notebooklm.google.com/", { waitUntil: "domcontentloaded", timeout: 60000 });
    await sleep(2500);
  }
  const created = await clickByText(page, ["new notebook"], "new_notebook")
    || await clickByText(page, ["create notebook"], "create_notebook");
  if (!created) throw new Error("New notebook button not found");
  await page.waitForFunction(
    () => /\/notebook\/[0-9a-f-]{8,}/i.test(location.href),
    { timeout: 20000 }
  ).catch(() => {});
  if (!/\/notebook\/[0-9a-f-]{8,}/i.test(page.url())) {
    throw new Error(`New notebook did not open (still at ${page.url()})`);
  }
  await sleep(2500);
  await renameNotebook(page, notebook.title);
  await addSourcesByUrl(page, notebook.files);
  await sleep(4000);
  await pasteStudioPrompt(page, promptText());
}

async function main() {
  const args = parseArgs(process.argv);
  if (!fs.existsSync(args.urls)) {
    throw new Error(`urls.json missing: ${args.urls}. Run utility_scripts/resolve_pgx_uc_drive_urls.py first.`);
  }
  const pack = loadJson(args.urls);
  let notebooks = pack.notebooks || [];
  if (args.only) {
    notebooks = notebooks.filter((nb) => nb.uc.toUpperCase().startsWith(args.only) || nb.title.toUpperCase().includes(args.only));
  }
  if (!notebooks.length) throw new Error("No notebooks selected");
  for (const nb of notebooks) {
    if (!nb.files || !nb.files.length) throw new Error(`${nb.uc} has no resolved Drive URLs`);
    const bad = nb.files.filter((f) => !/^https:\/\/drive\.google\.com\/file\/d\/[^/]+\/view/.test(f.url));
    if (bad.length) throw new Error(`${nb.uc} has invalid Drive view URLs`);
  }

  if (args.dryRun) {
    for (const nb of notebooks) {
      console.log(nb.title, nb.file_count, "urls");
      for (const file of nb.files) console.log(" ", file.rel, file.url);
    }
    return;
  }

  console.log(`Connecting to existing Chrome at ${args.browserUrl}`);
  const browser = await puppeteer.connect({
    browserURL: args.browserUrl,
    defaultViewport: null,
  });
  let failed = false;
  try {
    const page = await findNotebookPage(browser);
    page.setDefaultTimeout(20000);
    console.log("start", page.url());
    await dumpPage(page, "attach_start");
    const ready = await waitSignedIn(page, args.waitLoginMs);
    if (!ready) {
      await dumpPage(page, "attach_need_login");
      throw new Error(
        "Debug Chrome is not signed into NotebookLM as jerome.dixon90@gmail.com. "
        + "In the 9226 window, finish Google sign-in (never canallc), then re-run."
      );
    }
    await dumpPage(page, "attach_signed_in");

    for (const notebook of notebooks) {
      console.log("====", notebook.title, notebook.file_count, "sources ====");
      ensureDirs(notebook.uc);
      await createNotebook(page, notebook);
      await dumpPage(page, `${notebook.uc.toLowerCase()}_sources`);
      if (!args.skipGenerate) {
        await generateStudio(page);
        await dumpPage(page, `${notebook.uc.toLowerCase()}_studio`);
      }
      if (!args.skipDownload) {
        await downloadIfPresent(page, notebook.uc);
      }
      await sleep(2000);
    }
  } catch (err) {
    failed = true;
    console.error("FAIL", err.message);
  } finally {
    await browser.disconnect();
    console.log("disconnected (Chrome left open)");
  }
  if (failed) process.exitCode = 1;
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
