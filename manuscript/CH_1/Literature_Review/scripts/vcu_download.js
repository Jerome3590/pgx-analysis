/**
 * vcu_download.js
 * Puppeteer-based downloader for paywalled articles via VCU Library EZProxy.
 *
 * Flow:
 *   1. Read credentials from secrets/secrets.txt
 *   2. Load saved session cookies (secrets/session_cookies.json) if present
 *   3. For each article in screened_missing_fulltext.csv:
 *        a. Navigate to proxy URL (DOI path or title search fallback)
 *        b. If not authenticated → VCU SSO login → wait for Duo push
 *        c. Detect PDF response → save to data/scholar_pdfs/{hsh_id}.pdf
 *        d. Log result to scripts/vcu_download_log.csv
 *
 * Usage:
 *   node scripts/vcu_download.js                         # download all
 *   node scripts/vcu_download.js --login-only            # login + save cookies only
 *   node scripts/vcu_download.js --duo-passcode=398635   # pass current Duo code directly
 *   node scripts/vcu_download.js --limit 10              # first 10 only (test)
 *   node scripts/vcu_download.js --headed                # show browser
 */

const puppeteer  = require('puppeteer-extra');
const StealthPlugin = require('puppeteer-extra-plugin-stealth');
puppeteer.use(StealthPlugin());
const fs         = require('fs');
const path       = require('path');
const { parse }  = require('csv-parse/sync');
const { stringify } = require('csv-stringify/sync');

// ── Paths ─────────────────────────────────────────────────────────────────────
const ROOT         = path.resolve(__dirname, '..');
const SECRETS_FILE = path.join(ROOT, 'secrets', 'secrets.txt');
const COOKIES_FILE = path.join(ROOT, 'secrets', 'session_cookies.json');
const INPUT_CSV    = path.join(ROOT, 'scripts', 'screened_doi_map.csv');
const UPAYWALL_LOG = path.join(ROOT, 'scripts', 'unpaywall_log.csv');
const LOG_FILE     = path.join(ROOT, 'scripts', 'vcu_download_log.csv');
const PDF_DIR      = path.join(ROOT, 'data', 'scholar_pdfs');
fs.mkdirSync(PDF_DIR,                          { recursive: true });
fs.mkdirSync(path.join(ROOT, 'secrets'),       { recursive: true });

// ── Parse args ────────────────────────────────────────────────────────────────
const args        = process.argv.slice(2);
const LOGIN_ONLY  = args.includes('--login-only');
const HEADED      = args.includes('--headed') || LOGIN_ONLY;
const LIMIT       = (() => { const i = args.indexOf('--limit'); return i >= 0 ? parseInt(args[i+1]) : 0; })();
const DUO_ARG     = (() => { const m = args.find(a => a.startsWith('--duo-passcode=')); return m ? m.split('=')[1] : ''; })();
const INPUT_ARG   = (() => { const i = args.indexOf('--input'); return i >= 0 ? args[i+1] : null; })();
if (DUO_ARG) process.env.DUO_PASSCODE = DUO_ARG;  // env var consumed by waitForDuo

// ── Load credentials ──────────────────────────────────────────────────────────
function loadSecrets() {
  if (!fs.existsSync(SECRETS_FILE)) {
    console.error(`ERROR: ${SECRETS_FILE} not found.`);
    console.error(`Copy secrets.example.txt → secrets/secrets.txt and fill in your credentials.`);
    process.exit(1);
  }
  const lines = fs.readFileSync(SECRETS_FILE, 'utf8').split('\n');
  const cfg   = {};
  for (const line of lines) {
    const m = line.match(/^\s*([^#=\s][^=]*)=(.*)$/);
    if (m) cfg[m[1].trim()] = m[2].trim();
  }
  if (!cfg.username || !cfg.password) {
    console.error('ERROR: secrets.txt must contain username= and password=');
    process.exit(1);
  }
  cfg.proxy_base = cfg.proxy_base || 'https://proxy.library.vcu.edu/login?url=';
  return cfg;
}

// ── DOI → publisher URL resolution ───────────────────────────────────────────
const https = require('https');
const http  = require('http');

function resolveDoi(doi) {
  return new Promise((resolve) => {
    const url = `https://doi.org/${doi}`;
    function follow(u, hops) {
      if (hops > 8) return resolve(null);
      const mod = u.startsWith('https') ? https : http;
      const req = mod.get(u, { headers: { 'Accept': 'text/html' } }, res => {
        if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
          const loc = res.headers.location;
          follow(loc.startsWith('http') ? loc : new URL(loc, u).href, hops + 1);
        } else {
          resolve(u);
        }
        res.resume();
      });
      req.on('error', () => resolve(null));
      req.setTimeout(8000, () => { req.destroy(); resolve(null); });
    }
    follow(url, 0);
  });
}

// ── Load candidates ───────────────────────────────────────────────────────────
function loadCandidates() {
  // --input accepts any CSV with screened_pmc_id + doi + title columns
  // (screened_doi_map.csv or vcu_queue_with_dois.csv)
  const csvFile = INPUT_ARG ? path.resolve(ROOT, INPUT_ARG) : INPUT_CSV;
  if (!fs.existsSync(csvFile)) {
    console.error(`ERROR: ${csvFile} not found.`);
    if (!INPUT_ARG) console.error(`Run: python scripts/_check_doi_match.py`);
    else            console.error(`Run: python scripts/_build_vcu_doi_map.py`);
    process.exit(1);
  }

  // Skip PDF files already on disk OR already logged OK
  const done = new Set();
  if (fs.existsSync(LOG_FILE)) {
    try {
      const rows = parse(fs.readFileSync(LOG_FILE, 'utf8'), { columns: true, skip_empty_lines: true, relax_column_count: true });
      for (const row of rows) {
        if (row.status === 'ok') done.add(row.hsh_id);
      }
    } catch (_) { /* corrupt log — ignore, will append fresh */ }
  }
  // Also skip if JSON already exists in scholar_json/
  const scholarJsonDir = path.join(ROOT, 'data', 'scholar_json');
  const jsonSet = new Set();
  if (fs.existsSync(scholarJsonDir)) {
    for (const f of fs.readdirSync(scholarJsonDir)) {
      if (f.endsWith('.json')) jsonSet.add(f.replace(/\.json$/, ''));
    }
  }

  // Supports both screened_doi_map.csv and vcu_queue_with_dois.csv schemas
  const rows = parse(fs.readFileSync(csvFile, 'utf8'), { columns: true, skip_empty_lines: true });
  const candidates = rows
    .filter(r => {
      const id  = r.screened_pmc_id || r.pmc_id || '';
      const doi = r.doi || '';
      return id && doi && !done.has(id) && !jsonSet.has(id);
    })
    .map(r => ({
      hsh_id: r.screened_pmc_id || r.pmc_id || `article_${r.article_id}`,
      title:  r.title || '',
      doi:    r.doi,
    }));

  console.log(`Input CSV:  ${csvFile}  (${rows.length} rows, ${candidates.length} need download)`);
  return LIMIT > 0 ? candidates.slice(0, LIMIT) : candidates;
}

// ── CSV log helpers ───────────────────────────────────────────────────────────
const LOG_FIELDS = ['hsh_id', 'title', 'doi', 'proxy_url', 'status', 'bytes', 'timestamp'];

function appendLog(row) {
  const header = !fs.existsSync(LOG_FILE);
  const line   = stringify([row], { header, columns: LOG_FIELDS, record_delimiter: '\r\n' });
  fs.appendFileSync(LOG_FILE, line, 'utf8');
}

// ── VCU SSO login ──────────────────────────────────────────────────────────────
// Selectors to try for each field (order = priority).
// Run: node scripts/vcu_dump_html.js  to capture the actual page HTML
// then update these selectors if needed.
const SELECTORS = {
  // Microsoft Azure AD (most common for VCU)
  username: ['#i0116', '#userNameInput', 'input[name="loginfmt"]', 'input[type="email"]'],
  nextBtn:  ['#idSIButton9', 'input[type="submit"]', 'button[type="submit"]'],
  password: ['#i0118', '#passwordInput', 'input[name="passwd"]', 'input[type="password"]'],
  signIn:   ['#idSIButton9', 'input[type="submit"]', 'button[type="submit"]'],
  // Shibboleth / CAS fallback
  userCas:  ['#username', 'input[name="username"]', 'input[name="j_username"]'],
  passCas:  ['#password', 'input[name="password"]', 'input[name="j_password"]'],
};

async function trySelector(page, selectors, timeout = 3000) {
  for (const sel of selectors) {
    try {
      await page.waitForSelector(sel, { timeout });
      return sel;
    } catch (_) { /* try next */ }
  }
  return null;
}

async function fillAndClick(page, selectors, value) {
  const sel = await trySelector(page, selectors);
  if (!sel) throw new Error(`None of selectors found: ${selectors.join(', ')}`);
  await page.click(sel, { clickCount: 3 });
  await page.type(sel, value, { delay: 50 });
  return sel;
}

async function doVcuLogin(page, creds) {
  console.log('  → VCU CAS login at:', page.url());
  try {
    // VCU CAS — username + password + submit all on one page
    await page.waitForSelector('#username', { timeout: 15000 });
    await page.click('#username', { clickCount: 3 });
    await page.type('#username', creds.username, { delay: 50 });

    await page.click('#password', { clickCount: 3 });
    await page.type('#password', creds.password, { delay: 50 });

    await Promise.all([
      page.waitForNavigation({ waitUntil: 'networkidle2', timeout: 30000 }).catch(() => {}),
      page.click('#submitBtn'),
    ]);

    console.log('  → Post-CAS URL:', page.url());

    // ── Duo 2FA (if present) ───────────────────────────────────────────────
    if (await isDuoPage(page)) {
      console.log('  → Handling Duo 2FA...');
      await waitForDuo(page, creds);
    }

    console.log('  ✓ Login complete — final URL:', page.url());
  } catch (err) {
    console.error(`  ✗ Login error: ${err.message}`);
    console.error(`    Current URL: ${page.url()}`);
    throw err;
  }
}

async function isDuoPage(page) {
  const url = page.url();
  if (url.includes('duosecurity') || url.includes('duo.com')) return true;
  // Duo Universal Prompt embeds inline — check for known elements
  const hasDuoEl = await page.$('[data-testid="push-send-button"], [data-testid="passcode-option-link"], #duo-frame, iframe[src*="duo"]').catch(() => null);
  return hasDuoEl !== null;
}

async function promptPasscode(prompt) {
  return new Promise(resolve => {
    process.stdout.write(prompt);
    process.stdin.resume();
    process.stdin.setEncoding('utf8');
    process.stdin.once('data', data => {
      process.stdin.pause();
      resolve(data.trim());
    });
  });
}

async function waitForDuo(page, creds, timeoutMs = 120000) {
  // Duo can appear as: iframe, redirect to duosecurity.com, or inline Universal Prompt
  const deadline = Date.now() + timeoutMs;
  let passcodeAttempted = false;

  while (Date.now() < deadline) {
    const url = page.url();

    // If we've moved past all auth domains, Duo is done
    if (!url.includes('microsoftonline') && !url.includes('cas.vcu.edu') &&
        !url.includes('shibboleth')       && !url.includes('duosecurity') &&
        !url.includes('login.vcu.edu')) {
      return;
    }

    // ── Duo Universal Prompt (new Duo UI, inline) ───────────────────────────
    try {
      // "Other options" → "Passcode" button in new Duo Universal Prompt
      const passcodeBtn = await page.$('[data-testid="passcode-option-link"], button[aria-label*="passcode" i], button[aria-label*="Passcode" i]');
      if (passcodeBtn && !passcodeAttempted) {
        passcodeAttempted = true;
        await passcodeBtn.click();
        await new Promise(r => setTimeout(r, 800));
        const code = process.env.DUO_PASSCODE ||
                     await promptPasscode('  Enter Duo passcode (from Duo app → Passcode): ');
        const input = await page.$('input[name="passcode"], input[placeholder*="code" i], input[type="number"]');
        if (input) {
          await input.click({ clickCount: 3 });
          await input.type(code, { delay: 80 });
          const submitBtn = await page.$('button[type="submit"], button[aria-label*="Log in" i], #passcode-submit');
          if (submitBtn) await submitBtn.click();
          passcodeAttempted = true;
        }
        await new Promise(r => setTimeout(r, 2000));
        continue;
      }

      // "Send Me a Push" as fallback if passcode button not found
      const pushBtn = await page.$('[data-testid="push-send-button"], button[aria-label*="Duo Push" i]');
      if (pushBtn && !passcodeAttempted) {
        await pushBtn.click();
        console.log('  → Duo push sent — approve on your phone...');
      }
    } catch (_) { /* element not found */ }

    // ── Duo legacy iframe ───────────────────────────────────────────────────
    const duoFrame = page.frames().find(f =>
      f.url().includes('duosecurity') || f.url().includes('duo.com'));
    if (duoFrame && !passcodeAttempted) {
      try {
        // Click "Enter a Passcode" tab
        const pcTab = await duoFrame.$('a[href="#passcode"], .passcode-label, button.use-passcode');
        if (pcTab) {
          await pcTab.click();
          await new Promise(r => setTimeout(r, 600));
          const code = process.env.DUO_PASSCODE ||
                       await promptPasscode('  Enter Duo passcode (from Duo app → Passcode): ');
          const pcInput = await duoFrame.$('input[name="passcode"]');
          if (pcInput) {
            await pcInput.click({ clickCount: 3 });
            await duoFrame.type('input[name="passcode"]', code, { delay: 80 });
            await duoFrame.click('button[id="passcode"] , button[type="submit"]');
            passcodeAttempted = true;
          }
        } else {
          // Fall back to push
          const pushBtn = await duoFrame.$('.push-label, button.positive, [aria-label*="push" i]');
          if (pushBtn) {
            await pushBtn.click();
            console.log('  → Duo push sent — approve on your phone...');
          }
        }
      } catch (_) { /* try next tick */ }
    }

    await new Promise(r => setTimeout(r, 2500));
  }
  throw new Error(`Duo 2FA timed out after ${timeoutMs/1000}s`);
}

// ── Publisher PDF URL patterns ────────────────────────────────────────────────
function guessPdfUrl(articleUrl) {
  // Most publishers use /doi/pdf/ instead of /doi/abs/ or /doi/full/
  const pdfUrl = articleUrl
    .replace(/\/doi\/abs\//,  '/doi/pdf/')
    .replace(/\/doi\/full\//, '/doi/pdf/')
    .replace(/\/doi\/epdf\//, '/doi/pdf/');
  if (pdfUrl !== articleUrl) return pdfUrl;

  // Wiley: pdfdirect (matches both direct and EZProxy proxied hostnames)
  if (/onlinelibrary[.\-]wiley/i.test(articleUrl))
    return articleUrl.replace('/doi/', '/doi/pdfdirect/');

  // ScienceDirect: use pdfft download link (works via proxy)
  if (/sciencedirect[.-]com/i.test(articleUrl) && !/pdfft/.test(articleUrl))
    return articleUrl.replace(/(\/pii\/[^/?#]+).*$/, '$1/pdfft?isDTMRedir=true&download=true');

  // BMJ: /content/X/Y.full → /content/X/Y.full.pdf
  if (/bmj[.-]com/i.test(articleUrl) && !articleUrl.endsWith('.pdf'))
    return articleUrl.replace(/(\.full)?$/, '.full.pdf');

  // bioRxiv/medRxiv: /content/XXXX → /content/XXXX.full.pdf
  if (/biorxiv|medrxiv/i.test(articleUrl) && !articleUrl.endsWith('.pdf'))
    return articleUrl.replace(/(v\d+)?$/, '$1.full.pdf');

  return null;
}

// ── In-browser fetch (carries proxy session cookies) ─────────────────────────
async function browserFetch(page, url, destPath) {
  const result = await page.evaluate(async (u) => {
    try {
      const r = await fetch(u, { credentials: 'include', redirect: 'follow' });
      if (!r.ok) return { ok: false, status: r.status };
      const ct = r.headers.get('content-type') || '';
      if (!ct.includes('pdf')) return { ok: false, ct };
      const ab  = await r.arrayBuffer();
      return { ok: true, bytes: Array.from(new Uint8Array(ab)) };
    } catch (e) { return { ok: false, err: e.message }; }
  }, url);
  if (result && result.ok && result.bytes && result.bytes.length > 1024) {
    const buf = Buffer.from(result.bytes);
    fs.writeFileSync(destPath, buf);
    return buf.length;
  }
  return 0;
}

// ── PDF detection & download ──────────────────────────────────────────────────
async function downloadPdf(page, destPath) {
  const articleUrl = page.url();
  process.stdout.write(`\n    proxy→ ${articleUrl.substring(0, 100)}\n`);

  // 1. Direct PDF URL (article already IS a PDF)
  const ct = await page.evaluate(() => document.contentType || '').catch(() => '');
  if (ct.includes('pdf') || articleUrl.endsWith('.pdf')) {
    const n = await browserFetch(page, articleUrl, destPath);
    if (n > 1024) return n;
  }

  // 2. Publisher-specific guessed PDF URL
  const pdfUrl = guessPdfUrl(articleUrl);
  if (pdfUrl && pdfUrl !== articleUrl) {
    const n = await browserFetch(page, pdfUrl, destPath);
    if (n > 1024) return n;
    // Also try navigating to PDF URL (some publishers require cookie from article page)
    try {
      let captured = 0;
      const handler = async r => {
        if ((r.headers()['content-type'] || '').includes('pdf') && !captured) {
          try { const buf = await r.buffer(); fs.writeFileSync(destPath, buf); captured = buf.length; } catch (_) {}
        }
      };
      page.on('response', handler);
      await page.goto(pdfUrl, { waitUntil: 'networkidle2', timeout: 30000 }).catch(() => {});
      await new Promise(r => setTimeout(r, 2000));
      page.off('response', handler);
      if (captured > 1024) return captured;
      // If we landed on a PDF page
      const ct2 = await page.evaluate(() => document.contentType || '').catch(() => '');
      if (ct2.includes('pdf')) {
        const n2 = await browserFetch(page, page.url(), destPath);
        if (n2 > 1024) return n2;
      }
    } catch (_) {}
  }

  // 3. Scrape all PDF-bearing links from the article page (navigate back first)
  try {
    if (page.url() !== articleUrl)
      await page.goto(articleUrl, { waitUntil: 'networkidle2', timeout: 30000 }).catch(() => {});
    const links = await page.evaluate(() =>
      [...document.querySelectorAll('a[href]')]
        .map(a => a.href)
        .filter(h => /pdf/i.test(h) && h.startsWith('http'))
        .slice(0, 10)
    );
    for (const link of links) {
      const n = await browserFetch(page, link, destPath);
      if (n > 1024) return n;
    }
  } catch (_) {}

  return 0;
}

// ── Auth check ────────────────────────────────────────────────────────────────
async function isLoginPage(page) {
  const url = page.url();
  // Only match VCU's own CAS / SSO domains — never publisher login forms
  return url.includes('login.vcu.edu') ||
         url.includes('cas.vcu.edu')   ||
         url.includes('shibboleth.vcu.edu');
}

// ── Main ──────────────────────────────────────────────────────────────────────
(async () => {
  const creds      = loadSecrets();
  const candidates = LOGIN_ONLY ? [] : loadCandidates();
  console.log(`VCU Library Downloader`);
  console.log(`Proxy:      ${creds.proxy_base}`);
  console.log(`Candidates: ${candidates.length}`);
  console.log(`Log:        ${LOG_FILE}\n`);

  const browser = await puppeteer.launch({
    headless:  HEADED ? false : 'new',
    args:      ['--no-sandbox', '--disable-setuid-sandbox', '--ignore-certificate-errors'],
    defaultViewport: { width: 1280, height: 900 },
  });

  const page = await browser.newPage();
  page.setDefaultTimeout(60000);

  // ── CDP session for full-jar cookie operations ────────────────────────────
  const cdpClient = await page.target().createCDPSession();
  await cdpClient.send('Network.enable');

  async function saveAllCookies() {
    const { cookies } = await cdpClient.send('Network.getAllCookies');
    fs.writeFileSync(COOKIES_FILE, JSON.stringify(cookies, null, 2));
    console.log(`  ✓ Saved ${cookies.length} cookies to ${COOKIES_FILE}`);
    return cookies.length;
  }

  async function loadAllCookies() {
    if (!fs.existsSync(COOKIES_FILE)) return 0;
    const cookies = JSON.parse(fs.readFileSync(COOKIES_FILE, 'utf8'));
    if (!cookies.length) return 0;
    await cdpClient.send('Network.setCookies', { cookies });
    console.log(`Loaded ${cookies.length} saved cookies from ${COOKIES_FILE}`);
    return cookies.length;
  }

  // ── Load saved cookies ──────────────────────────────────────────────────────
  let loggedIn = false;
  const loadedCount = await loadAllCookies();
  if (loadedCount > 0) loggedIn = true;

  // ── Login-only mode ─────────────────────────────────────────────────────────
  if (LOGIN_ONLY) {
    console.log('Login-only mode: navigating through proxy to establish session...');
    // Use a real article DOI so EZProxy sets its session cookie
    const testUrl = creds.proxy_base + encodeURIComponent('https://doi.org/10.1377/hlthaff.2025.00253');
    await page.goto(testUrl, { waitUntil: 'networkidle2', timeout: 45000 });
    if (await isLoginPage(page)) {
      await doVcuLogin(page, creds);
    }
    await new Promise(r => setTimeout(r, 3000)); // let proxy set session cookie
    await saveAllCookies();
    await browser.close();
    return;
  }

  // ── Download loop ───────────────────────────────────────────────────────────
  let found = 0;
  for (let i = 0; i < candidates.length; i++) {
    const { hsh_id, title, doi } = candidates[i];
    const label = title.substring(0, 60);
    process.stdout.write(`[${String(i+1).padStart(5)}/${candidates.length}] ${label.padEnd(60)}  `);

    let proxyUrl = '';
    if (doi) {
      // Resolve DOI → publisher URL, then build direct EZProxy URL
      // Format: https://publisher.host.proxy.library.vcu.edu/path
      const publisherUrl = await resolveDoi(doi);
      if (!publisherUrl) {
        process.stdout.write('doi_resolve_failed\n');
        appendLog({ hsh_id, title: label, doi, proxy_url: '', status: 'doi_resolve_failed', bytes: 0, timestamp: new Date().toISOString() });
        continue;
      }
      try {
        let targetUrl = publisherUrl;
        // Elsevier linkinghub is a redirect broker — map directly to ScienceDirect
        const linkingHub = targetUrl.match(/linkinghub\.elsevier\.com\/retrieve\/pii\/([^?#/]+)/i);
        if (linkingHub) {
          targetUrl = `https://www.sciencedirect.com/science/article/pii/${linkingHub[1]}`;
        }
        const u      = new URL(targetUrl);
        const ezHost = u.hostname + '.proxy.library.vcu.edu';
        proxyUrl = `https://${ezHost}${u.pathname}${u.search}`;
      } catch (_) {
        proxyUrl = creds.proxy_base + encodeURIComponent(publisherUrl);
      }
      process.stdout.write(`\n    proxy→ ${proxyUrl.substring(0, 100)}\n    `);
    } else {
      process.stdout.write('no_doi — skipped\n');
      appendLog({ hsh_id, title: label, doi, proxy_url: '', status: 'no_doi', bytes: 0, timestamp: new Date().toISOString() });
      continue;
    }

    const destPath = path.join(PDF_DIR, `${hsh_id}.pdf`);
    try {
      // domcontentloaded fires once HTML is parsed — avoids hanging on
      // publishers (SAGE, T&F) that keep firing analytics/tracking requests
      // indefinitely and never reach networkidle2.
      try {
        await page.goto(proxyUrl, { waitUntil: 'domcontentloaded', timeout: 90000 });
      } catch (navErr) {
        if (!navErr.message.includes('timeout')) throw navErr;
        process.stdout.write('(nav-timeout, attempting pdf anyway) ');
      }

      // Handle login if prompted
      if (!loggedIn || await isLoginPage(page)) {
        await doVcuLogin(page, creds);
        loggedIn = true;
        await saveAllCookies();
        try {
          await page.goto(proxyUrl, { waitUntil: 'domcontentloaded', timeout: 90000 });
        } catch (navErr) {
          if (!navErr.message.includes('timeout')) throw navErr;
        }
      }

      const bytes = await downloadPdf(page, destPath);
      if (bytes > 1024) {
        found++;
        process.stdout.write(`OK  ${(bytes/1024).toFixed(1)} KB\n`);
        appendLog({ hsh_id, title: label, doi, proxy_url: proxyUrl, status: 'ok', bytes, timestamp: new Date().toISOString() });
      } else {
        process.stdout.write('no_pdf\n');
        appendLog({ hsh_id, title: label, doi, proxy_url: proxyUrl, status: 'no_pdf', bytes: 0, timestamp: new Date().toISOString() });
      }

    } catch (err) {
      process.stdout.write(`error: ${err.message.substring(0, 60)}\n`);
      appendLog({ hsh_id, title: label, doi, proxy_url: proxyUrl, status: 'error', bytes: 0, timestamp: new Date().toISOString() });
    }

    // Small delay between requests to avoid rate limiting
    await new Promise(r => setTimeout(r, 2000));
  }

  // Save final cookies (full jar) — guard against CDP session already closed
  try { await saveAllCookies(); } catch (_) {}

  await browser.close();

  console.log(`\n── VCU Download complete ─────────────────────────────`);
  console.log(`  Processed:  ${candidates.length}`);
  console.log(`  Found PDFs: ${found}  (${(found/Math.max(candidates.length,1)*100).toFixed(1)}%)`);
  console.log(`  Log:        ${LOG_FILE}`);
  console.log(`  PDFs:       ${PDF_DIR}/`);
})();
