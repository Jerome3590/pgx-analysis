/**
 * vcu_dump_html.js
 * Opens a headed browser, navigates to VCU library pages, and dumps HTML
 * at each step so selectors can be identified for vcu_download.js.
 *
 * Usage:
 *   node scripts/vcu_dump_html.js
 *
 * Output files (in secrets/ — gitignored):
 *   secrets/dump_01_proxy_entry.html     — EZProxy entry page
 *   secrets/dump_02_sso_login.html       — VCU SSO / username step
 *   secrets/dump_03_password.html        — password step
 *   secrets/dump_04_duo.html             — Duo 2FA page/iframe
 *   secrets/dump_05_article.html         — landed article page (post-auth)
 */

const puppeteer = require('puppeteer');
const fs        = require('fs');
const path      = require('path');

const ROOT         = path.resolve(__dirname, '..');
const SECRETS_FILE = path.join(ROOT, 'secrets', 'secrets.txt');
const DUMP_DIR     = path.join(ROOT, 'secrets');
fs.mkdirSync(DUMP_DIR, { recursive: true });

// Test article — Health Affairs (paywalled, will trigger VCU proxy auth)
const TEST_DOI    = '10.1377/hlthaff.2025.00253';
const PROXY_BASE  = 'https://proxy.library.vcu.edu/login?url=';
const TARGET_URL  = PROXY_BASE + encodeURIComponent(`https://doi.org/${TEST_DOI}`);

function loadSecrets() {
  if (!fs.existsSync(SECRETS_FILE)) {
    console.error(`ERROR: ${SECRETS_FILE} not found. Copy secrets.example.txt → secrets/secrets.txt`);
    process.exit(1);
  }
  const lines = fs.readFileSync(SECRETS_FILE, 'utf8').split('\n');
  const cfg = {};
  for (const line of lines) {
    const m = line.match(/^\s*([^#=\s][^=]*)=(.*)$/);
    if (m) cfg[m[1].trim()] = m[2].trim();
  }
  return cfg;
}

async function dumpPage(page, filename, label) {
  const html = await page.content();
  const dest = path.join(DUMP_DIR, filename);
  fs.writeFileSync(dest, html, 'utf8');
  console.log(`  ✓ Dumped ${label} → ${dest}  (${(html.length/1024).toFixed(1)} KB)`);
  return dest;
}

async function waitForKeypress(prompt) {
  process.stdout.write(`\n[PAUSE] ${prompt}\nPress ENTER when ready...`);
  return new Promise(resolve => {
    process.stdin.resume();
    process.stdin.setEncoding('utf8');
    process.stdin.once('data', () => {
      process.stdin.pause();
      resolve();
    });
  });
}

(async () => {
  const creds = loadSecrets();
  console.log(`VCU HTML Dumper`);
  console.log(`Target: ${TARGET_URL}\n`);

  const browser = await puppeteer.launch({
    headless: false,
    args:     ['--no-sandbox'],
    defaultViewport: { width: 1280, height: 900 },
  });
  const page = await browser.newPage();
  page.setDefaultTimeout(60000);

  // ── Step 1: Navigate to proxy entry ────────────────────────────────────────
  console.log('Step 1: Navigating to EZProxy entry...');
  await page.goto(TARGET_URL, { waitUntil: 'networkidle2', timeout: 30000 });
  await dumpPage(page, 'dump_01_proxy_entry.html', 'EZProxy entry / first redirect');
  console.log(`  Current URL: ${page.url()}`);

  // ── Step 2: SSO username step ───────────────────────────────────────────────
  await waitForKeypress('If a login page appeared, DO NOT type yet. Just let us capture it.');
  await dumpPage(page, 'dump_02_sso_login.html', 'SSO login page (username step)');
  console.log(`  Current URL: ${page.url()}`);

  // ── Step 3: Type username, capture password step ───────────────────────────
  console.log('\nStep 3: Type your VCU eID in the browser, then press ENTER here.');
  console.log(`  Username hint: ${creds.username || '(set in secrets.txt)'}`);
  await waitForKeypress('Type your username in the browser, click Next, then press ENTER here.');
  await dumpPage(page, 'dump_03_password.html', 'SSO password step');
  console.log(`  Current URL: ${page.url()}`);

  // ── Step 4: Type password, capture Duo step ────────────────────────────────
  console.log('\nStep 4: Type your password in the browser, click Sign in, then press ENTER here.');
  await waitForKeypress('Type password in browser, click Sign In, then press ENTER here.');
  await dumpPage(page, 'dump_04_duo.html', 'Duo 2FA page');
  console.log(`  Current URL: ${page.url()}`);

  // ── Step 5: Complete Duo, capture article page ─────────────────────────────
  console.log('\nStep 5: Complete Duo 2FA on your phone.');
  await waitForKeypress('After Duo approval and redirect to article, press ENTER here.');
  await dumpPage(page, 'dump_05_article.html', 'Landed article page (post-auth)');
  console.log(`  Current URL: ${page.url()}`);

  await browser.close();

  console.log('\n── Done ─────────────────────────────────────────────────');
  console.log('HTML dumps saved to: secrets/dump_0*.html');
  console.log('Send these files (or paste their contents) so selectors can be confirmed.');
})();
