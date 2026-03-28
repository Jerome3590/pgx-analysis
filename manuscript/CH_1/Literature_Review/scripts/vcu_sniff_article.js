/**
 * Navigate to one article via VCU proxy using saved cookies,
 * dump the final URL chain, content-type, and any PDF links found.
 */
const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');
const { parse } = require('csv-parse/sync');

const ROOT        = path.resolve(__dirname, '..');
const COOKIES_FILE = path.join(ROOT, 'secrets', 'session_cookies.json');
const DOI_MAP     = path.join(ROOT, 'scripts', 'screened_doi_map.csv');
const OUT         = path.join(ROOT, 'secrets', 'sniff_article.html');

// Pick first DOI from the map
const rows = parse(fs.readFileSync(DOI_MAP, 'utf8'), { columns: true, skip_empty_lines: true });
const { doi, title } = rows[0];
const PROXY_BASE = 'https://proxy.library.vcu.edu/login?url=';
const TARGET     = PROXY_BASE + encodeURIComponent(`https://doi.org/${doi}`);

(async () => {
  console.log('Article:', title);
  console.log('DOI:    ', doi);
  console.log('URL:    ', TARGET);

  const browser = await puppeteer.launch({ headless: 'new', args: ['--no-sandbox'] });
  const page    = await browser.newPage();
  const cdp     = await page.target().createCDPSession();
  await cdp.send('Network.enable');

  // Load saved cookies
  const cookies = JSON.parse(fs.readFileSync(COOKIES_FILE, 'utf8'));
  await cdp.send('Network.setCookies', { cookies });
  console.log(`\nLoaded ${cookies.length} cookies`);

  // Track redirects
  const urls = [];
  page.on('request',  r => { if (r.isNavigationRequest()) urls.push('→ ' + r.url().substring(0, 120)); });
  page.on('response', r => { if (r.request().isNavigationRequest()) urls.push('← ' + r.status() + ' ' + r.url().substring(0, 120)); });

  await page.goto(TARGET, { waitUntil: 'networkidle2', timeout: 45000 }).catch(e => console.log('nav error:', e.message));
  await new Promise(r => setTimeout(r, 3000));

  const finalUrl   = page.url();
  const contentType = await page.evaluate(() => document.contentType || '');
  const html       = await page.content();

  // Find PDF links
  const pdfLinks = await page.evaluate(() =>
    [...document.querySelectorAll('a[href]')]
      .map(a => ({ text: a.innerText?.trim().substring(0, 50), href: a.href }))
      .filter(a => a.href.toLowerCase().includes('pdf') || a.text?.toLowerCase().includes('pdf'))
      .slice(0, 20)
  );

  // Find download buttons
  const dlBtns = await page.evaluate(() =>
    [...document.querySelectorAll('a,button')]
      .map(el => ({ tag: el.tagName, text: el.innerText?.trim().substring(0, 60), href: el.href || '', class: el.className.substring(0,60) }))
      .filter(el => /download|full.?text|pdf|article/i.test(el.text + el.href + el.class))
      .slice(0, 15)
  );

  fs.writeFileSync(OUT, html, 'utf8');
  console.log('\n── Redirect chain:');
  urls.forEach(u => console.log('  ' + u));
  console.log('\n── Final URL:    ', finalUrl);
  console.log('── Content-type:', contentType);
  console.log(`── HTML size:    ${(html.length/1024).toFixed(1)} KB`);
  console.log('\n── PDF links found:');
  pdfLinks.forEach(l => console.log(' ', JSON.stringify(l)));
  console.log('\n── Download/Article buttons:');
  dlBtns.forEach(b => console.log(' ', JSON.stringify(b)));
  console.log('\nFull page saved to:', OUT);

  await browser.close();
})();
