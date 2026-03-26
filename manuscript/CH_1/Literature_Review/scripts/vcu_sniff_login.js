/**
 * vcu_sniff_login.js — navigate to VCU proxy, wait 8s for redirects to settle,
 * dump HTML + URL + all input/button selectors found on the page.
 */
const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const OUT  = path.join(ROOT, 'secrets', 'sniff_login.html');
fs.mkdirSync(path.join(ROOT, 'secrets'), { recursive: true });

const PROXY_URL = 'https://proxy.library.vcu.edu/login?url=' +
  encodeURIComponent('https://doi.org/10.1377/hlthaff.2025.00253');

(async () => {
  const browser = await puppeteer.launch({ headless: 'new', args: ['--no-sandbox'] });
  const page    = await browser.newPage();
  page.setDefaultTimeout(30000);

  console.log('Navigating to:', PROXY_URL);
  await page.goto(PROXY_URL, { waitUntil: 'networkidle2', timeout: 30000 }).catch(() => {});
  await new Promise(r => setTimeout(r, 5000)); // wait for JS redirects

  const url  = page.url();
  const html = await page.content();

  // Extract all input + button selectors
  const fields = await page.evaluate(() => {
    const inputs  = [...document.querySelectorAll('input')].map(el => ({
      tag: 'input', id: el.id, name: el.name, type: el.type,
      placeholder: el.placeholder, 'aria-label': el.getAttribute('aria-label'),
      class: el.className.substring(0, 60),
    }));
    const buttons = [...document.querySelectorAll('button,input[type=submit]')].map(el => ({
      tag: el.tagName, id: el.id, type: el.type, text: el.innerText?.substring(0, 40),
      'aria-label': el.getAttribute('aria-label'), class: el.className.substring(0, 60),
    }));
    return { inputs, buttons };
  });

  fs.writeFileSync(OUT, html, 'utf8');
  console.log('\nLanded URL:', url);
  console.log('\nInputs found:');
  fields.inputs.forEach(f => console.log(' ', JSON.stringify(f)));
  console.log('\nButtons found:');
  fields.buttons.forEach(b => console.log(' ', JSON.stringify(b)));
  console.log('\nHTML saved to:', OUT, `(${(html.length/1024).toFixed(1)} KB)`);

  await browser.close();
})();
