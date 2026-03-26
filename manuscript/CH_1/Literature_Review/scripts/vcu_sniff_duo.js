/**
 * Sniff Duo Universal Prompt (frameless v4) elements.
 * This navigates directly to the Duo URL captured from the login flow.
 * NOTE: The SID/TX tokens are session-specific — run this quickly after seeing
 * the Duo URL in the login output.
 *
 * Pass the full Duo URL as the first argument:
 *   node scripts/vcu_sniff_duo.js "https://api-xxxx.duosecurity.com/frame/..."
 */
const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');

const ROOT    = path.resolve(__dirname, '..');
const OUT     = path.join(ROOT, 'secrets', 'sniff_duo.html');
const DUO_URL = process.argv[2];

if (!DUO_URL) {
  console.error('Usage: node scripts/vcu_sniff_duo.js "<duo_url>"');
  process.exit(1);
}

(async () => {
  const browser = await puppeteer.launch({ headless: 'new', args: ['--no-sandbox'] });
  const page    = await browser.newPage();
  page.setDefaultTimeout(20000);

  console.log('Navigating to Duo URL...');
  await page.goto(DUO_URL, { waitUntil: 'networkidle2', timeout: 20000 }).catch(() => {});
  await new Promise(r => setTimeout(r, 4000));

  const url  = page.url();
  const html = await page.content();

  const fields = await page.evaluate(() => {
    const inputs  = [...document.querySelectorAll('input')].map(el => ({
      id: el.id, name: el.name, type: el.type, placeholder: el.placeholder,
      'data-testid': el.getAttribute('data-testid'), 'aria-label': el.getAttribute('aria-label'),
    }));
    const buttons = [...document.querySelectorAll('button')].map(el => ({
      id: el.id, type: el.type, text: el.innerText?.substring(0, 60).trim(),
      'data-testid': el.getAttribute('data-testid'), 'aria-label': el.getAttribute('aria-label'),
      class: el.className.substring(0, 80),
    }));
    const links = [...document.querySelectorAll('a[role=button],a.btn')].map(el => ({
      text: el.innerText?.substring(0, 60).trim(),
      'data-testid': el.getAttribute('data-testid'), href: el.href?.substring(0, 80),
    }));
    return { inputs, buttons, links };
  });

  fs.writeFileSync(OUT, html, 'utf8');
  console.log('\nLanded URL:', url);
  console.log('\nInputs:');  fields.inputs.forEach(f  => console.log(' ', JSON.stringify(f)));
  console.log('\nButtons:'); fields.buttons.forEach(b => console.log(' ', JSON.stringify(b)));
  console.log('\nLinks:');   fields.links.forEach(l   => console.log(' ', JSON.stringify(l)));
  console.log('\nHTML saved to:', OUT);

  await browser.close();
})();
