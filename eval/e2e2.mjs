// New full-screen visual search (preview theme): pick a photo -> results come by themselves ("الكل" guesses the category).
// Screenshots go to shots/ (published to the `shots` branch by e2e2.yml).
import { chromium, devices } from 'playwright';
import fs from 'fs';
const URL0 = 'https://sa-jewel.com/?preview_theme_id=132002938938';
const Q = 'https://raw.githubusercontent.com/jewelodoo/vs-index/queries/';
const tests = [['t/16.jpg', '408996'], ['t/ch.jpg', 'product-4957']];
fs.mkdirSync('shots', { recursive: true });
const b = await chromium.launch();
const ctx = await b.newContext({ ...devices['iPhone 13'] });
const page = await ctx.newPage();
page.on('pageerror', e => console.log('PAGEERROR', e.message));
page.on('console', m => { if (m.type() === 'error') console.log('CONSOLE', m.text().slice(0, 200)); });
await page.goto(URL0, { waitUntil: 'load' }); await page.waitForTimeout(4000);
const ranks = async t => page.evaluate(t => { const as = [...document.querySelectorAll('.jvs-res a')]; return { n: as.length, rank: as.findIndex(a => a.getAttribute('href').includes(t)) + 1, guess: [...document.querySelectorAll('.jvs-cats button.guess')].map(x => x.textContent).join('+'), top: as.slice(0, 4).map(a => a.getAttribute('href').split('/products/')[1].split('?')[0]) }; }, t);
let k = 0;
for (const [file, target] of tests) {
  k++;
  await page.evaluate(() => (document.querySelector('.jvs-btn') || document.querySelector('.jvs-in')).click());
  await page.waitForTimeout(1200);
  await page.screenshot({ path: `shots/${k}a_pick.png` });
  const buf = Buffer.from(await (await fetch(Q + file)).arrayBuffer());
  const t0 = Date.now();
  await page.setInputFiles('.jvs-pick label:nth-child(2) input', { name: file, mimeType: 'image/jpeg', buffer: buf });
  await page.waitForTimeout(700);
  await page.screenshot({ path: `shots/${k}b_loading.png` });
  const ok = await page.waitForSelector('.jvs-res a', { timeout: 90000 }).then(() => true).catch(() => false);
  if (!ok) { console.log('NORES', file, await page.evaluate(() => document.querySelector('.jvs-msg').textContent)); continue; }
  await page.waitForTimeout(1500);
  console.log('E2E2', file, 'auto_ms=' + (Date.now() - t0), JSON.stringify(await ranks(target)), 'box=' + JSON.stringify(await page.evaluate(() => { const s = document.querySelector('.jvs-sel'); return s.style.cssText; })));
  await page.screenshot({ path: `shots/${k}c_results.png` });
  // drag the top-left corner a little and see the search run again
  const c = await page.locator('.jvs-sel i.b').boundingBox();
  if (c) {
    await page.mouse.move(c.x + 22, c.y + 22); await page.mouse.down(); await page.mouse.move(c.x + 50, c.y + 40, { steps: 6 }); await page.mouse.up();
    await page.waitForTimeout(6000);
    console.log('E2E2-DRAG', file, JSON.stringify(await ranks(target)));
    await page.screenshot({ path: `shots/${k}d_after_drag.png` });
  }
  // open the panel over the photo
  await page.click('.jvs-grab'); await page.waitForTimeout(600);
  await page.screenshot({ path: `shots/${k}e_panel_big.png` });
  const named = await page.evaluate(() => /claude|anthropic/i.test(document.documentElement.outerHTML));
  console.log('E2E2-UI', file, 'big=' + await page.evaluate(() => document.querySelector('.jvs-full').classList.contains('big')), 'ai_button=' + await page.evaluate(() => getComputedStyle(document.querySelector('.jvs-ai')).display), 'name_in_page=' + named);
  await page.click('.jvs-fx');
  await page.reload({ waitUntil: 'load' }); await page.waitForTimeout(20000);
}
// desktop look
const d = await (await b.newContext({ viewport: { width: 1280, height: 800 } })).newPage();
await d.goto(URL0, { waitUntil: 'load' }); await d.waitForTimeout(4000);
await d.evaluate(() => (document.querySelector('.jvs-btn') || document.querySelector('.jvs-in')).click()); await d.waitForTimeout(800);
await d.setInputFiles('.jvs-pick label:nth-child(2) input', { name: 'x.jpg', mimeType: 'image/jpeg', buffer: Buffer.from(await (await fetch(Q + 't/16.jpg')).arrayBuffer()) });
await d.waitForSelector('.jvs-res a', { timeout: 90000 }).catch(() => {}); await d.waitForTimeout(2000);
await d.screenshot({ path: 'shots/9_desktop.png' });
await b.close();
