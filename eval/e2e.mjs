// run: auto-crop check
// End-to-end: open the preview theme on a phone-sized Chrome, run a visual search, report rank and timings.
import { chromium, devices } from 'playwright';
const URL0 = 'https://sa-jewel.com/?preview_theme_id=132002938938';
const Q = 'https://raw.githubusercontent.com/jewelodoo/vs-index/queries/';
const tests = [['dior_full.jpg', 'copy-of-443', null], ['dior_sel.jpg', 'copy-of-443', null], ['dior_full.jpg', 'copy-of-443', [0.12, 0.27, 0.9, 0.75]]];
const b = await chromium.launch();
const ctx = await b.newContext({ ...devices['iPhone 13'] });
const page = await ctx.newPage();
page.on('pageerror', e => console.log('PAGEERROR', e.message));
page.on('console', m => { if (m.type() === 'error') console.log('CONSOLE', m.text().slice(0, 200)); });
await page.goto(URL0, { waitUntil: 'load' }); await page.waitForTimeout(4000);
console.log('snippet', await page.evaluate(() => [...document.scripts].some(s => s.textContent.includes('embedMany'))));
for (const [file, target, sel] of tests) {
  await page.evaluate(() => (document.querySelector('.jvs-btn') || document.querySelector('.jvs-in')).click());
  await page.waitForTimeout(1500);
  const buf = Buffer.from(await (await fetch(Q + file)).arrayBuffer());
  const t0 = Date.now();
  await page.setInputFiles('.jvs-pick label:nth-child(2) input', { name: file, mimeType: 'image/jpeg', buffer: buf });
  await page.waitForTimeout(400);
  const hint = await page.evaluate(() => { const h = document.querySelector('.jvs-hint'); return h && h.classList.contains('on'); });
  await page.evaluate(() => [...document.querySelectorAll('.jvs-cats button')].find(x => x.textContent === 'حلق').click());
  if (sel) {
    const box = await page.locator('.jvs-stage canvas').boundingBox();
    await page.mouse.move(box.x + box.width * sel[0], box.y + box.height * sel[1]); await page.mouse.down();
    await page.mouse.move(box.x + box.width * sel[2], box.y + box.height * sel[3], { steps: 8 }); await page.mouse.up();
  }
  await page.waitForTimeout(3000); // shopper choosing the category
  const t1 = Date.now();
  await page.click('.jvs-go');
  await page.waitForSelector('.jvs-res a', { timeout: 90000 });
  const t2 = Date.now();
  const r = await page.evaluate(t => { const as = [...document.querySelectorAll('.jvs-res a')]; return { n: as.length, rank: as.findIndex(a => a.getAttribute('href').includes(t)) + 1, top: as.slice(0, 3).map(a => a.textContent.trim().slice(0, 30)) }; }, target);
  console.log('E2E', file, sel ? 'selected' : 'whole', 'hint=' + hint, 'wait_after_click_ms=' + (t2 - t1), 'from_pick_ms=' + (t2 - t0), JSON.stringify(r));
  await page.evaluate(() => document.querySelector('.jvs-x').click());
  await page.reload({ waitUntil: 'load' }); await page.waitForTimeout(3000);
}
await b.close();
