// Full-screen visual search (preview theme): pick a photo -> «وش نوع القطعة؟» -> pick the kind -> results; AI button with its spinner.
// Screenshots go to shots/ (published to the `shots` branch by e2e2.yml).
import { chromium, devices } from 'playwright';
import fs from 'fs';
const URL0 = 'https://sa-jewel.com/?preview_theme_id=132002938938';
const Q = 'https://raw.githubusercontent.com/jewelodoo/vs-index/queries/';
const tests = [['t/17.jpg', 'product-4804', 'حلق']];
fs.mkdirSync('shots', { recursive: true });
const b = await chromium.launch();
const ctx = await b.newContext({ ...devices['iPhone 13'] });
const page = await ctx.newPage();
page.on('pageerror', e => console.log('PAGEERROR', e.message));
page.on('console', m => { if (m.type() === 'error') console.log('CONSOLE', m.text().slice(0, 200)); });
await page.goto(URL0, { waitUntil: 'load' }); await page.waitForTimeout(4000);
const ranks = async t => page.evaluate(t => { const as = [...document.querySelectorAll('.jvs-res a')]; return { n: as.length, rank: as.findIndex(a => a.getAttribute('href').includes(t)) + 1, on: (document.querySelector('.jvs-cats button.on') || {}).textContent, chips: [...document.querySelectorAll('.jvs-cats button')].map(x => x.textContent).join('|'), top: as.slice(0, 4).map(a => a.getAttribute('href').split('/products/')[1].split('?')[0]) }; }, t);
const idle = () => page.waitForFunction(() => !document.querySelector('.jvs-busy').classList.contains('on'), null, { timeout: 60000 }).catch(() => {});
let k = 0;
for (const [file, target, chip] of tests) {
  k++;
  await page.evaluate(() => (document.querySelector('.jvs-btn') || document.querySelector('.jvs-in')).click());
  await page.waitForTimeout(1200);
  const buf = Buffer.from(await (await fetch(Q + file)).arrayBuffer());
  await page.setInputFiles('.jvs-pick label:nth-child(2) input', { name: file, mimeType: 'image/jpeg', buffer: buf });
  await page.waitForTimeout(2500);
  console.log('E2E2-ASK', file, 'ask=' + await page.evaluate(() => document.querySelector('.jvs-full').classList.contains('ask')), 'results=' + await page.evaluate(() => document.querySelectorAll('.jvs-res a').length), JSON.stringify(await ranks(target)));
  await page.screenshot({ path: `shots/${k}a_ask.png` });
  const t0 = Date.now();
  await page.evaluate(c => [...document.querySelectorAll('.jvs-cats button')].find(x => x.textContent.trim() === c).click(), chip);
  const ok = await page.waitForSelector('.jvs-res a', { timeout: 90000 }).then(() => true).catch(() => false);
  await idle(); await page.waitForTimeout(800);
  console.log('E2E2', file, chip, 'ok=' + ok, 'ms=' + (Date.now() - t0), JSON.stringify(await ranks(target)), 'ai_shown=' + await page.evaluate(() => getComputedStyle(document.querySelector('.jvs-ai')).display));
  await page.screenshot({ path: `shots/${k}b_results.png` });
  const [resp] = await Promise.all([page.waitForResponse(r => r.url().includes('/vs/rerank') && r.request().method() === 'POST', { timeout: 90000 }).catch(() => null), (async () => { await page.evaluate(() => document.querySelector('.jvs-ai').click()); await page.waitForTimeout(700); console.log('E2E2-WAIT', file, 'spinner=' + await page.evaluate(() => document.querySelector('.jvs-wait').classList.contains('on'))); await page.screenshot({ path: `shots/${k}c_ai_wait.png` }); })()]);
  await page.waitForFunction(() => !document.querySelector('.jvs-wait').classList.contains('on'), null, { timeout: 60000 }).catch(() => {}); await page.waitForTimeout(1500);
  console.log('E2E2-AI', file, 'status=' + (resp && resp.status()), JSON.stringify(await ranks(target)), 'exact=' + await page.evaluate(() => (document.querySelector('.jvs-res a.x') || {}).textContent || null), 'name_in_page=' + await page.evaluate(() => /claude|anthropic/i.test(document.documentElement.outerHTML)));
  await page.screenshot({ path: `shots/${k}d_ai_done.png` });
  // learning log: the AI match is saved; opening a product is saved
  const cnt = async () => (await (await fetch('https://jewel-app-ic3o.onrender.com/vs/log/count')).text());
  console.log('E2E2-LOG after AI', await cnt());
  await page.evaluate(() => { const a = document.querySelector('.jvs-res a:nth-child(2)'); a.addEventListener('click', e => e.preventDefault(), { once: true }); a.click(); });
  await page.waitForTimeout(3000);
  console.log('E2E2-LOG after open', await cnt());
  // next photo keeps the kind
  await page.setInputFiles('.jvs-new input', { name: file, mimeType: 'image/jpeg', buffer: buf }); await page.waitForTimeout(1500);
  console.log('E2E2-NEXT', file, 'ask=' + await page.evaluate(() => document.querySelector('.jvs-full').classList.contains('ask')), 'on=' + await page.evaluate(() => (document.querySelector('.jvs-cats button.on') || {}).textContent));
}
await b.close();
