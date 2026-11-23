// run: auto-crop check
// End-to-end: open the preview theme on a phone-sized Chrome, run a visual search, report rank and timings.
import { chromium, devices } from 'playwright';
const URL0 = 'https://sa-jewel.com/?preview_theme_id=132002938938';
const Q = 'https://raw.githubusercontent.com/jewelodoo/vs-index/queries/';
const tests = [['t/12.jpg', 'copy-14', null, 'أساور'], ['t/13.jpg', '408609', null, 'أساور'], ['t/14.jpg', '408563', null, 'أساور'], ['t/10.jpg', '408996', null, 'أساور'], ['t/ch.jpg', 'product-4957', null, 'حلق'], ['t/bv.jpg', '408879', null, 'أساور']];
const b = await chromium.launch();
const ctx = await b.newContext({ ...devices['iPhone 13'] });
const page = await ctx.newPage();
page.on('pageerror', e => console.log('PAGEERROR', e.message));
page.on('console', m => { if (m.type() === 'error') console.log('CONSOLE', m.text().slice(0, 200)); });
await page.goto(URL0, { waitUntil: 'load' }); await page.waitForTimeout(4000);
console.log('snippet', await page.evaluate(() => [...document.scripts].some(s => s.textContent.includes('embedMany'))));
for (const [file, target, sel, cat] of tests) {
  await page.evaluate(() => (document.querySelector('.jvs-btn') || document.querySelector('.jvs-in')).click());
  await page.waitForTimeout(1500);
  const buf = Buffer.from(await (await fetch(Q + file)).arrayBuffer());
  const t0 = Date.now();
  await page.setInputFiles('.jvs-pick label:nth-child(2) input', { name: file, mimeType: 'image/jpeg', buffer: buf });
  await page.waitForTimeout(400);
  const hint = await page.evaluate(() => { const h = document.querySelector('.jvs-hint'); return h && h.classList.contains('on'); });
  await page.waitForSelector('.jvs-cats button', { timeout: 30000 }).catch(() => {});
  console.log('cats', await page.evaluate(c => [...document.querySelectorAll('.jvs-cats button')].map(x => x.textContent).join('|')));
  await page.evaluate(c => [...document.querySelectorAll('.jvs-cats button')].find(x => x.textContent.trim() === c).click(), cat);
  if (sel) {
    const box = await page.locator('.jvs-stage canvas').boundingBox();
    await page.mouse.move(box.x + box.width * sel[0], box.y + box.height * sel[1]); await page.mouse.down();
    await page.mouse.move(box.x + box.width * sel[2], box.y + box.height * sel[3], { steps: 8 }); await page.mouse.up();
  }
  await page.waitForTimeout(3000); // shopper choosing the category
  const t1 = Date.now();
  await page.click('.jvs-go');
  const okRes = await page.waitForSelector('.jvs-res a', { timeout: 60000 }).then(() => true).catch(() => false);
  if (!okRes) { console.log('NORES', JSON.stringify(await page.evaluate(() => ({ msg: document.querySelector('.jvs-msg').textContent, go: document.querySelector('.jvs-go').disabled, sel: document.querySelector('.jvs-stage') && getComputedStyle(document.querySelector('.jvs-stage')).display })))); continue; }
  const t2 = Date.now();
  const r = await page.evaluate(t => { const as = [...document.querySelectorAll('.jvs-res a')]; return { n: as.length, rank: as.findIndex(a => a.getAttribute('href').includes(t)) + 1, top: as.slice(0, 8).map(a => a.getAttribute('href').split('/products/')[1].split('?')[0] + ' ' + a.textContent.trim().slice(0, 40)) }; }, target);
  const box = await page.evaluate(() => window.__jvsBox ? Object.values(window.__jvsBox).map(v => v.toFixed(2)) : null);
  console.log('E2E', file, 'box=' + JSON.stringify(box), sel ? 'selected' : 'whole', 'hint=' + hint, 'wait_after_click_ms=' + (t2 - t1), 'from_pick_ms=' + (t2 - t0), JSON.stringify(r));
  const ai = await page.evaluate(() => { const b = document.querySelector('.jvs-ai'); return b && getComputedStyle(b).display !== 'none'; });
  let aiRes = null;
  if (ai) {
    const t3 = Date.now();
    const [resp] = await Promise.all([page.waitForResponse(r => r.url().includes('/vs/rerank') && r.request().method() === 'POST', { timeout: 90000 }).catch(() => null), page.click('.jvs-ai')]);
    if (resp) { await page.waitForTimeout(2500); aiRes = { status: resp.status(), body: (await resp.text()).slice(0, 120), ms: Date.now() - t3, msg: await page.evaluate(() => document.querySelector('.jvs-msg').textContent), exact: await page.evaluate(() => (document.querySelector('.jvs-res a.x') || {}).textContent || null), rank: await page.evaluate(t => [...document.querySelectorAll('.jvs-res a')].findIndex(a => a.getAttribute('href').includes(t)) + 1, target), top: await page.evaluate(() => [...document.querySelectorAll('.jvs-res a')].slice(0, 8).map(a => a.getAttribute('href').split('/products/')[1].split('?')[0] + ' ' + a.textContent.trim().slice(0, 40))), info: await page.evaluate(t => { const i = window.__jvsAi; return i ? { brand: i.brand, err: i.err, cands: i.cands.length, picks: i.picks, scanned: i.scanned, sure: i.sure, why: i.why, ms: i.ms, target_in_cands: i.cands.indexOf(t) + 1 } : null; }, target) }; }
  }
  const after = await page.evaluate(() => getComputedStyle(document.querySelector('.jvs-ai')).display);
  await page.click('.jvs-go'); await page.waitForTimeout(6000);
  const again = await page.evaluate(() => getComputedStyle(document.querySelector('.jvs-ai')).display);
  const named = await page.evaluate(() => /claude|anthropic/i.test(document.documentElement.outerHTML));
  console.log('E2E-AI', file, 'exact_ok=' + (aiRes && aiRes.exact ? 'see top[0]' : 'none'), 'button=' + ai, 'after_press=' + after, 'after_new_search_same_photo=' + again, 'model_name_in_page=' + named, JSON.stringify(aiRes));
  await page.evaluate(() => document.querySelector('.jvs-x').click());
  await page.reload({ waitUntil: 'load' }); await page.waitForTimeout(25000); // the store answers 429 if photos run back to back
}
await b.close();
