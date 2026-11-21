"""Full theme protocol (7 index crops per image, query = 7 crops of the photo + 7 crops of the auto-cropped piece)
for DINOv2-small and DINOv3-small, plus SigLIP B/16 (whole + piece) as an optional second stage.
Prints the rank of the right product for each model and for combinations. Usage: python eval/full_compare.py <category>"""
import io, os, re, sys, time, numpy as np, requests, onnxruntime as ort, concurrent.futures as cf
from PIL import Image
CAT = sys.argv[1]
RAW = "https://raw.githubusercontent.com/jewelodoo/vs-index/"
IMN = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
MODELS = {
  "v2": ("https://huggingface.co/Xenova/dinov2-small/resolve/main/onnx/model_quantized.onnx", "cls", IMN),
  "v3": ("https://huggingface.co/onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX/resolve/main/onnx/model_quantized.onnx", "cls", IMN),
  "sg": ("https://huggingface.co/Xenova/siglip-base-patch16-224/resolve/main/onnx/vision_model_quantized.onnx", "pool", ([.5, .5, .5], [.5, .5, .5])),
}
CROPS = requests.get(RAW + "data/manifest.json").json()["crops"]
sess = {}
for n, (url, kind, (m, s)) in MODELS.items():
    os.makedirs(n, exist_ok=True); fn = os.path.join(n, url.rsplit("/", 1)[1])
    open(fn, "wb").write(requests.get(url, timeout=600).content)
    rr = requests.get(url + "_data", timeout=600)
    if rr.ok: open(fn + "_data", "wb").write(rr.content)
    sess[n] = (ort.InferenceSession(fn, providers=["CPUExecutionProvider"]), kind, np.array(m, np.float32), np.array(s, np.float32))
def run(n, ims):
    se, kind, M, S = sess[n]
    x = np.stack([(np.asarray(i.convert("RGB").resize((224, 224), Image.BILINEAR), np.float32) / 255 - M) / S for i in ims]).transpose(0, 3, 1, 2)
    outs = dict(zip([o.name for o in se.get_outputs()], se.run(None, {se.get_inputs()[0].name: x})))
    if kind == "cls": t = outs[se.get_outputs()[0].name]; v = t[:, 0, :]
    else:
        t = None; v = outs.get("image_embeds", outs.get("pooler_output"))
        if v is None: v = list(outs.values())[0]
        if v.ndim == 3: v = v.mean(1)
    v = v.astype(np.float32); return v / np.linalg.norm(v, axis=1, keepdims=True), t
def crops(im, spec):
    w, h = im.size; return [im.crop((int((w - w*s)*fx), int((h - h*s)*fy), int((w - w*s)*fx + w*s), int((h - h*s)*fy + h*s))) for s, fx, fy in spec]
def autobox(tok, keep=.15, margin=.12):
    P = tok[1:]; g = int(len(P) ** .5); Pn = P / np.linalg.norm(P, axis=1, keepdims=True)
    c = Pn - Pn.mean(0); s = c @ np.linalg.svd(c, full_matrices=False)[2][0]; G = s.reshape(g, g)
    if G[g//4:3*g//4, g//4:3*g//4].mean() < np.r_[G[0], G[-1], G[:, 0], G[:, -1]].mean(): s = -s
    m = (s >= np.sort(s)[::-1][int(len(s) * keep)]).reshape(g, g); ys, xs = np.where(m)
    x0, x1, y0, y1 = xs.min()/g, (xs.max()+1)/g, ys.min()/g, (ys.max()+1)/g; mx, my = (x1-x0)*margin, (y1-y0)*margin
    return max(0, x0-mx), max(0, y0-my), min(1, x1+mx), min(1, y1+my)
def piece(im, tok):
    b = autobox(tok); w, h = im.size; return im.crop((int(b[0]*w), int(b[1]*h), int(b[2]*w), int(b[3]*h)))
S_ = requests.Session(); S_.headers["User-Agent"] = "jewel-vs-test"
prods = {}; page = 1
while True:
    ps = S_.get(f"https://sa-jewel.com/products.json?limit=250&page={page}").json()["products"]
    if not ps: break
    for p in ps: prods[p["handle"]] = p
    page += 1; time.sleep(.3)
TITLE = {h: p["title"] for h, p in prods.items()}
hs = requests.get(RAW + f"data/{CAT}.json").json()["handles"]
urls = [(k, i["src"]) for k, h in enumerate(hs) for i in prods.get(h, {}).get("images", [])[:2]]
def fetch(a):
    k, u = a
    for t in range(3):
        try:
            r = S_.get(u + ("&" if "?" in u else "?") + "width=448", timeout=60)
            if r.ok: return k, Image.open(io.BytesIO(r.content)).convert("RGB")
        except Exception: time.sleep(1)
    return k, None
IDX = {"v2": [], "v3": [], "sg": []}; OWN = {"v2": [], "v3": [], "sg": []}; t0 = time.time(); ms = {n: 0.0 for n in sess}
with cf.ThreadPoolExecutor(16) as ex:
    for k, im in ex.map(fetch, urls):
        if im is None: continue
        cs = crops(im, CROPS)
        for n in ("v2", "v3"):
            a = time.time(); v, t = run(n, cs); ms[n] += time.time() - a; IDX[n].append(v); OWN[n] += [k] * len(v)
            if n == "v2": tok = t[0]
        a = time.time(); v, _ = run("sg", [im, piece(im, tok)]); ms["sg"] += time.time() - a; IDX["sg"].append(v); OWN["sg"] += [k, k]
nimg = len(OWN["sg"]) // 2
print("INDEX", CAT, len(hs), "products", nimg, "images", round(time.time() - t0), "s; ms per image:", {n: round(1000 * ms[n] / max(nimg, 1)) for n in ms}, flush=True)
for n in IDX: IDX[n] = np.concatenate(IDX[n]); OWN[n] = np.array(OWN[n])
TESTS = {"bracelets": [("t/2.jpg", r"كارتير"), ("t/3.jpg", r"بولغاري.*خيط|خيط.*بولغاري"), ("t/6.jpg", r"فان كليف"), ("t/8.jpg", r"فندي زركون بيق لوقو"),
                       ("t/9.jpg", r"لويس فيتون.*سمبل"), ("t/10.jpg", r"^اسواره سواروفسكي اريانا غراندي بانقل ماستر$")],
         "necklaces": [("t/1.jpg", r"دورين 707"), ("t/7.jpg", r"سواروفسكي.*ملون")],
         "earrings": [("t/5.jpg", r"سي اتش|كارولينا"), ("dior_full.jpg", r"ديور لولو ذهبي بالشعار"), ("dior_sel.jpg", r"ديور لولو ذهبي بالشعار")],
         "rings": [("t/4.jpg", r"توري"), ("t/11.jpg", r"فندي")]}[CAT]
def prodscore(n, Q):
    S = Q @ IDX[n].T; P = np.full((len(Q), len(hs)), -9.0); np.maximum.at(P.T, OWN[n], S.T); return P.max(0)
def z(a): return (a - a.mean()) / (a.std() + 1e-9)
def rr(a): r = np.empty(len(a)); r[np.argsort(-a)] = np.arange(1, len(a) + 1); return r
def rank(a, rx):
    order = np.argsort(-a); hits = [j for j, i in enumerate(order) if re.search(rx, TITLE.get(hs[i], ""))]; return hits[0] + 1 if hits else 999
for f, rx in TESTS:
    im = Image.open(io.BytesIO(requests.get(RAW + "queries/" + f).content)).convert("RGB"); im.thumbnail((900, 900))
    cs = crops(im, CROPS); _, t = run("v2", cs); pc = piece(im, t[0]); qs = cs + crops(pc, CROPS)
    s2 = prodscore("v2", run("v2", qs)[0]); s3 = prodscore("v3", run("v3", qs)[0]); sg = prodscore("sg", run("sg", [im, pc])[0])
    fz = z(s2) + z(s3); frr = 1 / (10 + rr(s2)) + 1 / (10 + rr(s3))
    top = np.argsort(-fz)[:50]; two = fz.copy(); two[top] += 10 + z(sg)[top]          # stage 2: SigLIP re-ranks the top 50 of the vote
    three = z(s2) + z(s3) + z(sg)
    print("FULL", CAT, f, f"v2={rank(s2, rx)} v3={rank(s3, rx)} sg={rank(sg, rx)} vote_z={rank(fz, rx)} vote_rr={rank(frr, rx)} vote+sg_top50={rank(two, rx)} all3={rank(three, rx)}", flush=True)
