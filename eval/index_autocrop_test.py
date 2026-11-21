"""Would adding an auto-cropped 'piece' vector per product image (index side) help?
Bracelets + necklaces only: build piece vectors for every product image, then rank the shopper photos."""
import io, re, time, numpy as np, requests, onnxruntime as ort, concurrent.futures as cf
from PIL import Image
import importlib.util, sys
spec = importlib.util.spec_from_file_location("pt", "eval/photo_test_lib.py")
RAW = "https://raw.githubusercontent.com/jewelodoo/vs-index/"
M = requests.get(RAW + "data/manifest.json").json(); d = M["dim"]
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
open("m.onnx", "wb").write(requests.get(M["model"]).content)
sess = ort.InferenceSession("m.onnx", providers=["CPUExecutionProvider"]); inp = sess.get_inputs()[0].name
pb = np.frombuffer(requests.get(RAW + "data/pca.bin").content, np.float32); D = len(pb) // (d + 1); mean, W = pb[:D], pb[D:].reshape(D, d)
S_ = requests.Session(); S_.headers["User-Agent"] = "jewel-vs-test"
prods = {}; page = 1
while True:
    ps = S_.get(f"https://sa-jewel.com/products.json?limit=250&page={page}").json()["products"]
    if not ps: break
    for p in ps: prods[p["handle"]] = p
    page += 1; time.sleep(.3)
def run(ims):
    x = np.stack([(np.asarray(i.convert("RGB").resize((224, 224), Image.BILINEAR), np.float32) / 255 - MEAN) / STD for i in ims]).transpose(0, 3, 1, 2)
    return sess.run(None, {inp: x})[0]
def proj(o):
    o = o[:, 0, :]; o = o / np.linalg.norm(o, axis=1, keepdims=True); p = (o - mean) @ W; return p / np.linalg.norm(p, axis=1, keepdims=True)
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
CATS = ["bracelets", "necklaces"]
data = {}
for cat in CATS:
    meta = requests.get(RAW + f"data/{cat}.json").json(); V = np.frombuffer(requests.get(RAW + f"data/{cat}.bin").content, np.int8).reshape(-1, d).astype(np.float32) / 127
    hs = meta["handles"]; owner = np.repeat(np.arange(len(hs)), meta["counts"])
    urls = [(k, i["src"]) for k, h in enumerate(hs) for i in prods.get(h, {}).get("images", [])[:2]]
    def fetch(a):
        k, u = a
        for t in range(3):
            try:
                r = S_.get(u + ("&" if "?" in u else "?") + "width=448", timeout=60)
                if r.ok: return k, Image.open(io.BytesIO(r.content)).convert("RGB")
            except Exception: pass
        return k, None
    pv, po = [], []
    with cf.ThreadPoolExecutor(16) as ex:
        for k, im in ex.map(fetch, urls):
            if im is None: continue
            tok = run([im])[0]; pv.append(proj(run([piece(im, tok)]))[0]); po.append(k)
    data[cat] = (hs, V, owner, np.array(pv), np.array(po)); print(cat, "products", len(hs), "piece vectors", len(pv), flush=True)
TITLE = {h: p["title"] for h, p in prods.items()}
TESTS = [("2.jpg", "bracelets", r"كارتير"), ("3.jpg", "bracelets", r"بولغاري.*خيط|خيط.*بولغاري"), ("6.jpg", "bracelets", r"فان كليف"),
         ("8.jpg", "bracelets", r"فندي زركون بيق لوقو"), ("9.jpg", "bracelets", r"لويس فيتون.*سمبل"), ("10.jpg", "bracelets", r"^اسواره سواروفسكي اريانا غراندي بانقل ماستر$"),
         ("1.jpg", "necklaces", r"دورين 707"), ("7.jpg", "necklaces", r"سواروفسكي عقد ملون$")]
for f, cat, rx in TESTS:
    hs, V, owner, PV, PO = data[cat]
    im = Image.open(io.BytesIO(requests.get(RAW + "queries/t/" + f).content)).convert("RGB"); im.thumbnail((900, 900))
    o = run(crops(im, M["crops"])); base = proj(o); Qa = proj(run(crops(piece(im, o[0]), M["crops"]))); Q = np.vstack([base, Qa])
    def score(useP):
        S = Q @ V.T; P = np.full((len(Q), len(hs)), -9.0); np.maximum.at(P.T, owner, S.T)
        if useP:
            S2 = Q @ PV.T; np.maximum.at(P.T, PO, S2.T)
        a = P.max(0); order = np.argsort(-a)
        hits = [k for k, i in enumerate(order) if re.search(rx, TITLE.get(hs[i], ""))]
        return hits[0] + 1 if hits else 999
    print("IDXTEST", f, "current", score(False), "with_piece_index", score(True), flush=True)
