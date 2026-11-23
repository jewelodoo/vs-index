"""Shopper-photo test: rank of the first matching product for several query strategies (as in the theme)."""
import io, re, json, time, numpy as np, requests, onnxruntime as ort
from PIL import Image
RAW = "https://raw.githubusercontent.com/jewelodoo/vs-index/"
import os as _os
DATA_DIR = _os.environ.get("DATA_DIR")  # test a local index (e.g. out_dual) instead of the published one
class _R:
    def __init__(s, b): s.content = b
    def json(s): import json as _j; return _j.loads(s.content)
def _get(u, *a, **k):
    if DATA_DIR and u.startswith(RAW + "data/"): return _R(open(_os.path.join(DATA_DIR, u[len(RAW) + 5:]), "rb").read())
    return requests.get(u, *a, **k)
M = _get(RAW + "data/manifest.json").json(); d = M["dim"]
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
open("m.onnx", "wb").write(requests.get(M["model"]).content)
sess = ort.InferenceSession("m.onnx", providers=["CPUExecutionProvider"]); inp = sess.get_inputs()[0].name
pb = np.frombuffer(_get(RAW + "data/pca.bin").content, np.float32); D = len(pb) // (d + 1); mean, W = pb[:D], pb[D:].reshape(D, d)
titles = {}
page = 1
while True:
    ps = requests.get(f"https://sa-jewel.com/products.json?limit=250&page={page}", headers={"User-Agent": "jewel-vs-test"}).json()["products"]
    if not ps: break
    for p in ps: titles[p["handle"]] = p["title"]
    page += 1; time.sleep(.3)
def run(ims):
    x = np.stack([(np.asarray(i.convert("RGB").resize((224, 224), Image.BILINEAR), np.float32) / 255 - MEAN) / STD for i in ims]).transpose(0, 3, 1, 2)
    return sess.run(None, {inp: x})[0]
def proj(o):
    o = o[:, 0, :]; o = o / np.linalg.norm(o, axis=1, keepdims=True); p = (o - mean) @ W; return p / np.linalg.norm(p, axis=1, keepdims=True)
def crops(im, spec):
    w, h = im.size; return [im.crop((int((w - w*s)*fx), int((h - h*s)*fy), int((w - w*s)*fx + w*s), int((h - h*s)*fy + h*s))) for s, fx, fy in spec]
def autobox(tok, method="pca", keep=.2, margin=.12):
    cls, P = tok[0], tok[1:]; g = int(len(P) ** .5); Pn = P / np.linalg.norm(P, axis=1, keepdims=True)
    if method == "pca":
        c = Pn - Pn.mean(0); s = c @ np.linalg.svd(c, full_matrices=False)[2][0]; G = s.reshape(g, g)
        if G[g//4:3*g//4, g//4:3*g//4].mean() < np.r_[G[0], G[-1], G[:, 0], G[:, -1]].mean(): s = -s
    else: s = Pn @ (cls / np.linalg.norm(cls))
    m = (s >= np.sort(s)[::-1][int(len(s) * keep)]).reshape(g, g); ys, xs = np.where(m)
    x0, x1, y0, y1 = xs.min()/g, (xs.max()+1)/g, ys.min()/g, (ys.max()+1)/g; mx, my = (x1-x0)*margin, (y1-y0)*margin
    return max(0, x0-mx), max(0, y0-my), min(1, x1+mx), min(1, y1+my)
TESTS = [  # file, category, regex on the product title that counts as a hit
 ("1.jpg", "necklaces", r"دورين 707"), ("2.jpg", "bracelets", r"كارتير"), ("3.jpg", "bracelets", r"بولغاري.*خيط|خيط.*بولغاري"),
 ("4.jpg", "rings", r"توري"), ("5.jpg", "earrings", r"سي اتش|كارولينا"), ("6.jpg", "bracelets", r"فان كليف"),
 ("7.jpg", "necklaces", r"سواروفسكي.*ملون"), ("8.jpg", "bracelets", r"فندي زركون بيق لوقو"), ("9.jpg", "bracelets", r"لويس فيتون.*سمبل"),
 ("10.jpg", "bracelets", r"سواروفسكي"), ("11.jpg", "rings", r"فندي"),
 ("../dior_full.jpg", "earrings", r"ديور لولو ذهبي بالشعار"), ("../dior_sel.jpg", "earrings", r"ديور لولو ذهبي بالشعار"),
 ("12.jpg", "bracelets", "handle:copy-14"), ("13.jpg", "bracelets", "handle:408609"),
 ("14.jpg", "bracelets", "handle:408563"), ("15.jpg", "bracelets", "handle:copy-14|408609|408563|408879"), ("16.jpg", "bracelets", "handle:408996"), ("17.jpg", "earrings", "handle:product-4804")]  # 15 = four bracelets stacked: any of them counts  # 23 Nov: new shopper photos from here on (rx "handle:<h>" = that exact product)
cache = {}
def catdata(cat):
    if cat not in cache:
        meta = _get(RAW + f"data/{cat}.json").json(); V = np.frombuffer(_get(RAW + f"data/{cat}.bin").content, np.int8).reshape(-1, d).astype(np.float32) / 127
        cache[cat] = (meta["handles"], V, np.repeat(np.arange(len(meta["counts"])), meta["counts"]))
    return cache[cat]
def rank(Q, cat, rx):
    hs, V, owner = catdata(cat); S = Q @ V.T; P = np.full((len(Q), len(hs)), -9.0); np.maximum.at(P.T, owner, S.T); a = P.max(0)
    order = np.argsort(-a); hits = [k for k, i in enumerate(order) if (hs[i] in rx[7:].split("|") if rx.startswith("handle:") else re.search(rx, titles.get(hs[i], "")))]
    return (hits[0] + 1 if hits else 999), [titles.get(hs[i], hs[i])[:28] for i in order[:3]]
STRATS = ["full7", "auto_pca_.08", "auto_pca_.1", "auto_pca_.12", "auto_pca_.15", "auto_pca_.2"]
tot = {s: [] for s in STRATS}
for f, cat, rx in TESTS:
    im = Image.open(io.BytesIO(requests.get(RAW + "queries/t/" + f).content)).convert("RGB"); im.thumbnail((900, 900))
    o = run(crops(im, M["crops"])); base = proj(o); tok = o[0]
    line = []
    for s in STRATS:
        if s == "full7": Q = base
        else:
            parts = s.split("_"); box = autobox(tok, parts[1], float(parts[2])); w, h = im.size
            piece = im.crop((int(box[0]*w), int(box[1]*h), int(box[2]*w), int(box[3]*h)))
            Qa = proj(run(crops(piece, M["crops"])))
            Q = Qa if parts[0] == "autoonly" else np.vstack([base, Qa])
        r, top = rank(Q, cat, rx); tot[s].append(r); line.append(f"{s}={r}")
    print("PHOTO", f, cat, " ".join(line), "| top(full7):", rank(base, cat, rx)[1])
for s in STRATS:
    r = np.array(tot[s]); print(f"SUMMARY13 {s:16s} top1={int((r[:13]<=1).sum())}/13 new={list(r[13:])}"); print(f"SUMMARY {s:16s} top1={np.mean(r<=1):.0%} top2={np.mean(r<=2):.0%} top5={np.mean(r<=5):.0%} top10={np.mean(r<=10):.0%} ranks={list(r)}")
