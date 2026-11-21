"""Rank of the true product for real shopper photos, for several query strategies.
Uses the published index (data branch) exactly as the theme does."""
import io, json, numpy as np, requests, onnxruntime as ort
from PIL import Image
RAW = "https://raw.githubusercontent.com/jewelodoo/vs-index/"
M = requests.get(RAW + "data/manifest.json").json()
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
open("m.onnx", "wb").write(requests.get(M["model"]).content)
sess = ort.InferenceSession("m.onnx", providers=["CPUExecutionProvider"]); inp = sess.get_inputs()[0].name
pb = np.frombuffer(requests.get(RAW + "data/pca.bin").content, np.float32); d = M["dim"]; D = len(pb) // (d + 1)
mean, W = pb[:D], pb[D:].reshape(D, d)
def emb(ims):
    x = np.stack([(np.asarray(i.convert("RGB").resize((224, 224), Image.BILINEAR), np.float32) / 255 - MEAN) / STD for i in ims]).transpose(0, 3, 1, 2)
    o = sess.run(None, {inp: x})[0][:, 0, :]; o /= np.linalg.norm(o, axis=1, keepdims=True)
    p = (o - mean) @ W; return p / np.linalg.norm(p, axis=1, keepdims=True)
def crops(im, spec):
    w, h = im.size; out = []
    for s, fx, fy in spec:
        cw, ch = w * s, h * s; x, y = (w - cw) * fx, (h - ch) * fy; out.append(im.crop((int(x), int(y), int(x + cw), int(y + ch))))
    return out
GRID = [(0.6, fx, fy) for fx in (0, .5, 1) for fy in (0, .5, 1)] + [(0.4, fx, fy) for fx in (.25, .5, .75) for fy in (.25, .5, .75)]
STRATS = {"two": [(1, .5, .5), (.75, .5, .5)], "index7": M["crops"], "index7+grid": M["crops"] + GRID}
TESTS = [("dior_full.jpg", "earrings", "copy-of-443"), ("dior_sel.jpg", "earrings", "copy-of-443")]
cache = {}
for f, cat, target in TESTS:
    if cat not in cache:
        meta = requests.get(RAW + f"data/{cat}.json").json(); v = np.frombuffer(requests.get(RAW + f"data/{cat}.bin").content, np.int8).reshape(-1, d).astype(np.float32) / 127
        cache[cat] = (meta, v, np.repeat(np.arange(len(meta["counts"])), meta["counts"]))
    meta, V, owner = cache[cat]; hs = meta["handles"]
    im = Image.open(io.BytesIO(requests.get(RAW + "queries/" + f).content))
    for sn, spec in STRATS.items():
        Q = emb(crops(im, spec)); S = Q @ V.T          # (q, all crops)
        P = np.full((len(Q), len(hs)), -9.0)
        np.maximum.at(P.T, owner, S.T)                 # per query crop, best crop of each product
        aggs = {"max": P.max(0), "meanmax": P.mean(0), "top3": np.sort(P, 0)[-3:].mean(0)}
        for an, a in aggs.items():
            order = np.argsort(-a); rank = int(np.where(np.array(hs)[order] == target)[0][0]) + 1
            print(f"RESULT {f:14s} {sn:12s} {an:8s} rank={rank:4d} top3={[hs[i] for i in order[:3]]}")

# ---- auto-crop: find the piece from DINOv2 patch tokens, then search the crop ----
def tokens(im):
    x = ((np.asarray(im.convert("RGB").resize((224, 224), Image.BILINEAR), np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)[None]
    return sess.run(None, {inp: x})[0][0]          # (257, 384)
def autobox(im, method, keep=0.25, margin=0.12):
    t = tokens(im); cls, P = t[0], t[1:]; g = int(len(P) ** .5)
    Pn = P / np.linalg.norm(P, axis=1, keepdims=True)
    if method == "pca":
        c = Pn - Pn.mean(0); u = np.linalg.svd(c, full_matrices=False)[2][0]; s = c @ u
        grid = s.reshape(g, g); centre = grid[g//4:3*g//4, g//4:3*g//4].mean(); border = np.r_[grid[0], grid[-1], grid[:, 0], grid[:, -1]].mean()
        if centre < border: s = -s
    else:  # "cls": patches most like the image summary
        s = Pn @ (cls / np.linalg.norm(cls))
    m = (s >= np.quantile(s, 1 - keep)).reshape(g, g)
    ys, xs = np.where(m); x0, x1, y0, y1 = xs.min() / g, (xs.max() + 1) / g, ys.min() / g, (ys.max() + 1) / g
    w, h = im.size; mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    box = (max(0, x0 - mx) * w, max(0, y0 - my) * h, min(1, x1 + mx) * w, min(1, y1 + my) * h)
    return im.crop(tuple(int(v) for v in box)), [round(v / s_, 2) for v, s_ in zip(box, (w, h, w, h))]
for f, cat, target in TESTS:
    meta, V, owner = cache[cat]; hs = meta["handles"]
    im = Image.open(io.BytesIO(requests.get(RAW + "queries/" + f).content))
    for method in ("pca", "cls"):
        for keep in (0.2, 0.35, 0.5):
            crop, box = autobox(im, method, keep)
            for mode in ("crop", "crop+orig"):
                Q = emb(crops(crop, M["crops"]) + (crops(im, M["crops"]) if mode == "crop+orig" else []))
                S = Q @ V.T; P = np.full((len(Q), len(hs)), -9.0); np.maximum.at(P.T, owner, S.T); a = P.max(0)
                order = np.argsort(-a); rank = int(np.where(np.array(hs)[order] == target)[0][0]) + 1
                print(f"AUTO {f:14s} {method:4s} keep={keep} {mode:9s} box={box} rank={rank}")
