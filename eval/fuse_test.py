"""Blend test: whole-photo score + piece score per product (FUSE=1 index in DATA_DIR), theme protocol (7 crops + auto-crop 15%).
Prints the rank of the right product for several blends, for the 13 shopper photos + ch.jpg + bv.jpg."""
import io, re, os, json, numpy as np, requests
from PIL import Image
exec(open("eval/photo_test.py").read().split("cache = {}")[0].replace("TESTS = [", "TESTS0 = ["))
TESTS = TESTS0 + [("ch.jpg", "earrings", r"سي اتش"), ("bv.jpg", "bracelets", "handle:408879")]
DD = os.environ.get("DATA_DIR", "out_fuse")
C = {}
def cat(c):
    if c not in C:
        m = json.load(open(f"{DD}/{c}.json")); V = np.fromfile(f"{DD}/{c}.bin", np.int8).reshape(-1, d).astype(np.float32) / 127
        pm = json.load(open(f"{DD}/{c}.p.json")); P = np.fromfile(f"{DD}/{c}.p.bin", np.int8).reshape(-1, d).astype(np.float32) / 127
        C[c] = (m["handles"], V, np.repeat(np.arange(len(m["counts"])), m["counts"]), P, np.repeat(np.arange(len(pm["counts"])), pm["counts"]))
    return C[c]
def per_prod(Q, V, owner, n):
    S = (Q @ V.T).max(0); a = np.full(n, -9.0); np.maximum.at(a, owner, S); return a
def hit(h, rx): return h in rx[7:].split("|") if rx.startswith("handle:") else bool(re.search(rx, titles.get(h, "")))
BL = {}
def blend(name):
    def f(w, p):
        p = np.where(p < -1, w, p)  # no piece vector: use the whole score
        if name == "whole": return w
        if name == "max": return np.maximum(w, p)
        if name.startswith("a"): a = float(name[1:]); return a * w + (1 - a) * p
        if name.startswith("bonus"): b = float(name[5:]); return w + b * np.maximum(0, p - w)
        if name.startswith("top50_a"):
            a = float(name[7:]); s = w.copy(); top = np.argsort(-w)[:50]; s[top] = 10 + a * w[top] + (1 - a) * p[top]; return s
        if name == "rrf":
            rw = np.argsort(np.argsort(-w)); rp = np.argsort(np.argsort(-p)); return 1 / (60 + rw) + 1 / (60 + rp)
    return f
NAMES = ["whole", "max", "a0.9", "a0.8", "a0.7", "a0.6", "a0.5", "bonus0.3", "bonus0.5", "top50_a0.7", "top50_a0.5", "rrf"]
R = {n: [] for n in NAMES}
for f, c, rx in TESTS:
    im = Image.open(io.BytesIO(requests.get(RAW + "queries/t/" + f).content)).convert("RGB"); im.thumbnail((900, 900))
    o = run(crops(im, M["crops"])); base = proj(o); box = autobox(o[0], "pca", .15); w_, h_ = im.size
    piece = im.crop((int(box[0]*w_), int(box[1]*h_), int(box[2]*w_), int(box[3]*h_)))
    Q = np.vstack([base, proj(run(crops(piece, M["crops"])))])
    hs, V, ow, P, pow_ = cat(c); n = len(hs)
    ws, ps = per_prod(Q, V, ow, n), per_prod(Q, P, pow_, n)
    line = []
    for nm in NAMES:
        order = np.argsort(-blend(nm)(ws, ps)); r = next((k + 1 for k, i in enumerate(order) if hit(hs[i], rx)), 999)
        R[nm].append(r); line.append(f"{nm}={r}")
    print("PHOTO", f, " ".join(line))
for nm in NAMES:
    r = np.array(R[nm][:13]); x = R[nm][13:]
    print(f"SUMMARY {nm:11s} top1={int((r<=1).sum())}/13 top2={int((r<=2).sum())}/13 top5={int((r<=5).sum())}/13 ranks={list(r)} ch,bv={x}")
