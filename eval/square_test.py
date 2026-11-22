"""Square crops of the piece (no squashing): reuse the OWLv2 boxes of the 'Piece detection test' run.
  python eval/square_test.py shard I N  -> sq_I.npz
  python eval/square_test.py eval       -> ranks: squashed piece (old run) vs square piece, query piece squashed vs square"""
import io, os, re, sys, glob, time, numpy as np, requests, concurrent.futures as cf
from PIL import Image
sys.argv += [""] * 3
os.environ.setdefault("NO_DET", "1")
import importlib.util
spec = importlib.util.spec_from_file_location("dt", "eval/detect_test.py")
src = open("eval/detect_test.py").read().split("\nif sys.argv[1] == \"shard\":")[0]
g = {"__name__": "dt"}; exec(src, g)
B, tok, cls, get, S, RAW = g["B"], g["tok"], g["cls"], g["get"], g["S"], g["RAW"]
def square(im, box, margin=.12, pad=255):
    w, h = im.size; x0, y0, x1, y1 = box[0]*w, box[1]*h, box[2]*w, box[3]*h
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; side = max(x1 - x0, y1 - y0) * (1 + 2 * margin)
    c = Image.new("RGB", (int(side), int(side)), (pad, pad, pad))
    c.paste(im.crop((int(cx - side/2), int(cy - side/2), int(cx + side/2), int(cy + side/2))), (0, 0))
    return c
def load_det():
    U, BX, E = [], [], []
    for f in sorted(glob.glob("det/**/det_*.npz", recursive=True)):
        z = np.load(f); U += list(z["urls"]); BX += list(z["boxes"]); E += list(z["emb"])
    return U, BX, E
if sys.argv[1] == "shard":
    I, N = int(sys.argv[2]), int(sys.argv[3]); U, BX, _ = load_det(); idx = list(range(len(U)))[I::N]
    out = []
    with cf.ThreadPoolExecutor(8) as ex:
        for k, im in zip(idx, ex.map(lambda k: get(U[k], 768), idx)):
            if im is None: out.append(np.zeros((7, 384), np.float16)); continue
            out.append(cls(tok(B.crops(square(im, BX[k])))).astype(np.float16))
    np.savez(f"sq_{I}.npz", idx=np.array(idx), emb=np.stack(out)); sys.exit()
# eval
U, BX, E = load_det(); Esq = [None] * len(U)
for f in glob.glob("sq/**/sq_*.npz", recursive=True):
    z = np.load(f)
    for k, e in zip(z["idx"], z["emb"]): Esq[k] = e
pos = {u: k for k, u in enumerate(U)}
M = S.get(RAW + "data/manifest.json").json(); d = M["dim"]
pb = np.frombuffer(S.get(RAW + "data/pca.bin").content, np.float32); D = len(pb) // (d + 1); mean, W = pb[:D], pb[D:].reshape(D, d)
def proj(c):
    p = (np.asarray(c, np.float32) - mean) @ W; return p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-9)
meta = S.get(RAW + "data/bracelets.json").json(); hs = meta["handles"]
Vcur = np.frombuffer(S.get(RAW + "data/bracelets.bin").content, np.int8).reshape(-1, d).astype(np.float32) / 127
own_cur = np.repeat(np.arange(len(hs)), meta["counts"])
allp = {p["handle"]: p for p in B.feed("/products.json")}; titles = {h: p["title"] for h, p in allp.items()}
def build(EE):
    pv, po = [], []
    for j, h in enumerate(hs):
        for i in allp.get(h, {}).get("images", [])[:B.IMGS]:
            k = pos.get(i["src"])
            if k is not None and EE[k] is not None: pv.append(proj(EE[k])); po += [j] * 7
    return np.concatenate(pv), np.array(po)
IDX = {"cur": (Vcur, own_cur), "piece_squash": build(E), "piece_square": build(Esq)}
IDX["square+squash"] = (np.vstack([IDX["piece_square"][0], IDX["piece_squash"][0]]), np.r_[IDX["piece_square"][1], IDX["piece_squash"][1]])
autobox = g.get("autobox")
def autobox(t, keep=.15):
    P = t[1:]; gg = int(len(P) ** .5); Pn = P / np.linalg.norm(P, axis=1, keepdims=True); c = Pn - Pn.mean(0)
    s = c @ np.linalg.svd(c, full_matrices=False)[2][0]; G = s.reshape(gg, gg)
    if G[gg//4:3*gg//4, gg//4:3*gg//4].mean() < np.r_[G[0], G[-1], G[:, 0], G[:, -1]].mean(): s = -s
    m = (s >= np.sort(s)[::-1][int(len(s) * keep)]).reshape(gg, gg); ys, xs = np.where(m)
    return xs.min()/gg, ys.min()/gg, (xs.max()+1)/gg, (ys.max()+1)/gg
TESTS = [("2.jpg", r"كارتير"), ("3.jpg", r"بولغاري.*خيط|خيط.*بولغاري"), ("6.jpg", r"فان كليف"), ("8.jpg", r"فندي زركون بيق لوقو"),
         ("9.jpg", r"لويس فيتون.*سمبل"), ("10.jpg", r"اريانا غراندي بانقل"), ("bv.jpg", r"بولغاري نيو ستايل زركون")]
def rank(Q, V, own, rx):
    Sm = Q @ V.T; P = np.full((len(Q), len(hs)), -9.0); np.maximum.at(P.T, own, Sm.T); order = np.argsort(-P.max(0))
    return next((k + 1 for k, i in enumerate(order) if re.search(rx, titles.get(hs[i], ""))), 999)
tot = {}
for f, rx in TESTS:
    im = Image.open(io.BytesIO(S.get(RAW + "queries/t/" + f).content)).convert("RGB"); im.thumbnail((900, 900))
    o = tok(B.crops(im)); base = proj(cls(o)); bx = autobox(o[0]); w, h = im.size
    q_squash = proj(cls(tok(B.crops(im.crop((int(bx[0]*w), int(bx[1]*h), int(bx[2]*w), int(bx[3]*h)))))))
    q_square = proj(cls(tok(B.crops(square(im, bx)))))
    Qs = {"q_squash": np.vstack([base, q_squash]), "q_square": np.vstack([base, q_square]), "q_square_only": q_square, "q_both": np.vstack([base, q_squash, q_square])}
    line = []
    for qn, Q in Qs.items():
        for vn, (V, own) in IDX.items():
            r = rank(Q, V, own, rx); tot.setdefault(f"{qn} x {vn}", []).append(r); line.append(f"{qn}|{vn}={r}")
    print("PHOTO", f, " ".join(line), flush=True)
for k, r in tot.items():
    r = np.array(r); print(f"SUMMARY {k:30s} top1={np.sum(r<=1)}/{len(r)} top3={np.sum(r<=3)} sum_log={np.log(r).sum():.1f} ranks={[int(x) for x in r]}")
