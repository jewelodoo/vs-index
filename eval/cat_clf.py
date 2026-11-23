"""Which kind of piece is in the photo? A small linear classifier on DINOv2 features (384 -> categories), for the theme's «الكل».
Train: the store's photos (cache/emb_dino.pkl, 7 crops each) + fake shopper photos (synth/*.npz: jewel cut out on a wrist/cloth).
Test: the shopper photos of eval/photo_test.py (their category is known), theme protocol (whole + auto-crop), against the
current vote (40 closest products in all.bin). Writes out_clf/clf.bin (float32: W [K,384] then b [K]) + clf.json (keys)."""
import os, io, glob, pickle, json, zlib, numpy as np, torch, requests
from PIL import Image
import sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import build_index as B

allp, members = B.catalog()
keys = [k for k, _, _ in B.CATS if members.get(k)]; ki = {k: i for i, k in enumerate(keys)}
cache = pickle.load(open(B.CACHE, "rb"))
X, Y, G = [], [], []  # G: product hash (hold out 10% of products)
for k in keys:
    for h in members[k]:
        g = zlib.crc32(h.encode()) % 10
        for u in B.img_urls(allp[h]):
            v = cache.get(u)
            if v is None: continue
            X.append(v.astype(np.float32)); Y += [ki[k]] * len(v); G += [g] * len(v)
X = np.concatenate(X); Y = np.array(Y); G = np.array(G); S = np.zeros(len(Y), bool)
hc = {h: k for k in keys for h in members[k]}
for f in sorted(glob.glob("synth/synth_*.npz")):
    z = np.load(f)
    for q, h in zip(z["Q"], z["H"]):
        h = str(h)
        if h not in hc: continue
        X = np.concatenate([X, q.astype(np.float32)]); Y = np.concatenate([Y, [ki[hc[h]]] * len(q)])
        G = np.concatenate([G, [zlib.crc32(h.encode()) % 10] * len(q)]); S = np.concatenate([S, np.ones(len(q), bool)])
print("samples", len(X), "synthetic", int(S.sum()), "classes", keys, flush=True)
tr, te = G != 0, G == 0
cnt = np.bincount(Y[tr], minlength=len(keys)).astype(np.float32); wcls = torch.tensor((cnt.sum() / (len(keys) * np.maximum(cnt, 1))) ** .5)
Xt, Yt = torch.tensor(X[tr]), torch.tensor(Y[tr])
lin = torch.nn.Linear(384, len(keys)); opt = torch.optim.Adam(lin.parameters(), lr=3e-3, weight_decay=1e-5)
for ep in range(30):
    perm = torch.randperm(len(Xt))
    for i in range(0, len(Xt), 4096):
        b = perm[i:i + 4096]; loss = torch.nn.functional.cross_entropy(lin(Xt[b] * 10), Yt[b], weight=wcls)
        opt.zero_grad(); loss.backward(); opt.step()
    if ep % 5 == 4:
        with torch.no_grad():
            p = lin(torch.tensor(X[te]) * 10).argmax(1).numpy()
        print(f"epoch {ep} heldout acc store={np.mean(p[~S[te]] == Y[te][~S[te]]):.1%} synthetic={np.mean(p[S[te]] == Y[te][S[te]]):.1%}", flush=True)
W = (lin.weight.detach().numpy() * 10).astype(np.float32); b = lin.bias.detach().numpy().astype(np.float32)
os.makedirs("out_clf", exist_ok=True); np.concatenate([W.ravel(), b]).astype(np.float32).tofile("out_clf/clf.bin")
json.dump({"keys": keys, "dim": 384}, open("out_clf/clf.json", "w"))

# shopper photos: classifier vs the current vote
exec(open("eval/photo_test.py").read().split("cache = {}")[0].replace("TESTS = [", "TESTS0 = ["), g := {})
tests = g["TESTS0"] + [("ch.jpg", "earrings", ""), ("bv.jpg", "bracelets", "")]
M = g["M"]; RAW = g["RAW"]
A = np.fromfile(io.BytesIO(g["_get"](RAW + "data/all.bin").content), np.int8) if False else np.frombuffer(g["_get"](RAW + "data/all.bin").content, np.int8)
acats = g["_get"](RAW + "data/all.json").json()["cats"]; A = A.reshape(len(acats), -1).astype(np.float32) / 127
mcats = [c["key"] for c in M["cats"]]
ok_c = ok_v = 0
for f, c, _ in tests:
    im = Image.open(io.BytesIO(requests.get(RAW + "queries/t/" + f).content)).convert("RGB"); im.thumbnail((900, 900))
    o = g["run"](g["crops"](im, M["crops"])); raw = o[:, 0, :] / np.linalg.norm(o[:, 0, :], axis=1, keepdims=True)
    box = g["autobox"](o[0], "pca", .15); w, h = im.size
    if (box[2] - box[0]) * (box[3] - box[1]) < .85:
        p2 = g["run"](g["crops"](im.crop((int(box[0]*w), int(box[1]*h), int(box[2]*w), int(box[3]*h))), M["crops"]))
        raw = np.vstack([raw, p2[:, 0, :] / np.linalg.norm(p2[:, 0, :], axis=1, keepdims=True)])
    logits = raw @ W.T + b; pc = keys[int(np.argmax(logits.mean(0)))]
    pr = np.exp(logits - logits.max(1, keepdims=True)); pr /= pr.sum(1, keepdims=True); conf = pr.mean(0); top2 = [keys[i] for i in np.argsort(-conf)[:2]]
    Q = g["proj"](np.concatenate([o, p2]) if raw.shape[0] > 7 else o)
    sc = (Q @ A.T).max(0); top = np.argsort(-sc)[:40]; v = {}
    for r, p in enumerate(top): v[acats[p]] = v.get(acats[p], 0) + 1 / (r + 5)
    pv = mcats[max(v, key=v.get)]
    ok_c += pc == c; ok_v += pv == c
    print(f"PHOTO {f:18s} true={c:10s} clf={pc:10s} ({conf.max():.2f}, top2={'+'.join(top2)}) vote={pv}")
print(f"SUMMARY clf={ok_c}/{len(tests)} vote={ok_v}/{len(tests)}")
