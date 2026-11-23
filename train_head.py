"""Train the 384->96 projection (the theme's pca.bin) so a jewel on a wrist/cloth lands next to the same jewel on its box.
Starts from PCA; shopper-like queries = synth/*.npz (synth.py); keeps real matches with a second loss (photo 2 of a product -> photo 1)
and stays near PCA (L2). Writes head.npz (mean, W) for build_index.py (HEAD=head.npz). Prints held-out top-1 before/after."""
import os, glob, pickle, zlib, numpy as np, torch
import build_index as B

torch.manual_seed(0); rng = np.random.default_rng(0)
STEPS = int(os.environ.get("HEAD_STEPS", "3000")); LAM = float(os.environ.get("HEAD_LAM", "1.0")); TAU = 0.05; BS = 192
allp, members = B.catalog()
cache = pickle.load(open(B.CACHE, "rb")); print("cache", len(cache))
JEWEL = ["bracelets", "necklaces", "earrings", "rings", "sets", "other"]
prods, pcat, pimg = [], [], []  # per product: (n,384) vectors, category, image id per vector
for c in JEWEL:
    for h in members.get(c, []):
        vs = [(k, cache[u].astype(np.float32)) for k, u in enumerate(B.img_urls(allp[h])) if cache.get(u) is not None]
        if not vs: continue
        prods.append((h, np.concatenate([v for _, v in vs]), np.concatenate([np.full(len(v), k) for k, v in vs]))); pcat.append(c)
hidx = {p[0]: i for i, p in enumerate(prods)}; pcat = np.array(pcat)
Qs, Qp = [], []
for f in sorted(glob.glob("synth/synth_*.npz")):
    z = np.load(f)
    for q, h in zip(z["Q"], z["H"]):
        if str(h) in hidx: Qs.append(q.astype(np.float32)); Qp.append(hidx[str(h)])
Qs = np.stack(Qs); Qp = np.array(Qp); print("synthetic queries", len(Qs), "products", len(prods))
hold = np.array([zlib.crc32(p[0].encode()) % 10 == 0 for p in prods])  # 10% of products held out
X = np.concatenate([p[1] for p in prods]); samp = X[rng.choice(len(X), min(len(X), 40000), replace=False)]
mean = samp.mean(0); _, _, Vt = np.linalg.svd(samp - mean, full_matrices=False); W0 = Vt[:B.DIM].T.astype(np.float32)
NV = 14
def pad(ix, img=None):
    out = np.zeros((len(ix), NV, 384), np.float32); m = np.zeros((len(ix), NV), bool)
    for r, i in enumerate(ix):
        v = prods[i][1] if img is None else prods[i][1][prods[i][2] == img]
        v = v[:NV]; out[r, :len(v)] = v; m[r, :len(v)] = True
    return torch.tensor(out), torch.tensor(m)
mt = torch.tensor(mean)
W = torch.nn.Parameter(torch.tensor(W0.copy())); W0t = torch.tensor(W0)
def proj(x, Wm): y = (x - mt) @ Wm; return y / (y.norm(dim=-1, keepdim=True) + 1e-9)
def score(q, P, M, Wm):  # q (b,7,384), P (c,NV,384) -> (b,c) max over all pairs
    a, b = proj(q, Wm), proj(P, Wm); s = torch.einsum("bid,cjd->bcij", a, b)
    s = s.masked_fill(~M[None, :, None, :], -9); return s.amax(dim=(2, 3))
opt = torch.optim.Adam([W], lr=float(os.environ.get("HEAD_LR", "5e-4")))
train_q = np.where(~hold[Qp])[0]
two = [i for i, p in enumerate(prods) if not hold[i] and len(set(p[2].tolist())) > 1]
by_cat = {c: train_q[pcat[Qp[train_q]] == c] for c in JEWEL}
by_cat = {c: v for c, v in by_cat.items() if len(v) > BS}
def evaluate(Wm, tag):
    with torch.no_grad():
        res = []
        for c in JEWEL:
            cand = np.where(pcat == c)[0]
            if not len(cand): continue
            qi = np.where(hold[Qp] & (pcat[Qp] == c))[0][:300]
            if not len(qi): continue
            P, M = pad(cand); pos = {p: k for k, p in enumerate(cand)}
            for k in range(0, len(qi), 64):
                S = score(torch.tensor(Qs[qi[k:k + 64]]), P, M, Wm).numpy()
                for r, q in enumerate(qi[k:k + 64]):
                    res.append(int((S[r] > S[r, pos[Qp[q]]]).sum()) + 1)
        r = np.array(res); print(f"HELDOUT {tag} n={len(r)} top1={np.mean(r <= 1):.1%} top5={np.mean(r <= 5):.1%} top20={np.mean(r <= 20):.1%} median={int(np.median(r))}", flush=True)
evaluate(W0t, "pca")
cats = list(by_cat)
for step in range(STEPS):
    c = cats[step % len(cats)]; qb = rng.choice(by_cat[c], BS, replace=False)
    pi = Qp[qb]; _, first = np.unique(pi, return_index=True); qb, pi = qb[first], pi[first]  # one query per product
    P, M = pad(pi); S = score(torch.tensor(Qs[qb]), P, M, W) / TAU
    loss = torch.nn.functional.cross_entropy(S, torch.arange(len(pi)))
    rb = rng.choice(two, min(BS, len(two)), replace=False)  # real: photo 2 -> photo 1 of the same product
    q2, _ = pad(rb, img=1); P1, M1 = pad(rb, img=0)
    S2 = score(q2[:, :7], P1, M1, W) / TAU
    loss = loss + torch.nn.functional.cross_entropy(S2, torch.arange(len(rb))) + LAM * ((W - W0t) ** 2).sum()
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 250 == 0: print("step", step, round(float(loss), 4), flush=True)
    if step and step % 1000 == 0: evaluate(W.detach(), f"step{step}")
evaluate(W.detach(), "trained")
np.savez("head.npz", mean=mean.astype(np.float32), W=W.detach().numpy().astype(np.float32))
print("saved head.npz")
