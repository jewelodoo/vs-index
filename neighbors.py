"""Product-to-product neighbours for the complementary-products app.
For every product: the K most similar products in each OTHER jewellery category.
Also evaluates against hand-made complementary groups (eval/groups.json): recall@K."""
import json, numpy as np

NB_CATS = ["rings", "earrings", "bracelets", "necklaces", "sets"]
K = 20

def build(prod, cat_of, mean, W, out="out"):
    """prod: list of (id, handle, cat, vecs[n,D] float32 unit) ; returns neighbours dict"""
    P = [p for p in prod if p[2] in NB_CATS]
    V = [((p[3] - mean) @ W) for p in P]
    V = [v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9) for v in V]
    full = np.stack([v[0] for v in V]).astype(np.float32)          # main image, whole frame
    allv = np.concatenate(V).astype(np.float32)                     # every crop of every image
    owner = np.concatenate([np.full(len(v), i) for i, v in enumerate(V)])
    starts = np.r_[0, np.cumsum([len(v) for v in V])[:-1]]
    cats = np.array([p[2] for p in P])
    n = len(P)
    variants = {}
    # A: whole main image vs whole main image
    variants["full"] = lambda qi: full[qi] @ full.T
    # B: max over all crop pairs (query crops x target crops)
    def maxcrop(qi):
        S = np.concatenate([V[i] for i in qi]) @ allv.T                 # (sum q crops, all crops)
        S = np.maximum.reduceat(S, starts, axis=1)                      # per target product
        qo = np.r_[0, np.cumsum([len(V[i]) for i in qi])[:-1]]
        return np.maximum.reduceat(S, qo, axis=0)                       # per query product
    variants["maxcrop"] = maxcrop
    # C: average of A and B
    variants["mix"] = lambda qi: 0.5 * variants["full"](qi) + 0.5 * maxcrop(qi)

    ids = np.array([p[0] for p in P])
    pos = {int(x): i for i, x in enumerate(ids)}
    groups = json.load(open("eval/groups.json"))
    res = {}
    def topk(scores, qi):
        nb = {}
        for c in NB_CATS:
            if c == cats[qi]: continue
            m = np.where(cats == c)[0]
            if not len(m): continue
            s = scores[m]; o = np.argsort(-s)[:K]
            nb[c] = (m[o], s[o])
        return nb
    evalq = sorted({pos[x] for g in groups.values() for x in g if x in pos})
    for name, fn in variants.items():
        hit = tot = 0
        for b in range(0, len(evalq), 64):
            qi = evalq[b:b + 64]; S = fn(qi)
            for r, q in enumerate(qi):
                g = next(gg for gg in groups.values() if int(ids[q]) in gg)
                nb = topk(S[r], q); found = {int(ids[j]) for c in nb for j in nb[c][0]}
                gold = [x for x in g if x in pos and x != int(ids[q]) and cats[pos[x]] != cats[q]]
                hit += sum(x in found for x in gold); tot += len(gold)
        res[name] = round(hit / max(tot, 1), 3)
    best = max(res, key=res.get)
    print("recall@%d (other categories) per variant:" % K, res, "missing from index:", sum(x not in pos for g in groups.values() for x in g), "-> using", best)
    fn = variants[best]
    nbr = {}
    for b in range(0, n, 48):
        qi = list(range(b, min(n, b + 48))); S = fn(qi)
        for r, q in enumerate(qi):
            nb = topk(S[r], q)
            nbr[str(int(ids[q]))] = {c: [[int(ids[j]), round(float(s), 3)] for j, s in zip(*nb[c])] for c in nb}
    meta = {"k": K, "variant": best, "recall": res, "categories": NB_CATS,
            "products": {str(int(p[0])): {"handle": p[1], "cat": p[2]} for p in P}}
    json.dump({"meta": meta, "neighbors": nbr}, open(f"{out}/neighbors.json", "w"), separators=(",", ":"), ensure_ascii=False)
    json.dump(res, open(f"{out}/eval.json", "w"))
    return res
