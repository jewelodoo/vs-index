"""Product-to-product neighbours for the complementary-products app.
For every product: the K most similar products in each OTHER jewellery category.
Also evaluates against hand-made complementary groups (eval/groups.json): recall@K."""
import json, re, collections, numpy as np

# Brand from the product title (Arabic spellings), else from the vendor. Every piece of a set comes from one brand.
BRANDS = {
 "hermes": ["هيرمز", "هرمز", "هيرميس", "hermes"], "bulgari": ["بولغاري", "بلغاري", "بولقاري", "بولجاري", "bvlgari", "bulgari"],
 "vca": ["فان كليف", "فانكليف", "van cleef", "vca"], "cartier": ["كارتير", "كارتيير", "كارتيه", "cartier"],
 "tiffany": ["تيفاني", "تيفني", "tiffany"], "chaumet": ["شوميه", "شومية", "شومي", "chaumet"], "messika": ["ميسيكا", "مسيكا", "messika"],
 "lv": ["لويس فيتون", "لوي فيتون", "louis vuitton", "lv"], "dior": ["ديور", "dior"], "chanel": ["شانيل", "شانل", "chanel"],
 "boucheron": ["بوشرون", "boucheron"], "swarovski": ["سواروفسكي", "سواروفيسكي", "swarovski"], "fendi": ["فندي", "fendi"],
 "gucci": ["قوتشي", "غوتشي", "قوشي", "gucci"], "ch": ["سي اتش", "كارولينا", "carolina"], "ysl": ["سان لوران", "ysl"],
 "valentino": ["فالنتينو", "valentino"], "tory": ["توري", "tory"], "mk": ["مايكل كورس", "michael kors"], "coach": ["كوتش", "coach"],
 "prada": ["برادا", "prada"], "givenchy": ["جيفنشي", "givenchy"], "balenciaga": ["بالنسياقا", "balenciaga"], "dolce": ["دولتشي", "dolce"],
 "pandora": ["باندورا", "pandora"], "piaget": ["بياجيه", "piaget"], "chopard": ["شوبارد", "chopard"], "graff": ["قراف", "graff"],
 "nomination": ["نوميناشن", "نومينيشن", "nomination"], "apm": ["apm", "اي بي ام"], "roberto": ["روبرتو كوين", "roberto coin"],
 "montblanc": ["مونت بلانك", "مون بلان", "mont blank", "montblanc"], "aigner": ["اقنر", "ايغنر", "aigner"], "rolex": ["رولكس", "rolex"], "armani": ["ارماني", "armani"],
 "marc": ["مارك جيكوبس", "marc jacobs"], "celine": ["سيلين", "celine"], "miumiu": ["ميو ميو", "miu miu"], "versace": ["فيرزاتشي", "فرزاتشي", "versace"],
}
GENERIC = {"", "اكسسوارات جويل", "new", "hold", "jewel", "شنط", "اكسسوارات"}
def norm(t): return re.sub(r"[\sـ-]+", " ", str(t or "").lower().replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ة", "ه").replace("ى", "ي")).strip()
_BR = [(b, norm(w)) for b, ws in BRANDS.items() for w in ws]
def brand_of(title, vendor=""):
    t = " " + norm(title) + " "
    for b, w in _BR:
        if (" " + w + " ") in t if len(w) <= 3 else w in t: return b
    v = norm(vendor)
    for b, w in _BR:
        if w == v or (len(w) > 3 and w in v): return b
    return None if v in GENERIC or re.search(r"متوفر|لون|الوان|جويل", v) else "v:" + v

NB_CATS = ["rings", "earrings", "bracelets", "necklaces", "sets"]
K = 20

def build(prod, allp, mean, W, out="out"):
    """prod: list of (id, handle, cat, vecs[n,D] float32 unit); allp: handle -> product (title, vendor) ; returns neighbours dict"""
    P = [p for p in prod if p[2] in NB_CATS]
    allp = allp or {}
    brand = np.array([brand_of(allp.get(p[1], {}).get("title", ""), allp.get(p[1], {}).get("vendor", "")) or "" for p in P], dtype=object)
    print("brands:", collections.Counter(brand).most_common(40))
    print("vendors without a known brand:", collections.Counter(allp.get(p[1], {}).get("vendor", "") for p, b in zip(P, brand) if not b or b.startswith("v:")).most_common(25))
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
    # D: same brand first (when the product's brand is known), then by look (maxcrop)
    def brandcrop(qi):
        S = maxcrop(qi)
        for r, q in enumerate(qi):
            if brand[q]: S[r] = S[r] + 10 * (brand == brand[q])
        return S
    variants["brand"] = brandcrop

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
                g = {x for gg in groups.values() if int(ids[q]) in gg for x in gg}  # a piece can be in two sets
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
            nbr[str(int(ids[q]))] = {c: [[int(ids[j]), round(float(s) % 10 if s >= 9 else float(s), 3), int(bool(brand[q]) and brand[j] == brand[q])] for j, s in zip(*nb[c])] for c in nb}
    meta = {"k": K, "variant": best, "recall": res, "categories": NB_CATS,
            "products": {str(int(p[0])): {"handle": p[1], "cat": p[2], "brand": brand[i] or None} for i, p in enumerate(P)},
            "row": "[product id, look score, same brand 1/0]"}
    json.dump({"meta": meta, "neighbors": nbr}, open(f"{out}/neighbors.json", "w"), separators=(",", ":"), ensure_ascii=False)
    json.dump(res, open(f"{out}/eval.json", "w"))
    return res
