"""Background removal test (bracelets): cut the jewel out (rembg models) on a plain white background, in the store's photos
and in the shopper photo, then compare with DINOv2 as the theme does (7 crops, max similarity, own PCA per variant).
  python eval/bg_test.py shard I N  -> bg/bg_I.npz  (per store photo and variant: 7 crop vectors)
  python eval/bg_test.py eval       -> ranks of the right product per shopper photo and variant
Variants: <model>:whole = background removed from the whole photo; <model>:piece = OWLv2 box first (pieces.py), then removed."""
import os, sys, io, re, json, numpy as np, requests, concurrent.futures as cf
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import build_index as B, pieces
from rembg import new_session, remove

MODELS = os.environ.get("BG_MODELS", "isnet-general-use,u2netp").split(",")
SESS = {m: new_session(m) for m in MODELS}

def clean(im, m):
    """The jewel on white, cut to its own size (+8%) and padded to a square. Nothing found: the photo as it is."""
    a = np.asarray(remove(im, session=SESS[m], only_mask=True).convert("L"), np.float32) / 255
    if (a > .5).mean() < .003: return im
    ys, xs = np.where(a > .5); h, w = a.shape
    white = np.full((h, w, 3), 255, np.float32); rgb = np.asarray(im.convert("RGB"), np.float32)
    comp = Image.fromarray((rgb * a[..., None] + white * (1 - a[..., None])).astype(np.uint8))
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max(); mx, my = int((x1 - x0) * .08) + 2, int((y1 - y0) * .08) + 2
    comp = comp.crop((max(0, x0 - mx), max(0, y0 - my), min(w, x1 + mx), min(h, y1 + my)))
    s = max(comp.size); sq = Image.new("RGB", (s, s), "white"); sq.paste(comp, ((s - comp.width) // 2, (s - comp.height) // 2)); return sq

def variants(im, box):
    out = {}
    for m in MODELS:
        out[f"{m}:whole"] = clean(im, m)
        if box: out[f"{m}:piece"] = clean(pieces.piece(im, box), m)
    return out

def dino():
    import onnxruntime as ort
    if not os.path.exists("model.onnx"): open("model.onnx", "wb").write(B.S.get(B.MODEL_URL, timeout=300).content)
    sess = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"]); inp = sess.get_inputs()[0].name
    def embed(ims):
        x = np.stack([(np.asarray(i.convert("RGB").resize((B.SIZE, B.SIZE), Image.BILINEAR), np.float32) / 255 - B.MEAN) / B.STD for i in ims]).transpose(0, 3, 1, 2)
        o = sess.run(None, {inp: x})[0][:, 0, :]; return o / np.linalg.norm(o, axis=1, keepdims=True)
    return embed

def shard(I, N):
    allp, members = B.catalog(); boxes = pieces.load_boxes()
    todo = sorted((h, u) for h in members["bracelets"] for u in B.img_urls(allp[h]))[I::N]; print("photos", len(todo))
    embed = dino(); S = requests.Session(); S.headers["User-Agent"] = "jewel-visual-search-indexer"
    H, V, looks = [], {}, []
    for k in range(0, len(todo), 32):
        chunk = todo[k:k + 32]
        with cf.ThreadPoolExecutor(16) as ex: ims = list(ex.map(lambda t: pieces.fetch(t[1], 768, S), chunk))
        for (h, u), im in zip(chunk, ims):
            if im is None: continue
            vs = variants(im, boxes.get(u)); H.append(h)
            for name in [f"{m}:{k2}" for m in MODELS for k2 in ("whole", "piece")]:
                V.setdefault(name, []).append(embed(B.crops(vs.get(name, vs[name.split(":")[0] + ":whole"]))).astype(np.float16))
            if len(looks) < 8: looks += [im.resize((200, 200))] + [vs[n].resize((200, 200)) for n in sorted(vs)]
        print("done", k + len(chunk), flush=True)
    os.makedirs("bg", exist_ok=True)
    np.savez_compressed(f"bg/bg_{I}.npz", H=np.array(H), **{n.replace(":", "__"): np.stack(v) for n, v in V.items()})
    if looks:
        cols = 1 + 2 * len(MODELS); sheet = Image.new("RGB", (200 * cols, 200 * ((len(looks) + cols - 1) // cols)), "white")
        for j, im in enumerate(looks): sheet.paste(im, ((j % cols) * 200, (j // cols) * 200))
        sheet.save(f"bg/look_{I}.jpg", quality=80)

def evaluate():
    import glob
    exec(open("eval/photo_test.py").read().split("cache = {}")[0].replace("TESTS = [", "TESTS0 = ["), g := {})
    tests = [t for t in g["TESTS0"] if t[1] == "bracelets"] + [("bv.jpg", "bracelets", "handle:408879")]
    def hit(h, rx):
        return h in rx[7:].split("|") if rx.startswith("handle:") else bool(re.search(rx, g["titles"].get(h, "")))
    Z = [np.load(f) for f in sorted(glob.glob("bg/bg_*.npz"))]
    H = np.concatenate([z["H"] for z in Z]); names = [k for k in Z[0].files if k != "H"]
    hs = sorted(set(H)); hi = {h: i for i, h in enumerate(hs)}; owner = np.array([hi[h] for h in H])
    embed = dino(); RAWQ = g["RAW"] + "queries/t/"
    Q = {}
    for f, c, rx in tests:
        im = Image.open(io.BytesIO(requests.get(RAWQ + f).content)).convert("RGB"); im.thumbnail((900, 900))
        Q[f] = (im, variants(im, None), rx)
    res = {}
    for n in names:
        X = np.concatenate([z[n] for z in Z]).astype(np.float32)  # (photos, 7, 384)
        flat = X.reshape(-1, 384); samp = flat[np.random.default_rng(0).choice(len(flat), min(len(flat), 40000), replace=False)]
        mean = samp.mean(0); W = np.linalg.svd(samp - mean, full_matrices=False)[2][:B.DIM].T
        P = (flat - mean) @ W; P /= np.linalg.norm(P, axis=1, keepdims=True); pown = np.repeat(owner, 7)
        m = n.split("__")[0]
        for f, (im, qv, rx) in Q.items():
            q = embed(B.crops(qv[f"{m}:whole"])); q = (q - mean) @ W; q /= np.linalg.norm(q, axis=1, keepdims=True)
            s = (q @ P.T).max(0); best = np.full(len(hs), -9.0); np.maximum.at(best, pown, s); order = np.argsort(-best)
            res.setdefault(n, {})[f] = next((k + 1 for k, i in enumerate(order) if hit(hs[i], rx)), 999)
        print("VARIANT", n, res[n], flush=True)
    for n, r in res.items():
        v = np.array(list(r.values())); print(f"SUMMARY {n:28s} top1={int((v <= 1).sum())}/{len(v)} top5={int((v <= 5).sum())} top20={int((v <= 20).sum())} ranks={list(r.values())}")
    os.makedirs("bg", exist_ok=True)
    for f, (im, qv, rx) in Q.items():
        row = [im.resize((200, 200))] + [qv[k].resize((200, 200)) for k in sorted(qv)]
        sheet = Image.new("RGB", (200 * len(row), 200), "white")
        for j, x in enumerate(row): sheet.paste(x, (j * 200, 0))
        sheet.save(f"bg/q_{f.replace('/', '_')}", quality=80)

if __name__ == "__main__":
    if sys.argv[1] == "shard": shard(int(sys.argv[2]), int(sys.argv[3]))
    else: evaluate()
