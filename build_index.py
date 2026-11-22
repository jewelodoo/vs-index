"""Builds the Jewel visual-search index from the store's public product feed.
Output (in ./out): manifest.json, pca.bin, <cat>.json, <cat>.bin
No store credentials are used: only the public /products.json feeds."""
import os, sys, json, time, pickle, hashlib, io, concurrent.futures as cf
import numpy as np, requests
from PIL import Image
import onnxruntime as ort
import neighbors, pieces

STORE = os.environ.get("STORE", "https://sa-jewel.com")
MODEL_URL = "https://huggingface.co/Xenova/dinov2-small/resolve/main/onnx/model_quantized.onnx"  # DINOv2-small, int8
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
SIZE = 224
DIM = 96            # PCA output size
IMGS = 2            # images per product
CACHE = "cache/emb_dino.pkl"         # whole photo (PIECES=0)
CACHE_P = "cache/emb_piece.pkl"      # the piece only, key "url|box"
PIECES = os.environ.get("PIECES", "1") == "1"   # index the jewel, not the photo (OWLv2 boxes, see pieces.py)
DETECT_MAX = int(os.environ.get("DETECT_MAX", "400"))  # photos to detect inside the daily build; more waits for boxes.yml
CATS = [  # key, Arabic label, collection handle (None = everything else)
    ("bracelets", "أساور", "accessories-bracelet"),
    ("necklaces", "سلاسل", "accessories-necklace"),
    ("earrings", "حلق", "accessories-earrings"),
    ("rings", "خواتم", "accessories-rings"),
    ("sets", "أطقم", "jewelry-set"),
    ("bags", "شنط", "bags"),
    ("other", "أخرى", None),
]
CROPS = [(1.0, .5, .5), (0.6, .5, .5), (0.6, .5, .8), (0.45, .5, .6), (0.45, .5, .85), (0.45, .2, .75), (0.45, .8, .75)]
S = requests.Session(); S.headers["User-Agent"] = "jewel-visual-search-indexer"

def feed(path):
    out, page = [], 1
    while True:
        for t in range(5):
            r = S.get(f"{STORE}{path}?limit=250&page={page}", timeout=60)
            if r.status_code == 200: break
            time.sleep(5 * (t + 1))
        r.raise_for_status()
        ps = r.json().get("products", [])
        if not ps: return out
        out += ps; page += 1; time.sleep(0.5)

def crops(im):
    w, h = im.size; res = []
    for s, fx, fy in CROPS:
        cw, ch = int(w * s), int(h * s); x = int((w - cw) * fx); y = int((h - ch) * fy)
        res.append(im.crop((x, y, x + cw, y + ch)))
    return res

def img_urls(p):
    return [i["src"] for i in p.get("images", [])[:IMGS]]

def catalog():
    allp = {p["handle"]: p for p in feed("/products.json")}
    print("products", len(allp))
    members, used = {}, set()
    for key, _, coll in CATS:
        if coll:
            hs = [p["handle"] for p in feed(f"/collections/{coll}/products.json") if p["handle"] in allp]
            used.update(hs)
        else:
            hs = [h for h in allp if h not in used]
        members[key] = hs
        print(key, len(hs))
    return allp, members

def main():
    global PIECES
    os.makedirs("out", exist_ok=True); os.makedirs("cache", exist_ok=True)
    if not os.path.exists("model.onnx"):
        open("model.onnx", "wb").write(S.get(MODEL_URL, timeout=300).content)
    sess = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    def embed(ims):
        x = np.stack([(np.asarray(i.resize((SIZE, SIZE), Image.BILINEAR), np.float32) / 255 - MEAN) / STD for i in ims]).transpose(0, 3, 1, 2)
        o = sess.run(None, {inp: x})[0][:, 0, :]
        return o / np.linalg.norm(o, axis=1, keepdims=True)
    allp, members = catalog()
    boxes = {}
    if PIECES:
        boxes = pieces.load_boxes()
        if not boxes: print("no boxes yet (run boxes.yml): whole photos"); PIECES = False
    if PIECES:
        missing = sorted({(u, cat) for cat, hs in members.items() for h in hs for u in img_urls(allp[h]) if u not in boxes})
        print("boxes", len(boxes), "photos without a box", len(missing))
        if missing and len(missing) <= DETECT_MAX:
            try:
                boxes.update(pieces.detect(missing)); os.makedirs("out_boxes", exist_ok=True)
                json.dump(boxes, open("out_boxes/boxes.json", "w"), separators=(",", ":"))
            except Exception as e: print("detection skipped:", e)
    def key(u):  # cache key: the photo, or the photo + where the piece is
        b = boxes.get(u)
        return f"{u}|{','.join(map(str, b[:4]))}" if PIECES and b else u
    cpath = CACHE_P if PIECES else CACHE
    cache = pickle.load(open(cpath, "rb")) if os.path.exists(cpath) else {}
    want = {u for p in allp.values() for u in img_urls(p)}
    todo = sorted(u for u in want if key(u) not in cache)
    print("new images", len(todo))
    def fetch(u):
        sep = "&" if "?" in u else "?"
        for t in range(3):
            try:
                r = S.get(f"{u}{sep}width={768 if PIECES else 448}", timeout=60)  # a piece cut from the photo needs the bigger photo
                if r.ok: return u, Image.open(io.BytesIO(r.content)).convert("RGB")
            except Exception: pass
            time.sleep(2)
        return u, None
    done = 0
    with cf.ThreadPoolExecutor(12) as ex:
        for u, im in ex.map(fetch, todo):
            if im is None: continue
            cache[key(u)] = embed(crops(pieces.piece(im, boxes.get(u)) if PIECES else im)).astype(np.float16); done += 1
            if done % 500 == 0:
                print("embedded", done); pickle.dump(cache, open(cpath, "wb"))
    live = {key(u) for u in want}
    cache = {k: v for k, v in cache.items() if k in live}
    pickle.dump(cache, open(cpath, "wb"))
    emb = lambda u: cache.get(key(u))

    X = np.concatenate([v for v in cache.values()]).astype(np.float32)
    rng = np.random.default_rng(0)
    sample = X[rng.choice(len(X), min(len(X), 40000), replace=False)]
    mean = sample.mean(0)
    _, _, Vt = np.linalg.svd(sample - mean, full_matrices=False)
    W = Vt[:DIM].T.astype(np.float32)
    with open("out/pca.bin", "wb") as f:
        f.write(mean.astype(np.float32).tobytes()); f.write(W.tobytes())
    version = hashlib.sha1(str(time.time()).encode()).hexdigest()[:10]
    cats = []
    for key, label, _ in CATS:
        hs, counts, vecs = [], [], []
        for h in members[key]:
            vv = [emb(u) for u in img_urls(allp[h]) if emb(u) is not None]
            if not vv: continue
            v = (np.concatenate(vv).astype(np.float32) - mean) @ W
            v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
            hs.append(h); counts.append(len(v)); vecs.append(v)
        if not hs: continue
        V = np.concatenate(vecs)
        q = np.clip(np.round(V * 127), -127, 127).astype(np.int8)
        q.tofile(f"out/{key}.bin")
        json.dump({"handles": hs, "ids": [allp[h]["id"] for h in hs], "counts": counts}, open(f"out/{key}.json", "w"), separators=(",", ":"))
        cats.append({"key": key, "label": label, "products": len(hs), "vectors": int(len(V))})
    json.dump({"version": version, "dim": DIM, "arch": "dinov2-small", "size": SIZE, "model": MODEL_URL, "crops": CROPS, "pieces": PIECES, "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "cats": cats},
              open("out/manifest.json", "w"), ensure_ascii=False, indent=1)
    print("done", cats)
    prod = []
    for key, _, _ in CATS:
        for h in members[key]:
            vv = [emb(u) for u in img_urls(allp[h]) if emb(u) is not None]
            if vv: prod.append((allp[h]["id"], h, key, np.concatenate(vv).astype(np.float32)))
    neighbors.build(prod, None, mean, W)

if __name__ == "__main__":
    main()
