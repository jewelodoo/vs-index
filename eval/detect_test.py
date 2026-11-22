"""Index the piece, not the box: OWLv2 finds the jewel in each product photo, DINOv2 embeds 7 crops of that piece.
  python eval/detect_test.py shard I N   -> det_I.npz  (bracelet products, 2 photos each)
  python eval/detect_test.py eval        -> ranks of the hard shopper photos: current index vs piece index vs both"""
import io, os, re, sys, glob, json, time, numpy as np, requests, onnxruntime as ort
from PIL import Image, ImageDraw
sys.path.insert(0, "."); import build_index as B
RAW = "https://raw.githubusercontent.com/jewelodoo/vs-index/"
S = requests.Session(); S.headers["User-Agent"] = "jewel-vs-test"
if not os.path.exists("model.onnx"): open("model.onnx", "wb").write(S.get(B.MODEL_URL, timeout=300).content)
sess = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"]); inp = sess.get_inputs()[0].name
def tok(ims):
    x = np.stack([(np.asarray(i.convert("RGB").resize((224, 224), Image.BILINEAR), np.float32) / 255 - B.MEAN) / B.STD for i in ims]).transpose(0, 3, 1, 2)
    return sess.run(None, {inp: x})[0]
def cls(o):
    o = o[:, 0, :]; return o / np.linalg.norm(o, axis=1, keepdims=True)
def get(u, w=768):
    for t in range(3):
        try:
            r = S.get(u + ("&" if "?" in u else "?") + f"width={w}", timeout=60)
            if r.ok: return Image.open(io.BytesIO(r.content)).convert("RGB")
        except Exception: pass
        time.sleep(2)
QUERIES = ["a bracelet", "a chain bracelet", "a bangle", "a necklace", "an earring", "a ring", "a piece of jewelry"]
def detector():
    import torch
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble"); model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").eval()
    torch.set_num_threads(os.cpu_count())
    def find(im):
        with torch.no_grad():
            enc = proc(text=[QUERIES], images=im, return_tensors="pt"); out = model(**enc)
        side = max(im.size)  # OWLv2 pads to a square
        res = proc.post_process_object_detection(out, threshold=0.05, target_sizes=torch.tensor([[side, side]]))[0]
        if not len(res["scores"]): return None, 0.0
        k = int(res["scores"].argmax()); x0, y0, x1, y1 = res["boxes"][k].tolist(); w, h = im.size
        return (max(0, x0) / w, max(0, y0) / h, min(w, x1) / w, min(h, y1) / h), float(res["scores"][k])
    return find
def piece(im, box, margin=.12):
    w, h = im.size; x0, y0, x1, y1 = box; mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    return im.crop((int(max(0, x0 - mx) * w), int(max(0, y0 - my) * h), int(min(1, x1 + mx) * w), int(min(1, y1 + my) * h)))

if sys.argv[1] == "shard":
    I, N = int(sys.argv[2]), int(sys.argv[3])
    allp = {p["handle"]: p for p in B.feed("/collections/accessories-bracelet/products.json")}
    urls = sorted({i["src"] for p in allp.values() for i in p.get("images", [])[:B.IMGS]})[I::N]
    find = detector(); U, BX, SC, E = [], [], [], []; t0 = time.time()
    for k, u in enumerate(urls):
        im = get(u)
        if im is None: continue
        box, sc = find(im)
        pc = piece(im, box) if box else im
        U.append(u); BX.append(box or (0, 0, 1, 1)); SC.append(sc); E.append(cls(tok(B.crops(pc))).astype(np.float16))
        if k % 50 == 0: print(k, len(urls), round(time.time() - t0), "s", flush=True)
    np.savez(f"det_{I}.npz", urls=np.array(U), boxes=np.array(BX), scores=np.array(SC), emb=np.stack(E))
    sys.exit()

# ---- eval
U, BX, SC, E = [], [], [], []
for f in glob.glob("det/**/det_*.npz", recursive=True):
    z = np.load(f); U += list(z["urls"]); BX += list(z["boxes"]); SC += list(z["scores"]); E += list(z["emb"])
pos = {u: k for k, u in enumerate(U)}
print("images", len(U), "found", sum(s > 0 for s in SC), "median score", np.median(SC))
M = S.get(RAW + "data/manifest.json").json(); d = M["dim"]
pb = np.frombuffer(S.get(RAW + "data/pca.bin").content, np.float32); D = len(pb) // (d + 1); mean, W = pb[:D], pb[D:].reshape(D, d)
def proj(c):
    p = (c.astype(np.float32) - mean) @ W; return p / np.linalg.norm(p, axis=1, keepdims=True)
meta = S.get(RAW + "data/bracelets.json").json(); hs = meta["handles"]
Vcur = np.frombuffer(S.get(RAW + "data/bracelets.bin").content, np.int8).reshape(-1, d).astype(np.float32) / 127
own_cur = np.repeat(np.arange(len(hs)), meta["counts"])
allp = {p["handle"]: p for p in B.feed("/products.json")}
titles = {h: p["title"] for h, p in allp.items()}
pv, po = [], []
for j, h in enumerate(hs):
    for i in allp.get(h, {}).get("images", [])[:B.IMGS]:
        if i["src"] in pos: pv.append(proj(E[pos[i["src"]]])); po += [j] * 7
Vp, own_p = np.concatenate(pv), np.array(po)
exec(open("eval/photo_test.py").read().split("def autobox")[1].split("TESTS = [")[0].join(["def autobox", ""]) if False else "")
def autobox(t, keep=.15, margin=.12):
    P = t[1:]; g = int(len(P) ** .5); Pn = P / np.linalg.norm(P, axis=1, keepdims=True); c = Pn - Pn.mean(0)
    s = c @ np.linalg.svd(c, full_matrices=False)[2][0]; G = s.reshape(g, g)
    if G[g//4:3*g//4, g//4:3*g//4].mean() < np.r_[G[0], G[-1], G[:, 0], G[:, -1]].mean(): s = -s
    m = (s >= np.sort(s)[::-1][int(len(s) * keep)]).reshape(g, g); ys, xs = np.where(m)
    return xs.min()/g, ys.min()/g, (xs.max()+1)/g, (ys.max()+1)/g
TESTS = [("2.jpg", r"كارتير"), ("3.jpg", r"بولغاري.*خيط|خيط.*بولغاري"), ("6.jpg", r"فان كليف"), ("8.jpg", r"فندي زركون بيق لوقو"),
         ("9.jpg", r"لويس فيتون.*سمبل"), ("10.jpg", r"اريانا غراندي بانقل"), ("bv.jpg", r"بولغاري نيو ستايل زركون")]
find = detector()
def rank(Q, V, own, rx):
    Sm = Q @ V.T; P = np.full((len(Q), len(hs)), -9.0); np.maximum.at(P.T, own, Sm.T); order = np.argsort(-P.max(0))
    return next((k + 1 for k, i in enumerate(order) if re.search(rx, titles.get(hs[i], ""))), 999)
tot = {}
for f, rx in TESTS:
    im = Image.open(io.BytesIO(S.get(RAW + "queries/t/" + f).content)).convert("RGB"); im.thumbnail((900, 900))
    o = tok(B.crops(im)); base = proj(cls(o))
    qa = proj(cls(tok(B.crops(piece(im, autobox(o[0]))))))
    qb, qs = find(im); qd = proj(cls(tok(B.crops(piece(im, qb))))) if qb else qa
    Qs = {"theme(base+auto)": np.vstack([base, qa]), "owl_query": qd, "owl_query+base": np.vstack([base, qd])}
    line = []
    for qn, Q in Qs.items():
        for vn, (V, own) in {"cur": (Vcur, own_cur), "piece": (Vp, own_p), "both": (np.vstack([Vcur, Vp]), np.r_[own_cur, own_p])}.items():
            r = rank(Q, V, own, rx); tot.setdefault(f"{qn} x {vn}", []).append(r); line.append(f"{qn}|{vn}={r}")
    print("PHOTO", f, "qbox", qb and [round(v, 2) for v in qb], round(qs, 2), " ".join(line), flush=True)
for k, r in tot.items():
    r = np.array(r); print(f"SUMMARY {k:32s} top1={np.sum(r<=1)}/{len(r)} top3={np.sum(r<=3)} ranks={list(r)}")
# a contact sheet of detections to eyeball
os.makedirs("sheets", exist_ok=True); rng = np.random.default_rng(1); pick = rng.choice(len(U), 24, replace=False)
sheet = Image.new("RGB", (6 * 200, 4 * 200), "white")
for k, j in enumerate(pick):
    im = get(U[j], 400); 
    if im is None: continue
    im.thumbnail((200, 200)); dr = ImageDraw.Draw(im); w, h = im.size; x0, y0, x1, y1 = BX[j]
    dr.rectangle((x0*w, y0*h, x1*w, y1*h), outline="red", width=3); sheet.paste(im, ((k % 6) * 200, (k // 6) * 200))
sheet.save("sheets/detections.jpg", quality=85)
