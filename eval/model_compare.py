"""Compare image models for shopper photos (different background/worn) vs catalogue photos.
Same protocol for every model: index = whole image + auto-cropped piece (box from DINOv2-small tokens),
query = whole photo + auto-cropped piece; score = best pair. Usage: python eval/model_compare.py <category>"""
import io, re, sys, time, numpy as np, requests, onnxruntime as ort, concurrent.futures as cf
from PIL import Image
CAT = sys.argv[1]
RAW = "https://raw.githubusercontent.com/jewelodoo/vs-index/"
HF = "https://huggingface.co/Xenova/"
MODELS = {
  "dinov2_small": (HF + "dinov2-small/resolve/main/onnx/model_quantized.onnx", "cls", [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
  "dinov2_base":  (HF + "dinov2-base/resolve/main/onnx/model_quantized.onnx", "cls", [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
  "clip_b32":     (HF + "clip-vit-base-patch32/resolve/main/onnx/vision_model_quantized.onnx", "image_embeds", [0.4815, 0.4578, 0.4082], [0.2686, 0.2613, 0.2758]),
  "clip_b16":     (HF + "clip-vit-base-patch16/resolve/main/onnx/vision_model_quantized.onnx", "image_embeds", [0.4815, 0.4578, 0.4082], [0.2686, 0.2613, 0.2758]),
}
sess = {}
for name, (url, out, m, s) in MODELS.items():
    try:
        r = requests.get(url, timeout=600); r.raise_for_status(); open(name + ".onnx", "wb").write(r.content)
        se = ort.InferenceSession(name + ".onnx", providers=["CPUExecutionProvider"])
        sess[name] = (se, out, np.array(m, np.float32), np.array(s, np.float32)); print("MODEL", name, round(len(r.content) / 1e6, 1), "MB", flush=True)
    except Exception as e: print("MODEL", name, "unavailable", e, flush=True)
def run(name, ims):
    se, out, M, S = sess[name]
    x = np.stack([(np.asarray(i.convert("RGB").resize((224, 224), Image.BICUBIC), np.float32) / 255 - M) / S for i in ims]).transpose(0, 3, 1, 2)
    outs = {o.name: v for o, v in zip(se.get_outputs(), se.run(None, {se.get_inputs()[0].name: x}))}
    if out == "cls": v = outs[se.get_outputs()[0].name][:, 0, :]; tok = outs[se.get_outputs()[0].name]
    else: v = outs.get("image_embeds", list(outs.values())[0]); tok = None
    v = v / np.linalg.norm(v, axis=1, keepdims=True); return v, tok
def autobox(tok, keep=.15, margin=.12):
    P = tok[1:]; g = int(len(P) ** .5); Pn = P / np.linalg.norm(P, axis=1, keepdims=True)
    c = Pn - Pn.mean(0); s = c @ np.linalg.svd(c, full_matrices=False)[2][0]; G = s.reshape(g, g)
    if G[g//4:3*g//4, g//4:3*g//4].mean() < np.r_[G[0], G[-1], G[:, 0], G[:, -1]].mean(): s = -s
    m = (s >= np.sort(s)[::-1][int(len(s) * keep)]).reshape(g, g); ys, xs = np.where(m)
    x0, x1, y0, y1 = xs.min()/g, (xs.max()+1)/g, ys.min()/g, (ys.max()+1)/g; mx, my = (x1-x0)*margin, (y1-y0)*margin
    return max(0, x0-mx), max(0, y0-my), min(1, x1+mx), min(1, y1+my)
def views(im):
    v, tok = run("dinov2_small", [im]); b = autobox(tok[0]); w, h = im.size
    piece = im.crop((int(b[0]*w), int(b[1]*h), int(b[2]*w), int(b[3]*h)))
    return [im, piece]
S_ = requests.Session(); S_.headers["User-Agent"] = "jewel-vs-test"
prods = {}; page = 1
while True:
    ps = S_.get(f"https://sa-jewel.com/products.json?limit=250&page={page}").json()["products"]
    if not ps: break
    for p in ps: prods[p["handle"]] = p
    page += 1; time.sleep(.3)
TITLE = {h: p["title"] for h, p in prods.items()}
hs = requests.get(RAW + f"data/{CAT}.json").json()["handles"]
urls = [(k, i["src"]) for k, h in enumerate(hs) for i in prods.get(h, {}).get("images", [])[:2]]
def fetch(a):
    k, u = a
    for t in range(3):
        try:
            r = S_.get(u + ("&" if "?" in u else "?") + "width=448", timeout=60)
            if r.ok: return k, Image.open(io.BytesIO(r.content)).convert("RGB")
        except Exception: time.sleep(1)
    return k, None
IDX = {n: [] for n in sess}; OWN = []
t0 = time.time()
with cf.ThreadPoolExecutor(16) as ex:
    for k, im in ex.map(fetch, urls):
        if im is None: continue
        vw = views(im)
        for n in sess: IDX[n].append(run(n, vw)[0])
        OWN += [k, k]
print("INDEX", CAT, len(hs), "products", len(OWN) // 2, "images", round(time.time() - t0), "s", flush=True)
for n in sess: IDX[n] = np.concatenate(IDX[n])
OWN = np.array(OWN)
TESTS = {"bracelets": [("t/2.jpg", r"كارتير"), ("t/3.jpg", r"بولغاري.*خيط|خيط.*بولغاري"), ("t/6.jpg", r"فان كليف"), ("t/8.jpg", r"فندي زركون بيق لوقو"),
                       ("t/9.jpg", r"لويس فيتون.*سمبل"), ("t/10.jpg", r"^اسواره سواروفسكي اريانا غراندي بانقل ماستر$")],
         "necklaces": [("t/1.jpg", r"دورين 707"), ("t/7.jpg", r"سواروفسكي.*ملون")],
         "earrings": [("t/5.jpg", r"سي اتش|كارولينا"), ("dior_full.jpg", r"ديور لولو ذهبي بالشعار"), ("dior_sel.jpg", r"ديور لولو ذهبي بالشعار")],
         "rings": [("t/4.jpg", r"توري"), ("t/11.jpg", r"فندي")]}[CAT]
for f, rx in TESTS:
    im = Image.open(io.BytesIO(requests.get(RAW + "queries/" + f).content)).convert("RGB"); im.thumbnail((900, 900)); vw = views(im)
    res = []
    for n in sess:
        Q = run(n, vw)[0]; Sm = Q @ IDX[n].T; P = np.full((len(Q), len(hs)), -9.0); np.maximum.at(P.T, OWN, Sm.T); a = P.max(0)
        order = np.argsort(-a); hits = [j for j, i in enumerate(order) if re.search(rx, TITLE.get(hs[i], ""))]
        res.append(f"{n}={hits[0] + 1 if hits else 999}")
    print("CMP", CAT, f, " ".join(res), flush=True)
