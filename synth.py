"""Fake shopper photos from the store's own photos: cut the jewel out of the product photo (OWLv2 box + GrabCut mask)
and put it on a wrist, a cloth or a table. Then DINOv2 embeds them with the theme's 7 crops.
  python synth.py I N   -> synth/synth_I.npz (Q: [n,7,384] fp16, handle per row) + synth/look_I.jpg (first examples)
Used by .github/workflows/head.yml to train the projection (train_head.py) so the box behind the jewel matters less."""
import os, sys, io, json, random, time, concurrent.futures as cf
import numpy as np, requests, cv2, onnxruntime as ort
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_index as B, pieces

SKIN = [(241, 194, 168), (234, 186, 156), (224, 172, 140), (210, 158, 124), (198, 134, 98), (255, 222, 204), (176, 118, 86), (232, 200, 180)]
PER_PRODUCT = int(os.environ.get("SYNTH_PER_PRODUCT", "2"))
JEWEL = ["bracelets", "necklaces", "earrings", "rings", "sets", "other"]

def background(S, rnd):
    kind = rnd.random()
    base = tuple(int(x) for x in rnd.choice([(235, 235, 235), (245, 243, 240), (220, 222, 225), (250, 250, 250), (230, 225, 215), (205, 205, 210)]))
    bg = Image.new("RGB", (S, S), base)
    g = np.linspace(rnd.uniform(.85, 1), rnd.uniform(.9, 1.1), S)[:, None, None]
    bg = Image.fromarray(np.clip(np.asarray(bg, np.float32) * g, 0, 255).astype(np.uint8))
    if kind < .6:  # a wrist / forearm across the photo
        d = ImageDraw.Draw(bg); skin = rnd.choice(SKIN); ang = np.radians(rnd.uniform(-35, 35)); wid = S * rnd.uniform(.4, .65)
        cx, cy = S / 2 + rnd.uniform(-.1, .1) * S, S / 2 + rnd.uniform(-.1, .1) * S
        dx, dy = np.cos(ang) * S, np.sin(ang) * S; nx, ny = -np.sin(ang) * wid / 2, np.cos(ang) * wid / 2
        mask = Image.new("L", (S, S), 0)
        ImageDraw.Draw(mask).polygon([(cx - dx + nx, cy - dy + ny), (cx + dx + nx, cy + dy + ny), (cx + dx - nx, cy + dy - ny), (cx - dx - nx, cy - dy - ny)], fill=255)
        if rnd.random() < .5:  # back of the hand at one end
            hx, hy = cx + dx * rnd.choice([-.45, .45]), cy + dy * .45
            ImageDraw.Draw(mask).ellipse([hx - wid * .8, hy - wid * .7, hx + wid * .8, hy + wid * .7], fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(S * .01))
        yy, xx = np.mgrid[0:S, 0:S]; t = ((xx - cx) * np.sin(ang) * -1 + (yy - cy) * np.cos(ang)) / (wid / 2)  # across the arm: shading
        shade = (1 - .18 * np.clip(np.abs(t), 0, 1) ** 2)[..., None] * rnd.uniform(.9, 1.08)
        arm = np.clip(np.array(skin, np.float32)[None, None] * shade + np.random.normal(0, 3, (S, S, 1)), 0, 255).astype(np.uint8)
        bg = Image.composite(Image.fromarray(arm), bg, mask)
    elif kind < .8:  # cloth
        c = np.array(rnd.choice([(40, 40, 45), (120, 30, 40), (200, 190, 170), (60, 80, 120), (230, 220, 200), (90, 60, 50)]), np.float32)
        n = cv2.GaussianBlur(np.random.normal(0, 18, (S, S, 1)).astype(np.float32), (0, 0), rnd.uniform(1, 4))
        bg = Image.fromarray(np.clip(c[None, None] + (n if n.ndim == 3 else n[..., None]), 0, 255).astype(np.uint8))
    return bg

def cut(im, box, rnd):
    """The jewel with a soft mask (GrabCut inside the OWLv2 box)."""
    w, h = im.size; x0, y0, x1, y1 = box[:4]; mx, my = (x1 - x0) * .04, (y1 - y0) * .04
    c = im.crop((int(max(0, x0 - mx) * w), int(max(0, y0 - my) * h), int(min(1, x1 + mx) * w), int(min(1, y1 + my) * h)))
    a = np.asarray(c)[:, :, ::-1].copy(); H, W = a.shape[:2]
    if H < 20 or W < 20: return None
    # background colours = the crop's border (k-means, 3 colours); pixels far from all of them are probably the jewel
    lab = cv2.cvtColor(a, cv2.COLOR_BGR2LAB).astype(np.float32); bw = max(2, int(min(H, W) * .04))
    border = np.concatenate([lab[:bw].reshape(-1, 3), lab[-bw:].reshape(-1, 3), lab[:, :bw].reshape(-1, 3), lab[:, -bw:].reshape(-1, 3)])
    _, _, cen = cv2.kmeans(border, 3, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 2, cv2.KMEANS_PP_CENTERS)
    dist = np.min(np.linalg.norm(lab[:, :, None, :] - cen[None, None], axis=3), axis=2)
    m = np.where(dist > 28, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8); m[dist > 55] = cv2.GC_FGD
    m[:bw] = m[-bw:] = cv2.GC_BGD; m[:, :bw] = m[:, -bw:] = cv2.GC_BGD
    try:
        if (m == cv2.GC_FGD).sum() + (m == cv2.GC_PR_FGD).sum() < 30: raise ValueError
        cv2.grabCut(a, m, None, np.zeros((1, 65)), np.zeros((1, 65)), 4, cv2.GC_INIT_WITH_MASK)
        fg = ((m == 1) | (m == 3)).astype(np.uint8) * 255
    except Exception:
        fg = np.full((H, W), 255, np.uint8)
    k = max(3, int(min(H, W) * .012) | 1)  # drop thin box edges, keep the big pieces
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    n, lab_, st, _ = cv2.connectedComponentsWithStats(fg)
    if n > 1:
        big = st[1:, cv2.CC_STAT_AREA].max(); keep = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= .03 * big]
        fg = np.isin(lab_, keep).astype(np.uint8) * 255
    f = fg.mean() / 255
    if f < .04 or f > .97: fg = np.full((H, W), 255, np.uint8)  # GrabCut failed: keep the rectangle, feathered
    alpha = Image.fromarray(fg).filter(ImageFilter.GaussianBlur(1.2))
    c = c.convert("RGBA"); c.putalpha(alpha); bb = alpha.getbbox()
    return c.crop(bb) if bb else c

def fake(im, box, rnd):
    p = cut(im, box, rnd)
    if p is None: return None
    S = rnd.choice([448, 512, 600]); bg = background(S, rnd)
    p = p.rotate(rnd.uniform(-30, 30), expand=True, resample=Image.BICUBIC)
    if rnd.random() < .5: p = p.transpose(Image.FLIP_LEFT_RIGHT)
    tw = int(S * rnd.uniform(.35, .85)); th = int(p.height * tw / max(1, p.width))
    if th > S * .9: th = int(S * .9); tw = int(p.width * th / max(1, p.height))
    p = p.resize((max(8, tw), max(8, th)), Image.BICUBIC)
    x = int((S - p.width) * rnd.uniform(.25, .75)); y = int((S - p.height) * rnd.uniform(.3, .75))
    bg.paste(p, (x, y), p)
    out = ImageEnhance.Brightness(bg).enhance(rnd.uniform(.8, 1.15))
    out = ImageEnhance.Color(out).enhance(rnd.uniform(.8, 1.15))
    if rnd.random() < .5: out = out.filter(ImageFilter.GaussianBlur(rnd.uniform(.3, 1.2)))
    buf = io.BytesIO(); out.save(buf, "JPEG", quality=rnd.randint(60, 90)); return Image.open(buf).convert("RGB")

def main(I, N):
    allp, members = B.catalog(); boxes = pieces.load_boxes(); print("boxes", len(boxes))
    todo = []
    for cat in JEWEL:
        for h in members.get(cat, []):
            us = [u for u in B.img_urls(allp[h]) if boxes.get(u) and boxes[u][4] >= .15][:PER_PRODUCT]
            todo += [(h, cat, u) for u in us]
    todo.sort(); todo = todo[I::N]; print("shard", I, "photos", len(todo))
    if not os.path.exists("model.onnx"): open("model.onnx", "wb").write(B.S.get(B.MODEL_URL, timeout=300).content)
    sess = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"]); inp = sess.get_inputs()[0].name
    def embed(ims):
        x = np.stack([(np.asarray(i.resize((B.SIZE, B.SIZE), Image.BILINEAR), np.float32) / 255 - B.MEAN) / B.STD for i in ims]).transpose(0, 3, 1, 2)
        o = sess.run(None, {inp: x})[0][:, 0, :]; return o / np.linalg.norm(o, axis=1, keepdims=True)
    S = requests.Session(); S.headers["User-Agent"] = "jewel-visual-search-indexer"
    Q, H, C, looks = [], [], [], []
    for k in range(0, len(todo), 48):
        chunk = todo[k:k + 48]
        with cf.ThreadPoolExecutor(16) as ex: ims = list(ex.map(lambda t: pieces.fetch(t[2], 768, S), chunk))
        for (h, cat, u), im in zip(chunk, ims):
            if im is None: continue
            rnd = random.Random(hash(u) & 0xffffffff)
            f = fake(im, boxes[u], rnd)
            if f is None: continue
            Q.append(embed(B.crops(f)).astype(np.float16)); H.append(h); C.append(cat)
            if len(looks) < 24: looks.append(f.resize((256, 256)))
        print("done", k + len(chunk), "of", len(todo), flush=True)
    os.makedirs("synth", exist_ok=True)
    np.savez_compressed(f"synth/synth_{I}.npz", Q=np.stack(Q), H=np.array(H), C=np.array(C))
    if looks:
        sheet = Image.new("RGB", (256 * 6, 256 * ((len(looks) + 5) // 6)), "white")
        for j, im in enumerate(looks): sheet.paste(im, ((j % 6) * 256, (j // 6) * 256))
        sheet.save(f"synth/look_{I}.jpg", quality=80)
    print("saved", len(Q))

if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
