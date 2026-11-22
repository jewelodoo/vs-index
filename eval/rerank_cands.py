"""Second-stage test: which candidates would Claude see? Theme protocol (7 crops + auto-crop 15%),
then top 12 overall + best of the brand Claude names, up to 24. Saves a contact sheet per photo."""
import io, re, json, time, os, numpy as np, requests
from PIL import Image, ImageDraw, ImageFont
exec(open("eval/photo_test.py").read().split("TESTS = [")[0])  # model, pca, titles, run/proj/crops/autobox
imgs = {}
page = 1
while True:
    ps = requests.get(f"https://sa-jewel.com/products.json?limit=250&page={page}", headers={"User-Agent": "jewel-vs-test"}).json()["products"]
    if not ps: break
    for p in ps:
        if p.get("images"): imgs[p["handle"]] = p["images"][0]["src"]
    page += 1; time.sleep(.3)
TESTS = [("1.jpg","necklaces",r"دورين 707","دورين"),("2.jpg","bracelets",r"كارتير","كارتير"),("3.jpg","bracelets",r"بولغاري.*خيط|خيط.*بولغاري","بولغاري"),
 ("4.jpg","rings",r"توري","توري"),("5.jpg","earrings",r"سي اتش|كارولينا","كارولينا|سي اتش"),("6.jpg","bracelets",r"فان كليف","فان كليف"),
 ("7.jpg","necklaces",r"سواروفسكي.*ملون","سواروفسكي"),("8.jpg","bracelets",r"فندي زركون بيق لوقو","فندي"),("9.jpg","bracelets",r"لويس فيتون.*سمبل","لويس فيتون"),
 ("10.jpg","bracelets",r"سواروفسكي","سواروفسكي"),("11.jpg","rings",r"فندي","فندي"),
 ("../dior_full.jpg","earrings",r"ديور لولو ذهبي بالشعار","ديور"),("../dior_sel.jpg","earrings",r"ديور لولو ذهبي بالشعار","ديور")]
cache = {}
def catdata(cat):
    if cat not in cache:
        meta = requests.get(RAW + f"data/{cat}.json").json(); V = np.frombuffer(requests.get(RAW + f"data/{cat}.bin").content, np.int8).reshape(-1, d).astype(np.float32) / 127
        cache[cat] = (meta["handles"], V, np.repeat(np.arange(len(meta["counts"])), meta["counts"]))
    return cache[cat]
os.makedirs("sheets", exist_ok=True)
try: font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
except Exception: font = ImageFont.load_default()
def thumb(url, s=240):
    try:
        im = Image.open(io.BytesIO(requests.get(url + ("&" if "?" in url else "?") + "width=320", timeout=20).content)).convert("RGB")
    except Exception: im = Image.new("RGB", (s, s), "white")
    im.thumbnail((s, s)); c = Image.new("RGB", (s, s), "white"); c.paste(im, ((s - im.width)//2, (s - im.height)//2)); return c
out = []
for f, cat, rx, brand in TESTS:
    im = Image.open(io.BytesIO(requests.get(RAW + "queries/t/" + f).content)).convert("RGB"); im.thumbnail((900, 900))
    o = run(crops(im, M["crops"])); base = proj(o); box = autobox(o[0], "pca", .15); w, h = im.size
    piece = im.crop((int(box[0]*w), int(box[1]*h), int(box[2]*w), int(box[3]*h))); Q = np.vstack([base, proj(run(crops(piece, M["crops"])))])
    hs, V, owner = catdata(cat); S = Q @ V.T; P = np.full((len(Q), len(hs)), -9.0); np.maximum.at(P.T, owner, S.T); order = list(np.argsort(-P.max(0)))[:400]
    vis_rank = next((k + 1 for k, i in enumerate(order) if re.search(rx, titles.get(hs[i], ""))), 999)
    cands = order[:12]; cands += [i for i in order if i not in cands and re.search(brand, titles.get(hs[i], ""))][:24 - len(cands)]
    cands += [i for i in order if i not in cands][:24 - len(cands)]
    in_c = next((k + 1 for k, i in enumerate(cands) if re.search(rx, titles.get(hs[i], ""))), 0)
    sheet = Image.new("RGB", (6 * 250, 260 + 4 * 250 + 10), "white"); q = im.copy(); q.thumbnail((500, 250)); sheet.paste(q, (10, 5)); dr = ImageDraw.Draw(sheet)
    for k, i in enumerate(cands):
        x, y = 5 + (k % 6) * 250, 265 + (k // 6) * 250; sheet.paste(thumb(imgs.get(hs[i], "")), (x, y)); dr.rectangle((x, y, x + 34, y + 28), fill="black"); dr.text((x + 4, y + 2), str(k + 1), fill="white", font=font)
    name = re.sub(r"\W", "_", f.split("/")[-1]); sheet.save(f"sheets/{name}.jpg", quality=85)
    out.append({"photo": f, "visual_rank": vis_rank, "pos_in_24": in_c, "cands": [[k + 1, titles.get(hs[i], hs[i])] for k, i in enumerate(cands)]})
    print("PHOTO", f, "visual_rank", vis_rank, "in_candidates_at", in_c)
json.dump(out, open("sheets/cands.json", "w"), ensure_ascii=False, indent=1)
