"""Rank of one product for one shopper photo, theme protocol (7 crops + auto-crop 15%). Also lists the product's own index entries."""
import io, os, json, numpy as np, requests
from PIL import Image
exec(open("eval/photo_test.py").read().split("TESTS = [")[0])
F, CAT, H = os.environ.get("PHOTO", "bv.jpg"), os.environ.get("CAT", "bracelets"), os.environ.get("HANDLE", "408879")
meta = _get(RAW + f"data/{CAT}.json").json(); hs = meta["handles"]
V = np.frombuffer(_get(RAW + f"data/{CAT}.bin").content, np.int8).reshape(-1, d).astype(np.float32) / 127
owner = np.repeat(np.arange(len(meta["counts"])), meta["counts"])
print("IN_INDEX", H in hs, "vectors", meta["counts"][hs.index(H)] if H in hs else 0)
im = Image.open(io.BytesIO(requests.get(RAW + "queries/t/" + F).content)).convert("RGB"); im.thumbnail((900, 900))
o = run(crops(im, M["crops"])); base = proj(o); box = autobox(o[0], "pca", .15); w, h = im.size
piece = im.crop((int(box[0]*w), int(box[1]*h), int(box[2]*w), int(box[3]*h)))
for name, Q in [("whole", base), ("whole+auto", np.vstack([base, proj(run(crops(piece, M["crops"])))]))]:
    S = Q @ V.T; P = np.full((len(Q), len(hs)), -9.0); np.maximum.at(P.T, owner, S.T); sc = P.max(0); order = list(np.argsort(-sc))
    r = order.index(hs.index(H)) + 1 if H in hs else -1
    print("RANK", name, r, "score", round(float(sc[hs.index(H)]), 3), "top5", [(titles.get(hs[i], hs[i]), round(float(sc[i]), 3)) for i in order[:5]])
