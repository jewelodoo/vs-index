"""Automatic learning for the visual search (25 Nov).

Shoppers' photos often look nothing like the store photos (on a hand, on a card, no brand box), so the plain
DINOv2 match misses them while the AI second look finds them. Every night this adds the numbers of those
shopper photos to the product they turned out to be, as extra "photos" of that product in the index.
The next shopper with a similar photo then finds it with the normal search, no AI needed.

Sources (both optional):
  1. The learning log of the Jewel app (GET /vs/log/export?key=VS_LOG_KEY). The theme sends the raw DINOv2
     numbers of the search (never the photo) when the AI marks a result «نفس القطعة» (kind "exact"), or when the
     shopper opens a product from the results (kind "open", with its place in the list).
  2. Photos in learn/<product handle>/ in this repository: the very same piece.
     Photos in learn-similar/<product handle>/: a look-alike (the piece itself is not sold), shown first but never «نفس القطعة».

«نفس القطعة» in the theme needs a score >= 0.94. Look-alikes and "open" searches (the shopper may open a look-alike)
are stored at SIMILAR x their score (0.9), so they still come first but can never reach that mark; "exact" and
learn/ photos are stored as they are.

Trusted: every "exact"; an "open" only from the first OPEN_RANK places. At most PER_PRODUCT log searches per
product (newest first), near-identical searches (cosine >= 0.985) counted once.
"""
import os, io, glob, base64, time
import numpy as np, requests
from PIL import Image

LOG_URL = os.environ.get("VS_LOG_URL", "https://jewel-app-ic3o.onrender.com/vs/log/export")
OPEN_RANK = int(os.environ.get("LEARN_OPEN_RANK", "6"))
PER_PRODUCT = int(os.environ.get("LEARN_PER_PRODUCT", "20"))
SEED_DIR = "learn"
SIMILAR_DIR = "learn-similar"
SIMILAR = float(os.environ.get("LEARN_SIMILAR", "0.9"))


def _f16(b64, n, dim):
    a = np.frombuffer(base64.b64decode(b64), dtype="<f2").astype(np.float32)
    if a.size != n * dim: return None
    a = a.reshape(n, dim)
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)


def log_rows():
    key = os.environ.get("VS_LOG_KEY", "").strip()
    if not key:
        print("learn: no VS_LOG_KEY, the shoppers' log is skipped"); return []
    rows, after = [], 0
    for _ in range(100):  # 5000 a page
        try:
            r = requests.get(LOG_URL, params={"key": key, "after": after}, timeout=90)
        except Exception as e:
            print("learn: log not reachable:", e); break
        if r.status_code == 404: print("learn: the app refused the key (VS_LOG_KEY differs in Render?)"); break
        if not r.ok: print("learn: log error", r.status_code); break
        page = r.json().get("rows", [])
        rows += page
        if len(page) < 5000: break
        after = page[-1]["id"]; time.sleep(1)
    print("learn: log rows", len(rows))
    return rows


def seeds(embed, crops, folder, scale):
    out = {}
    for d in sorted(glob.glob(f"{folder}/*/")):
        h = os.path.basename(os.path.normpath(d))
        for f in sorted(glob.glob(d + "*")):
            if not f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")): continue
            try:
                im = Image.open(f).convert("RGB"); im.thumbnail((900, 900))
                out.setdefault(h, []).append((embed(crops(im)).astype(np.float32), scale))
            except Exception as e: print("learn: seed photo skipped", f, e)
    print("learn:", folder, "photos for", len(out), "products")
    return out


def learned(handles, embed, crops):
    """handle -> list of ((k, 384) array, scale) to add next to the product's own photo vectors (scale: 1 same piece, SIMILAR look-alike)."""
    out = {}
    for folder, scale in ((SEED_DIR, 1.0), (SIMILAR_DIR, SIMILAR)):
        for h, v in seeds(embed, crops, folder, scale).items():
            if h in handles: out.setdefault(h, []).extend(v)
    kept, skipped = {}, {"untrusted": 0, "unknown": 0, "same": 0, "cap": 0, "bad": 0}
    for row in sorted(log_rows(), key=lambda r: -int(r["id"])):  # newest first
        h = row.get("handle")
        if h not in handles: skipped["unknown"] += 1; continue
        if row.get("kind") != "exact" and not (row.get("kind") == "open" and row.get("rank") and int(row["rank"]) <= OPEN_RANK):
            skipped["untrusted"] += 1; continue
        if int(row.get("dim") or 0) != 384: skipped["bad"] += 1; continue
        v = _f16(row["vec"], int(row["n"]), 384)
        if v is None: skipped["bad"] += 1; continue
        have = kept.setdefault(h, [])
        if len(have) >= PER_PRODUCT: skipped["cap"] += 1; continue
        if any(float(v[0] @ x[0][0]) >= 0.985 for x in have): skipped["same"] += 1; continue
        have.append((v, 1.0 if row.get("kind") == "exact" else SIMILAR))
    for h, vs in kept.items(): out.setdefault(h, []).extend(vs)
    print("learn: shopper searches kept", sum(len(v) for v in kept.values()), "for", len(kept), "products; skipped", skipped)
    print("learn: products with learned photos", len(out))
    return out
