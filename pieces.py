"""Where is the jewel in a product photo? OWLv2 (free, open-vocabulary detector) returns one box per photo,
so the index embeds the piece itself instead of the box, card or background around it.
Boxes live in the `boxes` branch (boxes.json: url -> [x0, y0, x1, y1, score], fractions of the photo).
  python pieces.py shard I N   -> boxes_I.json for photos that have no box yet (used by .github/workflows/boxes.yml)
  python pieces.py merge       -> out_boxes/boxes.json = old boxes + all boxes_*.json"""
import os, sys, io, json, glob, time
import requests
from PIL import Image

RAW_BOXES = "https://raw.githubusercontent.com/{repo}/boxes/boxes.json"
QUERIES = {  # what to look for, per index category
    "bracelets": ["a bracelet", "a chain bracelet", "a bangle", "a piece of jewelry"],
    "necklaces": ["a necklace", "a pendant necklace", "a chain necklace", "a piece of jewelry"],
    "earrings": ["an earring", "a pair of earrings", "a piece of jewelry"],
    "rings": ["a ring", "a piece of jewelry"],
    "sets": ["a jewelry set", "a necklace", "a bracelet", "an earring", "a ring"],
    "bags": ["a handbag", "a bag", "a wallet"],
    "other": ["a piece of jewelry", "an accessory", "a watch", "a hair clip"],
}
UNION = {"sets"}  # a set is several pieces: keep every strong box together

def load_boxes(repo=None):
    repo = repo or os.environ.get("GITHUB_REPOSITORY", "jewelodoo/vs-index")
    try:
        r = requests.get(RAW_BOXES.format(repo=repo) + f"?t={int(time.time())}", timeout=60)
        return r.json() if r.ok else {}
    except Exception:
        return {}

def detector():
    import torch
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    name = "google/owlv2-base-patch16-ensemble"
    proc = Owlv2Processor.from_pretrained(name); model = Owlv2ForObjectDetection.from_pretrained(name).eval()
    torch.set_num_threads(os.cpu_count() or 2)
    def find(im, cat):
        q = QUERIES.get(cat, QUERIES["other"])
        with torch.no_grad():
            out = model(**proc(text=[q], images=im, return_tensors="pt"))
        side = max(im.size); ts = torch.tensor([[side, side]])  # OWLv2 pads to a square
        pp = getattr(proc, "post_process_grounded_object_detection", None)
        res = (pp(outputs=out, threshold=0.05, target_sizes=ts) if pp else proc.image_processor.post_process_object_detection(out, threshold=0.05, target_sizes=ts))[0]
        if not len(res["scores"]): return None
        sc = res["scores"].tolist(); bx = res["boxes"].tolist(); best = max(sc); w, h = im.size
        keep = [b for b, s in zip(bx, sc) if s >= 0.5 * best] if cat in UNION else [bx[sc.index(best)]]
        x0 = min(b[0] for b in keep); y0 = min(b[1] for b in keep); x1 = max(b[2] for b in keep); y1 = max(b[3] for b in keep)
        return [round(max(0, x0) / w, 4), round(max(0, y0) / h, 4), round(min(w, x1) / w, 4), round(min(h, y1) / h, 4), round(best, 3)]
    return find

def piece(im, box, margin=.12):
    """The photo cut to the piece (+12% each side). No box, or a box that is nearly the whole photo: the photo."""
    if not box: return im
    x0, y0, x1, y1 = box[:4]
    if (x1 - x0) * (y1 - y0) > .85 or x1 - x0 < .03 or y1 - y0 < .03: return im
    w, h = im.size; mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    return im.crop((int(max(0, x0 - mx) * w), int(max(0, y0 - my) * h), int(min(1, x1 + mx) * w), int(min(1, y1 + my) * h)))

def fetch(u, w=768, S=requests):
    for t in range(3):
        try:
            r = S.get(u + ("&" if "?" in u else "?") + f"width={w}", timeout=60)
            if r.ok: return Image.open(io.BytesIO(r.content)).convert("RGB")
        except Exception: pass
        time.sleep(2)

def detect(todo, find=None, log_every=50):
    """todo: [(url, cat)] -> {url: box or None}"""
    find = find or detector(); out = {}; t0 = time.time()
    for k, (u, cat) in enumerate(todo):
        im = fetch(u)
        out[u] = find(im, cat) if im is not None else None
        if k % log_every == 0: print("boxes", k, len(todo), round(time.time() - t0), "s", flush=True)
    return out

if __name__ == "__main__":
    import build_index as B
    if sys.argv[1] == "shard":
        I, N = int(sys.argv[2]), int(sys.argv[3])
        allp, members = B.catalog()
        have = load_boxes()
        todo = sorted({(u, cat) for cat, hs in members.items() for h in hs for u in B.img_urls(allp[h]) if u not in have})[I::N]
        print("to detect", len(todo))
        json.dump(detect(todo), open(f"boxes_{I}.json", "w"))
    elif sys.argv[1] == "merge":
        allb = load_boxes()
        for f in glob.glob("parts/**/boxes_*.json", recursive=True): allb.update(json.load(open(f)))
        os.makedirs("out_boxes", exist_ok=True); json.dump(allb, open("out_boxes/boxes.json", "w"), separators=(",", ":"))
        print("boxes", len(allb), "with a piece", sum(1 for v in allb.values() if v))
