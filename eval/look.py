"""Contact sheet of a few products' store photos (all images, 300px) -> sheets/look.jpg, to compare with shopper photos."""
import io, sys, requests
from PIL import Image, ImageDraw
H = sys.argv[1].split(",")
rows = []
for h in H:
    p = requests.get(f"https://sa-jewel.com/products/{h}.js", headers={"User-Agent": "jewel-vs-test"}).json()
    ims = []
    for u in p["images"][:8]:
        u = ("https:" + u if u.startswith("//") else u); u += ("&" if "?" in u else "?") + "width=300"
        ims.append(Image.open(io.BytesIO(requests.get(u).content)).convert("RGB").resize((300, 300)))
    print(h, p["title"], len(p["images"]))
    rows.append(ims)
W = max(len(r) for r in rows) * 300
sheet = Image.new("RGB", (W, 300 * len(rows)), "white")
for y, r in enumerate(rows):
    for x, im in enumerate(r): sheet.paste(im, (x * 300, y * 300))
import os; os.makedirs("sheets", exist_ok=True); sheet.save("sheets/look.jpg", quality=80)
