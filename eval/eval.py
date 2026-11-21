"""Compare MobileNetV3 (MediaPipe) vs DINOv2-small (ONNX) for product photo retrieval on the store's bracelets."""
import io, json, random, time, numpy as np, requests, onnxruntime as ort, mediapipe as mp
from PIL import Image, ImageEnhance, ImageFilter
from mediapipe.tasks.python import vision, BaseOptions
S = requests.Session()
STORE = "https://sa-jewel.com"
ps = S.get(f"{STORE}/collections/accessories-bracelet/products.json?limit=250", timeout=60).json()["products"]
ps = [p for p in ps if len(p["images"]) >= 3][:200]
def get(u):
    r = S.get(u + ("&" if "?" in u else "?") + "width=448", timeout=60); return Image.open(io.BytesIO(r.content)).convert("RGB")
imgs = [[get(p["images"][j]["src"]) for j in range(3)] for p in ps]
print("products", len(ps), flush=True)
IX = [(1.0,.5,.5),(0.6,.5,.5),(0.6,.5,.8),(0.45,.5,.6),(0.45,.5,.85),(0.45,.2,.75),(0.45,.8,.75)]
def crops(im, spec):
    w, h = im.size; o = []
    for s, fx, fy in spec:
        cw, ch = int(w*s), int(h*s); x = int((w-cw)*fx); y = int((h-ch)*fy); o.append(im.crop((x, y, x+cw, y+ch)))
    return o
open("mn.tflite","wb").write(S.get("https://storage.googleapis.com/mediapipe-models/image_embedder/mobilenet_v3_large/float32/latest/mobilenet_v3_large.tflite").content)
mpe = vision.ImageEmbedder.create_from_options(vision.ImageEmbedderOptions(base_options=BaseOptions(model_asset_path="mn.tflite"), l2_normalize=True))
def mn(ims): return np.stack([mpe.embed(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(np.asarray(i)))).embeddings[0].embedding for i in ims])
MEAN = np.array([0.485,0.456,0.406],np.float32); STD = np.array([0.229,0.224,0.225],np.float32)
sessions = {}
for name, url in [("dino_q", "https://huggingface.co/Xenova/dinov2-small/resolve/main/onnx/model_quantized.onnx"),
                  ("dino_fp32", "https://huggingface.co/Xenova/dinov2-small/resolve/main/onnx/model.onnx")]:
    b = S.get(url, timeout=300).content; open(name+".onnx","wb").write(b); print(name, "MB", round(len(b)/1e6,1), flush=True)
    sessions[name] = ort.InferenceSession(name+".onnx", providers=["CPUExecutionProvider"])
def dino(name):
    sess = sessions[name]; inp = sess.get_inputs()[0].name
    def f(ims):
        out = []
        for k in range(0, len(ims), 32):
            x = np.stack([(np.asarray(i.resize((224,224), Image.BICUBIC), np.float32)/255 - MEAN)/STD for i in ims[k:k+32]]).transpose(0,3,1,2)
            o = sess.run(None, {inp: x})[0][:, 0, :]
            out.append(o)
        o = np.concatenate(out); return o/np.linalg.norm(o, axis=1, keepdims=True)
    return f
random.seed(5)
def phone(im, tight):
    w, h = im.size; s = random.uniform(.35,.5) if tight else random.uniform(.8,1)
    cw, ch = int(w*s), int(h*s); x = int((w-cw)*random.uniform(.3,.7)); y = int((h-ch)*random.uniform(.4,.8))
    im = im.crop((x,y,x+cw,y+ch)).rotate(random.uniform(-15,15), fillcolor=(230,225,220))
    return ImageEnhance.Brightness(im).enhance(random.uniform(.7,1.2)).filter(ImageFilter.GaussianBlur(random.uniform(0,1)))
queries = {t: [phone(imgs[i][2], t) for i in range(len(ps))] for t in (True, False)}
QC = [(1,.5,.5),(.75,.5,.5)]
for name, fn in [("mobilenet", mn), ("dino_q", dino("dino_q")), ("dino_fp32", dino("dino_fp32"))]:
    t = time.time()
    X = []; L = []
    for i in range(len(ps)):
        for j in range(2):
            v = fn(crops(imgs[i][j], IX)); X.append(v); L += [i]*len(v)
    X = np.concatenate(X); L = np.array(L)
    per = (time.time()-t)/len(X)
    for single in (True, False):
        Xi = X[::7] if single else X; Li = L[::7] if single else L
        for tight in (True, False):
            t1 = t5 = 0
            for i, q in enumerate(queries[tight]):
                Q = fn(crops(q, QC)); s = (Xi @ Q.T).max(1)
                best = np.full(len(ps), -9.0); np.maximum.at(best, Li, s); o = np.argsort(-best)[:5]
                t1 += o[0] == i; t5 += i in o
            n = len(ps)
            print(f"RESULT {name:10s} index={'full-only' if single else 'multicrop'} query={'tight' if tight else 'loose'} top1={100*t1/n:.0f}% top5={100*t5/n:.0f}%  ({per*1000:.0f} ms/img)", flush=True)
