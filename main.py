from fastapi import FastAPI, UploadFile, File, HTTPException
from ultralytics import YOLO
from PIL import Image
import io
import torch
import clip
import base64
import requests
import numpy as np
from pydantic import BaseModel
from typing import List

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

yolo_model = YOLO('yolov8n.pt') 
clip_model, preprocess = clip.load("ViT-B/32", device=device)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

CATEGORY_MAPPING = {
    "shirt": "top", "t-shirt": "top", "jacket": "top", "coat": "top", "sweater": "top", "dress": "top",
    "pants": "bottom", "jeans": "bottom", "shorts": "bottom", "skirt": "bottom",
    "sneakers": "shoes", "boots": "shoes"
}

# Highly distinctive prompts — maximise inter-class separation for CLIP zero-shot
CLIP_CATEGORY_PROMPTS = {
    "top": [
        "a plain crew-neck t-shirt on a white background",
        "a fitted blouse with front buttons",
        "a chunky knit sweater or wool pullover",
        "a zip-up hoodie or cotton sweatshirt",
        "a tailored blazer or sport jacket worn by a model",
        "a long winter coat or heavy parka outerwear",
    ],
    "bottom": [
        "blue denim jeans full length on a white background",
        "wide-leg linen trousers with a high waist",
        "tailored suit trousers or formal dress pants",
        "casual chino shorts above the knee",
        "cargo pants with large patch pockets on the sides",
    ],
    "skirt": [
        "a midi-length pleated fabric skirt",
        "a short mini flared skirt on a model",
        "a long flowing maxi skirt",
        "a tight knee-length pencil skirt",
    ],
    "dress": [
        "a full-length evening gown worn by a woman",
        "a sleeveless casual summer sundress",
        "a short bodycon cocktail dress",
        "a floral wrap dress with a tied waist",
    ],
    "shoes": [
        "white leather sneakers isolated on white background",
        "high-heel stiletto pumps on a shelf",
        "ankle boots with a chunky thick sole",
        "flat leather oxford shoes from the side",
        "open-toe strappy sandals on a white background",
    ],
}

# Pre-compute text features once at startup so inference stays fast
_category_text_features: dict = {}
with torch.no_grad():
    for _cat, _texts in CLIP_CATEGORY_PROMPTS.items():
        _tokens = clip.tokenize(_texts).to(device)
        _feats = clip_model.encode_text(_tokens)
        _feats /= _feats.norm(dim=-1, keepdim=True)
        _category_text_features[_cat] = _feats

# Multiple prompts per color → averaged into one strong color centroid.
# More prompts = more robust colour direction in CLIP's joint embedding space.
COLOR_TEXT_PROMPTS = {
    "black":      ["a black top", "a solid black t-shirt", "black clothing item", "a dark black garment on white background"],
    "white":      ["a white top", "a pure white t-shirt", "white clothing item", "a bright white garment on white background"],
    "beige":      ["a beige top", "a cream colored t-shirt", "beige neutral clothing", "a sand-colored garment"],
    "grey":       ["a grey top", "a gray t-shirt", "grey clothing item", "a charcoal grey garment"],
    "navy":       ["a navy blue top", "a dark navy t-shirt", "navy blue clothing", "a deep navy colored garment"],
    "red":        ["a red top", "a bright red t-shirt", "red clothing item", "a vivid red garment"],
    "burgundy":   ["a burgundy top", "a dark wine red t-shirt", "burgundy clothing", "a deep burgundy garment"],
    "brown":      ["a brown top", "an earthy brown t-shirt", "brown clothing item", "a caramel brown garment"],
    "olive":      ["an olive top", "an olive green t-shirt", "olive drab clothing", "an army green garment"],
    "light_blue": ["a light blue top", "a sky blue t-shirt", "light blue clothing item", "a pale blue garment"],
    "pink":       ["a pink top", "a soft pink t-shirt", "pink clothing item", "a rose pink garment"],
    "green":      ["a green top", "a forest green t-shirt", "green clothing item", "an emerald green garment"],
    "yellow":     ["a yellow top", "a bright yellow t-shirt", "yellow clothing item", "a golden yellow garment"],
}

# Pre-compute and average all color text embeddings at startup
_color_text_features: dict = {}
with torch.no_grad():
    for _color, _prompts in COLOR_TEXT_PROMPTS.items():
        _tokens = clip.tokenize(_prompts).to(device)
        _feats = clip_model.encode_text(_tokens)
        _feats /= _feats.norm(dim=-1, keepdim=True)
        # Average the prompt embeddings → single robust colour centroid
        _centroid = _feats.mean(dim=0, keepdim=True)
        _centroid /= _centroid.norm(dim=-1, keepdim=True)
        _color_text_features[_color] = _centroid

FASHION_COLORS = {
    "black": (0, 0, 0), "white": (245, 245, 245), "beige": (222, 199, 166),
    "grey": (128, 128, 128), "brown": (101, 67, 33), "olive": (85, 107, 47),
    "navy": (0, 0, 128), "light_blue": (135, 206, 250), "red": (200, 0, 0),
    "burgundy": (128, 0, 32), "pink": (255, 182, 193), "green": (34, 139, 34),
    "yellow": (255, 215, 0)
}

def get_fashion_color(pil_img):
    # ── Step 1: centre crop ───────────────────────────────────────────────────
    # Focus on the middle 80 % width × 70 % height so the model's face/neck
    # above and jeans/legs below don't bleed into the garment colour reading.
    w, h = pil_img.size
    cx0, cx1 = int(w * 0.10), int(w * 0.90)
    cy0, cy1 = int(h * 0.15), int(h * 0.85)
    pil_img = pil_img.crop((cx0, cy0, cx1, cy1))

    img_small = pil_img.convert("RGB").resize((100, 100))
    pixels = np.array(img_small).reshape(-1, 3).astype(np.float32)

    brightness = pixels.mean(axis=1)
    saturation = pixels.max(axis=1) - pixels.min(axis=1)

    # Layer 1: bright pixels (> 200 on all channels) → background
    is_bright_bg = (pixels[:, 0] > 200) & (pixels[:, 1] > 200) & (pixels[:, 2] > 200)
    # Layer 2: neutral-grey pixels bright enough to be background (not dark clothing)
    is_grey_bg   = (saturation < 20) & (brightness > 150)
    # Layer 3: skin tones — warm (R significantly > B), medium brightness
    is_skin = (
        (pixels[:, 0] > 100) &
        (pixels[:, 0] > pixels[:, 2]) &
        (pixels[:, 0] - pixels[:, 2] > 30) &
        (brightness > 90) & (brightness < 210)
    )

    garment_pixels = pixels[~(is_bright_bg | is_grey_bg | is_skin)]

    if len(garment_pixels) < 100:
        garment_pixels = pixels[brightness <= 210]
    if len(garment_pixels) < 50:
        garment_pixels = pixels

    avg_color = garment_pixels.mean(axis=0)
    r, g, b = float(avg_color[0]), float(avg_color[1]), float(avg_color[2])
    avg_brightness = (r + g + b) / 3.0

    # ── Step 2: dark-colour rule ──────────────────────────────────────────────
    # Very dark averages are always black or navy — skip palette matching which
    # confuses near-black dark-browns as "brown".
    if avg_brightness < 60:
        result = "navy" if (b > r + 25) else "black"
        print(f"🎨 Detected color: {result} (dark-rule)  avg_rgb=({r:.0f},{g:.0f},{b:.0f})")
        return result

    # ── Step 3: palette matching ──────────────────────────────────────────────
    min_distance = float('inf')
    closest_color = "other"
    for color_name, rgb_value in FASHION_COLORS.items():
        distance = (r - rgb_value[0])**2 + (g - rgb_value[1])**2 + (b - rgb_value[2])**2
        if distance < min_distance:
            min_distance = distance
            closest_color = color_name

    # ── Step 4: post-correction ───────────────────────────────────────────────
    # Warm colours (R >> B) that ended up as "grey" are actually beige/tan.
    if closest_color == "grey" and (r - b) > 30:
        closest_color = "beige"

    closest_color = closest_color.lower()
    print(f"🎨 Detected color: {closest_color}  avg_rgb=({r:.0f},{g:.0f},{b:.0f})")
    return closest_color


def detect_category_clip(pil_img) -> str:
    """Zero-shot CLIP classification → one of: top, bottom, skirt, dress, shoes."""
    image_input = preprocess(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        best_cat = "top"
        best_score = -1.0
        for cat, text_feats in _category_text_features.items():
            sims = (image_features @ text_feats.T).squeeze(0)
            score = float(sims.max())
            if score > best_score:
                best_score = score
                best_cat = cat
    return best_cat


def get_category_group(yolo_label):
    return CATEGORY_MAPPING.get(yolo_label.lower(), "other")

def get_color_vector(color: str) -> list:
    """Return the pre-computed CLIP text centroid for a colour as a plain list.
    This is sent to the backend so it can re-rank candidates by colour affinity
    WITHOUT modifying the query embedding (which would break image-image similarity)."""
    feats = _color_text_features.get(color)
    if feats is None:
        return []
    return feats.cpu().numpy().flatten().tolist()


def process_image_logic(img):
    results = yolo_model(img)
    found_items = []

    if len(results[0].boxes) == 0:
        image_input = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            # Pure image embedding — no blending (keeps image-image similarity intact)
            embedding = image_features.cpu().numpy().flatten().tolist()

        color = get_fashion_color(img)
        category_group = detect_category_clip(img)

        found_items.append({
            "category": "other",
            "categoryGroup": category_group,
            "confidence": 1.0,
            "embedding": embedding,
            "color": color,
            "colorVector": get_color_vector(color),
        })
    else:
        for r in results:
            for box in r.boxes:
                label = yolo_model.names[int(box.cls)]
                conf = float(box.conf)
                if conf > 0.2:
                    coords = box.xyxy[0].tolist()
                    crop_img = img.crop((coords[0], coords[1], coords[2], coords[3]))
                    image_input = preprocess(crop_img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        image_features = clip_model.encode_image(image_input)
                        image_features /= image_features.norm(dim=-1, keepdim=True)
                        embedding = image_features.cpu().numpy().flatten().tolist()

                    color = get_fashion_color(crop_img)
                    category_group = detect_category_clip(crop_img)

                    found_items.append({
                        "category": get_category_group(label),
                        "categoryGroup": category_group,
                        "confidence": conf,
                        "embedding": embedding,
                        "color": color,
                        "colorVector": get_color_vector(color),
                    })
    return found_items

class URLRequest(BaseModel):
    image_url: str

@app.post("/process-url")
async def process_url(data: URLRequest):
    try:
        response = requests.get(data.image_url, headers=HEADERS, timeout=15)
        if response.status_code == 403:
            raise HTTPException(status_code=403, detail="The website blocked the image request (403 Forbidden)")
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content)).convert("RGB")
        return {"items": process_image_logic(img)}
    except Exception as e:
        print(f"Error processing URL {data.image_url}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract-color")
async def extract_color_from_url(data: URLRequest):
    try:
        response = requests.get(data.image_url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content)).convert("RGB")
        detected_color = get_fashion_color(img)
        return {"color": detected_color}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# ---------------------------------------------------

@app.post("/process-look")
async def process_look_file(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    return {"items": process_image_logic(img)}

class ImageRequest(BaseModel):
    image: str

@app.post("/process-look-base64")
async def process_look_base64(data: ImageRequest):
    base64_data = data.image.split(",")[1] if "," in data.image else data.image
    img_bytes = base64.b64decode(base64_data)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return {"items": process_image_logic(img)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

#source venv/Scripts/activate
#pip install fastapi uvicorn python-multipart
#python main.py