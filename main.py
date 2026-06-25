from fastapi import FastAPI, UploadFile, File, HTTPException
from ultralytics import YOLO
from PIL import Image
import io
import torch
from transformers import CLIPModel, CLIPProcessor
import base64
import requests
import numpy as np
from pydantic import BaseModel
from typing import List

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

yolo_model = YOLO('yolov8n.pt')

# fashion-clip: CLIP ViT-B/32 fine-tuned on 700k fashion image-text pairs.
# Same 512-dim output as the generic ViT-B/32 → no MongoDB index changes needed.
print("Loading fashion-clip model (first run downloads ~600MB from HuggingFace)...")
clip_model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip").to(device)
clip_processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
clip_model.eval()
print("fashion-clip loaded.")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

CATEGORY_MAPPING = {
    "shirt": "top", "t-shirt": "top", "jacket": "top", "coat": "top", "sweater": "top", "dress": "top",
    "pants": "bottom", "jeans": "bottom", "shorts": "bottom", "skirt": "bottom",
    "sneakers": "shoes", "boots": "shoes"
}

# Highly distinctive prompts — maximise inter-class separation for zero-shot.
# Key rule: "top" must cover ALL torso-only garments, including strapless/tube
# styles, so they are never mis-routed to "dress" (which is full-length/skirted).
CLIP_CATEGORY_PROMPTS = {
    "top": [
        "a plain crew-neck t-shirt on a white background",
        "a fitted blouse with front buttons",
        "a chunky knit sweater or wool pullover",
        "a zip-up hoodie or cotton sweatshirt",
        "a tailored blazer or sport jacket worn by a model",
        "a long winter coat or heavy parka outerwear",
        "a strapless tube top or bandeau top worn by a woman",
        "a fitted corset-style strapless top cropped at the waist",
        "an off-shoulder crop top showing bare shoulders",
        "a sleeveless tank top or camisole",
        "a tight bodysuit with no straps on a model",
        "a soft lavender v-neck t-shirt on a model",
        "a dusty pink mauve cotton tee with short sleeves",
        "a light purple heathered jersey top",
    ],
    "bottom": [
        "blue denim jeans full length on a white background",
        "wide-leg linen trousers with a high waist",
        "tailored suit trousers or formal dress pants",
        "casual chino shorts above the knee",
        "cargo pants with large patch pockets on the sides",
        "flowy printed shorts with an elastic waistband above the knee",
        "boho floral mini shorts with a drawstring waist",
        "patterned shorts showing both legs with a crotch seam",
    ],
    "skirt": [
        "a midi-length pleated fabric skirt",
        "a short mini flared skirt on a model",
        "a long flowing maxi skirt",
        "a tight knee-length pencil skirt",
    ],
    "dress": [
        "a full-length evening gown that covers the legs worn by a woman",
        "a sleeveless casual summer sundress with a skirt below the knee",
        "a short bodycon cocktail dress with a skirt",
        "a floral wrap dress with a tied waist and a flowing skirt",
        "a midi dress reaching below the knee on a model",
        "a black sleeveless maxi column dress reaching the ankles on a model",
        "a long black tank-style slip dress full length worn by a woman",
        "a minimalist black maxi dress with wide straps and straight silhouette",
    ],
    "shoes": [
        "white leather sneakers isolated on white background",
        "high-heel stiletto pumps on a shelf",
        "ankle boots with a chunky thick sole",
        "flat leather oxford shoes from the side",
        "open-toe strappy sandals on a white background",
        "nude pink cross strap slide sandals with cork footbed",
        "beige flat slide sandals with toe loop on feet",
        "open toe leather flat sandals worn on feet",
        "cork footbed slide sandals with crossed straps",
    ],
}

# Shoe sub-type prompts — distinguish slide sandals from birkenstocks, heels, etc.
CLIP_SHOE_STYLE_PROMPTS = {
    "slide_sandal": [
        "flat cross strap slide sandals with toe loop and cork footbed on feet",
        "nude pink leather slide sandals with crossed straps and toe ring",
        "open toe flat mule slide sandals worn on feet",
    ],
    "birkenstock": [
        "double buckle strap birkenstock cork sandals on feet",
        "two wide buckled leather straps sandals with cork sole",
        "leopard print double buckle birkenstock sandals on feet",
    ],
    "heeled_sandal": [
        "high heel strappy dress sandals with ankle strap on feet",
        "kitten heel sandals with decorative flower on the toe",
    ],
    "espadrille": [
        "closed toe espadrille flat shoes with woven jute rope sole",
        "beige canvas espadrille loafers on feet",
    ],
    "puffy_slide": [
        "puffy quilted cross strap pillow slide sandals on feet",
        "thick padded strap slide sandals on feet",
    ],
}

# Pre-compute text features once at startup so inference stays fast
def _encode_texts(texts: list) -> torch.Tensor:
    """Encode a list of text strings → normalized (N, 512) tensor.

    Uses the underlying text_model + text_projection directly to avoid
    transformers version differences in get_text_features() return types.
    """
    inputs = clip_processor(
        text=texts, return_tensors="pt", padding=True, truncation=True
    ).to(device)
    with torch.no_grad():
        text_outputs = clip_model.text_model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
        )
        pooled = text_outputs.pooler_output          # (N, hidden_size)
        feats = clip_model.text_projection(pooled)   # (N, 512)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats

_category_text_features: dict = {}
for _cat, _texts in CLIP_CATEGORY_PROMPTS.items():
    _feats = _encode_texts(_texts)
    _category_text_features[_cat] = _feats

_shoe_style_text_features: dict = {}
for _style, _texts in CLIP_SHOE_STYLE_PROMPTS.items():
    _shoe_style_text_features[_style] = _encode_texts(_texts)

def _encode_image(pil_img: Image.Image) -> torch.Tensor:
    """Encode a PIL image → normalized (1, 512) tensor.

    Uses the underlying vision_model + visual_projection directly to avoid
    transformers version differences in get_image_features() return types.
    """
    inputs = clip_processor(images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        vision_outputs = clip_model.vision_model(
            pixel_values=inputs.pixel_values,
        )
        pooled = vision_outputs.pooler_output            # (1, hidden_size)
        feats = clip_model.visual_projection(pooled)     # (1, 512)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats

# Multiple prompts per color → averaged into one strong color centroid.
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
    "lavender":   ["a lavender top", "a dusty purple blouse", "a lilac clothing item", "a soft mauve purple garment"],
    "purple":     ["a purple top", "a deep violet blouse", "a rich purple clothing item", "a vivid purple garment"],
}

_color_text_features: dict = {}
for _color, _prompts in COLOR_TEXT_PROMPTS.items():
    _feats = _encode_texts(_prompts)
    _centroid = _feats.mean(dim=0, keepdim=True)
    _centroid = _centroid / _centroid.norm(dim=-1, keepdim=True)
    _color_text_features[_color] = _centroid

# ── Fabric detection ─────────────────────────────────────────────────────────
# These prompts let fashion-clip distinguish denim/jersey/knit/woven at query
# time. The detected fabric is then combined with color to produce a specific
# re-ranking vector ("grey denim jeans") instead of the generic plain color
# vector ("grey"), so sweatpants and chinos score lower for a denim query.
CLIP_FABRIC_PROMPTS = {
    "denim": [
        "blue denim jeans woven cotton fabric",
        "washed denim jeans product photo",
        "denim jeans on a model",
    ],
    "jersey": [
        "soft cotton jersey sweatpants joggers",
        "fleece jersey fabric athletic pants",
        "cotton jersey knit sportswear",
    ],
    "knit": [
        "chunky ribbed knit sweater knitwear",
        "cable knit wool pullover sweater",
        "ribbed knit fabric top",
    ],
    "woven": [
        "tailored woven fabric dress trousers",
        "structured woven chino dress pants",
        "smooth woven fabric formal trousers",
    ],
    "linen": [
        "lightweight linen fabric trousers",
        "natural linen material clothing",
        "linen blend pants summer",
    ],
    "leather": [
        "leather or faux leather pants jacket",
        "PU leather material clothing",
        "genuine leather fashion item",
    ],
}

_fabric_text_features: dict = {}
for _fab, _texts in CLIP_FABRIC_PROMPTS.items():
    _feats = _encode_texts(_texts)
    _centroid = _feats.mean(dim=0, keepdim=True)
    _centroid = _centroid / _centroid.norm(dim=-1, keepdim=True)
    _fabric_text_features[_fab] = _centroid

# Template used to build a combined color+fabric re-ranking prompt.
# Category-specific: "lavender linen trousers" on a shirt crop boosts pants in results.
FABRIC_COLOR_TEMPLATES_BOTTOM = {
    "denim":   "{color} denim jeans",
    "jersey":  "{color} jersey sweatpants",
    "knit":    "{color} knit sweater",
    "woven":   "{color} woven dress pants",
    "linen":   "{color} linen trousers",
    "leather": "{color} leather pants",
}

FABRIC_COLOR_TEMPLATES_TOP = {
    "denim":   "{color} denim jacket top",
    "jersey":  "{color} cotton t-shirt top",
    "knit":    "{color} knit sweater top",
    "woven":   "{color} woven blouse top",
    "linen":   "{color} linen shirt top",
    "leather": "{color} leather top",
}

FASHION_COLORS = {
    "black": (0, 0, 0), "white": (245, 245, 245), "beige": (222, 199, 166),
    "grey": (128, 128, 128), "brown": (101, 67, 33), "olive": (85, 107, 47),
    "navy": (0, 0, 128), "light_blue": (135, 206, 250), "red": (200, 0, 0),
    "burgundy": (128, 0, 32), "pink": (255, 182, 193), "green": (34, 139, 34),
    "yellow": (255, 215, 0),
    "lavender": (200, 162, 200),   # dusty lavender / muted purple-pink
    "purple":   (128, 60, 160),    # medium-dark vivid purple
}

def get_fashion_color(pil_img, category_group=None):
    # ── Step 1: centre crop ───────────────────────────────────────────────────
    # Tops/bottoms: middle 70% × central 60% — avoids belt and shoes bleeding in.
    # Shoes: lower 55% — focus on foot/sandal pixels, less skin variance.
    w, h = pil_img.size
    cx0, cx1 = int(w * 0.15), int(w * 0.85)
    if category_group == "shoes":
        cy0, cy1 = int(h * 0.40), int(h * 0.95)
    else:
        cy0, cy1 = int(h * 0.20), int(h * 0.80)
    pil_img = pil_img.crop((cx0, cy0, cx1, cy1))

    img_small = pil_img.convert("RGB").resize((100, 100))
    pixels = np.array(img_small).reshape(-1, 3).astype(np.float32)

    brightness = pixels.mean(axis=1)
    saturation = pixels.max(axis=1) - pixels.min(axis=1)

    # Layer 1: only mask near-pure-white pixels as background.
    is_bright_bg = (pixels[:, 0] > 235) & (pixels[:, 1] > 235) & (pixels[:, 2] > 230)
    # Layer 2: neutral-grey background pixels
    is_grey_bg   = (saturation < 15) & (brightness > 225)
    # Layer 3: skin tones
    is_skin = (
        (pixels[:, 0] > 100) &
        (pixels[:, 0] > pixels[:, 2]) &
        (pixels[:, 0] - pixels[:, 2] > 30) &
        (brightness > 90) & (brightness < 210)
    )

    garment_pixels = pixels[~(is_bright_bg | is_grey_bg | is_skin)]

    if len(garment_pixels) < 100:
        garment_pixels = pixels[brightness <= 240]
    if len(garment_pixels) < 50:
        garment_pixels = pixels

    avg_color = garment_pixels.mean(axis=0)
    r, g, b = float(avg_color[0]), float(avg_color[1]), float(avg_color[2])
    avg_brightness = (r + g + b) / 3.0

    # ── Step 1.5: stripe / pattern detection ─────────────────────────────────
    # Navy+white stripes average to a flat grey because the mixed pixel average
    # has near-zero saturation. Instead of using the overall average, re-derive
    # the colour from the DARK pixels only (brightness < 120).
    # Dark pixels of navy/white stripes average to ~(30,35,80) → dark-rule → "navy"
    # Dark pixels of black/white stripes average to ~(10,10,10) → dark-rule → "black"
    # This lets the normal colour pipeline filter correctly for the stripe colour.
    garment_brightness = garment_pixels.mean(axis=1)
    brightness_std = float(garment_brightness.std())

    # Light floral / patchwork prints — skip for shoes (straps + skin variance
    # falsely triggers this rule on solid-colour sandals).
    skip_pattern = category_group == "shoes"
    if not skip_pattern and brightness_std > 38 and avg_brightness > 155 and float(saturation.std()) > 18:
        print(f"🎨 Detected color: pattern (light-floral)  brightness_std={brightness_std:.1f}, avg={avg_brightness:.0f}")
        return "pattern", False

    if not skip_pattern and brightness_std > 50:
        dark_pixels = garment_pixels[garment_brightness < 120]
        if len(dark_pixels) >= 30:
            dark_avg = dark_pixels.mean(axis=0)
            r, g, b = float(dark_avg[0]), float(dark_avg[1]), float(dark_avg[2])
            avg_brightness = (r + g + b) / 3.0
            # Only treat as a simple stripe when the dark pixels are TRULY dark (< 65).
            # Floral/complex prints have darker flower petals averaging ~70-90 brightness —
            # using those as the "stripe colour" gives wrong results (e.g. brown flowers on
            # cream shorts → detected as brown, shows brown skirts instead of floral shorts).
            # For those, skip colour filtering and rely on visual similarity only.
            if avg_brightness >= 65:
                print(f"🎨 Detected color: pattern (complex-print)  brightness_std={brightness_std:.1f}, dark_avg={avg_brightness:.0f}")
                return "pattern", False
            print(f"🎨 Pattern detected (std={brightness_std:.1f}), re-scoring dark pixels avg=({r:.0f},{g:.0f},{b:.0f})")
            # Navy stripe pixels are often very dark (47,46,54) — a blue bias of only
            # 7 points. The global dark-rule requires b > r+15, which is too strict here.
            # Use a lower threshold of b > r+5 specifically for pattern-detected items.
            if avg_brightness < 60 and b > r + 5:
                print(f"🎨 Detected color: navy (dark-rule-pattern)  avg_rgb=({r:.0f},{g:.0f},{b:.0f})")
                return "navy", True   # is_stripe=True → caller uses stripe-specific vector
            # Fall through — normal rules now run on the dark-pixel average (non-navy stripes)
        else:
            # Mostly light-coloured print (e.g. pastel on white) — no dominant dark colour
            print(f"🎨 Detected color: pattern (light-print)  brightness_std={brightness_std:.1f}")
            return "pattern", False

    # ── Step 2: bright-colour rule (white garments) ───────────────────────────
    if avg_brightness > 190 and saturation.mean() < 25:
        print(f"🎨 Detected color: white (bright-rule)  avg_rgb=({r:.0f},{g:.0f},{b:.0f})")
        return "white", False

    # ── Step 3: dark-colour rule ──────────────────────────────────────────────
    if avg_brightness < 60:
        result = "navy" if (b > r + 15) else "black"
        print(f"🎨 Detected color: {result} (dark-rule)  avg_rgb=({r:.0f},{g:.0f},{b:.0f})")
        return result, False

    # ── Step 3.5: low-saturation grey detection ──────────────────────────────
    # Garments with near-zero colour saturation are grey/charcoal regardless of
    # which palette point wins in Euclidean distance. Without this rule,
    # dark grey (83,84,88) ends up classified as olive because the grey palette
    # reference (128,128,128) is much brighter — the distance to olive (85,107,47)
    # is accidentally smaller in RGB space even though olive has 60-point saturation.
    avg_saturation = max(r, g, b) - min(r, g, b)
    if avg_saturation < 20:
        print(f"🎨 Detected color: grey (low-sat rule)  avg_rgb=({r:.0f},{g:.0f},{b:.0f})")
        return "grey", False

    # ── Step 4: palette matching ──────────────────────────────────────────────
    min_distance = float('inf')
    closest_color = "other"
    for color_name, rgb_value in FASHION_COLORS.items():
        distance = (r - rgb_value[0])**2 + (g - rgb_value[1])**2 + (b - rgb_value[2])**2
        if distance < min_distance:
            min_distance = distance
            closest_color = color_name

    # ── Step 5: post-correction ───────────────────────────────────────────────
    if closest_color == "grey" and (r - b) > 30:
        closest_color = "beige"

    if closest_color == "grey" and (b - r) > 20:
        closest_color = "light_blue"

    closest_color = closest_color.lower()
    print(f"🎨 Detected color: {closest_color}  avg_rgb=({r:.0f},{g:.0f},{b:.0f})")
    return closest_color, False


def _score_categories(pil_img) -> dict:
    """Return max CLIP similarity per category for an image crop."""
    image_features = _encode_image(pil_img)
    scores = {}
    for cat, text_feats in _category_text_features.items():
        sims = (image_features @ text_feats.T).squeeze(0)
        scores[cat] = float(sims.max())
    return scores


def detect_category_clip(pil_img) -> str:
    """Zero-shot fashion-clip classification → one of: top, bottom, skirt, dress, shoes."""
    w, h = pil_img.size

    if h > w * 0.85:
        # Portrait: score full body + zones. Upper half of a maxi dress looks like a tank
        # top — always check full-image dress score before trusting upper-body "top".
        full_scores = _score_categories(pil_img)
        upper_scores = _score_categories(pil_img.crop((0, 0, w, int(h * 0.55))))
        lower_scores = _score_categories(pil_img.crop((0, int(h * 0.42), w, h)))

        lower_best = max(lower_scores, key=lower_scores.get)
        upper_best = max(upper_scores, key=upper_scores.get)

        if lower_best == "shoes" and lower_scores["shoes"] >= lower_scores.get("bottom", 0):
            best_cat = "shoes"
            all_scores = lower_scores
            region = "lower-feet"
        elif (full_scores["dress"] >= full_scores["top"] - 0.015
              and full_scores["dress"] >= max(
                  full_scores.get("skirt", 0),
                  full_scores.get("bottom", 0) - 0.01,
              )):
            best_cat = "dress"
            all_scores = full_scores
            region = "full-dress"
        elif upper_best == "top" and upper_scores["top"] > upper_scores.get("bottom", 0) + 0.01:
            if full_scores["dress"] < full_scores["top"] - 0.02:
                best_cat = "top"
                all_scores = upper_scores
                region = "upper-body"
            else:
                best_cat = "dress"
                all_scores = full_scores
                region = "full-dress-override"
        else:
            merged = {
                cat: max(lower_scores[cat], upper_scores[cat], full_scores[cat])
                for cat in lower_scores
            }
            best_cat = max(merged, key=merged.get)
            all_scores = merged
            region = "portrait-merged"
    else:
        all_scores = _score_categories(pil_img)
        best_cat = max(all_scores, key=all_scores.get)
        region = "full"

    rounded = {k: round(v, 4) for k, v in all_scores.items()}
    scores_str = "  ".join(f"{c}={s}" for c, s in sorted(rounded.items(), key=lambda x: -x[1]))
    print(f"📐 Category scores ({region}): {scores_str}  → winner: {best_cat}")
    return best_cat


def detect_shoe_style_clip(pil_img) -> str:
    """Classify shoe sub-type when category is shoes (slide vs birkenstock vs heel…)."""
    w, h = pil_img.size
    crop = pil_img.crop((0, int(h * 0.35), w, h)) if h > w * 0.85 else pil_img
    image_features = _encode_image(crop)
    best_style = "slide_sandal"
    best_score = -1.0
    scores = {}
    for style, text_feats in _shoe_style_text_features.items():
        sims = (image_features @ text_feats.T).squeeze(0)
        score = float(sims.max())
        scores[style] = score
        if score > best_score:
            best_score = score
            best_style = style
    rounded = {k: round(v, 3) for k, v in sorted(scores.items(), key=lambda x: -x[1])}
    print(f"👡 Shoe style: {best_style}  scores: {rounded}")
    return best_style


SHOE_STYLE_COLOR_PHRASES = {
    "slide_sandal": {
        "beige":  "beige nude pink cross strap slide sandals with toe loop cork footbed",
        "pink":   "dusty pink nude cross strap slide sandals with toe loop cork footbed",
        "white":  "white leather cross strap slide sandals with cork footbed",
        "brown":  "tan brown leather cross strap slide sandals with cork footbed",
        "black":  "black leather cross strap slide sandals with cork footbed",
        "grey":   "grey suede cross strap slide sandals with cork footbed",
    },
    "birkenstock": {
        "beige":  "beige double buckle birkenstock cork sandals",
        "brown":  "brown leather birkenstock two strap sandals",
        "black":  "black birkenstock cork sandals with double buckles",
    },
    "heeled_sandal": {
        "beige":  "beige kitten heel strappy dress sandals",
        "black":  "black high heel strappy dress sandals",
        "pink":   "pink heeled strappy sandals with ankle strap",
    },
    "espadrille": {
        "beige":  "beige closed toe espadrille flats with jute sole",
        "white":  "white canvas espadrille loafers with rope sole",
    },
    "puffy_slide": {
        "pink":   "pink puffy quilted cross strap pillow slide sandals",
        "beige":  "beige puffy padded cross strap slide sandals",
        "black":  "black puffy quilted slide sandals",
    },
}


def get_shoe_style_color_vector(color: str, shoe_style: str) -> list:
    style_phrases = SHOE_STYLE_COLOR_PHRASES.get(shoe_style, {})
    phrase = style_phrases.get(color)
    if not phrase:
        color_display = color.replace("_", " ")
        style_display = shoe_style.replace("_", " ")
        phrase = f"{color_display} {style_display} sandals on feet"
    feats = _encode_texts([phrase])
    return feats.cpu().numpy().flatten().tolist()


def get_shoe_style_contrast_vector(shoe_style: str) -> list:
    """Centroid of wrong shoe styles — birkenstocks/heels score high, slides score low."""
    phrases = []
    for style, texts in CLIP_SHOE_STYLE_PROMPTS.items():
        if style != shoe_style:
            phrases.extend(texts[:2])
    if not phrases:
        return []
    feats = _encode_texts(phrases)
    centroid = feats.mean(dim=0, keepdim=True)
    centroid = centroid / centroid.norm(dim=-1, keepdim=True)
    return centroid.cpu().numpy().flatten().tolist()


def get_category_group(yolo_label):
    return CATEGORY_MAPPING.get(yolo_label.lower(), "other")

def get_color_vector(color: str) -> list:
    """Return the pre-computed fashion-clip text centroid for a colour as a plain list."""
    feats = _color_text_features.get(color)
    if feats is None:
        return []
    return feats.cpu().numpy().flatten().tolist()


def detect_fabric_clip(pil_img) -> str:
    """Zero-shot fabric detection → denim | jersey | knit | woven | linen | leather | other."""
    image_features = _encode_image(pil_img)
    best_fab = "other"
    best_score = -1.0
    for fab, text_feats in _fabric_text_features.items():
        sims = (image_features @ text_feats.T).squeeze(0)
        score = float(sims.max())
        if score > best_score:
            best_score = score
            best_fab = fab
    print(f"🧵 Fabric: {best_fab} (score={best_score:.4f})")
    return best_fab


def get_fabric_color_vector(color: str, fabric: str, category_group: str = "other") -> list:
    """Combined color+fabric text embedding for more specific re-ranking.

    Uses category-specific templates so a lavender shirt does not get
    "lavender linen trousers" and return pants in the results.
    """
    # Highly specific phrases for pastel tops — plain white tees score lower
    # on these than on generic "lavender linen shirt top".
    TOP_SPECIFIC_COLOR_PHRASES = {
        "lavender":   "soft dusty lavender mauve purple cotton v-neck t-shirt",
        "purple":     "rich purple violet cotton t-shirt top",
        "pink":       "soft dusty pink rose cotton t-shirt top",
        "light_blue": "light sky blue cotton t-shirt top",
    }
    SHOE_COLOR_PHRASES = {
        "beige":      "beige nude pink cross strap slide sandals with cork footbed",
        "pink":       "dusty pink nude slide sandals with toe loop strap",
        "white":      "white leather flat slide sandals",
        "brown":      "tan brown leather flat sandals",
        "black":      "black leather flat slide sandals",
        "grey":       "grey suede flat slide sandals",
    }
    if category_group == "shoes" and color in SHOE_COLOR_PHRASES:
        feats = _encode_texts([SHOE_COLOR_PHRASES[color]])
        return feats.cpu().numpy().flatten().tolist()
    if category_group == "top" and color in TOP_SPECIFIC_COLOR_PHRASES:
        feats = _encode_texts([TOP_SPECIFIC_COLOR_PHRASES[color]])
        return feats.cpu().numpy().flatten().tolist()

    if category_group == "top":
        templates = FABRIC_COLOR_TEMPLATES_TOP
        fallback = f"{color.replace('_', ' ')} cotton t-shirt top"
    elif category_group == "bottom":
        templates = FABRIC_COLOR_TEMPLATES_BOTTOM
        fallback = f"{color.replace('_', ' ')} pants bottom"
    else:
        templates = FABRIC_COLOR_TEMPLATES_BOTTOM
        fallback = color.replace("_", " ")

    template = templates.get(fabric)
    if template and color in COLOR_TEXT_PROMPTS:
        color_display = color.replace("_", " ")
        phrase = template.format(color=color_display)
    else:
        phrase = fallback

    feats = _encode_texts([phrase])
    return feats.cpu().numpy().flatten().tolist()


# For these colours, send reference vectors so the backend can drop near-misses
# (white tees, grey/beige neutrals scoring too high on pastel queries).
COLORS_CONTRAST_WITH_WHITE = {
    "lavender", "purple", "pink", "light_blue", "yellow", "green", "red", "burgundy",
}
PASTEL_COLORS = {"lavender", "purple", "pink"}


def get_extra_contrast_vectors(color: str) -> dict:
    extras = {}
    if color in COLORS_CONTRAST_WITH_WHITE:
        white = get_color_vector("white")
        if white:
            extras["contrastColorVector"] = white
    if color in PASTEL_COLORS:
        grey = get_color_vector("grey")
        beige = get_color_vector("beige")
        blue = get_color_vector("light_blue")
        if grey:
            extras["greyContrastVector"] = grey
        if beige:
            extras["beigeContrastVector"] = beige
        if blue:
            extras["blueContrastVector"] = blue
    return extras


def get_top_contrast_vector() -> list:
    """Centroid of top/tank prompts — tanks score high, dresses score low."""
    phrases = [
        "a sleeveless black tank top or camisole on a model",
        "a cropped fitted top worn by a woman",
        "a black halter neck crop top",
        "a ribbed sleeveless tank top",
    ]
    feats = _encode_texts(phrases)
    centroid = feats.mean(dim=0, keepdim=True)
    centroid = centroid / centroid.norm(dim=-1, keepdim=True)
    return centroid.cpu().numpy().flatten().tolist()


DRESS_COLOR_PHRASES = {
    "black":      "black sleeveless maxi column dress reaching the ankles",
    "white":      "white maxi slip dress full length on a model",
    "beige":      "beige linen maxi dress full length",
    "red":        "red midi dress with a flowing skirt",
    "burgundy":   "burgundy maxi dress full length",
    "pink":       "pink maxi slip dress full length",
    "light_blue": "light blue maxi sundress full length",
    "lavender":   "lavender maxi dress full length",
}


def get_dress_wrong_garment_contrast_vector() -> list:
    """Tops and skirts score high; one-piece dresses score low."""
    phrases = [
        "a sleeveless black tank top or camisole on a model",
        "a cropped fitted top worn by a woman",
        "a long black maxi skirt worn with a separate top",
        "a pleated midi skirt on a model",
        "a denim tiered maxi skirt with t-shirt",
        "a beige linen maxi skirt with drawstring waist",
    ]
    feats = _encode_texts(phrases)
    centroid = feats.mean(dim=0, keepdim=True)
    centroid = centroid / centroid.norm(dim=-1, keepdim=True)
    return centroid.cpu().numpy().flatten().tolist()

def get_dress_color_vector(color: str) -> list:
    phrase = DRESS_COLOR_PHRASES.get(
        color,
        f"{color.replace('_', ' ')} maxi dress full length on a model",
    )
    feats = _encode_texts([phrase])
    return feats.cpu().numpy().flatten().tolist()


def _attach_color_vectors(item: dict, color: str, is_stripe: bool, category_group: str, fabric: str, shoe_style: str = None) -> dict:
    # Shoes: style-specific vectors take priority over pattern mis-detection
    if category_group == "shoes" and shoe_style:
        item["shoeStyle"] = shoe_style
        shoe_color = color if color != "pattern" else "beige"
        item["colorVector"] = get_shoe_style_color_vector(shoe_color, shoe_style)
        style_contrast = get_shoe_style_contrast_vector(shoe_style)
        if style_contrast:
            item["styleContrastVector"] = style_contrast
        for key, vec in get_extra_contrast_vectors(shoe_color).items():
            if vec:
                item[key] = vec
        return item

    if category_group == "dress" and color != "pattern":
        item["colorVector"] = get_dress_color_vector(color)
        top_contrast = get_top_contrast_vector()
        if top_contrast:
            item["contrastColorVector"] = top_contrast
        for key, vec in get_extra_contrast_vectors(color).items():
            if vec:
                item[key] = vec
        return item

    if color == "pattern":
        item["colorVector"] = get_pattern_color_vector(category_group)
        solid = get_solid_contrast_vector(category_group)
        if solid:
            item["contrastColorVector"] = solid
        return item

    item["colorVector"] = (
        get_stripe_color_vector(color) if is_stripe
        else get_fabric_color_vector(color, fabric, category_group)
    )
    for key, vec in get_extra_contrast_vectors(color).items():
        if vec:
            item[key] = vec
    return item


# Stripe-specific re-ranking phrases.
# for striped items and LOW for plain navy or plain white — preventing the
# MIN_SURVIVORS safety net from pulling in solid-colour shirts.
STRIPE_COLOR_PHRASES = {
    "navy":       "navy blue and white horizontal striped top",
    "black":      "black and white striped top",
    "red":        "red and white striped top",
    "burgundy":   "burgundy and white striped top",
    "olive":      "olive green and white striped top",
    "green":      "green and white striped top",
    "grey":       "grey and white striped top",
    "beige":      "beige and cream striped top",
    "brown":      "brown and cream striped top",
    "light_blue": "light blue and white striped top",
    "lavender":   "lavender and white striped top",
    "purple":     "purple and white striped top",
}


def get_stripe_color_vector(color: str) -> list:
    """Text embedding for a stripe-specific phrase.

    More discriminative than the plain color vector: striped products
    score high, solid products of the same color score significantly lower.
    """
    phrase = STRIPE_COLOR_PHRASES.get(
        color,
        f"{color.replace('_', ' ')} and white striped garment"
    )
    feats = _encode_texts([phrase])
    return feats.cpu().numpy().flatten().tolist()


PATTERN_COLOR_PHRASES = {
    "bottom": "floral print boho shorts with colorful pattern and drawstring waist",
    "top":    "floral print patterned blouse with colorful motifs",
    "skirt":  "floral print midi skirt with colorful pattern",
    "dress":  "floral print dress with colorful pattern",
}

SOLID_GARMENT_PHRASES = {
    "bottom": "plain solid beige cotton shorts with no print",
    "top":    "plain solid cotton t-shirt with no print",
    "skirt":  "plain solid skirt with no print",
    "dress":  "plain solid dress with no print",
}


def get_pattern_color_vector(category_group: str) -> list:
    phrase = PATTERN_COLOR_PHRASES.get(
        category_group,
        "colorful floral print garment with pattern"
    )
    feats = _encode_texts([phrase])
    return feats.cpu().numpy().flatten().tolist()


def get_solid_contrast_vector(category_group: str) -> list:
    phrase = SOLID_GARMENT_PHRASES.get(
        category_group,
        "plain solid garment with no print"
    )
    feats = _encode_texts([phrase])
    return feats.cpu().numpy().flatten().tolist()


def process_image_logic(img):
    results = yolo_model(img)
    found_items = []

    if len(results[0].boxes) == 0:
        image_features = _encode_image(img)
        embedding = image_features.cpu().numpy().flatten().tolist()

        category_group = detect_category_clip(img)
        fabric = detect_fabric_clip(img)
        color, is_stripe = get_fashion_color(img, category_group)
        shoe_style = detect_shoe_style_clip(img) if category_group == "shoes" else None
        found_items.append(_attach_color_vectors({
            "category": "other",
            "categoryGroup": category_group,
            "fabricGroup": fabric,
            "confidence": 1.0,
            "embedding": embedding,
            "color": color,
        }, color, is_stripe, category_group, fabric, shoe_style))
    else:
        for r in results:
            for box in r.boxes:
                label = yolo_model.names[int(box.cls)]
                conf = float(box.conf)
                if conf > 0.2:
                    coords = box.xyxy[0].tolist()
                    crop_img = img.crop((coords[0], coords[1], coords[2], coords[3]))

                    image_features = _encode_image(crop_img)
                    embedding = image_features.cpu().numpy().flatten().tolist()

                    category_group = detect_category_clip(crop_img)
                    fabric = detect_fabric_clip(crop_img)
                    color, is_stripe = get_fashion_color(crop_img, category_group)
                    shoe_style = detect_shoe_style_clip(crop_img) if category_group == "shoes" else None
                    found_items.append(_attach_color_vectors({
                        "category": get_category_group(label),
                        "categoryGroup": category_group,
                        "fabricGroup": fabric,
                        "confidence": conf,
                        "embedding": embedding,
                        "color": color,
                    }, color, is_stripe, category_group, fabric, shoe_style))
    return found_items

class URLRequest(BaseModel):
    image_url: str

@app.post("/process-url")
async def process_url(data: URLRequest):
    """Used by enrichProducts.js. Skips YOLO — product images are already clean."""
    try:
        response = requests.get(data.image_url, headers=HEADERS, timeout=15)
        if response.status_code == 403:
            raise HTTPException(status_code=403, detail="The website blocked the image request (403 Forbidden)")
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content)).convert("RGB")

        image_features = _encode_image(img)
        embedding = image_features.cpu().numpy().flatten().tolist()

        color, _ = get_fashion_color(img)
        category_group = detect_category_clip(img)

        return {"items": [{
            "embedding": embedding,
            "color": color,
            "categoryGroup": category_group,
            "colorVector": get_color_vector(color),
            "confidence": 1.0,
            "category": "product",
        }]}
    except Exception as e:
        print(f"Error processing URL {data.image_url}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract-color")
async def extract_color_from_url(data: URLRequest):
    try:
        response = requests.get(data.image_url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content)).convert("RGB")
        detected_color, _ = get_fashion_color(img)
        return {"color": detected_color}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
#pip install fastapi uvicorn python-multipart transformers accelerate
#python main.py
