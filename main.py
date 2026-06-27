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
        "white pointed toe kitten heel ankle boots on feet",
        "cream leather heeled booties worn with trousers",
        "flat leather oxford shoes from the side",
        "open-toe strappy sandals on a white background",
        "nude pink cross strap slide sandals with cork footbed",
        "beige flat slide sandals with toe loop on feet",
        "open toe leather flat sandals worn on feet",
        "cork footbed slide sandals with crossed straps",
    ],
    "belt": [
        "black leather belt with silver buckle on jeans waist",
        "brown leather waist belt with metal buckle",
        "thin black leather belt worn through belt loops",
        "classic leather belt buckle close up on denim jeans",
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
    "heeled_boot": [
        "white pointed toe kitten heel ankle boots on feet",
        "cream leather heeled booties with small tapered heel",
        "black ankle boots with a slim stiletto heel on feet",
        "black sock boots with stretch knit shaft and block heel on feet",
        "black pointed toe ankle booties with medium block heel",
    ],
    "flat_shoe": [
        "beige suede ballet flats with round toe on feet",
        "black leather ballerina flat shoes on feet",
        "white canvas sneakers on feet",
    ],
}

CLIP_TOP_STYLE_PROMPTS = {
    "strapless": [
        "black strapless tube top bandeau with bare shoulders no straps",
        "strapless corset-style tube top cropped at the waist",
        "sleeveless strapless bandeau top showing bare shoulders",
        "black strapless top with stomach cutout bare shoulders",
        "strapless bandeau tube top no shoulder straps on a model",
    ],
    "tank": [
        "black sleeveless tank top with thin shoulder straps",
        "ribbed cotton camisole with spaghetti shoulder straps",
        "scoop neck tank top with visible shoulder straps",
    ],
    "halter": [
        "black halter neck top with straps tied behind the neck",
        "halter neck ribbed crop top with neck straps",
        "high neck halter top with straps around the neck",
    ],
}

CLIP_BOTTOM_LENGTH_PROMPTS = {
    "shorts": [
        "beige linen shorts above the knee mid-thigh length on a model",
        "tailored chino shorts showing bare legs above the knee",
        "casual cotton shorts mid-thigh length with structured waistband",
        "linen shorts above the knee with visible leg above hem",
    ],
    "long_pants": [
        "wide-leg linen trousers full length to the ankle on a model",
        "tailored suit pants long trousers reaching the floor",
        "flowy linen pants full length covering the legs to the ankle",
        "formal dress trousers full length on a model",
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

_top_style_text_features: dict = {}
for _style, _texts in CLIP_TOP_STYLE_PROMPTS.items():
    _top_style_text_features[_style] = _encode_texts(_texts)

_bottom_length_text_features: dict = {}
for _length, _texts in CLIP_BOTTOM_LENGTH_PROMPTS.items():
    _bottom_length_text_features[_length] = _encode_texts(_texts)

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
    "tan": (158, 135, 108),
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

    # ── Step 3.3: black shoes on pavement (avoid grey misread) ────────────────
    if category_group == "shoes" and avg_brightness < 110 and max(r, g, b) - min(r, g, b) < 35:
        print(f"🎨 Detected color: black (shoe-dark rule)  avg_rgb=({r:.0f},{g:.0f},{b:.0f})")
        return "black", False

    # ── Step 3.4: nude / dusty pink sandals ───────────────────────────────────
    if category_group == "shoes":
        avg_saturation = max(r, g, b) - min(r, g, b)
        if r > g > b and (r - b) > 12 and 130 <= avg_brightness <= 240 and avg_saturation < 90:
            if (r - g) < 40 and (g - b) > 8:
                print(f"🎨 Detected color: pink (nude-sandal rule)  avg_rgb=({r:.0f},{g:.0f},{b:.0f})")
                return "pink", False
            print(f"🎨 Detected color: beige (nude-sandal rule)  avg_rgb=({r:.0f},{g:.0f},{b:.0f})")
            return "beige", False

    # ── Step 3.5: warm tan / khaki / beige linen ─────────────────────────────
    # Natural-light neutrals have low saturation and r > g > b — otherwise
    # misclassified as grey (e.g. avg_rgb=(125,110,101) pleated linen shorts).
    avg_saturation = max(r, g, b) - min(r, g, b)
    if (
        r >= g >= b
        and (r - b) >= 10
        and 90 <= avg_brightness <= 210
        and avg_saturation < 55
    ):
        print(f"🎨 Detected color: beige (warm-neutral rule)  avg_rgb=({r:.0f},{g:.0f},{b:.0f})")
        return "beige", False

    # ── Step 3.6: low-saturation grey detection ──────────────────────────────
    # Garments with near-zero colour saturation are grey/charcoal regardless of
    # which palette point wins in Euclidean distance. Without this rule,
    # dark grey (83,84,88) ends up classified as olive because the grey palette
    # reference (128,128,128) is much brighter — the distance to olive (85,107,47)
    # is accidentally smaller in RGB space even though olive has 60-point saturation.
    # Warm tan/beige is handled above — only cool neutrals fall through to grey here.
    if avg_saturation < 20 and not (r >= g >= b and (r - b) >= 8):
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
    if closest_color in ("grey", "tan") and (r - b) > 15:
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



def _shorts_with_belt_override(pil_img) -> bool:
    """Shorts/trousers worn with a belt — lower body dominates, not a belt-only search."""
    w, h = pil_img.size
    if h <= w * 0.85:
        return False
    lower = pil_img.crop((0, int(h * 0.32), w, h))
    lower_scores = _score_categories(lower)
    if lower_scores.get("bottom", 0) < 0.26:
        return False
    length = detect_bottom_length_clip(pil_img)
    return length == "shorts"


def _feet_shoes_override(pil_img) -> tuple[bool, dict]:
    """Foot/sandal crops — lower zone should win over bottom/trouser misreads."""
    w, h = pil_img.size
    feet = pil_img.crop((int(w * 0.05), int(h * 0.38), int(w * 0.95), h))
    feet_scores = _score_categories(feet)
    shoes = feet_scores.get("shoes", 0)
    bottom = feet_scores.get("bottom", 0)
    top = feet_scores.get("top", 0)
    rival = max(bottom, top, feet_scores.get("dress", 0), feet_scores.get("skirt", 0))
    if shoes >= rival + 0.008 and shoes >= 0.22:
        return True, feet_scores
    return False, feet_scores


def detect_category_clip(pil_img) -> str:
    """Zero-shot fashion-clip classification → top, bottom, skirt, dress, shoes, belt."""
    w, h = pil_img.size

    belt_bands = [
        pil_img.crop((int(w * 0.05), int(h * 0.34), int(w * 0.95), int(h * 0.58))),
        pil_img.crop((int(w * 0.05), int(h * 0.22), int(w * 0.95), int(h * 0.78))),
    ]
    if w >= h * 0.85:
        belt_bands.append(pil_img)

    belt_score = 0.0
    bottom_score = 0.0
    top_score = 0.0
    waist_scores = {}
    for band in belt_bands:
        band_scores = _score_categories(band)
        belt_score = max(belt_score, band_scores.get("belt", 0))
        bottom_score = max(bottom_score, band_scores.get("bottom", 0))
        top_score = max(top_score, band_scores.get("top", 0))
        if not waist_scores or band_scores.get("belt", 0) >= waist_scores.get("belt", 0):
            waist_scores = band_scores

    if belt_score >= bottom_score + 0.005 and belt_score >= top_score - 0.005:
        if _shorts_with_belt_override(pil_img):
            print("📐 Belt skipped — shorts-with-belt override (garment is bottom/shorts)")
        else:
            rounded = {k: round(v, 4) for k, v in sorted(waist_scores.items(), key=lambda x: -x[1])}
            print(f"📐 Category scores (waist-belt): {rounded}  → winner: belt")
            return "belt"

    feet_override, feet_scores = _feet_shoes_override(pil_img)
    if feet_override:
        rounded = {k: round(v, 4) for k, v in sorted(feet_scores.items(), key=lambda x: -x[1])}
        print(f"📐 Category scores (feet-override): {rounded}  → winner: shoes")
        return "shoes"

    if h > w * 0.85:
        # Portrait: score full body + zones. Upper half of a maxi dress looks like a tank
        # top — always check full-image dress score before trusting upper-body "top".
        full_scores = _score_categories(pil_img)
        upper_scores = _score_categories(pil_img.crop((0, 0, w, int(h * 0.55))))
        lower_scores = _score_categories(pil_img.crop((0, int(h * 0.42), w, h)))

        lower_best = max(lower_scores, key=lower_scores.get)
        upper_best = max(upper_scores, key=upper_scores.get)

        if lower_best == "shoes" and lower_scores["shoes"] >= lower_scores.get("bottom", 0) + 0.01:
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

    if best_cat == "belt" and _shorts_with_belt_override(pil_img):
        best_cat = "bottom"
        region = "shorts-over-belt"
    rounded = {k: round(v, 4) for k, v in all_scores.items()}
    scores_str = "  ".join(f"{c}={s}" for c, s in sorted(rounded.items(), key=lambda x: -x[1]))
    print(f"📐 Category scores ({region}): {scores_str}  → winner: {best_cat}")
    return best_cat


def detect_shoe_style_clip(pil_img) -> str:
    """Classify shoe sub-type when category is shoes (slide vs birkenstock vs heel.)."""
    w, h = pil_img.size
    cy0 = int(h * 0.30) if h > w * 0.6 else int(h * 0.10)
    crops = [pil_img.crop((int(w * 0.05), cy0, int(w * 0.95), h))]
    if h > w * 0.55:
        crops.append(pil_img.crop((0, int(h * 0.12), w, h)))

    scores = {}
    for style, text_feats in _shoe_style_text_features.items():
        style_best = -1.0
        for crop in crops:
            sims = (_encode_image(crop) @ text_feats.T).squeeze(0)
            style_best = max(style_best, float(sims.max()))
        scores[style] = style_best

    open_shoe_styles = ("espadrille", "flat_shoe", "slide_sandal", "birkenstock", "puffy_slide", "heeled_sandal")
    boot_score = scores.get("heeled_boot", 0.0)
    slide_score = max(scores.get("slide_sandal", 0.0), scores.get("puffy_slide", 0.0))
    open_best = max(scores.get(s, 0.0) for s in open_shoe_styles)

    if boot_score >= slide_score - 0.035 or boot_score >= open_best - 0.035:
        best_style = "heeled_boot"
    else:
        best_style = max(scores, key=scores.get)

    rounded = {k: round(v, 3) for k, v in sorted(scores.items(), key=lambda x: -x[1])}
    print(f"?? Shoe style: {best_style}  scores: {rounded}")
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
    "heeled_boot": {
        "white":  "white pointed toe kitten heel ankle boots on feet",
        "black":  "black sock boots with pointed toe and block heel ankle booties on feet",
        "beige":  "beige cream leather heeled ankle boots with small heel",
        "brown":  "brown leather heeled ankle booties on feet",
        "grey":   "grey suede heeled ankle boots on feet",
    },
    "flat_shoe": {
        "white":  "white leather ballet flat shoes on feet",
        "black":  "black ballerina flat shoes on feet",
        "beige":  "beige suede ballet flats on feet",
    },
}


def get_shoe_style_color_vector(color: str, shoe_style: str) -> list:
    style_phrases = SHOE_STYLE_COLOR_PHRASES.get(shoe_style, {})
    phrase = style_phrases.get(color)
    if not phrase:
        color_display = color.replace("_", " ")
        style_display = shoe_style.replace("_", " ")
        if shoe_style == "heeled_boot":
            phrase = f"{color_display} pointed toe heeled ankle boots with block heel on feet"
        elif shoe_style == "flat_shoe":
            phrase = f"{color_display} ballet flat shoes on feet"
        else:
            phrase = f"{color_display} {style_display} sandals on feet"
    feats = _encode_texts([phrase])
    return feats.cpu().numpy().flatten().tolist()


def get_shoe_style_contrast_vector(shoe_style: str) -> list:
    """Centroid of wrong shoe styles — birkenstocks/heels score high, slides score low."""
    phrases = []
    for style, texts in CLIP_SHOE_STYLE_PROMPTS.items():
        if style != shoe_style:
            phrases.extend(texts[:2])
    if shoe_style in ("slide_sandal", "puffy_slide"):
        phrases.extend([
            "white leather sneakers on feet",
            "black ankle boots with chunky sole on feet",
            "high heel stiletto pumps on feet",
            "leopard print double buckle birkenstock sandals on feet",
        ])
    elif shoe_style == "birkenstock":
        phrases.extend([
            "flat cross strap slide sandals with toe loop on feet",
            "high heel strappy dress sandals with ankle strap on feet",
        ])
    elif shoe_style == "heeled_sandal":
        phrases.extend([
            "flat cross strap slide sandals with cork footbed on feet",
            "white leather sneakers on feet",
        ])
    elif shoe_style == "heeled_boot":
        phrases.extend([
            "beige suede ballet flats with round toe on feet",
            "white canvas sneakers on feet",
            "closed toe espadrille flat shoes with woven jute rope sole",
            "flat cross strap slide sandals with cork footbed on feet",
            "nude pink cross strap slide sandals on feet",
            "black leather ballerina flat shoes on feet",
            "denim platform slide sandals on feet",
        ])
    elif shoe_style == "flat_shoe":
        phrases.extend([
            "white pointed toe kitten heel ankle boots on feet",
            "black ankle boots with a slim stiletto heel on feet",
        ])
    if not phrases:
        return []
    feats = _encode_texts(phrases)
    centroid = feats.mean(dim=0, keepdim=True)
    centroid = centroid / centroid.norm(dim=-1, keepdim=True)
    return centroid.cpu().numpy().flatten().tolist()



TOP_STYLE_COLOR_PHRASES = {
    "strapless": {
        "black": "black strapless tube top bandeau with bare shoulders",
        "white": "white strapless bandeau tube top",
        "beige": "beige strapless tube top bandeau",
        "pink":  "pink strapless bandeau tube top",
        "red":   "red strapless tube top",
    },
    "tank": {
        "black": "black sleeveless tank top with shoulder straps",
        "white": "white ribbed tank top with thin straps",
        "beige": "beige cotton tank top with straps",
    },
    "halter": {
        "black": "black halter neck top with straps around the neck",
        "white": "white halter neck crop top",
        "beige": "beige halter neck top",
    },
}


def detect_top_style_clip(pil_img) -> str | None:
    """Classify top sub-type: strapless vs tank vs halter."""
    w, h = pil_img.size
    crop = pil_img.crop((0, 0, w, int(h * 0.72))) if h > w * 0.85 else pil_img
    shoulder = pil_img.crop((0, 0, w, int(h * 0.38)))
    scores = {}
    for style, text_feats in _top_style_text_features.items():
        body_sims = (_encode_image(crop) @ text_feats.T).squeeze(0)
        shoulder_sims = (_encode_image(shoulder) @ text_feats.T).squeeze(0)
        scores[style] = max(float(body_sims.max()), float(shoulder_sims.max()))
    best_style = max(scores, key=scores.get)
    best_score = scores[best_style]
    ordered = sorted(scores.values(), reverse=True)
    margin = (ordered[0] - ordered[1]) if len(ordered) > 1 else ordered[0]
    strapless_score = scores.get("strapless", 0)
    # Strapless tube/bandeau crops often score close to tank — still re-rank when strapless leads.
    if strapless_score >= best_score - 0.004 and strapless_score >= scores.get("halter", 0):
        best_style = "strapless"
        best_score = strapless_score
    strapless_score = scores.get("strapless", 0)
    tank_score = scores.get("tank", 0)
    halter_score = scores.get("halter", 0)
    if (
        strapless_score >= tank_score - 0.006
        and strapless_score >= halter_score - 0.020
        and (best_style in ("tank", "strapless") or strapless_score >= best_score - 0.010)
    ):
        best_style = "strapless"
        best_score = strapless_score
    rounded = {k: round(v, 3) for k, v in sorted(scores.items(), key=lambda x: -x[1])}
    if best_score < 0.12:
        print(f"Top style: none  scores: {rounded}")
        return None
    print(f"Top style: {best_style}  scores: {rounded}")
    return best_style


def get_top_style_color_vector(color: str, top_style: str) -> list:
    style_phrases = TOP_STYLE_COLOR_PHRASES.get(top_style, {})
    phrase = style_phrases.get(color)
    if not phrase:
        phrase = f"{color.replace('_', ' ')} {top_style.replace('_', ' ')} top"
    feats = _encode_texts([phrase])
    return feats.cpu().numpy().flatten().tolist()


def get_top_style_contrast_vector(top_style: str) -> list:
    phrases = []
    for style, texts in CLIP_TOP_STYLE_PROMPTS.items():
        if style != top_style:
            phrases.extend(texts[:2])
    if not phrases:
        return []
    feats = _encode_texts(phrases)
    centroid = feats.mean(dim=0, keepdim=True)
    centroid = centroid / centroid.norm(dim=-1, keepdim=True)
    return centroid.cpu().numpy().flatten().tolist()




def detect_bottom_length_clip(pil_img) -> str:
    """Classify bottom length when category is bottom (shorts vs long pants)."""
    image_features = _encode_image(pil_img)
    best_length = "long_pants"
    best_score = -1.0
    scores = {}
    for length, text_feats in _bottom_length_text_features.items():
        sims = (image_features @ text_feats.T).squeeze(0)
        score = float(sims.max())
        scores[length] = score
        if score > best_score:
            best_score = score
            best_length = length
    rounded = {k: round(v, 3) for k, v in sorted(scores.items(), key=lambda x: -x[1])}
    print(f"Shorts/pants length: {best_length}  scores: {rounded}")
    return best_length


BOTTOM_LENGTH_FABRIC_PHRASES = {
    "shorts": {
        "linen":  "{color} pleated linen shorts above the knee with structured waistband",
        "denim":  "{color} denim shorts mid-thigh length",
        "woven":  "{color} pleated tailored shorts above the knee",
        "jersey": "{color} cotton jersey shorts mid-thigh",
        "knit":   "{color} knit shorts above the knee",
    },
    "long_pants": {
        "linen":  "{color} linen trousers full length to the ankle",
        "denim":  "{color} denim jeans full length",
        "woven":  "{color} tailored dress trousers full length",
        "jersey": "{color} jersey jogger pants full length",
        "knit":   "{color} knit trousers full length",
    },
}


def get_bottom_length_color_vector(color: str, fabric: str, bottom_length: str) -> list:
    fabric_phrases = BOTTOM_LENGTH_FABRIC_PHRASES.get(bottom_length, {})
    template = fabric_phrases.get(fabric)
    color_display = color.replace("_", " ")
    if template:
        phrase = template.format(color=color_display)
    else:
        length_display = "shorts above the knee" if bottom_length == "shorts" else "trousers full length"
        phrase = f"{color_display} {fabric} {length_display}"
    feats = _encode_texts([phrase])
    return feats.cpu().numpy().flatten().tolist()


def get_bottom_length_contrast_vector(bottom_length: str) -> list:
    """Wrong-length centroid — trousers score high on shorts queries and vice versa."""
    wrong = "long_pants" if bottom_length == "shorts" else "shorts"
    phrases = list(CLIP_BOTTOM_LENGTH_PROMPTS.get(wrong, []))
    if bottom_length == "shorts":
        phrases.extend([
            "a plain crew-neck t-shirt on a white background",
            "a fitted blouse with front buttons",
            "a sleeveless tank top or camisole",
            "a ribbed sleeveless crop top worn by a woman",
        ])
    if not phrases:
        return []
    feats = _encode_texts(phrases)
    centroid = feats.mean(dim=0, keepdim=True)
    centroid = centroid / centroid.norm(dim=-1, keepdim=True)
    return centroid.cpu().numpy().flatten().tolist()



def get_belt_color_vector(color: str) -> list:
    phrases = {
        "black": "black leather belt with silver buckle on jeans",
        "brown": "brown leather belt with brass buckle",
        "beige": "beige tan leather belt with metal buckle",
        "burgundy": "burgundy leather belt with gold buckle",
        "white": "white leather belt with silver buckle",
    }
    phrase = phrases.get(color, f"{color.replace('_', ' ')} leather belt with metal buckle")
    feats = _encode_texts([phrase])
    return feats.cpu().numpy().flatten().tolist()


def get_belt_contrast_vector() -> list:
    phrases = [
        "blue denim jeans full length on a model",
        "black high waist denim jeans",
        "wide-leg linen trousers full length",
        "tailored suit pants long trousers",
    ]
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
WARM_NEUTRAL_COLORS = {"beige", "tan", "brown", "olive"}


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
    if color in WARM_NEUTRAL_COLORS:
        grey = get_color_vector("grey")
        navy = get_color_vector("navy")
        if grey:
            extras["greyContrastVector"] = grey
        if navy:
            extras["contrastColorVector"] = navy
    if color == "black":
        brown = get_color_vector("brown")
        if brown:
            extras["beigeContrastVector"] = brown
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


def _attach_color_vectors(item: dict, color: str, is_stripe: bool, category_group: str, fabric: str, shoe_style: str = None, bottom_length: str = None, top_style: str = None) -> dict:
    if category_group == "bottom" and bottom_length:
        item["bottomLength"] = bottom_length
        if color == "pattern":
            item["colorVector"] = get_pattern_color_vector(category_group)
            solid = get_solid_contrast_vector(category_group)
            if solid:
                item["contrastColorVector"] = solid
        else:
            item["colorVector"] = get_bottom_length_color_vector(color, fabric, bottom_length)
            for key, vec in get_extra_contrast_vectors(color).items():
                if vec:
                    item[key] = vec
        length_contrast = get_bottom_length_contrast_vector(bottom_length)
        if length_contrast:
            item["styleContrastVector"] = length_contrast
        return item
    if category_group == "belt" and color != "pattern":
        item["colorVector"] = get_belt_color_vector(color)
        contrast = get_belt_contrast_vector()
        if contrast:
            item["styleContrastVector"] = contrast
        return item

    if category_group == "top" and top_style and color != "pattern":
        item["topStyle"] = top_style
        item["colorVector"] = get_top_style_color_vector(color, top_style)
        style_contrast = get_top_style_contrast_vector(top_style)
        if style_contrast:
            item["styleContrastVector"] = style_contrast
        for key, vec in get_extra_contrast_vectors(color).items():
            if vec:
                item[key] = vec
        return item

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
        top_style = detect_top_style_clip(img) if category_group == "top" else None
        bottom_length = detect_bottom_length_clip(img) if category_group == "bottom" else None
        found_items.append(_attach_color_vectors({
            "category": "other",
            "categoryGroup": category_group,
            "fabricGroup": fabric,
            "confidence": 1.0,
            "embedding": embedding,
            "color": color,
        }, color, is_stripe, category_group, fabric, shoe_style, bottom_length, top_style))
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
                    top_style = detect_top_style_clip(crop_img) if category_group == "top" else None
                    bottom_length = detect_bottom_length_clip(crop_img) if category_group == "bottom" else None
                    found_items.append(_attach_color_vectors({
                        "category": get_category_group(label),
                        "categoryGroup": category_group,
                        "fabricGroup": fabric,
                        "confidence": conf,
                        "embedding": embedding,
                        "color": color,
                    }, color, is_stripe, category_group, fabric, shoe_style, bottom_length, top_style))
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
