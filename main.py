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
from typing import List, Optional

# Garment slots returned by Find Total Look (display order)
TOTAL_LOOK_SLOT_ORDER = ["dress", "top", "belt", "bottom", "shorts", "skirt", "shoes"]
MAX_TOTAL_LOOK_ITEMS = 4

# COCO classes that are never outfit garments
_IGNORED_YOLO_CLASSES = {
    "person", "backpack", "handbag", "tie", "umbrella", "suitcase",
}

# Accessory classes whose pixels contaminate garment zone colour analysis
_BAG_YOLO_CLASSES = {"handbag", "backpack", "suitcase"}


def _paint_out_boxes(
    img: Image.Image,
    boxes: list,
    fill: tuple = (245, 245, 245),
) -> Image.Image:
    """Return a copy of *img* with each bounding box filled with *fill*.

    The default fill (245, 245, 245) is above the background-masking threshold
    used in get_fashion_color (R>235 & G>235 & B>230), so painted pixels are
    automatically excluded from garment pixel statistics — they vanish as
    "bright background" and never inflate dark_frac or brightness_std.
    """
    if not boxes:
        return img
    arr = np.array(img.convert("RGB"), dtype=np.uint8).copy()
    ih, iw = arr.shape[:2]
    for x0, y0, x1, y1 in boxes:
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(iw, int(x1)), min(ih, int(y1))
        if x1 > x0 and y1 > y0:
            arr[y0:y1, x0:x1] = fill
    return Image.fromarray(arr)

# Portrait zone crops when YOLO misses individual garments (y0/y1 as fraction of height)
# Keep zones non-overlapping — a wide top zone pulls in belt/waist and misreads shirt colour.
_TOTAL_LOOK_ZONES = [
    ("top",    0.05, 0.38, {"top"}),
    ("belt",   0.38, 0.50, {"belt"}),
    ("bottom", 0.44, 0.76, {"bottom", "shorts", "skirt"}),
    ("shoes",  0.72, 0.98, {"shoes"}),
]

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
        "a white sleeveless mock neck crop top on a model",
        "a black shiny leather bomber jacket with gathered sleeves",
        "a dark navy denim sleeveless vest with buttons and waist tie",
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
        "a black high waisted mini skirt with front patch pockets",
        "a black leather mini skirt on a model",
        "a white midi skirt with black polka dots and lace trim",
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
        "tan brown leather H cutout mule slide sandals on feet",
        "beige leather flat slide sandals with open toe on feet",
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
    "flip_flop": [
        "beige tan leather thong flip flop sandals on feet",
        "nude tan toe post flip flop sandals on feet",
        "brown leather flat flip flop sandals on feet",
        "black thong flip flop sandals on feet",
        "black leather flip flop sandal with thin kitten heel on feet",
        "flat black toe post flip flop sandals on feet",
    ],
    "flat_shoe": [
        "black pointed toe ballet flat shoes on feet",
        "black leather ballerina flats closed toe on feet",
        "nude pointed toe flat pumps on feet",
        "beige suede ballet flats with round toe on feet",
        "black leather ballerina flat shoes on feet",
        "white canvas sneakers on feet",
    ],
}

CLIP_TOP_STYLE_PROMPTS = {
    "tshirt": [
        "plain white cotton t-shirt with short sleeves and crew neck no stripes",
        "oversized white cotton t-shirt with short sleeves and crew neck",
        "loose fit white tee shirt with short sleeves tucked into jeans",
        "casual plain crew neck t-shirt with short sleeves solid color",
    ],
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
        "white sleeveless mock neck crop top high neckline",
        "white ribbed sleeveless tank top mock neck",
    ],
    "halter": [
        "black halter neck top with straps tied behind the neck",
        "halter neck ribbed crop top with neck straps",
        "high neck halter top with straps around the neck",
    ],
    "coat": [
        "black shiny leather bomber jacket with gathered sleeves",
        "black leather coat with high collar on a model",
        "black faux leather jacket cropped at the waist",
        "black leather outerwear jacket with voluminous sleeves",
    ],
    "vest": [
        "dark navy denim sleeveless vest with buttons and waist tie on a model",
        "black denim waistcoat vest worn with a long grey skirt",
        "sleeveless denim jacket vest with front button placket",
    ],
    "shirt": [
        "light blue denim button down shirt with long sleeves and chest pockets on a model",
        "medium wash denim shirt with collar and front button placket long sleeves",
        "classic denim shirt jacket with long sleeves and two chest pockets",
        "blue denim long sleeve shirt with buttons and patch pockets",
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

CLIP_SKIRT_LENGTH_PROMPTS = {
    "mini": [
        "high waisted mini skirt above the knee with front patch pockets",
        "short mini skirt mid thigh length on a model",
        "structured mini skirt with belt loops well above the knee",
        "denim mini skirt exposing most of the thigh",
    ],
    "midi": [
        "pleated midi skirt ending below the knee on a model",
        "knee length A-line midi skirt on a model",
        "midi skirt that falls between the knee and ankle",
        "asymmetric hem midi skirt below the knee",
    ],
    "maxi": [
        "long maxi skirt reaching the floor on a model",
        "flowing maxi skirt covering the ankles and feet",
        "full length maxi skirt touching the ground",
        "satin maxi skirt with side slit floor length",
        "a skirt that goes all the way to the ankles",
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

_skirt_length_text_features: dict = {}
for _length, _texts in CLIP_SKIRT_LENGTH_PROMPTS.items():
    _skirt_length_text_features[_length] = _encode_texts(_texts)

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
    # White prompts: neutral, crisp, cool — no warm tones, no sandy hue.
    "white":      [
        "a crisp bright white cotton t-shirt on a model",
        "a solid white tee shirt with pure neutral cool white color",
        "a plain white garment with no warm yellow or beige tones",
        "a bright white blouse with neutral pure white hue",
    ],
    # Beige prompts: warm, sandy, taupe — earthy undertone distinguishes from crisp white.
    "beige":      [
        "a warm sandy beige cotton t-shirt with earthy warm hue",
        "an earthy taupe outdoor shirt with warm khaki sandy tone",
        "a soft sand-colored top with warm caramel neutral undertone",
        "an outdoor linen shirt in warm beige with yellowish warm cast",
    ],
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

# ── Solid-vs-pattern CLIP tiebreaker ─────────────────────────────────────────
# Pixel-based pattern rules fail when dots/motifs are smaller than ~2 pixels in
# the 100x100 resize (< 1% dark coverage → brightness_std < 20).  CLIP operates
# at full resolution and reliably distinguishes solid from printed garments.
CLIP_SOLID_VS_PATTERN_PROMPTS = {
    "solid": [
        "a plain solid white garment with no print or pattern, clean uniform color",
        "a smooth solid colored skirt or top, no design, no motif",
    ],
    "pattern": [
        "a white garment with black polka dots printed on the fabric",
        "a light colored skirt or top with dark polka dots or printed pattern",
        "a white fabric with repeating printed motifs, dots, or floral design",
    ],
}
_solid_pattern_features: dict = {}
for _key, _texts in CLIP_SOLID_VS_PATTERN_PROMPTS.items():
    _feats = _encode_texts(_texts)
    _centroid = _feats.mean(dim=0, keepdim=True)
    _centroid = _centroid / _centroid.norm(dim=-1, keepdim=True)
    _solid_pattern_features[_key] = _centroid


def _clip_is_patterned(pil_img: Image.Image, margin: float = 0.010) -> bool:
    """Return True when CLIP scores 'patterned' > 'solid' by at least `margin`.

    Used as a last-resort gate before returning 'white' for garments whose
    polka-dot or print is too small to register in the 100×100 pixel analysis.
    """
    image_features = _encode_image(pil_img)
    pattern_score = float((image_features @ _solid_pattern_features["pattern"].T).squeeze(0).max())
    solid_score   = float((image_features @ _solid_pattern_features["solid"].T).squeeze(0).max())
    print(f"🔍 Solid-vs-pattern CLIP: pattern={pattern_score:.3f}, solid={solid_score:.3f}")
    return pattern_score >= solid_score + margin


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


def _classify_stripe_dark_pixels(garment_pixels, garment_brightness) -> Optional[tuple]:
    """Return (stripe_color, True) from dark band pixels, or None if not a simple stripe."""
    for dark_cutoff in (100, 120):
        dark_pixels = garment_pixels[garment_brightness < dark_cutoff]
        if len(dark_pixels) < 20:
            continue
        dr, dg, db = dark_pixels.mean(axis=0)
        r, g, b = float(dr), float(dg), float(db)
        avg_brightness = (r + g + b) / 3.0
        if avg_brightness >= 72:
            continue
        print(f"🎨 Pattern detected (stripe), re-scoring dark pixels avg=({r:.0f},{g:.0f},{b:.0f})")
        if r >= g - 6 and (r - b) >= 16 and g >= b + 6:
            continue
        if avg_brightness < 60 and b > r + 5:
            # Only call it black if the blue dominance is very weak (washed-out dark pixel,
            # not true navy). A real navy stripe has b-r >= 8.
            if (b - r) < 8 and abs(r - g) < 12:
                return "black", True
            return "navy", True
        return "black", True
    return None


def _is_simple_horizontal_stripe(garment_pixels, garment_brightness) -> bool:
    """True navy/black + white horizontal stripes — not ribbing, shadows, or skirt bleed."""
    bright_frac = float((garment_brightness > 175).sum()) / max(len(garment_brightness), 1)
    dark_frac = float((garment_brightness < 120).sum()) / max(len(garment_brightness), 1)
    if bright_frac < 0.45 or dark_frac < 0.06:
        return False
    bright_px = garment_pixels[garment_brightness > 175]
    if len(bright_px) >= 20:
        bright_sat = float((bright_px.max(axis=1) - bright_px.min(axis=1)).mean())
        # Ribbed/knit white tops add texture to bright pixels; stripe bands stay flat.
        if bright_sat > 28:
            return False
    return True


def _try_bright_white_top(garment_pixels, garment_brightness) -> Optional[tuple]:
    """White tee when bright shirt pixels beat dark hair/shadow in a mixed crop."""
    n = max(len(garment_brightness), 1)
    bright_mask = garment_brightness > 170
    bright_px = garment_pixels[bright_mask]
    bright_frac = float(bright_mask.sum()) / n
    if len(bright_px) < 25 or bright_frac < 0.22:
        return None
    br, bg, bb = bright_px.mean(axis=0)
    bright_avg = (float(br) + float(bg) + float(bb)) / 3.0
    bright_sat = max(br, bg, bb) - min(br, bg, bb)
    neutral = abs(br - bg) < 12 and abs(bg - bb) < 12 and abs(br - bb) < 16
    if neutral and bright_avg >= 218 and bright_frac >= 0.18:
        print(f"🎨 Detected color: white (bright-white-top) bright_avg={bright_avg:.0f}")
        return "white", False
    # Studio white tee with hair/skin masked out — bright cluster still neutral.
    if (
        bright_frac >= 0.40
        and bright_avg >= 186
        and min(br, bg, bb) >= 180
        and bright_sat < 30
        and neutral
    ):
        print(f"🎨 Detected color: white (bright-white-top, studio) bright_avg={bright_avg:.0f}")
        return "white", False
    # Outdoor / mixed crop — shirt bright pixels mixed with shadow.
    if (
        bright_frac >= 0.22
        and bright_frac <= 0.90
        and bright_avg >= 200
        and min(br, bg, bb) >= 195
        and bright_sat < 40
        and neutral
    ):
        print(f"🎨 Detected color: white (bright-white-top, warm-lit) bright_avg={bright_avg:.0f}")
        return "white", False
    return None


def _try_warm_beige_top(garment_pixels, garment_brightness) -> Optional[tuple]:
    """Outdoor taupe / khaki / sand linen tees — strict warm cluster, not white."""
    warm = garment_pixels[
        (garment_brightness >= 95)
        & (garment_brightness <= 225)
        & (garment_pixels[:, 0] >= garment_pixels[:, 2] + 6)
        & (garment_pixels[:, 0] >= garment_pixels[:, 1] - 18)
    ]
    if len(warm) < max(22, int(len(garment_pixels) * 0.16)):
        return None
    wr, wg, wb = warm.mean(axis=0)
    warm_avg = (float(wr) + float(wg) + float(wb)) / 3.0
    warm_sat = max(wr, wg, wb) - min(wr, wg, wb)
    if wr >= wg >= wb and (wr - wb) >= 12 and 105 <= warm_avg <= 195 and warm_sat < 70:
        print(f"🎨 Detected color: beige (warm-beige-top rule)  warm_avg={warm_avg:.0f}")
        return "beige", False
    return None


def _try_solid_white_top(garment_pixels, garment_brightness) -> Optional[tuple]:
    """Ribbed or solid white crop tops — high bright fraction, not alternating stripe bands."""
    if _is_simple_horizontal_stripe(garment_pixels, garment_brightness):
        return None
    white_hit = _try_bright_white_top(garment_pixels, garment_brightness)
    if white_hit:
        return white_hit
    bright_frac = float((garment_brightness > 175).sum()) / max(len(garment_brightness), 1)
    if bright_frac < 0.22:
        return None
    bright_px = garment_pixels[garment_brightness > 175]
    if len(bright_px) < 20:
        return None
    br, bg, bb = bright_px.mean(axis=0)
    bright_avg = (float(br) + float(bg) + float(bb)) / 3.0
    bright_sat = max(br, bg, bb) - min(br, bg, bb)
    neutral = abs(br - bg) < 12 and abs(bg - bb) < 12
    if bright_avg >= 228 and bright_sat < 35 and neutral:
        print(f"🎨 Detected color: white (solid-white-top rule)  bright_frac={bright_frac:.2f}")
        return "white", False
    if bright_avg >= 205 and bright_sat < 18 and neutral:
        print(f"🎨 Detected color: white (solid-white-top rule)  bright_frac={bright_frac:.2f}")
        return "white", False
    return None


def _try_stripe_color(garment_pixels, garment_brightness, brightness_std, category_group) -> Optional[tuple]:
    if category_group != "top" or brightness_std <= 24:
        return None
    if not _is_simple_horizontal_stripe(garment_pixels, garment_brightness):
        return None
    stripe = _classify_stripe_dark_pixels(garment_pixels, garment_brightness)
    if stripe:
        color, is_stripe = stripe
        print(f"🎨 Detected color: {color} (top-stripe rule)")
        return color, is_stripe
    return None


def get_fashion_color(pil_img, category_group=None):
    # ── Step 1: centre crop ───────────────────────────────────────────────────
    # Tops/bottoms: middle 70% × central 60% — avoids belt and shoes bleeding in.
    # Shoes: lower 55% — focus on foot/sandal pixels, less skin variance.
    w, h = pil_img.size
    cx0, cx1 = int(w * 0.15), int(w * 0.85)
    if category_group == "top":
        cx0, cx1 = int(w * 0.22), int(w * 0.78)
    if category_group == "shoes":
        if h <= w * 1.3:
            cy0, cy1 = int(h * 0.35), int(h * 0.98)
        else:
            cy0, cy1 = int(h * 0.55), int(h * 0.95)
    elif category_group == "top":
        # Zone crops are short horizontal bands — use most of the band so stripes aren't clipped away.
        if h < w * 0.55:
            cy0, cy1 = int(h * 0.22), int(h * 0.92)
        else:
            # Tall person bbox — upper chest only; skip waist/dark pants at bottom of crop.
            cy0, cy1 = int(h * 0.10), int(h * 0.52)
    elif category_group == "belt":
        # Thin waist band — centre strip only, skip shorts above/below.
        cx0, cx1 = int(w * 0.18), int(w * 0.82)
        cy0, cy1 = int(h * 0.30), int(h * 0.70)
    elif category_group == "skirt":
        cy0, cy1 = int(h * 0.08), int(h * 0.88)
    else:
        cy0, cy1 = int(h * 0.20), int(h * 0.80)
    pil_img = pil_img.crop((cx0, cy0, cx1, cy1))

    img_small = pil_img.convert("RGB").resize((100, 100))
    pixels = np.array(img_small).reshape(-1, 3).astype(np.float32)

    brightness = pixels.mean(axis=1)
    saturation = pixels.max(axis=1) - pixels.min(axis=1)

    # Layer 1: mask near-pure-white background — but never strip white from top crops.
    if category_group == "top":
        is_bright_bg = np.zeros(len(pixels), dtype=bool)
        is_grey_bg = (saturation < 12) & (brightness > 248)
        is_sky = (
            (brightness > 205)
            & (pixels[:, 2] >= pixels[:, 0] - 12)
            & (saturation < 42)
        )
    else:
        is_bright_bg = (pixels[:, 0] > 235) & (pixels[:, 1] > 235) & (pixels[:, 2] > 230)
        is_grey_bg   = (saturation < 15) & (brightness > 225)
        is_sky = np.zeros(len(pixels), dtype=bool)
    # Layer 3: skin tones
    # For tops, use a tighter brightness cap (< 185 instead of < 210) so that
    # bright warm-toned fabrics (pale yellow, cream, apricot at brightness 185-210)
    # are NOT removed as skin.  Actual skin sits at 90-180; fabric highlights at 185+.
    _skin_bright_cap = 185 if category_group == "top" else 210
    is_skin = (
        (pixels[:, 0] > 100) &
        (pixels[:, 0] > pixels[:, 2]) &
        (pixels[:, 0] - pixels[:, 2] > 30) &
        (brightness > 90) & (brightness < _skin_bright_cap)
    )
    # Shoe crops often include bright floor tiles — drop them before averaging.
    if category_group == "shoes":
        is_floor = (brightness > 190) & (saturation < 28)
        is_denim = (
            (pixels[:, 2] > pixels[:, 0] + 10)
            & (brightness < 160)
            & (saturation > 8)
        )
    else:
        is_floor = np.zeros(len(pixels), dtype=bool)
        is_denim = np.zeros(len(pixels), dtype=bool)

    garment_pixels = pixels[~(is_bright_bg | is_grey_bg | is_skin | is_floor | is_denim | is_sky)]

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

    if category_group == "skirt":
        dark_frac = float((garment_brightness < 85).sum()) / max(len(garment_brightness), 1)
        bright_frac = float((garment_brightness > 175).sum()) / max(len(garment_brightness), 1)
        if dark_frac >= 0.18:
            print(f"🎨 Detected color: black (dark-skirt rule)  dark_frac={dark_frac:.2f}")
            return "black", False
        skirt_avg_sat = float(max(r, g, b) - min(r, g, b))
        # Suede/linen panel seams add brightness variance — not a polka/print skirt.
        if skirt_avg_sat < 12 and dark_frac < 0.07 and 110 <= avg_brightness <= 215:
            print(
                f"🎨 Detected color: grey (textured-solid-skirt rule)  "
                f"sat={skirt_avg_sat:.1f}, dark={dark_frac:.2f}, avg={avg_brightness:.0f}"
            )
            return "grey", False
        if brightness_std > 15 and dark_frac >= 0.06 and bright_frac >= 0.10:
            print(f"🎨 Detected color: pattern (polka/print skirt)  std={brightness_std:.1f}, dark={dark_frac:.2f}")
            return "pattern", False
        if brightness_std > 18 and avg_brightness > 175 and dark_frac >= 0.05:
            print(f"🎨 Detected color: pattern (polka/print skirt)  std={brightness_std:.1f}, avg={avg_brightness:.0f}")
            return "pattern", False
        if bright_frac >= 0.20 and dark_frac >= 0.05 and brightness_std > 12:
            print(f"🎨 Detected color: pattern (polka/print skirt)  bright={bright_frac:.2f}, dark={dark_frac:.2f}")
            return "pattern", False
        # Sparse polka-dots / small print on a light skirt (2–4% dark coverage).
        # Dark-fraction falls below the 0.05 threshold above but brightness std is
        # still elevated and the fabric is predominantly bright — must fire before
        # the grey rule below which would otherwise win on the same pixel stats.
        if brightness_std > 22 and avg_brightness > 175 and dark_frac >= 0.02 and bright_frac >= 0.45:
            print(f"🎨 Detected color: pattern (sparse-dot skirt)  std={brightness_std:.1f}, dark={dark_frac:.2f}")
            return "pattern", False
        if skirt_avg_sat < 25 and 115 <= avg_brightness <= 210 and dark_frac < 0.12:
            print(f"🎨 Detected color: grey (solid-skirt rule)  avg_rgb=({r:.0f},{g:.0f},{b:.0f})")
            return "grey", False
        if brightness_std > 26 and bright_frac >= 0.70 and dark_frac >= 0.03:
            print(f"🎨 Detected color: pattern (polka/print skirt)  std={brightness_std:.1f}, bright={bright_frac:.2f}")
            return "pattern", False
        if brightness_std > 24 and bright_frac >= 0.40 and dark_frac >= 0.04:
            print(f"🎨 Detected color: pattern (polka/print skirt)  std={brightness_std:.1f}, bright={bright_frac:.2f}")
            return "pattern", False
        if brightness_std > 22 and bright_frac >= 0.10 and dark_frac >= 0.04:
            print(f"🎨 Detected color: pattern (polka/print skirt)  std={brightness_std:.1f}")
            return "pattern", False
        bright_frac = float((garment_brightness > 175).sum()) / max(len(garment_brightness), 1)
        if bright_frac >= 0.18 and brightness_std > 35:
            print(f"🎨 Detected color: white (bright-skirt rule)  bright_frac={bright_frac:.2f}")
            return "white", False

    if category_group == "shoes":
        # ── Dark-strap priority check (must run BEFORE warm check) ────────────
        # When skin (warm) dominates the foot zone the warm-strap rule would
        # mis-classify dark-strapped shoes (black flip-flops, black sandals) as
        # "beige". A visible dark strap/sole (≥ 3 % of garment pixels, avg < 68)
        # is a stronger signal than warm foot skin → return "black" first.
        dark_strap_px = garment_pixels[garment_brightness < 75]
        dark_strap_frac = float(len(dark_strap_px)) / max(len(garment_pixels), 1)
        if dark_strap_frac >= 0.030 and len(dark_strap_px) >= 10:
            dr, dg, db = dark_strap_px.mean(axis=0)
            if (float(dr) + float(dg) + float(db)) / 3.0 < 68:
                print(f"🎨 Detected color: black (dark-strap-priority rule)  dark_frac={dark_strap_frac:.3f}")
                return "black", False

        # Warm strap / sole rule (tan, beige, brown flip-flops)
        warm_hit = _try_warm_shoe_strap_color(garment_pixels, garment_brightness)
        if warm_hit:
            return warm_hit

        dark_frac = float((garment_brightness < 80).sum()) / max(len(garment_brightness), 1)
        warm_strap_px = garment_pixels[
            (garment_brightness >= 88)
            & (garment_brightness <= 220)
            & (garment_pixels[:, 0] >= garment_pixels[:, 2] - 6)
        ]
        if (
            len(dark_strap_px) >= max(10, int(len(garment_pixels) * 0.025))
            and len(warm_strap_px) < len(dark_strap_px) * 2
        ):
            dr, dg, db = dark_strap_px.mean(axis=0)
            if (float(dr) + float(dg) + float(db)) / 3.0 < 68:
                print(f"🎨 Detected color: black (dark-strap-shoe rule)  dark_px={len(dark_strap_px)}")
                return "black", False
        if dark_frac >= 0.10:
            print(f"🎨 Detected color: black (dark-shoe rule)  dark_frac={dark_frac:.2f}")
            return "black", False

    # Light floral / patchwork prints — skip for shoes (straps + skin variance
    # falsely triggers this rule on solid-colour sandals).
    skip_pattern = category_group == "shoes"
    # Top zone crops mix shirt + hair — still allow stripe detection via top-stripe rule above.
    skip_stripe_dark = False
    # Tan leather sandals — warm strap pixels dominate over shadow/floor.
    if category_group == "shoes" and brightness_std > 25:
        shoe_dark_frac = float((garment_brightness < 80).sum()) / max(len(garment_brightness), 1)
        if shoe_dark_frac < 0.08:
            warm = garment_pixels[
            (garment_brightness >= 90)
            & (garment_brightness <= 215)
            & (garment_pixels[:, 0] >= garment_pixels[:, 2])
        ]
        if len(warm) >= max(20, int(len(garment_pixels) * 0.12)):
            wr, wg, wb = warm.mean(axis=0)
            warm_avg = (float(wr) + float(wg) + float(wb)) / 3.0
            warm_sat = max(wr, wg, wb) - min(wr, wg, wb)
            if wr >= wg >= wb and (wr - wb) >= 6 and 85 <= warm_avg <= 210 and warm_sat < 70:
                print(f"🎨 Detected color: beige (warm-shoe rule)  warm_avg={warm_avg:.0f}")
                return "beige", False
    # ── Top colour pipeline (single ordered pass) ─────────────────────────────
    if category_group == "top":
        # 1. Stripe — must precede bright-white: white stripe pixels fool _try_bright_white_top.
        if brightness_std > 24:
            stripe_hit = _try_stripe_color(garment_pixels, garment_brightness, brightness_std, category_group)
            if stripe_hit:
                return stripe_hit
        # 1b. Polka-dot / printed top: bright background + discrete dark elements.
        # Must run AFTER stripe check (stripes already handled with a specific colour)
        # but BEFORE _try_bright_white_top, which would otherwise claim the white
        # pixels and return "white" without ever seeing the dot pattern.
        _dark_frac_top = float((garment_brightness < 80).sum()) / max(len(garment_brightness), 1)
        _bright_frac_top = float((garment_brightness > 175).sum()) / max(len(garment_brightness), 1)
        if _bright_frac_top >= 0.30 and _dark_frac_top >= 0.04 and brightness_std > 20:
            print(f"🎨 Detected color: pattern (polka-dot/print top)  std={brightness_std:.1f}, dark={_dark_frac_top:.2f}")
            return "pattern", False
        # 2. Solid white and mixed-crop white
        white_hit = _try_bright_white_top(garment_pixels, garment_brightness)
        if white_hit:
            return white_hit
        white_hit = _try_solid_white_top(garment_pixels, garment_brightness)
        if white_hit:
            return white_hit
        # 3. Warm beige / taupe
        beige_hit = _try_warm_beige_top(garment_pixels, garment_brightness)
        if beige_hit:
            return beige_hit
        # 4. Light-wash denim / chambray
        # Threshold starts at 100 (not 130) because dark-area shadows on denim
        # can pull avg_brightness below 130 even on a medium-wash jacket.
        avg_sat = float(max(r, g, b) - min(r, g, b))
        if 100 <= avg_brightness <= 235 and avg_sat >= 10 and b >= r + 8 and b >= g - 8:
            print(f"🎨 Detected color: light_blue (denim-top rule)  avg_rgb=({r:.0f},{g:.0f},{b:.0f})")
            return "light_blue", False
        # 5. Lavender / dusty purple
        purple_px = garment_pixels[
            (garment_pixels[:, 2] >= garment_pixels[:, 0])
            & (garment_brightness > 115)
            & (garment_brightness < 240)
        ]
        if len(purple_px) >= max(20, int(len(garment_pixels) * 0.10)):
            pr, pg, pb = purple_px.mean(axis=0)
            purple_sat = max(pr, pg, pb) - min(pr, pg, pb)
            # True lavender/purple has R ≥ G (pink-purple hue).
            # Blue denim has G > R (blue-teal hue) — exclude it here; the
            # light_blue rule above already handles denim correctly.
            if pb >= pr + 12 and purple_sat >= 18 and purple_sat < 55 and pr >= pg:
                print(f"🎨 Detected color: lavender (purple-top rule)  avg_rgb=({pr:.0f},{pg:.0f},{pb:.0f})")
                return "lavender", False
        # 6. Mixed-crop fallback — white or beige tee tucked into dark jeans.
        # The full-crop average is dragged dark by pants pixels. Re-score only the
        # bright-pixel cluster (the shirt) to retrieve the actual garment colour.
        if brightness_std > 35:
            bright_garment = garment_pixels[garment_brightness > 170]
            dark_garment = garment_pixels[garment_brightness < 90]
            if (
                len(bright_garment) >= max(25, int(len(garment_pixels) * 0.15))
                and len(bright_garment) > len(dark_garment)
            ):
                white_hit = _try_bright_white_top(bright_garment, bright_garment.mean(axis=1))
                if white_hit:
                    return white_hit
                beige_hit = _try_warm_beige_top(bright_garment, bright_garment.mean(axis=1))
                if beige_hit:
                    return beige_hit
                br, bg, bb = bright_garment.mean(axis=0)
                bright_avg = (float(br) + float(bg) + float(bb)) / 3.0
                bright_sat = max(br, bg, bb) - min(br, bg, bb)
                neutral = abs(br - bg) < 12 and abs(bg - bb) < 12
                if bright_avg > 168 and bright_sat < 45 and neutral and bright_avg >= 218:
                    print(f"🎨 Detected color: white (bright-top rule)  bright_avg={bright_avg:.0f}")
                    return "white", False
        # 7. Bright-fraction last resort — covers low-variance crops where
        # _try_bright_white_top (22% threshold) doesn't fire but the garment
        # is clearly mostly bright and neutral (e.g. overexposed/studio white top).
        bright_frac = float((garment_brightness > 175).sum()) / max(len(garment_brightness), 1)
        dark_frac_top = float((garment_brightness < 80).sum()) / max(len(garment_brightness), 1)
        if bright_frac >= 0.85 and brightness_std < 20:
            pass  # background bleed, not a white garment
        elif bright_frac >= 0.07 and dark_frac_top < 0.04 and brightness_std < 28:
            bright_px = garment_pixels[garment_brightness > 175]
            if len(bright_px) >= 20:
                br, bg, bb = bright_px.mean(axis=0)
                bright_sat = max(br, bg, bb) - min(br, bg, bb)
                neutral = abs(br - bg) < 12 and abs(bg - bb) < 12
                if neutral and bright_sat < 35:
                    print(f"🎨 Detected color: white (bright-fraction-top rule)  bright_frac={bright_frac:.2f}")
                    return "white", False
    # White shorts/skirts with a dark belt: bright garment pixels dominate — use them, not belt.
    if category_group in ("bottom", "skirt") and brightness_std > 45:
        bright_garment = garment_pixels[garment_brightness > 165]
        dark_frac_bg = float((garment_brightness < 85).sum()) / max(len(garment_brightness), 1)
        if category_group == "skirt" and dark_frac_bg >= 0.04 and brightness_std > 22:
            print(f"🎨 Detected color: pattern (polka/print skirt)  std={brightness_std:.1f}")
            return "pattern", False
        if len(bright_garment) >= max(30, int(len(garment_pixels) * 0.22)):
            br, bg, bb = bright_garment.mean(axis=0)
            bright_avg = (float(br) + float(bg) + float(bb)) / 3.0
            bright_sat = max(br, bg, bb) - min(br, bg, bb)
            if bright_avg > 182 and bright_sat < 40:
                print(f"🎨 Detected color: white (bright-bottom rule)  bright_avg={bright_avg:.0f}")
                return "white", False
    if not skip_pattern and category_group == "bottom" and brightness_std > 32 and avg_brightness > 145:
        print(f"🎨 Detected color: pattern (printed-bottom)  std={brightness_std:.1f}")
        return "pattern", False
    # Dark-background print: light polka-dots / pattern elements on dark bottoms/dresses.
    # Thresholds are tighter than the skirt rules because dark denim naturally has
    # specular highlights — we only fire when the bright fraction is clearly too large
    # to be explained by fabric sheen (>= 10%) and std is high (> 32).
    if not skip_pattern and category_group in ("bottom", "dress") and brightness_std > 32:
        _light_on_dark_bright = float((garment_brightness > 170).sum()) / max(len(garment_brightness), 1)
        if avg_brightness < 120 and _light_on_dark_bright >= 0.10:
            print(f"🎨 Detected color: pattern (light-on-dark print)  std={brightness_std:.1f}, avg={avg_brightness:.0f}")
            return "pattern", False
    # Monochrome high-contrast print: black polka-dots on white, zebra, houndstooth.
    # The light-floral rule below requires saturation.std() > 18, but black+white prints
    # have near-zero saturation on BOTH colours, so saturation variance is ~0 and that
    # rule misses them entirely.  This rule fires on high brightness std + both dark and
    # bright pixels present in meaningful proportions, regardless of saturation.
    if not skip_pattern and category_group not in ("top", "skirt"):
        _mono_dark = float((garment_brightness < 80).sum()) / max(len(garment_brightness), 1)
        _mono_bright = float((garment_brightness > 175).sum()) / max(len(garment_brightness), 1)
        if brightness_std > 26 and avg_brightness > 130 and _mono_dark >= 0.03 and _mono_bright >= 0.35:
            print(f"🎨 Detected color: pattern (mono-contrast print)  std={brightness_std:.1f}")
            return "pattern", False
    if not skip_pattern and brightness_std > 38 and avg_brightness > 155 and float(saturation.std()) > 18:
        print(f"🎨 Detected color: pattern (light-floral)  brightness_std={brightness_std:.1f}, avg={avg_brightness:.0f}")
        return "pattern", False

    if not skip_pattern and not skip_stripe_dark and brightness_std > 50:
        if category_group == "top":
            white_hit = _try_bright_white_top(garment_pixels, garment_brightness)
            if white_hit:
                return white_hit
            beige_hit = _try_warm_beige_top(garment_pixels, garment_brightness)
            if beige_hit:
                return beige_hit
        stripe = _classify_stripe_dark_pixels(garment_pixels, garment_brightness)
        if stripe:
            return stripe
        dark_pixels = garment_pixels[garment_brightness < 120]
        if len(dark_pixels) >= 30:
            # Guard: if ≥ 72 % of the crop is bright the garment is fundamentally
            # light-coloured.  Any dark pixels come from an accessory or shadow
            # (e.g. a bag handle crossing the vest zone), NOT from the garment's
            # own stripe/pattern.  Reassigning r,g,b to those dark pixels would
            # corrupt every downstream rule (step 3 avg_brightness < 60 would fire
            # on the accessory colour instead of the garment colour).
            _bright_guard = float((garment_brightness > 175).sum()) / max(len(garment_brightness), 1)
            if _bright_guard >= 0.72:
                print(f"🎨 Stripe re-score skipped (bright_frac={_bright_guard:.2f}) — light garment, dark pixels are intruder")
                # Leave r, g, b, avg_brightness as the overall garment averages so
                # the bright-colour / palette rules correctly classify the garment.
            else:
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
                if category_group == "top":
                    beige_hit = _try_warm_beige_top(garment_pixels, garment_brightness)
                    if beige_hit:
                        return beige_hit
                    white_hit = _try_bright_white_top(garment_pixels, garment_brightness)
                    if white_hit:
                        return white_hit
                # Navy stripe pixels are often very dark (47,46,54) — a blue bias of only
                # 7 points. The global dark-rule requires b > r+15, which is too strict here.
                # Use a lower threshold of b > r+5 specifically for pattern-detected items.
                if avg_brightness < 60 and b > r + 5:
                    if (b - r) < 12 and abs(r - g) < 15:
                        print(f"🎨 Detected color: black (dark-neutral-pattern)  avg_rgb=({r:.0f},{g:.0f},{b:.0f})")
                        return "black", True
                    print(f"🎨 Detected color: navy (dark-rule-pattern)  avg_rgb=({r:.0f},{g:.0f},{b:.0f})")
                    return "navy", True   # is_stripe=True → caller uses stripe-specific vector
                # Fall through — normal rules now run on the dark-pixel average (non-navy stripes)
        else:
            # Mostly light-coloured print (e.g. pastel on white) — no dominant dark colour
            print(f"🎨 Detected color: pattern (light-print)  brightness_std={brightness_std:.1f}")
            return "pattern", False

    # ── Step 2: bright-colour rule (white garments) ───────────────────────────
    top_white_threshold = 175 if category_group == "top" else 190
    if avg_brightness > top_white_threshold and saturation.mean() < 30:
        # Pixel analysis at 100×100 cannot detect sparse small dots/prints.
        # CLIP sees the full-resolution crop and reliably spots polka-dot /
        # printed garments that the pixel stats miss entirely.
        if category_group in ("skirt", "top", "bottom", "dress") and _clip_is_patterned(pil_img):
            print(f"🎨 Detected color: pattern (CLIP override of bright-rule)  avg_rgb=({r:.0f},{g:.0f},{b:.0f})")
            return "pattern", False
        print(f"🎨 Detected color: white (bright-rule)  avg_rgb=({r:.0f},{g:.0f},{b:.0f})")
        return "white", False

    # ── Step 3: dark-colour rule ──────────────────────────────────────────────
    if avg_brightness < 60:
        if category_group == "top":
            white_hit = _try_bright_white_top(garment_pixels, garment_brightness)
            if white_hit:
                return white_hit
        if r >= g >= b and (r - b) >= 12:
            print(f"🎨 Detected color: beige (warm-dark-top rule)  avg_rgb=({r:.0f},{g:.0f},{b:.0f})")
            return "beige", False
        result = "navy" if (b > r + 15) else "black"
        print(f"🎨 Detected color: {result} (dark-rule)  avg_rgb=({r:.0f},{g:.0f},{b:.0f})")
        return result, False

    # ── Step 3.4: nude / tan sandals (before shoe-dark — shadows read as black) ─
    if category_group == "shoes":
        avg_saturation = max(r, g, b) - min(r, g, b)
        if r > g > b and (r - b) > 10 and 80 <= avg_brightness <= 240 and avg_saturation < 95:
            if (r - g) < 45 and (g - b) > 6:
                print(f"🎨 Detected color: pink (nude-sandal rule)  avg_rgb=({r:.0f},{g:.0f},{b:.0f})")
                return "pink", False
            print(f"🎨 Detected color: beige (nude-sandal rule)  avg_rgb=({r:.0f},{g:.0f},{b:.0f})")
            return "beige", False
        if r >= g >= b and (r - b) >= 8 and 85 <= avg_brightness <= 200 and avg_saturation < 60:
            print(f"🎨 Detected color: beige (tan-sandal rule)  avg_rgb=({r:.0f},{g:.0f},{b:.0f})")
            return "beige", False

    # ── Step 3.3: black shoes on pavement (avoid grey misread) ────────────────
    if category_group == "shoes" and avg_brightness < 95 and max(r, g, b) - min(r, g, b) < 30:
        print(f"🎨 Detected color: black (shoe-dark rule)  avg_rgb=({r:.0f},{g:.0f},{b:.0f})")
        return "black", False

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


def _foot_skin_and_dark(pil_img: Image.Image) -> tuple:
    """Estimate visible foot skin vs dark shoe pixels (excludes bright studio floor)."""
    w, h = pil_img.size
    foot = pil_img.crop((int(w * 0.12), int(h * 0.45), int(w * 0.88), h))
    arr = np.array(foot.resize((72, 48)).convert("RGB"), dtype=np.float32)
    br = arr.mean(axis=2).reshape(-1)
    sat = (arr.max(axis=2) - arr.min(axis=2)).reshape(-1)
    shoe_mask = br < 210
    br_s = br[shoe_mask] if shoe_mask.sum() > 50 else br
    sat_s = sat[shoe_mask] if shoe_mask.sum() > 50 else sat
    skin_frac = float(((br_s > 100) & (br_s < 200) & (sat_s > 10)).sum()) / max(len(br_s), 1)
    dark_frac = float((br_s < 75).sum()) / max(len(br_s), 1)
    return skin_frac, dark_frac


def _try_warm_shoe_strap_color(garment_pixels, garment_brightness) -> Optional[tuple]:
    """Tan/beige thong straps and cork soles — not black slides."""
    warm = garment_pixels[
        (garment_brightness >= 88)
        & (garment_brightness <= 220)
        & (garment_pixels[:, 0] >= garment_pixels[:, 2] - 6)
    ]
    if len(warm) < max(20, int(len(garment_pixels) * 0.14)):
        return None
    wr, wg, wb = warm.mean(axis=0)
    warm_avg = (float(wr) + float(wg) + float(wb)) / 3.0
    warm_sat = max(wr, wg, wb) - min(wr, wg, wb)
    if wr >= wg - 10 and (wr - wb) >= 3 and 92 <= warm_avg <= 198 and warm_sat < 78:
        color = "beige" if warm_avg >= 138 else "brown"
        print(f"🎨 Detected color: {color} (warm-shoe-strap rule)  warm_avg={warm_avg:.0f}")
        return color, False
    return None


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

    open_shoe_styles = ("espadrille", "flat_shoe", "slide_sandal", "birkenstock", "puffy_slide", "heeled_sandal", "flip_flop")
    boot_score = scores.get("heeled_boot", 0.0)
    flat_score = scores.get("flat_shoe", 0.0)
    slide_score = max(scores.get("slide_sandal", 0.0), scores.get("puffy_slide", 0.0))
    open_best_style = max(open_shoe_styles, key=lambda s: scores.get(s, 0.0))
    open_best = scores.get(open_best_style, 0.0)
    skin_frac, dark_frac = _foot_skin_and_dark(pil_img)
    closed_toe = skin_frac < 0.14 and dark_frac >= 0.03

    if closed_toe and flat_score >= open_best - 0.04:
        best_style = "flat_shoe"
    elif boot_score >= max(open_best, slide_score) + 0.025 and boot_score >= 0.28:
        best_style = "heeled_boot"
    else:
        best_style = open_best_style

    flip_score = scores.get("flip_flop", 0.0)
    if best_style not in ("heeled_boot", "flat_shoe") and boot_score < 0.32:
        if flip_score >= scores.get(best_style, 0) - 0.012:
            best_style = "flip_flop"
        elif skin_frac >= 0.32 and flip_score >= scores.get(best_style, 0) - 0.06:
            if best_style in ("slide_sandal", "puffy_slide", "birkenstock", "espadrille"):
                best_style = "flip_flop"
        elif (
            _looks_like_flip_flop(pil_img)
            and best_style in ("slide_sandal", "espadrille", "birkenstock", "puffy_slide")
            and boot_score < 0.28
        ):
            best_style = "flip_flop"

    rounded = {k: round(v, 3) for k, v in sorted(scores.items(), key=lambda x: -x[1])}
    print(f"?? Shoe style: {best_style}  scores: {rounded}")
    return best_style


def _looks_like_flip_flop(pil_img: Image.Image) -> bool:
    """Toe-post sandals: visible skin with a small dark thong strap."""
    w, h = pil_img.size
    foot = pil_img.crop((int(w * 0.08), int(h * 0.30), int(w * 0.92), h))
    arr = np.array(foot.resize((72, 48)).convert("RGB"), dtype=np.float32)
    br = arr.mean(axis=2).reshape(-1)
    sat = (arr.max(axis=2) - arr.min(axis=2)).reshape(-1)
    skin_frac = float(((br > 105) & (br < 215) & (sat > 8)).sum()) / len(br)
    dark_frac = float((br < 72).sum()) / len(br)
    return skin_frac >= 0.15 and 0.003 <= dark_frac <= 0.35


def _refine_shoe_color(crop_img: Image.Image, color: str, shoe_style: Optional[str]) -> str:
    """Read strap colour from foot zone — tan flip-flops, not denim hem or floor average."""
    w, h = crop_img.size
    foot = crop_img.crop((int(w * 0.08), int(h * 0.28), int(w * 0.92), h))
    arr = np.array(foot.convert("RGB"), dtype=np.float32).reshape(-1, 3)
    br = arr.mean(axis=1)
    dark_px = arr[(br < 78) & ~((arr[:, 2] > arr[:, 0] + 10) & (br < 160))]
    dark_frac = float(len(dark_px)) / max(len(arr), 1)

    # Dark-strap priority (runs on unmasked pixels so threshold is 4%, not 3%).
    # A black strap/sole covers 10-15% of the foot zone; beige-flip-flop shadows
    # only ~1-2%.  Check this BEFORE the warm/skin rule so foot skin can never
    # override a clearly dark shoe.
    if dark_frac >= 0.040 and len(dark_px) >= 10:
        dr, dg, db = dark_px.mean(axis=0)
        if (float(dr) + float(dg) + float(db)) / 3.0 < 68:
            print(f"👟 Shoe refine → black ({shoe_style or 'shoe'}) [dark-strap]")
            return "black"

    warm_px = arr[
        (br >= 88) & (br <= 215)
        & (arr[:, 0] >= arr[:, 2] - 6)
        & ~((arr[:, 2] > arr[:, 0] + 10) & (br < 160))
    ]
    if len(warm_px) >= max(18, int(len(arr) * 0.12)):
        wr, wg, wb = warm_px.mean(axis=0)
        warm_avg = (float(wr) + float(wg) + float(wb)) / 3.0
        if len(dark_px) < len(warm_px) * 0.35 and 95 <= warm_avg <= 198:
            warm_color = "beige" if warm_avg >= 138 else "brown"
            print(f"👟 Shoe refine → {warm_color} ({shoe_style or 'shoe'})")
            return warm_color
    if len(dark_px) >= max(8, int(len(arr) * 0.012)):
        dr, dg, db = dark_px.mean(axis=0)
        if (float(dr) + float(dg) + float(db)) / 3.0 < 68:
            print(f"👟 Shoe refine → black ({shoe_style or 'shoe'})")
            return "black"
    return color


SHOE_STYLE_COLOR_PHRASES = {
    "slide_sandal": {
        "beige":  "beige tan leather H cutout mule slide sandals on feet",
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
    "flip_flop": {
        "black":  "black thong flip flop sandals with thin kitten heel on feet",
        "white":  "white flat flip flop toe post sandals on feet",
        "beige":  "beige leather flip flop thong sandals on feet",
        "brown":  "brown leather flip flop sandals on feet",
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
        elif shoe_style == "flip_flop":
            phrase = f"{color_display} thong flip flop sandals on feet"
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
    elif shoe_style == "flip_flop":
        phrases.extend([
            "black leather cross strap slide sandals with cork footbed",
            "black high heel strappy dress sandals with ankle strap",
            "black quilted puffy slide sandals with shearling lining",
            "beige double buckle birkenstock cork sandals",
        ])
    if not phrases:
        return []
    feats = _encode_texts(phrases)
    centroid = feats.mean(dim=0, keepdim=True)
    centroid = centroid / centroid.norm(dim=-1, keepdim=True)
    return centroid.cpu().numpy().flatten().tolist()



TOP_STYLE_COLOR_PHRASES = {
    "tshirt": {
        "beige": "beige oversized cotton t-shirt with short sleeves crew neck",
        "white": "plain white cotton t-shirt with short sleeves crew neck no print",
        "black": "black cotton t-shirt with short sleeves crew neck",
        "grey":  "grey oversized tee shirt with short sleeves",
        "lavender": "soft dusty lavender purple v-neck cotton t-shirt on a model",
        "purple": "light purple lavender cotton t-shirt with v-neck on a model",
        "pink":  "dusty pink mauve cotton t-shirt with short sleeves",
    },
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
    "coat": {
        "black": "black shiny leather bomber jacket with high collar",
        "brown": "brown leather coat jacket on a model",
        "beige": "beige trench coat outerwear",
    },
    "vest": {
        "black": "dark navy denim sleeveless vest with buttons and waist tie",
        "light_blue": "blue denim sleeveless vest with button front",
        "grey": "grey denim waistcoat vest on a model",
    },
    "shirt": {
        "light_blue": "light blue denim button down shirt with long sleeves and chest pockets",
        "navy": "dark indigo denim shirt with long sleeves and front buttons",
        "black": "black denim shirt with long sleeves and collar",
        "grey": "grey denim shirt with long sleeves and button placket",
        "white": "white denim shirt with long sleeves and chest pockets",
    },
}


def detect_top_style_clip(pil_img) -> str | None:
    """Classify top sub-type: tshirt vs tank vs halter vs strapless."""
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
    tshirt_score = scores.get("tshirt", 0)
    tank_score = scores.get("tank", 0)
    strapless_score = scores.get("strapless", 0)
    # Oversized tees with sleeves often score near strapless — prefer tshirt when close.
    if tshirt_score >= best_score - 0.012 and tshirt_score >= strapless_score - 0.008:
        best_style = "tshirt"
        best_score = tshirt_score
    elif (
        best_style == "halter"
        and tank_score >= best_score - 0.020
    ):
        best_style = "tank"
        best_score = tank_score
    elif (
        best_style == "strapless"
        and tank_score >= strapless_score - 0.010
        and tshirt_score >= strapless_score - 0.015
    ):
        best_style = "tshirt" if tshirt_score >= tank_score else "tank"
        best_score = scores[best_style]
    rounded = {k: round(v, 3) for k, v in sorted(scores.items(), key=lambda x: -x[1])}
    if best_score < 0.12:
        print(f"Top style: none  scores: {rounded}")
        return None
    print(f"Top style: {best_style}  scores: {rounded}")
    return best_style


def _resolve_denim_top_style(crop_img: Image.Image, clip_style: str | None) -> str:
    """Long-sleeve denim shirts vs sleeveless denim vests."""
    image_features = _encode_image(crop_img)
    shirt_prompts = CLIP_TOP_STYLE_PROMPTS["shirt"]
    vest_prompts = CLIP_TOP_STYLE_PROMPTS["vest"]
    shirt_s = float((image_features @ _encode_texts(shirt_prompts).T).max())
    vest_s = float((image_features @ _encode_texts(vest_prompts).T).max())
    if shirt_s >= vest_s - 0.012:
        print(f"👕 Denim top → shirt (shirt={shirt_s:.3f}, vest={vest_s:.3f})")
        return "shirt"
    print(f"👕 Denim top → vest (shirt={shirt_s:.3f}, vest={vest_s:.3f})")
    return "vest" if clip_style in (None, "vest", "coat") else clip_style


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
    if top_style == "tshirt":
        phrases.extend([
            "navy and white horizontal striped polo shirt with collar",
            "black and white horizontal striped t-shirt",
            "rust red orange boxy cotton t-shirt",
            "navy blue short sleeve t-shirt on model",
            "flared wide leg denim jeans pants on model",
            "black short sleeve t-shirt on model",
        ])
    if top_style in ("halter", "tank", "strapless"):
        phrases.extend([
            "light purple lavender ribbed sleeveless top with zipper",
            "light blue cotton tank top on model",
            "dusty pink mauve sleeveless top",
            "lavender sleeveless crop top",
        ])
    if top_style == "coat":
        phrases.extend([
            "white cotton t-shirt on model",
            "white ribbed tank top on model",
            "white short sleeve tee on model",
            "light blue denim shirt on model",
        ])
    if top_style == "shirt":
        phrases.extend([
            "blue denim sleeveless vest with button front",
            "sleeveless denim waistcoat vest on a model",
            "plain white cotton sleeveless shirt on model",
            "white ribbed tank top on model",
            "black sleeveless tank top with thin shoulder straps",
        ])
    if not phrases:
        return []
    feats = _encode_texts(phrases)
    centroid = feats.mean(dim=0, keepdim=True)
    centroid = centroid / centroid.norm(dim=-1, keepdim=True)
    return centroid.cpu().numpy().flatten().tolist()




def _detect_pastel_top_color_clip(crop_img: Image.Image) -> Optional[str]:
    """CLIP vote for lavender/purple/pink vs white on pastel outdoor tops."""
    image_features = _encode_image(crop_img)
    lavender_prompts = [
        "a soft lavender v-neck t-shirt on a model",
        "a light purple heathered jersey top",
        "soft dusty lavender mauve purple cotton v-neck t-shirt on a model",
    ]
    white_prompts = [
        "plain white cotton t-shirt with short sleeves and crew neck no stripes",
        "a plain crew-neck t-shirt on a white background",
    ]
    lav_feats = _encode_texts(lavender_prompts)
    white_feats = _encode_texts(white_prompts)
    lavender_s = float((image_features @ lav_feats.T).max())
    white_s = float((image_features @ white_feats.T).max())
    if lavender_s >= white_s + 0.012 and lavender_s >= 0.24:
        print(f"🎨 Pastel top CLIP: lavender  lav={lavender_s:.3f} white={white_s:.3f}")
        return "lavender"
    scores = {}
    for color in ("lavender", "purple", "pink", "white"):
        prompts = COLOR_TEXT_PROMPTS.get(color, [color])
        feats = _encode_texts(prompts[:3])
        scores[color] = float((image_features @ feats.T).max())
    best_pastel = max(("lavender", "purple", "pink"), key=lambda c: scores[c])
    if scores[best_pastel] >= scores["white"] + 0.012 and scores[best_pastel] >= 0.22:
        rounded = {k: round(v, 3) for k, v in scores.items()}
        print(f"🎨 Pastel top CLIP: {best_pastel}  scores: {rounded}")
        return best_pastel
    return None


def _detect_denim_vest_top_clip(crop_img: Image.Image) -> Optional[tuple]:
    """Sleeveless denim waistcoat — beats white-wall false positives on studio shots."""
    image_features = _encode_image(crop_img)
    vest_prompts = [
        "dark navy denim sleeveless vest with buttons and waist tie on a model",
        "black denim waistcoat vest worn with a long grey skirt",
        "sleeveless denim jacket vest with front button placket",
    ]
    compare_prompts = {
        "tank": ["black sleeveless tank top with thin shoulder straps", "ribbed cotton camisole with spaghetti shoulder straps"],
        "dress": ["a midi dress reaching below the knee on a model", "a sleeveless casual summer sundress"],
        "white": COLOR_TEXT_PROMPTS.get("white", ["a white top"])[:2],
    }
    vest_s = float((image_features @ _encode_texts(vest_prompts).T).max())
    scores = {"vest": vest_s}
    for name, texts in compare_prompts.items():
        scores[name] = float((image_features @ _encode_texts(texts).T).max())
    rivals = max(scores[k] for k in ("tank", "dress", "white"))
    if vest_s >= rivals - 0.02 and vest_s >= 0.26:
        print(f"👕 Denim vest CLIP  scores: {{{', '.join(f'{k}: {v:.3f}' for k,v in scores.items())}}}")
        return "denim", "black", "vest"
    return None


def _looks_like_separate_top_skirt(img: Image.Image) -> bool:
    """Two-piece look: sleeveless top + long skirt (not a one-piece dress)."""
    w, h = img.size
    if h <= w * 0.85:
        return False
    upper = _score_categories(img.crop((0, 0, w, int(h * 0.55))))
    lower = _score_categories(img.crop((0, int(h * 0.38), w, h)))
    top_ok = upper.get("top", 0) >= 0.24
    skirt_ok = (
        lower.get("skirt", 0) >= 0.24
        and lower.get("skirt", 0) >= lower.get("dress", 0) - 0.02
    )
    return top_ok and skirt_ok


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


def detect_skirt_length_clip(pil_img) -> tuple:
    """Classify skirt length: mini vs midi vs maxi.

    Returns (best_length: str, scores: dict) so callers can inspect raw scores
    for confidence-gated upgrade logic.
    """
    image_features = _encode_image(pil_img)
    best_length = "midi"
    best_score = -1.0
    scores = {}
    for length, text_feats in _skirt_length_text_features.items():
        sims = (image_features @ text_feats.T).squeeze(0)
        score = float(sims.max())
        scores[length] = score
        if score > best_score:
            best_score = score
            best_length = length
    mini_score = scores.get("mini", 0)
    midi_score = scores.get("midi", 0)
    maxi_score = scores.get("maxi", 0)
    # Let argmax win; apply a conservative maxi boost only since floor-length garments
    # are systematically under-scored by CLIP relative to mini/midi prompts.
    if maxi_score >= midi_score + 0.010 and maxi_score >= mini_score:
        best_length = "maxi"
    rounded = {k: round(v, 3) for k, v in sorted(scores.items(), key=lambda x: -x[1])}
    print(f"Skirt length: {best_length}  scores: {rounded}")
    return best_length, scores


SKIRT_LENGTH_COLOR_PHRASES = {
    "mini": {
        "black": "black high waisted mini skirt above the knee with front patch pockets",
        "white": "white structured mini skirt above the knee high waisted",
        "beige": "beige linen mini skirt above the knee",
        "navy":  "navy blue mini skirt above the knee",
    },
    "midi": {
        "black": "black pleated midi skirt below the knee",
        "white": "white midi skirt below the knee",
        "pattern": "white midi skirt with black polka dots and lace trim",
    },
    "maxi": {
        "black": "long black maxi skirt floor length",
        "white": "white maxi skirt floor length",
        "pattern": "cream maxi skirt with black polka dots and lace hem",
    },
}


def get_skirt_length_color_vector(color: str, fabric: str, skirt_length: str) -> list:
    length_phrases = SKIRT_LENGTH_COLOR_PHRASES.get(skirt_length, {})
    phrase = length_phrases.get(color)
    if not phrase:
        length_label = {
            "mini": "mini skirt above the knee",
            "midi": "midi skirt below the knee",
            "maxi": "maxi skirt floor length",
        }.get(skirt_length, "skirt")
        phrase = f"{color.replace('_', ' ')} {length_label}"
    feats = _encode_texts([phrase])
    return feats.cpu().numpy().flatten().tolist()


def get_skirt_length_contrast_vector(skirt_length: str) -> list:
    wrong_phrases = {
        "mini": [
            "long black maxi skirt floor length on a model",
            "black pleated midi skirt below the knee",
            "flowing maxi skirt reaching the ankles",
            "black satin maxi skirt with side slit full length",
        ],
        "midi": [
            "short black mini skirt above the knee mid thigh",
            "plain white mini skirt above the knee",
            "long black maxi skirt floor length",
        ],
        "maxi": [
            "short black mini skirt above the knee",
            "plain white mini skirt mid thigh",
            "black pleated midi skirt below the knee",
        ],
    }
    phrases = wrong_phrases.get(skirt_length, [])
    if not phrases:
        return []
    feats = _encode_texts(phrases)
    centroid = feats.mean(dim=0, keepdim=True)
    centroid = centroid / centroid.norm(dim=-1, keepdim=True)
    return centroid.cpu().numpy().flatten().tolist()


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
    phrases_map = {
        "black": [
            "black leather belt with silver buckle on jeans waist",
            "thin black leather belt with silver metal buckle",
            "black waist belt with round silver buckle on mini skirt",
        ],
        "brown": ["brown leather belt with brass buckle"],
        "beige": ["beige tan leather belt with metal buckle"],
        "burgundy": ["burgundy leather belt with gold buckle"],
        "white": ["white leather belt with silver buckle"],
    }
    phrases = phrases_map.get(color, [f"{color.replace('_', ' ')} leather belt with metal buckle"])
    feats = _encode_texts(phrases if isinstance(phrases, list) else [phrases])
    if len(phrases) > 1:
        centroid = feats.mean(dim=0, keepdim=True)
        centroid = centroid / centroid.norm(dim=-1, keepdim=True)
        return centroid.cpu().numpy().flatten().tolist()
    return feats.cpu().numpy().flatten().tolist()


def get_belt_wrong_color_contrast_vector(color: str) -> list:
    """Wrong-color belt phrases — brown belts score high, black belts score low."""
    wrong = {
        "black": [
            "brown leather belt with brass buckle on jeans",
            "tan beige leather waist belt with gold buckle",
            "cognac brown leather belt on model waist",
            "camel brown leather belt with metal buckle",
        ],
        "brown": [
            "black leather belt with silver buckle on jeans waist",
            "thin black leather belt with silver metal buckle",
        ],
        "white": [
            "brown leather belt with brass buckle",
            "black leather belt with silver buckle",
        ],
    }
    phrases = wrong.get(color, [])
    if not phrases:
        return []
    feats = _encode_texts(phrases)
    centroid = feats.mean(dim=0, keepdim=True)
    centroid = centroid / centroid.norm(dim=-1, keepdim=True)
    return centroid.cpu().numpy().flatten().tolist()


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
    if best_score < 0.26:
        return "woven"
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
    elif category_group == "skirt":
        phrase = f"{color.replace('_', ' ')} high waisted mini skirt above the knee"
        feats = _encode_texts([phrase])
        return feats.cpu().numpy().flatten().tolist()
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
    if color == "white":
        wrong = _encode_texts([
            "navy blue cotton t-shirt on model",
            "black short sleeve t-shirt",
            "rust red orange cotton t-shirt",
            "navy and white horizontal striped polo shirt",
            "black and white striped t-shirt",
        ])
        centroid = wrong.mean(dim=0, keepdim=True)
        centroid = centroid / centroid.norm(dim=-1, keepdim=True)
        extras["contrastColorVector"] = centroid.cpu().numpy().flatten().tolist()
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


def _attach_color_vectors(item: dict, color: str, is_stripe: bool, category_group: str, fabric: str, shoe_style: str = None, bottom_length: str = None, top_style: str = None, skirt_length: str = None) -> dict:
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
    if category_group == "skirt" and skirt_length and color != "pattern":
        item["skirtLength"] = skirt_length
        item["colorVector"] = get_skirt_length_color_vector(color, fabric, skirt_length)
        length_contrast = get_skirt_length_contrast_vector(skirt_length)
        if length_contrast:
            item["styleContrastVector"] = length_contrast
        for key, vec in get_extra_contrast_vectors(color).items():
            if vec:
                item[key] = vec
        return item
    if category_group == "skirt" and skirt_length and color == "pattern":
        item["skirtLength"] = skirt_length
        item["colorVector"] = get_skirt_length_color_vector("pattern", fabric, skirt_length)
        length_contrast = get_skirt_length_contrast_vector(skirt_length)
        if length_contrast:
            item["styleContrastVector"] = length_contrast
        solid = get_solid_contrast_vector("skirt")
        if solid:
            item["contrastColorVector"] = solid
        return item
    if category_group == "belt" and color != "pattern":
        item["colorVector"] = get_belt_color_vector(color)
        contrast = get_belt_contrast_vector()
        if contrast:
            item["styleContrastVector"] = contrast
        wrong_color = get_belt_wrong_color_contrast_vector(color)
        if wrong_color:
            item["brownContrastVector"] = wrong_color
        for key, vec in get_extra_contrast_vectors(color).items():
            if vec:
                item[key] = vec
        return item

    if category_group == "top" and top_style and color != "pattern":
        item["topStyle"] = top_style
        if is_stripe:
            item["isStripe"] = True
        item["colorVector"] = get_top_style_color_vector(color, top_style) if not is_stripe else get_stripe_color_vector(color)
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
    "skirt":  "white midi skirt with black polka dots and lace trim",
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


def _crop_to_base64(pil_img: Image.Image, quality: int = 82) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _slot_id(category_group: str, bottom_length: Optional[str] = None) -> Optional[str]:
    if category_group == "bottom" and bottom_length == "shorts":
        return "shorts"
    if category_group in TOTAL_LOOK_SLOT_ORDER:
        return category_group
    return None


def _bbox_area_fraction(bbox, img_w: int, img_h: int) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, (x2 - x1) * (y2 - y1) / max(img_w * img_h, 1))


def _detect_stripe_top_clip(crop_img: Image.Image) -> Optional[tuple]:
    """CLIP fallback when pixel rules misread a striped tee as solid white."""
    image_features = _encode_image(crop_img)
    stripe_prompts = [
        "navy and white horizontal striped short sleeve t-shirt on a model",
        "black and white horizontal striped tee shirt on a model",
        "navy blue striped cotton t-shirt with white stripes",
    ]
    white_prompts = [
        "plain white cotton t-shirt with short sleeves and crew neck no stripes",
        "solid white tee shirt with no pattern on a white background",
    ]
    stripe_s = float((image_features @ _encode_texts(stripe_prompts).T).max())
    white_s = float((image_features @ _encode_texts(white_prompts).T).max())
    if stripe_s >= white_s - 0.010 and stripe_s >= 0.22 and stripe_s > white_s + 0.012:
        color = "navy" if stripe_s >= 0.23 else "black"
        print(f"👕 Stripe top CLIP: {color}  stripe={stripe_s:.3f} white={white_s:.3f}")
        return color, True
    return None


def _refine_top_attributes(crop_img: Image.Image, fabric: str, color: str, is_stripe: bool = False) -> tuple:
    """Outdoor/studio top zones mix background — re-check centre panel for white vs black."""
    if is_stripe:
        return fabric, color, True
    w, h = crop_img.size
    panel = crop_img.crop((int(w * 0.22), int(h * 0.08), int(w * 0.78), int(h * 0.58)))
    arr = np.array(panel.resize((80, 80)).convert("RGB"), dtype=np.float32).reshape(-1, 3)
    br = arr.mean(axis=1)
    sat = arr.max(axis=1) - arr.min(axis=1)
    bright_frac = float(((br > 175) & (sat < 50)).sum()) / len(br)
    dark_frac = float((br < 72).sum()) / len(br)
    panel_avg = float(br.mean())
    if fabric == "leather":
        if bright_frac >= 0.22 and dark_frac < 0.04 and panel_avg > 195:
            return "leather", "white", False
        print(f"👕 Top refine → black leather coat (dark_frac={dark_frac:.2f}, avg={panel_avg:.0f})")
        return "leather", "black", False
    if fabric == "denim":
        return "denim", ("light_blue" if color in ("white", "grey") else color), False
    panel_std = float(br.std())
    if bright_frac >= 0.07:
        if color in ("lavender", "purple", "pink"):
            return fabric, color, False
        white_hit = _try_bright_white_top(arr, br)
        if white_hit:
            return "jersey", "white", False
        if color in ("beige", "brown", "tan"):
            beige_hit = _try_warm_beige_top(arr, br)
            if beige_hit:
                return "jersey", beige_hit[0], False
            return fabric, color, False
        if color == "white":
            return fabric, "white", False
        beige_hit = _try_warm_beige_top(arr, br)
        if beige_hit:
            return "jersey", beige_hit[0], False
        stripe = _classify_stripe_dark_pixels(arr, br)
        if stripe and _is_simple_horizontal_stripe(arr, br):
            stripe_color, stripe_flag = stripe
            print(f"👕 Top refine → {stripe_color} stripe (panel stripe check)")
            return "jersey", stripe_color, stripe_flag
        if color == "white" and panel_std > 16 and dark_frac >= 0.025:
            return fabric, color, False
        if bright_frac >= 0.80 and panel_std < 18 and dark_frac < 0.03 and color != "white":
            return fabric, color, False
        if bright_frac >= 0.80 and panel_std < 18 and dark_frac < 0.03:
            white_hit = _try_bright_white_top(arr, br)
            if white_hit:
                return "jersey", "white", False
            beige_hit = _try_warm_beige_top(arr, br)
            if beige_hit:
                return "jersey", beige_hit[0], False
        print(f"👕 Top refine → white jersey (bright_frac={bright_frac:.2f})")
        return "jersey", "white", False
    if dark_frac >= 0.40 and bright_frac < 0.08 and panel_avg < 82:
        print(f"👕 Top refine → black leather (dark_frac={dark_frac:.2f}, avg={panel_avg:.0f})")
        return "leather", "black", False
    return fabric, color, False


def _resolve_zone_force_category(zone_name: str, scores: dict, crop: Optional[Image.Image] = None) -> Optional[str]:
    if zone_name == "bottom":
        skirt_s = scores.get("skirt", 0)
        bottom_s = scores.get("bottom", 0)
        if bottom_s >= skirt_s + 0.012:
            return "bottom"
        if crop is not None:
            arr = np.array(crop.resize((80, 80)).convert("RGB"), dtype=np.float32).reshape(-1, 3)
            br = arr.mean(axis=1)
            bright_frac = float((br > 175).sum()) / len(br)
            dark_frac = float((br < 85).sum()) / len(br)
            if br.std() > 24 and bright_frac >= 0.70 and dark_frac >= 0.03:
                return "skirt"
            if br.std() > 24 and bright_frac >= 0.40 and dark_frac >= 0.04:
                return "skirt"
            if br.std() > 28 and bright_frac >= 0.12 and dark_frac >= 0.05:
                return "skirt"
        if skirt_s >= 0.17 and skirt_s >= bottom_s - 0.05:
            return "skirt"
        if max(skirt_s, bottom_s) >= 0.17:
            return "skirt" if skirt_s >= bottom_s else "bottom"
        return None
    return _ZONE_FORCE_CATEGORY.get(zone_name)


def _detect_white_vs_beige_top_clip(crop_img: Image.Image, pixel_color: str) -> str:
    """CLIP tiebreaker called when pixel rules returned "white" on a top.

    Guards against warm-lit / outdoor beige tops being misclassified as white.
    Beige pixel detections are already strict — this only corrects white→beige.
    Only overrides when CLIP has clear confidence (margin ≥ 0.018).
    """
    image_features = _encode_image(crop_img)
    white_prompts = [
        "a crisp bright white cotton t-shirt with neutral pure white color",
        "a solid white tee shirt with cool bright neutral hue no warm tones",
        "a plain bright white garment with neutral cool white color",
    ]
    beige_prompts = [
        "a warm sandy beige cotton t-shirt with earthy warm hue",
        "an earthy taupe outdoor shirt with warm khaki sandy undertone",
        "a soft sand-colored top with warm caramel neutral hue",
    ]
    white_feats = _encode_texts(white_prompts)
    beige_feats = _encode_texts(beige_prompts)
    white_s = float((image_features @ white_feats.T).max())
    beige_s = float((image_features @ beige_feats.T).max())
    print(f"🎨 White/beige CLIP: white={white_s:.3f} beige={beige_s:.3f} (pixel={pixel_color})")
    if beige_s >= white_s + 0.018:
        print(f"🎨 → overriding to beige (CLIP margin {beige_s - white_s:+.3f})")
        return "beige"
    return "white"


def _analyze_garment_crop(
    crop_img: Image.Image,
    bbox: list,
    source: str,
    yolo_label: Optional[str] = None,
    yolo_conf: float = 0.0,
    force_category: Optional[str] = None,
    length_hint_img: Optional[Image.Image] = None,
) -> Optional[dict]:
    """Run full CLIP pipeline on one crop and return a Total Look item dict."""
    category_group = force_category or detect_category_clip(crop_img)
    fabric = detect_fabric_clip(crop_img)
    color, is_stripe = get_fashion_color(crop_img, category_group)
    if fabric == "denim" and category_group == "top" and color == "white":
        color, is_stripe = "light_blue", False
    elif fabric == "denim" and category_group == "bottom" and color == "white":
        color, is_stripe = "light_blue", False
    if category_group == "belt" and fabric == "denim":
        print("⏭ Belt crop rejected — denim waist (drawstring/co-ord), not a leather belt")
        return None
    vest = None
    top_style_from_vest = None
    if category_group == "top":
        vest = None
        top_style_from_vest = None
        if fabric != "leather":
            vest = _detect_denim_vest_top_clip(crop_img)
        pastel = None
        if vest:
            fabric, color, top_style_from_vest = vest
        else:
            top_style_from_vest = None
            if fabric not in ("leather", "denim") or color in ("lavender", "purple", "pink"):
                pastel = _detect_pastel_top_color_clip(crop_img)
                if pastel:
                    color = pastel
                    fabric = "jersey"
            if not is_stripe and color == "white":
                clip_stripe = _detect_stripe_top_clip(crop_img)
                if clip_stripe:
                    color, is_stripe = clip_stripe
                    fabric = "jersey"
            fabric, color, refined_stripe = _refine_top_attributes(crop_img, fabric, color, is_stripe)
            if refined_stripe:
                is_stripe = True
            if pastel and color == "white":
                color = pastel
            if not is_stripe and color == "white":
                clip_stripe = _detect_stripe_top_clip(crop_img)
                if clip_stripe:
                    color, is_stripe = clip_stripe
                    fabric = "jersey"
            if color == "lavender":
                image_features = _encode_image(crop_img)
                white_prompts = [
                    "plain white ribbed tank crop top on model",
                    "white sleeveless high neck crop top on model",
                ]
                lav_prompts = [
                    "soft lavender v-neck t-shirt on a model",
                    "light purple heathered jersey top",
                ]
                white_s = float((image_features @ _encode_texts(white_prompts).T).max())
                lav_s = float((image_features @ _encode_texts(lav_prompts).T).max())
                if white_s >= lav_s - 0.008:
                    print(f"👕 Top CLIP refine → white (was lavender, white={white_s:.3f} lav={lav_s:.3f})")
                    color = "white"
                    fabric = "jersey"
        if color in ("light_blue", "grey") and fabric not in ("leather", "denim") and not vest:
            w, h = crop_img.size
            panel = crop_img.crop((int(w * 0.15), int(h * 0.05), int(w * 0.85), int(h * 0.60)))
            arr = np.array(panel.resize((60, 60)).convert("RGB"), dtype=np.float32).reshape(-1, 3)
            br = arr.mean(axis=1)
            if float((br > 175).sum()) / len(br) >= 0.07:
                color = "white"
                fabric = "jersey"
        if fabric == "leather" and color in ("black", "brown", "burgundy"):
            color = "black"
        if fabric == "denim" and color in ("white", "grey"):
            color = "light_blue"
        # CLIP tiebreaker: only called when pixel rules returned "white" to guard
        # against warm-lit / outdoor beige tops being misread as white.
        # Beige pixel detections (_try_warm_beige_top) are strict enough to trust.
        if color == "white" and not is_stripe and fabric not in ("denim", "leather"):
            color = _detect_white_vs_beige_top_clip(crop_img, color)

    shoe_style = detect_shoe_style_clip(crop_img) if category_group == "shoes" else None
    if category_group == "shoes":
        color = _refine_shoe_color(crop_img, color, shoe_style)
    top_style = detect_top_style_clip(crop_img) if category_group == "top" else None
    if category_group == "top" and vest:
        top_style = top_style_from_vest
    elif category_group == "top" and fabric == "denim":
        top_style = _resolve_denim_top_style(crop_img, top_style)
    elif category_group == "top" and fabric == "leather":
        top_style = "coat"
    bottom_length = detect_bottom_length_clip(crop_img) if category_group == "bottom" else None
    if category_group == "skirt":
        skirt_length, _ = detect_skirt_length_clip(crop_img)
        # When the standard zone crop (44-76%) says "mini", verify against a taller
        # crop (44-95%) that captures the full garment length.  Only upgrade if the
        # longer-length label beats mini by ≥ 0.012 on the taller crop — this rules
        # out false upgrades caused by boots/bare-legs below a real mini skirt.
        if skirt_length == "mini" and length_hint_img is not None:
            tall_length, tall_scores = detect_skirt_length_clip(length_hint_img)
            if tall_length in ("midi", "maxi"):
                mini_s = tall_scores.get("mini", 0)
                tall_s = tall_scores.get(tall_length, 0)
                if tall_s >= mini_s + 0.012:
                    print(f"📐 Skirt length upgraded: mini → {tall_length} (tall-crop margin={tall_s - mini_s:.3f})")
                    skirt_length = tall_length
    else:
        skirt_length = None
    if category_group == "skirt" and color == "grey":
        midi_s = maxi_s = 0.0
        image_features = _encode_image(crop_img)
        for length, text_feats in _skirt_length_text_features.items():
            sims = (image_features @ text_feats.T).squeeze(0)
            score = float(sims.max())
            if length == "midi":
                midi_s = score
            elif length == "maxi":
                maxi_s = score
        if maxi_s >= midi_s - 0.08:
            skirt_length = "maxi"
    if category_group == "skirt" and color == "pattern":
        midi_s = maxi_s = mini_s = 0.0
        image_features = _encode_image(crop_img)
        for length, text_feats in _skirt_length_text_features.items():
            sims = (image_features @ text_feats.T).squeeze(0)
            score = float(sims.max())
            if length == "mini":
                mini_s = score
            elif length == "midi":
                midi_s = score
            elif length == "maxi":
                maxi_s = score
        if maxi_s >= midi_s - 0.010 and maxi_s >= mini_s - 0.015:
            skirt_length = "maxi"
        elif midi_s >= mini_s - 0.010:
            skirt_length = "midi"

    slot_id = _slot_id(category_group, bottom_length)
    if not slot_id:
        return None

    cat_score = _score_categories(crop_img).get(category_group, 0.0)
    if category_group == "belt" and cat_score < 0.24:
        return None
    confidence = max(yolo_conf, cat_score)

    item = {
        "slotId": slot_id,
        "category": get_category_group(yolo_label) if yolo_label else category_group,
        "categoryGroup": category_group,
        "fabricGroup": fabric,
        "confidence": round(confidence, 4),
        "embedding": _encode_image(crop_img).cpu().numpy().flatten().tolist(),
        "color": color,
        "bbox": [round(v, 1) for v in bbox],
        "detectionSource": source,
        "cropBase64": _crop_to_base64(crop_img),
    }
    return _attach_color_vectors(
        item, color, is_stripe, category_group, fabric,
        shoe_style, bottom_length, top_style, skirt_length,
    )


def _slot_pick_score(item: dict) -> float:
    """Prefer portrait zone crops; penalize YOLO shoe crops contaminated by floor."""
    score = float(item.get("confidence", 0))
    source = item.get("detectionSource", "")
    if source.startswith("zone-"):
        score += 0.04
    if item.get("slotId") == "top" and item.get("color") == "white" and source.startswith("zone-"):
        score -= 0.12
    if item.get("slotId") == "top" and item.get("color") == "beige":
        score += 0.03
    if item.get("slotId") == "shoes" and source == "yolo" and item.get("color") in ("white", "grey"):
        score -= 0.10
    return score


def _dedupe_by_slot(candidates: list) -> dict:
    by_slot = {}
    for item in candidates:
        sid = item["slotId"]
        if sid not in by_slot or _slot_pick_score(item) > _slot_pick_score(by_slot[sid]):
            by_slot[sid] = item
    return by_slot


def _apply_dress_exclusive_rule(by_slot: dict) -> dict:
    dress = by_slot.get("dress")
    if dress and dress["confidence"] >= 0.24:
        for key in ("top", "bottom", "shorts", "skirt"):
            by_slot.pop(key, None)
    return by_slot


def _detect_full_dress(img: Image.Image) -> Optional[dict]:
    w, h = img.size
    if h <= w * 0.85:
        return None
    if _looks_like_separate_top_skirt(img):
        print("👗 Full-dress skipped — separate top + skirt look")
        return None
    full_scores = _score_categories(img)
    dress_score = full_scores.get("dress", 0.0)
    top_score = full_scores.get("top", 0.0)
    skirt_score = full_scores.get("skirt", 0.0)
    if dress_score < 0.26 or dress_score < top_score - 0.012:
        return None
    if skirt_score > dress_score + 0.015:
        print("👗 Full-dress skipped — skirt beats dress (separate top + skirt)")
        return None
    item = _analyze_garment_crop(img, [0, 0, w, h], "full-dress")
    if item and item["categoryGroup"] == "dress":
        return item
    return None


def _yolo_garment_candidates(img: Image.Image) -> tuple:
    """Return (garment_candidates, bag_boxes).

    bag_boxes is a list of (x0,y0,x1,y1) ints for every handbag / backpack /
    suitcase detected in the image.  These are passed to _zone_garment_candidates
    so zone crops can be cleaned before colour analysis.

    Two-pass design: bags are collected first so we can paint them out of the
    image before cropping individual garments.  This prevents the bag handle
    (visible inside the vest YOLO box, for example) from contaminating the
    garment's colour statistics.
    """
    results = yolo_model(img)

    # Pass 1 — collect bag boxes and raw garment detections
    bag_boxes = []
    garment_detections = []
    for r in results:
        for box in r.boxes:
            label = yolo_model.names[int(box.cls)].lower()
            conf = float(box.conf)
            if conf < 0.25:
                continue
            coords = box.xyxy[0].tolist()
            if label in _BAG_YOLO_CLASSES:
                bag_boxes.append(tuple(int(c) for c in coords))
            elif label not in _IGNORED_YOLO_CLASSES:
                garment_detections.append((label, conf, coords))

    if bag_boxes:
        print(f"🎒 Bag occlusion: masking {len(bag_boxes)} accessory box(es) from YOLO + zone crops")

    # Create bag-masked image so garment crops are clean
    clean_img = _paint_out_boxes(img, bag_boxes)

    # Pass 2 — analyse each garment crop using the cleaned image
    candidates = []
    for label, conf, coords in garment_detections:
        crop = clean_img.crop(tuple(coords))
        item = _analyze_garment_crop(
            crop, coords, "yolo",
            yolo_label=label, yolo_conf=conf,
        )
        if item:
            candidates.append(item)

    return candidates, bag_boxes


_ZONE_FORCE_CATEGORY = {
    "top": "top",
    "belt": "belt",
    "bottom": "bottom",
    "shoes": "shoes",
}


def _filter_spurious_belt(by_slot: dict) -> dict:
    """Drop belt when the waist crop is really denim shorts/pants (drawstring, co-ord)."""
    belt = by_slot.get("belt")
    if not belt:
        return by_slot
    if belt.get("fabricGroup") == "denim":
        print("⏭ Dropping belt slot — denim waist, not a leather belt")
        del by_slot["belt"]
        return by_slot
    bottom_item = by_slot.get("shorts") or by_slot.get("bottom")
    if bottom_item and belt.get("color") == bottom_item.get("color"):
        if bottom_item.get("fabricGroup") == "denim" or belt.get("confidence", 0) < 0.28:
            print("⏭ Dropping belt slot — waist matches bottom (no separate belt)")
            del by_slot["belt"]
    return by_slot


def _is_matching_coord_set(top: dict, bottom_item: dict) -> bool:
    return (
        top.get("fabricGroup") == "denim"
        and bottom_item.get("fabricGroup") == "denim"
        and top.get("color") == bottom_item.get("color")
    )


def _pick_primary_person_bbox(img: Image.Image) -> Optional[tuple]:
    """Largest YOLO person box — used for zone crops on landscape uploads."""
    results = yolo_model(img)
    best_coords = None
    best_area = 0.0
    for r in results:
        for box in r.boxes:
            label = yolo_model.names[int(box.cls)].lower()
            if label != "person":
                continue
            conf = float(box.conf)
            if conf < 0.35:
                continue
            coords = box.xyxy[0].tolist()
            area = (coords[2] - coords[0]) * (coords[3] - coords[1])
            if area > best_area:
                best_area = area
                best_coords = tuple(coords)
    return best_coords


def _zone_garment_candidates(
    img: Image.Image,
    region: Optional[tuple] = None,
    bag_boxes: Optional[list] = None,
) -> list:
    w, h = img.size
    if region is None:
        if h <= w * 0.85:
            return []
        rx0, ry0, rx1, ry1 = 0, 0, w, h
    else:
        rx0, ry0, rx1, ry1 = region
    rw = rx1 - rx0
    rh = ry1 - ry0
    if rw < 8 or rh < 8:
        return []

    # Paint detected bags/backpacks out of the image before cropping zones.
    # The fill colour (245,245,245) is above the bright-background threshold
    # in get_fashion_color, so painted pixels are excluded from garment colour
    # statistics — they never inflate dark_frac or distort avg_brightness.
    base_img = _paint_out_boxes(img, bag_boxes or [])

    candidates = []
    for zone_name, y0, y1, allowed_slots in _TOTAL_LOOK_ZONES:
        bbox = [
            int(rx0 + rw * 0.05),
            int(ry0 + rh * y0),
            int(rx0 + rw * 0.95),
            int(ry0 + rh * y1),
        ]
        crop = base_img.crop(tuple(bbox))
        scores = _score_categories(crop)
        if zone_name == "belt":
            belt_s = scores.get("belt", 0)
            bottom_s = scores.get("bottom", 0)
            if belt_s < 0.24 or belt_s < bottom_s + 0.04:
                continue
            if detect_fabric_clip(crop) == "denim":
                continue
        elif zone_name == "bottom":
            if max(scores.get("skirt", 0), scores.get("bottom", 0)) < 0.17:
                continue
        elif max(scores.values()) < 0.21:
            continue
        forced = _resolve_zone_force_category(zone_name, scores, crop)
        if not forced:
            continue
        # For the bottom/skirt zone, pass a taller crop (44%→95%) as a length
        # hint so detect_skirt_length_clip sees the full length of long skirts.
        # The standard zone crop stops at 76%, cutting off ankle-length skirts
        # and causing them to be mis-labelled as mini.
        length_hint = None
        if zone_name == "bottom":
            tall_bbox = [
                int(rx0 + rw * 0.05),
                int(ry0 + rh * 0.44),
                int(rx0 + rw * 0.95),
                int(ry0 + rh * 0.95),
            ]
            length_hint = base_img.crop(tuple(tall_bbox))
        item = _analyze_garment_crop(
            crop, bbox, f"zone-{zone_name}", force_category=forced,
            length_hint_img=length_hint,
        )
        if item and item["slotId"] in allowed_slots:
            candidates.append(item)
    return candidates


def _finalize_total_look_slots(by_slot: dict) -> list:
    """Pick final slots — co-ord sets, belted looks, or default ordered items (up to 4)."""
    belt = by_slot.get("belt")
    top = by_slot.get("top")
    shorts = by_slot.get("shorts")
    bottom = by_slot.get("bottom")
    skirt = by_slot.get("skirt")
    shoes = by_slot.get("shoes")
    bottom_item = shorts or bottom

    if top and shorts and shoes and not skirt and not belt:
        print("👗 Look archetype: top + shorts + shoes")
        return [top, shorts, shoes]

    if top and skirt and shoes and not belt and not bottom_item:
        print("👗 Look archetype: top + skirt + shoes")
        return [top, skirt, shoes]

    if top and belt and skirt and belt["confidence"] >= 0.17:
        items = [top, belt, skirt]
        if shoes:
            items.append(shoes)
        print(f"👗 Look archetype: top + belt + skirt (+ shoes={bool(shoes)})")
        return items

    if top and belt and shorts and not skirt and belt["confidence"] >= 0.17:
        print("👗 Look archetype: top + belt + shorts (waist-detail outfit)")
        return [top, belt, shorts]

    if top and bottom_item and _is_matching_coord_set(top, bottom_item):
        print("👗 Look archetype: denim co-ord set → top + bottom only")
        return [top, bottom_item]

    items = []
    for slot_id in TOTAL_LOOK_SLOT_ORDER:
        if slot_id in by_slot:
            items.append(by_slot[slot_id])
        if len(items) >= MAX_TOTAL_LOOK_ITEMS:
            break
    return items


def process_total_look_logic(img: Image.Image) -> dict:
    """Detect multiple garments in one full-body image for Find Total Look."""
    w, h = img.size
    methods = []
    person_bbox = _pick_primary_person_bbox(img) if h <= w * 0.85 else None

    dress_item = _detect_full_dress(img)
    if dress_item:
        return {
            "items": [dress_item],
            "detectionMeta": {
                "method": ["full-dress"],
                "rawCandidateCount": 1,
                "garmentCount": 1,
                "slots": ["dress"],
            },
        }

    candidates = []
    yolo_cands, bag_boxes = _yolo_garment_candidates(img)
    if yolo_cands:
        methods.append("yolo")
        candidates.extend(yolo_cands)

    if h > w * 0.85:
        zone_cands = _zone_garment_candidates(img, bag_boxes=bag_boxes)
        if zone_cands:
            methods.append("zone")
            candidates.extend(zone_cands)
    elif person_bbox:
        zone_cands = _zone_garment_candidates(img, region=person_bbox, bag_boxes=bag_boxes)
        if zone_cands:
            methods.append("person-zone")
            candidates.extend(zone_cands)

    raw_count = len(candidates)

    if not candidates:
        methods.append("full-fallback")
        fallback_bbox = [0, 0, w, h]
        fallback_img = img
        if person_bbox:
            x0, y0, x1, y1 = person_bbox
            pw, ph = x1 - x0, y1 - y0
            fallback_bbox = [
                int(x0 + pw * 0.05),
                int(y0 + ph * 0.05),
                int(x1 - pw * 0.05),
                int(y0 + ph * 0.42),
            ]
            fallback_img = img.crop(tuple(fallback_bbox))
        fallback = _analyze_garment_crop(
            fallback_img, fallback_bbox, "full-fallback", force_category="top",
        )
        items = [fallback] if fallback else []
    else:
        by_slot = _apply_dress_exclusive_rule(_dedupe_by_slot(candidates))
        by_slot = _filter_spurious_belt(by_slot)
        items = _finalize_total_look_slots(by_slot)

    return {
        "items": items,
        "detectionMeta": {
            "method": methods or ["none"],
            "rawCandidateCount": raw_count,
            "garmentCount": len(items),
            "slots": [i["slotId"] for i in items],
        },
    }


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
        # Collect bag boxes first so garment crops can be cleaned
        single_bag_boxes = []
        for r in results:
            for box in r.boxes:
                lbl = yolo_model.names[int(box.cls)].lower()
                if float(box.conf) > 0.2 and lbl in _BAG_YOLO_CLASSES:
                    single_bag_boxes.append(tuple(int(c) for c in box.xyxy[0].tolist()))
        clean_img_single = _paint_out_boxes(img, single_bag_boxes)

        for r in results:
            for box in r.boxes:
                label = yolo_model.names[int(box.cls)]
                conf = float(box.conf)
                if conf > 0.2:
                    coords = box.xyxy[0].tolist()
                    crop_img = clean_img_single.crop((coords[0], coords[1], coords[2], coords[3]))

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


@app.post("/process-total-look-base64")
async def process_total_look_base64(data: ImageRequest):
    """Find Total Look: detect all garments in one full-body image."""
    base64_data = data.image.split(",")[1] if "," in data.image else data.image
    img_bytes = base64.b64decode(base64_data)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return process_total_look_logic(img)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

#source venv/Scripts/activate
#pip install fastapi uvicorn python-multipart transformers accelerate
#python main.py
