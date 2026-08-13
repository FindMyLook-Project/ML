"""
Service layer for orchestrating the Total Look logic.
Handles image cropping, YOLO detection, zone logic, deduplication,
and assembling the final list of detected garments.
"""

import io
import base64
import numpy as np
from PIL import Image
from typing import Optional, Tuple, List, Dict
from services.pose_estimator import extract_anatomical_zones

# Import internal ML and analysis services
from services.ml_service import yolo_model, encode_image, encode_texts
from services.garment_classifier import (
    score_categories, looks_like_separate_top_skirt, detect_fabric_clip,
    detect_bottom_length_clip, detect_skirt_length_clip, detect_top_style_clip,
    resolve_denim_top_style, detect_denim_vest_top_clip
)
from services.color_analyzer import (
    get_fashion_color, refine_top_attributes, foot_skin_and_dark
)
from services.vector_builder import attach_color_vectors
from config.prompts import (
    TOTAL_LOOK_SLOT_ORDER, MAX_TOTAL_LOOK_ITEMS, TOTAL_LOOK_ZONES,
    BAG_YOLO_CLASSES, IGNORED_YOLO_CLASSES, ZONE_FORCE_CATEGORY,
    CATEGORY_MAPPING, COLOR_TEXT_PROMPTS, PATTERN_TEXTURE_PROMPTS
)

# -------------------------------------------------------------------------
# 1. Image Processing & BBox Helpers
# -------------------------------------------------------------------------
def paint_out_boxes(img: Image.Image, boxes: list, fill: tuple = (245, 245, 245)) -> Image.Image:
    """Mask detected accessory boxes to prevent them from contaminating garment color."""
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

def crop_to_base64(pil_img: Image.Image, quality: int = 82) -> str:
    """Convert a cropped PIL image to a base64 JPEG string for the frontend."""
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

def get_category_group(yolo_label: str) -> str:
    return CATEGORY_MAPPING.get(yolo_label.lower(), "other")

def get_slot_id(category_group: str, bottom_length: Optional[str] = None) -> Optional[str]:
    if category_group == "bottom" and bottom_length == "shorts":
        return "shorts"
    if category_group in TOTAL_LOOK_SLOT_ORDER:
        return category_group
    return None

def pick_primary_person_bbox(img: Image.Image) -> Optional[tuple]:
    """Find the largest person bounding box for zone crops on landscape uploads."""
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

# -------------------------------------------------------------------------
# 2. CLIP-Based Context Overrides
# -------------------------------------------------------------------------
def shorts_with_belt_override(pil_img: Image.Image) -> bool:
    w, h = pil_img.size
    if h <= w * 0.85:
        return False
    lower = pil_img.crop((0, int(h * 0.32), w, h))
    lower_scores = score_categories(lower)
    if lower_scores.get("bottom", 0) < 0.26:
        return False
    length = detect_bottom_length_clip(pil_img)
    return length == "shorts"

def feet_shoes_override(pil_img: Image.Image) -> Tuple[bool, dict]:
    w, h = pil_img.size
    feet = pil_img.crop((int(w * 0.05), int(h * 0.38), int(w * 0.95), h))
    feet_scores = score_categories(feet)
    shoes = feet_scores.get("shoes", 0)
    rival = max(feet_scores.get("bottom", 0), feet_scores.get("top", 0), feet_scores.get("dress", 0), feet_scores.get("skirt", 0))
    if shoes >= rival + 0.008 and shoes >= 0.22:
        return True, feet_scores
    return False, feet_scores

def detect_category_clip(pil_img: Image.Image) -> str:
    """Zero-shot fashion-clip classification for primary category."""
    w, h = pil_img.size
    belt_bands = [
        pil_img.crop((int(w * 0.05), int(h * 0.34), int(w * 0.95), int(h * 0.58))),
        pil_img.crop((int(w * 0.05), int(h * 0.22), int(w * 0.95), int(h * 0.78))),
    ]
    if w >= h * 0.85:
        belt_bands.append(pil_img)

    belt_score, bottom_score, top_score = 0.0, 0.0, 0.0
    waist_scores = {}
    
    for band in belt_bands:
        band_scores = score_categories(band)
        belt_score = max(belt_score, band_scores.get("belt", 0))
        bottom_score = max(bottom_score, band_scores.get("bottom", 0))
        top_score = max(top_score, band_scores.get("top", 0))
        if not waist_scores or band_scores.get("belt", 0) >= waist_scores.get("belt", 0):
            waist_scores = band_scores

    if belt_score >= bottom_score + 0.005 and belt_score >= top_score - 0.005:
        if shorts_with_belt_override(pil_img):
            print("📐 Belt skipped — shorts-with-belt override (garment is bottom/shorts)")
        elif h > w * 1.2:
            print("📐 Belt skipped — image is tall, likely trousers/jeans, not a belt")
        else:
            return "belt"

    feet_override, feet_scores = feet_shoes_override(pil_img)
    if feet_override:
        return "shoes"

    if h > w * 0.85:
        full_scores = score_categories(pil_img)
        upper_scores = score_categories(pil_img.crop((0, 0, w, int(h * 0.55))))
        lower_scores = score_categories(pil_img.crop((0, int(h * 0.42), w, h)))

        lower_best = max(lower_scores, key=lower_scores.get)
        upper_best = max(upper_scores, key=upper_scores.get)

        if lower_best == "shoes" and lower_scores["shoes"] >= lower_scores.get("bottom", 0) + 0.01:
            best_cat = "shoes"
            all_scores = lower_scores
        elif (full_scores["dress"] >= full_scores["top"] - 0.005
              and full_scores["dress"] >= full_scores.get("bottom", 0) + 0.015
              and full_scores["dress"] >= full_scores.get("skirt", 0) + 0.015):
            best_cat = "dress"
            all_scores = full_scores
        elif upper_best == "top" and upper_scores["top"] > upper_scores.get("bottom", 0) + 0.01:
            if full_scores["dress"] < full_scores["top"] - 0.02:
                best_cat = "top"
                all_scores = upper_scores
            else:
                best_cat = "dress"
                all_scores = full_scores
        else:
            merged = {cat: max(lower_scores.get(cat, 0), upper_scores.get(cat, 0), full_scores.get(cat, 0)) for cat in lower_scores}
            best_cat = max(merged, key=merged.get)
            all_scores = merged
    else:
        all_scores = score_categories(pil_img)
        best_cat = max(all_scores, key=all_scores.get)

    if best_cat == "belt":
        if shorts_with_belt_override(pil_img):
            best_cat = "bottom"
        elif h > w * 1.2:
            best_cat = "bottom"
        elif detect_fabric_clip(pil_img) == "denim":
            best_cat = "bottom"
            
    if best_cat == "skirt" and detect_fabric_clip(pil_img) == "leather":
        top_s = all_scores.get("top", 0.0)
        skirt_s = all_scores.get("skirt", 0.0)
        if top_s >= skirt_s - 0.04:
            best_cat = "top"

    return best_cat

def resolve_zone_force_category(zone_name: str, scores: dict, crop: Optional[Image.Image] = None) -> Optional[str]:
    """Force categories based on predefined crop zones."""
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
    return ZONE_FORCE_CATEGORY.get(zone_name)

def looks_like_flip_flop(pil_img: Image.Image) -> bool:
    """Helper for shoe style classification."""
    w, h = pil_img.size
    foot = pil_img.crop((int(w * 0.08), int(h * 0.30), int(w * 0.92), h))
    arr = np.array(foot.resize((72, 48)).convert("RGB"), dtype=np.float32)
    br = arr.mean(axis=2).reshape(-1)
    sat = (arr.max(axis=2) - arr.min(axis=2)).reshape(-1)
    skin_frac = float(((br > 105) & (br < 215) & (sat > 8)).sum()) / len(br)
    dark_frac = float((br < 72).sum()) / len(br)
    return skin_frac >= 0.15 and 0.003 <= dark_frac <= 0.35

def detect_shoe_style_clip(pil_img: Image.Image) -> str:
    from config.prompts import CLIP_SHOE_STYLE_PROMPTS
    from services.ml_service import shoe_style_text_features
    w, h = pil_img.size
    cy0 = int(h * 0.30) if h > w * 0.6 else int(h * 0.10)
    crops = [pil_img.crop((int(w * 0.05), cy0, int(w * 0.95), h))]
    if h > w * 0.55:
        crops.append(pil_img.crop((0, int(h * 0.12), w, h)))

    scores = {}
    for style, text_feats in shoe_style_text_features.items():
        style_best = -1.0
        for crop in crops:
            sims = (encode_image(crop) @ text_feats.T).squeeze(0)
            style_best = max(style_best, float(sims.max()))
        scores[style] = style_best

    open_shoe_styles = ("espadrille", "flat_shoe", "slide_sandal", "birkenstock", "puffy_slide", "heeled_sandal", "flip_flop")
    boot_score = scores.get("heeled_boot", 0.0)
    flat_score = scores.get("flat_shoe", 0.0)
    slide_score = max(scores.get("slide_sandal", 0.0), scores.get("puffy_slide", 0.0))
    open_best_style = max(open_shoe_styles, key=lambda s: scores.get(s, 0.0))
    open_best = scores.get(open_best_style, 0.0)
    skin_frac, dark_frac = foot_skin_and_dark(pil_img)
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
        elif looks_like_flip_flop(pil_img) and best_style in ("slide_sandal", "espadrille", "birkenstock", "puffy_slide") and boot_score < 0.28:
            best_style = "flip_flop"

    print(f"👟 Shoe style: {best_style}")
    return best_style

def detect_pastel_top_color_clip(crop_img: Image.Image) -> Optional[str]:
    image_features = encode_image(crop_img)
    
    # 1. Legacy robust check for lavender
    lavender_prompts = ["a soft lavender v-neck t-shirt on a model", "a light purple heathered jersey top", "soft dusty lavender mauve purple cotton v-neck t-shirt"]
    white_prompts = ["plain white cotton t-shirt with short sleeves", "a plain crew-neck t-shirt on a white background"]
    
    lavender_s = float((image_features @ encode_texts(lavender_prompts).T).max())
    white_s = float((image_features @ encode_texts(white_prompts).T).max())
    if lavender_s >= white_s + 0.012 and lavender_s >= 0.24:
        return "lavender"
        
    # 2. Advanced check for all pastels vs neutrals
    scores = {}
    for color in ("lavender", "purple", "pink", "yellow", "white", "beige"):
        prompts = COLOR_TEXT_PROMPTS.get(color, [color])
        scores[color] = float((image_features @ encode_texts(prompts[:3]).T).max())
        
    best_pastel = max(("lavender", "purple", "pink", "yellow"), key=lambda c: scores[c])
    rival = max(scores["white"], scores["beige"])
    
    # Only override if the AI is highly confident it's a pastel/yellow and NOT white or beige
    if scores[best_pastel] >= rival + 0.008 and scores[best_pastel] >= 0.22:
        return best_pastel
        
    return None

def detect_white_vs_beige_top_clip(crop_img: Image.Image, pixel_color: str) -> str:
    image_features = encode_image(crop_img)
    white_prompts = ["a crisp bright white cotton t-shirt", "a solid white tee shirt", "a plain bright white garment"]
    beige_prompts = ["a warm sandy beige cotton t-shirt", "an earthy taupe outdoor shirt", "a soft sand-colored top"]
    yellow_prompts = ["a soft pale yellow satin top", "a light buttery yellow halter top", "a pale yellow pastel garment"]
    
    white_s = float((image_features @ encode_texts(white_prompts).T).max())
    beige_s = float((image_features @ encode_texts(beige_prompts).T).max())
    yellow_s = float((image_features @ encode_texts(yellow_prompts).T).max())
    
    if yellow_s >= white_s + 0.010 and yellow_s >= beige_s + 0.005:
        return "yellow"
        
    if beige_s >= white_s + 0.018:
        return "beige"
        
    return "white"

def detect_stripe_top_clip(crop_img: Image.Image) -> Optional[Tuple[str, bool]]:
    image_features = encode_image(crop_img)
    stripe_prompts = ["navy and white horizontal striped short sleeve t-shirt", "black and white horizontal striped tee shirt", "navy blue striped cotton t-shirt"]
    white_prompts = ["plain white cotton t-shirt", "solid white tee shirt"]
    
    stripe_s = float((image_features @ encode_texts(stripe_prompts).T).max())
    white_s = float((image_features @ encode_texts(white_prompts).T).max())
    if stripe_s >= white_s - 0.010 and stripe_s >= 0.22 and stripe_s > white_s + 0.012:
        return ("navy" if stripe_s >= 0.23 else "black", True)
    return None

# -------------------------------------------------------------------------
# 3. Core Processing Pipeline (Analysis & Cropping)
# -------------------------------------------------------------------------
def analyze_garment_crop(crop_img: Image.Image, bbox: list, source: str, yolo_label: Optional[str] = None, yolo_conf: float = 0.0, force_category: Optional[str] = None, length_hint_img: Optional[Image.Image] = None) -> Optional[dict]:
    """Run full pipeline on a single crop to produce a Total Look item."""
    category_group = force_category or detect_category_clip(crop_img)
    fabric = detect_fabric_clip(crop_img)

    if category_group == "shoes":
        if fabric == "sequin":
            return None
        skin_frac, dark_frac = foot_skin_and_dark(crop_img)
        if skin_frac > 0.40 and dark_frac < 0.05:
            return None
        if source == "yolo" and score_categories(crop_img).get("shoes", 0.0) < 0.17:
            return None

    color, is_stripe = get_fashion_color(crop_img, category_group, fabric)
    
    if fabric == "denim" and category_group in ("top", "bottom") and color == "white":
        color, is_stripe = "light_blue", False
    if category_group == "belt" and fabric == "denim":
        return None

    vest = None
    top_style_from_vest = None
    
    if category_group == "top":
        if fabric != "leather":
            vest = detect_denim_vest_top_clip(crop_img)
        if vest:
            fabric, color, top_style_from_vest = vest
        else:
            if fabric not in ("leather", "denim") or color in ("lavender", "purple", "pink"):
                pastel = detect_pastel_top_color_clip(crop_img)
                if pastel:
                    color, fabric = pastel, "jersey"
                    
            if not is_stripe and color == "white":
                clip_stripe = detect_stripe_top_clip(crop_img)
                if clip_stripe:
                    color, is_stripe = clip_stripe[0], clip_stripe[1]
                    fabric = "jersey"
                    
            fabric, color, refined_stripe = refine_top_attributes(crop_img, fabric, color, is_stripe)
            if refined_stripe:
                is_stripe = True
                
            if color == "lavender":
                image_features = encode_image(crop_img)
                white_s = float((image_features @ encode_texts(["plain white ribbed tank crop top on model", "white sleeveless high neck crop top"]).T).max())
                lav_s = float((image_features @ encode_texts(["soft lavender v-neck t-shirt", "light purple heathered jersey top"]).T).max())
                if white_s >= lav_s - 0.008:
                    color, fabric = "white", "jersey"
                    
        if color in ("light_blue", "grey") and fabric not in ("leather", "denim") and not vest:
            w, h = crop_img.size
            panel = crop_img.crop((int(w * 0.15), int(h * 0.05), int(w * 0.85), int(h * 0.60)))
            arr = np.array(panel.resize((60, 60)).convert("RGB"), dtype=np.float32).reshape(-1, 3)
            br = arr.mean(axis=1)
            if float((br > 175).sum()) / len(br) >= 0.07:
                color, fabric = "white", "jersey"
                
        if fabric == "leather" and color in ("black", "brown", "burgundy", "grey"):
            color = "black"
        if fabric == "denim" and color in ("white", "grey"):
            color = "light_blue"
            
        if color == "white" and not is_stripe and fabric not in ("denim", "leather"):
            color = detect_white_vs_beige_top_clip(crop_img, color)

    shoe_style = detect_shoe_style_clip(crop_img) if category_group == "shoes" else None
    top_style = detect_top_style_clip(crop_img) if category_group == "top" else None
    
    if category_group == "top":
        if vest:
            top_style = top_style_from_vest
        elif fabric == "denim":
            top_style = resolve_denim_top_style(crop_img, top_style)
        elif fabric == "leather":
            top_style = "coat"
            
    bottom_length = detect_bottom_length_clip(crop_img) if category_group == "bottom" else None
    
    if category_group == "skirt":
        skirt_length, _ = detect_skirt_length_clip(crop_img)
        if skirt_length == "mini" and length_hint_img is not None:
            tall_length, tall_scores = detect_skirt_length_clip(length_hint_img)
            if tall_length in ("midi", "maxi"):
                if tall_scores.get(tall_length, 0) >= tall_scores.get("mini", 0) + 0.012:
                    skirt_length = tall_length
    else:
        skirt_length = None

    slot_id = get_slot_id(category_group, bottom_length)
    if not slot_id:
        return None

    cat_score = score_categories(crop_img).get(category_group, 0.0)
    if category_group == "belt" and cat_score < 0.24:
        return None
    confidence = max(yolo_conf, cat_score)

    final_category = get_category_group(yolo_label) if yolo_label else category_group
    if category_group == "top" and top_style == "coat":
        final_category = "leather_jacket" if fabric == "leather" else "jacket"
    elif category_group == "top" and top_style == "shirt":
        final_category = "shirt"

    item = {
        "slotId": slot_id,
        "category": final_category, 
        "categoryGroup": category_group,
        "fabricGroup": fabric,
        "confidence": round(confidence, 4),
        "embedding": encode_image(crop_img).cpu().numpy().flatten().tolist(),
        "color": color,
        "bbox": [round(v, 1) for v in bbox],
        "detectionSource": source,
        "cropBase64": crop_to_base64(crop_img),
    }

    # 🛑 --- Protected Confidence Boost for Bright Bottoms --- 🛑
    if category_group in ("bottom", "shorts", "skirt") and color in ("white", "beige", "light-blue", "cream"):
        item["confidence"] = min(1.0, round(item["confidence"] + 0.12, 4))
    # ---------------------------------------------------------

    return attach_color_vectors(item, color, is_stripe, category_group, fabric, shoe_style, bottom_length, top_style, skirt_length)

# -------------------------------------------------------------------------
# 4. Orchestration & Filtering Logic
# -------------------------------------------------------------------------
def slot_pick_score(item: dict) -> float:
    score = float(item.get("confidence", 0))
    source = item.get("detectionSource", "")
    if source.startswith("zone-"): score += 0.04
    if item.get("slotId") == "top" and item.get("color") == "white" and source.startswith("zone-"): score -= 0.12
    if item.get("slotId") == "top" and item.get("color") == "beige": score += 0.03
    if item.get("slotId") == "shoes" and source == "yolo" and item.get("color") in ("white", "grey"): score -= 0.10
    return score

def dedupe_by_slot(candidates: list) -> dict:
    by_slot = {}
    for item in candidates:
        sid = item["slotId"]
        if sid not in by_slot or slot_pick_score(item) > slot_pick_score(by_slot[sid]):
            by_slot[sid] = item
    return by_slot

def apply_dress_exclusive_rule(by_slot: dict) -> dict:
    dress = by_slot.get("dress")
    if dress and dress["confidence"] >= 0.24:
        for key in ("top", "bottom", "shorts", "skirt"):
            by_slot.pop(key, None)
    return by_slot

def filter_spurious_belt(by_slot: dict) -> dict:
    belt = by_slot.get("belt")
    if not belt: return by_slot
    if belt.get("fabricGroup") == "denim":
        del by_slot["belt"]
        return by_slot
    bottom_item = by_slot.get("shorts") or by_slot.get("bottom")
    if bottom_item and belt.get("color") == bottom_item.get("color"):
        if bottom_item.get("fabricGroup") == "denim" or belt.get("confidence", 0) < 0.28:
            del by_slot["belt"]
    return by_slot

def detect_match_set_vs_one_piece(pil_img: Image.Image, top_bbox: list, bottom_bbox: list) -> str:
    """Uses CLIP on the waistline to differentiate a 2-piece set from a 1-piece jumpsuit."""
    w, h = pil_img.size
    
    # 1. Crop only the waist area - the seam between top and bottom
    y0 = max(0, int(top_bbox[1] + (top_bbox[3] - top_bbox[1]) * 0.6))
    y1 = min(h, int(bottom_bbox[1] + (bottom_bbox[3] - bottom_bbox[1]) * 0.4))
    
    # Fallback protection in case bounding boxes overlap too much or collide
    if y1 <= y0 + 10:
        y0, y1 = int(h * 0.40), int(h * 0.60)
        
    waist_crop = pil_img.crop((0, y0, w, y1))
    image_features = encode_image(waist_crop)
    
    # 2. Ask the model what it sees at the intersection
    one_piece_prompts = ["a one-piece romper", "a one-piece jumpsuit", "connected fabric at the waist", "a dress"]
    two_piece_prompts = ["a two-piece matching set", "bare midriff skin between top and bottom", "separate top and shorts set", "two-piece co-ord"]
    
    one_piece_score = float((image_features @ encode_texts(one_piece_prompts).T).max())
    two_piece_score = float((image_features @ encode_texts(two_piece_prompts).T).max())
    
    print(f"🔍 Match-Set Check: two_piece={two_piece_score:.3f}, one_piece={one_piece_score:.3f}")
    
    if two_piece_score >= one_piece_score + 0.005:
        return "match_set"
    return "one_piece"

def merge_to_one_piece(candidates: list, img: Image.Image = None) -> list:
    """
    Hybrid logic: Merges top and bottom using text color match AND visual similarity.
    Includes a high-similarity override to catch identical patterned fabrics that 
    were misclassified by basic pixel color rules.
    """
    top_item = None
    bottom_item = None
    
    for c in candidates:
        if c.get("slotId") == "top":
            top_item = c
        elif c.get("slotId") in ["bottom", "shorts", "skirt"]:
            bottom_item = c
            
    if not top_item or not bottom_item:
        return candidates
        
    top_color = top_item.get("color")
    bottom_color = bottom_item.get("color")
    
    # 1. Textual Match
    text_match = (top_color and bottom_color and top_color == bottom_color)
    
    # 2. Visual Similarity Match (Calculate regardless of text match)
    import numpy as np
    top_emb = np.array(top_item.get("embedding", []))
    bot_emb = np.array(bottom_item.get("embedding", []))
    similarity = 0.0
    
    if len(top_emb) > 0 and len(bot_emb) > 0:
        similarity = float(np.dot(top_emb, bot_emb) / (np.linalg.norm(top_emb) * np.linalg.norm(bot_emb)))
        print(f"🔍 Hybrid Check - Visual Similarity: {similarity:.3f}")

    # 3. Decision Logic
    is_matching = False
    
    if text_match and similarity >= 0.76:
        # Standard merge: colors match in text, and they look reasonably similar
        print("✨ Standard Merge Triggered (Text + Similarity)")
        is_matching = True
    elif similarity > 0.77:
        # Override merge: colors text differ (e.g. white vs grey), but the fabric is virtually identical
        print("🚀 Override Merge Triggered! Visual similarity is extremely high despite text differences.")
        is_matching = True

    if is_matching:
        bottom_cat = bottom_item.get("category", "")
        bottom_slot = bottom_item.get("slotId", "")
        
        if bottom_cat == "shorts" or bottom_slot == "shorts":
            new_category = "romper"
        elif bottom_cat in ["long_pants", "pants", "bottom"] or bottom_slot == "bottom":
            new_category = "jumpsuit"
        else:
            new_category = "dress"
            
        print(f"👗 Transforming Outfit into One-Piece: {new_category}")
        
        top_item["category"] = new_category
        top_item["slotId"] = "dress"
        top_item["categoryGroup"] = "dress"
        
        if "topStyle" in top_item:
            del top_item["topStyle"]
            
        return [c for c in candidates if c != bottom_item]
        
    return candidates

def yolo_garment_candidates(img: Image.Image) -> tuple:
    results = yolo_model(img)
    bag_boxes = []
    garment_detections = []
    for r in results:
        for box in r.boxes:
            label = yolo_model.names[int(box.cls)].lower()
            conf = float(box.conf)
            if conf < 0.25: continue
            coords = box.xyxy[0].tolist()
            if label in BAG_YOLO_CLASSES:
                bag_boxes.append(tuple(int(c) for c in coords))
            elif label not in IGNORED_YOLO_CLASSES:
                garment_detections.append((label, conf, coords))

    clean_img = paint_out_boxes(img, bag_boxes)
    candidates = []
    for label, conf, coords in garment_detections:
        crop = clean_img.crop(tuple(coords))
        item = analyze_garment_crop(crop, coords, "yolo", yolo_label=label, yolo_conf=conf)
        if item: candidates.append(item)
    return candidates, bag_boxes

def zone_garment_candidates(img: Image.Image, region: Optional[tuple] = None, bag_boxes: Optional[list] = None) -> list:
    w, h = img.size
    if region is None:
        if h <= w * 0.85: return []
        rx0, ry0, rx1, ry1 = 0, 0, w, h
    else:
        rx0, ry0, rx1, ry1 = region
        
    rw, rh = rx1 - rx0, ry1 - ry0
    if rw < 8 or rh < 8: return []
    base_img = paint_out_boxes(img, bag_boxes or [])
    
    person_img = base_img.crop((rx0, ry0, rx1, ry1))
    
    dynamic_zones = extract_anatomical_zones(person_img)
    
    zones_to_process = []
    
    if dynamic_zones:
        print("🧍 Using intelligent anatomical zones from MediaPipe!")
        allowed_slots_map = {
            "top": {"top"},
            "bottom": {"bottom", "shorts", "skirt"},
            "shoes": {"shoes"}
        }
        for z_name, (x0, y0, x1, y1) in dynamic_zones.items():
            abs_bbox = [rx0 + x0, ry0 + y0, rx0 + x1, ry0 + y1]
            zones_to_process.append((z_name, abs_bbox, allowed_slots_map[z_name]))
            
        belt_y0, belt_y1 = 0.42, 0.55
        belt_bbox = [int(rx0 + rw * 0.05), int(ry0 + rh * belt_y0), int(rx0 + rw * 0.95), int(ry0 + rh * belt_y1)]
        zones_to_process.append(("belt", belt_bbox, {"belt"}))
        
    else:
        print("⚠️ MediaPipe failed, falling back to static percentage zones.")
        for zone_name, y0, y1, allowed_slots in TOTAL_LOOK_ZONES:
            bbox = [int(rx0 + rw * 0.05), int(ry0 + rh * y0), int(rx0 + rw * 0.95), int(ry0 + rh * y1)]
            zones_to_process.append((zone_name, bbox, allowed_slots))
            
    candidates = []
    
    for zone_name, bbox, allowed_slots in zones_to_process:
        crop = base_img.crop(tuple(bbox))
        scores = score_categories(crop)
        
        if zone_name == "shoes":
            shoe_s, rival_s = scores.get("shoes", 0.0), max(scores.get("bottom", 0.0), scores.get("skirt", 0.0))
            if rival_s > shoe_s + 0.01 or shoe_s < 0.17: continue
            skin_frac, dark_frac = foot_skin_and_dark(crop)
            if skin_frac > 0.45 and dark_frac < 0.05 and shoe_s < 0.25: continue
        elif zone_name == "belt":
            belt_s, bottom_s = scores.get("belt", 0), scores.get("bottom", 0)
            if belt_s < 0.24 or belt_s < bottom_s + 0.04 or detect_fabric_clip(crop) == "denim": continue
        elif zone_name == "bottom" and max(scores.get("skirt", 0), scores.get("bottom", 0)) < 0.17: continue
        elif zone_name == "lower_bottom" and scores.get("bottom", 0) < 0.17: continue
        elif max(scores.values()) < 0.21: continue
            
        forced = resolve_zone_force_category(zone_name, scores, crop)
        if not forced: continue
        
        length_hint = None
        if zone_name == "bottom":
            tall_bbox = [int(rx0 + rw * 0.05), int(ry0 + rh * 0.44), int(rx0 + rw * 0.95), int(ry0 + rh * 0.95)]
            length_hint = base_img.crop(tuple(tall_bbox))
            
        item = analyze_garment_crop(crop, bbox, f"zone-{zone_name}", force_category=forced, length_hint_img=length_hint)
        if item and item["slotId"] in allowed_slots:
            candidates.append(item)

    candidates = merge_to_one_piece(candidates, base_img)
            
    return candidates

def finalize_total_look_slots(by_slot: dict) -> list:
    belt = by_slot.get("belt")
    top = by_slot.get("top")
    shorts = by_slot.get("shorts")
    bottom = by_slot.get("bottom")
    skirt = by_slot.get("skirt")
    shoes = by_slot.get("shoes")
    bottom_item = shorts or bottom

    if top and skirt:
        if top.get("fabricGroup") == "leather" and skirt.get("fabricGroup") == "leather" and top.get("color") == skirt.get("color"):
            del by_slot["skirt"]
            skirt = None

    if top and shorts and shoes and not skirt and not belt: return [top, shorts, shoes]
    if top and belt and skirt and belt["confidence"] >= 0.17:
        items = [top, belt, skirt]
        if shoes: items.append(shoes)
        return items
    if top and belt and shorts and not skirt and belt["confidence"] >= 0.17: return [top, belt, shorts]
    if top and bottom_item and top.get("fabricGroup") == "denim" and bottom_item.get("fabricGroup") == "denim" and top.get("color") == bottom_item.get("color"):
        return [top, bottom_item]

    items = []
    for slot_id in TOTAL_LOOK_SLOT_ORDER:
        if slot_id in by_slot: items.append(by_slot[slot_id])
        if len(items) >= MAX_TOTAL_LOOK_ITEMS: break
    return items

def detect_full_dress(img: Image.Image) -> Optional[dict]:
    w, h = img.size
    if h <= w * 0.85 or looks_like_separate_top_skirt(img): return None
    full_scores = score_categories(img)
    dress_score = full_scores.get("dress", 0.0)
    if dress_score < 0.26 or dress_score < full_scores.get("top", 0.0) - 0.012 or full_scores.get("skirt", 0.0) > dress_score + 0.015:
        return None
    item = analyze_garment_crop(img, [0, 0, w, h], "full-dress")
    return item if item and item["categoryGroup"] == "dress" else None

# -------------------------------------------------------------------------
# 5. Public Entry Points (Used by main.py)
# -------------------------------------------------------------------------
def process_total_look_logic(img: Image.Image) -> dict:
    w, h = img.size
    methods = []
    person_bbox = pick_primary_person_bbox(img) if h <= w * 0.85 else None

    dress_item = detect_full_dress(img)
    if dress_item:
        return {"items": [dress_item], "detectionMeta": {"method": ["full-dress"], "rawCandidateCount": 1, "garmentCount": 1, "slots": ["dress"]}}

    candidates = []
    yolo_cands, bag_boxes = yolo_garment_candidates(img)
    if yolo_cands:
        methods.append("yolo")
        candidates.extend(yolo_cands)

    if h > w * 0.85:
        zone_cands = zone_garment_candidates(img, bag_boxes=bag_boxes)
        if zone_cands:
            methods.append("zone")
            candidates.extend(zone_cands)
    elif person_bbox:
        zone_cands = zone_garment_candidates(img, region=person_bbox, bag_boxes=bag_boxes)
        if zone_cands:
            methods.append("person-zone")
            candidates.extend(zone_cands)

    candidates = [c for c in candidates if c.get("color") != "none"]
    raw_count = len(candidates)

    if not candidates:
        methods.append("full-fallback")
        fallback_bbox = [0, 0, w, h]
        fallback_img = img
        if person_bbox:
            x0, y0, x1, y1 = person_bbox
            pw, ph = x1 - x0, y1 - y0
            fallback_bbox = [int(x0 + pw * 0.05), int(y0 + ph * 0.05), int(x1 - pw * 0.05), int(y0 + ph * 0.42)]
            fallback_img = img.crop(tuple(fallback_bbox))
        fallback = analyze_garment_crop(fallback_img, fallback_bbox, "full-fallback", force_category="top")
        items = [fallback] if fallback else []
    else:
        by_slot = apply_dress_exclusive_rule(dedupe_by_slot(candidates))
        by_slot = filter_spurious_belt(by_slot)
        items = finalize_total_look_slots(by_slot)

    return {
        "items": items,
        "detectionMeta": {
            "method": methods or ["none"],
            "rawCandidateCount": raw_count,
            "garmentCount": len(items),
            "slots": [i["slotId"] for i in items],
        },
    }

def process_image_logic(img: Image.Image) -> list:
    """Used for single-garment processing when YOLO detects individual items."""
    results = yolo_model(img)
    found_items = []

    if len(results[0].boxes) == 0:
        embedding = encode_image(img).cpu().numpy().flatten().tolist()
        category_group = detect_category_clip(img)
        fabric = detect_fabric_clip(img)
        color, is_stripe = get_fashion_color(img, category_group, fabric)
        if color == "none":
          return found_items
        shoe_style = detect_shoe_style_clip(img) if category_group == "shoes" else None
        top_style = detect_top_style_clip(img) if category_group == "top" else None
        bottom_length = detect_bottom_length_clip(img) if category_group == "bottom" else None
        
        found_items.append(attach_color_vectors({
            "category": "other", "categoryGroup": category_group, "fabricGroup": fabric,
            "confidence": 1.0, "embedding": embedding, "color": color,
        }, color, is_stripe, category_group, fabric, shoe_style, bottom_length, top_style))
    else:
        single_bag_boxes = []
        for r in results:
            for box in r.boxes:
                lbl = yolo_model.names[int(box.cls)].lower()
                if float(box.conf) > 0.2 and lbl in BAG_YOLO_CLASSES:
                    single_bag_boxes.append(tuple(int(c) for c in box.xyxy[0].tolist()))
        clean_img_single = paint_out_boxes(img, single_bag_boxes)

        for r in results:
            for box in r.boxes:
                label = yolo_model.names[int(box.cls)]
                conf = float(box.conf)
                if conf > 0.2:
                    coords = box.xyxy[0].tolist()
                    crop_img = clean_img_single.crop((coords[0], coords[1], coords[2], coords[3]))
                    embedding = encode_image(crop_img).cpu().numpy().flatten().tolist()
                    category_group = detect_category_clip(crop_img)
                    fabric = detect_fabric_clip(crop_img)
                    color, is_stripe = get_fashion_color(crop_img, category_group, fabric)
                    if color == "none":
                        continue
                    shoe_style = detect_shoe_style_clip(crop_img) if category_group == "shoes" else None
                    top_style = detect_top_style_clip(crop_img) if category_group == "top" else None
                    bottom_length = detect_bottom_length_clip(crop_img) if category_group == "bottom" else None
                    
                    found_items.append(attach_color_vectors({
                        "category": get_category_group(label), "categoryGroup": category_group,
                        "fabricGroup": fabric, "confidence": conf, "embedding": embedding, "color": color,
                    }, color, is_stripe, category_group, fabric, shoe_style, bottom_length, top_style))
    return found_items