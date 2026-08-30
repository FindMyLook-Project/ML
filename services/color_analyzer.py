"""
Service layer for pixel-level color analysis, pattern detection, and brightness calculations.
Contains helper functions for specific garment edge-cases (stripes, bright whites, shoe straps).
"""
import cv2
import numpy as np
from PIL import Image
from typing import Optional, Tuple

# Import ML functions for CLIP fallbacks
from services.ml_service import encode_image, encode_texts, solid_pattern_features
from config.prompts import COLOR_TEXT_PROMPTS

# -------------------------------------------------------------------------
# 1. CLIP-Based Color & Pattern Fallbacks
# -------------------------------------------------------------------------
def detect_sequin_color_clip(pil_img: Image.Image) -> str:
    """
    Detect sequin fabric color using CLIP to bypass pixel reflection issues.
    """
    image_features = encode_image(pil_img)
    sequin_colors = ["green", "silver", "gold", "pink", "black", "blue", "red", "purple", "white"]
    scores = {}
    
    for color in sequin_colors:
        prompts = [
            f"shiny {color} sequin fabric",
            f"glittery {color} sequins on a model",
            f"sparkling {color} embellished garment"
        ]
        feats = encode_texts(prompts)
        scores[color] = float((image_features @ feats.T).max())
        
    best_color = max(sequin_colors, key=scores.get)
    print(f"✨ Sequin color CLIP: {best_color} with scores {scores}")
    return best_color

def clip_is_patterned(pil_img: Image.Image, margin: float = 0.010) -> bool:
    """
    Return True when CLIP scores 'patterned' > 'solid' by at least the specified margin.
    Acts as a last-resort gate for small prints missed by 100x100 pixel analysis.
    """
    image_features = encode_image(pil_img)
    pattern_score = float((image_features @ solid_pattern_features["pattern"].T).squeeze(0).max())
    solid_score   = float((image_features @ solid_pattern_features["solid"].T).squeeze(0).max())
    print(f"🔍 Solid-vs-pattern CLIP: pattern={pattern_score:.3f}, solid={solid_score:.3f}")
    return pattern_score >= solid_score + margin

# -------------------------------------------------------------------------
# 2. Pattern & Stripe Pixel Analysis
# -------------------------------------------------------------------------
def is_simple_horizontal_stripe(garment_pixels: np.ndarray, garment_brightness: np.ndarray) -> bool:
    """Check for true navy/black + white horizontal stripes (not ribbing or shadows)."""
    bright_frac = float((garment_brightness > 175).sum()) / max(len(garment_brightness), 1)
    dark_frac = float((garment_brightness < 120).sum()) / max(len(garment_brightness), 1)
    
    if bright_frac < 0.45 or dark_frac < 0.06:
        return False
        
    bright_px = garment_pixels[garment_brightness > 175]
    if len(bright_px) >= 20:
        bright_sat = float((bright_px.max(axis=1) - bright_px.min(axis=1)).mean())
        if bright_sat > 28:
            return False
            
    return True

def classify_stripe_dark_pixels(garment_pixels: np.ndarray, garment_brightness: np.ndarray) -> Optional[Tuple[str, bool]]:
    """Analyze dark bands to return (stripe_color, True), or None if not a simple stripe."""
    total_pixels = max(len(garment_pixels), 1)
    
    for dark_cutoff in (100, 120):
        dark_pixels = garment_pixels[garment_brightness < dark_cutoff]
        
        dark_frac = float(len(dark_pixels)) / total_pixels
        
        if dark_frac < 0.15 or len(dark_pixels) < 20:
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
            if (b - r) < 8 and abs(r - g) < 12:
                return "black", True
            return "navy", True
            
        return "black", True
    return None

def try_stripe_color(garment_pixels: np.ndarray, garment_brightness: np.ndarray, brightness_std: float, category_group: str) -> Optional[Tuple[str, bool]]:
    """Pipeline executor for stripe detection on tops."""
    if category_group != "top" or brightness_std <= 24:
        return None
    if not is_simple_horizontal_stripe(garment_pixels, garment_brightness):
        return None
        
    stripe = classify_stripe_dark_pixels(garment_pixels, garment_brightness)
    if stripe:
        color, is_stripe_flag = stripe
        print(f"🎨 Detected color: {color} (top-stripe rule)")
        return color, is_stripe_flag
    return None

# -------------------------------------------------------------------------
# 3. Top Garment Specific Color Rules
# -------------------------------------------------------------------------

def try_bright_white_top(garment_pixels: np.ndarray, garment_brightness: np.ndarray) -> Optional[Tuple[str, bool]]:
    """Detect white tees when bright shirt pixels beat dark hair/shadow in a mixed crop."""
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
        
    if bright_frac >= 0.40 and bright_avg >= 186 and min(br, bg, bb) >= 180 and bright_sat < 30 and neutral:
        print(f"🎨 Detected color: white (bright-white-top, studio) bright_avg={bright_avg:.0f}")
        return "white", False
        
    if bright_frac >= 0.22 and bright_frac <= 0.90 and bright_avg >= 200 and min(br, bg, bb) >= 195 and bright_sat < 40 and neutral:
        print(f"🎨 Detected color: white (bright-white-top, warm-lit) bright_avg={bright_avg:.0f}")
        return "white", False
        
    return None

def try_warm_beige_top(garment_pixels: np.ndarray, garment_brightness: np.ndarray) -> Optional[Tuple[str, bool]]:
    """Detect outdoor taupe / khaki / sand linen tees using a strict warm cluster."""
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

def try_solid_white_top(garment_pixels: np.ndarray, garment_brightness: np.ndarray) -> Optional[Tuple[str, bool]]:
    """Detect ribbed or solid white crop tops based on high bright fraction."""
    if is_simple_horizontal_stripe(garment_pixels, garment_brightness):
        return None
    
    white_hit = try_bright_white_top(garment_pixels, garment_brightness)
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
    
    if neutral and bright_sat < 35 and bright_avg >= 205:
        print(f"🎨 Detected color: white (solid-white-top rule)  bright_frac={bright_frac:.2f}")
        return "white", False
    return None

def refine_top_attributes(crop_img: Image.Image, fabric: str, color: str, is_stripe: bool = False) -> Tuple[str, str, bool]:
    """Re-check centre panel for white vs black outdoor/studio top zones mixing with background."""
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
        # Protected Brown Fix
        if color != "brown":
            print(f"👕 Top refine → black leather coat (dark_frac={dark_frac:.2f}, avg={panel_avg:.0f})")
            return "leather", "black", False
        return fabric, color, False
        
    if fabric == "denim":
        return "denim", ("light_blue" if color in ("white", "grey") else color), False
        
    panel_std = float(br.std())
    
    if bright_frac >= 0.07:
        if color in ("lavender", "purple", "pink"):
            return fabric, color, False
            
        white_hit = try_bright_white_top(arr, br)
        if white_hit:
            return "jersey", "white", False
            
        if color in ("beige", "brown", "tan"):
            beige_hit = try_warm_beige_top(arr, br)
            if beige_hit:
                return "jersey", beige_hit[0], False
            return fabric, color, False
            
        if color == "white":
            return fabric, "white", False
            
        beige_hit = try_warm_beige_top(arr, br)
        if beige_hit:
            return "jersey", beige_hit[0], False
            
        stripe = classify_stripe_dark_pixels(arr, br)
        if stripe and is_simple_horizontal_stripe(arr, br):
            stripe_color, stripe_flag = stripe
            print(f"👕 Top refine → {stripe_color} stripe (panel stripe check)")
            return "jersey", stripe_color, stripe_flag
            
        if color == "white" and panel_std > 16 and dark_frac >= 0.025:
            return fabric, color, False
            
        if bright_frac >= 0.80 and panel_std < 18 and dark_frac < 0.03 and color != "white":
            return fabric, color, False
            
        if bright_frac >= 0.80 and panel_std < 18 and dark_frac < 0.03:
            white_hit = try_bright_white_top(arr, br)
            if white_hit:
                return "jersey", "white", False
            beige_hit = try_warm_beige_top(arr, br)
            if beige_hit:
                return "jersey", beige_hit[0], False
                
        print(f"👕 Top refine → white jersey (bright_frac={bright_frac:.2f})")
        return "jersey", "white", False
        
    # Protected Brown Fix 2
    #if dark_frac >= 0.40 and bright_frac < 0.08 and panel_avg < 82:
        #if color != "brown":
            #print(f"👕 Top refine → black leather (dark_frac={dark_frac:.2f}, avg={panel_avg:.0f})")
            #return "leather", "black", False
            
    return fabric, color, False

# -------------------------------------------------------------------------
# 4. Shoe & Skin Pixel Helpers
# -------------------------------------------------------------------------
def foot_skin_and_dark(pil_img: Image.Image) -> Tuple[float, float]:
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

def try_warm_shoe_strap_color(garment_pixels: np.ndarray, garment_brightness: np.ndarray) -> Optional[Tuple[str, bool]]:
    """Detect tan/beige thong straps and cork soles (distinguishes from black slides)."""
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

def refine_shoe_color(crop_img: Image.Image, color: str, shoe_style: Optional[str]) -> str:
    """Read strap colour from foot zone explicitly (ignores floor/denim hem)."""
    w, h = crop_img.size
    foot = crop_img.crop((int(w * 0.08), int(h * 0.28), int(w * 0.92), h))
    arr = np.array(foot.convert("RGB"), dtype=np.float32).reshape(-1, 3)
    br = arr.mean(axis=1)
    
    dark_px = arr[(br < 78) & ~((arr[:, 2] > arr[:, 0] + 10) & (br < 160))]
    dark_frac = float(len(dark_px)) / max(len(arr), 1)

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


def remove_skin_tones(image_bgr):
    """
    detects skin tone pixels in YCrCb space,
    and returns a mask where all skin is rendered black.
    """
    ycrcb_img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    
    lower_skin = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin = np.array([255, 173, 127], dtype=np.uint8)
    
    skin_mask = cv2.inRange(ycrcb_img, lower_skin, upper_skin)
    non_skin_mask = cv2.bitwise_not(skin_mask)
    result_img = cv2.bitwise_and(image_bgr, image_bgr, mask=non_skin_mask)
    
    return result_img, non_skin_mask
# -------------------------------------------------------------------------
# 5. Main Color Extraction Pipeline
# -------------------------------------------------------------------------
def get_fashion_color(pil_img: Image.Image, category_group: Optional[str] = None, fabric: Optional[str] = None) -> Tuple[str, bool]:
    """
    Main pipeline for color and pattern extraction.
    Masks background, skin, and floor pixels, calculates color statistics,
    and classifies the color based on fashion-specific rules and palette matching.
    
    Args:
        pil_img (Image.Image): The cropped image of the garment.
        category_group (str): The garment category (e.g., 'top', 'shoes').
        fabric (str): The detected fabric type.
        
    Returns:
        Tuple[str, bool]: (detected_color, is_stripe_flag).
    """
    if fabric == "sequin":
        detected_sequin_color = detect_sequin_color_clip(pil_img)
        return detected_sequin_color, False

    # ── Step 1: Centre Crop ───────────────────────────────────────────────────
    # Tops/bottoms: middle 70% × central 60% — avoids belt and shoes bleeding in.
    # Shoes: lower 55% — focus on foot/sandal pixels, less skin variance.
# ── Step 1: Centre Crop ───────────────────────────────────────────────────
    w, h = pil_img.size
    cx0, cx1 = int(w * 0.15), int(w * 0.85)
    
    if category_group == "top":
        # Narrow the X-axis for tops to strictly hit the chest, avoiding armpits/background
        cx0, cx1 = int(w * 0.30), int(w * 0.70) 
        if h < w * 0.55:
            cy0, cy1 = int(h * 0.22), int(h * 0.92)
        else:
            # Catch strapless (chest) and crop tops (midriff), but avoid neck (0-20%) and pants (65-100%)
            cy0, cy1 = int(h * 0.22), int(h * 0.50)
            
    elif category_group == "shoes":
        # Focus on foot/sandal pixels, less skin variance.
        if h <= w * 1.3:
            cy0, cy1 = int(h * 0.35), int(h * 0.98)
        else:
            cy0, cy1 = int(h * 0.55), int(h * 0.95)
            
    elif category_group == "belt":
        # Thin waist band — centre strip only, skip shorts above/below.
        cx0, cx1 = int(w * 0.18), int(w * 0.82)
        cy0, cy1 = int(h * 0.30), int(h * 0.70)
        
    elif category_group == "skirt":
        # Skirts need the full length to detect patterns/polka dots accurately
        cy0, cy1 = int(h * 0.08), int(h * 0.88)
        
    elif category_group in ("bottom", "shorts"):
        # Pants/Shorts: Sample the upper thighs/pelvis to avoid knees and chairs when seated!
        cx0, cx1 = int(w * 0.25), int(w * 0.75)
        cy0, cy1 = int(h * 0.05), int(h * 0.40)
        
    else:
        cy0, cy1 = int(h * 0.20), int(h * 0.80)
        
    pil_img = pil_img.crop((cx0, cy0, cx1, cy1))

    img_small = pil_img.convert("RGB").resize((100, 100))
    pixels = np.array(img_small).reshape(-1, 3).astype(np.float32)

    brightness = pixels.mean(axis=1)
    saturation = pixels.max(axis=1) - pixels.min(axis=1)

    # Layer 1: Mask near-pure-white background — but never strip white from top crops.
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
        
    # Layer 3: Skin tones
    # For tops, use a tighter brightness cap (< 185 instead of < 210) so bright warm fabrics aren't stripped.
    img_bgr = cv2.cvtColor(np.array(img_small), cv2.COLOR_RGB2BGR)
    _, non_skin_mask = remove_skin_tones(img_bgr)
    is_skin = (non_skin_mask == 0).flatten()

    skin_fraction = float(is_skin.sum()) / len(is_skin)
    if category_group == "belt" and skin_fraction > 0.40:
        print(f"🛑 False positive belt detected (Belly! skin_fraction={skin_fraction:.2f}). Skipping.")
        return "none", False

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

    # ── Step 1.5: Stripe / Pattern Detection ─────────────────────────────────
    garment_brightness = garment_pixels.mean(axis=1)
    brightness_std = float(garment_brightness.std())

    if category_group == "skirt":
        dark_frac = float((garment_brightness < 85).sum()) / max(len(garment_brightness), 1)
        bright_frac = float((garment_brightness > 175).sum()) / max(len(garment_brightness), 1)
        if dark_frac >= 0.18:
            if r > b + 15 and r > g + 5:
                print(f"🎨 Detected color: brown (dark-skirt override)  dark_frac={dark_frac:.2f}")
                return "brown", False
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
            
        # Sparse polka-dots / small print on a light skirt
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
        # ── Dark-strap priority check ─────────────────────────────────────────
        dark_strap_px = garment_pixels[garment_brightness < 75]
        dark_strap_frac = float(len(dark_strap_px)) / max(len(garment_pixels), 1)
        if dark_strap_frac >= 0.030 and len(dark_strap_px) >= 10:
            dr, dg, db = dark_strap_px.mean(axis=0)
            if (float(dr) + float(dg) + float(db)) / 3.0 < 68:
                print(f"🎨 Detected color: black (dark-strap-priority rule)  dark_frac={dark_strap_frac:.3f}")
                return "black", False

        # Warm strap / sole rule
        warm_hit = try_warm_shoe_strap_color(garment_pixels, garment_brightness)
        if warm_hit:
            return warm_hit

        dark_frac = float((garment_brightness < 80).sum()) / max(len(garment_brightness), 1)
        warm_strap_px = garment_pixels[
            (garment_brightness >= 88)
            & (garment_brightness <= 220)
            & (garment_pixels[:, 0] >= garment_pixels[:, 2] - 6)
        ]
        if (len(dark_strap_px) >= max(10, int(len(garment_pixels) * 0.025)) and len(warm_strap_px) < len(dark_strap_px) * 2):
            dr, dg, db = dark_strap_px.mean(axis=0)
            if (float(dr) + float(dg) + float(db)) / 3.0 < 68:
                print(f"🎨 Detected color: black (dark-strap-shoe rule)  dark_px={len(dark_strap_px)}")
                return "black", False
                
        if dark_frac >= 0.10:
            print(f"🎨 Detected color: black (dark-shoe rule)  dark_frac={dark_frac:.2f}")
            return "black", False

    skip_pattern = category_group == "shoes" or fabric in ("denim", "leather", "sequin")
    skip_stripe_dark = False
    
    # Tan leather sandals — warm strap pixels dominate over shadow/floor
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
        # 1. Priority Check: Solid white and mixed-crop white (Moved UP!)
        white_hit = try_bright_white_top(garment_pixels, garment_brightness)
        if white_hit:
            return white_hit
        
        # 2. Stripe
        if brightness_std > 24 and fabric not in ("leather", "sequin"):
            stripe_hit = try_stripe_color(garment_pixels, garment_brightness, brightness_std, category_group)
            if stripe_hit:
                return stripe_hit
                
        # 2b. Polka-dot / printed top
        _dark_frac_top = float((garment_brightness < 80).sum()) / max(len(garment_brightness), 1)
        _bright_frac_top = float((garment_brightness > 175).sum()) / max(len(garment_brightness), 1)
        if _bright_frac_top >= 0.30 and _dark_frac_top >= 0.04 and brightness_std > 20:
            print(f"🎨 Detected color: pattern (polka-dot/print top)  std={brightness_std:.1f}, dark={_dark_frac_top:.2f}")
            return "pattern", False
            
        # 3. Solid white and mixed-crop white
        white_hit = try_bright_white_top(garment_pixels, garment_brightness)
        if white_hit:
            return white_hit
        white_hit = try_solid_white_top(garment_pixels, garment_brightness)
        if white_hit:
            return white_hit
            
        # 4. Warm beige / taupe
        beige_hit = try_warm_beige_top(garment_pixels, garment_brightness)
        if beige_hit:
            return beige_hit
            
        # 5. Light-wash denim / chambray
        avg_sat = float(max(r, g, b) - min(r, g, b))
        if 100 <= avg_brightness <= 235 and avg_sat >= 10 and b >= r + 8 and b >= g - 8:
            print(f"🎨 Detected color: light_blue (denim-top rule)  avg_rgb=({r:.0f},{g:.0f},{b:.0f})")
            return "light_blue", False
            
        # 6. Lavender / dusty purple
        purple_px = garment_pixels[
            (garment_pixels[:, 2] >= garment_pixels[:, 0])
            & (garment_brightness > 115)
            & (garment_brightness < 240)
        ]
        if len(purple_px) >= max(20, int(len(garment_pixels) * 0.10)):
            pr, pg, pb = purple_px.mean(axis=0)
            purple_sat = max(pr, pg, pb) - min(pr, pg, pb)
            if pb >= pr + 12 and purple_sat >= 18 and purple_sat < 55 and pr >= pg:
                print(f"🎨 Detected color: lavender (purple-top rule)  avg_rgb=({pr:.0f},{pg:.0f},{pb:.0f})")
                return "lavender", False
                
        # 7. Mixed-crop fallback
        if brightness_std > 35:
            bright_garment = garment_pixels[garment_brightness > 170]
            dark_garment = garment_pixels[garment_brightness < 90]
            if len(bright_garment) >= max(25, int(len(garment_pixels) * 0.15)) and len(bright_garment) > len(dark_garment):
                white_hit = try_bright_white_top(bright_garment, bright_garment.mean(axis=1))
                if white_hit:
                    return white_hit
                beige_hit = try_warm_beige_top(bright_garment, bright_garment.mean(axis=1))
                if beige_hit:
                    return beige_hit
                br, bg, bb = bright_garment.mean(axis=0)
                bright_avg = (float(br) + float(bg) + float(bb)) / 3.0
                bright_sat = max(br, bg, bb) - min(br, bg, bb)
                neutral = abs(br - bg) < 12 and abs(bg - bb) < 12
                if bright_avg > 168 and bright_sat < 45 and neutral and bright_avg >= 218:
                    print(f"🎨 Detected color: white (bright-top rule)  bright_avg={bright_avg:.0f}")
                    return "white", False
                    
        # 8. Bright-fraction last resort
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

    # White shorts/skirts with a dark belt: bright garment pixels dominate
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
        
    # Dark-background print
    if not skip_pattern and category_group in ("bottom", "dress") and brightness_std > 32:
        _light_on_dark_bright = float((garment_brightness > 170).sum()) / max(len(garment_brightness), 1)
        if avg_brightness < 120 and _light_on_dark_bright >= 0.10:
            print(f"🎨 Detected color: pattern (light-on-dark print)  std={brightness_std:.1f}, avg={avg_brightness:.0f}")
            return "pattern", False
            
    # Monochrome high-contrast print
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
            white_hit = try_bright_white_top(garment_pixels, garment_brightness)
            if white_hit:
                return white_hit
            beige_hit = try_warm_beige_top(garment_pixels, garment_brightness)
            if beige_hit:
                return beige_hit
                
        stripe = classify_stripe_dark_pixels(garment_pixels, garment_brightness)
        if stripe:
            return stripe
            
        dark_pixels = garment_pixels[garment_brightness < 120]
        if len(dark_pixels) >= 30:
            _bright_guard = float((garment_brightness > 175).sum()) / max(len(garment_brightness), 1)
            if _bright_guard >= 0.72:
                print(f"🎨 Stripe re-score skipped (bright_frac={_bright_guard:.2f}) — light garment, dark pixels are intruder")
            else:
                dark_avg = dark_pixels.mean(axis=0)
                r, g, b = float(dark_avg[0]), float(dark_avg[1]), float(dark_avg[2])
                avg_brightness = (r + g + b) / 3.0
                
                if avg_brightness >= 65:
                    print(f"🎨 Detected color: pattern (complex-print)  brightness_std={brightness_std:.1f}, dark_avg={avg_brightness:.0f}")
                    return "pattern", False
                    
                print(f"🎨 Pattern detected (std={brightness_std:.1f}), re-scoring dark pixels avg=({r:.0f},{g:.0f},{b:.0f})")
                if category_group == "top":
                    beige_hit = try_warm_beige_top(garment_pixels, garment_brightness)
                    if beige_hit:
                        return beige_hit
                    white_hit = try_bright_white_top(garment_pixels, garment_brightness)
                    if white_hit:
                        return white_hit
                        
                if avg_brightness < 60 and b > r + 5:
                    if (b - r) < 12 and abs(r - g) < 15:
                        print(f"🎨 Detected color: black (dark-neutral-pattern)  avg_rgb=({r:.0f},{g:.0f},{b:.0f})")
                        return "black", True
                    print(f"🎨 Detected color: navy (dark-rule-pattern)  avg_rgb=({r:.0f},{g:.0f},{b:.0f})")
                    return "navy", True  
        else:
            print(f"🎨 Detected color: pattern (light-print)  brightness_std={brightness_std:.1f}")
            return "pattern", False

    # ── Step 2: Bright-colour rule (white garments) ───────────────────────────
    top_white_threshold = 175 if category_group == "top" else 190
    if avg_brightness > top_white_threshold and saturation.mean() < 30:
        if category_group in ("skirt", "top", "bottom", "dress") and clip_is_patterned(pil_img):
            print(f"🎨 Detected color: pattern (CLIP override of bright-rule)  avg_rgb=({r:.0f},{g:.0f},{b:.0f})")
            return "pattern", False
        print(f"🎨 Detected color: white (bright-rule)  avg_rgb=({r:.0f},{g:.0f},{b:.0f})")
        return "white", False

    # ── Step 3: Dark-colour rule ──────────────────────────────────────────────
    if avg_brightness < 60:
        if category_group == "top":
            white_hit = try_bright_white_top(garment_pixels, garment_brightness)
            if white_hit:
                return white_hit
                
        if r >= g >= b and (r - b) >= 12:
            detected = "brown" if avg_brightness >= 35 else "black"
            print(f"🎨 Detected color: {detected} (warm-dark rule)  avg_rgb=({r:.0f},{g:.0f},{b:.0f})")
            return detected, False
            
        result = "navy" if (b > r + 15) else "black"
        print(f"🎨 Detected color: {result} (dark-rule)  avg_rgb=({r:.0f},{g:.0f},{b:.0f})")
        return result, False

    # ── Step 3.4: Nude / tan sandals ──────────────────────────────────────────
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

    # ── Step 3.3: Black shoes on pavement ─────────────────────────────────────
    if category_group == "shoes" and avg_brightness < 95 and max(r, g, b) - min(r, g, b) < 30:
        print(f"🎨 Detected color: black (shoe-dark rule)  avg_rgb=({r:.0f},{g:.0f},{b:.0f})")
        return "black", False

    # ── Step 3.5: Warm tan / khaki / beige linen ─────────────────────────────
    avg_saturation = max(r, g, b) - min(r, g, b)
    if r >= g >= b and (r - b) >= 10 and 90 <= avg_brightness <= 210 and avg_saturation < 55:
        print(f"🎨 Detected color: beige (warm-neutral rule)  avg_rgb=({r:.0f},{g:.0f},{b:.0f})")
        return "beige", False

    # ── Step 3.6: Low-saturation grey detection ──────────────────────────────
    if avg_saturation < 20 and not (r >= g >= b and (r - b) >= 12):
        print(f"🎨 Detected color: grey (low-sat rule)  avg_rgb=({r:.0f},{g:.0f},{b:.0f})")
        return "grey", False

    # ── Step 4: Palette matching ──────────────────────────────────────────────
    from config.prompts import FASHION_COLORS
    min_distance = float('inf')
    closest_color = "other"
    
    for color_name, rgb_value in FASHION_COLORS.items():
        distance = (r - rgb_value[0])**2 + (g - rgb_value[1])**2 + (b - rgb_value[2])**2
        if distance < min_distance:
            min_distance = distance
            closest_color = color_name

    # ── Step 5: Post-correction ───────────────────────────────────────────────
    if closest_color in ("grey", "tan") and (r - b) > 15:
        closest_color = "beige"

    if closest_color == "grey" and (b - r) > 20:
        closest_color = "light_blue"

    closest_color = closest_color.lower()
    print(f"🎨 Detected color: {closest_color}  avg_rgb=({r:.0f},{g:.0f},{b:.0f})")
    
    return closest_color, False