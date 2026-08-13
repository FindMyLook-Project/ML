"""
Service layer for garment classification using zero-shot CLIP.
Handles categories, fabric types, styles (shoes/tops), and lengths.
"""

from typing import Optional, Tuple, Dict
from PIL import Image

# Import ML functions and pre-computed features
from services.ml_service import (
    encode_image,
    encode_texts,
    category_text_features,
    shoe_style_text_features,
    top_style_text_features,
    bottom_length_text_features,
    skirt_length_text_features,
    fabric_text_features,
    color_text_features
)
from config.prompts import COLOR_TEXT_PROMPTS, CLIP_TOP_STYLE_PROMPTS

# -------------------------------------------------------------------------
# 1. Helper Functions
# -------------------------------------------------------------------------
def score_categories(pil_img: Image.Image) -> Dict[str, float]:
    """
    Return max CLIP similarity per category for an image crop.
    
    Args:
        pil_img (Image.Image): The cropped image.
        
    Returns:
        dict: Mapping of category names to their similarity scores.
    """
    image_features = encode_image(pil_img)
    scores = {}
    for cat, text_feats in category_text_features.items():
        sims = (image_features @ text_feats.T).squeeze(0)
        scores[cat] = float(sims.max())
    return scores

def looks_like_separate_top_skirt(img: Image.Image) -> bool:
    """Evaluate if a two-piece look is present (sleeveless top + long skirt)."""
    w, h = img.size
    if h <= w * 0.85:
        return False
    upper = score_categories(img.crop((0, 0, w, int(h * 0.55))))
    lower = score_categories(img.crop((0, int(h * 0.38), w, h)))
    top_ok = upper.get("top", 0) >= 0.24
    skirt_ok = (
        lower.get("skirt", 0) >= 0.24
        and lower.get("skirt", 0) >= lower.get("dress", 0) - 0.02
    )
    return top_ok and skirt_ok

# -------------------------------------------------------------------------
# 2. Fabric Classification
# -------------------------------------------------------------------------
def detect_fabric_clip(pil_img: Image.Image) -> str:
    """
    Zero-shot fabric detection.
    
    Returns:
        str: denim | jersey | knit | woven | linen | leather | sequin | other.
    """
    image_features = encode_image(pil_img)
    best_fab = "other"
    best_score = -1.0
    
    for fab, text_feats in fabric_text_features.items():
        sims = (image_features @ text_feats.T).squeeze(0)
        score = float(sims.max())
        if score > best_score:
            best_score = score
            best_fab = fab
            
    print(f"🧵 Fabric: {best_fab} (score={best_score:.4f})")
    
    # Sequin override based on visual complexity
    if best_fab == "sequin" and best_score >= 0.22:
        return best_fab
        
    if best_score < 0.26:
        return "woven"
        
    return best_fab

# -------------------------------------------------------------------------
# 3. Length Classifications
# -------------------------------------------------------------------------
def detect_bottom_length_clip(pil_img: Image.Image) -> str:
    """Classify bottom length (shorts vs long pants)."""
    image_features = encode_image(pil_img)
    best_length = "long_pants"
    best_score = -1.0
    scores = {}
    
    for length, text_feats in bottom_length_text_features.items():
        sims = (image_features @ text_feats.T).squeeze(0)
        score = float(sims.max())
        scores[length] = score
        if score > best_score:
            best_score = score
            best_length = length
            
    rounded = {k: round(v, 3) for k, v in sorted(scores.items(), key=lambda x: -x[1])}
    print(f"👖 Shorts/pants length: {best_length}  scores: {rounded}")
    return best_length

def detect_skirt_length_clip(pil_img: Image.Image) -> Tuple[str, dict]:
    """
    Classify skirt length (mini vs midi vs maxi) with aspect ratio adjustments.
    """
    image_features = encode_image(pil_img)
    w, h = pil_img.size
    aspect_ratio = h / max(w, 1)

    scores = {}
    for length, text_feats in skirt_length_text_features.items():
        sims = (image_features @ text_feats.T).squeeze(0)
        scores[length] = float(sims.max())

    # Aspect ratio heuristics for skirts
    if aspect_ratio < 1.1:
        scores["mini"] = scores.get("mini", 0.0) + 0.06
        scores["midi"] = scores.get("midi", 0.0) - 0.03
        scores["maxi"] = scores.get("maxi", 0.0) - 0.08
    elif aspect_ratio > 1.8:
        scores["maxi"] = scores.get("maxi", 0.0) + 0.03
        scores["midi"] = scores.get("midi", 0.0) + 0.02

    best_length = max(scores, key=scores.get)
    mini_score = scores.get("mini", 0)
    midi_score = scores.get("midi", 0)
    maxi_score = scores.get("maxi", 0)
    
    # Boost maxi if it confidently beats midi and is not mini
    if best_length != "mini" and maxi_score >= midi_score + 0.010 and maxi_score >= mini_score:
        best_length = "maxi"

    rounded = {k: round(v, 3) for k, v in sorted(scores.items(), key=lambda x: -x[1])}
    print(f"👗 Skirt length: {best_length}  aspect_ratio: {aspect_ratio:.2f}  scores: {rounded}")
    return best_length, scores

# -------------------------------------------------------------------------
# 4. Style Classifications
# -------------------------------------------------------------------------
def detect_top_style_clip(pil_img: Image.Image) -> Optional[str]:
    """Classify top sub-type (tshirt vs tank vs halter vs strapless)."""
    w, h = pil_img.size
    crop = pil_img.crop((0, 0, w, int(h * 0.72))) if h > w * 0.85 else pil_img
    shoulder = pil_img.crop((0, 0, w, int(h * 0.38)))
    
    scores = {}
    for style, text_feats in top_style_text_features.items():
        body_sims = (encode_image(crop) @ text_feats.T).squeeze(0)
        shoulder_sims = (encode_image(shoulder) @ text_feats.T).squeeze(0)
        scores[style] = max(float(body_sims.max()), float(shoulder_sims.max()))
        
    best_style = max(scores, key=scores.get)
    best_score = scores[best_style]
    tshirt_score = scores.get("tshirt", 0)
    tank_score = scores.get("tank", 0)
    strapless_score = scores.get("strapless", 0)
    
    if tshirt_score >= best_score - 0.012 and tshirt_score >= strapless_score - 0.008:
        best_style = "tshirt"
        best_score = tshirt_score
    elif best_style == "halter" and tank_score >= best_score - 0.020:
        best_style = "tank"
        best_score = tank_score
    elif best_style == "strapless" and tank_score >= strapless_score - 0.010 and tshirt_score >= strapless_score - 0.015:
        best_style = "tshirt" if tshirt_score >= tank_score else "tank"
        best_score = scores[best_style]
        
    rounded = {k: round(v, 3) for k, v in sorted(scores.items(), key=lambda x: -x[1])}
    if best_score < 0.12:
        print(f"👕 Top style: none  scores: {rounded}")
        return None
        
    print(f"👕 Top style: {best_style}  scores: {rounded}")
    return best_style

def resolve_denim_top_style(crop_img: Image.Image, clip_style: Optional[str]) -> str:
    """Differentiate long-sleeve denim shirts from sleeveless denim vests."""
    image_features = encode_image(crop_img)
    shirt_prompts = CLIP_TOP_STYLE_PROMPTS["shirt"]
    vest_prompts = CLIP_TOP_STYLE_PROMPTS["vest"]
    
    shirt_s = float((image_features @ encode_texts(shirt_prompts).T).max())
    vest_s = float((image_features @ encode_texts(vest_prompts).T).max())
    
    if shirt_s >= vest_s - 0.012:
        print(f"👕 Denim top → shirt (shirt={shirt_s:.3f}, vest={vest_s:.3f})")
        return "shirt"
        
    print(f"👕 Denim top → vest (shirt={shirt_s:.3f}, vest={vest_s:.3f})")
    return "vest" if clip_style in (None, "vest", "coat") else clip_style

def detect_denim_vest_top_clip(crop_img: Image.Image) -> Optional[Tuple[str, str, str]]:
    """Detect sleeveless denim waistcoat to avoid white-wall false positives."""
    image_features = encode_image(crop_img)
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
    
    vest_s = float((image_features @ encode_texts(vest_prompts).T).max())
    scores = {"vest": vest_s}
    for name, texts in compare_prompts.items():
        scores[name] = float((image_features @ encode_texts(texts).T).max())
        
    rivals = max(scores[k] for k in ("tank", "dress", "white"))
    if vest_s >= rivals - 0.02 and vest_s >= 0.295:
        print(f"👕 Denim vest CLIP  scores: {{{', '.join(f'{k}: {v:.3f}' for k,v in scores.items())}}}")
        return "denim", "black", "vest"
    return None