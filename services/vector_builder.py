"""
Service layer for generating CLIP text embeddings (vectors).
Transforms detected attributes (color, fabric, style) into specialized 512-dimensional
search vectors to be sent to the backend for database similarity search.
"""

from typing import Dict, List, Optional

from services.ml_service import encode_texts, color_text_features
from config.prompts import (
    SHOE_STYLE_COLOR_PHRASES,
    CLIP_SHOE_STYLE_PROMPTS,
    TOP_STYLE_COLOR_PHRASES,
    CLIP_TOP_STYLE_PROMPTS,
    SKIRT_LENGTH_COLOR_PHRASES,
    CLIP_BOTTOM_LENGTH_PROMPTS,
    BOTTOM_LENGTH_FABRIC_PHRASES,
    FABRIC_COLOR_TEMPLATES_TOP,
    FABRIC_COLOR_TEMPLATES_BOTTOM,
    COLOR_TEXT_PROMPTS,
    COLORS_CONTRAST_WITH_WHITE,
    PASTEL_COLORS,
    WARM_NEUTRAL_COLORS,
    STRIPE_COLOR_PHRASES,
    PATTERN_COLOR_PHRASES,
    SOLID_GARMENT_PHRASES,
    DRESS_COLOR_PHRASES
)

# -------------------------------------------------------------------------
# 1. Base Color & Fabric Vectors
# -------------------------------------------------------------------------
def get_color_vector(color: str) -> list:
    """Return the pre-computed fashion-clip text centroid for a colour as a plain list."""
    feats = color_text_features.get(color)
    if feats is None:
        return []
    return feats.cpu().numpy().flatten().tolist()

def get_fabric_color_vector(color: str, fabric: str, category_group: str = "other") -> list:
    """Combined color+fabric text embedding for more specific re-ranking."""
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
        feats = encode_texts([SHOE_COLOR_PHRASES[color]])
        return feats.cpu().numpy().flatten().tolist()
        
    if category_group == "top" and color in TOP_SPECIFIC_COLOR_PHRASES:
        feats = encode_texts([TOP_SPECIFIC_COLOR_PHRASES[color]])
        return feats.cpu().numpy().flatten().tolist()

    if category_group == "top":
        templates = FABRIC_COLOR_TEMPLATES_TOP
        fallback = f"{color.replace('_', ' ')} cotton t-shirt top"
    elif category_group == "bottom":
        templates = FABRIC_COLOR_TEMPLATES_BOTTOM
        fallback = f"{color.replace('_', ' ')} pants bottom"
    elif category_group == "skirt":
        phrase = f"{color.replace('_', ' ')} high waisted mini skirt above the knee"
        feats = encode_texts([phrase])
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

    feats = encode_texts([phrase])
    return feats.cpu().numpy().flatten().tolist()

# -------------------------------------------------------------------------
# 2. Category-Specific Vectors (Shoes, Tops, Bottoms, Skirts, Belts, Dresses)
# -------------------------------------------------------------------------
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
    feats = encode_texts([phrase])
    return feats.cpu().numpy().flatten().tolist()

def get_shoe_style_contrast_vector(shoe_style: str) -> list:
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
        phrases.extend(["flat cross strap slide sandals with toe loop on feet", "high heel strappy dress sandals with ankle strap on feet"])
    elif shoe_style == "heeled_sandal":
        phrases.extend(["flat cross strap slide sandals with cork footbed on feet", "white leather sneakers on feet"])
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
        phrases.extend(["white pointed toe kitten heel ankle boots on feet", "black ankle boots with a slim stiletto heel on feet"])
    elif shoe_style == "flip_flop":
        phrases.extend([
            "black leather cross strap slide sandals with cork footbed",
            "black high heel strappy dress sandals with ankle strap",
            "black quilted puffy slide sandals with shearling lining",
            "beige double buckle birkenstock cork sandals",
        ])
        
    if not phrases:
        return []
        
    feats = encode_texts(phrases)
    centroid = feats.mean(dim=0, keepdim=True)
    centroid = centroid / centroid.norm(dim=-1, keepdim=True)
    return centroid.cpu().numpy().flatten().tolist()

def get_top_style_color_vector(color: str, top_style: str) -> list:
    style_phrases = TOP_STYLE_COLOR_PHRASES.get(top_style, {})
    phrase = style_phrases.get(color)
    if not phrase:
        phrase = f"{color.replace('_', ' ')} {top_style.replace('_', ' ')} top"
    feats = encode_texts([phrase])
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
        phrases.extend(["white cotton t-shirt on model", "white ribbed tank top on model", "white short sleeve tee on model", "light blue denim shirt on model"])
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
        
    feats = encode_texts(phrases)
    centroid = feats.mean(dim=0, keepdim=True)
    centroid = centroid / centroid.norm(dim=-1, keepdim=True)
    return centroid.cpu().numpy().flatten().tolist()

def get_skirt_length_color_vector(color: str, fabric: str, skirt_length: str) -> list:
    length_phrases = SKIRT_LENGTH_COLOR_PHRASES.get(skirt_length, {})
    phrase = length_phrases.get(color)
    
    if not phrase:
        length_label = {
            "mini": "mini skirt above the knee",
            "midi": "midi skirt below the knee",
            "maxi": "maxi skirt floor length",
        }.get(skirt_length, "skirt")
        
        if fabric in ("sequin", "leather"):
            phrase = f"{color.replace('_', ' ')} {fabric} {length_label}"
        else:
            phrase = f"{color.replace('_', ' ')} {length_label}"
    else:
        if fabric == "sequin":
            phrase = phrase.replace("skirt", "sequin skirt")
            
    feats = encode_texts([phrase])
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
        
    feats = encode_texts(phrases)
    centroid = feats.mean(dim=0, keepdim=True)
    centroid = centroid / centroid.norm(dim=-1, keepdim=True)
    return centroid.cpu().numpy().flatten().tolist()

def get_bottom_length_color_vector(color: str, fabric: str, bottom_length: str) -> list:
    fabric_phrases = BOTTOM_LENGTH_FABRIC_PHRASES.get(bottom_length, {})
    template = fabric_phrases.get(fabric)
    color_display = color.replace("_", " ")
    
    if template:
        phrase = template.format(color=color_display)
    else:
        length_display = "shorts above the knee" if bottom_length == "shorts" else "trousers full length"
        phrase = f"{color_display} {fabric} {length_display}"
        
    feats = encode_texts([phrase])
    return feats.cpu().numpy().flatten().tolist()

def get_bottom_length_contrast_vector(bottom_length: str) -> list:
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
        
    feats = encode_texts(phrases)
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
    feats = encode_texts(phrases if isinstance(phrases, list) else [phrases])
    
    if len(phrases) > 1:
        centroid = feats.mean(dim=0, keepdim=True)
        centroid = centroid / centroid.norm(dim=-1, keepdim=True)
        return centroid.cpu().numpy().flatten().tolist()
        
    return feats.cpu().numpy().flatten().tolist()

def get_belt_wrong_color_contrast_vector(color: str) -> list:
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
        
    feats = encode_texts(phrases)
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
    feats = encode_texts(phrases)
    centroid = feats.mean(dim=0, keepdim=True)
    centroid = centroid / centroid.norm(dim=-1, keepdim=True)
    return centroid.cpu().numpy().flatten().tolist()

def get_dress_color_vector(color: str) -> list:
    phrase = DRESS_COLOR_PHRASES.get(
        color,
        f"{color.replace('_', ' ')} maxi dress full length on a model",
    )
    feats = encode_texts([phrase])
    return feats.cpu().numpy().flatten().tolist()

def get_dress_wrong_garment_contrast_vector() -> list:
    phrases = [
        "a sleeveless black tank top or camisole on a model",
        "a cropped fitted top worn by a woman",
        "a long black maxi skirt worn with a separate top",
        "a pleated midi skirt on a model",
        "a denim tiered maxi skirt with t-shirt",
        "a beige linen maxi skirt with drawstring waist",
    ]
    feats = encode_texts(phrases)
    centroid = feats.mean(dim=0, keepdim=True)
    centroid = centroid / centroid.norm(dim=-1, keepdim=True)
    return centroid.cpu().numpy().flatten().tolist()

# -------------------------------------------------------------------------
# 3. Patterns & Stripes
# -------------------------------------------------------------------------
def get_stripe_color_vector(color: str) -> list:
    phrase = STRIPE_COLOR_PHRASES.get(
        color,
        f"{color.replace('_', ' ')} and white striped garment"
    )
    feats = encode_texts([phrase])
    return feats.cpu().numpy().flatten().tolist()

def get_pattern_color_vector(category_group: str) -> list:
    phrase = PATTERN_COLOR_PHRASES.get(
        category_group,
        "colorful floral print garment with pattern"
    )
    feats = encode_texts([phrase])
    return feats.cpu().numpy().flatten().tolist()

def get_solid_contrast_vector(category_group: str) -> list:
    phrase = SOLID_GARMENT_PHRASES.get(
        category_group,
        "plain solid garment with no print"
    )
    feats = encode_texts([phrase])
    return feats.cpu().numpy().flatten().tolist()

# -------------------------------------------------------------------------
# 4. Contrast Overrides & Combinations
# -------------------------------------------------------------------------
def get_extra_contrast_vectors(color: str) -> dict:
    """Send reference vectors so backend can drop near-misses on complex colors."""
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
        wrong = encode_texts([
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
    phrases = [
        "a sleeveless black tank top or camisole on a model",
        "a cropped fitted top worn by a woman",
        "a black halter neck crop top",
        "a ribbed sleeveless tank top",
    ]
    feats = encode_texts(phrases)
    centroid = feats.mean(dim=0, keepdim=True)
    centroid = centroid / centroid.norm(dim=-1, keepdim=True)
    return centroid.cpu().numpy().flatten().tolist()

# -------------------------------------------------------------------------
# 5. Master Output Builder
# -------------------------------------------------------------------------
def attach_color_vectors(item: dict, color: str, is_stripe: bool, category_group: str, fabric: str, 
                         shoe_style: Optional[str] = None, bottom_length: Optional[str] = None, 
                         top_style: Optional[str] = None, skirt_length: Optional[str] = None) -> dict:
    """
    Orchestrates the generation of all necessary vectors for a processed item
    and attaches them to the output dictionary before returning to the frontend.
    """
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

    # Default fallback
    item["colorVector"] = (
        get_stripe_color_vector(color) if is_stripe
        else get_fabric_color_vector(color, fabric, category_group)
    )
    for key, vec in get_extra_contrast_vectors(color).items():
        if vec:
            item[key] = vec
            
    return item