"""
Configuration file containing all constants, mappings, and text prompts
used by the fashion-clip model for zero-shot classification and embeddings.
"""

# Garment slots returned by Find Total Look (display order)
TOTAL_LOOK_SLOT_ORDER = ["dress", "top", "belt", "bottom", "shorts", "skirt", "shoes"]
MAX_TOTAL_LOOK_ITEMS = 4

# COCO classes that are never outfit garments
IGNORED_YOLO_CLASSES = {
    "person", "backpack", "handbag", "tie", "umbrella", "suitcase", "cell phone", "bottle", "cup",
    "couch", "tv", "potted plant", "chair", "bed", "dining table", "laptop", "book", "vase", 
    "tennis racket", "sports ball", "bicycle", "car", "motorcycle", "bench", "dog", "cat"
}

# Accessory classes whose pixels contaminate garment zone colour analysis
BAG_YOLO_CLASSES = {"handbag", "backpack", "suitcase", "cell phone", "bottle", "cup"}

# Portrait zone crops when YOLO misses individual garments (y0/y1 as fraction of height)
TOTAL_LOOK_ZONES = [
    ("top",    0.15, 0.48, {"top"}),     
    ("belt",   0.42, 0.55, {"belt"}),   
    ("bottom", 0.48, 0.78, {"bottom", "shorts", "skirt"}),
    ("lower_bottom", 0.60, 0.95, {"bottom"}),
    ("shoes",  0.72, 0.98, {"shoes"}),
]

ZONE_FORCE_CATEGORY = {
    "top": "top",
    "belt": "belt",
    "bottom": "bottom",
    "lower_bottom": "bottom",
    "shoes": "shoes",
}

CATEGORY_MAPPING = {
    "shirt": "top", "t-shirt": "top", "jacket": "top", "coat": "top", "sweater": "top", "dress": "top",
    "pants": "bottom", "jeans": "bottom", "shorts": "bottom", "skirt": "bottom",
    "sneakers": "shoes", "boots": "shoes"
}

# Highly distinctive prompts — maximise inter-class separation for zero-shot.
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
        "a black leather biker jacket with zippers outerwear",
        "a black leather jacket on a model",
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
        "a black leather mini skirt showing bare legs",
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
        "a tailored suit waistcoat vest with front buttons",
        "a linen buttoned vest sleeveless tailored top",
        "dark navy denim sleeveless vest with buttons and waist tie on a model",
        "black denim waistcoat vest worn with a long grey skirt",
        "beige tailored linen waistcoat vest with front buttons",
        "white tailored suit vest sleeveless top",
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

COLOR_TEXT_PROMPTS = {
    "black":      ["a black top", "a solid black t-shirt", "black clothing item", "a dark black garment on white background"],
    "white":      ["a crisp bright white cotton t-shirt on a model", "a solid white tee shirt with pure neutral cool white color", "a plain white garment with no warm yellow or beige tones", "a bright white blouse with neutral pure white hue"],
    "beige":      ["a warm sandy beige cotton t-shirt with earthy warm hue", "an earthy taupe outdoor shirt with warm khaki sandy tone", "a soft sand-colored top with warm caramel neutral undertone", "an outdoor linen shirt in warm beige with yellowish warm cast"],
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

CLIP_FABRIC_PROMPTS = {
    "denim":  ["blue denim jeans woven cotton fabric", "washed denim jeans product photo", "denim jeans on a model", "grey washed denim jeans", "black denim jacket or pants"],
    "jersey": ["soft cotton jersey sweatpants joggers", "fleece jersey fabric athletic pants", "cotton jersey knit sportswear"],
    "knit":   ["chunky ribbed knit sweater knitwear", "cable knit wool pullover sweater", "ribbed knit fabric top"],
    "woven":  ["tailored woven fabric dress trousers", "structured woven chino dress pants", "smooth woven fabric formal trousers"],
    "linen":  ["lightweight linen fabric trousers", "natural linen material clothing", "linen blend pants summer"],
    "leather":["leather or faux leather pants jacket", "PU leather material clothing", "genuine leather fashion item"],
    "sequin": ["shiny sequin fabric garment", "glittery metallic sequins on clothing", "sparkling embellished party fabric"]
}

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
    "tan": (158, 135, 108), "grey": (128, 128, 128), "brown": (101, 67, 33), 
    "olive": (85, 107, 47), "navy": (0, 0, 128), "light_blue": (135, 206, 250), 
    "red": (200, 0, 0), "burgundy": (128, 0, 32), "pink": (255, 182, 193), 
    "green": (34, 139, 34), "yellow": (255, 215, 0), "lavender": (200, 162, 200), 
    "purple": (128, 60, 160),
}

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
        "black": "black leather jacket outerwear",
        "brown": "brown leather jacket outerwear",
        "grey": "grey leather jacket outerwear",
        "white": "white leather jacket outerwear",
        "beige": "beige trench coat outerwear",
    },
    "vest": {
        "beige": "beige tailored linen waistcoat vest with front buttons",
        "white": "white tailored suit vest sleeveless top",
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

COLORS_CONTRAST_WITH_WHITE = {
    "lavender", "purple", "pink", "light_blue", "yellow", "green", "red", "burgundy",
}
PASTEL_COLORS = {"lavender", "purple", "pink"}
WARM_NEUTRAL_COLORS = {"beige", "tan", "brown", "olive"}

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

PATTERN_COLOR_PHRASES = {
    "bottom": "floral print boho shorts with colorful pattern and drawstring waist",
    "top":    "floral print patterned blouse with colorful motifs",
    "skirt":  "white midi skirt with black polka dots and lace trim",
    "dress":  "floral print dress with colorful pattern",
}

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

SOLID_GARMENT_PHRASES = {
    "bottom": "plain solid beige cotton shorts with no print",
    "top":    "plain solid cotton t-shirt with no print",
    "skirt":  "plain solid skirt with no print",
    "dress":  "plain solid dress with no print",
}

# --- Micro-Patterns and Textures Dictionary ---
PATTERN_TEXTURE_PROMPTS = {
    "checkered": [
        "a checkered pattern garment", 
        "a gingham plaid outfit", 
        "a tartan plaid fabric",
        "small grid pattern fabric"
    ],
    "polka_dot": [
        "a garment with small polka dots", 
        "a dotted pattern fabric", 
        "tiny dots print",
        "a dress with small spots"
    ],
    "floral": [
        "a floral print garment", 
        "a fabric with small flowers", 
        "ditsy floral pattern"
    ],
    "animal_print": [
        "leopard print fabric", 
        "zebra stripe pattern", 
        "snake skin print pattern"
    ],
    "solid": [
        "a solid color garment", 
        "a plain unpatterned fabric", 
        "a solid block color without any pattern"
    ]
}