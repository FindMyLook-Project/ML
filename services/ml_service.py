"""
Service layer for initializing and interacting with Machine Learning models.
Handles YOLO and CLIP loading, tensor encoding, and pre-computation of text features.
"""

import torch
from ultralytics import YOLO
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

# Import the configured prompts and mappings
from config.prompts import (
    CLIP_CATEGORY_PROMPTS,
    CLIP_SHOE_STYLE_PROMPTS,
    CLIP_TOP_STYLE_PROMPTS,
    CLIP_BOTTOM_LENGTH_PROMPTS,
    CLIP_SKIRT_LENGTH_PROMPTS,
    COLOR_TEXT_PROMPTS,
    CLIP_FABRIC_PROMPTS,
    CLIP_SOLID_VS_PATTERN_PROMPTS
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device for ML operations: {device}")

# -------------------------------------------------------------------------
# 1. Initialize Models
# -------------------------------------------------------------------------
print("Loading YOLOv8 model...")
yolo_model = YOLO('yolov8n.pt')

print("Loading fashion-clip model (first run downloads ~600MB from HuggingFace)...")
clip_model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip").to(device)
clip_processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
clip_model.eval()
print("✅ fashion-clip loaded successfully.")


# -------------------------------------------------------------------------
# 2. Core Encoding Functions
# -------------------------------------------------------------------------
def encode_texts(texts: list) -> torch.Tensor:
    """
    Encode a list of text strings into a normalized (N, 512) tensor using CLIP.
    
    Args:
        texts (list): A list of text prompts to encode.
        
    Returns:
        torch.Tensor: Normalized feature tensor for the provided texts.
    """
    inputs = clip_processor(
        text=texts, return_tensors="pt", padding=True, truncation=True
    ).to(device)
    
    with torch.no_grad():
        text_outputs = clip_model.text_model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
        )
        pooled = text_outputs.pooler_output         # (N, hidden_size)
        feats = clip_model.text_projection(pooled)  # (N, 512)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        
    return feats


def encode_image(pil_img: Image.Image) -> torch.Tensor:
    """
    Encode a PIL image into a normalized (1, 512) tensor using CLIP.
    
    Args:
        pil_img (Image.Image): The cropped image to encode.
        
    Returns:
        torch.Tensor: Normalized feature tensor for the image.
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


# -------------------------------------------------------------------------
# 3. Pre-computed Feature Registries (Computed once at startup)
# -------------------------------------------------------------------------
category_text_features = {}
for cat, texts in CLIP_CATEGORY_PROMPTS.items():
    category_text_features[cat] = encode_texts(texts)

shoe_style_text_features = {}
for style, texts in CLIP_SHOE_STYLE_PROMPTS.items():
    shoe_style_text_features[style] = encode_texts(texts)

top_style_text_features = {}
for style, texts in CLIP_TOP_STYLE_PROMPTS.items():
    top_style_text_features[style] = encode_texts(texts)

bottom_length_text_features = {}
for length, texts in CLIP_BOTTOM_LENGTH_PROMPTS.items():
    bottom_length_text_features[length] = encode_texts(texts)

skirt_length_text_features = {}
for length, texts in CLIP_SKIRT_LENGTH_PROMPTS.items():
    skirt_length_text_features[length] = encode_texts(texts)

color_text_features = {}
for color, prompts in COLOR_TEXT_PROMPTS.items():
    feats = encode_texts(prompts)
    centroid = feats.mean(dim=0, keepdim=True)
    centroid = centroid / centroid.norm(dim=-1, keepdim=True)
    color_text_features[color] = centroid

fabric_text_features = {}
for fab, texts in CLIP_FABRIC_PROMPTS.items():
    feats = encode_texts(texts)
    centroid = feats.mean(dim=0, keepdim=True)
    centroid = centroid / centroid.norm(dim=-1, keepdim=True)
    fabric_text_features[fab] = centroid

solid_pattern_features = {}
for key, texts in CLIP_SOLID_VS_PATTERN_PROMPTS.items():
    feats = encode_texts(texts)
    centroid = feats.mean(dim=0, keepdim=True)
    centroid = centroid / centroid.norm(dim=-1, keepdim=True)
    solid_pattern_features[key] = centroid