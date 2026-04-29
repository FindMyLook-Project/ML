from fastapi import FastAPI, UploadFile, File, HTTPException
from ultralytics import YOLO
from PIL import Image
import io
import torch
import clip
import base64
import requests
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

FASHION_COLORS = {
    "black": (0, 0, 0), "white": (245, 245, 245), "beige": (222, 199, 166),
    "grey": (128, 128, 128), "brown": (101, 67, 33), "olive": (85, 107, 47),
    "navy": (0, 0, 128), "light_blue": (135, 206, 250), "red": (200, 0, 0),
    "burgundy": (128, 0, 32), "pink": (255, 182, 193), "green": (34, 139, 34),
    "yellow": (255, 215, 0)
}

def get_fashion_color(pil_img):
    img_1x1 = pil_img.resize((1, 1))
    avg_color = img_1x1.getpixel((0, 0)) 
    min_distance = float('inf')
    closest_color = "other"
    for color_name, rgb_value in FASHION_COLORS.items():
        distance = sum((a - b) ** 2 for a, b in zip(avg_color, rgb_value))
        if distance < min_distance:
            min_distance = distance
            closest_color = color_name
    return closest_color
# ------------------------------------------------

def get_category_group(yolo_label):
    return CATEGORY_MAPPING.get(yolo_label.lower(), "other")

def process_image_logic(img):
    results = yolo_model(img)
    found_items = []
    
    if len(results[0].boxes) == 0:
        image_input = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            embedding = image_features.cpu().numpy().flatten().tolist()
            
        color = get_fashion_color(img)
        found_items.append({"category": "other", "confidence": 1.0, "embedding": embedding, "color": color})
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
                    
                    found_items.append({
                        "category": get_category_group(label),
                        "confidence": conf,
                        "embedding": embedding,
                        "color": color 
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