from fastapi import FastAPI, Body
import uvicorn
from ultralytics import YOLO
from PIL import Image
import io
import torch
import clip
import base64
from pydantic import BaseModel

app = FastAPI()

class ImageRequest(BaseModel):
    image: str

device = "cuda" if torch.cuda.is_available() else "cpu"
yolo_model = YOLO('yolov8n.pt') 
clip_model, preprocess = clip.load("ViT-B/32", device=device)

@app.post("/process-look")
async def process_look(data: ImageRequest):
    base64_data = data.image
    if "," in base64_data:
        base64_data = base64_data.split(",")[1]
    
    img_bytes = base64.b64decode(base64_data)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    
    results = yolo_model(img)
    
    found_items = []
    
    if len(results[0].boxes) == 0:
        image_input = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            embedding = image_features.cpu().numpy().flatten().tolist()
            
        found_items.append({
            "category": "unknown",
            "confidence": 1.0,
            "embedding": embedding 
        })
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

                    found_items.append({
                        "category": label,
                        "confidence": conf,
                        "embedding": embedding 
                    })

    return {
        "items_found": len(found_items),
        "items": found_items
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)