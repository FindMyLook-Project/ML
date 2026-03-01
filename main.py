from fastapi import FastAPI, File, UploadFile
import uvicorn
from ultralytics import YOLO
from PIL import Image
import io
import torch
import clip

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"
yolo_model = YOLO('yolov8n.pt') 
clip_model, preprocess = clip.load("ViT-B/32", device=device)

@app.post("/process-look")
async def process_look(file: UploadFile = File(...)):
    content = await file.read()
    img = Image.open(io.BytesIO(content)).convert("RGB")
    
    results = yolo_model(img)
    
    found_items = []
    for r in results:
        for box in r.boxes:
            label = yolo_model.names[int(box.cls)]
            conf = float(box.conf)
            
            if conf > 0.3:
                coords = box.xyxy[0].tolist() # [x1, y1, x2, y2]
                
                crop_img = img.crop((coords[0], coords[1], coords[2], coords[3]))
                
                image_input = preprocess(crop_img).unsqueeze(0).to(device)
                with torch.no_grad():
                    image_features = clip_model.encode_image(image_input)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    embedding = image_features.cpu().numpy().flatten().tolist()

                found_items.append({
                    "category": label,
                    "confidence": conf,
                    "bbox": coords,
                    "embedding": embedding 
                })

    return {
        "items_found": len(found_items),
        "items": found_items
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)