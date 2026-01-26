from fastapi import FastAPI, File, UploadFile
import uvicorn
from ultralytics import YOLO
from PIL import Image
import io
import numpy as np

app = FastAPI()

# טעינת מודל YOLO - בטעינה הראשונה הוא יוריד קובץ קטן מהאינטרנט
model = YOLO('yolov8n.pt') 

@app.get("/")
def home():
    return {"status": "FindMyLook ML Service is active"}

@app.post("/process-look")
async def process_look(file: UploadFile = File(...)):
    # 1. קריאת הקובץ והפיכתו לתמונה
    content = await file.read()
    img = Image.open(io.BytesIO(content)).convert("RGB")
    
    # 2. הרצת זיהוי אובייקטים
    results = model(img)
    
    found_items = []
    for r in results:
        for box in r.boxes:
            label = model.names[int(box.cls)]
            conf = float(box.conf)
            
            # סינון: ניקח רק פריטים בביטחון גבוה (מעל 30%)
            if conf > 0.3:
                coords = box.xyxy[0].tolist() # [x1, y1, x2, y2]
                
                found_items.append({
                    "category": label,
                    "confidence": conf,
                    "bbox": coords,
                    "embedding": [0.1] * 512 # Mock וקטור - הסטודנטית תשלים כאן CLIP
                })

    return {
        "items_found": len(found_items),
        "items": found_items
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)