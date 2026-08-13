"""
Main entry point for the Find My Look ML Server.
Handles FastAPI routing, request validation, and delegates logic to internal services.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import requests
import io
import base64
from PIL import Image

# Import core processing functions from the services architecture
from services.ml_service import encode_image
from services.garment_classifier import detect_fabric_clip
from services.color_analyzer import get_fashion_color
from services.vector_builder import get_color_vector

from services.total_look_orchestrator import (
    process_total_look_logic, 
    process_image_logic, 
    detect_category_clip
)

app = FastAPI(title="Find My Look ML Server")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

class URLRequest(BaseModel):
    image_url: str

class ImageRequest(BaseModel):
    image: str

# -------------------------------------------------------------------------
# API Routes
# -------------------------------------------------------------------------
@app.get("/")
async def health_check():
    """Health check endpoint for Docker to verify the service is ready."""
    return {"status": "ok", "message": "ML Server is up and running securely"}    

@app.post("/process-url")
async def process_url(data: URLRequest):
    """Used by enrichProducts.js. Skips YOLO — product images are already clean."""
    try:
        response = requests.get(data.image_url, headers=HEADERS, timeout=15)
        if response.status_code == 403:
            raise HTTPException(status_code=403, detail="The website blocked the image request (403 Forbidden)")
        response.raise_for_status()
        
        img = Image.open(io.BytesIO(response.content)).convert("RGB")
        embedding = encode_image(img).cpu().numpy().flatten().tolist()
        
        category_group = detect_category_clip(img)
        fabric = detect_fabric_clip(img)
        color, _ = get_fashion_color(img, category_group, fabric)
        
        return {"items": [{
            "embedding": embedding,
            "color": color,
            "categoryGroup": category_group,
            "colorVector": get_color_vector(color),
            "confidence": 1.0,
            "category": "product",
        }]}
    except Exception as e:
        print(f"Error processing URL {data.image_url}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract-color")
async def extract_color_from_url(data: URLRequest):
    """Extracts only the primary fashion color from an image URL."""
    try:
        response = requests.get(data.image_url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content)).convert("RGB")
        
        category_group = detect_category_clip(img)
        fabric = detect_fabric_clip(img)
        detected_color, _ = get_fashion_color(img, category_group, fabric)
        
        return {"color": detected_color}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-look")
async def process_look_file(file: UploadFile = File(...)):
    """Process a directly uploaded image file (Multi-part form)."""
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    return {"items": process_image_logic(img)}

@app.post("/process-look-base64")
async def process_look_base64(data: ImageRequest):
    """Process a single garment from a base64 string."""
    base64_data = data.image.split(",")[1] if "," in data.image else data.image
    img_bytes = base64.b64decode(base64_data)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return {"items": process_image_logic(img)}

@app.post("/process-total-look-base64")
async def process_total_look_base64_endpoint(data: ImageRequest):
    """Find Total Look: detect all garments in one full-body image."""
    base64_data = data.image.split(",")[1] if "," in data.image else data.image
    img_bytes = base64.b64decode(base64_data)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return process_total_look_logic(img)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


#.\venv\Scripts\activate
