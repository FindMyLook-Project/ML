# FIND MY LOOK - Machine Learning Service

Welcome to the Machine Learning repository for **FIND MY LOOK**. 
This Python-based service acts as the computer vision and AI analysis engine for the application. It processes uploaded images to detect garments, classify features, extract fashion-specific colors, and generate semantic vectors (embeddings) for similarity search.

## Note for the Evaluator
This service is designed to run asynchronously alongside the **Backend (Node.js)** and **Frontend (React)**. To evaluate the full system flow, please ensure this FastAPI server is running. It handles both single-item crops and full-body "Total Look" pose estimations.

## Tech Stack & Models
* **FastAPI:** High-performance API framework serving the ML endpoints.
* **YOLOv8 (Ultralytics):** Real-time object detection model fine-tuned for garment bounding boxes.
* **MediaPipe:** Anatomical pose estimation used to dynamically calculate body zones (shoulders, hips, knees) for full-body outfit mapping.
* **OpenAI CLIP (fashion-clip):** Zero-shot classification and image/text embedding model used to generate semantic vectors and classify complex styles (e.g., fabric types, skirt lengths).
* **OpenCV & Pillow:** Image processing and pixel-level color analysis.

## Installation & Setup

Ensure you have **Python 3.8+** installed on your machine.

**1. Clone the Repository**
```bash
git clone [https://github.com/FindMyLook-Project/ml-service.git](https://github.com/FindMyLook-Project/ml-service.git)
cd ml-service

2. Create a Virtual Environment:
python -m venv venv

3. Activate the Virtual Environment
Windows (PowerShell): .\venv\Scripts\activate
Windows (Git Bash): source venv/Scripts/activate
macOS/Linux: source venv/bin/activate

4. Install Dependencies
pip install -r requirements.txt

5. Run the ML Server
python main.py

The server will start on http://localhost:8000.
(Note: The first run may take a few minutes as it downloads the fashion-clip model weights from HuggingFace).

-----------------------------------------------------------------------
Key Architectural Components
- Anatomical Pose Estimation (pose_estimator.py): Replaces static image crops with dynamic bounding boxes based on human joints, ensuring accurate "Total Look" detection even when the user is not perfectly centered.

- Zero-Shot Classification (garment_classifier.py): Leverages CLIP's semantic understanding to classify attributes that are difficult for traditional models (e.g., distinguishing a denim shirt from a denim vest, or a mini skirt from a maxi skirt).

- Smart Color Taxonomy (color_analyzer.py): Combines pixel-level analysis (masking skin, floor, and background) with CLIP fallbacks to accurately identify complex patterns, stripes, and subtle fashion hues (like lavender or beige).

- Vector Builder (vector_builder.py): Translates detected visual features into 512-dimensional text vectors, allowing the Node.js backend to perform highly accurate semantic searches against the MongoDB database.