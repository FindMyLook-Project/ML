"""
Service layer for anatomical body pose estimation using Google MediaPipe.
Extracts precise bounding boxes for top, bottom, and shoes based on human joints
(shoulders, hips, knees, ankles) to avoid hardcoded static image zones.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Dict, Optional, Tuple

import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils  
pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=1,
    min_detection_confidence=0.5
)

def _get_landmark_y(landmarks, indices: list, h: int) -> Optional[int]:
    """Helper to get the average Y coordinate (in pixels) for given landmark indices."""
    valid_points = []
    for idx in indices:
        lm = landmarks.landmark[idx]
        if lm.visibility > 0.5:  # Only use visible joints
            valid_points.append(lm.y * h)
    if not valid_points:
        return None
    return int(sum(valid_points) / len(valid_points))

def _get_landmark_x_bounds(landmarks, indices: list, w: int, padding_ratio: float = 0.1) -> Tuple[int, int]:
    """Helper to get the dynamic min and max X coordinates based on human joints."""
    valid_x = []
    for idx in indices:
        lm = landmarks.landmark[idx]
        if lm.visibility > 0.5:
            valid_x.append(lm.x * w)
            
    if not valid_x:
        return int(w * 0.15), int(w * 0.85)
        
    min_x = min(valid_x)
    max_x = max(valid_x)
    padding = w * padding_ratio 
    
    return max(0, int(min_x - padding)), min(w, int(max_x + padding))

def extract_anatomical_zones(pil_img: Image.Image) -> Dict[str, Tuple[int, int, int, int]]:
    """
    Analyzes the image to find human joints and calculates bounding boxes for garments.
    
    Args:
        pil_img (Image.Image): The original full-body image.
        
    Returns:
        dict: Mapping of zone names ('top', 'bottom', 'shoes') to bbox coordinates (x0, y0, x1, y1).
    """
    w, h = pil_img.size
    
    # Convert PIL Image to RGB array for MediaPipe (cv2 uses BGR, but MediaPipe needs RGB)
    image_array = np.array(pil_img)
    
    results = pose.process(image_array)
    
    if not results.pose_landmarks:
        print("🧍 Pose Estimation: No person detected by MediaPipe.")
        return {}

    landmarks = results.pose_landmarks
    # ==========================================
    # --- beggining of debug ---
    # ==========================================
  
    annotated_image = image_array.copy()
    
    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS
    )
    
    annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite("debug_pose.jpg", annotated_image_bgr)
    # ==========================================
    # --- end of debug ---
    # ==========================================

    # MediaPipe Landmark Indices:
    # 11, 12: Shoulders
    # 23, 24: Hips
    # 25, 26: Knees
    # 27, 28: Ankles
    # 31, 32: Foot indices
    
    shoulders_y = _get_landmark_y(landmarks, [11, 12], h)
    hips_y = _get_landmark_y(landmarks, [23, 24], h)
    knees_y = _get_landmark_y(landmarks, [25, 26], h)
    ankles_y = _get_landmark_y(landmarks, [27, 28], h)
    feet_y = _get_landmark_y(landmarks, [31, 32], h)
    
    zones = {}
    
    # --- Top Zone (Shoulders to slightly below Hips) ---
    if shoulders_y is not None and hips_y is not None:
        top_y0 = max(0, shoulders_y - int(h * 0.05)) # Little above shoulders for collar
        top_y1 = min(h, hips_y + int(h * 0.08))      # Slightly below hips for shirt hem
        top_x0, top_x1 = _get_landmark_x_bounds(landmarks, [11, 12, 13, 14, 23, 24], w, padding_ratio=0.12)
        zones["top"] = (top_x0, top_y0, top_x1, top_y1)
        
    # --- Bottom Zone (Waist to Ankles/Knees) ---
    if hips_y is not None:
        bottom_y0 = max(0, hips_y - int(h * 0.02))
        # If ankles are visible, go down to them. Otherwise, default to knees or bottom of image
        if ankles_y is not None:
            bottom_y1 = min(h, ankles_y)
        elif knees_y is not None:
            bottom_y1 = min(h, knees_y + int(h * 0.15))
        else:
            bottom_y1 = int(h * 0.95)
            
        bottom_x0, bottom_x1 = _get_landmark_x_bounds(landmarks, [23, 24, 25, 26, 27, 28], w, padding_ratio=0.08)
        zones["bottom"] = (bottom_x0, bottom_y0, bottom_x1, bottom_y1)
        
    # --- Shoes Zone (Ankles to Bottom of Feet) ---
    if ankles_y is not None:
        shoes_y0 = max(0, ankles_y - int(h * 0.05))
        shoes_y1 = feet_y + int(h * 0.05) if feet_y else int(h * 0.98)
        shoes_x0, shoes_x1 = _get_landmark_x_bounds(landmarks, [27, 28, 31, 32], w, padding_ratio=0.06)
        zones["shoes"] = (shoes_x0, shoes_y0, shoes_x1, min(h, shoes_y1))        
    safe_zones = {}
    for name, (x0, y0, x1, y1) in zones.items():
        real_x0, real_x1 = min(x0, x1), max(x0, x1)
        real_y0, real_y1 = min(y0, y1), max(y0, y1)
        
        if (real_x1 - real_x0) >= 10 and (real_y1 - real_y0) >= 10:
            safe_zones[name] = (real_x0, real_y0, real_x1, real_y1)
        else:
            print(f"⚠️ Zone '{name}' skipped (area too small to crop).")
   
    if "top" not in safe_zones or "bottom" not in safe_zones:
        print("⚠️ MediaPipe failed to map essential body zones (likely a mirror selfie). Aborting pose estimation.")
        return {}
            
    print(f"🧍 Pose Estimation Success! Zones found: {list(safe_zones.keys())}")
    return safe_zones