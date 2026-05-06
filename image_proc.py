import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import os
import numpy as np
# נתיב לתיקייה שמכילה את התמונות ונתיב לקובץ המודל שהורדת
folder_path = './images'  
model_path = '/models/pose_landmarker_full.task' # הנתיב החדש והבטוח בדוקר
# הגדרות ה-API החדש של MediaPipe למצב תמונות סטטיות
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE)

# פתיחת המודל דרך מנהל ההקשר (with) כדי להבטיח שחרור משאבים בסוף
with vision.PoseLandmarker.create_from_options(options) as landmarker:
    
    # --- שלב החימום (Warm-up) ---
    # ב-API החדש התמונות חייבות להיות עטופות באובייקט mp.Image
    dummy_image_np = np.zeros((480, 640, 3), dtype=np.uint8)
    mp_dummy_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=dummy_image_np)
    landmarker.detect(mp_dummy_image)
    
    print("Model warm-up complete. Starting measurements...\n")
    print("-" * 40)

    total_time_ms = 0
    valid_images_count = 0

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Warning: Could not read {filename}")
            continue
        # --- חילוץ הרזולוציה של התמונה ---
        # הפונקציה shape מחזירה (height, width, channels)
        height, width, _ = image.shape

        # המרה ל-RGB ויצירת אובייקט תמונה של MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        # --- תחילת המדידה ---
        start_time = time.perf_counter()
        
        # זיהוי תנוחה
        detection_result = landmarker.detect(mp_image)
        
        # --- סיום המדידה ---
        end_time = time.perf_counter()

        processing_time_ms = (end_time - start_time) * 1000
        print(f"Image: {filename} | Resolution: {width}x{height} | Processing time: {processing_time_ms:.2f} ms")        
        total_time_ms += processing_time_ms
        valid_images_count += 1

    print("-" * 40)
    if valid_images_count > 0:
        average_time = total_time_ms / valid_images_count
        print(f"Total images processed: {valid_images_count}")
        print(f"Average processing time: {average_time:.2f} ms")
    else:
        print("No valid images were found and processed.")