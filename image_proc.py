import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import os
import numpy as np

# הגדרת נתיבים
folder_path = './images'  
output_folder = './output_images' 
model_path = '/models/pose_landmarker_full.task' 

os.makedirs(output_folder, exist_ok=True)

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE)

with vision.PoseLandmarker.create_from_options(options) as landmarker:
    
    # --- שלב החימום ---
    dummy_image_np = np.zeros((480, 640, 3), dtype=np.uint8)
    mp_dummy_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=dummy_image_np)
    landmarker.detect(mp_dummy_image)
    
    print("Model warm-up complete. Starting measurements...\n")
    print("-" * 40)

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)
        
        if image is None:
            continue
            
        height, width, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        start_time = time.perf_counter()
        detection_result = landmarker.detect(mp_image)
        end_time = time.perf_counter()

        processing_time_ms = (end_time - start_time) * 1000
        print(f"Image: {filename} | Resolution: {width}x{height} | Processing time: {processing_time_ms:.2f} ms")
        
        # --- ציור ושמירת התמונה (בעזרת OpenCV ישירות!) ---
        if detection_result.pose_landmarks:
            annotated_image = image.copy()
            
            # הגדרה ידנית ובטוחה של כל החיבורים בשלד (ללא תלות ב-solutions השבור)
            POSE_CONNECTIONS = [
                (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10), 
                (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19), 
                (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20), (11, 23), 
                (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), 
                (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
            ]
            
            for pose_landmarks in detection_result.pose_landmarks:
                
                # שלב 1: המרת כל הנקודות לפיקסלים אמיתיים
                pixel_landmarks = []
                for landmark in pose_landmarks:
                    cx = int(landmark.x * width)
                    cy = int(landmark.y * height)
                    pixel_landmarks.append((cx, cy))
                    
                # שלב 2: ציור הקווים (השלד)
                for connection in POSE_CONNECTIONS:
                    start_idx = connection[0]
                    end_idx = connection[1]
                    
                    if start_idx < len(pixel_landmarks) and end_idx < len(pixel_landmarks):
                        pt1 = pixel_landmarks[start_idx]
                        pt2 = pixel_landmarks[end_idx]
                        # ציור קו ירוק בעובי 2 פיקסלים
                        cv2.line(annotated_image, pt1, pt2, (0, 255, 0), 2)
                        
                # שלב 3: ציור הנקודות (המפרקים)
                for pt in pixel_landmarks:
                    # ציור עיגול אדום
                    cv2.circle(annotated_image, pt, 4, (0, 0, 255), -1)

            output_path = os.path.join(output_folder, f"annotated_{filename}")
            cv2.imwrite(output_path, annotated_image)
            print(f"    -> Pose Drawn! Image saved to: {output_path}")
        else:
            print("    -> No pose detected. Image not saved.")