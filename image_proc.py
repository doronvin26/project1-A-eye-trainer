import cv2
import mediapipe as mp
import time
import os
import numpy as np
# נתיב לתיקייה שמכילה את התמונות
folder_path = './images'
# הגדרת המודל
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# --- שלב החימום (Warm-up) ---
# ניצור תמונת "דמי" שחורה ונעביר אותה במודל רק כדי לאתחל אותו
dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
pose.process(dummy_image)
print("Model warm-up complete. Starting measurements...\n")
print("-" * 40)

total_time_ms = 0
valid_images_count = 0

# מעבר על כל הקבצים בתיקייה
for filename in os.listdir(folder_path):
    # סינון קבצים שאינם תמונות (אפשר להוסיף סיומות נוספות אם צריך)
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
        
    image_path = os.path.join(folder_path, filename)
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Warning: Could not read {filename}")
        continue
        
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # --- תחילת המדידה לתמונה ספציפית ---
    start_time = time.perf_counter()
    
    results = pose.process(image_rgb)
    
    # --- סיום המדידה ---
    end_time = time.perf_counter()

    # חישוב והדפסת הזמן
    processing_time_ms = (end_time - start_time) * 1000
    print(f"Image: {filename} | Processing time: {processing_time_ms:.2f} ms")
    
    # צבירת נתונים לממוצע
    total_time_ms += processing_time_ms
    valid_images_count += 1

print("-" * 40)

# סיכום נתונים
if valid_images_count > 0:
    average_time = total_time_ms / valid_images_count
    print(f"Total images processed: {valid_images_count}")
    print(f"Average processing time: {average_time:.2f} ms")
else:
    print("No valid images were found and processed.")

# סגירת המשאבים
pose.close()
