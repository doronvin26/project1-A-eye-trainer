import cv2
import time
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO

# ==========================================
# User Configuration
# ==========================================
VIDEO_PATH = "11good.mp4" # Test video path
OUTPUT_DIR = "OUTPUT_model_comparison"  # Directory to save extracted frames
NUM_FRAMES_TO_EXTRACT = 10     # Number of frames to extract

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# Helper function: Calculate frame indices to extract
# ==========================================
def get_target_frames(video_path, num_frames):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    if total_frames == 0:
        print("Error: Could not read video frame count.")
        return []
        
    # Divide the video into equal intervals to get exactly the requested number of frames
    target_frames = [int(i * (total_frames - 1) / (num_frames - 1)) for i in range(num_frames)]
    return target_frames

# ==========================================
# 1. Benchmark function for MediaPipe
# ==========================================
def benchmark_mediapipe(video_path, target_frames):
    print("Running MediaPipe Benchmark...")
    
    # Use the model path downloaded in the Dockerfile, or download it locally
    model_path = '/models/pose_landmarker_full.task'
    if not os.path.exists(model_path):
        model_path = 'pose_landmarker_full.task'
        if not os.path.exists(model_path):
            print("Downloading MediaPipe model...")
            import urllib.request
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
                model_path
            )

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE)
        
    POSE_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10), 
        (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19), 
        (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20), (11, 23), 
        (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), 
        (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
    ]
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 1
    
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        start_time = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            results = landmarker.detect(mp_image)
            
            # If the current frame is in our target list - save it
            if frame_count in target_frames:
                frame_to_save = frame.copy()
                height, width, _ = frame_to_save.shape
                
                if results.pose_landmarks:
                    for pose_landmarks in results.pose_landmarks:
                        pixel_landmarks = [(int(lm.x * width), int(lm.y * height)) for lm in pose_landmarks]
                        
                        for connection in POSE_CONNECTIONS:
                            start_idx, end_idx = connection
                            if start_idx < len(pixel_landmarks) and end_idx < len(pixel_landmarks):
                                cv2.line(frame_to_save, pixel_landmarks[start_idx], pixel_landmarks[end_idx], (0, 255, 0), 2)
                                
                        for pt in pixel_landmarks:
                            cv2.circle(frame_to_save, pt, 4, (0, 0, 255), -1)

                filename = f"frame_{saved_count:02d}_MediaPipe.jpg"
                cv2.imwrite(os.path.join(OUTPUT_DIR, filename), frame_to_save)
                saved_count += 1
                    
            frame_count += 1

        total_time = time.time() - start_time
        
    fps = frame_count / total_time if total_time > 0 else 0
    cap.release()
    
    return fps

# ==========================================
# 2. Benchmark function for YOLO-Pose
# ==========================================
def benchmark_yolo(video_path, target_frames):
    print("Running YOLOv8-Pose Benchmark...")
    model = YOLO('yolov8n-pose.pt') 
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 1
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        results = model(frame, verbose=False)
        
        # If the current frame is in our target list - save it
        if frame_count in target_frames:
            frame_to_save = results[0].plot() 
            filename = f"frame_{saved_count:02d}_YOLOv8n.jpg"
            cv2.imwrite(os.path.join(OUTPUT_DIR, filename), frame_to_save)
            saved_count += 1
                
        frame_count += 1

    total_time = time.time() - start_time
    fps = frame_count / total_time if total_time > 0 else 0
    cap.release()
    
    return fps

# ==========================================
# Execution and printing results
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video file '{VIDEO_PATH}' not found.")
    else:
        # Calculate the frames we will take from the video
        frames_to_extract = get_target_frames(VIDEO_PATH, NUM_FRAMES_TO_EXTRACT)
        print(f"Video detected. Extracting frame indices: {frames_to_extract}\n")
        
        # Run the models
        fps_mp = benchmark_mediapipe(VIDEO_PATH, frames_to_extract)
        fps_yolo = benchmark_yolo(VIDEO_PATH, frames_to_extract)
        
        print("\n" + "="*40)
        print("📊 BENCHMARK RESULTS")
        print("="*40)
        print(f"MediaPipe FPS : {fps_mp:.2f}")
        print(f"YOLOv8n FPS   : {fps_yolo:.2f}")
        print("="*40)
        print(f"✅ Extracted frames are saved in the '{OUTPUT_DIR}' folder.")