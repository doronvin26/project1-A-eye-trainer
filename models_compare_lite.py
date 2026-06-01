#not working!!!!
import cv2
import time
import os
import urllib.request
import numpy as np

# Prevent OpenMP library conflicts (Segmentation fault - code 139)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Import YOLO (PyTorch) before TensorFlow to avoid memory/thread allocation crashes
from ultralytics import YOLO
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

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
# 1. Benchmark function for MediaPipe (LITE VERSION)
# ==========================================
def benchmark_mediapipe_lite(video_path, target_frames):
    print("Running MediaPipe Lite Benchmark...")
    
    # Use the LITE model path
    model_path = 'pose_landmarker_lite.task'
    if not os.path.exists(model_path):
        print("Downloading MediaPipe Lite model...")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
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

                filename = f"frame_{saved_count:02d}_MediaPipe_Lite.jpg"
                cv2.imwrite(os.path.join(OUTPUT_DIR, filename), frame_to_save)
                saved_count += 1
                    
            frame_count += 1

        total_time = time.time() - start_time
        
    fps = frame_count / total_time if total_time > 0 else 0
    cap.release()
    
    return fps

# ==========================================
# 2. Benchmark function for YOLO-Pose (NANO VERSION)
# ==========================================
def benchmark_yolo_nano(video_path, target_frames):
    print("Running YOLOv8 Nano Benchmark...")
    # 'n' stands for nano, which is the lightest version of YOLOv8
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
# 3. Benchmark function for MoveNet (LIGHTNING VERSION)
# ==========================================
def benchmark_movenet_lightning(video_path, target_frames):
    print("Running MoveNet Lightning Benchmark...")
    
    model_path = 'movenet_lightning.tflite'
    if not os.path.exists(model_path):
        print("Downloading MoveNet Lightning model...")
        urllib.request.urlretrieve(
            "https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/3?lite-format=tflite",
            model_path
        )
        
    # Initialize TensorFlow Lite interpreter
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # MoveNet COCO 17 Keypoint connections
    MOVENET_CONNECTIONS = [
        (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7), (7, 9), 
        (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), 
        (11, 13), (13, 15), (12, 14), (14, 16)
    ]
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 1
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        height, width, _ = frame.shape
        
        # MoveNet Lightning requires input size of 192x192
        input_image = cv2.resize(frame, (192, 192))
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = np.expand_dims(input_image, axis=0)
        input_image = tf.cast(input_image, dtype=tf.int32)
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_image)
        interpreter.invoke()
        keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
        
        # If the current frame is in our target list - save it
        if frame_count in target_frames:
            frame_to_save = frame.copy()
            
            # Extract keypoints
            keypoints = keypoints_with_scores[0][0]
            pixel_landmarks = []
            
            for kp in keypoints:
                y, x, score = kp
                # Confidence threshold check
                if score > 0.3: 
                    pixel_landmarks.append((int(x * width), int(y * height)))
                else:
                    pixel_landmarks.append(None)
            
            # Draw connections
            for connection in MOVENET_CONNECTIONS:
                start_idx, end_idx = connection
                pt1 = pixel_landmarks[start_idx]
                pt2 = pixel_landmarks[end_idx]
                if pt1 is not None and pt2 is not None:
                    cv2.line(frame_to_save, pt1, pt2, (0, 255, 0), 2)
            
            # Draw points
            for pt in pixel_landmarks:
                if pt is not None:
                    cv2.circle(frame_to_save, pt, 4, (0, 0, 255), -1)
                    
            filename = f"frame_{saved_count:02d}_MoveNet_Lightning.jpg"
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
        fps_mp_lite = benchmark_mediapipe_lite(VIDEO_PATH, frames_to_extract)
        fps_yolo_nano = benchmark_yolo_nano(VIDEO_PATH, frames_to_extract)
        fps_movenet_lightning = benchmark_movenet_lightning(VIDEO_PATH, frames_to_extract)
        
        print("\n" + "="*40)
        print("📊 BENCHMARK RESULTS")
        print("="*40)
        print(f"MediaPipe Lite FPS    : {fps_mp_lite:.2f}")
        print(f"YOLOv8 Nano FPS       : {fps_yolo_nano:.2f}")
        print(f"MoveNet Lightning FPS : {fps_movenet_lightning:.2f}")
        print("="*40)
        print(f"✅ Extracted frames are saved in the '{OUTPUT_DIR}' folder.")