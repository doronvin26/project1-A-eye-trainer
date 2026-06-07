import streamlit as st
import cv2
import tempfile
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pandas as pd
import os
from pathlib import Path
from functools import lru_cache

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="A-EYE TRAINER: Video Labeling",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CONSTANTS
# ============================================================================

# MediaPipe Pose Landmarks (33 keypoints)
POSE_LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index"
]

# Pose connections for skeleton drawing
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20), (11, 23),
    (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29),
    (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
]

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def init_session_state():
    """Initialize or validate session state."""
    if "video_path" not in st.session_state:
        st.session_state.video_path = None
    if "video_name" not in st.session_state:
        st.session_state.video_name = None
    if "frame_count" not in st.session_state:
        st.session_state.frame_count = 0
    if "current_frame_idx" not in st.session_state:
        st.session_state.current_frame_idx = 0
    if "landmarks_cache" not in st.session_state:
        st.session_state.landmarks_cache = None
    if "rendered_frames_cache" not in st.session_state:
        st.session_state.rendered_frames_cache = None
    if "frame_labels" not in st.session_state:
        st.session_state.frame_labels = {}
    if "last_viewed_frame" not in st.session_state:
        st.session_state.last_viewed_frame = -1

init_session_state()

# ============================================================================
# UTILITIES
# ============================================================================

@lru_cache(maxsize=1)
def load_mediapipe_model():
    """Load the MediaPipe Pose Landmarker model."""
    model_path = "pose_landmarker_lite.task"
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return None
    
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1
    )
    return vision.PoseLandmarker.create_from_options(options)

def extract_and_render_video(video_path):
    """
    Extract MediaPipe landmarks for all frames AND render them.
    Returns:
    - landmarks_list: List of landmark arrays (one per frame)
    - rendered_frames: List of pre-rendered frames with skeletons drawn
    - frame_count: Total number of frames
    """
    cap = cv2.VideoCapture(video_path)
    landmarks_list = []
    rendered_frames = []
    frame_count = 0
    
    landmarker = load_mediapipe_model()
    if landmarker is None:
        return None, None, 0
    
    with st.spinner("🔄 Processing video... Extracting landmarks and rendering frames"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            
            # Detect pose landmarks
            detection_result = landmarker.detect(mp_image)
            
            if detection_result.pose_landmarks and len(detection_result.pose_landmarks) > 0:
                landmarks = detection_result.pose_landmarks[0]
                landmark_data = [
                    {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
                    for lm in landmarks
                ]
            else:
                # If no pose detected, store empty landmarks
                landmark_data = [
                    {"x": 0, "y": 0, "z": 0, "visibility": 0}
                    for _ in range(33)
                ]
            
            landmarks_list.append(landmark_data)
            
            # Pre-render frame with skeleton
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            rendered_frame = draw_pose_on_frame(frame_bgr, landmark_data)
            rendered_frames.append(rendered_frame)
            
            frame_count += 1
            
            # Update progress
            progress = min(frame_count / 300, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Processing frame {frame_count}...")
        
        progress_bar.empty()
        status_text.empty()
    
    cap.release()
    return landmarks_list, rendered_frames, frame_count

def draw_pose_on_frame(frame, landmarks):
    """
    Draw MediaPipe skeleton on frame.
    
    Args:
        frame: OpenCV frame (BGR)
        landmarks: List of 33 landmark dicts with x, y, z, visibility
    
    Returns:
        Annotated frame with skeleton drawn
    """
    annotated = frame.copy()
    height, width = frame.shape[:2]
    
    if not landmarks:
        return annotated
    
    # Convert normalized coordinates to pixel coordinates
    pixel_landmarks = []
    for landmark in landmarks:
        x_px = int(landmark["x"] * width)
        y_px = int(landmark["y"] * height)
        pixel_landmarks.append((x_px, y_px))
    
    # Draw skeleton connections
    for connection in POSE_CONNECTIONS:
        start_idx, end_idx = connection
        if start_idx < len(pixel_landmarks) and end_idx < len(pixel_landmarks):
            pt1 = pixel_landmarks[start_idx]
            pt2 = pixel_landmarks[end_idx]
            cv2.line(annotated, pt1, pt2, (0, 255, 0), 2)
    
    # Draw keypoints
    for pt in pixel_landmarks:
        cv2.circle(annotated, pt, 5, (0, 0, 255), -1)
    
    return annotated

def get_label_for_frame(frame_idx):
    """Get the label dictionary for a specific frame."""
    if frame_idx not in st.session_state.frame_labels:
        st.session_state.frame_labels[frame_idx] = {
            "is_valid_frame": True,
            "pushup_phase": "High",
            "hips_position": "Good"
        }
    return st.session_state.frame_labels[frame_idx]

def get_default_labels_for_frame(frame_idx):
    """
    Get default labels for a frame.
    If previous frame was labeled, use its labels. Otherwise, use standard defaults.
    """
    if frame_idx == 0:
        # First frame: use standard defaults
        return {"is_valid_frame": True, "pushup_phase": "High", "hips_position": "Good"}
    
    # Check if previous frame has labels
    if (frame_idx - 1) in st.session_state.frame_labels:
        prev_labels = st.session_state.frame_labels[frame_idx - 1]
        # Auto-carry, but respect the "bad frame" flag
        if not prev_labels.get("is_valid_frame", True):
            # If previous frame is marked as bad, reset to defaults
            return {"is_valid_frame": True, "pushup_phase": "High", "hips_position": "Good"}
        else:
            return {
                "is_valid_frame": prev_labels.get("is_valid_frame", True),
                "pushup_phase": prev_labels.get("pushup_phase", "High"),
                "hips_position": prev_labels.get("hips_position", "Good")
            }
    
    # Fall back to standard defaults
    return {"is_valid_frame": True, "pushup_phase": "High", "hips_position": "Good"}

def update_frame_label(frame_idx, key, value):
    """Instantly update a single label for a frame."""
    if frame_idx not in st.session_state.frame_labels:
        st.session_state.frame_labels[frame_idx] = get_default_labels_for_frame(frame_idx)
    st.session_state.frame_labels[frame_idx][key] = value

def export_all_frames_to_csv():
    """Export all frames with their labels and landmarks to CSV."""
    if st.session_state.landmarks_cache is None or st.session_state.frame_count == 0:
        st.error("No video data to export.")
        return None
    
    data = []
    for frame_idx in range(st.session_state.frame_count):
        row = {"frame_index": frame_idx}
        
        # Get labels (or use defaults if not labeled)
        if frame_idx in st.session_state.frame_labels:
            labels = st.session_state.frame_labels[frame_idx]
        else:
            labels = get_default_labels_for_frame(frame_idx)
        
        row["is_valid_frame"] = labels.get("is_valid_frame", True)
        
        if labels.get("is_valid_frame", True):
            row["pushup_phase"] = labels.get("pushup_phase", "High")
            row["hips_position"] = labels.get("hips_position", "Good")
        else:
            row["pushup_phase"] = np.nan
            row["hips_position"] = np.nan
        
        # Add landmarks
        if frame_idx < len(st.session_state.landmarks_cache):
            landmarks = st.session_state.landmarks_cache[frame_idx]
            for i, landmark_name in enumerate(POSE_LANDMARK_NAMES):
                if i < len(landmarks):
                    row[f"{landmark_name}_x"] = landmarks[i]["x"]
                    row[f"{landmark_name}_y"] = landmarks[i]["y"]
                    row[f"{landmark_name}_z"] = landmarks[i]["z"]
                    row[f"{landmark_name}_visibility"] = landmarks[i]["visibility"]
        
        data.append(row)
    
    df = pd.DataFrame(data)
    return df

def export_labeled_so_far_to_csv():
    """Export only the frames that have been viewed/labeled up to the current point."""
    if st.session_state.landmarks_cache is None:
        st.error("No video data to export.")
        return None
    
    # Export frames from 0 to last_viewed_frame (inclusive)
    last_viewed = st.session_state.last_viewed_frame
    if last_viewed < 0:
        st.warning("No frames have been labeled yet.")
        return None
    
    data = []
    for frame_idx in range(last_viewed + 1):
        row = {"frame_index": frame_idx}
        
        # Get labels
        if frame_idx in st.session_state.frame_labels:
            labels = st.session_state.frame_labels[frame_idx]
        else:
            labels = get_default_labels_for_frame(frame_idx)
        
        row["is_valid_frame"] = labels.get("is_valid_frame", True)
        
        if labels.get("is_valid_frame", True):
            row["pushup_phase"] = labels.get("pushup_phase", "High")
            row["hips_position"] = labels.get("hips_position", "Good")
        else:
            row["pushup_phase"] = np.nan
            row["hips_position"] = np.nan
        
        # Add landmarks
        if frame_idx < len(st.session_state.landmarks_cache):
            landmarks = st.session_state.landmarks_cache[frame_idx]
            for i, landmark_name in enumerate(POSE_LANDMARK_NAMES):
                if i < len(landmarks):
                    row[f"{landmark_name}_x"] = landmarks[i]["x"]
                    row[f"{landmark_name}_y"] = landmarks[i]["y"]
                    row[f"{landmark_name}_z"] = landmarks[i]["z"]
                    row[f"{landmark_name}_visibility"] = landmarks[i]["visibility"]
        
        data.append(row)
    
    df = pd.DataFrame(data)
    return df

# ============================================================================
# MAIN APPLICATION
# ============================================================================

st.title("🎥 A-EYE TRAINER: Video Labeling System")
st.write("Upload a video, label push-up frames in real-time, and export results as CSV.")

# Sidebar for video upload
with st.sidebar:
    st.header("📁 Video Upload")
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    
    if uploaded_video is not None:
        # Save uploaded video to temp file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        tfile.flush()
        
        # Only process if it's a new video
        if st.session_state.video_path != tfile.name:
            st.session_state.video_path = tfile.name
            st.session_state.video_name = uploaded_video.name.split(".")[0]
            st.session_state.frame_labels = {}
            st.session_state.last_viewed_frame = -1
            st.session_state.current_frame_idx = 0
            
            # Extract landmarks AND render frames
            landmarks, rendered_frames, frame_count = extract_and_render_video(tfile.name)
            st.session_state.landmarks_cache = landmarks
            st.session_state.rendered_frames_cache = rendered_frames
            st.session_state.frame_count = frame_count
            
            st.success(f"✅ Video loaded: {frame_count} frames (pre-rendered)")
        
        st.divider()
        st.subheader("📊 Statistics")
        st.metric("Total Frames", st.session_state.frame_count)
        st.metric("Labeled Frames", len(st.session_state.frame_labels))
        st.metric("Last Viewed Frame", st.session_state.last_viewed_frame)

# Main content area
if st.session_state.video_path is not None:
    # Layout: left panel (video + controls), right panel (labels)
    col_video, col_labels = st.columns([3, 1])
    
    with col_video:
        st.subheader("🎬 Video Player")
        
        # Frame slider
        frame_idx = st.slider(
            "Frame Navigation",
            min_value=0,
            max_value=st.session_state.frame_count - 1,
            value=st.session_state.current_frame_idx,
            step=1
        )
        st.session_state.current_frame_idx = frame_idx
        st.session_state.last_viewed_frame = max(st.session_state.last_viewed_frame, frame_idx)
        
        # Navigation buttons
        col_prev, col_next = st.columns(2)
        with col_prev:
            if st.button("⬅️ Previous Frame"):
                if st.session_state.current_frame_idx > 0:
                    st.session_state.current_frame_idx -= 1
                    st.session_state.last_viewed_frame = max(
                        st.session_state.last_viewed_frame,
                        st.session_state.current_frame_idx
                    )
                    st.rerun()
        
        with col_next:
            if st.button("Next Frame ➡️"):
                if st.session_state.current_frame_idx < st.session_state.frame_count - 1:
                    st.session_state.current_frame_idx += 1
                    st.session_state.last_viewed_frame = max(
                        st.session_state.last_viewed_frame,
                        st.session_state.current_frame_idx
                    )
                    st.rerun()
        
        # Display PRE-RENDERED frame with skeleton (instant display!)
        if st.session_state.rendered_frames_cache and st.session_state.current_frame_idx < len(st.session_state.rendered_frames_cache):
            # Display cached rendered frame directly
            rendered_frame_bgr = st.session_state.rendered_frames_cache[st.session_state.current_frame_idx]
            rendered_frame_rgb = cv2.cvtColor(rendered_frame_bgr, cv2.COLOR_BGR2RGB)
            st.image(rendered_frame_rgb, use_column_width=True, caption=f"Frame {st.session_state.current_frame_idx}")
        
        st.info(f"🎯 Current Frame: {st.session_state.current_frame_idx} / {st.session_state.frame_count - 1}")
    
    with col_labels:
        st.subheader("🏷️ Labels")
        
        # Get current frame labels with smart defaults
        current_labels = get_label_for_frame(st.session_state.current_frame_idx)
        
        # If frame hasn't been labeled, initialize with auto-carry defaults
        if st.session_state.current_frame_idx not in st.session_state.frame_labels:
            default_labels = get_default_labels_for_frame(st.session_state.current_frame_idx)
            st.session_state.frame_labels[st.session_state.current_frame_idx] = default_labels
            current_labels = default_labels
        
        # Bad Frame checkbox
        is_valid = current_labels.get("is_valid_frame", True)
        bad_frame = st.checkbox(
            "❌ Exclude Frame / Bad Frame",
            value=not is_valid,
            key=f"bad_frame_{st.session_state.current_frame_idx}"
        )
        
        # Update instantly
        update_frame_label(st.session_state.current_frame_idx, "is_valid_frame", not bad_frame)
        
        # Enable/disable other inputs based on bad frame status
        if not bad_frame:
            # Push-up Phase
            pushup_phase = st.radio(
                "Push-up Phase",
                options=["High", "Medium", "Low"],
                index=["High", "Medium", "Low"].index(current_labels.get("pushup_phase", "High")),
                key=f"pushup_{st.session_state.current_frame_idx}"
            )
            update_frame_label(st.session_state.current_frame_idx, "pushup_phase", pushup_phase)
            
            # Hips Position
            hips_pos = st.radio(
                "Hips Position",
                options=["Good", "Too High", "Too Low"],
                index=["Good", "Too High", "Too Low"].index(current_labels.get("hips_position", "Good")),
                key=f"hips_{st.session_state.current_frame_idx}"
            )
            update_frame_label(st.session_state.current_frame_idx, "hips_position", hips_pos)
        else:
            st.write("*(inputs disabled for bad frames)*")
    
    st.divider()
    
    # Export section
    st.subheader("📥 Export Data")
    col_export1, col_export2 = st.columns(2)
    
    with col_export1:
        if st.button("📊 Export All Frames to CSV"):
            df = export_all_frames_to_csv()
            if df is not None:
                csv_filename = f"{st.session_state.video_name}_all_labels.csv"
                st.download_button(
                    label="⬇️ Download CSV (All Frames)",
                    data=df.to_csv(index=False),
                    file_name=csv_filename,
                    mime="text/csv"
                )
                st.success(f"✅ Ready to export {len(df)} frames")
    
    with col_export2:
        if st.button("📊 Export Labeled So Far to CSV"):
            df = export_labeled_so_far_to_csv()
            if df is not None:
                csv_filename = f"{st.session_state.video_name}_partial_labels.csv"
                st.download_button(
                    label="⬇️ Download CSV (Partial)",
                    data=df.to_csv(index=False),
                    file_name=csv_filename,
                    mime="text/csv"
                )
                st.success(f"✅ Ready to export {len(df)} frames (up to frame {st.session_state.last_viewed_frame})")

else:
    st.info("👉 Upload a video file to get started!")
