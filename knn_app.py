import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import tempfile
import cv2
import math # הוספנו לצורך חישובי הזוויות

# --- הייבוא היציב של MediaPipe מהאפליקציה שעובדת לך ---
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(page_title="A-EYE TRAINER: KNN Predictor", layout="wide")

# מיפוי השמות של MediaPipe לאינדקסים האמיתיים שלהם
MP_LANDMARK_MAP = {
    "nose": 0, "left_eye_inner": 1, "left_eye": 2, "left_eye_outer": 3,
    "right_eye_inner": 4, "right_eye": 5, "right_eye_outer": 6,
    "left_ear": 7, "right_ear": 8, "mouth_left": 9, "mouth_right": 10,
    "left_shoulder": 11, "right_shoulder": 12, "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16, "left_pinky": 17, "right_pinky": 18,
    "left_index": 19, "right_index": 20, "left_thumb": 21, "right_thumb": 22,
    "left_hip": 23, "right_hip": 24, "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28, "left_heel": 29, "right_heel": 30,
    "left_foot_index": 31, "right_foot_index": 32
}
ALL_LANDMARKS = list(MP_LANDMARK_MAP.keys())

# ==========================================
# HELPER FUNCTIONS
# ==========================================
@st.cache_data
def load_training_data():
    """קורא אך ורק את הקבצים מתיקיית data לצורך אימון המודל"""
    path = os.path.join("data", "*.csv")
    files = glob.glob(path)
    if not files: return None
    
    dfs = [pd.read_csv(f).assign(source_file=os.path.basename(f)) for f in files]
    full_data = pd.concat(dfs, ignore_index=True)
    full_data = full_data[full_data['is_valid_frame'] == True]
    full_data = full_data.dropna(subset=['pushup_phase', 'hips_position'])
    return full_data

def calc_angle_live(pose_landmarks, p1_idx, p2_idx, p3_idx):
    """מחשב את הזווית מתוך אובייקט landmarks של Tasks API בזמן אמת"""
    try:
        x1, y1 = pose_landmarks[p1_idx].x, pose_landmarks[p1_idx].y
        x2, y2 = pose_landmarks[p2_idx].x, pose_landmarks[p2_idx].y
        x3, y3 = pose_landmarks[p3_idx].x, pose_landmarks[p3_idx].y
        
        radians = math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2)
        angle = abs(math.degrees(radians))
        
        if angle > 180.0:
            angle = 360.0 - angle
            
        return angle
    except Exception:
        return 0.0

def get_feature_columns(selected_landmarks, use_engineered_features=False):
    """בונה את שמות העמודות, עם אופציה להוסיף את הפיצ'רים החדשים"""
    cols = []
    for lm in selected_landmarks:
        cols.extend([f"{lm}_x", f"{lm}_y", f"{lm}_z", f"{lm}_visibility"])
        
    if use_engineered_features:
        cols.extend(['left_body_angle', 'right_body_angle', 'left_angle_elbow','right_angle_elbow'])
        # הוסף כאן בעתיד עוד עמודות...
        
    return cols

def extract_features_from_task(pose_landmarks, selected_landmarks, use_engineered_features=False):
    """חולץ פיצ'רים ומחשב בזמן אמת את הפיצ'רים המהונדסים אם התבקש"""
    features = []
    for lm_name in selected_landmarks:
        idx = MP_LANDMARK_MAP[lm_name]
        lm = pose_landmarks[idx]
        features.extend([lm.x, lm.y, lm.z, lm.visibility])
        
    if use_engineered_features:
        # חישוב הזוויות על בסיס אינדקסים קבועים
        left_angle = calc_angle_live(pose_landmarks, MP_LANDMARK_MAP["left_shoulder"], MP_LANDMARK_MAP["left_hip"], MP_LANDMARK_MAP["left_heel"])
        right_angle = calc_angle_live(pose_landmarks, MP_LANDMARK_MAP["right_shoulder"], MP_LANDMARK_MAP["right_hip"], MP_LANDMARK_MAP["right_heel"])
        left_angle_elbow = calc_angle_live(pose_landmarks, MP_LANDMARK_MAP["left_shoulder"], MP_LANDMARK_MAP["left_elbow"], MP_LANDMARK_MAP["left_wrist"])
        right_angle_elbow = calc_angle_live(pose_landmarks, MP_LANDMARK_MAP["right_shoulder"], MP_LANDMARK_MAP["right_elbow"], MP_LANDMARK_MAP["right_wrist"])
        features.extend([left_angle, right_angle,left_angle_elbow,right_angle_elbow])
        # חישובים עתידיים יתווספו לכאן...
        
    return np.array(features).reshape(1, -1)

def get_prediction_stats(actual_arr, pred_arr, frame_indices):
    correct_mask = actual_arr == pred_arr
    good_preds = np.sum(correct_mask)
    bad_preds = len(correct_mask) - good_preds
    
    bad_streaks = []
    current_streak = 0
    for is_correct in correct_mask:
        if not is_correct:
            current_streak += 1
        else:
            if current_streak > 0:
                bad_streaks.append(current_streak)
            current_streak = 0
    if current_streak > 0:
        bad_streaks.append(current_streak)
        
    max_bad_streak = max(bad_streaks) if bad_streaks else 0
    bad_frames = frame_indices[~correct_mask].tolist()
    
    return good_preds, bad_preds, max_bad_streak, bad_frames

# ==========================================
# UI & SIDEBAR (MODEL TRAINING)
# ==========================================
st.title("🤖 A-EYE TRAINER: KNN Model")

# נוודא שקובץ המודל אכן קיים בתיקייה
MODEL_PATH = 'pose_landmarker_heavy.task'
if not os.path.exists(MODEL_PATH):
    st.error(f"⚠️ Model file '{MODEL_PATH}' not found in the root directory! Please add it.")
    st.stop()

train_data = load_training_data()
if train_data is None:
    st.error("No CSV files found in 'data' folder! The model needs data to train.")
    st.stop()

st.sidebar.header("⚙️ Model Parameters")
k_neighbors = st.sidebar.slider("Select K (Neighbors):", 1, 50, 3, 1)

default_landmarks = ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip"]
selected_landmarks = st.sidebar.multiselect("Select Landmarks:", ALL_LANDMARKS, default=default_landmarks)

# כפתור חדש להפעלת הפיצ'רים המחושבים
use_engineered_features = st.sidebar.checkbox("Include Engineered Features (Angles)", value=True)

if not selected_landmarks:
    st.warning("Please select landmarks.")
    st.stop()

# --- אימון המודל ---
feature_cols = get_feature_columns(selected_landmarks, use_engineered_features)

# וידוא שהעמודות החדשות קיימות ב-CSV (למקרה ששכחו להריץ את סקריפט הטיוב)
missing_cols = [col for col in feature_cols if col not in train_data.columns]
if missing_cols:
    st.sidebar.error(f"⚠️ Missing columns in training data: {missing_cols}. Did you run the CSV angles script?")
    st.stop()

X_train = train_data[feature_cols]
y_phase_train = train_data['pushup_phase']
y_hips_train = train_data['hips_position']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_cols)
# 3. הזרקת המשקלים: נותנים משקל גבוה יותר לפיצ'רים המועדפים
if 'left_body_angle' in X_train_scaled_df.columns:
    X_train_scaled_df['left_body_angle'] *= 1
    X_train_scaled_df['right_body_angle'] *= 1
    X_train_scaled_df['left_angle_elbow'] *= 1
    X_train_scaled_df['right_angle_elbow'] *= 1

# 4. אימון ה-KNN על הדאטה עם המשקלים המעודכנים

knn_phase = KNeighborsClassifier(n_neighbors=k_neighbors).fit(X_train_scaled_df, y_phase_train)
knn_hips = KNeighborsClassifier(n_neighbors=k_neighbors).fit(X_train_scaled_df, y_hips_train)

st.sidebar.success(f"Models trained successfully on {len(train_data)} frames from 'data' folder.")

# ==========================================
# TABS SETUP
# ==========================================
tab_csv, tab_video = st.tabs(["📊 Validate on Unseen CSV", "🎥 Test on New Video"])

# ==========================================
# TAB 1: UNSEEN CSV VALIDATION & STATS
# ==========================================
with tab_csv:
    st.subheader("Upload a NEW Test CSV to evaluate the model")
    
    uploaded_csv = st.file_uploader("Upload Test CSV", type=['csv'])
    
    if uploaded_csv is not None:
        test_df = pd.read_csv(uploaded_csv)
        test_df = test_df[test_df['is_valid_frame'] == True].dropna(subset=['pushup_phase', 'hips_position']).reset_index(drop=True)
        
        if len(test_df) == 0:
            st.warning("No valid frames with labels found in the uploaded CSV.")
        else:
            # וידוא עמודות לפני שמנסים לנבא
            if not all(col in test_df.columns for col in feature_cols):
                st.error("The uploaded test CSV is missing required columns (e.g., angles). Please update it.")
            else:
                X_test = test_df[feature_cols]
                X_test_scaled = scaler.transform(X_test)
                
                test_df['pred_phase'] = knn_phase.predict(X_test_scaled)
                test_df['pred_hips'] = knn_hips.predict(X_test_scaled)
                
                st.markdown("### 📈 Prediction Statistics")
                
                p_good, p_bad, p_streak, p_frames = get_prediction_stats(test_df['pushup_phase'].values, test_df['pred_phase'].values, test_df['frame_index'].values)
                h_good, h_bad, h_streak, h_frames = get_prediction_stats(test_df['hips_position'].values, test_df['pred_hips'].values, test_df['frame_index'].values)
                
                col_stat1, col_stat2 = st.columns(2)
                
                with col_stat1:
                    st.markdown("**🎯 Pushup Phase Stats**")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Correct", p_good)
                    c2.metric("Incorrect", p_bad)
                    c3.metric("Max Bad Streak", p_streak)
                    with st.expander("Show incorrect frame indices"):
                        st.write(p_frames if p_frames else "No errors! 🎉")

                with col_stat2:
                    st.markdown("**🎯 Hips Position Stats**")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Correct", h_good)
                    c2.metric("Incorrect", h_bad)
                    c3.metric("Max Bad Streak", h_streak)
                    with st.expander("Show incorrect frame indices"):
                        st.write(h_frames if h_frames else "No errors! 🎉")
                        
                st.divider()
                
                st.markdown("### 🔍 Frame-by-Frame Inspector")
                csv_frame_idx = st.slider("Navigate Test Frames:", 0, len(test_df) - 1, 0, key="test_csv_slider")
                current_row = test_df.iloc[csv_frame_idx]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Pushup Phase:** Actual: `{current_row['pushup_phase']}` | Pred: `{current_row['pred_phase']}` {'🟢' if current_row['pushup_phase']==current_row['pred_phase'] else '🔴'}")
                with col2:
                    st.markdown(f"**Hips:** Actual: `{current_row['hips_position']}` | Pred: `{current_row['pred_hips']}` {'🟢' if current_row['hips_position']==current_row['pred_hips'] else '🔴'}")

# ==========================================
# TAB 2: LIVE VIDEO PREDICTION
# ==========================================
with tab_video:
    st.subheader("Upload a video to see real-time KNN predictions")
    
    uploaded_video = st.file_uploader("Upload MP4 video", type=['mp4', 'mov', 'avi'])
    
    if uploaded_video is not None:
        if 'current_video_name' not in st.session_state or st.session_state.current_video_name != uploaded_video.name:
            st.session_state.current_video_name = uploaded_video.name
            st.session_state.vid_frames = []
            st.session_state.vid_features = []
            st.session_state.vid_frame_idx = 0
            
            with st.spinner("Processing video with MediaPipe Tasks API... Please wait."):
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded_video.read())
                tfile.close()
                
                base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
                options = vision.PoseLandmarkerOptions(
                    base_options=base_options,
                    output_segmentation_masks=False
                )
                
                with vision.PoseLandmarker.create_from_options(options) as landmarker:
                    cap = cv2.VideoCapture(tfile.name)
                    
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret: break
                        
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                        
                        detection_result = landmarker.detect(mp_image)
                        
                        if detection_result.pose_landmarks:
                            pose_landmarks = detection_result.pose_landmarks[0]
                            # העברנו את משתנה ההפעלה של הפיצ'רים החדשים לתוך הפונקציה
                            features = extract_features_from_task(pose_landmarks, selected_landmarks, use_engineered_features)
                            st.session_state.vid_features.append(features)
                            
                            h, w, _ = frame_rgb.shape
                            for lm_name in selected_landmarks:
                                idx = MP_LANDMARK_MAP[lm_name]
                                lm = pose_landmarks[idx]
                                cx, cy = int(lm.x * w), int(lm.y * h)
                                cv2.circle(frame_rgb, (cx, cy), 6, (0, 255, 0), -1)
                        else:
                            st.session_state.vid_features.append(None)
                            
                        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 50])
                        st.session_state.vid_frames.append(buffer.tobytes())
                        
                    cap.release()
                os.unlink(tfile.name)
        
        # --- UI הצגת הסרטון והחיזוי ---
        if st.session_state.vid_frames:
            if "vid_slider" not in st.session_state:
                st.session_state.vid_slider = 0

            def next_vid_frame():
                if st.session_state.vid_slider < len(st.session_state.vid_frames) - 1:
                    st.session_state.vid_slider += 1
                    
            def prev_vid_frame():
                if st.session_state.vid_slider > 0:
                    st.session_state.vid_slider -= 1

            frame_idx = st.slider("Video Frame", 0, len(st.session_state.vid_frames) - 1, key="vid_slider")
            st.session_state.vid_frame_idx = frame_idx
            
            c1, c2, c3 = st.columns([1,2,1])
            with c1: st.button("⬅️ Prev", on_click=prev_vid_frame, use_container_width=True)
            with c3: st.button("Next ➡️", on_click=next_vid_frame, use_container_width=True)
            
            current_feat = st.session_state.vid_features[st.session_state.vid_frame_idx]
            
            if current_feat is not None:
                scaled_feat = scaler.transform(current_feat)
                pred_phase_vid = knn_phase.predict(scaled_feat)[0]
                pred_hips_vid = knn_hips.predict(scaled_feat)[0]
                
                st.success(f"🎯 **KNN Prediction:** Phase = `{pred_phase_vid}` | Hips = `{pred_hips_vid}`")
            else:
                st.warning("⚠️ No skeleton detected in this frame - Cannot predict.")

            img_bytes = st.session_state.vid_frames[st.session_state.vid_frame_idx]
            st.image(img_bytes, use_container_width=True)