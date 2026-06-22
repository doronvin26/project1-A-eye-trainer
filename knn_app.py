import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import tempfile
import cv2
import math

# ALON PATCH: EVALUATION METRICS - יבוא ספריות לגרפים ומדדים
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.set_page_config(page_title="A-EYE TRAINER: KNN Predictor", layout="wide")

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

@st.cache_data
def load_training_data():
    path = os.path.join("data", "*.csv")
    files = glob.glob(path)
    if not files: return None
    dfs = [pd.read_csv(f).assign(source_file=os.path.basename(f)) for f in files]
    full_data = pd.concat(dfs, ignore_index=True)
    full_data = full_data[full_data['is_valid_frame'] == True].dropna(subset=['pushup_phase', 'hips_position'])
    return full_data

# --- 2D HELPER FUNCTIONS ---
def calc_dist(pose_landmarks, p1_idx, p2_idx):
    try:
        x1, y1 = pose_landmarks[p1_idx].x, pose_landmarks[p1_idx].y
        x2, y2 = pose_landmarks[p2_idx].x, pose_landmarks[p2_idx].y
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    except Exception: return 0.0
    
def calc_angle_live(pose_landmarks, p1_idx, p2_idx, p3_idx):
    try:
        x1, y1 = pose_landmarks[p1_idx].x, pose_landmarks[p1_idx].y
        x2, y2 = pose_landmarks[p2_idx].x, pose_landmarks[p2_idx].y
        x3, y3 = pose_landmarks[p3_idx].x, pose_landmarks[p3_idx].y
        radians = math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2)
        angle = abs(math.degrees(radians))
        if angle > 180.0: angle = 360.0 - angle
        return angle
    except Exception: return 0.0

def calc_y_diff_live(pose_landmarks, p1_idx, p2_idx):
    try: return pose_landmarks[p1_idx].y - pose_landmarks[p2_idx].y
    except Exception: return 0.0

def calc_hip_deviation_live(pose_landmarks, shoulder_idx, knee_idx, hip_idx):
    try:
        expected = (pose_landmarks[shoulder_idx].y + pose_landmarks[knee_idx].y) / 2.0
        return pose_landmarks[hip_idx].y - expected
    except Exception: return 0.0

# --- 3D HELPER FUNCTIONS ---
def get_mp_3d_point(world_landmarks, idx):
    try:
        lm = world_landmarks[idx]
        return np.array([lm.x, lm.y, lm.z])
    except Exception: return None

def calc_mp_dist_3d(world_landmarks, idx1, idx2):
    p1 = get_mp_3d_point(world_landmarks, idx1)
    p2 = get_mp_3d_point(world_landmarks, idx2)
    if p1 is None or p2 is None: return 0.0
    return np.linalg.norm(p1 - p2)

def calc_mp_angle_3d(p1, p2, p3):
    if p1 is None or p2 is None or p3 is None: return 0.0
    v1 = p1 - p2
    v2 = p3 - p2
    v1_u = v1 / (np.linalg.norm(v1) + 1e-8)
    v2_u = v2 / (np.linalg.norm(v2) + 1e-8)
    angle_rad = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    return np.degrees(angle_rad)

def calc_mp_hip_line_error_3d(shoulder_center, hip_center, ankle_center):
    if shoulder_center is None or hip_center is None or ankle_center is None: return 0.0
    line_vec = ankle_center - shoulder_center
    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-8: return 0.0
    
    line_dir = line_vec / line_len
    vec_to_hip = hip_center - shoulder_center
    projection_length = np.dot(vec_to_hip, line_dir)
    projected_point = shoulder_center + projection_length * line_dir
    
    diff_vec = hip_center - projected_point
    distance = np.linalg.norm(diff_vec)
    sign = 1.0 if diff_vec[1] > 0 else -1.0
    return distance * sign

def get_feature_columns(selected_landmarks, selected_engineered_features, use_centered):
    cols = []
    for lm in selected_landmarks:
        if use_centered:
            cols.extend([f"{lm}_centered_x", f"{lm}_centered_y", f"{lm}_centered_z", f"{lm}_visibility"])
        else:
            cols.extend([f"{lm}_x", f"{lm}_y", f"{lm}_z", f"{lm}_visibility"])
        
    cols.extend(selected_engineered_features)
    return cols

def extract_features_from_task(pose_landmarks, world_landmarks, selected_landmarks, selected_engineered_features, use_centered, temporal_state):
    features = []
    
    l_hip_idx = MP_LANDMARK_MAP["left_hip"]
    r_hip_idx = MP_LANDMARK_MAP["right_hip"]
    mid_x = (pose_landmarks[l_hip_idx].x + pose_landmarks[r_hip_idx].x) / 2.0
    mid_y = (pose_landmarks[l_hip_idx].y + pose_landmarks[r_hip_idx].y) / 2.0
    mid_z = (pose_landmarks[l_hip_idx].z + pose_landmarks[r_hip_idx].z) / 2.0

    for lm_name in selected_landmarks:
        idx = MP_LANDMARK_MAP[lm_name]
        lm = pose_landmarks[idx]
        if use_centered:
            features.extend([lm.x - mid_x, lm.y - mid_y, lm.z - mid_z, lm.visibility])
        else:
            features.extend([lm.x, lm.y, lm.z, lm.visibility])
        
    l_shoulder = MP_LANDMARK_MAP["left_shoulder"]
    r_shoulder = MP_LANDMARK_MAP["right_shoulder"]
    l_torso = calc_dist(pose_landmarks, l_shoulder, l_hip_idx)
    r_torso = calc_dist(pose_landmarks, r_shoulder, r_hip_idx)
    avg_torso = (l_torso + r_torso) / 2.0
    if avg_torso == 0: avg_torso = 0.0001
        
    eng_feat_dict = {}

    left_elbow_ang = calc_angle_live(pose_landmarks, l_shoulder, MP_LANDMARK_MAP["left_elbow"], MP_LANDMARK_MAP["left_wrist"])
    right_elbow_ang = calc_angle_live(pose_landmarks, r_shoulder, MP_LANDMARK_MAP["right_elbow"], MP_LANDMARK_MAP["right_wrist"])
    left_body_ang = calc_angle_live(pose_landmarks, l_shoulder, l_hip_idx, MP_LANDMARK_MAP["left_heel"])
    right_body_ang = calc_angle_live(pose_landmarks, r_shoulder, r_hip_idx, MP_LANDMARK_MAP["right_heel"])

    eng_feat_dict['left_body_angle'] = left_body_ang
    eng_feat_dict['right_body_angle'] = right_body_ang
    eng_feat_dict['left_angle_elbow'] = left_elbow_ang
    eng_feat_dict['right_angle_elbow'] = right_elbow_ang
    
    eng_feat_dict['left_knee_angle'] = calc_angle_live(pose_landmarks, l_hip_idx, MP_LANDMARK_MAP["left_knee"], MP_LANDMARK_MAP["left_ankle"])
    eng_feat_dict['right_knee_angle'] = calc_angle_live(pose_landmarks, r_hip_idx, MP_LANDMARK_MAP["right_knee"], MP_LANDMARK_MAP["right_ankle"])
    eng_feat_dict['neck_angle'] = calc_angle_live(pose_landmarks, MP_LANDMARK_MAP["nose"], l_shoulder, l_hip_idx)

    eng_feat_dict['left_arm_ratio'] = calc_dist(pose_landmarks, l_shoulder, MP_LANDMARK_MAP["left_wrist"]) / avg_torso
    eng_feat_dict['right_arm_ratio'] = calc_dist(pose_landmarks, r_shoulder, MP_LANDMARK_MAP["right_wrist"]) / avg_torso
    
    eng_feat_dict['left_shoulder_elbow_y_norm'] = calc_y_diff_live(pose_landmarks, l_shoulder, MP_LANDMARK_MAP["left_elbow"]) / avg_torso
    eng_feat_dict['right_shoulder_elbow_y_norm'] = calc_y_diff_live(pose_landmarks, r_shoulder, MP_LANDMARK_MAP["right_elbow"]) / avg_torso
    
    eng_feat_dict['left_hip_deviation_norm'] = calc_hip_deviation_live(pose_landmarks, l_shoulder, MP_LANDMARK_MAP["left_knee"], l_hip_idx) / avg_torso
    eng_feat_dict['right_hip_deviation_norm'] = calc_hip_deviation_live(pose_landmarks, r_shoulder, MP_LANDMARK_MAP["right_knee"], r_hip_idx) / avg_torso

    wl = world_landmarks if world_landmarks is not None else pose_landmarks
    ls_3d = get_mp_3d_point(wl, l_shoulder)
    rs_3d = get_mp_3d_point(wl, r_shoulder)
    lh_3d = get_mp_3d_point(wl, l_hip_idx)
    rh_3d = get_mp_3d_point(wl, r_hip_idx)
    la_3d = get_mp_3d_point(wl, MP_LANDMARK_MAP["left_ankle"])
    ra_3d = get_mp_3d_point(wl, MP_LANDMARK_MAP["right_ankle"])
    
    s_center = (ls_3d + rs_3d) / 2.0 if ls_3d is not None and rs_3d is not None else None
    h_center = (lh_3d + rh_3d) / 2.0 if lh_3d is not None and rh_3d is not None else None
    a_center = (la_3d + ra_3d) / 2.0 if la_3d is not None and ra_3d is not None else None

    body_align_val = calc_mp_angle_3d(s_center, h_center, a_center)
    hip_line_error_val = calc_mp_hip_line_error_3d(s_center, h_center, a_center)
    
    eng_feat_dict['body_alignment_angle'] = body_align_val
    eng_feat_dict['hip_line_error'] = hip_line_error_val

    le_3d = get_mp_3d_point(wl, MP_LANDMARK_MAP["left_elbow"])
    re_3d = get_mp_3d_point(wl, MP_LANDMARK_MAP["right_elbow"])
    lw_3d = get_mp_3d_point(wl, MP_LANDMARK_MAP["left_wrist"])
    rw_3d = get_mp_3d_point(wl, MP_LANDMARK_MAP["right_wrist"])
    
    l_arm_len = calc_mp_dist_3d(wl, l_shoulder, MP_LANDMARK_MAP["left_elbow"]) + calc_mp_dist_3d(wl, MP_LANDMARK_MAP["left_elbow"], MP_LANDMARK_MAP["left_wrist"])
    r_arm_len = calc_mp_dist_3d(wl, r_shoulder, MP_LANDMARK_MAP["right_elbow"]) + calc_mp_dist_3d(wl, MP_LANDMARK_MAP["right_elbow"], MP_LANDMARK_MAP["right_wrist"])
    
    l_comp = calc_mp_dist_3d(wl, l_shoulder, MP_LANDMARK_MAP["left_wrist"]) / l_arm_len if l_arm_len > 0 else 0.0
    r_comp = calc_mp_dist_3d(wl, r_shoulder, MP_LANDMARK_MAP["right_wrist"]) / r_arm_len if r_arm_len > 0 else 0.0
    
    eng_feat_dict['left_arm_compression'] = l_comp
    eng_feat_dict['right_arm_compression'] = r_comp
    eng_feat_dict['avg_arm_compression'] = (l_comp + r_comp) / 2.0
    
    eng_feat_dict['elbow_symmetry'] = abs(left_elbow_ang - right_elbow_ang)
    eng_feat_dict['avg_elbow_angle'] = (left_elbow_ang + right_elbow_ang) / 2.0
    eng_feat_dict['avg_body_angle'] = (left_body_ang + right_body_ang) / 2.0

    if temporal_state["l_elbow"] is None:
        eng_feat_dict['avg_delta_elbow_angle'] = 0.0
        eng_feat_dict['delta_hip_line_error'] = 0.0
        eng_feat_dict['delta_body_alignment_angle'] = 0.0
    else:
        d_l_elbow = left_elbow_ang - temporal_state["l_elbow"]
        d_r_elbow = right_elbow_ang - temporal_state["r_elbow"]
        eng_feat_dict['avg_delta_elbow_angle'] = (d_l_elbow + d_r_elbow) / 2.0
        eng_feat_dict['delta_hip_line_error'] = hip_line_error_val - temporal_state["hip_err"]
        eng_feat_dict['delta_body_alignment_angle'] = body_align_val - temporal_state["body_align"]
        
    temporal_state["l_elbow"] = left_elbow_ang
    temporal_state["r_elbow"] = right_elbow_ang
    temporal_state["hip_err"] = hip_line_error_val
    temporal_state["body_align"] = body_align_val

    for feat_name in selected_engineered_features:
        features.append(eng_feat_dict[feat_name])

    return np.array(features).reshape(1, -1)

def get_prediction_stats(actual_arr, pred_arr, frame_indices):
    correct_mask = actual_arr == pred_arr
    good_preds = np.sum(correct_mask)
    bad_preds = len(correct_mask) - good_preds
    bad_streaks = []
    current_streak = 0
    for is_correct in correct_mask:
        if not is_correct: current_streak += 1
        else:
            if current_streak > 0: bad_streaks.append(current_streak)
            current_streak = 0
    if current_streak > 0: bad_streaks.append(current_streak)
    return good_preds, bad_preds, max(bad_streaks) if bad_streaks else 0, frame_indices[~correct_mask].tolist()

# ALON PATCH: EVALUATION METRICS - פונקציה לייצור הגרפים והצגתם במסך
def display_evaluation_charts(y_true, y_pred, title_prefix):
    labels = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax1, annot_kws={"size": 14})
    ax1.set_title(f'{title_prefix} - Confusion Matrix', fontsize=16, fontweight='bold', pad=15)
    ax1.set_xlabel('Predicted Label', fontsize=12)
    ax1.set_ylabel('Actual Label', fontsize=12)
    
    # 2. Precision & Recall Bar Chart
    x = np.arange(len(labels))
    width = 0.35
    ax2.bar(x - width/2, precision, width, label='Precision', color='royalblue')
    ax2.bar(x + width/2, recall, width, label='Recall', color='lightcoral')
    
    ax2.set_title(f'{title_prefix} - Precision & Recall', fontsize=16, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=12)
    ax2.set_ylim(0, 1.1)
    ax2.legend(fontsize=12)
    
    # הוספת הערכים מעל לעמודות לנוחות
    for i in range(len(labels)):
        ax2.text(x[i] - width/2, precision[i] + 0.02, f'{precision[i]:.2f}', ha='center', va='bottom', fontsize=10)
        ax2.text(x[i] + width/2, recall[i] + 0.02, f'{recall[i]:.2f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    st.pyplot(fig)

st.title("🤖 A-EYE TRAINER: Interactive KNN Predictor")

MODEL_PATH = 'pose_landmarker_heavy.task'
if not os.path.exists(MODEL_PATH):
    st.error(f"⚠️ Model file '{MODEL_PATH}' not found! Please add it.")
    st.stop()

train_data = load_training_data()
if train_data is None:
    st.error("No CSV files found in 'data' folder!")
    st.stop()

st.sidebar.header("🕹️ Pipeline Mode")
pipeline_mode = st.sidebar.radio(
    "Select Configuration Mode:", 
    ["Classic 2D Mode (Legacy)", "Advanced 3D Mode (New)"],
    help="Toggle between original capabilities and the newly introduced 3D features and PCA options."
)

st.sidebar.header("⚙️ KNN Parameters")
k_neighbors_phase = st.sidebar.slider("Select K (Phase):", 1, 50, 3, 1)
k_neighbors_hip = st.sidebar.slider("Select K (Hips):", 1, 50, 3, 1)

if pipeline_mode == "Advanced 3D Mode (New)":
    use_pca = st.sidebar.checkbox("Use PCA (Dimensionality Reduction)", value=True)
else:
    use_pca = True

st.sidebar.markdown("### 🧍 Base Landmarks")
use_centered_coords = st.sidebar.checkbox("Center Coordinates (Translation Invariant)", value=True)
default_landmarks = ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip"]
selected_landmarks = st.sidebar.multiselect("Select Coordinates:", ALL_LANDMARKS, default=default_landmarks)

st.sidebar.markdown("### 🧬 Individual Engineered Features")

CLASSIC_FEATURES = [
    'left_body_angle', 'right_body_angle', 
    'left_angle_elbow', 'right_angle_elbow',
    'left_knee_angle', 'right_knee_angle', 'neck_angle',
    'left_arm_ratio', 'right_arm_ratio',
    'left_shoulder_elbow_y_norm', 'right_shoulder_elbow_y_norm',
    'left_hip_deviation_norm', 'right_hip_deviation_norm'
]

ADVANCED_FEATURES = [
    'body_alignment_angle', 'hip_line_error', 
    'left_arm_compression', 'right_arm_compression', 'avg_arm_compression',
    'elbow_symmetry', 'avg_elbow_angle', 'avg_body_angle',
    'avg_delta_elbow_angle', 'delta_hip_line_error', 'delta_body_alignment_angle'
]

available_features = CLASSIC_FEATURES + ADVANCED_FEATURES if pipeline_mode == "Advanced 3D Mode (New)" else CLASSIC_FEATURES

default_eng_features = [
    'left_body_angle', 'right_body_angle', 
    'left_angle_elbow', 'right_angle_elbow', 
    'left_hip_deviation_norm', 'right_hip_deviation_norm'
]

selected_engineered_features = st.sidebar.multiselect(
    "Select individual mathematical features to extract:", 
    available_features, 
    default=default_eng_features
)

feature_cols = get_feature_columns(selected_landmarks, selected_engineered_features, use_centered_coords)

missing_cols = [col for col in feature_cols if col not in train_data.columns]
if missing_cols:
    st.sidebar.error(f"⚠️ Missing columns: {missing_cols[:3]}... Did you run the CSV script?")
    st.stop()

X_train = train_data[feature_cols]
y_phase_train = train_data['pushup_phase']
y_hips_train = train_data['hips_position']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_cols)

if use_pca:
    pca = PCA(n_components=0.95)
    X_train_final = pca.fit_transform(X_train_scaled_df)
else:
    pca = None
    X_train_final = X_train_scaled_df.values

knn_phase = KNeighborsClassifier(n_neighbors=k_neighbors_phase).fit(X_train_final, y_phase_train)
knn_hips = KNeighborsClassifier(n_neighbors=k_neighbors_hip).fit(X_train_final, y_hips_train)

st.sidebar.success(f"Models trained successfully on {len(train_data)} frames.\n\nFeatures: {X_train_final.shape[1]}")

tab_csv, tab_video = st.tabs(["📊 Validate on Unseen CSV", "🎥 Test on New Video"])

with tab_csv:
    st.subheader("Upload a NEW Test CSV to evaluate the model")
    uploaded_csv = st.file_uploader("Upload Test CSV", type=['csv'])
    if uploaded_csv is not None:
        test_df = pd.read_csv(uploaded_csv)
        test_df = test_df[test_df['is_valid_frame'] == True].dropna(subset=['pushup_phase', 'hips_position']).reset_index(drop=True)
        if len(test_df) == 0:
            st.warning("No valid frames with labels found.")
        else:
            if not all(col in test_df.columns for col in feature_cols):
                st.error("Test CSV missing columns. Run CSV script on test data too.")
            else:
                X_test = test_df[feature_cols]
                X_test_scaled = scaler.transform(X_test)
                
                if use_pca:
                    X_test_final = pca.transform(X_test_scaled)
                else:
                    X_test_final = X_test_scaled

                test_df['pred_phase'] = knn_phase.predict(X_test_final)
                test_df['pred_hips'] = knn_hips.predict(X_test_final)
                
                st.markdown("### 📈 Basic Prediction Statistics")
                p_good, p_bad, p_streak, p_frames = get_prediction_stats(test_df['pushup_phase'].values, test_df['pred_phase'].values, test_df['frame_index'].values)
                h_good, h_bad, h_streak, h_frames = get_prediction_stats(test_df['hips_position'].values, test_df['pred_hips'].values, test_df['frame_index'].values)
                
                c_stat1, c_stat2 = st.columns(2)
                with c_stat1:
                    st.markdown("**🎯 Pushup Phase Stats**")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Correct", p_good); c2.metric("Incorrect", p_bad); c3.metric("Max Streak", p_streak)
                with c_stat2:
                    st.markdown("**🎯 Hips Position Stats**")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Correct", h_good); c2.metric("Incorrect", h_bad); c3.metric("Max Streak", h_streak)

                # ALON PATCH: EVALUATION METRICS - קריאה להצגת הגרפים
                st.divider()
                st.markdown("### 📊 Advanced Classification Metrics (Confusion & Precision/Recall)")
                
                display_evaluation_charts(test_df['pushup_phase'].values, test_df['pred_phase'].values, "Pushup Phase")
                display_evaluation_charts(test_df['hips_position'].values, test_df['pred_hips'].values, "Hips Position")
                
                st.divider()
                st.markdown("### 🔍 Frame-by-Frame Inspector")
                csv_frame_idx = st.slider("Navigate Test Frames:", 0, len(test_df) - 1, 0, key="test_csv_slider")
                curr = test_df.iloc[csv_frame_idx]
                st.write(f"**Phase:** Actual: `{curr['pushup_phase']}` | Pred: `{curr['pred_phase']}` {'🟢' if curr['pushup_phase']==curr['pred_phase'] else '🔴'}")
                st.write(f"**Hips:** Actual: `{curr['hips_position']}` | Pred: `{curr['pred_hips']}` {'🟢' if curr['hips_position']==curr['pred_hips'] else '🔴'}")

with tab_video:
    st.subheader("Upload a video to see real-time KNN predictions")
    uploaded_video = st.file_uploader("Upload MP4 video", type=['mp4', 'mov', 'avi'])
    if uploaded_video is not None:
        if 'current_video_name' not in st.session_state or st.session_state.current_video_name != uploaded_video.name:
            st.session_state.current_video_name = uploaded_video.name
            st.session_state.vid_frames, st.session_state.vid_features = [], []
            
            temporal_state = {"l_elbow": None, "r_elbow": None, "hip_err": None, "body_align": None}

            with st.spinner("Processing video..."):
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded_video.read())
                tfile.close()
                options = vision.PoseLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=MODEL_PATH), output_segmentation_masks=False)
                with vision.PoseLandmarker.create_from_options(options) as landmarker:
                    cap = cv2.VideoCapture(tfile.name)
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret: break
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb))
                        if res.pose_landmarks:
                            world_landmarks = res.pose_world_landmarks[0] if res.pose_world_landmarks else None
                            
                            features = extract_features_from_task(res.pose_landmarks[0], world_landmarks, selected_landmarks, selected_engineered_features, use_centered_coords, temporal_state)
                            
                            st.session_state.vid_features.append(features)
                            h, w, _ = frame_rgb.shape
                            for lm_name in selected_landmarks:
                                lm = res.pose_landmarks[0][MP_LANDMARK_MAP[lm_name]]
                                cv2.circle(frame_rgb, (int(lm.x * w), int(lm.y * h)), 6, (0, 255, 0), -1)
                        else:
                            st.session_state.vid_features.append(None)
                        _, buf = cv2.imencode('.jpg', cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 50])
                        st.session_state.vid_frames.append(buf.tobytes())
                    cap.release()
                os.unlink(tfile.name)
        
        if st.session_state.vid_frames:
            if "vid_slider" not in st.session_state: st.session_state.vid_slider = 0
            def nxt(): 
                if st.session_state.vid_slider < len(st.session_state.vid_frames) - 1: st.session_state.vid_slider += 1
            def prv(): 
                if st.session_state.vid_slider > 0: st.session_state.vid_slider -= 1

            idx = st.slider("Frame", 0, len(st.session_state.vid_frames) - 1, key="vid_slider")
            st.session_state.vid_frame_idx = idx
            
            c1, c2, c3 = st.columns([1,2,1])
            with c1: st.button("⬅️ Prev", on_click=prv, use_container_width=True)
            with c3: st.button("Next ➡️", on_click=nxt, use_container_width=True)
            
            feat = st.session_state.vid_features[idx]
            if feat is not None:
                scaled_feat = scaler.transform(feat)
                final_feat = pca.transform(scaled_feat) if use_pca else scaled_feat
                st.success(f"🎯 **KNN Prediction:** Phase = `{knn_phase.predict(final_feat)[0]}` | Hips = `{knn_hips.predict(final_feat)[0]}`")
            else:
                st.warning("⚠️ No skeleton detected.")
            st.image(st.session_state.vid_frames[idx], use_container_width=True)