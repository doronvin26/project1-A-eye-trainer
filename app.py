import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import cv2
import time
from collections import deque
import tempfile
import math
import av

# Mediapipe & ML
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# WebRTC for live video
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

st.set_page_config(page_title="A-EYE TRAINER", layout="wide")
USE_GEBUG_INFORMATION  = False
# ==========================================
# DEBUG MODULE
# ==========================================
class FrameDebugData:
    def __init__(self, frame_num, is_processed, timestamp_ms, delta_ms, phase_pred, hip_pred, mp_time, feat_time, pred_time, sm_time, rep_count, sm_state, hip_sm_status):
        self.frame_num = frame_num
        self.is_processed = is_processed
        self.timestamp_ms = timestamp_ms
        self.delta_ms = delta_ms
        self.phase_pred = phase_pred
        self.hip_pred = hip_pred
        self.mp_time = mp_time
        self.feat_time = feat_time
        self.pred_time = pred_time
        self.sm_time = sm_time
        self.rep_count = rep_count
        self.sm_state = sm_state
        self.hip_sm_status = hip_sm_status
        
    def __str__(self):
        status_str = "Yes " if self.is_processed else "Skip"
        return (f"Frame: {self.frame_num:04d} | Processed: {status_str} | TS(ms): {self.timestamp_ms:06d} | Delta: {self.delta_ms:04d} | "
                f"Phase: {self.phase_pred:<15} | Hip: {self.hip_pred:<15} | "
                f"Reps: {self.rep_count:02d} | RepState: {self.sm_state:<15} | HipStatus: {self.hip_sm_status:<10} | "
                f"Times(ms) -> MP: {self.mp_time:04.1f}, Feat: {self.feat_time:04.1f}, "
                f"Predict: {self.pred_time:04.1f}, States: {self.sm_time:04.1f}")

def flush_debug_queue(queue, source_info):
    if not queue: 
        return
    
    # וידוא שתיקיית debug קיימת
    os.makedirs("debug", exist_ok=True)
    
    # יצירת שם קובץ עם תאריך וחותמת זמן
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join("debug", f"debug_log_{timestamp}.txt")
    
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"Source: {source_info}\n")
            f.write(f"Date & Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*90 + "\n")
            for item in queue:
                f.write(str(item) + "\n")
        print(f"✅ Debug log saved successfully to {filename}")
    except Exception as e:
        print(f"❌ Error saving debug log: {e}")


# ==========================================
# CONSTANTS & CONFIG
# ==========================================
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

USE_CENTERED_COORDS = True
# USE_PCA = True
# USE_PCA_PUSHUP_PHASE = False
# SELECTED_LANDMARKS = ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip"]
# SELECTED_ENG_FEATURES = [
#     'left_body_angle', 'right_body_angle', 
#     'left_angle_elbow', 'right_angle_elbow', 
#     'left_hip_deviation_norm', 'right_hip_deviation_norm',
#     'avg_elbow_angle', 'avg_delta_elbow_angle', 
#     'delta_hip_line_error', 'delta_body_alignment_angle'
# ]

MODELS_CONFIG = {
    "phase": {
        "target_col": "pushup_phase",
        "features": ["left_angle_elbow","avg_elbow_angle",
                                  "left_arm_ratio","right_arm_ratio",
                                  "left_shoulder_elbow_y_norm","right_shoulder_elbow_y_norm",
                                  "delta_left_elbow_angle"],
        "use_pca": False,
        "pca_components": 1,
        "k_neighbors": 11
    },
    "hips": {
        "target_col": "hips_position",
        "features": [
            "left_body_angle", "right_body_angle", 
            "left_hip_deviation_norm", "right_hip_deviation_norm",
            "delta_hip_line_error", "delta_body_alignment_angle"
        ],
        "use_pca": True,
        "pca_components": 0.95,
        "k_neighbors": 18
    }
}

# ==========================================
# 1. STATE MACHINES LOGIC
# ==========================================
class MajorityVoting:
    def __init__(self, window_size=5):
        self.q = deque(maxlen=window_size)
    def push(self, val):
        self.q.append(val)
    def get_majority(self):
        if len(self.q) < 3: return None
        counts = {}
        for v in self.q: counts[v] = counts.get(v, 0) + 1
        for v, count in counts.items():
            if count >= 3: return v
        return None

class RepStateMachine:
    def __init__(self):
        self.state = 'idle'
        self.vote = MajorityVoting(5)
        self.rep_count = 0
        self.last_issue = ""
        
    def process(self, phase_val):
        self.vote.push(str(phase_val).strip().lower())
        maj = self.vote.get_majority()
        output = None
        self.last_issue = ""
        
        if maj is None: return output, self.state
            
        if self.state == 'idle':
            if 'high' in maj: 
                self.state = 'HIGH'
                
        elif self.state == 'HIGH':
            if 'medium' in maj or 'mid' in maj: 
                self.state = 'descending_mid'
            elif 'low' in maj: 
                output = -1
                self.last_issue = "Sequence Error!"
                
        elif self.state == 'descending_mid':
            if 'low' in maj: 
                self.state = 'LOW'
            elif 'high' in maj:
                self.state = 'HIGH'
                output = 2
                self.last_issue = "Half way up!"
                
        elif self.state == 'LOW':
            if 'medium' in maj or 'mid' in maj: 
                self.state = 'ascending_mid'
            elif 'high' in maj:
                self.state = 'HIGH'
                output = -1
                self.last_issue = "Sequence Error!"
                
        elif self.state == 'ascending_mid':
            if 'high' in maj:
                self.state = 'HIGH'
                output = 1
                self.rep_count += 1
            elif 'low' in maj:
                self.state = 'LOW'
                output = 3
                self.last_issue = "Half way down!"
                
        return output, self.state

class HipStateMachine:
    def __init__(self):
        self.vote = MajorityVoting(5)
        self.status = "Good"
        
    def process(self, hip_val):
        self.vote.push(str(hip_val).strip().lower())
        maj = self.vote.get_majority()
        
        if maj is None: return -1
        
        if 'high' in maj: 
            self.status = "Too High"
            return 1
        elif 'low' in maj: 
            self.status = "Too Low"
            return 2
        elif 'good' in maj: 
            self.status = "Good Form"
            return 0
        return -1

# ==========================================
# 2. FEATURE EXTRACTION HELPERS
# ==========================================
def get_pt3d(world_landmarks, idx):
    lm = world_landmarks[idx]
    return np.array([lm.x, lm.y, lm.z])

def calc_angle_3d(p1, p2, p3):
    if p1 is None or p2 is None or p3 is None: return 0.0
    v1 = p1 - p2
    v2 = p3 - p2
    v1_u = v1 / (np.linalg.norm(v1) + 1e-8)
    v2_u = v2 / (np.linalg.norm(v2) + 1e-8)
    angle_rad = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    return np.degrees(angle_rad)

def calc_hip_line_error_3d(shoulder_center, hip_center, ankle_center):
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

def calc_hip_deviation_live(pose_landmarks, shoulder_idx, knee_idx, hip_idx):
    try:
        expected = (pose_landmarks[shoulder_idx].y + pose_landmarks[knee_idx].y) / 2.0
        return pose_landmarks[hip_idx].y - expected
    except Exception: return 0.0

def extract_features_from_task(pose_landmarks, world_landmarks, cache):
    eng_feat_dict = {}
    
    # --- 1. אינדקסים בסיסיים של נקודות הציון ---
    l_shoulder = MP_LANDMARK_MAP["left_shoulder"]
    r_shoulder = MP_LANDMARK_MAP["right_shoulder"]
    l_hip_idx = MP_LANDMARK_MAP["left_hip"]
    r_hip_idx = MP_LANDMARK_MAP["right_hip"]
    
    # --- 2. מדידות פלג גוף עליון (2D) כבסיס לנרמול ---
    l_torso = calc_dist(pose_landmarks, l_shoulder, l_hip_idx)
    r_torso = calc_dist(pose_landmarks, r_shoulder, r_hip_idx)
    avg_torso = (l_torso + r_torso) / 2.0
    if avg_torso == 0: avg_torso = 0.0001
    eng_feat_dict['avg_torso_px'] = avg_torso
        
    # --- 3. פיצ'רים דו-ממדיים בסיסיים (מתוך הקוד הישן) ---
    eng_feat_dict['left_body_angle'] = calc_angle_live(pose_landmarks, l_shoulder, l_hip_idx, MP_LANDMARK_MAP["left_heel"])
    eng_feat_dict['right_body_angle'] = calc_angle_live(pose_landmarks, r_shoulder, r_hip_idx, MP_LANDMARK_MAP["right_heel"])
    eng_feat_dict['avg_body_angle'] = (eng_feat_dict['left_body_angle'] + eng_feat_dict['right_body_angle']) / 2.0

    eng_feat_dict['left_angle_elbow'] = calc_angle_live(pose_landmarks, l_shoulder, MP_LANDMARK_MAP["left_elbow"], MP_LANDMARK_MAP["left_wrist"])
    eng_feat_dict['right_angle_elbow'] = calc_angle_live(pose_landmarks, r_shoulder, MP_LANDMARK_MAP["right_elbow"], MP_LANDMARK_MAP["right_wrist"])
    eng_feat_dict['avg_elbow_angle'] = (eng_feat_dict['left_angle_elbow'] + eng_feat_dict['right_angle_elbow']) / 2.0
    eng_feat_dict['elbow_symmetry'] = abs(eng_feat_dict['left_angle_elbow'] - eng_feat_dict['right_angle_elbow'])
    
    eng_feat_dict['left_hip_deviation_norm'] = calc_hip_deviation_live(pose_landmarks, l_shoulder, MP_LANDMARK_MAP["left_knee"], l_hip_idx) / avg_torso
    eng_feat_dict['right_hip_deviation_norm'] = calc_hip_deviation_live(pose_landmarks, r_shoulder, MP_LANDMARK_MAP["right_knee"], r_hip_idx) / avg_torso

    # --- 4. תוספות 2D: מרחקים, יחסים ו-Y ---
    eng_feat_dict['left_arm_distance'] = calc_dist(pose_landmarks, l_shoulder, MP_LANDMARK_MAP["left_wrist"])
    eng_feat_dict['right_arm_distance'] = calc_dist(pose_landmarks, r_shoulder, MP_LANDMARK_MAP["right_wrist"])
    eng_feat_dict['left_arm_index_shoulder'] = calc_dist(pose_landmarks, l_shoulder, MP_LANDMARK_MAP["left_index"])
    eng_feat_dict['right_arm_index_shoulder'] = calc_dist(pose_landmarks, r_shoulder, MP_LANDMARK_MAP["right_index"])

    eng_feat_dict['left_arm_ratio'] = eng_feat_dict['left_arm_distance'] / avg_torso
    eng_feat_dict['right_arm_ratio'] = eng_feat_dict['right_arm_distance'] / avg_torso
    
    left_shoulder_y = pose_landmarks[l_shoulder].y
    left_elbow_y = pose_landmarks[MP_LANDMARK_MAP["left_elbow"]].y
    right_shoulder_y = pose_landmarks[r_shoulder].y
    right_elbow_y = pose_landmarks[MP_LANDMARK_MAP["right_elbow"]].y
    eng_feat_dict['left_shoulder_elbow_y_norm'] = (left_shoulder_y - left_elbow_y) / avg_torso
    eng_feat_dict['right_shoulder_elbow_y_norm'] = (right_shoulder_y - right_elbow_y) / avg_torso

    # --- 5. תוספות 2D: זוויות נוספות ---
    eng_feat_dict['left_wrist_shoulder_hip'] = calc_angle_live(pose_landmarks, MP_LANDMARK_MAP["left_wrist"], l_shoulder, l_hip_idx)
    eng_feat_dict['right_wrist_shoulder_hip'] = calc_angle_live(pose_landmarks, MP_LANDMARK_MAP["right_wrist"], r_shoulder, r_hip_idx)
    eng_feat_dict['left_knee_angle'] = calc_angle_live(pose_landmarks, l_hip_idx, MP_LANDMARK_MAP["left_knee"], MP_LANDMARK_MAP["left_ankle"])
    eng_feat_dict['right_knee_angle'] = calc_angle_live(pose_landmarks, r_hip_idx, MP_LANDMARK_MAP["right_knee"], MP_LANDMARK_MAP["right_ankle"])
    eng_feat_dict['neck_angle'] = calc_angle_live(pose_landmarks, MP_LANDMARK_MAP["nose"], l_shoulder, l_hip_idx)

    # --- 6. פיצ'רים תלת-ממדיים (3D) ---
    if world_landmarks:
        # חילוץ בטוח של כל הנקודות ב-3D
        ls = get_pt3d(world_landmarks, MP_LANDMARK_MAP['left_shoulder'])
        rs = get_pt3d(world_landmarks, MP_LANDMARK_MAP['right_shoulder'])
        lh = get_pt3d(world_landmarks, MP_LANDMARK_MAP['left_hip'])
        rh = get_pt3d(world_landmarks, MP_LANDMARK_MAP['right_hip'])
        la = get_pt3d(world_landmarks, MP_LANDMARK_MAP['left_ankle'])
        ra = get_pt3d(world_landmarks, MP_LANDMARK_MAP['right_ankle'])
        le = get_pt3d(world_landmarks, MP_LANDMARK_MAP['left_elbow'])
        re = get_pt3d(world_landmarks, MP_LANDMARK_MAP['right_elbow'])
        lw = get_pt3d(world_landmarks, MP_LANDMARK_MAP['left_wrist'])
        rw = get_pt3d(world_landmarks, MP_LANDMARK_MAP['right_wrist'])
        
        shoulder_center = (ls + rs) / 2.0 if ls is not None and rs is not None else None
        hip_center = (lh + rh) / 2.0 if lh is not None and rh is not None else None
        ankle_center = (la + ra) / 2.0 if la is not None and ra is not None else None
        
        # חישוב יישור גוף וקו אגן
        eng_feat_dict['body_alignment_angle'] = calc_angle_3d(shoulder_center, hip_center, ankle_center)
        eng_feat_dict['hip_line_error'] = calc_hip_line_error_3d(shoulder_center, hip_center, ankle_center)
        
        # פונקציית עזר פנימית בטוחה למרחק ב-3D
        def dist3d(p1, p2): 
            if p1 is None or p2 is None: return 0.0
            return np.linalg.norm(p1 - p2)
            
        # דחיסת זרוע
        l_arm_len = dist3d(ls, le) + dist3d(le, lw)
        r_arm_len = dist3d(rs, re) + dist3d(re, rw)
        
        eng_feat_dict['left_arm_compression'] = dist3d(ls, lw) / l_arm_len if l_arm_len > 0 else 0.0
        eng_feat_dict['right_arm_compression'] = dist3d(rs, rw) / r_arm_len if r_arm_len > 0 else 0.0
        eng_feat_dict['avg_arm_compression'] = (eng_feat_dict['left_arm_compression'] + eng_feat_dict['right_arm_compression']) / 2.0
        
    else:
        # ערכי ברירת מחדל אם אין זיהוי 3D בפריים הנוכחי
        eng_feat_dict['body_alignment_angle'] = 0.0
        eng_feat_dict['hip_line_error'] = 0.0
        eng_feat_dict['left_arm_compression'] = 0.0
        eng_feat_dict['right_arm_compression'] = 0.0
        eng_feat_dict['avg_arm_compression'] = 0.0

    # --- 7. דלתאות (Deltas) מבוססות זמן בעזרת ה-Cache ---
    prev_avg_elbow = cache.get('prev_avg_elbow_angle', eng_feat_dict['avg_elbow_angle'])
    prev_l_elbow = cache.get('prev_left_angle_elbow', eng_feat_dict['left_angle_elbow'])
    prev_r_elbow = cache.get('prev_right_angle_elbow', eng_feat_dict['right_angle_elbow'])
    prev_hip_error = cache.get('prev_hip_line_error', eng_feat_dict['hip_line_error'])
    prev_body_align = cache.get('prev_body_alignment', eng_feat_dict['body_alignment_angle'])
    
    eng_feat_dict['delta_left_elbow_angle'] = eng_feat_dict['left_angle_elbow'] - prev_l_elbow
    eng_feat_dict['delta_right_elbow_angle'] = eng_feat_dict['right_angle_elbow'] - prev_r_elbow
    eng_feat_dict['avg_delta_elbow_angle'] = eng_feat_dict['avg_elbow_angle'] - prev_avg_elbow
    eng_feat_dict['delta_hip_line_error'] = eng_feat_dict['hip_line_error'] - prev_hip_error
    eng_feat_dict['delta_body_alignment_angle'] = eng_feat_dict['body_alignment_angle'] - prev_body_align
    
    # עדכון ה-Cache לפריים הבא
    cache['prev_avg_elbow_angle'] = eng_feat_dict['avg_elbow_angle']
    cache['prev_left_angle_elbow'] = eng_feat_dict['left_angle_elbow']
    cache['prev_right_angle_elbow'] = eng_feat_dict['right_angle_elbow']
    cache['prev_hip_line_error'] = eng_feat_dict['hip_line_error']
    cache['prev_body_alignment'] = eng_feat_dict['body_alignment_angle']

    return eng_feat_dict
    features = []
    l_hip_idx = MP_LANDMARK_MAP["left_hip"]
    r_hip_idx = MP_LANDMARK_MAP["right_hip"]
    mid_x = (pose_landmarks[l_hip_idx].x + pose_landmarks[r_hip_idx].x) / 2.0
    mid_y = (pose_landmarks[l_hip_idx].y + pose_landmarks[r_hip_idx].y) / 2.0
    mid_z = (pose_landmarks[l_hip_idx].z + pose_landmarks[r_hip_idx].z) / 2.0

    # for lm_name in selected_landmarks:
    #     idx = MP_LANDMARK_MAP[lm_name]
    #     lm = pose_landmarks[idx]
    #     if use_centered:
    #         features.extend([lm.x - mid_x, lm.y - mid_y, lm.z - mid_z, lm.visibility])
    #     else:
    #         features.extend([lm.x, lm.y, lm.z, lm.visibility])
        
    l_shoulder = MP_LANDMARK_MAP["left_shoulder"]
    r_shoulder = MP_LANDMARK_MAP["right_shoulder"]
    l_torso = calc_dist(pose_landmarks, l_shoulder, l_hip_idx)
    r_torso = calc_dist(pose_landmarks, r_shoulder, r_hip_idx)
    avg_torso = (l_torso + r_torso) / 2.0
    if avg_torso == 0: avg_torso = 0.0001
        
    eng_feat_dict = {}
    eng_feat_dict['left_body_angle'] = calc_angle_live(pose_landmarks, l_shoulder, l_hip_idx, MP_LANDMARK_MAP["left_heel"])
    eng_feat_dict['right_body_angle'] = calc_angle_live(pose_landmarks, r_shoulder, r_hip_idx, MP_LANDMARK_MAP["right_heel"])
    eng_feat_dict['left_angle_elbow'] = calc_angle_live(pose_landmarks, l_shoulder, MP_LANDMARK_MAP["left_elbow"], MP_LANDMARK_MAP["left_wrist"])
    eng_feat_dict['right_angle_elbow'] = calc_angle_live(pose_landmarks, r_shoulder, MP_LANDMARK_MAP["right_elbow"], MP_LANDMARK_MAP["right_wrist"])
    
    eng_feat_dict['left_hip_deviation_norm'] = calc_hip_deviation_live(pose_landmarks, l_shoulder, MP_LANDMARK_MAP["left_knee"], l_hip_idx) / avg_torso
    eng_feat_dict['right_hip_deviation_norm'] = calc_hip_deviation_live(pose_landmarks, r_shoulder, MP_LANDMARK_MAP["right_knee"], r_hip_idx) / avg_torso

    # --- הפיצ'רים החדשים שלך ---
    if world_landmarks:
        # שליפת כל הנקודות הדרושות ב-3D (כתפיים, מרפקים, שורשי כף יד)
        ls = get_pt3d(world_landmarks, MP_LANDMARK_MAP['left_shoulder'])
        rs = get_pt3d(world_landmarks, MP_LANDMARK_MAP['right_shoulder'])
        le = get_pt3d(world_landmarks, MP_LANDMARK_MAP['left_elbow'])
        re = get_pt3d(world_landmarks, MP_LANDMARK_MAP['right_elbow'])
        lw = get_pt3d(world_landmarks, MP_LANDMARK_MAP['left_wrist'])
        rw = get_pt3d(world_landmarks, MP_LANDMARK_MAP['right_wrist'])
        
        # פונקציית עזר קטנה לחישוב מרחק 3D בתוך הפונקציה
        def dist3d(p1, p2): 
            if p1 is None or p2 is None: return 0.0
            return np.linalg.norm(p1 - p2)
        
        l_arm_len = dist3d(ls, le) + dist3d(le, lw)
        r_arm_len = dist3d(rs, re) + dist3d(re, rw)
        
        eng_feat_dict['left_arm_compression'] = dist3d(ls, lw) / l_arm_len if l_arm_len > 0 else 0.0
        eng_feat_dict['right_arm_compression'] = dist3d(rs, rw) / r_arm_len if r_arm_len > 0 else 0.0
        eng_feat_dict['avg_arm_compression'] = (eng_feat_dict['left_arm_compression'] + eng_feat_dict['right_arm_compression']) / 2.0
    else:
        eng_feat_dict['left_arm_compression'] = 0.0
        eng_feat_dict['right_arm_compression'] = 0.0
        eng_feat_dict['avg_arm_compression'] = 0.0

    # 1. ממוצע זווית מרפקים
    avg_elbow = (eng_feat_dict['left_angle_elbow'] + eng_feat_dict['right_angle_elbow']) / 2.0
    eng_feat_dict['avg_elbow_angle'] = avg_elbow

# 1. מרחקים 2D
    eng_feat_dict['left_arm_distance'] = calc_dist(pose_landmarks, l_shoulder, MP_LANDMARK_MAP["left_wrist"])
    eng_feat_dict['right_arm_distance'] = calc_dist(pose_landmarks, r_shoulder, MP_LANDMARK_MAP["right_wrist"])
    eng_feat_dict['left_arm_index_shoulder'] = calc_dist(pose_landmarks, l_shoulder, MP_LANDMARK_MAP["left_index"])
    eng_feat_dict['right_arm_index_shoulder'] = calc_dist(pose_landmarks, r_shoulder, MP_LANDMARK_MAP["right_index"])

    # 2. יחסים ומנרמלים 2D
    eng_feat_dict['left_arm_ratio'] = eng_feat_dict['left_arm_distance'] / avg_torso
    eng_feat_dict['right_arm_ratio'] = eng_feat_dict['right_arm_distance'] / avg_torso
    
    # חישוב הפרשי Y מנורמלים
    left_shoulder_y = pose_landmarks[l_shoulder].y
    left_elbow_y = pose_landmarks[MP_LANDMARK_MAP["left_elbow"]].y
    right_shoulder_y = pose_landmarks[r_shoulder].y
    right_elbow_y = pose_landmarks[MP_LANDMARK_MAP["right_elbow"]].y
    eng_feat_dict['left_shoulder_elbow_y_norm'] = (left_shoulder_y - left_elbow_y) / avg_torso
    eng_feat_dict['right_shoulder_elbow_y_norm'] = (right_shoulder_y - right_elbow_y) / avg_torso

    # 3. זוויות 2D נוספות
    eng_feat_dict['left_wrist_shoulder_hip'] = calc_angle_live(pose_landmarks, MP_LANDMARK_MAP["left_wrist"], l_shoulder, l_hip_idx)
    eng_feat_dict['right_wrist_shoulder_hip'] = calc_angle_live(pose_landmarks, MP_LANDMARK_MAP["right_wrist"], r_shoulder, r_hip_idx)
    eng_feat_dict['left_knee_angle'] = calc_angle_live(pose_landmarks, l_hip_idx, MP_LANDMARK_MAP["left_knee"], MP_LANDMARK_MAP["left_ankle"])
    eng_feat_dict['right_knee_angle'] = calc_angle_live(pose_landmarks, r_hip_idx, MP_LANDMARK_MAP["right_knee"], MP_LANDMARK_MAP["right_ankle"])
    eng_feat_dict['neck_angle'] = calc_angle_live(pose_landmarks, MP_LANDMARK_MAP["nose"], l_shoulder, l_hip_idx)
    
    eng_feat_dict['elbow_symmetry'] = abs(eng_feat_dict['left_angle_elbow'] - eng_feat_dict['right_angle_elbow'])
    eng_feat_dict['avg_body_angle'] = (eng_feat_dict['left_body_angle'] + eng_feat_dict['right_body_angle']) / 2.0

    # 4. פיצ'רים תלת-ממדיים (יש למזג את זה לתוך בלוק ה- if world_landmarks הקיים)
    if world_landmarks:
        # שליפת נקודות נוספות למרפקים ושורש כף היד ב-3D
        le = get_pt3d(world_landmarks, MP_LANDMARK_MAP['left_elbow'])
        re = get_pt3d(world_landmarks, MP_LANDMARK_MAP['right_elbow'])
        lw = get_pt3d(world_landmarks, MP_LANDMARK_MAP['left_wrist'])
        rw = get_pt3d(world_landmarks, MP_LANDMARK_MAP['right_wrist'])
        
        # פונקציית עזר קטנה לחישוב מרחק 3D בתוך הפונקציה
        def dist3d(p1, p2): return np.linalg.norm(p1 - p2)
        
        l_arm_len = dist3d(ls, le) + dist3d(le, lw)
        r_arm_len = dist3d(rs, re) + dist3d(re, rw)
        
        eng_feat_dict['left_arm_compression'] = dist3d(ls, lw) / l_arm_len if l_arm_len > 0 else 0.0
        eng_feat_dict['right_arm_compression'] = dist3d(rs, rw) / r_arm_len if r_arm_len > 0 else 0.0
        eng_feat_dict['avg_arm_compression'] = (eng_feat_dict['left_arm_compression'] + eng_feat_dict['right_arm_compression']) / 2.0
    else:
        eng_feat_dict['left_arm_compression'] = 0.0
        eng_feat_dict['right_arm_compression'] = 0.0
        eng_feat_dict['avg_arm_compression'] = 0.0

    # 5. דלתאות (Deltas) נוספות מבוססות זמן
    prev_l_elbow = cache.get('prev_left_angle_elbow', eng_feat_dict['left_angle_elbow'])
    prev_r_elbow = cache.get('prev_right_angle_elbow', eng_feat_dict['right_angle_elbow'])
    
    eng_feat_dict['delta_left_elbow_angle'] = eng_feat_dict['left_angle_elbow'] - prev_l_elbow
    eng_feat_dict['delta_right_elbow_angle'] = eng_feat_dict['right_angle_elbow'] - prev_r_elbow
    
    # עדכון ה-Cache לפריים הבא
    cache['prev_left_angle_elbow'] = eng_feat_dict['left_angle_elbow']
    cache['prev_right_angle_elbow'] = eng_feat_dict['right_angle_elbow']



    
# 3. חישוב דלתאות זמניות (Deltas) בעזרת ה-cache
    prev_avg_elbow = cache.get('prev_avg_elbow_angle', avg_elbow)
    prev_hip_error = cache.get('prev_hip_line_error', hip_error)
    prev_body_align = cache.get('prev_body_alignment', body_align)
    
    # שימוש בשמות המדויקים כפי שהם ב-CSV
    eng_feat_dict['avg_delta_elbow_angle'] = avg_elbow - prev_avg_elbow
    eng_feat_dict['delta_hip_line_error'] = hip_error - prev_hip_error
    eng_feat_dict['delta_body_alignment_angle'] = body_align - prev_body_align
    
    # עדכון ה-cache לפריים הבא
    cache['prev_avg_elbow_angle'] = avg_elbow
    cache['prev_hip_line_error'] = hip_error
    cache['prev_body_alignment'] = body_align


    return eng_feat_dict
# ==========================================
# 3. ML MODELS SETUP
# ==========================================
def get_feature_columns(selected_landmarks, selected_engineered_features, use_centered):
    cols = []
    # for lm in selected_landmarks:
    #     if use_centered:
    #         cols.extend([f"{lm}_centered_x", f"{lm}_centered_y", f"{lm}_centered_z", f"{lm}_visibility"])
    #     else:
    #         cols.extend([f"{lm}_x", f"{lm}_y", f"{lm}_z", f"{lm}_visibility"])
    cols.extend(selected_engineered_features)
    return cols

@st.cache_resource
@st.cache_resource
def load_and_train_models():
    path = os.path.join("data", "*.csv")
    files = glob.glob(path)
    if not files: return None
    dfs = [pd.read_csv(f) for f in files]
    full_data = pd.concat(dfs, ignore_index=True)
    
    full_data = full_data[full_data['is_valid_frame'] == True]
    
    trained_models = {}
    
    for model_name, config in MODELS_CONFIG.items():
        
        model_data = full_data.dropna(subset=[config["target_col"]])
        
        X_train = model_data[config["features"]]
        y_train = model_data[config["target_col"]]
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.values)    
        
        pca = None
        if config["use_pca"]:
            pca = PCA(n_components=config["pca_components"])
            X_train_final = pca.fit_transform(X_train_scaled)
        else:
            X_train_final = X_train_scaled

        knn = KNeighborsClassifier(n_neighbors=config["k_neighbors"]).fit(X_train_final, y_train)
        
        trained_models[model_name] = {
            "scaler": scaler,
            "pca": pca,
            "knn": knn
        }
        
    return trained_models

trained_models = load_and_train_models()

MODEL_PATH = 'pose_landmarker_heavy.task'
if not os.path.exists(MODEL_PATH):
    st.error(f"⚠️ Model file '{MODEL_PATH}' not found!")
    st.stop()

@st.cache_resource
def get_landmarker():
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH), 
        running_mode=vision.RunningMode.VIDEO,
        output_segmentation_masks=False,
        num_poses=1,                    
        min_pose_detection_confidence=0.5, 
        min_tracking_confidence=0.5      
    )
    return vision.PoseLandmarker.create_from_options(options)

landmarker = get_landmarker()

# ==========================================
# 4. UNIFIED PROCESSING PIPELINE
# ==========================================
from mediapipe.tasks.python.components.containers import NormalizedRect

def process_frame(frame, rep_sm, hip_sm, cache, timestamp_ms, is_live=False, start_time=0, process_this_frame=True, frame_num=0):
    frame = cv2.resize(frame, (640, 480))
    h, w, _ = frame.shape
    debug_info = None
    
    if is_live:
        elapsed = time.time() - start_time
        if elapsed < 10:
            countdown = int(10 - elapsed)
            cv2.putText(frame, f"Get Redfdfady! {countdown}s", (int(w/2) - 150, int(h/2)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 165, 255), 4)
            return frame, debug_info
    
    if process_this_frame:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        t_mp_start = time.time()
        res = landmarker.detect_for_video(mp_image, timestamp_ms)
        mp_time = (time.time() - t_mp_start) * 1000
        delta_ms = 0
        feat_time, pred_time, sm_time = 0.0, 0.0, 0.0
        phase_pred_str, hip_pred_str = "None", "None"
        last_ts = cache.get('last_processed_ts', 0)
        if last_ts > 0:
            delta_ms = timestamp_ms - last_ts
        if res.pose_landmarks:
            pose_landmarks = res.pose_landmarks[0]
            world_landmarks = res.pose_world_landmarks[0] if res.pose_world_landmarks else None
                
            t_feat_start = time.time()
            all_features_dict = extract_features_from_task(
                pose_landmarks, world_landmarks,cache
            )
            feat_time = (time.time() - t_feat_start) * 1000
            
            t_pred_start = time.time()
            
            # --- 1. מסלול חיזוי שלב שכיבת סמיכה (Phase) ---
            phase_config = MODELS_CONFIG["phase"]
            phase_model = trained_models["phase"]
            
            # שולפים רק את הפיצ'רים הרלוונטיים ומסדרים אותם כמטריצה של שורה אחת
            phase_features = np.array([[all_features_dict.get(f, 0.0) for f in phase_config["features"]]])
            
            # נרמול ו-PCA (אם יש) שמיוחדים למודל הזה
            phase_scaled = phase_model["scaler"].transform(phase_features)
            phase_final = phase_model["pca"].transform(phase_scaled) if phase_model["pca"] else phase_scaled
            phase_pred = phase_model["knn"].predict(phase_final)[0]

            # --- 2. מסלול חיזוי מנח אגן (Hips) ---
            hips_config = MODELS_CONFIG["hips"]
            hips_model = trained_models["hips"]
            
            # שולפים רק את הפיצ'רים הרלוונטיים לאגן
            hips_features = np.array([[all_features_dict.get(f, 0.0) for f in hips_config["features"]]])
            
            # נרמול ו-PCA (אם יש) שמיוחדים למודל האגן
            hips_scaled = hips_model["scaler"].transform(hips_features)
            hips_final = hips_model["pca"].transform(hips_scaled) if hips_model["pca"] else hips_scaled
            hip_pred = hips_model["knn"].predict(hips_final)[0]


            
            phase_pred_str = str(phase_pred)
            hip_pred_str = str(hip_pred)
            pred_time = (time.time() - t_pred_start) * 1000

            t_sm_start = time.time()
            rep_sm.process(phase_pred)
            hip_sm.process(hip_pred)
            sm_time = (time.time() - t_sm_start) * 1000
            
            cache['last_landmarks'] = pose_landmarks
        else:
            cache['last_landmarks'] = None

        # יצירת מופע של אובייקט ה-Debug
       # debug_info = FrameDebugData(frame_num, phase_pred_str, hip_pred_str, mp_time, feat_time, pred_time, sm_time)
        # חילוץ הנתונים ממכונות המצבים
        current_rep_count = rep_sm.rep_count
        current_sm_state = rep_sm.state
        current_hip_status = hip_sm.status

        debug_info = FrameDebugData(
            frame_num, process_this_frame, timestamp_ms, delta_ms, 
            phase_pred_str, hip_pred_str, mp_time, feat_time, pred_time, sm_time, 
            current_rep_count, current_sm_state, current_hip_status
        )

    # ציור על גבי הוידאו
    if cache.get('last_landmarks'):
        # for lm_name in SELECTED_LANDMARKS:
        #     lm = cache['last_landmarks'][MP_LANDMARK_MAP[lm_name]]
        #     cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 6, (0, 255, 0), -1)
        pass
    else:
        cv2.putText(frame, "No Pose Detected", (int(w/2) - 150, int(h/2)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.rectangle(frame, (10, 200), (300, 350), (0, 0, 0), -1)
    cv2.putText(frame, f"Reps: {rep_sm.rep_count}", (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    color = (0, 255, 0) if hip_sm.status == "Good Form" else (0, 0, 255)
    cv2.putText(frame, f"Hips: {hip_sm.status}", (20,300), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    if rep_sm.last_issue:
        cv2.putText(frame, f"Alert: {rep_sm.last_issue}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        
    return frame, debug_info

# ==========================================
# 5. UI & TABS
# ==========================================
st.title("🤖 A-EYE TRAINER: Live & Offline Predictor")

if 'offline_rep_sm' not in st.session_state:
    st.session_state.offline_rep_sm = RepStateMachine()
if 'offline_hip_sm' not in st.session_state:
    st.session_state.offline_hip_sm = HipStateMachine()
if 'global_timestamp_ms' not in st.session_state:
    st.session_state.global_timestamp_ms = int(time.time() * 1000)

tab_live, tab_video = st.tabs(["🔴 Live Camera", "🎥 Upload Video"])
    
with tab_live:
    st.subheader("Live Real-Time Processing")
    
    class VideoProcessor(VideoTransformerBase):
        def __init__(self):
            self.start_time = time.time()
            self.rep_sm = RepStateMachine()
            self.hip_sm = HipStateMachine()
            self.cache = {'last_landmarks': None}
            self.last_process_time = 0
            self.fps = 30
            self.last_timestamp_ms = -1 
            
            # הוספת משתנים לתור ה-Debug
            self.frame_count = 0
            self.debug_queue = []
            
        def recv(self, frame):
            self.frame_count += 1
            
            img = frame.to_ndarray(format="bgr24")
            h, w, _ = img.shape
            current_time = time.time()
            
            elapsed = current_time - self.start_time
            if elapsed < 10:
                countdown = int(10 - elapsed)
                cv2.putText(img, f"Get Ready! {countdown}s", (int(w/2) - 150, int(h/2)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 165, 255), 4)
                cv2.putText(img, f"Place camera at 9 o'clock", (int(w/2) - 130, int(h/2) + 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                return av.VideoFrame.from_ndarray(img, format="bgr24")
            
            timestamp_ms = int(current_time * 1000)
            if timestamp_ms <= self.last_timestamp_ms:
                timestamp_ms = self.last_timestamp_ms + 1
            self.last_timestamp_ms = timestamp_ms
            
            process_this_frame = (current_time - self.last_process_time) >= 1/self.fps
            if process_this_frame:
                self.last_process_time = current_time
                
            # קבלת האובייקט המעובד ואובייקט ה-Debug
            processed_img, debug_info = process_frame(
                img, self.rep_sm, self.hip_sm, self.cache, timestamp_ms,
                is_live=True, start_time=self.start_time, process_this_frame=process_this_frame, frame_num=self.frame_count
            )
            
            # הכנסה לתור במידה ועובד פריים
            if debug_info:
                self.debug_queue.append(debug_info)
            
            return av.VideoFrame.from_ndarray(processed_img, format="bgr24")
            
        # פונקציה מובנית שנקראת אוטומטית כשהחיבור מתנתק / השידור נעצר
        def on_ended(self):
            flush_debug_queue(self.debug_queue, "Live Camera")

    webrtc_streamer(
        key="a-eye-live", 
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        
    )

with tab_video:
    st.subheader("Offline Video Processing")
    uploaded_video = st.file_uploader("Upload an MP4 video", type=['mp4', 'mov', 'avi'])
    
    if uploaded_video is not None:
        if st.button("Start Processing Video"):
            st.session_state.offline_rep_sm = RepStateMachine()
            st.session_state.offline_hip_sm = HipStateMachine()
            
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_video.read())
            tfile.close()
            
            cap = cv2.VideoCapture(tfile.name)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0: fps = 10
            
            st_frame = st.empty() 
            frame_count = 0
            offline_cache = {'last_landmarks': None}
            frame_skip_interval =2
            
            # יצירת תור ה-Debug לוידאו האופליין
            offline_debug_queue = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                frame_count += 1
                
                if (frame_count % frame_skip_interval == 0):
                    st.session_state.global_timestamp_ms += int((frame_skip_interval / fps) * 1000)
                
                timestamp_ms = st.session_state.global_timestamp_ms
                process_this_frame = (frame_count % frame_skip_interval == 0)
                
                # קבלת התמונה ואובייקט ה-Debug
                processed_frame, debug_info = process_frame(
                    frame, st.session_state.offline_rep_sm, st.session_state.offline_hip_sm, 
                    offline_cache, timestamp_ms, is_live=False, process_this_frame=process_this_frame, frame_num=frame_count
                )
                
                # הכנסה לתור
                if debug_info:
                    offline_debug_queue.append(debug_info)
                
                rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                st_frame.image(rgb_frame, channels="RGB", width="stretch")
                
            cap.release()
            os.unlink(tfile.name)
            
            # ריקון התור לקובץ בסוף הריצה
            flush_debug_queue(offline_debug_queue, f"Offline Video - {uploaded_video.name}")
            
            st.success(f"✅ Processing complete! Total Reps: {st.session_state.offline_rep_sm.rep_count}")