import pandas as pd
import numpy as np
import glob
import os

def calculate_angle(row, p1, p2, p3):
    """Original 2D angle calculation."""
    try:
        x1, y1 = float(row[f'{p1}_x']), float(row[f'{p1}_y']) 
        x2, y2 = float(row[f'{p2}_x']), float(row[f'{p2}_y']) 
        x3, y3 = float(row[f'{p3}_x']), float(row[f'{p3}_y']) 
        
        if pd.isna(x1) or pd.isna(x2) or pd.isna(x3):
            return np.nan
            
        radians = np.arctan2(y3 - y2, x3 - x2) - np.arctan2(y1 - y2, x1 - x2)
        angle = np.abs(np.degrees(radians))
        if angle > 180.0:
            angle = 360.0 - angle
        return angle
    except Exception:
        return np.nan

def calculate_distance(row, p1, p2):
    """Original 2D distance calculation."""
    try:
        x1, y1 = float(row[f'{p1}_x']), float(row[f'{p1}_y'])
        x2, y2 = float(row[f'{p2}_x']), float(row[f'{p2}_y'])
        
        if pd.isna(x1) or pd.isna(x2):
            return np.nan
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    except Exception:
        return np.nan

def calculate_y_difference(row, p1, p2):
    """Original 2D vertical difference."""
    try:
        y1 = float(row[f'{p1}_y'])
        y2 = float(row[f'{p2}_y'])
        if pd.isna(y1) or pd.isna(y2): return np.nan
        return y1 - y2
    except Exception: return np.nan

def calculate_hip_deviation(row, p_shoulder, p_knee, p_hip):
    """Original 2D hip deviation."""
    try:
        shoulder_y = float(row[f'{p_shoulder}_y'])
        knee_y = float(row[f'{p_knee}_y'])
        hip_y = float(row[f'{p_hip}_y'])
        if pd.isna(shoulder_y) or pd.isna(knee_y) or pd.isna(hip_y): return np.nan
        expected_hip_y = (shoulder_y + knee_y) / 2.0
        return hip_y - expected_hip_y
    except Exception: return np.nan

# ==============================================================================
# 3D HELPER FUNCTIONS & BIOMECHANICAL CALCULATIONS
# ==============================================================================

def get_3d_point(row, p_name):
    try:
        x, y, z = float(row[f'{p_name}_x']), float(row[f'{p_name}_y']), float(row[f'{p_name}_z'])
        if pd.isna(x) or pd.isna(y) or pd.isna(z): return None
        return np.array([x, y, z])
    except Exception: return None

def calc_dist_3d(p1, p2):
    if p1 is None or p2 is None: return np.nan
    return np.linalg.norm(p1 - p2)

def calc_angle_3d(p1, p2, p3):
    if p1 is None or p2 is None or p3 is None: return np.nan
    v1 = p1 - p2
    v2 = p3 - p2
    v1_u = v1 / (np.linalg.norm(v1) + 1e-8)
    v2_u = v2 / (np.linalg.norm(v2) + 1e-8)
    angle_rad = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    return np.degrees(angle_rad)

def calc_hip_line_error_3d(shoulder_center, hip_center, ankle_center):
    if shoulder_center is None or hip_center is None or ankle_center is None: return np.nan
    line_vec = ankle_center - shoulder_center
    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-8: return np.nan
    
    line_dir = line_vec / line_len
    vec_to_hip = hip_center - shoulder_center
    projection_length = np.dot(vec_to_hip, line_dir)
    projected_point = shoulder_center + projection_length * line_dir
    
    diff_vec = hip_center - projected_point
    distance = np.linalg.norm(diff_vec)
    sign = 1.0 if diff_vec[1] > 0 else -1.0
    return distance * sign

# ==============================================================================

def process_all_csvs(target_folders):
    total_processed = 0

    for folder in target_folders:
        print(f"\n🔍 Scanning directory: '{folder}'")
        path = os.path.join(folder, "*.csv")
        csv_files = glob.glob(path)
        if not csv_files: continue

        print(f"📂 Found {len(csv_files)} CSV files in '{folder}'. Starting to process...")

        for file in csv_files:
            df = pd.read_csv(file)
            
            if 'left_shoulder_x' in df.columns and 'left_hip_x' in df.columns and 'left_heel_x' in df.columns:
                
                # --- CENTERING CALCS ---
                df['mid_hip_x'] = (df['left_hip_x'] + df['right_hip_x']) / 2.0
                df['mid_hip_y'] = (df['left_hip_y'] + df['right_hip_y']) / 2.0
                df['mid_hip_z'] = (df['left_hip_z'] + df['right_hip_z']) / 2.0

                landmarks = [col.replace('_visibility', '') for col in df.columns if col.endswith('_visibility')]
                for lm in landmarks:
                    df[f'{lm}_centered_x'] = df[f'{lm}_x'] - df['mid_hip_x']
                    df[f'{lm}_centered_y'] = df[f'{lm}_y'] - df['mid_hip_y']
                    df[f'{lm}_centered_z'] = df[f'{lm}_z'] - df['mid_hip_z']

                # --- 1. EXISTING 2D FEATURES ---
                df['left_body_angle'] = df.apply(lambda row: calculate_angle(row, 'left_shoulder', 'left_hip', 'left_heel') , axis=1)
                df['right_body_angle'] = df.apply(lambda row: calculate_angle(row, 'right_shoulder', 'right_hip', 'right_heel'), axis=1)
                df['left_angle_elbow'] = df.apply(lambda row: calculate_angle(row, 'left_shoulder', 'left_elbow', 'left_wrist'), axis=1)
                df['right_angle_elbow'] = df.apply(lambda row: calculate_angle(row, 'right_shoulder', 'right_elbow', 'right_wrist'), axis=1)
                df['right_wrist_shoulder_hip'] = df.apply(lambda row: calculate_angle(row, 'right_wrist', 'right_shoulder', 'right_hip'), axis=1)
                df['left_wrist_shoulder_hip'] = df.apply(lambda row: calculate_angle(row, 'left_wrist', 'left_shoulder', 'left_hip'), axis=1)
                
                df['left_arm_distance'] = df.apply(lambda row: calculate_distance(row, 'left_shoulder', 'left_wrist'), axis=1)
                df['right_arm_distance'] = df.apply(lambda row: calculate_distance(row, 'right_shoulder', 'right_wrist'), axis=1)
                df['left_arm_index_shoulder'] = df.apply(lambda row: calculate_distance(row, 'left_shoulder', 'left_index'), axis=1)
                df['right_arm_index_shoulder'] = df.apply(lambda row: calculate_distance(row, 'right_shoulder', 'right_index'), axis=1)
                
                df['left_torso_px'] = df.apply(lambda row: calculate_distance(row, 'left_shoulder', 'left_hip'), axis=1)
                df['right_torso_px'] = df.apply(lambda row: calculate_distance(row, 'right_shoulder', 'right_hip'), axis=1)
                df['avg_torso_px'] = (df['left_torso_px'] + df['right_torso_px']) / 2.0
                df['avg_torso_px'] = df['avg_torso_px'].replace(0, np.nan)

                df['left_arm_ratio'] = df['left_arm_distance'] / df['avg_torso_px']
                df['right_arm_ratio'] = df['right_arm_distance'] / df['avg_torso_px']
                df['left_shoulder_elbow_y_norm'] = df.apply(lambda row: calculate_y_difference(row, 'left_shoulder', 'left_elbow'), axis=1) / df['avg_torso_px']
                df['right_shoulder_elbow_y_norm'] = df.apply(lambda row: calculate_y_difference(row, 'right_shoulder', 'right_elbow'), axis=1) / df['avg_torso_px']
                df['left_hip_deviation_norm'] = df.apply(lambda row: calculate_hip_deviation(row, 'left_shoulder', 'left_knee', 'left_hip'), axis=1) / df['avg_torso_px']
                df['right_hip_deviation_norm'] = df.apply(lambda row: calculate_hip_deviation(row, 'right_shoulder', 'right_knee', 'right_hip'), axis=1) / df['avg_torso_px']

                df['left_knee_angle'] = df.apply(lambda row: calculate_angle(row, 'left_hip', 'left_knee', 'left_ankle'), axis=1)
                df['right_knee_angle'] = df.apply(lambda row: calculate_angle(row, 'right_hip', 'right_knee', 'right_ankle'), axis=1)
                if 'nose_x' in df.columns:
                    df['neck_angle'] = df.apply(lambda row: calculate_angle(row, 'nose', 'left_shoulder', 'left_hip'), axis=1)

                # --- 2. ADVANCED 3D FEATURES ---
                def compute_centers_and_features(row):
                    ls, rs = get_3d_point(row, 'left_shoulder'), get_3d_point(row, 'right_shoulder')
                    lh, rh = get_3d_point(row, 'left_hip'), get_3d_point(row, 'right_hip')
                    la, ra = get_3d_point(row, 'left_ankle'), get_3d_point(row, 'right_ankle')
                    le, re = get_3d_point(row, 'left_elbow'), get_3d_point(row, 'right_elbow')
                    lw, rw = get_3d_point(row, 'left_wrist'), get_3d_point(row, 'right_wrist')
                    
                    shoulder_center = (ls + rs) / 2.0 if ls is not None and rs is not None else None
                    hip_center = (lh + rh) / 2.0 if lh is not None and rh is not None else None
                    ankle_center = (la + ra) / 2.0 if la is not None and ra is not None else None

                    body_align = calc_angle_3d(shoulder_center, hip_center, ankle_center)
                    hip_error = calc_hip_line_error_3d(shoulder_center, hip_center, ankle_center)
                    
                    l_arm_len = calc_dist_3d(ls, le) + calc_dist_3d(le, lw)
                    r_arm_len = calc_dist_3d(rs, re) + calc_dist_3d(re, rw)
                    l_comp = calc_dist_3d(ls, lw) / l_arm_len if l_arm_len and l_arm_len > 0 else np.nan
                    r_comp = calc_dist_3d(rs, rw) / r_arm_len if r_arm_len and r_arm_len > 0 else np.nan
                    avg_comp = (l_comp + r_comp) / 2.0 if not np.isnan(l_comp) and not np.isnan(r_comp) else np.nan

                    return pd.Series([body_align, hip_error, l_comp, r_comp, avg_comp])

                df[['body_alignment_angle', 'hip_line_error', 'left_arm_compression', 
                    'right_arm_compression', 'avg_arm_compression']] = df.apply(compute_centers_and_features, axis=1)

                df['elbow_symmetry'] = np.abs(df['left_angle_elbow'] - df['right_angle_elbow'])
                df['avg_elbow_angle'] = (df['left_angle_elbow'] + df['right_angle_elbow']) / 2.0
                df['avg_body_angle'] = (df['left_body_angle'] + df['right_body_angle']) / 2.0

                # ==============================================================================
                # ALON SECOND TRY: TEMPORAL MOTION FEATURES (Velocities via Frame Deltas)
                # Biomechanical meaning: Tracks whether angles are expanding or collapsing over time.
                # ==============================================================================
                df['delta_left_elbow_angle'] = df['left_angle_elbow'] - df['left_angle_elbow'].shift(1)
                df['delta_right_elbow_angle'] = df['right_angle_elbow'] - df['right_angle_elbow'].shift(1)
                df['avg_delta_elbow_angle'] = (df['delta_left_elbow_angle'] + df['delta_right_elbow_angle']) / 2.0
                
                df['delta_hip_line_error'] = df['hip_line_error'] - df['hip_line_error'].shift(1)
                df['delta_body_alignment_angle'] = df['body_alignment_angle'] - df['body_alignment_angle'].shift(1)
                
                # Handling the first frame (no previous frame available = delta is 0)
                temporal_cols = ['delta_left_elbow_angle', 'delta_right_elbow_angle', 'avg_delta_elbow_angle', 'delta_hip_line_error', 'delta_body_alignment_angle']
                df[temporal_cols] = df[temporal_cols].fillna(0)
                # ==============================================================================

                df.to_csv(file, index=False)
                print(f"✅ Processed and updated: {file}")
                total_processed += 1
            else:
                print(f"⚠️ Skipped {file} - missing required coordinate columns.")

    print(f"\n🎉 All done! Successfully updated {total_processed} files in their original locations.")

if __name__ == "__main__":
    folders_to_process = ["data", "test data"]
    process_all_csvs(folders_to_process)