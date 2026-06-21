import pandas as pd
import numpy as np
import glob
import os
from sklearn.pipeline import Pipeline
from sklearn.decomposition import KernelPCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# ==========================================
# 1. הגדרות ופיצ'רים
# ==========================================
default_landmarks = [
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", 
    "left_wrist", "right_wrist", "left_hip", "right_hip"
]

engineered_features = [
    'left_body_angle', 'right_body_angle', 
    'left_angle_elbow', 'right_angle_elbow',
    'left_arm_distance', 'right_arm_distance'
]

def get_feature_columns():
    cols = []
    for lm in default_landmarks:
        cols.extend([f"{lm}_x", f"{lm}_y", f"{lm}_z", f"{lm}_visibility"])
    cols.extend(engineered_features)
    return cols

# ==========================================
# 2. פונקציות עזר מתוך האפליקציה
# ==========================================
def load_data(folder_name):
    path = os.path.join(folder_name, "*.csv")
    files = glob.glob(path)
    if not files: 
        return None
    
    dfs = [pd.read_csv(f).assign(source_file=os.path.basename(f)) for f in files]
    full_data = pd.concat(dfs, ignore_index=True)
    full_data = full_data[full_data['is_valid_frame'] == True].dropna(subset=['pushup_phase', 'hips_position'])
    return full_data

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
# 3. הפעלת החיפוש, ההערכה ושמירת הקובץ
# ==========================================
def optimize_and_evaluate(target_col, X_train, y_train, X_test, y_test, test_frames):
    print(f"\n🔍 Optimizing Model for: {target_col.upper()}...")
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('kpca', KernelPCA()),
        ('knn', KNeighborsClassifier())
    ])

    param_grid = {
        'kpca__n_components': [6, 10, 15],
        'kpca__kernel': ['linear', 'rbf'],
        'knn__n_neighbors': [3, 5, 7],
        'knn__weights': ['uniform', 'distance']
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=1, verbose=3)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    print("\n🏆 Best Parameters Found:")
    for param, value in grid_search.best_params_.items():
        print(f"   - {param}: {value}")
    
    print(f"\n📊 Evaluating on Test Data...")
    predictions = best_model.predict(X_test)
    
    good, bad, streak, bad_frames_list = get_prediction_stats(y_test.values, predictions, test_frames.values)
    accuracy = (good / (good + bad)) * 100
    
    print(f"   🎯 Accuracy: {accuracy:.2f}%")
    print(f"   ✅ Correct Frames: {good}")
    print(f"   ❌ Incorrect Frames: {bad}")
    print(f"   ⚠️ Max Bad Streak: {streak}")

    # ==========================================
    # --- התוספת החדשה: שמירת התוצאות לקובץ TXT ---
    # ==========================================
    # יצירת שם קובץ תקין דינמי לפי שם המודל
    file_safe_name = f"results_{target_col.lower().replace(' ', '_')}.txt"
    
    with open(file_safe_name, "w", encoding="utf-8") as f:
        f.write(f"==================================================\n")
        f.write(f"📊 OPTIMIZATION RESULTS FOR: {target_col.upper()}\n")
        f.write(f"==================================================\n\n")
        
        f.write("🏆 BEST PARAMETERS FOUND:\n")
        for param, value in grid_search.best_params_.items():
            f.write(f" - {param}: {value}\n")
            
        f.write("\n📈 TEST SET PERFORMANCE STATISTICS:\n")
        f.write(f" - Final Accuracy: {accuracy:.2f}%\n")
        f.write(f" - Total Valid Test Frames: {good + bad}\n")
        f.write(f" - Correct Predictions: {good}\n")
        f.write(f" - Incorrect Predictions: {bad}\n")
        f.write(f" - Maximum Consecutive Error Streak: {streak}\n\n")
        
        if bad > 0:
            f.write("📋 INCORRECT FRAME INDICES:\n")
            f.write(f" {str(bad_frames_list)}\n")
        else:
            f.write("🎉 Perfect Prediction - No errors encountered on the test set!\n")
            
    print(f"💾 Results successfully saved to: {file_safe_name}")
    # ==========================================

    return best_model

if __name__ == "__main__":
    print("🚀 Starting Optimized Hyperparameter Optimization Script...")
    
    feature_cols = get_feature_columns()
    
    train_df = load_data("data")
    test_df = load_data("test data")
    
    if train_df is None or test_df is None:
        print("❌ Error: Missing 'data' or 'test data' folders/csvs.")
        exit()
        
    print(f"📂 Loaded {len(train_df)} train frames and {len(test_df)} test frames.")
    
    # דילול לשם מהירות
    MAX_SAMPLES = 1500
    if len(train_df) > MAX_SAMPLES:
        print(f"⚡ Subsampling training data from {len(train_df)} to {MAX_SAMPLES} frames for exponential speedup...")
        train_df = train_df.sample(n=MAX_SAMPLES, random_state=42).reset_index(drop=True)
    
    X_train = train_df[feature_cols]
    y_train_phase = train_df['pushup_phase']
    y_train_hips = train_df['hips_position']
    
    X_test = test_df[feature_cols]
    y_test_phase = test_df['pushup_phase']
    y_test_hips = test_df['hips_position']
    test_frames = test_df['frame_index']
    
    print("\n" + "="*50)
    optimize_and_evaluate("Pushup Phase", X_train, y_train_phase, X_test, y_test_phase, test_frames)
    
    print("\n" + "="*50)
    optimize_and_evaluate("Hips Position", X_train, y_train_hips, X_test, y_test_hips, test_frames)