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
    # "left_shoulder", "right_shoulder",
    # "left_elbow", "right_elbow", 
    # "left_wrist", "right_wrist",
    # "left_hip", "right_hip",
    # "left_ankle", "right_ankle", 
    # "left_heel", "right_heel", 
]

engineered_features = [
    # --- Classic 2D Features ---
    'left_body_angle', 'right_body_angle', 
    'left_angle_elbow', 'right_angle_elbow',
    
    # 'left_knee_angle', 'right_knee_angle', 'neck_angle',
    # 'left_arm_ratio', 'right_arm_ratio',
    # 'left_shoulder_elbow_y_norm', 'right_shoulder_elbow_y_norm',
    # 'left_hip_deviation_norm', 'right_hip_deviation_norm',
    
    # --- Advanced 3D Features ---
    #'body_alignment_angle', 
    'hip_line_error', 
    #'left_arm_compression', 'right_arm_compression', 'avg_arm_compression',
    #'elbow_symmetry',
    'avg_elbow_angle', 
    #'avg_body_angle',
    
    # --- Temporal Motion Features ---
    'avg_delta_elbow_angle', 'delta_hip_line_error', 'delta_body_alignment_angle'
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
# 3. הפעלת החיפוש, ההערכה וכתיבה לדו"ח המלא
# ==========================================
def optimize_and_evaluate(target_col, X_train, y_train, X_test, y_test, test_frames, output_file):
    print(f"⏳ Working on '{target_col.upper()}'... (This may take a few minutes)")
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('kpca', KernelPCA()),
        ('knn', KNeighborsClassifier())
    ])

    param_grid = [
        # מילון 1: רק עבור קרנל לינארי (בלי גאמה!)
        {
            'kpca__n_components': [4, 6, 8],
            'kpca__kernel': ['linear'],
            'knn__n_neighbors': [3, 5, 7, 12, 15, 18, 20],
            'knn__weights': ['uniform', 'distance']
        },
        # מילון 2: רק עבור קרנל RBF (עם גאמה)
        {
            'kpca__n_components': [4, 6, 8],
            'kpca__kernel': ['rbf'],
            'kpca__gamma': [None, 0.01, 0.1, 0.5],
            'knn__n_neighbors': [3, 5, 7, 12, 15, 18, 20],
            'knn__weights': ['uniform', 'distance']
        }
    ]

    # verbose=0 משתיק לחלוטין את הטרמינל בזמן החישובים
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=0)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    # חיזוי על נתוני הטסט עם המודל המנצח
    predictions = best_model.predict(X_test)
    good, bad, streak, bad_frames_list = get_prediction_stats(y_test.values, predictions, test_frames.values)
    accuracy = (good / (good + bad)) * 100

    # שליפת כל המודלים שנבדקו וסידור שלהם מהטוב לגרוע
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df = results_df.sort_values(by='rank_test_score')

    # ==========================================
    # --- כתיבת הדו"ח המלא לקובץ ---
    # ==========================================
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(f"==================================================\n")
        f.write(f"🎯 OPTIMIZATION REPORT FOR: {target_col.upper()}\n")
        f.write(f"==================================================\n\n")
        
        # בלוק 1: המודל המנצח והסטטיסטיקות שלו
        f.write("🌟 THE WINNING MODEL (BEST PARAMETERS):\n")
        for param, value in grid_search.best_params_.items():
            f.write(f" - {param}: {value}\n")
            
        f.write("\n📈 TEST SET PERFORMANCE (WINNER):\n")
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
            
        f.write("\n--------------------------------------------------\n")
        
        # בלוק 2: רשימת כל המודלים שנבדקו במהלך החיפוש
        f.write("🔬 ALL TESTED COMBINATIONS (Ranked by internal accuracy):\n")
        for index, row in results_df.iterrows():
            rank = row['rank_test_score']
            mean_acc = row['mean_test_score'] * 100
            params_dict = row['params']
            
            # הדפסה קריאה של כל שילוב
            f.write(f" [Rank {rank}] Accuracy: {mean_acc:.2f}% | Params: {params_dict}\n")
            
        f.write("\n\n\n") 
            
    print(f"✅ Finished '{target_col.upper()}'. Results appended to file.")
    return best_model

if __name__ == "__main__":
    print("🚀 Starting Silent Hyperparameter Optimization Script...\n")
    
    feature_cols = get_feature_columns()
    
    train_df = load_data("data")
    test_df = load_data("test data")
    
    if train_df is None or test_df is None:
        print("❌ Error: Missing 'data' or 'test data' folders/csvs.")
        exit()
        
    MAX_SAMPLES = 1500
    if len(train_df) > MAX_SAMPLES:
        train_df = train_df.sample(n=MAX_SAMPLES, random_state=42).reset_index(drop=True)
    
    X_train = train_df[feature_cols]
    y_train_phase = train_df['pushup_phase']
    y_train_hips = train_df['hips_position']
    
    X_test = test_df[feature_cols]
    y_test_phase = test_df['pushup_phase']
    y_test_hips = test_df['hips_position']
    test_frames = test_df['frame_index']
    
    # ניקוי ויצירת הקובץ המאוחד מחדש
    shared_output_file = "kpca_optimization_results.txt"
    with open(shared_output_file, "w", encoding="utf-8") as fresh_file:
        fresh_file.write("==================================================\n")
        fresh_file.write("🚀 GLOBAL HYPERPARAMETER OPTIMIZATION REPORT\n")
        fresh_file.write("==================================================\n\n")
    
    optimize_and_evaluate("Pushup Phase", X_train, y_train_phase, X_test, y_test_phase, test_frames, shared_output_file)
    optimize_and_evaluate("Hips Position", X_train, y_train_hips, X_test, y_test_hips, test_frames, shared_output_file)
    
    print(f"\n🎉 All done! Open '{shared_output_file}' to view the full report.")