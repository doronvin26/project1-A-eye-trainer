import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.metrics import accuracy_score

# ==========================================
# 1. טעינת נתונים
# ==========================================
def load_training_data(data_folder):
    """קורא את כל קבצי ה-CSV לאימון, מוסיף עמודת מקור ומאחד ל-DataFrame אחד"""
    path = os.path.join(data_folder, "*.csv")
    files = glob.glob(path)
    
    if not files:
        raise FileNotFoundError(f"No CSV files found in '{data_folder}' folder.")
    
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df['source_file'] = os.path.basename(f)
        dfs.append(df)
        
    full_data = pd.concat(dfs, ignore_index=True)
    
    # סינון שורות לא רלוונטיות או חסרות תיוג
    full_data = full_data[full_data['is_valid_frame'] == True]
    full_data = full_data.dropna(subset=['pushup_phase', 'hips_position'])
    
    return full_data

def load_test_data(file_path):
    """קורא את קובץ ה-CSV המיועד לבדיקה (Test)"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Test file '{file_path}' not found.")
        
    df = pd.read_csv(file_path)
    df['source_file'] = os.path.basename(file_path)
    
    df = df[df['is_valid_frame'] == True]
    df = df.dropna(subset=['pushup_phase', 'hips_position'])
    
    return df

# ==========================================
# 2. שליפת פיצ'רים (ללא חישוב מחדש)
# ==========================================
def extract_features(df, feature_cols):
    """שולף מה-DataFrame רק את עמודות הפיצ'רים המבוקשות ואת התיוגים"""
    
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"The following required features are missing from the CSV: {missing_cols}")
        
    X = df[feature_cols]
    y_phase = df['pushup_phase']
    y_hips = df['hips_position']
    source_files = df['source_file']
    
    return X, y_phase, y_hips, source_files

# ==========================================
# 3. הדפסת תוצאות
# ==========================================
def print_evaluation_results(model_name, target_name, y_true, y_pred, source_files):
    """מחשב ומדפיס אחוזי שגיאה: כללי ולפי קובץ מקור (סרטון)"""
    print(f"\n[{model_name}] | Target: {target_name.upper()}")
    print("-" * 50)
    
    overall_error = 1.0 - accuracy_score(y_true, y_pred)
    print(f"Overall Error Rate: {overall_error:.2%}")
    
    results_df = pd.DataFrame({
        'Actual': np.array(y_true),
        'Predicted': np.array(y_pred),
        'Source_File': np.array(source_files)
    })
    
    results_df['Is_Error'] = results_df['Actual'] != results_df['Predicted']
    error_rates_per_file = results_df.groupby('Source_File')['Is_Error'].mean()
    
    print("Error Rate per Video (CSV):")
    for file, rate in error_rates_per_file.items():
        print(f"  - {file}: {rate:.2%}")
    print("=" * 50)

# ==========================================
# 4. מודלים
# ==========================================
def evaluate_knn(X_train, X_test, y_train, y_test, source_files_test, target_name, n_neighbors=5):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print_evaluation_results(f"Standard KNN (k={n_neighbors})", target_name, y_test, y_pred, source_files_test)

def evaluate_kdtree(X_train, X_test, y_train, y_test, source_files_test, target_name, n_neighbors=5, leaf_size=30):
    kdtree_model = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm='kd_tree', leaf_size=leaf_size)
    kdtree_model.fit(X_train, y_train)
    y_pred = kdtree_model.predict(X_test)
    print_evaluation_results(f"KD-Tree KNN (k={n_neighbors}, leaf={leaf_size})", target_name, y_test, y_pred, source_files_test)

def evaluate_radius_nn(X_train, X_test, y_train, y_test, source_files_test, target_name, radius=2.0):
    rnn = RadiusNeighborsClassifier(radius=radius, outlier_label='most_frequent')
    rnn.fit(X_train, y_train)
    y_pred = rnn.predict(X_test)
    print_evaluation_results(f"Radius NN (radius={radius})", target_name, y_test, y_pred, source_files_test)

# ==========================================
# הריצה המרכזית
# ==========================================
if __name__ == "__main__":
    
    # ==========================================
    # הגדרת הנתיבים (כאן אתה משנה את הנתיב בקוד!)
    # ==========================================
    TRAIN_DATA_DIR = "data"          
    TEST_CSV_FILE = "test data/test_alon_partial_labels.csv"    
    
    # 1. הגדרת רשימת הפיצ'רים המפורשת
    SELECTED_FEATURES = [
        # "left_shoulder_x", "left_shoulder_y", "left_shoulder_z", "left_shoulder_visibility",
        # "right_shoulder_x", "right_shoulder_y", "right_shoulder_z", "right_shoulder_visibility",
        # "left_elbow_x", "left_elbow_y", "left_elbow_z", "left_elbow_visibility",
        # "right_elbow_x", "right_elbow_y", "right_elbow_z", "right_elbow_visibility",
        # "left_wrist_x", "left_wrist_y", "left_wrist_z", "left_wrist_visibility",
        # "right_wrist_x", "right_wrist_y", "right_wrist_z", "right_wrist_visibility",
        # "left_hip_x", "left_hip_y", "left_hip_z", "left_hip_visibility",
        # "right_hip_x", "right_hip_y", "right_hip_z", "right_hip_visibility",
        "left_body_angle", "right_body_angle","right_angle_elbow", "left_angle_elbow",
        # "left_knee_x", "left_knee_y", "left_knee_z", "left_knee_visibility",
        # "right_knee_x", "right_knee_y", "right_knee_z", "right_knee_visibility",
        # "left_ankle_x", "left_ankle_y", "left_ankle_z", "left_ankle_visibility",
        # "right_ankle_x", "right_ankle_y", "right_ankle_z", "right_ankle_visibility"
    ]
    
    print(f"1. Loading Training Data (from folder: '{TRAIN_DATA_DIR}')...")
    train_df = load_training_data(TRAIN_DATA_DIR)
    print(f"Total valid frames for training: {len(train_df)}")
    
    print(f"2. Loading Test Data (from file: '{TEST_CSV_FILE}')...")
    test_df = load_test_data(TEST_CSV_FILE)
    print(f"Total valid frames for testing: {len(test_df)}")
    
    print("\n3. Extracting Features...")
    X_train_raw, y_train_phase, y_train_hips, src_train = extract_features(train_df, SELECTED_FEATURES)
    X_test_raw, y_test_phase, y_test_hips, src_test = extract_features(test_df, SELECTED_FEATURES)
    
    print("4. Scaling Features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    print("\n" + "#"*40)
    print("STARTING ALGORITHM EVALUATIONS")
    print("#"*40)

    # הרצת המודלים עבור שלב השכיבה
    evaluate_knn(X_train_scaled, X_test_scaled, y_train_phase, y_test_phase, src_test, target_name="Pushup Phase", n_neighbors=6)
   # evaluate_kdtree(X_train_scaled, X_test_scaled, y_train_phase, y_test_phase, src_test, target_name="Pushup Phase", n_neighbors=3, leaf_size=30)
    evaluate_radius_nn(X_train_scaled, X_test_scaled, y_train_phase, y_test_phase, src_test, target_name="Pushup Phase", radius=15)

    # הרצת המודלים עבור מנח הירכיים
    evaluate_knn(X_train_scaled, X_test_scaled, y_train_hips, y_test_hips, src_test, target_name="Hips Position", n_neighbors=6)
    #evaluate_kdtree(X_train_scaled, X_test_scaled, y_train_hips, y_test_hips, src_test, target_name="Hips Position", n_neighbors=3, leaf_size=30)
    evaluate_radius_nn(X_train_scaled, X_test_scaled, y_train_hips, y_test_hips, src_test, target_name="Hips Position", radius=0.5)
