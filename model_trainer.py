import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.metrics import accuracy_score

# ==========================================
# 1. טעינת ואיחוד נתונים
# ==========================================
def load_and_merge_data(data_folder="data"):
    """קורא את כל קבצי ה-CSV, מוסיף עמודת מקור ומאחד אותם ל-DataFrame אחד"""
    path = os.path.join(data_folder, "*.csv")
    files = glob.glob(path)
    
    if not files:
        raise FileNotFoundError(f"No CSV files found in '{data_folder}' folder.")
    
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df['source_file'] = os.path.basename(f) # שמירת שם קובץ המקור
        dfs.append(df)
        
    full_data = pd.concat(dfs, ignore_index=True)
    
    # סינון שורות לא רלוונטיות או חסרות תיוג
    full_data = full_data[full_data['is_valid_frame'] == True]
    full_data = full_data.dropna(subset=['pushup_phase', 'hips_position'])
    
    return full_data

# ==========================================
# 2. עיבוד מקדים ויצירת פיצ'רים מחושבים
# ==========================================
def preprocess_data(df, selected_landmarks):
    """מחשב זוויות, בוחר עמודות ומנרמל את הנתונים"""
    
    # חישוב זווית (כתף-מותן-עקב) בצורה וקטורית ומהירה לכל הטבלה בבת אחת
    def calc_angles(p1_y, p1_x, p2_y, p2_x, p3_y, p3_x):
        rad = np.arctan2(p3_y - p2_y, p3_x - p2_x) - np.arctan2(p1_y - p2_y, p1_x - p2_x)
        angle = np.abs(np.degrees(rad))
        return np.where(angle > 180.0, 360.0 - angle, angle)

    df['left_body_angle'] = calc_angles(
        df['left_shoulder_y'], df['left_shoulder_x'],
        df['left_hip_y'], df['left_hip_x'],
        df['left_heel_y'], df['left_heel_x']
    )
    
    df['right_body_angle'] = calc_angles(
        df['right_shoulder_y'], df['right_shoulder_x'],
        df['right_hip_y'], df['right_hip_x'],
        df['right_heel_y'], df['right_heel_x']
    )
    
    # יצירת רשימת הפיצ'רים הסופית
    feature_cols = []
    for lm in selected_landmarks:
        feature_cols.extend([f"{lm}_x", f"{lm}_y", f"{lm}_z", f"{lm}_visibility"])
    feature_cols.extend(['left_body_angle', 'right_body_angle'])
    
    X = df[feature_cols]
    y_phase = df['pushup_phase']
    y_hips = df['hips_position']
    source_files = df['source_file']
    
    # נרמול (Scaling)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y_phase, y_hips, source_files, scaler

# ==========================================
# 5. הדפסת תוצאות ושגיאות
# ==========================================
def print_evaluation_results(model_name, target_name, y_true, y_pred, source_files):
    """מחשב ומדפיס אחוזי שגיאה: כללי ולפי קובץ מקור (סרטון)"""
    print(f"\n[{model_name}] | Target: {target_name.upper()}")
    print("-" * 50)
    
    # חישוב אחוז שגיאה כללי (1 פחות אחוז ההצלחה)
    overall_error = 1.0 - accuracy_score(y_true, y_pred)
    print(f"Overall Error Rate: {overall_error:.2%}")
    
    # יצירת DataFrame לבדיקת שגיאות פר סרטון
    results_df = pd.DataFrame({
        'Actual': np.array(y_true),
        'Predicted': np.array(y_pred),
        'Source_File': np.array(source_files)
    })
    
    # עמודה בוליאנית: האם הייתה שגיאה בחיזוי?
    results_df['Is_Error'] = results_df['Actual'] != results_df['Predicted']
    
    # קיבוץ לפי סרטון וחישוב ממוצע השגיאות
    error_rates_per_file = results_df.groupby('Source_File')['Is_Error'].mean()
    
    print("Error Rate per Video (CSV):")
    for file, rate in error_rates_per_file.items():
        print(f"  - {file}: {rate:.2%}")
    print("=" * 50)

# ==========================================
# 4 & 6. מודלים (עם היפר-פרמטרים מוגדרים מראש)
# ==========================================
def evaluate_knn(X_train, X_test, y_train, y_test, source_files_test, target_name, n_neighbors=5):
    """מודל KNN קלאסי"""
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print_evaluation_results(f"Standard KNN (k={n_neighbors})", target_name, y_test, y_pred, source_files_test)

def evaluate_kdtree(X_train, X_test, y_train, y_test, source_files_test, target_name, n_neighbors=5, leaf_size=30):
    """מודל KNN מאולץ להשתמש בחיפוש KD-Tree"""
    kdtree_model = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm='kd_tree', leaf_size=leaf_size)
    kdtree_model.fit(X_train, y_train)
    y_pred = kdtree_model.predict(X_test)
    print_evaluation_results(f"KD-Tree KNN (k={n_neighbors}, leaf={leaf_size})", target_name, y_test, y_pred, source_files_test)

def evaluate_radius_nn(X_train, X_test, y_train, y_test, source_files_test, target_name, radius=2.0):
    """מודל מבוסס רדיוס (מקביל ל-KRNN) - מחפש שכנים בתוך טווח מוגדר"""
    # outlier_label מטפל במקרים בהם לפריים אין אף שכן בתוך הרדיוס המוגדר
    rnn = RadiusNeighborsClassifier(radius=radius, outlier_label='most_frequent')
    rnn.fit(X_train, y_train)
    y_pred = rnn.predict(X_test)
    print_evaluation_results(f"Radius NN (radius={radius})", target_name, y_test, y_pred, source_files_test)

# ==========================================
# הריצה המרכזית
# ==========================================
if __name__ == "__main__":
    print("1. Loading and merging CSVs...")
    df = load_and_merge_data("data")
    print(f"Total valid frames loaded: {len(df)}")
    
    # הגדרת איברי המטרה
    selected_landmarks = ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip"]
    
    print("2. Preprocessing & Feature Engineering...")
    X, y_phase, y_hips, source_files, scaler = preprocess_data(df, selected_landmarks)
    
    # 3. יצירת Train/Test Split (80/20) - עושים פעמיים, פעם ל-Phase ופעם ל-Hips
    print("3. Splitting Data (80% Train, 20% Test)...")
    X_train_p, X_test_p, y_train_p, y_test_p, src_train_p, src_test_p = train_test_split(
        X, y_phase, source_files, test_size=0.2, random_state=42
    )
    
    X_train_h, X_test_h, y_train_h, y_test_h, src_train_h, src_test_h = train_test_split(
        X, y_hips, source_files, test_size=0.2, random_state=42
    )

    print("\n" + "#"*40)
    print("STARTING ALGORITHM EVALUATIONS")
    print("#"*40)

    # הרצת המודלים עבור שלב השכיבה (Pushup Phase)
    evaluate_knn(X_train_p, X_test_p, y_train_p, y_test_p, src_test_p, target_name="Pushup Phase", n_neighbors=3)
    evaluate_kdtree(X_train_p, X_test_p, y_train_p, y_test_p, src_test_p, target_name="Pushup Phase", n_neighbors=3, leaf_size=30)
    evaluate_radius_nn(X_train_p, X_test_p, y_train_p, y_test_p, src_test_p, target_name="Pushup Phase", radius=2.5)

    # הרצת המודלים עבור מנח הירכיים (Hips Position)
    evaluate_knn(X_train_h, X_test_h, y_train_h, y_test_h, src_test_h, target_name="Hips Position", n_neighbors=3)
    evaluate_kdtree(X_train_h, X_test_h, y_train_h, y_test_h, src_test_h, target_name="Hips Position", n_neighbors=3, leaf_size=30)
    evaluate_radius_nn(X_train_h, X_test_h, y_train_h, y_test_h, src_test_h, target_name="Hips Position", radius=2.5)