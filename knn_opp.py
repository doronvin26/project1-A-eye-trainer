import os
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, confusion_matrix
import warnings

# התעלמות מאזהרות חלוקה באפס במידה ויש קלאסים ריקים בחלק מהקבצים
warnings.filterwarnings('ignore')

# ==========================================
# הגדרות מודולריות - ניתן לשנות בקלות בעתיד
# ==========================================
FEATURE_LIST_FILE = "Conditional probability/hips_position_feature_only.txt"
DATA_DIR = "data"                  
TARGET_COL = "hips_position"         
OUTPUT_DIR = "Conditional probability/knn_optimization_results"
OUTPUT_FILE_NAME = "hips_position.txt"
#hips_position
#pushup_phase

MAX_FEATURES = 10                 
PCA_VARIANCES = [0.6, 0.7, 0.8, 0.9, 0.95, 1.0] 
K_VALUES = [3, 5, 7, 9, 11, 15]

# MAX_FEATURES = 3                 
# PCA_VARIANCES = [0.95, 1.0] 
# K_VALUES = [15]


def extract_top_features(filepath, max_feat):
    
    features = []
    if not os.path.exists(filepath):
        print(f"Error: Feature file '{filepath}' not found.")
        return features
        
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if "|" in line and "Accuracy" in line:
               
                part1 = line.split('|')[0]
                feat_name = part1.split('.')[1].strip()
                features.append(feat_name)
                if len(features) >= max_feat:
                    break
    #print(f"Extracted features: {features}")
    return features

def load_all_data(data_dir, features_to_keep, target_col):
    
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        print(f"Error: No CSV files found in '{data_dir}'.")
        return []

    data_files = []
    cols_to_load = features_to_keep + [target_col]
    
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            # סינון עמודות והסרת שורות חסרות בפיצ'רים הרלוונטיים
            df = df[cols_to_load].dropna()
            if not df.empty:
                data_files.append({"filename": os.path.basename(file), "df": df})
            #print(df.head(5))
        except Exception as e:
            print(f"Skipping {file} due to error: {e}")
            
    return data_files

def run_optimization():
    print(f"Extracting top {MAX_FEATURES} features from {FEATURE_LIST_FILE}...")
    ranked_features = extract_top_features(FEATURE_LIST_FILE, MAX_FEATURES)
    if not ranked_features:
        return

    print("Loading datasets...")
    data_files = load_all_data(DATA_DIR, ranked_features, TARGET_COL)
    if not data_files:
        return
        
    print(f"Found {len(data_files)} valid CSV files for training/testing.\n")

    results = []
    total_combinations = len(ranked_features) * len(PCA_VARIANCES) * len(K_VALUES)
    current_combo = 0

    for num_features in range(1, len(ranked_features) + 1):
        current_features = ranked_features[:num_features]
        
        for pca_var in PCA_VARIANCES:
            
            for k in K_VALUES:
                current_combo += 1
                if current_combo % 10 == 0:
                    print(f"Processing combination {current_combo}/{total_combinations}...")

                mux_y_true = []
                mux_y_pred = []
                
                # מנגנון ולידציה: עוברים קובץ קובץ (Leave-One-Out)
                for test_data in data_files:
                    test_df = test_data["df"]
                    
                    # הרכבת סט האימון מכל שאר הקבצים
                    train_dfs = [d["df"] for d in data_files if d["filename"] != test_data["filename"]]
                    train_df = pd.concat(train_dfs, ignore_index=True)
                    
                    X_train = train_df[current_features].values
                    y_train = train_df[TARGET_COL].values
                    X_test = test_df[current_features].values
                    y_test = test_df[TARGET_COL].values
                    
                    # נרמול הנתונים (Scaler חייב להיות מותאם רק על סמך האימון)
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    # הפעלת PCA אם נדרש (1.0 משמעותו ללא PCA)
                    if pca_var < 1.0:
                        # וידוא שיש מספיק מימדים ל-PCA
                        n_components = min(X_train_scaled.shape[0], X_train_scaled.shape[1])
                        pca = PCA(n_components=pca_var)
                        try:
                            X_train_final = pca.fit_transform(X_train_scaled)
                            X_test_final = pca.transform(X_test_scaled)
                        except ValueError:
                            # אם הורדת המימדים נכשלה בשל חוסר התאמה מתמטית, נמשיך עם הנתונים המקוריים
                            X_train_final = X_train_scaled
                            X_test_final = X_test_scaled
                    else:
                        X_train_final = X_train_scaled
                        X_test_final = X_test_scaled
                        
                    # אימון וחיזוי KNN
                    knn = KNeighborsClassifier(n_neighbors=k)
                    knn.fit(X_train_final, y_train)
                    y_pred = knn.predict(X_test_final)
                    
                    # הוספת התוצאות מהקובץ הנוכחי לאוסף הגלובלי (mux)
                    mux_y_true.extend(y_test)
                    mux_y_pred.extend(y_pred)

                # חישוב מדדים על סמך כלל הקבצים יחד
                macro_f1 = f1_score(mux_y_true, mux_y_pred, average='macro')
                #print(knn.classes_)
                cm = confusion_matrix(mux_y_true, mux_y_pred)
                classes = np.unique(mux_y_true)
                
                results.append({
                    "num_features": num_features,
                    "feature_list": current_features,
                    "pca_variance": pca_var,
                    "k_value": k,
                    "f1_score": macro_f1,
                    "confusion_matrix": cm,
                    "classes": classes
                })

    print("Optimization finished. Sorting and saving results...")
    
    # מיון התוצאות מהציון הגבוה ביותר לנמוך ביותר
    results.sort(key=lambda x: x["f1_score"], reverse=True)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE_NAME)    
    
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"KNN Optimization Results for Target: {TARGET_COL}\n")
        f.write("="*60 + "\n\n")
        
        for idx, res in enumerate(results):
            f.write(f"Rank: {idx + 1}\n")
            f.write(f"Macro F1-Score: {res['f1_score']:.4f}\n")
            f.write(f"Number of Features: {res['num_features']}\n")
            f.write(f"K Value (KNN): {res['k_value']}\n")
            f.write(f"PCA Variance: {'No PCA' if res['pca_variance'] == 1.0 else res['pca_variance']}\n")
            f.write(f"Features Used: {', '.join(res['feature_list'])}\n")
            f.write(f"Classes Order: {res['classes']}\n")
            f.write("Global Confusion Matrix:\n")
            
            # כתיבת מטריצת הבלבול בצורה נקייה
            for row in res['confusion_matrix']:
                f.write(f"  {row}\n")
            f.write("-" * 60 + "\n")
            
    print(f"Done! Results successfully saved to {OUTPUT_FILE_NAME}.")

if __name__ == "__main__":
    run_optimization()