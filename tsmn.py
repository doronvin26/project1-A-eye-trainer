import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA

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
    'nose' 
]

engineered_features = [
    'left_body_angle', 'right_body_angle', 
    'left_angle_elbow', 'right_angle_elbow',
    #'left_arm_distance', 'right_arm_distance'
]

def get_feature_columns():
    cols = []
    for lm in default_landmarks:
        cols.extend([f"{lm}_x", f"{lm}_y", f"{lm}_z", f"{lm}_visibility"])
    cols.extend(engineered_features)
    return cols

# ==========================================
# 2. טעינת נתונים
# ==========================================
def load_data(folder_name):
    """טוען ומנקה נתונים מתיקייה ספציפית"""
    path = os.path.join(folder_name, "*.csv")
    files = glob.glob(path)
    if not files: 
        return None
    
    dfs = [pd.read_csv(f).assign(source_file=os.path.basename(f)) for f in files]
    full_data = pd.concat(dfs, ignore_index=True)
    full_data = full_data[full_data['is_valid_frame'] == True].dropna(subset=['pushup_phase', 'hips_position'])
    return full_data

# ==========================================
# 3. פונקציית צילום הרנטגן (UMAP Plotter)
# ==========================================
def generate_umap_visualization(data_features, labels, title, filename):
    """
    מקבלת מטריצת פיצ'רים (גולמית או מכווצת), מריצה UMAP, ושומרת תמונה של הגרף.
    """
    print(f"⏳ Generating UMAP for: {title}...")
    
    # הפעלת מנוע ה-UMAP לדחיסה ל-2 ממדים לצורך ציור
    reducer = umap.UMAP(n_components=2, random_state=42, min_dist=0.3, n_neighbors=15)
    embedding = reducer.fit_transform(data_features)
    
    # הגדרת סגנון הציור
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid")
    
    # ציור נקודות הנתונים, צבועות לפי שלב השכיבה
    scatter = sns.scatterplot(
        x=embedding[:, 0], 
        y=embedding[:, 1], 
        hue=labels, 
        palette="Set2", 
        s=30,          # גודל הנקודות
        alpha=0.7,     # שקיפות כדי לראות צפיפות
        edgecolor=None
    )
    
    # עיצוב הגרף
    plt.title(title, fontsize=18, fontweight='bold', pad=15)
    plt.xlabel('UMAP Dimension 1', fontsize=12)
    plt.ylabel('UMAP Dimension 2', fontsize=12)
    plt.legend(title='Pushup Phase', title_fontsize='13', fontsize='11', loc='best')
    
    # שמירת התמונה
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    
    print(f"✅ Saved visualization to: {filename}")

# ==========================================
# 4. המנוע הראשי
# ==========================================
if __name__ == "__main__":
    print("🚀 Starting UMAP Visualization Script...\n")
    
    feature_cols = get_feature_columns()
    train_df = load_data("data")
    
    if train_df is None:
        print("❌ Error: Missing 'data' folder or CSVs.")
        exit()
        
    # דילול הנתונים: UMAP מצייר נקודות, וענן של 6000 נקודות יהיה קשה לקריאה
    # נדגום 2000 נקודות כדי לקבל תמונה ברורה ונקייה
    MAX_SAMPLES = 2000
    if len(train_df) > MAX_SAMPLES:
        print(f"⚡ Subsampling data to {MAX_SAMPLES} points for clearer visualization...")
        train_df = train_df.sample(n=MAX_SAMPLES, random_state=42).reset_index(drop=True)
        
    X_raw = train_df[feature_cols]
    y_labels = train_df['pushup_phase']  # נצבע את הגרף לפי שלב השכיבה
    
    # נרמול חובה לפני כל הורדת ממדים
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    print("-" * 50)
    # ---------------------------------------------------------
    # רנטגן 1: UMAP על הנתונים הגולמיים המנורמלים
    # ---------------------------------------------------------
    generate_umap_visualization(
        data_features=X_scaled, 
        labels=y_labels, 
        title="UMAP Projection: Raw Engineered Features (34 Dimensions)", 
        filename="umap_1_raw_features.png"
    )
    
    print("-" * 50)
    # ---------------------------------------------------------
    # הפעלת KPCA (המודל המנצח מהאופטימיזציה שלך)
    # ---------------------------------------------------------
    print("⚙️ Applying Kernel PCA (rbf kernel, 10 components)...")
    # שים כאן את הפרמטרים שהיו המנצחים אצלך בקובץ ה-TXT!
    kpca = KernelPCA(n_components=6, kernel='linear', fit_inverse_transform=False)
    X_kpca = kpca.fit_transform(X_scaled)
    
    # ---------------------------------------------------------
    # רנטגן 2: UMAP על תוצרי ה-KPCA
    # ---------------------------------------------------------
    generate_umap_visualization(
        data_features=X_kpca, 
        labels=y_labels, 
        title="UMAP Projection: After Kernel PCA (10 Dimensions)", 
        filename="umap_2_after_kpca.png"
    )
    
    print("\n🎉 Done! Open the generated PNG images in your folder to compare the results.")