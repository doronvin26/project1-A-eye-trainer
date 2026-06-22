import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# ⚙️ פרמטרים להגדרה
# ==========================================
CSV_FILE_PATH = "test data/test_alon_partial_labels.csv" 

X_COLUMN = "left_arm_index_shoulder"   
Y_COLUMN = "hips_position"       
#HUE_COLUMN = "hips_position"     
#hips_position
#pushup_phase
# ==========================================
# 📊 פונקציית יצירת הגרף
# ==========================================


def create_scatter_plot(csv_path, x_col, y_col, hue_col=None):
    print(f"Loading data from: {csv_path}...")
    
    if not os.path.exists(csv_path):
        print(f"❌ Error: File '{csv_path}' not found!")
        return

    # טעינת הנתונים
    df = pd.read_csv(csv_path)

    missing_cols = [col for col in [x_col, y_col] if col not in df.columns]
    if missing_cols:
        print(f"❌ Error: The following columns are missing in the CSV: {missing_cols}")
        return
        
    if hue_col and hue_col not in df.columns:
        print(f"⚠️ Warning: Hue column '{hue_col}' not found. Plotting without colors.")
        hue_col = None

    print(f"Drawing scatter plot: {x_col} vs {y_col}...")

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    sns.scatterplot(
        data=df, 
        x=x_col, 
        y=y_col, 
        hue=hue_col, 
        palette="Set1", 
        alpha=0.7       
    )

    plt.title(f"Scatter Plot: {x_col} vs {y_col}", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    
    if hue_col:
        plt.legend(title=hue_col, loc='best')

    plt.tight_layout()

    # ==========================================
    # 💾 שמירת הגרף לתוך תיקיית pics
    # ==========================================
    # יצירת התיקייה אם היא לא קיימת
    os.makedirs("pics", exist_ok=True)
    
    # הגדרת נתיב השמירה המלא
    target_folder = os.path.join("pics", y_col)
    os.makedirs(target_folder, exist_ok=True)
    output_filename = os.path.join(target_folder, f"{x_col}_vs_{y_col}.png")
    
    # שמירת הקובץ
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Graph successfully saved to: {output_filename}")

# ==========================================
# 🚀 הפעלה
# ==========================================
if __name__ == "__main__":
    create_scatter_plot(CSV_FILE_PATH, X_COLUMN, Y_COLUMN, Y_COLUMN)