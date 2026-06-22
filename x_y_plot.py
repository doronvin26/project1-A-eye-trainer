import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

# ==========================================
# ⚙️ Configuration Parameters
# ==========================================
DATA_FOLDER = "data" 

TARGET_COLUMNS = ["pushup_phase", "hips_position"]

# 1. Base Landmarks to analyze raw/centered coordinates
BASE_LANDMARKS = [
    'left_shoulder', 'right_shoulder', 
    'left_elbow', 'right_elbow', 
    'left_wrist', 'right_wrist', 
    'left_hip', 'right_hip', 
    'left_knee', 'right_knee', 
    'left_ankle', 'right_ankle', 
    'nose'
]

# suffixes for both raw and centered 3D coordinates
COORD_SUFFIXES = ['_x', '_y', '_z', '_centered_x', '_centered_y', '_centered_z']

# Dynamically build the list of coordinate features (e.g., left_shoulder_x, left_shoulder_centered_z, etc.)
COORDINATE_FEATURES = [f"{lm}{suffix}" for lm in BASE_LANDMARKS for suffix in COORD_SUFFIXES]

# 2. Engineered Features (2D, 3D, and Normalized)
ENGINEERED_FEATURES = [
    # Basic Angles
    'left_body_angle', 'right_body_angle', 
    'left_angle_elbow', 'right_angle_elbow',
    'left_knee_angle', 'right_knee_angle', 'neck_angle',
    
    # Normalized Ratios & Y-Displacements
    'left_arm_ratio', 'right_arm_ratio',
    'left_shoulder_elbow_y_norm', 'right_shoulder_elbow_y_norm',
    'left_hip_deviation_norm', 'right_hip_deviation_norm',
    
    # ALON SECOND TRY: New 3D & Advanced Features
    'body_alignment_angle', 'hip_line_error', 
    'left_arm_compression', 'right_arm_compression', 'avg_arm_compression',
    'elbow_symmetry', 'avg_elbow_angle', 'avg_body_angle'
]

# Combine everything into one massive feature list
FEATURES_TO_PLOT = COORDINATE_FEATURES + ENGINEERED_FEATURES

# ==========================================
# 1. Load and Concatenate All Data
# ==========================================
def load_all_data(folder_path):
    print(f"📥 Scanning and loading all CSV files from directory: '{folder_path}'...")
    
    path = os.path.join(folder_path, "*.csv")
    csv_files = glob.glob(path)
    
    if not csv_files:
        print(f"❌ Error: No CSV files found in '{folder_path}'!")
        return None

    dfs = []
    for file in csv_files:
        df = pd.read_csv(file)
        if 'is_valid_frame' in df.columns:
            df = df[df['is_valid_frame'] == True]
        dfs.append(df)

    full_df = pd.concat(dfs, ignore_index=True)
    print(f"✅ Successfully merged {len(csv_files)} files. Total frames for analysis: {len(full_df)}")
    return full_df

# ==========================================
# 2. 📊 Scatter Plot Generation
# ==========================================
def create_scatter_plot(df, x_col, y_col):
    if x_col not in df.columns or y_col not in df.columns:
        # Silently skip if column doesn't exist to avoid terminal clutter
        return

    plot_df = df.dropna(subset=[x_col, y_col]).copy()
    
    if len(plot_df) == 0:
        return

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    sns.scatterplot(
        data=plot_df, 
        x=x_col, 
        y=y_col, 
        hue=y_col,      
        palette="Set1", 
        alpha=0.3       
    )

    plt.title(f"Separation Analysis: {x_col} vs {y_col}", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    plt.legend(title=y_col, loc='best')

    plt.tight_layout()

    # ==========================================
    # Smart Routing to PHASE and HIPS Folders
    # ==========================================
    folder_name = "PHASE" if y_col == "pushup_phase" else "HIPS"
    target_folder = os.path.join("pics", folder_name)
    os.makedirs(target_folder, exist_ok=True)
    
    output_filename = os.path.join(target_folder, f"{x_col}.png")
    
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()

# ==========================================
# 🚀 Main Execution
# ==========================================
if __name__ == "__main__":
    print(f"🚀 Starting feature analysis for {len(FEATURES_TO_PLOT)} distinct features...")
    
    master_df = load_all_data(DATA_FOLDER)
    
    if master_df is not None:
        for target in TARGET_COLUMNS:
            print(f"\n==========================================")
            print(f"📊 Generating separation plots for: {target.upper()}")
            print(f"==========================================")
            
            for feature in FEATURES_TO_PLOT:
                create_scatter_plot(master_df, feature, target)
                
        print("\n🎉 All plots generated successfully! Organized in pics/PHASE and pics/HIPS.")