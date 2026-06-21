import pandas as pd
import numpy as np
import glob
import os

def calculate_angle(row, p1, p2, p3):
    """
    מחשב את הזווית הפנימית (0-180) בין 3 נקודות: כתף, אגן (קודקוד), עקב.
    חישוב ב-2D (רק X ו-Y) נותן תוצאה הרבה יותר יציבה מחישוב בתלת-ממד.
    """
    try:
        x1, y1 = float(row[f'{p1}_x']), float(row[f'{p1}_y']) 
        x2, y2 = float(row[f'{p2}_x']), float(row[f'{p2}_y']) 
        x3, y3 = float(row[f'{p3}_x']), float(row[f'{p3}_y']) 
        
        if pd.isna(x1) or pd.isna(x2) or pd.isna(x3):
            return np.nan
            
        radians = np.arctan2(y3 - y2, x3 - x2) - np.arctan2(y1 - y2, x1 - x2)
        angle = np.abs(np.degrees(radians))
        
        return angle
    except Exception:
        return np.nan

def calculate_distance(row, p1, p2):
    """
    מחשב את המרחק האוקלידי הדו-ממדי בין שתי נקודות.
    """
    try:
        x1, y1 = float(row[f'{p1}_x']), float(row[f'{p1}_y'])
        x2, y2 = float(row[f'{p2}_x']), float(row[f'{p2}_y'])
        
        if pd.isna(x1) or pd.isna(x2):
            return np.nan
            
        # חישוב משפט פיתגורס (מרחק בין שתי נקודות)
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return distance
    except Exception:
        return np.nan

def process_all_csvs(target_folders):
    total_processed = 0

    for folder in target_folders:
        print(f"\n🔍 סורק את התיקייה: '{folder}'")
        
        path = os.path.join(folder, "*.csv")
        csv_files = glob.glob(path)
        
        if not csv_files:
            print(f"❌ No CSV files found inside '{folder}' folder.")
            continue

        print(f"📂 Found {len(csv_files)} CSV files in '{folder}'. Starting to process...")

        for file in csv_files:
            df = pd.read_csv(file)
            
            # מוודא שהעמודות הבסיסיות אכן קיימות בקובץ
            if 'left_shoulder_x' in df.columns and 'left_hip_x' in df.columns and 'left_heel_x' in df.columns:
                
                # --- חישובי זוויות ---
                df['left_body_angle'] = df.apply(
                    lambda row: calculate_angle(row, 'left_shoulder', 'left_hip', 'left_heel') , axis=1
                )
                df['right_body_angle'] = df.apply(
                    lambda row: calculate_angle(row, 'right_shoulder', 'right_hip', 'right_heel'), axis=1
                )
                df['left_angle_elbow'] = df.apply(
                    lambda row: calculate_angle(row, 'left_shoulder', 'left_elbow', 'left_wrist'), axis=1
                )
                df['right_angle_elbow'] = df.apply(
                    lambda row: calculate_angle(row, 'right_shoulder', 'right_elbow', 'right_wrist'), axis=1
                )
                
                # --- התוספת החדשה: חישובי מרחקים ---
                # מרחק יד שמאל
                df['left_arm_distance'] = df.apply(
                    lambda row: calculate_distance(row, 'left_shoulder', 'left_wrist'), axis=1
                )
                # מרחק יד ימין
                df['right_arm_distance'] = df.apply(
                    lambda row: calculate_distance(row, 'right_shoulder', 'right_wrist'), axis=1
                )
                
                # שמירת הקובץ מחדש
                df.to_csv(file, index=False)
                print(f"✅ Processed and updated: {file}")
                total_processed += 1
            else:
                print(f"⚠️ Skipped {file} - missing required coordinate columns.")

    print(f"\n🎉 All done! Successfully updated {total_processed} files in their original locations.")

if __name__ == "__main__":
    folders_to_process = ["data", "test data"]
    process_all_csvs(folders_to_process)