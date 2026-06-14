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
        # שליפת קואורדינטות X ו-Y של שלוש הנקודות
        x1, y1 = float(row[f'{p1}_x']), float(row[f'{p1}_y']) # כתף
        x2, y2 = float(row[f'{p2}_x']), float(row[f'{p2}_y']) # אגן (קודקוד הזווית)
        x3, y3 = float(row[f'{p3}_x']), float(row[f'{p3}_y']) # עקב
        
        # אם יש נתונים חסרים בפריים הספציפי הזה, נדלג עליו
        if pd.isna(x1) or pd.isna(x2) or pd.isna(x3):
            return np.nan
            
        # חישוב הזווית במעלות באמצעות arctan2
        radians = np.arctan2(y3 - y2, x3 - x2) - np.arctan2(y1 - y2, x1 - x2)
        angle = np.abs(np.degrees(radians))
        
        # אנחנו תמיד רוצים את הזווית הפנימית הקטנה מ-180 מעלות
        if angle > 180.0:
            angle = 360.0 - angle
            
        return angle
    except Exception:
        # במקרה של שגיאת המרה (למשל טקסט במקום מספר), נחזיר ערך ריק
        return np.nan

def process_all_csvs():
        
    path = "data/*.csv"
    csv_files = glob.glob(path)
    
    print(f"Files I found: {csv_files}")
    
    if not csv_files:
        print("❌ No CSV files found inside 'data' folder.")
        return
    path = os.path.join("data", "*.csv")
    csv_files = glob.glob(path)

    print(f"📂 Found {len(csv_files)} CSV files. Starting to process...\n")

    for file in csv_files:
        df = pd.read_csv(file)
        
        # מוודא שהעמודות הבסיסיות אכן קיימות בקובץ לפני שמנסים לחשב
        if 'left_shoulder_x' in df.columns and 'left_hip_x' in df.columns and 'left_heel_x' in df.columns:
            
            # חישוב זווית צד שמאל
            df['left_body_angle'] = df.apply(
                lambda row: calculate_angle(row, 'left_shoulder', 'left_hip', 'left_heel'), axis=1
            )
            
            # חישוב זווית צד ימין
            df['right_body_angle'] = df.apply(
                lambda row: calculate_angle(row, 'right_shoulder', 'right_hip', 'right_heel'), axis=1
            )
            
            # שמירת הקובץ מחדש (דורס את הישן, עכשיו הוא כולל את העמודות החדשות)
            df.to_csv(file, index=False)
            print(f"✅ Processed and updated: {os.path.basename(file)}")
        else:
            print(f"⚠️ Skipped {os.path.basename(file)} - missing required coordinate columns.")

    print("\n🎉 All files updated successfully! The new features are ready.")

if __name__ == "__main__":
    process_all_csvs()