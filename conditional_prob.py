import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
import os
import glob
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# Part 1: Load data and concatenate files
# ==========================================
path = os.path.join("data", "*.csv")
files = glob.glob(path)

if not files: 
    print("No CSV files found in the 'data' directory.")
else:
    # Read all files and concatenate them into a single dataframe
    dfs = [pd.read_csv(f) for f in files]
    full_data = pd.concat(dfs, ignore_index=True)
    
    # Filter valid frames and drop rows with missing target labels
    full_data = full_data[full_data['is_valid_frame'] == True].dropna(subset=['pushup_phase', 'hips_position'])

    # ==========================================
    # Part 2: Prepare features and targets
    # ==========================================
    # Select continuous (numeric) columns and exclude the frame index
    # רשימת הפיצ'רים שנבחרו לחישוב הסתברות מותנית
    selected_features = [
        # 1. 2D Features
        'left_body_angle', 'right_body_angle', 'left_angle_elbow', 'right_angle_elbow',
        'right_wrist_shoulder_hip', 'left_wrist_shoulder_hip',
        'left_arm_distance', 'right_arm_distance', 'left_arm_index_shoulder', 'right_arm_index_shoulder',
        'left_torso_px', 'right_torso_px', 'avg_torso_px',
        'left_arm_ratio', 'right_arm_ratio',
        'left_shoulder_elbow_y_norm', 'right_shoulder_elbow_y_norm',
        'left_hip_deviation_norm', 'right_hip_deviation_norm',
        'left_knee_angle', 'right_knee_angle', 'neck_angle',
        
        # 2. 3D & Biomechanical Features
        'body_alignment_angle', 'hip_line_error', 'left_arm_compression', 
        'right_arm_compression', 'avg_arm_compression',
        'elbow_symmetry', 'avg_elbow_angle', 'avg_body_angle',
        
        # 3. Temporal Features (Deltas)
        'delta_left_elbow_angle', 'delta_right_elbow_angle', 'avg_delta_elbow_angle', 
        'delta_hip_line_error', 'delta_body_alignment_angle'
    ]
    numeric_cols = full_data.select_dtypes(include=[np.number]).columns.tolist()
    #features = [c for c in numeric_cols if c != 'frame_index']
    features = selected_features

    targets = ['pushup_phase', 'hips_position']
    os.makedirs("Conditional probability", exist_ok=True)

    # ==========================================
    # Part 3 & 4: Calculate KDE, classify, and evaluate accuracy
    # ==========================================
    for target in targets:
        print(f"\n--- Analysis for target: {target} ---")
        print("Prior class distribution:")
        print(full_data[target].value_counts(normalize=True))
        results = []
        
        # Calculate the prior probability for each class P(y)
        class_counts = full_data[target].value_counts()
        priors = class_counts / len(full_data)
        classes = class_counts.index.tolist()
        print(f"Classes detected for {target}: {classes}")

        for feature in features:
            kdes = {}
            valid_feature = True
            
            # --- Build the Kernel Density Estimation (KDE) model ---
            for c in classes:
                data_c = full_data[full_data[target] == c][feature].dropna()
                
                # Ensure enough samples and variance to fit KDE
                if len(data_c) < 2 or data_c.var() < 1e-6:
                    valid_feature = False
                    break
                try:
                    # Reshape data to a 2D array as required by scikit-learn
                    X_train = data_c.values.reshape(-1, 1)
                    
                    # Initialize and fit the KDE model
                    kde = KernelDensity(kernel='gaussian', bandwidth='scott')
                    kde.fit(X_train)
                    kdes[c] = kde
                except Exception as e:
                    valid_feature = False
                    break
            
            if not valid_feature:
                continue
                
            # --- Apply Bayes' theorem on all data points ---
            feature_data = full_data[feature].values
            X_test = feature_data.reshape(-1, 1)
            posteriors = []
            
            for c in classes:
                # Get the log-likelihood from the model
                log_likelihood = kdes[c].score_samples(X_test) 
                
                # Convert log-likelihood back to normal likelihood P(x|y)
                likelihood = np.exp(log_likelihood) 
                
                # Calculate the numerator: P(x|y) * P(y)
                posterior = likelihood * priors[c] 
                posteriors.append(posterior)
            
            # Stack results and find the class with the highest probability
            posteriors = np.vstack(posteriors)
            predictions_idx = np.argmax(posteriors, axis=0)
            predictions = [classes[idx] for idx in predictions_idx]
            
            # Calculate the accuracy of the current feature
            accuracy = np.mean(predictions == full_data[target].values)
            results.append((feature, accuracy))
            cm = confusion_matrix(full_data[target].values, predictions, labels=classes)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
            plt.title(f'Confusion Matrix: {feature}')
            plt.ylabel('Actual'); plt.xlabel('Predicted')
            os.makedirs(f"Conditional probability/matrices_{target}", exist_ok=True)
            plt.savefig(f"Conditional probability/matrices_{target}/{feature}.png")
            plt.close()
        
        # ==========================================
        # Part 5: Sort results and save to file
        # ==========================================
        # Sort features by accuracy in descending order
        results.sort(key=lambda x: x[1], reverse=True)
        
        output_path = f"Conditional probability/{target}_feature_only.txt"
        
        with open(output_path, "w") as f:
            f.write(f"Feature ranking for target '{target}' based on KDE Conditional Probability Accuracy:\n")
            f.write("-" * 80 + "\n")
            for rank, (feature, acc) in enumerate(results, 1):
                f.write(f"{rank:3d}. {feature:40s} | Accuracy: {acc:.4f}\n")
        
        print(f"Saved {output_path} successfully.")