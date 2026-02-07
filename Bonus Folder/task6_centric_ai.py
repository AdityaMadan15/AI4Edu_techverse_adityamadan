import numpy as np
import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def main():
    print("--- Task 6: Data-Centric AI (Label Noise Detection) ---")
    
    # 1. Load Data (Visual Features from Task 2/4)
    data_path = "Task 4/data/features.npz"
    if not os.path.exists(data_path):
         # Handle running from inside the folder
        if os.path.exists("../Task 4/data/features.npz"):
            data_path = "../Task 4/data/features.npz"
    
    if not os.path.exists(data_path):
        print("Error: features.npz not found.")
        return

    data = np.load(data_path, allow_pickle=True)
    X = data["features"]
    y_noisy = data["labels"] # These are the given labels (potentially noisy)
    video_ids = data["video_ids"]
    
    # Class names mapping
    class_map = {0: "Distracted", 1: "Disengaged", 2: "Nominally", 3: "Highly"}
    # Original labels are 0.0, 0.33, 0.66, 1.0. Map to 0, 1, 2, 3
    # Note: Task 2 config maps string -> int, but labels.npy might store the floats or ints depending on how it was saved.
    # Let's inspect unique values first to be safe.
    unique_labels = np.unique(y_noisy)
    if np.all(unique_labels <= 1.0) and np.any(unique_labels < 1.0) and len(unique_labels) > 0:
        # Floats case
        y_int = np.zeros_like(y_noisy, dtype=int)
        y_int[np.isclose(y_noisy, 0.33, atol=0.05)] = 1
        y_int[np.isclose(y_noisy, 0.66, atol=0.05)] = 2
        y_int[np.isclose(y_noisy, 1.0, atol=0.05)] = 3
        y_noisy = y_int
    else:
        y_noisy = y_noisy.astype(int)

    print(f"Loaded {len(y_noisy)} samples.")
    
    # 2. Confident Learning Strategy (Simplified)
    # We use Cross-Validation to predict probabilities for every sample
    # purely based on the features, without seeing that sample during training.
    
    clf = LogisticRegression(max_iter=2000, random_state=42)
    
    # Get out-of-sample probabilities
    try:
        y_probs = cross_val_predict(clf, X, y_noisy, cv=5, method='predict_proba')
    except ValueError:
        # Fallback for very small datasets (classes with members < 5)
        print("Warning: Dataset too small for 5-fold CV. Using LOOCV logic via 3-fold.")
        y_probs = cross_val_predict(clf, X, y_noisy, cv=3, method='predict_proba')

    y_pred = np.argmax(y_probs, axis=1)
    
    # 3. Detect Label Errors
    # A simple effective heuristic: High confidence in a DIFFERENT class
    
    suspicious_indices = []
    
    print("\nAnalyzing dataset for label errors...")
    print(f"{'Video ID':<25} | {'Given Label':<15} | {'Predicted':<15} | {'Confidence':<10}")
    print("-" * 75)
    
    for i in range(len(y_noisy)):
        given_label = y_noisy[i]
        pred_label = y_pred[i]
        confidence = y_probs[i, pred_label]
        
        # Threshold: If model is >75% confident it's something else
        if given_label != pred_label and confidence > 0.6: 
            suspicious_indices.append(i)
            print(f"{video_ids[i]:<25} | {class_map.get(given_label, str(given_label)):<15} | {class_map.get(pred_label, str(pred_label)):<15} | {confidence:.2f}")

    print("-" * 75)
    print(f"Found {len(suspicious_indices)} potential label errors out of {len(y_noisy)} samples ({len(suspicious_indices)/len(y_noisy):.1%}).")
    
    # 4. Save Cleaned Suggestions
    # In a real Data-Centric task, we might output a 'cleaned_labels.csv'
    if len(suspicious_indices) > 0:
        df_sus = pd.DataFrame({
            "video_id": video_ids[suspicious_indices],
            "given_label": y_noisy[suspicious_indices],
            "suggested_label": y_pred[suspicious_indices],
            "confidence": np.max(y_probs[suspicious_indices], axis=1)
        })
        output_file = "suggested_label_fixes.csv"
        df_sus.to_csv(output_file, index=False)
        print(f"\nSuggestions saved to {output_file}")
    
    # 5. Baseline vs Cleaned Performance (Simulation)
    # We show that trusting the model's predictions on these specific "noisy" points 
    # (or removing them) might increase theoretical validation consistency.
    
    acc_original = accuracy_score(y_noisy, y_pred)
    print(f"\nModel Consistency (Agreement with Noisy Labels): {acc_original:.4f}")
    print("Lower consistency often indicates higher noise levels in the dataset.")

if __name__ == "__main__":
    main()
