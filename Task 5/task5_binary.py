import numpy as np
import os
import sys
import pandas as pd
from scipy.signal import find_peaks, welch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# --- Helper: rPPG Extraction (from Task 4) ---
def extract_rppg_features(csv_path, fs=30):
    if not os.path.exists(csv_path):
        return None
    
    try:
        df = pd.read_csv(csv_path)
        if "rppg_signal" not in df.columns: return None
        signal = df["rppg_signal"].values
    except:
        return None

    peaks, _ = find_peaks(signal, distance=fs//2)
    if len(peaks) < 2: return np.zeros(5)

    rr = np.diff(peaks) / fs
    hr = 60 / rr
    freqs, psd = welch(signal, fs=fs)
    lf_band = (freqs >= 0.04) & (freqs <= 0.15)
    hf_band = (freqs >= 0.15) & (freqs <= 0.4)
    lf_power = np.sum(psd[lf_band])
    hf_power = np.sum(psd[hf_band]) + 1e-6
    
    return np.array([np.mean(hr), np.std(hr), np.std(rr), np.var(signal), lf_power / hf_power])

def main():
    print("--- Task 5: Late Fusion (Binary) ---")
    
    data_path = "Task 4/data/features.npz"
    if not os.path.exists(data_path) and os.path.exists("../Task 4/data/features.npz"):
        data_path = "../Task 4/data/features.npz"
        
    if not os.path.exists(data_path):
        print("Error: features.npz not found.")
        return

    data = np.load(data_path, allow_pickle=True)
    X_viz = data["features"]
    labels = data["labels"]
    video_ids = data["video_ids"]
    
    # Binary Mapping: 0, 1 -> 0 (Not Engaged); 2, 3 -> 1 (Engaged)
    # Note: Task 2 config: 0=distracted, 1=disengaged, 2=nominally, 3=highly
    y = np.array([0 if l <= 1 else 1 for l in labels])
    
    print(f"Loaded {len(y)} samples. Class balance: {np.bincount(y)}")

    # Extract rPPG
    X_phys = []
    rppg_dirs = ["Task 4/data/rppg_signals", "../Task 4/data/rppg_signals", "data/rppg_signals"]
    rppg_dir = next((d for d in rppg_dirs if os.path.exists(d)), None)
    
    for vid in video_ids:
        feat = None
        if rppg_dir:
            paths = [os.path.join(rppg_dir, f"{vid}.csv"), os.path.join(rppg_dir, f"{vid}_pos.csv")]
            for p in paths:
                if os.path.exists(p):
                    feat = extract_rppg_features(p)
                    break
        X_phys.append(feat if feat is not None else np.zeros(5))
    
    X_phys = np.nan_to_num(np.array(X_phys))
    
    # Split
    indices = np.arange(len(y))
    X_train_idx, X_test_idx, y_train, y_test = train_test_split(
        indices, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train Models
    print("\nTraining Visual Model...")
    clf_viz = LogisticRegression(max_iter=1000)
    clf_viz.fit(X_viz[X_train_idx], y_train)
    acc_viz = clf_viz.score(X_viz[X_test_idx], y_test)
    print(f"Visual Acc: {acc_viz:.4f}")
    
    print("Training Physiological Model...")
    clf_phys = LogisticRegression(max_iter=1000)
    clf_phys.fit(X_phys[X_train_idx], y_train)
    acc_phys = clf_phys.score(X_phys[X_test_idx], y_test)
    print(f"Phys Acc: {acc_phys:.4f}")
    
    # Fusion
    p_viz = clf_viz.predict_proba(X_viz[X_test_idx])[:, 1]
    p_phys = clf_phys.predict_proba(X_phys[X_test_idx])[:, 1]
    
    p_final = (0.7 * p_viz) + (0.3 * p_phys)
    y_pred = (p_final > 0.5).astype(int)
    
    print(f"\nLate Fusion Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
