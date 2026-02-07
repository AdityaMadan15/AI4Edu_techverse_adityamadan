import numpy as np
import os
import sys
import pandas as pd
from scipy.signal import find_peaks, welch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# --- Helper: rPPG Extraction (from Task 4) ---
def extract_rppg_features(csv_path, fs=30):
    if not os.path.exists(csv_path):
        return None  # Handle missing file
    
    try:
        df = pd.read_csv(csv_path)
        if "rppg_signal" not in df.columns:
            return None
        signal = df["rppg_signal"].values
    except:
        return None

    # HR Proxy
    peaks, _ = find_peaks(signal, distance=fs//2)
    if len(peaks) < 2: return np.zeros(5) # Fallback

    rr = np.diff(peaks) / fs
    hr = 60 / rr
    
    # Frequency Domain
    freqs, psd = welch(signal, fs=fs)
    lf_band = (freqs >= 0.04) & (freqs <= 0.15)
    hf_band = (freqs >= 0.15) & (freqs <= 0.4)
    
    lf_power = np.sum(psd[lf_band])
    hf_power = np.sum(psd[hf_band]) + 1e-6
    
    return np.array([
        np.mean(hr),
        np.std(hr),
        np.std(rr), # HRV (SDNN)
        np.var(signal), # SQI proxy
        lf_power / hf_power
    ])

def main():
    print("--- Task 5: Late Fusion (Multiclass) ---")
    
    # 1. Load Data
    # Assuming the data was packed by Task 4 utility
    data_path = "Task 4/data/features.npz" 
    
    if not os.path.exists(data_path):
        # Fallback to checking local dir if run from inside Task 5
        if os.path.exists("../Task 4/data/features.npz"):
            data_path = "../Task 4/data/features.npz"
        else:
            print(f"Error: Could not find {data_path}. Please run Task 4 pack_features.py first.")
            return

    data = np.load(data_path, allow_pickle=True)
    X_viz = data["features"]
    y = data["labels"]
    video_ids = data["video_ids"]
    
    print(f"Loaded {len(y)} samples.")
    
    # 2. Extract rPPG Features
    print("Extracting rPPG (Physiological) features...")
    X_phys = []
    
    # Try multiple paths for rPPG signals
    rppg_dirs = ["Task 4/data/rppg_signals", "../Task 4/data/rppg_signals", "data/rppg_signals"]
    rppg_dir = next((d for d in rppg_dirs if os.path.exists(d)), None)
    
    if not rppg_dir:
        print("Warning: rPPG signals directory not found. Physiological features will be zeros.")
    
    valid_indices = []
    
    for i, vid in enumerate(video_ids):
        feat = None
        if rppg_dir:
            # Try plain ID or ID_pos
            paths_to_try = [
                os.path.join(rppg_dir, f"{vid}.csv"),
                os.path.join(rppg_dir, f"{vid}_pos.csv")
            ]
            for p in paths_to_try:
                if os.path.exists(p):
                    feat = extract_rppg_features(p)
                    if feat is not None: break
        
        if feat is None:
            # Fallback for missing signals - use mean imputation or zeros
            # ideally we should skip, but to keep X_viz aligned we use zeros
            feat = np.zeros(5) 
        
        X_phys.append(feat)

    X_phys = np.array(X_phys)
    
    # Handle NaNs
    X_phys = np.nan_to_num(X_phys)
    
    # 3. Split Data
    # We split indices to ensure we split X_viz and X_phys essentially the same way
    indices = np.arange(len(y))
    X_train_idx, X_test_idx, y_train, y_test = train_test_split(
        indices, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_viz_train, X_viz_test = X_viz[X_train_idx], X_viz[X_test_idx]
    X_phys_train, X_phys_test = X_phys[X_train_idx], X_phys[X_test_idx]
    
    # 4. Train Models
    print("\nTraining Visual Model (Logistic Regression)...")
    clf_viz = LogisticRegression(max_iter=1000, C=1.0)
    clf_viz.fit(X_viz_train, y_train)
    acc_viz = clf_viz.score(X_viz_test, y_test)
    print(f"Visual Only Accuracy: {acc_viz:.4f}")
    
    print("Training Physiological Model (Random Forest)...")
    # rPPG features are non-linear and noisy, RF is often robust
    clf_phys = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf_phys.fit(X_phys_train, y_train)
    acc_phys = clf_phys.score(X_phys_test, y_test)
    print(f"Physiological Only Accuracy: {acc_phys:.4f}")
    
    # 5. Late Fusion (Average Probabilities)
    print("\nPerforming Late Fusion...")
    
    # Get probabilities
    probs_viz = clf_viz.predict_proba(X_viz_test)
    probs_phys = clf_phys.predict_proba(X_phys_test)
    
    # Weighted Average (Visual is usually stronger, but we'll try equal weight first)
    # If Visual is much stronger, weight it higher: 0.7 * viz + 0.3 * phys
    final_probs = (0.7 * probs_viz) + (0.3 * probs_phys)
    
    y_pred = np.argmax(final_probs, axis=1)
    
    acc_fusion = accuracy_score(y_test, y_pred)
    print(f"Late Fusion Accuracy: {acc_fusion:.4f}")
    
    print("\nClassification Report (Fusion):")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
