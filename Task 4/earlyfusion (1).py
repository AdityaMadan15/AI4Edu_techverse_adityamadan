import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from physiological import extract_rppg_features

# Load visual features
data = np.load("data/features.npz", allow_pickle=True)
X_visual = data["features"]
labels = data["labels"]
video_ids = data["video_ids"]

# Map binary labels
y = np.array([0 if l in [0, 0.33] else 1 for l in labels])

X_rppg = []

for vid in video_ids:
    # Try different CSV naming patterns
    csv_path = f"data/rppg_signals/{vid}.csv"
    if not os.path.exists(csv_path):
        csv_path = f"data/rppg_signals/{vid}_pos.csv"  # Try POS version
    
    feat = extract_rppg_features(csv_path)
    if feat is None:
        feat = np.zeros(5)  # Use zeros for missing/invalid features
    X_rppg.append(feat)

X_rppg = np.array(X_rppg)

# --- Early Fusion ---
X_fused = np.concatenate([X_visual, X_rppg], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X_fused, y, test_size=0.2, random_state=42
)

clf = LogisticRegression(max_iter=3000)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Early Fusion Accuracy:", accuracy_score(y_test, y_pred))
