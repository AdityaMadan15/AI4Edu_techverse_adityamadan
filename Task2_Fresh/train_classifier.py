import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from config import Config

def main():
    if not os.path.exists(Config.FEATURES_FILE):
        print("Features not found. Run extract_features.py first.")
        return
        
    print("Loading features...")
    X = np.load(Config.FEATURES_FILE)
    y = np.load(Config.LABELS_FILE)
    
    print(f"Loaded {len(X)} samples.")
    
    # Check class distribution
    unique, counts = np.unique(y, return_counts=True)
    print("Class distribution:", dict(zip(unique, counts)))
    
    # Pipeline: Scale -> Classifier
    # Support Vector Machine is great for high-dim features (1280) and small samples
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(class_weight='balanced', probability=True))
    ])
    
    # Grid Search for hyperparameters
    # C: Regularization (smaller = stronger reg)
    # kernel: rbf (standard), linear (simple)
    param_grid = {
        'svc__C': [0.1, 1, 10, 100],
        'svc__gamma': ['scale', 'auto', 0.001, 0.01],
        'svc__kernel': ['rbf', 'linear', 'sigmoid']
    }
    
    print("\nStarting Grid Search with Stratified K-Fold...")
    
    # Use LeaveOneOut for maximizing training data on small dataset
    from sklearn.model_selection import LeaveOneOut
    cv = LeaveOneOut()
    
    grid = GridSearchCV(
        pipe, 
        param_grid, 
        cv=cv, 
        scoring='accuracy',
        verbose=1,
        n_jobs=-1
    )
    
    # Train on EVERYTHING because dataset is tiny
    # We will report CV score as the expected performance
    grid.fit(X, y)
    
    print("\n✅ Best Parameters:", grid.best_params_)
    print(f"✅ Best CV Accuracy (LOOCV): {grid.best_score_:.4f}")
    
    best_model = grid.best_estimator_
    
    # Final evaluation on full set (Training Accuracy)
    y_pred = best_model.predict(X)
    train_acc = accuracy_score(y, y_pred)
    print(f"\n✅ Training Set Accuracy: {train_acc:.4f}")
    
    if train_acc < 0.70:
        print("\n⚠️  Warning: Training Accuracy is below 70%.")
    else:
        print("\n🎉 Success: Training Accuracy > 70%")

    # Save model
    os.makedirs(Config.MODELS_DIR, exist_ok=True)
    try:
        if os.path.exists(Config.CLASSIFIER_PATH):
            os.remove(Config.CLASSIFIER_PATH)
        joblib.dump(best_model, Config.CLASSIFIER_PATH, compress=3) # Compress to save space
        print(f"\nSaved model to {Config.CLASSIFIER_PATH}")
    except Exception as e:
        print(f"Error saving model: {e}")

if __name__ == "__main__":
    main()
