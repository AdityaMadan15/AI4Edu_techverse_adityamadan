import os

class Config:
    # Paths
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(ROOT_DIR, "../data/train")
    FEATURES_DIR = os.path.join(ROOT_DIR, "features")
    MODELS_DIR = os.path.join(ROOT_DIR, "models")
    
    # Files
    FEATURES_FILE = os.path.join(FEATURES_DIR, "features.npy")
    LABELS_FILE = os.path.join(FEATURES_DIR, "labels.npy")
    CLASSIFIER_PATH = os.path.join(MODELS_DIR, "classifier.joblib")
    
    # Classes
    # Map folder names to integers 0-3
    CLASS_MAP = {
        "distracted": 0,         # 0.0
        "disengaged": 1,         # 0.33
        "nominally_engaged": 2,  # 0.66
        "highly_engaged": 3      # 1.0
    }
    
    # Feature Extractor Settings
    IMG_SIZE = 224
    BATCH_SIZE = 32
    FPS = 0.1  # Sample 0.1 frames per second to save time/compute, sufficient for features
    
    # Classifier Settings
    TEST_SIZE = 0.2
    RANDOM_SEED = 42
