"""Configuration for Task 2: Visual Multi-Class Classification"""
import torch

class Config:
    # Paths
    DATA_PATH = "../data/train/"
    MODEL_SAVE_PATH = "./checkpoints/"
    LOGS_PATH = "./logs/"
    
    # Model Architecture
    BACKBONE = "mobilenet_v2"
    HIDDEN_SIZE = 256
    LSTM_LAYERS = 2
    DROPOUT = 0.7
    IMG_SIZE = 224
    NUM_CLASSES = 4  # 4-class classification
    
    # Training Parameters
    BATCH_SIZE = 4
    NUM_EPOCHS = 25
    LEARNING_RATE = 0.0003
    WEIGHT_DECAY = 0.02
    LABEL_SMOOTHING = 0.1
    
    # Data Parameters
    FPS = 10
    MAX_FRAMES = 30
    MIN_FRAMES = 15
    
    # Class weights for imbalanced data
    # Based on dataset: 7 Distracted, 23 Disengaged, 22 Nominally, 22 Highly
    CLASS_WEIGHTS = torch.tensor([3.0, 1.0, 1.05, 1.05])
    
    # Label Mapping (4 classes)
    LABEL_MAP = {
        'distracted': 0,
        'disengaged': 1,
        'nominally_engaged': 2,
        'nominally': 2,
        'highly_engaged': 3,
        'highly': 3
    }
    
    CLASS_NAMES = ['Distracted', 'Disengaged', 'Nominally Engaged', 'Highly Engaged']
    
    # Training Settings
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 0
    VAL_SPLIT = 0.2
    EARLY_STOPPING_PATIENCE = 7
    SEED = 42
