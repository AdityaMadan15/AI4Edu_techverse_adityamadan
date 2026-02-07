"""Configuration for Task 1: Visual Binary Classification"""
import torch

class Config:
    # Paths
    DATA_PATH = "../data/train/"
    MODEL_SAVE_PATH = "./"
    LOGS_PATH = "./logs/"
    
    # Model Architecture
    BACKBONE = "mobilenet_v2"
    HIDDEN_SIZE = 256
    LSTM_LAYERS = 2
    DROPOUT = 0.6
    IMG_SIZE = 224
    NUM_CLASSES = 2
    
    # Training Parameters
    BATCH_SIZE = 4
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.0005
    WEIGHT_DECAY = 0.01
    
    # Data Parameters
    FPS = 5
    MAX_FRAMES = 20
    MIN_FRAMES = 10
    
    # Class weights for imbalanced data
    CLASS_WEIGHTS = torch.tensor([1.47, 1.0])
    
    # Label Mapping
    LABEL_MAP = {
        'distracted': 0,
        'disengaged': 0,
        'nominally_engaged': 1,
        'nominally': 1,
        'highly_engaged': 1,
        'highly': 1
    }
    
    CLASS_NAMES = ['Low Attention', 'High Attention']
    
    # Training Settings
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 0
    VAL_SPLIT = 0.15
    EARLY_STOPPING_PATIENCE = 8
    SEED = 42
