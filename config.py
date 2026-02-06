"""Configuration file for Phase A training"""

import torch

class Config:
    # Paths
    DATA_PATH = "data/"
    TRAIN_PATH = "data/train/"
    TEST_PATH = "data/test/"
    MODEL_SAVE_PATH = "models/"
    LOGS_PATH = "logs/"
    
    # Model Parameters
    BACKBONE = "resnet18"  # Options: resnet18, resnet50, efficientnet_b0 (using resnet18 - lighter, faster, DISK SPACE CONSTRAINED)
    IMG_SIZE = 224
    PRETRAINED = True
    USE_LSTM = True  # Use LSTM for temporal modeling (as per problem statement)
    
    # Training Parameters
    BATCH_SIZE = 8  # Reduced for memory efficiency
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    FPS = 10  # Increased from 5 to capture gaze movements
    MAX_FRAMES = 150  # Increased for 10 FPS
    
    # Task 1: Binary Classification
    TASK1_CLASSES = 2
    TASK1_CLASS_NAMES = ['Low Attention', 'High Attention']
    
    # Task 2: Multi-Class Classification  
    TASK2_CLASSES = 4
    TASK2_CLASS_NAMES = ['Distracted', 'Disengaged', 'Nominally Engaged', 'Highly Engaged']
    
    # Data Augmentation
    USE_AUGMENTATION = True
    HORIZONTAL_FLIP = 0.5
    ROTATION = 10
    
    # Training Settings
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 4
    EARLY_STOPPING_PATIENCE = 5
    
    # Validation Split
    VAL_SPLIT = 0.2
    
    # Label Mapping for Binary Task (Task 1)
    # Merge: 0, 0.33 -> 0 (Low Attention)
    # Merge: 0.66, 1 -> 1 (High Attention)
    BINARY_LABEL_MAP = {
        0: 0,      # Distracted -> Low
        0.33: 0,   # Disengaged -> Low
        0.66: 1,   # Nominally Engaged -> High
        1: 1       # Highly Engaged -> High
    }
    
    # Label Mapping for Multi-Class Task (Task 2)
    MULTICLASS_LABEL_MAP = {
        0: 0,      # Distracted
        0.33: 1,   # Disengaged
        0.66: 2,   # Nominally Engaged
        1: 3       # Highly Engaged
    }
