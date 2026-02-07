MODEL_CONFIG = {
    "cnn_backbone": "resnet18",
    "sequence_length": 45,
    "feature_dim": 515,
    "num_classes": 4,
    "lstm_hidden_size": 192,
    "lstm_num_layers": 2,
    "lstm_dropout": 0.4,
    "classifier_dropout": 0.4,
    "freeze_backbone_layers": 10,
    "distracted_class_index": 0,
}

TRAINING_CONFIG = {
    "learning_rate": 5e-5,
    "batch_size": 4,
    "num_epochs_phase1": 4,
    "num_epochs_phase2": 3,
    "num_epochs_phase3": 3,
    "weight_decay": 0.01,
    "grad_clip_norm": 2.5,
    "early_stopping_patience": 3,
    "num_workers": 0,
    "random_seeds": [1337],
    "class_weights": [12.0, 3.5, 3.7, 3.7],
    "tta_transforms": ["none", "hflip"],
    "synthetic_multiplier_class0": 2,
}

DATA_CONFIG = {
    "train_root": "data/train",
    "raw_root": "data/raw",
    "cache_dir": "Task2_Visual_Multiclass/cache",
    "video_ext": [".mp4", ".avi", ".mov"],
    "frame_size": (224, 224),
    "fps": 10,
}
