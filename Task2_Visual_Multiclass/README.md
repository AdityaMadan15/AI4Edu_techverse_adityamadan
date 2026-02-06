# Task 2: Visual Multi-Class Classification

## Objective
Classify student engagement into 4 granular levels:
- **Class 0**: Distracted
- **Class 1**: Disengaged  
- **Class 2**: Nominally Engaged
- **Class 3**: Highly Engaged

## Target Accuracy
**>70%** (Qualification: 65%)

## Architecture
- **Backbone**: MobileNetV2 (pretrained on ImageNet)
- **Temporal Module**: Temporal Attention Mechanism
- **Classifier**: 2-layer MLP with LayerNorm and Dropout

## Training
```bash
cd Task2_Visual_Multiclass
source ../venv/bin/activate
python train.py
```

## Prediction
```bash
# Single video
python predict.py --video_path /path/to/video.mp4

# Directory of videos
python predict.py --video_path /path/to/videos/ --output predictions.csv
```

## Files
- `config.py`: Configuration and hyperparameters
- `model.py`: MobileNetV2 + Temporal Attention model
- `data_loader.py`: Video dataset loader with augmentation
- `train.py`: Training script
- `predict.py`: Inference script
- `utils/`: Utility functions for metrics and visualization

## Results
Best model will be saved to `checkpoints/best_model.pth`
