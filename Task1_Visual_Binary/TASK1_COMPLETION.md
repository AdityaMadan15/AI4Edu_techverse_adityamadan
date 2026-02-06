# TASK 1: VISUAL BINARY CLASSIFICATION - COMPLETED ✓

## Achievement Summary
- **Target Accuracy**: ≥70%
- **Best Validation Accuracy**: **87.50%** ✓ (Epoch 5)
- **Consistent Performance**: 75%+ accuracy maintained across multiple epochs
- **Status**: **QUALIFICATION THRESHOLD MET**

## Model Architecture
- **Backbone**: MobileNetV2 (pre-trained on ImageNet)
- **Temporal Processing**: Temporal Attention Mechanism
- **Parameters**: 2,256,067 (efficient and lightweight)
- **Regularization**: LayerNorm, Dropout (0.6), Weight Decay (0.01)

## Dataset
- **Total Videos Used**: 57 videos
  - Class 0 (Low Attention): 23 videos
  - Class 1 (High Attention): 34 videos
- **Train/Val Split**: 85%/15%
- **Processing**: 20 frames per video @ 5 FPS

## Training Configuration
- **Batch Size**: 4
- **Learning Rate**: 0.0005 (AdamW optimizer)
- **Epochs**: 50 (with early stopping)
- **Data Augmentation**: Random flips, rotations, color jitter
- **Loss**: Cross-Entropy with class weights (1.47, 1.0) and label smoothing (0.1)

## Key Files
1. **model.py**: MobileNetV2 + Temporal Attention architecture
2. **train.py**: Training loop with metrics tracking
3. **predict.py**: Inference script with --simple flag for automated testing
4. **data_loader.py**: Video preprocessing and data loading
5. **config.py**: Hyperparameters and settings
6. **checkpoints/best_model.pth**: Best model (87.50% accuracy)

## Usage

### Training
```bash
cd Task1_Visual_Binary
source ../venv/bin/activate
python train.py --epochs 50
```

### Prediction (for judges/automated testing)
```bash
python predict.py --video_path <video_file> --simple
# Output: 0 or 1
```

### Prediction (verbose mode)
```bash
python predict.py --video_path <video_file>
# Shows detailed predictions and probabilities
```

## Performance Metrics (Best Epoch - Epoch 5)
- **Validation Accuracy**: 87.50%
- **F1-Score**: 0.9091
- **Precision** (Low Attention): 1.00
- **Recall** (Low Attention): 0.67
- **Precision** (High Attention): 0.83
- **Recall** (High Attention): 1.00

## Technical Fixes Applied
1. Fixed video file extension handling (.MP4 vs .mp4)
2. Removed 5 corrupted videos that couldn't be processed
3. Implemented frame padding to ensure consistent tensor sizes
4. Replaced BatchNorm with LayerNorm to handle varying batch sizes
5. Added drop_last=True to training loader to prevent single-sample batches
6. Enhanced error handling for video loading failures

## Submission Ready
- ✓ Model trained and saved
- ✓ Prediction script working with --simple flag
- ✓ Accuracy requirement met (87.50% >> 70%)
- ✓ Binary classification (0: Low Attention, 1: High Attention)
- ✓ All code documented and organized

## Next Steps
- Task 2: Multi-class Classification (4 classes)
- Task 3: Phase B (rPPG - Remote Photoplethysmography)

---
**Generated**: February 7, 2026
**Status**: ✓ COMPLETED AND QUALIFIED
