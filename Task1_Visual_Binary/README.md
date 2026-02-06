# Task 1: Visual Binary Classification

## Problem Statement
Classify student engagement into **High Attentiveness (1)** or **Low Attentiveness (0)** using visual cues only.

### Label Mapping
- **Class 0 (Low)**: Distracted (0) + Disengaged (0.33)
- **Class 1 (High)**: Nominally Engaged (0.66) + Highly Engaged (1)

### Qualification Criteria
- **Target Accuracy**: ≥ 70%
- **Primary Metric**: Accuracy
- **Secondary Metric**: F1-Score

---

## Model Architecture

```
Input Video (various FPS)
    ↓
Frame Extraction (10 FPS, max 100 frames)
    ↓
ResNet18 (pretrained, frozen early layers)
    → Extract spatial features per frame
    ↓
Bidirectional LSTM (2 layers, hidden=256)
    → Model temporal dynamics
    ↓
Classification Head (FC → ReLU → FC)
    ↓
Output: Binary Class (0 or 1)
```

**Key Features:**
- **Backbone**: ResNet18 pretrained on ImageNet
- **Temporal Model**: BiLSTM for sequence modeling
- **Input**: 10 FPS frame extraction, resized to 224×224
- **Data Augmentation**: RandomHorizontalFlip, ColorJitter
- **Class Weights**: [1.47, 1.0] to handle imbalance

---

## Dataset Structure

```
data/train/
  ├── distracted/       # Class 0
  ├── disengaged/       # Class 0
  ├── nominally_engaged/ # Class 1
  └── highly_engaged/    # Class 1
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Training

### Quick Start
```bash
python train.py --epochs 30
```

### Custom Training
```bash
python train.py --epochs 50
```

### Monitor Training
```bash
tensorboard --logdir=logs/
```

**Training automatically:**
- Splits data 80/20 (train/validation)
- Applies class weights for imbalance
- Saves best model to `checkpoints/best_model.pth`
- Generates confusion matrix plots
- Uses early stopping (patience=5)

---

## Inference

### Predict on a single video
```bash
# Detailed output (shows confidence, probabilities)
python predict.py --video_path /path/to/video.mp4

# Simple output (only class number: 0 or 1) - FOR JUDGES
python predict.py --video_path /path/to/video.mp4 --simple
```

### Use custom checkpoint
```bash
python predict.py --video_path video.mp4 --checkpoint checkpoints/best_model.pth
```

**Output Example (Verbose Mode):**
```
============================================================
PREDICTION RESULT
============================================================
Predicted Class: 1 (High Attention)
Confidence: 87.34%
Probabilities:
  Low Attention: 12.66%
  High Attention: 87.34%
============================================================
```

---

## Files

- **train.py**: Training script with full pipeline
- **predict.py**: Inference script for new videos
- **model.py**: ResNet18 + BiLSTM model definition
- **data_loader.py**: Video dataset and frame extraction
- **config.py**: All hyperparameters and settings
- **utils/metrics.py**: Evaluation metrics and visualization

---

## Results

### Performance Metrics
| Metric | Value |
|--------|-------|
| Validation Accuracy | TBD% |
| F1-Score | TBD |
| Qualification (≥70%) | TBD |

### Confusion Matrix
See `logs/confusion_matrix_epoch*.png`

---

## Technical Details

**Hyperparameters:**
- Batch Size: 4
- Learning Rate: 0.0001
- Weight Decay: 0.01
- Optimizer: AdamW
- Scheduler: ReduceLROnPlateau
- Max Frames: 100
- FPS: 10
- Image Size: 224×224

**Model Size:**
- Trainable Parameters: ~11M
- Checkpoint Size: ~45 MB

---

## Troubleshooting

**GPU Memory Issues:**
- Reduce batch size: `BATCH_SIZE = 2`
- Reduce max frames: `MAX_FRAMES = 50`

**Low Accuracy:**
- Increase epochs: `--epochs 50`
- Adjust learning rate in `config.py`
- Check data distribution

**Video Loading Errors:**
- Ensure videos are in supported formats (.mp4, .avi, .mov, .webm)
- Check video file permissions
- Verify opencv-python is installed

---

## Citation

```
@hackathon{techverse2026,
  title={Student Engagement Recognition},
  author={Team TechVerse},
  year={2026},
  event={IIT Ropar AI Hackathon - Pre-India AI Summit}
}
```
