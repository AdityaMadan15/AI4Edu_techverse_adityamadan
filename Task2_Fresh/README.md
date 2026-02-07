# Task 2: Visual Multiclass Classification

## Overview
This solution uses a fresh, robust approach for small datasets:
1. **Feature Extraction**: Uses a pre-trained **MobileNetV2** (ImageNet weights) to extract rich visual features from every video frame.
2. **Aggregation**: Averages features across time to create a single 1280-dimensional descriptor per video.
3. **Classification**: Uses a **Support Vector Machine (SVM)** (trained with LOOCV and grid search) to classify the features into 4 categories.

## Performance
- **Training Accuracy**: ~83% (Meets >70% requirement)
- **Model**: `models/classifier.joblib`

## How to Run

### Install Dependencies
```bash
pip install -r requirements.txt
```

### 1. Extract Features (If data changes)
```bash
python extract_features.py
```
This saves features to `features/features.npy`.

### 2. Train Model
```bash
python train_classifier.py
```
This saves the best model to `models/classifier.joblib`.

### 3. Predict
```bash
python predict.py --video_path <path_to_video> --simple
```
**Output:** Integer `0`, `1`, `2`, or `3`.

**Classes:**
- 0: Distracted
- 1: Disengaged
- 2: Nominally Engaged
- 3: Highly Engaged
