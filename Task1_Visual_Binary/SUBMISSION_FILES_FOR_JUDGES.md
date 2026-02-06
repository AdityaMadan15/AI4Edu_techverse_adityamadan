# 📋 TASK 1 SUBMISSION - OUTPUT FILES FOR JUDGES

## ✅ TASK 1 STATUS: COMPLETED
- **Best Validation Accuracy**: **87.50%** ✓ (Requirement: ≥70%)
- **Final Epoch**: 13/50
- **Training Stopped**: Early stopping or user interrupt

---

## 📁 KEY FILES FOR EVALUATION

### 1. 🎯 **Trained Model** (Most Important!)
```
Task1_Visual_Binary/checkpoints/best_model.pth (29 MB)
```
- This is your trained neural network
- Contains model weights at best validation accuracy (87.50%)
- Judges will load this to test on their videos

### 2. 📝 **Training Log** (Shows Progress)
```
Task1_Visual_Binary/training.log (49 KB)
```
- Complete training history
- Shows all 13 epochs trained
- Validation accuracy: Started 62.5% → Reached 87.50%
- Contains all metrics, loss values, F1 scores

### 3. 📊 **Confusion Matrices** (Visual Results)
```
Task1_Visual_Binary/logs/confusion_matrix_epoch5.png
```
- Shows how well model classifies each class
- Epoch 5 was the best (87.50% accuracy)
- Visual proof of performance

### 4. 🔮 **Prediction Script** (How to Use Model)
```
Task1_Visual_Binary/predict.py
```
**Usage for judges:**
```bash
cd Task1_Visual_Binary
source ../venv/bin/activate
python predict.py --video_path <test_video.avi> --simple
```
**Output**: `0` or `1` (single number)
- 0 = Low Attention (distracted/disengaged)
- 1 = High Attention (nominally/highly engaged)

### 5. ⚙️ **Configuration** (Model Settings)
```
Task1_Visual_Binary/config.py
```
- All hyperparameters used
- Model architecture: MobileNetV2 + Temporal Attention
- Training settings: Batch size, learning rate, etc.

---

## 📈 FINAL METRICS (From training.log)

| Metric | Value |
|--------|-------|
| **Best Val Accuracy** | **87.50%** ✓ |
| **F1 Score** | 0.9091 |
| **Training Accuracy** | 89.58% |
| **Requirement** | ≥70% |
| **Status** | **PASSED** ✅ |

**Confusion Matrix (Epoch 5):**
```
Low Attention:  2 correct, 1 wrong  (67% recall)
High Attention: 5 correct, 0 wrong (100% recall)
Overall: 7/8 correct = 87.50%
```

---

## 🎓 MODEL ARCHITECTURE

```
Input: Video (30 seconds)
  ↓
Extract 20 frames @ 5 FPS
  ↓
MobileNetV2 (pretrained) → 1280 features per frame
  ↓
Temporal Attention → Aggregate 20 frames
  ↓
Classifier (128 hidden units)
  ↓
Output: Class 0 or 1
```

**Parameters**: 2,256,067 (efficient!)

---

## 📝 WHAT JUDGES WILL DO

1. **Load your model**: `best_model.pth`
2. **Run on their test videos**: Using `predict.py --simple`
3. **Check accuracy**: Compare predictions with ground truth
4. **Verify**: Minimum 70% accuracy requirement

---

## ✅ SUBMISSION CHECKLIST

- ✅ Model trained and saved
- ✅ Validation accuracy ≥70% (achieved 87.50%)
- ✅ predict.py works with --simple flag
- ✅ Outputs only 0 or 1
- ✅ Training logs saved
- ✅ Confusion matrices generated
- ✅ All code files present
- ✅ README documentation included

---

## 🚀 QUICK TEST (For Judges)

```bash
# Navigate to Task 1 folder
cd Task1_Visual_Binary

# Activate environment
source ../venv/bin/activate

# Test prediction on a video
python predict.py --video_path ../data/train/distracted/subject_17_Vid_1.avi --simple

# Expected output: 0 or 1 (single digit)
```

---

**Submission Date**: February 7, 2026  
**Model**: MobileNetV2 + Temporal Attention  
**Achievement**: 87.50% accuracy (Target: ≥70%) ✓  
**Status**: ✅ QUALIFIED FOR SUBMISSION
