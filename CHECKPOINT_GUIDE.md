# ⏰ 1-HOUR CHECKPOINT GUIDE - PHASE A

## 🚨 CRITICAL: Dataset Setup First!

### Option 1: Folder Structure (Recommended)
```
data/
  train/
    distracted/          # Put all "Distracted" videos here
      video1.mp4
      video2.mp4
    disengaged/          # Put all "Disengaged" videos here
      video3.mp4
      video4.mp4
    nominally_engaged/   # Put all "Nominally Engaged" videos here
      video5.mp4
      video6.mp4
    highly_engaged/      # Put all "Highly Engaged" videos here
      video7.mp4
      video8.mp4
```

### Option 2: CSV Labels
```
data/
  train_videos/
    video1.mp4
    video2.mp4
    ...
  labels.csv
```

CSV format:
```csv
video_name,label
video1.mp4,0
video2.mp4,0.33
video3.mp4,0.66
video4.mp4,1
```

---

## ⚡ QUICK START (10 Minutes)

### 1. Install Dependencies (5 min)
```bash
pip install torch torchvision opencv-python pandas scikit-learn matplotlib seaborn pillow tqdm tensorboard albumentations timm
```

### 2. Create Data Folder
```bash
mkdir data
mkdir data\train
mkdir models
mkdir logs
```

### 3. Place Your Dataset
- Copy your 74 videos into the appropriate folder structure (Option 1)
- OR create a labels.csv file (Option 2)

---

## 🎯 TRAINING PHASE (40 Minutes)

### Task 1: Binary Classification (20 min)
Target: **70% accuracy**

```bash
# Start training
python train_task1_binary.py --data_path data/train/ --epochs 15 --batch_size 8

# If you have CSV labels:
python train_task1_binary.py --data_path data/train_videos/ --csv_path data/labels.csv --epochs 15 --batch_size 8
```

**Expected Output:**
- Training progress with accuracy
- Best model saved to `models/task1_best.pth`
- Should reach 70%+ accuracy by epoch 10-15

---

### Task 2: Multi-Class Classification (20 min)
Target: **65% accuracy**

```bash
# Start training
python train_task2_multiclass.py --data_path data/train/ --epochs 15 --batch_size 8

# If you have CSV labels:
python train_task2_multiclass.py --data_path data/train_videos/ --csv_path data/labels.csv --epochs 15 --batch_size 8
```

**Expected Output:**
- Training progress with accuracy
- Best model saved to `models/task2_best.pth`
- Should reach 65%+ accuracy by epoch 10-15

---

## 📊 MONITORING (During Training)

### Watch tensorboard:
```bash
tensorboard --logdir logs/
```
Open: http://localhost:6006

---

## ✅ CHECKPOINT DELIVERABLES

By end of 1 hour, you should have:

1. ✅ **Task 1 Model:** `models/task1_best.pth` (≥70% accuracy)
2. ✅ **Task 2 Model:** `models/task2_best.pth` (≥65% accuracy)
3. ✅ **Training Logs:** Accuracy curves in tensorboard
4. ✅ **Classification Reports:** Printed during training

---

## 🚨 TROUBLESHOOTING

### If GPU is not available:
Add `--batch_size 4` to reduce memory usage

### If videos won't load:
Check video formats (should be .mp4, .avi, or .mov)

### If accuracy is too low:
- Increase epochs: `--epochs 25`
- Try different learning rate: `--lr 0.0005`

### If training is too slow:
- Reduce batch size: `--batch_size 4`
- Use fewer frames per video (edit `dataset.py` line 31: `frames_per_video=15`)

---

## 📝 FOR CHECKPOINT PRESENTATION

### Show These Results:
1. Training curves (accuracy over epochs)
2. Final validation accuracy for both tasks
3. Confusion matrices
4. Classification reports

### Talk About:
- "We used ResNet-50 pretrained on ImageNet"
- "Transfer learning with temporal pooling across frames"
- "Task 1: Binary classification achieved X% accuracy (threshold: 70%)"
- "Task 2: Multi-class achieved Y% accuracy (threshold: 65%)"

---

## 🎯 AFTER CHECKPOINT

Once checkpoint passes, you'll work on:
- Phase B: rPPG signal extraction
- Phase C: Multimodal fusion (video + heart rate)
- Phase D: Deployment demo

---

## 💾 GIT COMMANDS

### Save your work:
```bash
git add .
git commit -m "Phase A: Task 1 & 2 completed - checkpoint ready"
git push origin aditya
```

**DO NOT push to main yet!**

---

## ⏱️ TIME ALLOCATION

- 0:00-0:10 → Setup & install dependencies
- 0:10-0:15 → Organize dataset
- 0:15-0:35 → Train Task 1 (binary)
- 0:35-0:55 → Train Task 2 (multi-class)
- 0:55-1:00 → Prepare presentation materials

---

**GOOD LUCK! YOU GOT THIS! 🚀**
