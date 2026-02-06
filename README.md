# Student Engagement Recognition - Phase A
**Team TechVerse | IIT Ropar AI Hackathon 2026**

## 🎯 Checkpoint Status: PHASE A

### Tasks Implemented:
- ✅ **Task 1:** Visual Binary Classification (Target: 70% accuracy)
- ✅ **Task 2:** Visual Multi-Class Classification (Target: 65% accuracy)

---

## 📁 Dataset Setup

Place your dataset in the following structure:
```
data/
  train/
    distracted/          # label 0
    disengaged/          # label 0.33
    nominally_engaged/   # label 0.66
    highly_engaged/      # label 1
  test/
    (videos for testing)
```

Or use video files with CSV labels:
```
data/
  train_videos/
  labels.csv  # columns: video_name, label
```

---

## 🚀 Quick Start (CHECKPOINT READY)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Task 1 (Binary - 70% target)
```bash
python train_task1_binary.py --data_path data/ --epochs 20 --batch_size 16
```

### 3. Train Task 2 (Multi-Class - 65% target)
```bash
python train_task2_multiclass.py --data_path data/ --epochs 20 --batch_size 16
```

### 4. Evaluate Models
```bash
python evaluate.py --task 1 --model_path models/task1_best.pth
python evaluate.py --task 2 --model_path models/task2_best.pth
```

---

## 🏗️ Architecture
- **Backbone:** ResNet-50 (pretrained on ImageNet)
- **Input:** Video frames (224x224)
- **Features:** Facial landmarks, frame features
- **Training:** Transfer learning + fine-tuning

---

## 📊 Model Performance
- Task 1 Binary: TBD after training
- Task 2 Multi-Class: TBD after training

---

## 👥 Team Branches
- `main` - Protected (merge only after approval)
- `aditya` - Your branch
- `member2` - Team member 2
- `member3` - Team member 3
- `member4` - Team member 4
