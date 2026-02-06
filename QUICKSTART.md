# 🚀 QUICK START - 1 HOUR CHECKPOINT

## ⚡ STEP-BY-STEP (DO THIS NOW!)

### **STEP 1: Download Dataset (5 min)**

1. Open this link in browser:
   ```
   https://drive.google.com/drive/folders/1r4zmPmOY6c6I3jJaYqzqpyTEeLI1MsSX
   ```

2. Click "Download" (top-right, or right-click → Download)

3. Extract the downloaded file to:
   ```
   D:\IIT Ropar\aiforeducation_techverse_adityamadan\data\raw\
   ```

---

### **STEP 2: Organize Dataset (2 min)**

```bash
python organize_dataset.py
```

This will organize videos into:
```
data/
  train/
    distracted/
    disengaged/
    nominally_engaged/
    highly_engaged/
```

---

### **STEP 3: Install Dependencies (3 min)**

```bash
pip install torch torchvision opencv-python pandas scikit-learn matplotlib seaborn pillow tqdm tensorboard albumentations timm gdown
```

Or use requirements file:
```bash
pip install -r requirements.txt
```

---

### **STEP 4: Test Model Loading (1 min)**

```bash
python model.py
```

Should output:
```
Input shape: torch.Size([2, 30, 3, 224, 224])
Output shape: torch.Size([2, 2])
Model parameters: XX.XXM
```

---

### **STEP 5: Train Task 1 - Binary (20 min)**

```bash
python train_task1_binary.py --data_path data/train/ --epochs 15 --batch_size 8
```

**Target: 70% accuracy**

Monitor progress - it will show:
- Epoch X/15
- Train accuracy
- Val accuracy
- Best model saved notification

---

### **STEP 6: Train Task 2 - Multi-Class (20 min)**

```bash
python train_task2_multiclass.py --data_path data/train/ --epochs 15 --batch_size 8
```

**Target: 65% accuracy**

---

### **STEP 7: Check Results (2 min)**

Look for these in the terminal:
```
✓ Saved best model with accuracy: 0.XXXX
✓ PASSED qualification threshold
```

Models saved to:
- `models/task1_best.pth` (Task 1)
- `models/task2_best.pth` (Task 2)

---

### **STEP 8: Prepare for Checkpoint (5 min)**

Create a summary of results:
- Task 1 Binary Accuracy: XX%
- Task 2 Multi-Class Accuracy: XX%
- Confusion matrices (printed during training)
- Training curves (open tensorboard)

```bash
tensorboard --logdir logs/
```

---

## 🆘 IF SOMETHING GOES WRONG

### Dataset Issues:
```bash
# List what's in data/raw/
dir data\raw

# Check if train folders have videos
dir data\train\distracted
dir data\train\disengaged
dir data\train\nominally_engaged
dir data\train\highly_engaged
```

### CUDA/GPU Issues:
If you see CUDA errors, use CPU:
```bash
# Add --batch_size 4 for CPU
python train_task1_binary.py --data_path data/train/ --epochs 15 --batch_size 4
```

### Out of Memory:
```bash
# Reduce batch size
python train_task1_binary.py --data_path data/train/ --epochs 15 --batch_size 4
```

---

## ✅ CHECKPOINT DELIVERABLES

Show your evaluator:
1. ✅ Task 1 model trained (≥70% accuracy)
2. ✅ Task 2 model trained (≥65% accuracy)  
3. ✅ Training logs showing improvement
4. ✅ Classification reports and confusion matrices
5. ✅ Saved model files in `models/` folder

---

## ⏰ TIME CHECK

Current time: **NOW**
Checkpoint: **1 hour from now**

- [0:00-0:10] Download & organize dataset
- [0:10-0:30] Train Task 1
- [0:30-0:50] Train Task 2  
- [0:50-1:00] Prepare results & presentation

---

## 🔥 AFTER CHECKPOINT PASSES

We'll implement:
- Phase B: rPPG heart rate extraction
- Phase C: Multimodal fusion
- Phase D: Demo app

But for now: **FOCUS ON PHASE A ONLY!**

---

**GO GO GO! ⚡**
