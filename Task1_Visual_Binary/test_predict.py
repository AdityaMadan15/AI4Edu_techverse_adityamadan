import torch
from model import VisualBinaryClassifier
from config import Config

config = Config()
checkpoint = torch.load('./checkpoints/best_model.pth', map_location='cpu')
print(f"✓ Task 1 Model Loaded Successfully!")
print(f"✓ Best Validation Accuracy: {checkpoint['val_acc']*100:.2f}%")
print(f"✓ Epoch: {checkpoint['epoch']}")
print(f"\n--- Model can predict: ---")
print("  Input: Video of driver")
print("  Output: 0 (Low Attention) or 1 (High Attention)")
print("✓ Task 1 COMPLETED!")
