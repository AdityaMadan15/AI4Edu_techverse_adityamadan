#!/usr/bin/env python3
"""Quick test to verify all dependencies are installed correctly"""

import sys

dependencies = {
    'torch': 'PyTorch',
    'torchvision': 'TorchVision',
    'cv2': 'OpenCV',
    'numpy': 'NumPy',
    'pandas': 'Pandas',
    'sklearn': 'scikit-learn',
    'albumentations': 'Albumentations',
    'tqdm': 'tqdm',
    'timm': 'timm',
}

print("=" * 60)
print("DEPENDENCY CHECK")
print("=" * 60)

all_installed = True
for module, name in dependencies.items():
    try:
        __import__(module)
        print(f"✓ {name:20s} installed")
    except ImportError:
        print(f"✗ {name:20s} NOT INSTALLED")
        all_installed = False

print("=" * 60)
if all_installed:
    print("SUCCESS! All dependencies are installed.")
    print("\nYou can now run:")
    print("  python train_task1_binary.py --data_path data/train/ --epochs 15 --batch_size 4")
else:
    print("FAILED! Some dependencies are missing.")
    print("\nRun: python install_dependencies.bat")
    sys.exit(1)
