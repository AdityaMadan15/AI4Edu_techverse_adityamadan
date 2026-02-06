"""
Download and organize dataset from Google Drive
Run this script to setup your dataset for Phase A training
"""

import os
import gdown
import zipfile
import shutil

# Google Drive folder link
GDRIVE_FOLDER = "https://drive.google.com/drive/folders/1r4zmPmOY6c6I3jJaYqzqpyTEeLI1MsSX"

print("=" * 60)
print("DATASET DOWNLOAD & SETUP")
print("=" * 60)

# Create data directories
os.makedirs("data/train", exist_ok=True)
os.makedirs("data/test", exist_ok=True)
os.makedirs("data/raw", exist_ok=True)

print("\n📥 DOWNLOAD INSTRUCTIONS:")
print("-" * 60)
print("1. Open this link in your browser:")
print(f"   {GDRIVE_FOLDER}")
print("\n2. Download all files/folders to: data/raw/")
print("\n3. After downloading, organize files into this structure:")
print("""
   data/
     train/
       distracted/          # Label 0
       disengaged/          # Label 0.33  
       nominally_engaged/   # Label 0.66
       highly_engaged/      # Label 1
""")

print("\nOR if you have a CSV file with labels:")
print("""
   data/
     train_videos/
       video1.mp4
       video2.mp4
       ...
     labels.csv  (columns: video_name, label)
""")

print("\n" + "=" * 60)
print("MANUAL STEPS:")
print("=" * 60)
print("1. Go to: https://drive.google.com/drive/folders/1r4zmPmOY6c6I3jJaYqzqpyTEeLI1MsSX")
print("2. Click 'Download' (top right)")
print("3. Extract to 'data/raw/' folder")
print("4. Run: python organize_dataset.py")
print("=" * 60)

# Alternative: If files are directly downloadable
print("\n💡 TIP: If you can get direct file IDs, add them here:")
print("Example: gdown.download_folder(id='YOUR_FOLDER_ID', output='data/raw/')")
