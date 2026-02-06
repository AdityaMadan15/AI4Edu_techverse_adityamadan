"""
Organize downloaded dataset into proper structure
Run this after downloading data from Google Drive
"""

import os
import shutil
import pandas as pd
from pathlib import Path

def organize_dataset():
    """Organize videos into train folders by class"""
    
    raw_path = Path("data/raw")
    train_path = Path("data/train")
    
    # Create class folders
    class_folders = {
        'distracted': train_path / 'distracted',
        'disengaged': train_path / 'disengaged', 
        'nominally_engaged': train_path / 'nominally_engaged',
        'highly_engaged': train_path / 'highly_engaged'
    }
    
    for folder in class_folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("DATASET ORGANIZATION")
    print("=" * 60)
    
    # Check if raw data exists
    if not raw_path.exists():
        print("❌ Error: data/raw/ folder not found!")
        print("\nPlease download dataset from Google Drive first:")
        print("https://drive.google.com/drive/folders/1r4zmPmOY6c6I3jJaYqzqpyTEeLI1MsSX")
        return
    
    # Video extensions to look for
    video_extensions = ['.mp4', '.avi', '.mov', '.webm', '.MP4', '.AVI', '.MOV', '.WEBM']
    
    # First, find the Excel label file
    print("\n📄 Looking for label files...")
    excel_files = list(raw_path.rglob("*.xlsx")) + list(raw_path.rglob("*.xls"))
    csv_files = list(raw_path.rglob("*.csv"))
    
    label_df = None
    
    # Try Excel first
    if excel_files:
        label_file = excel_files[0]
        print(f"   ✓ Found Excel file: {label_file.name}")
        try:
            label_df = pd.read_excel(label_file)
            print(f"   ✓ Loaded {len(label_df)} entries")
            print(f"   Columns: {list(label_df.columns)}")
        except Exception as e:
            print(f"   ✗ Error reading Excel: {e}")
    
    # Try CSV if Excel didn't work
    elif csv_files:
        label_file = csv_files[0]
        print(f"   ✓ Found CSV file: {label_file.name}")
        try:
            label_df = pd.read_csv(label_file)
            print(f"   ✓ Loaded {len(label_df)} entries")
            print(f"   Columns: {list(label_df.columns)}")
        except Exception as e:
            print(f"   ✗ Error reading CSV: {e}")
    
    if label_df is not None:
        # Map labels to folders
        label_map = {
            0: 'distracted',
            0.33: 'disengaged',
            0.66: 'nominally_engaged', 
            1: 'highly_engaged'
        }
        
        print("\n📦 Organizing videos by labels...")
        
        # Find all videos in raw folder recursively
        all_videos = {}
        for ext in video_extensions:
            for video_path in raw_path.rglob(f"*{ext}"):
                all_videos[video_path.name.lower()] = video_path
        
        print(f"   Found {len(all_videos)} total video files")
        
        copied_count = 0
        missing_count = 0
        
        for idx, row in label_df.iterrows():
            # Try different column names
            video_name = None
            if 'video_name' in row:
                video_name = row['video_name']
            elif 'filename' in row:
                video_name = row['filename']
            elif 'Video_name' in row:
                video_name = row['Video_name']
            else:
                # Use first column
                video_name = row.iloc[0]
            
            # Get label
            if 'label' in row:
                label = float(row['label'])
            elif 'Label' in row:
                label = float(row['Label'])
            else:
                # Try second column
                label = float(row.iloc[1])
            
            # Find class folder
            if label in label_map:
                class_name = label_map[label]
            else:
                print(f"   ⚠️  Unknown label {label} for {video_name}")
                continue
            
            # Find the actual video file
            video_name_lower = str(video_name).lower()
            video_path = None
            
            # Try exact match first
            if video_name_lower in all_videos:
                video_path = all_videos[video_name_lower]
            else:
                # Try without extension
                video_base = os.path.splitext(video_name_lower)[0]
                for vname, vpath in all_videos.items():
                    if os.path.splitext(vname)[0] == video_base:
                        video_path = vpath
                        break
            
            if video_path and video_path.exists():
                dest = class_folders[class_name] / video_path.name
                if not dest.exists():
                    shutil.copy2(video_path, dest)
                    copied_count += 1
                    print(f"   ✓ {video_path.name} → {class_name}/")
            else:
                missing_count += 1
                print(f"   ✗ Not found: {video_name}")
        
        print(f"\n   Copied: {copied_count} videos")
        if missing_count > 0:
            print(f"   Missing: {missing_count} videos")
    
    else:
        print("\n⚠️  No label file found!")
        print("   Please organize videos manually")
    
    # Summary
    print("\n" + "=" * 60)
    print("FINAL DATASET STRUCTURE:")
    print("=" * 60)
    
    total_videos = 0
    for class_name, folder in class_folders.items():
        videos = list(folder.glob("*.*"))
        count = len(videos)
        total_videos += count
        print(f"   {class_name:20s}: {count:3d} videos")
    
    print(f"\n   TOTAL: {total_videos} videos")
    
    if total_videos >= 70:
        print("\n✅ Dataset organized successfully!")
        print("\n🚀 Next step: Install dependencies")
        print("   pip install -r requirements.txt")
        print("\n🚀 Then start training:")
        print("   python train_task1_binary.py --data_path data/train/ --epochs 15 --batch_size 8")
    else:
        print("\n⚠️  Warning: Expected ~74 videos but found", total_videos)
        print("   Please check if all files are downloaded correctly")

if __name__ == "__main__":
    organize_dataset()
