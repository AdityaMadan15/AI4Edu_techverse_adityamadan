"""Dataset loader for student engagement videos"""

import os
import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

class EngagementVideoDataset(Dataset):
    """Dataset for loading video frames and labels"""
    
    def __init__(self, data_path, label_map, transform=None, mode='train', 
                 frames_per_video=50, fps=10, csv_path=None):
        """
        Args:
            data_path: Path to video files
            label_map: Dictionary mapping original labels to task-specific labels
            transform: Image transformations
            mode: 'train' or 'test'
            frames_per_video: Number of frames to extract per video
            csv_path: Path to CSV with labels (optional)
        """
        self.data_path = data_path
        self.label_map = label_map
        self.transform = transform
        self.mode = mode
        self.frames_per_video = frames_per_video
        self.max_frames = frames_per_video
        self.fps = fps
        
        # Load video paths and labels
        self.video_paths = []
        self.labels = []
        
        if csv_path and os.path.exists(csv_path):
            # Load from CSV
            df = pd.read_csv(csv_path)
            for idx, row in df.iterrows():
                video_path = os.path.join(data_path, row['video_name'])
                if os.path.exists(video_path):
                    self.video_paths.append(video_path)
                    self.labels.append(label_map[row['label']])
        else:
            # Load from folder structure: data/train/class_name/video.mp4
            for class_folder in os.listdir(data_path):
                class_path = os.path.join(data_path, class_folder)
                if os.path.isdir(class_path):
                    # Map folder name to label
                    if 'distracted' in class_folder.lower():
                        original_label = 0
                    elif 'disengaged' in class_folder.lower():
                        original_label = 0.33
                    elif 'nominally' in class_folder.lower():
                        original_label = 0.66
                    elif 'highly' in class_folder.lower():
                        original_label = 1
                    else:
                        continue
                    
                    # Get all videos in this class
                    for video_file in os.listdir(class_path):
                        if video_file.endswith(('.mp4', '.avi', '.mov', '.webm', '.MP4', '.AVI', '.MOV', '.WEBM')):
                            video_path = os.path.join(class_path, video_file)
                            self.video_paths.append(video_path)
                            self.labels.append(label_map[original_label])
        
        print(f"\n{'='*60}")
        print(f"DATASET LOADED: {mode.upper()} MODE")
        print(f"{'='*60}")
        print(f"Total videos: {len(self.video_paths)}")
        if len(self.video_paths) > 0:
            label_counts = np.bincount(self.labels)
            for i, count in enumerate(label_counts):
                print(f"  Class {i}: {count} videos")
        print(f"FPS: {self.fps} | Max frames per video: {self.max_frames}")
        print(f"{'='*60}\n")
    
    def __len__(self):
        return len(self.video_paths)
    
    def extract_frames(self, video_path):
        """Extract frames at specified FPS for better temporal analysis"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Warning: Could not open video {video_path}")
                return None
                
            frames = []
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            if original_fps == 0:
                original_fps = 30  # default fallback
            
            # Calculate frame interval for desired FPS
            frame_interval = max(1, int(original_fps / self.fps))
            
            if total_frames == 0:
                cap.release()
                return None
            
            # Sample frames at specified FPS (better for gaze tracking)
            frame_count = 0
            while len(frames) < self.frames_per_video:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                
                frame_count += 1
            
            cap.release()
            
            if len(frames) == 0:
                return None
            
            return frames
            
        except Exception as e:
            print(f"ERROR extracting frames from {video_path}: {e}")
            return None
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Extract frames (with progress indication)
        video_name = os.path.basename(video_path)
        if idx % 5 == 0:  # Print every 5th video to avoid terminal spam
            print(f"Processing video {idx+1}/{len(self.video_paths)}: {video_name}")
        
        frames = self.extract_frames(video_path)
        
        if frames is None:
            # Return a dummy sample if video loading fails
            dummy_frame = np.zeros((224, 224, 3), dtype=np.uint8)
            frames = [dummy_frame] * self.frames_per_video
        
        # Apply transforms to each frame
        transformed_frames = []
        for frame in frames:
            if self.transform:
                frame = self.transform(image=frame)['image']
            else:
                frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            transformed_frames.append(frame)
        
        # Stack frames: [T, C, H, W]
        video_tensor = torch.stack(transformed_frames)
        
        return video_tensor, label, video_path


def get_transforms(img_size=224, mode='train'):
    """Get image transformations for training/validation - Enhanced for robustness"""
    
    if mode == 'train':
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.4),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])


def get_dataloaders(data_path, label_map, batch_size=16, val_split=0.2, 
                   img_size=224, num_workers=4, fps=10, frames_per_video=50, csv_path=None):
    """Create train and validation dataloaders"""
    
    # Create dataset
    full_dataset = EngagementVideoDataset(
        data_path=data_path,
        label_map=label_map,
        transform=get_transforms(img_size, mode='train'),
        mode='train',
        frames_per_video=frames_per_video,
        fps=fps,
        csv_path=csv_path
    )
    
    # Split into train and validation
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    # CRITICAL: num_workers=0 to avoid Windows multiprocessing crash with OpenCV
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Changed from num_workers to 0 for Windows stability
        pin_memory=False  # Changed to False for CPU training
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Changed from num_workers to 0 for Windows stability
        pin_memory=False  # Changed to False for CPU training
    )
    
    return train_loader, val_loader
