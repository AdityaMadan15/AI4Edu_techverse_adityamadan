"""Data loader for video frames"""
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class VideoDataset(Dataset):
    def __init__(self, data_path, label_map, transform=None, fps=10, max_frames=100, min_frames=10, img_size=224):
        self.data_path = data_path
        self.label_map = label_map
        self.transform = transform
        self.fps = fps
        self.max_frames = max_frames
        self.min_frames = min_frames
        self.img_size = img_size
        
        self.video_paths = []
        self.labels = []
        self._load_dataset()
    
    def _load_dataset(self):
        if not os.path.exists(self.data_path):
            print(f"ERROR: Path not found: {self.data_path}")
            return
        
        for class_folder in os.listdir(self.data_path):
            class_path = os.path.join(self.data_path, class_folder)
            if not os.path.isdir(class_path):
                continue
            
            class_name = class_folder.lower().replace('_', ' ')
            
            # Map to binary label
            if 'distract' in class_name:
                label = 0
            elif 'disengage' in class_name:
                label = 0
            elif 'nominally' in class_name or 'nominal' in class_name:
                label = 1
            elif 'highly' in class_name or 'high' in class_name:
                label = 1
            else:
                continue
            
            for video_file in os.listdir(class_path):
                if video_file.lower().endswith(('.mp4', '.avi', '.mov', '.webm', '.wmv')):
                    self.video_paths.append(os.path.join(class_path, video_file))
                    self.labels.append(label)
        
        label_counts = np.bincount(self.labels)
        print(f"\n{'='*60}")
        print(f"DATASET LOADED")
        print(f"{'='*60}")
        print(f"Total videos: {len(self.video_paths)}")
        print(f"  Class 0 (Low Attention): {label_counts[0]} videos")
        print(f"  Class 1 (High Attention): {label_counts[1]} videos")
        print(f"FPS: {self.fps} | Max frames: {self.max_frames}")
        print(f"{'='*60}\n")
    
    def __len__(self):
        return len(self.video_paths)
    
    def extract_frames(self, video_path):
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"⚠️  Warning: Could not open video: {os.path.basename(video_path)}")
                return None
            
            # Quick test read to check if video is readable
            ret, test_frame = cap.read()
            if not ret or test_frame is None:
                print(f"⚠️  Warning: Cannot read frames from: {os.path.basename(video_path)}")
                cap.release()
                return None
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
            
            frames = []
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            if original_fps == 0:
                original_fps = 30
            
            frame_interval = max(1, int(original_fps / self.fps))
            frame_count = 0
            
            while len(frames) < self.max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (self.img_size, self.img_size))
                    frames.append(frame)
                
                frame_count += 1
            
            cap.release()
            
            # Pad or truncate to exactly max_frames
            if len(frames) < self.max_frames:
                # Pad with last frame or black frames
                if len(frames) > 0:
                    last_frame = frames[-1]
                    while len(frames) < self.max_frames:
                        frames.append(last_frame.copy())
                else:
                    return None
            elif len(frames) > self.max_frames:
                frames = frames[:self.max_frames]
            
            return frames if len(frames) >= self.min_frames else None
            
        except Exception as e:
            print(f"Error extracting {video_path}: {e}")
            return None
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Extract frames (no verbose output for speed)
        video_name = os.path.basename(video_path)
        
        frames = self.extract_frames(video_path)
        
        if frames is None or len(frames) != self.max_frames:
            # Create dummy frames with exact max_frames length
            frames = [np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)] * self.max_frames
        
        # Apply transforms
        transformed_frames = []
        for frame in frames:
            if self.transform:
                frame = Image.fromarray(frame)
                frame = self.transform(frame)
            else:
                frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            transformed_frames.append(frame)
        
        # Stack frames
        video_tensor = torch.stack(transformed_frames)
        
        return video_tensor, label, video_path


def get_transforms(mode='train', img_size=224):
    if mode == 'train':
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2)
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def get_dataloaders(config):
    dataset = VideoDataset(
        data_path=config.DATA_PATH,
        label_map=config.LABEL_MAP,
        transform=get_transforms('train', config.IMG_SIZE),
        fps=config.FPS,
        max_frames=config.MAX_FRAMES,
        min_frames=config.MIN_FRAMES,
        img_size=config.IMG_SIZE
    )
    
    # Split train/val
    dataset_size = len(dataset)
    val_size = int(dataset_size * config.VAL_SPLIT)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=False,
        drop_last=True  # Avoid single-sample batches that break BatchNorm
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=False,
        drop_last=False  # Keep all validation samples
    )
    
    return train_loader, val_loader
