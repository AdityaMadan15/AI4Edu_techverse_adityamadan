"""Data loader for video frames - 4 Class Classification"""
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
            
            # Map to 4-class label
            if 'distract' in class_name:
                label = 0  # Distracted
            elif 'disengage' in class_name:
                label = 1  # Disengaged
            elif 'nominally' in class_name or 'nominal' in class_name:
                label = 2  # Nominally Engaged
            elif 'highly' in class_name or 'high' in class_name:
                label = 3  # Highly Engaged
            else:
                continue
            
            for video_file in os.listdir(class_path):
                if video_file.lower().endswith(('.mp4', '.avi', '.mov', '.webm', '.wmv')):
                    self.video_paths.append(os.path.join(class_path, video_file))
                    self.labels.append(label)
        
        label_counts = np.bincount(self.labels)
        print(f"\n{'='*60}")
        print(f"DATASET LOADED - 4 CLASS")
        print(f"{'='*60}")
        print(f"Total videos: {len(self.video_paths)}")
        print(f"  Class 0 (Distracted): {label_counts[0]} videos")
        print(f"  Class 1 (Disengaged): {label_counts[1]} videos")
        print(f"  Class 2 (Nominally Engaged): {label_counts[2]} videos")
        print(f"  Class 3 (Highly Engaged): {label_counts[3]} videos")
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
            
            # Quick test read
            ret, test_frame = cap.read()
            if not ret or test_frame is None:
                print(f"⚠️  Warning: Cannot read frames from: {os.path.basename(video_path)}")
                cap.release()
                return None
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_step = max(1, int(original_fps / self.fps))
            
            frames = []
            frame_idx = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % frame_step == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_resized = cv2.resize(frame_rgb, (self.img_size, self.img_size))
                    frames.append(frame_resized)
                    
                    if len(frames) >= self.max_frames:
                        break
                
                frame_idx += 1
            
            cap.release()
            
            if len(frames) < self.min_frames:
                print(f"⚠️  Warning: Only {len(frames)} frames in {os.path.basename(video_path)}, padding...")
                while len(frames) < self.min_frames:
                    frames.append(frames[-1] if frames else np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8))
            
            return frames
        
        except Exception as e:
            print(f"❌ Error extracting frames from {os.path.basename(video_path)}: {str(e)}")
            return None
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        frames = self.extract_frames(video_path)
        if frames is None:
            frames = [np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)] * self.max_frames
        
        # Ensure exactly max_frames by padding or truncating
        if len(frames) < self.max_frames:
            # Pad with last frame
            while len(frames) < self.max_frames:
                frames.append(frames[-1] if frames else np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8))
        elif len(frames) > self.max_frames:
            # Truncate
            frames = frames[:self.max_frames]
        
        if self.transform:
            frames = [self.transform(Image.fromarray(frame)) for frame in frames]
        else:
            frames = [torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0 for frame in frames]
        
        video_tensor = torch.stack(frames)
        
        return video_tensor, label, os.path.basename(video_path)

def get_dataloaders(config):
    """Create train and validation dataloaders"""
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset once to get size and indices
    temp_dataset = VideoDataset(
        data_path=config.DATA_PATH,
        label_map=config.LABEL_MAP,
        transform=None,
        fps=config.FPS,
        max_frames=config.MAX_FRAMES,
        min_frames=config.MIN_FRAMES,
        img_size=config.IMG_SIZE
    )
    
    dataset_size = len(temp_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(config.VAL_SPLIT * dataset_size))
    
    np.random.seed(config.SEED)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    
    # Create separate datasets with different transforms
    train_data = VideoDataset(
        data_path=config.DATA_PATH,
        label_map=config.LABEL_MAP,
        transform=transform_train,
        fps=config.FPS,
        max_frames=config.MAX_FRAMES,
        min_frames=config.MIN_FRAMES,
        img_size=config.IMG_SIZE
    )
    
    val_data = VideoDataset(
        data_path=config.DATA_PATH,
        label_map=config.LABEL_MAP,
        transform=transform_val,
        fps=config.FPS,
        max_frames=config.MAX_FRAMES,
        min_frames=config.MIN_FRAMES,
        img_size=config.IMG_SIZE
    )
    
    train_dataset = torch.utils.data.Subset(train_data, train_indices)
    val_dataset = torch.utils.data.Subset(val_data, val_indices)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}\n")
    
    return train_loader, val_loader
