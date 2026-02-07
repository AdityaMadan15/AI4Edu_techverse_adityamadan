import os
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
from config import Config
from PIL import Image

def get_feature_extractor():
    # Load MobileNetV2 pretrained
    print("Loading MobileNetV2...")
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
    model = models.mobilenet_v2(weights=weights)
    
    # Remove classifier, we just want the features (bottleneck)
    # MobileNetV2 structure: features -> avgpool -> classifier
    # We want output of avgpool (which is 1280 dim)
    
    class FeatureExtractor(nn.Module):
        def __init__(self, original_model):
            super().__init__()
            self.features = original_model.features
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            
        def forward(self, x):
            x = self.features(x)
            x = self.avgpool(x)
            return torch.flatten(x, 1)
            
    return FeatureExtractor(model)

def extract_frames(video_path, fps=1):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps == 0: video_fps = 30 # fallback
    
    skip_frames = int(video_fps / fps)
    if skip_frames < 1: skip_frames = 1
    
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        if count % skip_frames == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
        count += 1
        
    cap.release()
    return frames

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup model
    model = get_feature_extractor()
    model = model.to(device)
    model.eval()
    
    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    X_features = []
    y_labels = []
    video_names = []
    
    # Iterate data
    if not os.path.exists(Config.DATA_DIR):
        print(f"Error: {Config.DATA_DIR} does not exist.")
        return

    print("Starting feature extraction...")
    
    total_videos = 0
    for class_name, label in Config.CLASS_MAP.items():
        class_dir = os.path.join(Config.DATA_DIR, class_name)
        if not os.path.exists(class_dir): continue
        total_videos += len([f for f in os.listdir(class_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.webm', '.mkv'))])

    with tqdm(total=total_videos) as pbar:
        for class_name, label in Config.CLASS_MAP.items():
            class_dir = os.path.join(Config.DATA_DIR, class_name)
            if not os.path.exists(class_dir):
                print(f"Skipping {class_name}, directory not found.")
                continue
                
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.mp4', '.avi', '.mov', '.webm', '.mkv')):
                    video_path = os.path.join(class_dir, fname)
                    
                    try:
                        # 1. Get frames
                        frames = extract_frames(video_path, fps=Config.FPS)
                        if not frames:
                            print(f"\nWarning: No frames extracted from {fname}")
                            continue
                            
                        # 2. Preprocess batch
                        batch = torch.stack([transform(f) for f in frames]).to(device)
                        
                        # 3. Extract features in batches to save memory
                        video_feats = []
                        with torch.no_grad():
                            # Process in chunks of 32 frames
                            for i in range(0, len(batch), Config.BATCH_SIZE):
                                sub_batch = batch[i:i+Config.BATCH_SIZE]
                                feats = model(sub_batch)
                                video_feats.append(feats.cpu().numpy())
                        
                        if not video_feats: continue
                        
                        video_feats = np.concatenate(video_feats, axis=0)
                        
                        # 4. Aggregate features (Mean Pooling over time)
                        # Result is 1 vector of size 1280 per video
                        avg_feat = np.mean(video_feats, axis=0)
                        
                        X_features.append(avg_feat)
                        y_labels.append(label)
                        video_names.append(fname)
                        
                    except Exception as e:
                        print(f"\nError processing {fname}: {e}")
                    
                    pbar.update(1)
    
    # Save
    os.makedirs(Config.FEATURES_DIR, exist_ok=True)
    
    X = np.array(X_features)
    y = np.array(y_labels)
    names = np.array(video_names)
    
    print(f"\nExtraction complete.")
    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    np.save(Config.FEATURES_FILE, X)
    np.save(Config.LABELS_FILE, y)
    np.save(os.path.join(Config.FEATURES_DIR, "video_names.npy"), names)
    print(f"Saved to {Config.FEATURES_DIR}")

if __name__ == "__main__":
    main()
