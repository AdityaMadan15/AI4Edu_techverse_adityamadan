#!/usr/bin/env python3
"""Simple prediction script for Task 2: 4-class classification"""
import sys
import argparse
from pathlib import Path
import torch
import cv2
import numpy as np
from torchvision import transforms

from model import VisualMultiClassifier
from config import Config

CLASS_NAMES = {
    0: "Distracted",
    1: "Disengaged", 
    2: "Nominally Engaged",
    3: "Highly Engaged"
}

MEAN_STD = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def load_model(checkpoint_path, device):
    """Load trained model"""
    config = Config()
    model = VisualMultiClassifier(
        num_classes=config.NUM_CLASSES,
        hidden_size=config.HIDDEN_SIZE,
        lstm_layers=config.LSTM_LAYERS,
        dropout=config.DROPOUT
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def extract_frames(video_path, img_size=224):
    """Extract frames from video"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        resized = cv2.resize(frame, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        frames.append(resized)
    
    cap.release()
    
    if not frames:
        raise RuntimeError(f"No frames read from {video_path}")
    
    return frames


def select_frames(frames, target_length=30):
    """Sample frames uniformly"""
    total = len(frames)
    if total <= target_length:
        idxs = np.linspace(0, total - 1, num=total, dtype=int).tolist()
        repeated = (idxs * ((target_length + total - 1) // total))[:target_length]
        return [frames[i] for i in repeated]
    
    step = total / target_length
    idxs = [int(i * step) for i in range(target_length)]
    return [frames[i] for i in idxs]


def preprocess_frames(frames, img_size=224):
    """Convert frames to tensor"""
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN_STD[0], std=MEAN_STD[1])
    ])
    
    tensors = []
    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = transform(rgb)
        tensors.append(tensor)
    
    return torch.stack(tensors)


def predict(video_path, checkpoint_path, simple=False):
    """Predict class for video"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = load_model(checkpoint_path, device)
    
    # Extract and process frames
    frames = extract_frames(video_path)
    selected = select_frames(frames, target_length=30)
    
    # Preprocess
    frame_tensor = preprocess_frames(selected).unsqueeze(0).to(device)
    
    # Predict
    with torch.inference_mode():
        logits = model(frame_tensor)
        probs = torch.softmax(logits, dim=-1)
        pred_class = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, pred_class].item()
    
    if simple:
        print(pred_class)
    else:
        print("="*60)
        print("TASK 2: MULTI-CLASS PREDICTION")
        print("="*60)
        print(f"Predicted Class: {pred_class} ({CLASS_NAMES[pred_class]})")
        print(f"Confidence: {confidence*100:.2f}%")
        print(f"\nProbabilities:")
        for i, prob in enumerate(probs[0].cpu().tolist()):
            print(f"  {CLASS_NAMES[i]}: {prob*100:.2f}%")
        print("="*60)
    
    return pred_class


def main():
    parser = argparse.ArgumentParser(description='Task 2: Multi-class prediction')
    parser.add_argument('--video_path', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, 
                       default='checkpoints/best_model.pth')
    parser.add_argument('--simple', action='store_true')
    
    args = parser.parse_args()
    predict(args.video_path, args.checkpoint, args.simple)


if __name__ == '__main__':
    main()
