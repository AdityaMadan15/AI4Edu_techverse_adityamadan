"""Prediction script for Task 1"""
import torch
import cv2
import numpy as np
from PIL import Image
import argparse

from config import Config
from model import VisualBinaryClassifier
from data_loader import get_transforms

class VideoPredictor:
    def __init__(self, checkpoint_path, config):
        self.config = config
        self.device = config.DEVICE
        self.transform = get_transforms('val', config.IMG_SIZE)
        
        # Load model
        self.model = VisualBinaryClassifier(
            num_classes=config.NUM_CLASSES,
            hidden_size=config.HIDDEN_SIZE,
            lstm_layers=config.LSTM_LAYERS,
            dropout=config.DROPOUT
        ).to(self.device)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✓ Loaded model from {checkpoint_path}")
        print(f"  Validation Accuracy: {checkpoint['val_acc']*100:.2f}%")
    
    def extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        frames = []
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        if original_fps == 0:
            original_fps = 30
        
        frame_interval = max(1, int(original_fps / self.config.FPS))
        frame_count = 0
        
        while len(frames) < self.config.MAX_FRAMES:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.config.IMG_SIZE, self.config.IMG_SIZE))
                frames.append(frame)
            
            frame_count += 1
        
        cap.release()
        return frames
    
    def predict(self, video_path):
        print(f"\nProcessing video: {video_path}")
        
        # Extract frames
        frames = self.extract_frames(video_path)
        print(f"Extracted {len(frames)} frames")
        
        if len(frames) < self.config.MIN_FRAMES:
            raise ValueError(f"Not enough frames: {len(frames)} < {self.config.MIN_FRAMES}")
        
        # Transform frames
        transformed_frames = []
        for frame in frames:
            frame = Image.fromarray(frame)
            frame = self.transform(frame)
            transformed_frames.append(frame)
        
        # Stack and add batch dimension
        video_tensor = torch.stack(transformed_frames).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(video_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        result = {
            'class': predicted_class,
            'class_name': self.config.CLASS_NAMES[predicted_class],
            'confidence': confidence,
            'probabilities': probabilities[0].cpu().numpy()
        }
        
        return result

def main(args):
    config = Config()
    
    predictor = VideoPredictor(args.checkpoint, config)
    result = predictor.predict(args.video_path)
    
    if args.simple:
        # Simple mode: Output ONLY the class number (for judges/automated testing)
        print(result['class'])
    else:
        # Verbose mode: Show detailed results
        print("\n" + "="*60)
        print("PREDICTION RESULT")
        print("="*60)
        print(f"Predicted Class: {result['class']} ({result['class_name']})")
        print(f"Confidence: {result['confidence']*100:.2f}%")
        print(f"Probabilities:")
        for i, prob in enumerate(result['probabilities']):
            print(f"  {config.CLASS_NAMES[i]}: {prob*100:.2f}%")
        print("="*60 + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict on a video')
    parser.add_argument('--video_path', type=str, required=True, help='Path to video file')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_model.pth', help='Model checkpoint')
    parser.add_argument('--simple', action='store_true', help='Output only class number (0 or 1) for automated testing')
    args = parser.parse_args()
    main(args)
