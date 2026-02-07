"""
INFERENCE DEMO - Task 1: Binary Classification
===============================================
This script demonstrates how to use the trained model to predict on ANY video.

Usage:
    python inference_demo.py path/to/your/video.avi
    python inference_demo.py path/to/your/video.mp4
    python inference_demo.py path/to/your/video.webm

Output:
    0 = Low Attention (distracted/disengaged)
    1 = High Attention (nominally/highly engaged)
"""

import sys
import os
import torch
import cv2
import numpy as np
from PIL import Image

from config import Config
from model import VisualBinaryClassifier
from data_loader import get_transforms


def predict_video(video_path, checkpoint_path='./best_model.pth'):
    """
    Predict engagement level from a video
    
    Args:
        video_path: Path to the video file
        checkpoint_path: Path to trained model weights
        
    Returns:
        dict with prediction results
    """
    config = Config()
    device = config.DEVICE
    
    # Load model
    print(f"\n{'='*60}")
    print("LOADING MODEL...")
    print(f"{'='*60}")
    
    model = VisualBinaryClassifier(
        num_classes=config.NUM_CLASSES,
        hidden_size=config.HIDDEN_SIZE,
        lstm_layers=config.LSTM_LAYERS,
        dropout=config.DROPOUT
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Model loaded successfully!")
    print(f"✓ Validation Accuracy: {checkpoint['val_acc']*100:.2f}%")
    
    # Extract frames
    print(f"\n{'='*60}")
    print("PROCESSING VIDEO...")
    print(f"{'='*60}")
    print(f"Video: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"❌ Cannot open video: {video_path}")
    
    frames = []
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps == 0:
        original_fps = 30
    
    frame_interval = max(1, int(original_fps / config.FPS))
    frame_count = 0
    
    while len(frames) < config.MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (config.IMG_SIZE, config.IMG_SIZE))
            frames.append(frame)
        
        frame_count += 1
    
    cap.release()
    
    print(f"✓ Extracted {len(frames)} frames at {config.FPS} FPS")
    
    if len(frames) < config.MIN_FRAMES:
        raise ValueError(f"❌ Not enough frames: {len(frames)} < {config.MIN_FRAMES}")
    
    # Transform frames
    transform = get_transforms('val', config.IMG_SIZE)
    transformed_frames = []
    for frame in frames:
        frame = Image.fromarray(frame)
        frame = transform(frame)
        transformed_frames.append(frame)
    
    # Stack and add batch dimension
    video_tensor = torch.stack(transformed_frames).unsqueeze(0).to(device)
    
    # Predict
    print(f"\n{'='*60}")
    print("MAKING PREDICTION...")
    print(f"{'='*60}")
    
    with torch.no_grad():
        outputs = model(video_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    # Results
    result = {
        'class': predicted_class,
        'class_name': config.CLASS_NAMES[predicted_class],
        'confidence': confidence,
        'probabilities': probabilities[0].cpu().numpy()
    }
    
    print(f"\n{'='*60}")
    print("PREDICTION RESULT")
    print(f"{'='*60}")
    print(f"📹 Video: {os.path.basename(video_path)}")
    print(f"🎯 Predicted Class: {result['class']}")
    print(f"📝 Class Name: {result['class_name']}")
    print(f"💯 Confidence: {result['confidence']*100:.2f}%")
    print(f"\n📊 Class Probabilities:")
    print(f"   {config.CLASS_NAMES[0]}: {result['probabilities'][0]*100:.2f}%")
    print(f"   {config.CLASS_NAMES[1]}: {result['probabilities'][1]*100:.2f}%")
    print(f"{'='*60}\n")
    
    return result


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("\n❌ ERROR: No video path provided!")
        print("\nUsage:")
        print("  python inference_demo.py <path_to_video>")
        print("\nExample:")
        print("  python inference_demo.py ../data/train/highly_engaged/subject_1_Vid_5.avi")
        print("  python inference_demo.py /path/to/your/new/video.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    if not os.path.exists(video_path):
        print(f"\n❌ ERROR: Video file not found: {video_path}")
        sys.exit(1)
    
    try:
        result = predict_video(video_path)
        
        # Simple output format (for automated testing)
        print("=" * 60)
        print("SIMPLE OUTPUT (for automated evaluation):")
        print(result['class'])
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
