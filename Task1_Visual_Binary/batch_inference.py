"""
Batch Inference - Run prediction on all videos in a folder
"""
import os
import sys
import glob
import torch
import cv2
import numpy as np
from PIL import Image

from config import Config
from model import VisualBinaryClassifier
from data_loader import get_transforms


def predict_single_video(video_path, model, config, device, transform):
    """Predict on a single video"""
    # Extract frames
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
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
    
    if len(frames) < config.MIN_FRAMES:
        return None
    
    # Transform frames
    transformed_frames = []
    for frame in frames:
        frame = Image.fromarray(frame)
        frame = transform(frame)
        transformed_frames.append(frame)
    
    # Stack and predict
    video_tensor = torch.stack(transformed_frames).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(video_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return {
        'class': predicted_class,
        'class_name': config.CLASS_NAMES[predicted_class],
        'confidence': confidence,
        'probabilities': probabilities[0].cpu().numpy()
    }


def main(folder_path):
    config = Config()
    device = config.DEVICE
    
    # Load model
    print(f"\n{'='*70}")
    print("LOADING MODEL...")
    print(f"{'='*70}")
    
    model = VisualBinaryClassifier(
        num_classes=config.NUM_CLASSES,
        hidden_size=config.HIDDEN_SIZE,
        lstm_layers=config.LSTM_LAYERS,
        dropout=config.DROPOUT
    ).to(device)
    
    checkpoint = torch.load('./best_model.pth', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Model loaded successfully!")
    print(f"✓ Validation Accuracy: {checkpoint['val_acc']*100:.2f}%")
    
    # Get transform
    transform = get_transforms('val', config.IMG_SIZE)
    
    # Find all video files
    video_extensions = ['*.mp4', '*.avi', '*.webm', '*.mov', '*.mkv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(folder_path, ext)))
    
    if not video_files:
        print(f"\n❌ No video files found in: {folder_path}")
        return
    
    print(f"\n{'='*70}")
    print(f"FOUND {len(video_files)} VIDEO(S) TO PROCESS")
    print(f"{'='*70}\n")
    
    # Process each video
    results = []
    for i, video_path in enumerate(video_files, 1):
        video_name = os.path.basename(video_path)
        print(f"[{i}/{len(video_files)}] Processing: {video_name}")
        
        result = predict_single_video(video_path, model, config, device, transform)
        
        if result is None:
            print(f"    ❌ Failed to process (insufficient frames or error)\n")
            continue
        
        results.append({
            'video': video_name,
            'result': result
        })
        
        print(f"    🎯 Prediction: {result['class']} ({result['class_name']})")
        print(f"    💯 Confidence: {result['confidence']*100:.2f}%\n")
    
    # Summary
    print(f"\n{'='*70}")
    print("BATCH INFERENCE RESULTS SUMMARY")
    print(f"{'='*70}\n")
    
    for item in results:
        video_name = item['video']
        result = item['result']
        
        print(f"📹 {video_name}")
        print(f"   Class: {result['class']} → {result['class_name']}")
        print(f"   Confidence: {result['confidence']*100:.2f}%")
        print(f"   Probabilities: Low={result['probabilities'][0]*100:.1f}% | High={result['probabilities'][1]*100:.1f}%")
        print()
    
    print(f"{'='*70}")
    print(f"✓ Successfully processed {len(results)}/{len(video_files)} videos")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    else:
        folder_path = '../videos for testing/'
    
    if not os.path.exists(folder_path):
        print(f"\n❌ Folder not found: {folder_path}")
        sys.exit(1)
    
    main(folder_path)
