import argparse
import torch
import cv2
import joblib
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from extract_features import get_feature_extractor, extract_frames
from config import Config

def predict_video(video_path, simple_output=False):
    # 1. Setup Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Feature Extractor
    cnn_model = get_feature_extractor()
    cnn_model = cnn_model.to(device)
    cnn_model.eval()
    
    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 2. Extract Frames
    if not simple_output: print(f"Processing {video_path}...")
    try:
        frames = extract_frames(video_path, fps=Config.FPS)
    except Exception as e:
        print(f"Error reading video: {e}")
        return
        
    if not frames:
        print("Error: No frames extracted.")
        return

    # 3. Extract Features
    batch = torch.stack([transform(f) for f in frames]).to(device)
    
    video_feats = []
    with torch.no_grad():
        for i in range(0, len(batch), Config.BATCH_SIZE):
            sub_batch = batch[i:i+Config.BATCH_SIZE]
            feats = cnn_model(sub_batch)
            video_feats.append(feats.cpu().numpy())
            
    video_feats = np.concatenate(video_feats, axis=0)
    
    # Mean pooling
    avg_feat = np.mean(video_feats, axis=0).reshape(1, -1)
    
    # 4. Predict using SVM/RF
    try:
        classifier = joblib.load(Config.CLASSIFIER_PATH)
    except FileNotFoundError:
        print("Error: Model not found. Run train_classifier.py first.")
        return

    prediction = classifier.predict(avg_feat)[0]
    probs = classifier.predict_proba(avg_feat)[0]
    
    # Map int back to label
    idx_to_label = {v: k for k, v in Config.CLASS_MAP.items()}
    label_name = idx_to_label[prediction]
    
    if simple_output:
        print(prediction)
    else:
        print(f"Prediction: {prediction} ({label_name})")
        print(f"Confidence: {probs[prediction]:.2f}")
        print(f"Probabilities: {probs}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True, help="Path to input video")
    parser.add_argument("--simple", action="store_true", help="Output only the class integer")
    args = parser.parse_args()
    
    predict_video(args.video_path, args.simple)
