"""Evaluation script for trained models"""

import os
import torch
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from tqdm import tqdm

from config import Config
from dataset import EngagementVideoDataset, get_transforms
from model import get_model

def evaluate_model(model_path, data_path, task, device):
    """Evaluate a trained model"""
    
    # Load config
    config = Config()
    
    # Set up label map based on task
    if task == 1:
        label_map = config.BINARY_LABEL_MAP
        num_classes = config.TASK1_CLASSES
        class_names = config.TASK1_CLASS_NAMES
    elif task == 2:
        label_map = config.MULTICLASS_LABEL_MAP
        num_classes = config.TASK2_CLASSES
        class_names = config.TASK2_CLASS_NAMES
    else:
        raise ValueError("Task must be 1 or 2")
    
    # Load model
    print(f"Loading model from: {model_path}")
    model = get_model(num_classes=num_classes, backbone=config.BACKBONE, device=device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model trained for {checkpoint['epoch']} epochs")
    print(f"Best validation accuracy: {checkpoint['val_acc']:.4f}")
    
    # Load test data
    print("\nLoading test data...")
    test_dataset = EngagementVideoDataset(
        data_path=data_path,
        label_map=label_map,
        transform=get_transforms(config.IMG_SIZE, mode='test'),
        mode='test'
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=2
    )
    
    # Evaluate
    print("\nEvaluating...")
    all_preds = []
    all_labels = []
    all_video_paths = []
    
    with torch.no_grad():
        for videos, labels, video_paths in tqdm(test_loader):
            videos = videos.to(device)
            outputs = model(videos)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_video_paths.extend(video_paths)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    print("\n" + "="*60)
    print(f"TASK {task} EVALUATION RESULTS")
    print("="*60)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score (Macro): {f1:.4f}")
    print("\n" + "="*60)
    print("Classification Report:")
    print("="*60)
    print(classification_report(all_labels, all_preds, target_names=class_names))
    print("\n" + "="*60)
    print("Confusion Matrix:")
    print("="*60)
    print(confusion_matrix(all_labels, all_preds))
    print("="*60)
    
    # Check qualification
    if task == 1:
        threshold = 0.70
    elif task == 2:
        threshold = 0.65
    
    if accuracy >= threshold:
        print(f"\n✓ QUALIFICATION THRESHOLD MET: {accuracy:.4f} >= {threshold:.2f}")
    else:
        print(f"\n✗ QUALIFICATION THRESHOLD NOT MET: {accuracy:.4f} < {threshold:.2f}")
    
    return accuracy, f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--task', type=int, required=True, choices=[1, 2],
                       help='Task number: 1 (binary) or 2 (multi-class)')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--data_path', type=str, default='data/test/',
                       help='Path to test data')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    evaluate_model(args.model_path, args.data_path, args.task, device)
