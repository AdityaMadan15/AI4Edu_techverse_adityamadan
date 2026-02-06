"""Metrics and visualization utilities"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import os

def plot_confusion_matrix(cm, class_names, save_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved confusion matrix: {save_path}")

def save_metrics(accuracy, logs_path):
    """Save final metrics to JSON"""
    metrics = {
        'accuracy': float(accuracy),
        'accuracy_percentage': f"{accuracy*100:.2f}%",
        'qualification_threshold': '70%',
        'passed': bool(accuracy >= 0.70)
    }
    
    save_path = os.path.join(logs_path, 'final_metrics.json')
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✓ Saved metrics: {save_path}")
