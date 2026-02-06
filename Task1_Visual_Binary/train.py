"""Training script for Task 1: Binary Classification"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import argparse
import random

from config import Config
from model import VisualBinaryClassifier
from data_loader import get_dataloaders
from utils.metrics import plot_confusion_matrix, save_metrics

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    print(f"\n{'='*60}")
    print(f"EPOCH {epoch}/{args.epochs} - TRAINING")
    print(f"{'='*60}")
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", unit="batch")
    for videos, labels, _ in pbar:
        videos = videos.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(videos)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average='binary')
    
    return epoch_loss, epoch_acc, epoch_f1

def validate(model, val_loader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    print(f"\n{'='*60}")
    print(f"EPOCH {epoch}/{args.epochs} - VALIDATION")
    print(f"{'='*60}")
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]", unit="batch")
        for videos, labels, _ in pbar:
            videos = videos.to(device)
            labels = labels.to(device)
            
            outputs = model(videos)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average='binary')
    
    return epoch_loss, epoch_acc, epoch_f1, all_preds, all_labels

def main(args):
    config = Config()
    set_seed(config.SEED)
    
    device = config.DEVICE
    print(f"\n{'='*60}")
    print(f"TASK 1: VISUAL BINARY CLASSIFICATION")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Backbone: {config.BACKBONE}")
    print(f"Target Accuracy: ≥70%")
    print(f"{'='*60}\n")
    
    # Create directories
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(config.LOGS_PATH, exist_ok=True)
    
    # Tensorboard
    writer = SummaryWriter(os.path.join(config.LOGS_PATH, 'task1'))
    
    # Data loaders
    train_loader, val_loader = get_dataloaders(config)
    
    # Model
    print(f"\n{'='*60}")
    print(f"BUILDING MODEL")
    print(f"{'='*60}")
    model = VisualBinaryClassifier(
        num_classes=config.NUM_CLASSES,
        hidden_size=config.HIDDEN_SIZE,
        lstm_layers=config.LSTM_LAYERS,
        dropout=config.DROPOUT
    ).to(device)
    
    print(f"Model created: MobileNetV2 + Temporal Attention")
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"{'='*60}\n")
    
    # Loss and optimizer
    class_weights = config.CLASS_WEIGHTS.to(device)
    # Label smoothing helps prevent overfitting
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
    
    # Training loop
    best_acc = 0.0
    patience_counter = 0
    
    print(f"\n{'='*60}")
    print(f"STARTING TRAINING")
    print(f"{'='*60}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"{'='*60}\n")
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_acc, val_f1, val_preds, val_labels = validate(
            model, val_loader, criterion, device, epoch
        )
        
        scheduler.step(val_acc)
        
        # Log
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('F1/train', train_f1, epoch)
        writer.add_scalar('F1/val', val_f1, epoch)
        
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch}/{args.epochs} RESULTS")
        print(f"{'='*60}")
        print(f"Train → Loss: {train_loss:.4f} | Acc: {train_acc*100:.2f}% | F1: {train_f1:.4f}")
        print(f"Val   → Loss: {val_loss:.4f} | Acc: {val_acc*100:.2f}% | F1: {val_f1:.4f}")
        if val_acc >= 0.70:
            print(f"✓ QUALIFICATION THRESHOLD MET: {val_acc*100:.2f}% ≥ 70%")
        else:
            print(f"⏳ Current: {val_acc*100:.2f}% | Need: 70% (Gap: {(0.70-val_acc)*100:.2f}%)")
        print(f"{'='*60}\n")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
                'config': config
            }
            torch.save(checkpoint, os.path.join(config.MODEL_SAVE_PATH, 'best_model.pth'))
            
            print(f"✓ SAVED BEST MODEL | Accuracy: {best_acc*100:.2f}%\n")
            
            # Print detailed metrics
            print("Classification Report:")
            print(classification_report(val_labels, val_preds, target_names=config.CLASS_NAMES))
            print("\nConfusion Matrix:")
            cm = confusion_matrix(val_labels, val_preds)
            print(cm)
            
            # Save confusion matrix plot
            plot_confusion_matrix(cm, config.CLASS_NAMES, 
                                os.path.join(config.LOGS_PATH, f'confusion_matrix_epoch{epoch}.png'))
        else:
            patience_counter += 1
        
        # Check qualification
        if val_acc >= 0.70:
            print(f"\n🎉 QUALIFICATION ACHIEVED: {val_acc*100:.2f}% ≥ 70%\n")
        
        # Early stopping
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    # Save final metrics
    save_metrics(best_acc, config.LOGS_PATH)
    
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETED")
    print(f"{'='*60}")
    print(f"Best Validation Accuracy: {best_acc*100:.2f}%")
    print(f"Qualification (≥70%): {'✓ PASSED' if best_acc >= 0.70 else '✗ NOT MET'}")
    print(f"{'='*60}\n")
    
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Task 1: Binary Classification')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    args = parser.parse_args()
    main(args)
