"""Training script for Task 2: Multi-Class Classification"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import argparse

from config import Config
from dataset import get_dataloaders
from model import get_model

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for videos, labels, _ in pbar:
        videos = videos.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(videos)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Metrics
        running_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': loss.item()})
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average='macro')
    
    return epoch_loss, epoch_acc, epoch_f1


def validate(model, val_loader, criterion, device, epoch):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
        for videos, labels, _ in pbar:
            videos = videos.to(device)
            labels = labels.to(device)
            
            outputs = model(videos)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average='macro')
    
    return epoch_loss, epoch_acc, epoch_f1, all_preds, all_labels


def main(args):
    # Configuration
    config = Config()
    device = config.DEVICE
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(config.LOGS_PATH, exist_ok=True)
    
    # Tensorboard
    writer = SummaryWriter(os.path.join(config.LOGS_PATH, 'task2_multiclass'))
    
    # Data loaders
    print("Loading dataset...")
    print(f"FPS: {config.FPS} (increased for gaze tracking)")
    train_loader, val_loader = get_dataloaders(
        data_path=args.data_path,
        label_map=config.MULTICLASS_LABEL_MAP,
        batch_size=args.batch_size,
        val_split=config.VAL_SPLIT,
        img_size=config.IMG_SIZE,
        num_workers=config.NUM_WORKERS,
        fps=config.FPS,
        frames_per_video=config.MAX_FRAMES,
        csv_path=args.csv_path
    )
    
    # Model
    print(f"Creating model with backbone: {config.BACKBONE}")
    print(f"Temporal modeling: {'LSTM (as per problem statement)' if config.USE_LSTM else 'Pooling'}")
    model = get_model(
        num_classes=config.TASK2_CLASSES,
        backbone=config.BACKBONE,
        pretrained=config.PRETRAINED,
        use_lstm=config.USE_LSTM,
        device=device
    )
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
    
    # Training loop
    best_acc = 0.0
    patience_counter = 0
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Target accuracy: 65% (Qualification threshold)\n")
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_acc, val_f1, val_preds, val_labels = validate(
            model, val_loader, criterion, device, epoch
        )
        
        # Scheduler step
        scheduler.step(val_acc)
        
        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('F1/train', train_f1, epoch)
        writer.add_scalar('F1/val', val_f1, epoch)
        
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
                'config': config
            }
            torch.save(checkpoint, os.path.join(config.MODEL_SAVE_PATH, 'task2_best.pth'))
            print(f"✓ Saved best model with accuracy: {best_acc:.4f}")
            
            # Print classification report
            print("\nClassification Report:")
            print(classification_report(val_labels, val_preds, 
                                      target_names=config.TASK2_CLASS_NAMES))
            print("\nConfusion Matrix:")
            print(confusion_matrix(val_labels, val_preds))
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break
        
        # Check qualification threshold
        if val_acc >= 0.65:
            print(f"\n🎉 ACHIEVED QUALIFICATION THRESHOLD: {val_acc:.4f} >= 65%")
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_acc:.4f}")
    print(f"Qualification threshold (65%): {'✓ PASSED' if best_acc >= 0.65 else '✗ NOT MET'}")
    
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Task 2: Multi-Class Classification')
    parser.add_argument('--data_path', type=str, default='data/train/',
                       help='Path to training data')
    parser.add_argument('--csv_path', type=str, default=None,
                       help='Path to CSV with labels (optional)')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    
    args = parser.parse_args()
    main(args)
