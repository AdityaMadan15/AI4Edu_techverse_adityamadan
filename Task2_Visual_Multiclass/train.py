"""Training script for Task 2: Multi-Class Classification (4 classes)"""
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
from model import VisualMultiClassifier
from data_loader import get_dataloaders
from utils.metrics import plot_confusion_matrix, save_metrics

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    print(f"\n{'='*60}")
    print(f"EPOCH {epoch}/{total_epochs} - TRAINING")
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
    epoch_f1 = f1_score(all_labels, all_preds, average='macro')
    
    return epoch_loss, epoch_acc, epoch_f1

def validate(model, val_loader, criterion, device, epoch, total_epochs):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    print(f"\n{'='*60}")
    print(f"EPOCH {epoch}/{total_epochs} - VALIDATION")
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
    epoch_f1 = f1_score(all_labels, all_preds, average='macro')
    
    return epoch_loss, epoch_acc, epoch_f1, all_preds, all_labels

def main(args):
    config = Config()
    set_seed(config.SEED)
    
    device = config.DEVICE
    print(f"\n{'='*60}")
    print(f"TASK 2: VISUAL MULTI-CLASS CLASSIFICATION (4 CLASSES)")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Backbone: {config.BACKBONE}")
    print(f"Classes: {config.CLASS_NAMES}")
    print(f"Learning Rate: {config.LEARNING_RATE}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Epochs: {config.NUM_EPOCHS}")
    print(f"{'='*60}\n")
    
    # Create directories
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(config.LOGS_PATH, exist_ok=True)
    
    # Load data
    train_loader, val_loader = get_dataloaders(config)
    
    # Initialize model
    model = VisualMultiClassifier(
        num_classes=config.NUM_CLASSES,
        hidden_size=config.HIDDEN_SIZE,
        lstm_layers=config.LSTM_LAYERS,
        dropout=config.DROPOUT
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=config.CLASS_WEIGHTS.to(device), label_smoothing=config.LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)
    
    # TensorBoard
    writer = SummaryWriter(config.LOGS_PATH)
    
    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(1, config.NUM_EPOCHS + 1):
        # Train
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, config.NUM_EPOCHS
        )
        
        # Validate
        val_loss, val_acc, val_f1, val_preds, val_labels = validate(
            model, val_loader, criterion, device, epoch, config.NUM_EPOCHS
        )
        
        # Update learning rate
        scheduler.step(val_acc)
        
        # Log metrics
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Val', val_acc, epoch)
        writer.add_scalar('F1/Train', train_f1, epoch)
        writer.add_scalar('F1/Val', val_f1, epoch)
        
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch} SUMMARY")
        print(f"{'='*60}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Train F1: {train_f1:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.2f}% | Val F1:   {val_f1:.4f}")
        print(f"{'='*60}\n")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
                'config': config
            }, os.path.join(config.MODEL_SAVE_PATH, 'best_model.pth'))
            
            print(f"✅ Best model saved! Validation Accuracy: {val_acc*100:.2f}%\n")
            
            # Save confusion matrix
            cm = confusion_matrix(val_labels, val_preds)
            plot_confusion_matrix(cm, config.CLASS_NAMES, 
                                os.path.join(config.MODEL_SAVE_PATH, 'confusion_matrix.png'))
            
            # Save classification report
            report = classification_report(val_labels, val_preds, 
                                          target_names=config.CLASS_NAMES, 
                                          digits=4)
            with open(os.path.join(config.MODEL_SAVE_PATH, 'classification_report.txt'), 'w') as f:
                f.write(report)
        else:
            patience_counter += 1
            print(f"⏳ No improvement for {patience_counter} epochs (Best: {best_val_acc*100:.2f}%)\n")
        
        # Early stopping
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"\n🛑 Early stopping triggered after {epoch} epochs")
            print(f"Best Validation Accuracy: {best_val_acc*100:.2f}%\n")
            break
    
    writer.close()
    
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETED")
    print(f"{'='*60}")
    print(f"Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print(f"Model saved to: {config.MODEL_SAVE_PATH}")
    print(f"{'='*60}\n")
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL CLASSIFICATION REPORT")
    print("="*60)
    print(report)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Task 2: Multi-Class Classification')
    args = parser.parse_args()
    main(args)
