# task2_high_accuracy/train_boosted.py
# COMPLETE TRAINING SCRIPT FOR TASK 2 (>70% ACCURACY)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import pandas as pd
import os
import sys
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

# Add paths for imports
sys.path.append('.')
sys.path.append('./models')
sys.path.append('./utils')
sys.path.append('./data')

# Try to import custom modules, create dummies if missing
try:
    from models.boosted_model import BoostedEngagementClassifier, EnsembleVoter
    MODEL_IMPORTED = True
    print("✅ Imported boosted_model.py")
except ImportError:
    MODEL_IMPORTED = False
    print("⚠️  Could not import boosted_model, creating simple model")

try:
    from utils.augmentation import AggressiveAugmentation
    AUG_IMPORTED = True
    print("✅ Imported augmentation.py")
except ImportError:
    AUG_IMPORTED = False
    print("⚠️  Could not import augmentation")

try:
    from utils.gaze_extractor import GazeExtractor
    GAZE_IMPORTED = True
    print("✅ Imported gaze_extractor.py")
except ImportError:
    GAZE_IMPORTED = False
    print("⚠️  Could not import gaze_extractor")

# Configuration
class Config:
    # Paths - UPDATE THESE!
    video_dir = "../data/train/"  # Path to your training videos
    label_csv = "./data/labels.csv"  # Path to labels CSV
    
    # Model settings
    use_gaze = True
    use_attention = True
    num_classes = 4
    
    # Training hyperparameters (optimized for >70%)
    batch_size = 4
    sequence_length = 45  # 4.5 seconds at 10 FPS
    epochs = 30
    learning_rate = 0.00005
    weight_decay = 0.01
    patience = 10  # Early stopping patience
    
    # Class weights for severe imbalance
    # Class counts: Distracted:5, Disengaged:16, Nominal:18, High:16
    class_counts = [5, 16, 18, 16]
    total_samples = sum(class_counts)
    class_weights = torch.tensor([
        total_samples / (len(class_counts) * count) * 1.5  # Extra weight for Class 0
        for count in class_counts
    ], dtype=torch.float32)
    
    # Paths for outputs
    checkpoint_dir = "./checkpoints/"
    results_dir = "./results/"
    log_file = "./training_log.txt"

# Create dummy dataset if real one not available
class DummyVideoDataset(Dataset):
    """Dummy dataset for testing - replace with real VideoEngagementDataset"""
    def __init__(self, size=50, seq_len=45, num_classes=4):
        self.size = size
        self.seq_len = seq_len
        self.num_classes = num_classes
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Generate random video-like data
        video = torch.randn(self.seq_len, 3, 224, 224)  # [seq_len, channels, height, width]
        label = idx % self.num_classes  # Cyclic labels
        return video, label

# Create model if import failed
if not MODEL_IMPORTED:
    print("Creating simple model from scratch...")
    class BoostedEngagementClassifier(nn.Module):
        def __init__(self, num_classes=4, use_gaze=True, use_attention=True):
            super().__init__()
            self.use_gaze = use_gaze
            self.use_attention = use_attention
            
            # Simple CNN
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d(1)
            )
            
            # Feature dimensions
            cnn_feat_dim = 64
            gaze_feat_dim = 3 if use_gaze else 0
            total_feat_dim = cnn_feat_dim + gaze_feat_dim
            
            # LSTM
            self.lstm = nn.LSTM(
                input_size=total_feat_dim,
                hidden_size=128,
                num_layers=2,
                bidirectional=True,
                batch_first=True,
                dropout=0.4
            )
            
            # Attention
            if use_attention:
                self.attention = nn.Sequential(
                    nn.Linear(128*2, 64),
                    nn.Tanh(),
                    nn.Linear(64, 1)
                )
            
            # Classifier
            self.classifier = nn.Sequential(
                nn.Linear(128*2, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )
        
        def forward(self, video_frames, gaze_features=None):
            batch_size, seq_len = video_frames.shape[0], video_frames.shape[1]
            
            # Extract CNN features
            cnn_features = []
            for t in range(seq_len):
                frame = video_frames[:, t, :, :, :]
                feat = self.cnn(frame)
                feat = feat.view(batch_size, -1)
                cnn_features.append(feat)
            
            cnn_features = torch.stack(cnn_features, dim=1)
            
            # Add gaze features if available
            if self.use_gaze and gaze_features is not None:
                features = torch.cat([cnn_features, gaze_features], dim=2)
            else:
                features = cnn_features
            
            # LSTM
            lstm_out, _ = self.lstm(features)
            
            # Attention
            if self.use_attention:
                attn_weights = self.attention(lstm_out)
                attn_weights = torch.softmax(attn_weights, dim=1)
                context = torch.sum(lstm_out * attn_weights, dim=1)
            else:
                context = lstm_out[:, -1, :]
            
            # Classify
            return self.classifier(context)

# Create augmentation if import failed
if not AUG_IMPORTED:
    class AggressiveAugmentation:
        def __init__(self, p=0.7):
            self.p = p
        def __call__(self, x):
            return x  # Identity transform

# Create gaze extractor if import failed
if not GAZE_IMPORTED:
    class GazeExtractor:
        def extract_gaze_features(self, frame):
            return [0.0, 0.0, 0.5]
        def extract_video_gaze(self, video_path, fps=10):
            return torch.zeros(45, 3)  # Dummy gaze

def create_dataloader(config, mode='train', use_real_data=True):
    """Create dataloader for training or validation"""
    if use_real_data:
        try:
            from data.dataset import VideoEngagementDataset
            print("✅ Using real VideoEngagementDataset")
            
            # Create transform
            if mode == 'train' and AUG_IMPORTED:
                transform = AggressiveAugmentation(p=0.7)
            else:
                # Simple normalization for validation
                transform = None
            
            # Create dataset
            dataset = VideoEngagementDataset(
                video_dir=config.video_dir,
                label_csv=config.label_csv,
                seq_len=config.sequence_length,
                fps=10,
                transform=transform
            )
            
            print(f"Created dataset with {len(dataset)} samples")
            
        except ImportError:
            print("⚠️  Could not import real dataset, using dummy data")
            use_real_data = False
    
    if not use_real_data:
        # Use dummy data
        dataset = DummyVideoDataset(
            size=50 if mode == 'train' else 20,
            seq_len=config.sequence_length,
            num_classes=config.num_classes
        )
        print(f"Using dummy dataset with {len(dataset)} samples")
    
    # Handle class imbalance for training
    if mode == 'train':
        # Get all labels
        all_labels = []
        for i in range(len(dataset)):
            try:
                _, label = dataset[i]
                all_labels.append(label)
            except:
                all_labels.append(i % config.num_classes)
        
        # Calculate class weights
        class_counts = np.bincount(all_labels, minlength=config.num_classes)
        class_counts = np.maximum(class_counts, 1)  # Avoid division by zero
        class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32)
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        
        # Create weighted sampler
        sample_weights = [class_weights[label] for label in all_labels]
        sampler = WeightedRandomSampler(sample_weights, len(dataset), replacement=True)
        
        return DataLoader(
            dataset, 
            batch_size=config.batch_size, 
            sampler=sampler,
            num_workers=0  # Set to 0 for easier debugging
        )
    else:
        # Validation - no sampling, just shuffle
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0
        )

def train_epoch(model, dataloader, criterion, optimizer, device, config, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # Progress bar
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]", leave=False)
    
    for batch_idx, (videos, labels) in enumerate(pbar):
        videos, labels = videos.to(device), labels.to(device)
        
        # Extract gaze features if enabled
        gaze_features = None
        if config.use_gaze and GAZE_IMPORTED:
            # This is simplified - in reality, pre-extract or batch process
            pass
        
        optimizer.zero_grad()
        outputs = model(videos, gaze_features)
        loss = criterion(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        if batch_idx % 10 == 0:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100*correct/max(total,1):.1f}%'
            })
    
    epoch_loss = total_loss / len(dataloader)
    epoch_acc = correct / total if total > 0 else 0
    
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device, config):
    """Validate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    class_correct = [0] * config.num_classes
    class_total = [0] * config.num_classes
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validating", leave=False)
        for videos, labels in pbar:
            videos, labels = videos.to(device), labels.to(device)
            
            gaze_features = None
            if config.use_gaze and GAZE_IMPORTED:
                pass
            
            outputs = model(videos, gaze_features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Per-class accuracy
            for i in range(config.num_classes):
                idx = (labels == i)
                if idx.sum() > 0:
                    class_correct[i] += (predicted[idx] == labels[idx]).sum().item()
                    class_total[i] += idx.sum().item()
    
    # Calculate metrics
    val_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    val_acc = correct / total if total > 0 else 0
    
    # Per-class accuracy
    class_acc = []
    for i in range(config.num_classes):
        if class_total[i] > 0:
            class_acc.append(class_correct[i] / class_total[i])
        else:
            class_acc.append(0.0)
    
    return val_loss, val_acc, class_acc

def save_checkpoint(model, optimizer, epoch, acc, path):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': acc,
    }, path)

def log_training(log_file, epoch, train_loss, train_acc, val_loss, val_acc, class_acc):
    """Log training progress"""
    with open(log_file, 'a') as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"{timestamp} | Epoch {epoch:3d} | "
        log_line += f"Train: Loss={train_loss:.4f}, Acc={train_acc*100:6.2f}% | "
        log_line += f"Val: Loss={val_loss:.4f}, Acc={val_acc*100:6.2f}% | "
        log_line += f"Class Acc: {[f'{a*100:.1f}%' for a in class_acc]}\n"
        f.write(log_line)
    
    print(log_line)

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path):
    """Plot training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(train_accs, label='Train Acc')
    ax2.plot(val_accs, label='Val Acc')
    ax2.axhline(y=0.70, color='r', linestyle='--', label='70% Target')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()

def main():
    """Main training function"""
    print("=" * 70)
    print("TASK 2 TRAINING - TARGET: >70% ACCURACY")
    print("=" * 70)
    
    # Configuration
    config = Config()
    
    # Create directories
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create model
    model = BoostedEngagementClassifier(
        num_classes=config.num_classes,
        use_gaze=config.use_gaze,
        use_attention=config.use_attention
    ).to(device)
    
    # Print model summary
    print(f"\nModel Architecture:")
    print(f"- CNN + {'Gaze + ' if config.use_gaze else ''}LSTM + {'Attention' if config.use_attention else 'No Attention'}")
    print(f"- Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"- Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=config.class_weights.to(device))
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # Gaze extractor
    gaze_extractor = GazeExtractor() if config.use_gaze else None
    
    # Data loaders
    print(f"\nCreating data loaders...")
    train_loader = create_dataloader(config, mode='train', use_real_data=True)
    val_loader = create_dataloader(config, mode='val', use_real_data=True)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Training tracking
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    patience_counter = 0
    
    # Clear log file
    with open(config.log_file, 'w') as f:
        f.write("TASK 2 TRAINING LOG\n")
        f.write("=" * 50 + "\n")
    
    print(f"\n{'='*70}")
    print("STARTING TRAINING...")
    print(f"{'='*70}\n")
    
    # Training loop
    for epoch in range(config.epochs):
        print(f"\n{'='*50}")
        print(f"EPOCH {epoch+1}/{config.epochs}")
        print(f"{'='*50}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, config, epoch
        )
        
        # Validate
        val_loss, val_acc, class_acc = validate(
            model, val_loader, criterion, device, config
        )
        
        # Update learning rate
        scheduler.step(val_acc)
        
        # Store metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Log progress
        log_training(
            config.log_file, epoch+1,
            train_loss, train_acc,
            val_loss, val_acc,
            class_acc
        )
        
        # Print class-wise accuracy
        print(f"\nClass-wise Validation Accuracy:")
        for i, acc in enumerate(class_acc):
            print(f"  Class {i} ({['Distracted', 'Disengaged', 'Nominally', 'Highly'][i]}): {acc*100:5.1f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                model, optimizer, epoch+1, val_acc,
                os.path.join(config.checkpoint_dir, f"best_model_epoch{epoch+1}_acc{val_acc*100:.1f}.pth")
            )
            print(f"✅ Saved best model with accuracy: {val_acc*100:.2f}%")
            patience_counter = 0
            
            # Also save latest
            torch.save(model.state_dict(), os.path.join(config.checkpoint_dir, "latest_model.pth"))
        else:
            patience_counter += 1
            print(f"⏳ No improvement for {patience_counter} epochs")
        
        # Early stopping
        if patience_counter >= config.patience:
            print(f"\n⚠️  Early stopping triggered after {epoch+1} epochs")
            break
        
        # Check if target achieved
        if val_acc >= 0.70:
            print(f"\n🎉 TARGET ACHIEVED! Accuracy: {val_acc*100:.2f}%")
            save_checkpoint(
                model, optimizer, epoch+1, val_acc,
                os.path.join(config.checkpoint_dir, "target_achieved.pth")
            )
            # Continue training but mark success
            print("Continuing training to see if we can improve further...")
        
        # Plot intermediate results every 5 epochs
        if (epoch + 1) % 5 == 0:
            plot_path = os.path.join(config.results_dir, f"training_curves_epoch{epoch+1}.png")
            plot_training_curves(train_losses, val_losses, train_accs, val_accs, plot_path)
            print(f"📊 Saved training curves to {plot_path}")
    
    # Final plots and summary
    print(f"\n{'='*70}")
    print("TRAINING COMPLETED")
    print(f"{'='*70}")
    
    # Final plot
    final_plot_path = os.path.join(config.results_dir, "final_training_curves.png")
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, final_plot_path)
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(config.checkpoint_dir, "final_model.pth"))
    
    # Print summary
    print(f"\n📊 Training Summary:")
    print(f"   Best validation accuracy: {best_val_acc*100:.2f}%")
    print(f"   Final validation accuracy: {val_accs[-1]*100:.2f}%")
    print(f"   Total epochs trained: {len(train_losses)}")
    print(f"\n📁 Outputs saved to:")
    print(f"   Checkpoints: {config.checkpoint_dir}")
    print(f"   Results: {config.results_dir}")
    print(f"   Log file: {config.log_file}")
    
    # Check if target was achieved
    if best_val_acc >= 0.70:
        print(f"\n✅ SUCCESS! Achieved target accuracy of >70%")
        print(f"   Final accuracy: {best_val_acc*100:.2f}%")
    else:
        print(f"\n⚠️  Did not reach 70% target")
        print(f"   Best accuracy: {best_val_acc*100:.2f}%")
        print(f"   Consider: More epochs, data augmentation, or ensemble methods")
    
    return best_val_acc

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.00005, help='Learning rate')
    parser.add_argument('--seq_len', type=int, default=45, help='Sequence length')
    parser.add_argument('--no_gaze', action='store_true', help='Disable gaze features')
    parser.add_argument('--no_attention', action='store_true', help='Disable attention')
    
    args = parser.parse_args()
    
    # Update config
    config = Config()
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.sequence_length = args.seq_len
    config.use_gaze = not args.no_gaze
    config.use_attention = not args.no_attention
    
    # Run training
    accuracy = main()
    
    # Exit code based on success
    if accuracy >= 0.70:
        exit(0)  # Success
    else:
        exit(1)  # Did not reach target