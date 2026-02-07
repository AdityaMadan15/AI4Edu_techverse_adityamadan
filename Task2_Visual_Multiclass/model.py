"""Model: MobileNetV2 + Temporal Attention for 4-Class Classification
Based on Task 1's proven architecture (87.50% accuracy)"""
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import MobileNet_V2_Weights

class TemporalAttention(nn.Module):
    """Lightweight attention mechanism for video frames"""
    def __init__(self, feature_dim):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.Tanh(),
            nn.Linear(feature_dim // 4, 1)
        )
    
    def forward(self, x):
        # x: [batch, frames, features]
        attn_weights = self.attention(x)  # [batch, frames, 1]
        attn_weights = torch.softmax(attn_weights, dim=1)
        weighted = x * attn_weights
        output = torch.sum(weighted, dim=1)  # [batch, features]
        return output

class VisualMultiClassifier(nn.Module):
    """Video Classifier for 4-Class Student Engagement"""
    def __init__(self, num_classes=2, hidden_size=256, lstm_layers=2, dropout=0.5):
        super(VisualMultiClassifier, self).__init__()
        
        # MobileNetV2 backbone (efficient and lightweight)
        mobilenet = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        self.feature_dim = mobilenet.classifier[1].in_features  # 1280
        mobilenet.classifier = nn.Identity()
        
        # Freeze early layers to prevent overfitting
        for param in mobilenet.features[:14].parameters():
            param.requires_grad = False
        # Train later layers
        for param in mobilenet.features[14:].parameters():
            param.requires_grad = True
        
        self.backbone = mobilenet
        
        # Temporal attention instead of LSTM (simpler)
        self.temporal_attention = TemporalAttention(self.feature_dim)
        
        # Classification head with LayerNorm (more stable than BatchNorm)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),  # LayerNorm instead of BatchNorm1d - no batch size issues
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        batch_size, num_frames, c, h, w = x.shape
        
        # Extract features per frame
        x = x.view(batch_size * num_frames, c, h, w)
        features = self.backbone(x)
        
        # Reshape for temporal processing
        features = features.view(batch_size, num_frames, self.feature_dim)
        
        # Temporal attention pooling
        features = self.temporal_attention(features)
        
        # Classification
        logits = self.classifier(features)
        
        return logits
