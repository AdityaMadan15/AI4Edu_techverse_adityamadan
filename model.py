"""Model architectures for Phase A tasks"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights, ResNet18_Weights, EfficientNet_B0_Weights

class VideoClassifier(nn.Module):
    """Video classifier using CNN backbone + LSTM for temporal modeling (as per problem statement)"""
    
    def __init__(self, num_classes, backbone='resnet50', pretrained=True, dropout=0.5, use_lstm=True):
        super(VideoClassifier, self).__init__()
        
        self.backbone_name = backbone
        self.use_lstm = use_lstm
        
        # Load pretrained backbone
        if backbone == 'resnet50':
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove final FC layer
            
        elif backbone == 'resnet18':
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
        elif backbone == 'efficientnet_b0':
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Temporal modeling (LSTM as per problem statement)
        if use_lstm:
            self.lstm = nn.LSTM(
                input_size=feature_dim,
                hidden_size=512,
                num_layers=2,
                batch_first=True,
                dropout=dropout if dropout > 0 else 0,
                bidirectional=True
            )
            lstm_output_dim = 512 * 2  # bidirectional
        else:
            # Fallback: Simple temporal pooling
            self.temporal_pool = nn.AdaptiveAvgPool1d(1)
            lstm_output_dim = feature_dim
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: Video tensor [batch_size, num_frames, channels, height, width]
        Returns:
            logits: [batch_size, num_classes]
        """
        batch_size, num_frames, c, h, w = x.shape
        
        # Reshape to process all frames: [batch_size * num_frames, C, H, W]
        x = x.view(batch_size * num_frames, c, h, w)
        
        # Extract features from each frame using CNN
        features = self.backbone(x)  # [batch_size * num_frames, feature_dim]
        
        # Reshape back: [batch_size, num_frames, feature_dim]
        feature_dim = features.shape[1]
        features = features.view(batch_size, num_frames, feature_dim)
        
        # Temporal modeling
        if self.use_lstm:
            # LSTM processes temporal sequence (as per problem statement)
            lstm_out, (h_n, c_n) = self.lstm(features)  # lstm_out: [batch_size, num_frames, hidden_size*2]
            # Use last hidden state from both directions
            features = lstm_out[:, -1, :]  # [batch_size, hidden_size*2]
        else:
            # Fallback: Simple temporal pooling
            features = features.permute(0, 2, 1)
            features = self.temporal_pool(features).squeeze(-1)  # [batch_size, feature_dim]
        
        # Classification
        logits = self.classifier(features)
        
        return logits


def get_model(num_classes, backbone='resnet50', pretrained=True, use_lstm=True, device='cuda'):
    """Factory function to create model (with LSTM as per problem statement)"""
    model = VideoClassifier(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained,
        use_lstm=use_lstm
    )
    model = model.to(device)
    return model


if __name__ == '__main__':
    # Test model
    model = get_model(num_classes=2, backbone='resnet50', device='cpu', use_lstm=True)
    
    # Dummy input: [batch=2, frames=30, channels=3, height=224, width=224]
    dummy_input = torch.randn(2, 30, 3, 224, 224)
    
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

