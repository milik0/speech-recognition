import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: [B, T, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return x

class Transformer(nn.Module):
    def __init__(self, input_dim=40, d_model=256, nhead=8, num_layers=6, dim_feedforward=1024, n_classes=29, dropout=0.1):
        """
        Transformer-based speech recognition model
        
        Args:
            input_dim: Input feature dimension (e.g., 40 for mel spectrograms)
            d_model: Dimension of the model
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            dim_feedforward: Dimension of feedforward network
            n_classes: Number of output classes
            dropout: Dropout rate
        """
        super().__init__()
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output classifier
        self.classifier = nn.Linear(d_model, n_classes)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [B, T, F] where F is input_dim (e.g., 40 mel bins)
        
        # Project to d_model
        x = self.input_projection(x)  # [B, T, d_model]
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x)  # [B, T, d_model]
        
        # Classification
        logits = self.classifier(x)  # [B, T, n_classes]
        
        return logits

