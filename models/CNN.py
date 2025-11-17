import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, n_classes=30, n_mels=40):  # 30 phonemes or characters
        super().__init__()
        
        # 1D convolutions along time axis
        # Input channels = n_mels (frequency bins treated as channels)
        self.conv1 = nn.Conv1d(n_mels, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool1d(2)  # pool along time axis
        
        self.dropout = nn.Dropout(0.3)
        
        # final classifier
        self.classifier = nn.Linear(512, n_classes)

    def forward(self, x):
        # x: [B, 1, F, T] from spectrogram (mel_spec gives [1, n_mels, time])
        # Reshape to [B, F, T] for 1D conv along time
        if x.dim() == 4:
            # Input is [B, 1, F, T]
            x = x.squeeze(1)  # -> [B, F, T]
        elif x.dim() == 3:
            # Input is already [B, F, T] or needs transpose
            # Check if last dim is n_mels (needs transpose)
            if x.shape[-1] == self.conv1.in_channels:
                x = x.transpose(1, 2)  # [B, T, F] -> [B, F, T]
        
        x = F.relu(self.conv1(x))
        x = self.pool(x)  # -> [B, 128, T/2]

        x = F.relu(self.conv2(x))
        x = self.pool(x)  # -> [B, 256, T/4]

        x = F.relu(self.conv3(x))
        x = self.pool(x)  # -> [B, 512, T/8]

        # Transpose to [B, T', C] for time-distributed classification
        x = x.transpose(1, 2)  # -> [B, T', 512]
        
        x = self.dropout(x)
        out = self.classifier(x)  # [B, T', n_classes]

        return out
