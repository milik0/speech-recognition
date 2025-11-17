import torch.nn as nn

class GRU(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=128, num_layers=2, n_classes=30):
        super().__init__()
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        self.classifier = nn.Linear(hidden_dim * 2, n_classes)

    def forward(self, x):
        # x: [B, T, F]
        out, _ = self.gru(x)       # out: [B, T, hidden*2]
        logits = self.classifier(out)
        return logits              # [B, T, n_classes]
