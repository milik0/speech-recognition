import torch.nn as nn
import torchaudio

class MLP(nn.Module):
    def __init__(self, input_size=13, hidden_size=256, output_size=29, num_layers=3):
        super(MLP, self).__init__()
        
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.2))
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
        
        layers.append(nn.Linear(hidden_size, output_size))
        
        self.network = nn.Sequential(*layers)
        self.loss_fn = nn.CTCLoss(blank=28, zero_infinity=True)
    
    def forward(self, x):
        return self.network(x)
    
    def loss(self, log_probs, targets, input_lengths, target_lengths):
        return self.loss_fn(log_probs, targets, input_lengths, target_lengths)
    