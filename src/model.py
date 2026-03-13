import torch
import torch.nn as nn

class LSTMWinPredictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # A smaller, single-layer LSTM is often more "sensitive" to changes
        self.lstm = nn.LSTM(input_dim, 32, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1]).squeeze(-1)