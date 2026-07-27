import torch
import torch.nn as nn

# -----------------------------
# Quantum PEFT Adapter
# -----------------------------
class QuantumPEFT(nn.Module):

    def __init__(self, hidden_dim=32, adapter_dim=8):
        super().__init__()

        self.down = nn.Linear(hidden_dim, adapter_dim)
        self.relu = nn.ReLU()
        self.up = nn.Linear(adapter_dim, hidden_dim)

    def forward(self, x):

        residual = x

        x = self.down(x)
        x = self.relu(x)
        x = self.up(x)

        return residual + x


# -----------------------------
# NS-QPEFT
# -----------------------------
class NSQPEFT(nn.Module):

    def __init__(self, input_size=9):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU()
        )

        self.qpeft = QuantumPEFT(
            hidden_dim=32,
            adapter_dim=8
        )

        self.predictor = nn.Linear(32,1)

    def forward(self,x):

        x = self.feature_extractor(x)

        x = self.qpeft(x)

        x = self.predictor(x)

        return x