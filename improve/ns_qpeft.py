import torch
import torch.nn as nn


# ----------------------------------
# Residual Block
# ----------------------------------

class ResidualBlock(nn.Module):

    def __init__(self, hidden_dim):

        super().__init__()

        self.block = nn.Sequential(

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim)

        )

        self.activation = nn.ReLU()

    def forward(self, x):

        identity = x

        out = self.block(x)

        out = out + identity

        out = self.activation(out)

        return out


# ----------------------------------
# Quantum PEFT Adapter
# ----------------------------------

class QuantumPEFT(nn.Module):

    def __init__(self,
                 hidden_dim=64,
                 adapter_dim=16):

        super().__init__()

        self.down_projection = nn.Linear(
            hidden_dim,
            adapter_dim
        )

        self.activation = nn.ReLU()

        self.up_projection = nn.Linear(
            adapter_dim,
            hidden_dim
        )

    def forward(self, x):

        identity = x

        x = self.down_projection(x)

        x = self.activation(x)

        x = self.up_projection(x)

        x = x + identity

        return x


# ----------------------------------
# NS-QPEFT
# ----------------------------------

class NSQPEFT(nn.Module):

    def __init__(self, input_size=12):

        super().__init__()

        # Feature Encoder

        self.encoder = nn.Sequential(

            nn.Linear(input_size, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU()

        )

        # Residual Learning

        self.residual1 = ResidualBlock(64)

        # Parameter Efficient Adapter

        self.adapter = QuantumPEFT(
            hidden_dim=64,
            adapter_dim=16
        )

        # Another Residual Layer

        self.residual2 = ResidualBlock(64)

        # Prediction Head

        self.predictor = nn.Sequential(

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)

        )

    def forward(self, x):

        x = self.encoder(x)

        x = self.residual1(x)

        x = self.adapter(x)

        x = self.residual2(x)

        x = self.predictor(x)

        return x