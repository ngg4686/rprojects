"""Driver identification from anonymous telemetry traces.

Classifies which driver produced a given lap telemetry sequence.
Architecture: 1D ResNet-style temporal CNN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock1D(nn.Module):
    """Residual block for 1D convolution."""

    def __init__(self, channels: int, kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = out + residual
        return F.relu(out)


class DriverClassifier(nn.Module):
    """1D ResNet for driver classification from telemetry.

    Input shape: (batch, channels, sequence_length)
    Output shape: (batch, num_drivers) — class logits.
    """

    def __init__(
        self,
        in_channels: int = 6,
        num_drivers: int = 20,
        base_channels: int = 64,
        num_blocks: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()

        # Stem: project input channels to base_channels
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
        )

        # Stack of residual blocks with pooling between stages
        stages = []
        ch = base_channels
        for i in range(num_blocks):
            stages.append(ResBlock1D(ch, dropout=dropout))
            if i < num_blocks - 1:
                next_ch = ch * 2
                stages.append(nn.Conv1d(ch, next_ch, kernel_size=1))
                stages.append(nn.MaxPool1d(2))
                ch = next_ch

        self.backbone = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(ch, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_drivers),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.backbone(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted driver index."""
        logits = self.forward(x)
        return logits.argmax(dim=-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities over drivers."""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)
