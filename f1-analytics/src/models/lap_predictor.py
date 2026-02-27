"""Lap time prediction model.

Predicts total lap time from a partial telemetry sequence (first N% of a lap).
Architectures: 1D CNN, LSTM/GRU, or Transformer encoder.
"""

import torch
import torch.nn as nn


class LapPredictorCNN(nn.Module):
    """1D CNN for lap time regression from telemetry sequences.

    Input shape: (batch, channels, sequence_length)
    Output shape: (batch, 1) — predicted lap time in seconds.
    """

    def __init__(
        self,
        in_channels: int = 6,
        hidden_channels: int = 64,
        num_blocks: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()

        layers = []
        ch_in = in_channels
        for i in range(num_blocks):
            ch_out = hidden_channels * (2 ** i)
            layers.extend(
                [
                    nn.Conv1d(ch_in, ch_out, kernel_size=7, padding=3),
                    nn.BatchNorm1d(ch_out),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Dropout(dropout),
                ]
            )
            ch_in = ch_out

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(ch_in, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x).squeeze(-1)


class LapPredictorLSTM(nn.Module):
    """Bidirectional LSTM for lap time regression.

    Input shape: (batch, channels, sequence_length)
    Output shape: (batch, 1)
    """

    def __init__(
        self,
        in_channels: int = 6,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, seq_len) -> (batch, seq_len, channels)
        x = x.permute(0, 2, 1)
        output, (h_n, _) = self.lstm(x)
        # Use final hidden states from both directions
        h_forward = h_n[-2]
        h_backward = h_n[-1]
        h = torch.cat([h_forward, h_backward], dim=-1)
        return self.head(h).squeeze(-1)


class LapPredictorTransformer(nn.Module):
    """Transformer encoder for lap time prediction.

    Splits the telemetry sequence into patches, embeds them, and uses
    a Transformer encoder + regression head.
    """

    def __init__(
        self,
        in_channels: int = 6,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        patch_size: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = nn.Linear(in_channels * patch_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, seq_len)
        batch_size = x.size(0)
        x = x.permute(0, 2, 1)  # (batch, seq_len, channels)

        # Create patches
        seq_len = x.size(1)
        n_patches = seq_len // self.patch_size
        x = x[:, : n_patches * self.patch_size, :]
        x = x.reshape(batch_size, n_patches, -1)  # (batch, n_patches, channels*patch_size)

        x = self.patch_embed(x)

        # Prepend CLS token
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)

        x = self.encoder(x)
        cls_out = x[:, 0]
        return self.head(cls_out).squeeze(-1)


# Convenience alias
LapPredictor = LapPredictorCNN
