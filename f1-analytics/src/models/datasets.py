"""PyTorch Dataset classes for F1 telemetry and race features.

Provides tensor-ready datasets for training deep learning models on
telemetry sequences and aggregated race feature vectors.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import TELEMETRY_CHANNELS

logger = logging.getLogger(__name__)


class TelemetryDataset(Dataset):
    """Dataset of fixed-length telemetry windows.

    Each sample is a (channels, window_size) tensor paired with a label
    (lap time in seconds for regression, or driver index for classification).
    """

    def __init__(
        self,
        windows: np.ndarray,
        labels: np.ndarray,
        normalize: bool = True,
    ):
        """
        Args:
            windows: Array of shape (N, window_size, channels).
            labels: Array of shape (N,) — lap times or driver indices.
            normalize: Whether to z-score normalize each channel.
        """
        self.windows = windows.astype(np.float32)
        self.labels = labels

        if normalize:
            mean = self.windows.mean(axis=(0, 1), keepdims=True)
            std = self.windows.std(axis=(0, 1), keepdims=True) + 1e-8
            self.windows = (self.windows - mean) / std
            self._mean = mean
            self._std = std

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Transpose to (channels, window_size) for Conv1d
        x = torch.from_numpy(self.windows[idx].T)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y

    @classmethod
    def from_preprocessed(
        cls,
        telemetry_windows: np.ndarray,
        lap_times: np.ndarray,
        fraction: float = 1.0,
    ) -> "TelemetryDataset":
        """Build a dataset, optionally using only a fraction of each window.

        Useful for the lap time predictor: use first N% of each lap.
        """
        if fraction < 1.0:
            cut = int(telemetry_windows.shape[1] * fraction)
            telemetry_windows = telemetry_windows[:, :cut, :]

        return cls(telemetry_windows, lap_times)


class RaceFeatureDataset(Dataset):
    """Dataset of aggregated per-driver-per-race feature vectors.

    Used for race outcome prediction models.
    """

    def __init__(
        self,
        features: pd.DataFrame,
        target_col: str = "position",
        feature_cols: Optional[list[str]] = None,
    ):
        """
        Args:
            features: DataFrame with one row per driver per race.
            target_col: Column to predict.
            feature_cols: Feature columns. If None, all numeric except target.
        """
        if feature_cols is None:
            feature_cols = [
                c
                for c in features.select_dtypes(include=[np.number]).columns
                if c != target_col
            ]

        self.feature_cols = feature_cols
        self.X = features[feature_cols].values.astype(np.float32)
        self.y = features[target_col].values.astype(np.float32)

        # Handle NaNs
        self.X = np.nan_to_num(self.X, nan=0.0)
        self.y = np.nan_to_num(self.y, nan=20.0)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx])


class DriverClassificationDataset(TelemetryDataset):
    """Telemetry dataset specialized for driver classification.

    Labels are integer driver indices. Includes a mapping from index to
    driver abbreviation.
    """

    def __init__(
        self,
        windows: np.ndarray,
        driver_labels: np.ndarray,
        driver_names: list[str],
        normalize: bool = True,
    ):
        # Map string labels to integer indices
        self.driver_to_idx = {name: i for i, name in enumerate(sorted(set(driver_names)))}
        self.idx_to_driver = {i: name for name, i in self.driver_to_idx.items()}
        self.num_drivers = len(self.driver_to_idx)

        int_labels = np.array([self.driver_to_idx[d] for d in driver_labels])
        super().__init__(windows, int_labels, normalize=normalize)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.windows[idx].T)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y
