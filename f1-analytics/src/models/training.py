"""Training loops and utilities.

Provides a reusable Trainer class with early stopping, learning rate
scheduling, checkpoint saving, and experiment logging.
"""

import csv
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from config import CHECKPOINT_DIR, DEFAULT_DEVICE, DEFAULT_EPOCHS, DEFAULT_LEARNING_RATE

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 7, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class ExperimentLogger:
    """Log training metrics to a CSV file."""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self._header_written = False

    def log(self, metrics: dict) -> None:
        write_header = not self._header_written and not self.log_path.exists()
        with open(self.log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            if write_header:
                writer.writeheader()
                self._header_written = True
            writer.writerow(metrics)


class Trainer:
    """Reusable training loop for PyTorch models.

    Supports:
    - Train/val loss tracking per epoch
    - Early stopping
    - Learning rate scheduling (ReduceLROnPlateau)
    - Model checkpoint saving
    - CSV experiment logging
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        criterion: Optional[nn.Module] = None,
        device: Optional[str] = None,
        checkpoint_dir: Optional[Path] = None,
        experiment_name: str = "experiment",
    ):
        self.device = torch.device(device or DEFAULT_DEVICE)
        self.model = model.to(self.device)
        self.optimizer = optimizer or torch.optim.Adam(
            model.parameters(), lr=DEFAULT_LEARNING_RATE
        )
        self.criterion = criterion or nn.MSELoss()
        self.checkpoint_dir = checkpoint_dir or CHECKPOINT_DIR
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.logger = ExperimentLogger(
            self.checkpoint_dir / f"{experiment_name}_log.csv"
        )

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Run one training epoch. Returns average loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for X, y in dataloader:
            X, y = X.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(X)
            loss = self.criterion(output, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> float:
        """Run validation. Returns average loss."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for X, y in dataloader:
            X, y = X.to(self.device), y.to(self.device)
            output = self.model(X)
            loss = self.criterion(output, y)
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = DEFAULT_EPOCHS,
        patience: int = 7,
    ) -> dict:
        """Full training loop with early stopping and checkpointing.

        Returns:
            Dict with training history (train_losses, val_losses, best_epoch).
        """
        scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", patience=3, factor=0.5
        )
        early_stopping = EarlyStopping(patience=patience)

        history = {"train_loss": [], "val_loss": [], "lr": [], "epoch_time": []}
        best_val_loss = np.inf

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            epoch_time = time.time() - t0

            current_lr = self.optimizer.param_groups[0]["lr"]
            scheduler.step(val_loss)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["lr"].append(current_lr)
            history["epoch_time"].append(epoch_time)

            self.logger.log(
                {
                    "epoch": epoch,
                    "train_loss": f"{train_loss:.6f}",
                    "val_loss": f"{val_loss:.6f}",
                    "lr": f"{current_lr:.2e}",
                    "time_s": f"{epoch_time:.1f}",
                }
            )

            logger.info(
                "Epoch %d/%d — train_loss=%.4f  val_loss=%.4f  lr=%.2e  (%.1fs)",
                epoch,
                epochs,
                train_loss,
                val_loss,
                current_lr,
                epoch_time,
            )

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(f"{self.experiment_name}_best.pt")

            if early_stopping.step(val_loss):
                logger.info("Early stopping triggered at epoch %d", epoch)
                break

        # Save final model
        self.save_checkpoint(f"{self.experiment_name}_final.pt")

        history["best_epoch"] = int(np.argmin(history["val_loss"])) + 1
        history["best_val_loss"] = best_val_loss
        return history

    def save_checkpoint(self, filename: str) -> Path:
        """Save model and optimizer state."""
        path = self.checkpoint_dir / filename
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )
        logger.info("Checkpoint saved to %s", path)
        return path

    def load_checkpoint(self, filename: str) -> None:
        """Load model and optimizer state from checkpoint."""
        path = self.checkpoint_dir / filename
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info("Checkpoint loaded from %s", path)

    @torch.no_grad()
    def predict(self, dataloader: DataLoader) -> np.ndarray:
        """Generate predictions for an entire dataloader."""
        self.model.eval()
        predictions = []
        for X, _ in dataloader:
            X = X.to(self.device)
            output = self.model(X)
            predictions.append(output.cpu().numpy())
        return np.concatenate(predictions)
