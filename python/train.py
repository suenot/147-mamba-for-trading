"""
Training Module for Mamba Trading Models

This module provides utilities for training Mamba models
on financial time series data.
"""

import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    batch_size: int = 32
    epochs: int = 100
    early_stopping_patience: int = 10
    warmup_steps: int = 100
    gradient_clip: float = 1.0
    device: str = "auto"
    seed: int = 42


class TradingDataset(Dataset):
    """
    PyTorch Dataset for trading data.

    Args:
        X: Feature array of shape (n_samples, seq_len, n_features)
        y: Label array of shape (n_samples,) or (n_samples, n_classes)
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y) if y.dtype in [np.int32, np.int64] else torch.FloatTensor(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class MambaTrainer:
    """
    Trainer for Mamba trading models.

    Handles:
    - Training loop with validation
    - Learning rate scheduling
    - Early stopping
    - Model checkpointing
    - Logging

    Example:
        >>> trainer = MambaTrainer(model, config)
        >>> history = trainer.fit(train_loader, val_loader)
        >>> trainer.save("model.pt")
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[TrainingConfig] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: Mamba model to train
            config: Training configuration
        """
        self.model = model
        self.config = config or TrainingConfig()

        # Set device
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)

        self.model = self.model.to(self.device)

        # Set seed for reproducibility
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Initialize scheduler
        self.scheduler = None

        # Training state
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            epochs: Number of epochs (overrides config)
            verbose: Whether to print progress

        Returns:
            Training history dictionary
        """
        epochs = epochs or self.config.epochs

        # Setup scheduler
        total_steps = len(train_loader) * epochs
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate,
            total_steps=total_steps,
            pct_start=0.1,
        )

        for epoch in range(epochs):
            # Training phase
            train_loss, train_acc = self._train_epoch(train_loader)
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)

            # Validation phase
            if val_loader is not None:
                val_loss, val_acc = self._validate(val_loader)
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)

                # Early stopping check
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self._save_checkpoint("best_model.pt")
                else:
                    self.patience_counter += 1

                if self.patience_counter >= self.config.early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break

                if verbose:
                    print(
                        f"Epoch {epoch + 1}/{epochs} - "
                        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                        f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                    )
            else:
                if verbose:
                    print(
                        f"Epoch {epoch + 1}/{epochs} - "
                        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}"
                    )

        return self.history

    def _train_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for X, y in loader:
            X = X.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(X)

            # Calculate loss
            if self.model.task == "classification":
                loss = F.cross_entropy(outputs, y)
                preds = outputs.argmax(dim=-1)
                correct += (preds == y).sum().item()
            else:
                loss = F.mse_loss(outputs.squeeze(), y)

            total += y.size(0)
            total_loss += loss.item() * y.size(0)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip,
                )

            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

        avg_loss = total_loss / total
        accuracy = correct / total if self.model.task == "classification" else 0.0

        return avg_loss, accuracy

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for X, y in loader:
            X = X.to(self.device)
            y = y.to(self.device)

            outputs = self.model(X)

            if self.model.task == "classification":
                loss = F.cross_entropy(outputs, y)
                preds = outputs.argmax(dim=-1)
                correct += (preds == y).sum().item()
            else:
                loss = F.mse_loss(outputs.squeeze(), y)

            total += y.size(0)
            total_loss += loss.item() * y.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total if self.model.task == "classification" else 0.0

        return avg_loss, accuracy

    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "history": self.history,
        }, filename)

    def save(self, path: str):
        """Save the trained model."""
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        """Load a trained model."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))


def create_data_loaders(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
    val_split: float = 0.2,
    test_split: float = 0.1,
    shuffle: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.

    Args:
        X: Feature array
        y: Label array
        batch_size: Batch size
        val_split: Validation split ratio
        test_split: Test split ratio
        shuffle: Whether to shuffle training data

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    dataset = TradingDataset(X, y)

    # Calculate split sizes
    n_samples = len(dataset)
    test_size = int(n_samples * test_split)
    val_size = int(n_samples * val_split)
    train_size = n_samples - test_size - val_size

    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader


def train_mamba_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    config: Optional[TrainingConfig] = None,
    verbose: bool = True,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Convenience function to train a Mamba model.

    Args:
        model: Mamba model
        X_train: Training features
        y_train: Training labels
        X_val: Optional validation features
        y_val: Optional validation labels
        config: Training configuration
        verbose: Whether to print progress

    Returns:
        Tuple of (trained_model, history)
    """
    config = config or TrainingConfig()

    # Create datasets
    train_dataset = TradingDataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )

    val_loader = None
    if X_val is not None and y_val is not None:
        val_dataset = TradingDataset(X_val, y_val)
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
        )

    # Create trainer and train
    trainer = MambaTrainer(model, config)
    history = trainer.fit(train_loader, val_loader, verbose=verbose)

    return model, history


class SharpeLoss(nn.Module):
    """
    Custom loss function based on Sharpe ratio.

    Maximizes risk-adjusted returns instead of minimizing prediction error.
    """

    def __init__(self, risk_free_rate: float = 0.02):
        super().__init__()
        self.risk_free_rate = risk_free_rate / 252  # Daily rate

    def forward(
        self,
        predictions: torch.Tensor,
        returns: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate negative Sharpe ratio as loss.

        Args:
            predictions: Model predictions (position sizes or signals)
            returns: Actual returns

        Returns:
            Negative Sharpe ratio (to minimize)
        """
        # Calculate portfolio returns based on predictions
        # Assuming predictions are in range [-1, 1] for position sizing
        positions = torch.tanh(predictions.squeeze())
        portfolio_returns = positions * returns

        # Calculate Sharpe ratio components
        excess_returns = portfolio_returns - self.risk_free_rate
        mean_return = excess_returns.mean()
        std_return = excess_returns.std() + 1e-8

        # Return negative Sharpe (we want to maximize Sharpe)
        sharpe = mean_return / std_return
        return -sharpe


def plot_training_history(history: Dict[str, List[float]], save_path: Optional[str] = None):
    """
    Plot training history.

    Args:
        history: Training history dictionary
        save_path: Optional path to save the figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required for plotting")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss plot
    ax1 = axes[0]
    ax1.plot(history["train_loss"], label="Train Loss")
    if "val_loss" in history and history["val_loss"]:
        ax1.plot(history["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy plot (for classification)
    ax2 = axes[1]
    if history.get("train_acc") and any(history["train_acc"]):
        ax2.plot(history["train_acc"], label="Train Acc")
        if history.get("val_acc") and any(history["val_acc"]):
            ax2.plot(history["val_acc"], label="Val Acc")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Training and Validation Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
