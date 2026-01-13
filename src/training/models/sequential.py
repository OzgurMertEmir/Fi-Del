"""
Sequential models using PyTorch.

Includes: LSTM, GRU, Transformer
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

from training.models.base import BaseModel

logger = logging.getLogger(__name__)


def get_device(device: str = "auto") -> str:
    """Determine the best available device."""
    import torch

    if device != "auto":
        return device

    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class PyTorchBaseModel(BaseModel):
    """Base class for PyTorch models."""

    def __init__(
        self,
        n_features: int,
        n_classes: int | None = 2,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        sequence_length: int = 60,
        **kwargs,
    ):
        super().__init__(n_features, n_classes)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.sequence_length = sequence_length
        self.device = get_device()

    def _create_dataloader(
        self, X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True
    ):
        """Create PyTorch DataLoader."""
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y) if self.is_classification else torch.FloatTensor(y)

        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        epochs: int = 100,
        batch_size: int = 256,
        learning_rate: float = 0.001,
        early_stopping_patience: int = 10,
        **kwargs,
    ) -> dict[str, Any]:
        """Train the PyTorch model."""
        import torch
        import torch.nn as nn

        logger.info(f"Training {self.__class__.__name__} on {self.device}...")

        self.model.to(self.device)

        # Loss function
        if self.is_classification:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=kwargs.get("weight_decay", 0.0001)
        )

        # Data loaders
        train_loader = self._create_dataloader(X_train, y_train, batch_size, shuffle=True)
        val_loader = None
        if X_val is not None:
            val_loader = self._create_dataloader(X_val, y_val, batch_size, shuffle=False)

        # Training loop
        history = {"train_loss": [], "val_loss": [], "val_acc": []}
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_X)

                if self.is_classification:
                    loss = criterion(outputs, batch_y)
                else:
                    loss = criterion(outputs.squeeze(), batch_y)

                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            history["train_loss"].append(train_loss)

            # Validation
            if val_loader:
                self.model.eval()
                val_loss = 0.0
                correct = 0
                total = 0

                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)

                        outputs = self.model(batch_X)

                        if self.is_classification:
                            loss = criterion(outputs, batch_y)
                            _, predicted = torch.max(outputs, 1)
                            correct += (predicted == batch_y).sum().item()
                            total += batch_y.size(0)
                        else:
                            loss = criterion(outputs.squeeze(), batch_y)

                        val_loss += loss.item()

                val_loss /= len(val_loader)
                history["val_loss"].append(val_loss)

                if self.is_classification:
                    val_acc = correct / total
                    history["val_acc"].append(val_acc)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

            if (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}"
                if val_loader:
                    msg += f" - Val Loss: {val_loss:.4f}"
                    if self.is_classification:
                        msg += f" - Val Acc: {val_acc:.4f}"
                logger.info(msg)

        # Restore best model
        if best_state:
            self.model.load_state_dict(best_state)

        self.model.to("cpu")  # Move back to CPU for inference
        self._is_fitted = True

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        import torch

        self.model.eval()
        X_tensor = torch.FloatTensor(X)

        with torch.no_grad():
            outputs = self.model(X_tensor)

            if self.is_classification:
                _, predicted = torch.max(outputs, 1)
                return predicted.numpy()
            return outputs.squeeze().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        """Predict class probabilities."""
        if not self.is_classification:
            return None

        import torch
        import torch.nn.functional as F

        self.model.eval()
        X_tensor = torch.FloatTensor(X)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            probs = F.softmax(outputs, dim=1)
            return probs.numpy()

    def save(self, path: str | Path) -> None:
        """Save model state."""
        import torch

        torch.save(
            {
                "model_state": self.model.state_dict(),
                "n_features": self.n_features,
                "n_classes": self.n_classes,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        """Load model state."""
        import torch

        checkpoint = torch.load(path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model_state"])
        self._is_fitted = True


class LSTMModel(PyTorchBaseModel):
    """LSTM model for sequence classification/regression."""

    def __init__(
        self,
        n_features: int,
        n_classes: int | None = 2,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        sequence_length: int = 60,
        **kwargs,
    ):
        super().__init__(n_features, n_classes, hidden_size, num_layers, dropout, sequence_length)
        self._build_model()

    def _build_model(self):
        """Build LSTM architecture."""
        import torch.nn as nn

        class LSTMNetwork(nn.Module):
            def __init__(self, n_features, hidden_size, num_layers, dropout, output_size, is_classification):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=n_features,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0,
                    batch_first=True,
                )
                self.fc = nn.Linear(hidden_size, output_size)
                self.is_classification = is_classification

            def forward(self, x):
                # x shape: (batch, seq_len, features)
                lstm_out, _ = self.lstm(x)
                # Take last time step
                last_hidden = lstm_out[:, -1, :]
                output = self.fc(last_hidden)
                return output

        output_size = self.n_classes if self.is_classification else 1
        self.model = LSTMNetwork(
            self.n_features,
            self.hidden_size,
            self.num_layers,
            self.dropout,
            output_size,
            self.is_classification,
        )


class GRUModel(PyTorchBaseModel):
    """GRU model for sequence classification/regression."""

    def __init__(
        self,
        n_features: int,
        n_classes: int | None = 2,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        sequence_length: int = 60,
        **kwargs,
    ):
        super().__init__(n_features, n_classes, hidden_size, num_layers, dropout, sequence_length)
        self._build_model()

    def _build_model(self):
        """Build GRU architecture."""
        import torch.nn as nn

        class GRUNetwork(nn.Module):
            def __init__(self, n_features, hidden_size, num_layers, dropout, output_size):
                super().__init__()
                self.gru = nn.GRU(
                    input_size=n_features,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0,
                    batch_first=True,
                )
                self.fc = nn.Linear(hidden_size, output_size)

            def forward(self, x):
                gru_out, _ = self.gru(x)
                last_hidden = gru_out[:, -1, :]
                return self.fc(last_hidden)

        output_size = self.n_classes if self.is_classification else 1
        self.model = GRUNetwork(
            self.n_features,
            self.hidden_size,
            self.num_layers,
            self.dropout,
            output_size,
        )


class TransformerModel(PyTorchBaseModel):
    """Transformer model for sequence classification/regression."""

    def __init__(
        self,
        n_features: int,
        n_classes: int | None = 2,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        num_heads: int = 4,
        sequence_length: int = 60,
        **kwargs,
    ):
        super().__init__(n_features, n_classes, hidden_size, num_layers, dropout, sequence_length)
        self.num_heads = num_heads
        self._build_model()

    def _build_model(self):
        """Build Transformer architecture."""
        import torch
        import torch.nn as nn

        class PositionalEncoding(nn.Module):
            def __init__(self, d_model, max_len=5000):
                super().__init__()
                pe = torch.zeros(max_len, d_model)
                position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                pe = pe.unsqueeze(0)
                self.register_buffer("pe", pe)

            def forward(self, x):
                return x + self.pe[:, : x.size(1), :]

        class TransformerNetwork(nn.Module):
            def __init__(self, n_features, hidden_size, num_layers, num_heads, dropout, output_size):
                super().__init__()

                # Project input features to hidden size
                self.input_projection = nn.Linear(n_features, hidden_size)

                # Positional encoding
                self.pos_encoder = PositionalEncoding(hidden_size)

                # Transformer encoder
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=num_heads,
                    dim_feedforward=hidden_size * 4,
                    dropout=dropout,
                    batch_first=True,
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

                # Output layer
                self.fc = nn.Linear(hidden_size, output_size)
                self.dropout = nn.Dropout(dropout)

            def forward(self, x):
                # x shape: (batch, seq_len, features)
                x = self.input_projection(x)
                x = self.pos_encoder(x)
                x = self.transformer(x)

                # Global average pooling over sequence
                x = x.mean(dim=1)
                x = self.dropout(x)
                return self.fc(x)

        output_size = self.n_classes if self.is_classification else 1
        self.model = TransformerNetwork(
            self.n_features,
            self.hidden_size,
            self.num_layers,
            self.num_heads,
            self.dropout,
            output_size,
        )
