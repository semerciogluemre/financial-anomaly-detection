"""
MLPAutoencoder — feed-forward autoencoder for tabular anomaly detection.

Architecture:
  Encoder: input_dim → 128 → 64 → 32  (ReLU + BatchNorm + Dropout)
  Decoder: 32 → 64 → 128 → input_dim  (mirrors encoder)

Trained only on non-fraud rows. Anomaly score = per-sample MSE.
Includes ReduceLROnPlateau scheduler for stable convergence.
"""

import logging
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class _MLP(nn.Module):
    """Internal encoder-decoder network."""

    def __init__(self, input_dim: int, dropout: float = 0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),        nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32),         nn.BatchNorm1d(32),  nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),         nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 128),        nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class MLPAutoencoder:
    """
    Tabular MLP autoencoder for unsupervised fraud anomaly detection.

    Parameters
    ----------
    input_dim   : int    Number of input features.
    dropout     : float  Dropout rate in encoder and decoder.
    device      : str    'mps', 'cuda', or 'cpu'. Auto-detected if None.
    """

    DROP_COLS = {"TransactionID", "isFraud", "TransactionDT"}

    def __init__(
        self,
        input_dim: int | None = None,
        dropout: float = 0.2,
        device: str | None = None,
    ):
        self.input_dim = input_dim
        self.dropout   = dropout
        self._net: _MLP | None = None
        self._scaler   = None

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)
        logger.info("MLPAutoencoder using device: %s", self.device)

    # ------------------------------------------------------------------
    def _to_array(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.drop(columns=[c for c in self.DROP_COLS if c in X.columns])
            X = X.select_dtypes(include="number")
        arr = np.array(X, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr

    # ------------------------------------------------------------------
    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        epochs: int = 50,
        lr: float = 0.001,
        batch_size: int = 512,
        run_name: str = "mlp_autoencoder_fit",
    ) -> "MLPAutoencoder":
        """Train on non-fraud rows only."""
        from sklearn.preprocessing import StandardScaler

        X_arr = self._to_array(X_train)
        y_arr = np.array(y_train, dtype=np.int32)

        # Filter to normal rows
        normal_mask = y_arr == 0
        X_normal    = X_arr[normal_mask]
        logger.info(
            "Training on %d normal rows (filtered from %d total)",
            len(X_normal), len(X_arr),
        )

        # Scale
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_normal)

        if self.input_dim is None:
            self.input_dim = X_scaled.shape[1]

        self._net = _MLP(self.input_dim, self.dropout).to(self.device)
        optimizer = torch.optim.Adam(self._net.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                      patience=5, min_lr=1e-5)
        criterion = nn.MSELoss()

        pin = self.device.type == "cuda"
        tensor  = torch.tensor(X_scaled, dtype=torch.float32)
        dataset = TensorDataset(tensor)
        loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=pin)

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({
                "model": "MLPAutoencoder", "input_dim": self.input_dim,
                "dropout": self.dropout, "epochs": epochs, "lr": lr,
                "batch_size": batch_size, "n_train_normal": len(X_normal),
            })

            for epoch in range(1, epochs + 1):
                self._net.train()
                epoch_loss = 0.0
                for (batch,) in loader:
                    batch = batch.to(self.device)
                    optimizer.zero_grad()
                    recon = self._net(batch)
                    loss  = criterion(recon, batch)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                avg = epoch_loss / max(len(loader), 1)
                scheduler.step(avg)

                if epoch % 10 == 0 or epoch == epochs:
                    lr_now = optimizer.param_groups[0]["lr"]
                    logger.info("Epoch %3d/%d  loss=%.6f  lr=%.2e", epoch, epochs, avg, lr_now)
                    mlflow.log_metric("train_loss", avg, step=epoch)

        logger.info("MLP-AE training complete.")
        return self

    # ------------------------------------------------------------------
    def reconstruction_error(
        self,
        X: pd.DataFrame | np.ndarray,
    ) -> np.ndarray:
        """Per-sample MSE anomaly score (higher = more anomalous)."""
        if self._net is None:
            raise RuntimeError("Not fitted. Call fit() first.")

        X_arr    = self._to_array(X)
        X_scaled = self._scaler.transform(X_arr)

        self._net.eval()
        all_errors = []
        batch_size = 1024

        with torch.no_grad():
            for i in range(0, len(X_scaled), batch_size):
                batch = torch.tensor(
                    X_scaled[i : i + batch_size], dtype=torch.float32
                ).to(self.device)
                recon  = self._net(batch)
                errors = ((batch - recon) ** 2).mean(dim=1).cpu().numpy()
                all_errors.append(errors)

        return np.concatenate(all_errors)

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        import pickle
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self._net.state_dict(),
            "input_dim":  self.input_dim,
            "dropout":    self.dropout,
            "scaler":     self._scaler,
        }, path)
        logger.info("MLPAutoencoder saved → %s", path)

    def load(self, path: str) -> "MLPAutoencoder":
        ckpt = torch.load(path, map_location=self.device)
        self.input_dim = ckpt["input_dim"]
        self.dropout   = ckpt["dropout"]
        self._scaler   = ckpt["scaler"]
        self._net      = _MLP(self.input_dim, self.dropout).to(self.device)
        self._net.load_state_dict(ckpt["state_dict"])
        self._net.eval()
        return self
