"""
ModelTrainer — trains LSTMAutoencoder on normal (non-fraud) transactions.

Design rationale:
    The autoencoder is trained exclusively on non-fraud sequences so it
    learns the manifold of "normal" behaviour.  At inference time, fraudulent
    transactions produce high reconstruction error because they fall outside
    that learned manifold.

MLflow integration:
    Every training run is logged to the local mlruns/ directory.
    Params: hidden_dim, latent_dim, seq_len, epochs, lr, threshold_percentile
    Metrics: train_loss (per epoch)
    Tags: threshold value after scoring
"""

import logging
from typing import Optional

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.lstm_ae import LSTMAutoencoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Prepares data, trains LSTMAutoencoder, scores sequences, and sets
    the anomaly threshold — all wired to MLflow.

    Parameters
    ----------
    feature_cols : list[str]
        Column names to use as model input (everything except IDs / label).
    device : str
        'cuda', 'mps', or 'cpu'.  Auto-detected if None.
    """

    def __init__(self, feature_cols: list[str], device: Optional[str] = None):
        self.feature_cols = feature_cols

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)
        self._scaler = None   # fitted during first prepare_sequences call
        logger.info("ModelTrainer using device: %s", self.device)

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def prepare_sequences(
        self,
        df: pd.DataFrame,
        seq_len: int = 10,
        fraud_col: str = "isFraud",
        train_on_normal: bool = True,
    ) -> DataLoader:
        """
        Convert the flat feature matrix into sliding-window sequences
        for LSTM input, optionally filtering to non-fraud rows only.

        Sequences are formed row-by-row with a sliding window of `seq_len`.
        Each sample is shape (seq_len, n_features).

        Parameters
        ----------
        df : pd.DataFrame
            Feature matrix from FeatureEngineer.build_feature_matrix().
        seq_len : int
            Number of consecutive timesteps per sequence.
        fraud_col : str
            Column name for the fraud label (used to filter normal rows).
        train_on_normal : bool
            If True, train only on non-fraud rows (unsupervised normal model).

        Returns
        -------
        DataLoader
        """
        data = df.copy()

        if train_on_normal and fraud_col in data.columns:
            n_before = len(data)
            data = data[data[fraud_col] == 0]
            logger.info(
                "Filtered to normal transactions: %d → %d rows", n_before, len(data)
            )

        # Drop non-numeric / identifier columns
        drop_cols = [c for c in ["TransactionID", "isFraud", "TransactionDT"] if c in data.columns]
        data = data.drop(columns=drop_cols)

        # Keep only requested feature cols that are present
        cols = [c for c in self.feature_cols if c in data.columns]
        X = data[cols].astype(np.float32).values  # (N, n_features)

        # Replace any residual NaN/Inf before scaling
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Standardise features — critical to prevent NaN loss in LSTM
        # Fit scaler on this split; store it so score() can reuse it
        if train_on_normal or self._scaler is None:
            from sklearn.preprocessing import StandardScaler
            self._scaler = StandardScaler()
            X = self._scaler.fit_transform(X)
            logger.info("StandardScaler fitted and applied.")
        else:
            X = self._scaler.transform(X)
            logger.info("Existing StandardScaler applied.")

        # Sliding window
        sequences = np.stack(
            [X[i : i + seq_len] for i in range(len(X) - seq_len + 1)],
            axis=0,
        )  # (N - seq_len + 1, seq_len, n_features)

        logger.info(
            "Sequences shape: %s  (seq_len=%d, n_features=%d)",
            sequences.shape, seq_len, sequences.shape[2],
        )

        tensor = torch.tensor(sequences, dtype=torch.float32)
        dataset = TensorDataset(tensor)
        # pin_memory is unsupported on MPS; use it only for CUDA
        pin = self.device.type == "cuda"
        return DataLoader(dataset, batch_size=512, shuffle=True, pin_memory=pin)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        model: LSTMAutoencoder,
        dataloader: DataLoader,
        epochs: int = 20,
        lr: float = 1e-3,
        run_name: str = "lstm_autoencoder",
    ) -> LSTMAutoencoder:
        """
        Train the LSTM autoencoder with Adam + MSE loss.

        All hyperparameters and per-epoch losses are logged to MLflow.

        Parameters
        ----------
        model : LSTMAutoencoder
        dataloader : DataLoader   Output of prepare_sequences().
        epochs : int
        lr : float
        run_name : str            MLflow run display name.

        Returns
        -------
        model : LSTMAutoencoder   Trained model (on self.device).
        """
        model = model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        with mlflow.start_run(run_name=run_name):
            # Log hyperparameters
            mlflow.log_params({
                "hidden_dim":          model.hidden_dim,
                "latent_dim":          model.latent_dim,
                "seq_len":             model.seq_len,
                "num_layers":          model.num_layers,
                "dropout":             model.dropout,
                "epochs":              epochs,
                "lr":                  lr,
                "optimizer":           "Adam",
                "loss":                "MSELoss",
            })

            for epoch in range(1, epochs + 1):
                model.train()
                epoch_loss = 0.0
                n_batches = 0

                for (batch,) in dataloader:
                    batch = batch.to(self.device)
                    optimizer.zero_grad()
                    recon = model(batch)
                    loss = criterion(recon, batch)
                    loss.backward()
                    # Gradient clipping — important for LSTM stability
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    epoch_loss += loss.item()
                    n_batches += 1

                avg_loss = epoch_loss / max(n_batches, 1)
                mlflow.log_metric("train_loss", avg_loss, step=epoch)

                if epoch % max(1, epochs // 10) == 0 or epoch == epochs:
                    logger.info("Epoch %3d / %d  |  loss: %.6f", epoch, epochs, avg_loss)

            logger.info("Training complete.")

        return model

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(
        self,
        model: LSTMAutoencoder,
        dataloader: DataLoader,
    ) -> np.ndarray:
        """
        Compute reconstruction errors for all samples in dataloader.

        Parameters
        ----------
        model : LSTMAutoencoder
        dataloader : DataLoader

        Returns
        -------
        errors : np.ndarray  shape (N,)  per-sample MSE.
        """
        model.eval()
        model = model.to(self.device)
        all_errors = []

        with torch.no_grad():
            for (batch,) in dataloader:
                batch = batch.to(self.device)
                errors = model.reconstruction_error(batch)
                all_errors.append(errors)

        return np.concatenate(all_errors)

    # ------------------------------------------------------------------
    # Threshold
    # ------------------------------------------------------------------

    def find_threshold(
        self,
        errors: np.ndarray,
        percentile: float = 95.0,
        run_name: str = "lstm_threshold",
    ) -> float:
        """
        Set the anomaly threshold at the given percentile of training errors.

        Anything above this value at inference time is flagged as anomalous.
        The threshold and score distribution stats are logged to MLflow.

        Parameters
        ----------
        errors : np.ndarray   Reconstruction errors from training data.
        percentile : float    Default 95 → flag top 5% as anomalous.
        run_name : str

        Returns
        -------
        threshold : float
        """
        threshold = float(np.percentile(errors, percentile))

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({
                "threshold_percentile": percentile,
                "threshold":            threshold,
            })
            mlflow.log_metrics({
                "error_mean":   float(errors.mean()),
                "error_std":    float(errors.std()),
                "error_min":    float(errors.min()),
                "error_max":    float(errors.max()),
                "error_p50":    float(np.percentile(errors, 50)),
                "error_p95":    float(np.percentile(errors, 95)),
                "error_p99":    float(np.percentile(errors, 99)),
            })

        logger.info(
            "Threshold set at %.4f  (p%.0f of training errors)", threshold, percentile
        )
        return threshold
