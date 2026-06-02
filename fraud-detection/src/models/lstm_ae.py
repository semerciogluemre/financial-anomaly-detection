"""
LSTM Autoencoder for unsupervised financial anomaly detection.

Architecture:
    Input (seq_len, batch, input_dim)
        → Encoder: 2-layer LSTM  → hidden state
        → Bottleneck: linear projection to latent_dim
        → Decoder: 2-layer LSTM  → per-step hidden states
        → Output: linear projection back to input_dim

Anomaly score = per-sample mean squared reconstruction error.
High error → the sequence deviates from learned "normal" behaviour.
"""

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class LSTMEncoder(nn.Module):
    """Two-layer LSTM encoder that compresses input sequences to a latent vector."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Compress last hidden state to latent space
        self.bottleneck = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: Tensor):
        """
        Parameters
        ----------
        x : Tensor  (batch, seq_len, input_dim)

        Returns
        -------
        latent : Tensor  (batch, latent_dim)
        hidden : tuple   raw LSTM hidden state (for decoder init)
        """
        _, (hidden, cell) = self.lstm(x)  # hidden: (num_layers, batch, hidden_dim)
        # Take the last layer's hidden state
        last_hidden = hidden[-1]           # (batch, hidden_dim)
        latent = self.bottleneck(last_hidden)  # (batch, latent_dim)
        return latent, (hidden, cell)


class LSTMDecoder(nn.Module):
    """Two-layer LSTM decoder that reconstructs the full sequence from a latent vector."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        output_dim: int,
        seq_len: int,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Expand latent back to hidden_dim for LSTM input
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, latent: Tensor) -> Tensor:
        """
        Parameters
        ----------
        latent : Tensor  (batch, latent_dim)

        Returns
        -------
        reconstruction : Tensor  (batch, seq_len, output_dim)
        """
        batch_size = latent.size(0)

        # Expand latent → (batch, hidden_dim), then repeat across seq_len
        h = self.latent_to_hidden(latent)          # (batch, hidden_dim)
        h = h.unsqueeze(1).repeat(1, self.seq_len, 1)  # (batch, seq_len, hidden_dim)

        lstm_out, _ = self.lstm(h)                 # (batch, seq_len, hidden_dim)
        reconstruction = self.output_layer(lstm_out)   # (batch, seq_len, output_dim)
        return reconstruction


class LSTMAutoencoder(nn.Module):
    """
    Full LSTM Autoencoder for sequence-level anomaly detection.

    Parameters
    ----------
    input_dim  : int   Number of features per timestep.
    hidden_dim : int   LSTM hidden size in encoder and decoder.
    latent_dim : int   Bottleneck dimensionality.
    seq_len    : int   Sequence length (number of timesteps).
    num_layers : int   Number of LSTM layers in encoder and decoder (default 2).
    dropout    : float Dropout between LSTM layers (default 0.2).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        seq_len: int,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.dropout = dropout

        self.encoder = LSTMEncoder(input_dim, hidden_dim, latent_dim, num_layers, dropout)
        self.decoder = LSTMDecoder(latent_dim, hidden_dim, input_dim, seq_len, num_layers, dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Encode then decode a batch of sequences.

        Parameters
        ----------
        x : Tensor  (batch, seq_len, input_dim)

        Returns
        -------
        reconstruction : Tensor  (batch, seq_len, input_dim)
        """
        latent, _ = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction

    def reconstruction_error(self, x: Tensor) -> np.ndarray:
        """
        Compute per-sample mean squared reconstruction error.

        This is the anomaly score: high error → anomalous sequence.

        Parameters
        ----------
        x : Tensor  (batch, seq_len, input_dim)  — can be on any device.

        Returns
        -------
        errors : np.ndarray  shape (batch,)  — MSE per sample, on CPU.
        """
        self.eval()
        with torch.no_grad():
            recon = self.forward(x)
            # MSE averaged over (seq_len, input_dim) per sample
            mse = ((x - recon) ** 2).mean(dim=(1, 2))  # (batch,)
        return mse.cpu().numpy()

    def encode(self, x: Tensor) -> Tensor:
        """Return the latent representation of x (useful for visualisation)."""
        self.eval()
        with torch.no_grad():
            latent, _ = self.encoder(x)
        return latent
