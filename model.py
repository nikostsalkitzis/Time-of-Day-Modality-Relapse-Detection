"""
The LSTM sequence autoencoder used for per-patient relapse detection, plus the
per-window reconstruction-error helpers. Architecture only: no data loading, no
training loop, no thresholding. train.py and the evaluation scripts import the
same reconstruction_error, so the anomaly score is computed identically
everywhere.

Shape contract:
    input  x : (batch, T, F)   where T = WINDOW_BINS, F = number of features
    output   : (batch, T, F)   reconstruction of the same shape

The two scoring helpers are consistent by construction:
    reconstruction_error              -> (batch,)    mean over time and features
    reconstruction_error_per_feature  -> (batch, F)  mean over time only
The mean over the feature axis of the per-feature version equals the scalar
version, so the per-feature error is an exact additive decomposition of the score
the pipeline already uses.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    """
    Encoder LSTM -> compressed latent vector -> decoder LSTM -> reconstruction.

    The encoder's final-layer hidden state is projected to a small bottleneck.
    That latent vector is repeated across all T timesteps and fed to the decoder
    LSTM, whose outputs are mapped back to the feature dimension.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 64,
                 latent_dim: int = 16,
                 num_layers: int = 1,
                 seq_len: int = 24,
                 dropout: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.seq_len = seq_len

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.enc_to_latent = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dec_to_out = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.encoder(x)          # h_n: (num_layers, batch, hidden)
        latent = self.enc_to_latent(h_n[-1])   # (batch, latent_dim)

        dec_in = latent.unsqueeze(1).repeat(1, self.seq_len, 1)  # (batch, T, latent)
        dec_out, _ = self.decoder(dec_in)       # (batch, T, hidden)
        x_hat = self.dec_to_out(dec_out)        # (batch, T, F)
        return x_hat


def reconstruction_error(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """
    Per-window mean-squared reconstruction error, averaged over time and
    feature dimensions. Returns a 1-D tensor of length `batch`.
    """
    return ((x - x_hat) ** 2).mean(dim=(1, 2))


def reconstruction_error_per_feature(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """
    Per-window, per-feature mean-squared reconstruction error, averaged over the
    time dimension only. Returns a 2-D tensor of shape (batch, F).

    The scalar score reopened along the feature axis. Because
    reconstruction_error_per_feature(x, x_hat).mean(dim=1) equals
    reconstruction_error(x, x_hat), partitioning these per-feature terms into
    physiological groups partitions the scalar score itself. Used only by the
    anomaly-signature pass, so no headline metric depends on it.
    """
    return ((x - x_hat) ** 2).mean(dim=1)
