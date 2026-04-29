"""
eegnet.py

EEGNet feature extractor (Lawhern et al. 2018).

EEGNet is a compact depthwise separable CNN designed for EEG classification.
It captures temporal frequency content and spatial (cross-channel) filters in
a small number of parameters, making it well suited as a shared feature
extractor for both the cross-subject CNN-LSTM and the DANN.

Reference:
    Lawhern et al. "EEGNet: a compact convolutional neural network for
    EEG-based brain-computer interfaces." J. Neural Eng. 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNet(nn.Module):
    """
    EEGNet feature extractor.

    Takes raw EEG windows of shape (batch, n_channels, window_samples) and
    produces a fixed-dimensional embedding vector per window.

    Parameters
    ----------
    n_channels : int
        Number of EEG channels (23 for CHB-MIT).
    window_samples : int
        Samples per window (1024 for 4s at 256 Hz).
    sfreq : float
        Sampling frequency. Used to set the temporal filter length (sfreq/2).
    F1 : int
        Number of temporal filters in the first conv layer.
    D : int
        Depthwise multiplier (number of spatial filters per temporal filter).
    F2 : int
        Number of pointwise filters in the separable conv layer. If None,
        defaults to F1 * D.
    dropout : float
        Dropout probability applied after each major block.
    embed_dim : int
        Output embedding dimension after the final linear projection.
        If None, the raw flattened EEGNet output is returned without projection.
    """

    def __init__(
        self,
        n_channels: int = 23,
        window_samples: int = 1024,
        sfreq: float = 256.0,
        F1: int = 8,
        D: int = 2,
        F2: int | None = None,
        dropout: float = 0.5,
        embed_dim: int = 128,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.window_samples = window_samples
        F2 = F2 or F1 * D

        # Block 1: temporal convolution + depthwise spatial convolution
        temporal_kernel = int(sfreq / 2)  # 128 samples = 0.5s receptive field
        self.block1 = nn.Sequential(
            # Temporal filter: shape-preserving convolution across time
            nn.Conv2d(1, F1, kernel_size=(1, temporal_kernel), padding=(0, temporal_kernel // 2), bias=False),
            nn.BatchNorm2d(F1),
            # Depthwise spatial filter: one filter per channel per F1 temporal filter
            nn.Conv2d(F1, F1 * D, kernel_size=(n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )

        # Block 2: depthwise separable convolution across time
        sep_kernel = 16
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, sep_kernel), padding=(0, sep_kernel // 2), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
        )

        # Compute flattened size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, window_samples)
            out = self.block2(self.block1(dummy))
            flat_size = out.view(1, -1).shape[1]

        self.embed_dim = embed_dim
        if embed_dim is not None:
            self.projection = nn.Linear(flat_size, embed_dim)
        else:
            self.projection = None
            self.embed_dim = flat_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (batch, n_channels, window_samples)

        Returns
        -------
        embedding : torch.Tensor, shape (batch, embed_dim)
        """
        # Add channel dimension for Conv2d: (batch, 1, n_channels, window_samples)
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.size(0), -1)
        if self.projection is not None:
            x = self.projection(x)
        return x
