"""
cnn_lstm.py

Architectures 2 and 3: EEGNet + bidirectional LSTM + linear classifier.

Used in two evaluation modes:
    - Subject-specific (Architecture 2): trained and tested on the same subject.
      Represents the performance ceiling when labeled data from the test subject
      is available.
    - Naive cross-subject (Architecture 3): trained on 22 subjects, tested on
      the held-out subject with no domain adaptation. Establishes the lower bound
      for cross-subject generalization.

The architecture is identical in both modes; only the data split changes.
"""

import torch
import torch.nn as nn

from src.models.eegnet import EEGNet


class CNNLSTM(nn.Module):
    """
    EEGNet feature extractor followed by a bidirectional LSTM for seizure detection.

    Processes a sequence of consecutive EEG windows to capture temporal context
    beyond the single 4-second window. The LSTM hidden state at the final time
    step is passed to a linear classification head.

    Parameters
    ----------
    n_channels : int
        EEG channels (23 for CHB-MIT).
    window_samples : int
        Samples per 4s window (1024).
    sfreq : float
    eegnet_F1 : int
        EEGNet temporal filter count.
    eegnet_D : int
        EEGNet depthwise multiplier.
    eegnet_embed_dim : int
        EEGNet output embedding dimension.
    lstm_hidden : int
        LSTM hidden size per direction.
    lstm_layers : int
    lstm_bidirectional : bool
    lstm_dropout : float
        Dropout between LSTM layers (only applied when lstm_layers > 1).
    n_classes : int
        2 for binary seizure/non-seizure.
    """

    def __init__(
        self,
        n_channels: int = 23,
        window_samples: int = 1024,
        sfreq: float = 256.0,
        eegnet_F1: int = 8,
        eegnet_D: int = 2,
        eegnet_embed_dim: int = 128,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        lstm_bidirectional: bool = True,
        lstm_dropout: float = 0.3,
        n_classes: int = 2,
    ):
        super().__init__()

        self.feature_extractor = EEGNet(
            n_channels=n_channels,
            window_samples=window_samples,
            sfreq=sfreq,
            F1=eegnet_F1,
            D=eegnet_D,
            embed_dim=eegnet_embed_dim,
        )

        self.lstm = nn.LSTM(
            input_size=eegnet_embed_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=lstm_bidirectional,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
        )

        lstm_out_dim = lstm_hidden * (2 if lstm_bidirectional else 1)
        self.classifier = nn.Linear(lstm_out_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Either shape (batch, n_channels, window_samples) for single-window
            classification, or (batch, seq_len, n_channels, window_samples) for
            sequence classification.

        Returns
        -------
        logits : torch.Tensor, shape (batch, n_classes)
        """
        if x.dim() == 3:
            # Single window: treat as sequence of length 1
            x = x.unsqueeze(1)

        batch, seq_len, n_ch, n_samp = x.shape

        # Apply EEGNet to each window in the sequence
        x_flat = x.view(batch * seq_len, n_ch, n_samp)
        embeddings = self.feature_extractor(x_flat)             # (batch*seq_len, embed_dim)
        embeddings = embeddings.view(batch, seq_len, -1)         # (batch, seq_len, embed_dim)

        lstm_out, _ = self.lstm(embeddings)                      # (batch, seq_len, lstm_out_dim)
        last_hidden = lstm_out[:, -1, :]                         # take final time step
        logits = self.classifier(last_hidden)
        return logits

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return EEGNet embeddings without the LSTM or classifier head.
        Used for feature visualization and DANN analysis.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, n_channels, window_samples)

        Returns
        -------
        embeddings : torch.Tensor, shape (batch, embed_dim)
        """
        return self.feature_extractor(x)
