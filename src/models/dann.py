"""
dann.py

Architecture 4: Domain Adversarial Neural Network (DANN).

The EEGNet feature extractor is shared between a seizure classifier (the task
head) and a subject-identity classifier (the domain head). A gradient reversal
layer (GRL) sits between the feature extractor and the domain head: during the
backward pass, it negates the gradient, forcing the extractor to produce
representations that are maximally uninformative about subject identity while
remaining discriminative for seizure detection.

Total loss:
    L = L_seizure - lambda * L_subject

where lambda is annealed from 0 to lambda_max over training following the
schedule in Ganin et al. 2016 (arXiv:1505.07818).

Reference:
    Ganin et al. "Domain-Adversarial Training of Neural Networks." JMLR 2016.
"""

import torch
import torch.nn as nn
from torch.autograd import Function

from src.models.eegnet import EEGNet


# ---------------------------------------------------------------------------
# Gradient reversal layer
# ---------------------------------------------------------------------------

class GradientReversalFunction(Function):
    """
    Reverses the sign of the gradient during the backward pass, scaled by lambda.

    In the forward pass, acts as the identity. In the backward pass, multiplies
    the upstream gradient by -lambda before passing it to the feature extractor.
    This encourages the extractor to produce domain-invariant representations.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.save_for_backward(torch.tensor(lambda_))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (lambda_,) = ctx.saved_tensors
        return -lambda_.item() * grad_output, None


class GradientReversalLayer(nn.Module):
    """Wraps GradientReversalFunction as a standard nn.Module."""

    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def set_lambda(self, lambda_: float):
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_)


# ---------------------------------------------------------------------------
# DANN
# ---------------------------------------------------------------------------

class DANN(nn.Module):
    """
    Domain Adversarial Neural Network for cross-subject seizure detection.

    Parameters
    ----------
    n_channels : int
    window_samples : int
    sfreq : float
    eegnet_F1 : int
    eegnet_D : int
    eegnet_embed_dim : int
    n_seizure_classes : int
        2 (binary: ictal vs. interictal).
    n_subjects : int
        Number of training subjects (22 for one LOSO fold). The domain
        classifier predicts which training subject a window came from.
    domain_hidden : int
        Hidden layer size for the domain classifier MLP.
    lambda_max : float
        Maximum adversarial strength (ceiling of the annealing schedule).
    """

    def __init__(
        self,
        n_channels: int = 23,
        window_samples: int = 1024,
        sfreq: float = 256.0,
        eegnet_F1: int = 8,
        eegnet_D: int = 2,
        eegnet_embed_dim: int = 128,
        n_seizure_classes: int = 2,
        n_subjects: int = 22,
        domain_hidden: int = 64,
        lambda_max: float = 1.0,
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

        # Task head: seizure vs. non-seizure
        self.seizure_classifier = nn.Sequential(
            nn.Linear(eegnet_embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_seizure_classes),
        )

        # Domain head: which training subject?
        self.grl = GradientReversalLayer(lambda_=0.0)  # starts at 0, annealed during training
        self.domain_classifier = nn.Sequential(
            nn.Linear(eegnet_embed_dim, domain_hidden),
            nn.ReLU(),
            nn.Linear(domain_hidden, n_subjects),
        )

    def set_lambda(self, lambda_: float):
        """Update the GRL adversarial strength. Called by the training loop."""
        self.grl.set_lambda(lambda_)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (batch, n_channels, window_samples)

        Returns
        -------
        seizure_logits : torch.Tensor, shape (batch, n_seizure_classes)
        domain_logits : torch.Tensor, shape (batch, n_subjects)
        """
        features = self.feature_extractor(x)
        seizure_logits = self.seizure_classifier(features)
        domain_logits = self.domain_classifier(self.grl(features))
        return seizure_logits, domain_logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return seizure logits only (inference mode, no domain head needed).

        Parameters
        ----------
        x : torch.Tensor, shape (batch, n_channels, window_samples)

        Returns
        -------
        seizure_logits : torch.Tensor, shape (batch, n_seizure_classes)
        """
        features = self.feature_extractor(x)
        return self.seizure_classifier(features)

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Return feature embeddings for visualization (e.g. t-SNE)."""
        return self.feature_extractor(x)


# ---------------------------------------------------------------------------
# Lambda annealing schedule (Ganin et al. 2016)
# ---------------------------------------------------------------------------

def compute_lambda(current_step: int, total_steps: int, lambda_max: float = 1.0) -> float:
    """
    Compute the current adversarial strength lambda using the sigmoid schedule
    from Ganin et al. 2016:

        lambda = lambda_max * (2 / (1 + exp(-10 * p)) - 1)

    where p = current_step / total_steps in [0, 1].

    Parameters
    ----------
    current_step : int
        Current training step (batch iteration across all epochs).
    total_steps : int
        Total training steps (n_epochs * n_batches_per_epoch).
    lambda_max : float

    Returns
    -------
    float
    """
    import math
    p = current_step / max(total_steps, 1)
    return lambda_max * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)
