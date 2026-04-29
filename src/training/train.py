"""
train.py

Unified training loop for CNN-LSTM and DANN. Handles:
    - Weighted random sampling for class imbalance
    - Early stopping on validation AUROC
    - DANN lambda annealing schedule
    - Checkpoint saving for best model weights

Usage:
    Called programmatically by evaluate.py, not run directly.
"""

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from src.models.dann import compute_lambda
from src.training.loso_cv import EEGWindowDataset, make_weighted_sampler


# ---------------------------------------------------------------------------
# CNN-LSTM training
# ---------------------------------------------------------------------------

def train_cnn_lstm(
    model: nn.Module,
    train_windows: np.ndarray,
    train_labels: np.ndarray,
    val_windows: np.ndarray,
    val_labels: np.ndarray,
    config: dict,
    device: torch.device,
) -> nn.Module:
    """
    Train the CNN-LSTM model with early stopping on validation AUROC.

    Parameters
    ----------
    model : nn.Module (CNNLSTM instance)
    train_windows : np.ndarray, shape (n_train, n_channels, window_samples)
    train_labels : np.ndarray, shape (n_train,)
    val_windows : np.ndarray
    val_labels : np.ndarray
    config : dict
        Training hyperparameters from configs/default.yaml.
    device : torch.device

    Returns
    -------
    nn.Module
        Model with best validation AUROC weights loaded.
    """
    train_dataset = EEGWindowDataset(train_windows, train_labels)
    val_dataset = EEGWindowDataset(val_windows, val_labels)

    sampler = make_weighted_sampler(
        train_labels,
        interictal_to_ictal_ratio=config.get("interictal_to_ictal_ratio", 10),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get("batch_size", 64),
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.get("learning_rate", 1e-3),
        weight_decay=config.get("weight_decay", 1e-4),
    )
    criterion = nn.CrossEntropyLoss()
    patience = config.get("early_stopping_patience", 7)

    model.to(device)
    best_auroc = -1.0
    best_weights = None
    epochs_without_improvement = 0

    for epoch in range(config.get("max_epochs", 50)):
        model.train()
        for windows, labels, _ in train_loader:
            windows, labels = windows.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(windows)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        val_auroc = _evaluate_auroc(model, val_loader, device)
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            best_weights = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_weights)
    return model


# ---------------------------------------------------------------------------
# DANN training
# ---------------------------------------------------------------------------

def train_dann(
    model: nn.Module,
    train_windows: np.ndarray,
    train_labels: np.ndarray,
    train_subject_ids: np.ndarray,
    val_windows: np.ndarray,
    val_labels: np.ndarray,
    config: dict,
    device: torch.device,
) -> nn.Module:
    """
    Train the DANN model with gradient reversal lambda annealing.

    The domain classifier loss is only computed on training windows (not
    validation or test). Subject ID labels are the integer training-subject
    index, as assigned by the LOSO fold generator.

    Parameters
    ----------
    model : nn.Module (DANN instance)
    train_windows : np.ndarray
    train_labels : np.ndarray
    train_subject_ids : np.ndarray
        Integer subject index for each training window.
    val_windows : np.ndarray
    val_labels : np.ndarray
    config : dict
    device : torch.device

    Returns
    -------
    nn.Module with best validation AUROC weights.
    """
    train_dataset = EEGWindowDataset(train_windows, train_labels, subject_ids=train_subject_ids)
    val_dataset = EEGWindowDataset(val_windows, val_labels)

    sampler = make_weighted_sampler(
        train_labels,
        interictal_to_ictal_ratio=config.get("interictal_to_ictal_ratio", 10),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get("batch_size", 64),
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.get("learning_rate", 1e-3),
        weight_decay=config.get("weight_decay", 1e-4),
    )
    seizure_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss()

    max_epochs = config.get("max_epochs", 50)
    total_steps = max_epochs * len(train_loader)
    lambda_max = config.get("dann_lambda_max", 1.0)
    patience = config.get("early_stopping_patience", 7)

    model.to(device)
    best_auroc = -1.0
    best_weights = None
    epochs_without_improvement = 0
    global_step = 0

    for epoch in range(max_epochs):
        model.train()
        for windows, labels, subject_ids in train_loader:
            windows = windows.to(device)
            labels = labels.to(device)
            subject_ids = subject_ids.to(device)

            lam = compute_lambda(global_step, total_steps, lambda_max)
            model.set_lambda(lam)
            global_step += 1

            optimizer.zero_grad()
            seizure_logits, domain_logits = model(windows)
            loss_seizure = seizure_criterion(seizure_logits, labels)
            loss_domain = domain_criterion(domain_logits, subject_ids)
            loss = loss_seizure - lam * loss_domain
            loss.backward()
            optimizer.step()

        val_auroc = _evaluate_auroc_dann(model, val_loader, device)
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            best_weights = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_weights)
    return model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _evaluate_auroc(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Compute AUROC on a DataLoader using CNN-LSTM (single head)."""
    model.eval()
    all_probs, all_labels = [], []
    for windows, labels, _ in loader:
        windows = windows.to(device)
        logits = model(windows)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())
    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)
    if y_true.sum() == 0:
        return 0.5
    return roc_auc_score(y_true, y_prob)


@torch.no_grad()
def _evaluate_auroc_dann(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Compute AUROC on a DataLoader using DANN (returns seizure logits only)."""
    model.eval()
    all_probs, all_labels = [], []
    for windows, labels, _ in loader:
        windows = windows.to(device)
        logits = model.predict(windows)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())
    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)
    if y_true.sum() == 0:
        return 0.5
    return roc_auc_score(y_true, y_prob)


@torch.no_grad()
def predict_proba(model: nn.Module, windows: np.ndarray, device: torch.device, batch_size: int = 256, is_dann: bool = False) -> np.ndarray:
    """
    Run inference and return ictal class probabilities.

    Parameters
    ----------
    model : nn.Module
    windows : np.ndarray, shape (n_windows, n_channels, window_samples)
    device : torch.device
    batch_size : int
    is_dann : bool
        If True, calls model.predict() instead of model().

    Returns
    -------
    probs : np.ndarray, shape (n_windows,)
    """
    model.eval()
    dataset = EEGWindowDataset(windows, np.zeros(len(windows), dtype=np.int8))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_probs = []
    for batch_windows, _, _ in loader:
        batch_windows = batch_windows.to(device)
        logits = model.predict(batch_windows) if is_dann else model(batch_windows)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        all_probs.append(probs)
    return np.concatenate(all_probs)
