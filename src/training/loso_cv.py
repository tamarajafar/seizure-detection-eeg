"""
loso_cv.py

Leave-one-subject-out cross-validation fold generator and dataset utilities.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class EEGWindowDataset(Dataset):
    """
    PyTorch Dataset wrapping preprocessed EEG windows and binary labels.

    Parameters
    ----------
    windows : np.ndarray, shape (n_windows, n_channels, window_samples)
    labels : np.ndarray, shape (n_windows,)
    subject_ids : np.ndarray, shape (n_windows,)
        Integer subject index for each window. Used by DANN.
    transform : callable, optional
        Applied to each window tensor before returning.
    """

    def __init__(
        self,
        windows: np.ndarray,
        labels: np.ndarray,
        subject_ids: np.ndarray | None = None,
        transform=None,
    ):
        self.windows = torch.from_numpy(windows).float()
        self.labels = torch.from_numpy(labels.astype(np.int64))
        self.subject_ids = (
            torch.from_numpy(subject_ids.astype(np.int64))
            if subject_ids is not None
            else torch.zeros(len(labels), dtype=torch.int64)
        )
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        window = self.windows[idx]
        if self.transform is not None:
            window = self.transform(window)
        return window, self.labels[idx], self.subject_ids[idx]


# ---------------------------------------------------------------------------
# Weighted sampler for class imbalance
# ---------------------------------------------------------------------------

def make_weighted_sampler(labels: np.ndarray, interictal_to_ictal_ratio: int = 10) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler that draws ictal and interictal samples at
    a controlled ratio without oversampling the minority class to full parity.

    Parameters
    ----------
    labels : np.ndarray, shape (n_windows,)
    interictal_to_ictal_ratio : int
        Number of interictal samples drawn per ictal sample on average.

    Returns
    -------
    WeightedRandomSampler
    """
    n_ictal = labels.sum()
    n_interictal = len(labels) - n_ictal

    # Weight each class inversely proportional to target ratio
    weight_ictal = 1.0
    weight_interictal = 1.0 / interictal_to_ictal_ratio

    weights = np.where(labels == 1, weight_ictal, weight_interictal).astype(np.float32)
    return WeightedRandomSampler(
        weights=torch.from_numpy(weights),
        num_samples=len(weights),
        replacement=True,
    )


# ---------------------------------------------------------------------------
# LOSO fold generator
# ---------------------------------------------------------------------------

def load_subject_arrays(processed_dir: Path, subject_id: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load preprocessed windows and labels for one subject from disk.

    Parameters
    ----------
    processed_dir : Path
    subject_id : str, e.g. "chb01"

    Returns
    -------
    windows : np.ndarray, shape (n_windows, n_channels, window_samples)
    labels : np.ndarray, shape (n_windows,)
    """
    data = np.load(processed_dir / f"{subject_id}.npz")
    return data["windows"], data["labels"]


def loso_folds(processed_dir: Path, val_fraction: float = 0.1, seed: int = 42):
    """
    Generator that yields one LOSO fold at a time.

    For each fold, one subject is held out as the test set. From the remaining
    subjects, val_fraction (by subject count) are held out as validation for
    early stopping and hyperparameter selection.

    Parameters
    ----------
    processed_dir : Path
        Directory containing chbXX.npz files.
    val_fraction : float
        Fraction of training subjects to use for validation.
    seed : int
        Random seed for validation split.

    Yields
    ------
    dict with keys:
        test_subject : str
        train_windows, train_labels, train_subject_ids : np.ndarray
        val_windows, val_labels, val_subject_ids : np.ndarray
        test_windows, test_labels : np.ndarray
        subject_id_map : dict mapping subject_id str to integer index
    """
    rng = np.random.default_rng(seed)

    subject_files = sorted(processed_dir.glob("chb*.npz"))
    subject_ids = [f.stem for f in subject_files]

    for test_idx, test_subject in enumerate(subject_ids):
        train_subjects = [s for s in subject_ids if s != test_subject]
        rng_local = np.random.default_rng(seed + test_idx)
        n_val = max(1, int(len(train_subjects) * val_fraction))
        val_subjects = list(rng_local.choice(train_subjects, size=n_val, replace=False))
        actual_train_subjects = [s for s in train_subjects if s not in val_subjects]

        # Integer ID map (used for DANN subject-identity classifier)
        id_map = {s: i for i, s in enumerate(train_subjects)}

        def _stack(subjects):
            all_w, all_l, all_s = [], [], []
            for s in subjects:
                w, l = load_subject_arrays(processed_dir, s)
                all_w.append(w)
                all_l.append(l)
                all_s.append(np.full(len(l), id_map.get(s, -1), dtype=np.int64))
            return np.concatenate(all_w), np.concatenate(all_l), np.concatenate(all_s)

        train_w, train_l, train_s = _stack(actual_train_subjects)
        val_w, val_l, val_s = _stack(val_subjects)
        test_w, test_l = load_subject_arrays(processed_dir, test_subject)

        yield {
            "test_subject": test_subject,
            "train_windows": train_w,
            "train_labels": train_l,
            "train_subject_ids": train_s,
            "val_windows": val_w,
            "val_labels": val_l,
            "val_subject_ids": val_s,
            "test_windows": test_w,
            "test_labels": test_l,
            "subject_id_map": id_map,
            "n_train_subjects": len(actual_train_subjects),
        }
