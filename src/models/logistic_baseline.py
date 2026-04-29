"""
logistic_baseline.py

Architecture 1: logistic regression on hand-crafted band-power features.

For each 4-second EEG window, computes band power in 5 clinically established
frequency bands across all 23 channels, producing a 115-dimensional feature
vector. A logistic regression classifier with balanced class weights is fit on
training windows and evaluated under LOSO-CV.
"""

import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

FREQ_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 40.0),
}


def band_power(psd: np.ndarray, freqs: np.ndarray, fmin: float, fmax: float) -> float:
    """
    Compute mean power spectral density within a frequency band using the
    trapezoidal rule.

    Parameters
    ----------
    psd : np.ndarray, shape (n_freqs,)
    freqs : np.ndarray, shape (n_freqs,)
    fmin, fmax : float

    Returns
    -------
    float
    """
    mask = (freqs >= fmin) & (freqs <= fmax)
    return np.trapz(psd[mask], freqs[mask])


def extract_features(window: np.ndarray, sfreq: float = 256.0) -> np.ndarray:
    """
    Extract a 115-dimensional band-power feature vector from a single window.

    For each of 23 channels, computes power in 5 frequency bands via Welch's
    method. Features are concatenated in channel-major order:
    [ch0_delta, ch0_theta, ..., ch0_gamma, ch1_delta, ..., ch22_gamma].

    Parameters
    ----------
    window : np.ndarray, shape (n_channels, window_samples)
    sfreq : float

    Returns
    -------
    features : np.ndarray, shape (n_channels * n_bands,) = (115,)
    """
    n_channels = window.shape[0]
    features = []

    for ch in range(n_channels):
        freqs, psd = welch(window[ch], fs=sfreq, nperseg=min(256, window.shape[1]))
        for band_name, (fmin, fmax) in FREQ_BANDS.items():
            features.append(band_power(psd, freqs, fmin, fmax))

    return np.array(features, dtype=np.float32)


def extract_features_batch(windows: np.ndarray, sfreq: float = 256.0) -> np.ndarray:
    """
    Extract band-power features for a batch of windows.

    Parameters
    ----------
    windows : np.ndarray, shape (n_windows, n_channels, window_samples)
    sfreq : float

    Returns
    -------
    features : np.ndarray, shape (n_windows, 115)
    """
    return np.stack([extract_features(w, sfreq) for w in windows], axis=0)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_logistic_pipeline(C: float = 1.0, max_iter: int = 1000) -> Pipeline:
    """
    Build a sklearn Pipeline: StandardScaler -> LogisticRegression.

    StandardScaler is included here because normalization statistics are
    computed from the training features only and applied to validation/test,
    consistent with the neural network preprocessing protocol.

    Parameters
    ----------
    C : float
        Inverse regularization strength.
    max_iter : int

    Returns
    -------
    sklearn Pipeline
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight="balanced",
            solver="lbfgs",
            n_jobs=-1,
            random_state=42,
        )),
    ])
