"""
spectrogram.py

Convert raw EEG windows to log-power spectrograms for CNN input.
Produces tensors of shape (n_channels, n_freq_bins, n_time_bins).
"""

import numpy as np
from scipy.signal import stft as scipy_stft


def compute_spectrogram(
    window: np.ndarray,
    sfreq: float = 256.0,
    stft_window: int = 256,
    stft_hop: int = 128,
    freq_max: float = 40.0,
) -> np.ndarray:
    """
    Compute a log-power spectrogram for a single multichannel EEG window.

    Uses a short-time Fourier transform with a Hann window. Frequencies above
    freq_max are discarded (matching the bandpass filter cutoff). Log power is
    computed as log(|STFT|^2 + epsilon) to avoid log(0).

    Parameters
    ----------
    window : np.ndarray, shape (n_channels, window_samples)
        A single preprocessed EEG window.
    sfreq : float
        Sampling frequency in Hz.
    stft_window : int
        STFT window length in samples.
    stft_hop : int
        STFT hop length in samples.
    freq_max : float
        Maximum frequency to retain (Hz). Bins above this are dropped.

    Returns
    -------
    spectrogram : np.ndarray, shape (n_channels, n_freq_bins, n_time_bins)
        Log-power spectrogram per channel.
    """
    epsilon = 1e-10
    n_channels = window.shape[0]
    spectrograms = []

    for ch in range(n_channels):
        freqs, times, Zxx = scipy_stft(
            window[ch],
            fs=sfreq,
            window="hann",
            nperseg=stft_window,
            noverlap=stft_window - stft_hop,
        )
        freq_mask = freqs <= freq_max
        log_power = np.log(np.abs(Zxx[freq_mask]) ** 2 + epsilon)
        spectrograms.append(log_power)

    return np.stack(spectrograms, axis=0)


def batch_spectrograms(
    windows: np.ndarray,
    sfreq: float = 256.0,
    stft_window: int = 256,
    stft_hop: int = 128,
    freq_max: float = 40.0,
) -> np.ndarray:
    """
    Compute log-power spectrograms for a batch of EEG windows.

    Parameters
    ----------
    windows : np.ndarray, shape (n_windows, n_channels, window_samples)
    sfreq : float
    stft_window : int
    stft_hop : int
    freq_max : float

    Returns
    -------
    specs : np.ndarray, shape (n_windows, n_channels, n_freq_bins, n_time_bins)
    """
    specs = [
        compute_spectrogram(windows[i], sfreq, stft_window, stft_hop, freq_max)
        for i in range(len(windows))
    ]
    return np.stack(specs, axis=0)


def spectrogram_shape(
    window_samples: int = 1024,
    sfreq: float = 256.0,
    stft_window: int = 256,
    stft_hop: int = 128,
    freq_max: float = 40.0,
    n_channels: int = 23,
) -> tuple[int, int, int]:
    """
    Return the (n_channels, n_freq_bins, n_time_bins) shape without computing
    a real spectrogram -- useful for model initialization.

    Parameters
    ----------
    window_samples : int
    sfreq : float
    stft_window : int
    stft_hop : int
    freq_max : float
    n_channels : int

    Returns
    -------
    tuple of (n_channels, n_freq_bins, n_time_bins)
    """
    dummy = np.zeros((n_channels, window_samples))
    spec = compute_spectrogram(dummy, sfreq, stft_window, stft_hop, freq_max)
    return spec.shape
