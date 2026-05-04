"""
pipeline.py

Full preprocessing pipeline: bandpass filter, 4-second segmentation,
seizure labeling, and z-score normalization. Writes processed arrays to disk.

Run as a script:
    python src/preprocessing/pipeline.py --data_dir data/raw --out_dir data/processed
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt
from tqdm import tqdm

from src.preprocessing.load_edf import load_all_subjects, load_edf, SubjectData


# ---------------------------------------------------------------------------
# Signal processing
# ---------------------------------------------------------------------------

def bandpass_filter(signal: np.ndarray, low: float, high: float, sfreq: float) -> np.ndarray:
    """
    Apply a zero-phase 4th-order Butterworth bandpass filter.

    Parameters
    ----------
    signal : np.ndarray, shape (n_channels, n_samples)
    low : float
        Lower cutoff frequency in Hz.
    high : float
        Upper cutoff frequency in Hz.
    sfreq : float
        Sampling frequency in Hz.

    Returns
    -------
    filtered : np.ndarray, same shape as signal
    """
    nyq = sfreq / 2.0
    sos = butter(4, [low / nyq, high / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, signal, axis=1)


# ---------------------------------------------------------------------------
# Segmentation and labeling
# ---------------------------------------------------------------------------

def segment_and_label(
    signal: np.ndarray,
    sfreq: float,
    seizure_intervals: list,
    window_sec: float = 4.0,
    overlap_thresh: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Segment a continuous EEG recording into non-overlapping windows and assign
    binary seizure labels.

    A window is labeled 1 (ictal) if the fraction of samples overlapping any
    annotated seizure interval exceeds overlap_thresh.

    Parameters
    ----------
    signal : np.ndarray, shape (n_channels, n_samples)
    sfreq : float
    seizure_intervals : list of (onset_sec, offset_sec) tuples
    window_sec : float
    overlap_thresh : float
        Minimum fraction of window samples that must overlap a seizure.

    Returns
    -------
    windows : np.ndarray, shape (n_windows, n_channels, window_samples)
    labels : np.ndarray, shape (n_windows,), dtype int8
    """
    window_samples = int(window_sec * sfreq)
    n_samples = signal.shape[1]
    n_windows = n_samples // window_samples

    windows = np.zeros((n_windows, signal.shape[0], window_samples), dtype=np.float32)
    labels = np.zeros(n_windows, dtype=np.int8)

    for i in range(n_windows):
        start = i * window_samples
        end = start + window_samples
        windows[i] = signal[:, start:end]

        # Compute seizure overlap for this window
        window_start_sec = start / sfreq
        window_end_sec = end / sfreq

        for onset_sec, offset_sec in seizure_intervals:
            overlap_start = max(window_start_sec, onset_sec)
            overlap_end = min(window_end_sec, offset_sec)
            if overlap_end > overlap_start:
                overlap_frac = (overlap_end - overlap_start) / window_sec
                if overlap_frac >= overlap_thresh:
                    labels[i] = 1
                    break

    return windows, labels


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def compute_normalization_stats(windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-channel mean and std from a set of windows for z-score normalization.

    Statistics are computed over all windows and all time samples, per channel.

    Parameters
    ----------
    windows : np.ndarray, shape (n_windows, n_channels, window_samples)

    Returns
    -------
    mean : np.ndarray, shape (n_channels,)
    std : np.ndarray, shape (n_channels,)
    """
    flat = windows.reshape(-1, windows.shape[2])  # (n_windows * n_channels, window_samples)
    # Reshape to (n_channels, n_windows * window_samples) for per-channel stats
    n_channels = windows.shape[1]
    reshaped = windows.transpose(1, 0, 2).reshape(n_channels, -1)
    mean = reshaped.mean(axis=1)
    std = reshaped.std(axis=1)
    std[std < 1e-8] = 1.0  # avoid division by zero for flat channels
    return mean, std


def apply_normalization(windows: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Z-score normalize windows using precomputed per-channel statistics.

    Parameters
    ----------
    windows : np.ndarray, shape (n_windows, n_channels, window_samples)
    mean : np.ndarray, shape (n_channels,)
    std : np.ndarray, shape (n_channels,)

    Returns
    -------
    normalized : np.ndarray, same shape as windows
    """
    return (windows - mean[np.newaxis, :, np.newaxis]) / std[np.newaxis, :, np.newaxis]


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def process_subject(
    subject_data: SubjectData,
    bandpass_low: float = 0.5,
    bandpass_high: float = 40.0,
    window_sec: float = 4.0,
    overlap_thresh: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the full preprocessing pipeline for one subject.

    Loads all EDF recordings, filters, segments, and labels them. Does NOT
    apply normalization (normalization stats must come from training subjects
    only and be applied separately).

    Parameters
    ----------
    subject_data : SubjectData
    bandpass_low : float
    bandpass_high : float
    window_sec : float
    overlap_thresh : float

    Returns
    -------
    all_windows : np.ndarray, shape (n_windows, n_channels, window_samples)
    all_labels : np.ndarray, shape (n_windows,)
    """
    all_windows = []
    all_labels = []

    for record in subject_data.records:
        signal, sfreq, _ = load_edf(record.edf_path)
        filtered = bandpass_filter(signal, bandpass_low, bandpass_high, sfreq)
        windows, labels = segment_and_label(
            filtered, sfreq, record.seizure_intervals,
            window_sec=window_sec, overlap_thresh=overlap_thresh
        )
        all_windows.append(windows)
        all_labels.append(labels)

    return np.concatenate(all_windows, axis=0), np.concatenate(all_labels, axis=0)


def run_pipeline(data_dir: Path, out_dir: Path, config: dict | None = None,
                 subjects: list[str] | None = None):
    """
    Preprocess CHB-MIT subjects and save segmented arrays to disk.

    Parameters
    ----------
    data_dir : Path
        Root of raw CHB-MIT data (contains chb01/, chb02/, ...).
    out_dir : Path
        Directory where processed arrays are written.
    config : dict, optional
        Override default preprocessing parameters. Keys: bandpass_low,
        bandpass_high, window_sec, overlap_thresh.
    subjects : list of str, optional
        Process only these subject IDs (e.g. ["chb01", "chb02"]).
        Default: process all subjects found in data_dir.
    """
    cfg = {
        "bandpass_low": 0.5,
        "bandpass_high": 40.0,
        "window_sec": 4.0,
        "overlap_thresh": 0.5,
    }
    if config:
        cfg.update(config)

    out_dir.mkdir(parents=True, exist_ok=True)
    all_subjects = load_all_subjects(data_dir)
    if subjects:
        keep = set(subjects)
        all_subjects = [s for s in all_subjects if s.subject_id in keep]

    for subject_data in tqdm(all_subjects, desc="Preprocessing subjects"):
        sid = subject_data.subject_id
        print(f"\n{sid}: {len(subject_data.records)} recordings, {subject_data.n_seizures} seizures")

        windows, labels = process_subject(subject_data, **cfg)
        ictal = labels.sum()
        print(f"  {len(labels)} windows total | {ictal} ictal | {len(labels) - ictal} interictal")

        np.savez_compressed(out_dir / f"{sid}.npz", windows=windows, labels=labels)

    print(f"\nPreprocessed data saved to {out_dir}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess CHB-MIT EEG data")
    parser.add_argument("--data_dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--subjects", nargs="+", default=None,
                        help="Process only these subjects (e.g. chb01 chb02). Default: all.")
    args = parser.parse_args()

    run_pipeline(args.data_dir, args.out_dir, subjects=args.subjects)
