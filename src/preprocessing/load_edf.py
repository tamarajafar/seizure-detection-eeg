"""
load_edf.py

Parse CHB-MIT .edf files and their companion summary files to extract raw EEG
signals and ground-truth seizure intervals.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import mne
import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SeizureRecord:
    """One EDF recording file with its annotated seizure intervals."""
    subject_id: str          # e.g. "chb01"
    edf_path: Path
    seizure_intervals: list  # list of (onset_sec, offset_sec) tuples


@dataclass
class SubjectData:
    """All recordings for one subject."""
    subject_id: str
    records: list = field(default_factory=list)

    @property
    def n_seizures(self):
        return sum(len(r.seizure_intervals) for r in self.records)


# ---------------------------------------------------------------------------
# Summary file parsing
# ---------------------------------------------------------------------------

def parse_summary(summary_path: Path) -> dict:
    """
    Parse a CHB-MIT subject summary file (e.g. chb01-summary.txt) and return
    a mapping from EDF filename to list of (onset_sec, offset_sec) tuples.

    Parameters
    ----------
    summary_path : Path
        Path to the *-summary.txt file for one subject.

    Returns
    -------
    dict
        Keys are EDF filenames (str, no directory), values are lists of
        (onset_sec, offset_sec) tuples. Files with no seizures map to [].
    """
    text = summary_path.read_text()
    records = {}

    # Split on "File Name:" to get one block per recording
    blocks = re.split(r"File Name:", text)[1:]

    for block in blocks:
        lines = block.strip().splitlines()
        filename = lines[0].strip()
        records[filename] = []

        # Find all seizure onset/offset pairs
        onsets = re.findall(r"Seizure(?:\s+\d+)?\s+Start\s+Time\s*:\s*(\d+)\s*seconds", block)
        offsets = re.findall(r"Seizure(?:\s+\d+)?\s+End\s+Time\s*:\s*(\d+)\s*seconds", block)

        for onset, offset in zip(onsets, offsets):
            records[filename].append((int(onset), int(offset)))

    return records


# ---------------------------------------------------------------------------
# EDF loading
# ---------------------------------------------------------------------------

STANDARD_CHANNELS = [
    "FP1-F7", "F7-T7",  "T7-P7",  "P7-O1",
    "FP1-F3", "F3-C3",  "C3-P3",  "P3-O1",
    "FP2-F4", "F4-C4",  "C4-P4",  "P4-O2",
    "FP2-F8", "F8-T8",  "T8-P8",  "P8-O2",
    "FZ-CZ",  "CZ-PZ",
    "P7-T7",  "T7-FT9", "FT9-FT10","FT10-T8",
    "T8-P8-0",
]


def load_edf(edf_path: Path, target_channels: list | None = None) -> tuple[np.ndarray, float, list]:
    """
    Load a single CHB-MIT EDF file and return the raw EEG array.

    The CHB-MIT dataset uses inconsistent channel naming across subjects.
    This function loads whatever channels are present, optionally reindexing
    to a target set and filling missing channels with zeros.

    Parameters
    ----------
    edf_path : Path
        Path to the .edf file.
    target_channels : list of str, optional
        If provided, reorder/subset channels to match this list. Missing
        channels are filled with zeros and a warning is issued.

    Returns
    -------
    signal : np.ndarray, shape (n_channels, n_samples)
        Raw EEG signal in microvolts.
    sfreq : float
        Sampling frequency in Hz.
    ch_names : list of str
        Channel names as they appear in (or were mapped to) the file.
    """
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    sfreq = raw.info["sfreq"]
    ch_names = raw.ch_names

    # Strip whitespace from channel names (CHB-MIT has trailing spaces)
    ch_names = [c.strip() for c in ch_names]
    raw.rename_channels({old: new for old, new in zip(raw.ch_names, ch_names)})

    # Drop non-EEG channels (e.g. ECG, annotations)
    eeg_channels = [c for c in ch_names if not any(x in c.upper() for x in ["ECG", "VNS", "EKG", "--"])]
    raw.pick_channels(eeg_channels)
    ch_names = raw.ch_names
    signal = raw.get_data() * 1e6  # convert V to uV

    if target_channels is not None:
        signal, ch_names = _align_channels(signal, ch_names, target_channels)

    return signal, sfreq, ch_names


def _align_channels(signal: np.ndarray, present: list, target: list) -> tuple[np.ndarray, list]:
    """
    Reorder signal rows to match target channel list, filling absent channels with zeros.

    Parameters
    ----------
    signal : np.ndarray, shape (n_present, n_samples)
    present : list of str
    target : list of str

    Returns
    -------
    aligned : np.ndarray, shape (len(target), n_samples)
    target : list of str
    """
    n_samples = signal.shape[1]
    present_upper = [c.upper() for c in present]
    aligned = np.zeros((len(target), n_samples), dtype=signal.dtype)

    for i, ch in enumerate(target):
        key = ch.upper()
        if key in present_upper:
            idx = present_upper.index(key)
            aligned[i] = signal[idx]
        else:
            import warnings
            warnings.warn(f"Channel '{ch}' not found in {present}; filling with zeros.")

    return aligned, target


# ---------------------------------------------------------------------------
# Subject loader
# ---------------------------------------------------------------------------

def load_subject(subject_dir: Path, target_channels: list | None = None) -> SubjectData:
    """
    Load all recordings for one CHB-MIT subject.

    Parameters
    ----------
    subject_dir : Path
        Directory for one subject, e.g. data/raw/chb01/.
    target_channels : list of str, optional
        Passed through to load_edf for channel alignment.

    Returns
    -------
    SubjectData
    """
    subject_id = subject_dir.name
    summary_files = list(subject_dir.glob("*-summary.txt"))
    if not summary_files:
        raise FileNotFoundError(f"No summary file found in {subject_dir}")

    seizure_map = parse_summary(summary_files[0])
    edf_files = sorted(subject_dir.glob("*.edf"))

    records = []
    for edf_path in edf_files:
        intervals = seizure_map.get(edf_path.name, [])
        records.append(SeizureRecord(
            subject_id=subject_id,
            edf_path=edf_path,
            seizure_intervals=intervals,
        ))

    return SubjectData(subject_id=subject_id, records=records)


def load_all_subjects(data_raw_dir: Path, target_channels: list | None = None) -> list[SubjectData]:
    """
    Load metadata for all subjects in the CHB-MIT dataset directory.

    Parameters
    ----------
    data_raw_dir : Path
        Root of the raw data directory (contains chb01/, chb02/, etc.).
        Supports both flat structure (chb01/, chb02/, ...) and nested
        PhysioNet structure (physionet.org/files/chbmit/1.0.0/chb01/, ...).
    target_channels : list of str, optional

    Returns
    -------
    list of SubjectData, sorted by subject_id
    """
    # Try to find chb* directories at root level
    subject_dirs = sorted([d for d in data_raw_dir.iterdir() if d.is_dir() and d.name.startswith("chb")])
    
    # If not found, search recursively (for nested PhysioNet structure)
    if not subject_dirs:
        import warnings
        warnings.warn(f"No chb* directories found at root of {data_raw_dir}. Searching recursively...")
        subject_dirs = sorted([d for d in data_raw_dir.rglob("chb*") if d.is_dir() and d.name.startswith("chb")])
        # Filter to only direct children of their parent (avoid duplicates)
        if subject_dirs:
            # Group by parent, keep only one level
            parent_groups = {}
            for d in subject_dirs:
                parent = d.parent
                if parent not in parent_groups:
                    parent_groups[parent] = []
                parent_groups[parent].append(d)
            # Use the deepest/most specific level
            subject_dirs = parent_groups[max(parent_groups.keys(), key=lambda p: len(p.parts))]
            subject_dirs = sorted(subject_dirs)
    
    subjects = []
    for d in subject_dirs:
        try:
            subjects.append(load_subject(d, target_channels=target_channels))
        except FileNotFoundError as e:
            import warnings
            warnings.warn(str(e))
    return subjects
