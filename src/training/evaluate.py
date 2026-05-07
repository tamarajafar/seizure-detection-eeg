"""
evaluate.py

LOSO-CV evaluation entry point for all four architectures.

Usage:
    python src/training/evaluate.py --arch logistic --config configs/default.yaml
    python src/training/evaluate.py --arch cnn_lstm --config configs/default.yaml
    python src/training/evaluate.py --arch cnn_lstm --subject_specific --config configs/default.yaml
    python src/training/evaluate.py --arch dann --config configs/default.yaml
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from src.models.cnn_lstm import CNNLSTM
from src.models.dann import DANN
from src.models.logistic_baseline import build_logistic_pipeline, extract_features_batch
from src.preprocessing.pipeline import compute_normalization_stats, apply_normalization
from src.training.loso_cv import load_subject_arrays
from src.training.train import train_cnn_lstm, train_dann, predict_proba
from src.utils.metrics import compute_metrics, aggregate_metrics, print_fold_summary


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    # Flatten nested config for convenience
    flat = {}
    for section in cfg.values():
        if isinstance(section, dict):
            flat.update(section)
    return flat


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Architecture 1: logistic regression baseline
# ---------------------------------------------------------------------------

def evaluate_logistic(processed_dir: Path, config: dict, results_dir: Path):
    print("\n=== Architecture 1: Logistic Regression Baseline ===\n")
    fold_metrics = []
    predictions = []

    sfreq = config.get("sample_rate", 256)

    # Discover subjects
    subject_files = sorted(processed_dir.glob("chb*.npz"))
    subject_ids = [f.stem for f in subject_files]
    print(f"Found {len(subject_ids)} subjects: {subject_ids}")

    # Extract features one subject at a time to avoid OOM
    print("Extracting band-power features per subject...")
    subject_feats = {}
    subject_labels = {}
    for sid in subject_ids:
        w, l = load_subject_arrays(processed_dir, sid)
        subject_feats[sid] = extract_features_batch(w, sfreq=sfreq)
        subject_labels[sid] = l
        del w  # free raw windows immediately
        print(f"  {sid}: {len(l)} windows -> {subject_feats[sid].shape[1]} features")

    # LOSO folds
    for test_sid in subject_ids:
        print(f"Fold: held-out = {test_sid}")

        train_feats = np.concatenate([subject_feats[s] for s in subject_ids if s != test_sid])
        train_labels = np.concatenate([subject_labels[s] for s in subject_ids if s != test_sid])

        pipeline = build_logistic_pipeline(
            C=config.get("C", 1.0),
            max_iter=config.get("max_iter", 1000),
        )
        pipeline.fit(train_feats, train_labels)
        y_prob = pipeline.predict_proba(subject_feats[test_sid])[:, 1]

        metrics = compute_metrics(subject_labels[test_sid], y_prob)
        fold_metrics.append(metrics)
        predictions.append({"y_true": subject_labels[test_sid], "y_prob": y_prob})
        print(f"  AUROC={metrics['auroc']:.3f}  Sens={metrics['sensitivity']:.3f}  n_ictal={metrics['n_ictal']}")

    print("\n--- LOSO Summary ---")
    print_fold_summary(fold_metrics, subject_ids)
    _save_results(fold_metrics, subject_ids, results_dir, "logistic", predictions=predictions)


# ---------------------------------------------------------------------------
# Architecture 2 and 3: CNN-LSTM
# ---------------------------------------------------------------------------

def _build_cnn_lstm(config):
    return CNNLSTM(
        n_channels=config.get("n_channels", 23),
        window_samples=config.get("window_samples", 1024),
        sfreq=config.get("sample_rate", 256),
        eegnet_F1=config.get("eegnet_f1", 8),
        eegnet_D=config.get("eegnet_d", 2),
        lstm_hidden=config.get("lstm_hidden", 128),
        lstm_layers=config.get("lstm_layers", 2),
        lstm_bidirectional=config.get("lstm_bidirectional", True),
    )


def evaluate_cnn_lstm_subject_specific(processed_dir: Path, config: dict, results_dir: Path):
    """Architecture 2: train and test on same subject with 90/10 split."""
    print("\n=== Architecture 2 (subject-specific) ===\n")

    device = get_device()
    print(f"Device: {device}")
    fold_metrics = []
    predictions = []

    subject_files = sorted(processed_dir.glob("chb*.npz"))
    subject_ids = [f.stem for f in subject_files]

    max_windows = config.get("max_windows_per_subject", 25_000)
    seed = config.get("seed", 42)

    for sid in subject_ids:
        print(f"Fold: subject = {sid}")
        windows, labels = load_subject_arrays(processed_dir, sid,
                                              max_windows=max_windows, seed=seed)
        print(f"  {len(labels)} windows loaded")

        # 90/10 train/val split
        rng = np.random.default_rng(seed)
        n = len(labels)
        idx = rng.permutation(n)
        split = int(0.9 * n)
        train_w, train_l = windows[idx[:split]], labels[idx[:split]]
        val_w, val_l = windows[idx[split:]], labels[idx[split:]]

        # Normalize
        mean, std = compute_normalization_stats(train_w)
        train_w = apply_normalization(train_w, mean, std)
        val_w = apply_normalization(val_w, mean, std)

        model = _build_cnn_lstm(config)
        torch.manual_seed(config.get("seed", 42))
        model = train_cnn_lstm(model, train_w, train_l, val_w, val_l, config, device)

        y_prob = predict_proba(model, val_w, device)
        metrics = compute_metrics(val_l, y_prob)
        fold_metrics.append(metrics)
        predictions.append({"y_true": val_l, "y_prob": y_prob})
        print(f"  AUROC={metrics['auroc']:.3f}  Sens={metrics['sensitivity']:.3f}  n_ictal={metrics['n_ictal']}")

        del windows, train_w, val_w  # free memory before next subject

    print("\n--- LOSO Summary: Architecture 2 (subject-specific) ---")
    print_fold_summary(fold_metrics, subject_ids)
    _save_results(fold_metrics, subject_ids, results_dir, "cnn_lstm_subject_specific", predictions=predictions)


def _compute_per_subject_cap(n_train_subjects: int, config: dict) -> int:
    """Cap per subject so total training data fits in ~30 GB RAM."""
    BYTES_PER_WINDOW = 23 * 1024 * 4  # ~94 KB per window (float32)
    MEM_BUDGET = 30 * (1024 ** 3)     # 30 GB
    budget_cap = MEM_BUDGET // (n_train_subjects * BYTES_PER_WINDOW)
    return min(config.get("max_windows_per_subject", 25_000), budget_cap)


def _load_and_concat(processed_dir: Path, subject_ids: list, max_win: int,
                     seed: int, mean: np.ndarray, std: np.ndarray,
                     include_subject_ids: bool = False):
    """Load subjects into a pre-allocated array to avoid 2x memory from concatenation."""
    # First pass: count total windows
    sizes = []
    for sid in subject_ids:
        data = np.load(processed_dir / f"{sid}.npz")
        n = min(len(data["labels"]), max_win)
        sizes.append(n)
        data.close()
    total = sum(sizes)

    # Pre-allocate output arrays
    train_w = np.empty((total, 23, 1024), dtype=np.float32)
    train_l = np.empty(total, dtype=np.int8)
    train_s = np.empty(total, dtype=np.int64) if include_subject_ids else None

    # Second pass: load and fill
    offset = 0
    for i, sid in enumerate(subject_ids):
        w, l = load_subject_arrays(processed_dir, sid, max_windows=max_win, seed=seed)
        w = apply_normalization(w, mean, std)
        n = len(l)
        train_w[offset:offset + n] = w
        train_l[offset:offset + n] = l
        if train_s is not None:
            train_s[offset:offset + n] = i
        offset += n
        del w, l

    if include_subject_ids:
        return train_w, train_l, train_s
    return train_w, train_l


def _load_subject_capped(processed_dir: Path, sid: str, max_windows: int, seed: int = 42):
    """Load subject arrays, subsampling at load time if over max_windows."""
    return load_subject_arrays(processed_dir, sid, max_windows=max_windows, seed=seed)


def _streaming_norm_stats(processed_dir: Path, subject_ids: list):
    """Compute per-channel mean/std across subjects without loading all at once."""
    n_channels = 23
    total_sum = np.zeros(n_channels)
    total_sq_sum = np.zeros(n_channels)
    total_count = 0

    for sid in subject_ids:
        w, _ = load_subject_arrays(processed_dir, sid)
        # w shape: (n_windows, n_channels, window_samples)
        n_samples = w.shape[0] * w.shape[2]
        total_sum += w.sum(axis=(0, 2))
        total_sq_sum += (w ** 2).sum(axis=(0, 2))
        total_count += n_samples
        del w

    mean = total_sum / total_count
    std = np.sqrt(total_sq_sum / total_count - mean ** 2)
    std[std < 1e-8] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def evaluate_cnn_lstm_cross_subject(processed_dir: Path, config: dict, results_dir: Path):
    """Architecture 3: cross-subject LOSO with CNN-LSTM."""
    print("\n=== Architecture 3 (cross-subject) ===\n")

    device = get_device()
    print(f"Device: {device}")

    subject_files = sorted(processed_dir.glob("chb*.npz"))
    subject_ids = [f.stem for f in subject_files]
    seed = config.get("seed", 42)

    tag = "cnn_lstm_cross_subject"
    fold_metrics, completed_sids = _load_checkpoint(results_dir, tag)
    predictions = []

    for test_idx, test_sid in enumerate(subject_ids):
        if test_sid in completed_sids:
            continue
        print(f"Fold: held-out = {test_sid}")

        train_sids = [s for s in subject_ids if s != test_sid]

        # Validation split: hold out ~10% of training subjects
        rng = np.random.default_rng(seed + test_idx)
        n_val = max(1, int(len(train_sids) * 0.1))
        val_sids = list(rng.choice(train_sids, size=n_val, replace=False))
        actual_train_sids = [s for s in train_sids if s not in val_sids]

        # Compute normalization from training subjects (streaming, no full load)
        print("  Computing normalization stats...")
        mean, std = _streaming_norm_stats(processed_dir, actual_train_sids)

        # Dynamic per-subject cap: keep total training data under ~30 GB
        max_win = _compute_per_subject_cap(len(actual_train_sids), config)
        print(f"  Loading training data ({len(actual_train_sids)} subjects, max {max_win}/subj)...")
        train_w, train_l = _load_and_concat(
            processed_dir, actual_train_sids, max_win, seed, mean, std)

        # Load and normalize validation data
        val_w, val_l = _load_and_concat(
            processed_dir, val_sids, max_win, seed, mean, std)

        # Load and normalize test data
        test_w, test_l = _load_subject_capped(processed_dir, test_sid, max_win, seed)
        test_w = apply_normalization(test_w, mean, std)

        model = _build_cnn_lstm(config)
        torch.manual_seed(seed)
        model = train_cnn_lstm(model, train_w, train_l, val_w, val_l, config, device)
        del train_w, train_l, val_w, val_l

        y_prob = predict_proba(model, test_w, device)
        metrics = compute_metrics(test_l, y_prob)
        fold_metrics.append(metrics)
        predictions.append({"y_true": test_l, "y_prob": y_prob})
        completed_sids.append(test_sid)
        print(f"  AUROC={metrics['auroc']:.3f}  Sens={metrics['sensitivity']:.3f}  n_ictal={metrics['n_ictal']}")
        del test_w, test_l, model
        _save_checkpoint(fold_metrics, completed_sids, results_dir, tag)

    print("\n--- LOSO Summary: Architecture 3 (cross-subject) ---")
    print_fold_summary(fold_metrics, subject_ids)
    _save_results(fold_metrics, subject_ids, results_dir, tag, predictions=predictions)


# ---------------------------------------------------------------------------
# Architecture 4: DANN
# ---------------------------------------------------------------------------

def evaluate_dann(processed_dir: Path, config: dict, results_dir: Path):
    print("\n=== Architecture 4: DANN ===\n")
    device = get_device()
    print(f"Device: {device}")

    subject_files = sorted(processed_dir.glob("chb*.npz"))
    subject_ids = [f.stem for f in subject_files]
    seed = config.get("seed", 42)

    tag = "dann"
    fold_metrics, completed_sids = _load_checkpoint(results_dir, tag)
    predictions = []

    for test_idx, test_sid in enumerate(subject_ids):
        if test_sid in completed_sids:
            continue
        train_sids = [s for s in subject_ids if s != test_sid]

        # Validation split
        rng = np.random.default_rng(seed + test_idx)
        n_val = max(1, int(len(train_sids) * 0.1))
        val_sids = list(rng.choice(train_sids, size=n_val, replace=False))
        actual_train_sids = [s for s in train_sids if s not in val_sids]

        n_train_subjects = len(actual_train_sids)
        print(f"Fold: held-out = {test_sid}  |  n_train_subjects = {n_train_subjects}")

        # Compute normalization from training subjects
        mean, std = _streaming_norm_stats(processed_dir, actual_train_sids)

        # Load training data with subject IDs (capped per subject)
        max_win = _compute_per_subject_cap(len(actual_train_sids), config)
        train_w, train_l, train_s = _load_and_concat(
            processed_dir, actual_train_sids, max_win, seed, mean, std,
            include_subject_ids=True)

        # Load validation data
        val_w, val_l = _load_and_concat(
            processed_dir, val_sids, max_win, seed, mean, std)

        # Load test data
        test_w, test_l = _load_subject_capped(processed_dir, test_sid, max_win, seed)
        test_w = apply_normalization(test_w, mean, std)

        model = DANN(
            n_channels=config.get("n_channels", 23),
            window_samples=config.get("window_samples", 1024),
            sfreq=config.get("sample_rate", 256),
            eegnet_F1=config.get("eegnet_f1", 8),
            eegnet_D=config.get("eegnet_d", 2),
            n_subjects=n_train_subjects,
            lambda_max=config.get("dann_lambda_max", 1.0),
        )

        torch.manual_seed(seed)
        model = train_dann(
            model,
            train_w, train_l, train_s,
            val_w, val_l,
            config, device,
        )
        del train_w, train_l, train_s, val_w, val_l

        y_prob = predict_proba(model, test_w, device, is_dann=True)
        metrics = compute_metrics(test_l, y_prob)
        fold_metrics.append(metrics)
        predictions.append({"y_true": test_l, "y_prob": y_prob})
        completed_sids.append(test_sid)
        print(f"  AUROC={metrics['auroc']:.3f}  Sens={metrics['sensitivity']:.3f}  n_ictal={metrics['n_ictal']}")
        del test_w, test_l, model
        _save_checkpoint(fold_metrics, completed_sids, results_dir, tag)

    print("\n--- LOSO Summary: DANN ---")
    print_fold_summary(fold_metrics, subject_ids)
    _save_results(fold_metrics, subject_ids, results_dir, tag, predictions=predictions)


# ---------------------------------------------------------------------------
# Results persistence
# ---------------------------------------------------------------------------

def _save_checkpoint(fold_metrics: list, completed_sids: list, results_dir: Path, tag: str):
    """Save intermediate fold results so jobs can resume after timeout."""
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {"completed": completed_sids, "fold_metrics": fold_metrics}
    ckpt_path = results_dir / f"{tag}_checkpoint.json"
    with open(ckpt_path, "w") as f:
        json.dump(ckpt, f)
    print(f"  Checkpoint saved ({len(completed_sids)} folds done)")


def _load_checkpoint(results_dir: Path, tag: str):
    """Load checkpoint if it exists. Returns (fold_metrics, completed_sids) or ([], [])."""
    ckpt_path = results_dir / f"{tag}_checkpoint.json"
    if ckpt_path.exists():
        with open(ckpt_path) as f:
            ckpt = json.load(f)
        print(f"  Resuming from checkpoint: {len(ckpt['completed'])} folds already done")
        return ckpt["fold_metrics"], ckpt["completed"]
    return [], []


def _save_results(fold_metrics: list, subject_ids: list, results_dir: Path, tag: str,
                  predictions: list | None = None):
    results_dir.mkdir(parents=True, exist_ok=True)
    agg = aggregate_metrics(fold_metrics)
    output = {
        "aggregate": agg,
        "per_fold": [{"subject": sid, **m} for sid, m in zip(subject_ids, fold_metrics)],
    }
    out_path = results_dir / f"{tag}_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    if predictions is not None:
        pred_path = results_dir / f"{tag}_predictions.npz"
        np.savez(
            pred_path,
            **{f"{sid}_y_true": p["y_true"] for sid, p in zip(subject_ids, predictions)},
            **{f"{sid}_y_prob": p["y_prob"] for sid, p in zip(subject_ids, predictions)},
        )
        print(f"Predictions saved to {pred_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LOSO-CV evaluation for seizure detection")
    parser.add_argument("--arch", choices=["logistic", "cnn_lstm", "dann"], required=True)
    parser.add_argument("--subject_specific", action="store_true",
                        help="Train and test on same subject (Architecture 2). Only valid with --arch cnn_lstm.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--processed_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--results_dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    config = load_config(args.config)
    torch.manual_seed(config.get("seed", 42))
    np.random.seed(config.get("seed", 42))

    t0 = time.time()
    if args.arch == "logistic":
        evaluate_logistic(args.processed_dir, config, args.results_dir)
    elif args.arch == "cnn_lstm":
        if args.subject_specific:
            evaluate_cnn_lstm_subject_specific(args.processed_dir, config, args.results_dir)
        else:
            evaluate_cnn_lstm_cross_subject(args.processed_dir, config, args.results_dir)
    elif args.arch == "dann":
        evaluate_dann(args.processed_dir, config, args.results_dir)

    print(f"\nTotal time: {(time.time() - t0) / 60:.1f} min")
