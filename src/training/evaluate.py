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
from src.training.loso_cv import loso_folds, load_subject_arrays
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
        print(f"  AUROC={metrics['auroc']:.3f}  Sens={metrics['sensitivity']:.3f}  n_ictal={metrics['n_ictal']}")

    print("\n--- LOSO Summary ---")
    print_fold_summary(fold_metrics, subject_ids)
    _save_results(fold_metrics, subject_ids, results_dir, "logistic")


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

    subject_files = sorted(processed_dir.glob("chb*.npz"))
    subject_ids = [f.stem for f in subject_files]

    for sid in subject_ids:
        print(f"Fold: subject = {sid}")
        windows, labels = load_subject_arrays(processed_dir, sid)

        # 90/10 train/val split
        rng = np.random.default_rng(config.get("seed", 42))
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
        print(f"  AUROC={metrics['auroc']:.3f}  Sens={metrics['sensitivity']:.3f}  n_ictal={metrics['n_ictal']}")

        del windows, train_w, val_w  # free memory before next subject

    print("\n--- LOSO Summary: Architecture 2 (subject-specific) ---")
    print_fold_summary(fold_metrics, subject_ids)
    _save_results(fold_metrics, subject_ids, results_dir, "cnn_lstm_subject_specific")


def evaluate_cnn_lstm_cross_subject(processed_dir: Path, config: dict, results_dir: Path):
    """Architecture 3: cross-subject LOSO with CNN-LSTM."""
    print("\n=== Architecture 3 (cross-subject) ===\n")

    device = get_device()
    print(f"Device: {device}")
    fold_metrics = []

    subject_files = sorted(processed_dir.glob("chb*.npz"))
    subject_ids = [f.stem for f in subject_files]

    for fold in loso_folds(processed_dir, seed=config.get("seed", 42)):
        sid = fold["test_subject"]
        print(f"Fold: held-out = {sid}")

        train_w, train_l = fold["train_windows"], fold["train_labels"]
        val_w, val_l = fold["val_windows"], fold["val_labels"]

        # Normalize using training set statistics only
        mean, std = compute_normalization_stats(train_w)
        train_w = apply_normalization(train_w, mean, std)
        val_w = apply_normalization(val_w, mean, std)
        test_w_norm = apply_normalization(fold["test_windows"], mean, std)

        model = _build_cnn_lstm(config)
        torch.manual_seed(config.get("seed", 42))
        model = train_cnn_lstm(model, train_w, train_l, val_w, val_l, config, device)

        y_prob = predict_proba(model, test_w_norm, device)
        metrics = compute_metrics(fold["test_labels"], y_prob)
        fold_metrics.append(metrics)
        print(f"  AUROC={metrics['auroc']:.3f}  Sens={metrics['sensitivity']:.3f}  n_ictal={metrics['n_ictal']}")

    print("\n--- LOSO Summary: Architecture 3 (cross-subject) ---")
    print_fold_summary(fold_metrics, subject_ids)
    _save_results(fold_metrics, subject_ids, results_dir, "cnn_lstm_cross_subject")


# ---------------------------------------------------------------------------
# Architecture 4: DANN
# ---------------------------------------------------------------------------

def evaluate_dann(processed_dir: Path, config: dict, results_dir: Path):
    print("\n=== Architecture 4: DANN ===\n")
    device = get_device()
    print(f"Device: {device}")
    fold_metrics = []
    subject_ids_out = []

    for fold in loso_folds(processed_dir, seed=config.get("seed", 42)):
        sid = fold["test_subject"]
        subject_ids_out.append(sid)
        n_train_subjects = fold["n_train_subjects"]
        print(f"Fold: held-out = {sid}  |  n_train_subjects = {n_train_subjects}")

        mean, std = compute_normalization_stats(fold["train_windows"])
        train_w = apply_normalization(fold["train_windows"], mean, std)
        val_w = apply_normalization(fold["val_windows"], mean, std)
        test_w = apply_normalization(fold["test_windows"], mean, std)

        model = DANN(
            n_channels=config.get("n_channels", 23),
            window_samples=config.get("window_samples", 1024),
            sfreq=config.get("sample_rate", 256),
            eegnet_F1=config.get("eegnet_f1", 8),
            eegnet_D=config.get("eegnet_d", 2),
            n_subjects=n_train_subjects,
            lambda_max=config.get("dann_lambda_max", 1.0),
        )

        torch.manual_seed(config.get("seed", 42))
        model = train_dann(
            model,
            train_w, fold["train_labels"], fold["train_subject_ids"],
            val_w, fold["val_labels"],
            config, device,
        )

        y_prob = predict_proba(model, test_w, device, is_dann=True)
        metrics = compute_metrics(fold["test_labels"], y_prob)
        fold_metrics.append(metrics)
        print(f"  AUROC={metrics['auroc']:.3f}  Sens={metrics['sensitivity']:.3f}  n_ictal={metrics['n_ictal']}")

    print("\n--- LOSO Summary: DANN ---")
    print_fold_summary(fold_metrics, subject_ids_out)
    _save_results(fold_metrics, subject_ids_out, results_dir, "dann")


# ---------------------------------------------------------------------------
# Results persistence
# ---------------------------------------------------------------------------

def _save_results(fold_metrics: list, subject_ids: list, results_dir: Path, tag: str):
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
