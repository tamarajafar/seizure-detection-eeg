"""
metrics.py

Evaluation metrics for binary seizure detection under class imbalance.
All functions operate on numpy arrays of true labels and predicted probabilities.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    f1_score,
    confusion_matrix,
    balanced_accuracy_score,
)


def _optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find the threshold that maximizes Youden's J (sensitivity + specificity - 1)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr  # Youden's J = sensitivity + specificity - 1
    return float(thresholds[np.argmax(j_scores)])


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: str = "optimal") -> dict:
    """
    Compute the full set of evaluation metrics for one fold.

    Parameters
    ----------
    y_true : np.ndarray, shape (n_windows,)
        Ground-truth binary labels (0 = interictal, 1 = ictal).
    y_prob : np.ndarray, shape (n_windows,)
        Predicted probability of the ictal class.
    threshold : str or float
        "optimal" to select via Youden's J statistic on the ROC curve,
        or a float for a fixed decision threshold.

    Returns
    -------
    dict with keys: auroc, sensitivity, specificity, f1, balanced_accuracy,
                    n_ictal, n_interictal, prevalence, threshold
    """
    n_ictal = int(y_true.sum())
    n_interictal = int(len(y_true) - n_ictal)

    if threshold == "optimal" and n_ictal > 0:
        thresh = _optimal_threshold(y_true, y_prob)
    else:
        thresh = float(threshold) if threshold != "optimal" else 0.5

    y_pred = (y_prob >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "auroc": roc_auc_score(y_true, y_prob) if n_ictal > 0 else float("nan"),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "n_ictal": n_ictal,
        "n_interictal": n_interictal,
        "prevalence": n_ictal / len(y_true) if len(y_true) > 0 else 0.0,
        "threshold": thresh,
    }


def aggregate_metrics(fold_metrics: list[dict]) -> dict:
    """
    Aggregate per-fold metrics into mean and standard deviation across folds.

    Folds with NaN AUROC (no seizures in test set) are excluded from the AUROC
    average but counted in the fold total.

    Parameters
    ----------
    fold_metrics : list of dict
        One dict per LOSO fold, as returned by compute_metrics.

    Returns
    -------
    dict with keys: {metric}_mean, {metric}_std for each numeric metric,
                    plus n_folds and n_folds_with_seizures.
    """
    keys = [k for k, v in fold_metrics[0].items() if isinstance(v, float)]
    result = {"n_folds": len(fold_metrics)}

    for key in keys:
        vals = np.array([m[key] for m in fold_metrics])
        valid = vals[~np.isnan(vals)]
        result[f"{key}_mean"] = float(np.mean(valid)) if len(valid) > 0 else float("nan")
        result[f"{key}_std"] = float(np.std(valid)) if len(valid) > 1 else 0.0

    result["n_folds_with_seizures"] = sum(
        1 for m in fold_metrics if not np.isnan(m["auroc"])
    )
    return result


def print_fold_summary(fold_metrics: list[dict], subject_ids: list[str]):
    """Print a formatted per-fold results table to stdout."""
    header = f"{'Subject':<10} {'AUROC':>7} {'Sens':>7} {'Spec':>7} {'F1':>7} {'Thresh':>7} {'N_ictal':>8}"
    print(header)
    print("-" * len(header))
    for sid, m in zip(subject_ids, fold_metrics):
        print(
            f"{sid:<10} "
            f"{m['auroc']:>7.3f} "
            f"{m['sensitivity']:>7.3f} "
            f"{m['specificity']:>7.3f} "
            f"{m['f1']:>7.3f} "
            f"{m.get('threshold', 0.5):>7.4f} "
            f"{m['n_ictal']:>8d}"
        )
    agg = aggregate_metrics(fold_metrics)
    print("-" * len(header))
    print(
        f"{'Mean':.<10} "
        f"{agg['auroc_mean']:>7.3f} "
        f"{agg['sensitivity_mean']:>7.3f} "
        f"{agg['specificity_mean']:>7.3f} "
        f"{agg['f1_mean']:>7.3f}"
    )
    print(
        f"{'Std':.<10} "
        f"{agg['auroc_std']:>7.3f} "
        f"{agg['sensitivity_std']:>7.3f} "
        f"{agg['specificity_std']:>7.3f} "
        f"{agg['f1_std']:>7.3f}"
    )
