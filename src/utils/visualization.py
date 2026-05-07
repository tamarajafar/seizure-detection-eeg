"""
visualization.py

Generate all publication-quality figures from LOSO-CV result files.

Figures produced:
    fig1_auroc_comparison     -- bar chart: mean AUROC ± SD per architecture
    fig2_metrics_comparison   -- grouped bar: sensitivity / specificity / F1
    fig3_per_subject_auroc    -- strip plot of per-fold AUROC across subjects
    fig4_roc_curves           -- pooled ROC curve per architecture
    fig5_confusion_matrices   -- normalized confusion matrices

Usage:
    from pathlib import Path
    from src.utils.visualization import generate_all_figures
    generate_all_figures(
        results_dir=Path("results"),
        figures_dir=Path("results/figures"),
    )
"""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix


# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

ARCH_ORDER = [
    "logistic",
    "cnn_lstm_subject_specific",
    "cnn_lstm_cross_subject",
    "dann",
]

ARCH_LABELS = {
    "logistic":                  "Arch 1\nLogistic",
    "cnn_lstm_subject_specific": "Arch 2\nCNN-LSTM\n(subject-specific)",
    "cnn_lstm_cross_subject":    "Arch 3\nCNN-LSTM\n(cross-subject)",
    "dann":                      "Arch 4\nDANN",
}

ARCH_COLORS = {
    "logistic":                  "#4477AA",
    "cnn_lstm_subject_specific": "#228833",
    "cnn_lstm_cross_subject":    "#CCBB44",
    "dann":                      "#EE6677",
}


def _style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)


def _save(fig, figures_dir: Path, name: str):
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}.pdf / .png")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(results_dir: Path) -> dict:
    """Load all available *_results.json files. Returns dict keyed by arch tag."""
    results = {}
    for tag in ARCH_ORDER:
        path = results_dir / f"{tag}_results.json"
        if path.exists():
            with open(path) as f:
                results[tag] = json.load(f)
    return results


def load_predictions(results_dir: Path) -> dict:
    """Load *_predictions.npz files for architectures that have them."""
    preds = {}
    for tag in ARCH_ORDER:
        path = results_dir / f"{tag}_predictions.npz"
        if path.exists():
            npz = np.load(path, allow_pickle=False)
            subject_ids = sorted(set(
                k.replace("_y_true", "").replace("_y_prob", "")
                for k in npz.files
            ))
            preds[tag] = {
                sid: {"y_true": npz[f"{sid}_y_true"], "y_prob": npz[f"{sid}_y_prob"]}
                for sid in subject_ids
            }
    return preds


# ---------------------------------------------------------------------------
# Figure 1: AUROC bar chart
# ---------------------------------------------------------------------------

def fig_auroc_comparison(results: dict, figures_dir: Path):
    available = [t for t in ARCH_ORDER if t in results]
    means  = [results[t]["aggregate"]["auroc_mean"] for t in available]
    stds   = [results[t]["aggregate"]["auroc_std"]  for t in available]
    colors = [ARCH_COLORS[t] for t in available]
    labels = [ARCH_LABELS[t] for t in available]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(available))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, width=0.55,
                  error_kw={"linewidth": 1.2})
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("AUROC (mean ± SD, 24 folds)", fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Chance")
    ax.legend(fontsize=9)
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, m + s + 0.02,
                f"{m:.3f}", ha="center", va="bottom", fontsize=8)
    _style_ax(ax)
    fig.tight_layout()
    _save(fig, figures_dir, "fig1_auroc_comparison")


# ---------------------------------------------------------------------------
# Figure 2: Grouped metric bar chart
# ---------------------------------------------------------------------------

def fig_metrics_comparison(results: dict, figures_dir: Path):
    available = [t for t in ARCH_ORDER if t in results]
    metric_keys   = ["sensitivity_mean", "specificity_mean", "f1_mean"]
    metric_labels = ["Sensitivity", "Specificity", "F1"]
    metric_colors = ["#4477AA", "#228833", "#EE6677"]

    n_arch = len(available)
    n_met  = len(metric_keys)
    x = np.arange(n_arch)
    width = 0.22

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (key, label, color) in enumerate(zip(metric_keys, metric_labels, metric_colors)):
        vals = [results[t]["aggregate"][key] for t in available]
        offset = (i - n_met / 2 + 0.5) * width
        ax.bar(x + offset, vals, width=width, label=label, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels([ARCH_LABELS[t] for t in available], fontsize=9)
    ax.set_ylabel("Score (mean across 24 folds)", fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=9, loc="upper right")
    _style_ax(ax)
    fig.tight_layout()
    _save(fig, figures_dir, "fig2_metrics_comparison")


# ---------------------------------------------------------------------------
# Figure 3: Per-subject AUROC strip plot
# ---------------------------------------------------------------------------

def fig_per_subject_auroc(results: dict, figures_dir: Path):
    available = [t for t in ARCH_ORDER if t in results]
    rng = np.random.default_rng(42)

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, tag in enumerate(available):
        aurocs = [f["auroc"] for f in results[tag]["per_fold"]
                  if not (isinstance(f["auroc"], float) and f["auroc"] != f["auroc"])]
        jitter = rng.uniform(-0.15, 0.15, len(aurocs))
        ax.scatter(np.full(len(aurocs), i) + jitter, aurocs,
                   color=ARCH_COLORS[tag], alpha=0.6, s=25, zorder=3)
        ax.plot([i - 0.25, i + 0.25], [np.mean(aurocs)] * 2,
                color=ARCH_COLORS[tag], linewidth=2.5, zorder=4)

    ax.set_xticks(range(len(available)))
    ax.set_xticklabels([ARCH_LABELS[t] for t in available], fontsize=9)
    ax.set_ylabel("AUROC per held-out subject", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    _style_ax(ax)
    fig.tight_layout()
    _save(fig, figures_dir, "fig3_per_subject_auroc")


# ---------------------------------------------------------------------------
# Figure 4: Pooled ROC curves
# ---------------------------------------------------------------------------

def fig_roc_curves(predictions: dict, figures_dir: Path):
    if not predictions:
        print("  No prediction files -- skipping ROC curves.")
        return

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Chance")

    for tag in ARCH_ORDER:
        if tag not in predictions:
            continue
        y_true = np.concatenate([d["y_true"] for d in predictions[tag].values()])
        y_prob = np.concatenate([d["y_prob"] for d in predictions[tag].values()])
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        label = ARCH_LABELS[tag].replace("\n", " ")
        ax.plot(fpr, tpr, color=ARCH_COLORS[tag], linewidth=1.8,
                label=f"{label} (AUC={roc_auc:.3f})")

    ax.set_xlabel("False positive rate", fontsize=11)
    ax.set_ylabel("True positive rate", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    _style_ax(ax)
    fig.tight_layout()
    _save(fig, figures_dir, "fig4_roc_curves")


# ---------------------------------------------------------------------------
# Figure 5: Confusion matrices
# ---------------------------------------------------------------------------

def fig_confusion_matrices(predictions: dict, figures_dir: Path):
    available = [t for t in ARCH_ORDER if t in predictions]
    if not available:
        print("  No prediction files -- skipping confusion matrices.")
        return

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.5))
    if n == 1:
        axes = [axes]

    for ax, tag in zip(axes, available):
        y_true = np.concatenate([d["y_true"] for d in predictions[tag].values()])
        y_pred = (np.concatenate([d["y_prob"] for d in predictions[tag].values()]) >= 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred, normalize="true")

        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Interictal", "Ictal"], fontsize=9)
        ax.set_yticklabels(["Interictal", "Ictal"], fontsize=9)
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("True", fontsize=10)
        ax.set_title(ARCH_LABELS[tag].replace("\n", " "), fontsize=10)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                        fontsize=12, color="white" if cm[i, j] > 0.5 else "black")

    fig.tight_layout()
    _save(fig, figures_dir, "fig5_confusion_matrices")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate_all_figures(results_dir: Path, figures_dir: Path):
    """
    Generate all figures from saved result files.

    Parameters
    ----------
    results_dir : Path
        Directory containing *_results.json and optionally *_predictions.npz.
    figures_dir : Path
        Output directory. PDFs and 300 dpi PNGs are written here.
    """
    np.random.seed(42)
    results     = load_results(results_dir)
    predictions = load_predictions(results_dir)

    print(f"Results available:     {list(results.keys())}")
    print(f"Predictions available: {list(predictions.keys())}")
    print()

    if results:
        fig_auroc_comparison(results, figures_dir)
        fig_metrics_comparison(results, figures_dir)
        fig_per_subject_auroc(results, figures_dir)

    fig_roc_curves(predictions, figures_dir)
    fig_confusion_matrices(predictions, figures_dir)

    print(f"\nDone. Figures saved to {figures_dir}")
