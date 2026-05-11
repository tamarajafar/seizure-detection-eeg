# Cross-Subject Seizure Detection from EEG

**EE 541: A Computational Introduction to Deep Learning**
University of Southern California, Spring 2026

**Team:** Tamara Jafar, Bayla Breningstall

---

## Problem

We train deep learning models to detect epileptic seizures from multichannel scalp EEG, with the primary challenge being generalization to patients unseen during training. The task is binary classification: given a 4-second EEG window from an unseen subject, predict whether it is ictal (seizure) or interictal (non-seizure).

This is evaluated under **leave-one-subject-out (LOSO) cross-validation** across 24 subjects from the CHB-MIT dataset, the only protocol that honestly measures cross-subject generalization.

---

## Architectures Compared

| Architecture | Description | Role |
|---|---|---|
| 1 | Logistic regression on band-power features | Floor / interpretable baseline |
| 2 | EEGNet + BiLSTM, subject-specific | Upper bound (not deployable) |
| 3 | EEGNet + BiLSTM, naive cross-subject | Lower bound for cross-subject |
| 4 | DANN (domain adversarial neural network) | Main proposed method |

---

## Repository Structure

```
seizure-detection-eeg/
├── Final_report.pdf          # final project report
├── Final_report.docx
├── configs/
│   └── default.yaml          # all hyperparameters in one place
├── data/
│   └── README.md             # download instructions (no data in repo)
├── docs/
│   └── model_card.md         # model documentation
├── scripts/                  # SLURM batch scripts for cluster
│   ├── run_preprocessing.sbatch
│   ├── run_logistic.sbatch
│   ├── run_cnn_lstm_subject.sbatch
│   ├── run_cnn_lstm_cross.sbatch
│   ├── run_dann.sbatch
│   └── data_inspector.py
├── results/                  # LOSO-CV results (24 folds per architecture)
│   ├── logistic_results.json
│   ├── cnn_lstm_subject_specific_results.json
│   ├── cnn_lstm_cross_subject_results.json
│   ├── cnn_lstm_cross_subject_predictions.npz
│   ├── dann_results.json
│   ├── dann_predictions.npz
│   └── figures/              # publication-quality figures (PDF + PNG)
│       ├── fig1_auroc_comparison.{pdf,png}
│       ├── fig2_metrics_comparison.{pdf,png}
│       ├── fig3_per_subject_auroc.{pdf,png}
│       ├── fig4_roc_curves.{pdf,png}
│       └── fig5_confusion_matrices.{pdf,png}
├── src/
│   ├── preprocessing/
│   │   ├── load_edf.py       # parse .edf files and seizure annotations
│   │   ├── pipeline.py       # filter, segment, normalize, label
│   │   ├── spectrogram.py    # STFT conversion for CNN input
│   │   └── ee541-seizure-detection-chb-mit-eeg.ipynb  # EDA notebook
│   ├── models/
│   │   ├── logistic_baseline.py  # Architecture 1: band-power features
│   │   ├── eegnet.py             # depthwise separable CNN feature extractor
│   │   ├── cnn_lstm.py           # Architectures 2 & 3: EEGNet + BiLSTM
│   │   └── dann.py               # Architecture 4: domain adversarial network
│   ├── training/
│   │   ├── evaluate.py       # LOSO evaluation loop (main entry point)
│   │   ├── train.py          # training loops with early stopping
│   │   └── loso_cv.py        # fold generator and dataset utilities
│   └── utils/
│       ├── metrics.py        # AUROC, sensitivity, optimal thresholding
│       └── visualization.py  # generate all figures from results
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/tamarajafar/seizure-detection-eeg.git
cd seizure-detection-eeg

conda create -n seizure python=3.11
conda activate seizure
pip install -r requirements.txt
```

Then download the CHB-MIT dataset see [data/README.md](data/README.md).

---

## Running Experiments

### Step 1: Preprocess

```bash
python src/preprocessing/pipeline.py \
    --data_dir data/raw/physionet.org/files/chbmit/1.0.0 \
    --out_dir data/processed
```

### Step 2: Train and evaluate (LOSO-CV)

```bash
# Architecture 1: logistic regression baseline
python src/training/evaluate.py --arch logistic --config configs/default.yaml

# Architecture 3: naive cross-subject CNN-LSTM
python src/training/evaluate.py --arch cnn_lstm --config configs/default.yaml

# Architecture 4: DANN
python src/training/evaluate.py --arch dann --config configs/default.yaml
```

Architecture 2 (subject-specific) trains a separate model per subject:

```bash
python src/training/evaluate.py --arch cnn_lstm --subject_specific --config configs/default.yaml
```

Results are written to `results/` as JSON files with per-fold metrics and predictions saved as compressed `.npz` arrays.

### Step 3: Generate figures

After evaluation completes, generate all publication figures:

```python
from pathlib import Path
from src.utils.visualization import generate_all_figures

generate_all_figures(
    results_dir=Path("results"),
    figures_dir=Path("results/figures")
)
```

This creates 5 figures (PDF + 300 dpi PNG):
- **fig1**: AUROC comparison bar chart
- **fig2**: Sensitivity/Specificity/F1 grouped bars
- **fig3**: Per-subject AUROC strip plots
- **fig4**: Pooled ROC curves
- **fig5**: Confusion matrices (using optimal thresholds)

---

## Reproducing Reported Results

All experiments use `seed: 42` (set in `configs/default.yaml`). Hardware differences may produce minor floating-point variation, but mean AUROC across 24 folds should match reported values within 0.5%.

---

## Dataset

CHB-MIT Scalp EEG Database (PhysioNet). See [data/README.md](data/README.md) for download instructions.

> Shoeb A. CHB-MIT Scalp EEG Database. PhysioNet (2010). https://doi.org/10.13026/C2K01R
