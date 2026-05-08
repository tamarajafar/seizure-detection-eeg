# Model Card: Cross-Subject Seizure Detection from EEG

**EE 541 Final Project Spring 2026**
Authors: Tamara Jafar, Bayla Breningstall

---

## Model Details

**Framework:** PyTorch 2.1, scikit-learn 1.4
**Training hardware:** Kaggle T4 GPU (16 GB VRAM)
**Evaluation protocol:** Leave-one-subject-out cross-validation, 24 folds

| Architecture | Description | Parameters (approx.) |
|---|---|---|
| 1: Logistic regression | Band-power features + L2 logistic regression | ~600 (115-D features x 2 classes) |
| 2: EEGNet + BiLSTM (subject-specific) | Learned EEG features, per-subject training | ~280,000 |
| 3: EEGNet + BiLSTM (cross-subject) | Same architecture, trained on 22 other subjects | ~280,000 |
| 4: DANN | EEGNet + BiLSTM + gradient reversal + domain head | ~320,000 |

### Design choices
- EEGNet depthwise separable CNN as feature extractor: compact, designed for EEG, few parameters relative to performance
- Bidirectional LSTM for temporal context across consecutive 4-second windows
- Gradient reversal layer (DANN) to learn subject-invariant representations
- Weighted random sampling (10:1 interictal:ictal) to address class imbalance without oversampling
- Early stopping on validation AUROC (patience=7) to prevent overfitting on training subjects

---

## Training Data

**Dataset:** CHB-MIT Scalp EEG Database (PhysioNet, Shoeb 2010)
- 24 pediatric subjects, ages 1.5-22 years
- 198 annotated seizures, ~916 hours total recording
- 23 EEG channels, 256 Hz, international 10-20 system

**Preprocessing:**
- Bandpass filtered 0.5-40 Hz (Butterworth 4th-order, zero-phase)
- Segmented into non-overlapping 4-second windows (1,024 samples)
- Window labeled ictal if >50% of samples overlap an annotated seizure
- Z-score normalized per channel using training-set statistics only

**Class distribution (approximate):**
- Interictal: ~800,000 windows (~98.5%)
- Ictal: ~12,000 windows (~1.5%)

**Known limitations:**
- Pediatric population only, generalization to adult epilepsy is untested
- Single institution (Children's Hospital Boston) site effects uncharacterized
- Seizure annotations are at the onset/offset level; pre-ictal state is not labeled

---

## Evaluation Protocol

Leave-one-subject-out cross-validation across all 24 subjects. For each fold,
23 subjects train the model and 1 unseen subject is the test set. This is the
only evaluation that honestly measures cross-subject generalization.

Primary metric: AUROC (area under the ROC curve), reported as mean ± SD across 24 folds.

---

## Performance Metrics

| Architecture | AUROC (mean ± SD) | Sensitivity | Specificity | F1 |
|---|---|---|---|---|
| 1: Logistic regression | 0.735 ± 0.206 | 0.457 ± 0.314 | 0.884 ± 0.144 | 0.054 ± 0.073 |
| 2: CNN-LSTM subject-specific | 0.975 ± 0.051 | 0.620 ± 0.368 | 0.963 ± 0.164 | 0.441 ± 0.305 |
| 3: CNN-LSTM cross-subject | 0.808 ± 0.216 | 0.119 ± 0.208 | 0.997 ± 0.007 | 0.097 ± 0.170 |
| 4: DANN | 0.783 ± 0.192 | 0.000 ± 0.000 | 0.998 ± 0.006 | 0.000 ± 0.000 |

**Domain shift penalty** (Arch 2 minus Arch 3): 0.167 AUROC points
**DANN vs naive cross-subject** (Arch 3 minus Arch 4): -0.025 AUROC points (DANN is worse)

All metrics are means across 24 LOSO folds. All experiments use random seed 42.

---

## Intended Use and Limitations

**Intended use:** Research comparison of domain generalization strategies for
EEG-based seizure detection. This is an academic project; the model is not
validated for clinical deployment.

**Where it works:** Subjects whose seizure semiology resembles the CHB-MIT
population (focal onset with or without secondary generalization, pediatric age).
Architectures 2 and 3 generalize well to subjects with broadband ictal power
shifts (chb01, chb09, chb10, chb11, chb22).

**Where it fails:**
- Subjects with novel seizure types not represented in training data (chb14 drops to AUROC 0.131 cross-subject)
- Subjects with low-contrast ictal signatures (chb12: AUROC 0.499-0.761 across all architectures)
- Very brief seizures (<4 seconds, below the window length)
- EEG recorded at substantially different sampling rates or with different channel sets

---

## Failure Modes

**chb12:** Near-chance performance across all architectures (AUROC 0.305-0.761). Highest seizure burden in dataset (380 ictal windows) but lowest detectability. Likely an intrinsic signal property, ictal spectral signature overlaps interictal baseline.

**chb14:** AUROC 0.968 subject-specific, 0.131 cross-subject (Architecture 3). Pure domain shift failure: the seizure signature is learnable but not represented in other patients' training data.

**DANN threshold collapse:** Architecture 4 produces zero sensitivity across all 24 subjects at threshold 0.5 despite AUROC 0.783. Adversarial training suppresses output magnitudes below the decision threshold. AUROC is preserved (rank order intact) but the model cannot produce positive predictions.

**F1/AUROC dissociation:** Many folds show AUROC > 0.99 but F1 = 0.0. This is a calibration artifact from 10:1 weighted sampling; threshold optimization on the validation set would recover F1.

---

## Fairness and Bias

The dataset is skewed toward specific pediatric seizure types collected at a
single US hospital. Models trained on this data may underperform for:
- Adult patients
- Patients from different geographic or demographic backgrounds
- Seizure types not well represented in the CHB-MIT dataset (e.g. absence seizures)

Class imbalance (~1.5% ictal) is addressed via weighted sampling and reported
separately on sensitivity and specificity rather than accuracy.

---

## Ethical Considerations

This model is developed for a course project and is not intended for clinical
deployment. Seizure detection in a clinical setting requires regulatory approval
(FDA 510(k) or equivalent), prospective clinical validation, and integration
with clinical workflows. Automated seizure detection that produces false
negatives in a clinical setting could delay treatment; this model's sensitivity
must be validated against a clinical standard before any deployment.
