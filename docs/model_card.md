# Model Card: Cross-Subject Seizure Detection from EEG

**EE 541 Final Project -- Spring 2026**
Authors: Tamara Jafar, Bayla Breningstall

*This card documents the best-performing model from our comparative study. Complete per-architecture results are in the final report.*

---

## Model Details

**Architecture:** [TO FILL: e.g. DANN / EEGNet + BiLSTM]
**Framework:** PyTorch [version]
**Parameters:** [TO FILL]
**Training hardware:** [TO FILL]
**Training time per LOSO fold:** [TO FILL]

### Design choices
- EEGNet depthwise separable CNN as feature extractor: compact, designed for EEG, few parameters relative to performance
- Bidirectional LSTM for temporal context across consecutive 4-second windows
- Gradient reversal layer (DANN) to learn subject-invariant representations
- Weighted random sampling (10:1 interictal:ictal) to address class imbalance without oversampling

---

## Training Data

**Dataset:** CHB-MIT Scalp EEG Database (PhysioNet, Shoeb 2010)
- 23 pediatric subjects, ages 1.5-22 years
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
- Pediatric population only -- generalization to adult epilepsy is untested
- Single institution (Children's Hospital Boston) -- site effects uncharacterized
- Seizure annotations are at the onset/offset level; pre-ictal state is not labeled

---

## Evaluation Protocol

Leave-one-subject-out cross-validation across all 23 subjects. For each fold,
22 subjects train the model and 1 unseen subject is the test set. This is the
only evaluation that honestly measures cross-subject generalization.

Primary metric: AUROC (area under the ROC curve), reported as mean ± SD across 23 folds.

---

## Performance Metrics

*To be filled after experiments complete (due 2026-05-08).*

| Architecture | AUROC (mean ± SD) | Sensitivity | Specificity | F1 |
|---|---|---|---|---|
| 1: Logistic regression | TBD | TBD | TBD | TBD |
| 2: CNN-LSTM subject-specific | TBD | TBD | TBD | TBD |
| 3: CNN-LSTM cross-subject | TBD | TBD | TBD | TBD |
| 4: DANN (best model) | TBD | TBD | TBD | TBD |

---

## Intended Use and Limitations

**Intended use:** Research comparison of domain generalization strategies for
EEG-based seizure detection. This is an academic project; the model is not
validated for clinical deployment.

**Where it works:** Subjects whose seizure semiology resembles the CHB-MIT
population (focal onset with or without secondary generalization, pediatric age).

**Where it fails:**
- Subjects with novel seizure types not represented in training data
- EEG recorded at substantially different sampling rates or with different channel sets
- Patients with extensive movement artifact or poor electrode contact
- Very brief seizures (<4 seconds, below the window length)

---

## Failure Modes

*To be filled after failure analysis (final report section).*

Known failure modes to investigate:
- Short seizures that fall within a single window boundary
- Subjects with low-amplitude focal seizures
- High-motion artifact windows misclassified as ictal

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
