# Cross-Subject Seizure Detection from EEG: A Comparative Study of Domain Generalization Approaches

**EE 541: A Computational Introduction to Deep Learning**
University of Southern California, Spring 2026

**Authors:** Tamara Jafar, Bayla Breningstall

---

## 1. Introduction

Epilepsy affects approximately 70 million people globally and is characterized by sudden, recurrent seizures arising from abnormal synchronous neural activity [1]. Early and accurate seizure detection is critical for patient safety, yet clinical practice still relies on manual review of electroencephalography (EEG) recordings by trained neurologists -- a process that is time-intensive, expensive, and unavailable in real time. Automated seizure detection using deep learning offers a path toward continuous, scalable monitoring, but clinical deployment requires a model that generalizes to patients who have never contributed labeled seizure data to training.

This problem is structurally identical to the Remaining Useful Life (RUL) prediction problem from which our topic is adapted. Both involve continuous monitoring of a multivariate sensor system, detection of a critical state transition (equipment failure or seizure onset), and the core challenge of building models that generalize across units with distinct individual characteristics. Just as a predictive maintenance model trained on one fleet of turbines may fail on a new turbine with different wear patterns, a seizure detector trained on N patients encounters systematic distribution shift when applied to an unseen patient whose spectral profile, amplitude range, and noise characteristics differ from the training population.

Three properties make this problem particularly difficult. First, severe class imbalance: seizures occupy approximately 1.5% of total recording time, so a classifier that always predicts non-seizure achieves 98.5% accuracy while being clinically useless. Second, inter-subject distribution shift: each patient's EEG has a distinct spectral fingerprint due to differences in seizure type, cortical anatomy, medication status, age, and electrode placement, causing models trained on other patients to encounter a different data distribution at test time. Third, temporal structure across multiple scales: seizure dynamics evolve over seconds within a seizure, while the interictal-to-ictal transition may unfold over minutes, requiring architectures that can capture both.

We compare four systems under identical leave-one-subject-out (LOSO) cross-validation: a logistic regression baseline on hand-crafted frequency features, a subject-specific EEGNet + BiLSTM that defines the performance ceiling, a naive cross-subject CNN-LSTM that defines the domain shift penalty, and a Domain Adversarial Neural Network (DANN) that explicitly optimizes for subject-invariant representations. Our central question is how much of the domain shift penalty DANN recovers relative to the naive cross-subject baseline, and whether adversarial training improves generalization on a task where inter-subject variability is the dominant source of error.

Key findings: the logistic baseline achieves AUROC 0.735 ± 0.206, the subject-specific upper bound reaches 0.975 ± 0.051, the naive cross-subject CNN-LSTM drops to 0.808 ± 0.216, and DANN achieves 0.783 ± 0.192 -- marginally below the naive cross-subject baseline. The 0.167-point domain shift penalty (Architecture 2 minus Architecture 3) quantifies the cost of cross-subject generalization, and DANN fails to recover it.

---

## 2. Related Work

### 2.1 Classical Feature-Based Approaches

Early automated seizure detection relied on hand-crafted features derived from the known spectral structure of ictal EEG. Subasi (2007) demonstrated that band-power features extracted via wavelet decomposition -- capturing delta (0.5-4 Hz), theta (4-8 Hz), alpha (8-13 Hz), beta (13-30 Hz), and gamma (30-40 Hz) activity -- are useful for discriminating seizure from non-seizure states [2]. These features are interpretable and computationally lightweight, and they serve as our Architecture 1 baseline. Their limitation is that they cannot capture spatiotemporal propagation patterns across channels that characterize seizure onset, and their cross-subject generalization is constrained by the fact that patients' baseline spectral profiles differ systematically.

### 2.2 Deep Learning for EEG Classification

Deep learning approaches learn features directly from raw EEG, avoiding manual feature engineering. Acharya et al. (2018) applied a deep CNN to single-channel EEG and reported approximately 90% accuracy, sensitivity, and specificity for distinguishing normal, preictal, and ictal states [3]. However, as Roy et al. (2019) documented in a systematic review, the overwhelming majority of deep learning EEG studies evaluate in an intra-subject setting -- training and testing on the same patient [4]. This produces strong performance figures that do not transfer to clinical deployment where patient-specific labeled data is unavailable. Roy et al. explicitly identify cross-subject generalization as the primary open challenge, motivating the structure of our experimental comparison.

### 2.3 EEGNet

Lawhern et al. (2018) introduced EEGNet, a compact depthwise separable CNN designed for EEG-based brain-computer interfaces [5]. Standard convolutions applied to multichannel EEG mix temporal and spatial information in a way that overfits on small per-subject datasets. EEGNet separates these: a temporal convolution identifies spectral features within each channel, a depthwise spatial convolution learns cross-channel combinations, and a pointwise convolution further reduces parameters. The result achieves competitive performance across multiple EEG paradigms with far fewer parameters than general CNNs. We use EEGNet as the shared feature extractor in Architectures 2, 3, and 4 because its compact design is appropriate given the limited labeled data per subject under LOSO evaluation.

### 2.4 Domain Adversarial Neural Networks

The core problem in cross-subject EEG is domain shift: the marginal distribution of EEG features differs across subjects even when the conditional relationship between features and seizure labels is stable. Ganin et al. (2016) introduced Domain-Adversarial Neural Networks (DANN) as a general solution to this class of problems [6]. A gradient reversal layer (GRL) sits between the feature extractor and a domain classifier. In the forward pass the GRL acts as identity; in backpropagation it negates gradients before passing them to the feature extractor. This simultaneously optimizes the feature extractor to produce seizure-discriminative representations (via the task head) and subject-invariant representations (via the negated domain gradient). We apply DANN with per-training-subject identity labels as the domain variable, hypothesizing that invariance to subject identity will improve generalization to held-out patients.

---

## 3. Dataset and Preprocessing

### 3.1 Dataset

We use the CHB-MIT Scalp EEG Database (Shoeb 2010, PhysioNet) [7], the standard benchmark for cross-subject seizure detection research. The dataset contains continuous scalp EEG recordings from 24 pediatric subjects (ages 1.5-22 years) with intractable focal epilepsy, collected at Children's Hospital Boston following withdrawal of seizure medication to provoke seizures for clinical characterization. Recordings are stored in European Data Format (EDF) with 23 channels following the international 10-20 electrode placement system, sampled at 256 Hz. Each subject has an accompanying summary file containing neurologist-verified seizure onset and offset times in seconds. One known data quality issue is that some recordings contain duplicate channel labels (specifically "T8-P8"); MNE handles this automatically by appending a running index.

The full dataset contains 198 annotated seizures across 24 subjects, with substantial variability in seizure burden: chb12 has 40 seizures and several subjects (chb02, chb07, chb17, chb19, chb22) have only 3. Total recording duration is approximately 916 hours. Subjects vary in seizure semiology, with focal onset, secondary generalization, and primary generalized patterns all represented.

**Class distribution.** After segmentation, the dataset contains approximately 812,000 windows total, of which approximately 12,000 are ictal (1.5%) and 800,000 are interictal (98.5%). Per-subject ictal prevalence ranges from 0.03% (chb04, 8 ictal windows) to 1.8% (chb12, 380 ictal windows). This variability means that performance on subjects with few ictal windows is highly sensitive to small changes in sensitivity.

### 3.2 Exploratory Data Analysis

Preliminary inspection of band-power distributions across subjects reveals the inter-subject spectral heterogeneity that drives the cross-subject generalization problem. Subjects vary considerably in their baseline theta and alpha power, and in the spectral shift associated with ictal activity. For some subjects (e.g. chb01, chb22) ictal epochs show a pronounced broadband power increase relative to interictal baseline, which makes frequency-band features discriminative. For others (e.g. chb12, chb14) the ictal spectral signature is less separable from the interictal baseline, which is consistent with their poor performance across all architectures. The class-imbalanced nature of the data motivated our choice of AUROC and sensitivity as primary metrics rather than accuracy.

### 3.3 Preprocessing Pipeline

**Bandpass filtering.** Raw EEG is bandpass filtered between 0.5 and 40 Hz using a 4th-order zero-phase Butterworth filter applied via second-order sections (SOS). The 0.5 Hz lower cutoff removes DC drift and slow movement artifact; the 40 Hz upper cutoff removes EMG contamination and power line harmonics. Zero-phase filtering (forward-backward application) is used to avoid phase distortion that would corrupt the temporal and spatial relationships between channels.

**Segmentation.** Filtered recordings are segmented into non-overlapping 4-second windows (1,024 samples at 256 Hz). Non-overlapping windows avoid creating highly correlated adjacent training examples. A window is labeled ictal (1) if more than 50% of its samples fall within a neurologist-annotated seizure interval, and interictal (0) otherwise. The 50% threshold ensures that labeled ictal windows contain predominantly seizure activity rather than transition periods.

**Normalization.** Each channel is z-score normalized using per-channel mean and standard deviation computed from the training subjects only. Normalization statistics are never computed from validation or test data. This is enforced structurally in the code: normalization is applied inside the evaluation loop after the LOSO fold is defined, not during preprocessing.

**Class imbalance handling.** Training uses weighted random sampling with a 10:1 interictal-to-ictal ratio. This down-weights the majority class rather than fully oversampling the minority class, which would introduce highly correlated duplicate ictal windows. The weighted sampler ensures the model sees ictal examples frequently enough to learn discriminative features without artificially equalizing the class prior, which would miscalibrate posterior probabilities.

### 3.4 Evaluation Protocol

We use leave-one-subject-out (LOSO) cross-validation across all 24 subjects. For each fold, 23 subjects serve as training data and the held-out subject is the test set. This is the only evaluation protocol that honestly measures cross-subject generalization, simulating the clinical scenario where a model trained on historical patients must perform on a new patient with no prior recordings. Within the 23 training subjects, approximately 10% (2-3 subjects) are held out as a validation set for early stopping and hyperparameter selection. Final metrics are reported as mean and standard deviation across all 24 folds. Primary metric is AUROC; secondary metrics are sensitivity, specificity, and F1.

---

## 4. Methodology

### 4.1 Architecture 1: Logistic Regression on Band-Power Features

**Feature extraction.** For each 4-second window, we compute band power in five frequency bands (delta: 0.5-4 Hz, theta: 4-8 Hz, alpha: 8-13 Hz, beta: 13-30 Hz, gamma: 30-40 Hz) across all 23 channels using Welch's method with a 256-sample window and 50% overlap. This produces a 115-dimensional feature vector per window (5 bands x 23 channels), concatenated in channel-major order.

**Model.** A logistic regression classifier with balanced class weights and L2 regularization (C=1.0) is fit using L-BFGS with a maximum of 1,000 iterations. Features are standardized using a StandardScaler fit on training features only.

**Rationale.** This baseline establishes what is achievable with domain expert features and a linear classifier. It is also the most interpretable architecture -- logistic regression weights directly indicate which frequency-band/channel combinations predict seizure activity. Because it uses no learnable representations and relies only on features known a priori to be relevant to seizure physiology, it provides a meaningful floor for the deep learning architectures.

```
Input: (23 channels x 1024 samples)
    |
    v
Welch PSD per channel (5 bands x 23 channels = 115 features)
    |
    v
StandardScaler (fit on training set)
    |
    v
Logistic Regression (C=1.0, balanced class weights)
    |
    v
Output: P(ictal)
```

### 4.2 Architecture 2: Subject-Specific EEGNet + BiLSTM (Upper Bound)

**EEGNet feature extractor.** EEGNet processes each 4-second EEG window (23 x 1024) and produces a 128-dimensional embedding. The architecture consists of:

- Block 1: Temporal convolution (F1=8 filters, kernel=128 samples = 0.5s) to capture spectral content, followed by depthwise spatial convolution (D=2, producing F1*D=16 filters) to learn cross-channel combinations, batch normalization, ELU activation, average pooling (kernel=4), and dropout (p=0.5).
- Block 2: Depthwise separable convolution (kernel=16 samples) followed by pointwise convolution to F2=16 filters, batch normalization, ELU, average pooling (kernel=8), dropout (p=0.5).
- Linear projection from the flattened output to a 128-dimensional embedding.

**BiLSTM.** A two-layer bidirectional LSTM (hidden size 128 per direction, total 256) processes a sequence of embeddings. In single-window mode (used throughout our experiments), each window is treated as a sequence of length 1; the BiLSTM provides the same linear transformation as a feedforward layer in this case but is included for architectural consistency and future extension to sequence-level classification.

**Classification head.** A linear layer maps the 256-dimensional BiLSTM output to 2 logits.

**Role.** Architecture 2 is trained and tested on data from the same subject using a 90/10 train/evaluation split. It represents the performance ceiling when labeled seizure data from the test patient is available, quantifying the maximum performance achievable with this architecture and the domain shift penalty incurred by cross-subject approaches.

```
Input: (23 x 1024)
    |
    v
EEGNet Block 1: Temporal Conv -> Depthwise Spatial Conv -> BN -> ELU -> AvgPool -> Dropout
    |
    v
EEGNet Block 2: DepthwiseSep Conv -> Pointwise Conv -> BN -> ELU -> AvgPool -> Dropout
    |
    v
Linear Projection -> (128-dim embedding)
    |
    v
BiLSTM (2 layers, hidden=128, bidirectional) -> (256-dim)
    |
    v
Linear -> (2 logits)
    |
    v
Output: P(ictal)
```

### 4.3 Architecture 3: Naive Cross-Subject EEGNet + BiLSTM (Lower Bound)

Identical architecture to Architecture 2, but trained on 22 subjects and tested on the held-out subject with no domain adaptation. Normalization statistics are computed from the training subjects. The gap between Architecture 2 and Architecture 3 AUROC directly measures the domain shift penalty -- the performance cost of applying a model trained on other patients to a new patient.

### 4.4 Architecture 4: Domain Adversarial Neural Network (DANN)

**Architecture.** The EEGNet feature extractor is shared between a seizure classification head and a domain (subject-identity) classification head. A gradient reversal layer (GRL) sits between the feature extractor and the domain head.

```
Input: (23 x 1024)
    |
    v
EEGNet Feature Extractor -> (128-dim embedding)
    |              |
    |              v
    |         Gradient Reversal Layer (scale = -lambda)
    |              |
    v              v
Seizure Head   Domain Head
(2-way)        (N_subjects-way)
    |
    v
Output: P(ictal)
```

**Training objective.** The total loss is:

L_total = L_seizure - lambda * L_subject

where L_seizure is binary cross-entropy for ictal/interictal classification and L_subject is cross-entropy for subject-identity classification across N_train training subjects (22 for each fold). The negation in the domain term is implemented via the GRL: gradients from L_subject are multiplied by -lambda before flowing into the feature extractor, while L_seizure gradients flow normally. The feature extractor is simultaneously pushed to encode seizure-discriminative information and to discard subject-identifying information.

**Lambda annealing.** Lambda is annealed from 0 to lambda_max=1.0 over training following the sigmoid schedule from Ganin et al.:

lambda(p) = lambda_max * (2 / (1 + exp(-10p)) - 1)

where p = current_step / total_steps in [0, 1]. Starting lambda at 0 allows the feature extractor to first learn useful seizure representations before adversarial pressure is applied; if lambda is large from the start, the domain classifier destabilizes training before the seizure head has learned anything.

### 4.5 Hyperparameters

All hyperparameters were set based on literature defaults (EEGNet paper, Ganin et al.) and validated on the held-out training subjects (the within-fold validation set). No grid search was performed due to computational constraints; a single configuration was used across all folds.

| Hyperparameter | Value | Rationale |
|---|---|---|
| Window length | 4 s (1024 samples) | Captures full seizure cycle at lowest relevant frequency (0.5 Hz); standard in literature |
| Batch size | 64 | Standard for EEG classification |
| Learning rate | 1e-3 | Adam default; stable across architectures |
| Weight decay | 1e-4 | L2 regularization to prevent overfitting on training subjects |
| Max epochs | 50 | Sufficient for convergence under early stopping |
| Early stopping patience | 7 epochs | On validation AUROC |
| Interictal:ictal sampling ratio | 10:1 | Reduces imbalance without full oversampling |
| EEGNet F1 | 8 | From Lawhern et al. recommendation |
| EEGNet D | 2 | From Lawhern et al. recommendation |
| Embedding dimension | 128 | Balance between capacity and overfitting risk |
| BiLSTM hidden size | 128 per direction | Standard for sequence modeling |
| BiLSTM layers | 2 | Depth for capturing temporal patterns |
| DANN lambda_max | 1.0 | From Ganin et al. |
| DANN annealing steps | Full training duration | Gradual ramp avoids early destabilization |
| Logistic C | 1.0 | Default L2 strength |
| Random seed | 42 | All experiments |

### 4.6 Training Procedures

**Optimizer.** Adam (learning rate 1e-3, weight decay 1e-4) for all deep learning models.

**Loss functions.** Binary cross-entropy for the seizure classification head in all architectures. Cross-entropy for the DANN domain head. Logistic regression uses L-BFGS with L2 regularization.

**Early stopping.** Training is monitored on validation AUROC (computed on held-out training subjects). Training halts when validation AUROC has not improved for 7 consecutive epochs; model weights are restored to the best-AUROC checkpoint. Validation AUROC is preferred over validation loss because it is more directly interpretable under class imbalance.

**Preventing overfitting.** Dropout (p=0.5) after each EEGNet block, L2 weight decay (1e-4), early stopping, and per-subject window caps (max 25,000 windows per training subject) to prevent high-seizure-burden subjects from dominating training.

**Checkpoint/resume.** For Architecture 3 and 4, per-fold checkpoints are saved after each completed fold. This allows training to resume from the last completed fold if a session times out, which is important for the 9-hour Kaggle session limit.

---

## 5. Results

### 5.1 Architecture 1: Logistic Regression Baseline

The logistic regression baseline achieves a mean AUROC of 0.735 +/- 0.206 across 24 LOSO folds (Table 1, Figure 1). Despite moderate discriminability at the ranking level, sensitivity is low at 0.457 +/- 0.314 and F1 is very low at 0.054 +/- 0.073 (Figure 2). The dissociation between AUROC and F1 reflects class imbalance: the logistic regression ranks ictal windows above interictal ones with moderate reliability, but at any fixed decision threshold the volume of false positives from the large interictal pool overwhelms the true positives.

Performance varies substantially across subjects (Table 6, Figure 3). The model succeeds on chb01 (AUROC=0.991), chb22 (AUROC=0.967), chb17 (AUROC=0.923), and chb18 (AUROC=0.927). It fails on chb12 (AUROC=0.305), chb14 (AUROC=0.337), chb16 (AUROC=0.391), and chb06 (AUROC=0.476), all near or below chance. The standard deviation of 0.206 across folds is comparable in magnitude to the mean improvement over chance (0.235), indicating that this model generalizes to some patients but not others in a way that is not predictable from subject-level metadata available to us.

**Table 1.** Architecture 1 (logistic regression) aggregate LOSO-CV results.

| Metric | Mean | SD |
|---|---|---|
| AUROC | 0.735 | 0.206 |
| Sensitivity | 0.457 | 0.314 |
| Specificity | 0.884 | 0.144 |
| F1 | 0.054 | 0.073 |
| Balanced Accuracy | 0.671 | 0.141 |

### 5.2 Architecture 2: Subject-Specific EEGNet + BiLSTM

The subject-specific CNN-LSTM achieves a mean AUROC of 0.975 +/- 0.051 across 24 folds (Table 2, Figure 1), a 0.240-point improvement over the logistic baseline. This confirms that the EEGNet + BiLSTM architecture learns substantially richer seizure representations when trained on the target subject's own data. Sensitivity rises to 0.620 +/- 0.368 and F1 to 0.441 +/- 0.305.

The architecture achieves perfect or near-perfect AUROC on many subjects: chb01, chb03, chb09, chb19, and chb22 all reach AUROC=1.000. Variance is dominated by a small number of difficult subjects. chb12 is again the worst fold (AUROC=0.761), consistent with its near-chance performance under the logistic baseline, suggesting that its seizure characteristics are genuinely difficult regardless of architecture. chb04 (0.926) and chb16 (0.940) also fall below the group mean.

The chb15 fold illustrates a systematic thresholding artifact: AUROC=0.997 but F1=0.0. The model ranks ictal windows near the top of its score distribution but predicts zero windows as ictal at threshold 0.5 because the ictal probability estimates are systematically miscalibrated toward low values. This is a consequence of training with 10:1 weighted sampling, which shifts the decision boundary. Threshold optimization on the validation set would recover F1 in these cases; AUROC is unaffected by calibration and remains the primary metric.

The 0.975 AUROC represents the performance ceiling for this architecture under the LOSO protocol. The gap between this value and Architecture 3 (Section 5.3) directly quantifies the domain shift penalty.

**Table 2.** Architecture 2 (subject-specific CNN-LSTM) aggregate LOSO-CV results.

| Metric | Mean | SD |
|---|---|---|
| AUROC | 0.975 | 0.051 |
| Sensitivity | 0.620 | 0.368 |
| Specificity | 0.963 | 0.164 |
| F1 | 0.441 | 0.305 |
| Balanced Accuracy | 0.792 | 0.184 |

### 5.3 Architecture 3: Naive Cross-Subject EEGNet + BiLSTM

The naive cross-subject CNN-LSTM achieves a mean AUROC of 0.808 ± 0.216 across 24 folds (Table 3, Figure 1), a 0.167-point drop from the subject-specific ceiling of 0.975. This gap is the domain shift penalty: the performance cost of applying a model trained on 22 other patients to a new patient without any adaptation. Sensitivity collapses to 0.119 ± 0.208 and F1 to 0.097 ± 0.170, reflecting severe thresholding failure -- the model ranks ictal windows above many interictal ones at the score level, but its calibrated outputs are too low to cross the 0.5 threshold for most subjects.

Per-subject AUROC (Table 6, Figure 3) shows that several subjects transfer well: chb01 (0.999), chb02 (0.998), chb09 (0.988), chb10 (0.982), and chb11 (0.983) all remain above 0.975 despite no access to target-subject data, suggesting their seizure signatures are consistent with those of training subjects. Generalization fails sharply for chb14 (0.131), chb20 (0.448), chb12 (0.499), and chb21 (0.618). chb14 is particularly striking: it achieves AUROC 0.968 under the subject-specific model but near-chance under cross-subject transfer, indicating its seizure signature is learnable from within-subject data but is not represented in the training distribution of other patients. chb12 remains near chance (0.499) under all cross-subject approaches, confirming that its failure is driven by intrinsic signal properties rather than distribution shift alone.

**Table 3.** Architecture 3 (naive cross-subject CNN-LSTM) aggregate LOSO-CV results.

| Metric | Mean | SD |
|---|---|---|
| AUROC | 0.808 | 0.216 |
| Sensitivity | 0.119 | 0.208 |
| Specificity | 0.997 | 0.007 |
| F1 | 0.097 | 0.170 |
| Balanced Accuracy | 0.558 | 0.104 |

### 5.4 Architecture 4: DANN

DANN achieves a mean AUROC of 0.783 ± 0.192 across 24 folds (Table 4, Figure 1) -- 0.025 points below the naive cross-subject baseline. The adversarial training does not recover the domain shift penalty; it slightly degrades ranking performance relative to Architecture 3. More importantly, sensitivity is 0.0 ± 0.0 and F1 is 0.0 ± 0.0 across all 24 folds. DANN predicts zero ictal windows at threshold 0.5 for every subject. This is a complete threshold collapse: the model's ictal probability scores rank windows in a meaningful order (AUROC > 0.5 for 22 of 24 subjects) but the scores are uniformly suppressed below 0.5, leaving the classifier effectively non-functional at any clinical operating threshold.

The most likely explanation is that adversarial training interferes with the calibration of the seizure classification head. As the GRL pushes the feature extractor toward subject-invariant representations, it also reduces the magnitude of seizure-discriminative features, flattening the output distribution of the seizure head. Because the class-imbalance structure of the problem requires ictal-predicting features to be large relative to the interictal baseline, even a modest suppression of feature magnitude by the domain gradient can push all ictal scores below threshold. This is consistent with the lambda annealing schedule: early in training the seizure head learns a useful representation, but as lambda ramps toward 1.0, the adversarial pressure steadily erodes calibration without a compensating benefit from subject-invariant features.

Per-subject AUROC is high for some subjects (chb10: 0.986, chb11: 0.981, chb03: 0.981, chb19: 0.975, chb09: 0.971) but near-chance for others (chb23: 0.430, chb12: 0.455, chb06: 0.556), with the same difficult subjects failing under all cross-subject approaches.

**Table 4.** Architecture 4 (DANN) aggregate LOSO-CV results.

| Metric | Mean | SD |
|---|---|---|
| AUROC | 0.783 | 0.192 |
| Sensitivity | 0.000 | 0.000 |
| Specificity | 0.998 | 0.006 |
| F1 | 0.000 | 0.000 |
| Balanced Accuracy | 0.499 | 0.003 |

### 5.5 Comparative Summary

**Table 5.** Aggregate AUROC across all four architectures.

| Architecture | AUROC Mean | AUROC SD | Sensitivity | Specificity | F1 |
|---|---|---|---|---|---|
| Arch 1: Logistic | 0.735 | 0.206 | 0.457 | 0.884 | 0.054 |
| Arch 2: CNN-LSTM (subject-specific) | 0.975 | 0.051 | 0.620 | 0.963 | 0.441 |
| Arch 3: CNN-LSTM (cross-subject) | 0.808 | 0.216 | 0.119 | 0.997 | 0.097 |
| Arch 4: DANN | 0.783 | 0.192 | 0.000 | 0.998 | 0.000 |

**Table 6.** Per-subject AUROC across all four architectures.

| Subject | N ictal | Arch 1 Logistic | Arch 2 CNN-LSTM SS | Arch 3 CNN-LSTM CS | Arch 4 DANN |
|---|---|---|---|---|---|
| chb01 | 78 | 0.991 | 1.000 | 0.999 | 0.892 |
| chb02 | 35 | 0.727 | 1.000 | 0.998 | 0.851 |
| chb03 | 79 | 0.773 | 1.000 | 0.940 | 0.981 |
| chb04 | 8 | 0.624 | 0.926 | 0.922 | 0.844 |
| chb05 | 93 | 0.623 | 0.997 | 0.934 | 0.914 |
| chb06 | 14 | 0.476 | 0.992 | 0.783 | 0.556 |
| chb07 | 36 | 0.706 | 0.999 | 0.886 | 0.918 |
| chb08 | 231 | 0.772 | 0.988 | 0.801 | 0.864 |
| chb09 | 20 | 0.776 | 1.000 | 0.988 | 0.971 |
| chb10 | 62 | 0.962 | 0.952 | 0.982 | 0.986 |
| chb11 | 159 | 0.926 | 1.000 | 0.983 | 0.981 |
| chb12 | 380 | 0.305 | 0.761 | 0.499 | 0.455 |
| chb13 | 112 | 0.842 | 0.981 | 0.791 | 0.671 |
| chb14 | 43 | 0.337 | 0.968 | 0.131 | 0.324 |
| chb15 | 330 | 0.823 | 0.997 | 0.839 | 0.853 |
| chb16 | 25 | 0.391 | 0.940 | 0.572 | 0.711 |
| chb17 | 74 | 0.923 | 0.999 | 0.875 | 0.917 |
| chb18 | 56 | 0.927 | 0.911 | 0.909 | 0.894 |
| chb19 | 54 | 0.712 | 1.000 | 0.623 | 0.975 |
| chb20 | 76 | 0.427 | 1.000 | 0.448 | 0.715 |
| chb21 | 39 | 0.842 | 1.000 | 0.618 | 0.590 |
| chb22 | 40 | 0.967 | 1.000 | 0.986 | 0.618 |
| chb23 | 108 | 0.944 | 0.999 | 0.979 | 0.430 |
| chb24 | 127 | 0.842 | 0.986 | 0.910 | 0.892 |
| **Mean** | | **0.735** | **0.975** | **0.808** | **0.783** |
| **SD** | | **0.206** | **0.051** | **0.216** | **0.192** |

Pooled ROC curves for Architectures 3 and 4 are shown in Figure 4; normalized confusion matrices are in Figure 5.

---

**Figure 1.** Mean AUROC ± SD across 24 LOSO folds for each architecture. Error bars show one standard deviation. Dashed line at 0.5 indicates chance. Architecture 2 (subject-specific CNN-LSTM) achieves the highest mean AUROC (0.975), establishing the performance ceiling. Architecture 1 (logistic regression, 0.735), Architecture 3 (naive cross-subject CNN-LSTM, 0.808), and Architecture 4 (DANN, 0.783) all fall substantially below this ceiling, with DANN performing marginally worse than the naive cross-subject baseline.

**Figure 2.** Mean sensitivity, specificity, and F1 across 24 LOSO folds for each architecture. The AUROC/F1 dissociation is visible across all architectures: high specificity coexists with low sensitivity and near-zero F1, a consequence of the 10:1 weighted sampling strategy and class-imbalanced evaluation at a fixed threshold of 0.5.

**Figure 3.** Per-subject AUROC for each architecture across all 24 held-out subjects. Each point represents one LOSO fold; horizontal bars show the mean. The wide spread within each architecture reflects the heterogeneity of seizure detectability across subjects. chb12 (lowest point, all architectures) and chb14 (low under cross-subject methods only) are identifiable as the most problematic subjects.

**Figure 4.** Pooled ROC curves for Architectures 3 (naive cross-subject CNN-LSTM) and 4 (DANN), constructed by concatenating predicted probabilities and true labels across all 24 LOSO folds. Pooled AUC values reflect aggregate ranking performance across the full dataset. Architectures 1 and 2 are omitted because raw prediction files were not saved for those runs.

**Figure 5.** Normalized confusion matrices for Architectures 3 and 4, pooled across all 24 LOSO folds. Values indicate the fraction of true positives and true negatives correctly classified at threshold 0.5. DANN's complete sensitivity failure (zero ictal detections at threshold 0.5 across all folds) is visible as a zero in the ictal/ictal cell.

---

### 5.6 Failure Analysis

**chb12 is the consistently hardest subject.** Across all four architectures, chb12 produces the lowest or near-lowest AUROC (logistic: 0.305, subject-specific CNN-LSTM: 0.761, cross-subject CNN-LSTM: 0.499, DANN: 0.455). chb12 has the highest seizure burden in the dataset (380 ictal windows, prevalence 1.8%), so data quantity is not the issue. The near-chance performance under cross-subject approaches (0.499, 0.455) and the modest performance even under the subject-specific model (0.761) indicates that chb12's ictal spectral signature overlaps substantially with its interictal baseline in a way that is not learnable from the EEGNet feature set. Domain adaptation cannot solve a problem where the signal is not separable.

**chb14 reveals a distinct failure mode.** chb14 achieves AUROC 0.968 under the subject-specific model but drops to 0.131 under cross-subject CNN-LSTM and 0.324 under DANN -- the worst generalization failure in the dataset. Its seizure signature is learnable from within-subject data but is not represented in the training subjects' combined distribution, meaning the cross-subject model has learned no features that transfer to chb14. This is a pure domain shift failure, not an intrinsic signal difficulty.

**Subjects with very few ictal windows are unreliable.** chb04 has 8 ictal windows total; the evaluation fold contains only 2 ictal windows (after the 90/10 split for Architecture 2). With 2 positive examples, the AUROC estimate is extremely noisy. This is a structural limitation of evaluating on small ictal counts, not a model failure.

**The F1/AUROC dissociation is systematic.** Across all architectures and most subjects, high AUROC coexists with low or zero F1. This arises because F1 depends on a fixed threshold (0.5), while AUROC is threshold-independent. The weighted sampler (10:1 ratio) shifts the decision boundary, and DANN's adversarial pressure further suppresses output magnitudes. Threshold calibration on the validation set would substantially improve F1 without changing AUROC.

---

## 6. Discussion

### 6.1 What Worked and Why

The subject-specific CNN-LSTM (Architecture 2) achieves near-perfect AUROC (0.975) when trained and evaluated on data from the same subject. This confirms that EEGNet's depthwise separable convolution is an effective inductive bias for EEG: separating temporal filtering (which captures frequency content) from spatial filtering (which captures cross-channel propagation patterns) is precisely the structure that distinguishes seizure from non-seizure EEG, where the key signal is both spectral (rhythmic high-amplitude oscillations) and spatial (propagation across cortical areas). The bidirectional LSTM adds temporal context across consecutive windows, though in single-window mode its benefit is primarily the nonlinear transformation it applies before the classification head.

The logistic regression's success on certain subjects (AUROC above 0.9 on chb01, chb10, chb11, chb17, chb18, chb22) confirms that band-power features are genuinely informative for seizure detection when the seizure type produces a clear broadband power shift. This is consistent with the electrophysiology: generalized tonic-clonic seizures and some focal seizures with secondary generalization produce large amplitude increases visible in all frequency bands simultaneously.

### 6.2 What Did Not Work and Why

The logistic baseline completely fails on several subjects (chb12, chb14, chb16, chb06 all near or below chance). These subjects likely have seizure types where the ictal spectral change is subtle or occurs in a frequency range that overlaps with normal activity, defeating a linear classifier on band-power features. This also illustrates why single-number accuracy metrics are misleading: a mean AUROC of 0.735 conceals the fact that the model is near-chance for roughly 25% of patients.

The high variance across subjects (SD=0.206 for Architecture 1, SD=0.051 for Architecture 2) reveals that generalization difficulty is not uniform. Models that appear to perform reasonably on average may be providing essentially no benefit for specific patient subpopulations. This has direct clinical implications: a cross-subject model cannot be deployed without knowing which patients it will fail on.

### 6.3 Data and Architecture Interactions

The class imbalance (1.5% ictal) interacts with LOSO evaluation in an important way. Subjects with few ictal windows (chb04: 8 windows total) produce unreliable per-fold metric estimates. AUROC computed from 2 positive examples has near-binary variance. This means the per-subject AUROC distribution in Table 6 conflates genuine model difficulty with estimation noise for low-seizure-burden subjects. A more robust evaluation would require minimum ictal count thresholds, but excluding subjects would bias the reported numbers toward easier cases.

The architectural choice of non-overlapping 4-second windows imposes a temporal resolution limit. A seizure that lasts 3 seconds will produce at most one window with more than 50% ictal overlap, even if the surrounding transition period contains discriminative signal. Overlapping windows or variable-length segmentation could recover this signal at the cost of introducing correlated training examples.

### 6.4 Comparison Across All Four Architectures

The full comparison reveals a clear performance ordering: Architecture 2 (subject-specific, 0.975) > Architecture 3 (cross-subject CNN-LSTM, 0.808) > Architecture 4 (DANN, 0.783) > Architecture 1 (logistic, 0.735). The domain shift penalty -- the gap between subject-specific and naive cross-subject performance -- is 0.167 AUROC points. DANN not only fails to recover this penalty but regresses by an additional 0.025 points relative to the naive baseline.

Two findings here are worth distinguishing. First, the naive cross-subject CNN-LSTM (Architecture 3) substantially outperforms logistic regression (0.808 vs. 0.735), confirming that learned representations generalize better across subjects than hand-crafted frequency features, even without domain adaptation. Second, DANN (Architecture 4) produces a lower AUROC than Architecture 3 and complete sensitivity collapse. This is not a subtle failure -- the adversarial objective actively degrades performance.

The likely explanation for DANN's failure is that the 22-way domain classification task (one class per training subject) is too fine-grained to learn a useful inductive bias for seizure detection. Enforcing invariance to individual subject identity -- including to seizure-relevant inter-subject differences -- discards discriminative features that would generalize. A more targeted form of domain adaptation that operates on coarser groupings (e.g., seizure type, age group, medication status) might preserve clinically relevant variance while suppressing irrelevant baseline differences. This remains speculative without the metadata to test it, but the DANN failure motivates reconsidering what the domain variable should encode.

### 6.5 Surprises and Insights

The most striking finding is DANN's complete sensitivity collapse across all 24 subjects. Despite ranking ictal windows above interictal ones with reasonable reliability (mean AUROC 0.783), DANN predicts zero ictal windows at threshold 0.5 for every subject -- a result that would be clinically indistinguishable from a trivially null model. This makes AUROC a misleading primary metric for DANN: a model that ranks correctly but never predicts positive is not useful. It also highlights a danger specific to class-imbalanced problems: the adversarial objective can suppress the magnitude of the seizure head's outputs without visibly degrading AUROC, which is rank-based and does not penalize calibration failure.

A second unexpected result is that the naive cross-subject CNN-LSTM (Architecture 3) outperforms DANN, reversing the expected ordering. The assumption motivating DANN -- that subject-invariant representations will generalize better -- does not hold here, at least under the standard implementation. Whether this is a fundamental limitation of the approach for seizure detection or a consequence of implementation choices (lambda schedule, domain granularity, architectural capacity) cannot be determined from these results alone.

The thresholding artifact (high AUROC, zero F1 on chb15 for Architecture 2) reinforces the broader point: AUROC and F1 can give completely contradictory impressions of model quality on imbalanced data, and reporting multiple metrics is not optional when class prevalence is 1.5%.

### 6.6 Limitations

**Window-level classification ignores seizure context.** Classifying each 4-second window independently discards sequential information about how the ictal state evolves. A seizure onset is typically preceded by pre-ictal changes that unfold over minutes; our architecture does not use this context.

**Single dataset, single institution.** All data comes from one hospital with one acquisition system. Generalization to EEG recorded with different hardware, different electrode montages, or adult patients is untested.

**No threshold optimization.** We report performance at a fixed threshold of 0.5, which is suboptimal for class-imbalanced tasks. Threshold calibration on the validation set could substantially improve sensitivity and F1 without changing AUROC.

**Computational constraints.** Training was limited to Kaggle T4 GPUs with session time limits. This precluded hyperparameter search; a systematic sweep over learning rate, regularization strength, and EEGNet filter counts could improve performance.

---

## 7. Conclusion

### 7.1 Summary

We implemented and evaluated four seizure detection systems under leave-one-subject-out cross-validation on the CHB-MIT scalp EEG dataset. A logistic regression baseline on band-power features achieves AUROC 0.735 ± 0.206. A subject-specific EEGNet + BiLSTM reaches 0.975 ± 0.051, establishing the performance ceiling and confirming that the task is highly learnable from within-subject data. The naive cross-subject CNN-LSTM drops to 0.808 ± 0.216, quantifying the domain shift penalty at 0.167 AUROC points. DANN achieves 0.783 ± 0.192 -- below the naive cross-subject baseline -- and produces zero sensitivity across all 24 folds, a complete failure of the adversarial training objective to produce a deployable classifier. The central finding is that adversarial domain adaptation in its standard formulation does not improve cross-subject seizure detection on this dataset, and may actively interfere with calibration in a class-imbalanced setting.

### 7.2 Required Reflections

**What question did we answer through this project?**

We answered: *how large is the domain shift penalty for EEG-based seizure detection, and can it be measured rigorously?* The comparison between Architecture 2 (subject-specific) and Architecture 3 (naive cross-subject) provides the first component of this answer -- a quantitative, per-subject measurement of how much performance is lost purely from generalizing across patients under an honest evaluation protocol. Many prior papers report cross-subject results but compare against within-subject baselines only at the group mean level, obscuring which patients drive the performance gap. Our per-subject breakdown in Table 6 reveals that some patients (chb12) are difficult for all methods, while others (chb01, chb22) are easy regardless of method. This decomposition informs whether domain adaptation is solving the right problem: if hard subjects are hard because of intrinsic signal properties rather than distribution shift, adversarial training cannot help them.

**Substantive extension.**

A concrete and meaningful extension would be to replace fixed 4-second non-overlapping windows with a seizure onset detection framing: use a sliding window with 50% overlap to compute ictal probability as a continuous score over time, then evaluate using time-to-detection metrics rather than window-level AUROC. Clinical EEG monitoring systems care about how quickly a seizure is detected after onset, not whether each 4-second block is correctly classified. This would require replacing AUROC with a detection latency distribution and false alarm rate per hour -- metrics that are directly clinically interpretable and would make the comparison between architectures more meaningful for real deployment. The preprocessing and model code are already compatible with this evaluation; only the evaluation script and metrics module would need modification.

A second extension would apply spectral normalization or instance normalization within the EEGNet feature extractor, rather than relying on training-set z-score normalization. Instance normalization removes the mean and variance of each test-time window independently, which would reduce the amplitude-scale component of inter-subject distribution shift without requiring access to training-subject statistics. This is particularly relevant for deployment scenarios where the new patient's recording characteristics are completely unknown.

**What are we still curious about?**

The most interesting open question is whether the subjects where all methods fail (chb12, chb14, chb16) fail for the same reason or different reasons. chb12 has high seizure burden but low detectability; chb14 has few seizures and near-chance logistic performance but recovers to AUROC=0.968 with a subject-specific model -- suggesting that its seizure signature is learnable but not cross-subject generalizable. Understanding whether these failure modes are physiologically interpretable (seizure type, medication effects, electrode placement) would inform which patients a cross-subject detection system can safely serve and which require subject-specific data collection before deployment.

We are also curious about whether DANN's adversarial objective is actually the right inductive bias for this problem. DANN enforces invariance to subject identity across the entire feature representation, but only some components of inter-subject EEG variability are irrelevant to seizure detection -- baseline spectral profiles, for instance. Seizure propagation patterns, which do vary across subjects, might also contain clinically meaningful information that should not be discarded. A more targeted approach might enforce invariance only in the frequency dimensions known to reflect baseline differences (e.g., resting alpha power) while preserving variance in dimensions relevant to seizure dynamics.

### 7.3 Contributions Statement

**Tamara Jafar:** Preprocessing pipeline implementation and validation, Architecture 1 (logistic regression) implementation and evaluation, Architecture 3 (cross-subject CNN-LSTM) training and evaluation, EEGNet model architecture, report writing (Introduction, Related Work, Dataset, Methodology, Results, Discussion, Conclusion).

**Bayla Breningstall:** Architecture 2 (subject-specific CNN-LSTM) implementation and evaluation, Architecture 4 (DANN) implementation and training, memory-efficient data loading and HPC job scripting, checkpoint/resume infrastructure, SLURM job script development and execution on USC CARC (Center for Advanced Research Computing) including environment setup, job scheduling, and output management across all architectures, repository organization.

---

## References

[1] R. Bandopadhyay et al., "Recent Developments in Diagnosis of Epilepsy: Scope of MicroRNA and Technological Advancements," *Biology*, vol. 10, no. 11, p. 1097, 2021.

[2] A. Subasi, "EEG signal classification using wavelet feature extraction and a mixture of expert model," *Expert Systems with Applications*, vol. 32, no. 4, pp. 1084-1093, 2007.

[3] U. R. Acharya, S. L. Oh, Y. Hagiwara, J. H. Tan, and H. Adeli, "Deep convolutional neural network for the automated detection and diagnosis of seizure using EEG signals," *Computers in Biology and Medicine*, vol. 100, pp. 270-278, 2018.

[4] Y. Roy, H. Banville, I. Albuquerque, A. Gramfort, T. H. Falk, and J. Faubert, "Deep learning-based electroencephalography analysis: a systematic review," *Journal of Neural Engineering*, vol. 16, no. 5, p. 051001, 2019.

[5] V. J. Lawhern, A. J. Solon, N. R. Waytowich, S. M. Gordon, C. P. Hung, and B. J. Lance, "EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces," *Journal of Neural Engineering*, vol. 15, no. 5, p. 056013, 2018.

[6] Y. Ganin et al., "Domain-Adversarial Training of Neural Networks," *Journal of Machine Learning Research*, vol. 17, pp. 1-35, 2016.

[7] A. Shoeb, "CHB-MIT Scalp EEG Database," PhysioNet, 2010. https://doi.org/10.13026/C2K01R
