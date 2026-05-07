# Cross-Subject Seizure Detection from EEG: A Comparative Study of Domain Generalization Approaches

**EE 541: A Computational Introduction to Deep Learning**
University of Southern California, Spring 2026

**Authors:** Tamara Jafar, Bayla Breningstall

---

## 1. Introduction

Epilepsy affects approximately 70 million people globally and is characterized by sudden, recurrent seizures arising from abnormal synchronous neural activity [1]. Early and accurate seizure detection is critical for patient safety, yet clinical practice still relies on manual review of electroencephalography (EEG) recordings by trained neurologists -- a process that is time-intensive, expensive, and unavailable in real time. Automated seizure detection using deep learning offers a path toward continuous, scalable monitoring, but clinical deployment requires a model that generalizes to new patients who have never contributed labeled seizure data to training.

This problem is structurally similar to the Remaining Useful Life (RUL) prediction problem from which our topic is adapted. Both tasks involve continuous monitoring of a multivariate sensor system (industrial machinery or the brain), detection of a critical state transition (equipment failure or seizure onset), and the core challenge of building models that generalize across units (machines or patients) with distinct individual characteristics. The parallel is precise: just as a predictive maintenance model trained on a fleet of turbines may fail on a new turbine with different wear patterns, a seizure detector trained on 22 patients' EEG will encounter systematic distribution shift when applied to an unseen patient whose spectral profile, signal amplitude, and noise characteristics differ from the training population.

The central challenge is not classification accuracy in the conventional sense. A model that always predicts "non-seizure" achieves 98% accuracy on this dataset because seizures occupy roughly 1.5% of total recording time. What matters is sensitivity on a minority class under severe distributional shift -- across subjects whose EEG characteristics differ due to seizure type, cortical anatomy, medication status, age, and electrode placement variability. Standard subject-specific models achieve near-perfect performance by training and testing on the same patient, but this requires labeled seizure recordings from every new patient before deployment, which defeats the clinical purpose of automated detection. The unsolved problem is cross-subject generalization.

We compare four systems under identical leave-one-subject-out (LOSO) cross-validation: (1) a logistic regression baseline on hand-crafted frequency-band features, (2) a subject-specific EEGNet + bidirectional LSTM that defines the performance ceiling, (3) a naive cross-subject version of the same architecture that defines the domain shift penalty, and (4) a domain adversarial neural network (DANN) that explicitly optimizes for subject-invariant representations. Our primary question is how much of the domain shift penalty DANN recovers relative to the naive cross-subject baseline.

---

## 2. Related Work

### 2.1 Classical Feature-Based Approaches

Early automated seizure detection relied on hand-crafted features derived from the known spectral structure of ictal EEG. Subasi (2007) demonstrated that band-power features extracted via wavelet decomposition -- capturing activity in the delta (0.5-4 Hz), theta (4-8 Hz), alpha (8-13 Hz), beta (13-30 Hz), and gamma (30-40 Hz) bands -- are useful for discriminating seizure from non-seizure states [2]. These features are interpretable and computationally lightweight, making them a natural baseline, but they cannot capture the spatiotemporal propagation patterns across channels that characterize seizure onset, and they require careful normalization to transfer across subjects.

### 2.2 Deep Learning for EEG Classification

Deep learning approaches learn features directly from raw or minimally processed EEG, avoiding the need for manual feature engineering. Acharya et al. (2018) applied a deep convolutional neural network to single-channel EEG classification and reported approximately 90% accuracy, sensitivity, and specificity for distinguishing normal, preictal, and ictal states [3]. This work established that CNNs can automatically discover spectral and temporal patterns relevant to seizure classification. However, as Roy et al. (2019) documented in a systematic review, the overwhelming majority of deep learning EEG studies evaluate in an intra-subject setting -- training and testing on the same patient -- which produces impressive performance figures that do not transfer to clinical deployment scenarios where patient-specific labeled data is unavailable [4]. The review explicitly identifies cross-subject generalization as the primary open challenge in the field.

### 2.3 EEGNet

Lawhern et al. (2018) introduced EEGNet, a compact depthwise separable convolutional neural network designed specifically for EEG-based brain-computer interfaces [5]. The key insight is that standard convolutions applied to multichannel EEG mix temporal frequency content and spatial (cross-channel) information in a way that is difficult to interpret and prone to overfitting. EEGNet separates these: a temporal convolution first identifies spectral features within each channel, a depthwise spatial convolution then learns cross-channel combinations, and a separable pointwise convolution further reduces parameter count. The result is a model with far fewer parameters than general CNNs that matches or exceeds their performance across multiple EEG paradigms. We use EEGNet as the shared feature extractor across our deep learning architectures because its compact design is appropriate for the data scale available per subject in LOSO evaluation.

### 2.4 Domain Adversarial Neural Networks

The fundamental problem in cross-subject EEG classification is domain shift: the marginal distribution of EEG features differs across subjects even when the conditional relationship between features and seizure labels is shared. Ganin et al. (2016) introduced Domain-Adversarial Neural Networks (DANN) as a general solution to this class of problems [6]. DANN adds a gradient reversal layer between the feature extractor and a domain classifier (in our case, a subject-identity classifier). During backpropagation, gradients from the domain classifier are negated before flowing into the feature extractor, explicitly penalizing the extractor for encoding subject-identifying information while the task classifier simultaneously rewards it for encoding seizure-discriminative information. The result is a representation that is informationally useful for seizure detection but invariant to subject identity -- exactly the property needed for cross-subject generalization. The adversarial strength is controlled by a scalar lambda annealed from 0 to 1 over training following the schedule in Ganin et al.

---

## 3. Dataset and Preprocessing

### 3.1 Dataset

We use the CHB-MIT Scalp EEG Database (Shoeb 2010, PhysioNet) [7], a widely used benchmark for seizure detection research. The dataset contains continuous scalp EEG recordings from 24 pediatric subjects (ages 1.5-22 years) with intractable focal epilepsy, collected at Children's Hospital Boston following withdrawal of seizure medication to provoke seizures for clinical characterization. Recordings are stored in European Data Format (EDF) with 23 channels following the international 10-20 electrode placement system, sampled at 256 Hz. Each subject has an accompanying summary file containing neurologist-verified seizure onset and offset times in seconds.

The full dataset contains 198 annotated seizures across all 24 subjects, with substantial variability: chb12 has 40 seizures and chb22 has 3. Total recording duration is approximately 916 hours. Subjects vary in seizure semiology, with some having focal onset with secondary generalization and others having primary generalized seizures. This variability in seizure type, combined with differences in age, medication washout schedule, and cortical anatomy, produces the inter-subject spectral heterogeneity that makes cross-subject generalization difficult.

Class distribution after segmentation is highly imbalanced: approximately 98.5% of windows are interictal and 1.5% are ictal, with per-subject prevalence ranging from 0.03% (chb04) to 1.8% (chb12). A naive classifier that always predicts non-seizure achieves 98.5% accuracy while being clinically useless. This is why we evaluate using AUROC, sensitivity, and specificity rather than accuracy.

### 3.2 Preprocessing

**Bandpass filtering.** Raw EEG is bandpass filtered between 0.5 and 40 Hz using a 4th-order zero-phase Butterworth filter applied via second-order sections (SOS) to avoid numerical instability at low cutoff frequencies. The 0.5 Hz lower bound removes DC drift and slow movement artifact; the 40 Hz upper bound removes EMG contamination and power line harmonics. Zero-phase filtering is used to preserve the temporal and spatial relationships between channels.

**Segmentation.** Filtered recordings are segmented into non-overlapping 4-second windows (1,024 samples at 256 Hz). Non-overlapping segmentation is used to avoid creating correlated training examples. A window is labeled ictal (1) if more than 50% of its samples fall within an annotated seizure interval, and interictal (0) otherwise.

**Normalization.** Each channel is z-score normalized using per-channel mean and standard deviation computed from the training subjects only. Normalization statistics are never computed on validation or test data to prevent information leakage across folds.

**Class imbalance handling.** Training uses weighted random sampling that draws ictal and interictal windows at a 1:10 ratio, down-weighting the majority class rather than fully oversampling the minority class. This preserves the realistic class distribution during evaluation while preventing the model from ignoring the minority class during training.

**Data volume.** After preprocessing across all 24 subjects, the dataset contains approximately 812,000 windows total: approximately 800,000 interictal and 12,000 ictal, confirming the approximately 66:1 imbalance.

### 3.3 Evaluation Protocol

We use leave-one-subject-out (LOSO) cross-validation across all 24 subjects. For each fold, 23 subjects serve as training data and the held-out subject is the test set. This is the only evaluation protocol that honestly measures cross-subject generalization, because it exactly simulates the clinical scenario where a model trained on historical patients must perform on a new patient with no prior recordings. We report AUROC, sensitivity, specificity, and F1 as mean and standard deviation across all 24 folds. Within training data, approximately 10% of training subjects serve as a validation set for early stopping and hyperparameter selection.

---

## 4. Methodology

### 4.1 Architecture 1: Logistic Regression on Band-Power Features

For each 4-second window, we compute band power in five frequency bands (delta: 0.5-4 Hz, theta: 4-8 Hz, alpha: 8-13 Hz, beta: 13-30 Hz, gamma: 30-40 Hz) across all 23 channels using Welch's method with a 256-sample window. This produces a 115-dimensional feature vector per window (5 bands x 23 channels), concatenated in channel-major order. A logistic regression classifier with balanced class weights and L2 regularization (C=1.0) is fit using L-BFGS. Features are standardized using a sklearn StandardScaler fit on the training set only.

This baseline establishes the performance achievable with domain expert features and a linear classifier, without any deep learning. It is also the most interpretable architecture: the logistic regression weights directly indicate which frequency bands and channels are most predictive of seizure activity.

### 4.2 Architecture 2: Subject-Specific EEGNet + BiLSTM (Upper Bound)

EEGNet processes each 4-second EEG segment to produce a fixed-dimensional embedding. The EEGNet architecture consists of a temporal convolution with a 128-sample (0.5 second) kernel that captures spectral content, followed by a depthwise spatial convolution across the 23 channels, followed by a separable pointwise convolution, with average pooling and dropout (p=0.5) after each block. This produces a 128-dimensional embedding per window.

A two-layer bidirectional LSTM (hidden size 128 per direction) processes a sequence of consecutive embeddings to capture temporal context beyond the single 4-second window, and a linear head produces the binary classification logit.

For Architecture 2 specifically, the model is trained and tested on data from the same subject using a 90/10 train/validation split, with the 10% split used for evaluation. This represents the performance ceiling achievable when labeled seizure data from the test subject is available. The result cannot be deployed for new patients -- it requires waiting until the patient has a recorded seizure -- but it quantifies the upper bound of what is achievable and thereby defines the domain shift penalty incurred by cross-subject approaches.

### 4.3 Architecture 3: Naive Cross-Subject EEGNet + BiLSTM (Lower Bound)

Identical architecture to Architecture 2, but trained on 22 subjects and evaluated on the held-out subject with no domain adaptation. The gap between Architecture 2 and Architecture 3 AUROC directly measures the domain shift penalty: how much performance is lost purely from applying a model trained on other patients to a new patient.

### 4.4 Architecture 4: Domain Adversarial Neural Network (DANN)

The EEGNet feature extractor is augmented with a gradient reversal layer (GRL) connecting to a subject-identity classifier (23-way softmax, one class per training subject). The seizure classifier receives the same 128-dimensional feature embedding. During forward propagation, the GRL acts as an identity function. During backpropagation, it negates the gradient by a scalar lambda before it reaches the feature extractor. The total training loss is:

$$\mathcal{L} = \mathcal{L}_{\text{seizure}} - \lambda \cdot \mathcal{L}_{\text{subject}}$$

The feature extractor is simultaneously pushed by the seizure classifier to encode seizure-discriminative information and pushed by the negated subject gradient to discard subject-identifying information. Lambda is annealed from 0 to 1 over training following the sigmoid schedule from Ganin et al.: $\lambda = \lambda_{\max} \cdot \frac{2}{1 + e^{-10p}} - 1$, where $p$ is the fraction of total training steps elapsed.

### 4.5 Training Details

All deep learning models are trained with the Adam optimizer (learning rate 1e-3, weight decay 1e-4) for up to 50 epochs with early stopping (patience=7 epochs) on validation AUROC. Batch size is 64. All experiments use seed 42 for reproducibility. Training was performed on GPU (NVIDIA T4 via Kaggle). Model selection across folds uses the epoch with best validation AUROC, and weights are restored before test-set evaluation.

---

## 5. Results

### 5.1 Architecture 1: Logistic Regression Baseline

The logistic regression baseline achieves a mean AUROC of 0.735 ± 0.206 across 24 LOSO folds (Table 1). Despite reasonable discriminability at the ranking level, sensitivity is low at 0.457 ± 0.314 and F1 is very low at 0.054 ± 0.073. The disparity between AUROC and F1 reflects the class imbalance: the logistic regression ranks ictal windows above interictal ones with moderate reliability, but at any fixed decision threshold the number of false positives from the large interictal pool overwhelms the small number of true positives.

Performance varies substantially across subjects. The model succeeds on chb01 (AUROC=0.991, sensitivity=0.910), chb22 (AUROC=0.967), chb17 (AUROC=0.923), and chb18 (AUROC=0.927). It fails on chb12 (AUROC=0.305), chb14 (AUROC=0.337), chb16 (AUROC=0.391), and chb06 (AUROC=0.476) -- all performing near or below chance. The across-fold standard deviation of 0.206 is nearly as large as the mean AUROC gap between this baseline and chance (0.235), indicating that a linear frequency-band model generalizes to some subjects but not others in an unpredictable way.

The low F1 (0.054) is important context for interpreting AUROC. In a clinical setting where a false negative (missed seizure) carries patient safety implications, sensitivity of 0.457 is unacceptable regardless of AUROC. The logistic regression systematically undersells ictal windows because the decision boundary learned from 22 subjects' frequency profiles does not align with the ictal spectral signature of the held-out subject.

**Table 1.** Logistic regression LOSO-CV results (mean ± SD, 24 folds).

| Metric | Mean | SD |
|---|---|---|
| AUROC | 0.735 | 0.206 |
| Sensitivity | 0.457 | 0.314 |
| Specificity | 0.884 | 0.144 |
| F1 | 0.054 | 0.073 |
| Balanced accuracy | 0.671 | 0.141 |

### 5.2 Architecture 2: Subject-Specific EEGNet + BiLSTM

The subject-specific CNN-LSTM achieves a mean AUROC of 0.975 ± 0.051 across 24 folds (Table 2). This is a 0.240-point improvement over the logistic baseline, confirming that the EEGNet + BiLSTM architecture learns substantially richer seizure representations when trained on the target subject's own data. Sensitivity rises to 0.620 ± 0.368 and F1 to 0.441 ± 0.305, both substantial improvements over the baseline.

The architecture achieves perfect or near-perfect AUROC on many subjects: chb01, chb03, chb09, chb19, and chb22 all reach AUROC=1.000. The remaining variance is dominated by a small number of difficult subjects. chb12 is the single worst fold (AUROC=0.761), consistent with its low performance under the logistic baseline, suggesting that its seizure characteristics are genuinely difficult to learn regardless of architecture. chb04 (AUROC=0.926), chb08 (AUROC=0.988), and chb16 (AUROC=0.940) are also below the group mean, though all are well above chance.

The low F1 on several subjects despite high AUROC reflects cases where the model ranks ictal windows correctly but the calibration threshold is set too conservatively, resulting in low recall. chb15 (AUROC=0.997, F1=0.0) is the clearest example: the model has near-perfect discriminability but predicts no windows as ictal at threshold 0.5. This is a thresholding artifact of the imbalanced dataset rather than a failure of the ranking score, and it would be remedied by threshold optimization on the validation set.

The 0.975 AUROC represents the performance ceiling for this task and dataset under the constraint of using 4-second non-overlapping windows and a single-site EEGNet + BiLSTM architecture. The gap between this value and the cross-subject architectures (Architectures 3 and 4) directly quantifies the domain shift penalty and the potential gain from domain adaptation.

**Table 2.** Subject-specific CNN-LSTM LOSO-CV results (mean ± SD, 24 folds).

| Metric | Mean | SD |
|---|---|---|
| AUROC | 0.975 | 0.051 |
| Sensitivity | 0.620 | 0.368 |
| Specificity | 0.963 | 0.164 |
| F1 | 0.441 | 0.305 |
| Balanced accuracy | 0.792 | 0.184 |

**Table 3.** Per-subject AUROC, Architectures 1 and 2.

| Subject | Arch 1 (Logistic) | Arch 2 (CNN-LSTM SS) |
|---|---|---|
| chb01 | 0.991 | 1.000 |
| chb02 | 0.727 | 1.000 |
| chb03 | 0.773 | 1.000 |
| chb04 | 0.624 | 0.926 |
| chb05 | 0.623 | 0.997 |
| chb06 | 0.476 | 0.992 |
| chb07 | 0.706 | 0.999 |
| chb08 | 0.772 | 0.988 |
| chb09 | 0.776 | 1.000 |
| chb10 | 0.962 | 0.952 |
| chb11 | 0.926 | 1.000 |
| chb12 | 0.305 | 0.761 |
| chb13 | 0.842 | 0.981 |
| chb14 | 0.337 | 0.968 |
| chb15 | 0.823 | 0.997 |
| chb16 | 0.391 | 0.940 |
| chb17 | 0.923 | 0.999 |
| chb18 | 0.927 | 0.911 |
| chb19 | 0.712 | 1.000 |
| chb20 | 0.427 | 1.000 |
| chb21 | 0.842 | 1.000 |
| chb22 | 0.967 | 1.000 |
| chb23 | 0.944 | 0.999 |
| chb24 | 0.842 | 0.986 |
| **Mean** | **0.735** | **0.975** |
| **SD** | **0.206** | **0.051** |

---

*Results for Architecture 3 (naive cross-subject CNN-LSTM) and Architecture 4 (DANN) to be added upon completion of training runs.*

---

## References

[1] R. Bandopadhyay et al., "Recent Developments in Diagnosis of Epilepsy: Scope of MicroRNA and Technological Advancements," *Biology*, vol. 10, no. 11, p. 1097, 2021.

[2] A. Subasi, "EEG signal classification using wavelet feature extraction and a mixture of expert model," *Expert Systems with Applications*, vol. 32, no. 4, pp. 1084-1093, 2007.

[3] U. R. Acharya, S. L. Oh, Y. Hagiwara, J. H. Tan, and H. Adeli, "Deep convolutional neural network for the automated detection and diagnosis of seizure using EEG signals," *Computers in Biology and Medicine*, vol. 100, pp. 270-278, 2018.

[4] Y. Roy, H. Banville, I. Albuquerque, A. Gramfort, T. H. Falk, and J. Faubert, "Deep learning-based electroencephalography analysis: a systematic review," *Journal of Neural Engineering*, vol. 16, no. 5, p. 051001, 2019.

[5] V. J. Lawhern, A. J. Solon, N. R. Waytowich, S. M. Gordon, C. P. Hung, and B. J. Lance, "EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces," *Journal of Neural Engineering*, vol. 15, no. 5, p. 056013, 2018.

[6] Y. Ganin et al., "Domain-Adversarial Training of Neural Networks," *Journal of Machine Learning Research*, vol. 17, pp. 1-35, 2016.

[7] A. Shoeb, "CHB-MIT Scalp EEG Database," PhysioNet, 2010. https://doi.org/10.13026/C2K01R
