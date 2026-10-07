# AudioGuard — Retraining, Audit & Generalization Report

**Date:** October 2026  
**Pipeline Compliance:** **100% Strict Adherence to Absolute Rule**  
$$\text{Audio Input} \xrightarrow{16\text{kHz Mono}} \text{Preprocessing} \xrightarrow{\text{MFCC (172-D)}} \text{StandardScaler} \xrightarrow{\text{RandomForestClassifier}} \text{REAL / FAKE / INCONCLUSIVE}$$  

---

## 1. Absolute Rule & ML Pipeline Compliance Confirmation

- [x] **No Pipeline Modifications:** The exact 6-stage architecture (`Audio` \(\rightarrow\) `Preprocessing` \(\rightarrow\) `MFCC` \(\rightarrow\) `StandardScaler` \(\rightarrow\) `RandomForestClassifier` \(\rightarrow\) `REAL / FAKE / INCONCLUSIVE`) was strictly preserved.
- [x] **No Secondary or Deep Learning Models:** Zero Wav2Vec, Wav2Vec2, Whisper, CNN, RNN, LSTM, GRU, Transformer, Vision Transformer, GAN, LLM, Isolation Forest, SVM, KNN, XGBoost, LightGBM, OOD detector, or feature-distance classifiers were introduced.
- [x] **No Feature Changes:** Feature vector size remains exactly **172 dimensions** (40 MFCCs mean/std, 40 Delta-MFCCs mean/std, 80 Mel-Spectrogram DB mean/std, Spectral Centroid, Bandwidth, Rolloff, ZCR, RMS Energy, Spectral Flatness).
- [x] **Single Scaler:** `StandardScaler` is fitted **exclusively on the training pool** and applied to validation, test, and external inference.

---

## 2. Comprehensive Dataset Audit

| Parameter | Original Dataset (`data/processed`) | Celebrity Dataset (`data/celebrity_audio`) | Combined Total |
| :--- | :--- | :--- | :--- |
| **Total Audio Files** | 1,650 | 1,715 | **3,365** |
| **REAL Audio Clips** | 930 | 798 | **1,728 (51.35%)** |
| **FAKE Audio Clips** | 720 | 917 | **1,637 (48.65%)** |
| **Training Samples** | 1,155 (651 Real / 504 Fake) | 1,035 (486 Real / 549 Fake) | **2,190 (1,137 Real / 1,053 Fake)** |
| **Validation Samples**| 247 (139 Real / 108 Fake) | 257 (117 Real / 140 Fake) | **504 (256 Real / 248 Fake)** |
| **Test Samples** | 248 (140 Real / 108 Fake) | 423 (195 Real / 228 Fake) | **671 (335 Real / 336 Fake)** |
| **Number of Speakers**| General Speech Pool | 100 Verified Celebrities | **100+ Speakers** |
| **Corrupted Files** | 0 | 0 | **0** |
| **Silent Files** | 0 | 0 | **0** |
| **SHA-256 Duplicates**| 0 | 0 | **0** |

---

## 3. Data Leakage & Speaker Independence Verification

- **Cross-Dataset SHA-256 Check:** 0 exact duplicate files across original and celebrity datasets.
- **Celebrity Speaker Independence:**
  - Train Speakers: **60**
  - Validation Speakers: **15**
  - Test Speakers: **25**
  - Train / Valid / Test Speaker Overlap: **0** (100% Speaker-Disjoint Partitioning).

---

## 4. Final Hyperparameters & Training Setup

### Preprocessing & Feature Extraction
- **Sample Rate & Format:** 16,000 Hz, Mono, Float32, Peak Normalized
- **Windowing:** 4.0-second sliding windows with 2.0-second hop
- **Feature Vector:** 172-dimensional statistical feature representation
- **Scaler:** `StandardScaler` fitted on `X_train_full` ($N=2,190$)

### Classifier Configuration (`sklearn.ensemble.RandomForestClassifier`)
- `n_estimators`: 400 decision trees
- `max_depth`: 16
- `min_samples_split`: 4
- `min_samples_leaf`: 2
- `max_features`: `"sqrt"`
- `class_weight`: `{0: 1.25, 1: 1.0}` (Optimized real-voice weighting to minimize false alarms on authentic speech)
- `random_state`: 42

### Threshold Calibration
- **Calibrated Decision Threshold ($T$):** `0.55` (tuned via validation set balanced accuracy)
- **Uncertainty Margin ($M$):** `0.05` ($0.50 - 0.60$ uncertainty band)

---

## 5. Measured Evaluation Performance (Dual & Combined Test Sets)

### Benchmark Metrics Table

| Metric | Original Test Set ($N=248$) | Celebrity Unseen Test ($N=423$) | Combined Test Set ($N=671$) |
| :--- | :--- | :--- | :--- |
| **Overall Accuracy** | **97.58%** | **92.43%** | **94.34%** |
| **Real Audio Accuracy** | **98.57% (138/140)** | **94.87% (185/195)** | **96.42% (323/335)** |
| **Fake Audio Accuracy** | **96.30% (104/108)** | **90.35% (206/228)** | **92.26% (310/336)** |
| **Precision** | **98.11%** | **95.37%** | **96.27%** |
| **Recall (Sensitivity)** | **96.30%** | **90.35%** | **92.26%** |
| **F1-Score** | **0.9720** | **0.9279** | **0.9422** |
| **ROC-AUC Score** | **0.9960** | **0.9724** | **0.9839** |
| **False Positive Rate** | **1.43%** | **5.13%** | **3.58%** |
| **Confusion Matrix** | TP=104, TN=138, FP=2, FN=4 | TP=206, TN=185, FP=10, FN=22 | TP=310, TN=323, FP=12, FN=26 |

---

## 6. External Audio Testing Protocol (`test_external_audio.py`)

A dedicated CLI utility [`test_external_audio.py`](file:///e:/deepfake_detection/test_external_audio.py) is provided to evaluate completely external audio files (e.g. friend's voice, phone recordings, laptop mic recordings, WhatsApp audio) **without adding them to the training set**:

```bash
python test_external_audio.py --file path/to/external_audio.wav
```

### Execution Guarantee
- Zero training leakage (audio file is never added to training data).
- Processed through the identical 6-stage pipeline (`Audio` \(\rightarrow\) `Preprocessing` \(\rightarrow\) `MFCC` \(\rightarrow\) `StandardScaler` \(\rightarrow\) `RandomForestClassifier` \(\rightarrow\) `Verdict`).
- Honest probability and 3-state verdict output (**AUTHENTIC AUDIO**, **AI-GENERATED AUDIO**, or **INCONCLUSIVE**).
