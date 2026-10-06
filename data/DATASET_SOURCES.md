# AudioGuard — Dataset Sources & Preprocessing Documentation

This document records the official dataset source, sample counts, validation, deduplication, licensing, and audio standardization protocol used in training the AudioGuard deepfake detection model.

---

## 1. Primary Dataset Information

- **Dataset Name:** `garystafford/deepfake-audio-detection`
- **Source Repository:** Hugging Face Hub (`https://huggingface.co/datasets/garystafford/deepfake-audio-detection`)
- **Domain:** Human Speech Authentication & Audio Deepfake Detection
- **Audio Encoding:** 16 kHz FLAC Audio
- **License / Access:** Publicly accessible research dataset

---

## 2. Sample Breakdown & Statistics

### Raw Download
- **Total Raw Samples:** 1,866 audio clips
- **Raw Real Clips:** 933 samples (genuine human speech)
- **Raw Fake Clips:** 933 samples (synthetic speech generated using neural TTS engines)

### Integrity & Deduplication Assessment
- **SHA-256 Hashing Audit:** 216 exact duplicate files detected and filtered out to prevent data contamination.
- **Corrupted / Unreadable Files:** 0 files.
- **Clean Valid Sample Pool:** **1,650 unique audio clips** (930 Real, 720 Fake).

### Stratified Train / Validation / Test Split (70% / 15% / 15%)

| Split | Total Samples | REAL Clips | FAKE Clips | Percentage |
| :--- | :--- | :--- | :--- | :--- |
| **Train Set** | 1,155 | 651 | 504 | 70.0% |
| **Validation Set** | 247 | 139 | 108 | 15.0% |
| **Test Set** | 248 | 140 | 108 | 15.0% |
| **Total** | **1,650** | **930** | **720** | **100.0%** |

---

## 3. Data Leakage Prevention Protocol

To guarantee realistic, generalization-focused model evaluation:
1. **Cryptographic SHA-256 Deduplication:** Every file's raw byte hash was computed prior to splitting. Duplicate recordings were removed so no sample appears in multiple splits.
2. **Stratified Sampling:** `sklearn.model_selection.train_test_split` with fixed random seed (`42`) was used to maintain the identical ~56.4% REAL to ~43.6% FAKE class balance across all splits.
3. **Strict Feature Caching Isolation:** Features for `train`, `valid`, and `test` splits were extracted and cached independently (`mfcc_train.joblib`, `mfcc_valid.joblib`, `mfcc_test.joblib`). `StandardScaler` parameters (mean and variance) are fitted *exclusively* on the Training set.

---

## 4. Audio Standardization & Preprocessing Pipeline

All dataset audio samples undergo standard normalization before feature extraction:
- **Resampling:** Uniform 16,000 Hz (16 kHz) single-channel (mono).
- **Amplitude Normalization:** Floating point 32-bit scaled to range \([-1.0, +1.0]\).
- **Windowing:** Fixed 4.0-second sliding windows with 50% overlap.
- **Feature Vector:** 172-dimensional statistical feature representation (MFCCs, Delta-MFCCs, Mel-Spectrogram stats, Spectral Centroid, Bandwidth, Rolloff, ZCR, RMS Energy, Spectral Flatness).
