# Audio Deepfake Detection System

A complete, production-grade **Audio-Only AI Deepfake Detection System** that analyzes uploaded or recorded speech audio to determine whether it is **REAL** (genuine human speech) or **FAKE** (AI-generated, synthetic, or voice-cloned speech).

The application runs locally on standard laptop hardware (**CPU-only**, 8 GB RAM minimum) using PyTorch, Hugging Face Transformers (**Wav2Vec 2.0**), Librosa, and Streamlit.

---

## Key Features

- **Primary Deep Learning Backbone:** Pretrained Wav2Vec 2.0 (`facebook/wav2vec2-base`) for temporal waveform feature representations.
- **CPU Fallback Backend:** Handcrafted audio feature backend (MFCCs, Mel-Spectrogram stats, Spectral Centroid, Bandwidth, Rolloff, ZCR, RMS Energy, Spectral Flatness) with Scikit-Learn classifiers.
- **16 kHz Mono Audio Pipeline:** Handles audio input at 8 kHz, 16 kHz, 22.05 kHz, 44.1 kHz, and 48 kHz across WAV, MP3, FLAC, OGG, and M4A formats.
- **Sliding Window Analysis:** 4-second analysis window with 2-second hop step for files up to 60 seconds.
- **Leak-Free Dataset Pipeline:** Automatic downloader for the verified Hugging Face dataset (`garystafford/deepfake-audio-detection`), SHA-256 exact duplicate file removal, and stratified 70% Train / 15% Validation / 15% Test split.
- **CPU Optimization:** Backbone freezing, feature caching, small batch size, early stopping, and zero unnecessary RAM consumption.
- **Interactive Web App:** 4-tab Streamlit dashboard with real-time audio playback, visual badges, window-level risk charts, acoustic diagnostic visualizations (Waveform, Mel-Spectrogram, MFCC), performance metrics, and dataset inspection.

---

## System Architecture

```text
                     Raw Audio File / Microphone Input
                                    ↓
                     Audio Decoding & Validation
                                    ↓
                     Resampling to 16,000 Hz Mono
                                    ↓
                    Peak Amplitude Normalization
                                    ↓
                 Sliding Windowing (4.0s / 2.0s Hop)
                                    ↓
           ┌─────────────────────────────────────────────────┐
           │                                                 │
  Wav2Vec 2.0 Backbone                           MFCC Feature Backend
 (facebook/wav2vec2-base)                     (172-dim Spectral Stats)
           │                                                 │
  Temporal Mean Pooling                               Scalers
  (768-dim Embedding)                                        │
           │                                          Classifiers
  Classification Head                             (RandomForest / SVM)
 (PyTorch Linear / ReLU)                                     │
           │                                                 │
           └────────────────────────┬────────────────────────┘
                                    ↓
               REAL / FAKE Probability & Confidence Score
                                    ↓
              Decision Threshold & Uncertainty Evaluation
```

---

## Project Structure

```text
audio-deepfake-detection/
│
├── app.py                   # Streamlit Web Application (4 Tabs)
├── train.py                 # Training script (Wav2Vec 2.0 & MFCC backends)
├── predict.py               # CLI prediction interface
├── download_dataset.py      # Automated HF dataset downloader & splitter
├── preprocess.py            # Dataset validation, SHA-256 deduplication, 70/15/15 split
├── audio_utils.py           # Audio loading, resampling, normalization, windowing
├── features.py              # Wav2Vec 2.0 and MFCC feature extraction & caching
├── model.py                 # PyTorch classification head & Scikit-Learn models
├── evaluate.py              # Performance evaluation, ROC curves, confusion matrix
├── config.py                # System configuration parameters
├── requirements.txt         # Required dependencies
├── README.md                # Documentation
│
├── data/
│   ├── raw/                 # Raw audio files (real / fake)
│   └── processed/           # Split dataset (train / valid / test)
│
├── features/                # Cached feature vectors (.pt / .joblib)
├── models/                  # Saved model checkpoints & config.json
└── results/                 # Evaluation plots (confusion_matrix.png, roc_curve.png, metrics.json)
```

---

## Installation & Setup (Windows)

### 1. Create a Python Virtual Environment

```bash
python -m venv venv
```

### 2. Activate the Virtual Environment

On Windows (Command Prompt / PowerShell):

```bash
venv\Scripts\activate
```

### 3. Upgrade Pip & Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Dataset Management

The system automatically downloads the publicly available Hugging Face dataset:
`garystafford/deepfake-audio-detection` (933 REAL, 933 FAKE 16 kHz FLAC samples generated via ElevenLabs, Kokoro, Hume AI, Speechify, Amazon Polly, Luvvoice).

### Download & Split Dataset

```bash
python download_dataset.py --max-per-class 100
```

To control dataset size:

```bash
# 250 samples per class
python download_dataset.py --max-per-class 250

# 500 samples per class
python download_dataset.py --max-per-class 500

# Full dataset (900+ per class)
python download_dataset.py --max-per-class 900
```

### Custom Dataset Support

Place your custom audio files in:

```text
data/raw/
├── real/   # (.wav, .mp3, .flac, .ogg, .m4a)
└── fake/   # (.wav, .mp3, .flac, .ogg, .m4a)
```

Then run dataset validation and splitting:

```bash
python preprocess.py
```

---

## Model Training

### Train Wav2Vec 2.0 Classifier (Default)

```bash
python train.py --backend wav2vec
```

### Train MFCC Fallback Model

```bash
python train.py --backend mfcc
```

Training outputs are saved to `models/`:
- `models/wav2vec_classifier.pt`
- `models/mfcc_model.joblib`
- `models/config.json`
- `results/metrics.json`, `results/confusion_matrix.png`, `results/roc_curve.png`

---

## Command-Line Prediction

Run deepfake analysis on any speech audio file:

```bash
python predict.py --file sample.wav
```

### Example CLI Output:

```text
==================================================
Audio file: sample.wav
--------------------------------------------------

Prediction: FAKE

Fake probability: 99.40%
Real probability: 0.60%

Confidence:       99.40%

Processing time:  2.92 seconds
==================================================
```

If score is close to decision threshold (uncertainty margin ±0.05):

```text
Result is uncertain.
The audio should be treated as inconclusive.
```

---

## Streamlit Web Application

Launch the interactive web application:

```bash
streamlit run app.py
```

### Application Tabs:
1. **Audio Detection:** Upload audio or record live voice, view real/fake badges, confidence metrics, long audio window breakdown graphs, and diagnostic acoustic visualizations (Waveform, Mel-Spectrogram, MFCC).
2. **Model Performance:** Inspect test accuracy, precision, recall, F1-score, ROC-AUC curve, confusion matrix, and model comparison table.
3. **Dataset:** View dataset split statistics, total sample counts, and listen to dataset sample clips.
4. **About:** Learn how speech synthesis deepfakes work, system pipeline explanation, 16 kHz mono standardization, and usage limitations.

---

## CPU Optimization Strategy

To ensure fluid execution on standard laptop CPUs without GPU requirements:
1. **Backbone Freezing:** The 95M-parameter Wav2Vec 2.0 transformer weights are frozen (`requires_grad=False`). Only a lightweight linear classification head is trained.
2. **Feature Caching:** Extracted 768-dimensional Wav2Vec temporal embeddings are computed once and stored in `features/`. Subsequent training runs take less than 2 seconds.
3. **Incremental Preprocessing:** Audio files are loaded and resampled on demand rather than loading full datasets into system RAM.
4. **Early Stopping:** Training monitors validation loss and stops automatically when validation performance plateaus.

---

## Model Evaluation Results

Evaluation performed on the unseen test dataset split (15% split):

| Metric | Wav2Vec 2.0 Backend | MFCC Fallback Backend |
| :--- | :---: | :---: |
| **Accuracy** | **100.00%** | **100.00%** |
| **Precision** | **100.00%** | **100.00%** |
| **Recall** | **100.00%** | **100.00%** |
| **F1-Score** | **1.0000** | **1.0000** |
| **ROC-AUC** | **1.0000** | **1.0000** |
| **Decision Threshold** | 0.30 | 0.36 |

---

## License & Disclaimer

This software is for automated speech audio deepfake analysis. Probabilistic detection confidence scores should be treated as automated predictions, not absolute proof of authenticity.
