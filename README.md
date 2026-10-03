# 🎙️ AudioGuard — AI Audio Deepfake Detection

> A modern web application for detecting whether speech audio appears to be genuine human speech or AI-generated.

AudioGuard is an audio-only machine learning application designed to analyze speech recordings and identify acoustic characteristics associated with synthetic or AI-generated speech. The system extracts acoustic features using **Mel-Frequency Cepstral Coefficients (MFCCs)** and classifies the audio using a trained **Random Forest** model.

---

## 📋 System Pipeline

```text
Audio Input (WAV, MP3, FLAC, OGG, M4A)
                   ↓
Audio Preprocessing & 16 kHz Mono Standardization
                   ↓
MFCC Feature Extraction (172-dim Spectral Statistics)
                   ↓
Trained Random Forest Classification Model
                   ↓
REAL / AI-GENERATED Prediction & Confidence Score
```

---

## ✨ Key Features

- 🎙️ **Audio Detection:** Upload speech recordings or record voice audio directly to detect synthetic or voice-cloned speech.
- 🧠 **MFCC Feature Extraction:** Captures 172-dimensional statistical acoustic descriptors (MFCCs, Delta MFCCs, Mel-Spectrogram stats, Spectral Centroid, Bandwidth, Rolloff, ZCR, RMS Energy, and Spectral Flatness).
- ⚡ **Random Forest Classifier:** Fast CPU-based machine learning inference with real-time prediction output.
- 📊 **Confidence & Probability Breakdown:** Displays model detection confidence percentage along with side-by-side REAL and FAKE class probabilities.
- 🔊 **Acoustic Visualizations:** Renders detailed Waveform, MFCC Heatmap, and Mel-Spectrogram plots for acoustic inspection.
- ⏱️ **Segment Analysis:** Divides audio longer than 4 seconds into sliding windows to analyze segment-by-segment risk and plot an *AI Probability Over Time* chart.
- 🔒 **Local Privacy-First Execution:** Audio is processed locally on your system and is not uploaded to external AI APIs or permanently stored.
- 📄 **Report Export:** Generate and download text summary reports for audio analysis results.

---

## 🛠️ Supported Formats & System Requirements

- **Supported Audio Formats:** WAV, MP3, FLAC, OGG, M4A
- **Maximum Audio Duration:** 60 seconds
- **Audio Standardization:** 16,000 Hz, Mono, Float32 Peak-Normalized Waveforms
- **Execution Requirements:** Standard Laptop CPU (Python 3.10 / 3.11 / 3.12, 8 GB RAM min, Windows / macOS / Linux)

---

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/ashashik2005-creator/AudioGuard.git
cd AudioGuard
```

### 2. Create and Activate Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📁 Dataset Preparation

AudioGuard uses the public Hugging Face dataset [`garystafford/deepfake-audio-detection`](https://huggingface.co/datasets/garystafford/deepfake-audio-detection).

To download, validate, deduplicate via SHA-256 hashing, and create a 70% Train / 15% Validation / 15% Test split:

```bash
python download_dataset.py --max-per-class 100
```

*Note: You can control sample size with `--max-per-class` (e.g. 100, 250, 500, 900).*

---

## 🏋️ Model Training

To train the Random Forest model on the extracted MFCC features:

```bash
python train.py --backend mfcc
```

The trained model checkpoint will be saved in `models/mfcc_model.joblib` and configured in `models/config.json`.

---

## 💻 Command-Line Prediction

Run deepfake analysis on any audio file via CLI:

```bash
python predict.py --file data/processed/test/fake/fake_0008.flac
```

### Example CLI Output:

```text
==================================================
Audio file: data\processed\test\fake\fake_0008.flac
--------------------------------------------------

Prediction: FAKE

Fake probability: 80.00%
Real probability: 20.00%

Confidence:       80.00%

Processing time:  1.27 seconds
==================================================
```

---

## 🌐 Running the Web Application

Launch the interactive web interface:

```bash
streamlit run app.py
```

Access the application in your browser at:
`http://localhost:8501`

---

## 📄 License & Disclaimer

AudioGuard generates probabilistic automated predictions based on acoustic feature analysis. Detection confidence scores represent statistical model outputs and should be treated as automated indicators rather than absolute proof of authenticity.
