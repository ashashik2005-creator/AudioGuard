# 🎙️ AudioGuard — AI Audio Deepfake Detection

> A modern audio-only web application that analyzes speech recordings and uses MFCC features with a trained Random Forest classifier to identify audio that appears genuine or AI-generated.

---

## 📌 Overview

**AudioGuard** is an AI-powered web application designed for audio deepfake detection. It accepts speech recordings in various formats, standardizes the audio to 16 kHz mono, extracts acoustic energy characteristics using Mel-Frequency Cepstral Coefficients (MFCCs), and processes them through a trained Random Forest classifier to generate a model-based prediction.

---

## ⚙️ How It Works

```text
🎧 Audio Input
       ↓
🔧 Audio Preprocessing
       ↓
🎵 MFCC Feature Extraction
       ↓
🌲 Random Forest Classifier
       ↓
🔍 Prediction
       ↓
REAL / AI-GENERATED
```

1. **Audio Input:** Accepts uploaded audio files or live microphone recordings.
2. **Audio Preprocessing:** Converts audio to 16 kHz mono and normalizes peak amplitude.
3. **MFCC Feature Extraction:** Extracts statistical spectral descriptors (MFCCs, Mel-spectrogram stats, spectral centroid, bandwidth, rolloff, ZCR, RMS energy, spectral flatness).
4. **Random Forest Classifier:** Evaluates the acoustic feature vector against trained decision boundaries.
5. **Prediction Output:** Displays classification (REAL or AI-GENERATED) and model confidence scores.

---

## ✨ Key Features

- 🎙️ **Upload Audio:** Upload speech audio files in multiple formats.
- 🎤 **Record Audio:** Capture live vocal audio directly via microphone.
- 🔊 **Audio Preview:** Integrated audio player with file name and size metadata.
- 🔍 **AI Deepfake Detection:** Instant classification into REAL or AI-GENERATED speech.
- 📊 **Prediction Confidence:** Displays model detection confidence percentage and class probability breakdown.
- 📈 **Waveform Visualization:** Render audio amplitude over time.
- 🎵 **MFCC Visualization:** View acoustic spectral feature heatmap.
- 🌈 **Mel-Spectrogram Visualization:** View log frequency spectrum distribution.
- ⏱️ **Audio Segment Analysis:** Evaluates long audio files in sliding windows with a time-series probability chart.
- 📄 **Analysis Report Download:** Export analysis results as a text summary report.
- 🔒 **Local Processing:** Audio is processed locally on your system and is not stored permanently.

---

## 🎧 Supported Audio

- **Supported Formats:** `WAV • MP3 • FLAC • OGG • M4A`
- **Maximum Duration:** `60 seconds`
- **Audio Specification:** `16,000 Hz, Mono`

---

## 🛠️ Technology

| Component | Technology |
| :--- | :--- |
| **Web Interface** | Streamlit |
| **Feature Extraction** | MFCC (Librosa) |
| **Machine Learning** | Random Forest (Scikit-Learn) |
| **Audio Processing** | Librosa, SoundFile, NumPy |
| **Model Storage** | Joblib |

---

## 📁 Dataset

AudioGuard is trained on the verified public dataset [`garystafford/deepfake-audio-detection`](https://huggingface.co/datasets/garystafford/deepfake-audio-detection) from Hugging Face, containing synthetic speech generated across multiple text-to-speech systems (including ElevenLabs, Kokoro, Hume AI, Speechify, Amazon Polly, and Luvvoice) and genuine human speech.

The dataset is partitioned into a **70% Train / 15% Validation / 15% Test** split with SHA-256 deduplication to prevent data leakage.

---

## 🚀 Running the Application

### 1. Clone & Setup

```bash
git clone https://github.com/ashashik2005-creator/AudioGuard.git
cd AudioGuard

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Launch Web Application

```bash
streamlit run app.py
```

Access the application at:
`http://localhost:8501`

---

## 📸 Screenshots

> Screenshots can be added here.

---

## 🧠 Model Information

- **Feature Extraction:** MFCC (172-dimensional statistical feature vector)
- **Classifier:** Random Forest Classifier
- **Output:** REAL / AI-GENERATED

The model generates predictions by identifying subtle statistical anomalies and spectral energy patterns in human vs. AI-generated speech.

---

## ⚠️ Limitations

- Prediction results depend on the characteristics of the training dataset.
- Detection performance may vary on unseen or newer speech synthesis models.
- Heavy background noise or low-quality recording conditions can affect model confidence.
- Results represent automated machine-learning predictions rather than absolute proof.

---

## ⚖️ Disclaimer

> AudioGuard provides automated machine-learning predictions based on audio characteristics. The result should be considered an analytical indication and not definitive proof of whether an audio recording is authentic or synthetic.
