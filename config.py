"""
Audio Deepfake Detection System - Configuration Settings
"""

import os
from pathlib import Path

# Project Base Directory
BASE_DIR = Path(__file__).resolve().parent

# Directory Paths
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_REAL_DIR = RAW_DIR / "real"
RAW_FAKE_DIR = RAW_DIR / "fake"

PROCESSED_DIR = DATA_DIR / "processed"
TRAIN_DIR = PROCESSED_DIR / "train"
VALID_DIR = PROCESSED_DIR / "valid"
TEST_DIR = PROCESSED_DIR / "test"

FEATURES_DIR = BASE_DIR / "features"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Ensure all directories exist
for path in [
    RAW_REAL_DIR, RAW_FAKE_DIR,
    TRAIN_DIR / "real", TRAIN_DIR / "fake",
    VALID_DIR / "real", VALID_DIR / "fake",
    TEST_DIR / "real", TEST_DIR / "fake",
    FEATURES_DIR, MODELS_DIR, RESULTS_DIR
]:
    path.mkdir(parents=True, exist_ok=True)

# Audio Preprocessing Parameters
SAMPLE_RATE = 16000
MONO = True
WINDOW_SECONDS = 4.0
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SECONDS)  # 64000 samples
HOP_SECONDS = 2.0
HOP_SAMPLES = int(SAMPLE_RATE * HOP_SECONDS)        # 32000 samples
MAX_AUDIO_SECONDS = 60.0
MAX_AUDIO_SAMPLES = int(SAMPLE_RATE * MAX_AUDIO_SECONDS)  # 960000 samples

# Supported Audio File Formats
SUPPORTED_FORMATS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

# Data Split Ratios
TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# Label Mappings
LABEL_TO_INT = {"real": 0, "fake": 1}
INT_TO_LABEL = {0: "REAL", 1: "FAKE"}

# Model Configurations
WAV2VEC_MODEL_NAME = "facebook/wav2vec2-base"
WAV2VEC_EMBEDDING_DIM = 768

# Training Hyperparameters (Optimized for CPU)
BATCH_SIZE = 4
EPOCHS = 10
EARLY_STOPPING_PATIENCE = 2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Decision Threshold & Uncertainty Margin
DEFAULT_DECISION_THRESHOLD = 0.50
UNCERTAINTY_MARGIN = 0.05  # Scores within [threshold - margin, threshold + margin] are inconclusive

# HuggingFace Dataset Source
HF_DATASET_NAME = "garystafford/deepfake-audio-detection"
