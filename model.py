"""
Deep Learning and Machine Learning Models for Audio Deepfake Classification
"""

import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

import config


class Wav2VecClassifierHead(nn.Module):
    """
    Lightweight classification head trained on top of Wav2Vec 2.0 temporal embeddings.
    Architecture: 768 -> 128 -> 1 (Logit output for BCEWithLogitsLoss)
    """

    def __init__(self, input_dim: int = config.WAV2VEC_EMBEDDING_DIM, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, 768]
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        logits = self.fc2(out)  # [batch_size, 1]
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns probability of FAKE class (0.0 to 1.0)."""
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        return probs


class FullWav2Vec2Classifier(nn.Module):
    """
    End-to-end Audio Deepfake Classifier combining Wav2Vec 2.0 backbone
    and the trained classification head.
    """

    def __init__(
        self,
        classifier_head: Wav2VecClassifierHead,
        model_name: str = config.WAV2VEC_MODEL_NAME,
        device: str = "cpu"
    ):
        super().__init__()
        self.device = torch.device(device)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.wav2vec = Wav2Vec2Model.from_pretrained(model_name).to(self.device)
        self.classifier_head = classifier_head.to(self.device)

        # Freeze Wav2Vec2 backbone
        self.wav2vec.eval()
        for param in self.wav2vec.parameters():
            param.requires_grad = False

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Input: waveforms tensor of shape [batch_size, samples] at 16 kHz
        Output: FAKE class probability tensor of shape [batch_size, 1]
        """
        # Feature extraction
        with torch.no_grad():
            # If input is raw numpy array or unnormalized tensor
            inputs = self.feature_extractor(
                [w.cpu().numpy() for w in waveforms],
                sampling_rate=config.SAMPLE_RATE,
                return_tensors="pt",
                padding=True
            )
            input_values = inputs.input_values.to(self.device)
            outputs = self.wav2vec(input_values)
            hidden_states = outputs.last_hidden_state  # [batch, seq_len, 768]
            pooled = torch.mean(hidden_states, dim=1)   # [batch, 768]

        probs = self.classifier_head.predict_proba(pooled)
        return probs


def build_mfcc_classifier(classifier_type: str = "rf") -> Pipeline:
    """
    Builds a Scikit-Learn pipeline for Handcrafted/MFCC feature classification.
    classifier_type options: 'rf' (RandomForest), 'svm' (Support Vector Machine), 'lr' (Logistic Regression)
    """
    scaler = StandardScaler()

    if classifier_type.lower() == "rf":
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            random_state=config.RANDOM_SEED,
            n_jobs=-1
        )
    elif classifier_type.lower() == "svm":
        clf = SVC(
            C=1.0,
            kernel="rbf",
            probability=True,
            random_state=config.RANDOM_SEED
        )
    elif classifier_type.lower() == "lr":
        clf = LogisticRegression(
            max_iter=1000,
            random_state=config.RANDOM_SEED
        )
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")

    pipeline = Pipeline([
        ("scaler", scaler),
        ("classifier", clf)
    ])
    return pipeline
