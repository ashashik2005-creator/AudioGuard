"""
Feature Extraction Utilities for Wav2Vec 2.0 and Handcrafted Audio Features (MFCC, Mel Spec, Spectral Stats)
"""

import os
import io
from pathlib import Path
from typing import Union, List, Tuple, Dict, Any, Optional

import numpy as np
import librosa
import torch
import joblib
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

import config
from audio_utils import load_and_preprocess_audio, pad_or_crop_waveform, extract_sliding_windows


class Wav2VecExtractor:
    """
    Wav2Vec 2.0 Feature Extractor using pretrained facebook/wav2vec2-base model.
    Extracts contextual temporal representations and applies temporal pooling.
    """

    def __init__(self, model_name: str = config.WAV2VEC_MODEL_NAME, device: str = "cpu"):
        self.device = torch.device(device)
        self.model_name = model_name

        # Load HuggingFace Wav2Vec2 feature extractor & model
        self.hf_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # Freeze Wav2Vec 2.0 backbone parameters for CPU efficiency
        for param in self.model.parameters():
            param.requires_grad = False

    def extract_window_embedding(self, waveform_chunk: np.ndarray) -> np.ndarray:
        """
        Extracts 768-dim temporal pooled embedding for a 4-second audio window (64000 samples).
        """
        # Ensure input length is padded or cropped to standard window size
        waveform_chunk = pad_or_crop_waveform(waveform_chunk, config.WINDOW_SAMPLES)

        inputs = self.hf_feature_extractor(
            waveform_chunk,
            sampling_rate=config.SAMPLE_RATE,
            return_tensors="pt"
        )
        input_values = inputs.input_values.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_values)
            # outputs.last_hidden_state: [1, seq_len, 768]
            hidden_states = outputs.last_hidden_state

            # Temporal Mean Pooling across time frames
            pooled_embedding = torch.mean(hidden_states, dim=1).squeeze(0)  # [768]

        return pooled_embedding.cpu().numpy().astype(np.float32)

    def extract_file_embedding(self, file_input: Union[str, Path, bytes, io.BytesIO]) -> np.ndarray:
        """
        Extracts embedding for an entire audio file.
        If file > 4s, extracts embeddings for all sliding windows and averages them.
        """
        waveform, sr, _ = load_and_preprocess_audio(file_input)
        windows = extract_sliding_windows(waveform)

        window_embeddings = []
        for win_wave, _, _ in windows:
            emb = self.extract_window_embedding(win_wave)
            window_embeddings.append(emb)

        # Average window embeddings across file
        file_embedding = np.mean(window_embeddings, axis=0)
        return file_embedding.astype(np.float32)

    def cache_split_features(
        self, split_dir: Union[str, Path], cache_file: Union[str, Path], force: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Extracts and caches Wav2Vec embeddings for a dataset split (train/valid/test).
        Returns (embeddings, labels, file_paths).
        """
        split_dir = Path(split_dir)
        cache_file = Path(cache_file)

        if cache_file.exists() and not force:
            print(f"Loading cached Wav2Vec features from {cache_file}...")
            cached_data = torch.load(cache_file, map_location="cpu", weights_only=False)
            return cached_data["embeddings"].numpy(), cached_data["labels"].numpy(), cached_data["file_paths"]

        print(f"Extracting Wav2Vec features from {split_dir}...")
        embeddings = []
        labels = []
        file_paths = []

        for label_str, label_int in config.LABEL_TO_INT.items():
            class_dir = split_dir / label_str
            if not class_dir.exists():
                continue

            audio_files = [
                f for f in class_dir.iterdir()
                if f.is_file() and f.suffix.lower() in config.SUPPORTED_FORMATS
            ]

            for audio_path in audio_files:
                try:
                    emb = self.extract_file_embedding(audio_path)
                    embeddings.append(emb)
                    labels.append(label_int)
                    file_paths.append(str(audio_path))
                except Exception as e:
                    print(f"Skipping corrupt or unreadable file {audio_path}: {e}")

        if len(embeddings) == 0:
            raise ValueError(f"No valid audio features extracted from {split_dir}")

        X = np.array(embeddings, dtype=np.float32)
        y = np.array(labels, dtype=np.int64)

        cache_data = {
            "embeddings": torch.tensor(X),
            "labels": torch.tensor(y),
            "file_paths": file_paths
        }
        torch.save(cache_data, cache_file)
        print(f"Successfully cached {len(X)} samples to {cache_file}")

        return X, y, file_paths


class HandcraftedFeatureExtractor:
    """
    Lightweight Handcrafted Audio Feature Extractor for CPU fallback model.
    Extracts MFCCs, Delta MFCCs, Mel Spectrogram stats, Spectral Centroid,
    Bandwidth, Rolloff, ZCR, RMS Energy, and Spectral Flatness.
    """

    def __init__(self, n_mfcc: int = 20, n_mels: int = 40):
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels

    def extract_waveform_features(self, waveform: np.ndarray, sr: int = config.SAMPLE_RATE) -> np.ndarray:
        """
        Extracts a concatenated 1D statistical feature vector from an audio waveform chunk.
        """
        # Ensure valid finite waveform
        if not np.isfinite(waveform).all():
            waveform = np.nan_to_num(waveform)

        features = []

        # 1. MFCCs (20 coefficients)
        mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=self.n_mfcc)
        features.extend([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)])

        # 2. Delta MFCCs (20 coefficients)
        delta_mfcc = librosa.feature.delta(mfcc)
        features.extend([np.mean(delta_mfcc, axis=1), np.std(delta_mfcc, axis=1)])

        # 3. Mel-Spectrogram Stats (40 bands)
        mel_spec = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=self.n_mels)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        features.extend([np.mean(mel_spec_db, axis=1), np.std(mel_spec_db, axis=1)])

        # 4. Spectral Centroid
        centroid = librosa.feature.spectral_centroid(y=waveform, sr=sr)
        features.extend([[np.mean(centroid)], [np.std(centroid)]])

        # 5. Spectral Bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(y=waveform, sr=sr)
        features.extend([[np.mean(bandwidth)], [np.std(bandwidth)]])

        # 6. Spectral Rolloff
        rolloff = librosa.feature.spectral_rolloff(y=waveform, sr=sr)
        features.extend([[np.mean(rolloff)], [np.std(rolloff)]])

        # 7. Zero-Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y=waveform)
        features.extend([[np.mean(zcr)], [np.std(zcr)]])

        # 8. RMS Energy
        rms = librosa.feature.rms(y=waveform)
        features.extend([[np.mean(rms)], [np.std(rms)]])

        # 9. Spectral Flatness
        flatness = librosa.feature.spectral_flatness(y=waveform)
        features.extend([[np.mean(flatness)], [np.std(flatness)]])

        # Concatenate into 1D float32 array
        feature_vector = np.concatenate([np.ravel(f) for f in features]).astype(np.float32)
        return feature_vector

    def extract_file_features(self, file_input: Union[str, Path, bytes, io.BytesIO]) -> np.ndarray:
        """
        Extracts handcrafted features for an audio file.
        """
        waveform, sr, _ = load_and_preprocess_audio(file_input)
        windows = extract_sliding_windows(waveform)

        window_feats = []
        for win_wave, _, _ in windows:
            feat = self.extract_waveform_features(win_wave, sr)
            window_feats.append(feat)

        file_feat = np.mean(window_feats, axis=0)
        return file_feat.astype(np.float32)

    def cache_split_features(
        self, split_dir: Union[str, Path], cache_file: Union[str, Path], force: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Extracts and caches handcrafted features for a dataset split using joblib.
        """
        split_dir = Path(split_dir)
        cache_file = Path(cache_file)

        if cache_file.exists() and not force:
            print(f"Loading cached MFCC features from {cache_file}...")
            cached_data = joblib.load(cache_file)
            return cached_data["X"], cached_data["y"], cached_data["file_paths"]

        print(f"Extracting MFCC & Handcrafted features from {split_dir}...")
        features = []
        labels = []
        file_paths = []

        for label_str, label_int in config.LABEL_TO_INT.items():
            class_dir = split_dir / label_str
            if not class_dir.exists():
                continue

            audio_files = [
                f for f in class_dir.iterdir()
                if f.is_file() and f.suffix.lower() in config.SUPPORTED_FORMATS
            ]

            for audio_path in audio_files:
                try:
                    feat = self.extract_file_features(audio_path)
                    features.append(feat)
                    labels.append(label_int)
                    file_paths.append(str(audio_path))
                except Exception as e:
                    print(f"Skipping corrupt audio file {audio_path}: {e}")

        if len(features) == 0:
            raise ValueError(f"No valid audio features extracted from {split_dir}")

        X = np.array(features, dtype=np.float32)
        y = np.array(labels, dtype=np.int64)

        cache_data = {"X": X, "y": y, "file_paths": file_paths}
        joblib.dump(cache_data, cache_file)
        print(f"Successfully cached {len(X)} MFCC samples to {cache_file}")

        return X, y, file_paths
