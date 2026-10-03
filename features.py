"""
Feature Extraction Utilities for Handcrafted Audio Features (MFCC, Mel Spec, Spectral Stats)
"""

import os
import io
from pathlib import Path
from typing import Union, List, Tuple, Dict, Any, Optional

import numpy as np
import librosa
import joblib

import config
from audio_utils import load_and_preprocess_audio, pad_or_crop_waveform, extract_sliding_windows


class HandcraftedFeatureExtractor:
    """
    Lightweight Handcrafted Audio Feature Extractor.
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
